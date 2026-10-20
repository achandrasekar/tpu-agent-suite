<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: v6e-measured-vpu-memory-limits
keyword: v6e-measured-vpu-memory-limits
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My elementwise, reduction or small-matmul work on TPU v6e is slow: what the vector unit, HBM access patterns, batch shape and VMEM limits actually cost

You are tuning the part of a TPU v6e workload that is not a big matmul:
chains of elementwise functions, reductions, gathers, strided reads,
random-number generation, or small and oddly batched matmuls. You want to
know which of the usual worries are real on this chip and which are not:
whether a transcendental is expensive, which way of writing an activation,
an exponential or a division is fastest inside a kernel, whether the order
you write independent operations in matters, at what point a streaming kernel stops
being limited by memory and starts being limited by the vector unit (VPU),
whether scattered or strided reads cost more than contiguous ones, which
axis to reduce along, what randomness costs, and whether short rows starve
the matrix unit (MXU), and what capping the compiler's scoped VMEM does
to a matmul. This page collects measurements of those questions. Every experiment ran on one v6e chip per
process, in two sessions on two different hosts. Some were plain PyTorch
through torch_tpu (eager, or `torch.compile` with the TPU backend); the
strided-read, reduction and roofline experiments were JAX Pallas kernels
called from PyTorch through torch_tpu's `pallas.jax_op`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process. Above one, the alternative is faster by that factor. Below one,
the alternative is slower, and the number is the share of the baseline's
speed it keeps. Where the two arms were meant to compute the same result,
the experiment checked that they did before timing them; where they were
meant to differ (random numbers against a constant, more arithmetic against
less), the text says so.

## Measured

### Long elementwise chains under torch.compile hit a fusion cliff

The experiment chained one operation many times over a single fp32 tensor
of about a million elements, compiled with `torch.compile`, at two chain
depths (256 and 1024 applications), and compared each chain with a chain of
adds of the same depth.

Read these ratios carefully. The add chain is not a fair "cheap operation"
baseline: XLA appears to fold a chain of near-identity adds (and
multiplies, divides and relus) into one operation (the compiled program
was not inspected), so the add arm measures little more than launching a
compiled graph. A chain of divides at
depth 1024 costs the same as the add chain, ratio
0.997,[^1] because it was folded
away too. The table below therefore does not rank operations by cost.

| chain | depth 256: share of the add chain's speed | depth 1024: share of the add chain's speed |
|---|---|---|
| tanh | 0.151[^2] | 0.00365[^3] |
| erf | 0.151[^4] | 0.00365[^5] |
| rsqrt | 0.151[^6] | 0.00365[^7] |
| sqrt | 0.0112[^8] | 0.0028[^9] |
| sigmoid | 0.0063[^10] | 0.0016[^11] |

What to take from this: tanh, erf and rsqrt gave identical ratios to four
digits at both depths, although erf is a polynomial on the vector unit and
tanh and rsqrt are single special-function operations. Time that does not
depend on the operation is not the operation's cost. Going from depth 256
to 1024 is four times the work but about 41 times the time (from roughly 7
to 270 times the folded graph), and at depth 1024 each operation costs
about one HBM round trip of the tensor. The likely reading is that the
chain stopped fitting in one fused loop, so each operation wrote and read
its intermediate through HBM; sqrt and sigmoid appear to cross that line
already at depth 256. So: under `torch.compile`, very long unbroken chains
on one tensor (hundreds of transcendentals) can fall off a fusion cliff.
For what individual transcendentals cost inside a kernel, see the latency
table in `index.py get v6e-measured-pallas-kernel-internals`.

### Don't hand-interleave independent elementwise chains

Four independent fp32 tensors each received sixteen tanh applications under
`torch.compile`. The baseline issued them round-robin across the tensors;
the alternative applied all sixteen to one tensor before moving to the next,
so every operation depended on the one before it. Both produce bit-identical
results.

- Order made no difference: ratios of
  1,[^12]
  1[^13] and
  1[^14] at the three largest
  tensor sizes. The compiler schedules the dataflow graph, not the order of
  your Python statements. Writing elementwise code in dependency order is
  fine.
- **Inside a Pallas kernel, interleaving by hand made things worse.** A
  Pallas recurrence over eight 128 by 128 fp32 tiles held in vector
  registers either ran two dependent single-accumulator loops one after the
  other, or interleaved two independent accumulators in one loop body to
  hide the vector unit's dependency latency. The interleaved loop kept
  0.944 of the speed at 256
  iterations[^15] and 0.943 at
  512:[^16] about 6% slower, where the experiment
  expected at least 1.2 times faster. Each 128 by 128 tile already gives
  the vector unit many independent register chains, so a second
  accumulator only doubles the live state. Don't hand-interleave
  recurrences for instruction-level parallelism; keep the plain dependent
  loops.

### Inside a Pallas loop, the obvious rewrite of an expensive op often loses

A set of small Pallas experiments ran a 16-step loop over bf16 tiles held in VMEM and swapped one formulation of an operation for another that should
be cheaper. Only one of the rewrites paid off; the rest were slower or no
faster.

- **Write the exponential as `exp2` with the constant folded in.** Replacing
  `jnp.exp(a * c1 - c2)` with `jnp.exp2` on constants pre-multiplied by
  log2(e) was 1.19× faster at 4096
  rows[^17] and 1.19× at
  8192,[^18] more than the experiment expected.
  Multiplying by log2(e) inside the loop and then calling `exp2` gained
  nothing (overall 1.01[^19]):
  the saving comes from folding the constant, not from the base.
- **Exact erf GELU was the fastest GELU.** The sigmoid approximation
  (`v * sigmoid(1.7 v)`) kept 0.841 of its
  speed[^20] and the tanh approximation
  (`jax.nn.gelu(approximate=True)`) only
  0.676.[^21] The
  approximations exist to be cheap elsewhere; in this one 16-step bf16
  Pallas loop they were not. Inside such a loop, try exact GELU first;
  under XLA, where most GELUs run, it was not measured.
- **Keep the division in the loop.** Normalising each tile by a
  loop-invariant row denominator, hoisting `lax.reciprocal` out of the loop
  and multiplying kept 0.935 of the speed of
  plain in-loop division,[^22] and reciprocal-then-multiply
  inside the loop kept 0.925.[^23]
  The experiment expected the hoist to win; it lost by about 6%. For a
  loop-invariant divisor, write plain division.
- **Masking with -inf costs nothing.** Filling masked positions with IEEE
  -inf, -1e30 or a clamp at -88 before `exp` made no measurable difference:
  -1e30 against -inf 1.01 at 4096
  rows[^24], -88 against -inf 1
  at 8192.[^25] The one exception, -1e30 at 8192 rows at
  0.955,[^26] is small and in the
  direction of -inf being faster. In these cells there was no slow path for
  -inf.

### Replace data-dependent selects with arithmetic on packed words

The same kind of loop measured two rewrites that change how many vector
instructions an element needs, and both paid off.

- **NF4 quantisation by binary search.** Finding each value's nearest of the
  16 NF4 centroids with a 4-level binary search over the midpoints was 2.53× faster than a linear scan of all
  16 distances.[^27] The linear scan in a Pallas kernel was
  itself far ahead of XLA's broadcast argmin over the same 16 centroids,
  which kept only 0.274 of its
  speed.[^28]
- **Sort keys and indices as one 32-bit word.** Packing a bf16 key and a
  16-bit index into one integer and sorting with plain max and min was
  2.82× faster at 2048 rows[^29]
  and 2.99× at 4096[^30] than
  keeping separate key and index arrays with masked selects. Both of those
  ran in VMEM; doing the split sort as passes through HBM was worse still,
  keeping 0.239 of the VMEM split sort's
  speed[^31] (a naive baseline).
- **How int4 nibbles are packed barely matters.** In a fused int4-weight
  GEMM, splitting each byte's nibbles across the two 128-lane halves was 1.03× faster than even/odd
  interleaving,[^32] about 3%. Unpacking inside the GEMM
  at all is what matters: dequantising the weights to HBM in a separate
  pass first kept 0.835 of the fused kernel's speed
  at 2048 rows[^33] and 0.93 at
  4096.[^34]

### Find where a streaming kernel stops being memory-bound

A Pallas kernel streamed a 64 MiB fp32 array through VMEM, reading and writing each element once, and applied k dependent multiply-add steps to
every element. The arms differ in k, so they do different amounts of
arithmetic on the same bytes; the point is to see where extra arithmetic
starts to cost time.

- **Four times the steps, from 4 to 16**, kept
  0.584 of the speed:[^35] slower,
  but by less than the extra arithmetic, so the knee where VPU time overtakes
  HBM time lies in this range.
- **From 16 to 64 steps**, the kernel kept
  0.265 of the speed,[^36] and
  **from 64 to 256 steps**, 0.254.[^37]
  Above the knee, time grows in direct proportion to the arithmetic: the
  kernel is VPU-bound.
- Going from one step to four produced no stored ratio, so it is not
  quoted here.

In practice: a streaming fp32 kernel can do a handful of dependent
multiply-adds per element for nothing beyond the memory traffic; by a few
dozen, the vector unit sets the pace. Fusing light elementwise work into a
memory-bound pass is free; fusing heavy work is not.

### Strided and scattered reads cost nothing extra

- **Strided blocks in Pallas.** A Pallas copy read a fixed number of blocks
  from a large bf16 array, either contiguous blocks or every 8th or 64th
  block (via the BlockSpec `index_map`). With 4 KiB blocks, every 8th block
  gave 1[^38] and every 64th
  1;[^39] with 64 KiB blocks,
  0.999[^40] and
  0.996.[^41] Skipping whole
  blocks in HBM is free at these block sizes; lay out your BlockSpec for the
  computation, not for contiguity of the blocks.
- **Gather order.** `torch.index_select` read the same set of rows from a
  bf16 table, once with the indices sorted and once shuffled, at row widths
  from 16 bytes to 4 KiB. Shuffling made no difference at any width
  (overall ratio 1[^42];
  the per-width ratios ranged from 0.999[^43]
  to 1.01[^44]). Don't sort gather indices for
  speed.
- **Gathering rows did not beat gathering columns.** Gathering 1024-wide embedding rows as contiguous blocks with a Pallas kernel driven by SMEM
  indices was expected to beat gathering columns from a transposed table and
  transposing back. It was slower: 0.869 of the
  column gather's speed for the smaller table[^45] and
  0.81 for the larger.[^46] That
  prediction was refuted. XLA's own row gather matched the column gather on
  the smaller table (1[^47]) and
  was 1.57× faster on the
  larger:[^48] for a plain embedding lookup, let XLA do it.

### Reduce along whichever axis suits you

A Pallas kernel reduced each fp32 block either across its rows or along
its lanes (the last axis), with outputs shaped so both arms write the same
bytes; sum and max were both run and are pooled per block shape.

- With 64 by 1024 blocks, reducing along lanes kept
  0.959 of the speed[^49] of
  reducing across rows.
- With 128 by 2048 blocks, the two were the same, ratio
  0.997.[^50]

The lane reduction's small penalty appears only on the smallest block and
disappears as blocks grow. The reduction is memory-bound either way.

### Random numbers cost far more than a plain write

`torch.rand` was timed against `torch.full` filling a tensor of the same
size, so both allocate and write the same bytes and the ratio isolates the
cost of generating the randomness. Six sizes ran, from 256 KiB to 256 MiB of fp32.

- At the four smaller sizes, random generation kept about half the speed
  of a constant fill: 0.493 at the
  smallest[^51] and 0.492 at the
  fourth.[^52]
- At the two largest sizes it kept
  0.31[^53] and
  0.31[^54] of the speed.
  Once both arms are saturated, random generation is limited by its own
  compute and falls further behind a store at memory speed; the two largest
  sizes gave the same ratio.

This is not a choice between the two (a constant is not a substitute for
noise), but it tells you what dropout masks and random initialisation cost.
Large random tensors are not free relative to the memory they occupy.

### Short or odd batch dimensions in a matmul barely matter

Two experiments held the arithmetic fixed and changed only how the rows were
grouped into a leading batch dimension, in one bf16 `torch.matmul`.

- **A tall matmul against a batch of short ones.** 2048 rows by 1024 times
  1024 by 1024, either as one matmul or reshaped into a batch of matmuls of
  height Mb. At Mb = 8 the batched form kept
  0.961 of the speed;[^55] from
  Mb = 32 upward it made no difference, for example
  1 at 32,[^56]
  0.997 at 128[^57] and
  0.995 at 1024.[^58] The
  experiment expected a small penalty at every height and saw none; the
  compiler appears to fold the batch back into the rows.
- **Blocking the rows of a larger matmul.** 2048 rows, contraction 4096,
  weight width 4096 or 8192, with the rows grouped into blocks of 1 to 2048.
  With the wider weight, every block size from 8 to 2048 rows matched one row
  per block: 1 at
  8,[^59] 0.998 at
  128[^60] and 0.998 at
  2048.[^61]
- **The one exception is two rows per block.** It kept
  0.855 of the speed with the 4096-wide
  weight[^62] and 0.922 with the
  8192-wide one[^63] (0.889
  on average[^64]). That is two cells and the cause is
  unknown; if your data arrives with a leading dimension of two, try
  folding the pairs into the rows before the matmul.

For this one `torch.matmul` shape family, padding or regrouping the batch
dimension to "fill the MXU" bought nothing.

### Capping scoped VMEM low slows a matmul several-fold

`--xla_tpu_scoped_vmem_limit_kib` caps how much VMEM the compiler may use
for a kernel's working tiles. Three experiments ran bf16 matmuls with the
compiler's default against a cap, each arm in its own process so libtpu
read its own flag, and both arms stamped into one sitting so they pair.

- **At a 4096-cube**, a 4 MiB cap kept 0.251 of the default's speed,[^65]
  and a 16 MiB cap 0.661.[^66] The experiment expected
  the larger cap to cost little; it cost a third of the speed. A 128 MiB
  cap was also run and produced no stored ratio.
- **Across square sizes**, the 4 MiB cap kept
  0.243 of the speed at
  2048[^67] and 0.302 at
  16384,[^68] with 4096 and 8192 in between
  (0.251[^69] and
  0.272[^70]): the cap costs three to four times the
  time throughout, easing slowly as the matmul grows.
- **Holding M and N at 8192 and varying the contraction depth K** from 128
  to 8192 moved the ratio by more than a factor of two: the capped arm kept
  0.434 of the speed at
  K = 128,[^71] 0.198 at
  512,[^72] 0.211 at
  2048[^73] and 0.271 at
  8192.[^74] The experiment expected the penalty to
  track how much operand reuse a shape offers; it did not.

In practice: capping scoped VMEM at 16 MiB or less cost 1.5 to 4 times the
time on these matmuls. Values between 16 MiB and the compiler's default
were not measured, so whether a cap slightly below the default can help a
particular shape is open. If you cap it (for example to leave VMEM to a
Pallas kernel), measure the matmuls around it.


## Believed, not measured

- The add, multiply, divide and relu chains cost nothing because XLA
  rewrites them into a single operation. This is a reading of the depth
  sweep; the compiled program was not inspected.
- The jump in transcendental cost between the two chain depths is believed
  to be the point where XLA stops fusing the chain into one loop and starts
  writing each intermediate to HBM. Where exactly that happens, and whether
  breaking a long chain into shorter fused pieces avoids it, is not known.
- The flat results for strided blocks and shuffled gathers suggest that the
  v6e memory system keeps enough requests in flight that the order of reads
  does not matter at these granularities, or that these operations are
  bound by per-call overhead or bandwidth rather than by locality.
- The fall in the random-number ratio at large sizes is consistent with the
  generator being arithmetic on the vector unit, competing with the store
  rather than running in a dedicated unit.
- Why a low cap costs what it does (a fixed tile the compiler falls back
  to, or a schedule that stops overlapping copies with compute) is not
  established.
- Why exact erf GELU beats its approximations, and why hoisting a
  reciprocal loses to in-loop division, is not established. The likely
  reading is that Mosaic lowers erf and division to sequences the SFU
  pipelines well, while the approximations add dependent VPU work and the
  hoisted (rows, 1) reciprocal costs a broadcast every step; none of this
  was profiled.
- The packed-word sort and the binary-search quantiser win by replacing
  masked selects with fewer, simpler vector instructions. Instruction
  counts were not measured.
- The batched and blocked matmuls cost the same as the tall matmul because
  XLA folds a contiguous leading dimension into the row dimension and emits
  one GEMM. Why a leading dimension of exactly two is slower, and why the
  smallest batch height is slightly slower, is not established; both are
  the only cases where the leading dimension is neither one nor a multiple
  of eight, or where the per-matmul height is very short.

## Never measured

- More than one chip, and chips other than the v6e.
- exp and log chains, which were excluded because they cannot be chained
  without rescaling; and bf16 elementwise chains (all chains were fp32).
- The chain depth at which fusion breaks, between the two depths tried.
- The roofline kernel in bf16, with other block sizes, or with
  independent rather than dependent steps.
- Strided reads with the largest block size planned, and strides other
  than every 8th and 64th block; gathers from tables larger than the one
  used, or along an axis other than the first.
- Reductions over the largest block shape planned, and reductions over
  more than one axis at once.
- Random numbers in bf16, normal rather than uniform samples, and random
  generation inside a Pallas kernel.
- Batch heights between the smallest and 32 in the tall-against-batched
  comparison, block sizes other than two that are not multiples of eight in
  the blocking comparison, and any of this through a Pallas kernel rather
  than XLA.
- Scoped-VMEM caps on anything but a single bf16 matmul (attention,
  convolutions, Pallas kernels that set their own VMEM limit), caps between
  16 MiB and the default, caps other than 4 MiB across shapes, and the
  smallest square sizes.
- More than two accumulators in the interleaving experiment, or bf16 tiles.
- The exp2, GELU, division and mask comparisons outside a Pallas loop (under
  XLA or `torch.compile`), in fp32 tiles, or at loop lengths other than 16.
- Other quantisation formats than NF4, sorts over other key widths, and
  int4 GEMMs at other shapes.
- End-to-end model throughput: these are single-operation timings.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.
[^1]: divide chain against add chain, depth 1024: ratio 0.997 [0.992, 1.01], 2 sessions; S = 1,048,580; depth = 1024.
[^2]: tanh chain against add chain, depth 256: ratio 0.151 [0.149, 0.154], 2 sessions; S = 1,048,580; depth = 256.
[^3]: tanh chain against add chain, depth 1024: ratio 0.00365 [0.0036, 0.0037], 2 sessions; S = 1,048,580; depth = 1024.
[^4]: erf chain against add chain, depth 256: ratio 0.151 [0.149, 0.154], 2 sessions; S = 1,048,580; depth = 256.
[^5]: erf chain against add chain, depth 1024: ratio 0.00365 [0.0036, 0.0037], 2 sessions; S = 1,048,580; depth = 1024.
[^6]: rsqrt chain against add chain, depth 256: ratio 0.151 [0.149, 0.154], 2 sessions; S = 1,048,580; depth = 256.
[^7]: rsqrt chain against add chain, depth 1024: ratio 0.00365 [0.0036, 0.0037], 2 sessions; S = 1,048,580; depth = 1024.
[^8]: sqrt chain against add chain, depth 256: ratio 0.0112 [0.011, 0.0114], 2 sessions; S = 1,048,580; depth = 256.
[^9]: sqrt chain against add chain, depth 1024: ratio 0.0028 [0.0028, 0.0028], 2 sessions; S = 1,048,580; depth = 1024.
[^10]: sigmoid chain against add chain, depth 256: ratio 0.0063 [0.0062, 0.0064], 2 sessions; S = 1,048,580; depth = 256.
[^11]: sigmoid chain against add chain, depth 1024: ratio 0.0016 [0.0016, 0.0016], 2 sessions; S = 1,048,580; depth = 1024.
[^12]: dependent order against round-robin, 262144 elements per tensor: ratio 1 [0.996, 1.02], 2 sessions; S = 262144; per_stream = 16; streams = 4.
[^13]: dependent order against round-robin, 1048576 elements per tensor: ratio 1 [0.999, 1], 2 sessions; S = 1,048,580; per_stream = 16; streams = 4.
[^14]: dependent order against round-robin, 4194304 elements per tensor: ratio 1 [0.999, 1], 2 sessions; S = 4,194,300; per_stream = 16; streams = 4.
[^15]: interleaved accumulators against dependent loops, 256 iterations: ratio 0.944 [0.944, 0.945], 2 sessions; B = 8; ITER = 256; M = 128; N = 128.
[^16]: interleaved accumulators against dependent loops, 512 iterations: ratio 0.943 [0.943, 0.943], 2 sessions; B = 8; ITER = 512; M = 128; N = 128.
[^17]: exp2 with folded constants against exp, 4096 rows: ratio 1.19 [1.19, 1.19], 2 sessions; M = 4096; N = 1024; S = 16.
[^18]: exp2 with folded constants against exp, 8192 rows: ratio 1.19 [1.19, 1.19], 2 sessions; M = 8192; N = 1024; S = 16.
[^19]: explicit log2(e) multiply then exp2 against exp, both sizes: median ratio 1.01, range [1.01, 1.01], over 2 cells, 2 sessions each.
[^20]: sigmoid GELU against exact erf GELU, both sizes: median ratio 0.841, cells 0.841 to 0.842, over 2 cells, 2 sessions each.
[^21]: tanh GELU against exact erf GELU, both sizes: median ratio 0.676, cells 0.675 to 0.677, over 2 cells, 2 sessions each.
[^22]: hoisted reciprocal against in-loop division, both sizes: median ratio 0.935, cells 0.934 to 0.936, over 2 cells, 2 sessions each.
[^23]: in-loop reciprocal and multiply against in-loop division, both sizes: median ratio 0.925, range [0.924, 0.926], over 2 cells, 2 sessions each.
[^24]: -1e30 against -inf, 4096 rows: ratio 1.01 [0.998, 1.01], 2 sessions; BM = 256; BN = 256; M = 4096; N = 256; STEPS = 16.
[^25]: clamp at -88 against -inf, 8192 rows: ratio 1 [0.998, 1], 2 sessions; BM = 256; BN = 256; M = 8192; N = 256; STEPS = 16.
[^26]: -1e30 against -inf, 8192 rows: ratio 0.955 [0.955, 0.955], 2 sessions; BM = 256; BN = 256; M = 8192; N = 256; STEPS = 16.
[^27]: binary search against linear scan, both sizes: median ratio 2.53, cells 2.52 to 2.53, over 2 cells, 2 sessions each.
[^28]: XLA broadcast argmin against linear scan, both sizes: median ratio 0.274, cells 0.272 to 0.276, over 2 cells, 2 sessions each.
[^29]: packed word sort against split arrays, 2048 rows: ratio 2.82 [2.79, 2.86], 2 sessions; BM = 256; BN = 256; M = 2048; N = 1024; STEPS = 16.
[^30]: packed word sort against split arrays, 4096 rows: ratio 2.99 [2.99, 2.99], 2 sessions; BM = 256; BN = 256; M = 4096; N = 1024; STEPS = 16.
[^31]: split sort through HBM against split sort in VMEM, both sizes: median ratio 0.239, cells 0.238 to 0.24, over 2 cells, 2 sessions each.
[^32]: split-half nibbles against interleaved, both sizes: median ratio 1.03, cells 1.03 to 1.03, over 2 cells, 2 sessions each.
[^33]: two-pass HBM dequant against fused interleaved unpack, 2048 rows: ratio 0.835 [0.83, 0.841], 2 sessions; K = 2048; M = 2048; N = 2048.
[^34]: two-pass HBM dequant against fused interleaved unpack, 4096 rows: ratio 0.93 [0.924, 0.937], 2 sessions; K = 2048; M = 4096; N = 2048.
[^35]: 16 multiply-add steps against 4: ratio 0.584 [0.577, 0.592], 2 sessions; block_rows = 2048; k_hi = 16; k_lo = 4; lanes = 128; rows = 131072.
[^36]: 64 multiply-add steps against 16: ratio 0.265 [0.265, 0.265], 2 sessions; block_rows = 2048; k_hi = 64; k_lo = 16; lanes = 128; rows = 131072.
[^37]: 256 multiply-add steps against 64: ratio 0.254 [0.254, 0.254], 2 sessions; block_rows = 2048; k_hi = 256; k_lo = 64; lanes = 128; rows = 131072.
[^38]: every 8th block against contiguous, smaller blocks: ratio 1 [0.999, 1], 2 sessions; block_rows = 16; blocks = 8192; cols = 128; rows_in = 8,388,610.
[^39]: every 64th block against contiguous, smaller blocks: ratio 1 [0.999, 1], 2 sessions; block_rows = 16; blocks = 8192; cols = 128; rows_in = 8,388,610.
[^40]: every 8th block against contiguous, larger blocks: ratio 0.999 [0.998, 0.999], 2 sessions; block_rows = 128; blocks = 512; cols = 256; rows_in = 4,194,300.
[^41]: every 64th block against contiguous, larger blocks: ratio 0.996 [0.996, 0.997], 2 sessions; block_rows = 128; blocks = 512; cols = 256; rows_in = 4,194,300.
[^42]: shuffled against sorted indices, all row widths: median ratio 1, cells 0.999 to 1.01, over 9 cells, 2 sessions each.
[^43]: shuffled against sorted indices, 32 columns: ratio 0.999 [0.998, 1], 2 sessions; C = 32; R = 8192; SRC = 32768.
[^44]: shuffled against sorted indices, 256 columns: ratio 1.01 [1.01, 1.01], 2 sessions; C = 256; R = 8192; SRC = 32768.
[^45]: Pallas row gather against column gather, smaller table: ratio 0.869 [0.83, 0.908], 2 sessions; D = 1024; G = 1024; N = 4096.
[^46]: Pallas row gather against column gather, larger table: ratio 0.81 [0.81, 0.81], 2 sessions; D = 1024; G = 2048; N = 8192.
[^47]: XLA row gather against column gather, smaller table: ratio 1 [0.995, 1.01], 2 sessions; D = 1024; G = 1024; N = 4096.
[^48]: XLA row gather against column gather, larger table: ratio 1.57 [1.57, 1.57], 2 sessions; D = 1024; G = 2048; N = 8192.
[^49]: lanes against rows, 64 by 1024 blocks: ratio 0.959 [0.957, 0.961], 2 sessions; C = 1024; R = 64; blocks = 256.
[^50]: lanes against rows, 128 by 2048 blocks: ratio 0.997 [0.986, 1], 2 sessions; C = 2048; R = 128; blocks = 64.
[^51]: torch.rand against torch.full, 65536 elements: ratio 0.493 [0.489, 0.496], 2 sessions; N = 65536.
[^52]: torch.rand against torch.full, 4194304 elements: ratio 0.492 [0.492, 0.493], 2 sessions; N = 4,194,300.
[^53]: torch.rand against torch.full, 16777216 elements: ratio 0.31 [0.309, 0.311], 2 sessions; N = 16,777,200.
[^54]: torch.rand against torch.full, 67108864 elements: ratio 0.31 [0.309, 0.311], 2 sessions; N = 67,108,900.
[^55]: batched against tall matmul, Mb = 8: ratio 0.961 [0.958, 0.963], 2 sessions; K = 1024; M = 2048; Mb = 8; N = 1024.
[^56]: batched against tall matmul, Mb = 32: ratio 1 [0.992, 1.01], 2 sessions; K = 1024; M = 2048; Mb = 32; N = 1024.
[^57]: batched against tall matmul, Mb = 128: ratio 0.997 [0.987, 1], 2 sessions; K = 1024; M = 2048; Mb = 128; N = 1024.
[^58]: batched against tall matmul, Mb = 1024: ratio 0.995 [0.987, 1], 2 sessions; K = 1024; M = 2048; Mb = 1024; N = 1024.
[^59]: 8 rows per block against 1, weight width 8192: ratio 1 [0.993, 1], 2 sessions; D = 8192; N = 2048; S = 4096.
[^60]: 128 rows per block against 1, weight width 8192: ratio 0.998 [0.993, 1], 2 sessions; D = 8192; N = 2048; S = 4096.
[^61]: 2048 rows per block against 1, weight width 8192: ratio 0.998 [0.993, 1], 2 sessions; D = 8192; N = 2048; S = 4096.
[^62]: 2 rows per block against 1, weight width 4096: ratio 0.855 [0.854, 0.857], 2 sessions; D = 4096; N = 2048; S = 4096.
[^63]: 2 rows per block against 1, weight width 8192: ratio 0.922 [0.922, 0.922], 2 sessions; D = 8192; N = 2048; S = 4096.
[^64]: 2 rows per block against 1, both weight widths: median ratio 0.889, cells 0.855 to 0.922, over 2 cells, 2 sessions each.
[^65]: 4 MiB cap against the default, 4096-cube: ratio 0.251 [0.247, 0.256], 2 sessions; K = 4096; M = 4096; N = 4096.
[^66]: 16 MiB cap against the default, 4096-cube: ratio 0.661 [0.656, 0.666], 2 sessions; K = 4096; M = 4096; N = 4096.
[^67]: 4 MiB cap against the default, 2048-cube: ratio 0.243 [0.237, 0.248], 2 sessions; K = 2048; M = 2048; N = 2048.
[^68]: 4 MiB cap against the default, 16384-cube: ratio 0.302 [0.295, 0.308], 2 sessions; K = 16384; M = 16384; N = 16384.
[^69]: 4 MiB cap against the default, 4096-cube: ratio 0.251 [0.246, 0.255], 2 sessions; K = 4096; M = 4096; N = 4096.
[^70]: 4 MiB cap against the default, 8192-cube: ratio 0.272 [0.265, 0.278], 2 sessions; K = 8192; M = 8192; N = 8192.
[^71]: 4 MiB cap against the default, M = N = 8192, K = 128: ratio 0.434 [0.428, 0.441], 2 sessions; K = 128; M = 8192; N = 8192.
[^72]: 4 MiB cap against the default, M = N = 8192, K = 512: ratio 0.198 [0.196, 0.201], 2 sessions; K = 512; M = 8192; N = 8192.
[^73]: 4 MiB cap against the default, M = N = 8192, K = 2048: ratio 0.211 [0.207, 0.215], 2 sessions; K = 2048; M = 8192; N = 8192.
[^74]: 4 MiB cap against the default, M = N = 8192, K = 8192: ratio 0.271 [0.267, 0.275], 2 sessions; K = 8192; M = 8192; N = 8192.
