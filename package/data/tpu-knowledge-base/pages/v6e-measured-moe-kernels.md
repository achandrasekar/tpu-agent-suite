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
entry: v6e-measured-moe-kernels
keyword: v6e-measured-moe-kernels
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My MoE layer on TPU v6e is slow: how to run the expert matmuls, gather rows, use quantized weights and pick the top-k (forward pass)

You are running a mixture-of-experts layer on a TPU v6e and its forward pass
or decode step is slower than the arithmetic says it should be. Everything
here is a forward-pass or decode timing; no backward pass was measured. Perhaps
the expert matmuls are padded to a fixed capacity, the decode step reads
every expert's weights to use a few of them, the expert weights are stored
in int8 or int4, the batch carries padding for variable-length sequences,
or the router's top-k is a full sort. This page collects measurements of
those choices on one v6e chip, plus the row gathers and
quantized-weight matmuls they are built from. Every experiment ran in a
single process on one v6e chip. The kernels were written in JAX Pallas and
called from PyTorch through torch_tpu's `pallas.jax_op`; most sections also
time the same computation left to XLA. For dense matmul dtypes and
quantization see `index.py get v6e-measured-matmul-quantization`; for general Pallas data
movement and padding see `index.py get v6e-measured-pallas-kernels`; for sampling and other
layer ops see `index.py get v6e-measured-model-ops`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process, in two separate sessions. Above one, the alternative is faster by
that factor. Below one, the alternative is slower, and the number is the
share of the baseline's speed it keeps. Each experiment checked every arm's
output against a full-precision reference before timing it, so the arms
being compared compute the same result. Several experiments were planned
with shapes or arms that differ from what ran; the shapes below are the ones
that ran.

## Measured

### Run routed experts as a ragged grouped matmul, not a padded batch

The workload was 8192 routed tokens over 32 experts, contraction depth
2048, output width 2048 or 4096, bf16. The baseline gathered tokens into a
padded three-dimensional batch with a capacity factor of 3.0 (room for
three times the average tokens per expert) and ran one batched matmul in
XLA.

- **A Pallas ragged grouped matmul** that reads expert boundaries from SMEM
  and computes only the real tokens is
  2.79× faster[^1]:
  2.33× at output width
  2048[^2] and 3.25× at
  4096.[^3]
- **Lowering the capacity factor** from 3.0 to 2.0 in the padded batch is
  1.39× faster.[^4]
  This is the cost of padding, not a free choice: a smaller capacity factor
  drops tokens once routing is uneven enough, and token dropping was not
  measured here.

The gain from going ragged roughly tracks the capacity factor it removes:
the baseline here computed three times the real tokens, and the best case
(3.25×) slightly exceeds that because the padded batch also pays to gather
tokens into it. Against a capacity factor near 1.25, expect much less. A
ragged grouped matmul has static shapes (the token buffer is fixed and the
group sizes are data), so it does not cause the recompilation that
dynamically sized per-expert slices would.

The more skewed the routing, the more padding costs. With 4096 tokens over
16 experts (model and output width 1024), padding every expert to the
busiest one's token count and running one `jnp.einsum`, against a Pallas
ragged grouped matmul over the packed tokens:
3.02× faster on
average,[^5] 2.47× when the
busiest expert got 1024 tokens[^6] and
3.57× when it got 2048.[^7]
The padded baseline here is the common default, not a strawman, so this is
the gain to expect from going ragged under skewed routing.

### Match the ragged matmul's row tile to the tokens each expert gets

The same kind of ragged grouped matmul ran over 64 experts, 8192 tokens,
output width 2048, contraction depth 2048 or 4096, with row tiles (tm) of
128, 256 or 512 tokens. Two routings were tested, each against its own
baseline. Every expert's token count is padded up to a whole number of row
tiles.

- **Uniform routing, 128 tokens per expert.** A tile larger than an
  expert's tokens only adds padded rows. Against a tile of 512, a tile of
  256 is 1.2×
  faster[^8] and a tile of 128 is
  1.42× faster.[^9]
  Both gains are well below the share of padded rows removed.
- **Skewed routing, eight hot experts with 1024 tokens each** and the other
  experts empty. Here every tile is full and a larger tile means fewer
  weight reloads. Against a tile of 128, a tile of 256 is
  1.73× faster[^10] and
  a tile of 512 is 2.84×
  faster.[^11]

Both results held at both contraction depths. Pick the tile from the
tokens per expert you actually see: no larger than a typical expert's
count, and as large as that allows. These are two routings at one shape
family; routings in between were not measured.

### Pad small expert bins to 128 rows, not 256

Eight experts, contraction depth and output width 2048, bf16, with 48 or
80 tokens per expert, so every expert fits in 128 rows. The expectation going
in was that padding each expert to 128 rows or to the MXU's native 256 would
cost the same, since both read the same weights in one pass.
Padding to 128 was 1.32× faster than 256
on average:[^12] 1.31× at 48
tokens[^13] and 1.32× at
80.[^14] That expectation was wrong on the fast side: the
gain was larger than the 1.25× it allowed for. Padding to 512 rows in two
passes was worse: 256 rows took 0.527 of
its time on average[^15] (0.527
at 48 tokens[^16] and 0.526 at
80[^17]). At this shape, when experts get fewer than 128 tokens, pad to
128, as the row-tile result above also says. Other expert counts and widths
were not measured.

### Prefer fewer, wider experts when the total width is fixed

At a fixed total intermediate width of 16384 and contraction depth 2048,
256 or 512 tokens were projected through the width split into 128 experts
of width 128, or into fewer, wider experts. Every arm did the same number
of FLOPs. Against 128 experts of width 128:

| experts × width | overall | 256 tokens | 512 tokens |
|---|---|---|---|
| 64 × 256 | 1.55×[^18] | 1.49×[^19] | 1.61×[^20] |
| 32 × 512 | 2.15×[^21] | 1.96×[^22] | 2.34×[^23] |
| 8 × 2048 | 2.99×[^24] | 2.51×[^25] | 3.47×[^26] |

Every step wider was faster, and the gap grew with more tokens. An expert
narrower than the MXU's 256 columns is the most expensive case. If your
model design allows it, give experts at least 256 columns; this measures
kernel cost only, not model quality.

### In decode, stream only the experts the batch touches

A decode step routed 16 token-expert pairs over 64 experts (contraction and
output width 2048, bf16), touching 8 or 16 distinct experts. The baseline,
in XLA, ran every expert and masked the unused results.

- **A Pallas kernel that reads the touched expert ids from SMEM** and
  streams only those experts' weights is
  6.09× faster on
  average[^27]: 8.1× with 8
  experts touched[^28] and 4.09×
  with 16.[^29] The gain tracks the share of experts the
  batch skips.
- **Gathering the touched experts' weights in XLA** first and then running
  the matmul is slower than the masked baseline: it keeps
  0.383 of its speed on
  average,[^30] 0.423 with 8
  touched[^31] and 0.342 with
  16.[^32] Do not materialise a gathered copy of the
  weights; index them in the kernel.

### Quantized expert weights help less than the byte count says

The expert projection had 8 experts, contraction and output width 2048, and
128 or 256 tokens per expert. All three arms were Pallas kernels over the
same underlying weights. Against bf16 weights:

- **int8 weights (W8A16)**, dequantized in VMEM:
  1.15× faster,[^33]
  1.17× at 128 tokens per
  expert[^34] and 1.13× at
  256.[^35]
- **packed int4 weights (W4A16)**, unpacked with shifts in VMEM:
  1.22× faster,[^36]
  1.23×[^37] and
  1.22×[^38] at the same two
  sizes.

Both were expected to gain far more from halving or quartering the weight
bytes, and neither did. In both kernels the MXU still runs bf16 (v6e
publishes no int4 rate, and these are weight-only kernels), so the dequant
or unpack work runs on the VPU and only the weight bytes shrink. At this
shape (8 experts, 2048 wide, 128 to 256 tokens per expert), quantizing
expert weights saved memory and bought a modest speedup; other shapes,
especially fewer tokens per expert, were not measured.

### If weights are int8, dequantize inside the matmul kernel, not in a pass before it

A W8A16 projection (512 tokens, contraction 4096, output width 4096 or 8192,
int8 weights with a per-channel scale) was run with a baseline that
dequantized the whole weight to bf16 in HBM and then multiplied. That
baseline is deliberately poor: it writes and rereads a full bf16 copy of the
weight, so the ratios below are the cost of a separate dequantization pass,
not a speedup over a well-written int8 kernel.

- **A single Pallas kernel that dequantizes each int8 tile on the VPU** while
  the MXU works on the previous one is 2.4×
  faster:[^39] 1.97× at width
  4096[^40] and 2.83× at
  8192.[^41]
- **A plain Pallas bf16 matmul over weights already stored in bf16** (twice
  the weight memory, no dequantization at all) is
  2.08× faster than the same
  baseline:[^42] 2.02× at
  4096[^43] and 2.15× at
  8192.[^44] This tells you the
  two-pass dequantization is the problem, not whether to quantize. Against
  the same baseline the fused int8 kernel gained about as much as the bf16
  matmul at width 4096 and more at 8192, but the two were not timed
  against each other directly.

### Fuse dynamic activation quantization into the int8 matmul's prologue

A dynamically quantized linear layer (contraction and output width 4096,
2048 or 4096 tokens, 256 by 512 tiles) needs a per-tile maximum and an int8
cast of the activations before the int8 matmul. The baseline did the
maximum and the cast as separate passes through HBM. Doing both inside the
Pallas kernel's prologue, in VMEM, is
1.43× faster on
average[^45] (1.45× at 2048 tokens, range 1.15× to
1.75×;[^46] 1.41× at 4096, range 1.11× to 1.71×[^47]). The range is
wide because the unfused baseline's time varied a lot between the two
sessions; the fused kernel's did not. Read it as "faster, by somewhere
between about 1.1× and 1.75×". The gain expected going in was larger and did
not hold.

### A grouped kernel inside a torch module, against a per-expert loop

A `torch.nn.Module` SwiGLU MoE layer (8 experts, 256 tokens each, model
width 1024, intermediate width 512 or 1024) replaced a Python loop that
dispatched each expert separately and stacked the results with one Pallas
grouped SwiGLU kernel. That kernel is
22× faster[^48]
(22×[^49] and
22.1×[^50] at the two widths).
The baseline is naive: one dispatch per expert across the torch_tpu
boundary.

A follow-up ran the same module against the standard alternative, one XLA
batched einsum over the expert dimension, as well as the loop:

- **Against the per-expert loop** the grouped kernel was again
  19.3× faster[^51]
  (20.7× at intermediate width
  512[^52] and 17.9× at
  1024).[^53]
- **Against the XLA batched einsum** it was
  1.34× faster at intermediate width
  1024;[^54] at 512 the two were within noise of each
  other. The expectation going in was that they would be within about 5%.

So with uniform routing, a single batched einsum already removes nearly
all of the loop's cost; the grouped kernel adds a third at the wider
expert and nothing measurable at the narrower one. The likely source of that
third is fusion rather than grouping: the kernel computes gate, up, SiLU and
down projections in one pass, while the einsum path writes the intermediate
activations to HBM and reads them back. Grouping alone was not isolated, and
a plain grouped matmul over uniform groups is not expected to beat the dense
einsum.

### Gathering rows: one-hot matmul or DMA, and how many rows per DMA

Expert dispatch, embedding lookups and paged caches all gather rows from a
table. Two experiments timed how.

**One-hot matmul against indirect DMA.** Rows were gathered from a table of
32768 rows of width 1024 or 2048 (bf16), either as a one-hot matrix times
the table in XLA or by a Pallas kernel issuing one DMA per row with indices
in SMEM.

- For 16 rows, the DMA kernel is 2.52×
  faster on average:[^55] 1.63×
  at width 1024[^56] and 3.41×
  at 2048.[^57] The one-hot matmul reads the whole table.
- For 1024 rows at width 1024, the one-row-per-DMA kernel keeps only
  0.362 of the one-hot matmul's
  speed.[^58] At width 2048 the two ran at about the same
  speed, and no ratio was recorded for that cell.

So a one-row-per-descriptor DMA gather wins for a handful of rows and loses
somewhere below a thousand rows.

**More rows per DMA descriptor.** Gathering 512 or 1024 rows from a table of
4096 rows of width 4096 (bf16) in Pallas, with indices in SMEM, grouping
rows into one descriptor beat one row per descriptor. The baseline issued one
8 KiB row per grid step, so most of the gain is the per-step overhead it
amortises, and one descriptor moves one contiguous block, so grouping needs
the indices to form contiguous runs:

- **8 rows per descriptor**: 5.6×
  faster on average,[^59] 4.45×
  for 512 rows[^60] and 6.74×
  for 1024.[^61]
- **16 rows per descriptor**: 6.67× on
  average,[^62] 4.47× for 512
  rows[^63] and 8.87× for
  1024.[^64]

At 512 rows the two bucket sizes are the same, so something other than
descriptor count limits the kernel there; at 1024 rows 16 per descriptor
pulls ahead. If the indices allow it (sorted or bucketed tokens, contiguous
pages), move several rows per descriptor. Nothing coarser than 16 rows was
tested, and this was a pure gather; in a kernel that overlaps its copies with
compute, coarser descriptors can cost overlap.

**A small page pool: stage it in VMEM rather than prefetching pages.**
Gathering and summing 16 or 32 active pages per sequence (16 sequences)
from a pool of 128 pages of 128 by 128 fp32 values, a Pallas kernel that
looked up the page table in SMEM and streamed only the active pages by
asynchronous DMA was slower than one that copied the whole pool into VMEM
and walked it with `lax.fori_loop`: it kept
0.283 of its speed on
average,[^65] 0.375 with 16
active pages[^66] and 0.191
with 32.[^67] The prediction was a 2× win for the
prefetch pipeline, and it was refuted. The whole pool is 8 MiB and fits in
VMEM, which is what makes staging it possible; for a pool too large for
VMEM this result says nothing, and per-page DMA may be the only choice.

**Skipping inactive blocks with `pl.when`: no gain here.** A sparse block
projection over 32 or 64 blocks of 256 by 256 fp32 values, a quarter of
them active, with four matmuls per active block. Reading the active flags
from SMEM and skipping inactive blocks with `@pl.when`, against computing
every block and masking with `jnp.where`, was
1.07× faster, but with an range from 0.996× to 1.09×, so no
measurable gain;[^68] skipping three quarters of the matmuls was expected to
give at least 1.26×. Do not expect `pl.when` to recover the skipped
arithmetic at this block size and density: it skips the compute, but the
inactive blocks are still copied in.

### Pack variable-length sequences before the projection

A padded batch of 32 sequences up to 512 tokens long (width 2048, projected
to 4096, bf16) held 4096 or 8192 real tokens, a quarter or half of the
slots. Compacting the real tokens with a Pallas kernel driven by the
cumulative sequence lengths in SMEM, then running the matmul, is
3.86× faster on
average[^69] than the full padded matmul:
4.89× at a quarter
full[^70] and 2.83× at
half.[^71] The emptier the batch, the bigger the gain.

### Router top-k: never fully sort

Taking the top 16 of 16384 or 32768 float32 logits for 64 rows, the
baseline sorted each row in full.

- **XLA's `lax.top_k`** is 10.2× faster on
  average:[^72] 8.06× at 16384
  columns[^73] and 12.3× at
  32768.[^74]
- **A Pallas radix-select kernel** over sign-flipped integer keys in VMEM is
  11× faster on
  average:[^75] 8.86× at
  16384[^76] and 13.2× at
  32768.[^77]

Both beat the sort by roughly the same margin. The two were not timed
against each other, so `lax.top_k` is the simple default and the custom
kernel is not shown to be worth writing.

## Believed, not measured

- Most gains above line up with fewer bytes read from HBM: ragged rather
  than padded tokens, only the touched experts' weights, int8 rather than a
  dequantized copy, packed rather than padded sequences. Bandwidth was not
  measured directly.
- Quantized expert weights gained less than their byte savings, perhaps
  because at these token counts the expert matmuls are not purely limited
  by weight bytes, or because the in-VMEM unpack and dequant work on the
  VPU takes back part of the saving. Neither has been checked.
- Narrow experts likely lose because a tile narrower than the MXU's 256
  columns leaves part of the array idle and because tokens are re-read once
  per expert.
- In the skewed routing case a larger row tile probably wins because each
  tile reloads its expert's weights, so fewer tiles means fewer reloads.
- Gathering expert weights in XLA probably loses because it writes a full
  copy of the touched weights to HBM before reading it back.
- The one-hot gather probably wins at many rows because one large
  contiguous read and a matmul beat many small descriptor-bound DMAs.
- The page-prefetch pipeline probably lost because each page is its own
  small DMA with its own wait, while staging copies the pool in a few large
  transfers; and `pl.when` probably saved little because the grid still
  visits every block and the inactive blocks' inputs are still copied in.
  Neither has been profiled.
- The fused activation-quantization baseline's session-to-session spread
  suggests its separate passes are sensitive to something in the
  environment; the cause is unknown.

## Never measured

- More than one chip, so no expert parallelism, all-to-all dispatch or
  token exchange between chips.
- The backward pass of any MoE kernel. Every timing is a forward pass or a
  decode step, so "training" here means the forward expert matmuls only.
- Token dropping, load-balancing loss, or model quality under any of these
  choices, including capacity factors and expert widths.
- The Pallas grouped SwiGLU kernel against XLA under skewed routing, where
  a batched einsum would have to pad (only uniform routing was measured
  against XLA).
- A page pool too large for VMEM, and `pl.when` at other block sizes or
  densities.
- The fused int8 dequant kernel timed directly against the bf16 matmul, and
  the Pallas top-k kernel timed directly against `lax.top_k`.
- The exact crossover between one-hot and DMA gathers between the two row
  counts tested, and multi-row descriptors against the one-hot matmul.
- fp8 expert weights, int4 with group-wise scales in the MoE kernel, and
  dynamic activation quantization for the expert matmuls.
- Row tiles under routings between uniform and the eight-hot skew, and
  routings measured from a real model.
- End-to-end MoE layer or model throughput.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.10.1 to 0.11.2, libtpu 0.0.41 to 0.0.46, torch 2.10.0 to 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: Pallas ragged grouped matmul against padded BMM at capacity 3.0: median ratio 2.79, cells 2.33 to 3.25, over 2 cells, 2 sessions each.
[^2]: ragged against capacity 3.0, output width 2048: ratio 2.33 [2.14, 2.53], 2 sessions; E = 32; K = 2048; M = 8192; N = 2048.
[^3]: ragged against capacity 3.0, output width 4096: ratio 3.25 [3.07, 3.44], 2 sessions; E = 32; K = 2048; M = 8192; N = 4096.
[^4]: capacity 2.0 against 3.0: median ratio 1.39, range [1.37, 1.44], over 2 cells, 2 sessions each.
[^5]: ragged grouped matmul against max-capacity padded einsum: median ratio 3.02, cells 2.47 to 3.57, over 2 cells, 2 sessions each.
[^6]: busiest expert 1024 tokens: ratio 2.47 [2.46, 2.47], 2 sessions; C_max = 1024; D = 1024; E = 16; N = 1024; T = 4096.
[^7]: busiest expert 2048 tokens: ratio 3.57 [3.56, 3.57], 2 sessions; C_max = 2048; D = 1024; E = 16; N = 1024; T = 4096.
[^8]: uniform routing, tile 256 against 512: median ratio 1.2, cells 1.2 to 1.21, over 2 cells, 2 sessions each.
[^9]: uniform routing, tile 128 against 512: median ratio 1.42, cells 1.41 to 1.43, over 2 cells, 2 sessions each.
[^10]: eight hot experts, tile 256 against 128: median ratio 1.73, range [1.72, 1.73], over 2 cells, 2 sessions each.
[^11]: eight hot experts, tile 512 against 128: median ratio 2.84, cells 2.82 to 2.85, over 2 cells, 2 sessions each.
[^12]: 128-row bins against 256: median ratio 1.32, cells 1.31 to 1.32, over 2 cells, 2 sessions each.
[^13]: 128 against 256, 48 tokens per expert: ratio 1.31 [1.31, 1.31], 2 sessions; E = 8; K = 2048; M_BIN = 128; N = 2048; T_VALID = 48.
[^14]: 128 against 256, 80 tokens per expert: ratio 1.32 [1.32, 1.32], 2 sessions; E = 8; K = 2048; M_BIN = 128; N = 2048; T_VALID = 80.
[^15]: 256-row bins against 512 in two passes: median ratio 0.527, range [0.526, 0.527], over 2 cells, 2 sessions each.
[^16]: 256 against 512, 48 tokens per expert: ratio 0.527 [0.527, 0.527], 2 sessions; E = 8; K = 2048; M_BIN = 128; N = 2048; T_VALID = 48.
[^17]: 256 against 512, 80 tokens per expert: ratio 0.526 [0.526, 0.526], 2 sessions; E = 8; K = 2048; M_BIN = 128; N = 2048; T_VALID = 80.
[^18]: 64 experts of width 256 against 128 of width 128: median ratio 1.55, range [1.48, 1.62], over 2 cells, 2 sessions each.
[^19]: 64 × 256 against 128 × 128, 256 tokens: ratio 1.49 [1.48, 1.49], 2 sessions; BM = 256; K = 2048; Total_Dffn = 16384.
[^20]: 64 × 256 against 128 × 128, 512 tokens: ratio 1.61 [1.6, 1.62], 2 sessions; BM = 512; K = 2048; Total_Dffn = 16384.
[^21]: 32 experts of width 512 against 128 of width 128: median ratio 2.15, range [1.96, 2.35], over 2 cells, 2 sessions each.
[^22]: 32 × 512 against 128 × 128, 256 tokens: ratio 1.96 [1.96, 1.97], 2 sessions; BM = 256; K = 2048; Total_Dffn = 16384.
[^23]: 32 × 512 against 128 × 128, 512 tokens: ratio 2.34 [2.33, 2.35], 2 sessions; BM = 512; K = 2048; Total_Dffn = 16384.
[^24]: 8 experts of width 2048 against 128 of width 128: median ratio 2.99, range [2.5, 3.5], over 2 cells, 2 sessions each.
[^25]: 8 × 2048 against 128 × 128, 256 tokens: ratio 2.51 [2.5, 2.52], 2 sessions; BM = 256; K = 2048; Total_Dffn = 16384.
[^26]: 8 × 2048 against 128 × 128, 512 tokens: ratio 3.47 [3.44, 3.5], 2 sessions; BM = 512; K = 2048; Total_Dffn = 16384.
[^27]: Pallas active-expert streaming against all experts masked: median ratio 6.09, cells 4.09 to 8.1, over 2 cells, 2 sessions each.
[^28]: Pallas streaming, 8 experts touched: ratio 8.1 [8.06, 8.14], 2 sessions; BM_dec = 16; E_total = 64; E_touched = 8; K = 2048; N = 2048.
[^29]: Pallas streaming, 16 experts touched: ratio 4.09 [4.08, 4.1], 2 sessions; BM_dec = 16; E_total = 64; E_touched = 16; K = 2048; N = 2048.
[^30]: XLA gather of touched experts against all experts masked: median ratio 0.383, cells 0.342 to 0.423, over 2 cells, 2 sessions each.
[^31]: XLA gather, 8 experts touched: ratio 0.423 [0.422, 0.424], 2 sessions; BM_dec = 16; E_total = 64; E_touched = 8; K = 2048; N = 2048.
[^32]: XLA gather, 16 experts touched: ratio 0.342 [0.342, 0.343], 2 sessions; BM_dec = 16; E_total = 64; E_touched = 16; K = 2048; N = 2048.
[^33]: W8A16 against bf16 experts: median ratio 1.15, cells 1.13 to 1.17, over 2 cells, 2 sessions each.
[^34]: W8A16, 128 tokens per expert: ratio 1.17 [1.17, 1.17], 2 sessions; E = 8; K = 2048; M_e = 128; N = 2048.
[^35]: W8A16, 256 tokens per expert: ratio 1.13 [1.13, 1.13], 2 sessions; E = 8; K = 2048; M_e = 256; N = 2048.
[^36]: W4A16 against bf16 experts: median ratio 1.22, cells 1.21 to 1.23, over 2 cells, 2 sessions each.
[^37]: W4A16, 128 tokens per expert: ratio 1.23 [1.22, 1.23], 2 sessions; E = 8; K = 2048; M_e = 128; N = 2048.
[^38]: W4A16, 256 tokens per expert: ratio 1.21 [1.21, 1.22], 2 sessions; E = 8; K = 2048; M_e = 256; N = 2048.
[^39]: pipelined VPU dequant kernel against two-pass HBM dequant: median ratio 2.4, cells 1.97 to 2.83, over 2 cells, 2 sessions each.
[^40]: pipelined kernel, output width 4096: ratio 1.97 [1.85, 2.09], 2 sessions; BM = 512; BN = 512; K = 4096; M = 512; N = 4096.
[^41]: pipelined kernel, output width 8192: ratio 2.83 [2.83, 2.83], 2 sessions; BM = 512; BN = 512; K = 4096; M = 512; N = 8192.
[^42]: bf16-weight Pallas matmul against two-pass HBM dequant: median ratio 2.08, cells 2.02 to 2.15, over 2 cells, 2 sessions each.
[^43]: bf16-weight matmul, output width 4096: ratio 2.02 [1.94, 2.1], 2 sessions; BM = 512; BN = 512; K = 4096; M = 512; N = 4096.
[^44]: bf16-weight matmul, output width 8192: ratio 2.15 [2.15, 2.15], 2 sessions; BM = 512; BN = 512; K = 4096; M = 512; N = 8192.
[^45]: fused prologue against separate HBM amax and quantize passes: median ratio 1.43, cells 1.41 to 1.45, over 2 cells, 2 sessions each.
[^46]: 2048 tokens: ratio 1.45 [1.15, 1.75], 2 sessions; BK = 512; BM = 256; BN = 512; K = 4096; M = 2048; N = 4096.
[^47]: 4096 tokens: ratio 1.41 [1.11, 1.71], 2 sessions; BK = 512; BM = 256; BN = 512; K = 4096; M = 4096; N = 4096.
[^48]: Pallas grouped SwiGLU against a per-expert loop (naive baseline): median ratio 22, range [20.3, 23.8], over 2 cells, 2 sessions each.
[^49]: intermediate width 512: ratio 22 [20.3, 23.8], 2 sessions; D = 1024; E = 8; F = 512; T_exp = 256.
[^50]: intermediate width 1024: ratio 22.1 [20.6, 23.5], 2 sessions; D = 1024; E = 8; F = 1024; T_exp = 256.
[^51]: grouped kernel against per-expert loop: median ratio 19.3, cells 17.9 to 20.7, over 2 cells, 2 sessions each.
[^52]: against loop, intermediate width 512: ratio 20.7 [20.5, 20.9], 2 sessions; D = 1024; E = 8; F = 512; T_exp = 256.
[^53]: against loop, intermediate width 1024: ratio 17.9 [17.9, 18], 2 sessions; D = 1024; E = 8; F = 1024; T_exp = 256.
[^54]: against XLA batched einsum, intermediate width 1024: ratio 1.34 [1.34, 1.35], 2 sessions; D = 1024; E = 8; F = 1024; T_exp = 256.
[^55]: 16 rows, indirect DMA against one-hot matmul: median ratio 2.52, cells 1.63 to 3.41, over 2 cells, 2 sessions each.
[^56]: 16 rows, width 1024: ratio 1.63 [1.47, 1.78], 2 sessions; D = 1024; V = 32768.
[^57]: 16 rows, width 2048: ratio 3.41 [3.28, 3.54], 2 sessions; D = 2048; V = 32768.
[^58]: 1024 rows, width 1024, indirect DMA against one-hot matmul: ratio 0.362 [0.361, 0.362], 2 sessions; D = 1024; V = 32768.
[^59]: 8 rows per descriptor against 1: median ratio 5.6, cells 4.45 to 6.74, over 2 cells, 2 sessions each.
[^60]: 8 rows per descriptor, 512 rows: ratio 4.45 [3.87, 5.03], 2 sessions; B = 512; D = 4096; V = 4096.
[^61]: 8 rows per descriptor, 1024 rows: ratio 6.74 [6.71, 6.78], 2 sessions; B = 1024; D = 4096; V = 4096.
[^62]: 16 rows per descriptor against 1: median ratio 6.67, range [3.87, 9.98], over 2 cells, 2 sessions each.
[^63]: 16 rows per descriptor, 512 rows: ratio 4.47 [3.87, 5.07], 2 sessions; B = 512; D = 4096; V = 4096.
[^64]: 16 rows per descriptor, 1024 rows: ratio 8.87 [7.75, 9.98], 2 sessions; B = 1024; D = 4096; V = 4096.
[^65]: SMEM-prefetched page DMA against VMEM-staged pool: median ratio 0.283, cells 0.191 to 0.375, over 2 cells, 2 sessions each.
[^66]: 16 active pages: ratio 0.375 [0.375, 0.376], 2 sessions; B = 16; H = 128; MAX_PAGES = 128; W = 128; num_pages = 16.
[^67]: 32 active pages: ratio 0.191 [0.191, 0.191], 2 sessions; B = 16; H = 128; MAX_PAGES = 128; W = 128; num_pages = 32.
[^68]: pl.when skipping against jnp.where masking: ratio 1.07 [0.996, 1.09], 2 sessions; 32 or 64 blocks of 256 by 256, a quarter active.
[^69]: SMEM-driven packing then matmul against padded matmul: median ratio 3.86, cells 2.83 to 4.89, over 2 cells, 2 sessions each.
[^70]: a quarter of slots used: ratio 4.89 [4.87, 4.92], 2 sessions; B = 32; D = 2048; N = 4096; S_max = 512; Total_Tokens = 4096.
[^71]: half of slots used: ratio 2.83 [2.81, 2.85], 2 sessions; B = 32; D = 2048; N = 4096; S_max = 512; Total_Tokens = 8192.
[^72]: lax.top_k against a full sort: median ratio 10.2, cells 8.06 to 12.3, over 2 cells, 2 sessions each.
[^73]: lax.top_k, 16384 columns: ratio 8.06 [7.94, 8.17], 2 sessions; B = 64; K = 16; V = 16384; out_cols = 128.
[^74]: lax.top_k, 32768 columns: ratio 12.3 [12.1, 12.5], 2 sessions; B = 64; K = 16; V = 32768; out_cols = 128.
[^75]: Pallas radix select against a full sort: median ratio 11, range [8.32, 14.1], over 2 cells, 2 sessions each.
[^76]: Pallas radix select, 16384 columns: ratio 8.86 [8.32, 9.39], 2 sessions; B = 64; K = 16; V = 16384; out_cols = 128.
[^77]: Pallas radix select, 32768 columns: ratio 13.2 [12.4, 14.1], 2 sessions; B = 64; K = 16; V = 32768; out_cols = 128.
