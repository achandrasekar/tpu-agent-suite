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
entry: v6e-measured-pallas-kernel-internals
keyword: v6e-measured-pallas-kernel-internals
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# I am writing the inside of a Pallas kernel on TPU v6e: which tile, buffering, scalar, loop and fusion choices actually change its speed

You are writing the body of a Pallas kernel for a TPU v6e: choosing a tile
height, deciding whether to alias the output onto the input, where to keep a
lookup table, how to shape a loop and its accumulators, whether to hand-fuse
an RMSNorm, softmax or SwiGLU block, and whether to call the kernel from
PyTorch or from JAX. This page collects measurements of those choices.
Every experiment ran on one v6e chip, in a single
process, in two sessions on two different hosts. Kernels were written in JAX
Pallas and, except where a section says otherwise, called from PyTorch
through torch_tpu's `pallas.jax_op`. Data movement, layout, grid order, int8
unpacking, padding and several specific kernels (state-space decode, top-k,
paged latent attention) are in `index.py get v6e-measured-pallas-kernels`;
this page is about what happens inside the kernel.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process. Above one, the alternative is faster by that factor. Below one, the
alternative is slower, and the number is the share of the baseline's speed
it keeps. Each experiment checked every arm's output against a reference
(exactly, for the elementwise kernels) before timing it, so the arms being
compared compute the same result.

## Measured

### Make the tile big enough, or don't write the kernel

The workload was relu over a 32768 by 512 float32 tensor: purely memory
bound. The baseline was `torch.relu` under `torch.compile`; the alternative
was the same relu as a Pallas kernel with a one-dimensional grid of row
tiles, each tile the full 512 columns wide. Both produce identical bytes;
the only difference is who chose the tiling.

| rows per tile | Pallas against compiled XLA |
|---|---|
| 8 | 0.0741[^1] |
| 32 | 0.257[^2] |
| 128 | 0.697[^3] |
| 512 | 0.994[^4] |
| 2048 | 0.999[^5] |

Small tiles are ruinous; from 512 rows up (1 MiB per tile ), the kernel ties
the compiler and never beats it. For this memory-bound relu the best the
hand-written kernel did was parity, and the worst was very much worse. If
you do write such a kernel, start at tiles of at least 512 full-width rows.
Only relu, at one array shape, was measured.

### Aliasing the output onto the input slowed a memory-bound relu

Same relu kernel, same tiles, same grid; the only change was passing
`input_output_aliases={0: 0}` so the kernel writes over its input instead of
into a fresh buffer. Every tile size got slower, and the penalty grew with
the tile:

- At 8 rows per tile the aliased kernel kept
  0.943 of the speed,[^6] at 32
  rows 0.822,[^7] at 128 rows
  0.631.[^8]
- From 512 rows up, exactly the tiles that make the kernel fast, it kept
  0.499 at 512,[^9]
  0.498 at 2048[^10] and
  0.497 at 4096:[^11] half the
  speed.
- Across all six tile sizes the median was 0.565.[^12]

For a memory-bound elementwise kernel that reads and writes every byte,
write to a separate output buffer unless memory is tight. This says nothing
about in-place updates that touch only part of a buffer, such as a KV-cache
write: aliasing a 256-row slice update was 3.47× faster than rewriting the
whole buffer (`index.py get v6e-measured-pallas-compile-and-buffers`).

### Leave automatic double buffering on; manual overlap pays only in a band

Pallas double-buffers BlockSpec inputs and outputs by default. In a bf16
matmul with 512 by 512 by 512 blocks and a float32 VMEM accumulator, forcing
the inputs to single buffering with `pl.Buffered(1)` kept
0.517 of the speed[^13]
over square problems of 2048, 4096 and 8192. Never single-buffer a block
whose index changes every grid step. `pl.Buffered(1)` makes the input copy,
the compute and the output copy run one after another, so the loss can
exceed 2× (it did in `index.py get v6e-measured-pallas-memory-pipeline`). A
block whose index never changes (weights, scalars) is not re-fetched, so
single-buffering it costs nothing and saves VMEM.

When the kernel drives its own copies (`pltpu.make_async_copy` from an
input left in HBM into two VMEM scratch buffers), issuing the second copy
before computing on the first is worth something only when copy and compute
are of similar size. The kernel copied two 128 kB half-blocks per grid step
and applied a chain of multiply-adds to each, with the chain length (the
arithmetic intensity, in operations per element) swept; the two arms
differed only in statement order.

| ops per element | issue-early against issue-late |
|---|---|
| 4 | 1[^14] |
| 16 | 1.08[^15] |
| 32 | 1.27[^16] |
| 64 | 1.22[^17] |
| 128 | 1.14[^18] |
| 256 | 1.08[^19] |
| 512 | 1.04[^20] |

Issuing early never hurt, so do it whenever you write manual DMA; but a
kernel that is mostly copy or mostly compute gains little. At best it gave
1.27×, well short of the 2× a perfect overlap of two equal stages would
allow. Mosaic followed source order here: it did not hoist the second copy
on its own.

### The `parallel` grid hint does nothing on v6e

v6e has one TensorCore per chip. Marking the matmul's first two grid axes
`parallel` instead of `arbitrary` gave a ratio of
1[^21] at every size.
It costs nothing either, so set it if the same kernel also targets a
multi-core chip.

### Put scalar tables in SMEM

The kernel chased a dependent chain through a 128-entry int32 table (each
read's address is the previous read's value), for 256 to 2048 reads per
grid step. Keeping the table in VMEM instead of SMEM kept
0.0895 of the speed at a chain of
256[^22], falling to 0.0694 at
2048.[^23] The gap does not keep growing: a law fitted
across all four depths, saturating in depth,[^24]
has it settling toward a floor as the shared launch cost dilutes out, so a
single average would understate the short chains and overstate the long
ones. This is the largest effect on this page. Page tables, indices, offsets and any
scalar that drives control flow or addressing belong in SMEM (via
`memory_space=pltpu.SMEM` or scalar prefetch). This measured latency of
dependent reads, not throughput of independent ones.

### Strided VMEM reads along sublanes are free

The kernel gathered 8 rows at a time, 32 times per grid step, from a 512 by
128 float32 block, either from consecutive rows or every S-th row; both arms
fetched the same block from HBM, so any VMEM cost could have been hidden
behind that copy. Dense time over strided time was
0.999 at stride 2,[^25]
1 at 4,[^26]
1 at 8,[^27]
0.999 at 16,[^28]
1 at 32[^29] and
1 at 64.[^30] In a kernel whose time is dominated by the HBM copy, don't
rearrange data to avoid a sublane stride. Lane-axis strides could not be
expressed (a strided ref slice needs a minor dimension of exactly 128) and
were not tested.

### Loop shape: unroll a little, and group accumulators however you like

- **Body length.** The same 8192 dependent multiply-adds on one vector
  register's worth of float32 data, as a loop whose body was 8 operations
  long against bodies of 32 to 8192 operations (fewer trips): the long body
  was 1.03× faster[^31]
  and already 1.02× at a body of
  32.[^32] A body of 8192 straight-line operations was no
  slower than one of 128 (1.03 at 8192 against
  1.03 at 128),[^33][^34]
  so no instruction-cache limit showed up. Unrolling a few dozen operations
  collects the gain; there is no reason to fear long bodies.
- **Accumulators.** A `fori_loop` carrying R separate (8, 128) float32
  accumulators against one (8R, 128) array, same arithmetic, for R from 4
  to 64: the split form was never slower, at
  1 for R = 4,[^35]
  1.08 for 8,[^36]
  1 for 16,[^37]
  1.06 for 32[^38] and
  1.05 for 64.[^39] Splitting one accumulator array into named pieces
  is free; write whichever form is clearer. (Adding more independent
  recurrences to the same loop is a different change: in
  `index.py get v6e-measured-vpu-memory-limits`, interleaving a second
  accumulator chain was 6% slower.)
- Mosaic did not fold these multiply-add chains away (XLA does), so a
  Pallas loop of dependent arithmetic really runs.

### Budget transcendentals as several adds each (bf16)

One (128, 128) bf16 tile, 16384 trips of a dependent loop, one operation
per trip; the baseline operation was an add. The share of the add's speed
each operation kept:

| operation | share of add's speed |
|---|---|
| multiply | 1[^40] |
| divide by a constant | 0.74[^41] |
| sqrt | 0.273[^42] |
| exp | 0.283[^43] |
| log | 0.283[^44] |
| reciprocal | 0.283[^45] |
| divide by a variable | 0.283[^46] |
| tanh | 0.295[^47] |
| sigmoid | 0.173[^48] |

Multiplies are as cheap as adds; exp, log, reciprocal, sqrt, tanh and
variable division all cost about the same, about 3.5 adds each; sigmoid
costs the most. These figures are latencies of a dependent chain, not
throughput, and are for bf16 only: the float32 cell produced no usable
figures. Do not read them as a reason to hoist a reciprocal: for a
loop-invariant divisor, plain division was faster than a hoisted reciprocal
and multiply (by 6%, measured directly in
`index.py get v6e-measured-vpu-memory-limits`).

### bf16 tiles: half the packed height is fine

bf16 packs 16 rows per tile. A Pallas copy of a 4096 by 4096 bf16 array
with blocks 8 rows high and twice as wide, against blocks 16 rows high (same
elements per block, same number of grid steps), gave
1.01[^49]: no
difference worth choosing on. An 8-row bf16 block is legal and costs
nothing extra. (Blocks not a multiple of 8 by 128 were refused at compile
time unless they span the whole array dimension, which Pallas also accepts;
fp8 blocks shorter than 32 rows were refused too.)

### Fused kernels: RMSNorm and softmax win, SwiGLU loses to torch.compile

- **RMSNorm** over 1024 rows of 4096 float32 values, against the same
  maths as a chain of eager PyTorch ops: one Pallas kernel was
  3.97× faster,[^50] and
  `torch.compile` of the eager chain was
  4.12× faster.[^51]
  Their ranges overlap, so this did not separate the Pallas kernel from
  `torch.compile`; the lesson is not to run the chain eagerly. Use
  `torch.compile` unless you need the kernel for another reason.
- **Softmax** over 16384 bf16 rows, a row-tiled Pallas kernel (max, exp,
  sum, divide in VMEM) against `jax.nn.softmax` compiled by XLA: the
  kernel was faster at every row length, from
  1.23× at rows of 512[^52] to
  1.77× at 2048[^53] and
  1.77× at 8192.[^54] The gain
  rises and then levels off rather than growing with row length: a
  saturating curve fitted across the three gives
  1.79× at rows of 4096.[^55]
  That curve rests on three points, and the tile height was different in
  each cell (256, 128 and 32 rows, the same for both arms within a cell),
  so row length and tile height are not separated. Against XLA it was
  worth writing; it was not compared with `torch.compile`.
- **SwiGLU** (both matmuls and the gate in one kernel, float32, 4096 by
  1024 activations, two 1024 by 1024 weights held whole in VMEM) against the
  same expression under `torch.compile`: the fused kernel was slower at
  every tile height, and how much slower depends on the tile, so no single
  figure describes it: a law fitted across all five heights,
  saturating in block\_rows,[^56] has the gap closing
  and then levelling off. It kept only
  0.457 at 64-row tiles[^57] and
  0.677 at 128,[^58] improving to
  0.862 at 256,[^59]
  0.903 at 512[^60] and
  0.901 at 1024.[^61] Keep the
  `torch.compile` version; if you write the fused kernel anyway, use tall
  tiles.

### Calling from PyTorch costs a fixed amount per call

The same Pallas RMSNorm kernel on bf16 rows of width 1024, called from
PyTorch through `torch.tpu.pallas.jax_op` against `jax.jit` from JAX
directly, on the same chip. Above one means JAX direct was faster:

- 1.27× at 16 rows,[^62]
  1.23× at 256[^63] and
  1.23× at 2048.[^64]
- 0.998 at 16384 rows:[^65]
  no difference.
- At 65536 and 262144 rows, 1.00:[^66]
  no difference.

For small kernels, the PyTorch bridge adds a per-call cost that JAX
avoids; once the kernel moves tens of megabytes per call it disappears.
Batch small calls together or fuse them into a larger kernel rather than
switching frameworks.

## Believed, not measured

- The tiling and accumulator results are consistent with a fixed cost per
  grid step that small tiles cannot amortise. The per-step cost itself was
  not measured.
- The aliasing penalty settling at half speed is consistent with the
  pipeline losing all overlap between reading the next tile and writing the
  current one, or with the alias forcing a hidden copy. The experiment
  cannot tell these apart; a profile would.
- Manual double buffering recovered well short of the most that hiding one
  copy could give. The output copy Pallas still manages, bandwidth shared
  between two copies in flight, or a per-copy issue cost may take the rest.
- The SMEM advantage is consistent with a VMEM scalar read needing a vector
  load, a lane extract and a hand-off to the scalar unit. Which part costs
  most is not established.
- The SwiGLU loss is consistent with the pipeline re-staging both resident
  weights at every grid step, which would explain why short tiles are
  worst. Why the fused kernel does not catch `torch.compile` even at tall
  tiles is not established.
- The bridge cost at small sizes is consistent with torch_tpu's per-op issue
  cost; JAX's own per-call cost on this host was not measured separately.

## Never measured

- More than one chip, other chips than the v6e, and multi-core grid
  splitting (v6e has one TensorCore).
- Per-step overhead isolated from tile size, and tiles for kernels other
  than relu and copies.
- Lane-axis strides in VMEM, and independent (rather than dependent) scalar
  or vector operation throughput.
- float32 operation latencies (the cell ran, but produced no usable
  figures), and integer VPU operations.
- In-place updates with kernels that are not memory bound, or that read
  and write different tiles.
- SwiGLU in bf16, other GLU variants, weights larger than the resident size
  tested, and a SwiGLU kernel that streams its weights instead of holding
  them.
- RMSNorm and softmax in other sizes and dtypes; softmax against
  `torch.compile`; backward passes of any kernel.
- Buffering deeper than two (`pallas_call` accepts only one or two
  buffers; `emit_pipeline` accepts more, untested here) and manual DMA with
  more than two copies in flight.
- End-to-end model throughput.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: Pallas relu against compiled XLA, 8-row tiles: ratio 0.0741 [0.073, 0.0751], 2 sessions; block_rows = 8; dim = 512; rows = 32768.
[^2]: Pallas relu against compiled XLA, 32-row tiles: ratio 0.257 [0.255, 0.26], 2 sessions; block_rows = 32; dim = 512; rows = 32768.
[^3]: Pallas relu against compiled XLA, 128-row tiles: ratio 0.697 [0.69, 0.705], 2 sessions; block_rows = 128; dim = 512; rows = 32768.
[^4]: Pallas relu against compiled XLA, 512-row tiles: ratio 0.994 [0.985, 1], 2 sessions; block_rows = 512; dim = 512; rows = 32768.
[^5]: Pallas relu against compiled XLA, 2048-row tiles: ratio 0.999 [0.987, 1.01], 2 sessions; block_rows = 2048; dim = 512; rows = 32768.
[^6]: aliased against fresh output, 8-row tiles: ratio 0.943 [0.943, 0.943], 2 sessions; block_rows = 8; dim = 512; rows = 32768.
[^7]: aliased against fresh output, 32-row tiles: ratio 0.822 [0.822, 0.822], 2 sessions; block_rows = 32; dim = 512; rows = 32768.
[^8]: aliased against fresh output, 128-row tiles: ratio 0.631 [0.631, 0.631], 2 sessions; block_rows = 128; dim = 512; rows = 32768.
[^9]: aliased against fresh output, 512-row tiles: ratio 0.499 [0.498, 0.5], 2 sessions; block_rows = 512; dim = 512; rows = 32768.
[^10]: aliased against fresh output, 2048-row tiles: ratio 0.498 [0.496, 0.499], 2 sessions; block_rows = 2048; dim = 512; rows = 32768.
[^11]: aliased against fresh output, 4096-row tiles: ratio 0.497 [0.497, 0.497], 2 sessions; block_rows = 4096; dim = 512; rows = 32768.
[^12]: aliased against fresh output, all tile sizes: median ratio 0.565, cells 0.497 to 0.943, over 6 cells, 2 sessions each.
[^13]: single-buffered inputs against default double buffering: median ratio 0.517, cells 0.512 to 0.53, over 3 cells, 2 sessions each.
[^14]: early against late second copy, 4 ops per element: ratio 1 [0.995, 1.01], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 4; rows = 8192.
[^15]: early against late second copy, 16 ops per element: ratio 1.08 [1.07, 1.09], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 16; rows = 8192.
[^16]: early against late second copy, 32 ops per element: ratio 1.27 [1.26, 1.29], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 32; rows = 8192.
[^17]: early against late second copy, 64 ops per element: ratio 1.22 [1.22, 1.22], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 64; rows = 8192.
[^18]: early against late second copy, 128 ops per element: ratio 1.14 [1.14, 1.14], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 128; rows = 8192.
[^19]: early against late second copy, 256 ops per element: ratio 1.08 [1.08, 1.08], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 256; rows = 8192.
[^20]: early against late second copy, 512 ops per element: ratio 1.04 [1.04, 1.04], 2 sessions; grid_steps = 16; half_rows = 256; lanes = 128; ops_per_elem = 512; rows = 8192.
[^21]: parallel against arbitrary dimension semantics: median ratio 1, range [0.999, 1], over 3 cells, 2 sessions each.
[^22]: VMEM against SMEM, chain of 256: ratio 0.0895 [0.0892, 0.0897], 2 sessions; depth = 256; grid_steps = 32; lanes = 128; rows = 8; table = 128.
[^23]: VMEM against SMEM, chain of 2048: ratio 0.0694 [0.0694, 0.0694], 2 sessions; depth = 2048; grid_steps = 32; lanes = 128; rows = 8; table = 128.
[^24]: VMEM table against SMEM table: saturating law fitted to four chain depths (256 to 2048), ratios 0.0895 down to 0.0694, i.e. SMEM 11× to 15× faster.
[^25]: strided against dense reads, stride 2: ratio 0.999 [0.994, 1], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 2.
[^26]: strided against dense reads, stride 4: ratio 1 [0.995, 1], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 4.
[^27]: strided against dense reads, stride 8: ratio 1 [0.995, 1], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 8.
[^28]: strided against dense reads, stride 16: ratio 0.999 [0.994, 1], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 16.
[^29]: strided against dense reads, stride 32: ratio 1 [0.997, 1.01], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 32.
[^30]: strided against dense reads, stride 64: ratio 1 [0.999, 1], 2 sessions; block_rows = 512; gathers = 32; lanes = 128; rows = 8192; stride = 64.
[^31]: long loop body against 8-op body, all lengths: median ratio 1.03, cells 1.02 to 1.03, over 5 cells, 2 sessions each.
[^32]: 32-op body against 8-op body: ratio 1.02 [1.02, 1.02], 2 sessions; body_ops = 32; lanes = 128; rows = 8192; total_ops = 8192.
[^33]: 8192-op body against 8-op body: ratio 1.03 [1.03, 1.03], 2 sessions; body_ops = 8192; lanes = 128; rows = 8192; total_ops = 8192.
[^34]: 128-op body against 8-op body: ratio 1.03 [1.03, 1.03], 2 sessions; body_ops = 128; lanes = 128; rows = 8192; total_ops = 8192.
[^35]: split against packed carry, R = 4: ratio 1 [0.999, 1], 2 sessions; accumulators = 4; block_rows = 32; iters = 512; lanes = 128; rows = 8192.
[^36]: split against packed carry, R = 8: ratio 1.08 [1.08, 1.08], 2 sessions; accumulators = 8; block_rows = 64; iters = 512; lanes = 128; rows = 8192.
[^37]: split against packed carry, R = 16: ratio 1 [1, 1], 2 sessions; accumulators = 16; block_rows = 128; iters = 512; lanes = 128; rows = 8192.
[^38]: split against packed carry, R = 32: ratio 1.06 [1.06, 1.06], 2 sessions; accumulators = 32; block_rows = 256; iters = 512; lanes = 128; rows = 8192.
[^39]: split against packed carry, R = 64: ratio 1.05 [1.05, 1.05], 2 sessions; accumulators = 64; block_rows = 512; iters = 512; lanes = 128; rows = 8192.
[^40]: bf16 multiply against add: ratio 1 [0.999, 1], 2 sessions; M = 128; N = 128; trips = 16384.
[^41]: bf16 divide by constant against add: ratio 0.74 [0.74, 0.741], 2 sessions; M = 128; N = 128; trips = 16384.
[^42]: bf16 sqrt against add: ratio 0.273 [0.273, 0.273], 2 sessions; M = 128; N = 128; trips = 16384.
[^43]: bf16 exp against add: ratio 0.283 [0.283, 0.283], 2 sessions; M = 128; N = 128; trips = 16384.
[^44]: bf16 log against add: ratio 0.283 [0.283, 0.283], 2 sessions; M = 128; N = 128; trips = 16384.
[^45]: bf16 reciprocal against add: ratio 0.283 [0.283, 0.283], 2 sessions; M = 128; N = 128; trips = 16384.
[^46]: bf16 divide by variable against add: ratio 0.283 [0.283, 0.283], 2 sessions; M = 128; N = 128; trips = 16384.
[^47]: bf16 tanh against add: ratio 0.295 [0.295, 0.295], 2 sessions; M = 128; N = 128; trips = 16384.
[^48]: bf16 sigmoid against add: ratio 0.173 [0.173, 0.173], 2 sessions; M = 128; N = 128; trips = 16384.
[^49]: 8-row bf16 blocks against 16-row blocks: median ratio 1.01, cells 1.01 to 1.01, over 3 cells, 2 sessions each.
[^50]: Pallas kernel against eager chain: ratio 3.97 [3.7, 4.24], 2 sessions; D = 4096; S = 1024.
[^51]: torch.compile against eager chain: ratio 4.12 [3.88, 4.36], 2 sessions; D = 4096; S = 1024.
[^52]: Pallas against XLA, rows of 512: ratio 1.23 [1.21, 1.24], 2 sessions; M = 16384; N = 512.
[^53]: Pallas against XLA, rows of 2048: ratio 1.77 [1.75, 1.8], 2 sessions; M = 16384; N = 2048.
[^54]: Pallas against XLA, rows of 8192: ratio 1.77 [1.75, 1.79], 2 sessions; M = 16384; N = 8192.
[^55]: Pallas against XLA softmax: saturating law fitted to three row lengths (512, 2048, 8192), one degree of freedom left; 4096 is interpolated, not measured.
[^56]: fused Pallas SwiGLU against torch.compile: saturating law fitted to the five tile heights in the text, 0.457 at 64 rows to 0.901 at 1024.
[^57]: fused against torch.compile, 64-row tiles: ratio 0.457 [0.457, 0.458], 2 sessions; H = 1024; K = 1024; M = 4096; block_rows = 64.
[^58]: fused against torch.compile, 128-row tiles: ratio 0.677 [0.676, 0.678], 2 sessions; H = 1024; K = 1024; M = 4096; block_rows = 128.
[^59]: fused against torch.compile, 256-row tiles: ratio 0.862 [0.862, 0.862], 2 sessions; H = 1024; K = 1024; M = 4096; block_rows = 256.
[^60]: fused against torch.compile, 512-row tiles: ratio 0.903 [0.902, 0.904], 2 sessions; H = 1024; K = 1024; M = 4096; block_rows = 512.
[^61]: fused against torch.compile, 1024-row tiles: ratio 0.901 [0.899, 0.903], 2 sessions; H = 1024; K = 1024; M = 4096; block_rows = 1024.
[^62]: JAX direct against jax_op, 16 rows: ratio 1.27 [1.21, 1.34], 2 sessions; D = 1024; R = 16; block_rows = 16.
[^63]: JAX direct against jax_op, 256 rows: ratio 1.23 [1.19, 1.26], 2 sessions; D = 1024; R = 256; block_rows = 256.
[^64]: JAX direct against jax_op, 2048 rows: ratio 1.23 [1.19, 1.26], 2 sessions; D = 1024; R = 2048; block_rows = 512.
[^65]: JAX direct against jax_op, 16384 rows: ratio 0.998 [0.993, 1.01], 2 sessions; D = 1024; R = 16384; block_rows = 512.
[^66]: JAX direct against jax_op, 65536 and 262144 rows: median ratio 1, cells 0.998 to 1, over 2 cells, 2 sessions each.
