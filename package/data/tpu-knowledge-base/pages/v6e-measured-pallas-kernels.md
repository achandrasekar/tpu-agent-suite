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
entry: v6e-measured-pallas-kernels
keyword: v6e-measured-pallas-kernels
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My Pallas kernel on TPU v6e is slower than it should be: data movement, layout, padding and which unit does the work

You have written a Pallas kernel for a TPU v6e, or you are about to, and it
is slower than the arithmetic says it should be. Perhaps a transpose or a
slice sits just before the `pallas_call`, your weights or activations are
stored in int8 and unpacked somewhere, your batches are padded to a power of
two, or you are not sure whether a small recurrent update belongs on the
matrix unit (MXU) or the vector unit (VPU). This page collects measurements
of those choices on one v6e chip. Every kernel
was written in JAX Pallas and called from PyTorch through torch_tpu's
`pallas.jax_op`; some sections also compare the same computation left to XLA.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process, in two separate sessions. Above one, the alternative is faster by
that factor. Below one, the alternative is slower, and the number is the
share of the baseline's speed it keeps. Each experiment checked every arm's
output against a full-precision reference before timing it, so the arms
being compared compute the same result.

## Measured

### Don't move data in HBM before the kernel

The single most expensive habit measured here is reshaping a tensor in
memory just before handing it to a Pallas kernel. The kernel was a streaming
elementwise pass over a bf16 matrix with 4096 columns and 8192 or 16384 rows,
in 512 by 512 tiles. The baseline transposed (or sliced) the input with
ordinary JAX operations and passed the result to `pallas_call`; XLA then
writes a rearranged copy to HBM and the kernel reads that copy.

- **Pass the input as it is stored** (no transpose at all):
  1.8× faster.[^1]
- **Do the transpose in the BlockSpec** instead, by swapping the axes in
  the `index_map` and the block shape so each tile is fetched from the
  untransposed array: 1.78×
  faster,[^2] which is almost all of the gain.
- **Take a column slice in the BlockSpec** rather than slicing
  `x[:, :N]` before the call: 1.72×
  faster.[^3]
- **A transposed operand in an elementwise kernel.** For `Z = X^T + Y` on
  float32 matrices of 2048 and 4096 square, transposing each 128 by 128
  tile of X in VMEM inside the one kernel is
  1.86× faster[^4] than
  a separate kernel that writes X^T to HBM followed by the add, and the gain
  is the same at both sizes. The prediction, from cutting five passes over
  HBM to three, was 1.67×; it came in above that.

The same holds for attention and projection layouts:

- **Multi-head projections.** With 32 heads, a contraction depth of 256 and
  an output width of 256, a Pallas kernel that reads activations stored
  head-major (`[H, M, D]`) is 1.51×
  faster[^5] than one that is given token-major
  (`[M, H, D]`) activations transposed into head-major in HBM first.
- **The key cache.** In a paged decode kernel (64 KV streams, 16 grouped
  queries each, head width 128, pages of 512 tokens), transposing the key
  cache in HBM before the kernel keeps only
  0.759 of the speed[^6]
  of leaving it sequence-major and transposing each tile inside the kernel.

### Rearranging a tile in VMEM is cheap; a transposed contraction is not free

Once a tile is in VMEM, rearranging it costs little:

- In the multi-head projection, a kernel that transposes its tile in VMEM so
  the contraction axis comes last is only
  1.06× slower than one whose data
  already arrives that way.[^7]
- In paged decode, storing the key cache pre-transposed made no difference
  while the kernel was limited by streaming pages from HBM: ratios of
  1.02[^8] and
  1.02[^9] at 4096 and 8192
  tokens of context.
- It starts to matter only when the kernel reuses each tile many times.
  With 16 query-key products per page inside VMEM, the pre-transposed layout
  is 1.11× faster.[^10]
- **Contracting a transposed axis needs no explicit transpose.** In a
  batched float32 matmul (M = N = 256, K = 512, 16 or 32 in the batch),
  `lax.dot_general` contracting the right-hand operand's minor axis
  (`X @ W^T`) costs the same as `jnp.swapaxes` on W in VMEM followed by
  `jnp.dot`, to within the noise floor (ratio
  1).[^11] So does
  contracting the left-hand operand's major axis (`A^T @ B`), ratio
  0.997.[^12] Given that
  you are going to contract a transposed axis, how you write it does not
  matter at that shape.
- **But the transposed contraction itself costs about 15% in a bf16
  GEMM.** That earlier result compared two spellings of the same transposed
  contraction. Measured against the plain orientation (`A @ B`, contracting
  the left operand's minor axis with the right operand's major axis), in a
  bf16 Pallas GEMM with 256 by 256 by 256 tiles (M of 2048 or 4096, N = K =
  2048), every transposed orientation was slower: contracting B's minor axis
  (`A @ B^T`) kept 0.857 of the
  speed,[^13] A's major axis (`A^T @ B`)
  0.822,[^14] and both
  (`A^T @ B^T`) 0.865.[^15]
  The same at both sizes: for example 0.824 and
  0.819 for `A^T @ B` at M = 2048 and
  4096.[^16][^17] The experiment expected no difference; the cost is small
  but the same at both sizes. If you control an operand's layout in HBM,
  store it in the orientation the contraction wants.

XLA, by contrast, chose its own layout: the same multi-head projection
written as a plain `einsum` ran at the same speed whichever axis order the
operands were given in (ratio 0.999).[^18]
In that one einsum XLA was insensitive to the axis order; in a Pallas kernel
the layout is yours to get right.

### Put the operand that doesn't change on the outer grid axis

When one input's BlockSpec does not depend on one of the grid axes, Mosaic
skips re-fetching it while only that axis moves. So iterate that axis
innermost, and keep the larger or invariant operand's axis outermost.

- **A matmul with unequal tiles** (an A tile of 1024 by 512 and a B tile of
  128 by 512, in bf16): putting A's axis outer, so the tile eight times
  larger is fetched once per row of the grid, is
  1.53× faster.[^19]
- **A per-row bias added to a streamed tile**: putting the row axis outer,
  so the bias tile is fetched once instead of once per column block, is
  1.34× faster.[^20]

### In a matmul kernel, keep the accumulator in place and fuse what shares a tile

- **Contract the whole depth in one pass.** For a bf16 matmul of 2048 by
  K by 2048 in 256 by 256 output tiles, contracting all of K in one step
  is faster than the usual pattern of stepping K in 256-deep blocks and
  adding each partial product into a float32 scratch accumulator in VMEM:
  1.33× overall,[^21]
  1.22× at K = 1024[^22] and
  1.45× at K = 2048.[^23] The
  gain grows with depth. The stepped version is the textbook Pallas matmul,
  so this is a comparison against a real alternative; it holds while a full
  K-deep tile fits in VMEM, and no depth beyond 2048 was tried.
- **Fuse both projections of a gated linear unit.** For GeGLU and ReGLU
  with M = N = 2048 in bf16, one Pallas kernel that runs the gate and up
  contractions on the same input tile and applies the gating in VMEM is
  1.34× faster for
  GeGLU[^24] and 1.35×
  for ReGLU[^25] than two matmul passes followed by a
  gating pass through HBM. The gain is larger at a contraction depth of
  2048 (1.38× GeGLU,[^26]
  1.4× ReGLU[^27]) than at
  4096 (1.29×,[^28]
  1.3×[^29]), where the
  matmuls take a larger share of the time. The baseline was the decomposed
  form; it was not compared with XLA compiling the same code.
- **Which gated unit: ReGLU is cheapest, SwiGLU and GeGLU cost the same.**
  At matched 256 by 256 tiles in one VMEM kernel (16 steps, 4096 or 8192
  rows, width 1024), ReGLU was 1.28×
  faster than GeGLU[^30] (1.28×
  and 1.28× at the two
  sizes[^31][^32]), and SwiGLU kept
  0.95 of GeGLU's
  speed.[^33] The same experiment's
  4.21× for the VMEM GeGLU kernel over a
  decomposed GeGLU[^34] is against a naive baseline:
  16 steps each writing an intermediate to HBM, which XLA would fuse. Read it as the cost of that decomposition, not as a
  gain over XLA.
- **Square tiles, by a little less than predicted.** For a bf16 matmul of M
  by 2048 by 2048, 256 by 256 tiles were 1.11×
  faster than 512 by 128 tiles,[^35]
  (1.11× at M = 2048,[^36]
  1.12× at 4096[^37]). The
  experiment expected at least 1.12×; it came in just under. Of the two
  rectangles, 128 by 512 kept 0.949 of the
  speed of 512 by 128.[^38]
- **For a batched matmul, XLA beat these Pallas kernels.** For B = 8 or 16
  bf16 matmuls of 1024 cubed, XLA's batched matmul was 2.73×
  faster than the better of two untuned Pallas kernels.[^39] Between
  the two, stepping one batch element per grid step was expected to match
  unrolling two batch elements inside the kernel body; it kept only 0.751
  of the speed[^40] (0.751 at
  B = 8, 0.751 at
  16[^41][^42]).

### Normalisation: keep the row in VMEM, and read the baselines carefully

- **DyT and RMSNorm** over 2048 or 4096 rows of 4096, eight steps: a VMEM
  kernel for dynamic tanh (DyT) was 3.36×
  faster,[^43] and a VMEM RMSNorm
  2.15× faster,[^44]
  than a decomposed DyT that writes every step's intermediate to HBM. That
  control is a naive baseline, since XLA would fuse those steps; the ratios are the cost of the decomposed form.
  The two kernels were not compared with each other directly, nor with XLA.
- **Ultra-wide RMSNorm** (1024 rows of 16384 or 32768): reading each full
  row into one VMEM tile was 1.58×
  faster,[^45] and a two-pass kernel chunked through VMEM
  1.26× faster,[^46]
  than a two-pass version through HBM. The single tile led at both widths
  (1.63× and 1.53×,
  against 1.3× and
  1.21×
  chunked[^47][^48][^49][^50]).
  The HBM two-pass control is a naive baseline.
- **Fusing a two-layer MLP did not pay where it was predicted to.** For
  `relu(X @ W1) @ W2` in float32 with all widths 128, keeping the hidden
  activation in VMEM inside one kernel was expected to be at least 1.26×
  faster than two kernels passing it through HBM, at both sizes. It was
  not: with 8192 rows the two cost the same to within
  the noise floor (ratio
  1),[^51] and only with
  16384 rows was the fused kernel faster,
  1.76×.[^52] Time both before
  fusing a narrow MLP.
- **What a long epilogue costs.** Appending a chain of 16 tanh and
  multiply-add stages to each output tile, instead of one, keeps
  0.513 of the speed at a contraction depth of
  512[^53] and 0.61 at
  1024.[^54] The two arms do different work, so this is
  the price of the epilogue, not a choice between them; a deeper
  contraction hides more of it.

### Stream int8 and unpack it in VMEM

The kernel applied a scale, a bias and `silu` to an int8 matrix (4096
columns, 8192 or 16384 rows) and wrote bf16. Unpacking int8 to float32
quadruples its size, so where that happens matters:

- **Unpack inside the kernel**, reading one byte per element from HBM,
  rather than converting to float32 in HBM first and streaming that:
  2.55× faster.[^55]
- Even when the float32 copy **already exists** in HBM, reading the int8
  and unpacking it in the kernel is 1.44×
  faster[^56] than reading the float32.
- **Keep the tiles large.** Halving the tile from 1024 to 512 rows kept only
  0.68 of the speed,[^57]
  although both tile sizes fit in VMEM.

### Pad less: the next multiple of 256 beat the next power of two

The test was a prefill projection (contraction 4096, output 8192, bf16) whose
true token counts sit just above a power of two: 1280, 2304 and 4352. The
baseline padded each up to the next power of two; the alternative padded
only to the next multiple of 256, which for these counts is no padding at
all. So what this shows is that less padding is faster. Padding to a
multiple of 128 was not tested, so it does not show that 256 (the v6e
MXU's width) is the granularity padding needs.

| true tokens | power of two | Pallas, 256-multiple speedup | XLA, 256-multiple speedup |
|---|---|---|---|
| 1280 | 2048 | 1.74×[^58] | 1.48×[^59] |
| 2304 | 4096 | 1.92×[^60] | 1.87×[^61] |
| 4352 | 8192 | 2.03×[^62] | 1.61×[^63] |

The Pallas gains track, and slightly exceed, the share of padded rows
removed.

The contraction depth behaves the same way from the other side: a depth
just under a multiple of 256 costs nearly as much as the multiple. Each run
was 12 steps of a 4096-row matmul, with the depth of each step varied.

- In XLA, a depth of 256 instead of 192 (a third more work) kept
  0.98 of the speed,[^64] and 512
  instead of 320 (three fifths more) kept
  0.928.[^65] Padding a
  ragged depth up to the next multiple of 256 is close to free.
- In a Pallas kernel with 1024 by 1024 output tiles, a block depth of 256
  instead of 384 was 1.33×
  faster[^66] and 512 instead of 384 kept
  0.861 of the speed.[^67]
  Time grew more slowly than depth, so deeper blocks did more work per
  second. These arms do different amounts of work: they tell you what a
  depth costs, not which depth to choose for a fixed problem.

### Small recurrent updates: VPU beat padded MXU calls, but XLA beat the kernel

The workload was one decode step of a linear-attention or state-space layer:
update a 128 by 128 bf16 state per stream with a decay and an outer product,
then multiply by the query, across 1024 or 2048 streams.

- Inside a Pallas kernel running eight recurrent steps, doing the update as
  vector multiply-adds on the VPU instead of padded `jnp.dot` calls on the
  MXU is 3.78×
  faster.[^68] The padded `jnp.dot` arm is a weak baseline: padding a
  rank-one update out to the MXU wastes most of each call.
- But for a single step, the fused Pallas VPU kernel kept only
  0.353 of the speed[^69]
  of leaving the update and the projection to XLA as two separate
  operations. A hand-written kernel is not automatically faster than XLA;
  time the XLA version first.

### Top-k: a sorting network is never slower than repeated argmax

The workload took the top k of 32 candidate vregs (8 by 128, float32) per
tile, across 32 or 64 tiles, either with a full 15-stage bitonic sorting
network or by repeating "find the maximum, mask it out" k times.

- For k = 1 and k = 2, the repeated argmax was no faster than the full sort:
  ratios of 1[^70] and
  1.01.[^71]
- For k = 16 it kept 0.662 of the sort's
  speed on average,[^72] from
  0.873 at 32 tiles[^73] to
  0.45 at 64.[^74]
- For k = 32 it kept 0.346 on
  average,[^75] from 0.46 at 32
  tiles[^76] to 0.233 at
  64.[^77]

At k = 1 and 2 the two tie; from k = 16 up the sorting network wins.
Only 32 candidates per tile were tested.

### Paged latent attention for decode: cache size first, then page size

The decode step served 32 sequences with 32 query heads, a DeepSeek-style
compressed latent of width 512 plus a rotary part of width 64, from a paged
cache whose pages were scattered at random through HBM, at 2048 and 4096
tokens of context.

- **Caching the compressed latent** instead of full per-head keys and
  values is the big win, whether the attention is written in XLA,
  7.88× faster,[^78] or
  as a Pallas kernel, 7.91×
  faster.[^79]
- **A Pallas kernel over the latent pages**, with the page table prefetched
  into SMEM, against XLA gathering the same pages: with pages of 128 tokens
  it gains little, 1.08×
  overall,[^80] from 1.03× at
  2048 tokens[^81] to 1.13× at
  4096.[^82] With pages of 256 tokens it is
  1.81× faster,[^83]
  from 1.72×[^84] to
  1.9×.[^85] If you write the
  kernel, prefer 256-token pages to 128; nothing larger was tested.

## Believed, not measured

- Most of the gains above are consistent with these kernels being limited
  by HBM traffic: each faster arm reads or writes fewer bytes. Bandwidth
  was not measured directly, and several gains fell short of what a
  pure byte count predicts (the grid-order matmul, the paged latent kernel)
  or exceeded it (256-multiple padding in Pallas).
- Why halving the int8 tile cost so much, when both sizes fit in VMEM, is
  not established. Per-step overhead in the grid is the usual suspect.
- Why XLA's two separate operations beat the fused single-step SSM kernel
  is not established either. XLA may simply map the update onto the
  hardware better than this particular kernel did.
- The one-pass contraction's gain growing with depth is consistent with
  each 256-deep step adding a read and write of the float32 accumulator in
  VMEM, so more steps cost more; those accesses were not measured.
- Why fusing the two-layer MLP paid at 16384 rows and not at 8192 is not
  established. Both sizes ran the same 256-row tile.
- The cost of a transposed contraction is consistent with Mosaic
  transposing tiles through the XLU before feeding the MXU, rather than the
  MXU loading them transposed for free; the generated code was not
  inspected. `pltpu.CompilerParams(fuse_transposed_lhs_in_matmul=True)` is
  a best-effort hint for a transposed left operand (off by default) and was
  not tried.

## Never measured

- More than one chip, and chips other than the v6e.
- Pallas `megacore` or core-parallel grid dimensions, and explicit manual
  DMA with `pltpu.async_copy`. Every kernel here used BlockSpec pipelining.
- Tile sizes other than those named in each section; none of the kernels
  was tuned, so better tiles may change the size of a gain.
- The backward pass. Every measurement is a forward pass or a decode step.
- fp8, int4 and other narrow types besides int8, and int8 matmuls (only
  the unpack and an elementwise epilogue were timed here).
- The fused gated linear units against XLA compiling the same JAX code,
  and SwiGLU in this form (`index.py get v6e-measured-pallas-kernel-internals`
  covers a SwiGLU kernel against `torch.compile`).
- The two spellings of a transposed contraction (`dot_general` against an
  explicit swap) in bf16 or at other shapes: both were float32 at M = N =
  256 and K = 512. And the transposed orientations against the plain one
  outside a bf16 GEMM at 256 by 256 by 256 tiles.
- DyT and RMSNorm kernels against each other or against XLA; batched
  matmul shapes other than 1024 cubed.
- Epilogue chains of lengths other than 1 and 16.
- Top-k over other candidate counts, and the crossover between iterative
  argmax and sorting below k = 16 other than at 1 and 2.
- Paged latent attention in prefill, page sizes other than 128 and 256,
  and contexts longer than 4096 tokens.
- End-to-end model throughput. These are kernel timings; how much of a
  step's time each kernel accounts for is open.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: contiguous input against a transpose before the call: median ratio 1.8, range [1.8, 1.81], over 2 cells, 2 sessions each.
[^2]: transpose in the BlockSpec: median ratio 1.78, range [1.78, 1.78], over 2 cells, 2 sessions each.
[^3]: column slice in the BlockSpec: median ratio 1.72, cells 1.71 to 1.72, over 2 cells, 2 sessions each.
[^4]: in-kernel tile transpose against a separate transpose kernel: median ratio 1.86, cells 1.85 to 1.86, over 2 cells, 2 sessions each.
[^5]: head-major input against an HBM pre-transpose: median ratio 1.51, range [1.5, 1.51], over 2 cells, 2 sessions each.
[^6]: HBM transpose before decode: median ratio 0.759, range [0.759, 0.76], over 2 cells, 2 sessions each.
[^7]: contraction axis transposed in VMEM: median ratio 1.06, cells 1.06 to 1.06, over 2 cells, 2 sessions each.
[^8]: pre-transposed K, 4096 tokens: ratio 1.02 [1.01, 1.02], 2 sessions; D = 128; M_q = 16; N_streams = 64; S = 4096; page_size = 512.
[^9]: pre-transposed K, 8192 tokens: ratio 1.02 [1.02, 1.02], 2 sessions; D = 128; M_q = 16; N_streams = 64; S = 8192; page_size = 512.
[^10]: 16 QK products per page in VMEM: median ratio 1.11, range [1.11, 1.11], over 2 cells, 2 sessions each.
[^11]: dot_general on W's minor axis against swapaxes then dot: median ratio 1, range [0.992, 1.01], over 2 cells, 2 sessions each.
[^12]: dot_general on A's major axis against swapaxes then dot: median ratio 0.997, range [0.986, 1.01], over 2 cells, 2 sessions each.
[^13]: A @ B^T against A @ B: median ratio 0.857, cells 0.855 to 0.86, over 2 cells, 2 sessions each.
[^14]: A^T @ B against A @ B: median ratio 0.822, range [0.813, 0.83], over 2 cells, 2 sessions each.
[^15]: A^T @ B^T against A @ B: median ratio 0.865, cells 0.863 to 0.867, over 2 cells, 2 sessions each.
[^16]: A^T @ B, M = 2048: ratio 0.824 [0.818, 0.83], 2 sessions; K = 2048; M = 2048; N = 2048.
[^17]: A^T @ B, M = 4096: ratio 0.819 [0.813, 0.825], 2 sessions; K = 2048; M = 4096; N = 2048.
[^18]: XLA einsum axis order: ratio 0.999 [0.991, 1.01], 2 sessions; BM = 512; D = 256; E = 256; H = 32; M = 8192.
[^19]: larger A tile outer: median ratio 1.53, range [1.51, 1.55], over 2 cells, 2 sessions each.
[^20]: invariant bias row outer: median ratio 1.34, cells 1.33 to 1.34, over 2 cells, 2 sessions each.
[^21]: full-depth contraction against 256-deep scratch steps: median ratio 1.33, range [1.22, 1.45], over 2 cells, 2 sessions each.
[^22]: K = 1024: ratio 1.22 [1.22, 1.22], 2 sessions; K = 1024; M = 2048; N = 2048.
[^23]: K = 2048: ratio 1.45 [1.45, 1.45], 2 sessions; K = 2048; M = 2048; N = 2048.
[^24]: fused GeGLU against two passes and HBM gating: median ratio 1.34, cells 1.29 to 1.38, over 2 cells, 2 sessions each.
[^25]: fused ReGLU against two passes and HBM gating: median ratio 1.35, cells 1.3 to 1.4, over 2 cells, 2 sessions each.
[^26]: GeGLU, K = 2048: ratio 1.38 [1.37, 1.4], 2 sessions; K = 2048; M = 2048; N = 2048.
[^27]: ReGLU, K = 2048: ratio 1.4 [1.39, 1.41], 2 sessions; K = 2048; M = 2048; N = 2048.
[^28]: GeGLU, K = 4096: ratio 1.29 [1.28, 1.3], 2 sessions; K = 4096; M = 2048; N = 2048.
[^29]: ReGLU, K = 4096: ratio 1.3 [1.29, 1.31], 2 sessions; K = 4096; M = 2048; N = 2048.
[^30]: ReGLU against GeGLU: median ratio 1.28, range [1.28, 1.28], over 2 cells, 2 sessions each.
[^31]: ReGLU against GeGLU, 4096 rows: ratio 1.28 [1.28, 1.28], 2 sessions; BM = 256; BN = 256; M = 4096; N = 1024; STEPS = 16.
[^32]: ReGLU against GeGLU, 8192 rows: ratio 1.28 [1.28, 1.28], 2 sessions; BM = 256; BN = 256; M = 8192; N = 1024; STEPS = 16.
[^33]: SwiGLU against GeGLU: median ratio 0.95, cells 0.95 to 0.95, over 2 cells, 2 sessions each.
[^34]: VMEM GeGLU against decomposed (naive): median ratio 4.21, range [4.21, 4.22], over 2 cells, 2 sessions each.
[^35]: 256x256 against 512x128: median ratio 1.11, cells 1.11 to 1.12, over 2 cells, 2 sessions each.
[^36]: M = 2048: ratio 1.11 [1.11, 1.12], 2 sessions; K = 2048; M = 2048; N = 2048.
[^37]: M = 4096: ratio 1.12 [1.11, 1.12], 2 sessions; K = 2048; M = 4096; N = 2048.
[^38]: 128x512 against 512x128: median ratio 0.949, cells 0.929 to 0.969, over 2 cells, 2 sessions each.
[^39]: XLA batched matmul against unrolled Pallas: median ratio 2.73, cells 2.73 to 2.73, over 2 cells, 2 sessions each.
[^40]: grid-pipelined against unrolled: median ratio 0.751, cells 0.751 to 0.751, over 2 cells, 2 sessions each.
[^41]: grid-pipelined, B = 8: ratio 0.751 [0.747, 0.755], 2 sessions; B = 8; K = 1024; M = 1024; N = 1024.
[^42]: grid-pipelined, B = 16: ratio 0.751 [0.746, 0.756], 2 sessions; B = 16; K = 1024; M = 1024; N = 1024.
[^43]: VMEM DyT against decomposed (naive): median ratio 3.36, cells 3.36 to 3.36, over 2 cells, 2 sessions each.
[^44]: VMEM RMSNorm against decomposed DyT (naive): median ratio 2.15, cells 2.14 to 2.16, over 2 cells, 2 sessions each.
[^45]: full-row tile against HBM two-pass: median ratio 1.58, cells 1.53 to 1.63, over 2 cells, 2 sessions each.
[^46]: chunked VMEM two-pass against HBM two-pass: median ratio 1.26, range [1.2, 1.3], over 2 cells, 2 sessions each.
[^47]: full row, width 16384: ratio 1.63 [1.62, 1.63], 2 sessions; BM = 16; D = 16384; M = 1024.
[^48]: full row, width 32768: ratio 1.53 [1.52, 1.54], 2 sessions; BM = 16; D = 32768; M = 1024.
[^49]: chunked, width 16384: ratio 1.3 [1.3, 1.3], 2 sessions; BM = 16; D = 16384; M = 1024.
[^50]: chunked, width 32768: ratio 1.21 [1.2, 1.22], 2 sessions; BM = 16; D = 32768; M = 1024.
[^51]: fused against two kernels, 8192 rows: ratio 1 [0.994, 1.01], 2 sessions; BM = 256; H = 128; K = 128; M = 8192; N = 128.
[^52]: fused against two kernels, 16384 rows: ratio 1.76 [1.73, 1.79], 2 sessions; BM = 256; H = 128; K = 128; M = 16384; N = 128.
[^53]: 16-stage against 1-stage epilogue, K = 512: ratio 0.513 [0.513, 0.514], 2 sessions; K = 512; M = 2048; N = 2048.
[^54]: 16-stage against 1-stage epilogue, K = 1024: ratio 0.61 [0.61, 0.611], 2 sessions; K = 1024; M = 2048; N = 2048.
[^55]: in-VMEM unpack against HBM-staged float32: median ratio 2.55, cells 2.54 to 2.57, over 2 cells, 2 sessions each.
[^56]: in-VMEM unpack against pre-unpacked float32: median ratio 1.44, cells 1.44 to 1.45, over 2 cells, 2 sessions each.
[^57]: 512-row tile against 1024: median ratio 0.68, range [0.67, 0.691], over 2 cells, 2 sessions each.
[^58]: Pallas, 1280 tokens: ratio 1.74 [1.74, 1.74], 2 sessions; K = 4096; M_256 = 1280; M_pow2 = 2048; M_true = 1280; N = 8192.
[^59]: XLA, 1280 tokens: ratio 1.48 [1.48, 1.48], 2 sessions; K = 4096; M_256 = 1280; M_pow2 = 2048; M_true = 1280; N = 8192.
[^60]: Pallas, 2304 tokens: ratio 1.92 [1.92, 1.92], 2 sessions; K = 4096; M_256 = 2304; M_pow2 = 4096; M_true = 2304; N = 8192.
[^61]: XLA, 2304 tokens: ratio 1.87 [1.87, 1.87], 2 sessions; K = 4096; M_256 = 2304; M_pow2 = 4096; M_true = 2304; N = 8192.
[^62]: Pallas, 4352 tokens: ratio 2.03 [2.03, 2.03], 2 sessions; K = 4096; M_256 = 4352; M_pow2 = 8192; M_true = 4352; N = 8192.
[^63]: XLA, 4352 tokens: ratio 1.61 [1.6, 1.62], 2 sessions; K = 4096; M_256 = 4352; M_pow2 = 8192; M_true = 4352; N = 8192.
[^64]: XLA depth 256 against 192: ratio 0.98 [0.98, 0.98], 2 sessions; BM = 1024; BN = 1024; M = 4096; N = 8192; R = 12.
[^65]: XLA depth 512 against 320: median ratio 0.928, range [0.926, 0.933], over 2 cells, 2 sessions each.
[^66]: Pallas block depth 256 against 384: median ratio 1.33, cells 1.33 to 1.34, over 2 cells, 2 sessions each.
[^67]: Pallas block depth 512 against 384: median ratio 0.861, cells 0.859 to 0.863, over 2 cells, 2 sessions each.
[^68]: eight steps, VPU against padded MXU: median ratio 3.78, cells 3.77 to 3.78, over 2 cells, 2 sessions each.
[^69]: fused Pallas VPU step against split XLA: median ratio 0.353, range [0.349, 0.358], over 2 cells, 2 sessions each.
[^70]: k = 1: ratio 1 [0.991, 1.01], 2 sessions; G = 32; N_cand = 32; lanes = 128; sublanes = 8.
[^71]: k = 2: median ratio 1.01, cells 1 to 1.01, over 2 cells, 2 sessions each.
[^72]: k = 16: median ratio 0.662, cells 0.45 to 0.873, over 2 cells, 2 sessions each.
[^73]: k = 16, 32 tiles: ratio 0.873 [0.824, 0.922], 2 sessions; G = 32; N_cand = 32; lanes = 128; sublanes = 8.
[^74]: k = 16, 64 tiles: ratio 0.45 [0.448, 0.452], 2 sessions; G = 64; N_cand = 32; lanes = 128; sublanes = 8.
[^75]: k = 32: median ratio 0.346, range [0.232, 0.485], over 2 cells, 2 sessions each.
[^76]: k = 32, 32 tiles: ratio 0.46 [0.433, 0.485], 2 sessions; G = 32; N_cand = 32; lanes = 128; sublanes = 8.
[^77]: k = 32, 64 tiles: ratio 0.233 [0.232, 0.234], 2 sessions; G = 64; N_cand = 32; lanes = 128; sublanes = 8.
[^78]: XLA latent cache against uncompressed paged attention: median ratio 7.88, cells 7.63 to 8.13, over 2 cells, 2 sessions each.
[^79]: Pallas latent cache against uncompressed paged attention: median ratio 7.91, cells 7.82 to 8.01, over 2 cells, 2 sessions each.
[^80]: Pallas pages of 128 against XLA gather: median ratio 1.08, cells 1.03 to 1.13, over 2 cells, 2 sessions each.
[^81]: pages of 128, 2048 tokens: ratio 1.03 [1.02, 1.04], 2 sessions; B = 32; H = 32; S = 2048; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^82]: pages of 128, 4096 tokens: ratio 1.13 [1.11, 1.14], 2 sessions; B = 32; H = 32; S = 4096; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^83]: Pallas pages of 256 against XLA gather: median ratio 1.81, range [1.7, 1.93], over 2 cells, 2 sessions each.
[^84]: pages of 256, 2048 tokens: ratio 1.72 [1.7, 1.75], 2 sessions; B = 32; H = 32; S = 2048; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^85]: pages of 256, 4096 tokens: ratio 1.9 [1.88, 1.93], 2 sessions; B = 32; H = 32; S = 4096; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
