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
entry: v6e-measured-pallas-skipping-overlap
keyword: v6e-measured-pallas-skipping-overlap
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My Pallas kernel on TPU v6e does less work but does not get faster: skipping tiles, ragged and packed batches, prefetching pages, and overlapping the MXU with everything else

You are writing a Pallas kernel for a TPU v6e and part of its work is
wasted: some tiles of a matmul are empty, a batch of packed documents or
ragged sequences leaves most attention tiles masked out, the grid is sized for
the largest input you will ever see, or the kernel pulls pages through a page
table. You want to skip that work, or hide one unit's work behind another's
(the memory transfers, the matrix unit (MXU), and the vector and special-function
units (VPU and SFU) that run softmax and activations). This page collects
measurements of those choices. Every experiment
ran in a single process on one v6e chip, with hand-written JAX Pallas kernels
called from PyTorch through torch_tpu's `pallas.jax_op`. Every arm's output
was checked against a full-precision PyTorch reference before any clock
started, and each arm was then timed over 20 samples after 5 warmups.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process, in two separate sessions. Above one, the alternative is faster by
that factor. Below one, the alternative is slower, and the number is the
share of the baseline's speed it keeps. Where the two arms do different
amounts of work (a different tile density, head width or number of
contractions), the ratio is a cost: how much the extra work slows you down,
not a choice between two ways of doing the same job. Each such place is
marked as a cost.

For kernel choices this page does not cover (grid order, layouts, padding,
which unit does the work) see `index.py get v6e-measured-pallas-kernels`. For tile sizes,
buffering, SMEM scalars and fusion inside the kernel see
`index.py get v6e-measured-pallas-kernel-internals`. For the attention kernels JAX ships
(splash, paged and ragged paged attention) see `index.py get v6e-measured-attention-serving`.

## Measured

### `pl.when` skips only the compute: a step that is still loaded still costs its copy

The pattern: put a mask or a counter in SMEM with `PrefetchScalarGridSpec`,
and wrap the tile's work in `@pl.when(...)` so inactive tiles return at once.
The grid stays the same size, so the BlockSpec pipeline still copies every
tile's inputs from HBM into VMEM, and every step still pays its fixed
overhead. Whether that saves anything depends on what limits a step. In
the matmuls below each step was copy-bound (a full-K weight block of about
4 MB against roughly half that time of MXU work), so skipping the compute
saved nothing. In the packed-document attention further down, each step
was compute-bound, and skipping the compute alone saved part of the time:
a skipped step still cost about 58% of a live one. To save the rest, keep
the skipped steps from loading anything: make the index map return the
previous step's block index for a skipped step, and Pallas does not
re-fetch a block whose index has not changed (or shrink the grid). Two
experiments tested `pl.when` on a matmul.

**A bitmask over a matmul with a SiLU epilogue (a cost).** M=16384, N=2048,
K=4096, bf16, 256x512 output tiles (a 64x4 grid). An int32 bitmask in SMEM
marks which tiles are active; active tiles do the dot, the SiLU and the
store, inactive tiles return. The baseline has every tile active. The arms
compute different amounts, so read these as the cost of the full kernel
compared with the sparse one:

- **25% of tiles active:** a ratio of 0.996, the same
  time as with every tile active.[^1]
- **12.5% of tiles active:** a ratio of 0.996, again
  the same time.[^2]

Skipping seven tiles in eight did not shorten the kernel at all.
The same experiment also ran half of the tiles active, and K=2048, but only
throughput and MXU utilisation ratios were stored for those, not times.

**A max-size grid with an SMEM early exit, against a grid sized to the
work.** M up to 32768 rows (128 row blocks of 256), N=2048, K=2048, bf16,
256x512 tiles, with only 16 or 32 of the 128 row blocks holding real rows.
The baseline runs the whole 128-block grid and computes every block,
including the empty ones. Both alternatives produce the same useful output:

- **Early exit:** an active-block count in SMEM, and
  `@pl.when(pl.program_id(0) < count)` around the dot and the store. It runs
  at 0.781 of the speed of simply computing
  every block,[^3] the same with 16 or 32 active blocks.
- **A grid sized to the active blocks:** compile and launch only the active
  row blocks. With 16 active blocks it is
  3.07× faster than the full
  grid.[^4]
  With twice as many active blocks it is 2.17×
  faster.[^5]

So in this one shape an early exit inside a static grid was worse than
doing the useless work, for a reason nobody has established (see below);
do not read it as a rule. Shrinking the grid paid. If the active size is known when you launch,
launch a grid of that size (bucket it if recompiles are a concern). If you
must keep a static grid, the next section shows what does work: make the
skipped steps stop loading data.

### Packed documents in attention: skip the copy, not just the math

Document-packed prefill attention: 32 batch-heads, 4096 tokens,
head width 128, bf16 with an fp32 online softmax, 256x256 tiles (16x16 tile
pairs per head). Each sequence packs 4 or 8 equal-length documents, so only
the tile pairs on the block diagonal hold any unmasked scores. The baseline
computes every tile pair and masks across documents with segment ids. Both
alternatives keep each tile's first and last document id in SMEM:

- **Skip the compute, keep the grid:** `@pl.when` around the MXU, VPU and SFU
  body when the tile's document ranges do not overlap. The grid still visits,
  and copies, every K and V tile. With 4 documents it is
  1.45× faster;[^6] with 8,
  1.58× faster.[^7]
- **Skip the copy as well:** the K and V index maps read the first KV tile of
  the query tile's document from SMEM, so the pipeline only ever loads tiles
  that overlap. With 4 documents it is 3.61×
  faster;[^8] with 8, 7.01×
  faster.[^9]

With 4 documents a quarter of the tile pairs are live and with 8 an eighth,
so the second version recovers almost all of the skippable time, while
skipping compute alone recovers a fraction of it. Put the skip in the index
map, so that skipped tiles are never fetched. This is the compute-bound
case from the first section: from these two cells, a skipped but still
loaded step cost about 58% of a live one.

### Ragged batches in flash attention: schedule a worklist, do not pad

A skewed batch of 8 documents, one of the maximum length and seven of an
eighth of it, causal attention, 4 heads of width 128, 128x128 tiles. The
baseline pads every document to the maximum length and gives each query tile
a loop over every KV tile, with no causal skipping, which is a weak
baseline. The alternative packs the real tokens end to end, puts a flat list
of the real query tiles in SMEM, and stops each tile's KV loop at the causal
edge of its own document, which cuts the number of tile steps by about an
order of magnitude.

- **Longest document 1024 tokens:** 7.22×
  faster.[^10] This cell was noisier than the other.
- **Longest document 2048 tokens:** 9.98×
  faster.[^11]

The gain grows with length. Part of it comes from causal skipping that the
padded baseline did not do, so it overstates what a worklist adds to a padded
kernel that already skips causal tiles. That comparison was not run.

### Block-sparse matmul: this kernel lost to dense

A 2048x2048 by 2048x2048 bf16 matmul whose left operand is block-sparse in
256x256 blocks, with 1 or 2 live blocks per block row (12.5% or 25%). The
baseline is the dense matmul on the MXU. The alternative is a Pallas
block-sparse-row kernel that stores only the live blocks and reads their
column indices from SMEM. This kernel ran at
0.563 of the dense matmul's
speed[^12]: 0.594 at one block per
row[^13] and 0.531 at
two.[^14] It did 4 to 8 times fewer MXU contractions
 and was
still slower. At this size, keep the dense matmul. This says nothing about
block-sparse kernels in general, or about larger matrices.

### Paged gathers: the XLA gather beat both Pallas kernels

The experiment was meant to cover paged decode attention; what actually ran
is simpler. A pool of 4096 pages, each 128 rows
of width 256 in bf16, a page table for 16 sequences of 128 or 256 pages, and
a contraction of every gathered page with a 256x256 weight, summed over
pages. The baseline gathers the pages into a contiguous HBM tensor with XLA
and then runs the einsum. The alternatives read the page table from SMEM and
let the BlockSpec index map fetch each page straight from the pool into VMEM:

- **One buffer** (`pl.Buffered(1)`, copy and compute in series):
  0.312 of the XLA path's
  speed.[^15]
- **Two buffers** (`pl.Buffered(2)`, next page copied during the current
  page's matmul): 0.617 of the
  speed.[^16]

Both page counts gave the same picture, though each cell's range is wide.
Double buffering clearly helped, but even then these kernels, which fetch one
128-row page per grid step, were slower than gathering first and
multiplying second, even though they read each page once instead of
writing a gathered copy and reading it back. Do not assume that
a per-page indirect DMA beats an XLA gather; time it at your page size.

### Overlapping the stages of a pipeline

A double-buffered Pallas pipeline over a 4096 or 8192 by 2048 bf16 tensor in
256x256 tiles, doing 4 rounds per tile of a 256x256 matmul followed by a
SiLU. The baseline is the same 4 rounds as 4 separate, unfused XLA
dispatches, each reading and writing its intermediate through HBM. That is a
deliberately weak baseline, not XLA's best fused version.

- **The full fused kernel** (copy, matmul and SiLU in one pipeline) is
  9.1× faster than the unfused
  stages.[^17] It did not vary with size.

The experiment also timed stripped-down versions of the kernel against the
same baseline. These do less work than the baseline, so they are not
alternatives to it; but because they share one baseline, they show what
each stage adds. The higher the figure, the faster the stripped kernel:

- **Copy only:** 12.5.[^18]
- **Copy plus the matmuls:** 11.[^19]
- **Copy plus the SiLUs:** 10.2.[^20]
- **All three:** 9.1, as above.[^17]

Neither the matmuls nor the SiLUs hid behind the copy: in time, the two
stages' costs add almost exactly. This is a ladder of stripped kernels
against one weak baseline, not a direct overlap measurement, and it is
one kernel at one tile size, but on this evidence the units did not
overlap their work here; budget for each stage's time rather than assuming
it is free.

### Adding contractions per tile, and fp8 on the MXU

A double-buffered kernel streams a 4096 or 8192 by 2048 bf16 tensor in
256x256 tiles and does R matmuls of 256x256 per tile. The baseline does one.

- **Two per tile instead of one:** 0.999, no
  difference.[^21]
- **Sixteen per tile instead of one (a cost):** the kernel keeps
  0.923 of its
  speed[^22] while doing sixteen times the matrix work.
  
Sixteen times the MXU work cost under a tenth more time, where a roofline
model with the published peak rates predicted several times more. Doing the
extra contractions inside a tile you have already loaded was close to free
in this one kernel, probably because it was paying per-step overhead
rather than running at its memory floor (see below); in a kernel that is
already at its floor, reuse may not be as cheap.

- **fp8 (`float8_e4m3fn`) instead of bf16** at sixteen contractions per
  tile: 0.973 of bf16's
  speed,[^23] slightly slower, in both sizes. In this one kernel fp8
  operands bought no matrix speed, which is consistent with v6e's fp8
  rate being listed equal to bf16's. Use fp8 for memory, if at all, not for
  MXU rate. The measured matmul and quantization choices are
  in `index.py get v6e-measured-matmul-quantization`.

### Two query tiles per KV step in flash attention

Flash attention forward, 16 batch-heads, head width 128, 2048 or 4096
tokens, bf16. The baseline gives each grid step one 128-row query tile and
runs its QK^T, exp and PV in a strict chain. The alternative holds two
128-row query tiles per grid step and interleaves their matmuls and
exponentials, so the MXU can work on one tile while the SFU works on the
other, and each loaded K and V tile serves both. It is
1.82× faster,[^24] the same
at both lengths.

Be careful what you credit. The alternative also doubles the query rows per
K and V load, which is the same as going from a 128-row to a 256-row query
block, and the next section shows that change alone giving a gain of the
same size (at head width 256).[^25]
The interleaving itself was not separated from the larger query block.

### Wide heads (D=256) in flash attention

Flash attention forward with head width 256 (as in Gemma 2 2B and 9B), 16
batch-heads, 128-row KV blocks, 2048 or 4096 tokens, bf16 with an fp32
output accumulator. The baseline uses 128-row query blocks.

- **256-row query blocks:** 1.9×
  faster.[^25]
- **512-row query blocks:** 1.77×
  faster.[^26]

Both larger blocks beat 128 rows by a wide margin, and 256 rows was a little
better than 512 at both lengths. At head width 256, use 256-row query
blocks (measured at one KV block size, 128 rows).

- **Head width 128 instead of 256 (a cost)**, both at 256-row query blocks:
  1.11× faster.[^27] The
  two arms compute different models, so this is not a choice; it says that
  doubling the head width cost only about a tenth more time, much less than
  doubling the matrix work would suggest. In these two cells, at one KV
  block size, wide heads were not punished much once the query block was
  large.

## Believed, not measured

- A grid step whose input copy takes longer than its compute is limited by
  the copy, so skipping its compute cannot shorten it. That fits the bitmask
  and early-exit results, where the tiles were still loaded, and the
  document and grid-size results, where they were not; but the copies were
  not timed on their own.
- Why the early exit made the kernel slower than computing every block is
  not known. A conditional around the dot and the store may change how the
  compiler schedules or pipelines the body.
- The single-contraction roofline kernel probably ran well above its memory
  floor, from per-step overhead, so the extra contractions fitted into time
  it was already spending. That would explain why sixteen times the work
  cost so little, and it means the experiment did not locate v6e's ridge
  point.
- The per-page Pallas kernels probably lost because each grid step moved one
  small page and paid a fixed per-step cost, while XLA's gather moves large
  contiguous blocks. Larger pages or several pages per step might change the
  result.
- The block-sparse kernel probably lost to per-block overhead and dynamic
  slicing on a problem small enough that the dense matmul is already short.
- In flash attention, a larger query block reuses each K and V tile over more
  rows and fills more of the 256-wide MXU, which is the likely reason both
  the 256-row blocks and the two-tile kernel gained so much over 128 rows.

## Never measured

- Skipping by remapping the index map for a plain block-sparse matmul (the
  method that worked for packed documents), against the dense matmul.
- An early exit that also redirects the skipped steps' index maps to a tile
  already in VMEM, so that nothing is copied.
- A worklist kernel against a padded kernel that already skips causal
  tiles, and against JAX's own ragged and splash kernels.
- Real paged decode attention with SMEM page prefetch, which the experiment
  was meant to cover but did not run. Also several pages
  per grid step, and other page sizes.
- Block-sparse matmul at larger sizes or with smaller blocks, and against
  the splash-style block-sparse kernels JAX ships.
- The separate effect of MXU and SFU interleaving, with the query block size
  held fixed.
- The backward pass of any of these kernels. Every timing here is a forward
  pass.
- fp8 at a memory-bound size, int8 on the MXU, and any precision effect on
  accuracy.
- More than one chip, and any chip other than v6e.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.10.1, libtpu 0.0.41, torch 2.10.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.
[^1]: all tiles vs one in four active: ratio 0.996 [0.984, 1.01], 2 sessions; BM = 256; BN = 512; K = 4096; M = 16384; N = 2048.
[^2]: all tiles vs one in eight active: ratio 0.996 [0.987, 1.01], 2 sessions; BM = 256; BN = 512; K = 4096; M = 16384; N = 2048.
[^3]: full max grid vs SMEM early exit, both cells: median ratio 0.781, cells 0.781 to 0.781, over 2 cells, 2 sessions each.
[^4]: full max grid vs shrunk exact grid: ratio 3.07 [3.06, 3.08], 2 sessions; G_active = 16; G_max = 128; K = 2048; M_max = 32768; N = 2048.
[^5]: full max grid vs shrunk exact grid: ratio 2.17 [2.16, 2.17], 2 sessions; G_active = 32; G_max = 128; K = 2048; M_max = 32768; N = 2048.
[^6]: dense doc mask vs SMEM no-op block skip: ratio 1.45 [1.45, 1.46], 2 sessions; BH = 32; B_kv = 256; B_q = 256; D = 128; N_docs = 4; S = 4096.
[^7]: dense doc mask vs SMEM no-op block skip: ratio 1.58 [1.57, 1.58], 2 sessions; BH = 32; B_kv = 256; B_q = 256; D = 128; N_docs = 8; S = 4096.
[^8]: dense doc mask vs SMEM document window skip: ratio 3.61 [3.61, 3.61], 2 sessions; BH = 32; B_kv = 256; B_q = 256; D = 128; N_docs = 4; S = 4096.
[^9]: dense doc mask vs SMEM document window skip: ratio 7.01 [7.01, 7.01], 2 sessions; BH = 32; B_kv = 256; B_q = 256; D = 128; N_docs = 8; S = 4096.
[^10]: padded grid vs SMEM worklist: ratio 7.22 [6.33, 8.1], 2 sessions; B = 8; D = 128; H = 4; S_max = 1024.
[^11]: padded grid vs SMEM worklist: ratio 9.98 [9.97, 10], 2 sessions; B = 8; D = 128; H = 4; S_max = 2048.
[^12]: dense MXU GEMM vs Pallas BSR SpMM, both densities: median ratio 0.563, cells 0.531 to 0.594, over 2 cells, 2 sessions each.
[^13]: 1 block per row: ratio 0.594 [0.524, 0.663], 2 sessions; K = 2048; M = 2048; N = 2048; block = 256; nnz_per_row = 1.
[^14]: 2 blocks per row: ratio 0.531 [0.483, 0.58], 2 sessions; K = 2048; M = 2048; N = 2048; block = 256; nnz_per_row = 2.
[^15]: XLA gather+matmul vs single-buffered SMEM indirect DMA: median ratio 0.312, range [0.185, 0.441], over 2 cells, 2 sessions each.
[^16]: XLA gather+matmul vs double-buffered SMEM indirect DMA: median ratio 0.617, range [0.366, 0.868], over 2 cells, 2 sessions each.
[^17]: unfused XLA stages vs full DMA+MXU+SFU kernel: median ratio 9.1, cells 9.04 to 9.15, over 2 cells, 2 sessions each.
[^18]: unfused XLA stages vs DMA-only kernel: median ratio 12.5, range [12.3, 12.7], over 2 cells, 2 sessions each.
[^19]: unfused XLA stages vs DMA+MXU kernel: median ratio 11, cells 10.9 to 11.1, over 2 cells, 2 sessions each.
[^20]: unfused XLA stages vs DMA+SFU kernel: median ratio 10.2, range [10, 10.3], over 2 cells, 2 sessions each.
[^21]: R=1 vs R=2 contractions per tile: median ratio 0.999, cells 0.999 to 1, over 2 cells, 2 sessions each.
[^22]: R=1 vs R=16 contractions per tile: median ratio 0.923, cells 0.923 to 0.924, over 2 cells, 2 sessions each.
[^23]: bf16 vs fp8 at R=16: median ratio 0.973, cells 0.971 to 0.975, over 2 cells, 2 sessions each.
[^24]: single 128-row tile vs dual-tile ping-pong: median ratio 1.82, cells 1.82 to 1.82, over 2 cells, 2 sessions each.
[^25]: D=256 query block 128 vs 256: median ratio 1.9, range [1.9, 1.9], over 2 cells, 2 sessions each.
[^26]: D=256 query block 128 vs 512: median ratio 1.77, range [1.76, 1.78], over 2 cells, 2 sessions each.
[^27]: D=256 vs D=128 at query block 256: median ratio 1.11, cells 1.1 to 1.11, over 2 cells, 2 sessions each.
