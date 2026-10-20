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
entry: v6e-measured-pallas-memory-pipeline
keyword: v6e-measured-pallas-memory-pipeline
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My Pallas kernel on TPU v6e spends its time moving data: buffering, block shape, tails, narrow minor axes and what does not matter

You have a Pallas kernel on a TPU v6e that does little arithmetic per byte:
an elementwise pass, a normalisation, a gate, a small matmul tile. It is
slower than the copy it amounts to, and you suspect the way tiles get in
and out of VMEM rather than the maths. Perhaps you turned the pipeline's
buffering down to save VMEM, chose a small block to keep things simple, cut
your array into thin row slices, padded a ragged row count on the host, or
padded 64-wide heads up to the 128-lane minor axis. This page collects
measurements of those choices on one v6e chip, and of two that turned out
not to matter at all. Every kernel was
written in JAX Pallas with BlockSpec pipelining and called from PyTorch
through torch_tpu's `pallas.jax_op`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip, in two separate
sessions on two hosts. Above one, the alternative is faster by that factor;
below one, it is slower and the number is the share of the baseline's speed
it keeps. Each experiment checked every arm's output against a
full-precision reference before timing it.

**Read the baselines.** Several experiments here were measured against a
baseline that no careful kernel author would choose: a single buffer where
Pallas double-buffers by default, 16 KB DMA chunks, 8-row slices, single
(8, 128) vector-register tiles. They are labelled below. Their ratios are
true, and they are the cost of getting the choice wrong, not a speedup you
will see over ordinary practice.

## Measured

### Keep the pipeline double-buffered: the default is right

Pallas pipelines a BlockSpec kernel by giving every input and output two
buffers, so the next tile's copy overlaps this tile's compute. Each
experiment below set the buffer count explicitly with `pl.Buffered`, and
each took a single buffer as its baseline. That baseline is not the
default: it is what you get by turning buffering down, for example to fit
a larger block in VMEM.

- **A streaming VPU kernel** over a bf16 array of 1024 columns, in 512 by
  1024 tiles, was 2.07× faster with two
  buffers on its input and its output than with one,[^1]
  and 1.56× faster with two on the input
  alone.[^2]
- **A tiled bf16 matmul** (256 by 256 by 256 tiles, contraction depth 1024)
  was 2.3× faster double-buffered on both
  sides than single-buffered,[^3] and
  1.87× faster with only its inputs
  double-buffered.[^4]
- **Starting from the default instead**, a triad kernel reading three bf16
  inputs and writing one: dropping the output to a single buffer kept
  0.787 of the default's
  speed,[^5] and dropping the three inputs to one buffer
  each kept 0.545.[^6]

So the output's buffers matter as well as the inputs'. Every experiment
expected at most a quarter's gain from overlapping the copy with the
compute, and measured far more. A gain above 2x is more than overlapping
two activities can give, and it follows from how a single buffer is
lowered: `pl.Buffered(1)` is a synchronous pipeline, so each step issues
its input copy, waits for it, computes, then copies the output out and
waits again. Three activities run one after another, each DMA's issue and
wait latency is exposed, and double buffering both sides overlaps all of
it. Double-buffering the inputs alone stayed under 2x (1.56x and 1.87x).

If VMEM is tight, shrinking the block is probably cheaper than taking a
buffer away from a block whose index changes every step, but no experiment
here compared the two, and none was compute-heavy. A block whose index
never changes (weights, scalars) is not re-fetched between steps, so a
second buffer for it only costs VMEM. An earlier buffering experiment on a
matmul is in `index.py get v6e-measured-pallas-kernel-internals`.

### Move data in large blocks

How much each copy carries mattered more than anything else measured here.

- **DMA chunk size.** Streaming a bf16 array of 2048 columns in 1 MB blocks
  (256 rows by 2048) rather than 16 KB blocks (8 rows by 1024) was
  36.8× faster.[^7] The
  16 KB baseline is far below anything a kernel would normally use; read
  the ratio as the price of tiny blocks.
- **Vector-register-sized tiles.** An elementwise fp32 polynomial over 2048
  columns, tiled at the minimum (8, 128) block, one vector register, was
  27.6× slower than the same kernel at
  (128, 512).[^8] Again a deliberately minimal baseline.
- **Row slices.** Streaming a bf16 array of 2048 rows through BlockSpec
  windows of 8 rows was the slowest of four window heights at every row
  width tried, 128 to 8192 columns. Against 8-row slices, 16-row slices
  were 1.81× faster,[^9]
  64-row blocks 2.88×
  faster[^10] and 256-row blocks faster still, by an
  amount that grows with the row width: 2.49× at
  128 columns[^11] and 5.04× at
  8192.[^12] A straight line fitted across the four
  widths holds them within a few percent and gives
  3.78× at 4096
  columns:[^13] there is no single "256-row speedup", it
  depends on how wide the rows are. Between 64 and 256 rows it made little
  difference except on the widest rows, where 256 led
  (4.09× for 64 rows[^14]).
- **Smaller tiles, not fewer passes.** In a streaming kernel over bf16
  matrices of 8192 to 32768 rows, 128 by 256 tiles kept
  0.41 of the speed of 256 by 512
  tiles.[^15]

Rule of thumb from these streaming kernels: make each block hundreds of
kilobytes, with at least 64 rows, before you reach for anything clever.
The next section is an exception: on ragged row counts a 128-row block
beat a 512-row one.

### Ragged row counts: the block size decided it, not the padding

The rows numbered 512k + 128 (2176, 4224 and 8320), against a natural
block of 512 rows by 1024 columns. The baseline padded the array on the
host to a multiple of 512 and sliced the result afterwards.

- **Masking the tail inside the kernel**, still with 512-row blocks, cost
  the same as padding on the host:
  0.999.[^16]
- **One bulk call at 512 rows plus a tail call at 128** gained little,
  1.06×.[^17]
- **128-row blocks that divide the row count exactly** were
  2.38× faster than the padded
  baseline.[^18]

Since avoiding the pad by masking bought nothing, the gain in the last case
came from the 128 by 1024 block itself, not from skipping the pad and the
slice. That is the opposite of the previous section's advice for this one
shape, and the experiment did not try a 128-row block on a row count that
512 divides, so it cannot say why. For a ragged row count, try a block
height that divides it before you pad.

### Narrow heads: pack two into the lanes, or let XLA do it

An elementwise gate over attention heads 64 wide (bf16, 16 or 32 heads, up
to 8192 tokens), where the minor axis is half the 128 lanes.

- Packing two heads side by side into a 128-wide minor axis was
  2.6× faster than zero-padding each head
  to 128 lanes,[^19] with no copy to make the packing.
- Leaving the gate to XLA on the 64-wide layout was faster still:
  6.96× faster than the padded Pallas
  kernel,[^20] from 3.07× at
  2048 tokens[^21] to 7.9× at
  8192.[^22]

For an elementwise operation this simple, XLA was the better choice in
every cell; if the gate lives inside a larger Pallas kernel, pack the heads
rather than padding them.

### Broadcasts belong in VMEM

An affine broadcast `(X + bias) * scale` over fp32 matrices of 2048 and
4096: splatting the bias and scale inside the kernel was
2.05× faster than first writing them out
as full 2-D arrays in HBM.[^23] The baseline is the
naive version; XLA would not normally materialise a broadcast like this.

### One output, and Pallas matches XLA

A STREAM-style triad reading 2, 4 or 8 bf16 inputs and writing their
weighted sum:

- A Pallas kernel writing one output ran at the same speed as XLA's fused
  version: 1 with four
  inputs[^24] and 1 with
  eight.[^25]
- Splitting the work into a kernel with two outputs kept only
  0.678 of the one-output kernel's
  speed.[^26]

For these pure streams Pallas matched XLA but did not beat it, at two
shapes; write one output per pass.

### Two things that do not matter

- **Grid order.** Walking 256 by 512 tiles of matrices up to 32768 by
  16384 column-major rather than row-major cost the same:
  1.[^27] At this tile size there was no
  HBM page-locality penalty to design around; other tile sizes were not
  compared across orders.
  Column-major order with smaller 128 by 256 tiles was slower, but only
  because of the tiles: it kept 0.41 of the
  speed,[^28] the same as row-major order at that size.
- **Where the index map gets its coordinates.** Reading each tile's block
  coordinates from a table in SMEM (scalar prefetch) rather than computing
  them from the grid indices cost the same:
  1.01.[^29] In these two cells (128 by 128 blocks) a lookup
  table cost nothing, so use one when the access pattern needs it.


## Believed, not measured

- The buffering results are what overlapping copy with compute predicts
  when the copy and the compute take similar time, plus the latency a
  synchronous single buffer exposes. No experiment here varied the ratio
  of compute to copy to test that.
- The large-block results fit a fixed cost per DMA issued, which small
  blocks pay many more times. The experiments changed block size and byte
  count together, so they cannot separate a per-DMA cost from HBM burst or
  page effects.
- Why 128-row blocks beat 512-row blocks on the ragged row counts is not
  established.

## Never measured

- Buffer counts above two, and `pl.Buffered` on a kernel whose compute
  clearly outweighs its copies.
- Manual DMA with `pltpu.async_copy` and semaphores instead of BlockSpec
  pipelining.
- Block heights between 16 and 64 rows, and a 128-row block on row counts
  that a 512-row block divides.
- Narrow heads inside a larger fused kernel such as attention, where XLA
  is not an option.
- More than one chip, other TPU generations, and the backward pass.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.
[^1]: two buffers on input and output against one: median ratio 2.07, cells 2.06 to 2.08, over 2 cells, 2 sessions each.
[^2]: two buffers on the input only against one: median ratio 1.56, range [1.56, 1.57], over 2 cells, 2 sessions each.
[^3]: matmul double-buffered against single-buffered: median ratio 2.3, range [2.3, 2.31], over 2 cells, 2 sessions each.
[^4]: matmul inputs double-buffered against single-buffered: median ratio 1.87, cells 1.86 to 1.87, over 2 cells, 2 sessions each.
[^5]: single-buffered output against the default: median ratio 0.787, cells 0.783 to 0.79, over 2 cells, 2 sessions each.
[^6]: single-buffered inputs against the default: median ratio 0.545, range [0.537, 0.55], over 2 cells, 2 sessions each.
[^7]: 1 MB blocks against 16 KB blocks: median ratio 36.8, cells 34.5 to 39.2, over 2 cells, 2 sessions each.
[^8]: (128, 512) tiles against (8, 128): median ratio 27.6, range [27, 28.1], over 2 cells, 2 sessions each.
[^9]: 16-row slices against 8-row: median ratio 1.81, cells 1.73 to 1.89, over 4 cells, 2 sessions each.
[^10]: 64-row blocks against 8-row slices: median ratio 2.88, cells 2.51 to 4.09, over 4 cells, 2 sessions each.
[^11]: 256-row blocks, 128 columns: ratio 2.49 [2.39, 2.59], 2 sessions; M = 2048; N = 128.
[^12]: 256-row blocks, 8192 columns: ratio 5.04 [4.98, 5.11], 2 sessions; M = 2048; N = 8192.
[^13]: 256-row blocks against 8-row slices: a straight line in row width fitted to the four measured widths (128, 512, 2048 and 8192 columns), every point within 2.5% of it; 3.78 at 4096 columns is read off the line, not measured.
[^14]: 64-row blocks, 8192 columns: ratio 4.09 [4.09, 4.09], 2 sessions; M = 2048; N = 8192.
[^15]: 128 by 256 tiles against 256 by 512: median ratio 0.41, range [0.407, 0.411], over 4 cells, 2 sessions each.
[^16]: in-kernel tail mask against host padding: median ratio 0.999, cells 0.997 to 1, over 4 cells, 2 sessions each.
[^17]: bulk call plus tail call against host padding: median ratio 1.06, range [1.02, 1.09], over 4 cells, 2 sessions each.
[^18]: exact 128-row blocks against host padding: median ratio 2.38, range [1.97, 2.48], over 4 cells, 2 sessions each.
[^19]: two heads packed against per-head padding: median ratio 2.6, range [2.41, 2.65], over 4 cells, 2 sessions each.
[^20]: XLA on the 64-wide layout against padded Pallas: median ratio 6.96, range [2.89, 7.97], over 4 cells, 2 sessions each.
[^21]: XLA against padded Pallas, 2048 tokens: ratio 3.07 [2.89, 3.24], 2 sessions; BS = 256; D = 64; H = 16; S = 2048.
[^22]: XLA against padded Pallas, 8192 tokens: ratio 7.9 [7.85, 7.95], 2 sessions; BS = 256; D = 64; H = 16; S = 8192.
[^23]: broadcast in VMEM against materialised in HBM: median ratio 2.05, cells 2.04 to 2.06, over 2 cells, 2 sessions each.
[^24]: XLA's fused triad against Pallas, four inputs: ratio 1 [0.992, 1.01], 2 sessions; BM = 512; BN = 1024; K_streams = 4; M = 2048; N = 4096.
[^25]: XLA's fused triad against Pallas, eight inputs: ratio 1 [0.999, 1.01], 2 sessions; BM = 512; BN = 1024; K_streams = 8; M = 4096; N = 4096.
[^26]: two outputs against one: median ratio 0.678, range [0.565, 0.706], over 4 cells, 2 sessions each.
[^27]: column-major against row-major order: median ratio 1, range [0.998, 1], over 4 cells, 2 sessions each.
[^28]: column-major 128 by 256 tiles against row-major 256 by 512: median ratio 0.41, range [0.407, 0.411], over 4 cells, 2 sessions each.
[^29]: affine index map against SMEM table: median ratio 1.01, range [1.01, 1.01], over 2 cells, 2 sessions each.
