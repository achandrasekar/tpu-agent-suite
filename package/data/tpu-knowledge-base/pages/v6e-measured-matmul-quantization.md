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
entry: v6e-measured-matmul-quantization
keyword: v6e-measured-matmul-quantization
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My matmuls on TPU v6e are slower than I expected, and I am not sure whether quantizing, reshaping or changing dtype will help

You have a model on a TPU v6e and most of its time goes into matrix
multiplies. You are deciding whether to quantize the weights to int8 or int4,
whether fp8 is worth storing, how big to make the blocks of a Pallas kernel,
what a decode step with only a few rows really costs, and what you give up by
keeping some matmuls in f32. This page collects measurements on one v6e chip
about those choices: which ones pay, which pay less than the arithmetic
suggests, and which make things slower.

Every number below is a ratio of times unless the sentence says it is a rate:
the time with the baseline divided by the time with the alternative, both
measured on the same chip in the same process, in two sessions on separate
machines. Above one, the alternative is faster by that factor. Below one, the
alternative is slower, and the number is the fraction of the baseline's speed
it keeps. Before anything was timed, each experiment wrote down a prediction
worked out from v6e's published characteristics (see *Believed, not
measured*); where the measurement agreed or disagreed with it, that is noted
in words.

## Measured

### Should I quantize to int8?

**Both operands in int8 (int8 times int8, accumulating in int32).** This compares
a bf16 matmul accumulating in f32 against the same shape with both inputs in
int8, so the MXU runs its int8 mode. The test data are small integers, so both
versions give exactly the same answer.

- **Large square matmuls in plain XLA** (`jnp.dot` at 4096 and 8192 on every
  side): int8 is 1.95× faster, the median of the two
  sizes,[^1] 2.03× at 4096[^2] and 1.87× at 8192.[^3] This is what the
  prediction expected: v6e's published int8 rate is twice its bf16 rate.
- **A few rows against a large weight** (128 rows, a K=8192 by N=16384 weight,
  limited by reading the weight from HBM): int8 is 1.94× faster, because the
  operands are half the bytes.[^4] This also matches the prediction.
- **A tiled Pallas kernel** at 8192 on every side (2048 by 2048 output tiles,
  K in steps of 512, the output tile accumulated in place): int8 is only
  1.54× faster.[^5] Here the int8 kernel reaches a smaller share of its own
  peak than the bf16 kernel does of its peak: as a rate, the bf16 kernel's
  share is 1.3× the int8 kernel's.[^6] The prediction expected the full
  doubling, so a hand-written kernel at this tiling leaves part of the int8
  gain unused, while XLA gets nearly all of it.

**Weights only in int8 (bf16 activations, per-channel scales).** The weights
are stored as int8 with one f32 scale per output column. Inside the matmul
each int8 weight is converted to bf16, multiplied in bf16 with an f32
accumulator, and the result is scaled. The MXU runs bf16 in both versions, so
the only thing saved is weight bytes read from HBM. The decode-shaped result
below is consistent with XLA fusing the int8-to-bf16 conversion into the
matmul, so that the weight is read at one byte per element; the compiled
program was not inspected to confirm it.

- **Decode-shaped** (16 rows, K=8192 by N=16384 weight, where reading the
  weight is nearly all the cost): in plain XLA the int8 weights are
  1.96× faster,[^7] and in a Pallas kernel that converts the int8 tile to
  bf16 in VMEM and applies the scales at the last K step, 1.95× faster.[^8]
  Both are what halving the weight bytes predicts: the conversion is hidden
  under the weight read.
- **Prefill-shaped** (8192 rows, K=4096 by N=4096, limited by arithmetic): in
  plain XLA the int8 weights are 1.06× faster,[^9] which is close to no
  change, as predicted: fewer weight bytes buy little when arithmetic is the
  limit, and the conversion costs nothing visible. The same shape in Pallas
  was not measured.

### Should I quantize weights to int4?

v6e's published specification gives a bf16 and an int8 rate and no int4
rate, so int4 is a storage format on v6e: it must be unpacked to bf16 (or
int8) before the MXU, and the most it can save is HBM bytes. The kernel here
is a Pallas kernel with int4 weights packed two to a byte. In VMEM each
weight tile is unpacked with integer shifts, multiplied by a bf16 scale shared
by a group of rows along K, and then contracted on the MXU in bf16 with an f32
accumulator. The weight is K=8192 by N=16384, with 64 or 128 rows of
activations, so the bf16 version is limited by reading the weight.

- **int4 (groups of 128) against bf16 weights:** 1.11× faster, the median of
  the two row counts,[^10] 1.1× at 64 rows[^11] and 1.11× at 128 rows.[^12]
  The prediction, from a weight a quarter the size, was a several-fold
  speedup. The measurement falls far short of it, and short of what
  weight-only int8 gained at a decode shape above. In this kernel the HBM
  read is evidently no longer what sets the pace once the weight shrinks; the
  unpacking and scaling in VMEM are the likely cost, though no experiment
  here separates them.
- **Group size.** With groups of 256 as the baseline, groups of 128 cost the
  same,[^13] groups of 64 run at 0.976 of the speed,[^14] and groups of 32 at
  0.935.[^15] Smaller groups add scale bytes and a little unpacking work, and
  the cost is about what the extra scale bytes predicted: small, with no
  cliff down to groups of 32.

### Is fp8 worth it?

**Not for arithmetic.** v6e's own specification lists only bf16 and int8
rates; the TPU7x comparison table lists v6e fp8 equal to bf16. So an fp8 matmul
cannot be faster than bf16 on this chip. In
practice it was much slower. A torch_tpu `torch.matmul` on fp8 operands
(`float8_e4m3fn`), compiled through XLA, against the identical matmul on bf16,
ran at 0.16 of bf16's speed at 2048 on a side,[^16] 0.144 at 4096[^17] and
0.158 at 8192.[^18] Only at 512 on a side, where neither is limited by
arithmetic, are they close (0.964).[^19] What makes the fp8 path slow was not
measured here; layout copies XLA inserts around 8-bit operands are one
documented cause (`index.py get xla-8bit-layout-copies`). Keep matmuls in
bf16, or int8 where it fits.

**As a storage format, the question is whether fp8 bytes move as cheaply as
any other bytes.** A Pallas kernel copies an array HBM to VMEM to HBM in
blocks; within each comparison every version moves the same total bytes in
blocks of the same byte size, only the element type differs.

- **fp8 against bf16:** the same cost, at every total size and block size
  measured: 1 with small blocks,[^20] 0.998 and 0.997 at mid
  sizes,[^21][^22] and 0.998 for the largest copy.[^23] The worry going in,
  that fp8 is packed four to a word and might need a relayout, did not show.
- **f32 against bf16:** also the same cost per byte, at all five
  sizes.[^24]

So in a Pallas copy a byte is a byte on v6e. Storing weights in fp8 halves
the bytes of bf16, and a step limited by moving those bytes should get
cheaper accordingly. That last step is reasoning, not a measurement: no
matmul with fp8 weights converted to bf16 is on this page.

### How should I shape the matmul?

**Pallas block size: use 256 or larger.** Each comparison computes the same
4096-cubed bf16 matmul with an f32 output used as the accumulator.

- **Square blocks of 256 instead of 128** (on all three sides):
  5.74× faster.[^25]
- **Square blocks of 512 instead of 256:** 4.04× faster.[^26]
  Both are close to the prediction, and the prediction said why: most of this
  is not MXU occupancy (v6e's MXU is 256 by 256) but the fixed cost of each
  grid step and re-reading tiles from HBM, which small blocks multiply. One
  shape was measured; how much a larger block buys at other shapes depends on
  how many grid steps it removes.
- **MXU occupancy on its own.** Here the blocks are held fixed (1024 by 1024
  output, 512 along K) and only the depth of each `jnp.dot` inside the kernel
  changes, with the pieces summed in f32. Dots 256 deep instead of 128 deep
  are 1.27× faster,[^27] less than predicted. Dots 512 deep instead of 256
  deep cost the same,[^28] as predicted. A contraction shallower than the
  MXU leaves part of it idle; deeper than 256 gains nothing.

**Unroll the K reduction in Pallas, within reason.** Each comparison computes
the same 16384 by 2048 times 2048 by 128 product, bf16 in and f32
accumulated, reducing K in sixteen chunks. The baseline writes the sixteen
dots out with a Python `for` loop, so they are unrolled into one block. The
alternative rolls them into `jax.lax.fori_loop`, one dot per iteration, which
Pallas lowers to a rolled loop unless `unroll=` is passed. Only the block
height (rows per grid step, and so the size of the live accumulator) changes
between comparisons.

- The rolled loop is slower at every block height: across the five heights it
  runs at 0.351 of the unrolled speed in the middle case.[^29] The penalty
  shrinks as blocks grow, from 0.216 of the speed at 8 rows[^30] and 0.247 at
  32,[^31] through 0.351 at 128,[^32] to 0.649 at 512[^33] and 0.809 at
  1024.[^34]
- The concern this experiment came from was the opposite: that unrolling
  keeps every partial sum live and spills registers once the accumulator is
  large. No sign of that appeared at any size tried. The penalty behaves like
  a roughly fixed cost per iteration, which bigger blocks amortise; that
  reading is an inference from the trend, not a separate measurement.
- Unrolling is not free at compile time. In a separate measurement on v6e,
  fully unrolling a Pallas MXU loop with Python `range()` made cold
  compilation 1.4× slower than `lax.fori_loop` at 8 steps and 6× slower at
  64 (`index.py get v6e-measured-pallas-compile-and-buffers`). For long loops,
  `lax.fori_loop(..., unroll=k)` is the middle ground.

**Decode-shaped matmuls: a few rows cost somewhat less than many.** Plain XLA
`x @ w` in torch_tpu against a fixed bf16 weight, changing only the number of
rows in `x`.

- With a 16384 by 16384 weight, one row took 0.886 of the time of 128
  rows[^35] and 0.825 of the time of 256 rows.[^36] Reading the weight
  dominates, so 128 to 256 times the rows cost only 13% to 21% more time.
  The prediction expected flat time up to 256 rows; the rise is larger than
  that.
- As a rate, the arithmetic done per second grows almost in step with the row
  count: one row reaches 0.125 of the rate of 8 rows[^37] and 0.0315 of the
  rate of 32.[^38] So if you can batch or pad decode rows up to a few
  hundred, each row gets much cheaper.
- Past the point where the work outweighs the weight read, time does rise:
  one row took 0.164 of the time of 2048 rows.[^39] That is more of a rise
  than a roofline at the published peak rates predicts.
- With a smaller 8192 by 8192 weight, 128 rows took 0.843 of the time of 256
  rows.[^40]

### What does f32 precision cost?

**In JAX, the precision setting on an f32 matmul.** An 8192-cubed f32
`jnp.dot`, with only `jax.lax.Precision` changed. On TPU, DEFAULT computes
the f32 matmul as one bf16 pass (both inputs rounded to bf16) and is the
least accurate; HIGH uses three bf16 passes and HIGHEST six, to recover f32
accuracy. Whether DEFAULT reached the MXU's peak at this size was not
measured.

- **HIGH instead of DEFAULT:** runs at 0.713 of DEFAULT's speed.[^41]
- **HIGHEST instead of DEFAULT:** runs at 0.526 of DEFAULT's speed.[^42]
- **HIGHEST instead of HIGH:** runs at 0.746 of HIGH's speed.[^43]

Pass counts are MXU work, not wall-clock time: HIGH took 1.4× and HIGHEST
1.9× DEFAULT's time, where three and six passes would suggest 3× and 6×. The
measured times fit a large fixed cost per matmul that does not scale with the
pass count, which also suggests DEFAULT f32 at this size runs well below the
MXU's bf16 peak; that reading is an inference, as no absolute rate was
recorded. Extra precision is still a real cost, so ask for HIGH or HIGHEST
only where you need the accuracy.

### Which dtype should a plain matmul use?

**In torch_tpu at its defaults,** a 2048-cubed `a @ b` with only the dtype
changed, f32 as the baseline.

- **bf16 instead of f32:** 1.99× faster.[^44] torch_tpu's default f32
  matmul precision is one bf16 pass, so the gain is not from fewer passes. At
  2048 on a side an f32 matmul is limited by memory rather than arithmetic,
  and bf16 halves the bytes; that is the likely source of the 2×.
- **fp16 instead of f32:** 1.49× faster.[^45] fp16 and bf16 were never timed
  against each other directly, but against the same f32 baseline fp16 gained
  less than bf16, against the prediction that the two 16-bit types would
  track each other. Prefer bf16.

## Believed, not measured

Every prediction above was worked out from v6e's published characteristics
together with what the chip reports about itself through Pallas: its peak
rates for bf16 and int8, its memory bandwidth, its one TensorCore with two
MXUs, and that its specification lists no fp8 or int4 rate (the TPU7x
comparison table lists v6e fp8 equal to bf16). Those figures are vendor
statements, not measurements here.

The predictions used a roofline model (each matmul costs the larger of its
arithmetic time and its HBM time) plus, for Pallas, a fixed cost per grid
step taken from an earlier tiling experiment on the same chip.

Where the roofline held: int8 compute in XLA, int8 and weight-only int8 in
decode shapes, Pallas block sizes, and copies costing the same per byte in
every dtype. Where it did not: the int8 Pallas kernel and the int4 kernel
(both well short of the predicted speedup), f32 precision (much cheaper than
the pass count), and fp16 (gaining less than bf16).

## Never measured

- Anything beyond one chip. Every measurement here is single-chip.
- Other TPU generations, for these experiments.
- Other sizes. Each experiment used one to a few shapes; whether the ratios
  hold at other shapes, and for whole models rather than single matmuls, is
  open.
- A matmul with fp8 weights converted to bf16 inside the kernel, the
  storage-only use of fp8. This page has fp8 copies and fp8 arithmetic, but
  not that combination.
- The cause of the fp8 matmul slowdown, and fp8 matmuls written directly in
  JAX rather than through torch_tpu.
- Accuracy of the quantized models. The experiments checked that each kernel
  computes the right product, not what int8 or int4 does to a model's output.
- int4 weights in plain XLA, or an int4 kernel with a different unpacking
  scheme. The one int4 kernel here is the only one timed.
- Weight-only int8 in a Pallas kernel at the prefill shape.
- Activation quantization other than int8 by int8, and per-tensor or
  per-group int8 scales.
- fp16 against bf16 directly, and absolute matmul rates (TFLOP/s) for any arm.
- Partial unrolling (`lax.fori_loop(..., unroll=k)`) of the K reduction.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: int8 by int8 against bf16, plain XLA, both square sizes: median ratio 1.95, range [1.87, 2.03], over 2 cells, 2 sessions each.
[^2]: int8 by int8 against bf16, plain XLA, 4096 cubed: ratio 2.03 [2.02, 2.03], 2 sessions; K = 4096; M = 4096; N = 4096.
[^3]: int8 by int8 against bf16, plain XLA, 8192 cubed: ratio 1.87 [1.87, 1.87], 2 sessions; K = 8192; M = 8192; N = 8192.
[^4]: int8 by int8 against bf16, plain XLA, 128 rows: ratio 1.94 [1.94, 1.94], 2 sessions; K = 8192; M = 128; N = 16384.
[^5]: int8 by int8 against bf16, tiled Pallas kernel, 8192 cubed: ratio 1.54 [1.53, 1.54], 1 cell, 2 sessions.
[^6]: share of own peak, bf16 kernel over int8 kernel, Pallas, 8192 cubed: ratio 1.3 [1.3, 1.3], 2 sessions; K = 8192; M = 8192; N = 8192.
[^7]: int8 weights against bf16 weights, plain XLA, decode shape: ratio 1.96 [1.94, 1.97], 2 sessions; K = 8192; M = 16; N = 16384.
[^8]: int8 weights against bf16 weights, Pallas, decode shape: ratio 1.95 [1.95, 1.96], 1 cell, 2 sessions; K = 8192; M = 16; N = 16384.
[^9]: int8 weights against bf16 weights, plain XLA, prefill shape: ratio 1.06 [1.06, 1.06], 2 sessions; K = 4096; M = 8192; N = 4096.
[^10]: int4 weights (groups of 128) against bf16 weights, both row counts: median ratio 1.11, cells 1.1 to 1.11, over 2 cells, 2 sessions each.
[^11]: int4 weights against bf16 weights, 64 rows: ratio 1.1 [1.08, 1.12], 2 sessions; K = 8192; M = 64; N = 16384; block_k = 2048; block_n = 2048.
[^12]: int4 weights against bf16 weights, 128 rows: ratio 1.11 [1.1, 1.12], 2 sessions; K = 8192; M = 128; N = 16384; block_k = 2048; block_n = 2048.
[^13]: int4 groups of 128 against groups of 256: median ratio 0.992, cells 0.992 to 0.993, over 2 cells, 2 sessions each.
[^14]: int4 groups of 64 against groups of 256: median ratio 0.976, cells 0.976 to 0.977, over 2 cells, 2 sessions each.
[^15]: int4 groups of 32 against groups of 256: median ratio 0.935, cells 0.934 to 0.936, over 2 cells, 2 sessions each.
[^16]: fp8 against bf16 torch.matmul, 2048 cubed: ratio 0.16 [0.158, 0.161], 2 sessions; K = 2048; M = 2048; N = 2048.
[^17]: fp8 against bf16 torch.matmul, 4096 cubed: ratio 0.144 [0.142, 0.145], 2 sessions; K = 4096; M = 4096; N = 4096.
[^18]: fp8 against bf16 torch.matmul, 8192 cubed: ratio 0.158 [0.157, 0.158], 2 sessions; K = 8192; M = 8192; N = 8192.
[^19]: fp8 against bf16 torch.matmul, 512 cubed: ratio 0.964 [0.964, 0.964], 2 sessions; K = 512; M = 512; N = 512.
[^20]: fp8 against bf16 Pallas copy, smallest blocks: ratio 1 [0.999, 1], 2 sessions; block_bytes = 262144; bytes = 67,108,900; cols = 1024.
[^21]: fp8 against bf16 Pallas copy, smallest total, mid blocks: ratio 0.998 [0.991, 1.01], 2 sessions; block_bytes = 1,048,580; bytes = 67,108,900; cols = 1024.
[^22]: fp8 against bf16 Pallas copy, mid total, mid blocks: ratio 0.997 [0.993, 1], 2 sessions; block_bytes = 1,048,580; bytes = 268,435,000; cols = 1024.
[^23]: fp8 against bf16 Pallas copy, largest total: ratio 0.998 [0.991, 1.01], 2 sessions; block_bytes = 1,048,580; bytes = 1,073,740,000; cols = 1024.
[^24]: f32 against bf16 Pallas copy, all five sizes: median ratio 0.998, cells 0.992 to 0.999, over 5 cells, 2 sessions each.
[^25]: Pallas blocks of 256 against 128: ratio 5.74 [5.74, 5.74], 2 sessions; K = 4096; M = 4096; N = 4096.
[^26]: Pallas blocks of 512 against 256: ratio 4.04 [4.04, 4.05], 1 cell, 2 sessions; K = 4096; M = 4096; N = 4096.
[^27]: dots 256 deep against 128 deep: ratio 1.27 [1.25, 1.28], 1 cell, 2 sessions; K = 4096; M = 4096; N = 4096.
[^28]: dots 512 deep against 256 deep: ratio 1 [0.997, 1], 2 sessions; K = 4096; M = 4096; N = 4096.
[^29]: fori_loop against unrolled K reduction, all block heights: median ratio 0.351, cells 0.216 to 0.809, over 5 cells, 2 sessions each.
[^30]: fori_loop against unrolled, 8-row blocks: ratio 0.216 [0.216, 0.216], 2 sessions; block_rows = 8; k = 2048; k_chunks = 16; lanes = 128; rows = 16384.
[^31]: fori_loop against unrolled, 32-row blocks: ratio 0.247 [0.247, 0.247], 2 sessions; block_rows = 32; k = 2048; k_chunks = 16; lanes = 128; rows = 16384.
[^32]: fori_loop against unrolled, 128-row blocks: ratio 0.351 [0.351, 0.351], 2 sessions; block_rows = 128; k = 2048; k_chunks = 16; lanes = 128; rows = 16384.
[^33]: fori_loop against unrolled, 512-row blocks: ratio 0.649 [0.639, 0.658], 2 sessions; block_rows = 512; k = 2048; k_chunks = 16; lanes = 128; rows = 16384.
[^34]: fori_loop against unrolled, 1024-row blocks: ratio 0.809 [0.809, 0.809], 2 sessions; block_rows = 1024; k = 2048; k_chunks = 16; lanes = 128; rows = 16384.
[^35]: 1 row against 128 rows, 16384 by 16384 weight (time): ratio 0.886 [0.882, 0.891], 2 sessions; K = 16384; N = 16384.
[^36]: 1 row against 256 rows, 16384 by 16384 weight (time): ratio 0.825 [0.822, 0.828], 2 sessions; K = 16384; N = 16384.
[^37]: 1 row against 8 rows, 16384 by 16384 weight (rate): ratio 0.125 [0.124, 0.126], 2 sessions; K = 16384; N = 16384.
[^38]: 1 row against 32 rows, 16384 by 16384 weight (rate): ratio 0.0315 [0.0313, 0.0317], 2 sessions; K = 16384; N = 16384.
[^39]: 1 row against 2048 rows, 16384 by 16384 weight (time): ratio 0.164 [0.163, 0.164], 1 cell, 2 sessions; K = 16384; N = 16384.
[^40]: 128 rows against 256 rows, 8192 by 8192 weight (time): ratio 0.843 [0.842, 0.845], 1 cell, 2 sessions; K = 8192; N = 8192.
[^41]: Precision.HIGH against DEFAULT, f32 jnp.dot: ratio 0.713 [0.703, 0.723], 1 cell, 2 sessions; K = 8192; M = 8192; N = 8192.
[^42]: Precision.HIGHEST against DEFAULT, f32 jnp.dot: ratio 0.526 [0.475, 0.578], 2 sessions; K = 8192; M = 8192; N = 8192.
[^43]: Precision.HIGHEST against HIGH, f32 jnp.dot: ratio 0.746 [0.669, 0.822], 2 sessions; K = 8192; M = 8192; N = 8192.
[^44]: bf16 against f32, torch_tpu a @ b at defaults: ratio 1.99 [1.98, 2], 1 cell, 2 sessions; K = 2048; M = 2048; N = 2048.
[^45]: fp16 against f32, torch_tpu a @ b at defaults: ratio 1.49 [1.49, 1.49], 2 sessions; K = 2048; M = 2048; N = 2048.
