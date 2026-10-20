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
entry: v6e-measured-pallas-compile-and-buffers
keyword: v6e-measured-pallas-compile-and-buffers
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# My Pallas kernel on TPU v6e takes long to compile, or the loop, launches and buffers around it cost more than the kernel itself

Your Pallas kernel is correct, but something around it is slow. The first
call takes seconds to compile and gets worse as you add steps to a loop.
Or the kernel itself is quick, and the step is not: elementwise work before
and after it, a loop you drive from Python, one launch per iteration. Or you
have tried the knobs people mention, a bigger VMEM limit, letting XLA fuse
into the kernel's inputs, `vmap` instead of a batch grid, aliasing the
output onto the input, and cannot tell which of them mattered. Or the
kernel is called from PyTorch and the casts, transposes and copies on the
way in cost as much as the kernel. This page
collects measurements of those choices on a TPU v6e.

Every experiment ran on one v6e chip, in a single process, in two sessions
on two different hosts. Kernels were written in JAX Pallas; except where a
section says otherwise they were called from PyTorch through torch_tpu's
`pallas.jax_op`. Every number is a ratio: the time with the baseline
divided by the time with the alternative, on the same chip in the same
process. Above one, the alternative is faster by that factor. Below one,
it is slower, and the number is the share of the baseline's speed it
keeps. The two compile-time experiments time cold compilation
(`jax.jit(...).lower().compile()` with a fresh cache key each time), not the
compiled kernel. Every arm was checked against a reference before it was
timed.

What the kernel does inside, tiles, scalar tables and accumulators, is in
`index.py get v6e-measured-pallas-kernel-internals`. How it moves data is in
`index.py get v6e-measured-pallas-kernels` and
`index.py get v6e-measured-pallas-memory-pipeline`.

## Measured

### Keep loops rolled: unrolling with `range()` costs compile time per step

A Pallas kernel whose MXU and VMEM-scratch loop over a 4096 by 2048 problem
was either kept rolled in `lax.fori_loop` (the baseline) or unrolled in
Python with `for i in range(N_steps)`. Below one means the unrolled kernel
took longer to compile:

| steps | rolled against fully unrolled | rolled against unrolled in chunks of 4 |
|---|---|---|
| 8 | 0.73[^1] | 0.788[^2] |
| 16 | 0.492[^3] | 0.786[^4] |
| 32 | 0.295[^5] | 0.785[^6] |
| 64 | 0.162[^7] | 0.789[^8] |

Full unrolling falls further behind the rolled loop with every step it
unrolls. Unrolling in chunks of 4 keeps a constant gap,
0.787 across all four step
counts:[^9] its cost does not grow with the number of
steps.

A power law fitted across the four step counts holds them within 6% and
puts the rolled loop's compile time at about 0.35
of the fully unrolled one's at 24 steps:[^10] the cost of
unrolling keeps growing with every step unrolled, with no sign of levelling
off by 64.

The same shows on a multi-stage MXU and VPU kernel: written as a compact
`fori_loop` over a one-dimensional grid, it compiled
4.71× faster than the same stages unrolled
in Python over a three-dimensional grid with four VMEM scratch
buffers,[^11] 3.89× at 16
stages[^12] and 5.54× at
24.[^13] If an agent or a sweep compiles many candidate
kernels, write the loop as a `fori_loop` and unroll only a small fixed
chunk.

### Compile a kernel once and keep it

An 8-stage VMEM loop kernel on bf16 tensors of 8192 and 16384 rows,
compiled cold each time with a fresh cache key, against running only the
staged part of ahead-of-time compilation (lower to IR and serialize it).
The two arms do different work: the staged path stops before the backend
compile that the cold arm pays for. The staged path was 24.1×
faster,[^14] 23.9× at 8192
rows[^15] and 24.3× at
16384.[^16] Loading an already compiled executable from a
cache was faster still; that comparison produced no usable figure, so it
carries no number here. Read both as the compile cost avoided, not as a faster kernel:
the cold compile is the baseline because it is what a fresh process pays. A
search that compiles the same candidate twice, or a service that restarts,
should keep the compiled executable.

### Run a multi-step loop inside one kernel, not one launch per step

- **An iterative update of a 128 by 128 state tile over a 2048 by 2048
  float32 tensor,** kept in VMEM across a `fori_loop` inside one kernel,
  against relaunching a one-step kernel that reads and writes the state
  through HBM each step: 5.8×
  faster,[^17] 3.93× at 4
  steps[^18] and 7.67× at
  8.[^19]
- **An 8-step elementwise recurrence on small bf16 tensors,** fused into one
  in-VMEM loop, against eight single-step kernels dispatched one after
  another from the host: 6.48×
  faster,[^20] 8.14× at 8192
  rows[^21] and 4.81× at
  16384.[^22]
- **Eight elementwise steps as eight chained `pallas_call`s inside one
  `jit`**, on bf16 tensors of width 1024, against the same eight steps as
  one kernel that keeps the tile in VMEM:
  3.48× faster,[^23]
  2.36× at 2048 rows[^24] and
  4.6× at 4096.[^25] Letting
  XLA fuse all eight steps did as well:
  3.57× faster than the chain
  overall,[^26] 2.37× and
  4.77× at the two
  sizes.[^27][^28] Being inside one
  `jit` does not make separate kernels free.

The first two baselines are the simplest way to write the loop, one launch
per step, not a tuned alternative. The gain grows with the number of steps in the first case and is largest for
the smaller tensor in the second, which is where a fixed cost per launch
weighs most. Read these as what fusing the loop saves over launching per
step, not as a gain over a tuned alternative. The chained-call case is the
opposite way round: its gap grows from the smaller tensor to the larger,
which points at the round trip of each step's output through HBM more than
at a fixed cost per call.

### Fuse the elementwise work around a kernel into it

- **Elementwise pre- and post-processing around a kernel on float32
  matrices**, moved inside the Pallas kernel instead of left to XLA around
  `pallas_call`: 1.36×
  faster.[^29]
- **A chain of elementwise Pallas layers**, run as one kernel that keeps the
  intermediate activations in VMEM instead of one kernel per layer:
  3.51× faster,[^30] from
  2.3× with 4 layers[^31] to
  3.65× with 8 on the largest
  tensor.[^32]
- **A scale-and-shift prologue feeding a kernel on bf16 tensors,** written
  inside the kernel by hand, against leaving it outside with
  `allow_input_fusion` off: 2.29× faster at 16384
  by 1024[^33] and 2.28× at
  16384 by 512,[^34] 1.35× at
  16384 by 256,[^35] and no faster,
  1, at 8192 by
  256.[^36] That is as fast as letting XLA fuse the whole
  thing with no Pallas kernel at all: 2.31×,
  2.29× and 1.34× on
  the first three shapes.[^37][^38][^39]
  The experiment expected a gain at every shape; the smallest shape showed
  none. Only one prologue type, a scale and shift, was tested.

### `allow_input_fusion` did not do that fusion for you

In the same experiment, turning `allow_input_fusion=[True, True]` on
instead of writing the prologue into the kernel made no difference at three
of the four shapes, 1 at 16384 by
1024,[^40] 1 at 16384 by
512[^41] and 1 at 8192 by
256,[^42] and gave 1.34× at
16384 by 256.[^43] Where the manual fusion paid most, the
flag paid nothing. For a scale-and-shift prologue, write it into the kernel;
other prologue shapes were not tried.

### Settings that made no difference

- **`vmem_limit_bytes`.** An isolated tiled bf16 matmul whose tiles need
  about 4 MiB of VMEM ran in the same time with the limit left at its
  default or set to 16, 32, 64 or 96 MiB: 1.00 against the default in every
  case,[^44][^45][^46][^47] inside the 1.6% noise floor at every shape. In an
  isolated kernel whose blocks already fit, raising the limit changes
  nothing. In a full program it also sets how much scoped VMEM XLA reserves
  for the call, which can affect the operations around it; that was not
  measured here.
- **`jax.vmap` over a `pallas_call`** against prepending the batch to the
  grid by hand, for a batched matmul and SiLU kernel:
  1.00.[^48] Use whichever
  reads better.
- **Hoisting a causal mask out of a `fori_loop`** gained
  1.05× for an `iota` mask[^49]
  and 1.05× for a float `tril`
  mask:[^50] small. What does matter is how the mask is
  built inside the loop: a float `tril` built there kept
  0.858 of the speed of an `iota`
  comparison.[^51] Build masks from `iota` comparisons;
  hoisting them is optional.

### Across the PyTorch boundary

When PyTorch calls a kernel through `pallas.jax_op`, what happens to the
tensors on the way in can cost as much as the kernel:

- **Casting fp32 operands to bf16 before a bf16 matmul kernel,** with
  `torch.autocast` producing bf16 operands up front, against an eager
  `.to(torch.bfloat16)` right before the call:
  1.27× faster,[^52]
  1.32× at 2048 rows[^53] and
  1.23× at 4096.[^54] Casting
  inside the kernel, in VMEM, also beat the eager cast,
  1.1× at 2048 rows;[^55] its
  4096-row cell produced no usable figure.
- **Feeding a kernel the transpose of a tensor,** with
  `.transpose().contiguous()` in PyTorch before the call as the baseline.
  Reading the transposed blocks through the kernel's BlockSpec and
  transposing them in VMEM was 1.34×
  faster,[^56] 1.34× at width
  2048[^57] and 1.34× at
  4096.[^58] Doing the transpose in JAX inside the
  `jax_op`, where XLA compiles it, was faster again:
  1.52× over the
  baseline,[^59] 1.54× and
  1.5× at the two
  widths.[^60][^61] When an operand must be transposed before a kernel
  called through this bridge, do it inside the JAX program (1.52×) or in
  the kernel (1.34×), not with a PyTorch `.contiguous()`.
- **Defensive `.clone()` of the inputs and outputs** around a kernel on
  bf16 tensors of 4096 by 4096, against handing the buffers over directly:
  1.00, the same.[^62] The experiment expected at least 1.26× for the
  direct handoff; at this size the clones cost nothing measurable. The
  8192-row cell produced no usable figure. The clones are a deliberately
  cautious baseline. The same experiment ran the operation as torch_tpu's own elementwise op, as
  a yardstick: the Pallas kernel with its clones took
  0.435 of its time at 4096
  rows[^63] and 0.435 at
  8192.[^64]

### Aliasing the output onto an input: three different answers

`input_output_aliases={0: 0}` asks Pallas to write the result over an input
buffer. Every case below was called from PyTorch through torch_tpu's
`pallas.jax_op`, and what it bought depended on what the kernel does:

- **Updating 256 rows of a large float32 state buffer in place,** against an
  out-of-place kernel that has to write the whole buffer:
  3.47× faster,[^65]
  3.03× with 2048 rows[^66] and
  3.9× with 4096.[^67] When
  aliasing lets the kernel touch only part of a buffer, it pays.
- **A chain of elementwise layers, each aliased onto its input,** against
  the same chain out of place:
  1.00.[^68] No
  difference: every layer still reads and writes the whole tensor.
- **A residual update, `y = x + 0.5 * res` on bf16 rows of width 2048,**
  aliased onto `x`, against writing a fresh output: at 4096 rows the aliased
  kernel kept 0.604 of the out-of-place
  speed.[^69] The experiment expected no difference. Only the 4096-row
  cell produced a usable figure.

So, across this bridge, alias when it lets the kernel skip work, as in
the slice update. When every byte is read and written anyway it bought
nothing, and in the residual update it cost speed. The same kind of loss
for an aliased relu is in `index.py get v6e-measured-pallas-kernel-internals`.
None of this was measured with kernels called from JAX directly.

## Believed, not measured

- The compile-time results are consistent with Mosaic's cost growing with
  the size of the kernel's lowered program: unrolling copies the loop body
  once per step, a `fori_loop` keeps one copy, and unrolling by a fixed
  chunk copies it a fixed number of times. The program size itself was not
  measured.
- The launch-per-step results are consistent with a fixed host and
  dispatch cost per kernel call plus a round trip of the state through HBM
  per step. The two were not separated.
- Why aliasing slowed the residual update is not established. One
  explanation is a protective copy made somewhere between PyTorch and the
  kernel, so that the donated input survives for any other reference to
  it; another is that the alias changes how the pipeline overlaps reading
  `res` with writing `x`. Neither has been tested. That defensive clones
  across the same bridge cost nothing measurable at a similar size weighs
  against the protective-copy explanation, but does not settle it: a copy
  made by the alias may not behave like a clone made in PyTorch. A sweep
  over the buffer size, which would show whether the loss tracks one extra
  read and write of the buffer, would separate them; it has not been run.
- Why a transpose done in JAX inside the `jax_op` beat one done in VMEM in
  the kernel is not established. XLA may fold it into the layout of the
  buffer it hands the kernel, so that no transpose runs at all.
- `allow_input_fusion` presumably fuses only some producer shapes into the
  kernel's inputs; which ones it accepted here is not known.

## Never measured

- Compile time against kernel run time for the same kernel, and whether
  the compilation cache removes the unrolling cost on a second run.
- How much a cached executable saves, as a number: the cache-hit
  comparison produced no usable figure.
- The PyTorch-boundary costs at other sizes and dtypes, and with
  `torch.compile` around the call.
- Unroll chunks other than 4, and loops longer than 64 steps.
- The residual-update aliasing at other sizes, in float32, called from JAX
  directly instead of through torch_tpu, and with a profile.
- `allow_input_fusion` with prologues other than a scale and shift, and
  epilogue fusion.
- `vmem_limit_bytes` for kernels whose blocks do not fit the default limit.
- `vmem_limit_bytes` in a full program, where it changes the scoped VMEM
  XLA reserves around the call.
- More than one chip, other chips than the v6e, and end-to-end models.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: rolled against fully unrolled, 8 steps: ratio 0.73 [0.73, 0.73], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 8.
[^2]: rolled against chunk-4 unroll, 8 steps: ratio 0.788 [0.774, 0.802], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 8.
[^3]: rolled against fully unrolled, 16 steps: ratio 0.492 [0.485, 0.5], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 16.
[^4]: rolled against chunk-4 unroll, 16 steps: ratio 0.786 [0.786, 0.786], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 16.
[^5]: rolled against fully unrolled, 32 steps: ratio 0.295 [0.294, 0.295], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 32.
[^6]: rolled against chunk-4 unroll, 32 steps: ratio 0.785 [0.783, 0.788], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 32.
[^7]: rolled against fully unrolled, 64 steps: ratio 0.162 [0.162, 0.163], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 64.
[^8]: rolled against chunk-4 unroll, 64 steps: ratio 0.789 [0.782, 0.796], 2 sessions; BM = 256; BN = 256; M = 4096; N = 2048; N_steps = 64.
[^9]: rolled against chunk-4 unroll, all step counts: median ratio 0.787, range [0.774, 0.802], over 4 cells, 2 sessions each.
[^10]: rolled fori_loop against full range() unroll: power law a·N^k with a = 3.47, k = -0.724, fitted to the four step counts in the table (worst residual 5.5%); 24 steps is interpolated, not measured.
[^11]: compact fori_loop against unrolled stages, both sizes: median ratio 4.71, cells 3.89 to 5.54, over 2 cells, 2 sessions each.
[^12]: 16 stages: ratio 3.89 [3.88, 3.9], 2 sessions; M = 1024; N = 256; STAGES = 16.
[^13]: 24 stages: ratio 5.54 [5.5, 5.58], 2 sessions; M = 1024; N = 256; STAGES = 24.
[^14]: staged AOT lower and serialize against cold compile, both sizes: median ratio 24.1, cells 23.9 to 24.3, over 2 cells, 2 sessions each.
[^15]: staged AOT against cold compile, 8192 rows: ratio 23.9 [23.8, 24.1], 2 sessions; BM = 512; BN = 256; M = 8192; N = 512; STAGES = 8.
[^16]: staged AOT against cold compile, 16384 rows: ratio 24.3 [23.9, 24.7], 2 sessions; BM = 512; BN = 256; M = 16384; N = 512; STAGES = 8.
[^17]: VMEM loop against per-step relaunch, both step counts: median ratio 5.8, range [3.93, 7.67], over 2 cells, 2 sessions each.
[^18]: 4 steps: ratio 3.93 [3.93, 3.93], 2 sessions; BM = 128; BN = 128; M = 2048; N = 2048; steps = 4.
[^19]: 8 steps: ratio 7.67 [7.67, 7.67], 2 sessions; BM = 128; BN = 128; M = 2048; N = 2048; steps = 8.
[^20]: fused loop against 8 host dispatches, both sizes: median ratio 6.48, range [4.68, 8.15], over 2 cells, 2 sessions each.
[^21]: 8192 rows: ratio 8.14 [8.13, 8.15], 2 sessions; M = 8192; N = 512; STEPS = 8.
[^22]: 16384 rows: ratio 4.81 [4.68, 4.95], 2 sessions; M = 16384; N = 512; STEPS = 8.
[^23]: one VMEM-fused kernel against 8 chained pallas_calls, both sizes: median ratio 3.48, range [2.36, 4.63], over 2 cells, 2 sessions each.
[^24]: VMEM-fused, 2048 rows: ratio 2.36 [2.36, 2.36], 2 sessions; K_CALLS = 8; M = 2048; N = 1024.
[^25]: VMEM-fused, 4096 rows: ratio 4.6 [4.56, 4.63], 2 sessions; K_CALLS = 8; M = 4096; N = 1024.
[^26]: XLA-fused against 8 chained pallas_calls, both sizes: median ratio 3.57, range [2.36, 4.95], over 2 cells, 2 sessions each.
[^27]: XLA-fused, 2048 rows: ratio 2.37 [2.36, 2.37], 2 sessions; K_CALLS = 8; M = 2048; N = 1024.
[^28]: XLA-fused, 4096 rows: ratio 4.77 [4.59, 4.95], 2 sessions; K_CALLS = 8; M = 4096; N = 1024.
[^29]: fused kernel against XLA around pallas_call: median ratio 1.36, cells 1.36 to 1.37, over 2 cells, 2 sessions each.
[^30]: one VMEM-fused kernel against one kernel per layer: median ratio 3.51, cells 2.3 to 3.65, over 4 cells, 2 sessions each.
[^31]: VMEM-fused, 4 layers: ratio 2.3 [2.28, 2.32], 2 sessions; BM = 512; BN = 1024; L_layers = 4; M = 4096; N = 1024.
[^32]: VMEM-fused, 8 layers, largest tensor: ratio 3.65 [3.61, 3.69], 2 sessions; BM = 512; BN = 1024; L_layers = 8; M = 8192; N = 2048.
[^33]: manual prologue, 16384 by 1024: ratio 2.29 [2.28, 2.29], 2 sessions; BM = 1024; M = 16384; N = 1024.
[^34]: manual prologue, 16384 by 512: ratio 2.28 [2.27, 2.28], 2 sessions; BM = 1024; M = 16384; N = 512.
[^35]: manual prologue, 16384 by 256: ratio 1.35 [1.34, 1.35], 2 sessions; BM = 1024; M = 16384; N = 256.
[^36]: manual prologue, 8192 by 256: ratio 1 [0.994, 1.01], 2 sessions; BM = 1024; M = 8192; N = 256.
[^37]: XLA-fused baseline, 16384 by 1024: ratio 2.31 [2.3, 2.31], 2 sessions; BM = 1024; M = 16384; N = 1024.
[^38]: XLA-fused baseline, 16384 by 512: ratio 2.29 [2.29, 2.3], 2 sessions; BM = 1024; M = 16384; N = 512.
[^39]: XLA-fused baseline, 16384 by 256: ratio 1.34 [1.34, 1.35], 2 sessions; BM = 1024; M = 16384; N = 256.
[^40]: flag on against off, 16384 by 1024: ratio 1 [0.993, 1.01], 2 sessions; BM = 1024; M = 16384; N = 1024.
[^41]: flag on against off, 16384 by 512: ratio 1 [0.994, 1.01], 2 sessions; BM = 1024; M = 16384; N = 512.
[^42]: flag on against off, 8192 by 256: ratio 1 [0.992, 1.01], 2 sessions; BM = 1024; M = 8192; N = 256.
[^43]: flag on against off, 16384 by 256: ratio 1.34 [1.34, 1.34], 2 sessions; BM = 1024; M = 16384; N = 256.
[^44]: 16 MiB against default: median ratio 1, cells 1 to 1, over 4 cells, 2 sessions each.
[^45]: 32 MiB against default: median ratio 1, cells 1 to 1, over 4 cells, 2 sessions each.
[^46]: 64 MiB against default: median ratio 1, cells 1 to 1, over 4 cells, 2 sessions each.
[^47]: 96 MiB against default: median ratio 1, cells 1 to 1, over 4 cells, 2 sessions each.
[^48]: manual batch grid against vmap: ratio 1 [0.999, 1], 2 sessions; B = 16; BM = 256; M = 1024; N = 512.
[^49]: iota mask hoisted against inside the loop: median ratio 1.05, cells 1.02 to 1.05, over 4 cells, 2 sessions each.
[^50]: tril mask hoisted against iota inside the loop: median ratio 1.05, cells 1.02 to 1.05, over 4 cells, 2 sessions each.
[^51]: tril mask inside against iota inside the loop: median ratio 0.858, range [0.857, 0.938], over 4 cells, 2 sessions each.
[^52]: autocast pre-cast against eager cast, both sizes: median ratio 1.27, range [1.23, 1.32], over 2 cells, 2 sessions each.
[^53]: autocast pre-cast, 2048 rows: ratio 1.32 [1.32, 1.32], 2 sessions; BK = 256; BM = 256; BN = 256; K = 1024; M = 2048; N = 2048.
[^54]: autocast pre-cast, 4096 rows: ratio 1.23 [1.23, 1.23], 2 sessions; BK = 256; BM = 256; BN = 256; K = 1024; M = 4096; N = 2048.
[^55]: in-kernel VMEM cast against eager cast, 2048 rows: ratio 1.1 [1.1, 1.1], 2 sessions; BK = 256; BM = 256; BN = 256; K = 1024; M = 2048; N = 2048.
[^56]: in-VMEM BlockSpec transpose against PyTorch transpose().contiguous(), both widths: median ratio 1.34, cells 1.34 to 1.34, over 2 cells, 2 sessions each.
[^57]: in-VMEM transpose, width 2048: ratio 1.34 [1.34, 1.34], 2 sessions; BT = 256; M = 4096; N = 2048.
[^58]: in-VMEM transpose, width 4096: ratio 1.34 [1.34, 1.34], 2 sessions; BT = 256; M = 4096; N = 4096.
[^59]: JAX transpose inside the jax_op against PyTorch transpose().contiguous(), both widths: median ratio 1.52, range [1.48, 1.56], over 2 cells, 2 sessions each.
[^60]: JAX transpose inside the jax_op, width 2048: ratio 1.54 [1.52, 1.56], 2 sessions; BT = 256; M = 4096; N = 2048.
[^61]: JAX transpose inside the jax_op, width 4096: ratio 1.5 [1.48, 1.52], 2 sessions; BT = 256; M = 4096; N = 4096.
[^62]: direct handoff against defensive clones, 4096 rows: ratio 1 [0.997, 1.01], 2 sessions; BM = 512; BN = 1024; M = 4096; N = 4096.
[^63]: Pallas with clones against torch_tpu's own elementwise op, 4096 rows: ratio 0.435 [0.435, 0.436], 2 sessions; BM = 512; BN = 1024; M = 4096; N = 4096.
[^64]: Pallas with clones against torch_tpu's own elementwise op, 8192 rows: ratio 0.435 [0.435, 0.436], 2 sessions; BM = 512; BN = 1024; M = 8192; N = 4096.
[^65]: aliased slice update against full out-of-place update: median ratio 3.47, range [3.03, 3.9], over 2 cells, 2 sessions each.
[^66]: 2048 rows: ratio 3.03 [3.03, 3.03], 2 sessions; BM = 128; BN = 128; M = 2048; N = 2048; UPD_M = 256.
[^67]: 4096 rows: ratio 3.9 [3.9, 3.9], 2 sessions; BM = 128; BN = 128; M = 4096; N = 2048; UPD_M = 256.
[^68]: aliased chain against out of place: median ratio 1, cells 1 to 1, over 4 cells, 2 sessions each.
[^69]: aliased against fresh output, 4096 rows: ratio 0.604 [0.601, 0.607], 2 sessions; BM = 256; M = 4096; N = 2048.
