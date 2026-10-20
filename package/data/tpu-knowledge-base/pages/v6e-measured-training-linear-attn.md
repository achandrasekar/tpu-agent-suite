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
entry: v6e-measured-training-linear-attn
keyword: v6e-measured-training-linear-attn
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# Is this training-step fusion or linear-attention kernel worth writing on TPU v6e? Attention backward, optimizer steps, loss, decoder layers, linear attention and state-space kernels

You are training a model on a TPU v6e and have read about a kernel that
promises a speedup: a FlashAttention backward that saves only the
log-sum-exp, a fused optimizer step, a loss that never writes the logits,
or a chunkwise linear-attention or state-space kernel. This page collects
measurements of those choices on one v6e chip. The kernels were written in
JAX Pallas or left to XLA, and called from PyTorch through torch_tpu's
`pallas.jax_op`. For RoPE, the vocabulary loss without the projection and
Mamba-2, see `index.py get v6e-measured-model-ops`; for layout, padding and
grid order inside a Pallas kernel, see
`index.py get v6e-measured-pallas-kernels`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative. Above one, the alternative is faster by that
factor; below one, it is slower, and the number is the share of the
baseline's speed it keeps. Each experiment checked every arm's output
against a full-precision PyTorch reference before timing it. Several of the
alternatives were a single hand-written kernel; where one lost, that says
this kernel lost, not that the technique cannot win.

## Measured

### Attention backward: these two Pallas kernels lost

Both experiments used 16 batch-heads of width 128.

- **Saving only the output and log-sum-exp** in a `torch.autograd.Function`
  around Pallas forward and backward kernels, and recomputing the
  probabilities in VMEM, kept 0.573 of the
  speed[^1] of a forward and backward that saved the full
  probability matrix in HBM: 0.666 at 1024
  tokens[^2] and 0.48 at
  2048.[^3] It fell further behind at the longer length,
  where the HBM version moves the most bytes.
- **A single-pass backward** holding dQ, dK and dV accumulators in VMEM kept
  0.607 of the speed[^4] of a
  two-kernel backward (a dQ pass, then a dK and dV pass). Only the 2048-token
  cell produced a result.

These say that these particular kernels were slow, not that saving the
log-sum-exp or a single-pass backward cannot work on v6e. If you need
attention backward, start from an existing kernel and time it against the
materialized version at your length.

### Optimizer steps: fuse them, but compare against XLA first

**AdamW** with fp32 master weights and moments and a bf16 copy of the
weights, from bf16 gradients, on 16.8 and 33.6 million parameters. The
baseline ran four separate passes (first moment, second moment, weight
update, downcast).

- **One fused XLA operation**, writing new arrays:
  2.56× faster,[^5] the
  same at both sizes.
- **A Pallas kernel** updating the state in place through
  `input_output_aliases` in the same single pass kept
  0.82 of the four-pass
  speed.[^6]

The four-pass baseline is a naive form that nobody would ship; read the
2.56× as what fusion buys over it. The one comparison that matters here is
XLA's fused step against this one hand-written in-place Pallas kernel, and
XLA won. Let XLA fuse the step unless you have a Pallas kernel that beats
it at your sizes.

**Muon's five Newton-Schulz steps** on square bf16 matrices of 512 and
1024. The baseline dispatched each step separately.

- **A Pallas kernel keeping the matrix and its Gram products in VMEM**
  across all five steps: 4.11×
  faster,[^7] 4.25× at
  512[^8] and 3.97× at
  1024.[^9]
- **All five steps in one XLA loop**: 2.36×
  on average,[^10] but the gain shrinks with size, from
  3.36× at 512[^11] to
  1.36× at 1024.[^12]

Both arms are measured against per-step dispatch, a weak baseline, and the
Pallas kernel was never timed against the XLA loop directly. Against the
same baseline it gained more than the XLA loop at both sizes, and by more
at 1024, so it is worth trying, but time it against the XLA loop before
committing to it.

### Elementwise backward: one fused pass, whether or not you cache the sigmoid

The SiLU backward on bf16 activations 8192 wide, at 4096 and 8192 rows.
The baseline computed the sigmoid, the derivative factor and the product in
three HBM passes.

- **One Pallas pass reading a sigmoid cached from the forward**:
  1.87× faster.[^13]
- **One Pallas pass recomputing the sigmoid** from the input:
  2.04× faster.[^14]

Both were the same at both sizes. Recomputing looks slightly better, but
the two were not timed against each other directly; either way, the win is
the single pass, and recomputing saves storing the sigmoid.

### Output projection plus cross-entropy: don't write the logits

The final projection and loss over 2048 tokens, model width 1024, and a
vocabulary of 16,384 or 32,768. The baseline wrote the fp32 logits to HBM
and read them back for the log-sum-exp. A Pallas kernel that tiles the
vocabulary weight, multiplies in VMEM and keeps an online log-sum-exp was
1.66× faster,[^15] the
same at both vocabulary sizes. This fuses the projection with the loss; the
loss alone, starting from logits already in HBM, is covered in
`index.py get v6e-measured-model-ops`, where XLA won.

### A whole decoder layer: fewer dispatches across the PyTorch boundary

A small decoder layer (RMSNorm, RoPE, attention, residual, RMSNorm, SwiGLU
MLP, residual; model width 512, 4 heads of width 128, MLP width 512) at 512
and 1024 tokens. The baseline dispatched five stages separately across the
torch_tpu boundary, a naive form that pays a host round trip per stage. One
compiled pipeline with a Pallas residual-RMSNorm-SwiGLU-residual kernel was
3.44× faster,[^16] about the same at both lengths. Read that as the cost of
five separate dispatches at this small width, not as the Pallas kernel's
lead over a compiled layer; no wide model and no compiled-XLA layer were
tried.

### Gated linear attention: fuse the decay mask in VMEM

Chunkwise gated linear attention (GLA) over 64 batch-heads, head width 128,
chunks of 128, at 2048 and 4096 tokens.

- **A Pallas kernel that builds the cumulative decay mask in VMEM** and keeps
  the 128 by 128 state there, against XLA writing the decay masks and
  intermediates to HBM: 3.44×
  faster,[^17] the same at both lengths.
- **What the gating costs**: the same Pallas kernel without gates (no decay
  stream, a plain causal mask, no exponentials) was
  1.19× faster[^18] than
  the gated one. That is a different model, so read it as the price of
  gating inside a fused kernel, not as advice to drop the gates.

### Chunk size in chunkwise linear attention: what each size costs

A Pallas chunkwise causal linear attention over 32 batch-heads, head width
128, at 2048 and 4096 tokens, with chunks of 64, 128, 256 and 512. All
sizes compute the same output but do different amounts of arithmetic:
larger chunks do more intra-chunk work and fewer sequential steps. Even so,
512 was the fastest here, and every ratio was the same at both lengths.

| chunk | against 512: share of its speed | against 64: speedup |
|---|---|---|
| 256 | 0.669[^19] | 3.2×[^20] |
| 128 | 0.394[^21] | 1.89×[^22] |

Time fell steadily as the chunk grew, the opposite of what counting
arithmetic predicts. If you write such a kernel, try large chunks first and
sweep at your shapes; nothing above 512 was tried.

### DeltaNet, the Gated Delta Rule and the SSM backward: worth writing, by much less than a token loop suggests

All of these used 16 batch-heads, head width 128, chunks of 128, at 512
and 1024 tokens. The first experiments compared the chunkwise kernels
against a token-by-token loop on the vector unit, which is a naive
baseline: it says a per-token loop is very slow on v6e, not what the kernel
is worth. Follow-ups ran the same kernels against the standard
alternative, an XLA implementation that loops over chunks with
`jax.lax.scan` and does the chunk's work with `jnp.einsum`. Against that,
the gains are real but one to two orders of magnitude smaller, and they
grow with sequence length.

| kernel | against the XLA chunked scan | against a token loop |
|---|---|---|
| DeltaNet forward (chunkwise WY on the MXU, state in VMEM) | 1.79×[^23]: 1.43× at 512 tokens,[^24] 2.16× at 1024 tokens[^25] | 84.5×[^26]: 56.6× at 512 tokens,[^27] 112× at 1024 tokens[^28] |
| Gated Delta Rule state update (fp32, state in VMEM) | 1.75×[^29]: 1.46× at 512 tokens,[^30] 2.04× at 1024 tokens[^31] | not run |
| Decayed SSM backward (chunkwise reverse adjoint on the MXU, recompute from chunk checkpoints) | 4.6×[^32]: 4.34× at 512 tokens,[^33] 4.87× at 1024 tokens[^34] (against `jax.grad` of the chunked scan) | 103×[^35]: 96.6× at 512 tokens,[^36] 109× at 1024 tokens[^37] |

The earlier experiments, against the token loop only, agree:

- **DeltaNet forward**, chunkwise WY against one rank-1 state update per
  token in a `fori_loop`: 79.6× faster on
  average,[^38] 52.8× at 512
  tokens[^39] and 106× at
  1024.[^40]
- **The SSM backward**, chunkwise reverse adjoint scan against a
  token-by-token reverse scan and recomputation:
  101× faster on
  average,[^41] 92.2× at
  512[^42] and 110× at
  1024.[^43]

Every gain the experiments expected held. If you have a chunked XLA version of one of
these recurrences, a fused Pallas kernel that keeps the state in VMEM is
worth what the first column shows, more for the backward than the forward,
and more as sequences get longer. Never judge such a
kernel by its speedup over a per-token loop: that number measures the
loop.

### State-space kernels: the state in VMEM, the epilogue and the discretization fused

Four more state-space experiments, at 16 batch-heads (8 or 16 for the
decay guard), head width 128, chunks of 128 (256 for the decay guard).
Only the first had a genuine standard baseline. The other three were
measured against a decomposed version that writes an intermediate to HBM
and reads it back, which is a naive baseline: XLA fuses
much of that on its own. Two of them also ran an XLA arm, and that is the
comparison to read.

- **Document-boundary resets in packed sequences**, against an XLA
  `lax.scan` over chunks (a genuine standard baseline): one Pallas kernel
  that keeps the 128 by 128 fp32 state in VMEM across chunks and applies the
  document mask and the chunk reset was
  1.82× faster on
  average,[^44] 1.35× at 512
  tokens[^45] and 2.29× at
  1024.[^46] A token-by-token loop was far slower again:
  the XLA scan took 0.0267 of its time at 512
  tokens[^47] and 0.0194 at
  1024[^48] (0.023 on
  average[^49]).
- **The Mamba-2 gated GroupRMSNorm epilogue**, fused into the end of each
  chunk of the scan, against writing the scan output to HBM and running a
  separate norm-and-gate kernel: 1.53×
  faster on average,[^50] 1.33×
  at 512 tokens[^51] and 1.74×
  at 1024.[^52] The XLA arm, a chunked scan followed by the
  epilogue, was slower than even the decomposed version: that version took
  0.856 of XLA's time at 512
  tokens[^53] and 0.842 at
  1024[^54] (0.849 on
  average[^55]). So against XLA the fused kernel's lead
  is larger still. Fuse the epilogue.
- **The clamped log-decay guard in chunkwise SSM**, computing the decay
  difference and its clamped exponential inside VMEM, against materializing
  the decay and score matrices in HBM: 1.48×
  faster on average,[^56] but only
  1.07× at 8 batch-heads[^57]
  against 1.89× at 16.[^58] The experiment expected at least 1.22×; the
  8-head cell fell short. The same kernel without the clamp measured
  1.07×[^59] and 1.89×[^60] against the same baseline (1.48 on
  average[^61]). The two were not timed against each other, so the clamp's
  cost is inferred from two ratios against a shared baseline rather than
  measured, but nothing suggests it costs anything. Keep the guard.
- **Softplus and zero-order-hold discretization**, all 8 state planes
  accumulated in VMEM, against writing the discretized planes to HBM before
  the scan: 1.81× faster on
  average,[^62] 1.21× at 2048
  rows[^63] and 2.42× at
  4096.[^64]
  The experiment expected at least 1.26×; the 2048-row cell fell short.
  An XLA arm with the same steps fused was itself
  faster than the decomposed version, 1.16× at
  2048[^65] and 1.54× at
  4096[^66] (1.35 on
  average[^67]), so the Pallas kernel's lead over XLA is
  small at 2048 rows and grows at 4096. Try XLA's fusion first at small
  sizes.

In each of these the gain grew with size: tokens, batch-heads or rows.
The decomposed baselines' headline ratios measure the cost of writing the
intermediate to HBM, not the kernel's lead over what XLA would do.

## Believed, not measured

- Most of the wins above remove trips through HBM: fewer passes over the
  optimizer state, the activations or the vocabulary, no materialized logits
  or decay masks. Bandwidth was not measured directly.
- Why the hand-written Pallas attention backward kernels and the in-place
  AdamW kernel lost is not established. Untuned tile sizes and grid order
  are the usual suspects; XLA may simply have scheduled the simpler
  versions better.
- The chunked XLA scan probably loses because each step of `lax.scan`
  writes the carried state and the chunk's intermediates back to HBM, while
  the fused kernels keep them in VMEM; the gap growing with length fits
  that reading. Not profiled.
- Large linear-attention chunks probably win because each chunk is a bigger
  matmul that fills the MXU better and the sequential carry between chunks
  is shorter. There is no counter data to confirm it.
- The large decoder-layer and Muon gains are consistent with fixed per-call
  dispatch cost dominating at these small sizes; the shrinking XLA Muon gain
  at the larger matrix fits that reading.

## Never measured

- More than one chip, and chips other than the v6e.
- The cached against the recomputed SiLU sigmoid directly, and the Muon
  Pallas kernel against the XLA loop directly.
- The chunkwise kernels against an existing tuned kernel library (only an
  XLA chunked scan was tried), and at sequence lengths above 1024.
- The decay-guard and discretization kernels against an XLA version of the
  same fusion, except where noted, and the state-space kernels' backward
  passes.
- Linear-attention chunks larger than 512, GLA chunks other than 128, and
  the chunkwise kernels at long sequence lengths.
- Attention backward at other sequence lengths for the single-pass kernel,
  and against an existing tuned FlashAttention backward.
- Optimizer steps other than AdamW and Muon, and either sharded across
  chips.
- Decoder layers at realistic model widths, and end-to-end step time.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.10.1 to 0.11.2, libtpu 0.0.41 to 0.0.46, torch 2.10.0 to 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: Pallas LSE-only autograd against materialized probabilities: median ratio 0.573, cells 0.48 to 0.666, over 2 cells, 2 sessions each.
[^2]: 1024 tokens: ratio 0.666 [0.628, 0.704], 2 sessions; BH = 16; D = 128; S = 1024.
[^3]: 2048 tokens: ratio 0.48 [0.437, 0.523], 2 sessions; BH = 16; D = 128; S = 2048.
[^4]: single-pass against two-kernel backward, 2048 tokens: ratio 0.607 [0.539, 0.675], 2 sessions; BH = 16; D = 128; S = 2048.
[^5]: fused XLA against four passes: median ratio 2.56, cells 2.56 to 2.56, over 2 cells, 2 sessions each.
[^6]: in-place Pallas against four passes: median ratio 0.82, cells 0.82 to 0.82, over 2 cells, 2 sessions each.
[^7]: VMEM-resident Pallas against per-step dispatch: median ratio 4.11, range [3.41, 4.97], over 2 cells, 2 sessions each.
[^8]: Pallas, 512: ratio 4.25 [3.53, 4.97], 2 sessions; M = 512; N = 512; steps = 5.
[^9]: Pallas, 1024: ratio 3.97 [3.41, 4.52], 2 sessions; M = 1024; N = 1024; steps = 5.
[^10]: one XLA loop against per-step dispatch: median ratio 2.36, cells 1.36 to 3.36, over 2 cells, 2 sessions each.
[^11]: XLA loop, 512: ratio 3.36 [3.29, 3.44], 2 sessions; M = 512; N = 512; steps = 5.
[^12]: XLA loop, 1024: ratio 1.36 [1.17, 1.56], 2 sessions; M = 1024; N = 1024; steps = 5.
[^13]: fused pass with cached sigmoid against three passes: median ratio 1.87, cells 1.87 to 1.88, over 2 cells, 2 sessions each.
[^14]: fused pass recomputing sigmoid against three passes: median ratio 2.04, cells 2.04 to 2.05, over 2 cells, 2 sessions each.
[^15]: online Pallas CE against materialized logits: median ratio 1.66, cells 1.66 to 1.67, over 2 cells, 2 sessions each.
[^16]: fused pipeline against five stage dispatches: median ratio 3.44, range [3.2, 3.85], over 2 cells, 2 sessions each.
[^17]: Pallas VMEM decay mask against XLA materialized masks: median ratio 3.44, range [3.32, 3.54], over 2 cells, 2 sessions each.
[^18]: ungated against gated Pallas kernel (cost of gating): median ratio 1.19, cells 1.18 to 1.19, over 2 cells, 2 sessions each.
[^19]: chunk 256 against 512: median ratio 0.669, range [0.659, 0.678], over 2 cells, 2 sessions each.
[^20]: chunk 256 against 64: median ratio 3.2, range [3.17, 3.25], over 2 cells, 2 sessions each.
[^21]: chunk 128 against 512: median ratio 0.394, cells 0.39 to 0.397, over 2 cells, 2 sessions each.
[^22]: chunk 128 against 64: median ratio 1.89, range [1.86, 1.9], over 2 cells, 2 sessions each.
[^23]: chunkwise WY against XLA chunked scan: median ratio 1.79, range [1.35, 2.16], over 2 cells, 2 sessions each.
[^24]: against XLA, 512 tokens: ratio 1.43 [1.35, 1.5], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^25]: against XLA, 1024 tokens: ratio 2.16 [2.16, 2.16], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^26]: chunkwise WY against token loop: median ratio 84.5, range [53.4, 112], over 2 cells, 2 sessions each.
[^27]: against token loop, 512 tokens: ratio 56.6 [53.4, 59.7], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^28]: against token loop, 1024 tokens: ratio 112 [112, 112], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^29]: Pallas VMEM state against XLA scan: median ratio 1.75, range [1.41, 2.04], over 2 cells, 2 sessions each.
[^30]: 512 tokens: ratio 1.46 [1.41, 1.52], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^31]: 1024 tokens: ratio 2.04 [2.04, 2.04], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^32]: chunkwise adjoint against jax.grad of XLA chunked scan: median ratio 4.6, range [4.2, 4.87], over 2 cells, 2 sessions each.
[^33]: against XLA, 512 tokens: ratio 4.34 [4.2, 4.47], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^34]: against XLA, 1024 tokens: ratio 4.87 [4.87, 4.87], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^35]: chunkwise adjoint against token reverse scan: median ratio 103, cells 96.6 to 109, over 2 cells, 2 sessions each.
[^36]: against token loop, 512 tokens: ratio 96.6 [93.5, 99.6], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^37]: against token loop, 1024 tokens: ratio 109 [109, 109], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^38]: chunkwise WY against token loop: median ratio 79.6, range [47.7, 112], over 2 cells, 2 sessions each.
[^39]: 512 tokens: ratio 52.8 [47.7, 58], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^40]: 1024 tokens: ratio 106 [101, 112], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^41]: chunkwise adjoint scan against token reverse scan: median ratio 101, cells 92.2 to 110, over 2 cells, 2 sessions each.
[^42]: 512 tokens: ratio 92.2 [88.9, 95.5], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^43]: 1024 tokens: ratio 110 [110, 111], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^44]: Pallas VMEM state against XLA chunked scan: median ratio 1.82, cells 1.35 to 2.29, over 2 cells, 2 sessions each.
[^45]: against XLA, 512 tokens: ratio 1.35 [1.29, 1.41], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^46]: against XLA, 1024 tokens: ratio 2.29 [2.28, 2.29], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^47]: XLA scan against token loop, 512 tokens: ratio 0.0267 [0.0266, 0.0267], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^48]: XLA scan against token loop, 1024 tokens: ratio 0.0194 [0.0194, 0.0194], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^49]: XLA scan against token loop: median ratio 0.023, range [0.0194, 0.0267], over 2 cells, 2 sessions each.
[^50]: fused epilogue against decomposed HBM version: median ratio 1.53, cells 1.33 to 1.74, over 2 cells, 2 sessions each.
[^51]: fused against decomposed, 512 tokens: ratio 1.33 [1.31, 1.34], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^52]: fused against decomposed, 1024 tokens: ratio 1.74 [1.74, 1.74], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^53]: decomposed against XLA chunked scan plus epilogue, 512 tokens: ratio 0.856 [0.856, 0.856], 2 sessions; BH = 16; C = 128; D = 128; S = 512.
[^54]: decomposed against XLA chunked scan plus epilogue, 1024 tokens: ratio 0.842 [0.841, 0.844], 2 sessions; BH = 16; C = 128; D = 128; S = 1024.
[^55]: decomposed against XLA: median ratio 0.849, cells 0.842 to 0.856, over 2 cells, 2 sessions each.
[^56]: clamped VMEM kernel against decomposed HBM version: median ratio 1.48, cells 1.07 to 1.89, over 2 cells, 2 sessions each.
[^57]: clamped, 8 batch-heads: ratio 1.07 [1.03, 1.11], 2 sessions; BH = 8; C = 256; D = 128; S = 1024.
[^58]: clamped, 16 batch-heads: ratio 1.89 [1.89, 1.89], 2 sessions; BH = 16; C = 256; D = 128; S = 1024.
[^59]: unclamped, 8 batch-heads: ratio 1.07 [1.03, 1.1], 2 sessions; BH = 8; C = 256; D = 128; S = 1024.
[^60]: unclamped, 16 batch-heads: ratio 1.89 [1.89, 1.89], 2 sessions; BH = 16; C = 256; D = 128; S = 1024.
[^61]: unclamped VMEM kernel against decomposed HBM version: median ratio 1.48, range [1.03, 1.89], over 2 cells, 2 sessions each.
[^62]: Pallas VMEM fusion against decomposed HBM version: median ratio 1.81, range [1.2, 2.44], over 2 cells, 2 sessions each.
[^63]: Pallas, 2048 rows: ratio 1.21 [1.2, 1.22], 2 sessions; BM = 256; D = 512; K_exp = 8; M = 2048.
[^64]: Pallas, 4096 rows: ratio 2.42 [2.4, 2.44], 2 sessions; BM = 256; D = 512; K_exp = 8; M = 4096.
[^65]: XLA fused, 2048 rows: ratio 1.16 [1.15, 1.18], 2 sessions; BM = 256; D = 512; K_exp = 8; M = 2048.
[^66]: XLA fused, 4096 rows: ratio 1.54 [1.52, 1.56], 2 sessions; BM = 256; D = 512; K_exp = 8; M = 4096.
[^67]: XLA fused against decomposed HBM version: median ratio 1.35, cells 1.16 to 1.54, over 2 cells, 2 sessions each.
