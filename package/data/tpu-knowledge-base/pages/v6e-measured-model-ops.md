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
entry: v6e-measured-model-ops
keyword: v6e-measured-model-ops
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# Which way should I build the non-attention parts of my model layer on TPU v6e? RoPE, LoRA and QKV, vocabulary loss and sampling, Mamba-2, activation dtype, epilogues and weight sharing

You are building or tuning a transformer or state-space model on a TPU v6e
and the attention kernel is already settled (that is covered in
`index.py get v6e-measured-attention-serving`). What is left are the smaller choices around it:
which rotary embedding layout to use, whether to fuse a LoRA adapter or the
Q, K and V projections, how to compute cross-entropy and sample over a large
vocabulary, how to run a Mamba-2 scan, whether to run activations in bf16,
what a fused SwiGLU epilogue costs, and what sharing a weight across matmuls
saves. This page collects measurements of those choices. Every experiment ran on one v6e chip, in a single process, in two
separate sessions; most compare plain JAX/XLA against a JAX Pallas kernel
called from PyTorch through torch_tpu's `pallas.jax_op`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative, both measured on the same chip in the same
process. Above one, the alternative is faster by that factor. Below one, the
alternative is slower, and the number is the share of the baseline's speed
it keeps. Where the arms compute the same result, each experiment checked
every arm against a full-precision reference before timing it; the
exceptions are said where they occur.

## Measured

### Use split-half RoPE, and keep a precomputed sin/cos table

The workload applied rotary position embedding to bf16 rows of width 256
(two halves of 128), with fp32 arithmetic, over 16,384 or 65,536
token-head rows. The baseline was GPT-J style interleaved RoPE in XLA, which
pairs adjacent elements (`x[:, 0::2]`, `x[:, 1::2]`) and so needs strided
gathers and a re-interleave.

- **Split-half (Llama/NeoX) layout in XLA**, which pairs the two contiguous
  128-wide halves: 8.15× faster on
  average,[^1] from 5.12× at
  16,384 rows[^2] to 11.2× at
  65,536.[^3]
- **Split-half layout as a Pallas kernel**: about the same,
  8.06× faster on
  average,[^4] from 5.12×[^5]
  to 11×.[^6] Both were timed against the same interleaved baseline rather
  than against each other, but the two ratios agree at both row counts: once
  the layout is split-half, writing a kernel buys nothing measurable over
  XLA.
- **If you must keep the interleaved layout**, a Pallas kernel that
  de-interleaves with permutation matmuls on the MXU is
  2.51× faster[^7] than
  XLA's strided version, at both row counts. That recovers well under half
  of what switching layout gives.

The two layouts are different RoPE conventions, not two ways of computing
one result: switching means your Q and K projection weights must be laid out
for split-half. If you control the checkpoint, use split-half.

Computing sin and cos on the fly inside the kernel, instead of streaming a
precomputed table from HBM, saves table reads but costs transcendental work.
It lost: on-the-fly synthesis kept 0.684 of
the table version's speed on average,[^8] and got worse
as the problem grew, 0.812 at 16,384
rows[^9] and 0.557 at
65,536.[^10] Keep a precomputed table rather than computing sin and cos per
element at run time. This is a different question from whether the table
should be a constant the compiler can see or a buffer passed in at run time;
that was not measured here.

### LoRA and Q/K/V fusion pays at 1024 tokens and loses at 4096

The projection had contraction 2048 and output 2048 in bf16, at 1024 or 4096
tokens, with 512 by 512 by 512 Pallas tiles. The baseline ran each matmul as
its own `jax_op` call.

**A base matmul plus a rank-128 LoRA adapter** (`X@W + (X@A)@B`), against
three separate calls (ratio of the three calls' time to the fused time;
above one the fusion is faster, below one it keeps that share of the speed):

| tokens | XLA, one graph | Pallas, accumulated in VMEM |
|---|---|---|
| 1024 | 2.41[^11] | 2.26[^12] |
| 4096 | 0.835[^13] | 0.7[^14] |

At 1024 tokens either fused form is more than twice as fast; at 4096 both
are slower than three separate calls, and the Pallas kernel is the slower of
the two. The Pallas kernel recomputes the small `X@A` product once per output
column block, so at 4096 tokens it does somewhat more arithmetic than the
baseline. If you fuse LoRA, use the XLA graph, and time it at your real
token count: at these shapes the answer flips between 1024 and 4096.

**One Pallas kernel computing Q, K and V from a shared X tile**, against three
separate projection calls: 1.12× faster at 1024
tokens,[^15] and keeping only
0.847 of the speed at 4096.[^16]
At 4096 tokens, three plain calls win.

### Vocabulary cross-entropy: a two-pass Pallas kernel does not beat XLA

The vocabulary was 131,072 columns in bf16, at a batch of 64 or 256 rows,
with Pallas tiles 64 rows by 4096 columns.

- **Log-sum-exp in two passes**, the same algorithm in XLA and in Pallas:
  the Pallas kernel kept 0.628 of XLA's
  speed,[^17] 0.651 at 64
  rows[^18] and 0.605 at
  256.[^19]
- **The cross-entropy plus z-loss gradient**: a chunked Pallas version that
  computes an online log-sum-exp and then overwrites the logits with the
  gradient in place (`input_output_aliases`) kept
  0.789 of the speed[^20]
  of XLA computing it with materialised intermediates, at both batch sizes.
  The in-place version saves memory, not time.

A single-pass online log-sum-exp kernel, the design usually recommended for
a Pallas cross-entropy (`index.py get kernel-streaming-reduction`), was not
measured, and neither was a kernel that fuses the LM-head matmul with the
cross-entropy so the logits are never written to HBM. Over logits already in
HBM, the two-pass Pallas kernel lost to XLA; whether the single-pass or fused
designs win is open.

### Sampling over the vocabulary: never sort the whole vocabulary

Same vocabulary, batch 64 or 256, top-k of 128 and top-p of 0.9.

**Drawing one token.** The baseline perturbed the logits with Gumbel noise,
took a softmax, a cumulative sum over all 131,072 columns, and picked the
crossing point. The alternative takes the argmax of the logits plus Gumbel
noise directly (the Gumbel-max trick), which draws from the same softmax
distribution without a softmax or a cumulative sum. The two samplers agree in
distribution; they return the same token only when they share their
randomness, which this page does not establish.

- In XLA: 18.8× faster on
  average,[^21] 17.2× at batch
  64[^22] and 20.3× at
  256.[^23]
- As a streaming Pallas kernel: 16.4×
  faster on average,[^24] from
  15×[^25] to
  17.9×,[^26] slightly less than
  the XLA version. Write `argmax(logits + gumbel)` in XLA.

**Top-k then top-p.** The baseline sorted all 131,072 columns, kept the top
128, and applied the nucleus mask.

- `jax.lax.top_k` with k = 128 instead of a full sort:
  2.7× faster on
  average,[^27] 2.9× at batch
  64[^28] and 2.5× at
  256.[^29]
- A Pallas kernel that keeps only the best four candidates in each of the
  128 lanes while streaming the vocabulary, then picks the top 128 from those
  512: 197× faster than the full sort on
  average,[^30] from 195×[^31]
  to 200×.[^32] This kernel is
  not an exact top-128: a true top-128 logit is lost whenever more than four
  of them fall in the same lane. It passed a tolerance check against the
  reference on random logits, but it does less work than the other two arms,
  so read its ratio as the price of an approximation, not as a faster way to
  compute the same thing. If you need exact top-k, use `lax.top_k`.

**Min-p filtering and normalization.** Over 16,384-wide bf16 logits at
batch 256 or 512 with min-p 0.1: one Pallas pass that finds the row
maximum, masks below the threshold and normalizes in VMEM, against a
three-pass version that writes to HBM between passes. The three-pass
baseline is closer to naive than to what a careful author would write, so
the XLA arm is the one to read.

- The single-pass kernel: 1.97× faster on
  average,[^33] 1.59× at batch
  256[^34] and 2.36× at
  512.[^35]
- XLA's own fused version barely improved on the three passes:
  1.06× on
  average,[^36] 1.08× at batch
  256[^37] and 1.04× at
  512.[^38] Here XLA does not fuse the passes for you, so
  the Pallas kernel is worth writing, more so at the larger batch.

### Mamba-2: chunk the scan, and fuse the prologue

The layer was Mamba-2's state-space duality (SSD) with 32 streams, head and
state width 128, chunks of 128 tokens, at 2048 or 4096 tokens, bf16
inputs and an fp32 state. The baseline scanned token by token with
`lax.scan`, which no careful implementation would do: read the two large
ratios below as the cost of that baseline, and compare them with each
other.

- **Chunked SSD in XLA**, which turns each chunk into 128-wide matmuls and
  scans only across chunks: 39.2×
  faster,[^39] the same at both lengths.
- **Chunked SSD as a Pallas kernel** that keeps the 128 by 128 state in VMEM
  across chunks: 56.7×
  faster,[^40] again the same at both lengths. The kernel and chunked XLA
  were each timed against the token scan, not against each other; the two
  ratios put the kernel about 1.45× ahead of chunked XLA at these shapes.
- **Fusing the prologue** (the width-4 causal depthwise convolution and SiLU
  gating) into the SSD kernel, instead of running it as a separate call that
  writes its output to HBM: 1.4×
  faster.[^41]

### In a Pallas loop, activations in bf16 were slower than in fp32

The kernel applied an activation (silu, gelu_quick, gelu_tanh or gelu_erf)
repeatedly to a block held in VMEM, 1000 or 11,000 passes, in fp32 or on the
bf16 cast of the same values, with the element count held fixed. Pooled over
the four activations, bf16 kept 0.69 of fp32's speed at 1000 passes, with
individual runs between 0.63 and 0.83,[^42] and the same at 11,000 passes
(runs between 0.63 and 0.83).[^43] The penalty did not shrink with more
passes, so in this loop it is paid on every evaluation, not once per kernel.
Inside a Pallas kernel that evaluates activations many times, keep the
activation math in fp32 and convert at the kernel's boundary. How this
carries over to a real MLP, where each activation runs once between matmuls,
was not measured. The arms agree to a 5% tolerance rather than exactly,
because they compute in different precisions.

### A fused SwiGLU epilogue is nearly free

A Pallas kernel accumulated two matmuls (4096 by 4096 output, depth 256,
1024 or 4096, bf16 in, fp32 accumulators) and ended either in a plain add or
in `silu(acc1) * acc2`. The SwiGLU ending kept
0.972[^44],
0.955[^45] and
0.986[^46] of the plain add's
speed at the three depths. In this kernel, fusing the gate cost 1.4% to
4.5%; other epilogues and kernels were not measured.

### Sharing a weight across matmuls pays only in a middle band of sizes

Four bf16 matmuls of 16 rows by depth K by 1024 columns, compiled into one
graph with `torch.compile`, used either one weight four times or four
different weights. The arms do the same arithmetic but multiply by
different matrices, so their outputs differ by design and were not
compared; this measures what distinct weights cost, not a way to compute the
same thing. At 16 rows the matmuls are limited by reading weights.

| depth K | four weights keep this share of one weight's speed |
|---|---|
| 2048 | 0.98[^47] |
| 4096 | 0.96[^48] |
| 8192 | 0.722[^49] |
| 16384 | 0.272[^50] |
| 32768 | 0.185[^51] |
| 65536 | 1[^52] |

Small weights cost the same either way, and at the largest size the saving
is gone again. In between, reusing one weight across these 16-row matmuls
was worth up to about 5× the speed. This was measured at one batch height
(16 rows) under `torch.compile`, the mechanism was not established, and why
the saving vanishes at depth 65536 is unexplained, so treat it as a reason
to measure weight reuse at your own shapes rather than as a rule.

## Believed, not measured

- The interleaved RoPE layout is slow because pairing adjacent elements
  needs strided access, which the TPU's vector registers do not support
  directly, so XLA inserts gathers and relayouts. Split-half pairs two
  contiguous halves that each fill the 128 lanes.
- On-the-fly sin/cos likely loses because the transcendental work costs
  more than the table reads it saves; why it worsens with more rows is not
  established.
- The LoRA and Q/K/V fusions likely win at 1024 tokens because three calls
  pay three dispatch overheads that dominate a small problem, and lose at
  4096 because the fused kernels' tiling or extra work costs more than the
  saved traffic once the matmuls dominate. Neither part has been isolated.
- XLA's own reductions over the vocabulary appear to be good enough that a
  hand-written Pallas kernel with the same algorithm does not beat them.
- The full sort is expensive because a sorting network over the whole
  vocabulary does far more work than selecting a few candidates.
- The bf16 activation penalty is consistent with the vector unit
  evaluating transcendentals in fp32, so each call converts bf16 in and
  out. The per-activation pattern reads as a cost paid at each
  transcendental; that was not tested directly.
- The weight-sharing band is consistent with a finite on-chip working set:
  when four weights fit, both arms read everything once; when only one fits,
  sharing saves three reads; when not even one fits, both arms stream from
  HBM. Where exactly the capacity sits was not located.

## Never measured

- More than one chip, and chips other than the v6e.
- The single-pass online log-sum-exp kernel, and an LM-head matmul fused with
  the cross-entropy. The experiment set out to time the single-pass kernel,
  but no result for it was recorded.
- The split-half Pallas RoPE kernel against split-half XLA directly, and the
  Mamba-2 kernel against chunked XLA directly.
- Exact top-k in Pallas, and sampling at other vocabulary sizes, batch sizes
  or values of k and p, or min-p at other thresholds.
- RoPE at other head widths, partial rotary, or fused into the Q/K
  projection.
- LoRA at other ranks, token counts between 1024 and 4096, other projection
  sizes, or several adapters batched together.
- Mamba-2 decode (one token at a time), other chunk sizes and state widths,
  and the backward pass.
- Activation dtype inside a real MLP, as opposed to a repeated loop, in XLA
  rather than Pallas, and other activations.
- The SwiGLU epilogue against XLA, and epilogues other than SwiGLU.
- Weight sharing at batch heights above 16 and at depths between the octave
  steps measured.
- Backward passes for everything here, and end-to-end model throughput.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: XLA split-half against XLA interleaved: median ratio 8.15, range [5.07, 11.3], over 2 cells, 2 sessions each.
[^2]: XLA split-half against XLA interleaved, 16,384 rows: ratio 5.12 [5.07, 5.17], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 16384.
[^3]: XLA split-half against XLA interleaved, 65,536 rows: ratio 11.2 [11, 11.3], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 65536.
[^4]: Pallas split-half against XLA interleaved: median ratio 8.06, range [5.11, 11.1], over 2 cells, 2 sessions each.
[^5]: Pallas split-half against XLA interleaved, 16,384 rows: ratio 5.12 [5.11, 5.13], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 16384.
[^6]: Pallas split-half against XLA interleaved, 65,536 rows: ratio 11 [10.9, 11.1], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 65536.
[^7]: Pallas interleaved against XLA interleaved: median ratio 2.51, cells 2.46 to 2.56, over 2 cells, 2 sessions each.
[^8]: in-kernel sin/cos against an HBM table: median ratio 0.684, range [0.552, 0.822], over 2 cells, 2 sessions each.
[^9]: in-kernel sin/cos against an HBM table, 16,384 rows: ratio 0.812 [0.803, 0.822], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 16384.
[^10]: in-kernel sin/cos against an HBM table, 65,536 rows: ratio 0.557 [0.552, 0.561], 2 sessions; BM = 1024; D = 256; D_half = 128; N = 65536.
[^11]: XLA fused LoRA against three calls, 1024 tokens: ratio 2.41 [2.38, 2.44], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 1024; N = 2048; R = 128.
[^12]: Pallas fused LoRA against three calls, 1024 tokens: ratio 2.26 [2.25, 2.26], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 1024; N = 2048; R = 128.
[^13]: XLA fused LoRA against three calls, 4096 tokens: ratio 0.835 [0.832, 0.839], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 4096; N = 2048; R = 128.
[^14]: Pallas fused LoRA against three calls, 4096 tokens: ratio 0.7 [0.689, 0.711], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 4096; N = 2048; R = 128.
[^15]: shared-X Pallas QKV against three calls, 1024 tokens: ratio 1.12 [1.08, 1.16], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 1024; N = 2048; R = 128.
[^16]: shared-X Pallas QKV against three calls, 4096 tokens: ratio 0.847 [0.845, 0.849], 2 sessions; BK = 512; BM = 512; BN = 512; K = 2048; M = 4096; N = 2048; R = 128.
[^17]: Pallas two-pass LSE against XLA two-pass LSE: median ratio 0.628, range [0.599, 0.666], over 2 cells, 2 sessions each.
[^18]: Pallas two-pass LSE against XLA, batch 64: ratio 0.651 [0.636, 0.666], 2 sessions; B = 64; BM = 64; BV = 4096; V = 131072.
[^19]: Pallas two-pass LSE against XLA, batch 256: ratio 0.605 [0.599, 0.611], 2 sessions; B = 256; BM = 64; BV = 4096; V = 131072.
[^20]: Pallas in-place CE gradient against XLA materialised: median ratio 0.789, cells 0.78 to 0.798, over 2 cells, 2 sessions each.
[^21]: XLA Gumbel-max against softmax and CDF: median ratio 18.8, cells 17.2 to 20.3, over 2 cells, 2 sessions each.
[^22]: XLA Gumbel-max against softmax and CDF, batch 64: ratio 17.2 [16.5, 17.9], 2 sessions; B = 64; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^23]: XLA Gumbel-max against softmax and CDF, batch 256: ratio 20.3 [20.3, 20.4], 2 sessions; B = 256; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^24]: Pallas Gumbel-max against softmax and CDF: median ratio 16.4, cells 15 to 17.9, over 2 cells, 2 sessions each.
[^25]: Pallas Gumbel-max against softmax and CDF, batch 64: ratio 15 [15, 15], 2 sessions; B = 64; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^26]: Pallas Gumbel-max against softmax and CDF, batch 256: ratio 17.9 [17.6, 18.1], 2 sessions; B = 256; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^27]: lax.top_k nucleus against full sort: median ratio 2.7, cells 2.5 to 2.9, over 2 cells, 2 sessions each.
[^28]: lax.top_k nucleus against full sort, batch 64: ratio 2.9 [2.9, 2.9], 2 sessions; B = 64; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^29]: lax.top_k nucleus against full sort, batch 256: ratio 2.5 [2.5, 2.5], 2 sessions; B = 256; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^30]: approximate Pallas lane top-k against full sort: median ratio 197, cells 195 to 200, over 2 cells, 2 sessions each.
[^31]: approximate Pallas lane top-k against full sort, batch 64: ratio 195 [195, 195], 2 sessions; B = 64; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^32]: approximate Pallas lane top-k against full sort, batch 256: ratio 200 [200, 200], 2 sessions; B = 256; K = 128; V = 131072; block_b = 64; block_v = 4096.
[^33]: min-p, single-pass Pallas in VMEM against three passes through HBM: median ratio 1.97, range [1.52, 2.36], over 2 cells, 2 sessions each.
[^34]: min-p, single-pass Pallas, batch 256: ratio 1.59 [1.52, 1.66], 2 sessions; B = 256; BM = 16; BN = 16384; V = 16384.
[^35]: min-p, single-pass Pallas, batch 512: ratio 2.36 [2.36, 2.36], 2 sessions; B = 512; BM = 16; BN = 16384; V = 16384.
[^36]: min-p, XLA fused against three passes through HBM: median ratio 1.06, range [1.03, 1.09], over 2 cells, 2 sessions each.
[^37]: min-p, XLA fused, batch 256: ratio 1.08 [1.06, 1.09], 2 sessions; B = 256; BM = 16; BN = 16384; V = 16384.
[^38]: min-p, XLA fused, batch 512: ratio 1.04 [1.03, 1.05], 2 sessions; B = 512; BM = 16; BN = 16384; V = 16384.
[^39]: Mamba-2, XLA chunked SSD against token-by-token lax.scan: median ratio 39.2, cells 39.1 to 39.2, over 2 cells, 2 sessions each.
[^40]: Mamba-2, Pallas SSD with VMEM state against token-by-token lax.scan: median ratio 56.7, range [56.2, 57.2], over 2 cells, 2 sessions each.
[^41]: Mamba-2, conv and SiLU prologue fused into the SSD kernel against a separate call: median ratio 1.4, range [1.4, 1.41], over 2 cells, 2 sessions each.
[^42]: bf16 against fp32, 1000 passes, four activations pooled: ratio 0.69, individual runs 0.627 to 0.828, 2 sessions; lanes = 128; passes = 1000; rows = 32768; sublanes = 128.
[^43]: bf16 against fp32, 11,000 passes, four activations pooled: ratio 0.688, individual runs 0.625 to 0.826, 2 sessions; lanes = 128; passes = 11000; rows = 32768; sublanes = 128.
[^44]: SwiGLU against add epilogue, depth 256: ratio 0.972 [0.971, 0.974], 2 sessions; K = 256; M = 4096; N = 4096.
[^45]: SwiGLU against add epilogue, depth 1024: ratio 0.955 [0.954, 0.956], 2 sessions; K = 1024; M = 4096; N = 4096.
[^46]: SwiGLU against add epilogue, depth 4096: ratio 0.986 [0.984, 0.988], 2 sessions; K = 4096; M = 4096; N = 4096.
[^47]: four weights against one shared, K 2048: ratio 0.98 [0.978, 0.981], 2 sessions; K = 2048; M = 16; N = 1024; reuse = 4.
[^48]: four weights against one shared, K 4096: ratio 0.96 [0.96, 0.961], 2 sessions; K = 4096; M = 16; N = 1024; reuse = 4.
[^49]: four weights against one shared, K 8192: ratio 0.722 [0.713, 0.732], 2 sessions; K = 8192; M = 16; N = 1024; reuse = 4.
[^50]: four weights against one shared, K 16384: ratio 0.272 [0.27, 0.274], 2 sessions; K = 16384; M = 16; N = 1024; reuse = 4.
[^51]: four weights against one shared, K 32768: ratio 0.185 [0.183, 0.187], 2 sessions; K = 32768; M = 16; N = 1024; reuse = 4.
[^52]: four weights against one shared, K 65536: ratio 1 [0.987, 1.01], 2 sessions; K = 65536; M = 16; N = 1024; reuse = 4.
