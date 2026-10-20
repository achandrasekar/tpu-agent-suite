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
entry: v6e-measured-decode-tricks
keyword: v6e-measured-decode-tricks
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# Is this decode trick worth writing on TPU v6e? An int4 KV cache, sparse page selection, the paged KV append, a unified prefill-and-decode kernel and speculative decoding

You are serving a model on a TPU v6e and have read about a decode trick
that promises a speedup: an int4 KV cache, attending only to the most
relevant pages, a fused paged append, a unified prefill-and-decode kernel,
batched speculative verification or a fused rejection resample. This page
collects measurements of those choices on one v6e chip. The kernels were
written in JAX Pallas or left to XLA, and called from PyTorch through
torch_tpu's `pallas.jax_op`. For choosing an attention kernel, KV head
count and page size, see `index.py get v6e-measured-attention-serving`; for
sampling over the vocabulary, see `index.py get v6e-measured-model-ops`.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative. Above one, the alternative is faster by that
factor; below one, it is slower, and the number is the share of the
baseline's speed it keeps. Each experiment checked every arm's output
against a full-precision PyTorch reference before timing it. Several of the
alternatives were a single hand-written kernel; where one lost, that says
this kernel lost, not that the technique cannot win.

## Measured

### Decode: an int4 KV cache helps, but much less than its bytes suggest

Grouped-query decode with 16 sequences, 4 KV heads with 4 query heads each,
head width 128, at 4096 and 16384 tokens of context. The baseline read a
bf16 KV cache. The int4 arms read keys quantized per channel and values
quantized per token, two values packed per byte, and unpacked them in VMEM.

- **Unpack, then scale in VMEM**: 1.23×
  faster on average,[^1] 1.27×
  at 4096 tokens[^2] and 1.19×
  at 16384.[^3]
- **Fold the scales into the query and the softmax probabilities** instead
  of multiplying every cache element: 1.22×
  on average,[^4] 1.27× at 4096
  tokens[^5] and 1.17× at
  16384.[^6] Hoisting the scales bought nothing extra.

The cache shrinks to about a quarter, but the gain is a fraction of that
and smaller at the longer context. Use int4 if you need the memory; do not
expect it to make decode several times faster.

### Decode: reading only the most relevant pages pays, and more with longer context

Paged decode over 64 query heads, head width 128, pages of 64 tokens, at
128 or 256 pages per sequence. The baseline attended over every page. The
Quest arm scored each page by the bounding box of its keys (per-channel
minimum and maximum), picked the top 16 pages with `lax.top_k`, and
attended over only those.

- Quest was 4.03× faster on
  average,[^7] 2.76× at 128
  pages[^8] and 5.29× at
  256.[^9]

This is not the same computation: Quest reads 16 pages where the baseline
reads all of them, so it is an approximation whose quality depends on your
model and prompts. The speed was measured; the quality on a real model was
not.

### Decode: leave the paged KV append to XLA (with this kernel)

Each decode step wrote one new key and value into the tail page of each
sequence and then ran paged decode attention: 32 sequences, 16 heads, head
width 128, pages of 128 tokens, at 1024 and 2048 tokens of context. The
baseline was XLA updating the page pool out of place and gathering pages.

- **A separate in-place Pallas append, then a Pallas paged decode kernel**
  kept 0.744 of XLA's
  speed.[^10]
- **One fused Pallas kernel** doing the in-place append and the decode kept
  0.733.[^11]

Both Pallas paths lost at both context lengths: the fused kernel kept
0.725 at 1024 tokens[^12] and 0.74 at 2048,[^13] the two-step path 0.723[^14]
and 0.766.[^15] Each cell's range is wide, about 0.60 to 0.85 for the
fused kernel and 0.60 to 0.89 for the two-step path, but every range sits
below one.
The two Pallas paths were not timed against each other directly.

### Mixed prefill and decode: one unified kernel is not a reliable win

A continuous-batching step held 2 prefill chunks of 128 tokens and 6
single-token decode requests, 8 heads of width 128. The baseline launched a
prefill kernel and a decode kernel and concatenated their outputs; the
alternative was one Pallas kernel over uniform tiles with per-row length
masks.

- At 1024 tokens of KV the unified kernel was
  1.42× faster,[^16] with a wide
  range.
- At 2048 tokens it kept 0.73 of the split
  version's speed.[^17]

The two cells point in opposite directions, so there is no rule here. Keep separate kernels unless you have timed this kernel at your
context lengths.

### Speculative decoding: score all draft tokens in one pass, and fuse the resample

Verification scored draft queries against a key cache of 8192 tokens for
16 sequences (head width 128) and kept the top 4 scores per draft token, at
8 and 16 draft tokens. The baseline ran one single-query pass per draft
token.

- **One batched XLA contraction** over all draft tokens, then top-k:
  17.2× faster on
  average,[^18] 13.1× at 8
  draft tokens[^19] and 21.4×
  at 16.[^20]
- **A fused Pallas kernel** that streams the key cache once and keeps a
  running top 4 in VMEM: 3.26×
  faster,[^21] 2.23× at
  8[^22] and 4.3× at
  16.[^23]

Both gains grow with the number of draft tokens. Against the same baseline
the batched XLA version gained far more than this Pallas kernel; the two
were not timed against each other directly, but start with XLA.

The rejection step resampled from the residual distribution, the positive
part of target minus draft probabilities, with Gumbel-max over a vocabulary
of 65,536 in fp32, for 64 and 128 draft tokens. A single Pallas pass that
streams both distributions and the noise once, using the shift invariance
of Gumbel-max to skip the normalization pass, was
1.62× faster[^24] than
three separate passes through HBM, the same at both
sizes.[^25][^26]

## Believed, not measured

- Most of the wins above remove trips through HBM: one read of the key
  cache for all draft tokens, one pass over both distributions for the
  resample. Bandwidth was not measured directly, and several results fell
  well short of a byte count (the int4 cache, the fused paged append) or
  exceeded it (the batched speculative scoring).
- The int4 cache probably gains less than its size because unpacking and
  scaling on the vector unit take time that the smaller read does not hide.
  This was not profiled.
- Why the hand-written Pallas paged-append kernels lost is not established.
  Untuned tile sizes and grid order are the usual suspects; XLA may simply
  have scheduled the simpler version better.

## Never measured

- More than one chip, and chips other than the v6e.
- Model quality: the accuracy cost of Quest page selection and of an int4
  KV cache on a real model. Only kernel time was measured.
- The fused and two-step Pallas paged-append paths against each other, and
  the batched XLA against the fused Pallas speculative top-k. No direct
  contrast was recorded for either.
- The unified prefill-and-decode kernel at context lengths other than 1024
  and 2048, and any decode trick end to end in a serving stack.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.10.1, libtpu 0.0.41, torch 2.10.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: int4 with in-VMEM scaling against bf16 cache: median ratio 1.23, range [1.18, 1.27], over 2 cells, 2 sessions each.
[^2]: in-VMEM scaling, 4096 tokens: ratio 1.27 [1.26, 1.27], 2 sessions; B = 16; D = 128; G = 4; H_kv = 4; S = 4096.
[^3]: in-VMEM scaling, 16384 tokens: ratio 1.19 [1.18, 1.19], 2 sessions; B = 16; D = 128; G = 4; H_kv = 4; S = 16384.
[^4]: int4 with hoisted scales against bf16 cache: median ratio 1.22, range [1.17, 1.27], over 2 cells, 2 sessions each.
[^5]: hoisted scales, 4096 tokens: ratio 1.27 [1.27, 1.27], 2 sessions; B = 16; D = 128; G = 4; H_kv = 4; S = 4096.
[^6]: hoisted scales, 16384 tokens: ratio 1.17 [1.17, 1.18], 2 sessions; B = 16; D = 128; G = 4; H_kv = 4; S = 16384.
[^7]: top-16 pages against dense paged decode: median ratio 4.03, cells 2.76 to 5.29, over 2 cells, 2 sessions each.
[^8]: 128 pages: ratio 2.76 [2.16, 3.37], 2 sessions; BH = 64; D = 128; N_pages = 128; P = 64; top_k = 16.
[^9]: 256 pages: ratio 5.29 [3.95, 6.64], 2 sessions; BH = 64; D = 128; N_pages = 256; P = 64; top_k = 16.
[^10]: two-step Pallas append and decode against XLA out-of-place: median ratio 0.744, range [0.602, 0.892], over 2 cells, 2 sessions each.
[^11]: fused Pallas append and decode against XLA out-of-place: median ratio 0.733, range [0.604, 0.861], over 2 cells, 2 sessions each.
[^12]: fused Pallas, 1024 tokens: ratio 0.725 [0.604, 0.846], 2 sessions; B = 32; D = 128; H = 16; P = 128; S = 1024.
[^13]: fused Pallas, 2048 tokens: ratio 0.74 [0.619, 0.861], 2 sessions; B = 32; D = 128; H = 16; P = 128; S = 2048.
[^14]: two-step Pallas, 1024 tokens: ratio 0.723 [0.602, 0.843], 2 sessions; B = 32; D = 128; H = 16; P = 128; S = 1024.
[^15]: two-step Pallas, 2048 tokens: ratio 0.766 [0.639, 0.892], 2 sessions; B = 32; D = 128; H = 16; P = 128; S = 2048.
[^16]: unified kernel against split kernels, 1024 KV tokens: ratio 1.42 [1.07, 1.77], 2 sessions; D = 128; H = 8; N_decode = 6; N_prefill = 2; S_kv = 1024.
[^17]: unified kernel against split kernels, 2048 KV tokens: ratio 0.73 [0.55, 0.91], 2 sessions; D = 128; H = 8; N_decode = 6; N_prefill = 2; S_kv = 2048.
[^18]: batched XLA against one pass per draft token: median ratio 17.2, range [11.3, 24.6], over 2 cells, 2 sessions each.
[^19]: batched XLA, 8 draft tokens: ratio 13.1 [11.3, 14.8], 2 sessions; B = 16; D = 128; K_top = 4; Q_spec = 8; S = 8192.
[^20]: batched XLA, 16 draft tokens: ratio 21.4 [18.1, 24.6], 2 sessions; B = 16; D = 128; K_top = 4; Q_spec = 16; S = 8192.
[^21]: fused Pallas against one pass per draft token: median ratio 3.26, range [1.89, 4.95], over 2 cells, 2 sessions each.
[^22]: fused Pallas, 8 draft tokens: ratio 2.23 [1.89, 2.56], 2 sessions; B = 16; D = 128; K_top = 4; Q_spec = 8; S = 8192.
[^23]: fused Pallas, 16 draft tokens: ratio 4.3 [3.66, 4.95], 2 sessions; B = 16; D = 128; K_top = 4; Q_spec = 16; S = 8192.
[^24]: single-pass Pallas against three HBM passes: median ratio 1.62, range [1.57, 1.65], over 2 cells, 2 sessions each.
[^25]: 64 draft tokens: ratio 1.6 [1.57, 1.64], 2 sessions; BD = 8; BV = 4096; N_draft = 64; V = 65536.
[^26]: 128 draft tokens: ratio 1.63 [1.62, 1.65], 2 sessions; BD = 8; BV = 4096; N_draft = 128; V = 65536.
