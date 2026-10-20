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
entry: v6e-measured-attention-serving
keyword: v6e-measured-attention-serving
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# Serving attention on TPU v6e: which kernel, how many KV heads, what page size

You are serving a transformer on a TPU v6e and attention is where the time
goes. Decode steps crawl as the KV cache grows, a long prompt takes far longer
to prefill than its size suggests, a batch of requests with very different
lengths wastes most of its work on padding, or you are trying to decide how
many KV heads to keep, what page size to give the KV cache, or whether
DeepSeek-style latent attention is worth the trouble. This page collects
measurements on one v6e chip of those choices, using
the attention kernels that ship with JAX (`paged_attention`,
`ragged_paged_attention` and `splash_attention`, all Pallas kernels called
from PyTorch through torch_tpu's `pallas.jax_op`), plus two
`scaled_dot_product_attention` choices under `torch.compile`, and, for anyone
writing their own attention kernel in Pallas, a set of choices inside one.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative. Above one, the alternative is faster by that
factor; below one, it is slower, and the number is the share of the
baseline's speed it keeps. Each experiment checked every arm's output
against a full-precision reference before timing it, so the arms being
compared compute the same attention. Most results come from one shape per
question; the shapes are in the text and footnotes, and a different batch,
context length or head count can move them.

## Measured

### Decode: fewer KV heads was the biggest lever measured

The decode step was JAX's `paged_attention` kernel answering one query token
for each of 16 sequences, each with 8192 tokens already in the KV cache
(pages of 128 tokens), with 32 query heads of width 128 in bf16. The baseline
gives each query head its own KV head (32 KV heads, plain multi-head
attention). The alternatives share each KV head across a group of query
heads, which shrinks the cache the kernel has to read by the same factor.

- **16 KV heads** (groups of two):
  1.87× faster.[^1]
- **8 KV heads** (groups of four):
  3.71× faster.[^2]
- **4 KV heads** (groups of eight):
  7.81× faster.[^3]
- **1 KV head** (multi-query attention):
  27.9× faster.[^4]

The speedup tracked the reduction in KV bytes almost one for one, which is
what you would expect if decode attention on v6e is limited by reading the
cache. That is one shape (16 sequences of 8192 tokens, 128-token pages) with
one cell per head count; at short contexts, where the cache read is a
smaller share of the step, expect less. These are kernel times only. Whether
your model keeps its quality with fewer KV heads is a training question that
was not measured.

If you write a grouped-query decode kernel yourself, compute the whole group
of query heads that share a KV head in one matrix multiply: stack the group's
queries as the rows of one (group, head width) block, so the MXU sees a real
matrix instead of a single row. With 16 query heads per KV head, 64 KV heads
and width 128, that was 11× faster with 1024
tokens of cache[^5] and 15.5×
faster with 2048,[^6] against a kernel that loops over the
query heads one at a time on the vector unit.[^7] The
experiment treated that loop as the standard way to do it, but it is a
naive version: no shipped kernel computes one query head per pass, and JAX's
`paged_attention` already groups the heads. Read the size of the gain as what
the naive loop costs, not as what folding buys over a sensible kernel.

### Writing the KV cache: fuse the rotary embedding into the append

Each decode step rotates the new keys with the rotary embedding and writes
them into the cache. Doing both in one Pallas kernel, with the rotation done
in on-chip memory on the way into the cache, against two kernels that write
the rotated keys to HBM and read them back to append them, was
1.61× faster with a 1024-token
cache[^8] and 1.62× faster with
2048[^9] (32 heads of width 128, in float32, 128 new
tokens at a time).[^10] Fusing saves one round trip of the
new keys through HBM, and the gain barely moved with cache length.

### Prefill: the number of KV heads made no difference

The prefill was JAX's `splash_attention` kernel over a 4096-token causal
prompt, again with 32 query heads, against the same baseline of 32 KV heads.
Going to 16, 8, 4 or 1 KV heads made no measurable difference: the ratios are
0.999,[^11]
0.998,[^12]
0.998[^13] and
0.999,[^14] all inside the
measurement's noise. That fits a long prefill being limited by the matrix
multiplies rather than by reading K and V, so grouping the heads saves
nothing here. This is one 4096-token causal prompt; at that shape the
number of KV heads is a decode and cache-memory choice, not a prefill one.

### Page size for paged decode attention: keep pages large

`paged_attention` keeps the KV cache in fixed-size pages scattered through a
pool, looks each page up in a page table, and copies one page per memory
transfer. The baseline uses 128-token pages; the alternatives use smaller
pages holding the same keys and values, with the kernel still working through
512 tokens at a time, so smaller pages only mean more, smaller transfers. The
model had 32 query heads over 8 KV heads. Each figure below is the median
over three shapes (16 sequences of 2048 or 8192 tokens, and 64 sequences of
2048 tokens), which agreed closely.

- **64-token pages** run at 0.909 of the
  speed of 128-token pages.[^15]
- **32-token pages** run at 0.96 of the
  speed.[^16]
- **16-token pages** run at 0.707 of the
  speed.[^17]

Every smaller page size was slower than 128-token pages, and 16-token pages
were by far the worst. The order is not simply smaller-is-slower: 32-token
pages lost less than 64-token pages, in all three shapes; why was not
measured. Use pages of at least 128 tokens (larger pages were not measured
for this kernel; 256-token pages tied for `ragged_paged_attention`, below)
unless memory fragmentation forces you smaller, and if it does, time your
own page size rather than assuming the next size down is the next best.

### Ragged batches: use the ragged kernel instead of padding

JAX's `ragged_paged_attention` takes a batch of sequences of any lengths,
each with its own number of new query tokens, straight from a paged cache.
The baseline is the usual workaround: copy the batch into a dense,
contiguous cache, pad every sequence to the longest one, and run masked
attention in XLA. Both were given the same sequences, with 32 query heads
over 8 KV heads. The ragged kernel used 64-token pages.

- **A decode-only batch** of 32 sequences, one new token each, with KV
  lengths stepping from 128 up to 4096 tokens: the ragged kernel is
  1.44× faster than padded dense
  attention.[^18]
- **A mixed batch**, the shape continuous batching produces: 28 decode
  sequences plus 4 prefill chunks of 128 new tokens each. Padding forces every
  decode sequence to carry 128 query slots, so almost all of the dense work is
  padding. The ragged kernel is 372× faster
  than padded dense attention.[^19]

The 372× is the cost of that padding, not a speedup of one good kernel over
another: the baseline computes 128 query slots for every decode sequence
that has one. If your server mixes prefill and decode in one batch, padding
it is not a small inefficiency; it is most of the cost.

### Page size for ragged paged attention: smaller matters less

The same ragged kernel was run at several page sizes against a 128-token
baseline, with the kernel always working through 512 tokens of cache at a
time. Each figure pools the decode-only and mixed batches above.

- **256-token pages:** no difference, a ratio of
  1.01, inside the noise.[^20]
- **64-token pages** run at 0.982 of the speed of
  128-token pages.[^21]
- **32-token pages** run at 0.943 of the
  speed.[^22]
- **16-token pages** run at 0.869 of the
  speed.[^23]

The direction is the same as for `paged_attention`, but the penalty for small
pages is smaller. Anything from 64 to 256 tokens is close to free; below that
you start to pay.

### Multi-head latent attention: absorb the up-projection

This is DeepSeek-style MLA decode: 32 sequences, 32 query heads, a compressed
latent cache of width 512 plus a separate rotary part of width 64, per-head
key and value width 128, bf16. The baseline decompresses the whole latent
cache into full per-head keys and values on every step and then runs
attention over them. The alternatives absorb the up-projection matrices into
the query and the output instead, so attention runs directly against the
small latent cache and nothing is decompressed: once as ordinary XLA
operations, and once with a single-pass Pallas kernel (FlashMLA) that streams
the latent cache through on-chip memory in 512-token blocks with an online
softmax.

- **Absorbed, in XLA:** with 2048 tokens of context,
  28.9× faster;[^24] with 4096,
  25.5× faster.[^25]
- **Absorbed, with the FlashMLA kernel:** with 2048 tokens of context,
  37.9× faster;[^26] with 4096,
  40.1× faster.[^27]

At this decode shape absorption mattered far more than the kernel:
decompressing the cache every step throws away the reason MLA exists, and
the absorbed form in plain XLA already captured most of the gain. This is
one batch and head configuration against the naive baseline that
decompresses every step. Once absorbed, the remaining choice is the
kernel. Against the same absorbed attention written as separate XLA
operations, which reads the latent cache twice and writes the full score
matrix to memory, the FlashMLA kernel is 1.32×
faster at 2048 tokens[^28] and
1.58× faster at 4096.[^29] Its
advantage grows with context.

That advantage is not general. A second experiment ran a fused Pallas
FlashMLA kernel at a different shape: 1024 queries at once against a latent
cache of width 128 with no rotary part, closer to a prefill chunk than to
decode. The experiment expected it to be at least 1.26× faster than the
same attention as separate XLA operations, and it was not. At 2048 tokens of
cache the two cost the same, a ratio of 1
(inside the noise);[^30] at 4096 the fused kernel ran at
0.827 of XLA's speed, so XLA was faster.[^31]
With many queries per step the score matrix is a matrix multiply XLA handles
well, and fusing buys nothing; time your own shape before writing the
kernel.

Inside an MLA kernel, the score is the sum of two products, one over the
compressed part and one over the separate rotary part. Computing them as two
MXU products and adding, against copying the two parts side by side in
on-chip memory and doing one product over the combined width, made no
difference: a ratio of 0.999, inside the noise, at
4096 tokens (8 heads, both parts of width 128).[^32]
Write whichever is simpler.

### Long prefill: never materialize the score matrix

Here the baseline is attention written in XLA that builds the full
sequence-by-sequence score matrix in HBM, against JAX's `splash_attention`,
which computes the same thing block by block in on-chip memory with an online
softmax. The shape was 2 sequences, 8 heads of width 128, bf16, 512-token
blocks, with no mask (every block computed).

- **4096 tokens:** splash is 21.3×
  faster.[^33]
- **8192 tokens:** splash is 151×
  faster.[^34]

The gap widens quickly with length, because the materialized score matrix
grows with the square of it.

### Block-sparse masks: let splash skip blocks

`splash_attention` takes its mask as a compile-time object and skips every
block the mask rules out entirely. The baseline here is splash with a full
mask; the alternatives are a causal mask and a local sliding window of 512
tokens. At 4096 tokens the causal mask leaves 36 of 64 blocks per head and
the window 15; at 8192 tokens, 136 of 256 and 31.

- **Causal:** 1.21× faster at 4096
  tokens[^35] and 1.3× at
  8192.[^36]
- **512-token sliding window:** 3.2× faster at
  4096 tokens[^37] and 6.22× at
  8192.[^38]

Skipping pays, and pays more as the sequence grows and the mask gets
sparser, but less than the share of skipped blocks alone would
suggest.[^39][^40] Why the
remaining blocks cost more than their share has not been measured.

### If you write your own flash attention kernel

These were measured on hand-written Pallas kernels rather than the shipped
ones, each against a variant of itself.

- **Full (non-causal) attention, as in a diffusion transformer.** A
  pipelined flash kernel with no mask, 256-row blocks, 8 heads of width 128,
  was 2.36× faster than attention that writes the
  score and probability matrices to HBM at 2048
  tokens,[^41] and 2.34× at
  4096.[^42][^43] The baseline is the
  weak one, attention written out in full, so this is the long-prefill
  lesson above for the non-causal case rather than a comparison of two good
  kernels. The gain is far smaller than splash's over materialized XLA
  attention there, and did not grow from 2048 to 4096; why was not
  measured.
- **Causal masking on the diagonal blocks.** Computing each full 256 by 256
  diagonal block on the MXU and masking the upper half on the vector unit
  was 3.23× faster with 8 diagonal
  blocks[^44] and 3.3× with
  16[^45] than splitting each diagonal block into the ten
  64 by 64 sub-blocks that hold any unmasked
  entries.[^46] Small MXU products cost more than the
  wasted half of a big one: do the whole block and mask it.
- **Skipping the rescale when the row maximum has not moved.** With an
  attention sink, the row maximum is usually set by the first block and
  stays put. Skipping the max update and the rescaling of the output
  accumulator on the blocks where it did not change was
  1.26× faster at 2048
  tokens[^47] and 1.27× at
  4096[^48] (8 heads, 128-row
  blocks).[^49] The experiment expected the gain to stay
  within 1.25×; at 4096 it went just past. It depends on the maximum really
  being stationary, which these inputs were built to make true.
- **Float32 accumulators with one final normalisation cost nothing at long
  context.** Keeping the running softmax normaliser and the output
  accumulator in float32 and normalising once at the end, against bf16
  accumulators normalised every step, ran at 1.02× the speed with 131072
  tokens of context[^50] (2 heads, 128 queries, 256-token key blocks). The
  two arms differ in two ways, the accumulator dtype and how often they
  normalise, and the two were not separated. So this shows the
  float32-plus-deferred-normalisation recipe costs nothing, not that float32
  alone is free. It is one cell; the 65536-token cell gave no usable
  result. Accuracy was compared but is not reported here.

### If you write attention in PyTorch under torch.compile

Two `scaled_dot_product_attention` choices, measured as compiled models on
the tpu backend:

- **Grouped-query attention:** passing `enable_gqa=True` with the smaller K
  and V, instead of copying K and V up to the query head count with
  `repeat_interleave`, steps 1.71×
  faster[^51] (16 query heads over 4 KV heads, 1024
  tokens). It compiles at 0.295 of the
  speed,[^52] so a short job may not earn the compile back.
  Copying with `expand` and `reshape` instead of `repeat_interleave` makes no
  difference.[^53]
- **Causal masking:** `is_causal=True` is
  1.05× faster than passing a mask
  tensor,[^54] and building the mask inside the compiled
  graph with `tril` is
  1.04× faster.[^55]
  Both are small gains.

## Believed, not measured

- Decode attention is generally understood to be limited by reading the KV
  cache from memory, not by arithmetic. The decode results above are
  consistent with that, but memory bandwidth was not measured directly.
- Block-sparse attention is expected to save work in proportion to the
  blocks its mask skips. The block-sparse results above gained less than
  that, and why was not measured.
- The absorbed form of multi-head latent attention is usually recommended
  for decode, and the explicit form for training and prefill. Only decode
  was measured here.

## Never measured

- Anything beyond one chip: context parallelism, ring attention, or KV heads
  sharded across chips.
- Contexts longer than those run here. Decode KV caches stopped at 8192
  tokens, prefill at 8192, the ragged batches at 4096 and MLA at 4096. The
  one exception is the float32-against-bf16 accumulator comparison, run at
  131072 tokens in a hand-written kernel; no shipped kernel was timed there.
- Other chips. Every figure here is v6e only.
- `flex_attention`, with or without a `score_mod`.
- The backward pass. Every measurement here is a forward pass, as in serving;
  none of these kernels was timed for training.
- Tuned block sizes, `megacore_mode`, attention soft-capping, sliding windows
  inside the paged kernels, and quantized (int8 or fp8) KV caches.
- Grouped-query attention inside the ragged kernel at any ratio other than
  32 query heads over 8 KV heads, and MLA in prefill.
- End-to-end serving throughput, for example through vLLM. These are kernel
  timings; how much of a request's latency they account for is open.
- Model quality with fewer KV heads or with MLA, which is a training question.

## Evidence

Every figure here was measured on a single TPU v6e chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: 16 KV heads: ratio 1.87, 1 cell, 2 sessions.
[^2]: 8 KV heads: ratio 3.71 [3.71, 3.71], 2 sessions; B = 16; Dh = 128; Hq = 32; S_kv = 8192; page = 128; pages_per_block = 8.
[^3]: 4 KV heads: ratio 7.81, 1 cell, 2 sessions.
[^4]: 1 KV head: ratio 27.9 [27.9, 27.9], 2 sessions; B = 16; Dh = 128; Hq = 32; S_kv = 8192; page = 128; pages_per_block = 8.
[^5]: folded group against per-head loop, 1024 tokens: ratio 11 [11, 11], 2 sessions; B_kv = 64; D = 128; G = 16; S_kv = 1024.
[^6]: folded group against per-head loop, 2048 tokens: ratio 15.5 [15.5, 15.5], 2 sessions; B_kv = 64; D = 128; G = 16; S_kv = 2048.
[^7]: both cache lengths: median ratio 13.3, cells 11 to 15.5, over 2 cells, 2 sessions each.
[^8]: fused against two kernels, 1024 tokens: ratio 1.61 [1.61, 1.61], 2 sessions; BH = 32; D = 128; S = 1024; TS = 128.
[^9]: fused against two kernels, 2048 tokens: ratio 1.62 [1.62, 1.62], 2 sessions; BH = 32; D = 128; S = 2048; TS = 128.
[^10]: both cache lengths: median ratio 1.62, cells 1.61 to 1.62, over 2 cells, 2 sessions each.
[^11]: 16 KV heads: ratio 0.999 [0.998, 1], 2 sessions; B = 1; Dh = 128; Hq = 32; S = 4096; block_kv = 512; block_q = 1024; causal = 1.
[^12]: 8 KV heads: ratio 0.998 [0.997, 1], 2 sessions; B = 1; Dh = 128; Hq = 32; S = 4096; block_kv = 512; block_q = 1024; causal = 1.
[^13]: 4 KV heads: ratio 0.998 [0.998, 1], 2 sessions; B = 1; Dh = 128; Hq = 32; S = 4096; block_kv = 512; block_q = 1024; causal = 1.
[^14]: 1 KV head: ratio 0.998, 1 cell, 2 sessions.
[^15]: 64-token pages: median ratio 0.909, range [0.898, 0.91], over 3 cells, 2 sessions each.
[^16]: 32-token pages: median ratio 0.96, cells 0.949 to 0.96, over 3 cells, 2 sessions each.
[^17]: 16-token pages: median ratio 0.707, cells 0.691 to 0.712, over 3 cells, 2 sessions each.
[^18]: ragged kernel against padded dense attention, decode-only batch: ratio 1.44 [1.43, 1.46], 2 sessions.
[^19]: ragged kernel against padded dense attention, mixed batch: ratio 372 [367, 377], 2 sessions.
[^20]: 256-token pages against 128, ragged kernel: ratio 1.01 [1.01, 1.01], 2 sessions.
[^21]: 64-token pages against 128, ragged kernel: ratio 0.982 [0.981, 0.982], 2 sessions.
[^22]: 32-token pages against 128, ragged kernel: ratio 0.943 [0.938, 0.95], 2 sessions.
[^23]: 16-token pages against 128, ragged kernel: ratio 0.869 [0.851, 0.887], 2 sessions.
[^24]: absorbed in XLA against decompressing every step, 2048 tokens: ratio 28.9 [28.8, 29], 2 sessions; B = 32; H = 32; S = 2048; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^25]: absorbed in XLA against decompressing every step, 4096 tokens: ratio 25.5 [25.4, 25.6], 2 sessions; B = 32; H = 32; S = 4096; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^26]: FlashMLA against decompressing every step, 2048 tokens: ratio 37.9 [37.5, 38.2], 2 sessions; B = 32; H = 32; S = 2048; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^27]: FlashMLA against decompressing every step, 4096 tokens: ratio 40.1 [39.7, 40.5], 2 sessions; B = 32; H = 32; S = 4096; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^28]: FlashMLA against absorbed XLA, 2048 tokens: ratio 1.32 [1.3, 1.34], 2 sessions; B = 32; H = 32; S = 2048; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^29]: FlashMLA against absorbed XLA, 4096 tokens: ratio 1.58 [1.56, 1.6], 2 sessions; B = 32; H = 32; S = 4096; block_kv = 512; d_c = 512; d_qk = 128; d_r = 64; d_v = 128.
[^30]: fused against XLA, 2048 tokens: ratio 1 [0.994, 1.01], 2 sessions; B_q = 1024; D_c = 128; S_kv = 2048.
[^31]: fused against XLA, 4096 tokens: ratio 0.827 [0.817, 0.837], 2 sessions; B_q = 1024; D_c = 128; S_kv = 4096.
[^32]: split against concatenated, 4096 tokens: ratio 0.999 [0.99, 1.01], 2 sessions; BH = 8; BQ = 256; D_NOPE = 128; D_PE = 128; SKV = 4096.
[^33]: splash against materialized XLA attention, 4096 tokens: ratio 21.3 [21.2, 21.5], 2 sessions; B = 2; D = 128; H = 8; S = 4096; block_kv = 512; block_q = 512.
[^34]: splash against materialized XLA attention, 8192 tokens: ratio 151 [149, 154], 2 sessions; B = 2; D = 128; H = 8; S = 8192; block_kv = 512; block_q = 512.
[^35]: causal mask against full mask, 4096 tokens: ratio 1.21 [1.2, 1.21], 2 sessions; B = 2; D = 128; H = 8; S = 4096; block_kv = 512; block_q = 512.
[^36]: causal mask against full mask, 8192 tokens: ratio 1.3 [1.3, 1.3], 2 sessions; B = 2; D = 128; H = 8; S = 8192; block_kv = 512; block_q = 512.
[^37]: 512-token window against full mask, 4096 tokens: ratio 3.2 [3.16, 3.23], 2 sessions; B = 2; D = 128; H = 8; S = 4096; block_kv = 512; block_q = 512.
[^38]: 512-token window against full mask, 8192 tokens: ratio 6.22 [6.16, 6.28], 2 sessions; B = 2; D = 128; H = 8; S = 8192; block_kv = 512; block_q = 512.
[^39]: causal across both lengths: median ratio 1.25, cells 1.21 to 1.3, over 2 cells, 2 sessions each.
[^40]: local 512 across both lengths: median ratio 4.71, cells 3.2 to 6.22, over 2 cells, 2 sessions each.
[^41]: flash against materialized, 2048 tokens: ratio 2.36 [2.36, 2.36], 2 sessions; BH = 8; BKV = 256; BQ = 256; D = 128; S = 2048.
[^42]: flash against materialized, 4096 tokens: ratio 2.34 [2.34, 2.34], 2 sessions; BH = 8; BKV = 256; BQ = 256; D = 128; S = 4096.
[^43]: both lengths: median ratio 2.35, range [2.34, 2.36], over 2 cells, 2 sessions each.
[^44]: full masked tile against 64x64 sub-blocks, 8 diagonal blocks: ratio 3.23 [3.23, 3.24], 2 sessions; BH = 8; BLK = 256; D = 128; N_diag = 8.
[^45]: full masked tile against 64x64 sub-blocks, 16 diagonal blocks: ratio 3.3 [3.3, 3.3], 2 sessions; BH = 8; BLK = 256; D = 128; N_diag = 16.
[^46]: both sizes: median ratio 3.27, cells 3.23 to 3.3, over 2 cells, 2 sessions each.
[^47]: lazy against eager, 2048 tokens: ratio 1.26 [1.26, 1.26], 2 sessions; BH = 8; BKV = 128; BQ = 128; D = 128; S = 2048.
[^48]: lazy against eager, 4096 tokens: ratio 1.27 [1.27, 1.27], 2 sessions; BH = 8; BKV = 128; BQ = 128; D = 128; S = 4096.
[^49]: both lengths: median ratio 1.27, range [1.26, 1.27], over 2 cells, 2 sessions each.
[^50]: float32 against bf16 accumulators, 131072 tokens: ratio 1.02 [1.02, 1.02], 2 sessions; BH = 2; BKV = 256; BQ = 128; D = 128; SKV = 131072.
[^51]: enable_gqa against repeat_interleave, step time: ratio 1.71 [1.71, 1.72], 2 sessions; B = 2; Dh = 128; Hkv = 4; Hq = 16; S = 1024.
[^52]: enable_gqa against repeat_interleave, compile time: ratio 0.295 [0.287, 0.302], 2 sessions; B = 2; Dh = 128; Hkv = 4; Hq = 16; S = 1024.
[^53]: expand and reshape against repeat_interleave, step time: ratio 0.999 [0.998, 0.999], 2 sessions; B = 2; Dh = 128; Hkv = 4; Hq = 16; S = 1024.
[^54]: is_causal against a mask tensor: ratio 1.05 [1.04, 1.05], 2 sessions; B = 4; Dh = 128; H = 8; S = 1024.
[^55]: mask built with tril in the graph against a mask tensor: ratio 1.04 [1.04, 1.04], 2 sessions; B = 4; Dh = 128; H = 8; S = 1024.
