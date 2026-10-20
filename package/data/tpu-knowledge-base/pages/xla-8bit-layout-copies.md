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
entry: xla-8bit-layout-copies
keyword: fp8-layout-copy
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-12
source_updated: not applicable
generator: authored
---

# 8-bit arrays on TPU: layout copies that make FP8 slower than BF16

An 8-bit array on TPU can sit in more than one tiled layout, and XLA
converts between them with a physical copy. If the layout an op produces
isn't the one the next op wants, the copy lands on the hot path. On v6e
that has made FP8 slower than BF16. The usual shape of the fix: keep
relayouts (concat, pad, reshape) in bf16 and convert to 8-bit once, on the
tensor in its final layout.

## The signature

8-bit tilings pack four values per 32-bit word (`index.py get
openxla-tiled-layout`). The two layouts seen here are `T(32,128)(4,1)`, the
XLA default for these weights, and `T(8,128)(4,1)`, which the `dot` wanted.
In optimized HLO the conversion shows up as a `copy_fusion` right before
the `dot`, often wrapped in a `bitcast_fusion`:

```hlotxt
%copy_fusion (input: f8e4m3fn[4096,4096]) -> f8e4m3fn[4096,4096] {
  %input = f8e4m3fn[4096,4096]{1,0:T(32,128)(4,1)S(1)} parameter(0)
  ROOT %copy.53 = f8e4m3fn[4096,4096]{1,0:T(8,128)(4,1)} copy(%input)
}
```

## Cases

**FSDP-unsharded FP8 weights (internal, not published).** An internal
investigation on v6e found FP8 matmuls about 3x slower than BF16. FSDP
rebuilds parameters as `torch.as_strided` views of its flat buffer, and
`as_strided` drops torch_tpu's out-of-band layout annotation. The weights
fell back to `T(32,128)(4,1)`, and XLA inserted the copy above before every
`dot`. The write-up gives no shapes beyond the 4096x4096 example and no
library versions. Its audit materialized a parameter and read back its
device layout, through torch_tpu internals that are not a public API. Check
the layout in the optimized HLO instead: an `f8e4m3fn` parameter in
`T(32,128)(4,1)` relaid out to `T(8,128)(4,1)` (a `copy` or a fusion) before
the `dot`.

**FP8 weights in vLLM on v6e (observed).** A tpu-agent hill-climb in
October 2026 (Gemma-4-31B-it, v6e-8, TP=4 x DP=2, vLLM 0.29.0 with torch_tpu)
tried `--quantization fp8`: per-channel `f8e4m3fn` weights and dynamic
per-token activations. The f8 x f8 dots compiled to convolution fusions with
an in-fusion `T(32,128)(4,1)` -> `T(8,128)(4,1)` relayout, running at about 6%
of HBM bandwidth. The decode step went from 26.27 ms to 96.45 ms, and output
throughput from 1,831.60 to 446.64 tok/s (-75.6%). Numerics had passed. Even
without the copies there was nothing to gain, because v6e's FP8 peak equals
its BF16 peak (`index.py get hw-tpu-v6e`, `index.py get
torch-tpu-precision-ladder`).

**FP8 KV-cache writes (observed, same session).** With an FP8 KV cache,
K and V were quantized to FP8 first and then concatenated, padded and
reshaped into the attention kernel's packed layout as 8-bit arrays. The
concat-and-pad fusion ran at 0.2% of roofline, about 159 µs per sliding
layer. Building K|V in bf16 and converting once, after the relayout, cut
KV write prep from 19.15 to 9.62 ms per prefill-heavy step. Output
throughput went from 1,771.03 to 1,831.60 tok/s (+3.4%), with greedy outputs
unchanged. The change was a local patch, not an upstream option. With
per-tensor scales, scaling before or after the concat is the same math.

## What to check

- After switching anything to 8-bit, grep the optimized HLO for `copy` or
  `copy_fusion` on `f8` or `s8` operands just before a `dot`, convolution or
  custom call.
- Before an 8-bit matmul on a chip, check its rate in that format. v6e runs
  INT8 at 2x BF16 and FP8 at 1x (`index.py get hw-tpu-v6e`).

## Evidence

The FSDP case comes from an internal TPU world-model wiki, which is not
published. Its page was read on 2026-10-09 and is not cited by location.
The two vLLM cases are one session's measurements, on no public page. The
8-bit tiling format is OpenXLA's documented one.
