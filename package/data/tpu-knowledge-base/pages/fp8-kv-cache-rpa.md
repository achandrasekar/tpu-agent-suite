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
entry: fp8-kv-cache-rpa
keyword: fp8-kv-cache
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-12
source_updated: not applicable
generator: authored
---

# FP8 KV cache with ragged paged attention: e4m3fn

An FP8 KV cache halves the bytes per cached token, so the same HBM holds
twice the tokens. With ragged paged attention (RPA), the one FP8 format with
recorded kernel-parity evidence is `float8_e4m3fn`. Ask for it by name:
`--kv-cache-dtype fp8_e4m3`. On a v6e vLLM + torch_tpu server, plain `fp8`
was logged as `float8_e5m2`, which nothing here has tested.

## Kernel parity (internal, not published)

An internal TPU kernel-experiment write-up (2026-07-30) checked RPA v3
against its reference implementation with a `float8_e4m3fn` KV cache and a
bf16 query:

- mixed prefill/decode batches, `q_len` 1 to 512, `kv_len` 16 to 2048
- head_dim 128, page sizes 16 and 128
- max abs difference 1.19e-7 against a 1e-2 tolerance: passed

Nothing was recorded for e5m2, for head_dim 256 or 512, or with a sliding
window.

The same write-up has kernel timings, left out here: it doesn't say which
TPU generation each row ran on.

## Observed on v6e

A tpu-agent hill-climb in October 2026 served Gemma-4-31B-it on one v6e-8
(TP=4 x DP=2, vLLM 0.29.0 with torch_tpu, batched RPA, page size 128,
8k in / 1k out at concurrency 128). These are that session's results, not
published figures:

| | bf16 KV | `--kv-cache-dtype fp8_e4m3` |
| --- | --- | --- |
| KV capacity per replica | 77,213 tokens | 154,456 tokens |
| output throughput (3 runs) | 1,309.49 ± 11.46 tok/s | 1,771.03 ± 27.89 tok/s (+35.2%) |
| TTFT p50 | 42.90 s | 13.85 s |
| TPOT p50 | 39.96 ms | 47.30 ms |

- Greedy outputs on fixed prompts, including a 3k-token needle, were
  identical to bf16. The scales were static and per-tensor (1.0).
- The gain came from capacity. More sequences fit, so the decode batch grew
  from 21 real sequences per replica to 45 and queueing time fell. Per-step
  latency rose.
- The model's full-attention layers hold one KV head per device. In bf16 the
  server padded their pages up to the sliding-window layers' page size. In FP8
  both page sizes came out at 256 KiB, so that padding went away too
  (`index.py get gemma-4-serving`).
- The first prefill profile with FP8 found the K and V quantize, concat and
  pad run as 8-bit XLA fusions at 0.2% of roofline: 19.15 ms of a 159 ms
  mixed step. See `index.py get xla-8bit-layout-copies` for the fix.

## Evidence

The parity sweep comes from an internal TPU
world-model wiki, which is not published. Its page was read on 2026-10-09
and is not cited by location. The "Observed" section is one session's
measurement, on no public page. RPA's own constraints (per-tensor scales,
packing) are in `index.py get ragged-paged-attention`.
