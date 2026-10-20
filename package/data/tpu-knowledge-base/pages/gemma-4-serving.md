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
entry: gemma-4-serving
keyword: gemma-4-tp
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-01
source_updated: not applicable
generator: authored
---

# Serving Gemma-4-31B-it on TPU: what its config decides

Gemma-4-31B-it's sliding-window layers and its global layers don't have the
same number of KV heads, and the smaller number sets how far tensor
parallelism (TP) can shard the cache. "16 KV heads", the figure a one-line
summary gives, is only the sliding-window layers'.

## The attention layout

From `text_config` in the model's `config.json` (`index.py get gemma-4-31b-it-config`):

| | sliding-window layers | global layers |
| --- | --- | --- |
| count, of 60 | 50 | 10, every 6th |
| KV heads | 16 | **4** (`num_global_key_value_heads`) |
| window | 1024 tokens | full context, up to 262144 |

Both have 32 query heads. The config also sets `attention_k_eq_v: true`.

## What it means for TP

- **TP of 2 or 4** shards every layer with no replication.
- **TP above 4** still divides the sliding-window layers, but the global layers
  have only 4 to split. Each device then holds a copy, and those layers' cache
  stops shrinking with TP. Going past the head count takes batch sharding
  instead (`index.py get concept-kv-cache`).
- **On TPU7x, TP counts devices.** Frameworks expose one device per
  TensorCore, so TP=4 spans half a `tpu7x-standard-4t` host, which has 8
  (`index.py get hw-tpu-7x`).
- **Which TP onboarding picks**, and how to override it, is the rule in
  `index.py get vllm-tp-dp-plan`: the largest TP that suits the heads and fits
  the weights, unless the user states one.

## Observed

A tpu-agent session served this model on TPU7x in September 2026 with
vLLM and torch_tpu. These are that session's results, not published figures:

- It chose TP=4, for the reason above.
- Teacher-forced against the Transformers model on CPU in bf16, the TPU
  server picked the same top token at 175 of 179 positions, and each mismatch
  was a near-tie (`index.py get serving-output-agreement`).
- The model is multimodal, but the session sent no image. Its checks and
  benchmarks cover text only.

## Evidence

The attention layout is the published `config.json`, read at
commit `842da3794eaa0b77d5f08bae87a17459d91ff475`. The TP consequences are
derived from it and the cache page above; they are arithmetic, not
measurements. The "Observed" section is one session and is on no public page.
