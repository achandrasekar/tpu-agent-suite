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
entry: vllm-torchtpu-attention-backend
keyword: custom-attention-backend
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-05
source_updated: not applicable
generator: authored
---

# `--attention-backend CUSTOM` on vllm-torchtpu: which kernel it selects

On the vllm-torchtpu plugin, `--attention-backend` is what picks the paged
attention kernel. `CUSTOM` selects the plugin's batched-RPA backend, which runs
tpu-inference's experimental `batched_rpa` kernel. Without the flag the plugin
falls back to `FLASH_ATTN` and the non-batched RPA kernel, and on Gemma-4-31B-it
that kernel fails to compile. This is why tpu-model-onboarding's vLLM recipe always
passes `--attention-backend CUSTOM`.

## Applies when

- The engine is `vllm` with the `vllm_torchtpu` platform plugin.
- You are deciding whether a launch needs `--attention-backend CUSTOM`, or
  which source file a paged attention kernel in a trace comes from.
- A server started without the flag fails at compile time with
  `Ran out of memory in memory space vmem` in an `RPA...` `pallas_call`.

## The dispatch chain

Read from the plugin source in the task image (vLLM 0.29.0, torch_tpu
0.1.1+release.2026.9.22). vllm-torchtpu is not public yet, so these paths are
the image's, not a repository's.

1. `--attention-backend CUSTOM` sets `attention_config.backend` to
   `AttentionBackendEnum.CUSTOM`.
2. The plugin registers `PallasBatchedRPAAttentionBackend` under that enum
   (`vllm_torchtpu/layers/adapter/attention.py`,
   `@register_backend(AttentionBackendEnum.CUSTOM)`). `TpuPlatform` chooses it
   over `PallasAttentionBackend` when the backend is `CUSTOM`
   (`vllm_torchtpu/platforms/tpu_platform.py`).
3. Its Impl sets `use_batched_rpa=True`, which the backend's docstring says
   "plumbs through `_pallas_rpa_kernel` → `attention()` →
   `sharded_ragged_paged_attention()` to pick the batched kernel function".
4. The kernel is tpu-inference's `kernels/experimental/batched_rpa`. The public
   half of the dispatch, in tpu-inference's `attention_interface.py`, is
   `index.py get attention-kernel-dispatch`.

Without the flag:

- The plugin logs `Cannot use AttentionBackendEnum.<X> backend on TPU. Setting
  to FLASH_ATTN.` and then `Using FLASH_ATTN backend.` `<X>` is whatever
  upstream vLLM chose. For Gemma-4 that is `TRITON_ATTN`, because its layers
  have two head sizes (256 and 512).
- `FLASH_ATTN` maps to `PallasAttentionBackend`, which does not set
  `use_batched_rpa`.

With the flag, the startup log says `Using CUSTOM backend.` once per worker.

Not determined here: how the per-call `use_batched_rpa` interacts with
tpu-inference's `USE_BATCHED_RPA_KERNEL` environment variable.

## Tell which kernel ran

- **Startup log:** `Using CUSTOM backend.` or `Using FLASH_ATTN backend.`
- **Kernel names**, in a trace or a compile error: the kernel that ran out of
  VMEM without the flag (below) was
  `RPAm-p_128-bq_512_512-bkv_2048_512-sw_1024`. That is the
  `RPA<case>-p_<page_size>-bq_...` form that `attention-kernel-dispatch` gives
  for RPA v3, not the `RPA<mode>-p<page_size>` form it gives for `batched_rpa`.
- When you profile a server, take the kernel's source file from the launch
  flags, not from a file with a similar name.

## Evidence: Gemma-4-31B-it with and without the flag

tpu-model-onboarding run `20261005T185158Z-40a60af3d0af`, on a TPU VM:

- **Hardware and stack:** `tpu7x-4`, 8 devices; vLLM 0.29.0; libtpu 0.0.47.
- **Launch:** `google/gemma-4-31B-it`, TP 2 x DP 4, `--max-model-len 2048`.

The two launch scripts differ only in the flag.

| | No flag | `--attention-backend CUSTOM` |
| --- | --- | --- |
| Backend in the log | `FLASH_ATTN` (upstream picked `TRITON_ATTN`) | `CUSTOM` |
| KV cache | block size 128, 117,749 tokens per replica | same |
| Outcome | Compile fails: `Ran out of memory in memory space vmem. Used 71.34M of 63.94M vmem.` The largest allocation is a 36 MiB scratch operand of `RPAm-p_128-bq_512_512-bkv_2048_512-sw_1024/pallas_call`; register spill slots add 11.34 MiB | Serves. All three requests return; the one-word France prompt answers `Paris` |

The same session's earlier no-flag attempts failed for reasons unrelated to
the attention backend, so they are not evidence either way:

- a tier-2 compilation cache directory it had no permission to create;
- the 600 s engine-ready timeout;
- `PjRtClient is not initialized`;
- a `head_dim` error from an older transformers in a different venv.

## Caveats

- **One model fails without the flag.** A hybrid linear-attention model in the
  same session started and generated without the flag at TP 4 x DP 2. So the
  flag is not always needed just to start a server. It is still the backend
  the recipe was validated with.
- **No performance comparison.** No run compared throughput or output agreement
  between the two backends on a model that serves under both.
- **TPU VM, not the GKE image.** The comparison ran on a TPU VM with its own
  plugin checkout. The dispatch chain was read from the GKE task image.

## Sources

- Run `20261005T185158Z-40a60af3d0af` (tpu-model-onboarding, 2026-10-05): the
  plugin-source introspection (`attn_introspect2.txt`), and the TPU-VM launch
  scripts, serve logs and responses for the two Gemma-4 runs (`vm6`, no flag;
  `vm7`, `CUSTOM`).
