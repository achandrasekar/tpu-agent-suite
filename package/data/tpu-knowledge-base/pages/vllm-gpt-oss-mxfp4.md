---
title: "Serve gpt-oss-20b on TPU with vLLM (vllm_torchtpu MXFP4 patch)"
type: playbook
tags: [onboarding, serving, vllm, moe, mxfp4, quantization, gpt-oss]
created: 2026-10-01
updated: 2026-10-13
sources: 3
human_reviewed: true
aliases: ["gpt-oss vllm", "gpt_oss_mxfp4", "vllm gpt-oss patch"]
target_imports:
- gpt_oss_mxfp4
---
# Serve gpt-oss-20b on TPU with vLLM (vllm_torchtpu MXFP4 patch)

`openai/gpt-oss-20b` serves correctly on TPU with vLLM after one launch-time
patch to the `vllm_torchtpu` plugin. Without it, vLLM refuses to start, fails
in attention, or returns NaN on every token. The patch is `gpt_oss_mxfp4.py` in
the shared vLLM patch directory, `scripts/common/vllm_patches/`, next to
`apply_vllm_patch.sh` and the `registry.json` entry that names it for this
model. `apply_vllm_patch.sh` runs it in every job that imports the plugin. The
task image is never modified.

`gpt_oss_mxfp4.py` is a patch script, not a fixed diff. Each fix finds its own
anchor in the installed plugin source, so one script covers several plugin
revisions and both plugin layouts (see *How the patch applies*).

## Applies when

- The model is `openai/gpt-oss-20b` and the engine is `vllm`.
- The vllm-engine-compatibility dry run, or `vllm serve`, fails with
  `gpt_oss_mxfp4 quantization is currently not supported in tpu`.
- Any later symptom in the triage list below.

## Validated environment

Proven on 2026-10-13 against this stack:

| Component | Value |
| --- | --- |
| Model | `openai/gpt-oss-20b` (public, safetensors, MXFP4 experts, no `trust_remote_code`) |
| vLLM | 0.29.0 |
| Plugin | `vllm_torchtpu` from the task image (`layers/adapter` + `layers/core` layout) |
| torch / torch_tpu | 2.13.0 (CPU build) / 0.1.1, release 2026.9.22; no torch_xla |
| transformers | 5.14.1 |
| JAX / jaxlib | 0.11.2 |
| Hardware | GKE job, tpu7x 2x2x1: 4 chips, 8 devices, 96 GiB HBM per device (as the runtime reported) |
| Serving config | TP 8 x DP 1, `--max-model-len 8192`, port 8000, `--attention-backend CUSTOM` |
| Environment | `USE_MOE_SPARSE_CORE=0`, `TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false` |
| Startup time | about 5.5 minutes from job start until the API server reports `Application startup complete` (weights already cached) |

The same script was also verified on the first stack this was proven on
(vLLM 0.22.1); see *Earlier stack* below.

## Validated serving settings

- **TP 8 x DP 1 is the setting to launch with.** It is what the
  vllm-engine-compatibility check recommends for this model on one 8-device
  host: the largest TP that divides the 64 attention heads and whose weight
  shard fits, so a single engine serves the model and compiles once. It also
  passed on the vLLM 0.22.1 image. Smaller TPs with more replicas serve
  correctly too, as the table shows; their throughput was not measured, so
  pass `--tp` for one only to measure it.
- **TP 1 does not start on vLLM 0.29.0.** Every worker fails at init with
  `RuntimeError: No TPU topology found for count: 1`: on this image torch_tpu
  maps tensor-parallel world sizes 2, 4 and 8 to a device topology, but not 1.
- **`USE_MOE_SPARSE_CORE=0`.** The patch pads experts to 2944, a multiple of
  128. At that width `moe_gmm` chooses the SparseCore
  `sc_ragged_gather_reduce_v3` combine, and that kernel fails to compile (see
  triage 7). This variable switches only the combine to plain JAX. Without
  expert parallelism, routing is unchanged. It was set for every layout in
  the table.

Layouts tested on vLLM 0.29.0 (2026-10-13), each with the checks under
*Verification*. Startup is from job start to the last API server reporting
`Application startup complete`, with the weights cached; HBM is per device
after the weights are loaded.

| Layout | Result | Startup | Engine init (compile) | HBM used | KV cache per replica |
| --- | --- | --- | --- | --- | --- |
| TP 8 x DP 1 | pass | 5.6 min | 211 s (88 s) | 5.6 GiB | 9.12 M tokens |
| TP 4 x DP 2 | pass | 5.8 min | 206 s (89 s) | 10.9 GiB | 8.53 M tokens |
| TP 2 x DP 4 | pass | 6.8 min | 242 s (113 s) | 21.4 GiB | 3.68 M tokens |
| TP 1 x DP 8 | no torch_tpu topology for 1 device | - | - | - | - |

TP 2 and TP 4 ran with the script before its attention-sink condition was
added for TP 8. That condition changes nothing when a device holds two or more
KV heads, which a unit test covers.

## What the patch changes

Each fix is named in the applier's log.

| Fix (log name) | Problem | Symptom without it | Change (file under `vllm_torchtpu/`) |
| --- | --- | --- | --- |
| `quantization-name-registration` | The plugin allowlist does not know vLLM's internal name `gpt_oss_mxfp4` | Startup error: `gpt_oss_mxfp4 quantization is currently not supported in tpu` | Add `gpt_oss_mxfp4` to `supported_quantization` (`platforms/tpu_platform.py`), add a `GPT_OSS_MXFP4` constant (`layers/core/quant_methods.py`), and map that name to the config below in `method_to_config` (`layers/adapter/quantization/__init__.py`) |
| `gpt-oss-config-class` | vLLM's gpt-oss weight loader takes its fused-expert path only when the config's `get_name()` is `gpt_oss_mxfp4` | Load error: `shard_id must be ['w1','w2','w3'] but got None` | New `VllmGptOssMxfp4Config` subclass whose `get_name()` returns `gpt_oss_mxfp4` (`layers/adapter/quantization/mxfp4.py`) |
| `expert-padding-128` | The `gmm_v2` MoE kernel rounds tile K/N up to 128 with bounds checks disabled. gpt-oss hidden = intermediate = 2880, which is not a multiple of 128 | Server healthy, but output is `!!!!...` (token 0) and logprobs are NaN | Zero-pad expert weights and biases from 2880 to 2944 in `process_weights_after_loading` (`mxfp4.py`) |
| `unloaded-moe-tail` | Under a flattened MoE TP of 8, each rank's intermediate slice is rounded up to 384, but the last rank loads only 192. vLLM relies on `torch.zeros` for the rest. The plugin's expert stager loads into reused `torch.empty` host buffers instead, so that tail holds stale bytes | All output NaN: `!!!!` text, and logprobs requests return 400 `nan` | Zero the never-loaded intermediate tail of `w13`/`w2` weights, scales and `w13` bias before dequantization (`mxfp4.py`, `_zero_gpt_oss_unloaded_intermediate`). It acts only on GPT-OSS MXFP4 layers of a rank that has a tail |
| `hd64-attention-sinks` | A head_dim 64 layer owns an hd64 KV pool whose last axis is padded to 128. The plugin treated that pool as a wider page and padded q/k/v to 128, which steered attention away from the hd64 kernel, the only kernel with attention-sink support | Engine fails on the first forward: `NotImplementedError: Attention sink support is only available when head_dim==64` | Do not widen q/k/v when the layer owns its hd64 pool, told by its row count, or when the layer has attention sinks: at TP 8 each device holds one KV head, and packing pads the hd64 pool to the row count of a wider pool (`layers/adapter/attention.py`) |
| `moe-forward-input-ids` | vLLM 0.22.1 passes `input_ids` to the MoE forward | `TypeError` on an unexpected `input_ids` keyword | Add `input_ids=None` to `_forward_monolithic_tpu` (`mxfp4.py`). Already upstream in the vLLM 0.29.0 image; logged `present, upstream` |

## How the patch applies

`bash apply_vllm_patch.sh gpt_oss_mxfp4.py` runs the script with the job's
Python, passing the directory that holds `vllm_torchtpu/`. That directory is
found with `importlib.util.find_spec`, which does not import the plugin.
Plugin files are located by module path. The script knows the
`layers/adapter` + `layers/core` layout and the older `layers/vllm` +
`layers/common` layout.

Every fix ends in one of three states, and each one is logged:

- `present, applied earlier by this patch`, `present, upstream` or
  `present, not needed`. The fix is skipped. Inserted code carries the comment
  `gpt-oss MXFP4 onboarding patch`, which is how the script tells its own
  changes from upstream ones.
- `applied`. The anchor was found and the change was made.
- Neither. The script exits 4 with `vllm patch does not apply`, names each fix
  whose anchor moved, and writes nothing.

All fixes are decided before any file is written. The plugin is either patched
in full or left untouched. The last log line is
`vllm patch applied: gpt_oss_mxfp4.py (<root>)` or
`vllm patch already applied: ...`. A second run changes nothing and reports
every fix as present.

On the vLLM 0.29.0 image the result is equivalent to the hand-rebased diff that
was verified there. One difference: the unloaded tail is zeroed on the layer
parameters through a host copy, not on the dequantization inputs.

## Triage: which failure is this?

Each failure appears only after the one above it is fixed.

1. **vLLM exits at config validation** with `Value error, gpt_oss_mxfp4
   quantization is currently not supported in tpu`: the patch was not applied
   (`quantization-name-registration`).
2. **Weight loading fails** with `shard_id must be ['w1','w2','w3'] but got
   None`: the config is reached but reports the wrong name
   (`gpt-oss-config-class`).
3. **Every worker fails at init** with `RuntimeError: No TPU topology found for
   count: 1`: the server was started with TP 1. Use TP 8 x DP 1 on one 8-device
   tpu7x host.
4. **The engine fails on the first forward** with `NotImplementedError:
   Attention sink support is only available when head_dim==64`: q/k/v were
   widened to 128 for the hd64 pool (`hd64-attention-sinks`). If the log shows
   that fix applied and the server runs TP 8, the script predates its
   attention-sink condition; use the current one.
5. **The engine crashes on the first forward** with `TypeError:
   VllmMxfp4MoEMethod._forward_monolithic_tpu() got an unexpected keyword
   argument 'input_ids'` (`moe-forward-input-ids`; vLLM 0.22.1 only).
6. **`/health` returns 200 but completions are `"!!!!!!!!..."`** (token id 0),
   and `logprobs` requests fail with `400 Out of range float values are not
   JSON compliant: nan`. Two causes look exactly alike:
   - `expert-padding-128` is missing (every DP/TP layout).
   - `unloaded-moe-tail` is missing. This happens only when the MoE TP size
     does not divide gpt-oss's 90 MX blocks evenly, as with an MoE TP of 8
     (TP 8 x DP 1, or TP 2 x DP 4, where MoE TP is flattened to 8).

   Check the applier's log for both fixes. A non-empty response is not proof
   of success.
7. **Compilation fails in `sc_ragged_gather_reduce_v3`** with
   `CompileTimeSparseCoreAllocationFailure: current allocation offset upper
   bound (3014655 words) exceeds the legitimate user allocatable offset upper
   bound`: SparseCore SPMEM overflowed at the padded width 1024 x 2944. Set
   `USE_MOE_SPARSE_CORE=0`.
8. **The fused GMM raises `size_n=5760 should be divisible by 2 * num_lanes`**:
   the weights were not padded at all. Apply `expert-padding-128` in full.
9. **`apply_vllm_patch.sh` exits 4** ("does not apply"): the lines under it
   name each fix whose anchor this plugin revision moved, and no file was
   changed. Check whether the image already fixes that problem before
   re-basing the fix in the script.

Upcasting to bf16 is not a workaround: the unquantized MoE method has the same
K-padding gap and also produces NaN.

## Root causes

The load path, with the patch applied:

1. The HF config says `quant_method: mxfp4`.
2. vLLM core (`GptOssMxfp4Config.override_quantization_method`) renames it to
   `gpt_oss_mxfp4`, so the gpt-oss layout in
   `vllm/model_executor/models/gpt_oss.py` is used.
3. The plugin's `TpuPlatform.supported_quantization` checks that name.
   **Unpatched, it stops here** (`quantization-name-registration`).
4. The plugin maps the name through `method_to_config` to
   `VllmGptOssMxfp4Config` (registered as `tpu-gpt_oss_mxfp4`).
5. `GptOssModel._load_weights_mxfp4` calls the expert layer's
   `weight_loader(shard_id=None)` (`FusedMoE` in vLLM 0.22.1, `RoutedExperts`
   in 0.29.0). That loader copies the fused all-experts tensors only when
   `quant_config.get_name() == "gpt_oss_mxfp4"`. **A plain `mxfp4` config
   fails here** (`gpt-oss-config-class`).
6. `VllmMxfp4MoEMethod.process_weights_after_loading` does these steps in
   order:
   1. Zeroes the never-loaded intermediate tail (`unloaded-moe-tail`).
   2. Dequantizes the experts to bf16 on CPU.
   3. De-interleaves gate_up into [gate | up] (the `swigluoai` path).
   4. Transposes.
   5. **Pads to 128** (`expert-padding-128`).
   6. Moves the weights to HBM.
7. At runtime, `_forward_monolithic_tpu` runs the fused MoE GMM (bf16 inputs,
   fp32 accumulation and biases).

Why the NaN happens:

- `gmm_v2` sets `tile_k = align_to(size_k, 128)` and launches its Pallas call
  with `disable_bounds_checks=True`.
- 2880 / 128 = 22.5, so both GMMs read 64 elements past the end of K and pick
  up uninitialized memory.
- `fused_moe_func` already pads activations `x` to w13's K and slices the
  output back to hidden. Padding the weights is therefore enough, and no kernel
  change is needed.
- Ablations: padding only the gate/up halves (N) still gives all NaN; no
  padding fails with the `size_n` error; padding both K and N gives correct
  output.
- With MoE TP 8, 2880 / 8 = 360 is rounded up to 384 per rank. Ranks 0 to 6
  load 384 each (2688 in total), and rank 7 loads the remaining 192. The other
  192 rows of rank 7 come from a reused `torch.empty` host buffer. One NaN e8m0
  scale or bias there turns that rank's partial sum into NaN. After the MoE
  all-reduce, every token is NaN. A layer-level comparison against CPU on TPU
  traced the NaN to exactly that tail.

## Steps

1. Ship `gpt_oss_mxfp4.py` and `apply_vllm_patch.sh` to the target as inputs,
   in the same directory.
2. Pass `--vllm-patch gpt_oss_mxfp4.py` to the vllm-engine-compatibility
   check, so its dry run sees the patched plugin. It recommends TP 8 x DP 1,
   the same as the registry entry's `serving`.
3. Write `launch_vllm.sh` so that it:
   - runs under `set -eo pipefail`, so a patch that fails stops the server;
   - exports `USE_MOE_SPARSE_CORE=0` and
     `TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false`;
   - runs `bash apply_vllm_patch.sh gpt_oss_mxfp4.py` before `vllm serve`;
   - serves with `--tensor-parallel-size 8 --data-parallel-size 1
     --max-model-len 8192 --attention-backend CUSTOM`.
4. Submit the recipe as `launch_vllm.sh`. Every backend runs a `.sh` recipe
   under bash. The applier runs the patch script with the configured Python,
   which `TPU_AGENT_SUITE_PYTHON` names (else `python3`).
5. Before the vLLM API server banner, the job log shows one line per fix, then
   `vllm patch applied` (or `already applied`).

## Verification

Do not accept a server on `/health` alone. Send each prompt with
`temperature: 0`, `logprobs: 5` and `max_tokens: 16`, at least 8 times in
parallel, so every data-parallel replica is hit. All texts must be identical,
and logprob requests must return 200.

| Prompt | Known-good text (matches a CPU transformers reference) | First-token top logprobs (CPU) |
| --- | --- | --- |
| `The capital of France is` | `" Paris.\"\n\nSure! Here's a simple example of a Python"` | ` Paris` -0.5, ` not` -3.31, `:` -3.5 |
| `Hello! Briefly, what is a tensor?` | `"\n\nA tensor is a mathematical object that generalizes scalars"` | `\n\n` -1.44, ` A` -2.75, ` How` -3.06 |

Results on 2026-10-13, 16 parallel requests per prompt, with
`gpt_oss_mxfp4.py`. In every run all 32 requests returned 200 and all 16 texts
per prompt were identical:

| Prompt | Text (every layout, both images) |
| --- | --- |
| `The capital of France is` | `" Paris.\"\n\nSure! Here's a simple example of a Python program that uses a"` |
| `Hello! Briefly, what is a tensor?` | `"\n\nA tensor is a mathematical object that generalizes scalars, vectors, and"` |

First-token top 3 in the most common response, and the largest top-3
deviation from CPU over all 32 requests:

| Stack | Layout | France | Tensor | Max deviation |
| --- | --- | --- | --- | --- |
| vLLM 0.29.0 | TP 8 x DP 1 | ` Paris` -0.4375, ` not` -3.3125, `:` -3.5 | `\n\n` -1.5625, ` A` -2.8125, ` How` -3.0625 | 0.19 |
| vLLM 0.29.0 | TP 4 x DP 2 | ` Paris` -0.5, ` not` -3.3125, `:` -3.5 | `\n\n` -1.5, ` A` -2.75, ` How` -3.0625 | 0.25 |
| vLLM 0.29.0 | TP 2 x DP 4 | ` Paris` -0.4375, ` not` -3.3125, `:` -3.5 | `\n\n` -1.5, ` A` -2.75, ` How` -3.0625 | 0.12 |
| vLLM 0.22.1 | TP 1 x DP 8 | ` Paris` -0.5, ` not` -3.3125, `:` -3.5 | `\n\n` -1.4375, ` A` -2.75, ` How` -3.0625 | 0.003 |
| vLLM 0.22.1 | TP 8 x DP 1 | ` Paris` -0.4375, ` not` -3.3125, `:` -3.5 | `\n\n` -1.5625, ` A` -2.75, ` And` -3.125 (` How` -3.125 next) | 0.19 |

Both texts match the known-good prefix, and the top-3 ranking matches CPU in
every run but one, where ` How` and ` And` tie exactly at -3.125 for third
place.

Across the 16 requests of one run, the text and ranking never changed, but
some first-token logprobs moved by one or two bf16 steps. The likely cause is
requests landing in batches of different sizes. It is not a failure.

Small reorderings in the top-5 tail are bf16 near-ties. With the fixed diff
on vLLM 0.22.1 (2026-10-01), ranks 2 and 3 of the France prompt swapped
relative to CPU, about 3.3 nats below the top token. A 400 that mentions `nan` on a logprobs request means
`expert-padding-128` or `unloaded-moe-tail` is missing (triage 6).

## Earlier stack (vLLM 0.22.1)

The first validation (2026-10-01) used a fixed diff, `gpt_oss_mxfp4.patch`. It
changed the same lines that `quantization-name-registration`,
`gpt-oss-config-class`, `expert-padding-128` and `moe-forward-input-ids`
change now, under the older `layers/vllm` + `layers/common` layout. The
applier refused that diff on any other plugin revision. The script replaced
it.

On 2026-10-13 the script was run on that same image:

| Component | Value |
| --- | --- |
| vLLM | 0.22.1 |
| Plugin | `vllm_torchtpu` from the task image (`layers/vllm` + `layers/common` layout) |
| torch / torch_tpu | 2.11.0 (CPU build) / 0.1.1.dev20260714 |
| transformers | 5.8.0 |
| JAX | 0.10.2 |
| Hardware | GKE job, tpu7x 2x2x1: 4 chips, 8 devices, 96 GiB HBM per device |

- **Apply log.** Every fix applied, except `hd64-attention-sinks`, logged
  `present, not needed`: this revision does not widen q/k/v to the KV pool.
  `moe-forward-input-ids` is applied here, not upstream. A second run
  reported every fix present and changed nothing.
- **TP 1 x DP 8**, `--max-model-len 8192 --attention-backend CUSTOM`, without
  `USE_MOE_SPARSE_CORE`: passed. All 8 replicas were ready about 10.6 minutes
  after job start, including a 1-minute weight download (engine init about
  330 s, of which about 140 s compile); 40.7 GiB HBM used per device;
  905 K tokens of KV cache per replica. MoE TP is 1, so no rank has an
  unloaded tail.
- **TP 8 x DP 1** with the validated serving settings, `USE_MOE_SPARSE_CORE=0`
  included: passed. Ready about 8.1 minutes after job start (engine init
  339 s, 93 s compile); 5.5 GiB HBM used per device.
- On this image the engine compatibility check reports no torch_tpu device
  layout for TP 1, yet TP 1 x DP 8 serves. The server is the test.

## Caveats

- **No native MXFP4 compute.** The checkpoint loads as MXFP4, but the experts
  are dequantized to bf16 at load. That costs about 40 GiB per replica and
  gives no MXFP4 speed benefit.
- **The padding bug is general.** Any MoE model whose hidden or intermediate
  size is not a multiple of 128 hits the `gmm_v2` out-of-bounds read. The
  unquantized MoE method (`VllmUnquantizedFusedMoEMethod`) pads only the N
  halves, not K. This patch does not fix that path.
- **The unloaded-tail bug is general too.** Any checkpoint whose loader writes
  only part of a parameter, behind the plugin's expert stager, can read stale
  host memory. The patch zeroes the tail for GPT-OSS MXFP4 layers only.
- **Only gpt-oss-20b was validated.** gpt-oss-120b shares the quantization and
  the 2880 width, but has not been run with this patch.
- **Formal parity was not run.** The evidence is the exact text match and the
  logprob comparison against CPU transformers above.
- **The generic inference check can read empty.** `verify_inference.py` asks
  through the chat template with a 64-token budget. gpt-oss can spend all of
  it in its reasoning channel, and the reply then has no content
  (`finish_reason: length`), which the check reports as an empty completion.
  This happened at TP 8 x DP 1 on vLLM 0.29.0 while the prompts above passed.
  Use the prompts above as the check for this model.
- **Upstream fixes, in order of preference:**
  - Make `gmm_v2` mask or zero-fill the partial K tile (or keep bounds checks
    on when K % 128 != 0).
  - Allocate the expert stager's buffers with `torch.zeros`, or zero the
    unwritten tail on flush.
  - Exempt hd64-pool layers from q/k/v widening.
  - Move K and N padding into a helper shared by the MXFP4 and unquantized MoE
    methods.
  - Add `gpt_oss_mxfp4` to the platform allowlist, and keep the subclass so
    `get_name()` still matches vLLM's loader check.

  As each fix lands upstream, the script logs it `present, upstream` and skips
  it.

## Sources

- tpu-model-onboarding fix run on the vLLM 0.22.1 stack (2026-10-01):
  the vLLM 0.22.1 patch, ablation logs, and the verification requests and
  responses for that stack.
- tpu-model-onboarding run on the vLLM 0.29.0 stack (2026-10-13): the fixes
  re-based for the new plugin layout, the layer-level NaN localisation, the
  failed TP 1 and SparseCore attempts, an engine-compatibility dry run with
  `gpt_oss_mxfp4.py`, and the verification above.
- tpu-model-onboarding runs on 2026-10-13 with the current
  `gpt_oss_mxfp4.py`: an onboarding run on the vLLM 0.29.0 stack following
  the patch registry end to end, the TP 2, 4 and 8 sweep (which found and
  fixed the TP 8 `hd64-attention-sinks` case), and the run on the vLLM 0.22.1
  image at TP 1 x DP 8 and TP 8 x DP 1.
