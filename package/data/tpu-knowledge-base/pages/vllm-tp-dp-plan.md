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
entry: vllm-tp-dp-plan
keyword: vllm-tp-dp-plan
url: https://github.com/google-pytorch/torchtpu-agents/blob/main/package/scripts/common/engine_compatibility_check.py
authority: first-party-verified
retrieved: 2026-10-13
source_updated: 2026-10-12
generator: authored
---

# Choosing TP and DP for a vLLM deployment on torch_tpu

tpu-model-onboarding does not search for a layout. The
`vllm-engine-compatibility` check (`engine_compatibility_check.py`) writes
`tp_plan.json` with one tensor-parallel size (TP) and one data-parallel size
(DP), and the recipe copies them into `deployments.json` unless the user
stated a layout or the model is in the patch registry. The plan takes the
**largest** TP that fits, so one engine serves the model.

## Terms

The same words are used here, in `tp_plan.json` and in onboarding's
references:

- **Device**: what torch_tpu counts as one device, and what
  `compute inspect` reports as `job_devices`. A tpu7x chip is two devices; a
  v6e chip is one.
- **TP**: `--tensor-parallel-size`, the devices one engine shards the model
  across.
- **Replica**: one vLLM engine with its own full copy of the weights, its own
  KV cache and its own compiled graphs. **DP** (`--data-parallel-size`) is the
  number of replicas.
- **TP x DP** in the plan equals the devices one deployment gets. The manifest
  can leave some idle: a TP the user states with no `--data-parallel-size` in
  this vLLM build, or a build that cannot run replicas, starts one engine.

size-serving-topology (`index.py get playbook-size-serving-topology`) asks a
different question: how big a slice to buy. Its "model parallelism `Y`" is TP
here, and its slice is the device count, which onboarding never changes. Its
pitfall against data parallelism in a generate configuration follows the
scaling book, which says data parallelism within one model "replicates our
parameters and doesn't help us load parameters faster", and that you are
"better off spinning up multiple copies of the model instead"
(`index.py get scaling-book-inference`). A vLLM replica is such a copy: it
shares no weights or KV cache with the other replicas. So that pitfall does
not rule out DP here.

## Applies when

- Choosing `--tensor-parallel-size` and `--data-parallel-size` for vLLM on the
  `vllm_torchtpu` plugin.
- Reading `tp_plan.json`, or deciding whether to pass `--tp` to override it.
- A server fails at the planned TP and you need the next layout to try.

## The rule

Read from `plan_parallelism`, `head_problem` and `main` in the script, and
from `torch-inference-coder.md`, at torchtpu-agents `5f408a087`.

1. **A TP the user states always wins** (`source: "user"`). The plan only
   records what it costs: idle devices, replica startup, an expected OOM.
   Next comes a model in the patch registry (`vllm_patches/registry.json`):
   its entry's `validated.serving` settings take precedence over the plan,
   and the manifest records `parallelism.source` as `registry`. The steps
   below are the plan, which decides every other model.
2. **Candidates** are the TPs that divide the device count and suit the
   heads:
   - query heads split evenly;
   - KV heads split evenly, or TP is a multiple of the KV-head count so they
     replicate (MLA models always replicate);
   - a hybrid model's Gated-DeltaNet key and value heads both split evenly
     (`linear_attention_heads`).
3. **Drop TPs the stack cannot start**: the ones the installed torch_tpu has
   no device layout for, and the ones vLLM rejects. The script asks
   `TpuPlatform._get_tpu_topology(tp)` itself, so it keeps no list of counts.
   For example, on a v6e 2x4 it refuses TP 2 with "No TPU topology found for
   count: 2". If that lookup fails any other way, no TP is ruled out. vLLM is
   asked only about the TP the plan picks, through a configuration dry run
   (`EngineArgs.create_engine_config`); a TP it refuses is dropped and the
   plan is redone. A dry run does not build the model's layers, so a failure
   that appears only when the engine starts is not caught here.
4. **Weight fit**: weights ÷ TP must be at most 80% of one device's HBM. The
   KV cache, a hybrid model's per-sequence state and working memory are not
   counted.
5. **Take the largest** remaining TP, with DP = devices ÷ TP. Replicas appear
   only when the heads, the layouts or vLLM rule out a TP that spans every
   device: a 12-head model on 8 devices gets TP 4 x DP 2.
6. **Record the alternative.** When a smaller TP also fits, the plan sets
   `smallest_fitting_tp` and `reason` says that it "may serve faster". Any
   benchmark of the deployment must then say it was measured at the larger
   TP. Measuring the smaller one is opt-in, with `--tp`.

If the weight size cannot be read, the plan takes the largest
head-compatible TP without checking the fit, and `reason` says so.

## Why largest, not smallest

The plan used to take the smallest TP that fits, which passed the weight
check and then failed to start. The check's regression cases keep two
examples on a tpu7x-4 (8 devices x 96 GiB), where that rule chose
TP 1 x DP 8:

- **Qwen3.8-27B** (51.75 GiB): its linear-attention state slots ran one device
  out of HBM at server startup.
- **GPT-OSS-20B**: eight engines compiling at once on one host ran past
  vLLM's engine-ready timeout.

Both now plan TP 8 x DP 1. Onboarding needs a server that starts on the
first try. The trade-off is throughput: for
a small model, a smaller TP with more replicas may serve more tokens per
second.

## When the planned TP fails

These are tpu-model-onboarding's retry rules (`torch-inference-coder.md`):

- **Memory error**: rerun the check with `--tp` set to the next divisor of the
  device count above the failed TP, and take both TP and DP from the new
  plan. Never change TP alone.
- **No larger TP**, or the new plan's `ok` is false: keep the layout and shrink
  each engine. A hybrid model lowers `max_num_seqs` first when
  `--max-num-seqs` is in `vllm_flags.txt`, since it keeps one state slot per
  concurrent sequence; otherwise, or for any other model, lower
  `max_model_len`.
- **Stop after two resubmissions** and hand the decision back to the user.

The rules do not yet cover an engine assertion, as opposed to a memory error,
at the planned TP (see the first caveat).

## Evidence

| Run | Hardware | Rule | Layout | Outcome |
| --- | --- | --- | --- | --- |
| Onboarding runs under the smallest-TP rule, kept as the check's regression cases | tpu7x-4, 8 x 96 GiB | smallest | TP 1 x DP 8 | Qwen3.8-27B OOM at startup; GPT-OSS-20B past the engine-ready timeout |
| A check with a newer `vllm_torchtpu` than the GKE image, 2026-10-12 | tpu7x, 8 x 96 GiB | largest | TP 8 x DP 1 | Qwen3.8-27B hits a GDN state assertion in `pool_adapters.py`; TP 4 x DP 1 serves |
| Patch registry entry for GPT-OSS-20B, validated 2026-10-13 (vLLM 0.29.0, torch_tpu 0.1.1 release 2026.9.22) | tpu7x 2x2x1, 8 x 96 GiB | registry | TP 8 x DP 1 | Serves; TP 2 x DP 4 and TP 4 x DP 2 also verified; TP 1 has no torch_tpu layout on this stack. On the earlier vLLM 0.22.1 stack it served at TP 1 x DP 8 |
| tpu-model-onboarding run `20261013T034030Z-692e0c582399` (Harbor job `af542c62`, tpu-agent-suite 0.5.0) | v6e 2x4, 8 x 32 GB | smallest | TP 4 x DP 2 (TP 2 has no layout) | Qwen3.8-27B served and answered; no throughput, because every benchmark attempt lost the node or sat Pending |

## Caveats

- **The largest TP is not verified for Qwen3.8-27B.** On v6e 2x4 the current
  rule plans TP 8 x DP 1: 24 query heads, 4 KV heads replicated twice, and 16
  and 48 linear-attention heads all divide 8. That is the layout that hit the
  plugin assertion on the newer plugin. The plan cannot catch it, since its
  vLLM check is a configuration dry run. Whether the GKE image's plugin has
  the same assertion is not known. GPT-OSS-20B, the other regression case,
  is validated at TP 8 on tpu7x (see Evidence).
- **No throughput comparison.** No run here measured the largest and the
  smallest fitting TP for the same model on the same image, so "largest"
  is about starting on the first try, not about speed.
- **The fit check counts weights only.** Expect less KV-cache room than 20% of
  HBM implies.

## Related

- How vLLM runs replicas (load balancing, one engine core per DP rank):
  `index.py get vllm-data-parallel-deployment`.
- The head-count rules vLLM itself applies: `index.py get vllm-model-config`
  and `index.py get vllm-qkv-parallel-linear`.
- Sizing a slice, batch and step-time floor from the model config:
  `index.py get playbook-size-serving-topology`.

## Sources

- `package/scripts/common/engine_compatibility_check.py` and its regression
  cases in `engine_compatibility_check_test.py`,
  `package/references/tpu-model-onboarding/torch-inference-coder.md` and
  `package/scripts/common/vllm_patches/registry.json`, torchtpu-agents
  `5f408a087`.
- The TP 8 assertion: a Qwen3.8-27B serve on tpu7x with a newer
  `vllm_torchtpu` than the GKE image, reported on 2026-10-12 when the
  largest-TP rule was reviewed.
- tpu-model-onboarding run `20261013T034030Z-692e0c582399` (2026-10-13), its
  `tp_plan.json`, serve attempts and benchmark failure table.
