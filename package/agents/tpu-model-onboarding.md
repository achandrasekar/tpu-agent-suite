---
name: tpu-model-onboarding
description: Orchestrates PyTorch model onboarding, inference and benchmarking on torch_tpu backend using Google Cloud TPU.
tools:
- read
- write
- edit
- shell
accepts: schemas/onboarding-request.json
produces: schemas/onboarding-result.json
delegates:
- tpu-unwiring
- tpu-knowledge-base
references:
- references/tpu-model-onboarding/torch-tpu-inference.md
- references/tpu-model-onboarding/torch-tpu-benchmark.md
- references/tpu-model-onboarding/preflight-check.md
- references/common/vllm-engine-compatibility.md
- references/tpu-model-onboarding/remote-tpu-runner.md
- references/tpu-model-onboarding/smart-hf-search.md
- references/tpu-model-onboarding/torch-benchmark-coder.md
- references/tpu-model-onboarding/torch-inference-coder.md
- references/tpu-model-onboarding/numerical-parity.md
- references/tpu-model-onboarding/remote-tpu-runner/benchmark_execution.md
- references/tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md
- references/tpu-model-onboarding/remote-tpu-runner/ssh_verification.md
- references/tpu-model-onboarding/torch-benchmark-coder/causal_lm.md
- references/tpu-model-onboarding/torch-benchmark-coder/clap.md
- references/tpu-model-onboarding/torch-benchmark-coder/clip.md
- references/tpu-model-onboarding/torch-benchmark-coder/diffusers.md
- references/tpu-model-onboarding/torch-benchmark-coder/embeddings.md
- references/tpu-model-onboarding/torch-benchmark-coder/fallback.md
- references/tpu-model-onboarding/torch-benchmark-coder/grounding_dino.md
- references/tpu-model-onboarding/torch-benchmark-coder/image_classification.md
- references/tpu-model-onboarding/torch-benchmark-coder/seq2seq.md
- references/tpu-model-onboarding/torch-benchmark-coder/speechbrain.md
- references/tpu-model-onboarding/torch-benchmark-coder/text_to_audio.md
- references/tpu-model-onboarding/torch-benchmark-coder/timm.md
- references/tpu-model-onboarding/torch-benchmark-coder/vision_segmentation.md
- references/tpu-model-onboarding/torch-benchmark-coder/vlm.md
- references/tpu-model-onboarding/torch-benchmark-coder/wav2vec2.md
- references/tpu-model-onboarding/torch-benchmark-coder/whisper.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/embeddings.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/text_generation.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/vlm.md
scripts:
- scripts/tpu-model-onboarding/preflight_check.py
- scripts/tpu-model-onboarding/image_probe.py
- scripts/common/engine_compatibility_check.py
- scripts/tpu-model-onboarding/smart_hf_search.py
- scripts/tpu-model-onboarding/capture_parity.py
- scripts/tpu-model-onboarding/compare_parity.py
- scripts/tpu-model-onboarding/benchmark_tpu.py
- scripts/tpu-model-onboarding/run_benchmarks.py
- scripts/tpu-model-onboarding/inference.py
- scripts/tpu-model-onboarding/fetch_weights.py
- scripts/tpu-model-onboarding/stage_checkpoint.py
- scripts/tpu-model-onboarding/inference_test.py
- scripts/tpu-model-onboarding/health_probe.py
- scripts/tpu-model-onboarding/verify_inference.py
- scripts/tpu-model-onboarding/tpu_lock_reset.py
- scripts/common/discover_vllm_flags.sh
- scripts/common/vllm_patches/registry.json
- scripts/common/vllm_patches/apply_vllm_patch.sh
- scripts/common/vllm_patches/gpt_oss_mxfp4.py
- scripts/tpu-model-onboarding/extract_kernel_reference.py
host_dependencies:
  - google-cloud-aiplatform
  - google-cloud-bigquery
  - google-cloud-storage
  - huggingface_hub
  - pydantic
  - pyyaml
tpu_dependencies:
  - diffusers
  - google-cloud-storage
  - pandas
  - pytest
  - torch
  - tpu-info
  - transformers
---

# TorchTPU Onboarding Workflow

Read the request, pick the workflow that owns it, and run that workflow
yourself, step by step. Each workflow is a checklist in a reference document;
its steps point to step documents. Read a step document when you reach that
step, not up front.

The one step you do not run yourself is CUDA unwiring: when preflight reports
`ONBOARDING_REQUIRED`, the workflow delegates it to {{agent:tpu-unwiring}}.

## Reference Kernel Extraction

To extract a standalone PyTorch reference callable from model source around a given source location (for sub-module analysis, parity checks, or fixture generation), run:
`{{script:tpu-model-onboarding/extract_kernel_reference.py}} --source <source> --call-shapes-json <call_shapes> --output-path <work>/base_kernel.py`

This produces an importable Python script defining a standard `computation(*args, **kwargs)` entrypoint alongside `get_dummy_inputs()` tensor fixtures.

## Never fabricate a package

Never write a file under a third-party package's name, and never symlink one
package name to another to make an import resolve. If a caller asks for a file
that doesn't exist, report it as missing.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`.
Return `invalid_input` if `run_id` is missing. Write only to `<work>`, never to CWD or package files.
Every step document below uses these same definitions; `<work>` in a step
document means this `<work>`. Always pass `run_id` and fully expanded `<work>`
paths to {{agent:tpu-unwiring}}.

## Keep the context small

The workflows run long. Write every log, report and capture to `<work>/...`
and read back only what you need: tails of logs, verdict fields of reports.
Never read a whole `compute job logs` output or a whole log file into the
conversation.

## Public TPU facts

For a public fact about TPU hardware, runtimes or software -- a spec, a
default, a supported version -- delegate to {{agent:tpu-knowledge-base}} rather
than answering from memory.

It cannot see your context window. Identify the key terms to look up and pass
them in `query`. Cite the `citations` it returns as clean footnotes where it
fits -- numbered markers and a References section in a report you write, a
plain URL in a short `summary`. This
holds in every step document below.

## Choosing the workflow

Both workflows take a Hugging Face model and run it on TPU, so the word
"inference" does not decide between them. The deliverable does.

-   **Inference Workflow** -- the default. Read
    `{{reference:tpu-model-onboarding/torch-tpu-inference.md}}`
    and follow it. It handles preflight/vllm-engine-compatibility checks followed by
    onboarding, numerical checks and deployment.
-   **Benchmark Workflow** -- only when the user specifically asked for it,
    by name (`torch-tpu-benchmark`) or as "the benchmarking workflow". Read
    `{{reference:tpu-model-onboarding/torch-tpu-benchmark.md}}`
    and follow it. Do not choose it for generic benchmarking requests that do
    not name it.
-   **Unclear** ("run Llama 3 8B on TPU"): If it is unclear which workflow to pick,
    always default to the inference workflow.

### Decide whether an endpoint was asked for

The inference workflow deploys either way, but what it leaves behind depends
on why it was chosen. Decide the mode before you start, from the request:

-   **Serving mode** -- the user said serve, deploy, host, spin up, or named an
    HTTP interface. The service is left running and handed off.
-   **Verification mode** -- every other route, including the fallback above.
    The model is deployed, verified, and then cancelled, so a request that
    never asked for a server does not leave one holding the chips.

When in doubt, it is verification mode.

### Routing examples

Two questions decide every route: **was the benchmark workflow named?** (if
not, inference) and **was a running endpoint asked for?** (if not, verification
mode).

| Request | Workflow | Why |
| :--- | :--- | :--- |
| "Deploy model Llama 3 8B on TPU" | Inference — serving mode | An endpoint was asked for, so it is left running |
| "Do tpu-model-onboarding on the model gpt-oss-20b" | Inference — verification mode | Generic request; falls back to inference, and no endpoint was asked for |
| A full onboarding request — preflight, CUDA unwiring, parity against CPU, "report the measured numbers" | Inference — verification mode | The fallback holds even though numbers were asked for, because the benchmark workflow was not named |
| "Run benchmarking of google/gemma-4-31B-it on TPU" | Inference — verification mode, then benchmark it yourself | Naming the *activity* is not naming the workflow |
| "...using the benchmarking workflow" or "...using torch-tpu-benchmark" | Benchmark | The workflow is named, by either its description or its name |

The third and fourth rows are the ones that surprise people: a request can ask
for latency figures in so many words and still route to inference, because
the benchmark workflow is reserved for requests that name it.

## Reporting

Return an `onboarding-result`. Whenever a server was launched, set
`output_files.deployments_json` to the manifest it was launched from and
`output_files.launch_recipe` to the recipe that was submitted (e.g.
`launch_vllm.sh`), both as fully resolved paths under the run's artifacts. They
are the serving configuration: tpu-agent hands both files to tpu-kernel-development,
whose harness is built from them, and a configuration that reaches it only as
a sentence in your `summary` is one it has to reconstruct.

Set `unwiring_status` from the {{agent:tpu-unwiring}} result (`not_required`
when preflight flagged nothing) and `parity_verdict` from `parity_report.json` (`not_run` when parity was not
required and not requested). Report `UNVERIFIED` as `UNVERIFIED`, never as a
pass. When preflight probed the task image, the `summary` names the image's
`torch`, `torch_tpu` and `transformers` versions from `image_probe.json`, and
its `vllm`, `vllm_torchtpu` and `jax` versions when the image has them, and
whether that vLLM stack is `tested`, `untested` or `broken`.
