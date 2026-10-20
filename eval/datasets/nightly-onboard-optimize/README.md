# Serving Onboarding & Optimization (`nightly-onboard-optimize`)

Serve a Hugging Face model on Cloud TPU with `torch_tpu` and vLLM, and prove
it. Where [`hf-onboarding-torchtpu`](../hf-onboarding-torchtpu/) asks for an
analysis and a benchmark script, this asks for a server that answers over
HTTP, measured numbers from that server, and a commit history that shows how
it got there.

## Tasks

### Onboarding (`onboarding-*`)

One task per model. The brief, the image and the verifier are identical
across all four; the model is the variable, so "size the topology yourself"
has more than one right answer and a run that hardcodes a slice stops
scoring.

| Task | Model | Params | Gated | Reachable with our key |
| :--- | :--- | :--- | :--- | :--- |
| [`onboarding-qwen3-8-27b`](./onboarding-qwen3-8-27b/) | `Qwen/Qwen3.8-27B` | 27B dense | no | yes |
| [`onboarding-gemma-4-31b-it`](./onboarding-gemma-4-31b-it/) | `google/gemma-4-31B-it` | 31.3B dense | no | yes |
| [`onboarding-mistral-small-24b`](./onboarding-mistral-small-24b/) | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | no | yes |
| [`onboarding-gpt-oss-20b`](./onboarding-gpt-oss-20b/) | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no | yes |

### Onboarding + Kernel Optimization (`tpu-inference-*-vllm`)

End-to-end tasks that require both onboarding a model onto `torch_tpu` + vLLM
and optimizing its serving throughput by profiling with XProf, authoring and
autotuning a custom Pallas kernel, integrating it into vLLM, and re-running
the full serving benchmark sweep.

| Task | Model | Params | Gated |
| :--- | :--- | :--- | :--- |
| [`tpu-inference-gemma-4-31b-it-vllm`](./tpu-inference-gemma-4-31b-it-vllm/) | `google/gemma-4-31B-it` | 31B dense | no |
| [`tpu-inference-gpt-oss-20b-vllm`](./tpu-inference-gpt-oss-20b-vllm/) | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no |
| [`tpu-inference-qwen3-8-27b-vllm`](./tpu-inference-qwen3-8-27b-vllm/) | `Qwen/Qwen3.8-27B` | 27B dense | no |

The model catalogue lives in
[`hf-onboarding-torchtpu/README.md`](../hf-onboarding-torchtpu/README.md) --
categories, target slices, recommended TP and gating for every model in scope.
The five here are a subset of it, not a second list.

Gating is per-account and manual for the marked model: the token in use has
`canReadGatedRepos`, so a 403 there means the account has not accepted that
model's terms, not that the key is wrong. Accept at the model page, then check
`https://huggingface.co/settings/gated-repos`. The briefs for gated models say
to report a 403 and stop rather than substitute another model -- a silent
substitution scores as a success on a model nobody asked for, and the verifier
cannot tell the difference.

## Arms

| Config | installs | |
| :--- | :--- | :--- |
| `claude-code/agent-suite` | the whole package | the agent-suite arm |
| `claude-code/baseline` | nothing | the control |

Every config installs the whole package; one whose name contains `baseline`
is a control arm and installs nothing.

## Environment

- **`HF_MODEL_URL`** -- set per task in its `task.toml`; the agent reads it
  from the environment rather than from the prompt.
- **`HF_TOKEN`** -- required for the gated models, ignored by the rest. The
  gated tasks declare `required_env = ["HF_TOKEN"]` under `[metadata]`. When
  it is unset *or empty* on the host, `run.py` drops those tasks from a
  dataset run (announced on stderr and in the GitHub step summary) so the
  rest still run, and refuses a task named alone with `--task`. A missing CI
  secret arrives as `""`, which `${HF_TOKEN:-}` forwards unchanged, and the
  trial would otherwise end at the first 401 inside the pod.
- The image is built from each task's `environment/Dockerfile`: the
  `torchtpu-vllm-dev` nightly plus node, git and tokamax. Harbor's GKE path
  never reads `task.toml`'s `docker_image` (`run.py:286`), so no image is
  pinned there.
- The tasks need a real v6e or v7 slice. Export the `GKE_*` variables for the
  shared cluster before running; `run.py` does not read a `.env` file.

## Scoring

`tests/test.sh` scores the checkable half of the brief and says which half
that is: artifacts under one directory, the image digest and HF revision
pinned, both required tables with a measured row, raw output retained, an HTTP
response carrying content, and three or more local commits. "Coherent output"
and a log that "reads as the story" are left to a reader.

Two prohibitions are disqualifying gates rather than points -- `torch_xla` or
`tpu_inference`, and committed credentials or weights -- so a run that
produced nothing earns nothing for satisfying them.

## Running

```bash
# one model, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite -t nightly-onboard-optimize/onboarding-qwen3-8-27b

# the same model, control arm
python3 eval/run.py -c claude-code/baseline -t nightly-onboard-optimize/onboarding-qwen3-8-27b

# every task in the nightly-onboard-optimize dataset
python3 eval/run.py -c claude-code/agent-suite -d nightly-onboard-optimize
```

### Provenance pages between trials

Each trial caches tpu-knowledge-base pages inside its own container, under
`/root/.tpu-agent-suite`, and they go when the container does. A trial is
scored on what it retrieved, not on what a sibling left behind.
`TPU_AGENT_SUITE_HOME` pins that path rather than moving it: every task
collects its artifacts from `/root/.tpu-agent-suite/runs` and nothing sets
`HOME`. Keep it equal to `~` for the container's user -- compiled hook
commands hardcode the `~/.tpu-agent-suite` spelling and do not honour it.

There used to be a shared host directory here, so that one trial's fetch
spared the next: every onboarding reads the same few vllm pages, and
docs.vllm.ai refused the third request in quick succession from one IP.
The lookup no longer fetches anything -- a url is summarised by the harness
CLI, from the CLI's own backend, not from this VM -- so there is no longer
a rate limit to share around, and nothing to configure.
