---
title: "Experiments policy: what torchtitan/experiments is, and why TPU support lives there"
type: doc-concept
provenance: doc
source: torchtitan/experiments/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Experiments policy

## Definition
`torchtitan/experiments/README.md` defines `experiments/` as the home for work that adds
significant value but is not `core`: a new model or architecture with its training infrastructure,
or an enhancement to the infrastructure. Each experiment has a clear theme, an owner, and its own
dependencies and tests; must reuse core (components, `train.py`, a new `ModelSpec`) rather than
copy it; depends on core **one-way** ("anything in `experiments` is optional for `core`", and core
development is not blocked by experiment breakage); must not contain vendor-specific code "such as
kernels written in a proprietary language"; and is removed once it has served its purpose (findings
delivered, features upstreamed to core or PyTorch) or goes stale. The companion
`.claude/rules/experiments.md` adds: looser standards but still `pre-commit` clean, never change core
to accommodate an experiment, use TorchTitan's config system (no custom argument parsing), and keep
unrelated features in separate folders.

## In torchtitan (grounded)
- **Registration.** Core models and experiments are resolved by
  [`ConfigManager._load_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L58 "torchtitan/config/manager#ConfigManager._load_config"): a short
  `--module` name must be in `torchtitan/models._supported_models` or
  `torchtitan/experiments._supported_experiments`; anything else is imported as a fully qualified
  path. The README's "Current experiments" table lists four experiments: `graph_trainer`,
  `torchft`, `transformers_modeling_backend` and `rl` (the `forge` row was removed along with the
  folder). The TPU, JAX and torchax experiments are in neither that table nor
  `_supported_experiments`, so they are always addressed by full path, e.g.
  `--module=torchtitan.experiments.tpu.llama3`. Conversely, `_supported_experiments` still lists
  `autoparallel.llama3` and `autoparallel.local_map_deepseek_v3`, although there is no
  `experiments/autoparallel/` folder at this pin (AutoParallel lives on as the
  `--compile.enable_autoparallel` option of [graph-trainer](graph-trainer.md)).
- **Reuse through extension points.** The TPU experiment reuses core the way the policy asks: its
  Qwen3 [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/__init__.py#L144 "torchtitan/experiments/tpu/qwen3/__init__#model_registry") calls the core Qwen3 registry
  and only swaps the FLOP accounting ([`MaxTextQwen3Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/model.py#L19 "torchtitan/experiments/tpu/qwen3/model#MaxTextQwen3Config")),
  its trainer [`TPUTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L42 "torchtitan/experiments/tpu/train#TPUTrainer") subclasses the core trainer, and its
  config [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig") extends the core config tree
  instead of adding a parser.
- **Experiment-owned kernels.** TPU kernels live inside the experiment (`experiments/tpu/kernels/`,
  e.g. [`grouped_matrix_multiply`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/kernels/gmm.py#L98 "torchtitan/experiments/tpu/kernels/gmm#grouped_matrix_multiply") and
  [`custom_jax_kernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/kernels/splash_attention.py#L27 "torchtitan/experiments/tpu/kernels/splash_attention#custom_jax_kernel")), wired in by
  [`apply_patches`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L667 "torchtitan/experiments/tpu/workarounds#apply_patches").

## Why it matters / when it applies
This policy is why TPU support is `experiments/tpu` and not a core device backend: principle 1 of
[repo-conventions](repo-conventions.md) keeps core PyTorch-native, and a TPU lane that needs the
`torch_tpu` runtime, JAX/Pallas kernels and TPU-specific launchers is exactly "an enhancement ...
with its own dependencies". The promotion path it implies is not "move the folder into core" but
"upstream the pieces": generic fixes go to core or PyTorch, and the experiment shrinks.

Two tensions a TPU contributor should know about:
- **Vendor-specific kernels.** The Pallas kernels are written in an open-source JAX DSL, but they
  are TPU-only; the README's rule prefers such kernels "hosted outside as dependency". Whether that
  would block upstream acceptance of `experiments/tpu` is not settled by the doc.
- **One-way dependency is violated in this fork.** Core files import `experiments/tpu`
  (`torchtitan/distributed/activation_checkpoint.py` and the core Flux model's `parallelize.py`,
  `config_registry.py` and `trainer.py`; details on
  [repo-conventions](repo-conventions.md) and [flux-model](flux-model.md)). The imported helpers
  (`utils`, `workarounds`, `spmd_utils`, `tpu_job_config`, `datasets`) import only PyTorch and
  TorchTitan, so no TPU runtime is needed, but core breaks if `experiments/tpu` is removed
  or renamed, which is what the one-way rule exists to prevent. A softer case is new at this pin:
  the shared SDPA path in `torchtitan/models/common/attention.py` imports
  `experiments.tpu.unflatten_shim` lazily inside a `try/except ImportError`, so it survives the
  folder's removal but still couples core attention to the TPU experiment. Adding more such imports widens the
  gap; prefer the override and `ModelSpec` hooks.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [job-configuration](../concepts/job-configuration.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [repo-conventions](repo-conventions.md),
  [model-registry](model-registry.md), [config-system-conventions](config-system-conventions.md),
  [tpu-training-recipes](tpu-training-recipes.md), [tpu-vm-provisioning](tpu-vm-provisioning.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md), [graph-trainer](graph-trainer.md).

## Source
Extracted from `torchtitan/experiments/README.md` (kept in place); the companion rules are in
`.claude/rules/experiments.md`.
