---
title: "Full configurations and torchtitan_recipes: runs are Python functions, the CLI is frozen"
type: doc-concept
provenance: doc
source: torchtitan/config/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Full configurations and torchtitan_recipes

## Definition
`torchtitan/config/README.md` describes a run as a **full configuration**: a Python function that
returns a complete [`Trainer.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L76 "torchtitan/trainer#Trainer.Config") holding the model, the parallelism degrees and every optimization choice.
"Building a new one is doing configuration programming with TorchTitan components." You select one
with `--module` (the module that defines the function) and `--config` (the function name). To
change behavior you write a new function, usually by calling an existing one and editing the
result, not by adding CLI flags. The `--section.option` flags still work and still take precedence,
but "only so existing scripts do not break"; the set is frozen and slated for removal.

## In torchtitan (grounded)
- **Selection.** [`ConfigManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L29 "torchtitan/config/manager#ConfigManager").[`parse_args`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L44 "torchtitan/config/manager#ConfigManager.parse_args") hands the args to [`_load_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L58 "torchtitan/config/manager#ConfigManager._load_config"), which requires both `--module` and `--config`.
  A short name (`llama3`) resolves to `<prefix>.<name>.config_registry` under `torchtitan.models`,
  `torchtitan.experiments` or the RL examples. Any other value is imported as a fully qualified
  module, trying `<module>.config_registry` first and then the module itself, so
  `--module torchtitan_recipes.tests.features` or an out-of-tree `my_company_configs.experiments`
  both work. The named function is called and its result becomes tyro's default, so remaining CLI
  flags edit it. `run_train.sh` passes `MODULE`/`CONFIG` (defaults `llama3` /
  `llama3_debugmodel`) through.
- **World size is not in the config.** The degrees live in [`ParallelismConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L123 "torchtitan/config/configs#ParallelismConfig"), but `NGPU` (or the launcher's
  process count) must still match their product.
- **`torchtitan_recipes` package.** A second top-level package next to `torchtitan`. At this pin it
  holds `tests/` (one module per integration suite: `features`, `models`, `h100`, `b200`, `flux`,
  `transformers_modeling_backend`, consumed by [`OverrideDefinitions`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L19 "tests/integration_tests/__init__#OverrideDefinitions")) and `muse_glimmer.py` (Muse Glimmer 30B CP
  recipes built from [`muse_glimmer_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L205 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_30b") plus a CP transform). Test configs derive from per-model baselines, for
  example [`llama3_debugmodel_fsdp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L408 "torchtitan_recipes/tests/features#llama3_debugmodel_fsdp2_cp2") builds on [`llama3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L63 "torchtitan/models/llama3/config_registry#llama3_debugmodel").
- **Per-model `config_registry.py`** modules are "the earlier location for the same thing". They
  keep working and "will eventually be deleted"; the model-size baselines (`llama3_8b` and the like)
  are to move to `torchtitan_recipes`, with no re-export shim planned.
- **Writing your own.** Add a function to a module named for the model, start from a
  `model_registry(...)` call such as [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L387 "torchtitan/models/llama3/__init__#model_registry") (which picks flavor and `attn_backend`), fill in a
  `Trainer.Config`, and name it on the command line.
- **The frozen CLI.** New features "express their knobs in the config tree instead". A new field on
  a component config, or in `configs.py` (which "is not closed", for options with no other home
  such as [`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank"), after discussing with the maintainers), needs `tyro.conf.Suppress`. Model-config
  fields need nothing, because [`model_spec`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L83 "torchtitan/trainer#Trainer.Config.model_spec") is suppressed. `tests/unit_tests/cpu/test_no_new_cli_options.py`
  enforces the frozen set.
- **Validation placement.** A simple invariant over one config's fields goes in its
  `__post_init__`. A check that spans sections goes in `torchtitan/config/validation.py`, for
  example [`validate_context_parallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/validation.py#L20 "torchtitan/config/validation#validate_context_parallel"), which [`Trainer.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L132 "torchtitan/trainer#Trainer.Config.__post_init__") calls after its local checks. The trainer re-runs the full validation
  after CLI and override edits ("Overrides may change any config field").
- **Optional transforms.** "Model config transforms are one supported way to modify a completed
  model config"; see [model-config-transforms](model-config-transforms.md).

## Why it matters / when it applies
- **One configuration per cluster and run.** "A different cluster usually means a different
  sharding layout, and therefore a different configuration." That needs no code change, only a new
  function. The README keeps recipes out of `torchtitan` because a configuration "changes on a
  different schedule from the library, and shipping one is not the same promise as shipping a
  class".
- **What gets shipped is deliberately small**: integration-test configs; golden configurations
  "verified on specific hardware, named for that hardware so a benchmark run is reproducible from
  its name alone"; and feature demos. The full product of model x degrees x optimization "is
  exponential", so your run is your own configuration, kept uncommitted or in your own package and
  selected with `--module`.
- **TPU reading.** TPU recipes are not in `torchtitan_recipes` at this pin. They live under
  `torchtitan/experiments/tpu` (per-topology recipe folders and TPU `config_registry` modules) and
  return a TPU subclass such as [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig"), so the experiments rule "use TorchTitan's config system"
  holds (see [experiments-policy](experiments-policy.md) and
  [tpu-training-recipes](tpu-training-recipes.md)). The README's model maps directly onto TPU
  work: a new slice shape or topology is a new function named for that hardware, derived from an
  existing one in a few lines, rather than a long CLI override string. CLI flags still work, but
  new TPU knobs belong in the config tree.

> [!warning] Doc vs code
> - The README says to "Re-run `pip install -e .`" so that `import torchtitan_recipes` works outside
>   the repository root. This checkout has no `pyproject.toml` or `setup.py` at the top level, so in
>   this fork running from the repository root (as `run_train.sh` and CI do) is the supported path.
> - The `torchtitan_recipes/llama3.py` and `llama3_8b_fsdp8_tp2_h200` names in the README are
>   illustrations. Only `muse_glimmer.py` and `tests/` exist in `torchtitan_recipes` at this pin,
>   and the model-size baselines are still in the per-model `config_registry.py` modules.

## Connections
- Code concepts: [job-configuration](../concepts/job-configuration.md) (the config tree,
  `ConfigManager`, overrides, transforms in depth),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md) (`ModelSpec`),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [context-parallel](../concepts/context-parallel.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/config/*`, `torchtitan_recipes/*`).
- Related doc-concepts: [config-system-conventions](config-system-conventions.md) (how fields are
  declared), [model-config-transforms](model-config-transforms.md),
  [model-registry](model-registry.md), [testing-strategy](testing-strategy.md) (test configs),
  [experiments-policy](experiments-policy.md), [tpu-training-recipes](tpu-training-recipes.md),
  [repo-conventions](repo-conventions.md).

## Source
Extracted from `torchtitan/config/README.md` (kept in place).
