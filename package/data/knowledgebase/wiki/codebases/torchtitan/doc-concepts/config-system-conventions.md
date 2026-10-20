---
title: "Config-system conventions: how TorchTitan config fields must be declared"
type: doc-concept
provenance: doc
source: .claude/rules/config.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Config-system conventions

## Definition
`.claude/rules/config.md` (scoped to `torchtitan/**`) is the maintainers' rule set for config
fields: no `None` default on a required field (give a meaningful default or make it required),
important and commonly used fields first, keyword-only arguments after the first positional, a
`num_` prefix for counts, descriptive names, and strings as selectors where that helps
extensibility (model selection, dataset selection). Two safety rules close it: values passed
through to external APIs must be valid on every code path, and an option that silently does not
take effect must warn. `torchtitan/config/README.md` adds the declaration rules that follow from the
frozen CLI: a new field still goes in the config tree, but it must be kept off the command line.

## In torchtitan (grounded)
- Configs are dataclasses nested inside the classes they build:
  [`Configurable`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configurable.py#L18 "torchtitan/config/configurable#Configurable") and its
  [`Configurable.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configurable.py#L41 "torchtitan/config/configurable#Configurable.Config"). A job is a
  [`Trainer.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L76 "torchtitan/trainer#Trainer.Config") tree selected by `--module` / `--config` and
  parsed by [`ConfigManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L29 "torchtitan/config/manager#ConfigManager"), whose
  [`_load_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L58 "torchtitan/config/manager#ConfigManager._load_config") resolves a short module name
  against the `torchtitan.models` / `torchtitan.experiments` allowlists or imports a fully
  qualified module path (see [job-configuration](../concepts/job-configuration.md)).
- **String selectors.** `--parallelism.spmd_backend` is a string
  ([`spmd_backend`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L199 "torchtitan/config/configs#ParallelismConfig.spmd_backend"): `"partial_dtensor"`
  or `"spmd_types"`) and attention backends are strings (`attn_backend="flex"`,
  `"varlen"`, `"sdpa"`) passed to each model's `model_registry`; the TPU recipes select `"sdpa"`
  this way rather than through a new class.
- **User-facing validation.** [`ParallelismConfig.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L286 "torchtitan/config/configs#ParallelismConfig.__post_init__")
  raises `ValueError` for an unknown `spmd_backend` or an empty CP load-balancer string, matching the
  `ValueError`-for-user-errors rule of [repo-conventions](repo-conventions.md).
- **Safe on every code path.** The rule's own example (DataLoader kwargs under `num_workers == 0`)
  no longer has a counterpart: the core loader is now
  [`GrainDataLoader`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L61 "torchtitan/components/data/loader#GrainDataLoader").
  The same rule shows up there instead: its
  [`__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L82 "torchtitan/components/data/loader#GrainDataLoader.__init__")
  raises `ValueError` when `repeat=False` is combined with data parallelism, because finite data
  "can exhaust ranks at different steps and hang collectives".
- **New fields are suppressed from the CLI.** The `--section.option` set is frozen. A new field on
  a component config, or in `configs.py` (for the few job-level options with no other home, such as
  [`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank"),
  added after discussing with the maintainers), is declared `Annotated[..., tyro.conf.Suppress]`.
  That keeps it settable from a recipe but off the command line.
  [`GrainDataLoader.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L65 "torchtitan/components/data/loader#GrainDataLoader.Config")
  is written this way (its `dataset`, `collator`, `shuffle`, `repeat`, `read_options` and
  `num_prefetch_batches` are all suppressed). Model-config fields need nothing, because
  [`Trainer.Config.model_spec`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L83 "torchtitan/trainer#Trainer.Config.model_spec")
  is suppressed and takes the whole model tree off the CLI. The unit test
  `tests/unit_tests/cpu/test_no_new_cli_options.py` pins the allowed option set. It skips config
  subclasses defined under `torchtitan/experiments` ("`torchtitan/experiments` sets its own rules").
- **Where validation goes.** A simple invariant over one config's fields goes in that config's
  `__post_init__`. A check that spans sections goes in `torchtitan/config/validation.py`, such as
  [`validate_context_parallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/validation.py#L20 "torchtitan/config/validation#validate_context_parallel"),
  which [`Trainer.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L132 "torchtitan/trainer#Trainer.Config.__post_init__")
  calls after its local checks (see [full-configurations-and-recipes](full-configurations-and-recipes.md)).

## Why it matters / when it applies
Configs are also the extension surface for TPU work: a TPU experiment adds knobs by subclassing
(the TPU lane uses [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig") with a nested
[`TPUConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L62 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig")), and the experiments rules forbid a
parallel argument parser. When adding a TPU knob, put it on the TPU config subclass, order it by
importance, and warn if it is ignored on some path (for example a flag that only applies under one
`compile_mode`). The CLI freeze test does not cover experiment subclasses, so a TPU knob stays
CLI-editable unless it is annotated `tyro.conf.Suppress`. Following the config README, prefer
setting it in a recipe function over adding another CLI flag.

**Where this checkout departs from the rule.** The fork's core
[`ParallelismConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L123 "torchtitan/config/configs#ParallelismConfig") starts with two TPU-oriented fields,
[`use_simple_fsdp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L124 "torchtitan/config/configs#ParallelismConfig.use_simple_fsdp") and
[`enable_tpu_topo_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L127 "torchtitan/config/configs#ParallelismConfig.enable_tpu_topo_mesh"), followed by
[`enable_min_cost_redistribute`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L132 "torchtitan/config/configs#ParallelismConfig.enable_min_cost_redistribute")
(a DTensor redistribution-planner switch whose docstring motivates it with multi-slice DCN), all ahead of the
data/tensor/pipeline degrees; that breaks "important fields first" and puts experiment knobs in a
core config, and `TPUConfig` separately carries its own `use_simple_fsdp`. Know which of the two a
code path reads before relying on either.

> [!warning] Doc vs code
> `.claude/rules/config.md` illustrates external-API safety with DataLoader kwargs and
> `num_workers == 0`. At this pin the core data loader is Grain-based and has no `num_workers`,
> `persistent_workers` or `prefetch_factor` config fields. The rule still applies, but its example
> is stale.

## Connections
- Code concepts: [job-configuration](../concepts/job-configuration.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [repo-conventions](repo-conventions.md),
  [experiments-policy](experiments-policy.md), [model-registry](model-registry.md),
  [full-configurations-and-recipes](full-configurations-and-recipes.md),
  [model-config-transforms](model-config-transforms.md).

## Source
Extracted from `.claude/rules/config.md` (kept in place). The field-declaration and
validation-placement rules come from `torchtitan/config/README.md`.
