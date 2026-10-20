---
title: "TorchTitan repo conventions: the maintainers' rules for structuring and changing code"
type: doc-concept
provenance: doc
source: .claude/CLAUDE.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# TorchTitan repo conventions

## Definition
`.claude/CLAUDE.md` is the maintainers' guide for agents working in TorchTitan. It sets seven core
principles: PyTorch-native training techniques only in core, root-cause before patching, reuse over
duplication, "don't leak experiments into core", protect battle-tested code paths, audit all
callsites (every model variant: llama3, llama4, qwen3, deepseek_v3, gpt_oss, flux), and no
speculative defensive checks. It adds a numerics bar for every change, a code-style section
(ASCII-only comments, naming, shape-suffix tensor names, code placement, errors) and PR
expectations. The top-level `README.md` frames the checkout itself: this is Google's distribution of
TorchTitan that adds Cloud TPU training through TorchTPU, with TPU support confined to
`torchtitan/experiments/tpu` and upstream TorchTitan otherwise working as documented. Path-scoped
details live in `.claude/rules/`; see
[config-system-conventions](config-system-conventions.md),
[distributed-conventions](distributed-conventions.md),
[model-authoring-conventions](model-authoring-conventions.md) and
[experiments-policy](experiments-policy.md).

## In torchtitan (grounded)
- **Numerics validation.** A non-computation change (activation checkpointing, refactors) must give
  *identical* loss before and after with `--debug.seed=42 --debug.deterministic`; a computation change
  needs loss convergence on a representative dataset such as C4. The guide points at
  `scripts/loss_compare.py`, which reads loss and grad_norm back from TensorBoard
  ([`extract_metrics_from_tensorboard`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L163 "scripts/loss_compare#extract_metrics_from_tensorboard"), command assembly in
  [`build_training_command`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L359 "scripts/loss_compare#build_training_command")) because stdout prints only five significant
  digits. `--debug.deterministic_warn_only` is forbidden. Performance runs need at least 10 steps.
- **Benchmark submissions.** `benchmarks/README.md` asks community performance reports to state the
  entity, model or theme, hardware and interconnect, the full training config (preferably the
  complete config printed by the run, because defaults not shown in a recipe can change), the
  versions and date of `torchtitan`, `torch`, `torchao` and other dependencies, and anything else
  needed to reproduce, in a file named `[model/theme]_[hardware]_[date/time]_[entity].md`. The
  print-config switch is [`DebugConfig.print_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L422 "torchtitan/config/configs#DebugConfig.print_config")
  (`--debug.print_config`) at this pin.
- **Code placement.** Model-agnostic parallelism helpers go in `torchtitan/distributed/` (for
  example [`NoParallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/tensor_parallel.py#L19 "torchtitan/distributed/tensor_parallel#NoParallel"),
  [`apply_fsdp_to_decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L189 "torchtitan/distributed/fsdp#apply_fsdp_to_decoder")); shared model components go in
  `torchtitan/models/common/` (for example [`Decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L67 "torchtitan/models/common/decoder#Decoder"),
  [`GroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L440 "torchtitan/models/common/moe#GroupedExperts")); only model-specific code goes in a model folder.
- **Naming: `axis` vs `dim`.** `axis`/`axes` names a specific `DeviceMesh` axis (TP, `dp_shard`);
  `dim` is for tensor dimensions and the mesh's shape. Upstream spellings such as
  `mesh_dim_names` are matched only at the call site. The mesh-owning class is
  [`ParallelDims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L105 "torchtitan/distributed/parallel_dims#ParallelDims").
- **Shape-suffix tensor names** (Noam Shazeer convention): `x_BLD`, `q_BLHK`, `out_THV`, with the
  letters describing *logical* dimensions (not the local shard) and a per-file legend. The attention
  sharding helpers key their layouts by such names (`q_THK`, `v_THV`, `q_BLHK`). The MoE code
  is the clearest example: [`GroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L440 "torchtitan/models/common/moe#GroupedExperts") holds `w1_EFD`, `w2_EDF`,
  `w3_EFD`, and [`TokenChoiceTopKRouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L586 "torchtitan/models/common/moe#TokenChoiceTopKRouter") works on `*_E` tensors.
- **Errors.** `ValueError` for user-facing errors (bad config), `assert` only for internal
  invariants, explicit validation of mesh axes and placements in distributed code, and a warning
  whenever user configuration is silently skipped.
- **Parameters and config.** Important parameters first, keyword-only after the first positional,
  no `None` defaults for required config fields, and `dataclasses.replace()` treated as a shallow
  copy.
- **PR bar.** `pre-commit run --all-files` first, a loss comparison for any non-trivial change, a
  "why" in the description, CPU unit tests at minimum (GPU integration tests for parallelism), and
  original checkpoints must still load after model changes.

## Why it matters / when it applies
A contributor adding TPU support is bound mainly by principle 4, "don't leak experiments into core"
("don't add `if experiment_x:` branches to core files"), and principle 1: core training and
parallelism code must not depend on non-PyTorch libraries. The upstream-intended path for TPU is
therefore an experiment folder that reuses core through its extension points (see
[experiments-policy](experiments-policy.md)), not TPU branches in `trainer.py` or `models/common/`.

**Where changes go in this fork.** The top-level `README.md` (and `CONTRIBUTING.md`, which it points
to) says TPU bugs and feature requests go to this repository's issue tracker and TPU changes go
here, under `torchtitan/experiments/tpu`; anything in shared TorchTitan code goes upstream to
`pytorch/torchtitan`, because the fork is kept in sync with it. Its "start here" list is the TPU
README (environment, building the TorchTPU wheel, single-host TPU VM and multi-host GKE runs), the
per-model, per-topology recipes under `torchtitan/experiments/tpu/recipes` (see
[tpu-training-recipes](tpu-training-recipes.md)) and the TorchTPU optimization guide; upstream's
README and docs apply unchanged for everything else. The practical reading: a core edit made only in
this fork is either reverted by the next upstream sync or becomes a hand-resolved conflict.

**This checkout disagrees with that rule in several places.** TPU support has partly leaked into
core:
- `torchtitan/distributed/activation_checkpoint.py` imports
  [`ptd_checkpoint_wrapper_with_early_stop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/utils.py#L14 "torchtitan/experiments/tpu/utils#ptd_checkpoint_wrapper_with_early_stop") in place of the
  upstream `checkpoint_wrapper` (the upstream import is left commented out).
- The core Flux model imports the TPU experiment: [`FluxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/trainer.py#L38 "torchtitan/models/flux/trainer#FluxTrainer")'s
  config inherits [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig"), the Flux recipes set
  a [`TPUConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L62 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig"), and
  [`parallelize_flux`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/parallelize.py#L52 "torchtitan/models/flux/parallelize#parallelize_flux") calls
  [`use_splash_attention_patch`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L33 "torchtitan/experiments/tpu/workarounds#use_splash_attention_patch") and
  [`apply_spmd_safe`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/spmd_utils.py#L17 "torchtitan/experiments/tpu/spmd_utils#apply_spmd_safe") (see [flux-model](flux-model.md)).
- Core has device-type branches for TPU:
  [`ScaledDotProductInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L393 "torchtitan/models/common/attention#ScaledDotProductInnerAttention") switches its SDPA backends to
  `OVERRIDEABLE` + `MATH` when a TPU device module is available, and
  [`init_device_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L24 "torchtitan/distributed/parallel_dims#init_device_mesh") builds a topology-aware mesh on
  `device_type == "tpu"`.
Treat these as fork-local deviations: they would not pass the upstream review bar this guide
describes, and they make core depend on `experiments/tpu`, which inverts the one-way dependency
the experiments policy requires. When adding new TPU behavior, prefer the experiment-side hooks
(overrides, a custom `ModelSpec`, `TPUTrainer`) over growing this list.

The numerics rule also has a TPU reading: the bit-identical-loss check assumes the same
parallelisms and deterministic kernels on the same hardware; comparing a TPU run against a GPU
baseline is a convergence comparison, not a bitwise one.

> [!warning] Doc vs code
> - Principle 6 lists `llama4` among the variants to audit; there is no `torchtitan/models/llama4`
>   at this pin. The core model folders are `deepseek_v3`, `deepseek_v4`, `flux`, `gpt_oss`,
>   `kimi_k2_7`, `kimi_k3`, `llama3`, `muse_glimmer`, `qwen3`, `qwen3_5`, `qwen3_6` and `qwen3_8`.
> - `benchmarks/README.md` points `--print_config` at an old upstream `torchtitan/config_manager.py`;
>   that file does not exist here, and the switch is `--debug.print_config`.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [job-configuration](../concepts/job-configuration.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [config-system-conventions](config-system-conventions.md),
  [distributed-conventions](distributed-conventions.md),
  [model-authoring-conventions](model-authoring-conventions.md),
  [experiments-policy](experiments-policy.md), [model-registry](model-registry.md),
  [full-configurations-and-recipes](full-configurations-and-recipes.md),
  [testing-strategy](testing-strategy.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md).

## Source
Extracted from `.claude/CLAUDE.md` (kept in place). The fork-scope and contribution routing come
from the top-level `README.md`; the benchmark-submission rules from `benchmarks/README.md`.
