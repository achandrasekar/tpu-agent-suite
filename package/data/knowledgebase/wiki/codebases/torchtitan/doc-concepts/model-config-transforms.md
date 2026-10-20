---
title: "Model config transforms: ordered, validated rewrites of a finished model config"
type: doc-concept
provenance: doc
source: torchtitan/config/transform/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Model config transforms

## Definition
`torchtitan/config/transform/README.md` defines a **model config transform** as a rewrite of a
complete model config tree, applied after `model_registry` has built the base model. Transforms are
"one supported way to build and maintain configs", and they are optional. The split of
responsibilities is the core idea. `model_registry` selects the base architecture, the attention
algorithm and the attention metadata format. A transform replaces or wraps configs inside the
built tree: context parallelism, TP GEMM backends, MoE communication backends, quantization and
LoRA. A CP transform "specializes the selected attention for distributed execution" but preserves
the selected algorithm and metadata format.

## In torchtitan (grounded)
- **Applying.** Set every training option first, then call [`apply_transforms`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/apply.py#L76 "torchtitan/config/transform/apply#apply_transforms") once. It deep-copies the
  `Trainer.Config`, rewrites `model_spec.model`, then re-runs `__post_init__` to validate the
  result. It returns the copy, so "the input config stays unchanged if a transform fails". Where
  there is no trainer config (a bare `ModelSpec`), [`transform_model_config_`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/apply.py#L61 "torchtitan/config/transform/apply#transform_model_config_") rewrites in place, returns the root and
  neither copies nor validates.
- **Ordering and conflicts.** Both entry points reject any pair where one transform lists the other
  in [`conflicts_with`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L26 "torchtitan/config/transform/base#ModelConfigTransform.conflicts_with"), then stable-sort by [`run_after`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L25 "torchtitan/config/transform/base#ModelConfigTransform.run_after") (class-level tuples on [`ModelConfigTransform`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L18 "torchtitan/config/transform/base#ModelConfigTransform")). A `run_after` cycle raises
  `ValueError`.
- **Writing one.** Subclass [`ModelConfigTransform`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L18 "torchtitan/config/transform/base#ModelConfigTransform") as a keyword-only dataclass and implement [`transform`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L29 "torchtitan/config/transform/base#ModelConfigTransform.transform"): rewrite in place
  and return the root, or return a different config only to replace the root. A transform sees only
  the model config, so any training or parallelism value it needs is passed in as a field.
  [`convert_config_type`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/base.py#L37 "torchtitan/config/transform/base#convert_config_type") swaps one config implementation for another while keeping its fields. It raises unless the
  replacement's `Config` inherits the current config type, which "preserves wrappers added by
  earlier transforms".
- **The one shipped transform.** [`ContextParallelTransform`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/context_parallel.py#L22 "torchtitan/config/transform/context_parallel#ContextParallelTransform") takes an [`inner_attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/transform/context_parallel.py#L32 "torchtitan/config/transform/context_parallel#ContextParallelTransform.inner_attention") class that must inherit [`CPInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/cp_attention.py#L39 "torchtitan/models/common/cp_attention#CPInnerAttention") (checked in
  `__post_init__`). It walks every `BaseAttention.Config` in the tree and converts its inner
  attention with `convert_config_type`. The CP inner attentions are [`KVAllGatherCPFlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/cp_attention.py#L55 "torchtitan/models/common/cp_attention#KVAllGatherCPFlexInnerAttention") (sharded Q, all-gathered
  K/V) and [`UlyssesCPFlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/cp_attention.py#L113 "torchtitan/models/common/cp_attention#UlyssesCPFlexInnerAttention") (sequence-to-head all-to-all). Both subclass [`FlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L233 "torchtitan/models/common/attention#FlexInnerAttention"), so the Flex algorithm and its
  `BlockMask` metadata are kept.
- **Validation after the last transform.** [`validate_context_parallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/validation.py#L20 "torchtitan/config/validation#validate_context_parallel"), called from [`Trainer.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L132 "torchtitan/trainer#Trainer.Config.__post_init__"), requires
  [`spmd_backend`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L199 "torchtitan/config/configs#ParallelismConfig.spmd_backend")` = "spmd_types"` when CP > 1. It also requires every attention to be a `CPInnerAttention`
  when CP > 1 (the error says "Apply ContextParallelTransform") and none when CP = 1. It requires one
  CP backend across all attentions, because "model inputs are sharded once", and for Ulysses a
  `None` load balancer plus a head count divisible by TP x CP.
- **Recipes that use it.** [`llama3_debugmodel_fsdp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L408 "torchtitan_recipes/tests/features#llama3_debugmodel_fsdp2_cp2") and `torchtitan_recipes/muse_glimmer.py` (all-gather CP 8 and
  Ulysses CP 2 variants of [`muse_glimmer_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L205 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_30b")) set the CP degree and then call `apply_transforms`.
- **Dependency direction.** `torchtitan/config/transform` may import other `torchtitan` packages;
  they must not import it, and recipes import and apply transforms. Shared types stay outside: for
  example, `CPInnerAttention` lives with the attention code in `torchtitan/models/common/`, and
  only the transform that installs it lives in the transform package.

## Why it matters / when it applies
- **Order of operations is part of the contract.** `__post_init__` also runs when a config is
  built, and the trainer validates again after command-line overrides. Set CP degree, load balancer
  and SPMD backend *before* `apply_transforms`, so that the final validation sees the real
  combination. Changing `context_parallel_degree` on the CLI after a recipe has applied CP is
  checked again, but switching CP on from the CLI alone fails validation, because the attention was
  never converted.
- **The base attention choice constrains CP.** The CP kernels at this pin are Flex-based, so a
  recipe that wants CP must start from a Flex `attn_backend` in `model_registry`. A model registered
  with [`VarlenInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L137 "torchtitan/models/common/attention#VarlenInnerAttention") (cumulative sequence offsets) or SDPA has no CP inner attention to convert to.
- **TPU reading.** No code under `torchtitan/experiments/tpu` subclasses `CPInnerAttention` or
  applies `ContextParallelTransform` at this pin, so TPU recipes (which pick SDPA or a TPU kernel
  override) get no CP from this layer. Enabling CP on TPU would take a `CPInnerAttention` subclass
  around the TPU attention kernel. Under the dependency rule that subclass would live next to the
  TPU attention code, and the transform would only install it *(unverified - agent inference from
  the transform contract)*. See [context-parallel](../concepts/context-parallel.md) and
  [long-context-training-recipe](long-context-training-recipe.md) (the TPU long-context recipe
  runs without CP).

> [!warning] Doc vs code
> - The README lists TP GEMM backends, MoE communication backends, quantization and LoRA as
>   transforms, and its example names `QuantizationTransform` and `LMHeadCastTransform`. Neither class
>   exists at this pin; `ContextParallelTransform` is the only transform. TP GEMM and MoE comm
>   backends are still `model_registry` keyword arguments (`tp_gemm_backend` on [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L387 "torchtitan/models/llama3/__init__#model_registry"),
>   `moe_comm_backend` on [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L686 "torchtitan/models/qwen3/__init__#model_registry")). Quantization and LoRA are still [`ModelConfigConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model.py#L19 "torchtitan/protocols/model#ModelConfigConverter") subclasses
>   ([`QuantizationConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/__init__.py#L21 "torchtitan/components/quantization/__init__#QuantizationConverter"), [`LoRAConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/lora.py#L141 "torchtitan/components/lora#LoRAConverter")) passed as `model_registry(converters=...)`.
> - The README example starts from `muse_glimmer_30b()` and adds CP 8. At this pin that recipe is
>   in `torchtitan_recipes/muse_glimmer.py`, which also sets the load balancer.

## Connections
- Code concepts: [job-configuration](../concepts/job-configuration.md) (transforms within the
  config flow), [context-parallel](../concepts/context-parallel.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md) (`ModelSpec`,
  `Module.Config`), [lora](../concepts/lora.md), [quantization](../concepts/quantization.md),
  [model-definitions](../concepts/model-definitions.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/config/transform/*`,
  `torchtitan/models/common/cp_attention`).
- Related doc-concepts: [full-configurations-and-recipes](full-configurations-and-recipes.md),
  [config-system-conventions](config-system-conventions.md), [model-registry](model-registry.md),
  [distributed-conventions](distributed-conventions.md),
  [long-context-training-recipe](long-context-training-recipe.md).

## Source
Extracted from `torchtitan/config/transform/README.md` (kept in place).
