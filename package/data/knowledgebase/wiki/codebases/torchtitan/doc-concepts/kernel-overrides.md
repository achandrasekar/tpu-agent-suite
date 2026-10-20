---
title: "Configurable overrides: swapping components and kernels by config (@override)"
type: doc-concept
provenance: doc
source: torchtitan/overrides/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Configurable overrides: swapping components and kernels by config (@override)

## Definition
The overrides README describes a way to "swap any `Configurable` in a torchtitan training
config - a model component, optimizer, loss, dataloader, and so on - for an alternative
implementation without editing in-repo code". An override module decorates a factory with
`@override(target=SomeComponent.Config, fqns=..., exact=...)` (keyword-only; there is no
separate name). The user lists each factory as a `module.function` target in
`--override.imports`. After the config is built and before any component is built, the
trainer walks the whole `Trainer.Config` tree and swaps every node the override claims for the
factory's output. Its core principles are explicit opt-in, strict scoping ("Each target names
exactly one override"), per-node (not per-class) conflict detection, and "configuration is
code": the factory bakes in defaults, with **optional per-entry kwargs** when two config trees
need the same override configured differently.

## In torchtitan (grounded)
- **API.** Everything lives in `torchtitan/config/override.py`:
  - [`override`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L175 "torchtitan/config/override#override")
    registers an [`Override`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L137 "torchtitan/config/override#Override")
    in a global table keyed by the factory's import path
    ([`name`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L156 "torchtitan/config/override#Override.name")
    = `module.function`); "a function may carry only one @override". A target that is not a
    `Configurable.Config` subclass (for example `ModelSpec`) raises `TypeError`.
  - [`derive`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L249 "torchtitan/config/override#derive")
    copies the fields that source and target share, then applies deltas. A delta that names an
    unknown field raises.
  - [`OverrideConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L61 "torchtitan/config/override#OverrideConfig")
    has a single [`imports`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L62 "torchtitan/config/override#OverrideConfig.imports")
    list of [`OverrideImport`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L57 "torchtitan/config/override#OverrideImport")
    entries: a `module.function` string or a `(target, kwargs)` tuple. On the CLI,
    [`parse_cli_imports`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L94 "torchtitan/config/override#parse_cli_imports")
    accepts space- or comma-separated targets and `target=<json-object>` tokens (quote them as
    one shell token), so values stay typed. The factory receives the kwargs after the matched
    config; a kwarg it does not accept raises `TypeError` at apply time. The README cites the RL
    trainer and generator activating one HybridEP dispatch override with opposite
    `capacity_factor` values.
- **Application.** [`Trainer.__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L344 "torchtitan/trainer#Trainer.__init__")
  calls [`apply_overrides`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L448 "torchtitan/config/override#apply_overrides")
  right after `model_config.update_from_config(...)` and before any build. That function does
  five things in order:
  1. Splits each target into module and function and imports the module
     ([`_resolve_target`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L313 "torchtitan/config/override#_resolve_target")).
     A target without a dot, or a failed import, raises.
  2. Looks each target up directly in the registry
     ([`_resolve_active`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L339 "torchtitan/config/override#_resolve_active")).
     A target naming no registered `@override` (for example a bare module path, the pre-bump
     form) raises, and so does listing the same target twice.
  3. Collects every claim from the original tree
     ([`_collect_claims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L386 "torchtitan/config/override#_collect_claims")).
     This walk uses [`Configurable.Config.traverse`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configurable.py#L75 "torchtitan/config/configurable#Configurable.Config.traverse"),
     and [`ModelSpec.traverse`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model_spec.py#L50 "torchtitan/protocols/model_spec#ModelSpec.traverse")
     exposes the model config under `model_spec.model`.
  4. Rejects claims on the same node or on an ancestor
     ([`_check_node_conflicts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L414 "torchtitan/config/override#_check_node_conflicts")).
  5. Calls `factory(cfg, **kwargs)` for each claim, swaps the node and logs
     `[Override] module.function: fqn Old -> New` (plus `with k=v` when kwargs were given).

  If the listed targets resolve but produce no replacement at all (for example an `fqns` glob
  that matches nothing), the code only logs a **warning**. It does not fail.
- **Targeting.** [`Override.matches`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L160 "torchtitan/config/override#Override.matches")
  runs `fnmatch` over the node's **full dotted path from the `Trainer.Config` root**, for
  example `model_spec.model.layers.0.feed_forward` or `optimizer`. `fnmatch` must match the
  whole string, and `*` also crosses `.`.
  - The README's globs start with `*.` (`*.layers.*.attention.*`, `*.layers.1[0-9].moe`), and
    that is the form that works.
  - The TPU `fused_loss` override confirms it: it uses the literal full path
    `fqns=["model_spec.model.lm_head"]`.
  - By default an override also claims subclasses of `target`. `exact=True` restricts it to the
    concrete class.
- **Ordering with converters.** [`ModelConfigConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model.py#L19 "torchtitan/protocols/model#ModelConfigConverter")
  converters (float8, MXFP8, LoRA) run earlier, inside `model_registry(...)`. Overrides
  therefore see the tree after conversion. The README's table: an override of `Linear.Config`
  collides with float8/LoRA, and one of `GroupedExperts.Config` may collide with the float8
  grouped-experts converter.

> [!warning] Doc vs code
> - The `@override` docstring still says the FQN is "the bare module FQN, e.g.
>   `layers.0.feed_forward`". Its example `fqns=["layers.1[0-9].moe"]` therefore matches
>   nothing, because no path starts with `layers.`. Such an override applies silently to zero
>   nodes, and you only get the "produced no replacements" warning if nothing else in the
>   listed targets matched either.
> - The README's "At a Glance" summary still says "only overrides registered by the listed
>   modules (provenance-checked) are applied" and shows kwargs as `module=<json>`. The code (and
>   the rest of the README) require a `module.function` target per override; a bare module path
>   raises "names no registered @override".
> - The README's file table lists only `fused_swiglu.py` and `helion_rope.py` as in-repo
>   overrides; at this pin `torchtitan/overrides/` also ships `offset_rmsnorm.py`,
>   `fused_mla.py` and `moe_token_dispatcher.py`.

## Why it matters / when it applies
The README gives three reasons for overrides: hardware-specific kernels that should not be held
to the core code-quality bar, larger fused regions (for example replacing all of `MoE.Config`
at once), and cheap team experiments. Each override author is responsible for three things. A
replacement module must declare its own `ShardingConfig` so that TP/FSDP compose. It must use
`register_state_dict_post_hook` / `register_load_state_dict_pre_hook` if it changes its
checkpoint layout. And it must wrap any custom kernel as a `torch.library` custom op with fake
and autograd registrations so that `torch.compile` can trace it. "We make no promise that any
core property survives an arbitrary override."

**On TPU.** This is how the TPU backend swaps in TPU-specific components without forking
models. The TPU recipes in `experiments/tpu/{llama3,qwen3,muse_glimmer}/config_registry.py`
append these `module.function` targets to `override.imports`:
- [`tpu_real_arithmetic_complex_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/real_rope.py#L90 "torchtitan/experiments/tpu/overrides/real_rope#tpu_real_arithmetic_complex_rope")
  writes complex RoPE in real arithmetic, so it can be traced inside `scan`.
- [`tpu_activation_dtype_cossin_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/bf16_rope.py#L86 "torchtitan/experiments/tpu/overrides/bf16_rope#tpu_activation_dtype_cossin_rope")
  evaluates RoPE in BF16. This matches MaxText, but it is not bit-exact with FP32.
- [`fused_loss_lm_head`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/fused_loss.py#L66 "torchtitan/experiments/tpu/overrides/fused_loss#fused_loss_lm_head")
  hands `(hidden, weight.T)` to [`PallasCrossEntropyLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/loss.py#L103 "torchtitan/experiments/tpu/loss#PallasCrossEntropyLoss");
  its sibling [`fused_loss_soft_capped_lm_head`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/fused_loss.py#L79 "torchtitan/experiments/tpu/overrides/fused_loss#fused_loss_soft_capped_lm_head")
  does the same for the Muse Glimmer soft-capped LM head (the cap is re-applied inside the loss
  kernel).
- [`tpu_discarded_token_counter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/discarded_token_counter.py#L52 "torchtitan/experiments/tpu/overrides/discarded_token_counter#tpu_discarded_token_counter")
  lets an MoE layer be traced inside `scan`.

All of them use `exact=True`. The in-repo CUDA examples (`fused_swiglu`, `helion_rope`,
`offset_rmsnorm`) do not help on TPU; see
[fused-swiglu-and-helion-rope](fused-swiglu-and-helion-rope.md). Do not list
`helion_cos_sin_rope` and `tpu_activation_dtype_cossin_rope` together: both claim every
`CosSinRoPE.Config` node, and the per-node conflict check will raise.

## Connections
- Code concepts: [job-configuration](../concepts/job-configuration.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [quantization](../concepts/quantization.md), [lora](../concepts/lora.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md).
- Module catalogs: [module map](../catalog/index.md) (`config/override`, `overrides/*`,
  `experiments/tpu/overrides/*`).
- Related doc-concepts: [fused-swiglu-and-helion-rope](fused-swiglu-and-helion-rope.md),
  [float8-training](float8-training.md), [mxfp8-training](mxfp8-training.md),
  [nvfp4-training](nvfp4-training.md), [rl-post-training](rl-post-training.md),
  [graph-trainer-autoresearch](graph-trainer-autoresearch.md).

## Source
Extracted from `torchtitan/overrides/README.md` (kept in place).
