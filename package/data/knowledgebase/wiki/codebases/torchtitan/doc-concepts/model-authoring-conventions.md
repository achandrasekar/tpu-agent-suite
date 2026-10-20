---
title: "Model-authoring conventions: minimal models, shared components, audit every variant"
type: doc-concept
provenance: doc
source: .claude/rules/models.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Model-authoring conventions

## Definition
`.claude/rules/models.md` (scoped to `torchtitan/models/**`) says model files hold architecture,
not training infrastructure; weight initialization belongs in the config or a dedicated init
function; original checkpoints must still load after any model change. A change to a shared
component (attention, normalization, MoE routing) must be applied to **all** variants (llama3,
llama4, qwen3, deepseek_v3, gpt_oss, flux, ...). Near-identical code is consolidated into
`torchtitan/models/common/` or `torchtitan/distributed/` with at most one general wrapper; a feature
only one model needs stays in that model's folder; routing and other control flow stays visible in
`forward`.

## In torchtitan (grounded)
- **Shared components** named by the rule all exist in `torchtitan/models/common/`:
  attention ([`GQAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L999 "torchtitan/models/common/attention#GQAttention"),
  [`BaseAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L745 "torchtitan/models/common/attention#BaseAttention")), decoder blocks
  ([`Decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L67 "torchtitan/models/common/decoder#Decoder"), [`TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L46 "torchtitan/models/common/decoder#TransformerBlock")),
  MoE ([`MoE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L934 "torchtitan/models/common/moe#MoE"), [`TokenChoiceTopKRouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L586 "torchtitan/models/common/moe#TokenChoiceTopKRouter"),
  [`GroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L440 "torchtitan/models/common/moe#GroupedExperts")) and RoPE
  ([`RoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/rope.py#L84 "torchtitan/models/common/rope#RoPE"), [`ComplexRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/rope.py#L185 "torchtitan/models/common/rope#ComplexRoPE"),
  [`CosSinRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/rope.py#L281 "torchtitan/models/common/rope#CosSinRoPE")).
  Normalization is [`RMSNorm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/nn_modules.py#L134 "torchtitan/models/common/nn_modules#RMSNorm")
  in `nn_modules.py`, and MoE is the single file `moe.py` (with `moe_sharding.py` and
  `token_dispatcher.py`), not the `rmsnorm.py` and `moe/` the rule names.
- **Init in config, not `__init__`.** Each module config carries a `param_init` dict, and
  [`init_states`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L109 "torchtitan/protocols/module#Module.init_states") walks the tree; buffers that need a device
  (RoPE caches, MoE counters) override [`_init_self_buffers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L242 "torchtitan/protocols/module#Module._init_self_buffers").
- **Standard folder.** The rule lists `config_registry.py`, `parallelize.py` and model files; in
  practice each core model also has `sharding.py`, `state_dict_adapter.py` and a `model_registry` in
  its `__init__.py` (the fuller layout is on [model-registry](model-registry.md)). The config README
  marks the per-model `config_registry.py` modules as the earlier home of full configurations: they
  keep working but "will eventually be deleted", with the model-size baselines moving to
  `torchtitan_recipes` (see [full-configurations-and-recipes](full-configurations-and-recipes.md)).
  Options that wrap or replace parts of a finished model config (CP today) are applied as
  [model-config-transforms](model-config-transforms.md), not added to the model folder.
- **Over-specialization is visible in the code.** Model-specific attention stays per model
  ([`deepseek_v3 Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L32 "torchtitan/models/deepseek_v3/model#Attention") for MLA,
  [`gpt_oss Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/model.py#L46 "torchtitan/models/gpt_oss/model#Attention") for sinks,
  [`Qwen35Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/model.py#L87 "torchtitan/models/qwen3_5/model#Qwen35Attention") and
  [`GatedDeltaNet`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L349 "torchtitan/models/qwen3_5/gdn#GatedDeltaNet"), now in its own
  `gdn.py`) while all of them subclass the shared
  `BaseAttention` or `Module`.

## Why it matters / when it applies
For TPU work the rule has three direct consequences:
- **Patch shared components once, for every model.** A TPU fix to attention or MoE belongs in one
  shared place (or one TPU-side override that works for every model), not a per-model copy. The TPU
  experiment follows this: its attention and grouped-matmul replacements are generic patches
  ([`use_splash_attention_patch`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L33 "torchtitan/experiments/tpu/workarounds#use_splash_attention_patch"),
  [`use_gmm_kernel_patch`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L536 "torchtitan/experiments/tpu/workarounds#use_gmm_kernel_patch"), applied through
  [`apply_patches`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L667 "torchtitan/experiments/tpu/workarounds#apply_patches")) and config overrides
  ([`tpu_real_arithmetic_complex_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/real_rope.py#L90 "torchtitan/experiments/tpu/overrides/real_rope#tpu_real_arithmetic_complex_rope"),
  [`fused_loss_lm_head`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/fused_loss.py#L66 "torchtitan/experiments/tpu/overrides/fused_loss#fused_loss_lm_head")) that rewrite the shared
  modules rather than forking model files.
- **Keep checkpoints loadable.** TPU overrides that change module structure (a fused loss head, real
  arithmetic RoPE) must keep the FQNs that the model's state-dict adapter maps, or HF checkpoints
  stop loading; see [state-dict-adapters](../concepts/state-dict-adapters.md).
- **Control flow in `forward`.** Data-dependent branching hidden in helpers is also what breaks full
  graph capture on TPU; keeping it in `forward` makes it visible when a graph break or recompile
  appears.

> [!warning] Doc vs code
> - "Audit all variants" lists `llama4`; there is no `torchtitan/models/llama4` at this pin. The core
>   variants to audit are `deepseek_v3`, `deepseek_v4`, `flux`, `gpt_oss`, `kimi_k2_7`, `kimi_k3`,
>   `llama3`, `muse_glimmer`, `qwen3`, `qwen3_5`, `qwen3_6` and `qwen3_8`.
> - The shared-components list names `rmsnorm.py` and a `moe/` directory under
>   `torchtitan/models/common/`. Neither exists: `RMSNorm` lives in `nn_modules.py` and MoE in
>   `moe.py`.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md), [repo-conventions](repo-conventions.md),
  [distributed-conventions](distributed-conventions.md), [qwen3-model](qwen3-model.md),
  [deepseek-v3-model](deepseek-v3-model.md), [qwen3-5-model](qwen3-5-model.md),
  [kernel-overrides](kernel-overrides.md), [model-config-transforms](model-config-transforms.md).

## Source
Extracted from `.claude/rules/models.md` (kept in place).
