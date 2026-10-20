---
title: "DeepSeek-V4 in TorchTitan: compressed sparse attention, HC branch mixing, MTP, and smoke-test status"
type: doc-concept
provenance: doc
source: torchtitan/models/deepseek_v4/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# DeepSeek-V4 model

## Definition
The DeepSeek-V4 README is a short component map plus a status note. Entry points are
`torchtitan.models.deepseek_v4.model_registry` and four training configs: `deepseek_v4_debugmodel`,
`deepseek_v4_mtp_debugmodel` (the optional multi-token-prediction path), `deepseek_v4_flash` and
`deepseek_v4_pro`. The files are `model.py` (decoder and block), `attention.py` ("sliding window
attention, heavily compressed attention, and compressed sparse attention"), `compressor.py` ("KV
compression and sparse-index selection"), `mhc.py` (branch mixing), `moe.py` (router and expert
wrapper), `sharding.py` (declarative TP, SP, EP, FSDP, DTensor and `spmd_types` sharding) and
`state_dict_adapter.py`. Status: the debug model has been smoke-tested on 4 GPUs with FSDP2, TP2,
EP2 and the `spmd_types` backend; "Checkpoint compatibility and larger-scale convergence validation
should be verified before using the larger configs for production training."

## In torchtitan (grounded)
- **Model and blocks.** [`DeepSeekV4Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/model.py#L101 "torchtitan/models/deepseek_v4/model#DeepSeekV4Model") carries hidden
  states as `[T, hc_mult, D]` (`hc_mult=4` branches) through
  [`DeepSeekV4TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/model.py#L30 "torchtitan/models/deepseek_v4/model#DeepSeekV4TransformerBlock") ("HC pre/post mixing around
  attention and FFN"). The mixing modules are [`HcSplitSinkhorn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/mhc.py#L16 "torchtitan/models/deepseek_v4/mhc#HcSplitSinkhorn")
  (mix logits -> pre, post and combination weights, Sinkhorn-normalized, 20 iterations by default),
  [`HcPre`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/mhc.py#L72 "torchtitan/models/deepseek_v4/mhc#HcPre"), [`HcPost`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/mhc.py#L120 "torchtitan/models/deepseek_v4/mhc#HcPost") and
  [`HcHead`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/mhc.py#L148 "torchtitan/models/deepseek_v4/mhc#HcHead").
- **Attention.** [`deepseek_v4 Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/attention.py#L353 "torchtitan/models/deepseek_v4/attention#Attention") has a
  low-rank query path (`wq_a -> q_norm -> wq_b`), a single `wkv`, a grouped low-rank output
  (`wo_a -> wo_b`) and a learned `attn_sink`; [`_make_v4_attn_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L172 "torchtitan/models/deepseek_v4/__init__#_make_v4_attn_config")
  picks the inner attention per layer from `compress_ratio`:
  [`SlidingWindowAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/attention.py#L280 "torchtitan/models/deepseek_v4/attention#SlidingWindowAttention") (ratio 1),
  [`HeavilyCompressedAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/attention.py#L301 "torchtitan/models/deepseek_v4/attention#HeavilyCompressedAttention") (HCA, any other ratio above 1; the docstring uses 128) or
  [`CompressedSparseAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/attention.py#L324 "torchtitan/models/deepseek_v4/attention#CompressedSparseAttention") (CSA, ratio 4). All three
  run on [`DSV4FlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/attention.py#L30 "torchtitan/models/deepseek_v4/attention#DSV4FlexInnerAttention"), which attends over the
  concatenation of sliding-window KV, compressed KV and a sink token: the window and HCA blocks are
  fixed `mask_mod` patterns, and CSA's per-query top-k compressed positions are "the only dynamic
  (index-based) part". Compressed KV comes from [`Compressor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/compressor.py#L31 "torchtitan/models/deepseek_v4/compressor#Compressor")
  and the CSA selection features from [`Indexer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/compressor.py#L120 "torchtitan/models/deepseek_v4/compressor#Indexer"); the
  docstring notes that the indexer auxiliary loss "is intentionally dropped for now".
- **MoE.** [`DeepSeekV4MoE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/moe.py#L94 "torchtitan/models/deepseek_v4/moe#DeepSeekV4MoE") with
  [`DeepSeekV4Router`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/moe.py#L34 "torchtitan/models/deepseek_v4/moe#DeepSeekV4Router"), a `TokenChoiceTopKRouter` subclass that
  uses a token-id hash table ([`_build_hash_routing_table`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/moe.py#L14 "torchtitan/models/deepseek_v4/moe#_build_hash_routing_table")) instead
  of learned selection on the first `n_hash_layers` (3) layers.
- **MTP.** [`MTPBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/mtp.py#L24 "torchtitan/models/deepseek_v4/mtp#MTPBlock") (one auxiliary depth, a V4 block with its
  own `e_proj` / `h_proj` and an `HcHead`), appended by
  [`_build_mtp_layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L535 "torchtitan/models/deepseek_v4/__init__#_build_mtp_layers") when `n_mtp_layers > 0`;
  [`deepseek_v4_mtp_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/config_registry.py#L70 "torchtitan/models/deepseek_v4/config_registry#deepseek_v4_mtp_debugmodel") sets `n_mtp_layers=1`
  and an MTP loss.
- **Registry, recipes, sharding.** [`deepseek_v4 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L1013 "torchtitan/models/deepseek_v4/__init__#model_registry")
  over [`deepseek_v4_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L1006 "torchtitan/models/deepseek_v4/__init__#deepseek_v4_configs") (`debugmodel` at 16384 context;
  [`_deepseek_v4_flash`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L742 "torchtitan/models/deepseek_v4/__init__#_deepseek_v4_flash") and
  [`_deepseek_v4_pro`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/__init__.py#L874 "torchtitan/models/deepseek_v4/__init__#_deepseek_v4_pro") registered at only 4096) reuses the
  DeepSeek-V3 parallelizer ([`parallelize_deepseekv3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/parallelize.py#L21 "torchtitan/models/deepseek_v3/parallelize#parallelize_deepseekv3"),
  re-exported as `parallelize_deepseek_v4`), the shared `pipeline_llm`, and
  [`DeepSeekV4StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/state_dict_adapter.py#L17 "torchtitan/models/deepseek_v4/state_dict_adapter#DeepSeekV4StateDictAdapter"), a subclass of the
  V3 adapter. Recipes: [`deepseek_v4_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/config_registry.py#L25 "torchtitan/models/deepseek_v4/config_registry#deepseek_v4_debugmodel"),
  [`deepseek_v4_flash`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/config_registry.py#L113 "torchtitan/models/deepseek_v4/config_registry#deepseek_v4_flash"),
  [`deepseek_v4_pro`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/config_registry.py#L156 "torchtitan/models/deepseek_v4/config_registry#deepseek_v4_pro"). Placements come from
  [`set_deepseek_v4_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/sharding.py#L308 "torchtitan/models/deepseek_v4/sharding#set_deepseek_v4_sharding_config"), including the sparse
  attention core ([`set_dsa_flex_attention_sharding`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v4/sharding.py#L76 "torchtitan/models/deepseek_v4/sharding#set_dsa_flex_attention_sharding")) and the
  compressor and indexer.
- **Smoke test.** The README's command uses
  [`data_parallel_shard_degree`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L160 "torchtitan/config/configs#ParallelismConfig.data_parallel_shard_degree"),
  [`tensor_parallel_degree`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L193 "torchtitan/config/configs#ParallelismConfig.tensor_parallel_degree") and
  [`expert_parallel_degree`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L326 "torchtitan/config/configs#ParallelismConfig.expert_parallel_degree") of 2, plus
  optionally `--parallelism.spmd_backend spmd_types`; CI runs the same shape as
  [`deepseek_v4_debugmodel_fsdp2_tp2_ep2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L176 "torchtitan_recipes/tests/models#deepseek_v4_debugmodel_fsdp2_tp2_ep2")
  (`deepseek_v4_fsdp+tp+ep` in `tests/integration_tests/models.py`).

> [!warning] Doc vs code
> - The README glosses `mhc.py` as "multi-token/head-coupled branch mixing modules". The code calls
>   them HC modules that mix `hc_mult` residual branches with Sinkhorn-normalized weights; nothing in
>   `mhc.py` is multi-token. Multi-token prediction is in `mtp.py`, which the README's component list
>   omits.
> - There is no `deepseek_v4/parallelize.py`; the model registry reuses the DeepSeek-V3 parallelizer.

## Why it matters / when it applies
DeepSeek-V4 is GPU-only in this checkout and has no `experiments/tpu` package. The blocker is the
attention core: every layer's inner attention is a `FlexInnerAttention` subclass whose sparsity is a
FlexAttention `BlockMask` plus `mask_mod` / `score_mod`, and `model_registry` exposes no
`attn_backend` switch. FlexAttention compiles through inductor/Triton, which is not the TPU path, so a
TPU port would need a Splash/Pallas kernel for window + compressed + sink attention and for CSA's
top-k gather (the class docstring already names override points: `_build_block_mask` or the whole
`forward`). The `[T, hc_mult, D]` residual layout also multiplies activation memory by `hc_mult` (4)
relative to a plain decoder. Treat the flash/pro configs as unvalidated, per the README.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [deepseek-v3-model](deepseek-v3-model.md), [model-registry](model-registry.md),
  [kernel-overrides](kernel-overrides.md), [distributed-conventions](distributed-conventions.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md).

## Source
Extracted from `torchtitan/models/deepseek_v4/README.md` (kept in place).
