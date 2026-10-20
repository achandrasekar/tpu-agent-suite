---
title: "Muse Glimmer in TorchTitan: iRoPE + per-layer sliding windows, gated GQA, soft-capped head, optional vision, and its TPU lane"
type: doc-concept
provenance: doc
source: torchtitan/models/muse_glimmer/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Muse Glimmer model

## Definition
The Muse Glimmer README describes a TorchTitan port of the Muse Glimmer **text decoder** plus an
optional **vision/multimodal** path (a vision encoder + adapter that `MuseGlimmerModel` owns and runs
inside `forward`). Four configs: `muse_glimmer_debugmodel` and `muse_glimmer_30b` (`dim=6656`, 52
layers, 32 query / 2 KV heads, head dim 128, vocab 202048), each with an `_mm` multimodal twin. The
architecture highlights are gain-centered RMSNorm (effective scale `weight + gain_center`, weight
initialized to 0), scaleless RMSNorm for QK-norm and the embedding norm, GQA with QK-norm before RoPE,
a tuned query scale and a sigmoid output gate before `wo`, **iRoPE** (RoPE skipped on every fourth,
NoPE, layer; complex RoPE, theta 500000), a cyclic per-layer **sliding-window vs global** attention
pattern "implemented with the **Flex** attention backend currently", post-norm residuals with a
SwiGLU FFN, and a **`SoftCappedLinear` lm_head** that applies the output multiplier and tanh soft-cap
inside the head "so it stays correct under `ChunkedLossWrapper`". Parallelism: FSDP, HSDP, TP, SP, CP,
`torch.compile` and PP; the multimodal path supports CP, TP+CP+SP and TP+CP+PP+SP with vision
computation replicated across CP. The debug config uses the `c4_test` dataset and the test tokenizer
under `./tests/assets/tokenizer`.

## In torchtitan (grounded)
- **Model.** [`MuseGlimmerModel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L262 "torchtitan/models/muse_glimmer/model#MuseGlimmerModel") with
  [`MuseGlimmerTransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L155 "torchtitan/models/muse_glimmer/model#MuseGlimmerTransformerBlock"),
  [`RMSGainCenterNorm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L58 "torchtitan/models/muse_glimmer/model#RMSGainCenterNorm"),
  [`EmbeddingWithNorm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L223 "torchtitan/models/muse_glimmer/model#EmbeddingWithNorm") (embedding and its norm travel together,
  so PP splits keep them on one stage) and
  [`SoftCappedLinear`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L190 "torchtitan/models/muse_glimmer/model#SoftCappedLinear"), whose docstring explains why the
  transform sits in the head: it then runs per chunk inside
  [`ChunkedLossWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/loss.py#L515 "torchtitan/components/loss#ChunkedLossWrapper"), which applies `lm_head` itself.
- **Attention.** [`_build_muse_glimmer_attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L134 "torchtitan/models/muse_glimmer/__init__#_build_muse_glimmer_attention") builds
  [`muse_glimmer Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/model.py#L79 "torchtitan/models/muse_glimmer/model#Attention") (unfused `wq` / `wkv`,
  scaleless QK-norm, sigmoid `o_gate`). [`_layer_use_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L122 "torchtitan/models/muse_glimmer/__init__#_layer_use_rope")
  implements iRoPE with the module constant `_EVERY_N_LAYERS_NOPE = 4`, counting NoPE layers backward
  from the last layer, and [`_layer_window_size`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L127 "torchtitan/models/muse_glimmer/__init__#_layer_window_size") reads the
  cyclic `window_pattern` aligned to the same phase (`[2048, 2048, 2048, 0]` in the 30B flavor, 0 =
  global). Flex carries the windows as a window-keyed dict of `BlockMask`s; varlen is also handled
  (each layer's window baked into its kernel argument), though every upstream recipe passes
  `attn_backend="flex"`.
- **Flavors and registry.** [`_muse_glimmer_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L353 "torchtitan/models/muse_glimmer/__init__#_muse_glimmer_config"),
  [`_muse_glimmer_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L445 "torchtitan/models/muse_glimmer/__init__#_muse_glimmer_30b"),
  [`muse_glimmer_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L518 "torchtitan/models/muse_glimmer/__init__#muse_glimmer_configs") (`debugmodel` and `debugmodel_mm`
  at 4096 context, `30B` and `30B_mm` at 16384) and
  [`muse_glimmer model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/__init__.py#L526 "torchtitan/models/muse_glimmer/__init__#model_registry"), whose
  `pipelining_fn` is [`pipeline_with_first_stage_modules`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L144 "torchtitan/distributed/pipeline_parallel#pipeline_with_first_stage_modules")
  with `vision_encoder`, `vision_adapter`, `vision_projection` and `perception_emb_norm` pinned to the
  first stage. Recipes: [`muse_glimmer_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L113 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_debugmodel"),
  [`muse_glimmer_debugmodel_mm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L154 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_debugmodel_mm"),
  [`muse_glimmer_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L205 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_30b"),
  [`muse_glimmer_30b_mm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L243 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_30b_mm").
- **Vision.** [`MuseGlimmerVisionEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/vision_encoder.py#L208 "torchtitan/models/muse_glimmer/vision_encoder#MuseGlimmerVisionEncoder") and
  [`MuseGlimmerVisionAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/vision_encoder.py#L188 "torchtitan/models/muse_glimmer/vision_encoder#MuseGlimmerVisionAdapter"); the "bank indices
  prepared before token sharding" have their own placement,
  [`vision_bank_indices_placement`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/sharding.py#L42 "torchtitan/models/muse_glimmer/sharding#vision_bank_indices_placement"), and the
  TP-replicated fusion is declared in [`_set_multimodal_sharding`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/sharding.py#L108 "torchtitan/models/muse_glimmer/sharding#_set_multimodal_sharding").
- **Parallelize and sharding.** [`parallelize_muse_glimmer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/parallelize.py#L30 "torchtitan/models/muse_glimmer/parallelize#parallelize_muse_glimmer")
  with [`set_muse_glimmer_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/sharding.py#L55 "torchtitan/models/muse_glimmer/sharding#set_muse_glimmer_sharding_config"); checkpoints go
  through [`MuseGlimmerStateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/state_dict_adapter.py#L106 "torchtitan/models/muse_glimmer/state_dict_adapter#MuseGlimmerStateDictAdapter"). CI covers
  the README's combinations with [`muse_glimmer_debugmodel_fsdp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L421 "torchtitan_recipes/tests/models#muse_glimmer_debugmodel_fsdp8"),
  [`muse_glimmer_debugmodel_fsdp2_tp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L427 "torchtitan_recipes/tests/models#muse_glimmer_debugmodel_fsdp2_tp2_cp2"),
  [`muse_glimmer_debugmodel_mm_fsdp2_tp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L445 "torchtitan_recipes/tests/models#muse_glimmer_debugmodel_mm_fsdp2_tp2") and
  [`muse_glimmer_debugmodel_mm_tp2_cp2_pp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L458 "torchtitan_recipes/tests/models#muse_glimmer_debugmodel_mm_tp2_cp2_pp2").

> [!warning] Doc vs code
> - The README writes `every_n_layers_nope = 4` as if it were a setting; in code it is the
>   module-level constant `_EVERY_N_LAYERS_NOPE`, not a config field.
> - "Implemented with the Flex attention backend currently" understates the code: the model also
>   builds varlen windows; only the shipped recipes are flex-only.

## Why it matters / when it applies
Muse Glimmer is one of the few new models with a TPU lane. Upstream windows only work on flex and
varlen, neither of which runs on TPU, and upstream's SDPA inner attention "silently ignores the
window, so every sliding-window layer would run as full causal attention". `experiments/tpu/muse_glimmer`
fixes this with [`use_tpu_attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/__init__.py#L49 "torchtitan/experiments/tpu/muse_glimmer/__init__#use_tpu_attention"), which swaps
every layer to [`TpuMuseGlimmerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/attention.py#L218 "torchtitan/experiments/tpu/muse_glimmer/attention#TpuMuseGlimmerAttention") (passes
`positions` so packed documents are masked) and
[`SlidingWindowScaledDotProductInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/attention.py#L118 "torchtitan/experiments/tpu/muse_glimmer/attention#SlidingWindowScaledDotProductInnerAttention"),
which applies the window explicitly and exposes `sliding_window_size = window_size - 1` for Splash's
`LocalMask` (the two window conventions differ by one). The
[`TPU model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/__init__.py#L114 "torchtitan/experiments/tpu/muse_glimmer/__init__#model_registry") builds the
upstream flavors with `attn_backend="sdpa"`. The 30B recipes
([`_muse_glimmer_30b_base`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L128 "torchtitan/experiments/tpu/muse_glimmer/config_registry#_muse_glimmer_30b_base"),
[`muse_glimmer_30b_full_logits`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L205 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_full_logits"),
[`muse_glimmer_30b_fused_ce`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L248 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_fused_ce")) use SimpleFSDP,
selective AC and JAX Pallas Splash with a local mask for the 39 sliding-window layers and causal for
the 13 global ones; a recipe comment records Tokamax Splash as ~5% slower because it has no
document-aware block skipping. The fused-CE variants
([`_enable_pallas_cross_entropy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L50 "torchtitan/experiments/tpu/muse_glimmer/config_registry#_enable_pallas_cross_entropy")) cannot keep the soft-cap in
`lm_head`, because the fused kernel receives only the hidden states and head weight: they copy the
head's soft-cap and output multiplier into the Pallas loss config and override `SoftCappedLinear`. The
TPU parallelizer ([`TPU parallelize_muse_glimmer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/infra/parallelize.py#L124 "torchtitan/experiments/tpu/muse_glimmer/infra/parallelize#parallelize_muse_glimmer"))
rejects CP and PP ("not supported for Muse Glimmer on TPU yet"), so the README's CP/PP coverage is
GPU-only; with 2 KV heads, Ulysses CP is also capped at degree 2 (see
[context-parallel](../concepts/context-parallel.md)). Measured TPU7x numbers are on
[ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md).

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [context-parallel](../concepts/context-parallel.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md),
  [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md), [tpu-training-recipes](tpu-training-recipes.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md), [qwen3-5-model](qwen3-5-model.md).

## Source
Extracted from `torchtitan/models/muse_glimmer/README.md` (kept in place).
