---
title: "Kimi K2.5 / K2.6 / K2.7 in TorchTitan: DeepSeek-V3 decoder + MoonViT3d, QK clipping with DistMuon"
type: doc-concept
provenance: doc
source: torchtitan/models/kimi_k2_7/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Kimi K2.7 model family

## Definition
The `kimi_k2_7` README covers Kimi K2.5, K2.6 and K2.7-Code, which "share an architecture that pairs
a **DeepSeek-V3-style** decoder (Multi-head Latent Attention + Mixture-of-Experts) with a
**MoonViT3d** vision encoder". The encoder uses a linear patch embedding, learnable 2D spatial
position embeddings (plus a sinusoidal temporal term for video), 2D RoPE, pre-norm blocks, a temporal
mean-pool with 2x2 spatial merge and a 2-layer MLP projector; projected vision embeddings are
scattered into the text sequence at runs of the shared media placeholder token. Variants are a debug
model, moonlight-16B-A3B (text only), Kimi-VL-A3B and Kimi-K2.5 (`dim=7168`, 61 layers, 384 experts
top-8). All recipes train with DistMuon plus the **QK clipping** update from the Kimi K2 technical
report (`threshold=100.0`, `alpha=0.5`). Parallelism: FSDP/HSDP, EP and PP are supported; TP exists
in the model but DistMuon recipes reject it; CP, video training and INT4 checkpoint loading are TODO.

## In torchtitan (grounded)
- **Model.** [`KimiK25Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/model.py#L45 "torchtitan/models/kimi_k2_7/model#KimiK25Model") builds its decoder layers with the
  DeepSeek-V3 MLA + MoE builder ([`build_mla_moe_layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L175 "torchtitan/models/deepseek_v3/__init__#build_mla_moe_layers"), with a QK-clip attention factory; see
  [deepseek-v3-model](deepseek-v3-model.md)) and owns a
  [`KimiK25VisionEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/vision_encoder.py#L472 "torchtitan/models/kimi_k2_7/vision_encoder#KimiK25VisionEncoder"): a
  [`MoonViTEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/vision_encoder.py#L328 "torchtitan/models/kimi_k2_7/vision_encoder#MoonViTEncoder"), the temporal-pool / spatial-merge
  step [`_tpool_patch_merger`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/vision_encoder.py#L191 "torchtitan/models/kimi_k2_7/vision_encoder#_tpool_patch_merger") and the
  [`VisionProjector`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/vision_encoder.py#L287 "torchtitan/models/kimi_k2_7/vision_encoder#VisionProjector"). The scatter uses the same shared
  helpers as Qwen3.5 ([`scatter_vision_embeds`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/multimodal.py#L115 "torchtitan/models/common/multimodal#scatter_vision_embeds")).
- **Naming.** The directory and `MODULE` are `kimi_k2_7`, but the code keeps K2.5 names: the
  `ModelSpec` is named `kimi_k2_5`, flavors live in
  [`kimi_k2_5_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/__init__.py#L498 "torchtitan/models/kimi_k2_7/__init__#kimi_k2_5_configs") (`debugmodel`, `moonlight-16B-A3B`,
  `Kimi-VL-A3B`, `Kimi-K2.5`; see [`_kimi_k2_5`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/__init__.py#L410 "torchtitan/models/kimi_k2_7/__init__#_kimi_k2_5")) and the
  recipes are [`kimi_k2_5_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L91 "torchtitan/models/kimi_k2_7/config_registry#kimi_k2_5_debugmodel"),
  [`moonlight_16b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L133 "torchtitan/models/kimi_k2_7/config_registry#moonlight_16b_a3b"),
  [`kimi_vl_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L176 "torchtitan/models/kimi_k2_7/config_registry#kimi_vl_a3b") and
  [`kimi_k2_5`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L222 "torchtitan/models/kimi_k2_7/config_registry#kimi_k2_5").
- **QK clipping.** Every MLA layer's inner attention is a
  [`QKClipFlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/qk_clip.py#L27 "torchtitan/models/kimi_k2_7/qk_clip#QKClipFlexInnerAttention"), which asks FlexAttention for the
  per-head maximum score on each training forward (microbatches, PP stages and AC recompute included);
  [`_qk_clip_attention_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/__init__.py#L192 "torchtitan/models/kimi_k2_7/__init__#_qk_clip_attention_config") raises unless
  `attn_backend == "flex"`. After each optimizer step the hook installed by
  [`register_qk_clip_hook`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/qk_clip.py#L180 "torchtitan/models/kimi_k2_7/qk_clip#register_qk_clip_hook") runs
  [`qk_clip`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/qk_clip.py#L137 "torchtitan/models/kimi_k2_7/qk_clip#qk_clip"): it all-reduces the maxima (MAX) over the `loss`
  mesh (the data- and context-parallel ranks), computes `scale = threshold / max(max_logit, threshold)`
  per head, and rescales the Q/K projection weights, with the NoPE query rows taking `scale ** alpha`,
  RoPE query rows the full scale and key rows `scale ** (1 - alpha)`. The registry's
  `post_optimizer_build_fn` ([`_register_optimizer_hooks`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/__init__.py#L198 "torchtitan/models/kimi_k2_7/__init__#_register_optimizer_hooks"))
  installs this next to the MoE load-balancing hook.
- **DistMuon.** [`_dist_muon_optimizer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L294 "torchtitan/models/kimi_k2_7/config_registry#_dist_muon_optimizer") builds the
  [FlexShard](flexshard-dist-muon.md) [`DistMuon`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L252 "torchtitan/distributed/flex_shard/dist_muon#DistMuon") param
  groups, with per-expert compute layouts for routed experts;
  [`_align_dist_muon_expert_compute_layouts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L472 "torchtitan/models/kimi_k2_7/config_registry#_align_dist_muon_expert_compute_layouts") rebuilds
  them when the CLI overrides `expert_parallel_degree`, because that decides between the 1D
  `dp_shard` and the 2D EP/EFSDP expert layout. [`_KimiTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L519 "torchtitan/models/kimi_k2_7/config_registry#_KimiTrainerConfig")
  fails at config parsing when `tensor_parallel_degree > 1` ("tensor parallelism can produce
  unsupported _StridedShard parameter layouts"). The PP limit in the README surfaces in
  [`OptimizersContainer._build_param_groups`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/optimizer/optimizer.py#L173 "torchtitan/components/optimizer/optimizer#OptimizersContainer._build_param_groups"), which
  raises "matched no parameters" when a stage holding only `norm` and `lm_head` leaves a Muon group
  empty.
- **Parallelize and PP.** [`parallelize_kimi_k2_5`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/parallelize.py#L36 "torchtitan/models/kimi_k2_7/parallelize#parallelize_kimi_k2_5") raises
  `NotImplementedError` for CP; PP goes through
  [`pipeline_with_first_stage_modules`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L144 "torchtitan/distributed/pipeline_parallel#pipeline_with_first_stage_modules") with the vision
  encoder on the first stage. CI runs
  [`kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L380 "torchtitan_recipes/tests/models#kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2") and
  [`kimi_k2_5_debugmodel_muon_fsdp8_ep8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L402 "torchtitan_recipes/tests/models#kimi_k2_5_debugmodel_muon_fsdp8_ep8").
- **Checkpoints and numerics.** [`KimiK25StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/state_dict_adapter.py#L31 "torchtitan/models/kimi_k2_7/state_dict_adapter#KimiK25StateDictAdapter")
  inherits the DeepSeek-V3 FP8 block-scale path, which is why the released INT4 group-quantized 1T
  checkpoints cannot be loaded yet (the 1T config trains from scratch). The HF comparison
  (`scripts/checkpoint_conversion/numerical_tests_kimi.py`) covers the Kimi-VL compatibility flavor,
  not the 1T checkpoints: vision-feature cosine 0.999977, end-to-end last-token KL 4.3e-2 with top-1
  match, and KL 5.3e-4 when routing is pinned to HF's choices (a diagnostic, "not a normal end-to-end
  parity result"). Logits are bit-identical across no-parallel / FSDP / FSDP+EP and within bf16
  tolerance with TP+SP.

## Why it matters / when it applies
There is no TPU package for this family, and two of its defining pieces depend on GPU-only paths:
QK clipping reads per-head max scores from FlexAttention and refuses any other backend, so a TPU run
with `attn_backend="sdpa"` (the TPU recipes' choice) cannot use the shipped recipes as-is; the
Splash path would need to return per-head max logits for the clip to work. The decoder inherits
DeepSeek-V3's TPU shape risks (192-wide MLA query/key heads, small per-expert groups with 384 experts
top-8). The reusable parts are model-agnostic: the clipping math only needs per-head maxima and
rescales weights after the step, and the MoonViT3d scatter has the same host-side placeholder search
as Qwen3.5 (see [qwen3-5-model](qwen3-5-model.md) on dynamic vision shapes).

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [context-parallel](../concepts/context-parallel.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [deepseek-v3-model](deepseek-v3-model.md), [kimi-k3-model](kimi-k3-model.md),
  [flexshard-dist-muon](flexshard-dist-muon.md), [qwen3-5-model](qwen3-5-model.md),
  [checkpoint-conversion](checkpoint-conversion.md), [model-registry](model-registry.md).

## Source
Extracted from `torchtitan/models/kimi_k2_7/README.md` (kept in place).
