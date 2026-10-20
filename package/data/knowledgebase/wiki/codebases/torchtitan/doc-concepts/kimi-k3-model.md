---
title: "Kimi K3 in TorchTitan: hybrid KDA + Gated MLA, attention residuals, LatentMoE, FSDP-only support"
type: doc-concept
provenance: doc
source: torchtitan/models/kimi_k3/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Kimi K3 model

## Definition
The Kimi K3 README describes a decoder that combines **Kimi Delta Attention (KDA)** linear-attention
layers with **Gated MLA** layers (69 KDA + 24 MLA), block-level **Attention Residuals (AttnRes)**,
**Stable LatentMoE** (16 of 896 experts per token, 2 shared experts, SiTU-GLU activation) and a
**MoonViT-V2** vision encoder. Its "Released Model Configuration" table follows the official model
card: 2.8T total / 104B active parameters, 93 layers (1 dense), hidden size 7168, 96 heads, context
1048576, LatentMoE dimension 3584 with 3072 hidden per expert, 160K vocabulary, and MXFP4 weights /
MXFP8 activations trained with quantization-aware training. Supported parallelism: FSDP2/HSDP only
(decoder per layer, vision encoder as a separate unit). Parity against HF on a reduced debug-sized
configuration: end-to-end KL 6.7634e-7 with 100% top-1/top-5 match, and vision features at cosine
1.000000.

## In torchtitan (grounded)
- **Decoder.** [`KimiK3Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/model.py#L289 "torchtitan/models/kimi_k3/model#KimiK3Model") (a `Decoder`) stacks
  [`KimiK3TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/model.py#L175 "torchtitan/models/kimi_k3/model#KimiK3TransformerBlock")s whose mixer is either
  [`KDA`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/kda.py#L182 "torchtitan/models/kimi_k3/kda#KDA") or [`KimiMLAAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/model.py#L58 "torchtitan/models/kimi_k3/model#KimiMLAAttention");
  [`_kimi_k3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/__init__.py#L511 "torchtitan/models/kimi_k3/__init__#_kimi_k3") places full attention at layers 3, 7, ..., 91 and 92
  (24 layers), so the other 69 are KDA. AttnRes is
  [`_apply_attention_residual`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/model.py#L152 "torchtitan/models/kimi_k3/model#_apply_attention_residual"): a softmax over the block residuals
  and the running prefix sum, computed in FP32, with a "TODO: Add TP Support".
- **KDA.** [`InnerKDA`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/kda.py#L112 "torchtitan/models/kimi_k3/kda#InnerKDA") runs the short convolution and
  [`KDAKernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/kda.py#L53 "torchtitan/models/kimi_k3/kda#KDAKernel"), which calls `chunk_kda` from `attn_gym.linear.kda`
  (a Triton kernel); `InnerKDA.Config` rejects any `head_dim` other than 128 ("Attention Gym KDA
  requires head_dim=128"). The output norm is [`KimiRMSNormGated`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/kda.py#L29 "torchtitan/models/kimi_k3/kda#KimiRMSNormGated").
- **LatentMoE.** [`KimiLatentMoE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/moe.py#L110 "torchtitan/models/kimi_k3/moe#KimiLatentMoE") extends the shared `MoE` with
  `routed_down -> routed_norm -> routed_up`: tokens are projected from `dim` to the latent width before
  the routed experts ([`KimiGroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/moe.py#L59 "torchtitan/models/kimi_k3/moe#KimiGroupedExperts")) and back afterwards,
  while shared experts ([`KimiFeedForward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/moe.py#L40 "torchtitan/models/kimi_k3/moe#KimiFeedForward")) run at full width;
  [`_latent_moe_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/__init__.py#L219 "torchtitan/models/kimi_k3/__init__#_latent_moe_config") builds it. SiTU-GLU is
  [`_situ_glu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/moe.py#L24 "torchtitan/models/kimi_k3/moe#_situ_glu") (`beta * tanh(gate / beta) * sigmoid(gate)`, optional
  tanh cap on the up branch, FP32).
- **Vision.** [`KimiK3VisionEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/vision_encoder.py#L47 "torchtitan/models/kimi_k3/vision_encoder#KimiK3VisionEncoder") with
  [`KimiK3VisionProjector`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/vision_encoder.py#L25 "torchtitan/models/kimi_k3/vision_encoder#KimiK3VisionProjector"), configured by
  [`_vision_encoder_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/__init__.py#L280 "torchtitan/models/kimi_k3/__init__#_vision_encoder_config").
- **Registry and recipes.** [`kimi_k3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/__init__.py#L554 "torchtitan/models/kimi_k3/__init__#model_registry")
  has two flavors, `debugmodel` and `Kimi-K3` (registered at a 262144 maximum context, below the
  released 1048576), sets `pipelining_fn=None`, and ships one recipe,
  [`kimi_k3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/config_registry.py#L62 "torchtitan/models/kimi_k3/config_registry#kimi_k3_debugmodel").
- **Parallelize.** [`parallelize_kimi_k3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/parallelize.py#L27 "torchtitan/models/kimi_k3/parallelize#parallelize_kimi_k3") raises
  `NotImplementedError` for TP, PP and CP ("Kimi K3 currently supports FSDP2 data parallelism only")
  and for model compilation ("does not support model compilation yet"). It does call
  `model.parallelize` under `spmd_types` or when EP is on, with expert placements from
  [`set_kimi_k3_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/sharding.py#L84 "torchtitan/models/kimi_k3/sharding#set_kimi_k3_sharding_config") (`enable_ep` follows
  `expert_parallel_degree`), so routed experts can be expert-parallel even though the README lists
  only FSDP2/HSDP.
- **Checkpoints.** [`KimiK3StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k3/state_dict_adapter.py#L26 "torchtitan/models/kimi_k3/state_dict_adapter#KimiK3StateDictAdapter") is the
  "HuggingFace checkpoint adapter for unquantized Kimi K3 weights"; it notes that the released index
  contains MXFP4 packed/scale FQNs while the adapter exports unquantized weights. Parity scripts:
  `scripts/checkpoint_conversion/numerical_tests_kimi_k3.py` and `tests/unit_tests/gpu/test_kimi_k3.py`.

> [!warning] Doc vs code
> - The README's configuration table describes the released checkpoint, not what TorchTitan trains:
>   there is no MXFP4/MXFP8 quantization-aware training path in `kimi_k3/`, the adapter handles only
>   unquantized weights, and the `Kimi-K3` flavor is registered at 262144 context rather than 1048576.
> - The README lists only FSDP2/HSDP; the parallelizer also supports EP for routed experts.
> - The prerequisites install `.ci/docker/requirements-vlm.txt`, which does not contain the KDA kernel
>   library; `attn-gym[linear]` comes from the base `.ci/docker/requirements.txt`.

## Why it matters / when it applies
Kimi K3 is the furthest core model from TPU in this checkout: no `experiments/tpu` package, a Triton
KDA kernel imported at module level for 69 of 93 layers (a TPU port needs a chunked KDA kernel, for
example in Pallas, much like Qwen3.5's GatedDeltaNet; see [qwen3-5-model](qwen3-5-model.md)), no
`torch.compile` support (the TPU backend compiles the model), and FSDP as the only dense
parallelism. On shapes, LatentMoE narrows routed-expert GEMMs to the latent width (3584 in the
released model, a multiple of 128) but 896 experts top-16 still gives small per-expert token groups
(see [model-definitions](../concepts/model-definitions.md)).

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [quantization](../concepts/quantization.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kimi-k2-7-model](kimi-k2-7-model.md), [qwen3-5-model](qwen3-5-model.md),
  [deepseek-v3-model](deepseek-v3-model.md), [checkpoint-conversion](checkpoint-conversion.md),
  [model-registry](model-registry.md).

## Source
Extracted from `torchtitan/models/kimi_k3/README.md` (kept in place).
