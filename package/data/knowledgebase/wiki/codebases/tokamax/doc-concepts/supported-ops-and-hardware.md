---
title: "Supported ops and hardware (the documented GPU/TPU support matrix)"
type: doc-concept
provenance: doc
source: docs/supported_ops.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Supported ops and hardware (the documented GPU/TPU support matrix)

## Definition
`docs/supported_ops.md` and the README's "Status" section give the user-facing support matrix.
GPU kernels: `tokamax.dot_product_attention` (FlashAttention), `tokamax.gated_linear_unit`
(SwiGLU and friends) and `tokamax.layer_norm` (LayerNorm and RMSNorm). Both GPU and TPU:
`tokamax.ragged_dot` (Mixture of Experts). The README alone adds one TPU kernel,
`tokamax.linear_softmax_cross_entropy_loss` (memory-efficient linear cross-entropy). The README
also stresses that the library is "heavily under development", so incomplete features and API
changes are to be expected. `docs/splash_attention.md` is a stub that says only that splash
attention "is currently an experimental TPU op within Tokamax".

## In tokamax (grounded)
The docs describe a subset of what is registered at the pin. The full registry, with every
backend and its hardware gate, is the [kernel catalog](../concepts/kernel-catalog.md). The
documented ops map to code like this:

| Documented op | Public entry | TPU kernel at the pin |
|---|---|---|
| `dot_product_attention` ("GPU") | [`dot_product_attention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/api.py#L81 "tokamax/_src/ops/attention/api#dot_product_attention") | yes: [`PallasMosaicTpuFlashAttention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L63 "tokamax/_src/ops/attention/pallas_mosaic_tpu#PallasMosaicTpuFlashAttention") with [`PallasMosaicTpuFlashAttentionVjp`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L63 "tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp#PallasMosaicTpuFlashAttentionVjp"), reached through `"mosaic"` |
| `gated_linear_unit` (GPU) | [`gated_linear_unit`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/gated_linear_unit/api.py#L64 "tokamax/_src/ops/gated_linear_unit/api#gated_linear_unit") | no (XLA fallback) |
| `layer_norm` (GPU) | [`layer_norm`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/normalization/api.py#L46 "tokamax/_src/ops/normalization/api#layer_norm") | no (XLA fallback) |
| `ragged_dot` (GPU and TPU) | [`ragged_dot`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/api.py#L77 "tokamax/_src/ops/ragged_dot/api#ragged_dot") | yes: [`PallasMosaicTpuRaggedDot`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L90 "tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu#PallasMosaicTpuRaggedDot") (GMM v1), plus `mosaic_tpu_v2` by name |
| `linear_softmax_cross_entropy_loss` (TPU, README only) | [`linear_softmax_cross_entropy_loss`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py#L46 "tokamax/_src/ops/linear_softmax_cross_entropy_loss/api#linear_softmax_cross_entropy_loss") | yes: [`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L37 "tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss"), first in the default order |

Neither doc lists the TPU-only ops that are also registered. Each has its own page under
`../concepts/`:
- splash attention: [`splash_attention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/tpu/splash_attention/api.py#L49 "tokamax/_src/ops/experimental/tpu/splash_attention/api#splash_attention")
- Kimi delta attention: [`kimi_delta_attention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/api.py#L49 "tokamax/_src/ops/experimental/kda/api#kimi_delta_attention")
- batched ragged paged attention: [`batched_ragged_paged_attention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/batched_rpa/api.py#L43 "tokamax/_src/ops/experimental/batched_rpa/api#batched_ragged_paged_attention")
- MLA: [`PallasTpuMultiHeadLatentAttention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L41 "tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu#PallasTpuMultiHeadLatentAttention")
- SparseCore top-k: [`top_k`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/tpu/topk/api.py#L40 "tokamax/_src/ops/experimental/tpu/topk/api#top_k")
- ragged gather-reduce: [`ragged_gather_reduce`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_gather_reduce/api.py#L46 "tokamax/_src/ops/ragged_gather_reduce/api#ragged_gather_reduce")
- fused conv1d + gated delta rule: [`PallasMosaicTpuCausalConv1dGatedDeltaRule`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/causal_conv1d_gated_delta_rule/pallas_mosaic_tpu.py#L43 "tokamax/_src/ops/causal_conv1d_gated_delta_rule/pallas_mosaic_tpu#PallasMosaicTpuCausalConv1dGatedDeltaRule")
- fused EP MoE: [`PallasMosaicTpuFusedMoe`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/fused_moe/pallas_mosaic_tpu.py#L41 "tokamax/_src/ops/experimental/fused_moe/pallas_mosaic_tpu#PallasMosaicTpuFusedMoe")

None of these is re-exported from the top-level `tokamax` package. They are imported from
their `tokamax._src...` modules.

> [!warning] The support matrix in the docs is out of date relative to the code
> - `docs/supported_ops.md` omits `linear_softmax_cross_entropy_loss`. The README lists it, and
>   the TPU kernel exists and is the default: the LSCE API registers
>   [`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py#L37 "tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu#PallasMosaicTpuLinearSoftmaxCrossEntropyLoss")
>   as `mosaic_tpu` and prepends it to `_DEFAULT_IMPLEMENTATION`.
> - Both docs list `dot_product_attention` as GPU-only, but the attention API registers
>   [`PallasMosaicTpuFlashAttention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L63 "tokamax/_src/ops/attention/pallas_mosaic_tpu#PallasMosaicTpuFlashAttention") as
>   `mosaic_tpu`. On any non-NVIDIA device, `"mosaic"` (first in the default order) resolves to
>   it. It is built on the splash attention kernel.
> - None of the experimental TPU ops above appear in either doc.

## Why it matters / when it applies
- For TPU users the docs undersell the library. Attention, MoE matmul, LM-head cross-entropy
  and the serving and linear-attention kernels all have Mosaic TPU backends. Check the
  [kernel catalog](../concepts/kernel-catalog.md) rather than the docs when deciding whether a
  call will hit a Pallas kernel or the XLA fallback.
- The "GPU-only" ops (`layer_norm`, `gated_linear_unit`) still run on TPU through
  `implementation=None`, as plain XLA.
- Two ops are reachable from PyTorch on TPU through the torch_tpu bridge: LSCE and batched RPA.

## Connections
- Code concepts: [kernel-catalog](../concepts/kernel-catalog.md) (the authoritative matrix),
  [dot-product-attention](../concepts/dot-product-attention.md),
  [splash-attention](../concepts/splash-attention.md),
  [ragged-dot-gmm](../concepts/ragged-dot-gmm.md), [gmm-v2](../concepts/gmm-v2.md),
  [linear-softmax-cross-entropy](../concepts/linear-softmax-cross-entropy.md),
  [kimi-delta-attention](../concepts/kimi-delta-attention.md),
  [batched-ragged-paged-attention](../concepts/batched-ragged-paged-attention.md),
  [multi-head-latent-attention](../concepts/multi-head-latent-attention.md),
  [tpu-topk](../concepts/tpu-topk.md),
  [ragged-gather-reduce](../concepts/ragged-gather-reduce.md),
  [gated-delta-rule](../concepts/gated-delta-rule.md), [fused-moe](../concepts/fused-moe.md),
  [torch-tpu-bridge](../concepts/torch-tpu-bridge.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [implementation-selection-and-fallback](implementation-selection-and-fallback.md),
  [kda-chunked-forward-pipeline](kda-chunked-forward-pipeline.md).

## Source
Extracted from `docs/supported_ops.md`, `README.md` ("Status") and `docs/splash_attention.md`
(one-line stub), kept in place.
