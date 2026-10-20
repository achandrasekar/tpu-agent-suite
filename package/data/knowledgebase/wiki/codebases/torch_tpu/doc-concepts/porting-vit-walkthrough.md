---
title: "Porting walkthrough: training a tiny ViT on MNIST with TorchTPU"
type: doc-concept
provenance: doc
source: docs/markdown/porting_vit_mnist.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Porting walkthrough: training a tiny ViT on MNIST with TorchTPU

## Definition
The ViT tutorial is the guide's end-to-end porting example. It takes an ordinary PyTorch
training script and runs it on `tpu`:
- **Model.** A "Tiny ViT" with `Conv2d` patch embedding (kernel = stride = 4 on 32x32 inputs),
  4 attention blocks (hidden 128, 4 heads), mean pooling and a linear head.
- **Data.** 1000 MNIST training images resized to 32x32.
- **Training.** Adam and cross-entropy, run in plain eager mode, followed by single-image
  inference.

Only two lines change from a CPU or GPU script: `device = torch.device("tpu")` and
`.to(device)` on the model and on each batch. The rest of the tutorial is the habits that make
eager TPU training efficient.

## In torch_tpu (grounded)
- **Attention written as explicit matmuls.** `TpuAttentionBlock` computes
  `softmax(q @ k^T / sqrt(d)) @ v` by hand, with a comment saying this "avoids SDP MATH
  fallback on TPU".
  - At this pin, `F.scaled_dot_product_attention` is routed by
    [`AtenFusedSdpChoice`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.cc#L123 "csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.cc#AtenFusedSdpChoice").
    It picks the Mosaic flash kernel when the inputs are static, `float32` or `bfloat16`,
    equal-rank, rank 4 or more, and pass the head-dim rule. Otherwise it picks the StableHLO
    overrideable path, which needs equal ranks, rank 4 or more, no dropout and floating-point
    inputs. It drops to PyTorch's math decomposition only when neither applies, and warns once.
  - The hand-written block is therefore a conservative choice, not a requirement. Its matmuls
    lower directly through the `bmm`/`mm` kernels.
- **Per-batch host-to-device copies.** Each batch is sliced on the host and moved with
  `.to(device)`. That goes through
  [`AtenCopyFrom`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/copy_from_aten_kernels.cc#L232 "csrc/ops/copy_from/copy_from_aten_kernels.cc#AtenCopyFrom")
  ([host-device-transfer](../concepts/host-device-transfer.md)). Every op in forward, backward
  and the optimizer step is recorded by
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp")
  without blocking the host.
- **Loss accumulated on device.** The loop adds `loss.detach()` into a device tensor and calls
  `.item()` once per epoch. `.item()` is
  [`AtenLocalScalarDense`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/copy_from_aten_kernels.cc#L254 "csrc/ops/copy_from/copy_from_aten_kernels.cc#AtenLocalScalarDense"),
  a device-to-host transfer that materializes all pending work
  ([deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)).
  Calling it every step would stall the host on each device step. Calling it once per epoch
  amortizes the "expensive TPU -> CPU DMA transfer".
- **Inference.** Inference runs under `torch.no_grad()` and brings back only the `argmax` with
  `.cpu().item()`.

## Why it matters / when it applies
- **It is the template for a first port.** Change only the device, keep the model in eager mode,
  then remove per-step host syncs (`.item()`, `.cpu()`, printing tensors) before reaching for
  `torch.compile`.
- **Keep scalar logging on device.** Logging the loss every step is the most common hidden
  serialization in ported training loops. Accumulate the detached loss on device and read it at
  epoch or logging-interval boundaries.
- **Fit SDPA to the flash conditions.** Code that uses `F.scaled_dot_product_attention` can
  keep it if the inputs meet those conditions. Watch for the one-time performance warning
  naming the unmet condition.
- **Silence a known warning if needed.** The tutorial filters a "`.grad` attribute of a Tensor
  that is not a leaf Tensor" warning raised during training. It is noise, not an error.
- **Five epochs is only a smoke test.** The tutorial notes the model needs about 30 epochs
  before it starts to converge.

## Connections
- Code concepts: [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [host-device-transfer](../concepts/host-device-transfer.md),
  [scaled-dot-product-attention](../concepts/scaled-dot-product-attention.md),
  [synchronization](../concepts/synchronization.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/ops/copy_from`,
  `csrc/ops/scaled_dot_product_attention`).
- Related doc-concepts: [tpu-device-onboarding](tpu-device-onboarding.md),
  [eager-execution-modes](eager-execution-modes.md),
  [async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md),
  [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md),
  [numerical-parity-checking](numerical-parity-checking.md).

## Source
Extracted from `docs/markdown/porting_vit_mnist.md` (kept in place).
