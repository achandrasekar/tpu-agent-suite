---
title: "Numerical parity checking: TPU versus a CPU reference"
type: doc-concept
provenance: doc
source: docs/markdown/numerical_parity.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Numerical parity checking: TPU versus a CPU reference

## Definition
When a model is ported to TPU, its outputs will differ slightly from a CPU (or GPU) reference.
The doc attributes this to hardware rather than to bugs:
- **`bfloat16`-native MXUs.** The MXU multiplies `bfloat16`, and `float32` matmuls are emulated
  with several `bfloat16` passes.
- **Accumulation paths.** These differ from GPU `float32`/TF32 accumulation.
- **Operation order.** The systolic array orders and rounds operations differently from
  sequential CPU loops.
- **Fusion.** XLA fuses ops such as multiply-add into single instructions.

A parity check therefore compares like with like and judges the *shape* of the mismatch, not
just its presence. The workflow is for verification only, not training:
1. Build the model on CPU.
2. Build an identical model on `tpu`.
3. Copy the CPU `state_dict` into the TPU model so both start from the same weights.
4. Cast both models to the same dtype (for example `bfloat16`).
5. Run the same input through both.
6. Compare with `torch.testing.assert_close(out_tpu.cpu(), out_cpu)`.

## In torch_tpu (grounded)
- **Weight sync.** `load_state_dict` copies each CPU tensor into the TPU parameter in place.
  That goes through [`AtenCopy_`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/copy_from_aten_kernels.cc#L268 "csrc/ops/copy_from/copy_from_aten_kernels.cc#AtenCopy_")
  and the host-to-device path
  ([host-device-transfer](../concepts/host-device-transfer.md)).
- **Bringing results back.** `out_tpu.cpu()` is a materialization point:
  [`CopyTpuToCpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/tpu_to_cpu.cc#L77 "csrc/ops/copy_from/tpu_to_cpu.cc#CopyTpuToCpu")
  forces the deferred graph to compile and run before the comparison
  ([deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)).
- **Matmul precision.** `float32` matmuls default to the 1-pass `DEFAULT` precision read by
  [`GetAndAddPrecisionTo`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/precision_context.cc#L76 "csrc/ops/precision_context.cc#GetAndAddPrecisionTo").
  This is the main reason a `float32` TPU run does not match a `float32` CPU run. Raising it is
  covered on [mxu-precision-modes](mxu-precision-modes.md).
- **Fused graphs.** Under `torch.compile`, fused graphs may keep `float32` intermediates only
  when excess precision is allowed
  ([excess-precision-and-out-dtype](excess-precision-and-out-dtype.md)). That setting is
  off by default in TorchTPU.

> [!warning] The doc says MXU accumulators "may operate in BFloat16" depending on the TPU
> generation. The precision guide in the same doc set
> ([mxu-precision-modes](mxu-precision-modes.md)) says MXU accumulation is `float32`. The code
> does not settle the hardware question; treat `float32` accumulation as the documented default.

## Why it matters / when it applies
- **Same dtype on both sides.** Always compare `bfloat16` TPU with `bfloat16` CPU; `bfloat16`
  against `float32` "will always yield significant mismatches".
- **Read the `assert_close` report.** It gives the mismatched-element percentage and the
  greatest absolute and relative differences with their indices.
  - A *high* mismatch percentage (the doc says above about 70%) with small magnitudes usually
    means systematic precision loss from fusion or rounding, not a logic bug.
  - A few large outliers are worth investigating.
  - Use the greatest differences to choose `atol`/`rtol`. Start from PyTorch's default
    tolerances, which the doc recommends as the baseline.
- **Check gradients too.** Run the same synchronized-weights pattern after `loss.backward()`
  and compare `model.weight.grad`.
- **Expect even a small layer to fail defaults.** The doc's `nn.Linear(10, 10)` example fails
  the default tolerances in `bfloat16` (for example 1 of 50 elements, relative difference 0.165
  against 0.016 allowed). That is expected drift, not a defect.
- **Isolate precision from bugs.** To tell the two apart in a `float32` model, rerun the
  comparison under `torch.tpu.precision(torch.tpu.Precision.HIGHEST)`. If the mismatch
  collapses, it was MXU precision. *(Agent inference; not in the doc.)*

## Connections
- Code concepts: [dtype-and-precision](../concepts/dtype-and-precision.md),
  [host-device-transfer](../concepts/host-device-transfer.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/ops/copy_from`, `csrc/ops`).
- Related doc-concepts: [mxu-precision-modes](mxu-precision-modes.md),
  [excess-precision-and-out-dtype](excess-precision-and-out-dtype.md),
  [tpu-hardware-model](tpu-hardware-model.md),
  [porting-vit-walkthrough](porting-vit-walkthrough.md),
  [debugging-ladder](debugging-ladder.md).

## Source
Extracted from `docs/markdown/numerical_parity.md` (kept in place).
