---
title: 'Area: tokamax/experimental'
type: area
area: 'tokamax/experimental'
description: 'The PyTorch bridge (TorchOp wrappers that run tokamax TPU kernels on torch_tpu tensors) and standalone TPU tuning scripts.'
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:49:08Z}
---
# Area: `tokamax/experimental`

## Purpose
`tokamax/experimental` is the PyTorch side of the library and a set of TPU tuning scripts. It
holds no kernels of its own: [`TorchOp`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/experimental/torch_tpu/ops/torch_op.py#L236 "tokamax/experimental/torch_tpu/ops/torch_op#TorchOp") wraps a
tokamax JAX op so that PyTorch tensors on a torch_tpu device can call its Pallas TPU kernel,
and the two wrapped ops today are batched ragged paged attention and the fused linear softmax
cross-entropy loss. How the wrapper converts tensors, picks a config and wires the backward is
on [torch-tpu-bridge](../concepts/torch-tpu-bridge.md); the kernels themselves are on
[batched-ragged-paged-attention](../concepts/batched-ragged-paged-attention.md) and
[linear-softmax-cross-entropy](../concepts/linear-softmax-cross-entropy.md).

## How the units connect
One unit. `torch_tpu/ops/torch_op.py` and `torch_utils.py` are the shared wrapper; each
`torch_tpu/ops/<op>/` directory holds a reference (`torch_base.py`) and the Pallas-backed
(`torch_pallas_mosaic_tpu.py`) wrapper for one op. `utils/tuning/tpu/` holds standalone
tune-jax sweeps for ragged dot and splash attention
([`main`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/experimental/utils/tuning/tpu/ragged_dot_benchmarking.py#L372 "tokamax/experimental/utils/tuning/tpu/ragged_dot_benchmarking#main")); they call the
kernels directly and do not go through tokamax's own [autotuning](../concepts/autotuning.md).

<!-- area:auto:begin -->
## Units
1 planned unit(s) in `tokamax/experimental`; deep units have their own mechanism page, the rest are sections below.

| Unit | Modules | Symbols | Callers outside | Entry points | Page |
|---|---|---|---|---|---|
| `tokamax/experimental` | 8 | 128 | 0 | (internal) | [torch-tpu-bridge](../concepts/torch-tpu-bridge.md) |

<!-- area:auto:end -->
