---
title: "Extending PyTorch on TPU: autograd.Function, ATen ops and custom ops"
type: doc-concept
provenance: doc
source: docs/markdown/customizing_autograd_via_torch_autograd_function.md
sources: [docs/markdown/customizing_autograd_via_torch_autograd_function.md, docs/markdown/background_on_aten_ops.md, docs/markdown/extending_modules_and_functions_via_composition.md, docs/markdown/quantized_sum.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Extending PyTorch on TPU: autograd.Function, ATen ops and custom ops

## Definition
The loose user-guide pages walk through ways to extend PyTorch, from easiest to most powerful:
composing existing functions and modules, overriding autograd with `torch.autograd.Function`,
then custom ops, ending in Pallas kernels. Composition, `autograd.Function` and the ATen-op
view are covered here; the custom-op steps have their own pages.
- **`torch.autograd.Function`** "enables developers to define a custom backward pass". The toy
  `SigmoidNoVanishing` subclasses it, overrides `forward` (saving the sigmoid result with
  `ctx.save_for_backward`) and `backward` (using `torch.where` with `>` and `<=` to replace the
  local gradient with 0.25 inside a band and 0.01 outside, so it never vanishes). The guide
  warns that in PyTorch 2.0 custom ops "have their own first class support and should not use
  `torch.autograd.Function`".
- **ATen ops as the IR.** Beyond functions and modules there is "a third class of callables in
  PyTorch: ops". Every call, including a custom autograd function, becomes ATen ops ("A Tensor
  Library"; thousands of them, forming an intermediate representation) and c10d ops (for
  collectives). For example `torch.nn.functional.linear` becomes `aten.t` plus `aten.addmm`.
  `torch._logging.set_logs(aot_graphs=True)` under `torch.compile(backend="aot_eager")` prints
  the AOT graph, and with it the exact ATen ops your Python lowered to. In the sigmoid exercise
  the comparisons show up as `aten.gt.Scalar` and `aten.le.Scalar`.

## In torch_tpu (grounded)
Composition and `autograd.Function` need nothing from TorchTPU beyond the ATen kernels they
lower to. ATen is exactly the interface TorchTPU implements:
- Every ATen op on a `tpu` tensor reaches a TorchTPU kernel registered for the PrivateUse1 key,
  written with the `TT_KERNEL` macro ([aten-kernel-authoring](../concepts/aten-kernel-authoring.md)).
  The guide's ops are concrete kernels:
  [`AtenAddmmOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/addmm/addmm_aten_kernels.cc#L294 "csrc/ops/addmm/addmm_aten_kernels.cc#AtenAddmmOut"),
  [`AtenGtScalarOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/binary_aten_kernels.cc#L1436 "csrc/ops/binary_aten_kernels.cc#AtenGtScalarOut"),
  [`AtenLeScalarOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/binary_aten_kernels.cc#L1543 "csrc/ops/binary_aten_kernels.cc#AtenLeScalarOut"),
  [`AtenSigmoidOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/sigmoid/sigmoid_aten_kernels.cc#L229 "csrc/ops/sigmoid/sigmoid_aten_kernels.cc#AtenSigmoidOut")
  and, for the built-in sigmoid's own backward,
  [`AtenSigmoidBackwardGradInput`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/sigmoid/sigmoid_aten_kernels.cc#L245 "csrc/ops/sigmoid/sigmoid_aten_kernels.cc#AtenSigmoidBackwardGradInput").
  Each kernel hands a StableHLO builder to
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp"),
  which records a deferred op ([eager-op-dispatch](../concepts/eager-op-dispatch.md)).
- A custom `backward` written in PyTorch ops is lowered the same way. The autograd engine calls
  it, its ATen ops dispatch to TorchTPU kernels, and in defer-and-fuse mode they fuse with the
  rest of the backward pass.
- c10d ops on TPU go to the `tpu_dist` process group
  ([`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu");
  [tpu-dist-process-group](tpu-dist-process-group.md)).
- The custom-op steps go through
  [`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op"),
  which registers a `torch.library.custom_op`. Its exported StableHLO is registered by
  [`PyRegisterCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/pallas/pallas_py.cc#L49 "csrc/internal/pallas/pallas_py.cc#PyRegisterCustomKernel")
  and inlined by
  [`CallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L608 "csrc/ops/custom_kernels.cc#CallCustomKernel")
  ([jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md)). Adding a first-class ATen kernel
  in C++ is the path in [aten-kernel-authoring](../concepts/aten-kernel-authoring.md).

## Composition: modules hold state, functions compute
The first step needs no new machinery. PyTorch has two kinds of built-in callables:
- **Functions** such as `torch.nn.functional.linear` are stateless; weights arrive as arguments.
- **Modules** such as `nn.Linear` subclass `torch.nn.Module`. They hold state (`nn.Parameter`
  weights, creation, save and load) and delegate the computation to a function. For example,
  `nn.Linear.forward` is just `F.linear(input, self.weight, self.bias)`.

Custom functionality follows the same split. Compose existing functions into a larger function,
and wrap it in an `nn.Module` subclass only when there is state to hold. The guide's toy is a
quantization-aware-training layer:
- `QATLinear` owns a `(64, 64)` `float32` weight (Kaiming-initialized) and its `forward` calls a
  `qat_linear(weight, x)` function.
- That function quantizes and dequantizes the weight, then calls `F.linear` with
  `(weight_qat - weight).detach() + weight`. The forward pass sees the quantized weight and the
  backward pass sees the master weight.

The quantization scheme itself (one-bit signs plus a scale, bit packing, a popcount-based sum)
is described on [one-bit-quantization-example](one-bit-quantization-example.md). On TorchTPU,
composition costs nothing extra: each composed function is ordinary ATen ops, dispatched and
fused as described above. The guide runs the example on CPU with
`torch.compile(backend="aot_eager")`. On `tpu` tensors the same code runs in eager mode, or
under `backend="tpu"`
([`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend")).

The custom-op chapter that follows (`quantized_sum`) gives the order of the remaining steps:
1. Python custom ops through the PyTorch custom-op API. Like ATen ops, they "compose well with
   `torch.compile` to avoid a graph break".
2. C++ HLO kernels, which the guide marks as forthcoming.
3. JAX kernels without Pallas, wrapped with `torch.tpu.pallas.jax_op`; `jax.lax` maps nearly
   1:1 onto StableHLO.
4. Pallas kernels ([jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md)).

## Why it matters / when it applies
- Pick the lowest step that solves the problem. Use composition for new math, and
  `autograd.Function` only when you need a different backward. Use a custom op when the
  operation must be opaque to the compiler or written outside PyTorch, and Pallas when you
  must control memory movement.
- Reading the ATen graph (`aot_graphs`) is the first check when onboarding a model. It shows
  which ops TorchTPU must lower, so you can spot unexpected ops (extra copies, transposes) and
  data-dependent shapes before looking at HLO. The guide runs it on CPU with `aot_eager`; the
  op list is the same on `tpu`.
- The ATen graph is also the top of the lowering stack. From there TorchTPU emits StableHLO and
  XLA takes over ([aten-to-llo-lowering-path](aten-to-llo-lowering-path.md)).

## Connections
- Code concepts: [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [aten-kernel-authoring](../concepts/aten-kernel-authoring.md),
  [device-backend-registration](../concepts/device-backend-registration.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [scan-and-custom-kernels](../concepts/scan-and-custom-kernels.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md).
- Module catalog: [catalog index](../catalog/index.md) (`csrc/ops`, `csrc/eager`,
  `src/torch_tpu/_internal/pallas`, `csrc/internal/pallas`).
- Related doc-concepts: [aten-to-llo-lowering-path](aten-to-llo-lowering-path.md),
  [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md),
  [pallas-kernel-autograd](pallas-kernel-autograd.md),
  [one-bit-quantization-example](one-bit-quantization-example.md).

## Source
Extracted from `docs/markdown/customizing_autograd_via_torch_autograd_function.md` and
`docs/markdown/background_on_aten_ops.md` (kept in place).
The composition section is from
`docs/markdown/extending_modules_and_functions_via_composition.md`, and the custom-op step order
is from `docs/markdown/quantized_sum.md` (kept in place).
