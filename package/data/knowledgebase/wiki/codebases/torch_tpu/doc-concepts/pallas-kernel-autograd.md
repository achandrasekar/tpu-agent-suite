---
title: "Autograd for Pallas custom kernels"
type: doc-concept
provenance: doc
source: docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md
sources: [docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md, docs/markdown/custom_pallas_kernel.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Autograd for Pallas custom kernels

## Definition
A Pallas kernel has no automatic derivative. To use one in a training loop, "you must manually
define the backward pass and register it with the forward op using `register_autograd`". The
user guide's rule is that both the forward and the backward kernel are defined with
`torch.tpu.pallas.jax_op`. That way each is a custom op with its own `register_fake`
implementation. The guide says this is "required because AOTAutograd traces both the forward and
backward graphs".

## In torch_tpu (grounded)
[`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op")
returns a
[`_JaxCustomOpDef`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L918 "src/torch_tpu/_internal/pallas/pallas#_JaxCustomOpDef"),
a subclass of PyTorch's `CustomOpDef`. The standard `register_autograd(backward,
setup_context=...)` and `register_fake` methods are therefore available on it. TorchTPU adds no
autograd machinery of its own here; the recipe is plain `torch.library`. The guide's pattern:
1. Write the backward Pallas kernel with one output ref per gradient (`g * y`, `g * x` for a
   multiply) and wrap it with `jax_op`, for example `"pallas::mul_vectors_backward"`. A
   multi-output `out_shape` list makes the op return a tuple of gradients.
2. Save the inputs in `setup_context` (`ctx.save_for_backward(x, y)`). In the backward function,
   call the backward op on `(grad, *ctx.saved_tensors)`.
3. Call `forward_op.register_autograd(backward_impl, setup_context=setup_context)`, then run the
   loss under `torch.compile(dynamic=False, backend="tpu")` and call `.backward()`.

The guide also calls `register_fake` on the backward op by hand, with
`torch.empty_like(x), torch.empty_like(y)`. This replaces the fake kernel that `jax_op`
registered automatically
([`fake_fn`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1650 "src/torch_tpu/_internal/pallas/pallas#jax_op.dec.fake_fn")).
That automatic fake kernel already infers output shapes by lowering the JAX export, so the manual
override is only needed when you want different behaviour, such as support for symbolic
dimensions.

Under `torch.compile`,
[`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend")
runs AOTAutograd with separate forward and backward compilers. Each compiler traces its graph
through the eager kernels, so the backward Pallas kernel is inlined into the backward program
exactly as the forward kernel is inlined into the forward program
([`CallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L608 "csrc/ops/custom_kernels.cc#CallCustomKernel")).

## Why it matters / when it applies
This applies whenever a hand-written kernel sits on the training path, for example a fused
attention variant or a quantize step that needs gradients. Inference-only kernels can skip it.
Defining the backward kernel as a `jax_op` as well, rather than as an opaque Python function,
keeps both graphs traceable and lets the backward kernel fuse into the compiled backward program.
The default fake kernel rejects symbolic dimensions, which is why the guide compiles with
`dynamic=False`.

## Connections
- Code concepts: [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md) (AOTAutograd forward/backward
  compilers), [fused-optimizers](../concepts/fused-optimizers.md) (the training step that consumes
  the gradients).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/pallas`).
- Related doc-concepts: [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md),
  [pallas-ops-under-torch-compile](pallas-ops-under-torch-compile.md).

## Source
Extracted from `docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md`
(kept in place).
`docs/markdown/custom_pallas_kernel.md` is an older, unnumbered export of the same Pallas chapter with identical content (kept in place).
