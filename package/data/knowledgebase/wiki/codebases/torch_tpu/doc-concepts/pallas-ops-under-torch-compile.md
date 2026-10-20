---
title: "Pallas custom ops under torch.compile"
type: doc-concept
provenance: doc
source: docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md
sources: [docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md, docs/markdown/custom_pallas_kernel.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Pallas custom ops under torch.compile

## Definition
The user guide says Pallas kernels "can be seamlessly compiled into larger graphs using
`torch.compile(backend="tpu")`". This works because `torch.tpu.pallas.jax_op` registers the
function as a `torch.library.custom_op` and supplies its `register_fake` implementation itself.
You can therefore call the op inside a compiled function without writing a meta kernel. The
guide's summary describes the result as "full graph fusion with custom kernels".

## In torch_tpu (grounded)
[`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op")
registers [`fake_fn`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1650 "src/torch_tpu/_internal/pallas/pallas#jax_op.dec.fake_fn")
as the fake kernel. `fake_fn` builds JAX placeholders for the arguments
([`JaxCallable._get_jax_placeholders`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1164 "src/torch_tpu/_internal/pallas/pallas#JaxCallable._get_jax_placeholders"))
and lowers the exported JAX function. It turns each output aval into an empty tensor on the
input's device with
[`torch_placeholder`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L220 "src/torch_tpu/_internal/pallas/pallas#torch_placeholder"),
which lets Dynamo and AOTAutograd propagate shapes without running the kernel. The op is
registered with `mutates_args=()`. A source comment explains that marking donated args as mutable
would make Dynamo copy the original value back, which "defeats the purpose of donating".

The `backend="tpu"` path starts at
[`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend").
It hands the graph to AOTAutograd with
[`StaticCompiler`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/compiler.py#L312 "src/torch_tpu/_internal/compile/compiler#StaticCompiler")
(or [`DynamicCompiler`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/dynamic/dynamic_compiler.py#L117 "src/torch_tpu/_internal/compile/dynamic/dynamic_compiler#DynamicCompiler")
when the graph has symbolic sizes). The static compiler traces the FX graph through the same
eager kernels as eager mode, using placeholder tensors. The custom op therefore takes its normal
eager path:
[`CallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L608 "csrc/ops/custom_kernels.cc#CallCustomKernel")
emits a call to the specialized kernel function, and the kernel ends up in the same StableHLO
program as the rest of the graph. There is no separate lowering for compiled mode.

## Why it matters / when it applies
- Pallas ops compose with `torch.compile`. A model that mixes ATen ops and custom kernels still
  compiles to one TPU program, which keeps the cross-op fusion the backend relies on.
- **Static shapes.** The default fake kernel raises a `RuntimeError` on symbolic (`SymInt`)
  dimensions ("Either remove symbolic dimensions or override this default implementation").
  The guide's training example accordingly uses `torch.compile(dynamic=False, backend="tpu")`.
  To trace with dynamic shapes, register your own fake kernel.
- Each distinct input shape still costs one trace and one registration of the kernel. The
  compiled program is then cached like any other compiled graph.

## Connections
- Code concepts: [torch-compile-backend](../concepts/torch-compile-backend.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md),
  [compilation-cache](../concepts/compilation-cache.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/pallas`,
  `src/torch_tpu/_internal/compile`).
- Related doc-concepts: [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md),
  [pallas-kernel-autograd](pallas-kernel-autograd.md).

## Source
Extracted from `docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md`
(kept in place).
`docs/markdown/custom_pallas_kernel.md` is an older, unnumbered export of the same Pallas chapter with identical content (kept in place).
