---
title: "TorchTPU architecture: from the PyTorch dispatcher to a TPU executable"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TorchTPU architecture: from the PyTorch dispatcher to a TPU executable

## Definition
The guide describes TorchTPU as "an open-source, native PyTorch backend for Google Cloud TPUs"
with three design principles. **Native PyTorch citizenship** means no JAX runtime dependency,
standard tensor view and in-place mutation semantics, PyTorch's one-process-per-device model,
and `torch.compile` through TorchDynamo. **Out-of-the-box usability** means a GPU workload is
onboarded by changing `device="cuda"` to `device="tpu"` and `backend="nccl"` to
`backend="tpu_dist"`. **Production-scale performance** comes from deferred-fusion eager mode, a
Dynamo-to-XLA compiler path, bounded dynamism and custom kernel APIs.

The guide gives the lowering pipeline in four steps:
1. PyTorch's dispatcher intercepts tensor and collective ops on the `"tpu"` device.
2. TorchTPU turns stride views and in-place writes into functional ops, through *view
   decomposition* and *view inversion*.
3. ATen and `c10d` ops are lowered to StableHLO, or to Mosaic for custom Pallas kernels.
4. A C++ PjRt client compiles the module with XLA into a TPU executable, caches it, and
   dispatches it for asynchronous execution.

## In torch_tpu (grounded)
- **Dispatcher to kernel (step 1).** ATen kernels registered for the TPU dispatch key funnel
  into [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp"), which attaches a StableHLO builder
  closure to the op's output buffers. Under the eager modes the op is then run immediately or
  deferred (see [eager-execution-modes](eager-execution-modes.md)). Collectives enter through the
  `c10d` backend [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu").
- **Views and in-place writes (step 2).** A strided view is rewritten as a sequence of
  functional view primitives by
  [`DecomposeIntoViewSequence`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view_decomposition/decomposition.cc#L1244 "csrc/ops/view_decomposition/decomposition.cc#DecomposeIntoViewSequence").
  A write through a view is pushed back into its base by inverting that sequence, in
  [`InvertViewSequence`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view_decomposition/inversion.cc#L470 "csrc/ops/view_decomposition/inversion.cc#InvertViewSequence").
- **StableHLO emission (step 3).** For eager execution, a traversal of the deferred-op graph
  emits one MLIR module per fused region, in
  [`Traversal.BuildMlirModule`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/traversal.cc#L583 "csrc/eager/traversal.cc#Traversal.BuildMlirModule"). For
  `torch.compile`, [`TraverseAndCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/compile/compiled_mode.cc#L357 "csrc/internal/compile/compiled_mode.cc#TraverseAndCompile")
  builds the module for a whole FX graph. Pallas kernels are registered as PyTorch custom ops with
  the experimental [`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op") (exposed as
  `torch.tpu.pallas.jax_op`) and lower to Mosaic rather than to StableHLO.
- **Compile, cache and run (step 4).**
  [`Compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L141 "csrc/common/compilation.cc#Compile") turns the module into a PjRt executable.
  [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache") stores executables across
  three tiers (see [multi-tier-compilation-cache](multi-tier-compilation-cache.md)). The process
  holds a single PjRt client, returned by [`GetPjRtClient`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_client.cc#L32 "csrc/pjrt/pjrt_client.cc#GetPjRtClient").

The guide also lists custom-kernel options beyond Pallas: the experimental Helion tile DSL and
kernels imported from Tokamax. These are named in the doc but have no single catalog entry point.

## Why it matters / when it applies
Each step is a place where performance or correctness can go wrong. Views and in-place updates
become copies unless view decomposition finds a cheap functional form. Every distinct StableHLO
module costs an XLA compile unless the cache hits. Asynchronous dispatch hides latency only until
something forces a host sync. When code is slow on TPU, ask which step pays the cost: dispatch
overhead (consider DeferAndFuse or `torch.compile`), recompilation (shapes, cache tiers), or
kernel quality (inspect the HLO, or write a Pallas kernel).

## Connections
- Code concepts: [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [view-decomposition](../concepts/view-decomposition.md),
  [stablehlo-op-builder](../concepts/stablehlo-op-builder.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [pjrt-client](../concepts/pjrt-client.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md).
- Module catalog: [catalog index](../catalog/index.md) (`csrc/eager`,
  `csrc/ops/view_decomposition`, `csrc/common`, `csrc/pjrt`, `csrc/internal/compile`).
- Related doc-concepts: [eager-execution-modes](eager-execution-modes.md),
  [tpu-hardware-model](tpu-hardware-model.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [tpu-device-onboarding](tpu-device-onboarding.md),
  [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place).
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
