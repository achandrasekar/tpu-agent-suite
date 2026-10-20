---
title: "Whole-block torch.compile fusion (FLUX milestone 3)"
type: doc-concept
provenance: doc
source: examples/tutorials/FLUX_optimization/README.md
sources: [examples/tutorials/FLUX_optimization/README.md, examples/classical_ml/README.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Whole-block torch.compile fusion (FLUX milestone 3)

## Definition
The FLUX tutorial compiles an entire `DoubleStreamBlock` with `torch.compile(backend="tpu")`.
In eager mode "every PyTorch operation launches an independent device kernel", so intermediate
activations round-trip through HBM and the host dispatches constantly; compiling the block lets
XLA fuse elementwise ops, normalizations and residual adds, and enables buffer donation and
asynchronous pipelining. Compile alone gave 4.57x over the eager baseline (344.50 ms); on top of
zero-copy views it took 578.03 ms to 113.31 ms (5.10x), and with fused RoPE reached 69.16 ms
(22.79x). The tutorial's ordering is the lesson: fix layouts first, then compile, because
compilation cannot remove copies the model asks for. The classical-ML recipe shows the other end:
for a small MLP that is already sync-efficient, compiled is only 1.03x faster than eager at
steady state, and its first window is slower (170 ms versus 69 ms) because it pays compilation.

## In torch_tpu (grounded)
- `torch.compile` without a backend is routed to
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend") by the loader's
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile"), which also sets `dynamic=False` (the backend
  only supports `dynamic=False`; `dynamic=True` raises an error). [`TpuBackend.__call__`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L654 "src/torch_tpu/_internal/compile/_backend#TpuBackend.__call__")
  and [`_compile_graph_module`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L866 "src/torch_tpu/_internal/compile/_backend#TpuBackend._compile_graph_module")
  turn the Dynamo FX graph into one StableHLO program.
- Buffer donation: [`StaticCompiler._compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/compiler.py#L340 "src/torch_tpu/_internal/compile/compiler#StaticCompiler._compile")
  accepts `donated_inputs`, and [`AnnotateBufferDonations`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/op_builder_utils.cc#L1224 "csrc/ops/op_builder_utils.cc#AnnotateBufferDonations") marks
  them on the module so XLA can reuse input buffers for outputs.
- Compiled programs go through [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache"); the first
  call compiles (the slow first window), later calls with the same shapes reuse the executable.

## Why it matters / when it applies
- Compile the largest region that has no graph breaks (a whole block or the whole step), after
  removing layout copies and materialized helper tensors.
- Always benchmark compiled code after warm-up steps
  ([step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md)).
- Small models that are already sync-efficient may gain little; large blocks with many
  elementwise ops gain the most.

## Connections
- Code concepts: [torch-compile-backend](../concepts/torch-compile-backend.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [mlir-optimization-passes](../concepts/mlir-optimization-passes.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md).
- Module catalogs: [module map](../catalog/index.md) (`src/torch_tpu/_internal/compile`,
  `csrc/ops`, `csrc/common`).
- Related doc-concepts: [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [zero-copy-views-vs-layout-copies](zero-copy-views-vs-layout-copies.md),
  [fused-algebraic-rope](fused-algebraic-rope.md),
  [async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md),
  [simple-fsdp](simple-fsdp.md).

## Source
Extracted from `examples/tutorials/FLUX_optimization/README.md` and
`examples/classical_ml/README.md` (kept in place).
