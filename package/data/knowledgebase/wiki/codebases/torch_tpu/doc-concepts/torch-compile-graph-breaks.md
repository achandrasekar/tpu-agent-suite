---
title: "Graph breaks and static shapes under torch.compile on TPU"
type: doc-concept
provenance: doc
source: docs/markdown/advanced_compile.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Graph breaks and static shapes under torch.compile on TPU

## Definition
The `torch.compile` guide names two rules for compiled code on TPU.
1. **Static shapes.** `torch_tpu` "requires static shapes when using `torch.compile`": always pass
   `dynamic=False`. Any input whose dimensions change (a short last batch, variable sequence
   lengths) triggers "a full recompilation every time the size changes". Pad or slice the data
   *outside* the compiled region so the compiled function always sees the same shapes.
2. **No graph breaks.** A graph break happens when TorchDynamo meets Python code, control flow
   or a library call it cannot trace into an FX graph, and execution falls back to the
   interpreter. On TPU this costs twice. XLA cannot fuse across the boundary, so intermediates
   are written to HBM and read back instead of staying on chip. And each fragment is a separate
   XLA executable that the host must schedule and dispatch on its own.

The guide lists three ways to find breaks: `torch._dynamo.explain(fn)(*args)` (break count,
reasons, stack traces), `torch.compile(fn, fullgraph=True)` (raises
`torch._dynamo.exc.Unsupported` at the first break) and `TORCH_LOGS="graph_breaks"` (streams
notices to stderr). Its main example is a plain training step: calling `loss.backward()` and
`optimizer.step()` inside the compiled function breaks the graph.

## In torch_tpu (grounded)
Graph-break detection is stock PyTorch (Dynamo) and has no TorchTPU symbol. What TorchTPU
adds is how each resulting piece is compiled:
- **One backend call per subgraph.** Dynamo hands each FX fragment separately to
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend"),
  which lowers it to one StableHLO module
  ([`TraverseAndCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/compile/compiled_mode.cc#L357 "csrc/internal/compile/compiled_mode.cc#TraverseAndCompile"))
  and one cache entry in
  [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache").
  N breaks mean N+1 executables, each with its own launch and its own HBM inputs and outputs. A
  fragment that computes no tensors (a seam between two breaks) compiles to a no-op artifact
  ([torch-compile-backend](../concepts/torch-compile-backend.md)).
- **`dynamic=False` is the default anyway.** The loader's
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile")
  sets `dynamic=False` for TPU backends when the caller does not pass `dynamic`, and
  `TpuBackend` rejects `dynamic=True`. The guide's flat "no dynamic shapes" rule is stricter
  than the code: bounded dynamism through `torch._dynamo.mark_dynamic` exists (see
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md)), while `dynamic=True` does not.
- **Optimizer breaks.** The guide recommends `capturable=True` for stateful optimizers in
  compiled steps. TorchTPU's forked AdamW and its Dynamo patches exist to remove the breaks that
  lazily created optimizer state causes ([fused-optimizers](../concepts/fused-optimizers.md)).
  To remove the backward and step breaks entirely, the guide uses
  [single-trace-trainer](single-trace-trainer.md).

## Why it matters / when it applies
- Run `torch._dynamo.explain` or `fullgraph=True` once on any function you compile. A break
  inside a hot loop splits fusion and adds host dispatch on every step.
- Treat a recurring recompile like a break: it means some input shape is not fixed. Use
  `drop_last=True`, pad to fixed buckets, or mark a bounded dynamic dimension.
- The first call to a compiled function pays the XLA compile ("1-2 minutes" for the guide's
  model). Warm up before timing; later calls with the same shapes hit the cache
  ([multi-tier-compilation-cache](multi-tier-compilation-cache.md)).

## Connections
- Code concepts: [torch-compile-backend](../concepts/torch-compile-backend.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md),
  [fused-optimizers](../concepts/fused-optimizers.md),
  [synchronization](../concepts/synchronization.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/compile`,
  `csrc/internal/compile`).
- Related doc-concepts: [single-trace-trainer](single-trace-trainer.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md),
  [debugging-ladder](debugging-ladder.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md).

## Source
Extracted from `docs/markdown/advanced_compile.md` (kept in place).
