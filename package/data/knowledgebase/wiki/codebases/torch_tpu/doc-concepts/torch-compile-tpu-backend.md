---
title: "The torch.compile \"tpu\" backend and bounded dynamism"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md, docs/markdown/advanced_compile.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# The torch.compile "tpu" backend and bounded dynamism

## Definition
TorchTPU "registers the `"tpu"` TorchDynamo backend" so that
`torch.compile(model, backend="tpu", dynamic=False)` lowers whole FX graphs to StableHLO and
compiles them with OpenXLA. The backend also supports bounded dynamism through
`torch._dynamo.mark_dynamic`, which "lets compiled graphs handle varying batch sizes or sequence
lengths up to a specified bound without recompiling". The guide notes a temporary limitation:
callers must pass `dynamic=False`, because bounded dynamism is what supports SymInts and it "is
not enabled by default yet".

## In torch_tpu (grounded)
- **Registration.** Importing the compile package registers an instance of
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend") under the name `"tpu"`.
  New options go through `torch.compile(..., backend="tpu", options={...})`. The constructor
  arguments are legacy.
- **Defaults patched by the loader.** When TorchTPU loads, it wraps `torch.compile` with
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile"). If no backend is given,
  the wrapper installs [`_default_backend_selector`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L123 "src/torch_tpu/_loader#_default_backend_selector"),
  which picks `"tpu"` when any example input is on a TPU and a CPU backend otherwise. For TPU
  backends it also sets `dynamic=False` when the caller did not pass `dynamic`. Requesting
  Inductor with TPU inputs raises, in
  [`_tpu_inductor_wrapper_call`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L130 "src/torch_tpu/_loader#_tpu_inductor_wrapper_call"). The one
  exception is Inductor configured with `tpu_backend="pallas"`.
- **Static vs bounded-dynamic compile.** `TpuBackend.__call__` rejects `dynamic=True` with
  `NotImplementedError`. Dynamism comes in only through dims marked dynamic, which Dynamo passes
  as SymInts. [`make_backend_compiler`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L432 "src/torch_tpu/_internal/compile/_backend#make_backend_compiler")
  checks the example inputs for SymInts with free symbols. If it finds any, it builds a dynamic
  compiler that compiles `num_dynamic_buckets` bucketed shapes, aligned to `bucket_size`, inside
  the declared bound. Otherwise it builds a static compiler. Both are wrapped in a split compiler.
- **Lowering.** The FX graph is traced in the internal `kInternalCompileFxGraph` eager mode and
  turned into one module by
  [`TraverseAndCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/compile/compiled_mode.cc#L357 "csrc/internal/compile/compiled_mode.cc#TraverseAndCompile"). It then goes
  through the same [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache") as eager
  ops.

## Why it matters / when it applies
`torch.compile` is the main route to peak throughput: one XLA program per graph instead of one
per op or per fused eager region. Varying input shapes are the usual source of recompiles. Mark
the varying dimension with `mark_dynamic` and realistic `min`/`max` bounds, so that one bounded
executable (or a few buckets) covers the whole range, and the cache tiers absorb the rest. Keep
`dynamic=False` or leave it unset. With TorchTPU loaded, an unset `dynamic` already becomes
`False` for TPU backends.

The `torch.compile` best-practices guide (`advanced_compile.md`) states the rule more bluntly:
"`torch_tpu` requires static shapes", so always pass `dynamic=False`, and pad or slice varying
inputs outside the compiled region. It does not mention `mark_dynamic`. It adds two practices
covered on their own pages: find and remove graph breaks
([torch-compile-graph-breaks](torch-compile-graph-breaks.md)), and compile the whole training
step as one graph with `SingleTraceTrainer` ([single-trace-trainer](single-trace-trainer.md)).

## Connections
- Code concepts: [torch-compile-backend](../concepts/torch-compile-backend.md),
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md),
  [eager-dynamic-shapes](../concepts/eager-dynamic-shapes.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [device-backend-registration](../concepts/device-backend-registration.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/compile`,
  `csrc/internal/compile`).
- Related doc-concepts: [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md),
  [eager-execution-modes](eager-execution-modes.md),
  [pallas-ops-under-torch-compile](pallas-ops-under-torch-compile.md),
  [torch-compile-graph-breaks](torch-compile-graph-breaks.md),
  [single-trace-trainer](single-trace-trainer.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place).
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
The paragraph on the `torch.compile` guide is from `docs/markdown/advanced_compile.md` (kept in place).
