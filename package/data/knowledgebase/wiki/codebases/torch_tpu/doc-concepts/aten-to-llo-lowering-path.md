---
title: "From ATen to HLO and LLO: the TPU lowering path and its dumps"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md
sources: [docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md, docs/markdown/background_on_torch_compile_on_tpus_via_xla.md, docs/markdown/background_on_torch_compile.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# From ATen to HLO and LLO: the TPU lowering path and its dumps

## Definition
TorchTPU compiles PyTorch with XLA rather than Inductor. The doc gives the path "from your
Python code to assembly on a TPU": PyTorch's dispatcher lowers Python to ATen and c10d ops
(unchanged); TorchTPU lowers those to StableHLO, "a dialect of MLIR that is device agnostic";
XLA lowers StableHLO to HLO, optimizes it over several passes, lowers HLO to device-specific
LLO (low-level ops) and optimizes that for the target TPU. Using
`torch.compile(backend="tpu")` on `relu(add(x, y))`, it shows the three artifacts to read: the
`*before_optimizations.txt` HLO module, where `add` and `maximum` (the ReLU) are separate
instructions; the `*after_codegen.txt` module, where XLA has fused both into one
`add_maximum_fusion` with `kind=kLoop` and `calls=%fused_computation` (a single
memory-efficient loop); and the LLO file `*add_maximum_fusion-01-original.txt`, where the fused
body is a `vadd.f32` and a `vmax.f32` between DMA transfers from HBM into VMEM and back.

## In torch_tpu (grounded)
The ATen-to-StableHLO step is TorchTPU's own; everything after it is XLA inside libtpu.
- In eager mode each kernel attaches a StableHLO builder closure to a deferred op through
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp"); at materialization the traversal emits one module
  for the recorded graph ([`Compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/traversal.cc#L1065 "csrc/eager/traversal.cc#Traversal.Compile")). For a
  `torch.compile` graph, [`TraverseAndCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/compile/compiled_mode.cc#L357 "csrc/internal/compile/compiled_mode.cc#TraverseAndCompile") does the
  same for the whole FX graph. See [stablehlo-op-builder](../concepts/stablehlo-op-builder.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md) and
  [torch-compile-backend](../concepts/torch-compile-backend.md).
- The `tt_jit_...` module names in the dump file names (the doc's
  `module_XXXX.tt_jit_pointwise_fusion_xla_...`) and the `op_name="add/add"` metadata come from
  the Python context captured at dispatch; see [xla-compilation-dumps](xla-compilation-dumps.md)
  and [python-context-and-op-names](../concepts/python-context-and-op-names.md).
- Handing the module to XLA is [`Compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L141 "csrc/common/compilation.cc#Compile"), with compiler options
  assembled by [`MakeCompilationSpecs`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L487 "csrc/common/compilation.cc#MakeCompilationSpecs")
  ([compilation-pipeline](../concepts/compilation-pipeline.md)); the resulting executable is
  cached across tiers by [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache")
  ([compilation-cache](../concepts/compilation-cache.md)). The HLO and LLO optimization passes
  and the fusion decision are XLA's and have no symbols in this package.
- One step the doc's list leaves out: before XLA sees a module, TorchTPU runs its own small
  MLIR pass pipeline over the StableHLO (CSE, canonicalization, a norm-weight rewrite), driven
  by [`RunTorchTpuOptimizationPasses`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/pipeline.cc#L61 "csrc/mlir/pipeline.cc#RunTorchTpuOptimizationPasses"); see
  [mlir-optimization-passes](../concepts/mlir-optimization-passes.md).

## Why it matters / when it applies
"The heart of creating highly optimized custom TPU kernels is gainful control of this final
outputted LLO", whether the kernel is written in StableHLO, JAX ops or Pallas. Reading the dumps
is how you check that XLA fused what you expected (a `kLoop` fusion rather than separate
kernels with HBM round trips between them) before deciding to write a custom kernel
([pallas-custom-kernels](../concepts/pallas-custom-kernels.md)).

## Why fusion is the thing to look for (background on torch.compile)
The companion page on `torch.compile` explains what the dumps are checking. Without fusion,
`relu(add(x, y))` reads and writes HBM twice in series: the add loads, computes and writes back,
then the ReLU reads that result back, computes and writes again. Fusion loads the data once,
applies both ops and writes once. On CUDA, Inductor does this by emitting one Triton kernel
(`triton_poi_fused_add_relu_0`) whose loads into registers, add, max and single store are
visible in the generated code (`TORCH_LOGS="output_code"`). The page notes that Inductor often
leaves matmuls unfused because cuBLAS is already well tuned. TorchTPU does not use Inductor.
The same fusion appears as XLA's `add_maximum_fusion` (`kind=kLoop`) in the HLO dump and as a
`vadd.f32`/`vmax.f32` pair between one HBM-to-VMEM DMA and one write back in the LLO dump.
Within one TorchTPU program, fusion is XLA's decision. TorchTPU's part is to put as many ops as
possible into one program: a whole FX graph under `torch.compile`, or a deferred region under
defer-and-fuse ([eager-execution-modes](eager-execution-modes.md),
[torch-compile-graph-breaks](torch-compile-graph-breaks.md)).

## Connections
- Code concepts: [stablehlo-op-builder](../concepts/stablehlo-op-builder.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [python-context-and-op-names](../concepts/python-context-and-op-names.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [mlir-optimization-passes](../concepts/mlir-optimization-passes.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/internal`,
  `csrc/common`).
- Related doc-concepts: [torchtpu-lowering-stack](torchtpu-lowering-stack.md) (the same
  pipeline from the architecture overview), [xla-compilation-dumps](xla-compilation-dumps.md),
  [extending-pytorch-on-tpu](extending-pytorch-on-tpu.md) (reading the ATen graph at the top),
  [torch-compile-graph-breaks](torch-compile-graph-breaks.md),
  [reading-tpu-profiles](reading-tpu-profiles.md),
  [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md`
(kept in place). `docs/markdown/background_on_torch_compile_on_tpus_via_xla.md` is an identical
copy of that chapter. The fusion background section is from
`docs/markdown/background_on_torch_compile.md`.
