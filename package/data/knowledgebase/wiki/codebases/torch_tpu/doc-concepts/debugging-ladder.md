---
title: "The TorchTPU debugging ladder"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md
sources: [docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md, docs/markdown/error_glossary.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# The TorchTPU debugging ladder

## Definition
The doc orders TorchTPU's diagnostic tools into five levels, each adding cost and detail:

```
Level 1: Python traceback            always on
Level 2: + C++ stack traces          TORCH_SHOW_CPP_STACKTRACES=1
Level 3: + ATen op audit             torch.profiler (CPU activity)
Level 4: + HLO / LLO dumps           XLA_FLAGS and LIBTPU_INIT_ARGS
Level 5: + synchronous debug eager   TORCH_TPU_LAUNCH_BLOCKING=1
```

Alongside it sit the stock PyTorch graph-break tools for `torch.compile`:
`torch._dynamo.explain(model)(input)` for a human-readable list of breaks and
`TORCH_LOGS="+dynamo"` for a detailed trace.

## In torch_tpu (grounded)
- **Level 2** is explained on
  [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md)
  ([`TorchShowCppStacktraces`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L34 "csrc/common/status_builder.cc#TorchShowCppStacktraces")).
- **Level 3**: the doc reads high counts of `aten::copy_` / `aten::_to_copy` in the
  `torch.profiler` op table as CPU round trips ("performance cliffs"). In TorchTPU those ops
  are the transfer kernels [`AtenCopy_`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/copy_from_aten_kernels.cc#L268 "csrc/ops/copy_from/copy_from_aten_kernels.cc#AtenCopy_") and
  [`AtenToCopy`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/to_copy/to_copy_aten_kernels.cc#L40 "csrc/ops/to_copy/to_copy_aten_kernels.cc#AtenToCopy"); a device-to-host copy ends in
  [`CopyTpuToCpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/tpu_to_cpu.cc#L77 "csrc/ops/copy_from/tpu_to_cpu.cc#CopyTpuToCpu"), which must materialize the source first
  ([host-device-transfer](../concepts/host-device-transfer.md)). Some kernels with
  data-dependent output sizes (for example `unique_consecutive`, `repeat_interleave`,
  `ctc_loss`) call it internally, so a round trip can come from an op rather than from user code. Note that an op with no TPU
  kernel does not silently fall back to CPU by default: the boxed fallback
  [`TpuMissingOpFallback`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L1134 "csrc/eager/tpu_aten_kernels.cc#TpuMissingOpFallback") raises `NotImplementedError` unless
  CPU fallback was turned on ([`EnableCpuFallback`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L1823 "csrc/eager/tpu_aten_kernels.cc#EnableCpuFallback"), described
  in the source as internal and for testing). So copy counts point at host reads in user code
  or in such kernels, not at missing kernels.
- **Level 4** is on [xla-compilation-dumps](xla-compilation-dumps.md) and
  [aten-to-llo-lowering-path](aten-to-llo-lowering-path.md).
- **Level 5** is on [launch-blocking-debug-eager](launch-blocking-debug-eager.md); it also turns
  on [tpu-debug-checks](tpu-debug-checks.md).

## Why it matters / when it applies
Climb only as far as needed: levels 1-2 are free on the success path, level 3 adds profiler
overhead, level 4 writes many files per compiled module and changes the compile-options key (so
it forces recompilation), and level 5 makes every op synchronous. For bug reports the doc asks
for level 1 and level 2 output.

## Connections
- Code concepts: [error-handling](../concepts/error-handling.md),
  [host-device-transfer](../concepts/host-device-transfer.md),
  [profiler](../concepts/profiler.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common`, `csrc/eager`, `csrc/ops`).
- Related doc-concepts: [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md),
  [launch-blocking-debug-eager](launch-blocking-debug-eager.md),
  [xla-compilation-dumps](xla-compilation-dumps.md),
  [common-onboarding-pitfalls](common-onboarding-pitfalls.md),
  [torch-profiler-on-tpu](torch-profiler-on-tpu.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md`
(kept in place).
`docs/markdown/error_glossary.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
