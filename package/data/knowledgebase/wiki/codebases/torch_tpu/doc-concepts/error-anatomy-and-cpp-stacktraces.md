---
title: "Anatomy of a TorchTPU error and TORCH_SHOW_CPP_STACKTRACES"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md
sources: [docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md, docs/markdown/error_glossary.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Anatomy of a TorchTPU error and TORCH_SHOW_CPP_STACKTRACES

## Definition
By default a TorchTPU failure shows only the Python traceback. Setting
`TORCH_SHOW_CPP_STACKTRACES=1` before importing `torch` adds two more sections to the
`RuntimeError`: a **C++ error trace**, listing where TorchTPU's C++ code first detected the
error and how it propagated (its topmost entry is the deepest point, the check that failed, for
example `ValidateTensorByteSize`), and a **captured traceback**, the full dispatcher path from
Python through ATen into TorchTPU (for `torch.ones(-1)`: `THPVariable_ones` to
`at::native::ones` to `AtenEmptyMemoryFormat`). The doc asks for the Python traceback plus this
C++ output in every bug report.

## In torch_tpu (grounded)
- The flag is read once per process by
  [`TorchShowCppStacktraces`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L34 "csrc/common/status_builder.cc#TorchShowCppStacktraces") (backed by
  [`kTorchShowCppStacktracesEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L88 "csrc/common/env_vars.h#kTorchShowCppStacktracesEnvVar"), a variable shared with PyTorch).
  Because the read is memoized, it must be set before TorchTPU first touches it; setting it
  later in the process has no effect.
- The C++ error trace is built incrementally: each error-propagating macro calls
  [`MaybeAddCppSourceLoc`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L148 "csrc/common/status_builder.cc#MaybeAddCppSourceLoc"), which, only when stack traces are
  enabled ([`CppStackTracesEnabled`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L50 "csrc/common/status_builder.cc#CppStackTracesEnabled")), appends `file:line:
  function()` to a payload on the `absl::Status`. When the status is turned into an exception,
  [`GetMessageWithCppErrorTrace`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L162 "csrc/common/status_builder.cc#GetMessageWithCppErrorTrace") renders it under "C++ error
  trace (starting from the origin)".
- [`MakeErrorMessage`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L1169 "csrc/common/error_utils.cc#MakeErrorMessage") assembles the final message: it prefixes the
  root op name recorded with the error ([`GetRootOpNamePayload`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/status_builder.cc#L142 "csrc/common/status_builder.cc#GetRootOpNamePayload")),
  appends the C++ trace, and decides whether to add PyTorch's own C++ backtrace.
  [`TranslateToC10ErrorAndThrow`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L1223 "csrc/common/error_utils.cc#TranslateToC10ErrorAndThrow") then maps the error code to the
  matching `c10::Error` subclass, so Python sees the same exception type PyTorch CPU would raise.
- The example's innermost check, [`ValidateTensorByteSize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/dtype.cc#L648 "csrc/common/dtype.cc#ValidateTensorByteSize"), is reached
  from the factory kernel [`AtenEmptyMemoryFormat`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/nullary_aten_kernels.cc#L106 "csrc/ops/nullary_aten_kernels.cc#AtenEmptyMemoryFormat").

## Why it matters / when it applies
Without the flag, a failure deep in a kernel or in XLA reads as a bare `RuntimeError` whose
Python line may be far from the cause (errors in deferred ops surface at materialization; see
[launch-blocking-debug-eager](launch-blocking-debug-eager.md)). With it, the top of the C++
trace names the exact validation that failed. It costs nothing on the success path, since the
trace is only collected for failing statuses.

## Connections
- Code concepts: [error-handling](../concepts/error-handling.md) (the `TtError` / `StatusBuilder`
  machinery in depth), [python-context-and-op-names](../concepts/python-context-and-op-names.md)
  (where the root op name comes from), [dtype-and-precision](../concepts/dtype-and-precision.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common`, `csrc/ops`).
- Related doc-concepts: [debugging-ladder](debugging-ladder.md),
  [launch-blocking-debug-eager](launch-blocking-debug-eager.md),
  [tpu-debug-checks](tpu-debug-checks.md), [environment-variables](environment-variables.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md`
(kept in place); the same variable is also listed in
`docs/markdown/6_reference_4_environment_variables.md`.
`docs/markdown/error_glossary.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
