---
title: "Common onboarding failures: shape/dtype errors, inference_mode, DataLoader workers"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md
sources: [docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md, docs/markdown/error_glossary.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Common onboarding failures: shape/dtype errors, inference_mode, DataLoader workers

## Definition
The doc's error glossary walks through four failures that code coming from CPU or GPU hits on
TPU:

- **Shape mismatch** (`a @ b` with inner dimensions 8 and 5) fails while lowering the op to
  StableHLO, and is reported when the result is materialized (`.cpu()`), not on the `@` line.
- **dtype mismatch** (`torch.mm` of `float32` and `int64`) likewise raises at materialization;
  the fix is an explicit cast.
- **`torch.compile` under `@torch.inference_mode()`** cannot capture graphs, because
  inference-mode tensors carry no version counters for AOTAutograd/Dynamo; use
  `@torch.no_grad()` instead. The doc runs the failing case in a subprocess because it can leave
  the dispatcher in a bad state.
- **`DataLoader(num_workers > 0)`** uses `fork` on Linux. Forking after the PjRt client is
  initialized duplicates multithreaded runtime state and the libtpu device lock, and workers
  hang before step 0 without an exception. Use `num_workers=0` or an asynchronous
  multi-worker loader such as Grain.

## In torch_tpu (grounded)
- The two mismatch cases are reported late because every op is recorded by
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp") and only built and compiled at materialization
  ([deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [stablehlo-op-builder](../concepts/stablehlo-op-builder.md)). To make them raise on the
  culprit line, use [launch-blocking-debug-eager](launch-blocking-debug-eager.md); to see which
  C++ check failed, use [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md).
  The exception type is chosen by [`TranslateToC10ErrorAndThrow`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L1223 "csrc/common/error_utils.cc#TranslateToC10ErrorAndThrow")
  to match what PyTorch on CPU raises.
- The `inference_mode` restriction comes from PyTorch's graph capture, not from TorchTPU code;
  the `tpu` backend that receives the captured graph is described on
  [torch-compile-backend](../concepts/torch-compile-backend.md).
- No code in the package guards against forking: there is no at-fork handler, so the
  DataLoader rule is a usage constraint. The process-level runtime that must not be forked is
  the PjRt client ([pjrt-client](../concepts/pjrt-client.md)).

## Why it matters / when it applies
These four cover most "it worked on GPU" first-run failures. The late-reported errors are the
most confusing: the Python line in the traceback is the materialization point, so read the
op name prefix and C++ trace rather than trusting the Python line.

## Connections
- Code concepts: [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [error-handling](../concepts/error-handling.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [pjrt-client](../concepts/pjrt-client.md),
  [dtype-and-precision](../concepts/dtype-and-precision.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/common`, `csrc/pjrt`).
- Related doc-concepts: [debugging-ladder](debugging-ladder.md),
  [launch-blocking-debug-eager](launch-blocking-debug-eager.md),
  [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md`
(kept in place).
`docs/markdown/error_glossary.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
