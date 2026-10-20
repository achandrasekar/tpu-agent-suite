---
title: "Synchronous debug eager (TORCH_TPU_LAUNCH_BLOCKING)"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md
sources: [docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md, docs/markdown/error_glossary.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Synchronous debug eager (TORCH_TPU_LAUNCH_BLOCKING)

## Definition
In the normal deferred mode an error is reported at the materialization point (for example
`.cpu()`), not on the line that recorded the bad op: the doc's shape-mismatch and dtype-mismatch
demos both raise only when `c.cpu()` runs. Synchronous debug eager, enabled with
`TORCH_TPU_LAUNCH_BLOCKING=1` (the TPU analogue of `CUDA_LAUNCH_BLOCKING`) or approximated by
calling `torch.accelerator.synchronize()` after every op, forces each op to run immediately so
the error lands on the culprit line. It is the doc's tool for OOM / crash attribution and
NaN/Inf hunting, and it is much slower than deferred mode.

## In torch_tpu (grounded)
- The env var ([`kTorchTpuLaunchBlockingEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L183 "csrc/common/env_vars.h#kTorchTpuLaunchBlockingEnvVar")) selects the process's
  default eager mode in [`GetDefaultEagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/eager_mode.cc#L28 "csrc/eager/eager_mode.cc#GetDefaultEagerMode"): launch-blocking wins
  and yields [`kDeferNeverAndLaunchBlocking`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L48 "csrc/common/context_states.h#EagerMode.kDeferNeverAndLaunchBlocking");
  otherwise `TORCH_TPU_DEFER_AND_FUSE` chooses between defer-and-fuse and defer-never. The
  choice is memoized, so the variable must be set before the backend initializes.
- The [`EagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L35 "csrc/common/context_states.h#EagerMode") enum documents the modes: defer-and-fuse
  defers everything that can be deferred (the performance mode), defer-never runs each op
  immediately like CUDA eager, and defer-never-and-launch-blocking additionally waits for one
  op's compilation before dispatching the next and runs extra runtime checks ("the worst
  execution performance").
- The mode can also be changed at runtime: [`SetEagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/eager_mode.cc#L56 "csrc/eager/eager_mode.cc#SetEagerMode") sets the
  global mode, [`GetEagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/eager_mode.cc#L51 "csrc/eager/eager_mode.cc#GetEagerMode") honors a context-manager override,
  and the Python enum exposes
  [`EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/execution_mode/__init__.py#L35 "src/torch_tpu/_internal/execution_mode/__init__#EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING").
- Launch-blocking mode turns on the expensive debug checks by default
  ([`GetEnableDebugChecks`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L811 "csrc/common/error_utils.cc#GetEnableDebugChecks")), unless
  `TORCH_TPU_ENABLE_DEBUG_CHECKS=0` overrides it; see [tpu-debug-checks](tpu-debug-checks.md).

## Why it matters / when it applies
Use it to bisect a failure or a numerical blow-up to one op, then turn it off: every op runs on its own,
with no fusion and with the host waiting on each compilation, so throughput is far worse than in
production modes. The doc places it at the top of its debugging ladder (level 5).

## Connections
- Code concepts: [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)
  (what deferral and materialization mean),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [synchronization](../concepts/synchronization.md),
  [error-handling](../concepts/error-handling.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/common`,
  `src/torch_tpu`).
- Related doc-concepts: [eager-execution-modes](eager-execution-modes.md) (all eager modes,
  where this one is called Debug), [debugging-ladder](debugging-ladder.md),
  [tpu-debug-checks](tpu-debug-checks.md),
  [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md),
  [environment-variables](environment-variables.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md`
(kept in place).
`docs/markdown/error_glossary.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
