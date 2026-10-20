---
title: "Expensive debug checks (TORCH_TPU_ENABLE_DEBUG_CHECKS)"
type: doc-concept
provenance: doc
source: docs/markdown/6_reference_4_environment_variables.md
sources: [docs/markdown/6_reference_4_environment_variables.md, docs/markdown/environment_variables.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Expensive debug checks (TORCH_TPU_ENABLE_DEBUG_CHECKS)

## Definition
`TORCH_TPU_ENABLE_DEBUG_CHECKS=1` enables internal invariant and bounds validation that TorchTPU
skips by default for speed. The doc's example is `torch.take` with an out-of-range index:
without the variable the TPU does not check, and the op may silently return garbage; with it, the
op raises `take(): expected indices to be in range`. The checks need extra host/device
synchronization, so they slow execution significantly.

## In torch_tpu (grounded)
- [`GetEnableDebugChecks`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L811 "csrc/common/error_utils.cc#GetEnableDebugChecks") is the single switch kernels consult. If
  the variable ([`kTorchTpuEnableDebugChecksEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L110 "csrc/common/env_vars.h#kTorchTpuEnableDebugChecksEnvVar")) is set, its value
  wins; if unset, checks are on exactly when the eager mode is
  [`kDeferNeverAndLaunchBlocking`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L48 "csrc/common/context_states.h#EagerMode.kDeferNeverAndLaunchBlocking"), so
  `TORCH_TPU_LAUNCH_BLOCKING=1` implies debug checks unless explicitly disabled with `0`.
- The `take` path shows the pattern: [`ValidateTakeInputs`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/take/take_aten_kernels.cc#L119 "csrc/ops/take/take_aten_kernels.cc#ValidateTakeInputs")
  runs [`ValidateIndices`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/take/take_aten_kernels.cc#L100 "csrc/ops/take/take_aten_kernels.cc#ValidateIndices") only when debug checks are on,
  and otherwise emits a one-time correctness warning that names the variable. `histc` uses the
  same warning. The kernel entry is [`AtenTake`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/take/take_aten_kernels.cc#L142 "csrc/ops/take/take_aten_kernels.cc#AtenTake").
- The value is read once ([`GetBooleanEnvOnce`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L419 "csrc/common/env_vars.h#GetBooleanEnvOnce")), so set it before the
  backend initializes.

## Why it matters / when it applies
PyTorch on CPU and CUDA validate indices on every call; TorchTPU does not, because a
data-dependent check needs the index values on the host and breaks deferred execution. Code that
relied on PyTorch raising for bad indices can therefore produce wrong results on TPU without an
error. Turn the checks on while onboarding or when results look corrupted, and off for
performance runs.

## Connections
- Code concepts: [error-handling](../concepts/error-handling.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [aten-kernel-authoring](../concepts/aten-kernel-authoring.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common`, `csrc/ops`).
- Related doc-concepts: [launch-blocking-debug-eager](launch-blocking-debug-eager.md),
  [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md),
  [environment-variables](environment-variables.md).

## Source
Extracted from `docs/markdown/6_reference_4_environment_variables.md` (kept in place).
`docs/markdown/environment_variables.md` is an older, unnumbered export of the environment-variables reference with identical content (kept in place).
