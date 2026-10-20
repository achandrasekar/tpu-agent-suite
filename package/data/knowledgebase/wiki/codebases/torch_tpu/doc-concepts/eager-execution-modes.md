---
title: "Eager execution modes: Strict, Debug and DeferAndFuse"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md, docs/markdown/eager_mode.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Eager execution modes: Strict, Debug and DeferAndFuse

## Definition
TorchTPU runs an unmodified PyTorch model in eager mode, and the user guide names three eager
execution modes. **Strict** (the default) "dispatches one ATen operator at a time asynchronously
to the TPU, matching standard PyTorch eager behavior on GPUs". **Debug**
(`TORCH_TPU_LAUNCH_BLOCKING=1`) "runs each operator synchronously with the CPU and enables bounds
checking". **DeferAndFuse** (`TORCH_TPU_DEFER_AND_FUSE=1`) "defers and fuses sequences of ATen
operators up to a materialization point (such as `.cpu()` or `.item()`), reducing High Bandwidth
Memory (HBM) traffic and accelerating eager execution".

## In torch_tpu (grounded)
The three guide names map onto the C++ enum
[`EagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L35 "csrc/common/context_states.h#EagerMode"): Strict is `kDeferNever`, Debug is
`kDeferNeverAndLaunchBlocking`, DeferAndFuse is `kDeferAndFuse`. The enum also has two internal
modes, `kInternalDeferAll` (every op must defer, so nothing runs until an explicit sync) and
`kInternalCompileFxGraph` (no on-device execution, used while tracing for `torch.compile` and
`torch.export`).

The process-wide default is chosen once, from the environment, when the mode is first read:
[`kTorchTpuLaunchBlockingEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L183 "csrc/common/env_vars.h#kTorchTpuLaunchBlockingEnvVar") wins and
selects Debug. Otherwise [`kTorchTpuDeferAndFuseEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L99 "csrc/common/env_vars.h#kTorchTpuDeferAndFuseEnvVar")
selects DeferAndFuse. With neither set the mode is `kDeferNever`, which is Strict.
[`GetEagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/eager_mode.cc#L51 "csrc/eager/eager_mode.cc#GetEagerMode") returns the innermost scoped override or
falls back to that global, and [`SetEagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/eager_mode.cc#L56 "csrc/eager/eager_mode.cc#SetEagerMode") replaces
the global. The "bounds checking" the guide mentions is
[`GetEnableDebugChecks`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/error_utils.cc#L811 "csrc/common/error_utils.cc#GetEnableDebugChecks"): it honours
[`kTorchTpuEnableDebugChecksEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L110 "csrc/common/env_vars.h#kTorchTpuEnableDebugChecksEnvVar")
when set, and otherwise turns the expensive checks on exactly when the mode is
`kDeferNeverAndLaunchBlocking`.

From Python the modes are exposed as the experimental enum
[`EagerMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/execution_mode/__init__.py#L30 "src/torch_tpu/_internal/execution_mode/__init__#EagerMode") (`DEFER_AND_FUSE`,
`DEFER_NEVER`, `DEFER_NEVER_AND_LAUNCH_BLOCKING`). The module-level property
[`eager_mode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/execution_mode/__init__.py#L92 "src/torch_tpu/_internal/execution_mode/__init__#_ExecutionModeModule.eager_mode")
gets and sets the global mode. The context manager
[`set_eager_mode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/execution_mode/__init__.py#L130 "src/torch_tpu/_internal/execution_mode/__init__#set_eager_mode") pushes a
scoped mode and pops it on exit. It is forbidden inside `torch.compile`, because "a compiled graph
cannot select how to dispatch eager ops".

In the deferred modes an op is recorded rather than run, and the work is flushed when a value
is needed. [`Materialize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/materialize.cc#L812 "csrc/eager/materialize.cc#Materialize") is the flush point, and
[deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)
covers it in detail.

## Why it matters / when it applies
- Start in Strict when onboarding a model. Behaviour matches CUDA eager, so a failure is
  reported against the op that caused it.
- Switch to Debug when an error or a numerical problem needs to be pinned to one op. Each op's
  compilation finishes before the next op is dispatched, execution is synchronous with the host,
  and the debug checks run, so this mode is the slowest. The source comment calls it "only for debugging".
- Use DeferAndFuse when eager throughput matters. Fused sequences save HBM round trips, but
  every materialization point (`.cpu()`, `.item()`, data-dependent Python control flow) ends a
  fused region and may trigger a compile. Avoid unnecessary host syncs in the training loop.
- The GKE example manifests export `TPU_LAUNCH_BLOCKING=1`. The variable TorchTPU actually reads
  is `TORCH_TPU_LAUNCH_BLOCKING`, so those manifests run in the default mode unless you rename
  the variable.

## Fused Eager in practice: materialization triggers and the compile trade-off
The eager-mode guide uses other names for the same three modes: **Strict Eager** (default; ops
"dispatched immediately and executed asynchronously on the device stream"), **Fused Eager**
(`TORCH_TPU_DEFER_AND_FUSE=1`) and **Debug Eager** (`TORCH_TPU_LAUNCH_BLOCKING=1`, "equivalent
to `CUDA_LAUNCH_BLOCKING=1`"). It stresses that the variables must be set in the shell, or in
`os.environ` before the first TPU tensor is created, because the runtime reads them on first
device touch. It describes Fused Eager in three steps. Element-wise and arithmetic ops are
recorded in a deferred buffer instead of being launched. A materialization trigger makes XLA
compile everything recorded so far into one fused cluster. Intermediates then stay in VMEM or
vector registers instead of making HBM round trips. Its timing demo shows this split: queueing
`x + 1.0; y * 2.0 - 0.5` takes microseconds, and the cost appears at `.cpu()`.

The guide lists these triggers:

```
Guide's trigger                               Reason recorded by Materialize
--------------------------------------------- ------------------------------------
.cpu(), .tolist(), .numpy()                   .cpu()          (kCpuTransfer)
.item()                                       item()          (kScalarConversion)
torch.accelerator.synchronize()               synchronize()   (kExplicitSync)
print(t) / repr(t)                            a host copy, so .cpu()
data-dependent Python control flow            a host read, so item() / .cpu()
"buffer thresholds" (internal limits)         see below
```

In the code every flush goes through
[`Materialize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/materialize.cc#L812 "csrc/eager/materialize.cc#Materialize") with a
[`MaterializationReason`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/structured_log_buffer.h#L36 "csrc/eager/structured_log_buffer.h#MaterializationReason").
That enum has no memory- or node-count threshold. The automatic flushes it names are
`autograd_post_backward` (end of a backward pass), `repeated_op_heuristic` (a repeating op
sequence was detected; the policy is chosen by
[`TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L122 "csrc/common/env_vars.h#kTorchTpuInternalDetectRepeatedOpsEnvVar"),
which defaults to `safe`, a split graph, with `aggressive` giving a full graph) and
`distributed_op`. Read the guide's "buffer thresholds" as these heuristics, not a size limit
you can tune.

**Fused Eager vs `torch.compile(backend="tpu")`.** Fused Eager fuses neighbouring ops on the fly
and keeps the eager workflow. `torch.compile` optimizes the whole graph ahead of time, across
forward and backward. The guide benchmarks ten repetitions of `(y + 1.0) * 2.0 - 0.5` on a
4096x4096 bf16 tensor under both, with warm-up, and reports Fused Eager as a fraction "of
compiled peak". Its recommendations:

```
Workload                         Configuration
-------------------------------- ---------------------------------------------
interactive prototyping          Strict Eager (variables unset)
debugging crashes / NaNs         TORCH_TPU_LAUNCH_BLOCKING=1
high-performance eager training  TORCH_TPU_DEFER_AND_FUSE=1
production serving, peak speed   torch.compile(model, backend="tpu")
```

See [torch-compile-graph-breaks](torch-compile-graph-breaks.md) for keeping a compiled region
whole, and [cuda-to-tpu-api-mapping](cuda-to-tpu-api-mapping.md) for `torch.accelerator.synchronize`.

## Connections
- Code concepts: [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [synchronization](../concepts/synchronization.md),
  [error-handling](../concepts/error-handling.md),
  [device-backend-registration](../concepts/device-backend-registration.md).
- Module catalog: [catalog index](../catalog/index.md) (`csrc/eager`, `csrc/common`,
  `src/torch_tpu/_internal/execution_mode`).
- Related doc-concepts: [launch-blocking-debug-eager](launch-blocking-debug-eager.md) (the Debug
  mode in depth), [tpu-debug-checks](tpu-debug-checks.md),
  [debugging-ladder](debugging-ladder.md),
  [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md),
  [gke-xpk-workloads](gke-xpk-workloads.md),
  [torch-compile-graph-breaks](torch-compile-graph-breaks.md),
  [cuda-to-tpu-api-mapping](cuda-to-tpu-api-mapping.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place).
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
The "Fused Eager in practice" section is from `docs/markdown/eager_mode.md` (kept in place).
