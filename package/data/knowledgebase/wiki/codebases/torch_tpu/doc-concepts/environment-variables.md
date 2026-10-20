---
title: "TorchTPU environment variables, by subsystem"
type: doc-concept
provenance: doc
source: docs/markdown/6_reference_4_environment_variables.md
sources: [docs/markdown/6_reference_4_environment_variables.md, docs/markdown/environment_variables.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TorchTPU environment variables, by subsystem

## Definition
TorchTPU is configured mostly through environment variables that reach three consumers: the
XLA compiler (`XLA_FLAGS`), the TPU driver libtpu (`LIBTPU_INIT_ARGS`), and TorchTPU itself
(the `TORCH_TPU_*` family plus a few shared PyTorch and launcher variables). The reference's
central rule is that they are process-global and **locked when the backend initializes**
(usually at `import torch` or the first `torch.device("tpu")`): to try a different value,
restart the process and set it before importing. This page groups the variables by the
subsystem they control and points to the page that explains each mechanism.

## In torch_tpu (grounded)
**Why they lock.** Every variable TorchTPU uses is declared once in `csrc/common/env_vars.h`
and must be registered in [`kEnvVarToStage`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L254 "csrc/common/env_vars.h#kEnvVarToStage"), which records its
stability stage. Reads go through [`GetEnvOnce`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L374 "csrc/common/env_vars.h#GetEnvOnce") and
[`GetBooleanEnvOnce`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L419 "csrc/common/env_vars.h#GetBooleanEnvOnce"), which memoize the first value per variable for
the life of the process and warn once when an experimental or deprecated variable is set. A
later `os.environ[...] = ...` is therefore invisible once the value has been read. Python code
reads the same registry through [`PyGetBooleanEnvOnce`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/env.cc#L40 "csrc/internal/env.cc#PyGetBooleanEnvOnce") and its siblings.

**Debug and safety** (set before `import torch`):

| Variable | Effect | Details |
|---|---|---|
| `TORCH_SHOW_CPP_STACKTRACES` ([`kTorchShowCppStacktracesEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L88 "csrc/common/env_vars.h#kTorchShowCppStacktracesEnvVar")) | Adds the C++ error trace and PyTorch's C++ backtrace to errors | [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md) |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` ([`kTorchTpuEnableDebugChecksEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L110 "csrc/common/env_vars.h#kTorchTpuEnableDebugChecksEnvVar")) | Expensive validation such as index bounds | [tpu-debug-checks](tpu-debug-checks.md) |
| `TORCH_TPU_LAUNCH_BLOCKING` ([`kTorchTpuLaunchBlockingEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L183 "csrc/common/env_vars.h#kTorchTpuLaunchBlockingEnvVar")) | Synchronous debug eager; also implies debug checks | [launch-blocking-debug-eager](launch-blocking-debug-eager.md) |

`TORCH_TPU_LAUNCH_BLOCKING` is not in the reference's tables; it comes from the error-information
guide. Its sibling `TORCH_TPU_DEFER_AND_FUSE` ([`kTorchTpuDeferAndFuseEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L99 "csrc/common/env_vars.h#kTorchTpuDeferAndFuseEnvVar"))
selects the defer-and-fuse eager mode; see [eager-execution-modes](eager-execution-modes.md).

**Compilation** (read when the first module is compiled):

| Variable | Effect | Details |
|---|---|---|
| `XLA_FLAGS` ([`kXlaFlagsEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L250 "csrc/common/env_vars.h#kXlaFlagsEnvVar")) | XLA compiler flags, e.g. HLO dumps | [xla-compilation-dumps](xla-compilation-dumps.md) |
| `LIBTPU_INIT_ARGS` ([`kLibtpuInitArgsEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L53 "csrc/common/env_vars.h#kLibtpuInitArgsEnvVar")) | libtpu flags, e.g. LLO dumps; extended by TorchTPU for distributed runs | [xla-compilation-dumps](xla-compilation-dumps.md) |

Both strings are fingerprinted into the compile-options cache key
([`MakeCompileOptionsKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L441 "csrc/common/compilation.cc#MakeCompileOptionsKey")), so changing them invalidates cached
executables ([compilation-cache-key](../concepts/compilation-cache-key.md)).

**Compilation caching** (set before `torch.device("tpu")`):

| Variable | Effect | Details |
|---|---|---|
| `TORCH_TPU_TIER2_COMPILATION_CACHE` ([`kTorchTpuTier2CompilationCacheEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L202 "csrc/common/env_vars.h#kTorchTpuTier2CompilationCacheEnvVar")) | Name of the host-local cache under `/dev/shm/torch_tpu_cache`; `disabled` turns it off; default on only when world size > 1 | [multi-tier-compilation-cache](multi-tier-compilation-cache.md) |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` ([`kTorchTpuTier3CompilationCacheRootEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L206 "csrc/common/env_vars.h#kTorchTpuTier3CompilationCacheRootEnvVar")) | Root of the shared persistent cache (e.g. a `gs://` path); off when unset | [multi-tier-compilation-cache](multi-tier-compilation-cache.md) |

**Hardware, topology and launch** (normally set by a launcher or the Cloud environment):

| Variable | Effect |
|---|---|
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` ([`kTorchTpuSlicebuilderAddressesEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L188 "csrc/common/env_vars.h#kTorchTpuSlicebuilderAddressesEnvVar")) | One `host:port` per rank for building the ICI network, identical on all ranks |
| `TORCH_TPU_TOPOLOGY` ([`kTorchTpuTopologyEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L208 "csrc/common/env_vars.h#kTorchTpuTopologyEnvVar")) | Physical mesh of the slice, e.g. `2x2x1` or `2,2,1` |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | Standard PyTorch launcher variables ([`kWorldSizeEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L248 "csrc/common/env_vars.h#kWorldSizeEnvVar") etc.) |

TorchTPU's helper [`set_tpu_launch_env`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L328 "src/torch_tpu/_internal/distributed/distributed#set_tpu_launch_env")
fills the TorchTPU-specific ones in the parent process before any launcher spawns workers,
never overwriting a value already set; [`_parse_topology`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L111 "src/torch_tpu/_internal/distributed/distributed#_parse_topology")
accepts `x` or `,` separators. See
[distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md).

**Profiling.** `TORCH_TPU_PROFILER_OUTPUT_DIR`
([`kTorchTpuProfilerOutputDirEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L186 "csrc/common/env_vars.h#kTorchTpuProfilerOutputDirEnvVar"), experimental) overrides where
XPlane traces are written; see [tpu-profiler-config](tpu-profiler-config.md).

## Why it matters / when it applies
Most "my setting had no effect" reports come from setting a variable after the backend read it,
for example inside a notebook cell after `import torch`, or in a worker after the launcher has
already initialized the runtime. Set variables in the launching shell or at the very top of the
entry script. `env_vars.h` registers many more variables than the reference lists; the
`TORCH_TPU_INTERNAL_*` ones are internal tuning knobs and may change without notice.

## Connections
- Code concepts: [error-handling](../concepts/error-handling.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [compilation-cache-key](../concepts/compilation-cache-key.md),
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md),
  [pjrt-client](../concepts/pjrt-client.md),
  [profiler](../concepts/profiler.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common` for `env_vars.h`,
  `csrc/internal`, `src/torch_tpu`).
- Related doc-concepts: [error-anatomy-and-cpp-stacktraces](error-anatomy-and-cpp-stacktraces.md),
  [tpu-debug-checks](tpu-debug-checks.md),
  [launch-blocking-debug-eager](launch-blocking-debug-eager.md),
  [eager-execution-modes](eager-execution-modes.md),
  [xla-compilation-dumps](xla-compilation-dumps.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md),
  [tpu-profiler-config](tpu-profiler-config.md),
  [debugging-ladder](debugging-ladder.md).

## Source
Extracted from `docs/markdown/6_reference_4_environment_variables.md` (kept in place).
`docs/markdown/environment_variables.md` is an older, unnumbered export of the environment-variables reference with identical content (kept in place).
