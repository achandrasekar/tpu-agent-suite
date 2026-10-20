---
title: "Multi-tier compilation cache (tiers 1, 2 and 3)"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md, docs/markdown/environment_variables.md, docs/markdown/compilation_cache.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Multi-tier compilation cache (tiers 1, 2 and 3)

## Definition
The guide lists a "multi-tier compilation cache" that "eliminates redundant XLA compilations
across three cache tiers":
- **Tier 1:** a process-local in-memory cache, always on.
- **Tier 2:** host-local shared memory, so the processes of a multi-device job on one host can
  reuse each other's compiles.
- **Tier 3:** a persistent remote cache (such as Cloud Storage), reused across hosts and across
  job restarts.

## In torch_tpu (grounded)
All three tiers sit behind [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache").
A request enters through
[`CompilationCache.GetOrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1015 "csrc/common/compilation_cache.cc#CompilationCache.GetOrCompile").
A tier-1 hit returns the in-process executable. On a miss, the work runs on a background
compilation thread pool. [`EnqueueCompilation`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1026 "csrc/common/compilation_cache.cc#CompilationCache.EnqueueCompilation")
shows the chain: try tier 2, then tier 3, then compile with XLA.
- **Tier 2.**
  [`ShouldUseTier2Cache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1118 "csrc/common/compilation_cache.cc#CompilationCache.ShouldUseTier2Cache")
  and [`GetFromTier2OrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1127 "csrc/common/compilation_cache.cc#CompilationCache.GetFromTier2OrCompile")
  handle the shared-memory tier.
  [`TORCH_TPU_TIER2_COMPILATION_CACHE`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L202 "csrc/common/env_vars.h#kTorchTpuTier2CompilationCacheEnvVar")
  names it, and the value `disabled` turns it off. When the variable is unset, tier 2 is enabled
  if tier 3 is enabled or the world size is greater than 1 (the multi-process case the guide
  describes), and disabled otherwise.
- **Tier 3.** [`GetFromTier3OrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1180 "csrc/common/compilation_cache.cc#CompilationCache.GetFromTier3OrCompile")
  reads and writes serialized executables under the root directory returned by
  [`GetTier3CacheRootDir`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache_utils.cc#L213 "csrc/common/compilation_cache_utils.cc#GetTier3CacheRootDir"). The tier
  is off unless
  [`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L206 "csrc/common/env_vars.h#kTorchTpuTier3CompilationCacheRootEnvVar")
  is set.

All tiers share one key: a fingerprint of the graph, the input shapes, dtypes and layouts, and
the compile options. [compilation-cache-key](../concepts/compilation-cache-key.md) describes it.

## Why it matters / when it applies
XLA compiles are expensive, and an eager model or a `torch.compile` graph can produce many of
them. Tier 1 removes repeats inside one process. On a host running one process per device
(8 processes on a 4-chip v7 host), tier 2 lets one rank compile and the others load the result.
Tier 3 removes warm-up compiles after a preemption or restart, and lets all hosts of a
multi-host job share one compile. To use it, point the tier-3 root at a shared bucket or volume.
A cache miss that keeps repeating usually means the key is changing, for example from varying
shapes. See [torch-compile-tpu-backend](torch-compile-tpu-backend.md) for bounded dynamism.

## Cold starts, warm hits and recompilation triggers
The compilation-cache guide frames the cache from the training loop's side. The first time a
sequence of ops runs, it is a **cold start**: record the graph, fingerprint it, miss, compile,
execute, store. For large models this costs seconds to minutes. Iteration 2 and later are
**warm hits** that execute at once. The guide says the fingerprint depends on input shapes
(input scalar values count too), input dtypes and the exact sequence of ATen ops. Change any of
them and the cache most likely misses and compiles again.

How that maps to the key: a
[`CompilationCacheKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L846 "csrc/common/cache_key.h#CompilationCacheKey")
holds a [`GraphKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L809 "csrc/common/cache_key.h#GraphKey"), which
is split into a [`ShapelessKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L757 "csrc/common/cache_key.h#ShapelessKey")
(op sequence, edges, dtypes) and a
[`DimensionsKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L785 "csrc/common/cache_key.h#DimensionsKey")
(shapes), plus the compile-options fingerprint. "Scalar values count" is true only for scalars a
kernel keeps as raw parameters. A [`PromotedScalar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L98 "csrc/common/cache_key.h#PromotedScalar")
becomes a device argument and is not keyed, and a
[`MaybePromotedScalar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L150 "csrc/common/cache_key.h#MaybePromotedScalar")
keys only the special values 0, 1 and -1
([compilation-cache-key](../concepts/compilation-cache-key.md)).

The guide's scenarios:

```
Input pattern                                  Unique graphs  Result
---------------------------------------------- -------------- ----------------------------
fixed batch 32, fixed seq len 128              1              always hits
fixed batch 32, 3 seq buckets (128/512/1024)   3              3 cold starts, then all hits
variable seq len (every batch different)       N              recompiles every new shape
```

- **Ragged last batch.** 100 samples with batch size 32 gives a last batch of 4, so every epoch
  ends with a recompile. Use `DataLoader(..., drop_last=True)`.
- **Data-dependent ops** such as `nonzero()` produce shapes that depend on values and can force
  new graphs. Bounded dynamism is the compiled-mode answer
  ([torch-compile-tpu-backend](torch-compile-tpu-backend.md)).
- **`num_workers=0`.** The guide's DataLoader also sets `num_workers=0`, because forking workers
  after the TPU runtime has started deadlocks. The device module refuses to re-initialize in a
  forked child ([`_lazy_init`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L588 "src/torch_tpu/_internal/device/_device_module#_DeviceModule._lazy_init")).
  Use Grain for asynchronous input pipelines. See
  [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md) and
  [common-onboarding-pitfalls](common-onboarding-pitfalls.md).
- **Warm-up before timing.** Run 2 to 3 iterations, each ending in a materialization such as
  `.cpu()`, before starting a timer, so compile time is not counted as step time
  ([step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md)).

## Configuring tiers 2 and 3 (environment variables reference)
The environment-variables reference demonstrates the two variables with local folders and
counts the `.bin` executables each tier writes. Details the code adds to it:
- `TORCH_TPU_TIER2_COMPILATION_CACHE` takes a cache **name**, not a path: the backend places it
  under [`kTier2CacheRootDir`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/tier2_compilation_cache.cc#L50 "csrc/common/tier2_compilation_cache.cc#kTier2CacheRootDir")
  (`/dev/shm/torch_tpu_cache`), in a subdirectory named after the TorchTPU binary fingerprint
  ([`GetTier2CompilationCachePath`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/tier2_compilation_cache.cc#L108 "csrc/common/tier2_compilation_cache.cc#GetTier2CompilationCachePath"),
  [`GetTorchTpuBinaryFingerprint`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache_utils.cc#L91 "csrc/common/compilation_cache_utils.cc#GetTorchTpuBinaryFingerprint")), so a new build never
  loads an old build's executables. The reference's quick-reference row suggests a path as the
  value; the doc's own snippet and the code treat it as a name.
  [`GetTier2CacheName`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/tier2_compilation_cache.cc#L64 "csrc/common/tier2_compilation_cache.cc#GetTier2CacheName") also forces tier 2 on, with
  the name `default`, whenever tier 3 is enabled.
- `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` is a root path, normally a `gs://` bucket shared by
  all hosts; the doc uses a local folder only to demonstrate.
- Cache writes are asynchronous (the doc waits a few seconds before counting files).
- Both values are read once and memoized, so they must be set before the backend initializes
  (the reference says before `torch.device("tpu")`).
- `XLA_FLAGS` and `LIBTPU_INIT_ARGS` are fingerprinted into the compile-options part of the
  key, so changing dump or tuning flags misses every tier; see
  [xla-compilation-dumps](xla-compilation-dumps.md). All variables are grouped on
  [environment-variables](environment-variables.md).

## Connections
- Code concepts: [compilation-cache](../concepts/compilation-cache.md),
  [compilation-cache-key](../concepts/compilation-cache-key.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [eager-dynamic-shapes](../concepts/eager-dynamic-shapes.md).
- Module catalog: [catalog index](../catalog/index.md) (`csrc/common`).
- Related doc-concepts: [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [eager-execution-modes](eager-execution-modes.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [gke-xpk-workloads](gke-xpk-workloads.md),
  [environment-variables](environment-variables.md),
  [xla-compilation-dumps](xla-compilation-dumps.md),
  [torch-compile-graph-breaks](torch-compile-graph-breaks.md),
  [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place). The
"Configuring tiers 2 and 3" section is from `docs/markdown/6_reference_4_environment_variables.md`.
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
`docs/markdown/environment_variables.md` is an older, unnumbered export of the environment-variables reference with identical content (kept in place).
The "Cold starts, warm hits and recompilation triggers" section is from `docs/markdown/compilation_cache.md` (kept in place).
