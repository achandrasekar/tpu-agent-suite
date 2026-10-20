---
title: "TPU training-loop hygiene: fork safety, static shapes, MXU alignment"
type: doc-concept
provenance: doc
source: examples/classical_ml/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TPU training-loop hygiene: fork safety, static shapes, MXU alignment

## Definition
The classical-ML recipe (MLP training on a Cloud TPU v6e, eager versus
`torch.compile(model, backend='tpu', dynamic=False)`, instrumented with TensorBoard and XProf)
lists three loop-level practices besides asynchronous execution: use
`DataLoader(num_workers=0)` (or Grain) because forked workers deadlock on the initialized TPU
runtime; keep batch shapes static with `drop_last=True` because "XLA compiles dedicated machine
code for each unique tensor dimension"; and align batch and hidden sizes to multiples of 128 to
fill the MXU systolic tiles. It also warns that the first compiled window pays tracing and XLA
compilation, so steady-state numbers should come from epoch 2 onward.

## In torch_tpu (grounded)
- **Fork safety.** The runtime tracks whether a process was forked after PjRt initialized
  ([`IsInBadFork`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.cc#L332 "csrc/pjrt/pjrt_state.cc#PjrtBackend.IsInBadFork"), fed by a
  [`BadForkChecker`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.h#L41 "csrc/pjrt/pjrt_state.h#BadForkChecker") hook). In such a child,
  [`_lazy_init`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L588 "src/torch_tpu/_internal/device/_device_module#_DeviceModule._lazy_init") raises
  "cannot access or re-initialize TPU in a forked subprocess" and tells you to use the `spawn`
  start method. The README's silent-deadlock warning is the case where a forked worker touches
  runtime state before that check; `num_workers=0` or a `spawn` context avoids both.
- **Static shapes.** Each distinct input shape is a distinct cache key
  ([`CompilationCacheKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L846 "csrc/common/cache_key.h#CompilationCacheKey")), so a short final batch triggers a fresh
  compile in both eager (per-op kernels) and compiled mode. The loader's
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile") defaults `dynamic=False` for the TPU backend
  (the only value the backend accepts), which makes this explicit. When shapes genuinely vary, declare bounds with
  [`MarkDynamic`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/dynamism/dynamism.cc#L38 "csrc/internal/dynamism/dynamism.cc#MarkDynamic") (bounded dynamism) rather than letting each
  size compile separately.
- **MXU alignment** is a hardware property, not a TorchTPU code path; no symbol implements it.
- **Warm-up.** The first window's cost is compilation through
  [`GetOrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1015 "csrc/common/compilation_cache.cc#CompilationCache.GetOrCompile"); later steps
  hit the cache.

## Why it matters / when it applies
These are the first things to check when onboarding a standard PyTorch training script:
DataLoader worker counts copied from GPU code, `drop_last=False`, and odd hidden sizes are common
and each costs either a hang, a recompilation per epoch, or wasted MXU capacity. The recipe's
benchmark (about 12k samples/s warm in both modes, 97.44% test accuracy) is a sanity baseline,
not a performance target.

## Connections
- Code concepts: [pjrt-client](../concepts/pjrt-client.md),
  [compilation-cache-key](../concepts/compilation-cache-key.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [eager-dynamic-shapes](../concepts/eager-dynamic-shapes.md),
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md),
  [device-backend-registration](../concepts/device-backend-registration.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/pjrt`, `csrc/common`,
  `src/torch_tpu/_internal/device`).
- Related doc-concepts: [common-onboarding-pitfalls](common-onboarding-pitfalls.md) (DataLoader
  workers and other first-run failures), [tpu-hardware-model](tpu-hardware-model.md) (MXU tiles),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md) (bounded dynamism),
  [async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md),
  [step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md).

## Source
Extracted from `examples/classical_ml/README.md` (kept in place).
