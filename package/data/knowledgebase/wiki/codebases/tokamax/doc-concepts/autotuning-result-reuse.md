---
title: "Reusing autotuning results: context manager, serialization, merging and determinism"
type: doc-concept
provenance: doc
source: docs/autotuning.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Reusing autotuning results: context manager, serialization, merging and determinism

## Definition
An `AutotuningResult` is the single output type of autotuning. Both `docs/autotuning.md` and the
README describe three ways to use it: "directly in your program as a context manager,
serialized and stored in your private cache for later use, or as part of the global
library-wide cache". Results can also be merged with `|`. The README explains why freezing
results matters: "autotuning is fundamentally non-deterministic: measuring kernel execution
times is noisy. As different configs chosen during autotuning can lead to different numerics,
this is a potential source of numerical non-determinism. Serializing and reusing fixed
autotuning results is a way to ensure the same numerics across sessions."

## In tokamax (grounded)
- [`AutotuningResult`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L96 "tokamax/_src/autotuning/api#AutotuningResult") is a frozen dataclass holding a
  `device_kind`, a tuple of `(BoundArguments, AutotuningData)` pairs, and the `tokamax_version`
  that produced it.
- **Context manager.** [`AutotuningResult.__enter__`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L161 "tokamax/_src/autotuning/api#AutotuningResult.__enter__") pushes an
  overlay keyed by `op -> device_kind -> autotuning_cache_key` onto a context-local stack. It
  also adds the result's id to a JAX config value, so a cached `jax.jit` trace from outside the
  context is not reused inside it. Per the class docstring, `default_config` checks the overlay
  first and falls back to the global cache, and when contexts are nested the innermost wins.
- **Serialization.** [`AutotuningResult.dumps`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L124 "tokamax/_src/autotuning/api#AutotuningResult.dumps") (with optional
  `prune_errors`) and [`AutotuningResult.loads`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L158 "tokamax/_src/autotuning/api#AutotuningResult.loads") round-trip through
  a pydantic adapter. [`AutotuningResult.load`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L154 "tokamax/_src/autotuning/api#AutotuningResult.load") reads from a file
  object. [`AutotuningResult.dump_cache_str`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L135 "tokamax/_src/autotuning/api#AutotuningResult.dump_cache_str") emits the per-op JSON
  layout used by the shipped [`AutotuningCache`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/cache.py#L53 "tokamax/_src/autotuning/cache#AutotuningCache"). That is the path
  to contributing results to the library-wide cache.
- **Merging.** [`AutotuningResult.__or__`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L179 "tokamax/_src/autotuning/api#AutotuningResult.__or__") requires both results to
  have the same `device_kind` (otherwise it raises `ValueError`). When the same cache key appears
  on both sides, the per-config data are merged, and the right-hand side wins where they
  overlap.

> [!warning] `AutotuningResult.load` takes a file object
> The doc's merge example calls `tokamax.AutotuningResult.load(path_to_file_1)`.
> [`AutotuningResult.load`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L154 "tokamax/_src/autotuning/api#AutotuningResult.load") is `cls.loads(fp.read())`, so pass an
> open file (`with open(path) as f: AutotuningResult.load(f)`), not a path string.

## Why it matters / when it applies
- Tuning cost: the search is exhaustive, so serializing the result avoids paying for it again
  on every launch.
- Numerics: different tile shapes change accumulation order. A frozen result pins the configs,
  and so pins bitwise numerics across runs. This matters for reproducible TPU training and for
  regression tests.
- Merging lets you tune parts of a model separately (forward and `jax.grad`, or separate
  profiles) on the same TPU generation and apply them together. Results are tied to a device
  kind, and merging across TPU generations is rejected.
- The shipped cache is consulted automatically. `tokamax_ignore_autotuning_cache`
  ([`ignore_autotuning_cache`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/config.py#L108 "tokamax/_src/config#ignore_autotuning_cache")) turns it off.

## Connections
- Code concepts: [autotuning](../concepts/autotuning.md) (overlay stack, cache layout,
  `dump_cache_str`), [op-framework](../concepts/op-framework.md) (`default_config`,
  `autotuning_cache_key`), [torch-tpu-bridge](../concepts/torch-tpu-bridge.md) (how cached
  configs reach the PyTorch wrappers).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [autotuning-search-space](autotuning-search-space.md),
  [autotuner-input-formats](autotuner-input-formats.md).

## Source
Extracted from `docs/autotuning.md` ("Autotuner Output Usage") and `README.md` /
`docs/basic_usage.md` ("Autotuning"), kept in place.
