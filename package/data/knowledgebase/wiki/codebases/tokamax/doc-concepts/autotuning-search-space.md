---
title: "Op Configs and the autotuning search space"
type: doc-concept
provenance: doc
source: docs/autotuning.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Op Configs and the autotuning search space

## Definition
`docs/autotuning.md` opens with the core premise: "Kernel performance may be significantly
impacted by implementation-specific hyperparameters (e.g., tile sizes, layouts, etc.), known as
Op Configs in Tokamax. The optimal values for these configs must be empirically determined for
every input of interest, via *autotuning*." Every `tokamax.Op` implementation declares its
tunables and their ranges in `_get_autotuning_configs`. The autotuner "runs an exhaustive search
across this space to identify the configuration with the best performance (lowest execution
time)". The README adds the extension point: users "can autotune their own kernels with
`tokamax.autotune` by inheriting from the `tokamax.Op` class and overriding the
`tokamax.Op._get_autotuning_configs` method".

## In tokamax (grounded)
- **The hook.** [`Op._get_autotuning_configs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L379 "tokamax/_src/ops/op#Op._get_autotuning_configs") returns an empty set by
  default, which means "nothing to tune". A backend overrides it to return a `set` of its
  frozen `Config` dataclass. [`BoundArguments.autotuning_configs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L573 "tokamax/_src/ops/op#BoundArguments.autotuning_configs") exposes
  that set for one concrete call.
- **A TPU example.** The doc's example link points at the ragged-dot Mosaic TPU backend.
  At the pin, that is [`PallasMosaicTpuRaggedDot._get_autotuning_configs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L435 "tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu#PallasMosaicTpuRaggedDot._get_autotuning_configs").
  It builds power-of-two `tile_m` / `tile_k` / `tile_n` ranges starting at 128 (the TPU lane
  width), plus divisors of the padded K and the full K. `tile_m` is capped at 1024 "to ensure
  reasonable compilation time". Other TPU ops build different spaces: the fused conv1d + GDN
  op derives its tile sizes from a VMEM footprint model, while KDA returns a single
  `Config(chunk_size=64)` from
  [`PallasMosaicTpuKimiDeltaAttention._get_autotuning_configs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L213 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#PallasMosaicTpuKimiDeltaAttention._get_autotuning_configs").
- **Search.** [`BoundArguments.autotune`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L593 "tokamax/_src/ops/op#BoundArguments.autotune") hands the set to
  [`Autotuner.autotune`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/autotuner.py#L126 "tokamax/_src/autotuning/autotuner#Autotuner.autotune"). It compiles every candidate on a
  thread pool (each compile has a timeout, 600 s by default), randomly initializes the inputs
  once, and then times the candidates one at a time. The result is an
  [`AutotuningData`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/autotuner.py#L38 "tokamax/_src/autotuning/autotuner#AutotuningData") map from config to `BenchmarkData` or
  exception. [`AutotuningData.fastest_config`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/autotuner.py#L49 "tokamax/_src/autotuning/autotuner#AutotuningData.fastest_config") picks the
  lowest median time.
- **When no tuned config exists.** [`BoundArguments.default_config`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L469 "tokamax/_src/ops/op#BoundArguments.default_config") tries,
  in order: an explicit `op.config`, the cached autotuning data, and then the fallback chosen by
  the [`autotuning_cache_miss_fallback`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/config.py#L82 "tokamax/_src/config#autotuning_cache_miss_fallback") flag. The flag's values are
  `heuristics` (the default, which calls [`Op._get_heuristics_config`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L359 "tokamax/_src/ops/op#Op._get_heuristics_config")),
  `autotune` (tune on the spot) and `error`. A kernel author therefore usually writes two
  methods: a cheap heuristic and a search space.

## Why it matters / when it applies
- On TPU the tunables are mostly block shapes that must fit in VMEM and line up with the
  (8, 128) sublane/lane tiling. The best block shape changes with the shapes and dtypes of the
  inputs, so tuning results are keyed by the op's canonical abstract arguments
  ([`BoundArguments.autotuning_cache_key`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L538 "tokamax/_src/ops/op#BoundArguments.autotuning_cache_key")) and the device kind.
- The search is exhaustive. The size of the space trades tuning time against coverage. This is
  why the ragged-dot space caps `tile_m`, and why the doc lists "autotuning can be expensive" as
  a reason to serialize results (see [autotuning-result-reuse](autotuning-result-reuse.md)).
- Shipped results live in [`AutotuningCache`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/cache.py#L53 "tokamax/_src/autotuning/cache#AutotuningCache"), one JSON file per
  device kind and op. The doc calls this "the global library-wide cache".

## Connections
- Code concepts: [autotuning](../concepts/autotuning.md) (the full Autotuner / cache
  mechanism), [op-framework](../concepts/op-framework.md) (`Config`, heuristics,
  `BoundArguments`), [ragged-dot-gmm](../concepts/ragged-dot-gmm.md) (the doc's example search
  space), [gmm-v2](../concepts/gmm-v2.md),
  [batched-ragged-paged-attention](../concepts/batched-ragged-paged-attention.md),
  [gated-delta-rule](../concepts/gated-delta-rule.md),
  [splash-attention](../concepts/splash-attention.md),
  [benchmarking-and-serialization](../concepts/benchmarking-and-serialization.md) (how each
  candidate is timed).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [autotuner-input-formats](autotuner-input-formats.md),
  [autotuning-result-reuse](autotuning-result-reuse.md),
  [benchmark-timing-methods](benchmark-timing-methods.md).

## Source
Extracted from `docs/autotuning.md` ("High-level Overview") and `README.md` / `docs/basic_usage.md`
("Autotuning"), kept in place.
