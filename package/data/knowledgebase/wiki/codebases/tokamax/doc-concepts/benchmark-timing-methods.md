---
title: "Device-time benchmarking: timing methods, iterations and thermal noise"
type: doc-concept
provenance: doc
source: docs/benchmarking.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Device-time benchmarking: timing methods, iterations and thermal noise

## Definition
All three docs start from the same observation: "JAX Python overhead is often much larger than
the actual accelerator kernel execution time", so timing `jax.block_until_ready(f(x))` is not
useful. `tokamax.benchmark` (and the runner returned by `compile_benchmark`) therefore lets you
choose a *timing method* that measures only device execution. `docs/benchmarking.md` "strongly
recommends" an XProf-based method for TPU kernels. That method "invokes the XProf profiler and
measures execution time on the hardware" with "almost no instrumentation overhead due to custom
full-stack support including the hardware and the compiler". On GPU, XProf goes through CUPTI,
or you can call CUPTI directly. Either adds "some variable overhead, typically up to 5%". The
doc also covers iteration count. More iterations reduce noise, but too many in a short window
can trigger thermal throttling on compute-heavy kernels. The suggested approach is "a small
number of iterations in each experiment with multiple spaced out experiments".

## In tokamax (grounded)
- [`benchmark`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L654 "tokamax/_src/benchmarking#benchmark") is `compile_benchmark(f, x)(x, ...)`.
  [`compile_benchmark`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L565 "tokamax/_src/benchmarking#compile_benchmark") jit-lowers and compiles once, recording
  lowering time, compile time and peak memory from the compiled module's memory analysis. It
  then returns a runner that takes `iterations` (default 5), `method` and `event_filter_regex`.
- Method keys come from the `TimingMethod` literal: `'wallclock'`, `'cupti'`, `'xprof'`,
  `'hermetic_xprof'`. They are dispatched through [`_TIMERS`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L545 "tokamax/_src/benchmarking#_TIMERS._TIMERS").
  With `method=None` the runner picks by platform from
  [`_DEFAULT_TIMING_METHOD`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L552 "tokamax/_src/benchmarking#_DEFAULT_TIMING_METHOD._DEFAULT_TIMING_METHOD"): `tpu -> 'hermetic_xprof'`,
  `gpu -> 'cupti'`, anything else -> `'wallclock'`. `'cupti'` raises on non-GPU platforms, and
  the XProf methods raise on platforms other than GPU and TPU.
- [`hermetic_xprof_timer`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L533 "tokamax/_src/benchmarking#hermetic_xprof_timer") wraps [`xprof_timer`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L512 "tokamax/_src/benchmarking#xprof_timer")
  with `use_jax_profiler=True` and never returns an XProf URL. The underlying
  [`XprofProfileSession`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L108 "tokamax/_src/benchmarking#XprofProfileSession") sums device op time as a disjoint interval
  union. `event_filter_regex` restricts that sum to matching XLA op names, for example to time a
  single kernel inside a larger program. [`cupti_timer`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L507 "tokamax/_src/benchmarking#cupti_timer") and
  [`wallclock_timer`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L497 "tokamax/_src/benchmarking#wallclock_timer") cover the other keys.
- The result is a [`BenchmarkData`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L77 "tokamax/_src/benchmarking#BenchmarkData") with per-iteration times and
  [`BenchmarkData.median_evaluation_time_ms`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L88 "tokamax/_src/benchmarking#BenchmarkData.median_evaluation_time_ms"). The autotuner ranks
  configs by that median.

> [!warning] `method=xprof_hermetic` is not a valid key
> `docs/benchmarking.md` recommends `method=xprof_hermetic` for TPU (and mentions
> `xprof_hermetic` for GPU). The code key is `'hermetic_xprof'`: see
> [`_TIMERS`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L545 "tokamax/_src/benchmarking#_TIMERS._TIMERS") and the docstring of
> [`benchmark`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L654 "tokamax/_src/benchmarking#benchmark") ("'hermetic_xprof' uses XProf as the profiler, and is
> the recommended timing method for TPU"). Passing `'xprof_hermetic'` falls through to
> `_TIMERS[method]` and raises `KeyError`. On TPU you can also just leave `method=None`, which
> already selects `'hermetic_xprof'`.

## Why it matters / when it applies
- For TPU kernel work, the device-only, hermetic XProf time is the number to optimize and
  compare against roofline estimates. The KDA design docs' HBM-utilization tables are an
  example. Wallclock includes dispatch and Python overhead, which dominates microsecond-scale
  kernels.
- Use `event_filter_regex` when the standardized function contains setup ops (casts,
  transposes) that you don't want counted against the kernel.
- Iteration policy is a judgement call. The runner's `iterations` (default 5) sets how many
  timed runs it records, and repeated spaced-out experiments beat one long burst when thermals
  matter.

## Connections
- Code concepts: [benchmarking-and-serialization](../concepts/benchmarking-and-serialization.md)
  (XProf session internals, disjoint interval union),
  [autotuning](../concepts/autotuning.md) (benchmarks inside the autotuner),
  [torch-tpu-bridge](../concepts/torch-tpu-bridge.md) (the TPU tuning and benchmarking scripts).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [standardized-benchmark-functions](standardized-benchmark-functions.md),
  [autotuning-search-space](autotuning-search-space.md),
  [kda-chunked-backward-fusion](kda-chunked-backward-fusion.md) (roofline measurements).

## Source
Extracted from `docs/benchmarking.md` ("Run Iterations", "Benchmarking Method"), `README.md` and
`docs/basic_usage.md` ("Benchmarking"), kept in place.
