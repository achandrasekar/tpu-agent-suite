---
title: "TpuProfilerConfig: tracer levels, output directory and hardware counters"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md
sources: [docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md, docs/markdown/performance_profiling.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TpuProfilerConfig: tracer levels, output directory and hardware counters

## Definition
`torch.tpu.profiler.TpuProfilerConfig` is the TPU-specific `experimental_config` for
`torch.profiler.profile()`. The doc's sanctioned options are `host_tracer_level` (default 2;
0=off, 1=critical, 2=all), `device_tracer_level` (default 1; 0=off, 1=on),
`python_tracer_level` (default 0) and `run_dir` (where `.xplane.pb` files go). Anything else is
passed through `experimental_options`, which is how fine-grained hardware performance counters
are enabled (for example `tpu_enable_periodic_counter_sampling` and
`tpu_tc_perf_counter_sampling_options`, Ironwood / TPU7x and later).

## In torch_tpu (grounded)
- [`TpuProfilerConfig`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/profiler/profiler_config.py#L89 "src/torch_tpu/_internal/profiler/profiler_config#TpuProfilerConfig") subclasses
  `torch.profiler._ExperimentalConfig` and is marked experimental. It serializes every option
  into one `custom_profiler_config` string of comma-separated `key:value` pairs; string values
  are quoted and escaped, and experimental keys may not clash with the stable ones or contain
  `:` / `,`. It also accepts a `worker_rank` (suffixes the XPlane file name with the rank) and
  `check_experimental_options` (default on: unknown experimental keys are rejected by the
  profiler backend). The docstring documents a level 3 for `host_tracer_level` (low-level XLA
  trace events) that the doc's table omits.
- On the C++ side the Kineto session parses that string with
  [`UpdateProfileOptions`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L280 "csrc/internal/profiler/tpu_profiler_plugin.cc#UpdateProfileOptions"): `run_dir` and
  `worker_rank` are pulled out, the three tracer levels go through
  [`UpdateProfileOption`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L202 "csrc/internal/profiler/tpu_profiler_plugin.cc#UpdateProfileOption") into the XLA
  `ProfileOptions`, and every other key becomes an advanced-configuration entry via
  [`SetAdvancedConfigValue`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L168 "csrc/internal/profiler/tpu_profiler_plugin.cc#SetAdvancedConfigValue") (quoted string,
  `true`/`false`, or integer). That pass-through is what makes the counter-sampling options work
  without TorchTPU knowing them.
- The output base directory is resolved by
  [`GetProfilerBaseOutputDir`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L332 "csrc/internal/profiler/tpu_profiler_plugin.cc#GetProfilerBaseOutputDir") in priority order:
  the `TORCH_TPU_PROFILER_OUTPUT_DIR` environment variable
  ([`kTorchTpuProfilerOutputDirEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L186 "csrc/common/env_vars.h#kTorchTpuProfilerOutputDirEnvVar")), then `run_dir` unless it is
  empty or `/tmp`, then `TMPDIR`, then `/tmp`. With no `run_dir`, the Python docstring says TPU
  traces follow Kineto's activities log directory so CPU and TPU traces stay co-located.

## Why it matters / when it applies
- The defaults mirror JAX/TensorFlow XLA profiling: host and device tracing on, Python tracing
  off to keep overhead down. Raise `python_tracer_level` only when you need Python frames.
- Hardware counters expose bottlenecks that are invisible in event traces, mainly for custom
  Pallas kernels ([pallas-custom-kernels](../concepts/pallas-custom-kernels.md)); counter
  indices and sampling parameters are documented in the OpenXLA kernel-profiling guide the doc
  links.
- Setting `TORCH_TPU_PROFILER_OUTPUT_DIR` overrides `run_dir` for every profile in the process,
  which is useful when a launcher wants all workers to write into one shared location.

## Connections
- Code concepts: [profiler](../concepts/profiler.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/internal` profiler plugin,
  `src/torch_tpu` `_internal/profiler`).
- Related doc-concepts: [torch-profiler-on-tpu](torch-profiler-on-tpu.md),
  [reading-tpu-profiles](reading-tpu-profiles.md),
  [environment-variables](environment-variables.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md`
(kept in place).
`docs/markdown/performance_profiling.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
