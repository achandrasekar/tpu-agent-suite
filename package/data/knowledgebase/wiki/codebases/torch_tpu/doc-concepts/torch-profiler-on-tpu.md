---
title: "Capturing TPU traces with torch.profiler and viewing them in xprof"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md
sources: [docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md, docs/markdown/performance_profiling.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Capturing TPU traces with torch.profiler and viewing them in xprof

## Definition
TorchTPU does not ship its own profiling API: the standard `torch.profiler.profile()` context
manager traces both host and TPU activity once `torch_tpu` is loaded. You list
`ProfilerActivity.CPU` (dispatch, data loading) and `ProfilerActivity.TPU` (compute, memory,
interconnect), set `on_trace_ready=torch.profiler.tensorboard_trace_handler(<dir>)`, optionally
pass a `TpuProfilerConfig` as `experimental_config`, and call `prof.step()` once per training
step. The traces are then opened in TensorBoard (with `tensorboard-plugin-profile`) or in the
standalone `xprof --logdir=<dir>` server, under the "Profile" tab.

## In torch_tpu (grounded)
- The TPU activity is a PyTorch `PrivateUse1` Kineto plugin: the
  [`TpuProfiler`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.h#L85 "csrc/internal/profiler/tpu_profiler_plugin.h#TpuProfiler") factory is registered with
  Kineto and hands out a [`TpuKinetoProfilerSession`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.h#L42 "csrc/internal/profiler/tpu_profiler_plugin.h#TpuKinetoProfilerSession")
  per profile, whose [`start`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L409 "csrc/internal/profiler/tpu_profiler_plugin.cc#TpuKinetoProfilerSession.start")
  and [`stop`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L485 "csrc/internal/profiler/tpu_profiler_plugin.cc#TpuKinetoProfilerSession.stop")
  drive the XLA profiler session and collect an XSpace.
- The Python-side activity enum that adds `TPU` next to `CPU` is
  [`ProfilerActivity`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/profiler/_api.py#L28 "src/torch_tpu/_internal/profiler/_api#ProfilerActivity")
  ([`ProfilerActivity.TPU`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/profiler/_api.py#L38 "src/torch_tpu/_internal/profiler/_api#ProfilerActivity.TPU")).
- Before collecting data, `stop` runs
  [`SynchronizeTpuDevicesBeforeStop`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L464 "csrc/internal/profiler/tpu_profiler_plugin.cc#SynchronizeTpuDevicesBeforeStop"), which
  synchronizes every addressable TPU device so asynchronously dispatched work lands in the
  trace. The doc's loop still calls `torch.accelerator.synchronize()` each step so step
  boundaries in the trace line up with device work (eager ops are deferred; see
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)).
- The `.xplane.pb` file is written under `<base>/plugins/profile/<timestamp>/<hostname>.xplane.pb`
  by [`GetXPlaneOutputPath`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.cc#L365 "csrc/internal/profiler/tpu_profiler_plugin.cc#GetXPlaneOutputPath"), the TensorBoard
  xprof layout, which is why `tensorboard --logdir=<dir>` finds it. How `<base>` is chosen is on
  [tpu-profiler-config](tpu-profiler-config.md).

## Why it matters / when it applies
- The profiler adds overhead (the doc estimates ~5-10% per step): profile a representative
  window of 100-1000 steps, not a whole run.
- TensorBoard needs `setuptools<81` (newer releases drop `pkg_resources`).
- The four views to use: Overview (step time, utilization, top ops), Trace Viewer (timeline;
  look for idle gaps), Op Profile (time per op), Memory Profile (HBM over time). What to look
  for in them is on [reading-tpu-profiles](reading-tpu-profiles.md).
- The doc's last check is that the log directory actually contains trace files; an empty
  directory means the configuration (activities or output directory) is wrong.

## Connections
- Code concepts: [profiler](../concepts/profiler.md) (the Kineto plugin, XProf session and
  callback handler in depth), [synchronization](../concepts/synchronization.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/internal` for the profiler plugin,
  `src/torch_tpu` for `_internal/profiler`).
- Related doc-concepts: [tpu-profiler-config](tpu-profiler-config.md),
  [reading-tpu-profiles](reading-tpu-profiles.md), [debugging-ladder](debugging-ladder.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md`
(kept in place). `docs/markdown/6_reference_3_xprof_for_torchtpu_reference.md` is a title-only
stub for the same topic at this pin.
`docs/markdown/performance_profiling.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
