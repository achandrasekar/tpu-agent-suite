---
title: "Step-scheduled profiling and benchmark hygiene on TPU"
type: doc-concept
provenance: doc
source: examples/classical_ml/README.md
sources: [examples/classical_ml/README.md, examples/tutorials/FLUX_optimization/README.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Step-scheduled profiling and benchmark hygiene on TPU

## Definition
Both the classical-ML recipe and the FLUX tutorial measure TPU performance the same way:
separate warm-up from timed steps (compilation happens on the first invocation), never insert
host synchronization or tensor prints inside the timed loop, and capture device time with
`torch.profiler.profile` instead. The classical-ML recipe uses
`torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=1)` with `profiler.step()` at each
window boundary, which skips cold compilation, bounds trace size and writes `ProfilerStep`
markers so TensorBoard's XProf Overview page can break step time into compute versus
input/host overhead. The FLUX test runs `--num_warmup_steps` before starting the timer, wraps the
benchmark loop in a profiler with CPU and TPU activities, calls `prof.step()` at step
boundaries, and only afterwards aggregates per-op device time (`self_device_time_total`) into a
top-operators table.

## In torch_tpu (grounded)
- PyTorch's profiler reaches the TPU through a Kineto plugin session,
  [`TpuKinetoProfilerSession`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/profiler/tpu_profiler_plugin.h#L42 "csrc/internal/profiler/tpu_profiler_plugin.h#TpuKinetoProfilerSession"), which emits XProf
  `.xplane.pb` traces (host threads, TPU device queues, HBM usage).
- The only public profiler API is `torch.tpu.profiler.TpuProfilerConfig` (passed to
  `torch.profiler.profile` as `experimental_config`); use the standard `torch.profiler` entry
  points, as both READMEs do. An older internal context manager,
  [`profile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/profiler/_api.py#L119 "src/torch_tpu/_internal/profiler/_api#profile"), is slated for removal; it is marked with
  [`forbid_in_torch_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L471 "src/torch_tpu/_internal/utils/annotations#forbid_in_torch_compile") because profiling from
  inside a compiled region would force eager execution and misrepresent compiled code. The same
  rule applies in general: start and stop profiling around the compiled call, not inside it.
- Warm-up exists because the first step goes through
  [`GetOrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1015 "csrc/common/compilation_cache.cc#CompilationCache.GetOrCompile") and pays
  compilation; later steps are cache hits.
- XLA module names in the trace encode where materialization was triggered:
  [`BuildModuleNameFromPyContext`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/op_builder_utils.cc#L1175 "csrc/ops/op_builder_utils.cc#BuildModuleNameFromPyContext") names eager modules
  `tt_jit_<file>_L<line>C<col>_<func>_<op>`, which is how the FLUX tutorial traces a copy back to
  its source line ([zero-copy-views-vs-layout-copies](zero-copy-views-vs-layout-copies.md)).

## Why it matters / when it applies
- A timed loop that calls `.item()` or `synchronize()` per step measures host round trips, not
  TPU execution ([async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md)).
- The first profiled epoch includes trace-flush overhead; report steady state from epoch 2 or
  after the warm-up steps.
- Host CPU/RAM and HBM telemetry are already in the XProf trace; the recipe logs only
  loss, step time and accuracy as TensorBoard scalars.

## Connections
- Code concepts: [profiler](../concepts/profiler.md),
  [synchronization](../concepts/synchronization.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [python-context-and-op-names](../concepts/python-context-and-op-names.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/internal/profiler`,
  `src/torch_tpu/_internal/profiler`).
- Related doc-concepts: [torch-profiler-on-tpu](torch-profiler-on-tpu.md) (capturing traces),
  [reading-tpu-profiles](reading-tpu-profiles.md) (interpreting them),
  [async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md),
  [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md).

## Source
Extracted from `examples/classical_ml/README.md` and
`examples/tutorials/FLUX_optimization/README.md` (kept in place).
