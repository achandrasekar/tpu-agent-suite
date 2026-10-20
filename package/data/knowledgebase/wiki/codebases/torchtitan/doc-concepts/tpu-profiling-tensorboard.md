---
title: "Profiling TorchTitan TPU runs for TensorBoard / XProf"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Profiling TorchTitan TPU runs for TensorBoard / XProf

## Definition
The README says that TorchTitan on TorchTPU "integrates with a profiler to export profiling data".
You turn profiling on with a CLI flag and tune the capture window with the related flags. Each
process writes its trace under the job's dump folder. With the GCS FUSE mount from
[tpu-multihost-launch](tpu-multihost-launch.md), the traces land in a bucket that TensorBoard reads
directly (`pip install tensorboard_plugin_profile tensorboard xprof`). The GKE example turns it on
with
`--profiler.enable_profiling --profiler.profile_freq=5 --profiler.profiler_warmup=3 --profiler.profiler_active=1`,
together with `--metrics.enable_tensorboard --metrics.log_freq=5`.

## In torchtitan (grounded)
- **Config.** The flags map to [`Profiler.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L129 "torchtitan/tools/profiler#Profiler.Config"):
  [`enable_profiling`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L130 "torchtitan/tools/profiler#Profiler.Config.enable_profiling"),
  [`profile_freq`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L136 "torchtitan/tools/profiler#Profiler.Config.profile_freq"),
  [`profiler_warmup`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L167 "torchtitan/tools/profiler#Profiler.Config.profiler_warmup")
  and [`profiler_active`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L160 "torchtitan/tools/profiler#Profiler.Config.profiler_active").
  This block is reached through the trainer's
  [`profiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L99 "torchtitan/trainer#Trainer.Config.profiler")
  field. [`Profiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L106 "torchtitan/tools/profiler#Profiler")
  is built with `base_folder=dump_folder`
  ([`dump_folder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L96 "torchtitan/trainer#Trainer.Config.dump_folder")).
  Traces go to `<dump_folder>/<save_traces_folder>`, where
  [`save_traces_folder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L133 "torchtitan/tools/profiler#Profiler.Config.save_traces_folder")
  defaults to [`PROFILE_DIR`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L31 "torchtitan/tools/profiler#PROFILE_DIR")
  (`profiling/traces`).
  [`build_torch_profiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L286 "torchtitan/tools/profiler#Profiler.build_torch_profiler")
  adds the TPU activity on TPU, which writes an XPlane device trace per rank under `rank_<r>`. The
  constraint `profile_freq >= warmup + active` is now checked when the config is built:
  [`Profiler.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/profiler.py#L193 "torchtitan/tools/profiler#Profiler.Config.__post_init__")
  raises a `ValueError` if profiling is enabled and the window does not fit. (It used to be an
  assert inside `build_torch_profiler`.) The README's `5 / 3 / 1` values satisfy it.
- **Optional JAX profiler.** When
  [`use_jax_profiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L70 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.use_jax_profiler")
  is set, the patched [`custom_enter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/profiler_workaround.py#L39 "torchtitan/experiments/tpu/profiler_workaround#custom_enter")
  swaps in [`JaxProfiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/jax_profiler.py#L14 "torchtitan/experiments/tpu/jax_profiler#JaxProfiler").
  It traces rank 0 only, into `<trace dir>/iteration_<n>/rank_<r>`. An absolute or URL
  `save_traces_folder` is used as-is, which is how you point it straight at `gs://`.
- **TensorBoard metrics.** `--metrics.enable_tensorboard` is
  [`enable_tensorboard`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/metrics.py#L290 "torchtitan/components/metrics#MetricsProcessor.Config.enable_tensorboard")
  and writes scalar logs at the
  [`log_freq`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/metrics.py#L284 "torchtitan/components/metrics#MetricsProcessor.Config.log_freq")
  cadence. These are separate from the profiler traces.
  [`MetricsProcessor.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/metrics.py#L310 "torchtitan/components/metrics#MetricsProcessor.Config.__post_init__")
  now rejects `log_freq <= 0`.

## Why it matters / when it applies
- **Getting traces into the bucket.** On GKE the bucket reaches the pods only through the job's
  `--mount "$BUCKET_NAME;/data;rw"` flag on `gcluster job submit`, which needs the cluster created
  with `enable_gcsfuse_csi=true` (see [tpu-multihost-launch](tpu-multihost-launch.md)). Point
  `--dump_folder` at a `/data/...` path so the traces land in the bucket.
- **Pick windows past compilation.** The first step on TPU is dominated by XLA compilation (see
  [torchtpu-backend](../concepts/torchtpu-backend.md)). The wait-plus-warmup part of each
  `profile_freq` cycle keeps the first recorded step away from the compile-heavy first step. Keep
  that margin when you shorten the cycle.
- **Treat memory and MFU numbers with care.** The README warns that "Memory and MFU metrics might be
  inaccurate". The profiling-and-tooling page explains that on TPU the peak-memory reset is a no-op
  and the MFU denominator comes from a device-name table.

> [!warning] Doc vs code
> - **Wrong flag spelling in the prose.** The README prose says "`--profiling.enable-profiling`" and
>   "`--job.dump_folder` + `profile_trace`". The config section is `profiler`, not `profiling`.
>   `dump_folder` is a top-level trainer field with no `job.` section. The default subdirectory is
>   `profiling/traces`. The README's own GKE command uses the correct `--profiler.*` spelling, so
>   copy that.
> - **The TensorBoard path is illustrative.**
>   `tensorboard --logdir=gcs://${BUCKET_NAME}/torchtitan-llama3-8b/rank_0` does not match the
>   default layout (`<dump_folder>/profiling/traces/...`), and the scheme should be `gs://`. Point
>   `--logdir` at the actual `save_traces_folder` under your dump folder.

## Connections
- Code concepts: [profiling-and-tooling](../concepts/profiling-and-tooling.md) (profiler schedule,
  XPlane output, TPU memory caveats), [metrics-and-observability](../concepts/metrics-and-observability.md)
  (TensorBoard scalars, MFU), [torchtpu-backend](../concepts/torchtpu-backend.md) (JAX-profiler hook).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/tools/profiler`,
  `torchtitan/experiments/tpu/profiler_workaround`, `torchtitan/experiments/tpu/jax_profiler`).
- Related doc-concepts: [tpu-multihost-launch](tpu-multihost-launch.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md).
- Runtime side: torch_tpu [profiler](../../torch_tpu/concepts/profiler.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (section "Profile workloads using
TensorBoard", plus the profiler flags in "Deploy the training workload"; kept in place).
