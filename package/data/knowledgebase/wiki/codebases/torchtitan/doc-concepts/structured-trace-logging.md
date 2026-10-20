---
title: "Structured trace logging: per-rank JSONL spans and Gantt traces"
type: doc-concept
provenance: doc
source: torchtitan/observability/structured_logger/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Structured trace logging: per-rank JSONL spans and Gantt traces

## Definition
The observability README describes "structured logging for distributed training". Every rank
emits JSONL events for phase timing, diagnostics and post-hoc analysis. Its stated design
principles:
- The output is "LLM friendly".
- It serves both SPMD pretraining and RL runs with many independent actors that share no
  process group.
- It is "invisible to users": there are no metric dictionaries to pass around.
- It must "never block training".
- Backends plug in through handler factories.

User code marks work with spans, instants and scalars, and stamps a step context once per step.
Every record then carries `rank`, `source`, `caller`, `time_us`, `step`, `relative_step` and
the optional `step_tags`.

## In torchtitan (grounded)
- **Init.** [`init_structured_logger`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L224 "torchtitan/observability/structured_logger/structured_logging#init_structured_logger")
  is idempotent ("Repeated calls do not duplicate handlers"). Rank defaults to `$RANK`, so it can
  run before `torch.distributed` is set up.
  With `enable=False` it turns every trace call into a no-op; that flag is wired to
  [`DebugConfig.enable_structured_logging`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L428 "torchtitan/config/configs#DebugConfig.enable_structured_logging")
  (default True). Handler factories come from `TITAN_STRUCT_LOGGER_HANDLERS`. When that
  variable is set, only the listed factories run. Otherwise the default is
  [`register_jsonl_handler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/jsonl_handler.py#L189 "torchtitan/observability/structured_logger/jsonl_handler#register_jsonl_handler"),
  and [`TraceJsonlHandler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/jsonl_handler.py#L168 "torchtitan/observability/structured_logger/jsonl_handler#TraceJsonlHandler")
  writes `{output_dir}/structured_logs/{source}.global_rank_{rank}.{timestamp}-{random}.jsonl`.
  Records are serialized by
  [`TraceJsonlFormatter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/jsonl_handler.py#L41 "torchtitan/observability/structured_logger/jsonl_handler#TraceJsonlFormatter").
  Every record may also carry an optional `context` field
  ([`ExtraFields`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L75 "torchtitan/observability/structured_logger/structured_logging#ExtraFields")
  gained `CONTEXT` at this pin). The core entry point [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/train.py#L48 "torchtitan/train#main")
  calls it with `source="training"`.
- **Emit.** There are three emitters:
  - [`log_trace_span`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L364 "torchtitan/observability/structured_logger/structured_logging#log_trace_span")
    works as a context manager or decorator, sync or async. It emits `_start`, `_end` (whose
    value is the elapsed wall time in ms) and `_error` on an exception.
  - [`log_trace_instant`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L339 "torchtitan/observability/structured_logger/structured_logging#log_trace_instant")
    emits a point-in-time marker.
  - [`log_trace_scalar`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L295 "torchtitan/observability/structured_logger/structured_logging#log_trace_scalar")
    emits `metric_value` records and skips non-numeric values with a warning.

  Payloads are built by [`event_extra`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L159 "torchtitan/observability/structured_logger/structured_logging#event_extra")
  and filtered by [`TraceEventsOnlyFilter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L180 "torchtitan/observability/structured_logger/structured_logging#TraceEventsOnlyFilter").
- **Records from other loggers.** Init now also installs
  [`_StructuredRecordForwarder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L88 "torchtitan/observability/structured_logger/structured_logging#_StructuredRecordForwarder")
  on Python's root logger. Any propagating logger that emits a record carrying `log_type_name`
  (built with `event_extra`) is bridged to the structured logger's handlers; the checkpoint
  backend's `torch_checkpointing` logger is the documented user. Native `log_trace_*` records are
  not duplicated, because the structured logger does not propagate.
- **Subprocesses.**
  [`get_structured_logger_subprocess_init_fn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/structured_logging.py#L151 "torchtitan/observability/structured_logger/structured_logging#get_structured_logger_subprocess_init_fn")
  returns a partial of `init_structured_logger` with the parent's `source`, `output_dir` and
  `rank` (or `None` when logging is disabled or has no handler). The async checkpoint saver
  passes it to its writer subprocess
  ([`_default_backend_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/torch_checkpointing.py#L180 "torchtitan/components/checkpointer/torch_checkpointing#_default_backend_config")),
  so checkpoint-write events land in the same per-rank JSONL stream.
- **Step context.** [`set_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/step_state.py#L101 "torchtitan/observability/structured_logger/step_state#set_step")
  stores the step and clears the tags.
  [`add_step_tag`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/step_state.py#L153 "torchtitan/observability/structured_logger/step_state#add_step_tag")
  annotates the current step (for example `gc`). Both keep their state in a ContextVar plus a
  module global, so asyncio tasks in RL actors each see a consistent step.
- **Analysis.** [`generate_gantt_trace`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/observability/structured_logger/gantt_generator.py#L25 "torchtitan/observability/structured_logger/gantt_generator#generate_gantt_trace")
  merges the per-rank files into a Chrome Trace JSON for Perfetto. Auto-named asyncio tasks are
  packed into `0..K-1` tracks, where K is the peak concurrency.
- **Instrumented phases.** The core trainer wraps `fetching_batch`, `fwd_bwd`, `optim`,
  `collect_dist_metrics`, `step` and others. The RL generator and weight sync add their own spans.

> [!warning] Doc vs code
> The README (unchanged at this pin) says only the `sl.*` helpers write to the structured
> logger and does not describe the root-logger forwarder or the subprocess initializer; at this
> pin records from other propagating loggers (the checkpoint backend) also reach the trace
> handlers.

## Why it matters / when it applies
When one rank is slow, per-rank stage timings are the fastest way to find it: the Gantt view
shows which rank or actor fell behind in which phase. Because the logs are queryable, an agent
can read them during or after a run. The step tags let an analysis drop steps that are not
representative (GC, eval).

**On TPU, keep three caveats in mind.**
1. The TPU entry point ([`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer"))
   calls `init_logger()` but never `init_structured_logger`. The spans that
   [`TPUTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L42 "torchtitan/experiments/tpu/train#TPUTrainer")
   inherits from the core trainer still fire, but the logger has no handler and does not
   propagate, so no JSONL is written. To get it, call `init_structured_logger` yourself or set
   up a handler.
2. Every emitter returns early under `torch.compiler.is_compiling()`, so nothing is recorded
   inside compiled regions.
3. A span measures host wall time with `timeit.default_timer`. On a backend that dispatches
   work asynchronously, such as TorchTPU's deferred execution, a `fwd_bwd` span can measure
   dispatch time rather than device time unless something inside the span synchronizes. Use
   the profiler for device timing (see [profiling-and-tooling](../concepts/profiling-and-tooling.md)).

## Connections
- Code concepts: [metrics-and-observability](../concepts/metrics-and-observability.md) (the
  deep page on this subsystem), [profiling-and-tooling](../concepts/profiling-and-tooling.md),
  [training-loop](../concepts/training-loop.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [checkpointing](../concepts/checkpointing.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`observability/structured_logger/*`).
- Related doc-concepts: [rl-metrics](rl-metrics.md).

## Source
Extracted from `torchtitan/observability/structured_logger/README.md` (kept in place).
