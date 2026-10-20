---
title: "RL metrics: typed Metric records reduced once per step"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/observability/metrics/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# RL metrics: typed Metric records reduced once per step

## Definition
The RL metrics README describes a small, typed metrics pipeline. Actors and the controller emit
`Metric(key, value)` records, where the value's type decides how it is reduced (`Mean`, `Max`,
`Min`, `Sum`, `Std`, `SummaryStats`, `NoReduce`). Nothing is reduced until
`MetricsProcessor.log(step, metrics)` is called. At that point the records are grouped and
reduced into one flat `dict[str, float]`. The full dict goes to every backend (W&B,
TensorBoard), and an allow-list selects which keys are printed to the console. vLLM engine-level
metrics are a separate pipeline: an OpenTelemetry stat logger inside each generator (see
[vLLM engine metrics](#vllm-engine-metrics-opentelemetry) below).

## In torchtitan (grounded)
- **Records.** [`Metric`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L89 "torchtitan/experiments/rl/observability/metrics/types#Metric")
  pairs a key with a [`MetricValue`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L30 "torchtitan/experiments/rl/observability/metrics/types#MetricValue").
  A `MetricValue` provides `output_suffix`, `from_list` and `reduce`. The concrete value types
  are [`Mean`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L108 "torchtitan/experiments/rl/observability/metrics/types#Mean"),
  [`Max`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L155 "torchtitan/experiments/rl/observability/metrics/types#Max"),
  [`Min`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L180 "torchtitan/experiments/rl/observability/metrics/types#Min"),
  [`Sum`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L203 "torchtitan/experiments/rl/observability/metrics/types#Sum"),
  [`Std`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L220 "torchtitan/experiments/rl/observability/metrics/types#Std"),
  [`SummaryStats`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L282 "torchtitan/experiments/rl/observability/metrics/types#SummaryStats")
  and [`NoReduce`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/types.py#L357 "torchtitan/experiments/rl/observability/metrics/types#NoReduce").
  `SummaryStats` emits `_max`, `_mean`, `_min`, `_std` and `_sum`. The leading underscore keeps
  these from colliding with plain `Mean`/`Max` records under the same key.
- **Reduction.** [`MetricsProcessor._aggregate_metrics`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/processor.py#L121 "torchtitan/experiments/rl/observability/metrics/processor#MetricsProcessor._aggregate_metrics")
  groups records by `(key, type(value))` and calls `reduce` on each group, which produces
  `key/suffix` names. A NaN result is dropped with a one-time warning, and two reductions that
  produce the same output key raise `ValueError`.
- **Dispatch.** [`MetricsProcessor.log`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/processor.py#L165 "torchtitan/experiments/rl/observability/metrics/processor#MetricsProcessor.log")
  prints through [`log_to_console`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/console.py#L96 "torchtitan/experiments/rl/observability/metrics/console#log_to_console")
  and then sends the full dict to each backend. A backend exception is logged, not raised.
  [`MetricsProcessor.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/processor.py#L51 "torchtitan/experiments/rl/observability/metrics/processor#MetricsProcessor.Config")
  holds the train/validation allow-lists, `enable_wandb`, `enable_tensorboard` (both need
  `log_dir`) and `wandb_project` (default `titan_rl`, applied through `setdefault`, so an
  existing `WANDB_PROJECT` wins).
- **Allow-list semantics.** The README calls `console_log_keys_train` an allow list of keys. In
  the code, [`_filter_allow_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/metrics/console.py#L38 "torchtitan/experiments/rl/observability/metrics/console#_filter_allow_list")
  compiles each entry as a **regex** and uses `re.search`. So `"loss"` matches every key that
  contains `loss`, and the default `"perf/"` entry pulls in a whole panel of keys. `None`
  prints every key and `[]` prints none. The shipped default is a curated list (perf panel,
  generator latencies, loss, reward, entropy, grad norm, lr, `bit_wise/logprob_diff/max`). It
  is not empty, as the README's short example might suggest.
- This processor is separate from the pretraining
  [`MetricsProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/metrics.py#L265 "torchtitan/components/metrics#MetricsProcessor").
  A TODO in the code says to unify the two.

## vLLM engine metrics (OpenTelemetry)
- **Logger and gate.** [`VllmOtelStatLogger`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L148 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger")
  is a vLLM `StatLoggerBase` whose metric selection follows vLLM's Prometheus logger. It is off by
  default: [`VLLMGenerator.Config.vllm_stat_logger`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/generator.py#L766 "torchtitan/experiments/rl/actors/generator#VLLMGenerator.Config.vllm_stat_logger")
  defaults to `None`, and only when it holds a
  [`VllmOtelStatLogger.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L155 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger.Config")
  does the generator build the logger, on TP rank 0 of each DP replica only (all TP ranks see the
  same engine aggregates). Once enabled, `OTEL_METRICS_EXPORTER` must be `jsonl` or `otlp`,
  otherwise the constructor raises; this explicit gate keeps inherited OpenTelemetry env vars from
  switching it on by accident. `OTEL_SDK_DISABLED=true` turns it off with a warning.
- **Instruments.** [`_create_counters`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L284 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger._create_counters"):
  cumulative counters `vllm.generation_tokens`, `vllm.prompt_tokens`, `vllm.cached_prompt_tokens`,
  `vllm.preempted_requests`, `vllm.finished_requests`, `vllm.prefix_cache_queries`,
  `vllm.prefix_cache_hits`.
  [`_create_histograms`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L303 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger._create_histograms"):
  millisecond histograms `vllm.time_to_first_token`, `vllm.inter_token_latency`, `vllm.decode_time`,
  `vllm.queue_time`, `vllm.e2e_latency`.
  [`_create_gauges`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L311 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger._create_gauges"):
  `vllm.kv_cache_usage`, `vllm.num_running_requests`, `vllm.num_waiting_requests` (latest scheduler
  state). [`record`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L331 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger.record")
  is the per-step vLLM callback that feeds them.
- **Derived rates.** Throughput and prefix-cache hit rate are computed in the backend from counter
  rates, never from raw cumulative values: `hit_rate = rate(prefix_cache_hits) /
  rate(prefix_cache_queries)`, summing each counter's rate across generators before dividing, and
  returning no value when the query rate is zero. Rates bound the result to the query window and
  survive counter resets.
- **Exporters.** `jsonl` (local debugging, no collector) writes compact one-line JSON records to
  `{dump_folder}/vllm_metrics/{generator_name}.rank{rank}.jsonl` and requires an output directory.
  `otlp` (distributed/production runs with a collector) uses the OTLP HTTP/protobuf exporter, needs
  `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` (metrics-specific wins), and
  rejects any protocol env var other than `http/protobuf`. Export runs on the OpenTelemetry SDK's
  background thread every `VLLM_LOG_STATS_INTERVAL` seconds (default 10), so vLLM's synchronous
  `log()` callback does no network I/O. Standard OTel variables (`OTEL_METRIC_EXPORT_TIMEOUT`,
  temporality preference, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`) still apply.
- **Resource attributes.** [`_build_resource_attributes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L267 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger._build_resource_attributes")
  attaches `model_name`, `hostname`, `rank`, `local_rank`, `world_size`, `dp_rank`, `tp_rank` and
  `generator_name`;
  [`extra_resource_attributes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/observability/vllm_otel_stat_logger.py#L156 "torchtitan/experiments/rl/observability/vllm_otel_stat_logger#VllmOtelStatLogger.Config.extra_resource_attributes")
  are applied last and override on key collisions.

> [!warning] Doc vs code
> The README's JSONL and OTLP example commands run `--config rl_dapo_qwen3_4b_math_8k` with only the
> `OTEL_*` env vars set. That recipe leaves `generator.vllm_stat_logger` at `None`, and the field is
> marked `tyro.conf.Suppress`, so it cannot be set from the CLI either. As written, the commands log
> "VllmOtelStatLogger inactive because vllm_stat_logger=None" and export nothing; the recipe (or a
> derived config function) must set `vllm_stat_logger=VllmOtelStatLogger.Config()`, as the README's
> own first paragraph on the gate says.

## Why it matters / when it applies
In RL, observations come from many actors at different rates. Emitting typed partial values
(for example `Mean` with a running total and count) lets values from different sources be
merged correctly at log time, with no need to pass pre-averaged dictionaries around. Use
`from_list` when the observations are already in a list, because it creates one record instead
of N. Use `NoReduce` for values that are already reduced, such as the trainer's loss.

**On TPU.** The TPU non-colocated GRPO path uses this pipeline unchanged.
[`RayController`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L123 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayController")
subclasses the RL [`Controller`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L260 "torchtitan/experiments/rl/controller#Controller")
("overriding ONLY setup() to inject Ray"), and that controller builds its `m.MetricsProcessor`
from `config.metrics`. [`RayTPUPolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/trainer.py#L120 "torchtitan/experiments/tpu/rl_nc_ray/trainer#RayTPUPolicyTrainer")
extends [`PolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L48 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer").
Reduction runs on the host over Python floats, so it adds no TPU work. However, any metric an
actor computes from a device tensor needs a host transfer before it can become a `Metric`, so
keep those transfers off the hot path. TPU pretraining recipes use the core
`components.metrics.MetricsProcessor`, not this one.

## Connections
- Code concepts: [metrics-and-observability](../concepts/metrics-and-observability.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [training-loop](../concepts/training-loop.md).
- Module catalogs: [module map](../catalog/index.md) (`experiments/rl/observability/metrics/*`,
  `experiments/rl/observability/vllm_otel_stat_logger`).
- Related doc-concepts: [structured-trace-logging](structured-trace-logging.md),
  [rl-post-training](rl-post-training.md) (the four observability views),
  [dapo-math-rl-recipe](dapo-math-rl-recipe.md) (the recipe the engine-metrics examples run).

## Source
Extracted from `torchtitan/experiments/rl/observability/metrics/README.md` (kept in place).
