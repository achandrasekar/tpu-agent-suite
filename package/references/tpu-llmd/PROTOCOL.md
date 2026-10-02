---
scripts:
  - scripts/tpu-llmd/evaluate_llmd_arm.py
  - scripts/tpu-llmd/render_protocol.py
---
# The protocol: what an llm-d study measures and how it judges

The study's protocol is rendered once by `{{script:tpu-llmd/render_protocol.py}}`
from the workload spec and the serving recipe, registered at `study start`,
and never edited by a round. This note explains each field so a reader can
tell a correct verdict from a convenient one.

## The primary metric

The user picks one of two in `objective.primary_metric`
(`{{schema:llmd-workload-spec.json}}`):

| metric | unit | direction | definition |
| :--- | :--- | :--- | :--- |
| `goodput_at_slo_per_chip` (default) | req/s/chip | maximize | requests per second that met **every** SLO bound of their tenant, summed over tenants, divided by the release's total chip count |
| `cost_per_1M_tokens_at_slo` | usd/1M-tokens | minimize | `chips_total x budget.chip_hour_cost / (good output tokens per hour / 1e6)`; needs `budget.chip_hour_cost` in the spec |

"Good" is decided per request by inference-perf's goodput constraints
(`ttft`, `tpot`, `itl`, `request_latency`, in seconds), which the renderer
writes into each tenant's bench config from the tenant's SLO
(https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/goodput.md).
The bench report (`{{schema:llmd-bench-report.json}}`) carries the metric with
its per-replicate values so a reader can see the spread, not just the mean.

Per chip, not per replica, because the point of the study is to spend the
user's chip budget well: a configuration that adds a replica and raises raw
throughput by less than the replica's share of chips has made things worse,
and raw throughput would call it a win.

## Admissibility gates

Every gate is `same_execution`: it must be measured by the same `deploy
bench` invocation that produced the primary metric. The renderer emits:

| gate name | metric | bound |
| :--- | :--- | :--- |
| `<tenant>.<slo key>` | the tenant's `ttft_p90_ms`, `tpot_p90_ms`, `e2e_p99_ms`, ... | the bound in the spec, ms |
| `<tenant>.error_rate` | failed / total requests | `availability.error_rate_max` |
| `budget.max_chips` | the release's chip total | `budget.max_chips` when set |

A gate failing makes the measurement **inadmissible** -- `verdict: invalid`
-- whatever the primary metric says. This is the mechanism that keeps a
multi-tenant study honest: raising the interactive tenant's goodput by
starving the batch tenant past its (looser) SLO fails `batch.tpot_p90_ms`
and is not a win.

The scorer (`{{script:tpu-llmd/evaluate_llmd_arm.py}} score`) appends every
protocol gate the report did not measure as `passed: false`, so a bench
run that lost a tenant's report cannot pass by omission.

## Replicates and noise

`replicate_count` (default 3) is the number of complete `slo`-stage runs the
bench performs per measurement, back to back, against the same release.
`noise_floor_pct` (default 5) is the delta the best point must clear. Both
are deliberately conservative: a fleet measurement includes the load
generator's own scheduling on the same cluster, the router's warm-up, and
prefix-cache state carried between replicates. A 3% improvement on a fleet
is indistinguishable from the second replicate having a warmer cache.

## Stopping rule

`max_rounds` (default 12) bounds the study; `plateau_rounds` (default 3)
closes it early after that many consecutive rounds without a win. The
orchestrator branches on the round number `study status` reports, which
counts attempted rounds including crashed ones, because a crashed round
still spent chips.

## The companion file

The protocol schema holds nothing llm-d specific, so `render_protocol.py`
writes `llmd-study.json` beside it with: `inner_sweep_budget` (how many
`reconfigure`+`bench` steps one arm may take, default 8), `max_chips`,
`max_chip_hours`, `chip_hour_cost`, `chips_per_replica`, the timeouts, and a
summary of the workload and recipe. A round reads it; a round never edits it.

## The baseline

The `baseline` arm is the capacity floor: `ceil(target_qps / knee_rps)`
replicas of the recipe behind the standalone router with the
`sticky-until-saturated` profile, inner-swept on the engine's batching flags.
Its `arm-result` becomes `study-baseline.json`:

| field | from |
| :--- | :--- |
| `execution_id` | the result's `execution_id` |
| `metric` | the result's `metric` (name, value, unit, replicates, `measured`) |
| `profile` | the fleet `metrics.json` path in `artifacts` -- the per-pod KV utilisation and queue depth that rank the next arms |
| `fingerprint.commit_sha` | the recipe's image tag; it is what was deployed |
| `fingerprint.config_digest` | the best ledger entry's `config_digest` |
| `fingerprint.cell` | the protocol's `measurement_cell` |

When the recipe has no `measured.knee_rps`, the floor is one replica and
the baseline arm's claim says so; the first measurement then tells the study
how far off that was.
