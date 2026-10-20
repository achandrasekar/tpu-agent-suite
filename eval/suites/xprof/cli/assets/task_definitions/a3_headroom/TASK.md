# Task A3 — How much is left on the table?

`inputs/trace` is an XProf trace of a real transformer training step on TPU v6e.

Every other question about a profile asks what is wrong. This one asks for the **ceiling**:
if this workload were optimized as far as it can go without changing the algorithm, how
much faster would it be — and which resource runs out first?

## What to write

Write it in **this directory** — the one containing `TASK.md` — as `answer.json`.
A relative path is resolved from here; do not write it to a scratch or temp location.

`answer.json`:

```json
{
  "measured_flop_rate_gflops": 0.0,
  "optimal_flop_rate_gflops": 0.0,
  "max_speedup_at_current_ai": 0.0,
  "peak_memory_usage_gib": 0.0,
  "memory_capacity_gib": 0.0,
  "hbm_free_gib": 0.0,
  "limiting_resource": "<hbm_bandwidth | compute | memory_capacity | interconnect | host>",
  "would_larger_batch_fit": true
}
```

`max_speedup_at_current_ai` is the achievable rate divided by the measured one, at this
workload's current arithmetic intensity.

## Also write

`REPORT.md`: the two ceilings you considered and why one binds before the other.

## Rules

- Do not edit or delete anything under `inputs/`.
