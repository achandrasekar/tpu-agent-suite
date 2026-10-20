# Task A9 — Three tools, three step times. Which one is the step time?

`inputs/trace` is an XProf trace of a real transformer training step on TPU v6e.

Three tools report step timing for this trace and they disagree — by nearly 5x on the time
and by an order of magnitude on the count. Reconcile them, and say which number answers
"how long does one training step take".

## What to write

Write it in **this directory** — the one containing `TASK.md` — as `answer.json`.
A relative path is resolved from here; do not write it to a scratch or temp location.

`answer.json`:

```json
{
  "overview_step_time_ms": 0.0,
  "avg_step_time_tool_ms": 0.0,
  "avg_step_time_tool_step_count": 0,
  "host_boundness_step_count": 0,
  "step_times_are_numerically_equal": true,
  "avg_step_time_counts": "<training_steps | xla_module_executions | unknown>",
  "training_step_time_ms": 0.0,
  "core_count": 0
}
```

`step_times_are_numerically_equal`: do the tools report the **same number** for step
time? This asks about the values, not about whether the difference is explainable.

`avg_step_time_counts` is what the dedicated step-time tool is actually counting.
`training_step_time_ms` is your answer to the colleague's question.

## Also write

`REPORT.md`: the reconciliation — what each tool counts, and the evidence that led you
there.

## Rules

- Do not edit or delete anything under `inputs/`.
- Do not average the disagreeing values. One of them answers the question; the others
  answer a different one.
