# Task A2 — Read the roofline, and prove the model closes

`inputs/trace` is an XProf trace of a real transformer training step on TPU v6e.

State what bounds the workload, and demonstrate the roofline model is self-consistent by
**recomputing the ridge point from the published hardware constants** and comparing it to
the value the tool reports.

## What to write

Write it in **this directory** — the one containing `TASK.md` — as `answer.json`.
A relative path is resolved from here; do not write it to a scratch or temp location.

`answer.json`:

```json
{
  "bound_by": "<HBM | Compute | Unknown>",
  "operational_intensity": 0.0,
  "peak_flop_rate_gflops": 0.0,
  "peak_hbm_bw_gibs": 0.0,
  "ridge_point_published": 0.0,
  "ridge_point_recomputed": 0.0,
  "is_above_ridge": true
}
```

`ridge_point_recomputed` must be **your own arithmetic** from the two peak constants — not
the published field copied across. `is_above_ridge` is whether this workload's operational
intensity sits above the ridge.

## Also write

`REPORT.md`: show the recomputation, including units.

## Rules

- Do not edit or delete anything under `inputs/`.
- If the two ridge-point values disagree, do not average them or pick one — work out why.
