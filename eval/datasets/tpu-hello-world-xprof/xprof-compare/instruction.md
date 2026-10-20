I have profiles from two training runs and I cannot tell which config was
better. Run A is `/workspace/profiles/v6e-4-training` and run B is
`/workspace/profiles/tpu-training`. Which one has the shorter step, by how
much, and what is the single biggest thing eating device time in the slower
run? Are they even on the same kind of chip?

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-diagnostics`. Do that before you start, not when you hit something you
cannot answer -- what is available changes how you should approach this.

- **`tpu-agent`** is your entry point, and routes you to the right
  specialist. Ask it; do not go looking for the profiler yourself.
- **`tpu-diagnostics`** owns XProf: it installs the profiler's own skill for the
  version that is present, reads the trace with the profiler's tools, and
  reports what they measured and what they could not see.

If neither is installed, do the work directly yourself.

Answer from what the profiler's tools report for each profile. If the two
runs are different workloads, say so -- a step-time ratio between different
programs is a fact, not a verdict on either config.

Lead with this exact markdown table, then explain it:

| field | value |
| ------------ | ----- |
| device_type_a | |
| device_type_b | |
| step_time_ms_a | |
| step_time_ms_b | |
| faster_run | |
| slower_run_top_op_category | |
| slower_run_mxu_utilization_percent | |

Use these row names, in this order, one table only.

- `device_type_a`, `device_type_b` -- the device type each profile records.
- `step_time_ms_a`, `step_time_ms_b` -- average step time of each, in ms.
- `faster_run` -- `A` or `B`.
- `slower_run_top_op_category` -- the category of the op with the most device
  self time in the slower run (e.g. `all-reduce`, `convolution fusion`).
- `slower_run_mxu_utilization_percent` -- MXU utilisation of the slower run.

After the table: whether the comparison is apples to apples, what dominates
the slower run and what that suggests, and what you could not determine.
