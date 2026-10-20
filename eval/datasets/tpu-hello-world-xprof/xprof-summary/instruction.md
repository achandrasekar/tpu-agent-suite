Here is an XProf profile from a training run: `/workspace/profiles/v6e-4-training`.
Can you summarise it for me? I want to know what it ran on, how long a step
takes, whether it is compute-bound or memory-bound, and which single op costs
the most device time.

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-diagnostics`. Do that before you start, not when you hit something you
cannot answer -- what is available changes how you should approach this.

- **`tpu-agent`** is your entry point, and routes you to the right
  specialist. Ask it; do not go looking for the profiler yourself.
- **`tpu-diagnostics`** owns XProf: it installs the profiler's own skill for the
  version that is present, reads the trace with the profiler's tools, and
  reports what they measured and what they could not see.

If neither is installed, do the work directly yourself.

Answer from what the profiler's tools report, not from what the numbers "should"
be. If a tool cannot produce a figure, say `N/A` for that row rather than
estimating it.

Lead with this exact markdown table, then explain it:

| field | value |
| ------------ | ----- |
| device_type | |
| device_core_count | |
| step_time_ms | |
| mxu_utilization_percent | |
| bound_by | |
| top_op | |

Use these row names, in this order, one table only.

- `device_type` -- the device type string as the profile records it.
- `device_core_count` -- number of device cores in the profile.
- `step_time_ms` -- average step time in milliseconds.
- `mxu_utilization_percent` -- MXU utilisation, percent.
- `bound_by` -- what the profile says bounds the workload (e.g. `HBM`,
  `Compute`).
- `top_op` -- the op with the most device self time, named as the trace names
  it.

After the table: two or three sentences on where the time goes and what you
would look at first, and anything the profile cannot tell you.
