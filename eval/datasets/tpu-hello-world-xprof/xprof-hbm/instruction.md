Can you give me a breakdown of HBM for this profile:
`/workspace/profiles/v6e-4-training`? I want to know how much HBM the chip has,
how much this run peaked at, roughly how that peak splits between the stack and
the heap, which program is holding it, and whether HBM is the thing limiting
the run.

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-diagnostics`. Do that before you start, not when you hit something you
cannot answer -- what is available changes how you should approach this.

- **`tpu-agent`** is your entry point, and routes you to the right
  specialist. Ask it; do not go looking for the profiler yourself.
- **`tpu-diagnostics`** owns XProf: it installs the profiler's own skill for the
  version that is present, reads the trace with the profiler's tools, and
  reports what they measured and what they could not see.

If neither is installed, do the work directly yourself.

Answer from what the profiler's memory tools report. Do not convert units the
tools did not use, and put `N/A` in any row a tool cannot fill.

Lead with this exact markdown table, then explain it:

| field | value |
| ------------ | ----- |
| hbm_capacity_gib | |
| peak_hbm_gib | |
| hbm_utilization_percent | |
| stack_reservation_gib | |
| heap_allocation_gib | |
| top_hbm_module | |
| bound_by | |

Use these row names, in this order, one table only.

- `hbm_capacity_gib` -- device memory capacity in GiB.
- `peak_hbm_gib` -- peak memory usage in GiB.
- `hbm_utilization_percent` -- peak usage as a percent of capacity.
- `stack_reservation_gib`, `heap_allocation_gib` -- the two halves of the peak.
- `top_hbm_module` -- the HLO module holding the most HBM at peak, named as
  the tools name it.
- `bound_by` -- what the profile says bounds the workload.

After the table: what the split means, whether there is headroom, and what
you would change first if there were not.
