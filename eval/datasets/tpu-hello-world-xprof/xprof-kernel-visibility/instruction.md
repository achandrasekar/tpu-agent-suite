I want to look inside the custom kernels in this profile:
`/workspace/profiles/v6e-4-training`. Does it actually have kernel-level data --
the LLO instruction analysis and per-kernel stats -- or only the XLA op view?
If it does not, tell me exactly what I would need to set to re-capture it with
that data, and what the current profile can still tell me about those kernels.

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-diagnostics`. Do that before you start, not when you hit something you
cannot answer -- what is available changes how you should approach this.

- **`tpu-agent`** is your entry point, and routes you to the right
  specialist. Ask it; do not go looking for the profiler yourself.
- **`tpu-diagnostics`** owns XProf: it installs the profiler's own skill for the
  version that is present, reads the trace with the profiler's tools, and
  reports what they measured and what they could not see.

If neither is installed, do the work directly yourself.

This is a question about what the profile contains, so answer from what the
profiler's kernel-level tools return when you run them on it -- including their
status when they have nothing to work with. Do not infer presence or absence
from trace line names, and do not recall the flags from memory; take them from
the tool's own output.

Lead with this exact markdown table, then explain it:

| field | value |
| ------------ | ----- |
| llo_data | |
| required_flags | |
| custom_call_flops_counted | |

Use these row names, in this order, one table only.

- `llo_data` -- `present` or `absent`, as the LLO analysis tool reports it.
- `required_flags` -- the exact `LIBTPU_INIT_ARGS` flags a re-capture needs,
  taken from the tool's `required_flags` field; leave out its
  `optional_flags`. `none` if the data is already present.
- `custom_call_flops_counted` -- `yes` if the profile's FLOP and utilisation
  figures include custom-call work, `no` if XLA has no cost model for it and
  those figures exclude it.

After the table: what the op-level view can and cannot say about the kernels,
and what would change once the data is present.
