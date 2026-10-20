# Task A5 — Which op dominates, and which line of my code produced it?

`inputs/trace` is an XProf trace of a real transformer training step on TPU v6e. The
program contains hundreds of operations across 242 KB of compiled HLO.

Find the single operation that consumes the most device time, and attribute it back to the
source line that generated it.

## What to write

Write it in **this directory** — the one containing `TASK.md` — as `answer.json`.
A relative path is resolved from here; do not write it to a scratch or temp location.

`answer.json`:

```json
{
  "top_op_name": "",
  "self_time_ms": 0.0,
  "occurrences": 0,
  "source_file_basename": "",
  "source_line": 0,
  "category": ""
}
```

`top_op_name` is the instruction name alone (the last path segment, e.g. `fusion.123`), not
the full `by_program/...` path. `source_file_basename` is the filename without directories.

## Also write

`REPORT.md`: how you ruled out the runner-up, and the total program time you checked
against.

## Rules

- Do not edit or delete anything under `inputs/`.
- Some listings contain both an op and a combined row for that op **and its duplicates**.
  Make sure you know which one you are reading.
