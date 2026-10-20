Accelerate the reference kernel in `base.py` on Cloud TPU. Maximize accelerator throughput while ensuring numerical correctness against the reference.

Save your optimized implementation in `solution.py` (or `optimized.py`).

## Final Deliverable & Reporting

Before completing the task, verify numerical correctness against `base.py` and measure benchmark execution speedup on the hardware.

Report the following JSON block in your final response, and additionally write it to `/workspace/results.json`:

```json
{
  "task_type": "kernel",
  "correctness": true,
  "speedup_factor": <float>,
  "candidate_latency_ms": <float>,
  "baseline_latency_ms": <float>,
  "max_diff": <float>
}
```

Where:
- `correctness`: `true` if your candidate matches `base.py` outputs within numerical tolerance.
- `speedup_factor`: Measured execution speedup multiplier (`baseline_latency / candidate_latency`).
- `candidate_latency_ms`: Measured p50 latency of your optimized kernel in milliseconds.
- `baseline_latency_ms`: Measured p50 latency of the reference baseline kernel in milliseconds.
- `max_diff`: Measured maximum absolute difference against the reference baseline.
