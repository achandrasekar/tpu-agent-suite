---
scripts:
  - scripts/tpu-model-onboarding/run_benchmarks.py
---
# Adaptive Benchmark Execution and Log Collection

Based on the hardware capabilities profiled, the execution will now be handled
by the orchestrator script.

## Step 3: Adaptive Benchmark Execution

1. Benchmark in compile mode: pass `--compile` below. Leave it off only when
    the request asks for eager mode. Do not stop to ask -- this runs inside a
    subagent, which has no user to put a question to.

2. Synchronously execute the orchestrator script `run_benchmarks.py` using `{{runtime}} compute run`.
   Only `--code` reaches the target on its own: `run_benchmarks.py` imports
   `benchmark_tpu.py` from its own directory and reads `models.json`, so both
   are declared with `--input`, and the results file is declared with
   `--output` so it comes back:
```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/run_benchmarks.py}} \
  --timeout 3600 \
  --input <work>/benchmark_tpu.py:benchmark_tpu.py \
  --input <models_json_path>:models.json \
  --output results.json:<work>/results.json \
  -- --models_json models.json --output_json results.json [--compile]
```

**NOTE**: The orchestrator handles all caching and fallback configurations
internally based on the provided JSON schema.

## Step 4: Collect Benchmarking Log Payload

1. Capture the complete stdout and stderr payload outputs from the remote execution into `<work>/benchmark.log`.
2. `results.json` is already in `<work>` -- `--output` above
    copied it back when the run finished. A run that exits 0 without producing
    it is reported as a failure by the runtime, so do not go looking for it on
    the target by hand.

## Step 5: Host Cleanup

1. Verify that no residue or temporary scripts remain on the target compute
   to prevent contamination of subsequent benchmarking sessions.
