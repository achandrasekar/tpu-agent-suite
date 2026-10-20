# OOM Forensics — Headroom Projection and Memory Mitigation

You are provided with an XProf profile trace of a model training run at batch size 4 in `inputs/trace`.

When the team attempts to scale this workload to batch size 8 on the same hardware accelerator ($31.25\,\text{GiB}$ High Bandwidth Memory capacity), the job crashes immediately with an **Out-Of-Memory (OOM)** error.

Your goal is to inspect the memory profile, identify the peak memory consumption, determine if doubling batch size fits within the $31.25\,\text{GiB}$ limit, and recommend the correct engineering mitigation.

## What to do

1. Inspect `inputs/trace` using memory diagnostic commands:
   - `xprof get_memory_profile trace --logdir=inputs`
   - `xprof get_peak_allocations trace --logdir=inputs`
2. Extract the peak memory usage at batch size 4 and compute the projected peak memory usage at $2\times$ batch size.
3. Determine whether batch size 8 fits in $31.25\,\text{GiB}$ ($32,000\,\text{MiB}$).
4. Prescribe the appropriate mitigation technique from `["remat", "donation", "sharding", "none"]`.
5. Write your findings to `answer.json` matching `answer.schema.json`.

## Evaluation Criteria

- Correct extraction of peak memory allocation.
- Accurate projection of scaled memory usage.
- Proper recommendation of rematerialization (`"remat"`) vs argument donation.
