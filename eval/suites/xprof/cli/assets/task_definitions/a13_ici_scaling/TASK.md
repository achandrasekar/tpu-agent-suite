# Interconnect Scaling — Multi-Device ICI Boundness Analysis

You are given an XProf trace of a multi-core training workload running across a multi-device TPU pod in `inputs/trace`.

Your objective is to determine whether the multi-device scaling is bound by computation or inter-chip interconnect (ICI) collective communication.

## What to do

1. Inspect `inputs/trace` using `xprof check_host_boundness trace --logdir=inputs` and `xprof get_overview trace --logdir=inputs`.
2. Extract the decomposed scaled time components:
   - `scaled_ici_time_ms`
   - `scaled_compute_time_ms`
   - `scaled_hbm_time_ms`
3. Compute the ratio of interconnect time to compute time (`ici_to_compute_ratio = scaled_ici_time_ms / scaled_compute_time_ms`).
4. Identify the dominant limiting factor (`"interconnect"` vs `"compute"` vs `"hbm"`) and prescribe the appropriate scaling remedy (`"sharding"`, `"pipelining"`, `"fusion"`, `"batching"`).
5. Write your findings to `answer.json` matching `answer.schema.json`.

## Evaluation Criteria

- Correct extraction of scaled time values within $\pm 1\%$.
- Accurate identification of the communication bottleneck ($>1.5\times$ compute time).
