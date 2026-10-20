# Autotune summary

The kernel-worker reads this at the end of Phase 4. Your goal is to summarize the autotuning results provided below, report the best
configuration and latency, and verify if the best configuration was applied if
the status was success.

Autotuning Results: {autotune_results}

Check the status of the autotuning results:

### Case 1: If the status is "success"

You must:

1.  Extract the `"best_config"` and `"best_time_ms"` from the results above.
2.  Verify that the best configuration was applied correctly to the kernel code
    by reading the file located at {optimized_kernel_path} using the `read`
    tool.
3.  Write a clear summary into your report. Do NOT list all tested
    configurations from `all_results`.

### Case 2: If the status is "failed" or "error"

You must:

1.  Report the error message.

In all cases, you must: Write a clear summary into your report. Do NOT list
all tested configurations from `all_results`.

Please use the following format for your summary:

### Autotuning Results

-   **Status**: [Success / Failed]
-   **Best Configuration**: `[JSON or description of best config]`
-   **Latency**: `[Time]` ms
-   **Applied to File**: [Yes / No]

### Output Requirement

You **must** use the `write` tool to save your autotuning summary report (including status, best configuration, latency, and verification of application) to the exact path provided in `{autotune_summary_path}`.

The `"success"` / `"failed"` values above belong to the `autotune_results`
file — they are the sweep's own record, written by the client script. Do not
rename or reinterpret them.

When the sweep failed, report no latency anywhere, only an account of why.
There was no best time, and reporting one would be inventing a measurement.

Once `{autotune_summary_path}` is written, continue to the kernel-worker's Phase 4.5.
