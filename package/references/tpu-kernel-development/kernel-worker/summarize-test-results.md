# Test results summary

The kernel-worker reads this at the end of Phase 3. Analyze the test execution
results `{test_results}` you just captured and write a comprehensive summary
with actionable recommendations.

**TPU VM Execution Requirement**: The results must come from the Phase 3
execution on the TPU VM. Summarize what that run printed; do not run anything
again to fill a gap.

## Test Results

{test_results}

## Your Task

Analyze these test results and provide a comprehensive report with the following
sections:

### 1. Overall Status

-   Clear statement: Did all tests pass, or were there failures?
-   Quick overview: compilation status, correctness status, performance status

### 2. Test Breakdown

Provide detailed analysis for each test category:

**Compilation Tests:**

-   Did the kernels compile successfully?
-   Were there any compilation errors or warnings?

**Correctness Tests:**

-   Did the optimized kernel produce correct results?
-   Were there numerical accuracy issues (tolerance problems)?
-   Did outputs match the baseline across different input sizes?

**Performance Tests:**

-   What was the performance comparison between base and optimized kernels?
-   Was there a speedup? How much?
-   Did performance meet expectations?

### 3. Detailed Error Analysis

If any test failed:

-   Include the **FULL traceback** and error message
-   Identify the root cause of the failure
-   Explain what the error means in plain language

### 4. Recommendations

Based on the test results, provide **specific, actionable recommendations** for
next steps.

**Recommendation Guidelines:**

-   If tests **passed**: Suggest next steps (profiling for bottlenecks, testing
    with more input sizes, production deployment considerations)
-   If **compilation failed**: Provide specific fixes based on the error (API
    signature issues, import problems, syntax errors)
-   If **correctness failed**: Suggest debugging approaches (check block
    boundaries, verify reduction operations, inspect memory access patterns,
    adjust tolerances)
-   If **performance is poor**: Suggest optimization opportunities (block size
    tuning, memory layout optimization, pipelining, prefetching)

**Important**:

-   Provide code examples or specific changes when possible
-   Prioritize recommendations by impact and ease of implementation

### Output Format

Structure the report as:

```
## Test Summary

[Overall status and quick overview]

## Detailed Results

### Compilation
[Compilation test results]

### Correctness
[Correctness test results]

### Performance
[Performance test results]

## Error Analysis

[If failures occurred, full tracebacks and explanations]

## Recommendations

[Numbered list of specific, actionable recommendations with code examples where applicable]
```

Provide a clear, actionable summary that helps the user understand what happened
and what to do next.

## Where the report goes

Append the report to `<work>/tpu_kernel_debug_history.md` under this
iteration's heading. Then return to Phase 3 step 4 of the kernel-worker: proceed to
Phase 4 when the kernel compiled and was numerically correct, otherwise to
Phase 5 with `test_ok` false.
