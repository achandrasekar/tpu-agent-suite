# Execution Architecture & Verification Protocol

How physical candidate optimization arms are executed, verified, and recorded.

## 1. Execution Architecture & Specialist Routing

A round (`tpu-performance-optimization`) is an ephemeral planner: it profiles the workload, decomposes bottlenecks, ranks addressable headroom, records candidate arms, and exits. Bottleneck kinds are defined once in `study-round.json` (`study.BOTTLENECK_KINDS`). Candidate execution follows a hybrid model:
1. **Specialist Blackbox Delegation**: Domain-specific arms owned by a dedicated specialist (`tpu-kernel-development` or `tpu-sharding`) are formulated as typed candidate briefs. The specialist executes its own optimization loop as a blackbox without polluting the planner's context.
2. **Generic Execution Agent (`tpu-performance-optimization-executor`)**: Unowned optimization kinds (see `study-round.json`) or parallel validation arms that explore alternatives outside established specialist boundaries are executed directly by `tpu-performance-optimization-executor`.
3. **Bounded Planning Context**: The planning agent never ingests raw compiler stderr, runtime logs, or large benchmark outputs.
4. **Reproducible Arm Telemetry**: Every arm has an isolated execution record with durable artifacts.
5. **Model Architecture & Engine Classification**:
   - **Native `vllm-torchtpu` Supported Models** (listed only in `VLLM_TORCHTPU_MODEL_SUPPORT.md`): run on the `vllm-torchtpu` engine (continuous batching, paged KV-cache, Pallas attention kernels, engine knobs).
   - **Other Models (Unsupported by `vllm-torchtpu`)**: Run as native PyTorch / `torch_tpu` without an inference engine and fall under `NATIVE_PYTORCH_PERFORMANCE.md` (5-phase ladder: static KV-cache, 128-byte tile padding alignment, compiler flags, op vectorization, and off-the-shelf kernels/`tpu-kernel-development` handoff).

## 2. Staged Accuracy & Microbenchmark Hierarchy

Before dispatching full cluster runs across dozens of accelerator chips, `tpu-performance-optimization-executor` follows a strict four-stage triage. A failure at any stage (syntax, memory bound, crash, compile error, OOM, accuracy mismatch, failed E2E gate) is a defect in the executor's implementation, not evidence against the hypothesis: the executor diagnoses it, fixes the implementation, and re-runs from the failed stage (the Repair Loop in `tpu-performance-optimization-executor`, bounded at 3 attempts per root cause and 6 per dispatch). Changing the tests, benchmark, evaluation harness, or gate thresholds is forbidden, and each re-run must follow an implementation change aimed at a named defect, never a bare re-roll of a noisy eval. The only early exit is an implementation that passes accuracy but shows no local gain.

```
[Candidate Arm: Evaluate Hypothesis & Inspect Baseline Recipe]
       │
       ▼
┌────────────────────────────────────────┐
│ Stage 1: Implement & Static Check      │ ── Fail ──> [Repair Loop]
│ (Apply Change, SRAM & HBM, Imports)    │
└──────────────────┬─────────────────────┘
                   │ Pass
                   ▼
┌────────────────────────────────────────┐
│ Stage 2: Accuracy & Numerical Check    │ ── Fail ──> [Repair Loop]
│ (Reference Parity, No NaN/Inf, Gates)  │
└──────────────────┬─────────────────────┘
                   │ Accuracy Passed
                   ▼
┌────────────────────────────────────────┐
│ Stage 3: Component Microbenchmark      │ ── Regr ──> [Loss/Tie: No Local Gain]
│ (Build/Run Targeted Bench, Prove Gain) │ ── Crash ─> [Repair Loop]
└──────────────────┬─────────────────────┘
                   │ Local Gain Proven
                   ▼
┌────────────────────────────────────────┐
│ Stage 4: E2E Cluster Execution         │ ── Crash / Gate Fail ──> [Repair Loop]
│                                        │ ── Done ──> [Measured Verdict (Win/Tie/Loss/Invalid)]
│ (Relaunch Baseline Recipe, N Replicate)│
└────────────────────────────────────────┘
```

1. **Stage 1: Implement Changes & Static / Memory Hierarchy Check**
   - Inspect the baseline job's launch command and recipe (`baseline.execution_id`), then apply the candidate code or config changes.
   - Check static tile dimensions, array padding alignment, and AST syntax/imports (`arm_microbench.py`).
   - Apply guided preflight review and optional static helper checks (canonical invocation: `arm_microbench.py --file <script> --gotchas [--tpu-version <v5e|v5p|v6e|v7x>] [--config <cfg.json>] [--run "<cmd>"]`): eliminate unguarded blocking device-to-host synchronization (`.item()`, `.cpu()`) on the hot loop path (including 1-level helper calls; `print`/`logging` calls are excluded), verify MXU lane and sublane dimension alignment to prevent XLA dynamic shape recompilation storms, and verify per-generation SparseCore SRAM and Core VMEM ceilings:

     | TPU Generation | MXU Lane Dim | MXU Sublane Dim | Core VMEM Ceiling | SparseCore SRAM Ceiling |
     | :--- | :--- | :--- | :--- | :--- |
     | `v5e` | 128 | 8 | 16.0 MiB | 128 KiB |
     | `v5p` | 128 | 8 | 32.0 MiB | 128 KiB |
     | `v6e` (`trillium`) | 256 | 8 | 32.0 MiB | 128 KiB |
     | `v7x` (`ironwood`, default) | 128 | 8 | 6.06 MiB | 128 KiB |
   - Verify memory footprint across the full **TPU Memory Hierarchy**: both **on-chip SRAM (TensorCore VMEM & SparseCore scratchpad spills)** and **device HBM (weights, KV cache, activations, and collective buffers)**.
   - If static preconditions fail, fix the implementation before reserving hardware.

2. **Stage 2: Accuracy & Numerical Parity Check**
   - Verify numerical parity against unoptimized reference outputs (e.g., `torch.testing.assert_close`, cosine similarity, SQNR, zero NaN/Inf, and protocol accuracy gates). This precedes the microbenchmark because a timing from an incorrect implementation measures nothing.
   - If accuracy or numerical tolerances fail, diagnose and fix the implementation; do not run benchmarks until it passes.

3. **Stage 3: Build & Run Targeted Component Microbenchmark (Prove Gain)**
   - Build or select a component microbenchmark tailored to the arm's `kind` (e.g. single-op kernel harness, multi-device collective microbenchmark, or scheduler step-loop trace) and execute it via `arm_microbench.py` or a minimal compute slice.
   - Prove that the change delivers a local speedup on the targeted bottleneck and does not trigger XLA compilation explosions or OOMs. If it crashes, fails to compile, or OOMs, fix the implementation. If the accuracy-verified implementation shows a clear local regression with no plausible E2E mechanism, return early: `loss` if regression exceeds the negative noise floor (`delta_pct < -noise_floor_pct`), or `tie` if within the noise floor (`abs(delta_pct) <= noise_floor_pct`).

4. **Stage 4: Full E2E Measurement on Target Cell**
   - Relaunch the full E2E workload on `protocol.measurement_cell` using the baseline recipe configuration modified only by the candidate arm.
   - For a served model, measure it with the baseline's exact `serving_benchmark.py` invocation (`SERVING_BENCHMARK.md`), which reports TTFT, TPOT, ITL, end-to-end latency and throughput under the names the protocol uses, and check accuracy on the same running server with `serving_eval.py` (an lm-eval score such as MMLU, gated against the baseline's).
   - If the run crashes or fails a gate, diagnose and fix the implementation, re-pass Stage 2, and relaunch; never change the tests, eval harness, or gate thresholds.
   - Run across at least `protocol.replicate_count` replicates and capture device profile traces, analysed by `tpu-diagnostics` through a `profiling-request` (`{{schema:profiling-request.json}}`).

## 3. Co-Located Same-Execution Guarantee

A common pathology in automated optimization is "split-run reporting": measuring throughput on an aggressive batch size that degenerates model quality, while reporting loss or accuracy from an unoptimized run.

To eliminate this:
* **One `execution_id` Rule**: The primary metric and every declared gate marked `same_execution: true` in the study protocol **MUST share the exact same `execution_id`**.
* The runtime (`record-round`) mechanically rejects any round where an admissibility gate was measured on a divergent job ID.

## 4. Verdict Decision Matrix

| Verdict | Condition | Meaning |
| :--- | :--- | :--- |
| **`win`** | Passes all declared admissibility gates, AND performance delta exceeds `protocol.noise_floor_pct` at $N \ge \text{replicate\_count}$. | Proven improvement clearing the noise floor. |
| **`tie`** | Passes all admissibility gates, but performance delta is within the noise floor margin ($[-\text{noise\_floor\_pct}, +\text{noise\_floor\_pct}]$), OR actual replicates $< \text{replicate\_count}$. | Inconclusive or inside noise margin; unconfirmed. |
| **`loss`** | Passes all admissibility gates, but performance regressed below the negative noise floor ($< -\text{noise\_floor\_pct}$). | Genuine performance regression on target cell. |
| **`invalid`** | Hardware execution aborted, crashed with OOM, corrupted telemetry, divergent/unproven `same_execution` gate, OR ANY admissibility gate failed (`passed == false`) on the final, repaired implementation. | Inadmissible configuration or untrustworthy measurement; cannot count. |

An implementation defect the executor could not repair within its budget is not a verdict: the result carries `status: "failed"` and no `verdict`, the round releases the arm with `study release-arm --failed`, which counts it in the arm's `failed_dispatches` and reopens the arm, or blocks it on its second failed dispatch.
