---
references:
  - references/tpu-sharding/optimization_memory.md
  - references/tpu-sharding/knob_wiki.md
---
# tpu-sharding Agent General Rules & Guidelines (`tpu-sharding.md`)

This file defines authoritative behavioral, execution, and writing rules for all subagents in the tpu-sharding ecosystem.

---

## 1. Writing, Reporting & Communication Style

* **Precise, Scientific & Objective Tone**: All generated reports (`report_<N>.md`, `state.json`, `optimization_memory.md`, takeaways, hypotheses, and status notifications) must use formal, objective, and scientific language.
* **Zero Filler or Marketing Terms**: Do not use subjective adjectives, buzzwords, fluff, or emotional language (e.g., avoid terms like "blazing fast", "massive improvement", "incredible result").
* **Empirical Data Focus**: State raw numerical values, exact performance metrics (e.g., TFLOPS/s/device, TTFT ms, TPOT ms, MFU %, step time), error stack traces, and technical root causes strictly as empirical facts.
* **Conciseness & Density**: Keep descriptions dense with technical details and free of unnecessary conversational text.

---

## 2. General Agent Execution Rules

* **Authoritative Rulebook**: All agents must read and consult this file before executing task workflows, generating files, or updating shared ledgers.
* **Knowledge Preservation**: Never overwrite past experiment trajectory logs or sub-run report histories unless explicitly instructed. Always append or maintain cumulative logs.
* **Physical Execution Integrity**: If a physical run fails or crashes, explicitly report the failure. DO NOT substitute, backfill, or replace the failed run's metrics with simulated or mock data.
* **Inaccessible Link Notification**: When the user provides a link for your reference, if you cannot open the link or do not have access to it, explicitly inform the user about this.

---

## 3. Directory Naming & `report_<N>.md` Standardized Schema

### 3.1 Artifact Root & Run Identifiers

Two distinct identifiers are in play. Do not conflate them:

* **`<run_id>`**: The session identifier, assigned at session start. It arrives in your task request and is shared by every tpu-sharding agent for the whole session. You never generate this yourself.
* **`$N$` (iteration)**: The local, zero-based counter for a single experiment within the session (`0` is the baseline). It names the per-experiment subdirectory.

All tpu-sharding output is written under the run's artifact root, referred to throughout as `<artifacts>`:

```
<artifacts> = {{runs}}/<run_id>/artifacts
```

Every experiment $N$ creates a simple numbered folder under `<artifacts>/`:
```
{{runs}}/<run_id>/artifacts/
├── state.json              # Master experiment ledger (conforms to {{schema:tpu-sharding-state.json}})
├── optimization_memory.md  # This run's copy, seeded from the packaged catalog
├── 0/                      # Baseline experiment 0
│   ├── recipe_0.<ext>
│   └── report_0.md
├── 1/                      # Experiment 1
│   ├── recipe_1.<ext>
│   └── report_1.md
└── 2/                      # Experiment 2
    ├── recipe_2.<ext>
    └── report_2.md
```

Never write experiment output anywhere else — in particular, never inside the installed package tree (`{{package}}`), which is read-only shipped content.

This includes the optimization memory. The catalog shipped at
`{{reference:tpu-sharding/optimization_memory.md}}` is a seed,
never a write target: every `query_tpu_sharding.py` invocation must pass
`--memory-file <artifacts>/optimization_memory.md`, which is copied from the
packaged catalog on first use. The knob wiki
(`{{reference:tpu-sharding/knob_wiki.md}}`) is read-only and is
always read from the package.

The seed `{{reference:tpu-sharding/optimization_memory.md}}` is simply meant to provide an initialization for `<artifacts>/optimization_memory.md`, which is what agents should actually access/modify during experiments. We refer to this copy of optimization memory, meant to be read from and written to during experiments, as `<memory>`:

```
<memory> = <artifacts>/optimization_memory.md
```

### 3.2 Standardized `report_<N>.md` Template
Every run's `report_<N>.md` file MUST adhere to this exact structure:

```markdown
# Run Report: Run <N> - <job_id> (Final Status: <SUCCESS | FAIL | TIMEOUT | PARTIAL_SUCCESS>)
**Run Directory Link**: [<N>/report_<N>.md](<artifacts>/<N>/report_<N>.md)
**Executed Recipe Link**: [recipe_<N>.<ext>](<artifacts>/<N>/recipe_<N>.<ext>)
**Backend**: <backend>
**Job ID**: <job_id>
**Total Running Time / Execution Duration**: <time_elapsed>
**Profiler Trace Path**: <path_to_profile_run_dir>
**XProf Session ID**: <session_id_timestamp_dir>

## Target Metric
- **<target_metric_name>**: <target_metric_value>

## Performance Metrics (Stable Steps Average)
### For Training Workloads:
- **Step Time (s)**: <step_time_value>
- **MFU (%)**: <mfu_value>%
- **TFLOPS/s/device**: <tflops_value>
- **Token Throughput (tok/s/dev)**: <tokens_per_sec_device_value>
- **Training Loss / Perplexity**: <loss_value>

### For Inference Workloads:
- **TTFT (ms)**: P50: <val> ms, P99: <val> ms
- **TPOT (ms)**: P50: <val> ms, P99: <val> ms
- **ITL (ms)**: P50: <val> ms, P99: <val> ms
- **E2E Latency (ms)**: P50: <val> ms, P99: <val> ms
- **Request Throughput (req/s)**: <req_per_sec_value>
- **Output Token Throughput (tokens/s)**: <output_tokens_per_sec_value>
- **KV-Cache Usage (%)**: <kv_cache_usage_pct>%

## XProf Analysis
Written by the package's XProf analyst, not by the Executor: the Orchestrator
hands it this run's trace after the run completes. Link it here rather than
copying numbers out of it.
- **Report**: [tpu_profiling_report.md](<artifacts>/<N>/tpu_profiling_report.md) — the full
  analysis: top 30 HLO ops by device time, what the tools could not see, and
  ranked levers.
- **KPI summary**, copied from that report so the Optimizer can compare
  candidate runs by scanning `report_<N>.md` alone, and only open the full
  report for the run it decides to branch from. `N/A` for anything the tools
  did not return — never an estimate.

| Duty cycle % | MXU util % | Roofline util % | Peak HBM (GiB) | TC idle (ms) | SC idle (ms) | Infeed (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<duty_cycle_percent>` | `<mxu_utilization_percent>` | `<roofline_utilization>` | `<peak_hbm_gib>` | `<tc_idle_ms_average>` | `<sc_idle_ms_average>` | `<tc_infeed_ms_average>` |

- If the run produced no trace, the Orchestrator dispatches no analyst and
  neither the report nor the table exists. Write `Not profiled: <reason>` in
  place of both — a dangling link reads as a lost report rather than a run
  that was never profiled.

## Sub-run Attempt History
### Attempt N.0 (Status: <Status>)
- **Recipe Link**: [recipe_<N>.<ext>](<artifacts>/<N>/recipe_<N>.<ext>)
- **Execution Duration**: <time_elapsed>
- **Error Log / Action taken**: `<details>`

## Optimizer Hypothesis & Detailed Analysis
- **Hypothesis**: <Hypothesis written by Optimizer prior to dispatch>
- **Detailed Takeaways**: <In-depth empirical analysis written by Optimizer post-run>
```

**Sub-run Attempt History**: carries one `### Attempt N.<k>` block per attempt.
Attempt `N.0` is the original launch and links the unsuffixed
`recipe_<N>.<ext>`; each self-healing relaunch `N.1`, `N.2`, … links its own
`recipe_<N>.<k>.<ext>`. Append attempts — do not overwrite an earlier one.

This template is the single source of truth for `report_<N>.md`. Agents that
write the file (Orchestrator, Optimizer, Executor) must render
this structure rather than restating their own copy of it, so the schema cannot
drift between them.
