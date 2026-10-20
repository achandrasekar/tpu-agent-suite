```yaml
id: STRAT-SC-07
title: Distribution sensitivity at constant work is a memory conflict, not a workload
hardware: tpu-v7x
unit: any
software: any
stage: [5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-07; raw/2026-09-sctopk-final_report.md §1
recheck: nothing on the diagnostic — but prove op count is constant across your distributions first
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-07 — Distribution sensitivity at constant work is a memory conflict

**Rule.** If latency moves with the input *distribution* while the operation count is provably
unchanged, stop modelling work and go look for a bank or port conflict. Use the **collapse of the
cross-distribution spread** as the confirming signature that you fixed the right thing, rather than
the headline delta on any one distribution.

**Mechanism.** Conflict severity tracks address *concentration*, which is exactly what changing a
distribution changes while leaving update counts fixed. No work-based account can produce a latency
that depends on values rather than counts, so the observation itself narrows the search to the
memory system.

**Evidence.** The L0 scatter cost **14.83%** of the call on `dsa_like` versus
**7.90%** on `uniform` at identical update counts. After the coprime padding of [[STRAT-SC-06]],
per-arm spread across four distributions went from **+7.87 / +7.68 / +7.00 / +6.83%** to within
**±1.5%**, and the incumbent's **40.0%** worst-vs-best spread
fell to **4.2…6.3%** end to end.

**Scope.** General, on any unit. It is also a **diagnostic-order** rule: a pre-LLO Mosaic dump
exposes no bank counter, so the instrument is a controlled sweep across distributions at fixed work,
not the profiler. Reaching for xprof first costs a job and answers nothing.

**Does not apply when** the operation count is not actually constant across your distributions —
prove that first, and expect to be wrong about it for any kernel with a data-dependent early exit.
If counts move, you have a workload difference and this rule says nothing.
