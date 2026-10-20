```yaml
id: STRAT-MLA-11
title: Before building a lever that shortens a grid, measure how much of the cost scales with that grid
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [1, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctraversal-final_report.md §1-§4; raw/2026-09-sctraversal-STRATEGY.md § STRAT-TRAV-02
recheck: nothing on the mechanism — re-derive which term of your own cost model the lever attacks
origin: campaign sc_traversal (DSA/MLA gather, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-11 — Localize fixed cost before building a lever against it

**Rule.** "Fixed cost binds" is a diagnosis, not a target. A kernel's time is
`fixed + per_step × steps + per_unit × work`, and a skip, truncation or compaction lever attacks
**only the middle term**. Before building one, sweep the grid extent at **constant live work** and
fit: the slope is the lever's entire ceiling and the intercept is what it can never reach. One
4-arm sweep, and it runs before the design, not after it.

**Mechanism.** A residual identified by subtraction — measured time minus modelled work — is
attributed, not measured, and it lands in the intercept and the slope indistinguishably. The two
imply opposite levers: a per-step cost is removed by shortening a grid, a per-call cost only by
issuing fewer calls, and no amount of grid work touches the second. Holding live work constant while
moving the grid extent is what separates them, because only the per-step term responds. The reason
this is worth a job is that the subtraction is usually done against a *variable-cost* model (bytes,
rows, FLOPs), so its residual is silently assumed to be the thing a grid lever reaches.

**Evidence.** A SparseCore gather residual of ~10 µs/layer had
been attributed to window traversal the skip predicated off but did not remove. Sweeping the token
bucket at constant live work (1 live token, 4 live windows) moved the outer grid 2 → 16 steps:
**`a = 17.06 µs`, `b = 0.280 µs/outer-step`, R² = 0.974** over 17.37 / 18.21 / 19.68 / 21.37 µs. Net
of live-row movement, **77% of the gather was invariant to the grid extent**; traversal was 11.4%,
and per window **8.75 ns against the 41 ns** the attribution implied — **4.7×**. The designed lever's
whole ceiling was **1.96 µs/layer, 4.9%** of the incumbent, so the campaign closed without writing a
variant. The same sweep refuted a sibling inference in passing: shrinking the bucket **did** cut
traversal exactly 4× as predicted, while the quantity anyone would ship against — total gather time —
moved **1.133×**. A confirmed linearity is not a confirmed lever.

**Scope.** General to any kernel with a grid and a candidate skip/truncation/compaction lever, on any
unit. The constants are this gather's; what transfers is the sweep and the reading of its two
coefficients. Sharpens [[STRAT-MLA-07]], whose fixed-cost diagnosis this confirms and whose
localization to traversal it refutes.

**Does not apply when** live work cannot be held constant while the grid extent moves — then the
slope confounds the two terms and the fit is not a ceiling. It also does not argue against
fixed-cost levers: it says a *grid* lever cannot be one. Where the intercept is the target, the
lever is call-count reduction, and [[STRAT-SC-09]] constrains which direction a cross-unit fusion
may go.
