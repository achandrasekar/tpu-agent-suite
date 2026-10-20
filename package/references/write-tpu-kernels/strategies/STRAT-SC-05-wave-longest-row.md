```yaml
id: STRAT-SC-05
title: A lockstep group costs its longest member, so mean extent does not model ragged work
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [1, 2]
lever: algorithm
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-05; raw/2026-09-sctopk-final_report.md §5; raw/2026-09-sctopk-WHITEBOARD-excerpt.md §7h
recheck: the lockstep width W on the target unit, and whether work is assigned to it by index
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-05 — A lockstep group costs its longest member

**Rule.** Where `W` units execute in lockstep, a group costs the **maximum extent over its `W`
members**, not the mean. Model ragged work as `E[max over W]`. A mean-based or bytes-based cost
model is wrong by the full raggedness factor, always optimistically, so it will manufacture a lever
that cannot exist.

**Mechanism.** Lockstep means every member waits for the slowest, so raggedness that would average
out across independent workers is instead rounded up per group to its worst member. `E[max]` over
even a modest `W` sits close to the distribution's ceiling: for `rl ~ U(0,n)` and `W = 32`,
`E[max over W] ≈ 0.97n`. **Halving the mean extent moves the clock by almost nothing.**

**Evidence.** On v7x SparseCore the wave is **W = 32 vector
subcores** (2 cores × 16) and membership is by index — row `w` of a wave is
`w * (cores * subcores) + flat`. `U(0,n_p)` rows measured **−11%** against the **−50%** a
bytes-based model predicted. That gap killed a "traced DMA extent" lever analytically, before it was
built.

**Scope.** General to any lockstep execution group. `W = 32` and the 0.97 constant are v7x
SparseCore's; `E[max over W]` is the part that transfers, and it moves with `W`.

**Does not apply when** work is assigned to groups by *extent* rather than by index. Sorting or
bucketing members by length is the fix this rule points at, and it shifts the governing statistic
from `E[max]` back toward the mean.
