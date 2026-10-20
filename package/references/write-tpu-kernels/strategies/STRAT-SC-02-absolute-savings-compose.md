```yaml
id: STRAT-SC-02
title: Predict a stacked optimization from absolute time saved, never from multiplied percentages
hardware: tpu-v7x
unit: any
software: any
stage: [1, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-02; raw/2026-09-sctopk-final_report.md §2
recheck: nothing — but confirm your levers touch disjoint work before assuming near-multiplicativity
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-02 — Absolute savings compose; percentages do not

**Rule.** When stacking two optimizations, forecast the composite from the **absolute time each
saves**, not from its percentage against a baseline that the other lever is moving. The intuition
"this lever is worth more once the baseline is smaller" is an artifact of the denominator, not a
property of the silicon.

**Mechanism.** Deleted work costs what it costs regardless of what else was deleted. A percentage is
a ratio to a moving denominator, so stacking levers changes the percentage without changing the
physics. Levers touching disjoint work therefore compose *near*-multiplicatively in ratio terms —
but the quantity that is actually stable is the numerator.

**Evidence.** The `cvt` fusion
saved **−238.7 µs** on the incumbent and **−240.6 µs** on a structurally different variant — 0.8%
apart — while its *percentage* differed by 1.2 pp. The composite landed within ±0.4 pp of
`(1+d1)(1+d2)−1` in **11/14** cells, worst excess 3.44 pp. On the full three-lever stack
composition held to **1.8 pp, always slightly better than
multiplicative**, and beat 30 frozen predictions in **26/30** cells.

**Scope.** General. Strongest where the levers touch disjoint passes; the residual interaction is
the part worth measuring rather than modelling.

**Does not apply when** two levers contend for the same resource — then the second saves less than
it did alone, and neither absolute nor multiplicative composition holds.
