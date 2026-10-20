```yaml
id: STRAT-MLA-07
title: A roofline ratio that widens as you optimize proves the floor models the wrong resource
hardware: tpu-v7x
unit: any
software: any
stage: [1, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-dsamla-WHITEBOARD-excerpt.md §19; raw/2026-09-dsamla-STRATEGY.md § STRAT-ROOF-01;
  raw/2026-09-sctraversal-final_report.md §1 (the correction below)
recheck: nothing on the mechanism — recompute your own floor at each step of the path, not once.
  **Then localize the fixed cost before acting on it — see the correction in Evidence.**
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-07 — Track the roofline ratio across the optimization, not at the start of it

**Rule.** Recompute the ratio of measured time to the analytic floor at **every step** of an
optimization. If removing work moves the kernel *further* from its floor, the remaining cost is not
the resource the floor models. That is a measurement, not a disappointment: it says the next lever
must attack per-invocation or per-traversal fixed cost, and that further work on the modelled
resource is capped.

**Mechanism.** A roofline divides measured time by a modelled resource. Let time be
`fixed + bytes/BW`. Cutting `bytes` shrinks the denominator of the floor proportionally while the
numerator falls only by the variable part, so the ratio **rises**. A monotonically rising ratio along
an optimization path is therefore direct evidence of a fixed-cost component, and its size bounds
that component from below. This inverts the usual reading: the gap is normally treated as headroom
to be closed, but a *widening* gap is the floor disqualifying itself.

**Evidence.** Along the shipped path the ratio to the
HBM-traffic floor went **3.8× → 8.2× → 12.8×**, as bytes fell 8× while time fell only 2.4×. The
mechanism was then located directly: the same machine retires a 640-B indirect row every **0.90 ns**
in the unskipped arm but only every **5.8–6.0 ns** in the skipped one, which places **~10 µs/layer**
in window traversal that the skip predicates off rather than removes. Confirmed not to be an
artifact of how the predicate was written — the `when` and `grid` spellings read **12.47 vs
12.35 µs**. That residual was named as the next investigation rather than claimed as a win.

**Correction — the rule held, the localization did not.**
The successor campaign swept the grid extent at constant live work and fit
`t = a + b·num_blocks`: **`a = 17.06 µs`, `b = 0.280 µs/outer-step`, R² = 0.974**. Net of live-row
movement, **77% of the gather is invariant to the grid extent** — the residual is **per-call fixed
cost, not window traversal**. Per window it is **8.75 ns**, not the ~41 ns implied above. Two
conditions separate the readings: the 12.35/12.47 µs pair is from the **rejected `(8,4)` geometry**
(at the shipping `(32,4)` the same cell is **19.68 µs**), and "predicated-off traversal" was an
attribution, never a measurement. **This vindicates the Rule** — the fixed-cost diagnosis was
correct — **and refutes the lever it appeared to name.** Traversal is attacked by shortening a grid;
per-call cost only by issuing fewer calls. Read the Rule as "attack fixed cost *once you have
localized it*"; localizing costs one 4-arm sweep and the successor campaign closed on it without
writing a variant. See [[STRAT-MLA-11]].

**Scope.** General to any roofline, on any unit. The diagnostic needs at least two points on an
optimization path, which is the only real cost of applying it.

**Does not apply when** the ratio is already near 1, where its movement is inside noise, or when the
work removed also removed fixed cost (fewer invocations, not smaller ones) — then both terms fall
and the ratio is uninformative rather than diagnostic. Complements [[STRAT-SC-08]], which says prove
the load binds *before* restructuring it; this says how to notice afterwards that it stopped binding.
