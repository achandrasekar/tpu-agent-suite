```yaml
id: STRAT-MLA-03
title: A knob swept on a shortened axis is biased when that axis is the one that penalizes it
hardware: any
unit: any
software: any
stage: [1, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-MEASURE-02; raw/2026-09-dsamla-WHITEBOARD-excerpt.md §22
recheck: nothing on the mechanism — re-derive which axis carries your knob's cost term
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-03 — Before shrinking a problem to sweep it, ask which term you shrank

**Rule.** Sweeping a knob at a reduced problem size is standard and usually fine. It stops being
fine when the axis you shortened is the one carrying the knob's **cost** rather than its benefit:
the sweep then understates the penalty by the shrink factor, and its trend does not extrapolate —
not in magnitude, and not in sign.

**Mechanism.** A knob's downside is often proportional to `extent / knob` — steps, launches, tiles,
passes. Its upside is usually proportional to something else entirely, often a fixed fraction of the
work. Shrinking `extent` scales the penalty term down while leaving the benefit term intact, so the
measurement is *biased toward the knob* by exactly the factor the axis was shortened. The result is
a clean, monotone, entirely convincing trend pointing the wrong way. Monotonicity is no defence
here; it is what the bias produces.

**Evidence.** A grid-truncation knob swept at bucket `P = 16`
against a chunk extent of `C = 64` exercised **16 grid steps where production has 64** — a 4×
understatement of the only term that penalizes a small setting. The smallest setting read **1.07×,
a win**. Re-measured at the production extent it read **0.89× at P=64 and 0.85× at P=128** — the
worst arm tested. Two conclusions drawn from the short axis
were retracted; the risk had been flagged in writing before the measurement, which is why the
re-measurement was run at all.

**Scope.** General. This is the *extent* sibling of [[STRAT-SC-12]], which is about measuring at the
wrong **scope** (which component). Both produce sign errors; this one is easier to miss, because
shrinking a problem to sweep it faster is normal practice and leaves no trace in the numbers.

**Does not apply when** the knob's cost provably does not scale with the shortened axis. That is a
claim to check, not assume — the reason an axis is worth shortening is usually that it is expensive,
and it is usually expensive because work is proportional to it.
