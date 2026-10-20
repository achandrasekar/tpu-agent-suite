```yaml
id: STRAT-MLA-10
title: A module-level constant must pay for itself on the shapes you did not measure
hardware: any
unit: any
software: any
stage: [3, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-GEOM-01; raw/2026-09-dsamla-WHITEBOARD-excerpt.md §22
recheck: grep the constant's callers and enumerate the shape classes compiled against it
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-10 — Tune through an argument; a global owes evidence on every shape it reaches

**Rule.** The evidence a tuning owes is set by its **blast radius**, not by the size of the diff.
Changing a call-site argument obliges you to measure that call site. Changing a module-level
constant obliges you to measure every shape class compiled against it, including the ones with no
test — and if you cannot, thread the value as an argument instead and leave the default alone.

**Mechanism.** The two changes look identical in review — one number, one line — which is what makes
the asymmetry easy to miss. A global is read by every caller at every shape, so a win on the shape
under optimization tells you about one point in the space the change actually covers. Worse, globals
that tune a *granularity* tend to have exactly the two-sided cost structure of [[STRAT-MLA-02]]: the
setting that helps the sparse shape is the setting that adds overhead to the dense one, so the
untested shapes are not merely unmeasured, they are *systematically* on the losing side.

**Evidence.** A gather-window geometry constant (32 → 8 row
subchunks) made a padding skip finer and more selective, worth **+0.81 µs/layer at decode (+2.6%)**
— and **−23.5 µs/layer at the full-bucket shape (−13%)**, where the skip is a no-op and the finer
window is pure overhead. Rejected despite winning the shape being optimized. The two tunings that
did ship were instead threaded as keyword arguments from the single dispatcher, leaving the kernel's
defaults untouched for any future caller and keeping each revertible on its own line.

**Scope.** General; this is about change blast radius, not about accelerators. It is also the
cheapest of the method rules to apply — the check is a grep for callers, run before the A/B rather
than after it.

**Does not apply when** the constant provably has one caller and one compiled shape. Verify that by
grep rather than by assumption — and prefer the argument anyway, since it costs nothing, documents
the tuning at the site that chose it, and keeps the revert local.
