```yaml
id: STRAT-MLA-01
title: Bucket padding is free only downstream of the stage that multiplies it out
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [2, 3]
lever: algorithm
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-SKIP-01; raw/2026-09-dsamla-final_report.md §2
recheck: **whether the upstream stage is on the critical path at all** — then the smallest bucket on your padding ladder, the live-row fraction that lands in it, and the traversal granularity the skip can reach
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-01 — Bucket padding is free only downstream of the mask

**Rule.** Padding a batch up to a compiled bucket size buys static shapes, and the stage that masks
the padding out pays nothing for it. Every stage **upstream** of that mask pays full price for rows
whose contribution is already known to be zero. When an indirect gather or prefetch sits upstream of
the mask, make it skip the padding at whatever granularity its traversal allows.

**Mechanism.** The masking stage knows which rows are live; the stage feeding it does not, because
liveness arrives as a runtime value while the gather's cost is set by its static extent. A gather's
work is `rows × fanout` regardless of how many rows survive, so its wasted fraction is exactly the
bucket's padding fraction. That fraction is negligible at prefill and nearly total at decode, where
one live row can sit in a bucket of 64. The *device-time* saving is therefore largest precisely
where the bucket mechanism is least justified, and it is **structurally zero** when the bucket is
full — which is what makes the change safe to enable unconditionally rather than behind a shape
branch. Whether any of that reaches the step clock is a separate question this paragraph does not
answer; see "A saving on a parallel engine" below, where the answer turned out to be *no at decode
and yes at prefill* — the reverse of what the padding fraction predicts.

**Evidence.** Skipping gather windows holding no live row, at
`fanout = 2048` cache rows per token: decode `R=1` in a `P=64` bucket **131.89 → 40.08 µs/layer
(3.29×)**; the gather in isolation **29.59 → 18.76 µs**. At
`R == P` it is neutral as predicted — **1.03×** at 64 and **1.00×** at 128 — because no window is
empty. Correctness unchanged: 52/52, real rows bit-identical to the unmodified kernel.

The **live fraction is the whole story**, confirmed by sweeping it (two jobs, agreeing to ~5%): at `S` live rows
in a bucket of 64 the composed speedup runs 3.39× / 3.58× / 3.32× / 3.08× / 2.06× / 1.44× / 1.00×
for `S = 1, 2, 4, 8, 16, 32, 64` — monotone in `S`, tracking the predicted `(P−S)/P` and reaching
unity exactly where the mechanism says the saving vanishes. Time *not* skipped is flat
(~133 µs/layer for `S ≤ 16`), which is the gather being priced by the bucket rather than by the live
set — the claim, read directly off the incumbent arm.

**The bucket the work actually lands in is the one to measure, and it is the ladder's floor.** The
same sweep at `P = 16` — the smallest compiled bucket, and therefore where every low-concurrency
decode step lands — gives 1.88× / 1.90× / 1.88× / 1.63× / 1.07× for `S = 1, 2, 4, 8, 16`. Same
mechanism, same curve shape, **roughly half the multiple**, because the absolute padding available
to skip is four times smaller. An entire campaign can be spent characterising a bucket the serving
configuration never reaches.

**A saving on a parallel engine is worth nothing if the work was already hidden — and an isolated
kernel benchmark structurally cannot tell you.** Every number above is device time for the gather
itself. In the serving program the gather runs on SparseCore while the TensorCore runs the
attention, and an in-program trace of both arms (rank0, 78 layers)
says the skip's **decode** saving is entirely absorbed by that overlap: SparseCore gather busy falls 2.837 → 1.367
ms/step (2.08×, exactly as the mechanism predicts) while the TensorCore's wait on the gather is
**0.0004 ms/step in the incumbent and 0.0003 ms/step in the candidate** — there was no exposed wait
to convert, so the decode critical path does not move. The same trace at **prefill**, where the
bucket is full and this entry predicts the skip is worthless, is where it actually pays: the gather
wait *is* exposed there, 51.18 → 33.90 ms/step (−17.28 ms), larger than the attention saving beside
it. **The critical-path value of this lever is inverted relative to its device-time value.** Do not
report a gather-only speedup as a step speedup; measure the consumer's wait, not the producer's
duration, and see [[STRAT-SC-12]].

**Scope.** Any bucketed serving kernel with a gather, prefetch or scatter stage ahead of its mask.
The granularity of the skip is the traversal window, not the row, so the realizable win is the
padding fraction *rounded down to whole windows* — a window coarser than the live set collapses the
win, which is the constraint that makes the window size the thing to check first.

**Does not apply when** the upstream stage's output is consumed by a cross-row reduction, where
padding rows must still be materialized to a neutral element, or when buckets are typically full.
It is also not a licence to shrink the window (see [[STRAT-MLA-10]]): a finer window makes the skip
more selective *and* costs real work at `R == P`, and that trade lost.

**Further evidence (2026-09-dsanew, source: raw/2026-09-dsanew-STRATEGY.md § V3, § B5).** Two more instances on the
sparse-MLA path, same rule. (1) Decode in a 16-token bucket with 1 live row: bounding both the native SC gather and the
attention grid to live rows cut attention **60.7 → 23.0 µs/layer (−62.6%)**; the two halves
interact (gather-only +12.3%, attention-only −39.8%) — the gather bound pays only once the consumer stops reading the
dead rows. (2) **Under data parallelism the bucket is set by the busiest rank**: a rank with one live decode token runs
the 1024-token program while a peer prefills, and the static-grid gather then moves all 16 × 64 × 2048 rows — the TC
waited **1016 µs/layer** on it in-model. Bounding every chunk's gather by live rows:
**1426 → 333 µs (−53…−77% by clock)** on that cell, null on full prefill chunks.
