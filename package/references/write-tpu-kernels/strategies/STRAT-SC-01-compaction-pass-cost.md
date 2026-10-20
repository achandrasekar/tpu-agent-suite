```yaml
id: STRAT-SC-01
title: A pass with cross-lane serializing ops in its body is priced by those ops, not by its trip count
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [2, 3]
lever: algorithm
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-01; raw/2026-09-sctopk-final_report.md §6
recheck: the per-chunk cost ratio of your serializing op to a plain masked pass on the target unit
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-01 — Price a pass by its serializing ops, not its trip count

**Rule.** "This deletes two passes" is a trip-count argument, and trip count does not price a pass
whose body contains **cross-lane serializing ops**. Cost the body first. A restructuring that trades
several cheap full-width passes for one pass with serializing ops in it can be a net loss even when
the trip-count arithmetic is correct.

**Mechanism.** A cross-lane op forces the lanes to agree before the chunk can retire, so its cost is
paid per chunk and does not amortize over the pass. Two such ops in a body make that body a small
multiple of a masked pass over the same width — the trip-count model prices the *iteration* while
the hardware prices the *body*.

**Evidence.** Survivor compaction
replaced **3** full-width masked passes with **1** compaction pass over the same width, so the
trip-count model predicted −2 passes. It measured **+6.53% median, slower in 14/14 cells**. The
body ran two `plsc.store_compressed` and two `plsc.all_reduce_population_count` per chunk and cost
**~3×** a masked-scatter chunk. The survivor-count model was right; the per-chunk cost model was
wrong.

**Scope.** General wherever a lane-cooperative primitive exists. On v7x SparseCore the ops are
`store_compressed` and `all_reduce_population_count`; the ~3× is theirs and must be re-measured
elsewhere.

**Does not apply when** the pass being deleted is itself serializing, or when the compaction shrinks
later passes enough to pay for itself — which is a measurement, not an assumption.
