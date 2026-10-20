```yaml
id: STRAT-SC-09
title: Unit choice follows what binds the algorithm, not which unit is wider
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [1, 2]
lever: algorithm
generality: unit-wide
status: measured
source: raw/2026-09-sctopk-final_report.md §6 — not in that campaign's STRATEGY.md;
  raw/2026-09-dsamla-STRATEGY.md § STRAT-DMA-02 (second instance, indirect gather)
recheck: what binds your algorithm at your parameters, and the effective bandwidth the wide unit actually achieves on it
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-09 — Unit choice follows what binds, not pipe width

**Rule.** Pick the execution unit from **what binds the algorithm**, not from which unit has more
throughput on paper. An algorithm bound by a parameter other than bandwidth will run the wide unit
far below its rated number, and the narrow unit that matches its access pattern wins by a margin
that looks impossible until you notice the wide unit was never bandwidth-bound to begin with.

**Mechanism.** A selection algorithm's work scales with the number of items selected, not with the
bytes scanned. A wide SIMD unit prices that as poor effective bandwidth — it moves the bytes at full
rate and then serializes on the selection — while a unit with gather/scatter and lane-private state
prices it as ordinary work.

**Evidence.** Top-k at **k=2048** is `k`-bound, and
SparseCore beat TensorCore by **8.70–22.97×**; XLA's TensorCore top-k ran at **4.4–10.6 GB/s
effective** against a unit rated orders of magnitude higher. The converse check on the same unit: a library SparseCore top-k forcing 16-way cooperation above
N=4096 issued ~1200 `subcore_barrier`s per call and ran **1.48× slower dense, 1.95× ragged** — the
right unit with the wrong cooperation granularity is not the right choice either.

A **second, independent instance** on a different algorithm:
rewriting a wide-row indirect gather as a TensorCore `pallas_call`, to avoid a SparseCore hop
entirely, paid **~8× per descriptor** and no blocking or layout change closed the gap. Here the
binding parameter is **descriptor issue rate**, not `k` — a different answer to the same question,
reaching the same unit. Rejected before composition.

**Scope.** v7x-generation SparseCore vs TensorCore. The two measured margins are specific to their
algorithms — large-`k` selection, and descriptor-bound indirect gather — and what transfers is the
question, *what binds?*, not either ratio. That two unrelated algorithms both turned out to be bound
by something other than bandwidth, and both landed on the narrow unit, is the reusable part.

**Does not apply when** `k` is small enough that the scan dominates. Then the algorithm really is
bandwidth-bound and the wide unit is correct.
