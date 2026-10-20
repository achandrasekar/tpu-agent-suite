```yaml
id: STRAT-MLA-13
title: Bounding a kernel's loops by the live length does not bound its DMA — and index-assigned work leaves the live units at full cost
hardware: tpu-v7x
unit: any
software: any
stage: [2, 3]
lever: algorithm
generality: general
status: measured
source: raw/2026-09-dsanew-STRATEGY.md § incumbent profile, § probe 03; raw/2026-09-dsanew-final_report.md § Levers V2
recheck: nothing on the mechanism — check whether your kernel's input BlockSpec/DMA extent is static while its body loops are length-bounded
origin: campaign dsa-mla-256k-c16, 2026-09
```

# STRAT-MLA-13 — A compute bound is not a DMA bound

**Rule.** A kernel whose body skips work past a runtime length can still pay for all of it: an
auto-pipelined `BlockSpec` (or any static-extent copy) moves the **whole block** whatever the body reads.
When the kernel is transfer-bound, its cost follows the static extent, not the live length. And fixing
the DMA alone is not enough if work is assigned by **index**: the live slices land on a subset of the
units, which keep their full cost while the rest go idle. Make the partition width itself dynamic so
the live range is spread over every unit, then bound each unit's DMA extent.

**Mechanism.** The pipeline emitter sizes transfers from the BlockSpec, which is static; the length only
reaches the loops inside the body. On a unit whose throughput is set by streaming (SparseCore), the
skipped compute was never the cost. Index assignment (`row·P + j` → unit `j mod W`) is the lockstep
problem of [[STRAT-SC-05]] in another form: the critical path is the busiest unit.

**Evidence.** A two-stage SparseCore top-k over `[b·16, 16512]` sub-rows, bounded by row length in
compute only, cost **3.41 ms at every context length** (C = 32k, 131k, 261k; ≈315 GB/s aggregate
streaming). Rebalanced (sub-row width = ⌈max_len/16⌉) with traced
DMA extents: **−55.2% at C=32k, −21.0% at C=131k, −0.2% at C=261k** (the null control, no dead context)
— the saving tracks the dead fraction, as the mechanism predicts. Top-k sets
bit-identical.

**Scope.** Any kernel with a static-extent input pipeline and a runtime live length. **Does not apply
when** the kernel is compute-bound on the live work — then the extra transfer is hidden (see [[STRAT-SC-08]]).
