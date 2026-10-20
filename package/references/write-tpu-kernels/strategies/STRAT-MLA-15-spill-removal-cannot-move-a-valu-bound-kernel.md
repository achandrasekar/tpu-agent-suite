```yaml
id: STRAT-MLA-15
title: Measured null — removing register spills does not move a kernel whose VALU issue binds; spills ride in free slots
hardware: tpu-v7x
unit: tensorcore
software: jax 0.10.2 / libtpu 0.0.44.1
stage: [3, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-dsanew-STRATEGY.md § B3 KILLED; raw/2026-09-dsanew-final_report.md § Rejected
recheck: static bundle count before/after the change — if it does not fall, the removed ops were co-issued
origin: campaign dsa-mla-256k-c16, 2026-09
```

# STRAT-MLA-15 — Price a spill by the slot it occupies, not by its count

**Rule.** A large spill count is not a lever unless spill traffic sits on the binding slot. In a VLIW
bundle, spill stores and reloads use the vst/vld slots; if VALU issue already fills the bundles, the spills
are co-issued for free and deleting them leaves the bundle count — and the clock — unchanged. Before
restructuring a kernel to remove spills, check which slot's op count sets the bundle floor.

**Mechanism.** Bundle count ≈ max over slots of (ops in that slot / slots per bundle), plus dependency
stalls. Removing ops from a non-binding slot changes neither term.

**Evidence.** An indexer scoring step (relu × w + acc over 32 heads, 3 VALU ops per score element) was
restructured into key sub-tiles that keep the accumulator in registers: spill stores **11,681 → 2,855**,
VALU ops 26,047 → 26,034, bundles **9,524 → 9,580**, per-call time +0.4%. A variant that cut bundles 7% still moved the slope only −0.6%. Bit-exact throughout.

**Scope.** Any VLIW-scheduled kernel. **Does not apply when** vst/vld is the binding slot — then spill
removal is a first-order lever.
