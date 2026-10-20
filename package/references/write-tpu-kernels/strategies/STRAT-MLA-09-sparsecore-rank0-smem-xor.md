```yaml
id: STRAT-MLA-09
title: On SparseCore a rank-0 load is legal only from SMEM, and an SMEM load only at rank 0
hardware: tpu-v7x
unit: sparsecore
software: "jax 0.10.2 / jaxlib 0.10.2 / libtpu 0.0.44.1"
stage: [4]
lever: layout
generality: unit-wide
status: source
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-LOWER-02
recheck: re-read the rank/memory-space check in the SparseCore lowering pass after a jax bump
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-09 — The SparseCore rank-0 / SMEM exclusive-or

**Rule.** On SparseCore the two properties are locked together: a **rank-0 (scalar) load is legal
only from SMEM**, and an **SMEM load is legal only at rank 0**. You therefore cannot read a scalar
out of a vector-memory `in_spec`, and you cannot read a vector out of SMEM. Neither half is
independently negotiable, which rules out the obvious shape of a per-window scalar predicate.

**Mechanism.** Enforced in the SparseCore lowering pass, at its rank/memory-space check
[source: `sc_lowering.py:125`]. The design it kills is the natural one: place one scalar predicate
per window in an SMEM `in_spec` and branch on it. The working shape instead is to **vector-load a
row subchunk into VMEM and lane-extract element 0** — the predicate arrives as a vector and is
narrowed afterwards, rather than being fetched as a scalar.

**Evidence.** [source: `sc_lowering.py:125`] read directly, and the SMEM-predicate design failed to
lower in a probe. This is a compile-time legality rule, so it
has no number; `status: source` rather than `measured` reflects that the rule is read out of the
compiler and the job only demonstrates it.

**Scope.** v7x SparseCore at the pinned toolchain. A lowering restriction is a fact about a build.
The workaround is cheap and is the thing to carry: if a predicate must cross into SparseCore, plan
for it to arrive as a vector.

**Does not apply on** TensorCore, where scalar reads from a `pl.ANY`/SMEM spec are routine — this
asymmetry is precisely what makes the restriction surprising to anyone porting a working TensorCore
Pallas kernel onto the unit.
