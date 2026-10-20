# STRATEGY — empirical SparseCore rules earned in this campaign

Append-only. Each entry is 4 lines: rule, mechanism, evidence, scope.

## STRAT-SC-01 — Price a `store_compressed` pass by its serializing ops, not its trip count

**Rule.** A cross-lane compaction chunk costs ~3× a masked-scatter chunk. Budget it that way before
choosing to compact.
**Mechanism.** `plsc.store_compressed` + `plsc.all_reduce_population_count` are cross-lane
serializing ops; a compaction body running two of each per chunk cannot be modelled as "one pass".
**Evidence.** [measured: j-e485902b-bf5c-49dd-98b4, j-abfde0e5-079d-482b-be98] Survivor compaction
replaced 3 full-width masked passes with 1 compaction pass over the same width and measured
**+6.53% median, slower 14/14 cells**. Trip-count model predicted −2 passes; the pass cost 3.
**Scope.** SparseCore v7x, any radix/filter kernel weighing compaction against re-scanning.

## STRAT-SC-02 — Absolute savings compose; percentages do not

**Rule.** When stacking two kernel optimizations, predict the composite from **absolute µs saved**,
not from multiplying percentage deltas against a moving baseline.
**Mechanism.** A deleted pass costs what it costs regardless of what else was deleted. The intuition
"this lever is worth more once the baseline is smaller" is a percentage artifact, not physics.
**Evidence.** [measured: same jobs] `cvt` fusion saved −238.7 µs on the incumbent and −240.6 µs on
the compaction variant (0.8% apart), while its *percentage* differed by 1.2 pp. Composite landed
within ±0.4 pp of `(1+d1)(1+d2)−1` in **11/14** cells (worst excess 3.44 pp) — near-multiplicative,
little interaction. [12/14 and "exactly" were the probe's own overstatement; corrected on review.]
**Scope.** General; strongest where levers touch disjoint passes.

## STRAT-SC-03 — Barrier the loop CARRY, never the captured invariants

**Rule.** In a chained-invocation clock, the data dependency must run through the loop carry. Then
assert `tpu_custom_call` occurrences in the HLO equal M exactly.
**Mechanism.** Barriering loop-invariant operands leaves XLA free to prove the kernel's inputs
unchanging; LICM hoists the call clean out of the `fori_loop` and M invocations execute **once**.
No correctness guard can catch this — a hoisted loop is idempotent and still returns the right answer.
**Evidence.** [measured: probe 04 vs arbitration job] A 2.8× phantom speedup and a physically
impossible sub-HBM-floor reading; HLO showed the async-start taking top-level operands instead of a
`while`-carry `%get-tuple-element`. Five reported numbers retracted.
**Scope.** Any Pallas microbenchmark that amortizes launch overhead by chaining. Derive per-call
cost from an **M-sweep slope**, never divide-by-M at a single M.

## STRAT-SC-04 — `emit_pipeline(..., scratches=)` is a silent no-op on the primitive branch

**Rule.** Pass scratch buffers to an `emit_pipeline` body by **closure capture**. Never via
`scratches=`.
**Mechanism.** The kwarg is real on the legacy branch and forwarded via `**kwargs`, but
`config.use_emit_pipeline_primitive` routes to a branch that drops unknown kwargs **with no error**.
Zero upstream call sites use it.
**Evidence.** [source: jax/_src/pallas/mosaic/pipeline.py:1768-1790,2061-2078; config.py:1906-1912]
Closure-capture form confirmed working in `tpu_sparsecore_pallas_test.py:2995-3025`.
**Scope.** All Pallas TPU pipelines. Also: the 2-D `Tiling.SPARSE_CORE` path is unproven at 16 lanes
(upstream test skips unless `num_lanes == 8`, open upstream bug 478865387) — use the 1-D `BlockSpec` form.

## STRAT-SC-05 — Raggedness is governed by the longest row in a 32-row wave

**Rule.** Do not model a ragged SparseCore kernel's cost from the *mean* row length. Use the
**maximum over the 32 rows sharing a wave** (`row = w*(cores*subcores) + flat`).
**Mechanism.** 32 vector subcores run in lockstep per wave; the wave costs its longest row. For
`rl ~ U(0,n)`, `E[max of 32] ≈ 0.97n` — so halving mean row length barely moves the clock.
**Evidence.** [measured] U(0,n_p) rows measured −11%, not the −50% a bytes-based model predicts.
This independently killed the "traced DMA extent" lever before it was built.
**Scope.** Any `VectorSubcoreMesh` kernel with per-row variable extent.

## STRAT-SC-06 — Never let a per-lane scatter stride be a multiple of the bank count

**Rule.** A per-lane private histogram addressed `lane * NUM_BUCKETS + bucket` serializes across
lanes whenever `NUM_BUCKETS % 32 == 0`. **Pad the stride to an odd number** — `lane * (NUM_BUCKETS+1)
+ bucket`. Costs 1 word per lane; buys up to 16-way parallelism back.
**Mechanism.** SparseCore VMEM has **32 banks**. With stride 256, all 16 lanes hitting the same
bucket produce addresses congruent mod 32 → one bank → 16-way serialization. Cost fits
`base + (ways−1) × 4.7 µs`. Stride 257 makes lane `l`'s address `l` mod 32, so 16 lanes hit 16
distinct banks regardless of bucket.
**Evidence.** [measured: j-764fb619-99e0-4b74-b283] −15.08% on `dsa_like`, −7.6% on `uniform` at
b·p=64; −17.24%/−9.54% at b·p=4096. 40/40 oracle PASS, rows_differing 0. `priv_vmem` 4112 → 4128.
**Scope.** Any SparseCore kernel with lane-private tables — histograms, counters, accumulators.

## STRAT-SC-07 — Distribution sensitivity at constant work is a memory conflict, not a workload

**Rule.** If a kernel's latency moves with the input *distribution* while the operation count is
provably unchanged, stop modelling work and go look for a bank or port conflict. Use the **collapse
of the cross-distribution spread** as the confirming signature of the fix.
**Mechanism.** Conflict severity tracks address *concentration*, which is exactly what a
distribution changes while leaving update counts fixed. Work-based explanations cannot produce this.
**Evidence.** [measured: j-3ce5902f, j-764fb619] The L0 scatter cost 14.83% on `dsa_like` vs 7.90%
on `uniform` at identical update counts. After padding, per-arm spread across four distributions
went from +7.87/+7.68/+7.00/+6.83% to within ±1.5%.
**Scope.** General. Also a diagnostic-order rule: a pre-LLO Mosaic dump exposes no bank counter, so
a controlled concurrency sweep is the instrument, not the profiler.
