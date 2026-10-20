# STRATEGY.md — DSA/MLA decode, GLM-5.2-FP8 on tpu7x

Append-only. Four lines per entry: what, why, the number, the job. Nulls count.

---

## STRAT-SKIP-01 — skip gather windows that hold no live row `[WIN]`

- **What:** A padded token bucket gathers `index_topk` cache rows per padding row that attention then multiplies by zero. Skip whole gather windows containing no live row (`pl.when` over a `VectorSubcoreMesh`).
- **Why:** At decode R=1 in a P=16..64 bucket, 94–98% of the gather is provably dead work. Skip granularity is the window (`row_chunk_size = num_lanes × _NUM_ROW_SUBCHUNKS`), so it is free when R==P.
- **Number:** gather-only 29.59 → 18.76 µs at the shipping geometry; composed decode 131.89 → 40.08 µs.
- **Job:** j-49a5dfe7 (slope), j-390366e4 (composed), j-b3a169bf (correctness, 52/52).

## STRAT-GRID-01 — a "batch size" that truncates a grid has an interior optimum `[WIN]`

- **What:** `attention_kernel_batch_size` is not a tiling knob. The grid stops at the last batch holding a real token, so the value is a *truncation granularity*: smaller wastes less padding but costs more grid steps.
- **Why:** Two opposing terms ⇒ a turnover, not a monotone. Finer truncation wins while padding dominates; per-grid-step cost wins once the chunk is full.
- **Number:** at chunk=64, 16→4 improves decode *and* prefill; 4→1 improves decode a further 20% but regresses prefill 11% (64 grid steps vs 4). Optimum = 4.
- **Job:** j-7b13c6ce (P=16 sweep), j-9773d307 (P=64/128, the turnover).

## STRAT-MEASURE-01 — a chained-call device slope must perturb the *indices* `[METHOD]`

- **What:** To defeat CSE in a K-chained device-time regression of a gather+compute kernel, rotate the **index** array (`tk = (topk + off + carry) % SLOTS`), not the value array (`q`).
- **Why:** The gather's result is a pure function of `(cache, indices, page_table)`. Perturbing `q` leaves that subgraph identical, so XLA hoists all K gathers to one and the slope measures compute only.
- **Number:** CSE'd slope read 31–35 µs and "attention cheaper than the gather it contains"; index-rotated slope read 64.16 µs. Self-check `tpu_custom_call` count per chained call: must be 3.00, not <2.5.
- **Job:** j-a8845b56 (CSE'd, void), j-49a5dfe7 (fixed, k/call=3.00 on 12/12).

## STRAT-MEASURE-02 — a monotone trend on a shortened axis says nothing about the long axis `[METHOD/NULL]`

- **What:** A knob swept at a reduced problem size extrapolated the wrong way when the axis shortened was the one carrying the cost.
- **Why:** Grid steps per chunk = `chunk_size / batch`. Measuring at P=16 with chunk=64 exercises 16 steps where production has 64 — a 4× understatement of the only term that penalises a small batch.
- **Number:** batch=1 read **1.07× (a win)** at P=16, and **0.89× / 0.85× (a regression)** at P=64 / P=128.
- **Job:** j-7b13c6ce (misleading), j-9773d307 (corrected).

## STRAT-DMA-01 — never pad a SparseCore gather index array with a constant `[HW RULE]`

- **What:** Fill unused index slots with distinct addresses, not a repeated one.
- **Why:** SparseCore reads that concentrate on one address serialise; scattered reads pipeline.
- **Number:** concentrated 40.1 ns/read vs scattered 0.97 ns/read = **41×**, r²=1.0000.
- **Job:** j-a4face93.

## STRAT-DMA-02 — a TensorCore Pallas kernel cannot stand in for a SparseCore indirect gather `[NULL]`

- **What:** Rewriting the DSA gather as a TensorCore `pallas_call` to avoid the SparseCore hop.
- **Why:** Per-descriptor issue cost is architectural, not a tuning artifact; the TC path pays ~8× per descriptor and no blocking/layout change closed it.
- **Number:** ~8× per descriptor. Rejected before any composition.
- **Job:** j-23e1bf57, probe `01-tc-gather-dma-issue-rate`.

## STRAT-DMA-03 — 338 GB/s is not the ceiling for *wide-row* indirect SparseCore reads `[REFINEMENT]`

- **What:** `hardware-facts.md:590` cites ~338 GB/s; `:619` attributes ~230 GB/s saturation to the 32-byte DMA granule.
- **Why:** The DSA gather reads 640-byte contiguous rows — 20 whole granules — so the granule penalty does not apply. The cited figure appears to describe narrow/partial-granule reads.
- **Number:** 709 GB/s sustained (32768 rows × 640 B in 29.59 µs), in four independent jobs.
- **Job:** j-74a24183, j-996520b8, j-48095a41, j-49a5dfe7. **Flagged, not applied** — the shared reference was deliberately left unamended.

## STRAT-LOWER-01 — two SparseCore Pallas constructs that do lower `[HW RULE]`

- **What:** (a) `pl.when` wrapping `pltpu.emit_pipeline` on a `VectorSubcoreMesh`; (b) a **traced** grid length for `emit_pipeline` on SparseCore.
- **Why:** Both were assumed unsupported and are the enabling primitives for any data-dependent skip on SparseCore.
- **Number:** both compile and are byte-exact vs stock; `when` and `grid` modes within 0.6 µs of each other at two geometries.
- **Job:** j-2c4dfcd3, j-49a5dfe7, j-b3a169bf (`lowering probe: skip_mode`, both PASS).

## STRAT-LOWER-02 — the SparseCore rank-0/SMEM XOR `[HW RULE]`

- **What:** On SparseCore a rank-0 load is legal *only* from SMEM, and an SMEM load is legal *only* at rank 0 (`sc_lowering.py:125`).
- **Why:** Kills the obvious "read one scalar predicate from an SMEM `in_spec`" design. Workaround: vector-load a row subchunk into VMEM, then lane-extract element 0.
- **Number:** n/a — a compile-time legality rule, found by lowering probe rather than measurement.
- **Job:** j-23e1bf57.

## STRAT-GEOM-01 — the gather window geometry retune, rejected `[NULL]`

- **What:** `_NUM_ROW_SUBCHUNKS` 32→8, which makes gather windows finer and the skip more selective.
- **Why rejected:** It is a *module-level* constant — it retunes the gather for every caller and every shape, including shapes never measured — and the finer window costs real work at R==P where skip is a no-op.
- **Number:** +0.81 µs/layer at decode (**+2.6%**) against **−23.5 µs/layer at prefill (−13%)**.
- **Job:** j-9773d307. A global constant must pay for itself on the shapes you did *not* measure.

## STRAT-ROOF-01 — DSA decode is fixed-cost bound, and optimizing moves it *away* from roofline `[DIAGNOSTIC]`

- **What:** Ratio to the HBM-traffic floor across the optimization path: 3.8× → 8.2× → 12.8×.
- **Why:** Bytes fell 8× while time fell 2.4×. Removing data movement from something whose cost is not data movement widens the roofline gap — the signature of fixed per-invocation/per-window cost.
- **Number:** the machine retires a 640-B indirect row every 0.90 ns in the stock arm; the skipped gather runs at 5.8–6.0 ns/row, so **~10 µs/layer is window traversal that the skip predicates off rather than skips**.
- **Job:** j-49a5dfe7 + §19. Next lever, ~0.78 ms/step, and `grid` mode does **not** capture it (12.35 vs 12.47 µs).
