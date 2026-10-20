# SparseCore gather traversal residual — final report

**Block.** `vllm-torchtpu/src/vllm_torchtpu/kernels/mla/sparse/dsa_gather.py` @ `61b1c0b8` with
`<campaign>/final/kernel_patch.diff` applied (the incumbent). GLM-5.2-FP8, 128k/1k, tpu7x `2x2x2`,
`topk=2048`, shipping gather geometry `(_NUM_ROW_SUBCHUNKS, _NUM_STREAMS) = (32, 4)`.
jax/jaxlib 0.10.2, libtpu 0.0.44.1.

**Outcome. No kernel.** The lever this campaign was opened to build has a measured ceiling of
**1.96 µs/layer — 4.9% of the incumbent** — and the roofline gate closes at **1.135×** of the
established floor, inside the 1.25× stop bar. One job, no code under `variants/`.

The campaign's value is that it **corrected the premise it was opened on**, and relocated the
largest remaining term in the block.

---

## 1. What was claimed on entry, and what it actually is

The campaign was opened on `[STRAT-MLA-07]`: *"roughly 10 µs/layer is window traversal that the skip
predicates off rather than removes"* — ~26% of the 40.08 µs/layer incumbent, larger than everything
the predecessor won.

Job #1 swept `P` at constant live work (`R=1`, 2048 live rows, 4 live windows), moving `num_blocks`
2 → 16 while leaving every other term untouched, and fit `t = a + b·num_blocks`
[measured: `j-02efd001-f3a4-4613-8a76`]:

| P | num_blocks | gather µs |
| --- | --- | --- |
| 16 | 2 | 17.37 |
| 32 | 4 | 18.21 |
| 64 | 8 | 19.68 |
| 128 | 16 | 21.37 |

**`a = 17.06 µs`, `b = 0.280 µs/outer-step`, R² = 0.974.** Refit independently on the main thread;
reproduces to 4 significant figures. Chain check: 1.00 `tpu_custom_call` per chained call, so the
clock is real [fact: STRAT-SC-03, STRAT-MLA-04].

Two errors in the entry premise, and they compound:

**(a) Mis-sourced.** The 12.35/12.47 µs pair the 10 µs was derived from is from the **rejected
`(8,4)` geometry**, not the shipping `(32,4)`. At shipping geometry the `R=1, P=64` gather is
**19.68 µs**.

**(b) Misattributed.** Subtracting live-row movement (`2048 × 0.90 ns = 1.84 µs`), **15.22 µs — 77%
of the gather — is invariant to the grid extent.** Traversal is **2.24 µs, 11.4%**. Per window that
is **8.75 ns**, not the 41 ns a per-window reading implied — **4.7× smaller**.

`[STRAT-MLA-07]`'s **rule is vindicated**: the widening roofline ratio did mean fixed-cost binding.
Its **localization is refuted**: the fixed cost is per-*call*, not per-*traversal*. Those name
different levers — a grid bound versus a call count — which is the whole reason this campaign
existed and the whole reason it closes.

## 2. The candidate that was designed and not built

Step 1–3 produced a design strictly simpler than the compaction sketch in `BRIEF.md`, and it is
recorded because the *reasoning* survives even though the lever does not.

The index stream is `(token, topk)` flattened with bucket padding at the end, so **the live region is
already a dense prefix `[0, R·topk)`**. Shortening the outer `emit_pipeline` grid to
`cdiv(R·topk, block_size)` therefore needs no compaction table and no output indirection — surviving
steps keep their `b`, so `out_row_base` keeps its meaning. At `R=1, P=64` that is 8 outer steps → 1,
using the traced grid length `[fact: STRAT-MLA-08]` already proved lowers (on the inner pipeline; the
outer one was the untested half).

Two independent facts said not to build the general version instead:

- `[fact: STRAT-SC-01]` prices a SparseCore compaction pass by its cross-lane serializing ops, not
  its trip count: `store_compressed` / `all_reduce_population_count` at ~3× a masked-scatter chunk,
  and that campaign's survivor compaction lost **+6.53%, slower in 14/14 cells**.
- The geometry made compaction unnecessary anyway (prefix, above).

**Ceiling, measured:** `b × (num_blocks − 1) = 0.280 × 7 = 1.96 µs/layer = 4.9%` of the incumbent.
At the conservative end of the slope CI (`b = 0.419`, t, df=2) it is 2.93 µs = **7.3%**. Both are
below the bar, so the design was not implemented.

## 3. The roofline gate

`WHITEBOARD.md §4` deferred the gate deliberately — the HBM floor had disqualified itself
[fact: STRAT-MLA-07], so "12.8× from floor" was not headroom. The gate was defined *before* the
measurement as `floor = a + b·live_steps + c·live_rows`, stopping at 1.25×.

**Measured 19.68 µs against a floor of 17.34 µs = 1.135×. The gate closes.**

## 4. Rejections and nulls

| rejected | why | evidence |
| --- | --- | --- |
| Prefix truncation of the outer gather grid | Whole ceiling 1.96 µs/layer = 4.9% of incumbent; 7.3% at the conservative CI. | `j-02efd001` |
| Compaction with a scalar-prefetch window table | Unnecessary (live region is a dense prefix) *and* carries a measured loss mechanism. | geometry + [STRAT-SC-01] |
| **Runner-side bucket tightening** (`BRIEF` lever 1) | Entered as an unmeasured inference that halving the bucket halves the residual. **The linearity is confirmed and the conclusion is still wrong**: traversal is exactly 4× smaller at P=16, but the gather total moves only **1.133×** (19.68 → 17.37 µs), not ~4×. | `j-02efd001` |
| `_NUM_ROW_SUBCHUNKS` retune | Not re-tested; already rejected by the predecessor as a module-level global. | [STRAT-MLA-10] |

## 5. Carried — the lever this campaign found but did not take

**15.22 µs/layer of per-call fixed cost on the gather alone — 38% of the 40.08 µs/layer incumbent.**
That is larger than everything this campaign or its predecessor won, and it is now the block's
dominant term. `hardware-facts.md` prices a TensorCore `pallas_call` intercept at 20–36 µs and states
that fusing calls beats any block tuning in that regime; the gather is one of **3** `tpu_custom_call`s
per layer. **Reducing the call count is the real lever.** Note the direction constraint: moving the
gather onto TensorCore to fuse it there was already rejected at ~8× per descriptor
[fact: STRAT-SC-09], so any fusion has to bring the consumer to SparseCore, not the reverse.

Two caveats on that number, both load-bearing:

1. **`a` is "invariant to `P`", not "irreducible".** The sweep held `topk` and live-row count
   constant, so anything scaling with those sits inside `a`. Whether part of the 15.22 µs tracks
   `topk` is untested, and is the cheapest next question.
2. **It was measured on the gather in isolation**, chained. Whether the same fixed cost is paid when
   the gather runs as one of three calls inside `sparse_ragged_paged_attention` is not established —
   3 × 15.22 µs would exceed the 40.08 µs/layer the whole layer costs, so the costs are plainly not
   uniform across the three calls. Resolve this before sizing any fusion work.

## 6. What this campaign spent

One job (`j-02efd001-f3a4-4613-8a76`), no e2e, nothing under `variants/`, no change to any source
file. `<campaign>/` was not touched.

## 7. Files

- `BRIEF.md` — entry brief (its §Target arithmetic is superseded by §1 above)
- `WHITEBOARD.md` — geometry, the three cost models, the prefix-truncation design
- `STRATEGY.md` — STRAT-TRAV-01…04
- `probes/01-cost-model/` — `probe.py`, `README.md`, `j-02efd001-f3a4-4613-8a76.txt`
