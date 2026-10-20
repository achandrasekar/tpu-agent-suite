# STRATEGY — SparseCore traversal residual campaign

Append-only. Promoted into `strategies/` at close.

---

## STRAT-TRAV-01 — measured null: prefix truncation of the outer gather grid

- **lever:** pipelining — shorten the outer `emit_pipeline` grid to the live index prefix
- **mutation:** traced outer grid length `cdiv(R*topk, block_size)` instead of the static
  `cdiv(P*topk, block_size)`; live region is a dense prefix so no compaction table is needed
- **result:** **rejected before implementation.** Whole ceiling at `R=1, P=64` is
  `b * (num_blocks - 1) = 0.280 * 7 = 1.96 us/layer` = **4.9%** of the 40.08 us/layer incumbent.
  At the conservative end of the slope CI (`b = 0.419`) it is 2.93 us = 7.3%.
- **evidence:** `[measured: j-02efd001-f3a4-4613-8a76]`

## STRAT-TRAV-02 — the SparseCore gather is per-call-fixed bound, not traversal bound

- **lever:** method / diagnostic
- **mechanism:** sweeping `P` at constant live work (`R=1`, 2048 live rows, 4 live windows) moves
  `num_blocks` 2 -> 16 and leaves every other term untouched. Fit `t = a + b*num_blocks`:
  **`a = 17.06 us`, `b = 0.280 us/outer-step`, R^2 = 0.974** over arms
  17.37 / 18.21 / 19.68 / 21.37 us.
- **result:** `a - 1.84 us` (live row movement) = **15.22 us invariant to the grid extent = 77% of
  the P=64 gather.** Traversal is 2.24 us, **11.4%**. Per-window cost is **8.75 ns**, not the 41 ns
  a per-window model of the predecessor's residual implied — **4.7x smaller**.
- **evidence:** `[measured: j-02efd001-f3a4-4613-8a76]`, chain check 1.00 `tpu_custom_call` per
  chained call

## STRAT-TRAV-03 — refutes BRIEF lever 1: bucket tightening does not move the gather

- **lever:** algorithm (runner-side bucket size)
- **claim under test:** traversal is linear in window count, so halving the decode bucket halves the
  residual — entered as an unmeasured inference, explicitly flagged as such.
- **result:** **the linearity is confirmed and the conclusion is still wrong.** `b` is constant, so
  traversal *is* proportional to `num_blocks` (0.56 us at P=16 vs 2.24 us at P=64, exactly 4x). But
  traversal is a small share, so the **gather total moves only 1.133x** from P=64 to P=16
  (19.68 -> 17.37 us), not ~4x.
- **evidence:** `[measured: j-02efd001-f3a4-4613-8a76]`

## STRAT-TRAV-04 — the predecessor's residual was misattributed, and partly mis-sourced

- Two independent errors, found by the sweep:
  1. **Mis-sourced.** The 12.35/12.47 us pair the residual was derived from is from the **rejected
     `(8,4)` gather geometry**, not the shipping `(32,4)`. At shipping geometry the `R=1, P=64`
     gather is **19.68 us** (harness cross-check 20.7 us; within 2.6% of the predecessor's 18.91 us
     at P=16).
  2. **Misattributed.** The residual was called "window traversal that the skip predicates off
     rather than removes". It is overwhelmingly **per-call fixed cost**: 77% invariant to grid
     extent. `[fact: STRAT-MLA-07]`'s *rule* (widening roofline ratio => fixed-cost bound) is
     **vindicated**; its *localization* to traversal is refuted.
- **why it matters:** the two diagnoses name different levers. Traversal is attacked by shortening a
  grid; per-call fixed cost is attacked only by issuing fewer calls. A campaign was opened on the
  first reading and closed by one 4-arm sweep.
- **evidence:** `[measured: j-02efd001-f3a4-4613-8a76]` against
  `[measured: j-49a5dfe7-e91c-407c-880a]`

## Carried, not pursued

- **15.22 us/layer of per-call fixed cost on the gather** is now the largest single term in the
  block — **38% of the 40.08 us/layer incumbent**, far larger than anything this campaign or its
  predecessor won. `hardware-facts.md` prices a TensorCore `pallas_call` intercept at 20-36 us and
  says fusing calls beats any block tuning in that regime. The gather is one of 3 `tpu_custom_call`s
  per layer. **Reducing the call count is the real lever**, and it is out of this campaign's remit.
  Note `[fact: STRAT-SC-09]`: moving the gather to TensorCore to fuse it was already rejected at
  ~8x per descriptor, so the fusion has to go the other way.
- **`a` is "invariant to `P`", which is not the same as "irreducible".** The sweep held `topk` and
  live-row count constant, so anything scaling with those sits inside `a`. Whether part of the
  15.22 us tracks `topk` is untested and is the cheapest next question.
