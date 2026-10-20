# WHITEBOARD — SparseCore window-traversal residual

Target, incumbent, established facts and caveats: see `BRIEF.md`. This file is the design.

## 1. Derived geometry (step 1)

All from `[source: vllm-torchtpu/src/vllm_torchtpu/kernels/mla/sparse/dsa_gather.py @ 61b1c0b8]`
plus `[fact: hardware-facts.md § SparseCore]`.

| quantity | expression | value at `R=1, P=64` |
| --- | --- | --- |
| `row_subchunk_size` | `sc_info.num_lanes` | 16 |
| `_NUM_ROW_SUBCHUNKS` | module constant | 32 |
| `row_chunk_size` (window) | `num_lanes * 32` | **512 indices** |
| `num_cores` (in-kernel) | `axis_size(core_axis, subcore_axis)` = `2 x 16` | **32** — this is the wave |
| `block_size` | `row_chunk_size * num_cores` | 16,384 |
| total indices | `P * topk` = `64 * 2048` | 131,072 |
| **`num_blocks` (outer grid steps)** | `cdiv(131072, 16384)` | **8** |
| total windows | `num_blocks * num_cores` | **256** |
| live windows | `R * topk / 512` = `1 * 2048 / 512` | **4** |

**The wave is the outer pipeline's inner dimension.** Each outer step dispatches 32 windows, one per
vector subcore, in lockstep. `[fact: STRAT-SC-05]` — a step costs its **longest member**, so a step
containing even one live window costs a full live window regardless of how many of its other 31
members were predicated off.

## 2. The residual, and the three cost models that fit it

Recomputed from `BRIEF.md`: live-row movement is `2048 x 0.90 ns = 1.84 us`, measured is
**12.35-12.47 us**, residual **10.5 us** `[measured: j-49a5dfe7]`.

Three models divide that 10.5 us differently. **They are not distinguishable from the existing
evidence, and they imply different candidates:**

| model | per-unit cost | what compaction can remove |
| --- | --- | --- |
| **A. per-window** | 10.5 / 256 = **41 ns/window** | 252 of 256 windows -> ~10.3 us |
| **B. per-outer-step** | 10.5 / 8 = **1.31 us/step** | 7 of 8 steps -> ~9.2 us |
| **C. per-call fixed** | 10.5 us flat | **nothing** — investigation stops |

Model C is not idle: `hardware-facts.md` carries a **20-36 us per-`pallas_call` device intercept**.
It cannot apply at full size here — the report measures 3 `tpu_custom_call`s inside a 40.08 us/layer
slope, and the gather alone at 12.4 us is below one intercept — so the SparseCore intercept must be
materially smaller than the TensorCore figure. But "smaller" is not "zero", and **any part of the
10.5 us that is per-call fixed is unreachable by any lever in this campaign.**

**Job #1 exists to separate A/B/C and nothing else.** Everything downstream is conditional on it.

### The check that a byte-count model already fails

Neither obvious variable cost explains 10.5 us, which is why the fixed/traversal split must be
measured rather than assumed:

- index-stream DMA: `131072 x 4 B = 512 KiB` at 338 GB/s `[fact: hardware-facts.md]` = **1.5 us**
- pure lockstep: 8 steps, 7 of them all-skipped, 1 costing a live window = far below 12.4 us

So ~9 us is unattributed. That is the finding to localize, not a number to model.

## 3. The design this geometry exposes — prefix truncation, not compaction

I entered this expecting the lever to be **compaction**: a scalar-prefetch table mapping live windows
to output rows. **The geometry says that is unnecessary.**

The index stream is `(token, topk)` flattened, so token `t` owns `[t*topk, (t+1)*topk)`. Bucket
padding is at the **end** of the token axis — confirmed by the incumbent pinning `kv_lens` with
`arange < cu_q_lens[distribution[2]]` `[source: kernel.py:744-758 @ patched]`. Therefore:

> **The live index region is already a dense prefix, `[0, R*topk)`.**

A dense prefix needs no compaction table and no output indirection. Shortening the outer grid to
`cdiv(R*topk, block_size)` leaves every surviving step's `b` unchanged, so
`out_row_base = (b * num_cores + core_index) * num_row_subchunks` keeps its meaning and the untouched
output rows are untouched for exactly the reason the incumbent's `pl.when` skip is already safe
(`load_bkv` zeroes at/past `kv_len`).

At `R=1, P=64` that is **8 outer steps -> 1**. It subsumes model B entirely and most of model A.

**This is strictly simpler than the compaction design in `BRIEF.md` lever 2, and it is the same
primitive `[fact: STRAT-MLA-08]` already proved lowers** — a traced grid length for `emit_pipeline`.
The untested half is that MLA-08 proved it on the **inner** pipeline; this applies it to the
**outer** one.

Interior dead windows (a live token whose own `kv_len < topk`) are *not* a prefix and are left to the
incumbent's existing per-window `pl.when`. The two compose: prefix truncation removes whole steps,
the existing predicate handles the ragged interior.

### A second, independent argument against compaction

`[fact: STRAT-SC-01]` prices a SparseCore compaction pass by its **cross-lane serializing ops**, not
its trip count. Real compaction here would need `plsc.store_compressed` /
`plsc.all_reduce_population_count`, measured at **~3x a masked-scatter chunk**, and that campaign's
survivor compaction lost **+6.53%, slower in 14/14 cells**, against a trip-count model that said it
should win by two passes.

Prefix truncation uses **neither primitive** — it changes a grid bound and moves no data to do it.
So the two candidate designs are not "simple vs general": the general one carries a measured
mechanism for losing. Compaction is demoted to a fallback, to be reconsidered only if Job #1 returns
model A *and* the interior (non-prefix) dead windows turn out to dominate.

### Carrying the scalar in

`dsa_gather` needs `live_indices` as a traced scalar. The caller already has it:
`cu_q_lens[distribution[2]]` minus the chunk's `start_pos`, clamped to the chunk, times `topk`.
Pass it down rather than deriving it in-kernel. A whole chunk past the live region yields a traced
grid of **0** — which is the `skip_mode="grid"` construct MLA-08 measured, so it is the same lowering.

## 4. Roofline gate — deferred, with cause

Step 1 says stop if baseline >= 80% of floor. **The HBM floor cannot arbitrate this.**
`[fact: STRAT-MLA-07]` — the ratio to that floor *widened* 3.8x -> 8.2x -> 12.8x along the
incumbent's path, which is that floor disqualifying itself as a model of what binds. Reading "12.8x
from floor" as "87% headroom" is precisely the error MLA-07 was written to stop.

The floor that governs here is **traversal + per-call fixed**, and its value is `a + b * live_steps`
from Job #1. The gate is therefore evaluated *after* Job #1, against:

```
floor = a + b * ceil(R*topk / block_size) + c * (live rows)
```

Gate: if measured 12.4 us <= 1.25 x floor, **stop and report** — the residual is fixed cost, not
traversal, and no candidate in this campaign reaches it.

## 5. Job #1 — the discriminating probe

**Core question.** Is the 10.5 us residual per-window, per-outer-step, or per-call fixed?

**Method.** Hold live rows constant at `R=1` (2048 live rows, 4 live windows) and sweep `P`, which
moves `num_blocks` linearly and leaves the live work untouched. Fit `t = a + b * num_blocks`.

| arm | P | num_blocks | windows | live windows |
| --- | --- | --- | --- | --- |
| 0 | 16 | 2 | 64 | 4 |
| 1 | 32 | 4 | 128 | 4 |
| 2 | 64 | 8 | 256 | 4 |
| 3 | 128 | 16 | 512 | 4 |

**Reading it.** `b` is the per-outer-step traversal cost and `a` is per-call fixed + live movement.
- `b ~ 1.3 us`, `R^2` high -> model B, prefix truncation ceiling is `b * (num_blocks - live_steps)`
- `b ~ 0` and `a ~ 12 us` -> **model C, stop the campaign**
- The same fit distinguishes A from B only if window count is varied independently of step count,
  which requires moving `_NUM_ROW_SUBCHUNKS` — a **module-level global** `[fact: STRAT-MLA-10]`.
  Deferred: A and B imply the same candidate, so the distinction does not change what is built.

**This also tests the `P`-linearity claim made to the user** (that the residual shrinks ~4x from
`P=64` to `P=16`), which is currently an unmeasured inference.

**Method constraints, from the wiki, all load-bearing here:**
- `[fact: STRAT-SC-03]` + `[fact: STRAT-MLA-04]` — chained clock must barrier the loop carry, and for
  a gather the carry must reach **the indices**, not the values. Reuse the incumbent's solved pattern
  from `<campaign>/instruments/` (read-only).
- `[fact: SKILL.md step 5.3]` — build every jitted callable **above** the timing loop; 4 arms x 4
  P-values is 16 distinct programs and each is a ~1 s compile.
- `[fact: STRAT-SC-13]` — gate the estimator's uncertainty, not the sample spread.
- Hold rep 0 out of the median; it pays the cold compile in every arm.

## 6. Whiteboard-Freeze checklist — NOT yet frozen

- [x] Exact shapes & dtypes locked — §1, derived from source at the pinned sha
- [x] Hardware facts verified — wave=32, window=512, SMEM 4 KiB, intercept 20-36 us
- [ ] **Roofline / gate evaluated** — blocked on Job #1 (§4)
- [ ] **Cost model established** — blocked on Job #1 (§5); model C ends the campaign
- [x] Physical working set — unchanged from incumbent; truncation allocates nothing new
- [x] Pipeline flow mapped — §3; outer grid shortened, inner pipeline untouched
- [ ] Reference idioms mined — outer-pipeline traced grid on SparseCore is the unproven construct
- [x] Hypothesis pre-formulated — §5

**Do not write anything under `variants/` until Job #1 returns.**
