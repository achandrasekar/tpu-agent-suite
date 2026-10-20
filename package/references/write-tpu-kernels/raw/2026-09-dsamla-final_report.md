# DSA/MLA decode — final report

**Block.** `vllm-torchtpu/src/vllm_torchtpu/kernels/mla/sparse/` (DSA gather + sparse MLA attention),
GLM-5.2-FP8, long context **128k/1k**, TPU v7x `2x2x2` (8 chips / 16 TensorCores, DP=16, one
sequence per rank). Checkout `61b1c0b8`, jax/jaxlib 0.10.2, libtpu 0.0.44.1.

**Outcome.** One composed candidate, two call-site tunings, **no change to any kernel default**.
Decode **1.96×–3.29×** per layer depending on bucket size; prefill **neutral**. Correctness closed
at 52/52 with real rows bit-identical to the unmodified kernel, and the change verified live on the
served entry point.

---

## 1. The premise, and what happened to it

The investigation was opened on a reported figure: *unchunked DSA 1.69 ms vs 16-chunked DSA
2.14 ms in decode, with DSA binding the whole DSA/MLA path.* The user asked for that number to be
**reconstructed rather than taken on trust**, so the baseline here was built from scratch by
sweeping the real decode geometry.

**The reconstruction does not reconcile with the quoted figures, and that is unresolved.** Measured
stock decode is **131.89 µs/layer** at a `P=64` bucket and **65.11 µs/layer** at `P=16`. Against 78
layers those project to ~10.3 ms and ~5.1 ms per step — 3–6× larger than either quoted number. I did
not find the configuration that produces 1.69/2.14 ms, and I did not reproduce the 16-chunk *penalty*
as a standalone effect.

What this does and does not invalidate:

- **The ratios stand.** Every speedup below is ship-vs-candidate measured in one process, interleaved,
  on identical inputs. A baseline that is offset from the user's does not affect a ratio taken
  against itself.
- **The absolute per-step projections do not stand** as a prediction of end-to-end gain. They are
  upper bounds from device-time slopes, not measured serving time.
- **The original framing was partly wrong, and usefully so.** The cost is not "loss due to chunking"
  to be recovered by unchunking. It is that a *padded* bucket makes the gather do work proportional
  to the bucket while only the live rows matter — see §2. Unchunking would not have fixed it;
  chunking is not what costs.

Reconciling the 1.69/2.14 ms figures needs the geometry they were taken at (bucket size, `index_topk`,
concurrency, and whether the figure is per-step or per-layer). That is the first thing to ask before
this work is projected onto a serving target.

## 2. What was changed, and the mechanism

Two independent tunings, both passed as **keyword arguments from the single dispatcher**, leaving the
sparse kernel's defaults untouched for any future caller ([[STRAT-MLA-10]]):

**(a) `skip_padding_gather=True` — skip gather windows holding no live row.** A token bucket is
padded up to a compiled size. Each padding row causes `index_topk = 2048` cache rows (640 B each) to
be gathered, which the attention then multiplies by zero. The gather runs *upstream* of the mask and
cannot know the rows are dead. At decode `R=1` in a `P=64` bucket, **98% of the gather is provably
dead work**. Skip granularity is the traversal window, so it is structurally a no-op when `R == P`,
which is why it is safe unconditionally rather than behind a shape branch. Promoted as
[[STRAT-MLA-01]]; the enabling lowering facts are [[STRAT-MLA-08]] and [[STRAT-MLA-09]].

**(b) `attention_kernel_batch_size=4` — finer grid truncation.** This knob is **not** a tiling
parameter. The attention grid stops at the last batch holding a real token, so the value is a
*truncation granularity*: smaller truncates padding more finely but walks more, narrower grid steps.
Grid steps per chunk = `chunk_size / batch`, and `gather_and_attention_chunk_size` defaults to 64.
Two opposing terms give an interior optimum at **4**. Promoted as [[STRAT-MLA-02]].

## 3. Result

Final composed A/B, **job `j-390366e4-b0d9-406c-8be8`** — one process, arms interleaved
(ship, cand, ship, cand) so drift cannot masquerade as a win, all callables warmed before timing.
Device-time slope from a chained-K regression; `tpu_custom_call` per chained call = 3.00 (`P≤64`)
and 5.00 (`P=128`), matching the static kernel count ([[STRAT-MLA-04]]).

| cell | ship µs/layer | cand µs/layer | speedup |
| --- | --- | --- | --- |
| decode `R=1` of `P=16` | 65.11 | 33.15 | **1.96×** |
| decode `R=1` of `P=64` | 131.89 | 40.08 | **3.29×** |
| prefill `R=64` of `P=64` | 185.15 | 180.07 | 1.03× — see below |
| prefill `R=128` of `P=128` | 304.52 | 304.78 | 1.00× |

The decode figure reproduces an independent job and script (`geom_final`: 131.90 → 40.16) to within
**0.2%**.

**Prefill is neutral, not a win.** The 1.03× above is inside the noise of the baseline itself: three
independent SHIP readings at `P=R=64` span **179.38 / 185.15 / 187.34 µs** — a 4.4% spread, wider
than the 1.03% delta. Taking the *lowest* SHIP reading (most favourable to the incumbent) the
candidate is 0.4% slower. The defensible claim is therefore **"prefill is not regressed beyond
measurement spread"**, and the 1.03× should not be quoted. The three candidate arms agreed to <1%,
so the spread is in the baseline arm, not the change.

## 4. Correctness

**Job `j-b3a169bf-dee1-4b17-bbab` — 52 PASS / 0 FAIL** at production geometry
(`row_chunk_size = 512`, the shipping `(32,4)`):

- Real rows **bit-identical** (`exact`, not tolerance) to the unmodified kernel at every batch value.
- Written padding exactly zero; rows the grid never reaches equal the input `q` bit-for-bit.
- Oracle 0.4–0.8 ulp against a 64-ulp bar, with the stock kernel scoring identically.
- Padding boundary tracks `_largest_divisor` at batch 16/8/4/2/1 **and across a chunk boundary**
  (`P=128` is two chunks of 64, so the second chunk's `batch_end` is computed against a non-zero
  `start_pos` — the case most likely to be wrong and the one a single-chunk test cannot see).
- Both lowering spellings (`when`, `grid`) verified as explicit arms.

**Served-path verification — job `j-9a7320bc-4d6b-4d51-8e64`, 8/8 PASS.** Every measurement above
called `sparse_ragged_paged_attention` directly; the shipped change is one level up in
`dispatch.ragged_paged_attention`, which had never been executed. This closes that gap: the
dispatcher runs, its real rows are bit-identical to the stock kernel, its output is bit-identical to
the *tuned* kernel over all rows, **and** tuned differs from stock in the padded region — the last
being the control that makes the others meaningful, since keywords silently dropped would make the
first checks pass vacuously.

That smoke test failed on its first run (`j-474c825a`) with non-finite output. The cause was my
harness, not the change: it built the fp8 KV cache from **random bytes**, and random e4m3 patterns
include NaN. The correctness suite already carried a comment warning of exactly this; I had copied
the cache builder from the suite's *gather* arms, which compare raw bytes and can afford any bit
pattern, into a test that does arithmetic. Fixed by quantizing real values, and the comparator is now
guarded by an explicit finiteness check so a future NaN localizes instead of failing three checks at
once.

## 5. Rejections and nulls

| rejected | why | evidence |
| --- | --- | --- |
| `attention_kernel_batch_size=1` | Wins decode a further ~20% but **regresses prefill ~11%** — a full chunk costs 64 grid steps instead of 4. Looked like a win at both ends when swept at `P=16`, which understates the penalty 4×. | `j-9773d307` corrects `j-7b13c6ce` |
| Gather window geometry retune (`_NUM_ROW_SUBCHUNKS` 32→8) | **+0.81 µs/layer decode (+2.6%)** against **−23.5 µs/layer at `R=64` (−13%)**. It is a *module-level global*: it retunes the gather for every caller and shape, including ones never measured. A 2.6% gain on one shape does not buy a global retune. | `j-9773d307` → [[STRAT-MLA-10]] |
| TensorCore `pallas_call` replacing the SparseCore gather | ~**8× per descriptor**; issue cost is architectural, and no blocking or layout change closed it. Rejected before composition. | `j-23e1bf57` → folded into [[STRAT-SC-09]] |
| Porting `csa_gather` / DSv4 | Needs `compress_ratio == 4` and `kv_lora_rank=448`; GLM's is 512. Separate kernel, no `SKIP_INDEX` support. Out of scope. | source read |

## 6. The roofline gate — it does not close, and that is the finding

Compute is irrelevant here (0.25 µs at `R=1`). The ratio to the HBM-traffic floor **widens** along
the optimization path: **3.8× → 8.2× → 12.8×**, as bytes fell 8× while time fell 2.4×. That widening
is the signature of fixed-cost binding, not of remaining headroom ([[STRAT-MLA-07]]).

Located directly: the machine retires a 640-B indirect row every **0.90 ns** in the stock arm but
every **5.8–6.0 ns** in the skipped one. So roughly **10 µs/layer is window traversal that the skip
predicates off rather than removes** — larger than everything won here, ~0.78 ms/step. The `grid`
spelling does not capture it either (12.35 vs 12.47 µs), so it is not an artifact of the predicate.

**Named as the next investigation, not claimed.** Reducing SparseCore window-traversal cost is a
separate piece of work and was declared out of remit.

## 7. Carried

- **`hardware-facts.md:590`'s 338 GB/s is not a ceiling at this access width** — five sightings at
  509–709 GB/s on 640-byte indirect rows. **Deliberately not applied** to the shared reference: three
  targeted jobs outrank a by-product measurement, and the absolute figure rests on byte accounting
  that was never settled. Flagged as [[STRAT-MLA-06]] and cross-referenced from [[STRAT-SC-08]] for
  someone to confirm on purpose.
- **The ~10 µs/layer SparseCore traversal residual** (§6) — the largest remaining lever found.
- **The 1.69/2.14 ms premise is reconstructed, not reproduced** (§1). Needs the originating geometry.
- **No end-to-end benchmark was run.** Every number here is a device-time slope. The ×78 projections
  are upper bounds.

## 8. Files

- Patch: `<campaign>/final/kernel_patch.diff` — 3 files, 194 insertions, 16 deletions, against
  `61b1c0b8`. Nothing committed.
- Instruments: `<campaign>/instruments/` (`unit_test.py`, `final_candidate.py`, `geom_final.py`,
  `prefill_batch.py`, `dispatch_smoke.py`, driver `run_floor.sh`).
- Promoted wiki entries: `STRAT-MLA-01` … `STRAT-MLA-10`, plus edits to `STRAT-SC-08` and
  `STRAT-SC-09`.
