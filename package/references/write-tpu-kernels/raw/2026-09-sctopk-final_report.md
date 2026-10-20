# SparseCore top-k — final report

**Target**: `vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` @ `486af81951cf3a202fb321c3144264437cb79266`
**Workload**: GLM-5.2-FP8 DSA indexer, long context — 128k/1k (`n_p=33280`, P=4) and 256k/1k (`n_p=32896`, P=8), k=2048
**Hardware**: TPU v7x (Ironwood), SparseCore, jax/jaxlib 0.10.2, libtpu 0.0.44.1
**Status**: complete. One candidate, measured whole, correctness verified by named jobs.

---

## 1. Result

`final/composite_full_stack.diff` — **867 lines (+592/−45), 5 files**, `git apply --check` clean at
`486af819`. The applied `sparsecore_topk.py` md5 `8ff7cb06c931fc0275b2a61cef4f6855` equals the
measured arm's source: **the shipped patch is the measured arm.**

Whole two-stage call, `dsa_like`, **full rows — the production setting** (see §4):

| cell | incumbent | +`cvt` fusion & padded scatter | **+`emit_pairs` (ship)** | +`payload_stage2` (flagged off) |
| --- | --- | --- | --- | --- |
| 128k decode (64 × 33280) | 119.1 µs | −15.58% | **−22.57%** | −28.53% |
| 256k decode (64 × 32896) | 127.7 µs | −17.45% | **−23.56%** | −28.63% |
| 128k prefill (4096 × 33280) | 5128.5 µs | −19.77% | **−25.36%** | −26.48% |
| 256k prefill (4096 × 32896) | 4969.1 µs | −20.09% | **−26.05%** | −26.23% |

[measured: j-721cbf97-e589-44f4-b138, replicated by j-8b0db944-bf26-4e44-a075 — 212 matched units
agreeing to 0.16% median and 0.08 pp on deltas]

Ragged rows do better, not worse: at `U(0,n_p)` −34.3…−56.3%, at `U(0,4096)` −62.7…−76.2%.

**Cross-distribution spread collapses from +40.0% to +4.2…+6.3%.** The incumbent is 40% slower on its
worst distribution than its best; the stack very nearly removes that dependence.

---

## 2. What the patch does — three levers

**H16 — padded scatter stride.** `addr = lane_iota * NUM_BUCKETS + bucket` with `NUM_BUCKETS = 256`
and **32 VMEM banks** puts all 16 lanes of a shared bucket on **one bank**, serializing the histogram
scatter 16-way. Stride 257 makes lane `l`'s address `≡ l (mod 32)`, so 16 lanes hit 16 distinct banks
regardless of bucket. Applied at all four radix levels. **+64 B per subcore.** Worth −12.98% to
−16.05% whole-call on its own.

**H4 — `cvt` fusion.** The standalone monotone-key conversion pass folds into the L0 histogram
traversal, which already walks the same words under the same guard and loop bound. One full-width
pass deleted.

**H5 — `emit_pairs`.** Stage 1 emits `(key, index)` pairs, deleting one of the two inter-stage XLA
gathers. Proven engaged per-cell: HLO gathers/invocation go 2 → 1 (→ 0 with `payload_stage2`) in
496/496 units.

**`payload_stage2`** ships in the same PR **behind its existing env flag, default off**: a further
−2.4…−6.0 pp at decode (−10.9 pp at short rows) and −0.12…−0.43 pp at prefill. Real at decode,
negligible at prefill, no VMEM cost over the shipping arm.

VMEM worst case **35.02% pooled** (45,904 words/subcore of the 2,097,152-word core ledger) against
the incumbent's 33.435%. The +1.59 pp is entirely `emit_pairs`' paired buffer; H16 costs +0.012 pp.
Zero spills in 248 units (advisory — the counter is a text search of the compiled module).

### Composition

Multiplicative to within **1.8 pp, always slightly better than multiplicative** (−1.22…−1.84 pp
versus probe 08's standalone `emit_pairs` at matched shapes). Against 30 cells of predictions frozen
before the run: median −3.97 pp, and **26/30 conservative** — the stack beat its own forecast.

The four large misses were missing *inputs*, not interactions, and both are explained:
`normal_positional_decay` was never measured for H16 by any earlier probe (it is the incumbent's
pathological distribution — 6780 vs 4852 µs — where H16 alone is worth −32.6%), and `U(0,n_p)` at
b·p=64 was scaled by a factor that assumed `emit_pairs` shrinks with batch, when the deleted gather
is fixed-size `p·k_p` and is therefore a *larger* fraction at small batch and short rows.

**A denominator subtlety worth recording, because the campaign predicted it backwards.** H16 does
*not* dilute on the whole call the way stage-1-only levers do: `PRIV_STRIDE` is kernel-wide, so
stage 2 gains too — it saves 15.0 µs whole against 12.0 µs direct, an effective dilution of 1.25, not
the 1.56 predicted from the call-size ratio. `emit_pairs` is the exact converse: **+0.04% on stage 1,
−8.28% on the whole call**, because it deletes work that lives *between* the stages. Measure the
whole call.

---

## 3. Correctness — verified, with positive controls

**The diff's own test suite passes on the pod**: `75 passed, 14 warnings in 18.38s`
[j-8fbfe53d-3cbd-446d-bcfc], run against composed arm 2, all 5 md5 gates matched, pod import check
reading `{PRIV_STRIDE: True, emit_pairs: True, payload_stage2: True, cvt_pass_deleted: True}`. This
is the gate that covers the `streamindex_topk.py` / `mla_attention_op.py` / `envs.py` plumbing that
no kernel oracle touches.

**Oracle** [j-ad94546c-f9d9-4c50-97a5]: `CORRECTNESS PASS` on all 4 shards, rc=0. **616 real-arm
units, 0 oracle failures, max `rows_differing` = 0.** Covers 4 distributions × {full, `U(0,n_p)`,
`U(0,4096)`} × both `write_empty_rows` × both widths, in-range `−inf` via explicit mask, all-negative
/ all-positive / all-equal rows, and `eff_len ∈ {0, 1, 16, k−1, k}`.

**The gate carries a fault-injection positive control, and that matters more than the pass here.**
`monotone_key` is an **involution**, so a double conversion yields the *raw bits*, not garbage — and
on full rows of a zero-mean distribution the top-2048 is entirely positive, where raw int32 order
agrees with float order. Measured on-pod:

> full oracle caught the conversion fault in **0/4 cells ← BLIND**
> `U(0,n_p)` … **0/4 cells ← BLIND**
> `U(0,4096)` … **4/4 cells**

Confirmed live by an `x_double_cvt` arm: discriminating cells **24/24 caught**, non-discriminating
**16/20 missed**. The shipping arm passes every cell the fault fails. That is what establishes
"converted exactly once"; a bare PASS would not have.

---

## 4. Which number applies to your workload

**Full rows.** `config.json` assumes `row_lengths == n` and the e2e config is 128k/c16, so every
decode row is near-maximal context. **−22.6% (128k decode) and −23.6% (256k decode) are the
conservative floor.** A mixed-length trace sits nearer `U(0,n_p)` and is worth roughly 2×.
`U(0,4096)` is unreachable at 128k and is in the matrix only because it is the sole setting whose
oracle can see a conversion fault.

The stack has **no raggedness weak spot**: H16 tracks the work (−15.58% → −13.16% → −1.61% as rows
shorten) while `emit_pairs` deletes a fixed-size gather and therefore grows as rows shorten. The two
are complementary across the whole range.

---

## 5. The literal request, answered

> "remove DMA and replace it with `emit_pipeline`"

**Measured null. ≤0.5%, not worth landing on its own.** [j-0852d569-16b9-4796-8139]

All three DMA treatments land on top of each other: `emit_pipeline` −6.21%/−4.97% (full|ragged),
hand-rolled double buffer −6.17%/−5.17%, traced DMA extent −6.15%/−4.74%. `emit_pipeline` versus a
hand-rolled double buffer is **−0.04%/+0.19%** — indistinguishable, which closes a long-open
SparseCore question as a null in its own right. The −6.2% those arms shared was **entirely the `cvt`
fusion riding along inside each of them**, and that fusion is in the shipping patch.

The load is not the bottleneck: stage 1 is compute-bound. Two further measured reasons the lever
cannot pay — **raggedness is governed by the longest row in a 32-row wave** (`E[max of 32] ≈ 0.97n`,
so ragged rows measured −11%, not −50%), and **SC streaming bandwidth is 338 GB/s per chip**
aggregate over all 32 subcores [j-74a24183], so there is no per-core escape hatch.

---

## 6. Rejected, with the number that rejected each

Every one was believed plausible before it was measured.

| lever | verdict | mechanism |
| --- | --- | --- |
| **DMA → `emit_pipeline`** | **null, ≤0.5%** [j-0852d569] | not the bottleneck; the −6.2% was the `cvt` fusion |
| **survivor compaction (H3+H13)** | **+6.53% median, slower 14/14** [j-e485902b, j-abfde0e5] | `compact_body` costs **~3× an emit pass** — two `store_compressed` + two popcounts per chunk. The `cnt` model was right; the per-chunk cost model was wrong |
| **adaptive radix window** | **slower in 11/12 cells** [j-c8205572] | levels *do* fall 3.06 → 2.06 and a level *is* worth 11.5–14.5%, but the machinery costs **+11…+18%** and the per-subcore tail eats the rest |
| **narrow radix digit (16×8)** | **+20.5%** [j-2c5aaa81, j-dc958adc] | active levels go 3.00 → 5.75; every active level re-scans full width |
| **wide radix digit (12-bit)** | **rejected analytically** | 4096 buckets fits VMEM (~50% pooled) but bookkeeping scales with bucket count: 16 × 4.3% ≈ 69% of the call |
| **tokamax SparseCore top-k** | **1.48× slower dense, 1.95× ragged** [j-970c76e0] | forced 16-way cooperation above N=4096 → ~1200 `subcore_barrier`s/call; no ragged support; rejects `n_p=32896` at trace time |
| **move to TensorCore** | **SparseCore wins 8.70–22.97×** [j-0499e0a9] | algorithm-bound in **k**, not bandwidth-bound; at k=2048 XLA top-k runs at 4.4–10.6 GB/s effective |
| **bookkeeping reduction** | **2–6% ceiling** [j-2c5aaa81] | 4.3% at L0, 5.6% across four levels — the residual was the scatter, ~8× larger |
| **lane-major scatter transpose** | **a wash vs padding** (+0.3 µs) [j-764fb619] | the transpose fold buys nothing; all its win was the scatter alone |

---

## 7. Carried — four items, each its own PR, none of them this one

1. **`vec_at(len_vmem, row_c)` reads 15 words past the end of `len_vmem`.** The `b_pad`-word buffer
   lacks the `LANES` lip that `glob_vmem` and `rowbuf_vmem` both have — a real **out-of-bounds VMEM
   read** under `disable_bounds_checks=True`. Pre-existing, predates every lever here, present
   identically in every arm, so it biased no delta. Harmless in practice (only lane 0 consumed;
   poisoning the overhang left 1592 windows unchanged). **A 16-word fix.**
2. **`instruments/unit_test.py` @ `f567d98` accepts `write_empty_rows` at `check()` line 158 and never
   reads it**, so it flags the unwritten slots of zero-eligible rows on the *untouched incumbent*. It
   voided a correctness job before being caught, and was worked around locally
   (`neutralize_unwritten_rows`, fired on 27/792 units, 2 rows each). The file is read-only and
   shared — **its owner must fix it.**
3. **The harness quarantine criterion is mis-specified.** It gates the *dispersion of the rounds*
   while the estimator is the *median* of those rounds, so adding rounds makes it fire more — 61/124
   cells (49.2%) tripped it on the p10–p90 conjunct alone, none on the ratio. Nothing was dropped;
   two full replicates were substituted as the error bar instead, agreeing to 0.16%. It should gate
   the uncertainty of the median (bootstrap CI or replicate agreement). **The ratio conjunct is sound
   and should be kept.**
4. **The correctness cell set was the campaign's weak link, not the kernel.** Two independent probes
   found real bugs invisible outside straddle-zero rows; one stated it would otherwise have shipped a
   wrong kernel with a green oracle. Empirical recipe correction: `U(k,3k)` straddles zero on only
   **8/16** rows on device; **`U(k,1.5k)` on 16/16**. Short rows, an in-range `−inf` mask, and a
   fault-injection positive control should be mandatory.

---

## 8. Method note worth keeping

**A chained-invocation clock must barrier the loop CARRY, never the captured invariants.** One probe
reported a 2.8× phantom speedup because it barriered loop-invariant operands: XLA proved the kernel's
inputs unchanging, LICM hoisted the call clean out of the `fori_loop`, and 20 nominal invocations
executed **once**. No correctness guard can catch this — a hoisted loop is idempotent and still
returns the right answer. It was caught by a physically impossible sub-HBM-floor reading and
arbitrated with an xprof device-time arm; five numbers were retracted.

The durable defence, used by every probe after it: **count `tpu_custom_call` occurrences in the HLO
and assert they equal the expected multiple of M** (2M here — the two-stage call emits two per
invocation; 496/496 units), and derive per-call cost from an **M-sweep least-squares slope**, never
divide-by-M. `T(M) = a + sM` with `a ≈ 250 µs`, so a healthy raw `T_hi/T_1` is 13–50 — a hoist reads
~1.0–1.6. Separately, **never stack heavy cell-groups on one shard**: three saturated at ~100 ms for
every M while HLO counts still scaled correctly, which is contention, not a hoist.

Full lever-by-lever history including superseded reasoning is in `../WHITEBOARD.md`; the
generalizable hardware rules are in `../STRATEGY.md`.

---

## 9. Next

- **Promote arm 2 to a PR** and schedule the **128k/c16 e2e** — every number here is a kernel-level
  measurement, and the end-to-end share of this kernel has not been re-measured since the stack.
- Ship `payload_stage2` in the same PR behind its default-off flag with the decode numbers attached.
- Route the four items in §7 separately.
