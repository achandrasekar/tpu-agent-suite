# STRATEGY — run log (dsa-mla-256k-c16)
- 2026-09-25 Job #1 j-0cb738d5: bench crashed on compiled.as_text() (jaxlib 0.10.2 vs libtpu 0.0.44.1 HLO proto: async-update arity RET_CHECK). Fix: read post-opt HLO from XLA_FLAGS dump. hardware: v7x; method.
- 2026-09-25 Job #1b j-0d1e2849: baseline recorded (whiteboard §1.6). Chained-clock hoist check flags a loop-invariant XLA gather (AssumeGatherIndicesInBound) that is legitimately hoistable — the check must distinguish invariant non-kernel ops from hoisted kernels.
- 2026-09-25 probe 02 j-1e32fec6: V1 SC/TC overlap via chunk_tokens (config). indexer c131072 chunk128 −28.8%, chunk256 −25.8%, chunk512 −17.9%;
  c32768 chunk256 −7.4% (pred −10…−20 refuted low); c261120 chunk256 −26.8% (pred −30…−45 refuted low). Identical top-k (overlap 1.0000).
  mechanism: SC top-k (~4 ms, C-independent) stays the binder; overlap can hide at most TC scoring (0.66 ms at C=32k, ~5 ms at 261k) minus
  per-chunk SC fixed cost (smaller chunks → more SC launches). lever: inter-unit concurrency; hardware v7x; phase prefill.
- 2026-09-25 probe 02: decode num_kv_pages_per_block=16 → VMEM OOM at compile (74.62M of 63.94M, 64 MiB scoped + 10 MiB spills). null on axis
  "decode pages=16"; compile-time limit, not timing.
- 2026-09-25 probe 02b j-abac8714: V5 decode num_kv_pages_per_block 2→8: indexer decode −14.0%/−13.4% (175.2→150.6 µs) [pred −13…−22 ✓];
  pages 4: −13.2%/−12.4%. mechanism: 128 steps of 512 KiB DMA waited per step (~1 TB/s) → 33 steps of 2 MiB; per-step fixed cost amortized.
  V1 chunk128: c32768 −7.4%, c261120 −30.4% (≥ chunk256 on every prefill cell). chunk decode: 0.0% (unchunked, as predicted).
- 2026-09-25 probe 04 j-7a60fc19: V3 decode real-row bound: attention decode −62.6%/−62.8% (62.6→23.4 µs) [pred −35…−60, beat range].
  Interaction: v3a (live gather only) +12.3%, v3b (batch-1 attention only) −39.8%, v3 −62.6% → levers interact (sum of singles −27.5%);
  mechanism: the atoms=1 live gather only pays once attention stops reading all 16 gathered rows; alone it adds per-batch SC overhead.
- 2026-09-25 probe 04: V4 REJECTED. prefill_c131072 v4 −1.4%, v34 −2.1% (CIs overlap; null on axis "module time, C=131072, SC gather chain
  ≈1.6 ms ‖ TC chain ≈1.45 ms" — TC body savings are hidden under the SC gather). decode v4a (two whole-body pl.when gating) +26.8%/+29.4%,
  v4 +26.8% → refuted refinement "0±3%". V4b (PV bf16 × 512) −1.8% decode/+1.2% prefill = null. V4b bit-identical (f32 DEFAULT = bf16 pass).
- 2026-09-25 probe 05 (A j-2dcfbed7, B j-2cfd6677): stage-A composite (21 idx + 78 attn layers) per step: decode 8.32–8.42 → 4.92–4.98 ms (−40.8…−41.3%),
  prefill c131072 −22.4/−21.1%, c32768 −27.5/−24.2%, c261120 −17.3/−17.6%. All frozen predictions in range. Ladder: R1 V2 −21.0/−19.8,
  R2 +V1 −43.6/−43.1, R3 +V5 −43.7/−43.0 (V5 decode-only ✓), LOO(no V2) −28.0/−27.3 → V2 worth 15.7 pp on top of V1 vs 21 alone: levers
  interact (both act on SC top-k's critical path), report the stack. Correctness: 15/15, faults wrapping candidate caught on all production cells.
- 2026-09-25 critical-review (preflight) PASS WITH CONDITIONS. Corrections: "all frozen predictions in range" is false in one row — B decode_first
  V3 −59.8% vs frozen −60…−65 (0.2 pp out). probe-05 replicate rule (≤2 pp) fails for c32768 composite (3.35 pp) and v3 c32768 attention (2.2 pp):
  cause = ±3% noise in unchanged prefill attention (V3 bit-identical at prefill); indexer-only c32768 Δ −25.5/−26.7%.
  Job A first pod 5xz9w killed mid-compile, rerun on ddg25 (A numbers from ddg25). V3 NaN-leak fault (g) wraps xla_ref, not the candidate.
- 2026-09-25 baseline e2e j-6d5dff1e: 90.75 out tok/s, TTFT 136.85 s, TPOT 42.63 ms, ITL 38.55. In-model xprof: SC top-k stage1 fully exposed
  (3.46 ms/layer at C=262k; TC idle); indexer K-cache insert `.at[].set` on u8[N,256,4,256] → XLA reshape whole cache to 2D + back = 777 µs/layer
  (16.3 ms per prefill step, at every C) — layout artifact; MLA q/out glue ≈305 µs/layer (≈24 ms/prefill step). Microbench vs in-model: idx 9.0 vs 9.0 ms
  (C=262k), MLA 1.9 vs 1.58 ms/layer (harness adds q copy/pad).
- 2026-09-25 B1 j-88dbb2ef: indexer K insert as Pallas TC kernel (aliased cache, 4-token-group DMAs): T=1024 792 → 35 µs/call (−95.6%), byte-exact 12/12,
  fault caught 12/12. T=16 decode 16.8 → 26.6 µs (+58%, refuted "≤ incumbent"): XLA scatter at T=16 is already in place on the 4D layout; the
  whole-cache u8 relayout (T(4,128)(4,1) layout → [435200,256]) appears only in the large-bucket program. mechanism: XLA scatter on a
  sub-32-bit tiled array falls back to a relayout of the whole operand at large update counts. → dispatch per static token bucket.
- 2026-09-25 incumbent e2e replicate j-2bbeb4a4: 91.04 tok/s, TTFT 136.24 s, TPOT 42.62 ms (vs j-6d5dff1e 90.75 / 136.85 / 42.63) → run-to-run spread ≈0.5%.
- 2026-09-25 probe 06 e2e stage A j-34b7fbc9 (first submit j-6a4e91bd died at device init, PjRtClient not initialized; resubmitted verbatim):
  102.30 out tok/s vs 90.75/91.04 (+12.5%); TTFT 120.24 s vs 136.85/136.24 (−11.9%) [pred −8…−13 ✓]; TPOT 38.92 vs 42.63/42.62 (−8.7%)
  [pred −3…−7, beat]; throughput pred +6…+11 beat. Code-ran proof: sc_topk_v2_*, StreamIdxTC-d-bq_1-bkvp_8, 8× chunked stage-1 per indexer call,
  sc_native_sc_gather_a1_…_live, MLA-…-bz_1 in rank0/rank8 traces. Noise (2 incumbent runs) ≈0.5%.
- 2026-09-25 glue attribution j-34b7fbc9 rank0 run 8141 (1024-token step, ~261k): MLA glue 25.0 ms/step; indexer glue 40.8 ms/step (XLA K insert 16.0,
  SC waits 15.0, stage-2 remap gather/reshape/select 13.1 of which overlaps the waits, empty StreamIdx launches 1.2). New: DP-padded mixed step
  (1 real token in T=1024 program) — native SC gather still gathers all 16×64×2048 rows, TC waits 1016 µs/layer → 79 ms per such step → B5.
  WHITEBOARD.v3 frozen (9af563dc…): rungs B4 (stage-2 global-index payload), B5 (live gather for all chunk sizes). Instruments v5: +mixed_decode_t1024 cell.
- 2026-09-25 B5 j-8baa6ce4: live gather for 64-token chunks (atoms 16, num_live_indices=(batch_end−start)·topk). mixed_decode_t1024 attention
  1426.3 → 332.9 µs (−76.7%) [pred −70…−90 ✓]; prefill_c131072 −0.5% (null ✓); decode HLO byte-identical (null by construction). 16/16 accuracy,
  native=0; attention faults wrapping b5 all caught. Remaining 333 µs on mixed unprofiled.
- 2026-09-25 B4 j-ea90ea93 (acc) j-6eb9ee98 (bench): stage-2 carries global indices as payload (stage 1 writes local+j·w in-kernel), remap gather deleted.
  vs stageA_port: c131072 −5.1% (pred −8…−12 refuted low: in-model remap was 626 µs exposed+contended, isolated 190 µs), c261120 −2.2%,
  decode −23.9% (beat), mixed_decode_t1024 −51.6%. Exact (bit-identical). Found pre-existing bug: sparsecore_topk b=1,n=264192 infinite recursion
  (stage 2 at n=65536 picks p=32, p·k_p==n) — unreachable at min bucket 16; carried.
- 2026-09-25 B3 KILLED (class: refuted; j-de6820cb bench, j-8023c2f5 static LLO). Key-sub-tiled scoring, bit-exact, spill stores 11,681→2,855 (kt128) but
  bundles 9,524→9,580 and slope +0.4% (c131072); kt256 −7% bundles yet slope −0.6%; kt512/tile store regress +4…+6%. mechanism: spills were
  co-issued in free VST/VLD slots; VALU (≈24.5k relu/×w/add = 3 ops/element) binds the step → removing spills cannot move the clock (null on axis
  "prefill scoring, stage-A wiring, KT∈{128,256,512}"). Also kt128 re-latches the k tile per head (vmatpush3.f8e4m3.xpose ×2048). Cheap arm-kill on a
  falsified premise (exempt from exit-review per the project instructions). Carried: VALU-op-count reduction (bf16 VALU or cross-head weight fold) = precision trade.
- 2026-09-25 critical-review B4/B5 preflight PASS WITH CONDITIONS — corrections: B4 decode −23.9% (slope) is NOT robust: M=1 +14.8%, unchained −6.8%
  → state range −7…+15%; B4 c131072 has a first-pod measurement −3.52% (vj77h) → state −3.5…−5.1%; B5 mixed: M=1 region doubles scalar ops
  (per-chunk live-count math hoisted out of the chain; tpu_custom_call-only hoist gate misses it) → state −53…−77%; j-ea90ea93 rc=1 (RecursionError
  in the b=1 edge case of the pre-existing bug) + rc=2, j-8baa6ce4 rc=2 — both rc=2 from d_idx blind at decode_s131073 (instrument property, blind with
  either arm; add to accepted-blind list with reason: seq_len 131073 → fault removes 1 key).
- 2026-09-25 e2e stage A+B1 j-353efca2 (image 6e8cbb7a5): 104.68 tok/s (+2.3% vs stage A 102.30; +15.2% vs incumbent avg 90.90), TTFT 116.78 s (−2.9% /
  −14.5%), TPOT 38.73 ms (−0.5% / −9.1%). B1 in-model: insert_indexer_k_cache 15.9 µs/call ×21 per 1024-token step (vs XLA 777 µs); u8[435200,256]
  relayout absent in all traces; T=16 decode keeps XLA scatter (gate 32). B1 TTFT −2.9% vs modelled ≈ −3% ✓.
- 2026-09-25 B2a (j-1d0275b0; exact j-e7c89c5d/j-1d0275b0; lowering fix after j-7827485a): split q/out interface (head-major ql_nope in, head-major out,
  q_pe concatenated in registers, sublane-strided loads on a u32 view, slab stored lane-tile-major [4,N,16,128]). Segment W_UK-out→W_UV-in:
  prefill −65 µs/layer chained (−3.6%, CI ±40), −121…−130 single-call; decode tie (T<64 falls back; split at T=16 costs +2.2 µs). Bit-exact 4/4, faults 3/3.
  NOT PORTED (deferred, carried): ≈ −5…−10 ms/prefill step (~1–2% TTFT) vs torch-side plumbing risk in attention.py. API facts: strided DMA via .at[] stride
  rejected (lowering.py:2084); strided load needs last dim 128 in the base memref ("The last dim size is not 128 in original base memref").
- 2026-09-25 e2e stage A+B1+B4+B5 j-064dd695 (image f1a803e86): 106.07 tok/s, TTFT 115.24 s, TPOT 38.31 ms, ITL 34.45 → vs incumbent mean +16.7% / −15.6% /
  −10.1%; vs stage A +3.7% / −4.2% / −1.6% [pred +2…+7 / −4…−7 / −2…+3 ✓ all]; vs A+B1 +1.3% / −1.3% / −1.1% (B4+B5 increment ≈2.6× noise).
  Code-ran proof: insert_indexer_k_cache 15.85 µs, 0 u8[435200,256]; sc_topk_v3_* only, 0 GatherScatterIndicesBitpacked; a16 gather all _live (2427:2426 with MLA).
- 2026-09-25 exit-review: V4 EXIT JUSTIFIED (ceiling; decode attn ≈1.8 ms/step total), B3 JUSTIFIED (refuted), one-sided MLA levers (dedup-alone, rope
  compaction) JUSTIFIED (MLA co-bound SC≈1.6/TC≈1.45 ms/layer → one-sided cap ≈10 ms/step; SC gather 848 GB/s ≥ cited practical 509–709 GB/s), B2a
  = `unfinished` (budget). EXIT PREMATURE on: indexer VALU precision lever (≈25 ms/step avg, ≈5% TTFT, p≈0.3) and MLA joint dedup (gated by unmeasured
  union ratio U/2048). Continuation dispatched: one real-activation dump (indexer q/w/k + top-k indices, prefill at C≥131k) → offline bf16-VALU top-k
  overlap test + union histogram per 64-token chunk.
- 2026-09-25 accuracy: incumbent same-window MMLU-Pro (limit 100, 1400 samples) = 0.7536 ± 0.0113 (j-507dac8a). Stage-A candidate eval j-aac3cb64 failed on
  scheduling (FailedScheduling, TPU capacity) → retry j-641409e7 running; final image f1a803e86 eval j-5f4c65a0 running. Gate: candidate ≥ 0.7386.
- 2026-09-25 accuracy stage A j-641409e7 (retry of j-aac3cb64, infra): MMLU-Pro limit100 0.7457 ± 0.0113 vs paired incumbent 0.7536 (j-507dac8a) → −0.79 pp,
  PASS at 1.5 pp; vs log baseline 0.7500 (j-012c51a9, different server shape) −0.43 pp. 89/1400 answers differ, SE(diff) 0.67 pp (1.2σ); stage A is
  kernel-exact, so attributed to serving non-determinism — incumbent-vs-incumbent eval spread NOT measured.
- 2026-09-25 accuracy final image f1a803e86 j-5f4c65a0: MMLU-Pro limit100 0.7479 ± 0.0114 vs paired incumbent 0.7536 (−0.57 pp), vs log 0.7500 (−0.21 pp) → PASS.
- 2026-09-25 final critical-review: ACCEPT WITH CORRECTIONS. Headline ESTABLISHED (f1a803e86 + 2 env vars +16.7% tok/s vs LKG, n=1 vs n=2). Corrections:
  code/config confound (V1 env exists at 22286cfcb; microbench config-only composite −13.9/−13.3% at 131k, −17.0/−16.7% at 261k, decode −6% vs −41%)
  → attribution arm dispatched (incumbent image + TPU_STREAMIDX_CHUNK_TOKENS=128); rungs < ~2% (B4+B5 +1.3%) PLAUSIBLE-UNCONFIRMED (noise from n=2, 1 df;
  tok/s spread 0.32%); tok/s/TTFT/TPOT deltas are one measurement (duration ≈ 4×(TTFT+1023·TPOT)); MMLU-Pro covers short prompts only — long-context top-k
  paths covered by kernel tests; V3 range −59.8…−62.8 incl. probe-05 replicates; "0 GatherScatterIndicesBitpacked" false (77 on concatenate ops) → "0 remap
  gathers"; probe-07 README frozen 08:55 AFTER A+B1 result published 08:48 → B1 share of the probe-07 prediction not blind.
- 2026-09-25 probe 08 real-activation dump (branch dsa_new_dump e7feb8a29, j-40ab18b9; 6 dumps, indexer layers 0/34 = model 0/34, C=131k–133k):
  (a) incumbent device top-k vs exact f64: overlap mean 0.9965–0.9970 (min 0.9917), swaps within 0.026σ of threshold; f32 vs f64 1.0000 → device scores
  differ from IEEE f32 by ~1e-3 rel (source unresolved; fp8 MXU accumulation suspected). (b) bf16 VALU (relu·w·sum all bf16) overlap vs f32 0.985–0.994
  mean → bf16 lever REFUTED on real data (worse than the incumbent's own device error); MXU-out-bf16 0.998 but cuts no VALU. (c) union per 64-token
  chunk U = 4.1–5.1 × 2048 (U/(64·2048) ≈ 0.065–0.074); 16-token blocks 2.6–3.0 × 2048; adjacent-token Jaccard 0.50–0.57 → joint-dedup gate PASSES
  (13–15× fewer unique rows than gathered). Exit-review gate text "U/2048 < 0.6" was ill-posed (U ≥ 2048); intended quantity is unique/gathered rows.
- 2026-09-25 attribution arm j-b8d65dbc (incumbent image 22286cfcb + TPU_STREAMIDX_CHUNK_TOKENS=128 only): 96.60 tok/s (+6.3% vs incumbent mean),
  TTFT 126.47 s (−7.4%), TPOT 42.11 ms (−1.2%). Split of the final +16.7%: config on main ≈ +6.3%; new code ≈ +9.8% on top (106.07/96.60),
  TTFT −8.9% and TPOT −9.0% beyond config. Chunking proven static (8× -c8 kernels, 0 unchunked) on rank0/rank8.
