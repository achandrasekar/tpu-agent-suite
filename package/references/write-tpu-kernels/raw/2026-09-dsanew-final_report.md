# final_report — GLM-5.2-FP8 DSA indexer + sparse MLA, cell 2x2x2/262144-1024/c16 (TPU v7x)

Status: completed (stage A + stage-B rungs B1, B4, B5). Final critical-review: ACCEPT WITH CORRECTIONS (applied). Exit-review continuation (probe 08) done. All planned measurements complete.

Checkout: vllm-torchtpu, branch `dsa_new` (pushed to origin), base main `22286cfcb`.
Commits: `4f1a81597` stage A · `62685bb28` zero-live tests · `6e8cbb7a5` B1 · `f1a803e86` B4+B5.
Serving: Cell-5 LKG command + `TPU_STREAMIDX_CHUNK_TOKENS=128` + `TPU_STREAMIDX_DECODE_KV_PAGES=8`.

## End-to-end (Cell-5 BENCH_CMD, 64 prompts, ENABLE_XPROF=1, clean pass)
| arm | job | image | out tok/s | TTFT med | TPOT med | ITL med |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | j-6d5dff1e | 22286cfcb | 90.75 | 136.85 s | 42.63 ms | 38.55 ms |
| incumbent (replicate) | j-2bbeb4a4 | 22286cfcb | 91.04 | 136.24 s | 42.62 ms | 38.56 ms |
| incumbent + CHUNK_TOKENS=128 only (config-only attribution) | j-b8d65dbc | 22286cfcb | 96.60 (+6.3%) | 126.47 s (−7.4%) | 42.11 ms (−1.2%) | 38.59 ms |
| stage A | j-34b7fbc9 | 4f1a81597 | 102.30 (+12.5%) | 120.24 s (−11.9%) | 38.92 ms (−8.7%) | 34.78 ms |
| stage A + B1 | j-353efca2 | 6e8cbb7a5 | 104.68 (+15.2%) | 116.78 s (−14.5%) | 38.73 ms (−9.1%) | 34.75 ms |
| stage A + B1 + B4 + B5 | j-064dd695 | f1a803e86 | **106.07 (+16.7%)** | **115.24 s (−15.6%)** | **38.31 ms (−10.1%)** | 34.45 ms |
Δ vs the incumbent mean (90.90 tok/s, 136.55 s, 42.63 ms). Every candidate arm = image **plus** `TPU_STREAMIDX_CHUNK_TOKENS=128` and
`TPU_STREAMIDX_DECODE_KV_PAGES=8` (both default-off in code; V1's env var already exists on the incumbent image). Noise from n=2 incumbent runs (1 df):
tok/s 0.32%, TTFT 0.45%, TPOT 0.02%. Candidates n=1, run sequentially. The three columns are one measurement, not three (duration ≈ 4 × (TTFT + 1023·TPOT)
at mns=1). Headline (final vs incumbent, +16.7%) is ESTABLISHED; code vs config split MEASURED (j-b8d65dbc): the overlap knob alone on main gives +6.3%; the new kernel code adds
≈ +9.8% on top (106.07 vs 96.60 tok/s; TTFT −8.9%, TPOT −9.0% beyond config — nearly all of the decode gain is code); B1 −2.9% TTFT observed and matches an independent model (761 µs × 21 × 256 ≈ −3.4%); B4+B5 +1.3% is
PLAUSIBLE-UNCONFIRMED (below what n=1 with a 1-df noise estimate can separate from drift).

## Levers (microbench: per-layer chained slope, one v7x chip, production shapes, vs verbatim incumbent in the same job)
| lever | change | measured | backing |
| --- | --- | --- | --- |
| V2 | SC top-k stage 1 streams only live context, rebalanced over 32 subcores (traced DMA offset/extent) | idx prefill −21.0/−19.8% (C=131k), −56/−54% (32k), 0 (261k, null control) | j-b59e427e, j-2dcfbed7/j-2cfd6677 |
| V1 | `TPU_STREAMIDX_CHUNK_TOKENS=128` (SC top-k ‖ TC scoring) | idx prefill −28.8% alone; with V2 −43.7/−43.0% (131k), −63/−64% (32k), −30% (261k) — levers interact (V2 worth 15.7 pp on top of V1 vs 21 alone) | j-1e32fec6, j-abac8714, probe 05 |
| V5 | `TPU_STREAMIDX_DECODE_KV_PAGES=8` | idx decode −14.0% (pages 16: VMEM OOM) | j-abac8714 |
| V3 | decode real-row bound (live gather atoms 1 + attention batch 1 for chunks ≤16) | attn decode −59.8…−62.8% across probes 04/05 (halves interact: +12.3% / −39.8% alone) | j-7a60fc19, j-2dcfbed7, j-2cfd6677 |
| B1 | Pallas indexer K-cache insert (buckets ≥ 32) | 792 → 34 µs/call at T=1024; in-model 777 → 15.9 µs; e2e TTFT −2.9% vs stage A | j-88dbb2ef, j-7c171f4d, j-353efca2 |
| B4 | stage-2 carries global indices (remap gather deleted) | idx vs stage A: −3.5…−5.1% (131k), −2.2% (261k), mixed T=1024 −51.6%; decode −7…+15% (clock-dependent, not robust) | j-6eb9ee98 |
| B5 | live gather for 64-token chunks | mixed T=1024 attn −53…−77%; prefill null | j-8baa6ce4 |
Composite 21·idx + 78·attn per step, stage A (two replicates): decode 8.32–8.42 → 4.92–4.98 ms (−41%); prefill C=131k −22.4/−21.1%, C=32k −27.5/−24.2% (misses the 2 pp replicate rule: noise in unchanged prefill attention; indexer-only −25.5/−26.7%),
C=261k −17.3/−17.6%. Config-only arm (V1+V5 without V2, `cfg_arms:chunk128_dpages8`): −13.9/−13.3% at 131k, −17.0/−16.7% at 261k, decode −6% — i.e.
at high C most of the stage-A prefill gain is the overlap config.

## Correctness
Regime coverage: MMLU-Pro prompts are far below 131k, so the long-context top-k paths V2/B4 change are covered by the kernel accuracy test, not by MMLU-Pro.
Accuracy test (instruments/, float64 oracle + native reference + fault control): every candidate 15/15 (16/16 with the mixed cell) — top-k sets identical
to the incumbent (overlap 1.0000), attention bit-identical on real rows (native=0); B1 byte-exact 12/12. Faults wrapping the candidate: caught on every
production cell; blind edge cells (fault ≈ no-op): c@decode_s2049, c@decode_s131071, d@decode_s131073. MMLU-Pro (`--limit 100`, 1400 samples, identical eval command and server shape, same window):
| image | job | score | Δ vs paired incumbent 0.7536 (j-507dac8a) | Δ vs log baseline 0.7500 (j-012c51a9) | gate 1.5 pp |
| --- | --- | --- | --- | --- | --- |
| stage A 4f1a81597 | j-641409e7 | 0.7457 ± 0.0113 | −0.79 pp | −0.43 pp | PASS |
| final f1a803e86 | j-5f4c65a0 | 0.7479 ± 0.0114 | −0.57 pp | −0.21 pp | PASS |
Stage A is kernel-exact; the 89/1400 differing answers (SE of the paired difference 0.67 pp) are attributed to serving non-determinism — an
incumbent-vs-incumbent eval spread was not measured.

## Rejected / killed
| lever | verdict | job |
| --- | --- | --- |
| V4 attention body (mask gating, PV bf16×512, scale fold) | null at prefill (module SC-gather-bound); +26.8% at decode (two-body pl.when) | j-7a60fc19 |
| B3 spill-free scoring (key sub-tiles) | refuted: spills co-issued in free slots; VALU (3 ops/elem) binds; bundles unchanged | j-de6820cb, j-8023c2f5 |
| V5 pages 16 | VMEM OOM at compile | j-1e32fec6 |
| B2a q/out glue fusion | measured −65 µs/layer (−3.6% segment), exact — DEFERRED, not ported (torch plumbing) | j-1d0275b0 |

## Exit-review (clean context) — stops
V4 EXIT JUSTIFIED (ceiling); B3 JUSTIFIED (refuted); one-sided MLA levers (dedup alone, rope compaction) JUSTIFIED — the MLA module is co-bound
(SC gather ≈1.6 ms ‖ TC ≈1.45 ms per layer; gather at 848 GB/s, above the cited 509–709 GB/s practical rate) so a one-sided lever caps at ≈10 ms/step;
B2a = unfinished (budget). EXIT PREMATURE on the indexer VALU precision lever and the MLA joint-dedup lever → continuation: real-activation dump
(probe 08) for a bf16-VALU top-k overlap test and a union histogram per 64-token chunk.

## Carried (each with a route)
- B2a split q/out interface — ready variant, exact, −65 µs/layer segment → /write-tpu-kernels port through layers/adapter/attention.py.
- Indexer scoring VALU (3 ops/elem binds; MXU floor 2.46 ms vs 4.9 ms at C=261k): bf16 VALU REFUTED on real activations (probe 08, j-40ab18b9: top-k
  overlap vs f32 0.985–0.994, worse than the incumbent's own 0.9965 vs exact). Remaining exact route: none found; stop class `refuted`.
- **MLA joint dedup — the strongest next lever, gate PASSED on real activations** (probe 08): a 64-token chunk's top-k sets union to 4.1–5.1 × 2048
  rows, i.e. the SC gather moves 13–15× more rows than unique (16-token blocks: 2.6–3.0 × 2048; adjacent-token Jaccard 0.50–0.57). Design: gather
  the union once per block, attend densely with a membership mask (also removes the per-token u32→u8 kv unpack that binds the TC body). Multi-day
  kernel redesign → next /write-tpu-kernels campaign (MLA prefill ≈123 ms/step is now the largest DSA+MLA row).
- Incumbent indexer device scores differ from IEEE f32 by ~1e-3 relative (top-k overlap 0.9965 vs exact; swaps ≤0.026σ from threshold) — source
  unresolved (fp8 MXU accumulation suspected) → informational, kernel owner.
- Pre-existing bug: `sparsecore_topk` b=1, n=264192 recurses forever at trace time (stage 2 at n=65536 picks p=32, p·k_p == n) — unreachable at min
  token bucket 16 → owner of kernels/deepseek_v4/sparsecore_topk.py.
- B1 gate (32 tokens) measured single-sequence only; buckets 32–512 with many decode sequences unmeasured (worst case ≈ +0.2 ms/step) → next cell with mns > 1.
- DCP twin `_build_streamidx_dcp_op` keeps decode pages 2 and the XLA insert (different slot mapping) → DCP campaign.
- Model-configs Cell 5: promote to LKG on main-lineage code (j-6d5dff1e 90.75 tok/s on 22286cfcb) and record the candidate command (+2 env vars) once merged.
- Experiment records of the incumbent, stage A and final runs → after critical-review.
- Machine hygiene: Docker holds ≈238 GB of reclaimable images / 96 GB build cache on the VM (disk hit 100% once during this campaign) → user's call.
