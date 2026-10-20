---
scripts:
- scripts/tpu-roofline/roofline.py
---
# Known bugs in this skill

Defects found and evidenced but **not yet fixed**. One entry per bug. Each carries where it lives,
what proves it, what it costs, and the proposed change — enough to fix without re-deriving.

A bug leaves this file when it is fixed *and* pinned by a selftest. "Fixed" without a test that
would have caught it is not fixed; the arithmetic here is exactly the kind that drifts back.

Opened 2026-09-21 from the GLM-5.2-FP8 8K/1K reconciliation: script vs a human workbook roofline
(`[8K/1K] GF 2x2x2 MoE a8w8`) vs a measured, profiled run.

**Reference measurement** — `j-eee42323-fe98-4411-b74f` (2026-09-15), GLM-5.2-FP8, v7x 2x2x2,
`tp1-dp16+EP`, `mnbt=1024`, `max-num-seqs=64`, ISL 8192 / OSL 1024, `c=1024`, `ENABLE_XPROF=1`.
Phase-separated xprof (`prefill_only` / `prefill_heavy` / `decode_only`), rank0, both TensorCores.

| | script | measured | ratio |
| --- | --: | --: | --: |
| decode step, per core | 30.56 ms | **69.44 ms** | 2.27× |
| prefill chunk (1024 tok), per core | 186.7 ms | **629.47 ms** | 3.37× |
| fleet throughput | 74,379 tok/s | **21,422 tok/s** | 3.47× |

The decode step is corroborated three ways to within 0.6 % — bench median ITL 69.12 ms, server-log
pure-decode window (661.8 tok/s ÷ 46 running) 69.5 ms, xprof TensorCore self-time 69.44 ms — so
host overhead is off the critical path and the device comparison is apples-to-apples.

---

## B2 — fused attention is aggregated as `Σ max(...)` instead of `max(Σ...)`

**Severity: high. STILL OPEN, and now easy to mistake for fixed.**

> **Warning added 2026-09-22.** `layer_rows` now emits a
> `MLA: Flash Attention QK+AV` **parent row with the three sub-ops indented under
> it** — which is step 3 of the proposed fix below, and looks like the whole thing.
> It is not. `_merge` SUMS the children's `roofline_ms`, i.e. it computes
> `Σ max(a,b)`, and its docstring argues for that explicitly. The parent row is
> presentational; the aggregation is unchanged. Measured on the 8K/1K decode cell:
> parent row **0.03607 ms**, `max(Σ)` floor **0.02509 ms** — a 0.011 ms/layer gap,
> **0.857 ms/step** across 78 layers. Whoever closes this must change `_merge`, not
> just check that a parent row exists.

`mla/mla.md:144` says:

> `> [!IMPORTANT] **Flow B Output:** Report all 10 sub-operations (Projections, QK/AV phases)
> separately. Do not combine.`

Right about *reporting*, wrong about *aggregation*. QK(nope), QK(rope) and AV are one fused flash
kernel, so the floor is a single `max(Σ hbm/BW, Σ flops/peak)`. `Σ max(a,b) ≥ max(Σa, Σb)`, and the
gap is exactly the amount by which a sub-op split over-charges a fused kernel.

The human workbook already does this correctly — §3.3 indents 10a/10b/10c under a parent carrying
the fused roofline (0.0250856 ms/layer), which is why its decode total is 26.56 ms where the sub-op
sum is 27.38 ms.

**Cost** (decode, per core, after B1 is fixed): 2.778 → 1.957 ms. B1+B2 together take the decode
step 30.56 → 28.84 ms (−5.6 %) and throughput 74,379 → 75,434 tok/s fleet (+1.4 % — prefill
dominates, so the headline barely moves). Prefill unchanged.

**What it changes that matters more than the ms.** Fused MLA attention (1.957 ms) drops below
`MLA: O Proj` (2.376 ms), so the decode op ranking is currently wrong — `/decompose-optimization`
ranks on share-of-step × headroom. And the boundness message flips from "HBM-bound with 2.2× over
compute" to "on the HBM/MXU knee at 1.04×", which is a different instruction to a kernel author.

**Proposed fix**

1. Replace `mla/mla.md:144` with:
   > `> [!IMPORTANT] **Flow B output and fusion.** Report all 10 sub-operations separately for FLOPs
   > and bytes — that attribution is the point. But QK(nope), QK(rope) and AV are **one fused flash
   > kernel**: their roofline is a single `max(Σ hbm/BW, Σ flops/peak)`, *not* the sum of three
   > per-op maxes. `Σ max(a,b) ≥ max(Σa, Σb)`, and the gap is precisely how much a sub-op split
   > over-charges a fused kernel. Present the three as indented children of one
   > `MLA: Flash Attention (QK+AV)` row that carries the roofline.`
2. Add the same rule to §2 (decode) and to `gated_mla/gated_mla.md`.
3. `{{script:tpu-roofline/roofline.py}}`: add `fuse: str | None = None` to `Op` (:~282); tag `mla_qk_nope`,
   `mla_qk_rope`, `mla_av` with `fuse="mla_attn"`; group by `(layer_class, fuse)` before the max in
   the aggregation (:~697) and emit parent + indented children.
4. `mla/EVAL.txtpb` — add a case asking for the *fused* QK+AV roofline. Nothing currently exercises
   this.

**Guard:** `test_fused_attention_is_max_of_sums_not_sum_of_maxes` — pin 1.957 ms against the
2.778 ms a sub-op split gives.

---

## B3 — the KV-pool gate over-states capacity by ~30 %, and reads as a note when it should read as a block

**Severity: high.** This is the one output that could have caught a mis-provisioned deployment.

For the reference run the script printed:

```
KV pool    = 96 GiB x 0.92 - weights - 4 GiB   =    24.81 GiB
KV gate    = pool / (kv_per_token x (ISL+OSL)) =     60.6 req/core
configured   64 req/core                        =      106% of the KV gate
```

vLLM's own sizing, from the server log of that run:

```
GPU KV cache size: 424,960 tokens
Maximum concurrency for 9,216 tokens per request: 46.11x
```

Truth is **46.11 req/core → the configured 64 is 139 % of capacity**. The run sat pinned at
`Running: 46, Waiting: 18, GPU KV cache usage: 100.0%` for its entire steady state; 18 of 64 offered
requests per engine never ran. Both this script and the human workbook model a decode batch the
machine refused to admit.

**Two causes.**

1. `Hardware.memory_utilization` defaults to **0.92** (`{{script:tpu-roofline/roofline.py}}:160`) while the cell runs
   `--gpu-memory-utilization=0.90`, and the config JSON has no field for it — so it diverges
   silently. Setting 0.90 gives 55.9 req/core.
2. `resident_overhead_gb = 4.0` (`{{script:tpu-roofline/roofline.py}}:161`) is roughly half the real resident
   overhead. Even at util 0.90 the script says ~515 k tokens where vLLM allocates 424,960 (1.21×).

**Proposed fix**

- Add `memory_utilization` to the `hardware` block of every config and make it **required**, not
  defaulted — a silent 0.92 against a deployed 0.90 is the failure mode.
- Re-derive `resident_overhead_gb` against a measured `GPU KV cache size` line rather than a guess.
  Consider reporting it as a range and printing the residual against vLLM's own number when one is
  available.
- Change the framing: over 100 % of the KV gate is not a footnote. Print it as a refusal or a loud
  warning, and feed the admitted batch (not the configured one) into the serving throughput.
  `serving.py` already computes an `admission` line — the roofline print should agree with it.

**Guard:** a test pinning the gate against the 424,960-token / 46.11× observation for this cell.

---

## B4 — op classes the model omits entirely: ~1/3 of the real step

**Severity: high for calibration, medium for ranking.** Not arithmetic errors — missing scope. The
README's known-gaps list adds to roughly 1.5×; the observed gap is 3.4×, so most of it is here.

Measured against the reference profile, per core:

| bucket | script decode | measured decode | script prefill /chunk | measured prefill /chunk |
| --- | --: | --: | --: | --: |
| MoE permute / unpermute / combine | 0.190 ms | 6.170 ms | 0.127 ms | 20.84 ms |
| MoE expert-order sorts, combine cast, copies, `select_n` | — | 3.314 ms | — | 55.7 ms |
| MoE router top-k (Pallas) | 0.0015 ms | 0.440 ms | 0.023 ms | 6.072 ms |
| `sc_dsa_gather` — the sparse KV gather | **absent** | 4.200 ms | **absent** | 51.395 ms |
| DSA indexer compute | 0.197 ms | 2.98 ms | 1.18 ms | ~21.1 ms |
| KV-cache write `scatter_fusion` | **absent** | 1.376 ms | **absent** | 22.63 ms |
| layout / copy / reshape / `as_strided` | **absent** | 3.407 ms | **absent** | 44.05 ms |
| ...of which **indexer KV-cache relayout** (see below) | — | — | **absent** | 25.9 ms |
| LM head + sampling | **absent** | 1.311 ms | — | — |

`sc_dsa_gather` is the important one conceptually: **DSA sparsity does not just buy compute and cost
nothing — it buys a gather**, worth 8.2 % of prefill and 6.1 % of decode. Neither this skill nor the
human workbook has a term for it.

**Proposed fix.** Add specs + ops for: the sparse index gather (new op under
`model/common/attention/dsa/`), the KV-cache write, and a layout/copy term. Recalibrate the MoE
permute term — it is 11× low in decode and the sorts around it are not modelled at all. Decide
whether the LM head and sampling belong in a per-step roofline at all (they are per-step, not
per-layer, and 1.9 % of the decode step).

**2026-09-22, the permute row.** Re-costed: the one `4 * T*K*D` op is now three
(`permute` 1.125U / `unpermute` 2.0U / `combine` 1.125U = 4.25U), matching
`j-5444b22d-79a3-440c-bd5b`'s `gather_fusion` → [expert GEMMs] → `gather` →
`convert_reduce_fusion`. The modelled figures above are the new three-op totals. **The gap widened
rather than closed** — 0.127 ms modelled against 20.84 measured at prefill, 164×, because the split
corrected the shape accounting and not the efficiency. The measured side now also splits: 6.298 /
6.877 / 7.663 ms. Whoever recalibrates has per-op measured numbers to work with.

**The layout row has a named instance.** `j-5444b22d`'s indexer KV-cache relayout is 25.9 ms/step
of the 44.05 — `reshape u8[768000,256] ↔ u8[750,256,4,256]` plus a `scatter_fusion`, 23
occurrences/step (the 21 indexer layers + 2), both directions of the same 187.5 MiB tensor, all
`Bound by: HBM`. Every indexer layer converts its K cache between a paged and a flat layout and
back, because the cache is stored paged and the indexer scans it contiguously. Two cautions before
building on it: it is reported at **3.6 GiB/s, 0.11 % of peak** (a 187.5 MiB copy should take
0.058 ms and takes ~0.60), which is either a strided relayout defeating coalescing or a
misattributed event; and its trailing dim is **256** where every GLM-5.2 config pins the indexer
cache at **128 B/token** (`D_idx 128 × fp8`), which the reference sheet agrees with. The shapes
cannot settle that — `768,000` factors as both `750 × 256 × 4` and `750 × 1024` — so the 256 may be
two tokens packed, a head pair, or the layout's tile width. **If it really is 256 B/token, then
`DSA Indexer: QK` — 5.107 ms of a 6.407 ms decode dense layer, the single largest decode op — is
2× light, and so is all indexer-cache residency.** One look at the cache allocation in the checkout
settles both; nothing in the profile will.

---

## B5 — collectives under-priced ~2× against measurement

**Severity: medium.** The volume model is careful; the exposure model is what breaks.

| | script | measured | ratio |
| --- | --: | --: | --: |
| decode, all three EP collectives | 6.635 ms | 13.204 ms | 1.99× |
| prefill, per 1024-tok chunk | 92.65 ms | 218.30 ms | 2.36× |

The script assumes collectives are **fully exposed**, which should make it pessimistic. They are
async on this build (`async_execution_thread="sparsecore"`) and it is still 2× low, because what the
TensorCore waits on is a straggler-inflated `async-done`, not wire time. Add 9.45 ms/step of
`barrier-cores` in decode. The assumption that breaks is perfect expert routing + a synchronised
fleet, not the bandwidth number.

Fixed cost per collective: on `j-484cd4c0` (GLM-5.2-FP8, 2x2x2, TP16), an alpha-beta fit across two
same-group all-gathers of different size gives **13.5 µs** (13.49 on mean times, 13.52 on min
times). The model charges only hops × `hop_latency`: 4.0 µs on that 4-axis slice at the configured
1.0 µs, 5.2 µs at 1.3 µs. Only the total was measured; how it splits between hop latency and launch
or barrier cost is not known. `launch_overhead` is where a measured remainder goes.

Also measured: the `collect = 2 × dispatch` dtype argument does not hold. Prefill 48.35 / 144.11
(3.0×); decode 6.111 / 5.775 (**0.94× — collect is cheaper than dispatch**).

**Proposed fix.** Not a formula change — an honesty change. Either model a straggler/imbalance term
explicitly, or state the measured 2× as a calibration factor in `COLLECTIVE.md` with this job id
behind it. Do not quietly scale the bandwidth.

---

## B8 — the router-logit all-gather width is unverified and may be ~32× over-priced

**Severity: unknown — this is a question, not yet a confirmed bug.**

`coll_ep_router` (`{{script:tpu-roofline/roofline.py}}:~600`) all-gathers `T × E` fp32 — all 256 router logits per
token. If the implementation gathers the top-8 *indices* instead, the volume is ~32× smaller.

Against measurement the script is 5× **low**, not high (prefill 5.13 modelled vs 25.84 ms measured;
decode 0.602 vs 1.318), so if the width is also wrong the underlying exposure error is larger still.
The op is real and not small — 4.1 % of the measured prefill chunk, more than the expert
down-projection GEMM.

**Proposed fix.** Read the vLLM-torchtpu EP dispatch path and pin the actual gathered tensor and
dtype in `parallelism/ep/ep.md`. Until then the number is a guess with a citation-shaped hole.

**2026-09-22.** `j-5444b22d-79a3-440c-bd5b` was dumped to settle this and **cannot**: that job runs
`tp16-dp1` with **no expert parallelism at all** — zero all-to-all, zero reduce-scatter, zero token
or router-logit all-gather across three phases at 100 % `hlo_stats` coverage. Its router weight is
a full unsharded `bf16[256,6144]` and its expert GEMMs are `gmm_v2-g_256`, so all 256 experts are
local and there is nothing to gather. The EP job is `j-484cd4c0` (`ep_size=16`, 16/256 experts per
rank, confirmed from its logs); that is the profile to dump for this question.

---

## B9 — `mla/mla.md:175` contradicts the spec's own op list

**Severity: low, but it flips a boundness label.**

Decode AV says `Memory: Write Output (fused with output projection): 0`, while the same spec carries
a separate **V Up Proj** op that reads that output back. Those cannot both be true.

The profile shows `W_UV` as a distinct `bmm` (0.765 ms/step, 78 occurrences), so the *separate*
treatment is right and line 175 is the stale one.

**2026-09-22.** The line *above* this one in the same block now reads `Memory: Read $c_{kv}$: **0**`
(the B1 fix). The section has been edited; this line has not. Do not read the edit as covering it. It is 4 MiB/layer — but with B1 and B2 applied it
decides whether the fused attention group reads HBM-bound (0.025086 ms/layer) or MXU-bound
(0.024053), so settle it deliberately rather than folding it into the B1 fix.

---
