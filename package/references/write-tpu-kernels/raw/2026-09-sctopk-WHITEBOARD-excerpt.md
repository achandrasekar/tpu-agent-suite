# WHITEBOARD — excerpt (§7h, §7h-old, §7j)

**This is an excerpt, not the campaign's whole `WHITEBOARD.md`.** The full 891-line lever-by-lever
log was not retained; the three sections below are, verbatim, because they are the only ones an
entry in `strategies/` leans on — and between them they are the campaign's *arbitration record*:
the measurement that retracted a probe's numbers, the reasoning as it stood before that retraction,
and the null that closed the DMA question. The original, if it still exists, is
`<campaign>/WHITEBOARD.md` (campaign `sparsecore-topk`).

| § | what it holds | cited by |
| --- | --- | --- |
| 7h | the LICM arbitration: probe 04 hoisted, five numbers retracted, `W = 32` / `E[max] ≈ 0.97n`, 338 GB/s aggregate | [[STRAT-SC-03]]; corroborates [[STRAT-SC-05]] and `../hardware-facts.md` |
| 7h-old | the same conflict as recorded *before* the arbitration — the reasoning that was wrong | [[STRAT-SC-03]] |
| 7j | the DMA question closed: `emit_pipeline` == hand-rolled, arms agreeing on what they share | [[STRAT-SC-08]], [[STRAT-SC-10]] |

---

### 7h. RESOLVED — probe 04's chained clock was hoisted out of its own loop
[measured: j-e351b144-4ecd-4eba-893f] 4/4 arms rc=0, run **sequentially on one uncontended pod** so
probe 01 (originally fanned out) and probe 04 (originally solo) are compared under identical
conditions.

| arm | µs | method |
| --- | --- | --- |
| 0 probe 01 harness verbatim | 86.30 (4.6%) | chained M=200, naive /M; slope 100→200 = 88.71 |
| 1 probe 04 harness verbatim | 26.40 (17.1%) | reproduces its 30.9 within run-to-run |
| 2 wall clock + `block_until_ready` | 88.7 (7.2%) | 250 reps / 20 warmup, minus measured empty-SC launch |
| **3 xprof device time — ARBITER** | **68.26 (0.5%)** | `async-done` of `sc_topk_direct_b64_n33280_k2048`, n=20; SC0/SC1 planes 68.05/67.69 |

**The arbiter vindicates probe 01 and the wall clock.** The traces are unambiguous: probe 01's
M=20 chain contains **20 kernel events at 70.06 µs each** (module 1535.4 µs = 76.8/call); probe 04's
M=20 chain contains **one** kernel event at 68.17 µs (module 107.9 µs *total*).

**Named mechanism — loop-invariant code motion.** Probe 04 barriered the **loop-invariant operands**
— `optimization_barrier((scores, lengths, dep))` — and fed the kernel `s, l`. XLA splits the barrier
tuple, proves `s, l` equal the invariant captures, and **hoists `call(s,l)` clean out of the while
body**, leaving `dep` dead. The HLO confirms it: probe 04's async-start takes top-level
`%reshape.0.scores_hbm.2` / `%lengths.1`, while probe 01's takes `%copy-done` /
`%get-tuple-element.101` off a 6-wide while carry. **20 nominal invocations executed once**, and
dividing one invocation plus a ~250 µs host round trip by M=10 yields 25–31 µs. Its own
`ORACLE_CHAINED` guard could not fire, because a hoisted loop is idempotent and still returns the
right answer. Probe 01 is clean because the barrier output it feeds the kernel is the loop **carry**,
data-dependent on the prior output.

Hypothesis (b) is dead — probe 01's chain adds **no** serialization penalty; its 20 traced events at
70.06 µs match 20 independent dispatches at 68.26 µs within 2.6%. (c) is dead — both call
`_sc_topk_direct` bare, identical kernel op name in the trace. (e) not a factor.

**τ SURVIVES, in its strongest form**: 2.723 (probe 01) / 2.696 (arm 0 here) / 2.715 (b·p=4096) /
**2.735 from arm 3's pure device time with no launch subtraction at all** — which removes the fitted
19 µs term from the derivation entirely. Probe 04's implied 0.48 ns is void.

**338 GB/s is PER CHIP** — aggregate over 2 SC cores × 16 vector subcores, all 32
[`08-sparsecore.md:974-980`, re-measured solo at 337 GB/s / 1% spread, `j-74a24183-08a9-49f3-a822`;
`hardware-facts.md:561-563`]. **The per-core 2× escape hatch does not exist.** So the 25.2 µs floor
was arithmetically right and probe 04's 24.8 µs really was impossible; the paradox dissolves because
24.8 = a 241 µs host round trip ÷ 10.

**HEADROOM: ~63–71%, not ≤18%.** Device time 68.3 µs is **2.71×** the 25.2 µs floor (2.40× the
~300 GB/s saturation floor); the 86 µs call is 3.42×. **The compaction lever survives.**

**Retract probe 04's chained numbers** — all five arms (30.9 / 45.7 / 48.4 / 24.8) are
host-round-trip ÷ 10; each `chained_total` sits within 2.5–4.1% of that arm's own *unchained* total.
Only its arm 2 (277.4 µs) escaped the hoist. **Its tokamax ranking (§7g) is unaffected** — same
harness both sides — but its absolute per-call figures are unusable, and with them go its
"≤18% headroom" and "do not write H3".

**Two findings to carry forward:**
- **Input-distribution sensitivity is 7.2%** (78.44 µs on `N(0,1)` vs 84.08 on `normal*3.0`) — the
  L0 bucket-occupancy effect §6 predicted, now measured. Quote the distribution with the number.
- **Raggedness is governed by the longest row in a wave, not the mean.** Rows map as
  `row = w*(cores*subcores) + flat`, one row per subcore, so a wave costs its **longest** row, and
  the max of 32 `U(0,n)` draws is ~0.97n. That explains probe 02's ragged 76.39 vs dense 86.20
  (−11%, not −50%) and is **inconsistent with any bytes-based account** — further confirmation that
  §7i/H1 is dead. It also means per-wave row-length homogeneity is the thing that matters; at
  prefill, consecutive rows already have near-equal causal lengths, so the current assignment is
  close to optimal there and this is **not** a lever worth pursuing.

**Measurement-discipline rule added for this campaign** (it would have cost a day here): a
chained-clock guard must be an **M-sweep slope, never divide-by-M at a single M**, and never an
output-correctness check on an idempotent loop — a hoisted loop returns the right answer. **Barrier
the loop carry, not the captured invariants.**

### 7h-old. The conflict as originally recorded

| source | shape | reported | corroboration |
| --- | --- | --- | --- |
| probe 01 arm 0 [j-3f7375f2] | 64 × 33280, k=2048 | **86.97 µs** | replicates at b·p=4096 (4358 µs); 4358/5669 = 77% of probe 03's whole call, consistent with the measured 85% stage-1 share |
| probe 04 arm 0 [j-970c76e0] | same | **30.9 µs** | ×64 → ~1978 µs, fits neither probe 01 nor probe 03 |

Probe 04 additionally reports the incumbent **below its own HBM floor** — 24.8 µs against a computed
25.2 µs — while the DMA at `:160` moves the full `slice_len` regardless of `row_lengths`, so its
ragged and dense arms move **identical bytes** and a faster ragged arm cannot be explained by fewer
of them. Either the clock or the 338 GB/s denominator is wrong; if that figure is per-*core* on a
2-core chip the floor halves and the paradox dissolves without touching any measurement.

**What rides on it.** Probe 04 concludes total headroom ≤18% and "do not write H3". Probe 01 + the
six-pass model puts the kernel at ~20–30% of its bandwidth floor with ~50% of stage 1 in the
refinement and emit passes. τ = 2.72 ns (§7b) also stands or falls here — if probe 04 is right,
τ ≈ 0.9 ns and the six-pass model must be rebuilt. An arbitration job is running with an **xprof
device-time** arm, which is immune to both host overhead and chained-clock artifacts.

**Meanwhile the H3+H13 variant build continues**, because its modelled saving is a pass count, which
is unaffected by which absolute clock is right.

### 7j. THE DMA QUESTION — CLOSED. Fusion is the lever; pipelining is a null.
[measured: j-0852d569-16b9-4796-8139; run 1 j-51001265-5d25-4f05-81d0 reported unresolved, not as a
result] 4/4 shards green, **48/48 oracle cells PASS**, chain length verified at [1, 20, 120] with
per-invocation cost read off the **M=20→120 slope** (not a divide-by-M), 45 rounds × 4 chips.
b=64, n_p=33280, k=2048. µs/invocation, full | ragged:

| arm | µs | Δ vs baseline | cross-chip spread |
| --- | --- | --- | --- |
| 0 baseline | 86.20 \| 76.39 | — | 2.25% \| 1.34% |
| 1 `emit_pipeline` + fused `cvt` | 80.84 \| 72.59 | **−6.21% \| −4.97%** | 0.62% \| 1.97% |
| 2 hand-rolled double buffer + fused `cvt` | 80.88 \| 72.44 | **−6.17% \| −5.17%** | 0.15% \| 1.66% |
| 3 traced DMA extent + fused `cvt` | 80.89 \| 72.77 | **−6.15% \| −4.74%** | 0.23% \| 1.31% |

**All three DMA treatments land on the same number.** At full rows arm 3's traced extent *equals*
`slice_len`, so arm 3 @ full is "baseline + `cvt` fusion, identical bytes" = −6.15%; arm 1 moves the
same bytes *hidden* = −6.21%. **pipeline − traced = −0.06%: hiding the DMA buys nothing.** At ragged
the three span **0.43%**, inside the 1.3–2.0% spread. 8 chunks beat 16 on both managers.

- **H2 `emit_pipeline` — REJECTED as a throughput lever.** vs hand-rolled: −0.04% full / +0.19%
  ragged, indistinguishable. The long-open "does `emit_pipeline` beat a hand-rolled double buffer on
  SparseCore" question is **answered: null.** It works, and it is not faster. (Wiring was via
  closure capture per the §2 `scratches=` trap; the oracle PASS is the wiring proof.)
- **H1 traced DMA extent — REJECTED.** Ragged halves the DMA bytes and buys nothing measurable
  (−4.74% vs −4.97% for the arm that moves *full* bytes). §7i's byte-count argument does not
  survive contact with the measurement.
- **The whole DMA lever is worth ≤0.5%, not the 4.91% ceiling** we had been carrying. The ceiling
  was an over-estimate, not just an upper bound.
- **H4 `cvt` fusion — ADOPTED, −6.2% measured**, reproduced on 4 chips at both raggedness settings.
  The entire win is the deleted pass. Probe 01 priced `cvt` at 6.0%; this recovers 6.2%. Two
  independent methods agreeing on the cheapest pass is a good check on both.
- **Inter-arm gaps (0.04–0.43%) are below spread and NOT individually resolved.** The −6.2%
  headline is resolved (0.15–2.4% spread against a 6% effect); the ranking *among* arms 1/2/3 is
  not, and must not be quoted as one.

**Adopt arm 3's shape**: fuse `cvt` into the L0 histogram, no pipeline, no staging buffer. It is the
simplest code of the three, wins at both raggedness settings, and carries no `emit_pipeline`
version-dependency. This is the answer to the original request: **the requested substitution is a
measured null; the fusion it was going to carry is the real −6.2%.**

**Baseline corroboration**: arm 0 at 86.20 µs independently confirms probe 01's 86.97 µs at the
same shape and refutes probe 04's 30.9 µs (§7h) — two unrelated harnesses on different jobs. Probe
02 also shows **no sub-floor paradox** at the same bytes, which is itself evidence about probe 04.

