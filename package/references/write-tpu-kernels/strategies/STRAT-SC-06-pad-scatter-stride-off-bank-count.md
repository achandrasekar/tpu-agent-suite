```yaml
id: STRAT-SC-06
title: A per-lane stride sharing a factor with the bank count collapses every lane onto one bank
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [3, 4]
lever: layout
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-06; raw/2026-09-sctopk-final_report.md §2 (H16)
recheck: the bank count on the target part, and the per-way conflict cost
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-06 — Pad a per-lane stride coprime to the bank count

**Rule.** When each lane owns a private region addressed `lane * STRIDE + offset`, the lanes land on
distinct banks only if **`STRIDE` is coprime to the bank count**. Share a factor and every lane
touching the same offset lands on one bank, serializing the access by the number of lanes. **Pad
`STRIDE` by one word.** The cost is one word per lane; the return is the full lane parallelism.

**Mechanism.** Bank index is the address modulo the bank count. If `STRIDE ≡ 0 (mod banks)` then
`lane * STRIDE + offset ≡ offset` for every lane — one bank, `lanes`-way serialization, independent
of the offset. If `gcd(STRIDE, banks) = 1` the map `lane ↦ lane * STRIDE (mod banks)` is a bijection,
so the lanes are spread regardless of which offset they chose. Conflict cost is roughly linear in
the number of ways.

**Evidence.** On v7x SparseCore the bank count is **32** —
**inferred from the conflict ladder below, not read out of a spec.** `SparseCoreInfo` declares
subcores and lanes but not banks; 32 is the value that reproduces the observed 16-way ceiling and the
odd-stride fix, and that is as far as the evidence goes (`hardware-facts.md` §10). Do not confuse it
with the 32 subcores of [[STRAT-SC-05]], which is a different 32 and *is* in source. A
private histogram at `lane * 256 + bucket` put all **16** lanes on one bank; stride **257** made
lane `l`'s address `≡ l (mod 32)`. Worth **−15.08%** on `dsa_like` and **−7.6%** on `uniform` at
b·p=64; **−17.24% / −9.54%** at b·p=4096; **−12.98…−16.05%** whole-call in the shipped stack. Per-way cost fitted `base + (ways−1) × 4.7 µs`. VMEM cost
4112 → 4128 words. 40/40 oracle PASS, `rows_differing` 0.

**Scope.** Any lane-private table — histograms, counters, accumulators — on any banked memory. The
bank count is the parameter; 32 is v7x SparseCore's and must be re-measured elsewhere.

**Does not apply when** lanes read their regions at *different* offsets by construction, which
already spreads them. It is the shared-offset case that collapses.
