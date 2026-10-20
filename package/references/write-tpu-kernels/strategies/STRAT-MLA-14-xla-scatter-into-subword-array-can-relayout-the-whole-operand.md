```yaml
id: STRAT-MLA-14
title: An XLA scatter into a sub-32-bit tiled array can relayout the entire operand — cost scales with the array, not the update
hardware: tpu-v7x
unit: tensorcore
software: jax 0.10.2 / libtpu 0.0.44.1 (XLA:TPU in libtpu)
stage: [2, 3]
lever: layout
generality: general
status: measured
source: raw/2026-09-dsanew-STRATEGY.md § baseline e2e, § B1; raw/2026-09-dsanew-final_report.md § Levers B1
recheck: dump post-opt HLO of the scatter at the production update count and look for reshape ops whose shape is the whole operand
origin: campaign dsa-mla-256k-c16, 2026-09
```

# STRAT-MLA-14 — Look for a whole-operand reshape around a small scatter

**Rule.** `cache.at[i, j, k].set(rows)` on a large u8 array can compile to "reshape the whole array to 2-D,
scatter, reshape back" — two full copies of the operand around a tiny update. Whether it does depends on
the update count, so a microbench at a small shape will not show it. When a cache write shows up at a cost
proportional to the cache, replace it with an aliased Pallas kernel that DMAs only the touched tiles.

**Mechanism.** The array's native layout (`u8[425,256,4,256]{…:T(4,128)(4,1)}`) does not match what the XLA
scatter emitter wants at large update counts, so layout assignment inserts relayouts of the whole operand;
at small update counts the same scatter runs in place. One group of 4 tokens is exactly one aligned tile
pair in the native layout, so a DMA per group needs no relayout at all.

**Evidence.** In-model: 342 + 125 + 297 µs = **777 µs per layer** for a 1024-row insert into a 111 MB
cache (16.3 ms per prefill step over 21 layers). Microbench across update
counts, XLA vs aliased Pallas: 12.9/12.1 (16), 30.4/15.5 (128), 81.1/25.0 (512), **792.1/34.4 µs (1024)** —
the jump at 1024 is the relayout; byte-exact 12/12. In-model
after the change: 15.9 µs/call, relayout absent; e2e TTFT −2.9%.

**Scope.** XLA scatter/dynamic-update into sub-32-bit tiled arrays on TPU. **Does not apply when** the
post-opt HLO shows the scatter in place (at T=16 here the XLA path won: 12.9 vs 12.1 µs single-seq,
19.0 vs 28.2 µs with 16 decode sequences) — dispatch per static shape.
