---
title: "All About Transformer Inference (scaling-book, Part 7)"
type: source
tags: [workload, performance, memory, parallelism]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 7 of [*How To Scale Your Model*](2025-scaling-book.md): prefill vs generation as two
different workloads, the critical batch size with precision factor `β`, attention
intensity (`T/2` in prefill, ~1 in generation), the step-time and throughput formulas, KV
cache memory and the techniques that shrink it, sharding rules for each phase (model
parallelism only during generation; FSDP impossible), sharding the KV cache, and inference
engine design (interleaved, disaggregated, continuous batching, prefix caching, JetStream,
speculative sampling).

## Identity & authority

- <https://jax-ml.github.io/scaling-book/inference> — fetched 2026-09-04; chapter last
  changed upstream 2026-05-20 (`37a2a7c`).
- Authority: authorial; the step-time model is validated qualitatively by one empirical
  batch-scaling plot (Appendix A: throughput rises until ~BS 240 per DP shard).

## Gaps & caveats

- Constants are v5e (`B_crit` 240, `β = W_hbm/W_ici ≈ 8` "for v5e and v6e").
- JetStream is presented as the reference engine; the autoresearch wiki notes it is being
  archived — treat as an architecture reference, not a deployment target
  *(unverified — agent inference from a sibling wiki)*.
- XLA overlap is imperfect: the book warns sharded matmuls take a latency hit above
  BS≈32 and rooflines are not sharp in practice.

## Residue (not extracted)

Seven worked problems (an invented 18.4B model: params, KV per token, max batch on a v5e
4x4, weight-load floor, MoE variant, expert sharding, 2D weight-stationary math). Lifted
conclusions: MoE raises the HBM critical batch by `E/k`; 2D weight-stationary sharding
beats 1D Megatron above ~72 chips for `F = 4D`.

## Connections

Created: [prefill-vs-decode](../concepts/prefill-vs-decode.md),
[kv-cache](../concepts/kv-cache.md).
Updated: [critical-batch-size](../concepts/critical-batch-size.md),
[tensor-parallelism](../concepts/tensor-parallelism.md) (generation limits),
[fsdp](../concepts/fsdp.md) (do not use for generation), [hbm](../concepts/hbm.md),
[quantization](../concepts/quantization.md),
[collective-costs](../concepts/collective-costs.md) (latency-bound activations).

## Sources

- `raw/docs/2025-scaling-book-inference.html`
