---
title: "All the Transformer Math You Need to Know (scaling-book, Part 4)"
type: source
tags: [workload, performance, memory]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 4 of [*How To Scale Your Model*](2025-scaling-book.md): FLOP counting for contractions,
the 6ND training rule, per-layer parameter and FLOP tables for MLP/attention/vocab, the
`T > 8D` crossover where attention dominates, MoE parameter accounting, gradient
checkpointing policies, KV-cache size, and an appendix on how Flash Attention works.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/transformers> — fetched 2026-09-04; chapter last
  changed upstream 2026-08-20 (`cf7e74d`); July 2026 copy-edit counted both per-layer
  layernorms in the parameter math (`808ce45`).
- Authority: authorial; formulas are architectural identities, not measurements.

## Gaps & caveats

- Assumes gated MLP (3DF params) and GQA; models with a single `W_in` have 2DF.
- Local/sliding attention and other attention variants change the `T > 8D` roofline
  (stated as a footnote only).

## Residue (not extracted)

Eight worked problems (param counts, KV sizes, GQA intensity, remat FLOPs, DeepSeek-V3
MFU, MoE critical batch). Two conclusions lifted: DeepSeek-V3's reported training implies
~22% FLOPs utilization; an MoE with int8 weights needs `B > 120·E/k` tokens.

## Connections

Created: [transformer-flops-accounting](../concepts/transformer-flops-accounting.md),
[kv-cache](../concepts/kv-cache.md),
[gradient-checkpointing](../concepts/gradient-checkpointing.md),
[expert-parallelism](../concepts/expert-parallelism.md).
Updated: [vmem](../concepts/vmem.md) (Flash Attention residency).

## Sources

- `raw/docs/2025-scaling-book-transformers.html`
