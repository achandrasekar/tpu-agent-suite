---
title: "TPU 8i (inference) — announced"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

Eighth-generation **inference** chip, announced April 2026, GA "later this year";
sibling of [TPU 8t](tpu-8t.md). Targets low-latency, multi-agent serving. **All
numbers below are vendor announcement claims**
([blog](../sources/2026-google-blog-tpu-8th-gen.md)); no Cloud docs spec sheet yet —
do not use for roofline math until datasheet figures land.

## Announced claims

| Claim | Value |
|---|---|
| Memory | 288 GB HBM + **384 MB on-chip SRAM** (3× previous gen; sized for reasoning-model KV cache, active working set on-chip) |
| Interconnect | ICI doubled to 19.2 Tb/s (MoE-motivated) |
| Topology | "Boardfly": 4 fully-connected chips → fully-connected groups of 8 boards → 36 groups per pod; >50% smaller network diameter — a departure from pure torus [ICI](../concepts/ici.md) |
| Collectives | on-chip Collectives Acceleration Engine (CAE): up to 5× lower collective latency |
| Economics | claimed 80% better perf/$ vs previous gen; up to 2× perf/W vs Ironwood |
| Host | Axion ARM-based CPUs; bare-metal access |
| Frameworks | JAX, MaxText, PyTorch, SGLang, vLLM |

Notable absence: no FLOPs figure and no HBM bandwidth number were announced.

## See also

- [TPU7x](tpu-7x.md) — current generation; [TPU 8t](tpu-8t.md) — training sibling

## Sources

- [Google blog: 8th-gen TPU announcement](../sources/2026-google-blog-tpu-8th-gen.md)
