---
title: "How to Think About GPUs (scaling-book, Part 12)"
type: source
tags: [hardware, interconnect, parallelism, onboarding]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 12 of [*How To Scale Your Model*](2025-scaling-book.md) (added 2025-08-18): GPU chip
anatomy mapped 1:1 onto TPU blocks, spec tables for V100–B200, NVLink nodes and the
InfiniBand fat-tree scale-out network, collective costs within and across nodes (SHARP,
empirical bandwidth shortfalls), and LLM scaling rooflines on GPUs (DP/TP/EP/PP) with
DeepSeek-V3 and LLaMA-3 as worked configurations.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/gpus> — fetched 2026-09-04; chapter last changed
  upstream 2026-07-12 (`3980377`, reader-reported errata); earlier July fixes corrected the
  spine-switch bandwidth (`60a6fa6`) and fat-tree per-node bandwidth (`ed80595`).
- Authority: authorial, reviewed by named GPU practitioners; NVIDIA specs are vendor
  numbers; the empirical AllReduce/AllGather plots are the book's own measurements.

## Gaps & caveats

- GPU numbers are H100/B200-era; GB200 NVL72 is covered in an appendix only.
- The "critical batch ~2500 tokens per GPU" rooflines assume the reference DGX SuperPod
  fat tree; Meta's LLaMA-3 cluster differed (footnote).
- Not an onboarding guide: the mapping table and the "what changes on TPU" reasoning were
  synthesized on [gpu-vs-tpu](../concepts/gpu-vs-tpu.md) and are marked where inferred.

## Residue (not extracted)

Five quizzes (H100 CUDA-core counts, vector FLOPs, matmul intensity, node bandwidth,
bisection bandwidth, sharded AllGather, SHARP byte accounting, LLaMA-3 70B on 4096 H100s,
Megatron-LM hyperparameters). Lifted: H100/B200 matmul critical intensity ≈ 295/281;
in-network reductions deliver ~30% not 2×.

## Connections

Created: [gpu-vs-tpu](../concepts/gpu-vs-tpu.md).
Updated: [collective-costs](../concepts/collective-costs.md) (GPU column),
[ici](../concepts/ici.md), [mxu](../concepts/mxu.md), [vmem](../concepts/vmem.md),
[vpu](../concepts/vpu.md), [expert-parallelism](../concepts/expert-parallelism.md),
[pipeline-parallelism](../concepts/pipeline-parallelism.md),
[data-parallelism](../concepts/data-parallelism.md).

## Sources

- `raw/docs/2025-scaling-book-gpus.html`
