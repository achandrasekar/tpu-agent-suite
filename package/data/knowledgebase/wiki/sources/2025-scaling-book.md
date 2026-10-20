---
title: "How To Scale Your Model (scaling-book) — book hub"
type: source
tags: [performance, parallelism, hardware, framework]
created: 2026-09-04
updated: 2026-09-04
sources: 2
human_reviewed: true
aliases: ["scaling-book", "jax-ml/scaling-book", "How to Scale Your Model"]
---

Google DeepMind's open textbook *How To Scale Your Model: A Systems View of LLMs on TPUs*
(Austin, Douglas, Frostig, Levskaya, Chen, Vikram, Lebron, Choy, Ramasesh, Webson, Pope;
first published 2025-02-04, continuously revised). One thesis carried through twelve
chapters: every scaling decision reduces to a roofline —
`max(T_math, T_comms)` — applied at the right bandwidth tier (HBM, ICI, PCIe, DCN).
This page is the book-level provenance record; each chapter has its own source page.

## Identity & authority

- Site: <https://jax-ml.github.io/scaling-book/> — source repo
  <https://github.com/jax-ml/scaling-book>.
- **Snapshot**: all thirteen pages fetched 2026-09-04; upstream HEAD at fetch time
  `ac72f02` (2026-09-03). Per-chapter last-change commits are recorded on each chapter
  page so a future lint can diff against upstream by commit rather than by date.
- Authority: authored by TPU/JAX practitioners at Google DeepMind; not peer reviewed;
  hardware numbers are book-reported (the [hardware pages](../hardware/tpu-v6e.md)
  from Cloud docs win for spec lookups). Worked numbers lean on v5e/v5p; TPU7x appears
  only in Part 2's spec tables (added upstream 2026-05).

## Chapter map

| Part | Source page | Feeds |
|---|---|---|
| 1 Rooflines | [roofline](2025-scaling-book-roofline.md) | [roofline-analysis](../concepts/roofline-analysis.md), [critical-batch-size](../concepts/critical-batch-size.md) |
| 2 TPUs | [tpus](2025-scaling-book-tpus.md) | hardware-block concept pages (ingested 2026-08-22) |
| 3 Sharded matrices | [sharding](2025-scaling-book-sharding.md) | [sharding-notation](../concepts/sharding-notation.md), [collective-costs](../concepts/collective-costs.md), [collective-matmul](../concepts/collective-matmul.md) |
| 4 Transformer math | [transformers](2025-scaling-book-transformers.md) | [transformer-flops-accounting](../concepts/transformer-flops-accounting.md), [kv-cache](../concepts/kv-cache.md), [gradient-checkpointing](../concepts/gradient-checkpointing.md), [expert-parallelism](../concepts/expert-parallelism.md) |
| 5 Training parallelism | [training](2025-scaling-book-training.md) | [parallelism-strategies](../concepts/parallelism-strategies.md) hub and its four pages |
| 6 Training LLaMA 3 | [applied-training](2025-scaling-book-applied-training.md) | [mfu](../concepts/mfu.md), [size-training-parallelism](../playbooks/size-training-parallelism.md) |
| 7 Inference | [inference](2025-scaling-book-inference.md) | [prefill-vs-decode](../concepts/prefill-vs-decode.md), [kv-cache](../concepts/kv-cache.md) |
| 8 Serving LLaMA 3 | [applied-inference](2025-scaling-book-applied-inference.md) | [size-serving-topology](../playbooks/size-serving-topology.md) |
| 9 Profiling | [profiling](2025-scaling-book-profiling.md) | [hlo](../concepts/hlo.md), [xprof](../codebases/xprof.md), [profile-tpu-program](../playbooks/profile-tpu-program.md) |
| 10 JAX | [jax-stuff](2025-scaling-book-jax-stuff.md) | [jax](../codebases/jax.md), [collective-matmul](../concepts/collective-matmul.md) |
| 11 Conclusion | this page (further-reading list below) | — |
| 12 GPUs | [gpus](2025-scaling-book-gpus.md) | [gpu-vs-tpu](../concepts/gpu-vs-tpu.md) |

## Gaps & caveats

- JAX-first: every code example is JAX. Nothing here states how torch_tpu exposes the
  same mechanisms; concept pages keep the mechanism framework-neutral and leave the
  TorchTPU mapping marked *not yet ingested*.
- No Pallas content: the book defers kernels to the Pallas series (fetch target below).
- Worked-problem sets (58 questions) were deliberately **not** captured (editorial
  decision 2026-09-04); the raw snapshots retain them.

## Further-reading fetch targets (from Part 11, not yet ingested)

TPU Deep Dive; *Efficiently Scaling Transformer Inference* (arXiv 2211.05102, the
inference chapter's inspiration); *Writing TPU Kernels with Pallas* series; JAX
*Distributed arrays and automatic parallelization* guide; Hugging Face Ultra-Scale
Playbook; *The Anatomy of Collective Communication* (better N-D/GPU collectives
writeup); Rafi Witten's High Performance LLMs 2024 slides; Stanford CS336.

## Connections

Created: the eleven chapter source pages listed above and every wiki page they feed.

## Sources

- `raw/docs/2025-scaling-book-index.html` — landing page (intro + outline), fetched
  2026-09-04; chapter file last changed upstream 2026-05-09 (`7a064c9`).
- `raw/docs/2025-scaling-book-conclusion.html` — acknowledgments + further reading,
  fetched 2026-09-04; last changed 2026-07-26 (`d8086c7`).
