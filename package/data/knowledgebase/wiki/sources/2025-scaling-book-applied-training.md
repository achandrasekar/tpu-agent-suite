---
title: "Training LLaMA 3 on TPUs (scaling-book, Part 6)"
type: source
tags: [workload, parallelism, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 6 of [*How To Scale Your Model*](2025-scaling-book.md): the Part 4/5 formulas applied to
LLaMA 3-70B on a v5p pod — parameter count from the config, FLOPs per token, training
time at 40% MFU, the memory budget (checkpoints dominate), and the FSDP/TP split that stays
compute-bound at a 4M-token batch.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/applied-training> — fetched 2026-09-04; chapter
  last changed upstream 2026-08-20 (`c6dd456`).
- Authority: authorial worked example; the LLaMA 3-70B config is from the public
  Hugging Face card.

## Gaps & caveats

- v5p-only numbers (8960-chip pod, 96 GB HBM). Re-derive on v6e/TPU7x with the
  [size-training-parallelism](../playbooks/size-training-parallelism.md) playbook.
- Two open worked problems (4-pod scaling, LLaMA 3-405B) have no answers upstream.

## Connections

Created: [mfu](../concepts/mfu.md),
[size-training-parallelism](../playbooks/size-training-parallelism.md).
Updated: [gradient-checkpointing](../concepts/gradient-checkpointing.md),
[tensor-parallelism](../concepts/tensor-parallelism.md), [fsdp](../concepts/fsdp.md),
[transformer-flops-accounting](../concepts/transformer-flops-accounting.md).

## Sources

- `raw/docs/2025-scaling-book-applied-training.html`
