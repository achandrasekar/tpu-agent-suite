---
title: "How to Parallelize a Transformer for Training (scaling-book, Part 5)"
type: source
tags: [parallelism, performance, interconnect]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 5 of [*How To Scale Your Model*](2025-scaling-book.md): data parallelism, FSDP (ZeRO),
tensor (Megatron) parallelism, mixed FSDP+TP, pipelining, and cross-pod scaling over DCN —
each as a sharding formula over the two FFW matmuls with a compute-bound condition derived
from `T_math` vs `T_comms`.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/training> — fetched 2026-09-04; chapter last
  changed upstream 2026-08-20 (`cf7e74d`); July 2026 fix to the `dW_in` gradient operand
  order (`46f7d5f`).
- Authority: authorial; thresholds are first-principles with v5p constants
  (`C/W_ici = 2550`, `C/W_dcn ≈ 73,440`).

## Gaps & caveats

- Models the Transformer as a stack of MLP blocks (no attention, no gating matmul);
  valid up to ~32k context per the book.
- Expert parallelism is explicitly out of scope here (covered from the GPU chapter and
  Part 4 on [expert-parallelism](../concepts/expert-parallelism.md)).
- All constants are v5p; per-generation values are derived on the concept pages from the
  Cloud-docs hardware pages and flagged as derived.

## Residue (not extracted)

Three worked problems on LLaMA-2 13B (param count, memory at 16M batch, sharding a 3M
batch on a 16x16x16 v5p slice). Backward-pass comms derivation (Appendix A) summarized
on [sharding-notation](../concepts/sharding-notation.md).

## Connections

Created: [parallelism-strategies](../concepts/parallelism-strategies.md) (hub),
[data-parallelism](../concepts/data-parallelism.md), [fsdp](../concepts/fsdp.md),
[tensor-parallelism](../concepts/tensor-parallelism.md),
[pipeline-parallelism](../concepts/pipeline-parallelism.md).
Updated: [dcn](../concepts/dcn.md), [ici](../concepts/ici.md),
[compute-communication-overlap](../concepts/compute-communication-overlap.md).

## Sources

- `raw/docs/2025-scaling-book-training.html`
