---
title: "Critical batch size (compute-bound thresholds)"
type: concept
tags: [performance, memory, interconnect, precision]
created: 2026-09-04
updated: 2026-09-04
sources: 10
human_reviewed: true
aliases: ["240 rule", "B_crit", "ICI arithmetic intensity", "critical intensity"]
---

The critical batch size is the per-replica **token** batch above which a Transformer
matmul is compute-bound rather than bound by loading its weights:
`B_crit = β · C / W`, where `C` is chip FLOPs/s, `W` the bandwidth the weights cross, and
`β = bits per parameter / bits per activation`. For bf16 on a TPU v5e fed from HBM this is
the book's "240 rule" ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md),
[scaling-book, Part 7](../sources/2025-scaling-book-inference.md)). The same ratio, with ICI or DCN
bandwidth in the denominator, sets every parallelism threshold.

## Why it is a batch size

A `bf16[B,D] · bf16[D,F]` matmul does `2BDF` FLOPs and moves `2BD + 2DF + 2BF` bytes; for
`B ≪ D, F` the bytes are dominated by the weights, so intensity ≈ `B` and the matmul is
compute-bound when `B > C/W_hbm` — 1.97e14 / 8.2e11 ≈ 240 on v5e, ~295 on an H100
([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md)). "Batch" counts tokens across all
sequences in a replica, and the rule is per copy of the weights: sharding a matmul scales
compute and bandwidth together ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md)).

## Per-generation values (derived)

Inputs come from the Cloud-docs [hardware pages](../hardware/tpu-v6e.md) (FLOPs, HBM) and
from the book's Part 2 tables (bidi ICI per axis); the ratios are arithmetic on those
cited numbers *(derived — agent inference, not stated by any source for v6e/TPU7x)*.

| Generation | bf16 FLOPs/s | HBM B/s | `α_hbm = C/W_hbm` | bidi ICI per axis | `α_ici = C/W_ici` | `β = W_hbm/W_ici` | axes |
|---|---|---|---|---|---|---|---|
| [v5e](../hardware/tpu-v5e.md) | 1.97e14 | 8.2e11 (book) / 800 GiBps (docs) | **240** (book) / ~230 | 9e10 | ~2,200 | ~9 (book: ~8) | 2 |
| [v5p](../hardware/tpu-v5p.md) | 4.59e14 | 2.765e12 | ~166 | 1.8e11 | 2,550 | ~15 | 3 |
| [v6e](../hardware/tpu-v6e.md) | 9.18e14 | 1.638e12 | ~560 | 1.8e11 | ~5,100 | ~9 (book: ~8) | 2 |
| [TPU7x](../hardware/tpu-7x.md) | 2.307e15 | 7.38e12 | ~313 | 1.8e11 | ~12,800 | ~41 | 3 |

ICI figures: [scaling-book, Part 2](../sources/2025-scaling-book-tpus.md) (v5e 4.5e10 one-way; v5p, v6e,
TPU7x 9e10 one-way, 1.8e11 bidi). The book's own v5e/v5p statements: 240 and 2,550
([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md),
[scaling-book, Part 5](../sources/2025-scaling-book-training.md)).

**What the table says about newer chips.** FLOPs grew ~5× from v5p to TPU7x while
per-axis ICI stayed at 1.8e11, so every ICI-bound threshold is ~5× tighter: FSDP or DP need
~4,300 tokens per chip over three axes on TPU7x versus 850 on v5p; tensor parallelism for
`F ≈ 28.7k` is compute-bound only to ~2 ways per axis; the mixed FSDP+TP floor
`α²/(M_X M_Y F)` rises from ~113 to ~2,900 tokens per chip. On v6e the same numbers are
~2,550 (two axes), ~5–6 ways, ~900. Treat these as planning estimates pending a TPU7x
collective benchmark *(derived — verify before relying on them)*. The HBM-side threshold
stays in the familiar range on TPU7x (~313) because HBM bandwidth grew with FLOPs, and the
`β` column is the decode model-parallel headroom used on
[tensor-parallelism](tensor-parallelism.md): the book gives ~8 for v5e/v6e
([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)); TPU7x's ~41 means far less
latency-bound sharding headroom.

## The same ratio, other denominators

| Where the weights (or activations) cross | Threshold | Example |
|---|---|---|
| HBM → MXU, single chip | `B > β·α_hbm` | 240 bf16 v5e; **120** with int8 weights + bf16 FLOPs; 240 again with int8×int8 ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md), [scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)) |
| VMEM-resident weights | `B ≳ 10–20` | ~22× HBM bandwidth ([vmem](vmem.md)) |
| ICI, data parallel / FSDP | `B/X > α_ici / M_X` | 2,550 (1 axis) / 850 (3 axes) on v5p ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)) |
| ICI, tensor parallel | `Y < M_Y·F/α_ici` | 8–16 way on v5p ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)) |
| ICI, 2-chip sharded matmul | `D > 2·C/W_ici` | `D > 8,755` on v5e — depends on `D`, not `B` ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md)) |
| DCN, data parallel across slices | `B/slice > C/W_dcn` | ~73k tokens per v5p slice ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)) |
| MoE (HBM) | `B > β·α_hbm·E/k` | 3,840 for DeepSeek-V3 int8 on v5e ([scaling-book, Part 4](../sources/2025-scaling-book-transformers.md)) |
| Per-example weights (`int8[B,D,F]`) | never — intensity ≈ 2 | always comms-bound ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md)) |

## Where it bites

- **Training**: per-replica batch below the ICI threshold means comms-bound no matter the
  MFU target; the fix is more TP or a larger batch
  ([parallelism-strategies](parallelism-strategies.md)).
- **Prefill**: prompts are usually longer than `B_crit`, so prefill is compute-bound
  ([prefill-vs-decode](prefill-vs-decode.md)).
- **Generation**: the total token batch (= concurrent requests) must exceed `B_crit`, which
  means 240 concurrent KV caches at bf16 — hard outside bulk inference; int8 weights halve it
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- Empirically throughput keeps rising until ~240 per data-parallel shard and then flattens;
  the roofline is not perfectly sharp because XLA overlaps imperfectly
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## See also

- [Roofline analysis](roofline-analysis.md), [Quantization](quantization.md),
  [ICI](ici.md), [HBM](hbm.md), [Expert parallelism](expert-parallelism.md)

## Sources

- [How To Scale Your Model, Part 1: Rooflines (scaling-book)](../sources/2025-scaling-book-roofline.md)
- [How To Scale Your Model, Part 2: How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [How To Scale Your Model, Part 4: Transformer Math (scaling-book)](../sources/2025-scaling-book-transformers.md)
- [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-inference.md)
- Hardware inputs: [v5e docs](../sources/2026-08-cloud-tpu-v5e-docs.md), [v5p docs](../sources/2026-08-cloud-tpu-v5p-docs.md), [v6e docs](../sources/2026-08-cloud-tpu-v6e-docs.md), [TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md)
