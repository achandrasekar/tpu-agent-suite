---
title: "HBM (High Bandwidth Memory)"
type: concept
tags: [hardware, memory]
created: 2026-08-22
updated: 2026-09-04
sources: 2
human_reviewed: true
---

HBM is the TPU chip's main memory: tens of GB holding the tensors the
[TensorCore](tensorcore.md) works on. Tensors stream HBM → [VMEM](vmem.md) → compute
units and back ([source](../sources/2025-scaling-book-tpus.md)).

## Role in performance

- HBM bandwidth (~1–2 TB/s; e.g. 2.8 TB/s on v5p) is the top of the communication
  hierarchy — faster than [ICI](ici.md), [PCIe](pcie.md), and [DCN](dcn.md) — and is
  the limiting rate for memory-bound workloads
  ([source](../sources/2025-scaling-book-tpus.md)).
- Sampling-latency floor: each decode step must load all resident parameters from HBM,
  so params ÷ aggregate HBM bandwidth lower-bounds per-step latency (~10 ms for 200B
  bf16 params on 32 v4p chips) ([source](../sources/2025-scaling-book-tpus.md)). See
  [Roofline analysis](roofline-analysis.md).
- A matmul fed from HBM needs arithmetic intensity ≈ 240 to be compute-bound, vs
  ~10–20 from VMEM ([source](../sources/2025-scaling-book-tpus.md)).

- Decode is HBM-bound on both terms: `T_step ≈ B·KV / W_hbm + max(2·B·params / C, params / W_hbm)`,
  with attention (the KV term) never compute-bound; throughput per chip saturates near
  the [critical batch size](critical-batch-size.md) and, at long context, KV loading
  rather than parameters dominates ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
  See [prefill vs decode](prefill-vs-decode.md).

Per-generation capacities and bandwidths: see the `wiki/hardware/` generation pages
([v4](../hardware/tpu-v4.md), [v5e](../hardware/tpu-v5e.md),
[v5p](../hardware/tpu-v5p.md), [v6e](../hardware/tpu-v6e.md),
[7x](../hardware/tpu-7x.md)).

## See also

- [VMEM](vmem.md) — the faster level below

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
