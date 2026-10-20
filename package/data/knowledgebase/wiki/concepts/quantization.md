---
title: "Quantization"
type: concept
tags: [precision, performance]
created: 2026-08-22
updated: 2026-09-04
sources: 9
human_reviewed: true
---

Quantization runs computation at reduced numerical precision to raise throughput and
shrink memory footprint. On TPU the payoff is native: generations that support it
execute int8 at ~2× and int4 at ~4× bf16 matmul throughput, while
[VPU](vpu.md) (vector) ops remain fp32
([source](../sources/2025-scaling-book-tpus.md)).

Lower precision also halves/quarters bytes moved, which relaxes every
[roofline](roofline-analysis.md): memory-bound workloads speed up from bandwidth
savings even before the FLOPs gain.

Native low-precision throughput varies by generation — check before assuming the 2×:

| Generation | Low-precision speedup vs bf16 |
|---|---|
| [v4](../hardware/tpu-v4.md) | int8 = 1× (275 TFLOPs either way) ([docs](../sources/2026-08-cloud-tpu-v4-docs.md)) |
| [v5e](../hardware/tpu-v5e.md) | int8 2× ([docs](../sources/2026-08-cloud-tpu-v5e-docs.md)) |
| [v5p](../hardware/tpu-v5p.md) | FP8 = 1× (459 TFLOPs either way) ([docs](../sources/2026-08-cloud-tpu-v5p-docs.md)) |
| [v6e](../hardware/tpu-v6e.md) | int8 2× ([docs](../sources/2026-08-cloud-tpu-v6e-docs.md)) |
| [TPU7x](../hardware/tpu-7x.md) | FP8 2× ([docs](../sources/2026-08-cloud-tpu-7x-docs.md)) |

*Methods (calibration, accuracy trade-offs, int4) await further sources.*

## Effect on rooflines (inference)

- Weight precision moves the [critical batch size](critical-batch-size.md):
  `B_crit = (bits per param / bits per activation) · C / W_hbm`. int8 weights with bf16
  FLOPs halve it (240 → 120 on v5e); int8 weights *and* int8 FLOPs land back at ~240
  because the int8 peak is 2× ([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md),
  [scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)). Quantizing weights
  losslessly is easier than low-precision arithmetic, so int8-weight/bf16-FLOPs is the
  common case ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- Tensor-parallel limits are precision-independent (comms and FLOPs halve together)
  ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)); but a lower `B_crit` lets a
  larger batch stay HBM-bound, which is what made int8/int4 LLaMA 3-70B serveable with
  pure model parallelism on a v5e 4×8 where bf16 was not
  ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- Memory fit decides the minimum topology: LLaMA 3-70B on v5e — bf16 140 GB → 4×4,
  int8 70 GB → 4×2, int4 35 GB → 2×2 ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- KV caches quantize too (int8/int4/fp8), usually post-training, cutting both bytes and
  bandwidth per step ([kv-cache](kv-cache.md)); GPUs report similar 2× fp8 gains
  ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## See also

- [MXU](mxu.md), [Roofline analysis](roofline-analysis.md),
  [Critical batch size](critical-batch-size.md), [KV cache](kv-cache.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- Cloud TPU docs: [v4](../sources/2026-08-cloud-tpu-v4-docs.md),
  [v5e](../sources/2026-08-cloud-tpu-v5e-docs.md),
  [v5p](../sources/2026-08-cloud-tpu-v5p-docs.md),
  [v6e](../sources/2026-08-cloud-tpu-v6e-docs.md),
  [TPU7x](../sources/2026-08-cloud-tpu-7x-docs.md)
- [How To Scale Your Model, Part 1: Rooflines (scaling-book)](../sources/2025-scaling-book-roofline.md),
  [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md),
  [How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-inference.md)
