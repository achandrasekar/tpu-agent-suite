---
title: "Matrix Multiply Unit (MXU)"
type: concept
tags: [hardware, performance]
created: 2026-08-22
updated: 2026-09-04
sources: 7
human_reviewed: false
---

The MXU is the matrix-multiplication engine at the heart of every
[TensorCore](tensorcore.md) — a [systolic array](systolic-array.md) that
delivers the vast majority of a TPU's FLOPs.

## Behavior

- Performs one `bf16[8,128] @ bf16[128,128] -> f32[8,128]` multiply every 8 cycles;
  ~5e13 bf16 FLOPs/s per MXU at 1.5 GHz on v5e
  ([source](../sources/2025-scaling-book-tpus.md)).
- The array is 128×128 on older generations; **256×256 on TPU v6e (Trillium)**
  ([source](../sources/2025-scaling-book-tpus.md)) **and later generations**
  *(human-reviewed assertion 2026-08-24 — not yet stated by an ingested source)*.
  The larger array quadruples FLOPs/cycle but doubles the tensor dimensions needed
  to saturate it.
- TensorCores carry multiple MXUs — 2 on v3
  ([source](../sources/2025-scaling-book-tpus.md)); 4 on
  [v4](../sources/2026-08-cloud-tpu-v4-docs.md)/[v5e](../sources/2026-08-cloud-tpu-v5e-docs.md)/[v5p](../sources/2026-08-cloud-tpu-v5p-docs.md);
  2 (of the larger 256×256 array) on [v6e](../sources/2026-08-cloud-tpu-v6e-docs.md) —
  so effective tiling dimensions should exceed array-size × #MXUs.
- Lower-precision inputs raise throughput where supported: int8 ≈ 2× and int4 ≈ 4× bf16
  ([source](../sources/2025-scaling-book-tpus.md)); see
  [Quantization](quantization.md).

## Optimization implications

- Operand dimensions must be ≥128 (256 on v6e+) or they are padded — wasted compute; see
  [Padding](padding.md).
- Feeding the MXU is the game: it reads from [VMEM](vmem.md), so arithmetic intensity
  vs HBM bandwidth decides compute- vs memory-bound; see
  [Roofline analysis](roofline-analysis.md).

## Versus GPU Tensor Cores

An H100 has 528 Tensor Cores (4 per SM), each ~1024 bf16 FLOPs/cycle — roughly an 8×8×8
matmul — versus 8 MXUs of 128×128 on a v5p; the GPU's matmul critical intensity is
similar (~295 vs 240 on v5e), and B200's Tensor Cores outgrew SMEM enough to need a new
TMEM ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)). See [GPU vs TPU](gpu-vs-tpu.md).

## See also

- [Systolic array](systolic-array.md) — how the 8-cycle matmul works
- [VPU](vpu.md) — everything the MXU doesn't do

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- Cloud TPU docs: [v4](../sources/2026-08-cloud-tpu-v4-docs.md),
  [v5e](../sources/2026-08-cloud-tpu-v5e-docs.md),
  [v5p](../sources/2026-08-cloud-tpu-v5p-docs.md),
  [v6e](../sources/2026-08-cloud-tpu-v6e-docs.md),
  [TPU7x](../sources/2026-08-cloud-tpu-7x-docs.md)
- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
