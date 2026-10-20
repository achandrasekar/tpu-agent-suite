---
title: "Matrix Multiply Unit (MXU)"
type: concept
tags: [hardware, performance]
created: 2026-08-22
updated: 2026-10-14
sources: 7
human_reviewed: false
---

The MXU is the matrix-multiplication engine at the heart of every
TensorCore (`index.py get concept-tensorcore`) — a systolic array (`index.py get concept-systolic-array`) that
delivers the vast majority of a TPU's FLOPs.

## Behavior

- Performs one `bf16[8,128] @ bf16[128,128] -> f32[8,128]` multiply every 8 cycles;
  ~5e13 bf16 FLOPs/s per MXU at 1.5 GHz on v5e
  (source (`index.py get scaling-book-tpus`)).
- The array is 128×128 on older generations; **256×256 on TPU v6e (Trillium)**
  (source (`index.py get scaling-book-tpus`)) **and later generations**
  *(human-reviewed assertion 2026-08-24 — not yet stated by an ingested source)*.
  The larger array quadruples FLOPs/cycle but doubles the tensor dimensions needed
  to saturate it.
- TensorCores carry multiple MXUs — 2 on v3
  (source (`index.py get scaling-book-tpus`)); 4 on
  v4 (`index.py get cloud-tpu-v4`)/v5e (`index.py get cloud-tpu-v5e`)/v5p (`index.py get cloud-tpu-v5p`);
  2 (of the larger 256×256 array) on v6e (`index.py get cloud-tpu-v6e`) —
  so effective tiling dimensions should exceed array-size × #MXUs.
- Lower-precision inputs raise throughput where the spec lists the rate: int8 ≈ 2× bf16
  on v5e, v5p and v6e (source (`index.py get scaling-book-tpus`)). No current Cloud TPU
  spec, v6e included, lists an int4 MXU rate, so int4 is a storage format there; see
  Quantization (`index.py get concept-quantization`).

## Optimization implications

- Operand dimensions must be ≥128 (256 on v6e+) or they are padded — wasted compute; see
  Padding (`index.py get padding`).
- Feeding the MXU is the game: it reads from VMEM (`index.py get concept-vmem`), so arithmetic intensity
  vs HBM bandwidth decides compute- vs memory-bound; see
  Roofline analysis (`index.py get concept-roofline-analysis`).

## Versus GPU Tensor Cores

An H100 has 528 Tensor Cores (4 per SM), each ~1024 bf16 FLOPs/cycle — roughly an 8×8×8
matmul — versus 8 MXUs of 128×128 on a v5p; the GPU's matmul critical intensity is
similar (~295 vs 240 on v5e), and B200's Tensor Cores outgrew SMEM enough to need a new
TMEM (scaling-book, Part 12 (`index.py get scaling-book-gpus`)). See GPU vs TPU (`index.py get concept-gpu-vs-tpu`).

## See also

- Systolic array (`index.py get concept-systolic-array`) — how the 8-cycle matmul works
- VPU (`index.py get vpu`) — everything the MXU doesn't do
- INT8 matmul in a Pallas kernel (`index.py get pallas-int8-matmul`) — what the int8 MXU path needs

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- Cloud TPU docs: v4 (`index.py get cloud-tpu-v4`),
  v5e (`index.py get cloud-tpu-v5e`),
  v5p (`index.py get cloud-tpu-v5p`),
  v6e (`index.py get cloud-tpu-v6e`),
  TPU7x (`index.py get cloud-tpu-7x`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
