---
title: "Quantization"
type: concept
tags: [precision, performance]
created: 2026-08-22
updated: 2026-10-14
sources: 9
human_reviewed: true
---

Quantization runs computation at reduced numerical precision to raise throughput and
shrink memory footprint. On TPU the payoff is native where the spec lists an int8 rate (v5e, v5p,
v6e): int8 runs at ~2× bf16 matmul throughput. The scaling book's "int4 at ~4×"
applies only to generations that publish an int4 rate, and no current Cloud TPU
spec, v6e included, lists one: there int4 is a storage format, unpacked to bf16
or int8 before the MXU, and saves only HBM bytes. Meanwhile
VPU (`index.py get vpu`) (vector) ops remain fp32
(source (`index.py get scaling-book-tpus`)).

Lower precision also halves/quarters bytes moved, which relaxes every
roofline (`index.py get concept-roofline-analysis`): memory-bound workloads speed up from bandwidth
savings even before the FLOPs gain.

These are peak and byte bounds. With weight-only or KV-cache quantization the MXU
still runs bf16 and the dequantize or unpack runs on the VPU; on v6e such kernels
measured 1.15–1.27× (int8 and int4 expert weights, int4 KV cache), so treat the
bandwidth saving as an upper bound (measured: `index.py get v6e-measured-moe-kernels`,
`index.py get v6e-measured-decode-tricks`). A W4A16 matmul was ~1.1× bf16 weights at
decode shapes (`index.py get v6e-measured-matmul-quantization`).

Native low-precision throughput varies by generation — check before assuming the 2×:

| Generation | Low-precision speedup vs bf16 |
|---|---|
| v4 (`index.py get hw-tpu-v4`) | int8 = 1× (275 TFLOPs either way) (docs (`index.py get cloud-tpu-v4`)) |
| v5e (`index.py get hw-tpu-v5e`) | int8 2× (docs (`index.py get cloud-tpu-v5e`)) |
| v5p (`index.py get hw-tpu-v5p`) | FP8 = 1× (459 TFLOPs either way) (docs (`index.py get cloud-tpu-v5p`)) |
| v6e (`index.py get hw-tpu-v6e`) | int8 2× (docs (`index.py get cloud-tpu-v6e`)) |
| TPU7x (`index.py get hw-tpu-7x`) | FP8 2× (docs (`index.py get cloud-tpu-7x`)) |

*Methods (calibration, accuracy trade-offs, int4) await further sources.*

## Effect on rooflines (inference)

- Weight precision moves the critical batch size (`index.py get concept-critical-batch-size`):
  `B_crit = (bits per param / bits per activation) · C / W_hbm`. int8 weights with bf16
  FLOPs halve it (240 → 120 on v5e); int8 weights *and* int8 FLOPs land back at ~240
  because the int8 peak is 2× (scaling-book, Part 1 (`index.py get scaling-book-roofline`),
  scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)). Quantizing weights
  losslessly is easier than low-precision arithmetic, so int8-weight/bf16-FLOPs is the
  common case (scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).
- Tensor-parallel limits are precision-independent (comms and FLOPs halve together)
  (scaling-book, Part 5 (`index.py get scaling-book-training`)); but a lower `B_crit` lets a
  larger batch stay HBM-bound, which is what made int8/int4 LLaMA 3-70B serveable with
  pure model parallelism on a v5e 4×8 where bf16 was not
  (scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).
- Memory fit decides the minimum topology: LLaMA 3-70B on v5e — bf16 140 GB → 4×4,
  int8 70 GB → 4×2, int4 35 GB → 2×2 (scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).
- KV caches quantize too (int8/int4/fp8), usually post-training, cutting both bytes and
  bandwidth per step (kv-cache (`index.py get concept-kv-cache`)); GPUs report similar 2× fp8 gains
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## See also

- MXU (`index.py get concept-mxu`), Roofline analysis (`index.py get concept-roofline-analysis`),
  Critical batch size (`index.py get concept-critical-batch-size`), KV cache (`index.py get concept-kv-cache`)
- FP8 slower than BF16 on v6e (`index.py get xla-8bit-layout-copies`) — the 8-bit relayout copies behind it
- INT8 matmul in a Pallas kernel (`index.py get pallas-int8-matmul`) — int32 accumulation, and W8A8 accuracy on one model

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- Cloud TPU docs: v4 (`index.py get cloud-tpu-v4`),
  v5e (`index.py get cloud-tpu-v5e`),
  v5p (`index.py get cloud-tpu-v5p`),
  v6e (`index.py get cloud-tpu-v6e`),
  TPU7x (`index.py get cloud-tpu-7x`)
- How To Scale Your Model, Part 1: Rooflines (scaling-book) (`index.py get scaling-book-roofline`),
  How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`),
  How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-inference`)
