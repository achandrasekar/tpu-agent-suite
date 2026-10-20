---
title: "TPU v4"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

Fourth-generation TPU: 3D-torus training chip, 4096-chip pods, [megacore]
(../concepts/megacore.md) (2 TensorCores, unified HBM). Legacy tier — Cloud TPU API in
maintenance mode; GKE recommended. Specs from
[Cloud TPU v4 docs](../sources/2026-08-cloud-tpu-v4-docs.md).

## Specifications

| Spec | Value |
|---|---|
| Peak compute / chip | 275 TFLOPs (bf16 **or** int8 — no int8 speedup) |
| HBM | 32 GiB HBM2, 1200 GBps |
| TensorCores / chip | 2 (megacore); 4 MXUs each |
| Measured power (min/mean/max) | 90 / 170 / 192 W |
| Pod size | 4096 chips; 1.1 bf16 exaFLOPs |
| All-reduce / bisection BW per pod | 1.1 PB/s / 24 TB/s |
| Interconnect | 3D mesh/torus, 6 [ICI](../concepts/ici.md) links per chip |
| Host link | x16 PCIe gen3, direct connect |

## Topologies

- AxBxC with A≤B≤C; all dims ≤4, or all multiples of 4. Slices ≥64 chips build from
  4x4x4 cubes; small slices: v4-8 (2x2x1) … v4-64 (2x4x4).
- Same chip count, multiple shapes (512 chips: 4x4x32 / 4x8x16 / 8x8x8; default ≈ most
  cubic). Match parallelism axes to topology dims — e.g. 8-way model parallelism fits
  8x16x16, not 4-way.
- **Twisted torus** (2A=B=C or 2A=2B=C, e.g. 4×4×8, 4×8×8, 12×12×24): up to ~70%
  (4×4×8) / ~40% (4×8×8) more bisection bandwidth; Cloud TPU API only; 1–2 pp MFU
  gain observed on FSDP MaxText.
- NUMA binding (`numactl --cpunodebind=0`) recommended for CPU-heavy input pipelines.

## See also

- [TPU v5p](tpu-v5p.md) — successor training chip

## Sources

- [Google Cloud TPU v4 documentation](../sources/2026-08-cloud-tpu-v4-docs.md)
