---
title: "TPU v4"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-10-03
human_reviewed: false
source_ids:
  - cloud-tpu-v4
  - xla-scoped-vmem
  - openxla-sparsecore
---

Fourth-generation TPU: 3D-torus training chip, 4096-chip pods, megacore (`index.py get megacore`) (2 TensorCores, unified HBM). Legacy tier *(knowledge base note)* — Cloud TPU API in
maintenance mode; GKE recommended. Specs from
Cloud TPU v4 docs (`index.py get cloud-tpu-v4`), last checked against the page on
2026-10-02. Anything the page does not state is marked *(knowledge base note)*.
Quota in `us-central2-b`, which currently offers only TPU v4, is approved by
hand: none is granted by default.

## Specifications

Primary source.[^cloud-tpu-v4]

| Spec | Value |
|---|---|
| Peak compute / chip | 275 TFLOPs (bf16 **or** int8 — no int8 speedup *(knowledge base note)*) |
| HBM | 32 GiB HBM2, 1200 GBps |
| TensorCores / chip | 2 (megacore); 4 MXUs each |
| Measured power (min/mean/max) | 90 / 170 / 192 W |
| Pod size | 4096 chips; 1.1 bf16 exaFLOPs |
| All-reduce / bisection BW per pod | 1.1 PB/s / 24 TB/s |
| Interconnect | 3D mesh/torus, 6 ICI (`index.py get concept-ici`) links per chip |
| Host link | x16 PCIe gen3, direct connect |
| Typical usable scoped VMEM | 16,384 KiB / 16,777,216 bytes (16 MiB, derived); [^xla-scoped-vmem] |

## Topologies

- AxBxC with A≤B≤C; all dims ≤4, or all multiples of 4. Larger slices build from
  4x4x4 cubes. The connections form a 3D mesh, configurable as a 3D torus where
  2A=B=C or 2A=2B=C and each dimension is a multiple of 4.
- A slice's name counts **TensorCores**, not chips: v4-8 is 4 chips. Small
  slices (under 64 chips) can be named either way; large ones come in steps of
  64 chips, and some shapes exist only as custom topologies, created with
  `--type` and `--topology`.

  | Name | Chips | Topology |
  |---|---|---|
  | v4-8 | 4 | 2x2x1 |
  | v4-16 | 8 | 2x2x2 |
  | v4-32 | 16 | 2x2x4 |
  | v4-64 | 32 | 2x4x4 |
  | v4-128 | 64 | 4x4x4 |
  | v4-256 | 128 | 4x4x8 |
  | v4-512 | 256 | 4x8x8 |
  | custom | 256 | 4x4x16 |
  | v4-1024 | 512 | 8x8x8 |
  | v4-1536 | 768 | 8x8x12 |
  | v4-2048 | 1024 | 8x8x16 |
  | custom | 1024 | 4x16x16 |
  | v4-4096 | 2048 | 8x16x16 |

  The page's large-slice table is a set of examples and ends with "…".
- Same chip count, multiple shapes (512 chips: 4x4x32 / 4x8x16 / 8x8x8; 2048
  chips: 4x4x128 / 4x8x64 / 4x16x32 / 8x16x16; default ≈ most cubic). Match parallelism axes to topology dims — e.g. 8-way model parallelism fits
  8x16x16, not 4-way.
- **Twisted torus** (2A=B=C or 2A=2B=C, e.g. 4×4×8, 4×8×8, 12×12×24): up to ~70%
  (4×4×8) / ~40% (4×8×8) more bisection bandwidth; Cloud TPU API only; 1–2 pp MFU
  gain observed on FSDP MaxText.
- NUMA binding (`numactl --cpunodebind=0`) recommended for CPU-heavy input pipelines.

## Memory and buffers

Scoped VMEM is a compiler allocation limit; see [memory and buffer sizing](../concepts/memory-buffers.md)
for allocation scopes, padded sizes and XProf inspection.

SparseCore: 4 per chip, 16 tiles each, SIMD width 8.
[^openxla-sparsecore].

## See also

- TPU v5p (`index.py get hw-tpu-v5p`) — successor training chip

<!-- sources:generated:start -->

## References

[^cloud-tpu-v4]: [Google Cloud TPU v4](https://docs.cloud.google.com/tpu/docs/v4). specification. Generation specifications, topology and host configurations. URL checked 2026-10-03.

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

<!-- sources:generated:end -->
