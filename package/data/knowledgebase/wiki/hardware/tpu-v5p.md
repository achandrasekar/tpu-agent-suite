---
title: "TPU v5p"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Fifth-generation performance training chip: [megacore](../concepts/megacore.md)
(2 TensorCores), 3D torus, 8960-chip pods — the largest pods before TPU7x. Specs from
[Cloud TPU v5p docs](../sources/2026-08-cloud-tpu-v5p-docs.md).

## Specifications

| Spec | Value |
|---|---|
| Peak compute / chip | 459 TFLOPs bf16; **FP8 also 459** (no low-precision speedup) |
| HBM | 95 GiB, 2765 GBps |
| TensorCores / chip | 2 (megacore); 4 MXUs each |
| [SparseCores](../concepts/sparsecore.md) / chip | 4 |
| [ICI](../concepts/ici.md) | 1200 GBps bidi per chip; 3D torus |
| DCN | 50 Gbps per chip |
| Pod | 8960 chips (2240 hosts, 140 cubes) |
| Max single slice | 6144 chips (16x16x24, 96 cubes); Multislice → 18432 |
| Host VM | 4 chips; 208 vCPU / 448 GB (½ usable under NUMA binding); 2 NUMA nodes; 200 Gbps NIC |

## Topologies

- 3D slices from 2x2x1 up; **slices below a full 4x4x4 cube have no wraparound links**
  (3D-connected but not a torus). Cube (rack) = 64 chips / 16 hosts.
- Twisted torus supported (Cloud TPU API; e.g. 4×4×8 ≈ +70% bisection BW).
- **ICI resiliency** (default-on ≥1 cube): inter-cube optical/OCS faults are routed
  around — availability up, ICI performance temporarily degraded; in-cube copper
  links unaffected.

## See also

- [Critical batch size](../concepts/critical-batch-size.md) — derived compute-bound thresholds for this generation

- [TPU v4](tpu-v4.md) — predecessor; [TPU7x](tpu-7x.md) — successor scale-out chip

## Sources

- [Google Cloud TPU v5p documentation](../sources/2026-08-cloud-tpu-v5p-docs.md)
