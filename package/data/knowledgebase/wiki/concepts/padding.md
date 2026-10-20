---
title: "Padding (MXU tiling requirement)"
type: concept
tags: [performance, compiler]
created: 2026-08-22
updated: 2026-09-04
sources: 1
human_reviewed: false
---

TPU matmul operands must fill the [MXU](mxu.md)'s systolic array: matrices
are padded to at least 128 per dimension — **256 on TPU v6e**
([source](../sources/2025-scaling-book-tpus.md)) **and later generations**
*(human-reviewed assertion 2026-08-24)* — and smaller axes are padded up in hardware,
spending FLOPs on zeros.

## Optimization implications

- Choose model dimensions (and shard sizes after partitioning!) that are multiples of
  128 (256 on v6e+); a sharded axis that drops below the tile size wastes proportional
  compute.
- With multiple MXUs per core (2 on v3, 4 on v4/v5), effective tiling wants dimensions
  above 128 × #MXUs ([source](../sources/2025-scaling-book-tpus.md)).
- On v6e the doubled array means shapes that saturated older chips may only 25%-fill
  the MXU ([source](../sources/2025-scaling-book-tpus.md)).

## See also

- [HLO](hlo.md) — where a tensor's tiling and padding are visible (`T(8,128)`)

- [Systolic array](systolic-array.md), [MXU](mxu.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
