---
title: "ICI (Inter-Chip Interconnect)"
type: concept
tags: [hardware, interconnect]
created: 2026-08-22
updated: 2026-09-04
sources: 6
human_reviewed: true
---

ICI is the direct chip-to-chip network inside a TPU pod — links that bypass the hosts
entirely. It is **nearest-neighbor only**: communication between distant chips hops
through intervening chips ([source](../sources/2025-scaling-book-tpus.md)).

## Topology

- **2D torus** (4 neighbors + edge wraparound): v2, v3, v5e, v6e. v5e/v6e pods max out
  at 16×16 and scale further only over [DCN](dcn.md)
  ([source](../sources/2025-scaling-book-tpus.md)).
- **3D torus** (6 neighbors): v4, v5p. Superpods (16×16×16 v4; 16×20×28 v5p) are built
  from reconfigurable 4×4×4 cubes joined by optical wraparound links
  ([source](../sources/2025-scaling-book-tpus.md)).
- A torus halves the max hop distance (N → N/2); a "twisted torus" reduces average
  distance further ([source](../sources/2025-scaling-book-tpus.md)).
- **Twisted torus** (v4/v5p, Cloud TPU API only; shapes where each dim = or 2× the
  smallest, e.g. 4×4×8, 4×8×8): rewires wraparound edges to make an asymmetric torus
  symmetric — up to ~70% (4×4×8) / ~40% (4×8×8) more bisection bandwidth, better load
  balancing, shorter routes; 1–2 pp MFU gain on FSDP MaxText
  ([v4 docs](../sources/2026-08-cloud-tpu-v4-docs.md),
  [v5p docs](../sources/2026-08-cloud-tpu-v5p-docs.md)).
- **ICI resiliency** (v4/v5p, default-on for ≥1-cube slices): inter-cube optical/OCS
  faults are routed around (in-cube copper unaffected) — better availability at the
  cost of temporarily degraded ICI performance
  ([v5p docs](../sources/2026-08-cloud-tpu-v5p-docs.md)).

## Wraparound rules (bite in practice)

- 3D-torus slices get wraparounds only as multiples of a full 4×4×4 cube (4×4×8 yes,
  2×2×4 no); v5e/v6e get a wraparound only on an axis of size 16 (8×16 → long axis
  only). **Missing wraparounds ≈ 2× communication time**
  ([source](../sources/2025-scaling-book-tpus.md)).
- Bidirectional ("bidi") bandwidth = 2× one-way, but only with a functioning ring —
  i.e. with wraparound. Roofline math usually uses bidi
  ([source](../sources/2025-scaling-book-tpus.md)).

## Numbers

~9e10 B/s one-way per link on v5p (3 axes/chip), ~4.5e10 on v5e/v4p; per-hop latency
~1 µs (v5e). Slower than [HBM](hbm.md) (~30×), far faster than [DCN](dcn.md)
([source](../sources/2025-scaling-book-tpus.md)). Transfer time = hop latency +
bytes / (ports × link BW); source and destination each have 2 usable ports in a
wrap-less 2D slice ([source](../sources/2025-scaling-book-tpus.md)).

## Collective costs on ICI

- A bandwidth-bound AllGather or ReduceScatter over a full ring costs `bytes / W_ici`
  regardless of axis size; AllReduce 2×; AllToAll ¼ — formulas on
  [collective-costs](collective-costs.md) ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- **ICI arithmetic intensity** `C / W_ici` (2,550 on v5p per axis) is the per-chip token
  batch needed for data-parallel/FSDP to be compute-bound; per-generation values on
  [critical-batch-size](critical-batch-size.md) ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- Latency floor ~1 µs per hop: buffers under ~45 kB (v5e) are latency-bound, which
  small-batch decode collectives routinely are ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- Measured: a v5e 8×16 AllGather reaches ~95% of the claimed link bandwidth at ~10 MB
  total, far smaller messages than GPUs need ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

## vs GPU fabrics

GPUs use switch hierarchies approximating all-to-all (8–72 GPUs directly linked, then
O(log N) hops); TPUs trade arbitrary point-to-point reach for constant per-device link
count/bandwidth — cheaper, simpler, larger max scale, but sharding layouts must respect
neighbor locality ([source](../sources/2025-scaling-book-tpus.md)).

## See also

- [DCN](dcn.md) — between slices; [Roofline analysis](roofline-analysis.md)
- [Collective costs](collective-costs.md), [GPU vs TPU](gpu-vs-tpu.md) (NVLink/InfiniBand numbers)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [Cloud TPU v4 docs](../sources/2026-08-cloud-tpu-v4-docs.md),
  [v5p docs](../sources/2026-08-cloud-tpu-v5p-docs.md)
- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md),
  [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md),
  [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
