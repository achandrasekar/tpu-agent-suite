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
through intervening chips (source (`index.py get scaling-book-tpus`)).

## Topology

- **2D torus** (4 neighbors + edge wraparound): v2, v3, v5e, v6e. v5e/v6e pods max out
  at 16×16 and scale further only over DCN (`index.py get concept-dcn`)
  (source (`index.py get scaling-book-tpus`)).
- **3D torus** (6 neighbors): v4, v5p. Superpods (16×16×16 v4; 16×20×28 v5p) are built
  from reconfigurable 4×4×4 cubes joined by optical wraparound links
  (source (`index.py get scaling-book-tpus`)).
- A torus halves the max hop distance (N → N/2); a "twisted torus" reduces average
  distance further (source (`index.py get scaling-book-tpus`)).
- **Twisted torus** (v4/v5p, Cloud TPU API only; shapes where each dim = or 2× the
  smallest, e.g. 4×4×8, 4×8×8): rewires wraparound edges to make an asymmetric torus
  symmetric — up to ~70% (4×4×8) / ~40% (4×8×8) more bisection bandwidth, better load
  balancing, shorter routes; 1–2 pp MFU gain on FSDP MaxText
  (v4 docs (`index.py get cloud-tpu-v4`),
  v5p docs (`index.py get cloud-tpu-v5p`)).
- **ICI resiliency** (v4/v5p, default-on for ≥1-cube slices): inter-cube optical/OCS
  faults are routed around (in-cube copper unaffected) — better availability at the
  cost of temporarily degraded ICI performance
  (v5p docs (`index.py get cloud-tpu-v5p`)).

## Wraparound rules (bite in practice)

- 3D-torus slices get wraparounds only as multiples of a full 4×4×4 cube (4×4×8 yes,
  2×2×4 no); v5e/v6e get a wraparound only on an axis of size 16 (8×16 → long axis
  only). **Missing wraparounds ≈ 2× communication time**
  (source (`index.py get scaling-book-tpus`)).
- Bidirectional ("bidi") bandwidth = 2× one-way, but only with a functioning ring —
  i.e. with wraparound. Roofline math usually uses bidi
  (source (`index.py get scaling-book-tpus`)).

## Numbers

~9e10 B/s one-way per link on v5p (3 axes/chip), ~4.5e10 on v5e/v4p; per-hop latency
~1 µs (v5e). Slower than HBM (`index.py get concept-hbm`) (~30×), far faster than DCN (`index.py get concept-dcn`)
(source (`index.py get scaling-book-tpus`)). Transfer time = hop latency +
bytes / (ports × link BW); source and destination each have 2 usable ports in a
wrap-less 2D slice (source (`index.py get scaling-book-tpus`)).

## Collective costs on ICI

- A bandwidth-bound AllGather or ReduceScatter over a full ring costs `bytes / W_ici`
  regardless of axis size; AllReduce 2×; AllToAll ¼ — formulas on
  collective-costs (`index.py get concept-collective-costs`) (scaling-book, Part 3 (`index.py get scaling-book-sharding`)).
- **ICI arithmetic intensity** `C / W_ici` (2,550 on v5p per axis) is the per-chip token
  batch needed for data-parallel/FSDP to be compute-bound; per-generation values on
  critical-batch-size (`index.py get concept-critical-batch-size`) (scaling-book, Part 5 (`index.py get scaling-book-training`)).
- Latency floor ~1 µs per hop: buffers under ~45 kB (v5e) are latency-bound, which
  small-batch decode collectives routinely are (scaling-book, Part 3 (`index.py get scaling-book-sharding`)).
- Measured: a v5e 8×16 AllGather reaches ~95% of the claimed link bandwidth at ~10 MB
  total, far smaller messages than GPUs need (scaling-book, Part 3 (`index.py get scaling-book-sharding`)).

## vs GPU fabrics

GPUs use switch hierarchies approximating all-to-all (8–72 GPUs directly linked, then
O(log N) hops); TPUs trade arbitrary point-to-point reach for constant per-device link
count/bandwidth — cheaper, simpler, larger max scale, but sharding layouts must respect
neighbor locality (source (`index.py get scaling-book-tpus`)).

## See also

- DCN (`index.py get concept-dcn`) — between slices; Roofline analysis (`index.py get concept-roofline-analysis`)
- Collective costs (`index.py get concept-collective-costs`), GPU vs TPU (`index.py get concept-gpu-vs-tpu`) (NVLink/InfiniBand numbers)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- Cloud TPU v4 docs (`index.py get cloud-tpu-v4`),
  v5p docs (`index.py get cloud-tpu-v5p`)
- How To Scale Your Model, Part 3: Sharded Matrices (scaling-book) (`index.py get scaling-book-sharding`),
  How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`),
  How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
