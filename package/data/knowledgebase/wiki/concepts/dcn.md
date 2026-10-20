---
title: "DCN (Data-Center Network)"
type: concept
tags: [hardware, interconnect]
created: 2026-08-22
updated: 2026-09-04
sources: 2
human_reviewed: true
---

DCN is the standard datacenter network connecting TPU **hosts** — the path between
slices and pods that [ICI](ici.md) can't reach. It is the slowest tier of the TPU
communication hierarchy ([source](../sources/2025-scaling-book-tpus.md)).

## Behavior

- ~6.25e9 B/s egress per TPU (12.5e9 on v6e/TPU7x; 3.125e9 on v5e) — ~15× below one
  ICI link ([source](../sources/2025-scaling-book-tpus.md)).
- DCN is **host-to-host**: a TPU-to-TPU transfer over DCN goes
  TPU → [PCIe](pcie.md) → host → DCN → host → PCIe → TPU
  ([source](../sources/2025-scaling-book-tpus.md)).

## Multi-slice training

A *slice* is an ICI-connected set of TPUs; slices (e.g. on different pods, or beyond a
v5e/v6e 16×16 torus) train together over DCN. Because DCN is far slower than ICI,
computation must be arranged so it rarely waits on DCN traffic — put the
communication-heavy parallelism axes inside slices
([source](../sources/2025-scaling-book-tpus.md)).

Roofline for the cross-slice axis: with FSDP/TP inside each ICI slice and data
parallelism across slices, the reduction is compute-bound when the batch **per slice**
exceeds `C / W_dcn` (`4.59e14 / 6.25e9 ≈ 73,440` tokens on v5p); DCN bandwidth scales with
slice size because every host adds NICs ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
Per-generation values on [critical-batch-size](critical-batch-size.md).

## See also

- [ICI](ici.md) — within a slice; [PCIe](pcie.md) — the host leg of every DCN transfer

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md)
