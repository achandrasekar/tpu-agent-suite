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
slices and pods that ICI (`index.py get concept-ici`) can't reach. It is the slowest tier of the TPU
communication hierarchy (source (`index.py get scaling-book-tpus`)).

## Behavior

- ~6.25e9 B/s egress per TPU (12.5e9 on v6e/TPU7x; 3.125e9 on v5e) — ~15× below one
  ICI link (source (`index.py get scaling-book-tpus`)).
- DCN is **host-to-host**: a TPU-to-TPU transfer over DCN goes
  TPU → PCIe (`index.py get concept-pcie`) → host → DCN → host → PCIe → TPU
  (source (`index.py get scaling-book-tpus`)).

## Multi-slice training

A *slice* is an ICI-connected set of TPUs; slices (e.g. on different pods, or beyond a
v5e/v6e 16×16 torus) train together over DCN. Because DCN is far slower than ICI,
computation must be arranged so it rarely waits on DCN traffic — put the
communication-heavy parallelism axes inside slices
(source (`index.py get scaling-book-tpus`)).

Roofline for the cross-slice axis: with FSDP/TP inside each ICI slice and data
parallelism across slices, the reduction is compute-bound when the batch **per slice**
exceeds `C / W_dcn` (`4.59e14 / 6.25e9 ≈ 73,440` tokens on v5p); DCN bandwidth scales with
slice size because every host adds NICs (scaling-book, Part 5 (`index.py get scaling-book-training`)).
Per-generation values on critical-batch-size (`index.py get concept-critical-batch-size`).

## See also

- ICI (`index.py get concept-ici`) — within a slice; PCIe (`index.py get concept-pcie`) — the host leg of every DCN transfer

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
