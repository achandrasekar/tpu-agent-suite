---
title: "HBM (High Bandwidth Memory)"
type: concept
tags: [hardware, memory]
created: 2026-08-22
updated: 2026-10-01
sources: 2
human_reviewed: true
---

HBM is the TPU chip's main memory: tens of GB holding the tensors the
TensorCore (`index.py get concept-tensorcore`) works on. Tensors stream HBM → VMEM (`index.py get concept-vmem`) → compute
units and back (source (`index.py get scaling-book-tpus`)).

## Role in performance

- HBM bandwidth (~1–2 TB/s; e.g. 2.8 TB/s on v5p) is the top of the communication
  hierarchy — faster than ICI (`index.py get concept-ici`), PCIe (`index.py get concept-pcie`), and DCN (`index.py get concept-dcn`) — and is
  the limiting rate for memory-bound workloads
  (source (`index.py get scaling-book-tpus`)).
- Sampling-latency floor: each decode step must load all resident parameters from HBM,
  so params ÷ aggregate HBM bandwidth lower-bounds per-step latency (~10 ms for 200B
  bf16 params on 32 v4p chips) (source (`index.py get scaling-book-tpus`)). See
  Roofline analysis (`index.py get concept-roofline-analysis`).
- A matmul fed from HBM needs arithmetic intensity ≈ 240 to be compute-bound, vs
  ~10–20 from VMEM (source (`index.py get scaling-book-tpus`)).

- Decode is HBM-bound on both terms: `T_step ≈ B·KV / W_hbm + max(2·B·params / C, params / W_hbm)`,
  with attention (the KV term) never compute-bound; throughput per chip saturates near
  the critical batch size (`index.py get concept-critical-batch-size`) and, at long context, KV loading
  rather than parameters dominates (scaling-book, Part 7 (`index.py get scaling-book-inference`)).
  See prefill vs decode (`index.py get concept-prefill-vs-decode`).

Per-generation capacities and bandwidths: see the hardware generation pages
(v4 (`index.py get hw-tpu-v4`), v5e (`index.py get hw-tpu-v5e`),
v5p (`index.py get hw-tpu-v5p`), v6e (`index.py get hw-tpu-v6e`),
7x (`index.py get hw-tpu-7x`)).

## See also

- VMEM (`index.py get concept-vmem`) — the faster level below

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
