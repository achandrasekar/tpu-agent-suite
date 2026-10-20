---
title: "SparseCore"
type: concept
tags: [hardware]
created: 2026-08-22
updated: 2026-10-01
sources: 3
human_reviewed: true
---

SparseCore is a per-chip accelerator block for sparse/embedding workloads. Counts per
chip: 4 on v5p (`index.py get hw-tpu-v5p`)
(docs (`index.py get cloud-tpu-v5p`)), 2 on
v6e (`index.py get hw-tpu-v6e`) (docs (`index.py get cloud-tpu-v6e`)),
4 on TPU7x (`index.py get hw-tpu-7x`) — 2 per chiplet
(docs (`index.py get cloud-tpu-7x`)).

How it works, which workloads suit it and how to program it are in two dedicated
entries rather than here:

- Pallas SparseCore guide (`index.py get pallas-sparsecore`) — its vector and scalar
  subcores and their memory spaces, which workloads suit it, and writing kernels for it.
- OpenXLA SparseCore guide (`index.py get openxla-sparsecore`) — the architecture,
  embedding-lookup preprocessing, configuration flags, and performance tuning.

## See also

- TensorCore (`index.py get concept-tensorcore`)

## Sources

- Cloud TPU docs: v5p (`index.py get cloud-tpu-v5p`),
  v6e (`index.py get cloud-tpu-v6e`),
  TPU7x (`index.py get cloud-tpu-7x`)
