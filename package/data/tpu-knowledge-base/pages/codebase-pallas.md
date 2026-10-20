---
title: "Pallas"
type: codebase
tags: [compiler, kernel]
created: 2026-08-22
updated: 2026-10-01
sources: 3
human_reviewed: false
---

Pallas is JAX's kernel-authoring extension — the primary way to write custom TPU
kernels. Official docs: Pallas overview (`index.py get pallas`). Beyond them: Pallas
kernel block sizes are constrained
by available VMEM (`index.py get concept-vmem`), whose buffer size is a tunable parameter —
stated in the TPU7x docs (`index.py get cloud-tpu-7x`); a property of the
Pallas/XLA stack rather than 7x silicon, so it should apply on other generations too
*(unverified — agent inference)*.

The scaling book positions Pallas as the escape hatch below JAX/HLO when the compiler's
schedule is not good enough, and defers all kernel content to the *Writing TPU Kernels
with Pallas* series; its only kernel exercise is
implementing Flash Attention in Pallas (scaling-book, Part 9 (`index.py get scaling-book-profiling`),
book hub (`index.py get scaling-book`)).

## Where the kernel content is

The official Pallas docs each have an entry:

- Quickstart (`index.py get pallas-quickstart`) — a first kernel, Refs, `pallas_call`,
  grids and BlockSpecs.
- TPU details (`index.py get pallas-tpu`) — HBM/VMEM/SMEM memory spaces, register
  tiling, multicore dimension semantics.
- Pipelining (`index.py get pallas-pipelining`) — overlapping HBM-to-VMEM copies with
  compute across a grid.
- Matmul (`index.py get pallas-matmul`) — block tiling and picking block sizes.
- Sparse kernels (`index.py get pallas-sparse`), distributed kernels
  (`index.py get pallas-distributed`), per-core programming
  (`index.py get pallas-core-map`), SparseCore (`index.py get pallas-sparsecore`).

Authored pages: the Pallas TPU reference (`index.py get pallas-tpu-reference`), a
mental model for TPU kernels (`index.py get pallas-core-mental-model`), validation and
profiling (`index.py get pallas-validation-and-xprof`), and rules learned from failed
optimization runs (`index.py get briefs`). Kernel libraries: Tokamax
(`index.py get tokamax`) and vLLM tpu-inference, whose ragged paged attention v3
kernel is `index.py get tpu-inference-kernels`.

## See also

- VMEM (`index.py get concept-vmem`), MXU (`index.py get concept-mxu`)
- Pallas Kernel (`index.py get pallas-kernel`)

## Sources

- Cloud TPU7x docs (`index.py get cloud-tpu-7x`)
- How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book) (`index.py get scaling-book-profiling`), book hub (`index.py get scaling-book`)
