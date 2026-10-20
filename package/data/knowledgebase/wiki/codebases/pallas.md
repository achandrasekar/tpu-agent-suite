---
title: "Pallas"
type: codebase
tags: [compiler, kernel, stub]
created: 2026-08-22
updated: 2026-09-04
sources: 3
human_reviewed: false
---

Pallas is JAX's kernel-authoring extension — the primary way to write custom TPU
kernels. Official docs: <https://docs.jax.dev/en/latest/pallas/index.html> (not yet
ingested). What ingested sources say so far: Pallas kernel block sizes are constrained
by available [VMEM](../concepts/vmem.md), whose buffer size is a tunable parameter —
stated in the [TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md); a property of the
Pallas/XLA stack rather than 7x silicon, so it should apply on other generations too
*(unverified — agent inference)*.

The scaling book positions Pallas as the escape hatch below JAX/HLO when the compiler's
schedule is not good enough, and defers all kernel content to the *Writing TPU Kernels
with Pallas* series (fetch target, not yet ingested); its only kernel exercise is
implementing Flash Attention in Pallas ([scaling-book, Part 9](../sources/2025-scaling-book-profiling.md),
[book hub](../sources/2025-scaling-book.md)).

*Stub — central to the kernel-optimization use case; expand on the first
Pallas-focused ingest (programming model, grid/BlockSpec, Mosaic backend, tuning
workflow).*

## See also

- [VMEM](../concepts/vmem.md), [MXU](../concepts/mxu.md)

## Sources

- [Cloud TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md)
- [How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book)](../sources/2025-scaling-book-profiling.md), [book hub](../sources/2025-scaling-book.md)
