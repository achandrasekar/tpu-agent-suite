---
title: "Compute–communication overlap"
type: concept
tags: [performance]
created: 2026-08-22
updated: 2026-09-04
sources: 4
human_reviewed: true
---

The TPU's default operating principle: memory transfers and computation run
concurrently, so a well-pipelined workload's time is the *max* of its stages, not the
sum. Every roofline estimate assumes it; every optimization that "hides" a cost
exploits it ([source](../sources/2025-scaling-book-tpus.md)).

## How it shows up

- **Within a matmul**: HBM→[VMEM](vmem.md) copies of the next chunks
  overlap with [MXU](mxu.md) work on the current ones, keeping matmuls
  compute-bound; partial results stream back without waiting for full arrays
  ([source](../sources/2025-scaling-book-tpus.md)).
- **Across operations**: [VMEM prefetching](vmem.md) — load the next
  layer's weights during the current layer's compute (e.g. FFN weights during
  attention) to hide weight-load cost when memory-bound
  ([source](../sources/2025-scaling-book-tpus.md)).
- **Across the system**: multi-stage transfers (PCIe load, ICI gather, HBM→MXU,
  FLOPs) pipeline into "bottlenecked by the slowest piece"
  ([source](../sources/2025-scaling-book-tpus.md)).

## Optimization implications

- Diagnose by comparing per-stage times: if the slowest stage ≉ total, overlap is
  broken and there's free performance in fixing the schedule.
- The systolic array's own pipelining (bubble only at start) is this principle at the
  hardware level; see [Systolic array](systolic-array.md).

## Critical-path vs asynchronous collectives

- Data-parallel gradient AllReduces and FSDP's ReduceScatters are **not on the critical
  path** — they can run whenever convenient; FSDP's weight AllGathers are prefetched a
  layer ahead. Tensor-parallel AllGathers/ReduceScatters **are** on the critical path,
  before and after each block ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- Hiding critical-path collectives needs a [collective matmul](collective-matmul.md):
  multiply one chunk while the ring moves the next
  ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- XLA's overlap is imperfect: HBM reads are not perfectly hidden and sharded matmuls
  take a latency hit above BS≈32 during generation, so rooflines are not sharp in
  practice ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## See also

- [Roofline analysis](roofline-analysis.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md),
  [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md),
  [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
