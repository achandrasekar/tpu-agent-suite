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
exploits it (source (`index.py get scaling-book-tpus`)).

## How it shows up

- **Within a matmul**: HBM→VMEM (`index.py get concept-vmem`) copies of the next chunks
  overlap with MXU (`index.py get concept-mxu`) work on the current ones, keeping matmuls
  compute-bound; partial results stream back without waiting for full arrays
  (source (`index.py get scaling-book-tpus`)).
- **Across operations**: VMEM prefetching (`index.py get concept-vmem`) — load the next
  layer's weights during the current layer's compute (e.g. FFN weights during
  attention) to hide weight-load cost when memory-bound
  (source (`index.py get scaling-book-tpus`)).
- **Across the system**: multi-stage transfers (PCIe load, ICI gather, HBM→MXU,
  FLOPs) pipeline into "bottlenecked by the slowest piece"
  (source (`index.py get scaling-book-tpus`)).

## Optimization implications

- Diagnose by comparing per-stage times: if the slowest stage ≉ total, overlap is
  broken and there's free performance in fixing the schedule.
- The systolic array's own pipelining (bubble only at start) is this principle at the
  hardware level; see Systolic array (`index.py get concept-systolic-array`).

## Critical-path vs asynchronous collectives

- Data-parallel gradient AllReduces and FSDP's ReduceScatters are **not on the critical
  path** — they can run whenever convenient; FSDP's weight AllGathers are prefetched a
  layer ahead. Tensor-parallel AllGathers/ReduceScatters **are** on the critical path,
  before and after each block (scaling-book, Part 5 (`index.py get scaling-book-training`)).
- Hiding critical-path collectives needs a collective matmul (`index.py get concept-collective-matmul`):
  multiply one chunk while the ring moves the next
  (scaling-book, Part 3 (`index.py get scaling-book-sharding`)).
- XLA's overlap is imperfect: HBM reads are not perfectly hidden and sharded matmuls
  take a latency hit above BS≈32 during generation, so rooflines are not sharp in
  practice (scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## See also

- Roofline analysis (`index.py get concept-roofline-analysis`)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- How To Scale Your Model, Part 3: Sharded Matrices (scaling-book) (`index.py get scaling-book-sharding`),
  How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`),
  How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
