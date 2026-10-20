---
title: "Collective matmul (overlapping sharded-matmul collectives)"
type: concept
tags: [parallelism, performance, kernel]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["AllGather matmul", "ReduceScatter matmul", "ppermute matmul"]
---

A collective matmul overlaps the AllGather (or ReduceScatter/AllReduce) that a sharded
matmul needs with the matmul itself: multiply the local chunk while the next chunk arrives
via a ring permute, accumulating as you go. It removes the blocking collective that
[tensor parallelism](tensor-parallelism.md) puts on the critical path
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md),
[scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).

## Mechanism

For `A[B_X, D_Y] · W[D, F_Y] → Out[B_X, F_Y]` the naive plan is `AllGather_Y(A)` then a local
matmul. The overlapped plan, per `Y` shard: multiply the local `A` chunk by the matching
`D`-chunk of `W`, and simultaneously `ppermute` `A` one step around the `Y` ring; after
`|Y|` steps every chunk has been multiplied and summed. In JAX this is a `shard_map`
containing a `fori_loop` over `jax.lax.dynamic_slice_in_dim` + `jax.lax.ppermute`
([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).

Measured on a v5e-8 (2×4 mesh, `B,D,F = 1024, 2048, 8192`): jit + blocking AllGather
311 µs; collective matmul 244 µs; unsharded-contracting baseline 224 µs — near-perfect
overlap ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).

## Variants

- **ReduceScatter matmul** for the down-projection `Tmp[B_X, F_Y] · W_out[F_Y, D] →
  Out[B_X, D_Y]`: permute the partial result as you accumulate
  ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- **AllReduce matmul** (tile over the output dim, `psum`); the book warns XLA may not make
  it faster than the baseline ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- **Bidirectional** rings halve the permute chain; the book leaves this as an exercise
  ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- Tiling over the batch or input dimension is also possible; the Mosaic GPU docs give a
  GPU example ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

## When to reach for it

- TP collectives on the critical path visibly bracketing matmuls in a profile
  ([profile-tpu-program](../playbooks/profile-tpu-program.md)) — the book's example
  profile showed a big blocking AllGather before the matmul.
- Prefill with Megatron sharding, where the book pairs collective matmuls (TP) with ring
  attention (sequence parallelism) to overlap comms
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- Not needed for FSDP's weight AllGathers, which are already off the critical path
  ([fsdp](fsdp.md)).

## See also

- [Compute–communication overlap](compute-communication-overlap.md),
  [Collective costs](collective-costs.md), [jax](../codebases/jax.md)

## Sources

- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md)
- [How To Scale Your Model, Part 10: Programming TPUs in JAX (scaling-book)](../sources/2025-scaling-book-jax-stuff.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
