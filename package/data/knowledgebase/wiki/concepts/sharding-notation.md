---
title: "Sharding notation and sharded matmul cases"
type: concept
tags: [parallelism]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["named-axis sharding", "PartitionSpec notation", "sharded matmul"]
---

The book's named-axis notation writes an array's sharding as subscripts on its logical
axes: `A[I_X, J_Y]` means axis `I` is partitioned across device-mesh axis `X` and `J` across
`Y`; an unsubscripted axis is replicated. A **mesh** names the physical axes of the slice
(`Mesh({'X': 4, 'Y': 2})`), and a **sharding** assigns mesh axes to array axes
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)). Every parallelism page in this wiki is
written in this notation.

## Rules of the notation

- **Global vs local shape**: `A[I_X, J_Y]` keeps its logical shape; each device holds
  `(|I|/|X|, |J|/|Y|)` ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- **Multiple mesh axes on one dimension**: `A[I_XY, J]` flattens `X` and `Y` into one
  larger axis; subscript order is the traversal order
  ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- **A mesh axis is spent once**: `A[I_X, J_X]` is forbidden — each mesh axis may shard at
  most one array axis ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- **Unreduced suffix**: `C[I, K]{U_X}` marks partial sums that still need summation over
  `X` ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- Mesh axes need not map 1:1 to hardware axes: a `{'X': 16, 'Y': 4}` mesh can describe a
  4×4×4 v5p cube with two hardware axes folded into `X`
  ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

In JAX this is literally `jax.make_mesh(axis_shapes, axis_names)` plus
`NamedSharding(mesh, PartitionSpec('X', 'Y'))`; `P(None, 'Y')` replicates the first
axis ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md); see [jax](../codebases/jax.md)).
The torch_tpu spelling is *not yet ingested*.

## The four sharded-matmul cases

Elementwise ops on sharded arrays cost nothing extra; only a contraction over a sharded
axis needs communication. Block-matmul algebra reduces every case to one of four
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)):

| Case | Shape pattern | What to do | Cost driver |
|---|---|---|---|
| 1 — neither contracting dim sharded | `A[I_X, J]·B[J, K_Y] → C[I_X, K_Y]` | multiply local blocks; output inherits input shardings | none |
| 2 — one contracting dim sharded | `A[I, J_X]·B[J, K] → C[I, K]` | **AllGather** the sharded operand along `X`, then local matmul | one AllGather of the operand |
| 3 — both contracting dims sharded on the same axis | `A[I, J_X]·B[J_X, K] → C[I, K]{U_X}` | local matmul gives partial sums; **AllReduce** (or **ReduceScatter** if a sharded output is wanted) | AllReduce of the output (2× an AllGather) |
| 4 — both operands share a non-contracting mesh axis | `A[I_X, J]·B[J, K_X]` | invalid as written (only diagonal blocks exist); AllGather one operand first | one AllGather |

Case 2 has an alternative: multiply local shards and AllReduce the partial sums (treat it
as case 3). The comms ratio of gather-then-matmul to matmul-then-reduce-scatter is `M/K`
(contracting size over output size), so the reduce path wins when the output is small
relative to the contracting dim, which is rare under FSDP where activations are
batch-sharded too ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

## Backward pass by the same rules

The gradient formulas are sharding-free identities (`dW_out = Tmp^T·dOut`,
`dTmp = dOut·W_out^T`, `dW_in = In^T·dTmp`, `dIn = dTmp·W_in^T`); apply the four cases to the
shardings each operand already has and the required collectives fall out. Because
AllGather and ReduceScatter are transposes of each other, an AllGather in the forward pass
becomes a ReduceScatter of the corresponding gradient in the backward pass
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md),
[scaling-book, Part 5](../sources/2025-scaling-book-training.md)).

## See also

- [Collective costs](collective-costs.md) — what each of those collectives costs
- [Parallelism strategies](parallelism-strategies.md) — the strategies as shardings
- [Collective matmul](collective-matmul.md) — overlapping the case-2/3 collectives

## Sources

- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md)
- [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md)
- [How To Scale Your Model, Part 10: Programming TPUs in JAX (scaling-book)](../sources/2025-scaling-book-jax-stuff.md)
