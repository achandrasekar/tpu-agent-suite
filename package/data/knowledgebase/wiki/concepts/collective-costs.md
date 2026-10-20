---
title: "Collective communication costs"
type: concept
tags: [parallelism, interconnect, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 4
human_reviewed: true
aliases: ["AllGather", "ReduceScatter", "AllReduce", "AllToAll", "collectives"]
---

TPUs use four collective primitives, and in the bandwidth-bound regime each costs
`bytes / bandwidth` **independent of how many devices the axis spans**, because a ring
moves `V/X` bytes per hop over `X/2` hops in each direction
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)). Knowing these four formulas is what
turns a sharding into a time estimate.

## Semantics and cost (TPU ring, bandwidth-bound)

`V` = total bytes of the array, `W_ici` = bidirectional ICI bandwidth per axis
(per-generation values in [ICI](ici.md)), `N_axes` = mesh axes the collective spans.

| Collective | Notation | What it does | Time |
|---|---|---|---|
| AllGather | `[A_X, B] → [A, B]` | removes a subscript; every device gets the full axis | `V / (W_ici · N_axes)` |
| ReduceScatter | `[A, B]{U_X} → [A_X, B]` | sums partial results, leaves them sharded on another axis | same as AllGather |
| AllReduce | `[A_X, B]{U_Y} → [A_X, B]` | sums partial results, replicated; = ReduceScatter + AllGather | `2 · V / (W_ici · N_axes)` |
| AllToAll | `[A, B_X] → [A_X, B]` | moves a subscript between axes (resharding, MoE routing) | `V · max(axis sizes) / (4 · N · W_ici)`; 1D: `V / (4 W_ici)` = ¼ AllGather |

All from [scaling-book, Part 3](../sources/2025-scaling-book-sharding.md). A unidirectional ring costs
`(X−1)/X · V/W` per direction; the bidi form above assumes a working ring, i.e. a wraparound
on that axis — without it (small v5e/v6e slices, non-cube v5p slices) expect ~2× ([ICI](ici.md)).
Spanning more axes multiplies bandwidth by `N_axes`; an array already sharded on another
axis costs `1/Y` as much because each axis moves `1/Y` of the bytes
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

## Latency-bound regime

Each ICI hop carries ~1 µs of fixed latency, so
`T = max(T_min · Σ|X_i| / 2, V / (W_ici · N_axes))` and buffers under
`W_uni × 1e-6` ≈ 45 kB on v5e are latency-bound
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)). This bites during decode: with
8-way model sharding the per-shard threshold is ~360 kB, and `BS=16 × D=8192` int8
activations are only 131 kB, so the activation collectives of small-batch generation are
latency-bound (`Y > BD/45,000` in int8)
([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)). Empirically a v5e 8×16 AllGather
reaches ~95% of the claimed link bandwidth at ~10 MB total (~625 kB per device)
([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).

## Duality and deferral

AllGather and ReduceScatter are transposes: the backward pass of an AllGather is a
ReduceScatter of the gradient and vice versa, so a sharding's backward comms are
determined by its forward comms. Decomposing AllReduce into ReduceScatter + AllGather
also lets the AllGather be deferred — keep the product sharded (`C[I, K_X]`) and gather
only if a consumer needs replication ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
This decomposition is exactly why FSDP costs the same as data parallelism
([fsdp](fsdp.md)).

## On GPUs (for comparison)

Within an NVLink node the same ring formulas apply with `W` = GPU egress bandwidth
(H100 450 GB/s, B200 900 GB/s); across nodes `W` = node egress over the InfiniBand fat
tree (400 GB/s on a DGX H100 SuperPod). Node-local AllToAll is 2× cheaper than the TPU ring
(`V/(8W)` vs `V/(4W)`), but cross-node AllToAll degrades >4×; in-network reductions
(SHARP) yield ~30% in practice, not 2×; measured AllReduce tops out near 370 GB/s of the
claimed 450 ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)). See [gpu-vs-tpu](gpu-vs-tpu.md).

## See also

- [ICI](ici.md), [DCN](dcn.md) — the bandwidths in the denominators
- [Sharding notation](sharding-notation.md) — which collective each matmul case needs
- [Collective matmul](collective-matmul.md) — hiding these behind compute

## Sources

- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
- [How To Scale Your Model, Part 2: How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md) (link bandwidths)
