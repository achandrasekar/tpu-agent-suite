---
title: "Collective communication costs"
type: concept
tags: [parallelism, interconnect, performance]
created: 2026-09-04
updated: 2026-10-14
sources: 4
human_reviewed: true
aliases: ["AllGather", "ReduceScatter", "AllReduce", "AllToAll", "collectives"]
---

TPUs use four collective primitives, and in the bandwidth-bound regime each costs
`bytes / bandwidth` **independent of how many devices the axis spans**, because a ring
moves `V/X` bytes per hop over `X/2` hops in each direction
(scaling-book, Part 3 (`index.py get scaling-book-sharding`)). Knowing these four formulas is what
turns a sharding into a time estimate.

## Semantics and cost (TPU ring, bandwidth-bound)

`V` = total bytes of the array, `W_ici` = bidirectional ICI bandwidth per axis
(per-generation values in ICI (`index.py get concept-ici`)), `N_axes` = mesh axes the collective spans.

| Collective | Notation | What it does | Time |
|---|---|---|---|
| AllGather | `[A_X, B] → [A, B]` | removes a subscript; every device gets the full axis | `V / (W_ici · N_axes)` |
| ReduceScatter | `[A, B]{U_X} → [A_X, B]` | sums partial results, leaves them sharded on another axis | same as AllGather |
| AllReduce | `[A_X, B]{U_Y} → [A_X, B]` | sums partial results, replicated; = ReduceScatter + AllGather | `2 · V / (W_ici · N_axes)` |
| AllToAll | `[A, B_X] → [A_X, B]` | moves a subscript between axes (resharding, MoE routing) | `V · max(axis sizes) / (4 · N · W_ici)`; 1D: `V / (4 W_ici)` = ¼ AllGather |

All from scaling-book, Part 3 (`index.py get scaling-book-sharding`). A unidirectional ring costs
`(X−1)/X · V/W` per direction; the bidi form above assumes a working ring, i.e. a wraparound
on that axis — without it (small v5e/v6e slices, non-cube v5p slices) expect ~2× (ICI (`index.py get concept-ici`)).
Spanning more axes multiplies bandwidth by `N_axes`; an array already sharded on another
axis costs `1/Y` as much because each axis moves `1/Y` of the bytes
(scaling-book, Part 3 (`index.py get scaling-book-sharding`)).

These are bandwidth models only, and the AllToAll row is the one measurement
has contradicted. On a 2x2 v6e slice (libtpu 0.0.46, torch_tpu
`all_to_all_single`, which lowers to one XLA all-to-all; bf16, 16 Mi elements
per rank), the all-to-all took ~2.2× as long as an all-gather and ~5.8× an
all-reduce of the same per-rank input, where the model predicts 0.25× and 0.5×.
Do not use the ¼ to budget an all-to-all on small v6e slices; measure it
(measured: `index.py get v6e-measured-collectives-dispatch`).

## Latency-bound regime

Each ICI hop carries ~1 µs of fixed latency, so
`T = max(T_min · Σ|X_i| / 2, V / (W_ici · N_axes))` and buffers under
`W_uni × 1e-6` ≈ 45 kB on v5e are latency-bound
(scaling-book, Part 3 (`index.py get scaling-book-sharding`)). This bites during decode: with
8-way model sharding the per-shard threshold is ~360 kB, and `BS=16 × D=8192` int8
activations are only 131 kB, so the activation collectives of small-batch generation are
latency-bound (`Y > BD/45,000` in int8)
(scaling-book, Part 7 (`index.py get scaling-book-inference`)). Empirically a v5e 8×16 AllGather
reaches ~95% of the claimed link bandwidth at ~10 MB total (~625 kB per device)
(scaling-book, Part 3 (`index.py get scaling-book-sharding`)).

## Duality and deferral

AllGather and ReduceScatter are transposes: the backward pass of an AllGather is a
ReduceScatter of the gradient and vice versa, so a sharding's backward comms are
determined by its forward comms. Decomposing AllReduce into ReduceScatter + AllGather
also lets the AllGather be deferred — keep the product sharded (`C[I, K_X]`) and gather
only if a consumer needs replication (scaling-book, Part 3 (`index.py get scaling-book-sharding`)).
This decomposition is exactly why FSDP costs the same as data parallelism
(fsdp (`index.py get concept-fsdp`)).

## On GPUs (for comparison)

Within an NVLink node the same ring formulas apply with `W` = GPU egress bandwidth
(H100 450 GB/s, B200 900 GB/s); across nodes `W` = node egress over the InfiniBand fat
tree (400 GB/s on a DGX H100 SuperPod). Node-local AllToAll is 2× cheaper than the TPU ring
(`V/(8W)` vs `V/(4W)`), but cross-node AllToAll degrades >4×; in-network reductions
(SHARP) yield ~30% in practice, not 2×; measured AllReduce tops out near 370 GB/s of the
claimed 450 (scaling-book, Part 12 (`index.py get scaling-book-gpus`)). See gpu-vs-tpu (`index.py get concept-gpu-vs-tpu`).

## See also

- ICI (`index.py get concept-ici`), DCN (`index.py get concept-dcn`) — the bandwidths in the denominators
- Sharding notation (`index.py get concept-sharding-notation`) — which collective each matmul case needs
- Collective matmul (`index.py get concept-collective-matmul`) — hiding these behind compute

## Sources

- How To Scale Your Model, Part 3: Sharded Matrices (scaling-book) (`index.py get scaling-book-sharding`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
- How To Scale Your Model, Part 2: How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`) (link bandwidths)
