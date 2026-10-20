---
title: "Expert parallelism and MoE routing"
type: concept
tags: [parallelism, workload, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 5
human_reviewed: true
aliases: ["EP", "mixture of experts", "MoE", "expert sharding"]
---

A mixture-of-experts layer replaces one dense MLP with `E` experts of which each token
activates `k` (sparsity `E/k`, typically 8–64; DeepSeek-V3 has `E = 256, k = 8`).
Parameters grow `O(E)` while FLOPs per token grow only `k×`, which changes every roofline
in this wiki by the factor `E/k`, and routing adds two AllToAlls per layer
(scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## Rooflines shift by E/k

- **HBM critical batch**: each weight matrix loads `E·D·F` bytes for `2·k·B·D·F` FLOPs, so
  compute-bound needs `k·B/E > α_hbm/β`, i.e. `B > 120·E/k` tokens with int8 weights on
  v5e — 3,840 for DeepSeek-V3, "a remarkably large batch at generation time"
  (scaling-book, Part 4 (`index.py get scaling-book-transformers`),
  scaling-book, Part 7 (`index.py get scaling-book-inference`)).
- **Data-parallel / FSDP threshold** inflates the same way: `B/X > (E/k) · C/W`
  (e.g. 32 × 2,475 ≈ 79k tokens per GPU for a `k=4, E=128` model)
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).
- KV cache size is unchanged — MoE touches only the MLP
  (scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## Expert sharding and its communication

Shard the expert axis: `W[E_Z, D_X, F_Y]` (`X` only during training as the FSDP axis).
Tokens must be sent to their experts' devices and back: one AllToAll before the block and
one after. On a bidirectional TPU ring an AllToAll costs ¼ of a comparable single-axis AllGather
(collective-costs (`index.py get concept-collective-costs`)), which is why the book treats routing as cheap (scaling-book, Part 4 (`index.py get scaling-book-transformers`),
scaling-book, Part 7 (`index.py get scaling-book-inference`)). A **ragged** AllToAll moves only the
`k` of `N` non-empty shards, cutting cost by ~`min(k/N, 1)`
(scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

On GPUs, `T_math = 4·B·k·D·F / (Z·C)` against `T_comms ≈ 4·B·D·(Z−8)/(W·Z) · min(8k/Z, 1)`
gives two viable regimes: small EP (≈2 nodes) with small `F`, or `F > 8·C/W_node` with EP up
to `E` nodes; DeepSeek-V3 (small `F`) chose 64-way EP over 8 nodes
(scaling-book, Part 12 (`index.py get scaling-book-gpus`)). The TPU-pod analogue — how far EP can
span over ICI on v6e/TPU7x — is not stated by the book *(gap; needs a source)*.

## Implementation sketch (JAX)

Local: loop over experts with masking (`scan` over `E`, wasteful) or sort tokens by expert
and use `jax.lax.ragged_dot`, never materializing `[S, D, F]`. Sharded: chunk the sorted
tokens, `all_to_all` each chunk to its expert device, local matmul, concatenate — inside a
`while_loop` to avoid padding (scaling-book, Part 10 (`index.py get scaling-book-jax`)). No
torch_tpu equivalent is ingested.

## See also

- Parallelism strategies (`index.py get concept-parallelism-strategies`) (hub),
  Critical batch size (`index.py get concept-critical-batch-size`), Collective costs (`index.py get concept-collective-costs`),
  Transformer FLOPs accounting (`index.py get concept-transformer-flops-accounting`)

## Sources

- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
- How To Scale Your Model, Part 10: Programming TPUs in JAX (scaling-book) (`index.py get scaling-book-jax`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
- How To Scale Your Model, Part 3: Sharded Matrices (scaling-book) (`index.py get scaling-book-sharding`) (AllToAll cost)
