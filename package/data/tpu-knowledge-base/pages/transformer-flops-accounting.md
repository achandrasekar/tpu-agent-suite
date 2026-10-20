---
title: "Transformer FLOPs and parameter accounting"
type: concept
tags: [workload, performance, memory]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["6ND rule", "transformer math", "parameter count", "FLOPs per token"]
---

Counting a Transformer's parameters and FLOPs from its config is the input to every sizing
decision in this wiki: FLOPs give training time and MFU, parameters give memory, and the
ratio between attention and MLP FLOPs decides when context length starts to matter
(scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## Counting rules

- A contraction costs `2 × ∏(all dims, batch and contracting dims counted once)` FLOPs;
  matmul compute is `O(N³)` on `O(N²)` bytes, which is why it scales
  (scaling-book, Part 4 (`index.py get scaling-book-transformers`)).
- Training = 3× inference per matmul: forward `2NPM`, backward `4NPM` (`dL/dB = Aᵀ·dL/dC`
  and `dL/dA = dL/dC·Bᵀ`), so **training FLOPs ≈ 6 · params · tokens**, inference ≈
  `2 · params · tokens` (scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## Per-layer table (gated MLP, GQA; `B` batch, `T` tokens, `D` model, `F` ff, `N` heads, `K` KV heads, `H` head dim, `V` vocab)

| Component | Params per layer | Training FLOPs per layer |
|---|---|---|
| MLP (`W_in1`, `W_in2`, `W_out`) | `3DF` | `18BTDF` |
| Attention projections `Q,K,V,O` | `2D(N+K)H` | `12BTD(N+K)H` |
| Dot-product attention | — | `12BT²NH` (halved by a causal mask, only with a kernel) |
| Layernorms (×2) | `2D` | `O(BTD)` |
| Vocab (total, not per layer) | `DV` (×2 if untied) | `6BTDV` |

(scaling-book, Part 4 (`index.py get scaling-book-transformers`).) Ignoring dot-product attention,
`(18BTDF + 12BTD(N+K)H)·L = 6 · tokens · params` — the 6ND rule.

## When attention dominates

With `F = 4D`, `D = NH`, `N = K`: attention FLOPs / matmul FLOPs = `T / 8D`, so
dot-product attention dominates only when **`T > 8D`** (~64k tokens for `D = 8k`; ~37k for
Gemma-27B's `D = 4608`). Attention FLOPs equal the QKVO projection FLOPs at `T = 2D`
(scaling-book, Part 4 (`index.py get scaling-book-transformers`)). Local attention and other
variants change this (scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## Worked: LLaMA 3-70B

`L=80, D=8192, F=28672, N=64, K=8, H=128, V=128256`: FFW `3·D·F·L = 56.3e9`, attention
`L·(2DNH + 2DKH) = 12e9`, vocab `2DV = 2.1e9` → 70.4B. The three MLP matrices are 56B of
70B, so "you can almost ignore everything else" for memory and FLOPs. `6 × 70e9 = 4.2e11`
FLOPs per token per training step (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).

## Attention intensity (why prefill and decode differ)

Per head, attention reads `Q[B,T,·]` and the `[B,S,·]` KV pair and does `4BSTD` FLOPs:
intensity `ST/(S+T)` → `T/2` in prefill (compute-bound past a few hundred tokens) and → 1
in generation (`T = 1`), always bandwidth-bound; with GQA group size `G` the decode
intensity rises toward `G` (scaling-book, Part 4 (`index.py get scaling-book-transformers`),
scaling-book, Part 7 (`index.py get scaling-book-inference`)). Flash Attention computes attention
in KV chunks with a running max, sum and output kept in VMEM (`index.py get concept-vmem`), so the `[B,N,T,S]`
matrix is never materialized; a softmax identity (`S·dS = dO·O`) makes the backward pass
chunk-local as well, which is what ring attention builds on
(scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## See also

- MFU (`index.py get concept-mfu`), KV cache (`index.py get concept-kv-cache`), Gradient checkpointing (`index.py get concept-gradient-checkpointing`),
  Expert parallelism (`index.py get concept-expert-parallelism`), Prefill vs decode (`index.py get concept-prefill-vs-decode`)

## Sources

- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
