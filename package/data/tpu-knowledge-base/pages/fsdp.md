---
title: "FSDP (fully-sharded data parallelism, ZeRO)"
type: concept
tags: [parallelism, memory, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 4
human_reviewed: true
aliases: ["ZeRO", "ZeRO-3", "fully sharded data parallel", "FSDP"]
---

FSDP shards activations along the batch **and** shards weights and optimizer state along
the same mesh axis (`In[B_X, D] · W_in[D_X, F] · W_out[F, D_X]`), AllGathering each weight
just before use in the forward pass and ReduceScattering gradients in the backward pass.
It costs exactly what data parallelism (`index.py get concept-data-parallelism`) costs in communication and
removes almost all of its replicated memory (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Why the roofline is unchanged

An AllReduce is a ReduceScatter followed by an AllGather, each half the cost. FSDP
performs that AllGather early (forward, weights) and that ReduceScatter late (backward,
gradients), moving the same bytes as DP's gradient AllReduce. Hence the compute-bound
condition is the same, `B / X > α / M_X` with `α = C / W_ici` (2,550 on v5p, per-generation
values on critical-batch-size (`index.py get concept-critical-batch-size`)), while per-device memory drops
to `1/X` of params, grads and optimizer state
(scaling-book, Part 5 (`index.py get scaling-book-training`)). ZeRO-1/2/3 shard optimizer state,
gradients, and weights respectively; since all three have the same comms volume the book's
advice is to always take ZeRO-3 (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Scheduling properties

- Weight AllGathers are not on the critical path: prefetch the next layer's weights during
  the current layer, then discard the gathered copy after the matmul
  (scaling-book, Part 5 (`index.py get scaling-book-training`); see
  compute–communication overlap (`index.py get concept-compute-communication-overlap`)).
- Gradient ReduceScatters are asynchronous, like DP's AllReduces
  (scaling-book, Part 5 (`index.py get scaling-book-training`)).
- The book's LLaMA-3 70B example: a 16M-token batch could spread over ~18.8k v5p chips
  (`16e6 / (2550/3)`) and train in ~17 days at 50% MFU
  (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## When FSDP is the wrong tool

- **Per-chip batch below `α/M_X`**: FSDP alone is comms-bound; add
  tensor parallelism (`index.py get concept-tensor-parallelism`), which shrinks the weight
  AllGathers by `1/Y` (scaling-book, Part 5 (`index.py get scaling-book-training`)). LLaMA-3 70B at a
  4M batch on 8,960 v5p chips gives 468 tokens/chip, below the 850 floor — FSDP alone fails
  even with sequence sharding (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).
- **Generation**: never. Decode is HBM-bound on weight loads, and FSDP moves weights over
  ICI, which is orders of magnitude slower — "accidentally leaving it on after training" is
  a common order-of-magnitude regression
  (scaling-book, Part 7 (`index.py get scaling-book-inference`)).
- **With pipelining**: ZeRO-3 needs a weight AllGather per microbatch, which pipelining
  cannot amortize; use ZeRO-1 with pipeline parallelism
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## See also

- Parallelism strategies (`index.py get concept-parallelism-strategies`) (hub),
  Collective costs (`index.py get concept-collective-costs`), Gradient checkpointing (`index.py get concept-gradient-checkpointing`)

## Sources

- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
