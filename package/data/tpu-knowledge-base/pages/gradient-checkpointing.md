---
title: "Gradient checkpointing (rematerialization)"
type: concept
tags: [memory, performance, workload]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["rematerialization", "remat", "activation checkpointing", "jax.checkpoint"]
---

Backpropagation trades compute for memory: it needs `O(L)` saved activations instead of
`O(L²)` recompute, but saving *every* intermediate is enormous — roughly 20 tensors per
layer, `2·20·B·T·D·L ≈ 84 TB` in bf16 for a 4M-token batch, `L=64, D=8192`. Checkpointing
keeps a subset and recomputes the rest in the backward pass
(scaling-book, Part 4 (`index.py get scaling-book-transformers`)). At scale it is the dominant memory
term, not the weights.

## Policies

| Policy | Saved per layer | FLOPs cost |
|---|---|---|
| Save everything | ~20 tensors | `6ND` (none extra) |
| Big matmuls only | ~7 (outputs of Q,K,V,O and the three FFW matmuls) | recompute attention matmuls (`4BT²NH`) and pointwise ops |
| Block remat | 1 (layer input) | forward pass repeated: ~`8ND` instead of `6ND` |

(scaling-book, Part 4 (`index.py get scaling-book-transformers`).) In JAX the knob is
`jax.checkpoint` / `jax.remat` (scaling-book, Part 4 (`index.py get scaling-book-transformers`)); the
torch_tpu spelling is *not yet ingested*.

## Where it sits in the memory budget

LLaMA 3-70B at a 4M-token batch with two checkpoints per layer: params 140 GB, Adam state
560 GB, **activation checkpoints ~10.5 TB** — checkpoints dominate even under a conservative
policy. That total fits ~117 v5p chips by memory, but spread over 8,960 chips it is 1.3 GB
per chip; 12 checkpoints per layer would still be ~8 GB. Training at pod scale is nowhere
near memory-bound, which is why the parallelism choice is driven by
communication rooflines (`index.py get concept-parallelism-strategies`), not by fitting
(scaling-book, Part 6 (`index.py get scaling-book-applied-training`),
scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Trade-off rule

Saving fewer tensors raises FLOPs toward `8ND` and lowers MFU (`index.py get concept-mfu`) as measured against
`6ND`; when per-chip HBM has headroom (as it does at pod scale) save more checkpoints per
layer instead (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).
The choice interacts with FSDP (`index.py get concept-fsdp`), which already shards params and optimizer state
`X` ways but leaves per-chip activations at `B/X` (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## See also

- Transformer FLOPs accounting (`index.py get concept-transformer-flops-accounting`), MFU (`index.py get concept-mfu`),
  HBM (`index.py get concept-hbm`), Size a training parallelism config (`index.py get playbook-size-training-parallelism`)

## Sources

- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
