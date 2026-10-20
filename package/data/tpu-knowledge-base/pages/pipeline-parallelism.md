---
title: "Pipeline parallelism"
type: concept
tags: [parallelism, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 2
human_reviewed: true
aliases: ["PP", "pipelining", "microbatching", "zero-bubble pipelining"]
---

Pipeline parallelism shards the *layer* axis (`W[L_Z, D, F]`), runs stage `i` on chip `i`,
and passes activations forward (and gradients backward) one hop at a time. Its
communication is tiny, which is why it dominates GPU training, but on TPU pods with dense
ICI it is rarely necessary and it complicates everything else
(scaling-book, Part 5 (`index.py get scaling-book-training`),
scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## Cost model

With `N_MB` microbatches and `N_stages` stages, each hop moves `2BD / N_MB` bytes and
there are `N_MB + N_stages − 2` hops, so per-layer comms is roughly `1.5 · 2BD / (W · L)`
— divided by the layer count, hence "basically free"
(scaling-book, Part 12 (`index.py get scaling-book-gpus`)). The cost is elsewhere:

- **Bubbles**: in a naive schedule stage 0 idles except at the start and end of the
  step. Microbatching shrinks the bubble; zero-bubble schedules (the book shows DeepSeek-V3's)
  hide it by interleaving the forward matmul, the `dL/dx` matmul and the `dL/dW` matmul so
  no FLOPs are stranded (scaling-book, Part 5 (`index.py get scaling-book-training`)).
- **Incompatible with ZeRO-3**: each microbatch would need its own weight AllGather with
  only `B/N_MB` tokens to amortize it; gradient reductions cannot start until the last
  microbatch clears a stage, leaving non-overlapped comms. Pair pipelining with ZeRO-1
  instead (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).
- **Code complexity**: microbatching and custom schedules do not fit automatic
  partitioners like GSPMD cleanly (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).
- Activation transfers between stages are on the critical path and shift stages relative
  to each other (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## When it earns its place

- Any model parallelism that spans nodes (PP, EP, TP) divides the cost of the outer
  data-parallel reduction by the number of nodes spanned; 16-way PP is how LLaMA-3
  trains at ~1k tokens per GPU, below the ~2.5k DP floor
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).
- On TPU the same reasoning applies across DCN-linked slices, but inside a pod the
  FSDP + TP (`index.py get concept-tensor-parallelism`) recipe usually reaches the
  compute-bound regime first (scaling-book, Part 5 (`index.py get scaling-book-training`)).

Recipes the book cites: DeepSeek-V3 = 64-way EP × 16-way PP × 2-way ZeRO-1 on 2,048 H800;
LLaMA-3 = 8-way TP × 16-way PP × 128-way ZeRO-1 on 16k H100
(scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## See also

- Parallelism strategies (`index.py get concept-parallelism-strategies`) (hub), FSDP (`index.py get concept-fsdp`),
  Expert parallelism (`index.py get concept-expert-parallelism`)

## Sources

- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
