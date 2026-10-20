---
title: "Tensor parallelism (Megatron / model parallelism)"
type: concept
tags: [parallelism, performance, interconnect]
created: 2026-09-04
updated: 2026-09-04
sources: 5
human_reviewed: true
aliases: ["TP", "Megatron sharding", "model parallelism", "1D model parallelism", "sequence parallelism"]
---

Tensor parallelism shards the feed-forward dimension of the weights and the model
dimension of the activations: `In[B, D_Y] · W_in[D, F_Y] · W_out[F_Y, D] → Out[B, D_Y]`.
It moves activations instead of weights — an AllGather of `In` before the first matmul and
a ReduceScatter of `Out` after the second, both on the critical path — and so unlocks a
smaller compute-bound batch than [FSDP](fsdp.md), at the price of a hard limit on how many
ways you can shard ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).

## Training cost model

Forward pass per layer: `T_math = 4·B·D·F / (Y·C)`, `T_comms = 4·B·D / W_ici`, so
compute-bound when

`Y < M_Y · F / α`, with `α = C / W_ici`

— independent of batch size and of precision (int8 halves both comms and FLOPs time)
([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). On v5p (`α = 2550`) that is 8–16 way
for most models: LLaMA-3 70B (`F ≈ 30k`) is fine 8-way and comms-bound 16-way; Gemma 7B
(`F ≈ 50k`) allows ~19-way ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). The
per-generation `α` table on [critical-batch-size](critical-batch-size.md) shows the limit
is several times tighter on v6e and TPU7x. The chaining of the two matmuls is what makes
this cheap: gather once before `W_in`, reduce-scatter once after `W_out`, no AllReduce in
between ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).

## Combining with FSDP

FSDP moves weights and TP moves activations, so each shrinks the other's transfer: TP's
`AllGather_Y([B_X, D_Y])` shrinks as `X` grows, FSDP's `AllGather_X([D_X, F_Y])` shrinks as
`Y` grows. With `N = X·Y` chips and `M_X`, `M_Y` axes, minimizing the larger of the two
comms gives

`X_opt = sqrt(B · M_X · N / (F · M_Y))`, and compute-bound when `B/N > α² / (M_X · M_Y · F)`

≈ 100 tokens/chip on v5p with `F = 32k` — about 8× lower than FSDP alone
([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). Worked: LLaMA-3 70B, 4M batch, 8,960
v5p chips → `X_opt ≈ 1618` → 2048-way FSDP (1024 data × 2 sequence) + 4-way TP, just above
the 113-token floor ([scaling-book, Part 6](../sources/2025-scaling-book-applied-training.md)).

## Sequence parallelism (section, not a page)

Sharding the sequence axis is data parallelism for the MLP ("tokens are tokens") and only
attention needs extra gathers, handled by ring attention; the book treats it as another DP
axis rather than a separate strategy ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
During prefill it is the second lever after TP once TP hits the ICI bound
([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## Inference: the only option for generation, with different limits

- Generation admits only model-parallel variants: FSDP moves weights over ICI (far slower
  than HBM), DP replicates without speeding loads, and a single token has no sequence to
  shard ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- **FLOPs-bound limit** as in training, `Y < M_Y·F/α` (~16 chips for LLaMA-3 70B with two
  axes on v5e) ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- **Latency-bound limit**: when decode is HBM-bound, sharding further still helps latency
  until ICI time exceeds weight-load time, `Y > F / (B · β)` with `β = W_hbm / W_ici` (~8 on
  v5e/v6e) — e.g. `F = 16k, B = 32` allows 64-way with little throughput cost
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)). On TPU7x `β` is several times
  larger, so this headroom shrinks (derived on [critical-batch-size](critical-batch-size.md)).
- Small activations make these collectives latency-bound (`Y > BD/45,000` in int8)
  ([collective-costs](collective-costs.md#latency-bound-regime)).
- Attention: shard `W_Q`/`W_O` over heads Megatron-style; KV weights are small enough to
  replicate beyond `K`-way ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)). Sharding
  the [KV cache](kv-cache.md) beyond the head count needs batch sharding plus two AllToAlls.
- **2D weight-stationary** (weights sharded on both `D` and `F`, activations on `D`; an
  AllGather over `YZ` then a ReduceScatter over `X`) beats 1D Megatron above
  `N > 32·(F/D)·(3/4)²` ≈ 72 chips for `F = 4D`, and its comms keep falling with `N`
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## Pitfalls

- TP collectives sit on the critical path; XLA overlaps them imperfectly, and the book
  reports a latency hit on sharded matmuls above BS≈32 in generation
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)). [Collective matmul](collective-matmul.md)
  is the fix when it matters.
- On GPUs TP is confined to one NVLink node (~8-way, `Y < F/2475`)
  ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)); TPU pods lift that limit to the ICI
  bound above.

## See also

- [Parallelism strategies](parallelism-strategies.md) (hub), [FSDP](fsdp.md),
  [Collective costs](collective-costs.md), [Prefill vs decode](prefill-vs-decode.md)

## Sources

- [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md)
- [How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-training.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-inference.md)
- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
