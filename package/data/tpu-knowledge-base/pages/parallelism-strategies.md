---
title: "Parallelism strategies (hub)"
type: concept
tags: [parallelism, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["distributed training", "sharding strategies", "model parallelism"]
---

Hub page: the four ways to split a Transformer across chips, each defined by how it
shards the two FFW matmuls `In[B,D] · W_in[D,F] · W_out[F,D]`, and the condition under
which each stays compute-bound. Every row links to a decision-sized page; this page holds
only the comparison (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Comparison

Notation: `B` = global batch in **tokens**, `X`/`Y`/`Z` = mesh-axis sizes, `M_X` = number of
hardware axes a strategy spans, `C` = chip FLOPs/s, `W_ici` = bidirectional ICI bandwidth
per axis, `α = C / W_ici` (ICI arithmetic intensity — per-generation values on
critical-batch-size (`index.py get concept-critical-batch-size`)).

| Strategy | Sharding formula | Comms per layer (fwd + bwd, bytes) | Compute-bound when | Page |
|---|---|---|---|---|
| Data parallelism | `In[B_X,D]·W_in[D,F]·W_out[F,D] → Out[B_X,D]` | `0 + 8DF` (gradient AllReduce, async) | `B/X > α/M_X` | data-parallelism (`index.py get concept-data-parallelism`) |
| FSDP / ZeRO-3 | `In[B_X,D]·W_in[D_X,F]·W_out[F,D_X] → Out[B_X,D]` | `4DF + 8DF` (weight AllGather, grad ReduceScatter) | `B/X > α/M_X` (same as DP, far less memory) | fsdp (`index.py get concept-fsdp`) |
| Tensor (Megatron) | `In[B,D_Y]·W_in[D,F_Y]·W_out[F_Y,D] → Out[B,D_Y]` | `4BD + 4BD` (activation AllGather + ReduceScatter, on the critical path) | `Y < M_Y·F/α` (batch-independent) | tensor-parallelism (`index.py get concept-tensor-parallelism`) |
| FSDP + TP | `In[B_X,D_Y]·W_in[D_X,F_Y]·W_out[F_Y,D_X] → Out[B_X,D_Y]` | `(4BD/X + 4DF/Y) + (8BD/X + 8DF/Y)` | `B/N > α²/(M_X·M_Y·F)` at `X_opt = sqrt(B·M_X·N / (F·M_Y))` | tensor-parallelism (`index.py get concept-tensor-parallelism`) |
| Pipeline | weights sharded over layers `L_Z`, microbatched activations | ~`1.5·2BD/L` per layer — nearly free | bubble- and schedule-bound, not comms-bound | pipeline-parallelism (`index.py get concept-pipeline-parallelism`) |
| Expert (MoE) | experts sharded `W[E_Z,D,F]`, two AllToAlls per layer | AllToAll ≈ ¼ AllGather on a bidi ring | critical batch scaled by `E/k` | expert-parallelism (`index.py get concept-expert-parallelism`) |

All rows from scaling-book, Part 5 (`index.py get scaling-book-training`) except Expert
(scaling-book, Part 4 (`index.py get scaling-book-transformers`),
scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## How to choose (the book's rules of thumb)

1. Model + Adam state fits one chip (`params < HBM/10`): the book says always use plain data
   parallelism, which is compute-bound once the per-chip batch exceeds `α/M_X`
   (scaling-book, Part 5 (`index.py get scaling-book-training`)).
2. Otherwise start with FSDP — same comms roofline as DP, a fraction of the memory.
3. Per-chip batch below `α/M_X`: add tensor parallelism up to `M_Y·F/α` (8–16 way on
   v5p for typical `F`), which lowers the floor to `α²/(M_X M_Y F)` ≈ 100 tokens/chip on
   v5p (scaling-book, Part 5 (`index.py get scaling-book-training`)).
4. Beyond one ICI slice: pure data parallelism over DCN (`index.py get concept-dcn`) needs
   `B/slice > C/W_dcn` (~73k tokens per v5p slice)
   (scaling-book, Part 5 (`index.py get scaling-book-training`)).
5. Pipelining is a GPU-first tool; on TPU pods it is rarely needed for comms reasons and
   complicates FSDP (scaling-book, Part 5 (`index.py get scaling-book-training`),
   scaling-book, Part 12 (`index.py get scaling-book-gpus`)).
6. **Inference is different**: generation admits only model parallelism — FSDP and DP are
   wrong there (prefill-vs-decode (`index.py get concept-prefill-vs-decode`)).

The thresholds tighten sharply on newer generations because FLOPs grew faster than
per-axis ICI (see the derived table on critical-batch-size (`index.py get concept-critical-batch-size`)).

## See also

- Sharding notation (`index.py get concept-sharding-notation`) — the algebra every row is written in
- Collective costs (`index.py get concept-collective-costs`) — where the comms numbers come from
- Compute–communication overlap (`index.py get concept-compute-communication-overlap`)
- Executable version: Size a training parallelism config (`index.py get playbook-size-training-parallelism`)

## Sources

- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
