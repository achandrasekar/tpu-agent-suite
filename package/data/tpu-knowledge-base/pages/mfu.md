---
title: "MFU (model FLOPs utilization) and time estimates"
type: concept
tags: [performance, workload]
created: 2026-09-04
updated: 2026-09-04
sources: 4
human_reviewed: true
aliases: ["model FLOPs utilization", "FLOPs utilization", "step time estimate"]
---

MFU is the fraction of a chip's peak FLOPs/s that useful model FLOPs actually consume.
Combined with the `6 · params · tokens` training rule it turns a model size, a token budget
and a chip count into a wall-clock estimate, and the book uses 40–50% as a realistic
target on TPU (scaling-book, Part 6 (`index.py get scaling-book-applied-training`),
scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Formulas

- Training FLOPs ≈ `6 · N_params · N_tokens` (2 forward + 4 backward per parameter);
  inference ≈ `2 · N_params · N_tokens`
  (transformer-flops-accounting (`index.py get concept-transformer-flops-accounting`)).
- Time ≈ `FLOPs / (N_chips · C · MFU)`. LLaMA-3 70B, 15T tokens: `6.3e24` FLOPs → 435
  years on one v5p, **44 days** on an 8,960-chip v5p pod at 40% MFU
  (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).
- With block rematerialization the forward pass is repeated, so budget ~`8ND` instead of
  `6ND` (gradient-checkpointing (`index.py get concept-gradient-checkpointing`)).
- Inverse use: reported compute reveals achieved utilization — DeepSeek-V3's 2.79M H800
  hours for 14.8T tokens on 37B activated parameters imply ~22% FLOPs utilization
  (scaling-book, Part 4 (`index.py get scaling-book-transformers`)).

## What caps it

- Being comms-bound: the critical batch size (`index.py get concept-critical-batch-size`) per chip must be
  exceeded for the chosen parallelism, otherwise MFU falls regardless of kernels
  (scaling-book, Part 5 (`index.py get scaling-book-training`)).
- Hardware ceiling: the book notes GPUs typically reach 80–85% of claimed peak while TPUs
  get closer to 95% in normal use (scaling-book, Part 1 (`index.py get scaling-book-roofline`)).
- Prefill throughput per chip is MFU-limited outright (always compute-bound); decode is
  not, because it is bandwidth-bound (prefill-vs-decode (`index.py get concept-prefill-vs-decode`)).

## Checking it

Compute the expected time of one matmul from its shape and the chip's peak, read the
measured time from a trace, compare: the book's example FFW matmul was predicted at
95.6 ms and measured at 96 ms (profile-tpu-program (`index.py get playbook-profile-tpu-program`)).

## See also

- Roofline analysis (`index.py get concept-roofline-analysis`), Critical batch size (`index.py get concept-critical-batch-size`),
  Size a training parallelism config (`index.py get playbook-size-training-parallelism`)

## Sources

- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 1: Rooflines (scaling-book) (`index.py get scaling-book-roofline`)
