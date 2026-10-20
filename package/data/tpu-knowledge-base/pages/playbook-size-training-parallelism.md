---
title: "Size a Training Parallelism Configuration"
type: playbook
tags: [parallelism, performance, memory, workload]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["training sharding recipe", "choose dp/fsdp/tp"]
target_imports:
- sharding_config
---
# Size a Training Parallelism Configuration

From a model config, a token batch, a chip count and a TPU generation, derive the
FSDP/TP/DP split that stays compute-bound, the per-chip memory, and the expected step
time. This is the book's Part 5/6 procedure written as steps
(scaling-book, Part 5 (`index.py get scaling-book-training`),
scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).

## Applies when

- Dense Transformer training (MoE: scale thresholds by `E/k`, see
  expert-parallelism (`index.py get concept-expert-parallelism`)).
- Inputs known: `L, D, F, N, K, H, V`; global batch `B` in tokens and sequence length `T`;
  chip count `N_chips` and slice shape; generation (`C`, HBM, ICI per axis, axes `M`).
- Framework-neutral: maps to tpu-sharding shorthand (`fsdp=`, `tp=`, `dp=`, `remat=`) or any
  mesh config; torch_tpu flag names are not yet ingested.

## Steps

1. **Count parameters and FLOPs.** `params ≈ L·(3DF + 2D(N+K)H) + 2DV`; training FLOPs
   `= 6·params·tokens` (`8·` with block remat). Signal: matches the published count (LLaMA
   3-70B → 70.4B) (transformer-flops-accounting (`index.py get concept-transformer-flops-accounting`)).
2. **Memory budget.** Params 2 B/param (bf16), Adam state 8 B/param, activation
   checkpoints `2·B·D·L·(checkpoints per layer)` bytes. Signal: total ÷ HBM per chip gives
   the minimum chip count by memory (LLaMA 3-70B, 4M batch, 2 ckpt/layer: 11.2 TB → 117 v5p)
   (gradient-checkpointing (`index.py get concept-gradient-checkpointing`)).
3. **Look up the generation's thresholds** on
   critical-batch-size (`index.py get concept-critical-batch-size`): `α_ici = C / W_ici`, axes `M`,
   `α_dcn` if multi-slice.
4. **Sequence-batch check.** DP/FSDP without sequence sharding is capped at `B/T` ways;
   beyond that you are sharding the sequence axis too (fine for the MLP; attention needs
   ring attention) (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).
5. **Try FSDP alone.** Compute-bound iff `B / N_chips > α_ici / M_X`. Signal: LLaMA 3-70B on
   8,960 v5p at 4M tokens gives 468 < 850 → fails
   (fsdp (`index.py get concept-fsdp`)).
6. **Add tensor parallelism.** Floor becomes `B/N > α² / (M_X·M_Y·F)`; if satisfied, pick
   `X_opt = sqrt(B · M_X · N / (F · M_Y))`, round to a power of two, `Y = N/X`; check
   `Y ≤ M_Y·F/α`. Signal: LLaMA 3-70B → `X_opt ≈ 1618` → FSDP 2048 (1024 data × 2 sequence),
   TP 4 (tensor-parallelism (`index.py get concept-tensor-parallelism`)).
7. **Multi-slice.** If chips span slices, put FSDP/TP inside each slice and DP across;
   require `B / slice > α_dcn` (~73k tokens on v5p)
   (data-parallelism (`index.py get concept-data-parallelism`)).
8. **Per-chip memory at the chosen split**: (params + state)/X + checkpoints/N. Signal:
   well under HBM (LLaMA 3-70B → ~1.3 GB/chip on a pod), so consider *less* aggressive remat
   to recover FLOPs (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).
9. **Step time and run time.** `T_step = 6·params·B / (N_chips · C · MFU)`; assume 40%
   MFU until measured (LLaMA 3-70B, 15T tokens, v5p pod: 44 days)
   (mfu (`index.py get concept-mfu`)).

## Pitfalls

- Using v5e/v5p thresholds on v6e/TPU7x: `α_ici` is ~2× / ~5× higher there, so FSDP-only
  configs that worked on v5p become comms-bound *(derived, see the table)*.
- Forgetting that `B` is tokens, not sequences.
- Pipelining to escape the floor: it breaks ZeRO-3 and adds bubbles; on a pod prefer more
  TP or a larger batch (pipeline-parallelism (`index.py get concept-pipeline-parallelism`)).
- Leaving FSDP enabled in the inference config later
  (prefill-vs-decode (`index.py get concept-prefill-vs-decode`)).

## Verification

Profile one step (profile-tpu-program (`index.py get playbook-profile-tpu-program`)): weight AllGathers
overlap (not on the critical path), TP AllGather/ReduceScatter are short relative to the
matmuls, achieved MFU within ~10 points of the assumption; collective overhead % and
HBM % as tracked by the optimizer agree with steps 6 and 8.

## See also

- Parallelism strategies (`index.py get concept-parallelism-strategies`), FSDP (`index.py get concept-fsdp`),
  Tensor parallelism (`index.py get concept-tensor-parallelism`), DCN (`index.py get concept-dcn`)

## Sources

- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
