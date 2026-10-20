---
title: "Case Study: Porting Hopper-Specialized LLM to Cloud TPU v7"
type: analysis
tags: [case-study, customer-onboarding, standard-intelligence, tpu-v7, fa3, uv-sync]
created: 2026-08-28
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases: [si_case_study, standard_intelligence_report]
---

# Case Study: Porting Hopper-Specialized LLM to Cloud TPU v7

This case study documents the replication procedure for porting the Hopper GPU-specialized Standard Intelligence (`si_torchtpu`) codebase to Google Cloud TPU v7 using the `torch_tpu` backend.

---

## 🔑 Key Engineering Solutions

1. **Dependency Clobbering Prevention:** `pyproject.toml` contained `pytorch-cu128`. Running `uv sync` destroyed `torch_tpu`. Resolved by provisioning via centralized requirements (`requirements-tpu.txt`) and installing non-CUDA packages manually via `uv pip install`.
2. **FlashAttention-3 Disabling:** Swapped `AttentionBackend.FA3` to `AttentionBackend.SDPA_MATH` to compile natively on TPU MXUs.
3. **KV-Cache Bypass:** Overrode FA3 assertions by routing autoregressive concatenation through native SDPA.
4. **CUDA API Stripping:** Removed `torch.cuda.set_device()`, `torch.cuda.memory_allocated()`, and switched `dist.init_process_group(backend='nccl')` to `backend='tpu_dist'`.

---

## 🔗 See also

- FlashAttention to SDPA (tpu-knowledge-base: `flash-attention-to-sdpa`)
- [TPU Distributed & Environment Hygiene](../playbooks/tpu-distributed-and-env-hygiene.md)
- CUDA API Stripping & Runtime (tpu-knowledge-base: `cuda-api-stripping-and-runtime`)
- [TPU7x (Ironwood) Hardware Specs](../hardware/tpu-7x.md)

---

## 📚 Sources

- `Standard Intelligence TPU Migration Report` — Step-by-step replication guide on Cloud TPU v7 (2026-08).\n