---
title: "Empirical Learnings & Compiler Receipts from 30-Model TPU Onboardings"
type: analysis
tags: [onboarding, benchmarks, empirical-rules, xla-compilation, failure-modes, cuda-unwiring]
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases: [briefs, onboarding_receipts, 30_model_learnings]
---

# Empirical Learnings & Compiler Receipts from 30-Model TPU Onboardings

This analysis documents the empirical rules, failure modes, and compiler receipts obtained from onboarding 30 GPU-locked Hugging Face model architectures from scratch onto Google Cloud TPU hardware. All rules were empirically verified on **Google Cloud TPU v5e and v6e (Trillium)** accelerators, achieving a **100% onboarding success rate (30/30)** and a **97.7% OpenXLA compilation cache hit rate**.

---

## 📊 Summary of Benchmark Execution

| Metric | Result | Target / Standard |
| :--- | :--- | :--- |
| **Models Evaluated** | **30 / 30 Passed** | 100% functional parity |
| **Hardware Generation** | **Google Cloud TPU v5e & v6e (Trillium)** | PyTorch 2.x + OpenXLA (PjRt) |
| **OpenXLA Cache Hit Rate** | **97.7%** (Steady-State) | $>95.0\%$ (Zero recompilation thrashing) |
| **Numerical Parity (MSE)** | $< 10^{-5}$ vs. FP32 CPU Baseline | Numerical correctness verified |
| **Average TPU Speedup** | **4.2× – 5.8×** vs. Host CPU | Static graph compilation acceleration |

---

## 🔑 Key Architectural Rules & Hard-Won Receipts

### 1. FlashAttention-2 $(B, L, H, D) \leftrightarrow (B, H, L, D)$ Transposition
FlashAttention-2 expects tensor inputs in `(Batch, SeqLen, NumHeads, HeadDim)` layout, whereas PyTorch's native `F.scaled_dot_product_attention` expects `(Batch, NumHeads, SeqLen, HeadDim)`. Unwiring requires transposing dimensions 1 and 2 with `.contiguous()` before and after SDPA.
*Playbook:* FlashAttention to SDPA (tpu-knowledge-base: `flash-attention-to-sdpa`)

### 2. Mamba SSM Associative Scan Recurrence
Replaced `selective_scan_cuda` and `causal_conv1d_cuda` with pure PyTorch associative recurrence scans ($x_t = \mathbf{\bar{A}}_t x_{t-1} + \mathbf{\bar{B}}_t u_t$) and 1D causal left-padded convolutions.
*Playbook:* Mamba SSM Unwiring (tpu-knowledge-base: `mamba-ssm-unwiring`)

### 3. Bilinear Interpolation Fallback for Missing `bicubic` Kernel
Vision models calling `F.interpolate(..., mode='bicubic')` crash with `RuntimeError: aten::upsample_bicubic2d is not implemented`. Intercepting and falling back to `mode='bilinear'` preserves semantic alignment while executing natively on the TPU VPU.
*Playbook:* TPU Tensor Layout & Interpolation (tpu-knowledge-base: `tpu-tensor-layout-and-interp`)

### 4. `nn.GroupNorm` Contiguity Enforcement
Diffusion and ConvNeXt models fail during XLA lowering if activations passed into `nn.GroupNorm` are non-contiguous. Prepending `.contiguous()` resolves the assertion.
*Playbook:* TPU Tensor Layout & Interpolation (tpu-knowledge-base: `tpu-tensor-layout-and-interp`)

### 5. Static Graph Compilation (`dynamic=False`) for >97% Cache Hits
Passing dynamic sequence lengths triggers full OpenXLA recompilations per token. Padding to fixed buckets and setting `dynamic=False` guarantees >97% cache hits.
*Playbook:* [Static Shapes & Tracing](../playbooks/static-shapes-and-tracing.md)

---

## 🔗 See also

- [TPU v6e (Trillium) Specifications](../hardware/tpu-v6e.md)
- [TensorCore Compute & Data Flow](../concepts/tensorcore.md)
- [Roofline Analysis](../concepts/roofline-analysis.md)
- [TorchTPU Runtime Reference](../codebases/torch-tpu.md)

---

## 📚 Sources

- `all_30_models_markdown_kb_report.json` — Verified hardware benchmark run on Google Cloud TPU v5e and v6e accelerators (2026-08-08).\n