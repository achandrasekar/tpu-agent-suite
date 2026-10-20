---
title: FlashAttention to Native PyTorch SDPA Conversion
type: playbook
tags:
- onboarding
- attention
- flash-attention
- fa2
- fa3
- sdpa
- pallas
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 2
human_reviewed: true
aliases:
- flash_attn_unwiring
- fa2_to_sdpa
- fa3_to_sdpa
target_imports:
- flash_attn
- flash_attn_interface
---
# FlashAttention to Native PyTorch SDPA Conversion

This playbook provides actionable recipes to unwire FlashAttention-2 and FlashAttention-3 (Hopper/CUDA-only `.cu` kernels) into native PyTorch `torch.nn.functional.scaled_dot_product_attention` (SDPA), which compiles natively to TPU MXU and Pallas kernels via OpenXLA.

---

## Applies when

- Preflight check flags `flash_attn`, `flash_attn_interface`, or `AttentionBackend.FA3`.
- Model runtime crashes with `ModuleNotFoundError: No module named 'flash_attn'`.
- Codebase contains assertions requiring FlashAttention for KV-cache operations (e.g. `assert self.kv_cache is None, 'kv_cache requires fa3'`).

---

## Steps & Transformation Recipes

### Step 1: Handle Input Tensor Transposition $(B, L, H, D) \leftrightarrow (B, H, L, D)$
FlashAttention expects input tensors in `(Batch, SeqLen, NumHeads, HeadDim)` layout, while PyTorch's native SDPA expects `(Batch, NumHeads, SeqLen, HeadDim)`.

Replace `flash_attn_func` calls with:
```python
import torch
import torch.nn.functional as F

def unwired_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    # 1. Transpose from (Batch, SeqLen, NumHeads, HeadDim) -> (Batch, NumHeads, SeqLen, HeadDim)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # 2. Native PyTorch SDPA (lowered to TPU MXU / Pallas)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
        scale=softmax_scale,
    )

    # 3. Transpose back to (Batch, SeqLen, NumHeads, HeadDim)
    return out.transpose(1, 2).contiguous()
```

### Step 2: Global Module Monkey-Patching for Remote Hugging Face Code
When models use `trust_remote_code=True`, inject dummy `flash_attn` module definitions into `sys.modules` before instantiating the model:
```python
import sys
import types

fa_module = types.ModuleType("flash_attn")
fa_module.flash_attn_func = unwired_flash_attn_func
fa_module.flash_attn_varlen_func = unwired_flash_attn_func

# Register aliases
sys.modules["flash_attn"] = fa_module
sys.modules["flash_attn.flash_attn_interface"] = fa_module
sys.modules["flash_attn.modules"] = types.ModuleType("flash_attn.modules")
sys.modules["flash_attn.modules.mha"] = types.ModuleType("flash_attn.modules.mha")
```

### Step 3: Handling FlashAttention-3 (FA3) & KV-Cache Assertions
In architectures that hardcode `AttentionBackend.FA3`:
1. Override attention backend enum to `AttentionBackend.SDPA_MATH` or `AttentionBackend.SDPA_FLASH`.
2. For generation without native Pallas KV-cache, bypass assertions by executing full-sequence autoregressive concatenation over SDPA.

---

## Mappings Table

| CUDA Construct | TorchTPU Equivalent | Caveats & Notes |
| :--- | :--- | :--- |
| `flash_attn_func(q, k, v, causal=...)` | `F.scaled_dot_product_attention` | Must transpose dimensions 1 and 2 before and after SDPA. |
| `flash_attn_varlen_func(...)` | Padded batched SDPA with attention mask | Pad sequences to static batch lengths for optimal OpenXLA cache hits. |
| `AttentionBackend.FA3` | `AttentionBackend.SDPA_MATH` | Supported natively on TPU MXU across v4, v5e, v6e, and v7. |

---

## Common Pitfalls

> [!CAUTION]
> **Missing `.contiguous()` after Transpose:**
> Calling `transpose(1, 2)` creates non-contiguous tensor strides. Failing to call `.contiguous()` before SDPA will cause XLA graph lowering errors.

---

## Verification

- **Numerical Check:** Run forward pass with identical weights on CPU and TPU; assert $\text{MSE}(y_{\text{tpu}}, y_{\text{cpu}}) < 10^{-5}$.
- **Compilation Check:** Confirm steady-state OpenXLA compilation cache hits without graph recompilation thrashing.

---

## See also

- TensorCore Architecture (`index.py get concept-tensorcore`)
- MXU Matmul Unit (`index.py get concept-mxu`)
- Static Shapes & Tracing (`index.py get playbook-static-shapes-and-tracing`)
- TPU v6e (Trillium) Hardware Specs (`index.py get hw-tpu-v6e`)

---

## Sources

- `2025-scaling-book-tpus` — Section on systolic arrays and attention tiling.
