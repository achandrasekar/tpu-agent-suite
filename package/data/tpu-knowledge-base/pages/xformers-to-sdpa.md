---
title: Meta xFormers to PyTorch SDPA Conversion
type: playbook
tags:
- onboarding
- attention
- xformers
- sdpa
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- xformers_unwiring
target_imports:
- xformers
---
# Meta xFormers to PyTorch SDPA Conversion

This playbook provides recipes to unwire Meta's `xformers.ops.memory_efficient_attention` into native PyTorch `F.scaled_dot_product_attention` for execution on Google Cloud TPUs.

---

## Applies when

- Preflight check flags `xformers` or `xformers.ops.memory_efficient_attention`.
- Vision transformer or diffusion model fails to initialize due to missing `xformers` CUDA binaries.

---

## Steps & Transformation Recipes

Replace `memory_efficient_attention` calls with:
```python
import torch
import torch.nn.functional as F

def unwired_memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    # xFormers layout: (Batch, SeqLen, Heads, Dim) -> (Batch, Heads, SeqLen, Dim)
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    mask = None
    if attn_bias is not None:
        if isinstance(attn_bias, torch.Tensor):
            mask = attn_bias.to(q.dtype)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=p if torch.is_grad_enabled() else 0.0, scale=scale
    )
    return out.transpose(1, 2).contiguous()
```

Inject module fallback:
```python
import sys
import types

xf_mod = types.ModuleType("xformers")
xf_ops = types.ModuleType("xformers.ops")
xf_ops.memory_efficient_attention = unwired_memory_efficient_attention
sys.modules["xformers"] = xf_mod
sys.modules["xformers.ops"] = xf_ops
```

---

## See also

- FlashAttention to SDPA (`index.py get flash-attention-to-sdpa`)
- TensorCore Architecture (`index.py get concept-tensorcore`)