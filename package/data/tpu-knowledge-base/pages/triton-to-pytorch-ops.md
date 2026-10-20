---
title: Triton JIT Kernels to Native PyTorch Ops Conversion
type: playbook
tags:
- onboarding
- triton
- custom-kernels
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- triton_unwiring
target_imports:
- triton
---
# Triton JIT Kernels to Native PyTorch Ops Conversion

This playbook describes how to replace proprietary `@triton.jit` GPU kernel decorators and Triton language operators with standard PyTorch tensor operations that compile seamlessly under OpenXLA.

---

## Applies when

- Preflight check flags `triton`, `@triton.jit`, or `triton.language`.
- Model uses custom Triton kernels for LayerNorm, RMSNorm, Rotary Position Embeddings (RoPE), or SwiGLU activations.

---

## Steps & Transformation Recipes

### Step 1: Strip Triton Decorators and Replace Custom Kernels
Replace custom Triton kernel calls with vectorized PyTorch functional operations:

```python
import torch
import torch.nn.functional as F

# 1. Triton RMSNorm -> Native PyTorch RMSNorm
def native_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# 2. Triton SwiGLU -> Native PyTorch SwiGLU
def native_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * x

# 3. Triton RoPE -> Native PyTorch RoPE
def native_apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, D)
    d_2 = x.shape[-1] // 2
    x1, x2 = x[..., :d_2], x[..., d_2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)
```

### Step 2: Inject Triton Mock Module
```python
import sys
import types

triton_mod = types.ModuleType("triton")
triton_mod.jit = lambda fn: fn  # No-op decorator
triton_lang = types.ModuleType("triton.language")
sys.modules["triton"] = triton_mod
sys.modules["triton.language"] = triton_lang
```

---

## See also

- APEX to Native Normalization (`index.py get apex-to-native-normalization`)
- TorchTPU Runtime Reference (`index.py get kb-torch-tpu-overview`)