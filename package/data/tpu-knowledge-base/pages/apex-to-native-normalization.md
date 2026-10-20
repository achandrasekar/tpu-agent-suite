---
title: NVIDIA APEX to Native PyTorch Normalization Conversion
type: playbook
tags:
- onboarding
- apex
- layernorm
- rmsnorm
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- apex_unwiring
target_imports:
- apex
---
# NVIDIA APEX to Native PyTorch Normalization Conversion

This playbook describes how to replace NVIDIA APEX fused operators (`FusedLayerNorm`, `FusedRMSNorm`) with standard `torch.nn` layers on TPU.

---

## Applies when

- Preflight check flags `apex` or `apex.normalization`.
- Codebase crashes with `ModuleNotFoundError: No module named 'apex'`.

---

## Steps & Transformation Recipes

Replace `apex.normalization.FusedLayerNorm` with `torch.nn.LayerNorm`:
```python
import sys
import types
import torch
import torch.nn as nn

class NativeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

apex_mod = types.ModuleType("apex")
apex_norm = types.ModuleType("apex.normalization")
apex_norm.FusedLayerNorm = nn.LayerNorm
apex_norm.FusedRMSNorm = NativeRMSNorm

sys.modules["apex"] = apex_mod
sys.modules["apex.normalization"] = apex_norm
```

---

## See also

- Triton to PyTorch Ops (`index.py get triton-to-pytorch-ops`)