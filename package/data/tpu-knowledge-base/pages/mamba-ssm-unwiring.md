---
title: Mamba State-Space Model (SSM) Unwiring
type: playbook
tags:
- onboarding
- mamba
- ssm
- selective-scan
- causal-conv1d
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- mamba_unwiring
- selective_scan_unwiring
target_imports:
- mamba_ssm
- causal_conv1d
- selective_scan_cuda
---
# Mamba State-Space Model (SSM) Unwiring

This playbook details how to unwire Mamba and State-Space Model (SSM) architectures by replacing custom CUDA C++ kernels (`selective_scan_cuda` and `causal_conv1d_cuda`) with pure PyTorch associative recurrence scans that compile cleanly on Google Cloud TPUs via OpenXLA.

---

## Applies when

- Preflight check flags `mamba_ssm`, `selective_scan_cuda`, or `causal_conv1d_cuda`.
- Model instantiation crashes with `ImportError: cannot import name 'selective_scan_cuda' from 'mamba_ssm'`.

---

## Steps & Transformation Recipes

### Step 1: Pure PyTorch Selective Scan Implementation
The continuous-time state space model parameters $(\mathbf{\Delta}, \mathbf{A}, \mathbf{B}, \mathbf{C})$ are discretized into discrete recurrence matrices:
$$\mathbf{\bar{A}}_t = \exp(\mathbf{\Delta}_t \mathbf{A}), \quad \mathbf{\bar{B}}_t = \mathbf{\Delta}_t \mathbf{B}_t$$
$$x_t = \mathbf{\bar{A}}_t x_{t-1} + \mathbf{\bar{B}}_t u_t, \quad y_t = \mathbf{C}_t x_t + \mathbf{D} u_t$$

Replace `selective_scan_fn` with:
```python
import torch
import torch.nn.functional as F

def selective_scan_pure_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    b, d, l = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretization
    deltaA = torch.exp(torch.einsum("b d l, d n -> b d l n", delta, A))
    deltaB_u = torch.einsum("b d l, b n l, b d l -> b d l n", delta, B, u)

    # Sequential associative recurrence scan
    x = torch.zeros((b, d, n), device=u.device, dtype=deltaA.dtype)
    ys = []
    for i in range(l):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum("b d n, b n -> b d", x, C[:, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z)
    return y.to(u.dtype)
```

### Step 2: Causal 1D Convolution Replacement
Replace `causal_conv1d_cuda` with PyTorch `F.conv1d` using explicit left-padding:
```python
def causal_conv1d_pure_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    # x shape: (B, D, L)
    seq_len = x.shape[-1]
    kernel_size = weight.shape[-1]
    # Left-pad sequence by (kernel_size - 1)
    x_padded = F.pad(x, (kernel_size - 1, 0))
    out = F.conv1d(x_padded, weight, bias=bias, groups=weight.shape[0])
    if activation == "silu":
        out = F.silu(out)
    return out[:, :, :seq_len]
```

### Step 3: Module Injection
Register mock `mamba_ssm` modules into `sys.modules`:
```python
import sys
import types

mamba_mod = types.ModuleType("mamba_ssm")
ops_mod = types.ModuleType("mamba_ssm.ops")
scan_mod = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
conv_mod = types.ModuleType("causal_conv1d")

scan_mod.selective_scan_fn = selective_scan_pure_pytorch
conv_mod.causal_conv1d_fn = causal_conv1d_pure_pytorch

sys.modules["mamba_ssm"] = mamba_mod
sys.modules["mamba_ssm.ops"] = ops_mod
sys.modules["mamba_ssm.ops.selective_scan_interface"] = scan_mod
sys.modules["causal_conv1d"] = conv_mod
```

---

## Verification

- **Recurrence Validation:** Verify outputs against FP32 CPU reference implementation with $\text{MSE} < 10^{-5}$.
- **Compilation Bounds:** Wrap inference calls with `torch.compile(dynamic=False)` to enable OpenXLA loop fusion across time steps.

---

## See also

- Static Shapes & Tracing (`index.py get playbook-static-shapes-and-tracing`)
- Roofline Analysis (`index.py get concept-roofline-analysis`)
- HBM Memory Bandwidth (`index.py get concept-hbm`)

---

## Sources

- 30-model onboarding runs (report not published) — Verified on Mamba-130M, Mamba-370M, and Mamba-790M on TPU v5e / v6e.