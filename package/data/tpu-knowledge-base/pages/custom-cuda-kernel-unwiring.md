---
title: Custom CUDA Kernel Unwiring & Native PyTorch Fallback
type: playbook
tags:
  - cuda
  - unwiring
  - pytorch
  - parity
  - tpu
target_imports:
  - custom_cuda
  - torch.ops
  - cuda_extension
created: 2026-09-22
updated: 2026-09-22
sources: 1
human_reviewed: true
---

# Custom CUDA Kernel Unwiring & Native PyTorch Fallback

When `preflight_check` discovers a custom or uncataloged CUDA kernel with no matching playbook (`playbook: null`), follow this procedure to synthesize a pure PyTorch replacement.

## 1. Reverse-Engineering Workflow
1. **Locate Op Definition**: Find custom C++/CUDA extensions, inline kernels (`torch.utils.cpp_extension.load_inline`), or `torch.ops.*` calls in the model repo.
2. **Identify Mathematical Intent**: Determine tensor shapes, inputs, and operation (e.g., custom activations, fused norms, RoPE variants, attention tweaks).
3. **Implement Pure PyTorch Op**: Author an equivalent replacement using standard `torch.*` operations without GPU-specific dependencies.
4. **Monkey-Patch in Harness**: Replace the model's CUDA op/module in `benchmark_tpu.py` or `server_patches.py` and declare `PATCHED_MODULES = ["module.path"]`.

## 2. Numerical Parity Verification
- Execute `numerical-parity` probe capture against CPU reference.
- Verify `max_rel` error is within noise floor (< 3-5%) and `top-1 agree` matches baseline.

## 3. Escalation on Unwiring Failure
If the custom op cannot be represented in PyTorch (e.g., closed-source binary or parity divergence):
- Set `status: "BLOCKED_ON_CUDA_KERNEL"` in `cuda_kernels_report.json`.
- Populate `failure_context` with failing symbol and error details.
- Handoff to `tpu-kernel-development` to synthesize a Pallas TPU kernel.
