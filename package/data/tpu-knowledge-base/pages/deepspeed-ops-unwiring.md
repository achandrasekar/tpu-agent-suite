---
title: DeepSpeed Custom Ops Unwiring
type: playbook
tags:
- onboarding
- deepspeed
- optimizers
- cuda-unwiring
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- deepspeed_unwiring
target_imports:
- deepspeed
---
# DeepSpeed Custom Ops Unwiring

This playbook details how to unwire DeepSpeed CUDA optimizers (`FusedAdam`, `CPUAdam`) and runtime wrappers for execution on Google Cloud TPUs.

---

## Applies when

- Preflight check flags `deepspeed` or `deepspeed.ops.adam`.

---

## Steps & Transformation Recipes

Replace DeepSpeed optimizers with PyTorch native implementations:
```python
import sys
import types
import torch

ds_mod = types.ModuleType("deepspeed")
ds_ops = types.ModuleType("deepspeed.ops")
ds_adam = types.ModuleType("deepspeed.ops.adam")
ds_adam.FusedAdam = torch.optim.AdamW

sys.modules["deepspeed"] = ds_mod
sys.modules["deepspeed.ops"] = ds_ops
sys.modules["deepspeed.ops.adam"] = ds_adam
```

---

## See also

- TPU Distributed & Environment Hygiene (`index.py get playbook-tpu-distributed-and-env-hygiene`)