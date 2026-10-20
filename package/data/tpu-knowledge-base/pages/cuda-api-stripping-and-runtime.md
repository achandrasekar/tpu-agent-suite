---
title: CUDA Device, Metric & Runtime API Stripping
type: playbook
tags:
- onboarding
- cuda-apis
- device-placement
- precision
- autocast
created: 2026-08-28
updated: 2026-10-14
sources: 1
human_reviewed: true
aliases:
- cuda_api_stripping
- device_binding_tpu
target_imports:
- cuda_api
---
# CUDA Device, Metric & Runtime API Stripping

This playbook details how to strip hardcoded NVIDIA CUDA device APIs, memory allocation metrics, and CUDA autocast decorators, replacing them with dynamic `torch.device('tpu')` bindings and native PyTorch precision controls (`torch.bfloat16`, `torch.autocast`).

---

## Applies when

- Codebase contains hardcoded calls to `torch.cuda.set_device()`, `torch.cuda.current_device()`, or `torch.cuda.is_available()`.
- Evaluation loops measure memory via `torch.cuda.memory_allocated()` or `torch.cuda.max_memory_allocated()`.
- Functions are decorated with `@torch.autocast(device_type='cuda')`.

---

## Steps & Transformation Recipes

### Step 1: Replace Device Assignment & Runtime Queries
Replace CUDA device strings with dynamic TPU device bindings, and translate CUDA device/stream/sync calls to PyTorch 2.11+ `torch.accelerator` (with fallback to `torch.tpu.*` on PyTorch < 2.11):
```python
import torch

# Replace: device = torch.device('cuda:0')
device = torch.device('tpu')

# Replace CUDA queries & controls with torch.accelerator (PyTorch 2.11+; torch.tpu.* fallback on < 2.11):
# torch.cuda.is_available()       -> torch.accelerator.is_available()
# torch.cuda.device_count()       -> torch.accelerator.device_count()
# torch.cuda.current_device()     -> torch.accelerator.current_device_index()
# torch.cuda.set_device(rank)     -> torch.accelerator.set_device_index(rank)
# with torch.cuda.device(rank):   -> with torch.accelerator.device_index(rank):
# torch.cuda.synchronize()        -> torch.accelerator.synchronize()
# torch.cuda.current_stream()     -> torch.accelerator.current_stream()
# torch.cuda.set_stream(s)        -> torch.accelerator.set_stream(s)
# torch.cuda.initial_seed()       -> torch.accelerator.random.initial_seed()  # PyTorch 2.13+ (torch.tpu fallback on < 2.13)
# torch.cuda.get_rng_state()      -> torch.accelerator.random.get_rng_state() # PyTorch 2.13+ (torch.tpu fallback on < 2.13)
```

### Step 2: Strip CUDA Memory Profiler Calls
Replace NVIDIA memory query functions with safe no-op or host memory counters:
```python
# Replace torch.cuda.memory_allocated() with 0 or host RAM query
def safe_memory_allocated():
    return 0

torch.cuda.memory_allocated = safe_memory_allocated
torch.cuda.max_memory_allocated = safe_memory_allocated
torch.cuda.reset_peak_memory_stats = lambda: None
```

### Step 3: TPU Precision Control
Replace CUDA autocast decorators and manual type casting with standard PyTorch TPU precision controls:
```python
import torch

# Option A: Model and tensor precision casting
model = model.to(torch.bfloat16).to("tpu")

# Option B: Standard PyTorch TPU autocast context
with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
    output = model(inputs)
```

---

## See also

- TorchTPU Runtime Reference (`index.py get kb-torch-tpu-overview`)
