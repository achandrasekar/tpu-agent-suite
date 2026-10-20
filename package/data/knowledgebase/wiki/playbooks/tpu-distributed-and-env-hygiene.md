---
title: TPU Distributed Backends & Environment Hygiene
type: playbook
tags:
- onboarding
- distributed
- nccl
- tpu-dist
- uv
- environment-hygiene
created: 2026-08-28
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- tpu_dist_onboarding
- uv_dependency_hygiene
target_imports:
- nccl_dist
---
# TPU Distributed Backends & Environment Hygiene

This playbook addresses two critical customer onboarding blockers: switching distributed communication backends from `nccl` to `tpu_dist`, and preventing destructive dependency clobbering when using `uv` on customer repositories.

---

## Applies when

- Customer training/eval loops hardcode `dist.init_process_group(backend='nccl')`.
- Customer repositories contain `pyproject.toml` or `requirements.txt` with explicit `pytorch-cu128` dependencies.

---

## Steps & Transformation Recipes

### 1. Switching Distributed Backend from `nccl` to `tpu_dist`
TPU accelerators communicate across 2D/3D torus topologies using the Google Inter-Chip Interconnect (ICI). Replace `nccl` with `tpu_dist`:

```python
import torch.distributed as dist

# Replace: dist.init_process_group(backend='nccl')
dist.init_process_group(backend='tpu_dist')
```

### 2. Dependency Clobbering Prevention (`uv sync` Hazard)
> [!CAUTION]
> **NEVER execute `uv sync` or `uv run` on a TPU VM when the project's `pyproject.toml` specifies CUDA wheels (`pytorch-cu128`, `flash-attn`).** Doing so will destructively overwrite the system `torch_tpu` runtime with standard x86 CUDA binaries.

**The Safe Recipe:**
1. Provision the environment using centralized requirements (`requirements-tpu.txt`).
2. Inject only downstream non-CUDA packages manually:
   ```bash
   uv pip install transformers jaxtyping beartype datasets pyarrow einops
   ```

---

## See also

- [ICI Interconnect & Topology](../concepts/ici.md)
- [Standard Intelligence Case Study](../analyses/standard-intelligence-case-study.md)\n