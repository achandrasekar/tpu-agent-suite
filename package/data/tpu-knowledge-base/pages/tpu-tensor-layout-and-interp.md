---
title: TPU Tensor Layout & Interpolation Fallback
type: playbook
tags:
- onboarding
- layout
- interpolation
- bicubic
- bilinear
- groupnorm
created: 2026-08-08
updated: 2026-08-28
sources: 1
human_reviewed: true
aliases:
- bicubic_fallback
- groupnorm_contiguity
target_imports:
- bicubic_interp
- groupnorm
---
# TPU Tensor Layout & Interpolation Fallback

This playbook resolves common runtime kernel crashes in vision, multimodal, and diffusion architectures on Google Cloud TPUs, specifically the missing `aten::upsample_bicubic2d` kernel and `nn.GroupNorm` non-contiguous view assertions.

---

## Applies when

- Vision model (DINOv2, SigLIP, ViT) crashes with: `RuntimeError: aten::upsample_bicubic2d is not implemented for TorchTPU / XLA`.
- Diffusion model crashes during GroupNorm lowering due to non-contiguous tensor strides.

---

## Steps & Transformation Recipes

### Step 1: Intercept Bicubic Interpolation Calls
Replace `mode='bicubic'` with `mode='bilinear'` in position embedding resamplers:
```python
import torch.nn.functional as F

_orig_interpolate = F.interpolate

def safe_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic':
        mode = 'bilinear'
        if align_corners is None:
            align_corners = False
    return _orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)

F.interpolate = safe_interpolate
```

### Step 2: Enforce Contiguity Before GroupNorm
```python
import torch.nn as nn

class ContiguousGroupNorm(nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.contiguous())

nn.GroupNorm = ContiguousGroupNorm
```

---

## See also

- VPU Vector Processing Unit (`index.py get vpu`)
- Padding & Alignment (`index.py get padding`)