---
title: "TensorCore"
type: concept
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

The TensorCore is the compute core of a TPU chip: a matrix-multiply specialist attached
to [HBM](hbm.md). Hub page — the components carry the detail.

## Components

| Unit | Role |
|---|---|
| [MXU](mxu.md) | Matrix multiplication via a [systolic array](systolic-array.md); the reason TPUs exist |
| [VPU](vpu.md) | Elementwise vector ops and reductions; also hosts VREGs, the scalar core, and the XLU |
| [VMEM](vmem.md) | Programmer-controlled on-chip scratchpad between HBM and the compute units |

Data flow: tensors stream HBM → VMEM → MXU/VPU and results stream back, with all stages
pipelined ([overlap](compute-communication-overlap.md)); the HBM↔VMEM and
VMEM↔MXU bandwidths set what runs efficiently
([source](../sources/2025-scaling-book-tpus.md)). On v4–v6 chips, two TensorCores
sharing memory form one logical accelerator ([megacore](megacore.md)).

## See also

- [ICI](ici.md), [PCIe](pcie.md), [DCN](dcn.md) — how chips connect outward
- [Roofline analysis](roofline-analysis.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
