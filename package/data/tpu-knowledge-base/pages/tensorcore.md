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
to HBM (`index.py get concept-hbm`). Hub page — the components carry the detail.

## Components

| Unit | Role |
|---|---|
| MXU (`index.py get concept-mxu`) | Matrix multiplication via a systolic array (`index.py get concept-systolic-array`); the reason TPUs exist |
| VPU (`index.py get vpu`) | Elementwise vector ops and reductions; also hosts VREGs, the scalar core, and the XLU |
| VMEM (`index.py get concept-vmem`) | Programmer-controlled on-chip scratchpad between HBM and the compute units |

Data flow: tensors stream HBM → VMEM → MXU/VPU and results stream back, with all stages
pipelined (overlap (`index.py get concept-compute-communication-overlap`)); the HBM↔VMEM and
VMEM↔MXU bandwidths set what runs efficiently
(source (`index.py get scaling-book-tpus`)). On v4–v6 chips, two TensorCores
sharing memory form one logical accelerator (megacore (`index.py get megacore`)).

## See also

- ICI (`index.py get concept-ici`), PCIe (`index.py get concept-pcie`), DCN (`index.py get concept-dcn`) — how chips connect outward
- Roofline analysis (`index.py get concept-roofline-analysis`)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
