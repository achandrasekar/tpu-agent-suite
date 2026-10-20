---
title: "Systolic array"
type: concept
tags: [hardware, performance]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

A systolic array is a 2D grid of multiply-add ALUs through which operands flow
rhythmically — the compute primitive inside every TPU MXU (`index.py get concept-mxu`).
It exploits matmul's O(n³) compute on O(n²) bytes, which is why it stays compute-bound
where ordinary ALUs starve on bandwidth
(source (`index.py get scaling-book-tpus`)).

## Mechanism

- 128×128 grid (16,384 ALUs; 256×256 on TPU v6e+). Weights (RHS, 128×128) load
  diagonally from above; activations (LHS, 8×128) stream in from the left; partial
  sums flow downward, each cell multiplying and accumulating
  (source (`index.py get scaling-book-tpus`)).
- Fully saturated: one `bf16[8,128] @ bf16[128,128] -> f32[8,128]` per 8 cycles
  (source (`index.py get scaling-book-tpus`)).
- An initial pipeline bubble while the first weights/activations load; after that,
  new operands stream in bubble-free — large matmuls pipeline efficiently
  (source (`index.py get scaling-book-tpus`)).

## Optimization implications

- Tensor dimensions must reach the array size (128, or 256 on v6e) or the remainder is
  padded (`index.py get padding`) — and with multiple MXUs per core, tiling dimensions should
  exceed 128 × #MXUs (source (`index.py get scaling-book-tpus`)).
- Small matmuls pay the pipeline bubble disproportionately; batch or fuse them.

## See also

- MXU (`index.py get concept-mxu`) — the hardware block wrapping this
- Roofline analysis (`index.py get concept-roofline-analysis`)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
