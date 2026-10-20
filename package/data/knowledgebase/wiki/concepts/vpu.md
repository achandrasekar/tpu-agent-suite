---
title: "Vector Processing Unit (VPU)"
type: concept
tags: [hardware, performance]
created: 2026-08-22
updated: 2026-09-04
sources: 2
human_reviewed: true
---

The VPU is the [TensorCore](tensorcore.md)'s vector arithmetic core: a 2D SIMD machine
handling everything the [MXU](mxu.md) isn't built for — elementwise ops (ReLU, add,
multiply) and reductions. VPU arithmetic runs in fp32 even when matmuls run in lower
precision ([source](../sources/2025-scaling-book-tpus.md)).

## Architecture (v5p numbers)

- Shape (8, 128): 128 **lanes** × 8 **sublanes**, 4 independent fp32 ALUs per
  (lane, sublane); most instructions execute in 1 cycle with 2-cycle latency
  ([source](../sources/2025-scaling-book-tpus.md)).
- Peak ≈ 8·128·4·1.75 GHz ≈ 7e12 FLOPs/s per core — **~30× below the MXU**, so any op
  forced through the VPU is expensive relative to matmul
  ([source](../sources/2025-scaling-book-tpus.md)).
- **VREGs**: 64 32-bit vector registers per core (~256 kB); 3 loads + 1 store to
  [VMEM](vmem.md) per cycle ([source](../sources/2025-scaling-book-tpus.md)).

## Sublane vs lane: reduction asymmetry

Intra-lane (sublane-axis, size 8) shuffles take ~1 cycle, so sublane reductions are
cheap (shuffle 4/2/1 + 3 adds). **Cross-lane reductions need the XLU ("cross lane
unit"), which is slow and expensive** — a layout consideration when placing the
reduction axis ([source](../sources/2025-scaling-book-tpus.md)).

## Scalar core

A single-threaded control unit fetches/dispatches all instructions and initiates
HBM→VMEM transfers — **one DMA request per cycle per core**, controlling ~4096 VPU ALUs,
4 MXUs, 2 XLUs, and the DMA engines. Efficient, but rules out interesting
data-dependent vectorization ([source](../sources/2025-scaling-book-tpus.md)).

GPU analogy: a VPU ALU ≈ a CUDA core; a VPU lane ≈ a warp scheduler
([source](../sources/2025-scaling-book-tpus.md)).

## Versus CUDA cores

A v5p has `2 × 4 × 8 × 128 = 8,192` vector ALUs; an H100 has `132 × 4 × 32 = 16,896` CUDA
cores at similar clocks. CUDA cores are SIMT — each thread has its own instruction
pointer, and divergent warps execute both paths with masking — while the VPU is strict
SIMD over contiguous blocks; GPU SMs also switch among up to 64 resident warps to hide
loads where the TPU is single-threaded ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## See also

- [MXU](mxu.md), [VMEM](vmem.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
