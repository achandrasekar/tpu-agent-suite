---
title: "Fused algebraic RoPE (FLUX milestone 2)"
type: doc-concept
provenance: doc
source: examples/tutorials/FLUX_optimization/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Fused algebraic RoPE (FLUX milestone 2)

## Definition
FLUX encodes positions with 3-axis rotary embeddings (`axes_dim=(16, 56, 56)`, summing to
`head_dim=128`). The reference builds a 5-D rotation-matrix tensor `(B, S, D/2, 2, 2)` by
stacking `[cos, -sin, sin, cos]` and applies it to queries and keys reshaped to 6-D, which shows
up in traces as large copy blocks. The optimized version stores only compact `[cos, sin]` pairs
`(B, S, D/2, 2)` (half the HBM footprint and bandwidth) and applies the rotation algebraically,
`(x0*cos - x1*sin, x0*sin + x1*cos)`, on a `view(b, s, h, d//2, 2)` of the input, then stacks and
views back. The tutorial's point generalizes: replace materialized helper matrices with direct
elementwise formulations XLA can fuse. With zero-copy views and compile already on, this cut
step time from 113.31 ms to 69.16 ms (1.64x more, 22.79x over baseline).

## In torch_tpu (grounded)
- In eager mode each elementwise op (`*`, `-`, `+`, `cos`, `sin`) is its own deferred op
  ([`BinaryOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/binary_aten_kernels.h#L124 "csrc/ops/binary_aten_kernels.h#BinaryOp"), [`UnaryOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/unary_aten_kernels.cc#L106 "csrc/ops/unary_aten_kernels.cc#UnaryOp"));
  materialized intermediates such as the 2x2 matrices round-trip through HBM.
- `torch.stack` lowers through [`AtenStack`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/cat/cat_aten_kernels.cc#L271 "csrc/ops/cat/cat_aten_kernels.cc#AtenStack") and the reshapes
  through [`AtenView`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view/view_aten_kernels.cc#L237 "csrc/ops/view/view_aten_kernels.cc#AtenView"); keeping the pair axis innermost
  makes the final `view(b, s, h, d)` a plain view rather than a layout copy
  ([zero-copy-views-vs-layout-copies](zero-copy-views-vs-layout-copies.md)).
- Under `torch.compile` the whole rotation becomes part of one StableHLO program, so XLA fuses
  it into the surrounding projection and attention ([whole-block-compile-fusion](whole-block-compile-fusion.md)).
  TorchTPU has no dedicated RoPE kernel; the gain comes entirely from the formulation.

## Why it matters / when it applies
Applies to any positional-embedding or small-matrix trick written for readability on GPU:
explicit rotation matrices, one-hot expansions, or high-rank reshapes for a 2x2 product. On TPU
these cost HBM bandwidth and block fusion. Express them as elementwise math on views. If a
formulation still does not fuse well, the next step is a custom kernel
([pallas-custom-kernels](../concepts/pallas-custom-kernels.md)); the tutorial lists custom
kernels as its fourth optimization theme.

## Connections
- Code concepts: [aten-kernel-authoring](../concepts/aten-kernel-authoring.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [view-decomposition](../concepts/view-decomposition.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/ops`, `csrc/ops/cat`,
  `csrc/ops/view`).
- Related doc-concepts: [zero-copy-views-vs-layout-copies](zero-copy-views-vs-layout-copies.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md).

## Source
Extracted from `examples/tutorials/FLUX_optimization/README.md` (kept in place).
