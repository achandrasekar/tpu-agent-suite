---
title: "Zero-copy views versus layout copies (FLUX milestone 1)"
type: doc-concept
provenance: doc
source: examples/tutorials/FLUX_optimization/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Zero-copy views versus layout copies (FLUX milestone 1)

## Definition
The FLUX optimization tutorial's first lesson: how a tensor is unpacked decides whether the TPU
moves data. In FLUX's `DoubleStreamBlock`, the fused QKV projection of shape `(B, L, 3*H*D)` was
split with `einops.rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=...)`, which moves the
`K=3` axis to the front and forces a physical HBM-to-HBM transposition (visible in the trace as a
`tt_jit_copy__as_strided_inverse` module). The fix keeps `3` adjacent to `(H, D)`:
`img_qkv.view(b, l, 3, H, D)` followed by `torch.unbind(img_qkv, dim=2)`, which yields three
tensors with adjusted offsets and strides over the existing buffer; the head merge becomes
`attn.transpose(1, 2).reshape(b, l, -1)`. In eager mode alone this took step time from
1576.01 ms to 578.03 ms (2.72x).

## In torch_tpu (grounded)
- XLA buffers have no strides, so TorchTPU represents every view as a computation over a
  contiguous base: [`DecomposeIntoViewSequence`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view_decomposition/decomposition.cc#L1244 "csrc/ops/view_decomposition/decomposition.cc#DecomposeIntoViewSequence")
  turns a strided layout into a [`ViewSequence`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view_decomposition/view_sequence.h#L84 "csrc/ops/view_decomposition/view_sequence.h#ViewSequence") of
  reshape, transpose, slice and similar primitives. Kernels such as
  [`AtenView`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/view/view_aten_kernels.cc#L237 "csrc/ops/view/view_aten_kernels.cc#AtenView") and
  [`AtenAsStrided`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/as_strided/as_strided_aten_kernels.cc#L76 "csrc/ops/as_strided/as_strided_aten_kernels.cc#AtenAsStrided") produce such views as cheap
  deferred ops that fuse with their consumers.
- A layout that cannot be expressed that way relative to the base, or a write into a strided
  destination, needs a real copy plus a view inversion: the `as_strided_inverse` part of the
  module name corresponds to [`kAsStridedInverse`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/op_names.h#L79 "csrc/ops/op_names.h#OpName.kAsStridedInverse"),
  built by [`MakeAsStridedInverseBuilder`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/device_buffer_utils.cc#L464 "csrc/eager/device_buffer_utils.cc#MakeAsStridedInverseBuilder").
- Module names are built by [`BuildModuleNameFromPyContext`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/op_builder_utils.cc#L1175 "csrc/ops/op_builder_utils.cc#BuildModuleNameFromPyContext") as
  `tt_jit_<file>_L<line>C<col>_<func>_<op>`, which is what lets you map a copy in the XProf trace
  back to the Python line that triggered it (the tutorial notes you can go either direction:
  suspect code first, or trace first).

## Why it matters / when it applies
- Reshape/rearrange patterns copied from GPU code (einops, `permute(...).contiguous()`) are
  candidates for hidden copies on TPU; each is small, but a transformer block repeats them per
  layer.
- Prefer layouts where the split axis is adjacent to the contiguous inner dimensions, so the
  result is a plain view (`view` plus `unbind`/slicing) rather than a transposition.
- Combined with `torch.compile` the gain compounds: 113.31 ms versus 344.50 ms for compile alone
  ([whole-block-compile-fusion](whole-block-compile-fusion.md)).

## Connections
- Code concepts: [view-decomposition](../concepts/view-decomposition.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [python-context-and-op-names](../concepts/python-context-and-op-names.md),
  [stablehlo-op-builder](../concepts/stablehlo-op-builder.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/ops/view_decomposition`,
  `csrc/ops/view`, `csrc/eager`).
- Related doc-concepts: [fused-algebraic-rope](fused-algebraic-rope.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md),
  [step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md).

## Source
Extracted from `examples/tutorials/FLUX_optimization/README.md` (kept in place).
