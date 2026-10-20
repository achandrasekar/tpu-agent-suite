---
title: "Excess precision (torch.tpu.allow_excess_precision) and out_dtype accumulators"
type: doc-concept
provenance: doc
source: docs/markdown/precision_management.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Excess precision (torch.tpu.allow_excess_precision) and out_dtype accumulators

## Definition
The MXU accumulates a `bfloat16` matmul in `float32`. Inside a fused computation, the XLA
compiler can keep that `float32` intermediate instead of rounding it to `bfloat16` before the
next op. The doc calls the retained bits **excess precision**. It is controlled by one global,
`torch.tpu.allow_excess_precision`, which sets XLA's `--xla_allow_excess_precision` and defaults
to `False` in TorchTPU:
- **`False`** keeps every intermediate downcast to `bfloat16`, so results match op-by-op
  rounding.
- **`True`** lets the compiler drop those downcasts and keep intermediates in `float32`.

A separate mechanism, the `out_dtype` argument on matmul ops (for example
`torch.bmm(a, b, out_dtype=torch.float32)` with `bfloat16` inputs), forces the result itself to
be returned at higher precision. `allow_excess_precision` only *permits* extra bits; `out_dtype`
*requests* them.

The doc demonstrates the difference with a bitwise test. `(1 + 2**-6) * (1 + 2**-7)` has a bit
at mantissa position 13, which `bfloat16` cannot hold. It multiplies 256x256 scaled identity
matrices in `bfloat16` on the MXU, then multiplies by a one-element `float32` tensor. A one-element
tensor is used rather than a Python scalar because PyTorch type promotion often ignores scalar
dtypes. Bit 13 survives only in a compiled graph with excess precision allowed. In eager mode,
and in compiled mode with the flag `False`, the `bfloat16` matmul output is rounded first.

## In torch_tpu (grounded)
- **Property to compile option.** The `torch.tpu` property
  [`allow_excess_precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L283 "src/torch_tpu/_internal/device/_device_module#_DeviceModuleMeta.allow_excess_precision")
  (experimental getter and setter) calls
  [`SetAllowExcessPrecision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/excess_precision.cc#L54 "csrc/common/excess_precision.cc#SetAllowExcessPrecision"),
  which stores a tri-state
  [`ExcessPrecisionState`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/excess_precision.cc#L29 "csrc/common/excess_precision.cc#ExcessPrecisionState")
  (unset, allow, disallow).
- **Default.** While the state is unset,
  [`GetAllowExcessPrecision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/excess_precision.cc#L59 "csrc/common/excess_precision.cc#GetAllowExcessPrecision")
  honors `xla_allow_excess_precision` only if `XLA_FLAGS` names it explicitly. Otherwise it
  returns `false`, overriding XLA's own default of `true`. That decision is memoized on first
  use.
- **Applied per compilation.**
  [`UpdateFromXlaFlags`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L218 "csrc/common/compilation.cc#UpdateFromXlaFlags")
  writes the current value into every compilation's debug options. The value is part of the
  build-options fingerprint folded into
  [`MakeCompileOptionsKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L441 "csrc/common/compilation.cc#MakeCompileOptionsKey"),
  so flipping it produces new cache keys and recompiles rather than reusing an executable built
  under the other setting.
- **`out_dtype`.** The `out_dtype` matmul overloads are
  [`AtenMmDtype`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/mm/mm_aten_kernels.cc#L131 "csrc/ops/mm/mm_aten_kernels.cc#AtenMmDtype")
  and [`AtenBmmDtype`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/bmm/bmm_aten_kernels.cc#L147 "csrc/ops/bmm/bmm_aten_kernels.cc#AtenBmmDtype").
  They pass the requested element type to
  [`BuildBmmShlo`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/bmm/bmm.cc#L31 "csrc/ops/bmm/bmm.cc#BuildBmmShlo")
  as the dot's result type.

## Why it matters / when it applies
- **Set it once, at the top of the script.** The doc warns that changing it mid-session
  recompiles. It also warns that the TPU XLA runtime "can latch the compiler option on the first
  compilation in a session, ignoring subsequent changes". The demo cells print a warning instead
  of failing when that happens, and call `torch.compiler.reset()` between settings.
- **Allowing excess precision is usually harmless.** It typically makes fused results *more*
  accurate. It makes TPU output differ bit-for-bit from an op-by-op reference, which matters for
  [numerical-parity-checking](numerical-parity-checking.md) and for reproducing eager results
  under `torch.compile`.
- **Use `out_dtype` for a wide result.** When you need a `float32` result from `bfloat16`
  inputs (loss accumulators, logits), `out_dtype` keeps the MXU's `float32` accumulator in the
  output without upcasting the inputs.

## Connections
- Code concepts: [dtype-and-precision](../concepts/dtype-and-precision.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md) (compile options and
  `XLA_FLAGS`), [compilation-cache-key](../concepts/compilation-cache-key.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common`, `csrc/ops/mm`,
  `csrc/ops/bmm`, `src/torch_tpu/_internal/device`).
- Related doc-concepts: [mxu-precision-modes](mxu-precision-modes.md),
  [numerical-parity-checking](numerical-parity-checking.md),
  [xla-compilation-dumps](xla-compilation-dumps.md) (the same `XLA_FLAGS` parsing),
  [environment-variables](environment-variables.md).

## Source
Extracted from `docs/markdown/precision_management.md` (kept in place).
