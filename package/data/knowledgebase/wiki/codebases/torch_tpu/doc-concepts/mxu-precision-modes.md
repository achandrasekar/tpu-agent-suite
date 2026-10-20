---
title: "MXU matmul precision modes: DEFAULT, HIGH and HIGHEST (1-, 3- and 6-pass)"
type: doc-concept
provenance: doc
source: docs/markdown/precision_management.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# MXU matmul precision modes: DEFAULT, HIGH and HIGHEST (1-, 3- and 6-pass)

## Definition
The TPU Matrix Multiply Unit (MXU) multiplies `bfloat16` inputs and accumulates in `float32`, so
a `float32` matmul cannot get all 24 bits of significand in one pass. TorchTPU lets you pick
how much of that precision to buy with `torch.tpu.Precision`:
- **`DEFAULT` (1-pass).** The fastest mode, and the default. Only the first 8 significand bits
  are multiplied, with `float32` accumulation.
- **`HIGH` (3-pass).** Several `bfloat16` passes approximate `float32`, multiplying the first
  16 bits.
- **`HIGHEST` (6-pass).** All 24 bits are multiplied, giving full `float32` precision. This is
  the slowest mode.

The passes come from writing each `float32` as a sum of three `bfloat16` pieces (high, medium,
low), after Henry, Tang and Heinecke. Expanding `(a+b+c)(d+e+f)` gives nine products. The three
lowest-order ones (med x low, low x med, low x low) underflow in `float32`, so six products
cover full precision. `torch.tpu.precision(mode)` is the context manager that applies a mode to
a block.

## In torch_tpu (grounded)
- **The Python surface.** [`Precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/precision/precision.py#L36 "src/torch_tpu/_internal/precision/precision#Precision")
  is the StableHLO `dot_general` precision enum bound from C++
  ([`Precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/precision/precision_impl.cc#L25 "csrc/internal/precision/precision_impl.cc#Precision")) and marked experimental.
  [`precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/precision/precision.py#L49 "src/torch_tpu/_internal/precision/precision#precision")
  validates the mode, pushes it with
  [`PyPushPrecision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/precision_context.cc#L45 "csrc/ops/precision_context.cc#PyPushPrecision")
  onto a per-thread stack of
  [`PrecisionContextState`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L70 "csrc/common/context_states.h#PrecisionContextState"),
  and pops it in a `finally`.
- **The global knob.** At device init, TorchTPU patches `torch.set_float32_matmul_precision`
  ([`_tpu_set_precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L326 "src/torch_tpu/_loader#_init_device_impl._tpu_set_precision")).
  `"medium"` maps to `DEFAULT`, `"high"` to `HIGH` and `"highest"` to `HIGHEST`, stored by
  [`PySetGlobalPrecision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/precision_context.cc#L37 "csrc/ops/precision_context.cc#PySetGlobalPrecision").
  The global starts at `DEFAULT`, not PyTorch's CPU default of `"highest"`. A context-manager
  value wins over the global.
  [`PyUnsafeGetPrecision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/precision_context.cc#L53 "csrc/ops/precision_context.cc#PyUnsafeGetPrecision")
  warns once when the global is above `DEFAULT`, suggesting `"medium"` for speed.
- **Where it applies.** Matmul-class kernels read the mode through
  [`GetAndAddPrecisionTo`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/precision_context.cc#L76 "csrc/ops/precision_context.cc#GetAndAddPrecisionTo"),
  which also writes it into the op's cache key. So each mode compiles and caches its own
  executable. Callers include `mm`, `bmm`, `addmm`, `addmv`, `linear`, scaled dot-product
  attention and native multi-head attention. For example,
  [`BuildMmShlo`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/mm/mm.cc#L36 "csrc/ops/mm/mm.cc#BuildMmShlo")
  and [`BuildBmmShlo`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/bmm/bmm.cc#L31 "csrc/ops/bmm/bmm.cc#BuildBmmShlo")
  take the precision as an argument and set it on the StableHLO dot.
- **Under `torch.compile`.** The context manager is not traceable at this pin. It calls
  `torch._dynamo.graph_break` with an explanatory message, so the surrounding frame runs eagerly
  (or raises under `fullgraph=True`).

## Why it matters / when it applies
- **The cost is usually smaller than 3x or 6x.** The doc gives two reasons. Modern models with
  large contracting dimensions are dominated by accumulation noise anyway. Older `float32`
  models are usually HBM-bandwidth-bound and leave MXU cycles idle, so the extra passes run on
  otherwise idle cycles.
- **Weights in `bfloat16` never need this.** The modes matter only for `float32` matmuls, which
  "rarely occur" in inference on recent models.
- **Vector units are separate.** VPU math on `float32` uses all 23 mantissa bits (not strictly
  IEEE in corner cases such as subnormals), so elementwise ops do not lose precision this way.
- **Raise precision locally.** For a `float32` numerical-parity check (see
  [numerical-parity-checking](numerical-parity-checking.md)), wrap just the sensitive matmuls
  in `torch.tpu.precision(Precision.HIGHEST)` rather than raising the global. The doc's timing
  cell compares compile plus run time of an `(8, 2048, 2048)` batched matmul across the three
  modes, and its accuracy cell compares max absolute and mean relative error against a CPU
  reference.

## Connections
- Code concepts: [dtype-and-precision](../concepts/dtype-and-precision.md) (the full precision
  and dtype mechanism), [compilation-cache-key](../concepts/compilation-cache-key.md) (why each
  mode is a separate cache entry),
  [scaled-dot-product-attention](../concepts/scaled-dot-product-attention.md),
  [stablehlo-op-builder](../concepts/stablehlo-op-builder.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/ops`, `csrc/internal/precision`,
  `src/torch_tpu/_internal/precision`).
- Related doc-concepts: [excess-precision-and-out-dtype](excess-precision-and-out-dtype.md),
  [tpu-hardware-model](tpu-hardware-model.md),
  [numerical-parity-checking](numerical-parity-checking.md),
  [api-stability-policy](api-stability-policy.md).

## Source
Extracted from `docs/markdown/precision_management.md` (kept in place).
