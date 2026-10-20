---
title: "One-bit quantization example: QAT, bit packing and a popcount sum"
type: doc-concept
provenance: doc
source: docs/markdown/extending_modules_and_functions_via_composition.md
sources: [docs/markdown/extending_modules_and_functions_via_composition.md, docs/markdown/quant.md, docs/markdown/quantized_sum.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# One-bit quantization example: QAT, bit packing and a popcount sum

## Definition
The composition and custom-op chapters of the user guide share one toy quantization format.
The guide says it is meant for teaching and "should not be considered realistic". The format:
- **Quantize.** `scale = mean(|w|)`, `signs = sign(w)`, with zeros forced to `+1`.
- **Dequantize.** `signs * scale`.
- **QAT trick.** `F.linear(x, (w_q - w).detach() + w)`. The forward pass uses the quantized
  weight `w_q`, and gradients flow to the `float32` master weight `w` (a straight-through
  estimator).
- **Packing.** `pack32` turns 32 signs into one `uint32`: `(signs == 1)` gives bits, which are
  multiplied by `2 ** arange(32)` and summed. Summing distinct powers of two is a bitwise-OR
  reduction. `unpack32` shifts right by `arange(32)`, masks with `& 1` and maps back to +1/-1.
  `torch.vmap(pack32)` extends this to matrices.
- **Sum without unpacking.** Each set bit stands for `+scale` and each clear bit for `-scale`,
  so the sum of a packed tensor is `(2 * popcount - total_bits) * scale`.

## In torch_tpu (grounded)
- **Ordinary ATen ops.** Every step except popcount is built from existing ops, so it runs on
  `tpu` without custom code ([extending-pytorch-on-tpu](extending-pytorch-on-tpu.md)). The
  relevant kernels include `pow` with a scalar base
  ([`AtenPowScalarOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/binary_aten_kernels.cc#L1702 "csrc/ops/binary_aten_kernels.cc#AtenPowScalarOut")),
  the dtype-carrying sum
  ([`AtenSumIntListOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/reductions/sum_aten_kernels.cc#L36 "csrc/ops/reductions/sum_aten_kernels.cc#AtenSumIntListOut"))
  and right shift
  ([`AtenBitwiseRightShiftTensorOut`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/binary_aten_kernels.cc#L1262 "csrc/ops/binary_aten_kernels.cc#AtenBitwiseRightShiftTensorOut")).
- **Native `uint32`.** `uint32` is a real device dtype on TPU:
  [`ToElementType`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/dtype.cc#L123 "csrc/common/dtype.cc#ToElementType")
  maps `kUInt16`/`kUInt32`/`kUInt64` to unsigned element types
  ([dtype-and-precision](../concepts/dtype-and-precision.md)). The TPU version of `pack32`
  builds the bit patterns directly as `uint32` on `values.device` and sums with
  `dtype=torch.uint32`. The CPU version detours through `int32` and `int64`.
- **Compiled quantize-and-pack.** The `quant` notebook wraps quantize plus `vmap(pack32)` in
  `torch.compile(backend="tpu")`, which is
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend")
  ([torch-compile-backend](../concepts/torch-compile-backend.md)). It runs on a
  `(1024, 32)` input, forces execution with `.cpu()` on both outputs, and lists the HLO and LLO
  dump directories. For the TPU build it drops the size, dtype and value asserts of the CPU
  version "to avoid compilation errors on TPU". Assertions that read tensor values are
  data-dependent Python control flow inside the compiled region.
- **Popcount from JAX.** The popcount comes from `jax.lax.population_count`, wrapped with
  [`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op")
  as `test_example::popcount`, then cast to `int32` and summed
  ([jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md)). The guide's check packs 7 ones
  and 25 minus-ones with `scale = 42.42`. It compares `quantized_sum(...).item()` with the
  Python sum.

## Why it matters / when it applies
- **Shows how much the native op set covers.** Even custom numeric formats can often be written
  with native ops: bit tricks, unsigned types and `vmap` all lower on TPU. A custom op is needed
  only for an operation with no ATen spelling. Here that is popcount, which the JAX step
  provides.
- **QAT pattern carries over.** The `(w_q - w).detach() + w` pattern is the standard
  straight-through estimator, and it is plain composition, so it needs nothing TPU-specific.
- **Pallas for production.** Real low-bit formats (`fp8`, int4) usually need a dedicated
  kernel; see [pallas-custom-kernels](../concepts/pallas-custom-kernels.md).
- **Dumps verify the lowering.** Both notebooks set `XLA_FLAGS` and `LIBTPU_INIT_ARGS` dump
  flags and turn on `torch._logging.set_logs(aot_graphs=True)`. You can follow the ATen graph,
  the HLO (with the `popcnt` instruction) and the LLO for these few lines
  ([xla-compilation-dumps](xla-compilation-dumps.md)).

## Connections
- Code concepts: [dtype-and-precision](../concepts/dtype-and-precision.md),
  [reductions](../concepts/reductions.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [scan-and-custom-kernels](../concepts/scan-and-custom-kernels.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/ops`, `csrc/ops/reductions`,
  `csrc/common`, `src/torch_tpu/_internal/pallas`).
- Related doc-concepts: [extending-pytorch-on-tpu](extending-pytorch-on-tpu.md),
  [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md),
  [xla-compilation-dumps](xla-compilation-dumps.md),
  [excess-precision-and-out-dtype](excess-precision-and-out-dtype.md).

## Source
Extracted from `docs/markdown/extending_modules_and_functions_via_composition.md` (the format
and `QATLinear`), `docs/markdown/quant.md` (compiled quantize-and-pack) and
`docs/markdown/quantized_sum.md` (the popcount sum), all kept in place.
