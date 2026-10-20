---
title: "Implementation selection and XLA fallback (the `implementation=` argument)"
type: doc-concept
provenance: doc
source: README.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Implementation selection and XLA fallback (the `implementation=` argument)

## Definition
Every public tokamax function takes an `implementation` argument. The README's "Using Tokamax"
section (repeated verbatim in `docs/basic_usage.md`) frames the two modes. With
`implementation=None`, "Tokamax is allowed to select the best implementation for each kernel
shape. It is even allowed to choose different implementations for the forward pass and gradient.
It will also always be supported, as it can fall back to an XLA implementation
`implementation='xla'`". Naming a backend instead (`"mosaic"`, `"triton"`, `"xla_chunked"`, ...)
means "try this kernel and throw an exception if this is unsupported for any reason". The
doc's examples of "unsupported" are FP64 inputs and older GPUs.

## In tokamax (grounded)
- **Dispatch loop.** Each op's `api.py` keeps an `IMPLEMENTATIONS` name -> op map and a
  `_DEFAULT_IMPLEMENTATIONS` tuple. Backends register only if their module imports. The public
  functions, for example [`dot_product_attention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/api.py#L81 "tokamax/_src/ops/attention/api#dot_product_attention") and
  [`ragged_dot_general`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/api.py#L150 "tokamax/_src/ops/ragged_dot/api#ragged_dot_general"), walk the requested names in order. A
  backend that cannot handle the arguments raises `NotImplementedError`, and the loop moves on to
  the next name. With `implementation=None`, the tuple ends in `"xla"`, the op family's reference
  `_fwd`. That is what makes the README's "always supported" promise hold.
- **The `"mosaic"` alias on TPU.** Both `dot_product_attention` and `ragged_dot` resolve
  `"mosaic"` to `mosaic_gpu` when the default device kind contains "NVIDIA", and to `mosaic_tpu`
  otherwise. So the README's H100 example (`implementation="mosaic"` -> Pallas Mosaic GPU)
  selects [`PallasMosaicTpuFlashAttention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L63 "tokamax/_src/ops/attention/pallas_mosaic_tpu#PallasMosaicTpuFlashAttention") when
  it runs on a TPU host. The same goes for
  [`PallasMosaicTpuRaggedDot`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L90 "tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu#PallasMosaicTpuRaggedDot") (GMM v1).
- **Device check.** [`Op.__call__`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L182 "tokamax/_src/ops/op#Op.__call__") binds the arguments, then rejects any
  device for which [`Op.supported_on`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L409 "tokamax/_src/ops/op#Op.supported_on") is false by raising
  `NotImplementedError("Not supported on <device_kind>.")`. This is the same exception the
  dispatch loop catches. TPU backends gate on generation here. For example, the KDA backend
  requires TPU v6 or later. The check is skipped when the op's `bypass_device_check` is set or
  `tokamax.config.cross_compile` is on (see
  [stablehlo-export-and-cross-compilation](stablehlo-export-and-cross-compilation.md)).
- **Forward and gradient can differ.** `Op.__call__` runs `_fwd` under a `jax.custom_vjp`
  and sends the backward pass to the op's own [`Op.vjp`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L174 "tokamax/_src/ops/op#Op.vjp"). That VJP is a
  separate `Op` with its own config and search space. This is why the README can say the
  forward and gradient implementations may be chosen independently, and why its "Evaluate the
  Gradient" example simply calls `jax.jit(jax.grad(loss))`. The gradient goes through the
  kernel's custom VJP, not through autodiff of the Pallas body.

## Why it matters / when it applies
- Use `implementation=None` for portable code: the same model runs on a TPU, a GPU or a CPU,
  and each call picks up the best registered kernel.
- Name an implementation when a silent XLA fallback would hide a problem, as in kernel
  benchmarks, perf regressions, or when you rely on a TPU kernel's memory behaviour. The
  linear softmax cross-entropy op is an example: its `"xla"` fallback materializes the full
  logits.
- On TPU, `layer_norm` and `gated_linear_unit` have no Mosaic TPU backend, so
  `implementation=None` lands on `"xla"`. The README example's `implementation="triton"` raises
  on a TPU.

## Connections
- Code concepts: [kernel-catalog](../concepts/kernel-catalog.md) (the per-op default order and
  what runs on TPU), [op-framework](../concepts/op-framework.md) (`Op.__call__`, `bind`, custom
  VJP), [dot-product-attention](../concepts/dot-product-attention.md),
  [ragged-dot-gmm](../concepts/ragged-dot-gmm.md),
  [linear-softmax-cross-entropy](../concepts/linear-softmax-cross-entropy.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [supported-ops-and-hardware](supported-ops-and-hardware.md),
  [autotuning-search-space](autotuning-search-space.md),
  [stablehlo-export-and-cross-compilation](stablehlo-export-and-cross-compilation.md).

## Source
Extracted from `README.md` ("Using Tokamax", "Evaluate the Gradient") and `docs/basic_usage.md`
(same text), kept in place.
