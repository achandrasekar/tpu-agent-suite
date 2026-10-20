---
title: "StableHLO export, serialization guarantees and CPU cross-compilation"
type: doc-concept
provenance: doc
source: README.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# StableHLO export, serialization guarantees and CPU cross-compilation

## Definition
The README's "Serialization" section says tokamax kernels "can be serialized to StableHLO".
Kernel calls are JAX custom calls, "which are by default banned in `jax.export`", so the
export needs `disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS`. The price is portability:
"functions serialized with Tokamax kernels lose the device-independence of standard StableHLO".
Tokamax makes two guarantees in return. A function serialized for a specific device is
guaranteed to run on exactly that device. And tokamax gives the same compatibility guarantee
as JAX custom calls, six months of backward compatibility. The "Cross-Compilation and CPU
Export" subsection adds that kernels can be exported or cross-compiled on a CPU host. There
are two ways to do it: name an implementation explicitly (e.g. `implementation="mosaic_tpu_v2"`),
which "automatically bypasses host device validation", or wrap the export in
`tokamax.config.cross_compile(True)`.

## In tokamax (grounded)
- [`DISABLE_JAX_EXPORT_CHECKS`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/hlo_utils.py#L32 "tokamax/_src/hlo_utils#DISABLE_JAX_EXPORT_CHECKS") is a tuple of
  `export.DisabledSafetyCheck.custom_call(...)` entries. At the pin it names only the Pallas
  Triton and Triton FFI custom-call targets, which are the GPU kernels. It names no Mosaic TPU
  target. The tuple is still the documented argument to pass for every export.
- **The device check that blocks CPU export.** [`Op.__call__`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L182 "tokamax/_src/ops/op#Op.__call__") raises
  `NotImplementedError` when [`Op.supported_on`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L409 "tokamax/_src/ops/op#Op.supported_on") is false for the input
  devices, or for the default device if no input is concrete. On a CPU host every TPU backend
  fails that check. The check is skipped when the op instance has `bypass_device_check=True`
  or when the [`cross_compile`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/config.py#L95 "tokamax/_src/config#cross_compile") option is on. `cross_compile` is a
  context-manager config option backed by the `--tokamax_cross_compile` flag. It also disables
  the GPU-side `has_triton_support()` / `has_mosaic_gpu_support()` probes.
- **Explicit implementation -> bypass.** [`ragged_dot_general`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/api.py#L150 "tokamax/_src/ops/ragged_dot/api#ragged_dot_general")
  has a `bypass_device_check` parameter. When it is `None` (the default), a single explicit
  implementation (a string, or a one-element sequence) sets it to `True`. Auto-selection and
  multi-name fallbacks keep the check. The code comment reads "a single explicit implementation
  bypasses them for CPU export".
- Every exported tokamax call keeps its op spec in the HLO, so an exported module can still be
  scanned with [`get_bound_args`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/hlo_utils.py#L103 "tokamax/_src/hlo_utils#get_bound_args") for autotuning (see
  [autotuner-input-formats](autotuner-input-formats.md)).

> [!warning] Scope of the explicit-implementation bypass
> The README describes the bypass as general. At the pin, the automatic
> `bypass_device_check` logic lives only in the ragged-dot API
> ([`ragged_dot_general`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/ragged_dot/api.py#L150 "tokamax/_src/ops/ragged_dot/api#ragged_dot_general")). For other ops (attention, LSCE,
> KDA, ...), CPU-host export of a TPU kernel needs `tokamax.config.cross_compile(True)`.

## Why it matters / when it applies
- Typical TPU use: lower and export a training or serving step on a CPU-only build host, then
  ship the StableHLO to TPU workers. Without the bypass, `implementation=None` dispatch on a CPU
  host skips every TPU kernel and silently exports the XLA fallback. With `cross_compile(True)`,
  or a single explicit implementation, the Mosaic TPU kernel itself gets serialized.
- Because the artifact is pinned to one device, export once per TPU generation you deploy to.
  Re-export when you upgrade tokamax or JAX beyond the six-month compatibility window.

## Connections
- Code concepts: [benchmarking-and-serialization](../concepts/benchmarking-and-serialization.md)
  (op specs in HLO, export story), [op-framework](../concepts/op-framework.md) (device check,
  `bypass_device_check`), [ragged-dot-gmm](../concepts/ragged-dot-gmm.md) (the
  `mosaic_tpu_v2` example), [gmm-v2](../concepts/gmm-v2.md),
  [mosaic-tpu-helpers](../concepts/mosaic-tpu-helpers.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [implementation-selection-and-fallback](implementation-selection-and-fallback.md),
  [autotuner-input-formats](autotuner-input-formats.md).

## Source
Extracted from `README.md` ("Serialization", "Cross-Compilation and CPU Export") and
`docs/basic_usage.md` ("Serialization"), kept in place.
