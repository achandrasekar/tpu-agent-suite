---
title: "Autotuner input formats: callables, BoundArguments and XProf profiles"
type: doc-concept
provenance: doc
source: docs/autotuning.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# Autotuner input formats: callables, BoundArguments and XProf profiles

## Definition
`tokamax.autotune` accepts three descriptions of "the inputs of interest". In every case it
returns an `AutotuningResult` with the best config for each op. The three forms in
`docs/autotuning.md` are:
1. **Callable function.** The autotuner "uses StableHLO metadata to extract information about
   every Op and its respective inputs". The function can mix tokamax and non-tokamax ops.
   Tokamax VJP ops have no direct API to call, so to tune them you pass the pullback:
   `tokamax.autotune(jax.grad(f))`.
2. **A sequence of `BoundArguments`.** Calling an op first "binds" it to its arguments, which
   "canonicalizes and validates the inputs, creating a `BoundArgument` object consisting of the
   op with its input arguments". A list of these can be tuned directly.
3. **A serialized list of `BoundArguments`.** The `xplane_to_bound_args.py` script reads an
   XProf xplane proto captured from a real profiling run and writes a JSON file of
   `BoundArguments`. That file can be loaded back and tuned, which covers every tokamax op that
   appeared in the profile.

## In tokamax (grounded)
- [`autotune`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L309 "tokamax/_src/autotuning/api#autotune") takes `f` as a callable, a list of
  `BoundArguments`, or a `jax.stages.Lowered`, plus `*args` for the callable case. For
  callables it calls [`get_bound_args`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/hlo_utils.py#L103 "tokamax/_src/hlo_utils#get_bound_args"), which lowers `f` and reads back
  the op spec JSON that every tokamax call writes into the HLO. Ops whose cache entry already
  exists are dropped unless `ignore_cache=True`. `all_implementations=True` expands each op to
  every implementation registered for the current device via
  [`get_op_implementations`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L287 "tokamax/_src/autotuning/api#get_op_implementations").
- [`Op.bind`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L317 "tokamax/_src/ops/op#Op.bind") produces the [`BoundArguments`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/op.py#L437 "tokamax/_src/ops/op#BoundArguments"): the op
  instance plus canonical abstract arguments. That value is hashable and JSON-serializable.
- The profile path: the CLI [`main`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/tools/xplane_to_bound_args.py#L73 "tokamax/_src/tools/xplane_to_bound_args#main") (flags
  `--xplane_file`, `--output_file`) pulls the HLO protos out of the `/host:metadata` plane
  ([`get_hlo_protos_from_xspace`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/tools/xplane_to_bound_args.py#L48 "tokamax/_src/tools/xplane_to_bound_args#get_hlo_protos_from_xspace")). It strips the
  config that was baked into each op spec and de-duplicates by op class and cache key. On the
  load side, [`bound_args_from_json`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L248 "tokamax/_src/autotuning/api#bound_args_from_json") parses a JSON *string* and
  [`bound_args_from_json_file`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L254 "tokamax/_src/autotuning/api#bound_args_from_json_file") reads a file. The reverse direction
  is [`bound_args_to_json`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L234 "tokamax/_src/autotuning/api#bound_args_to_json").

> [!warning] Doc examples that do not match the code at the pin
> - `tokamax.autotune(f, **args)`: [`autotune`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L309 "tokamax/_src/autotuning/api#autotune") forwards only
>   positional `*args`. Its own docstring says that to tune a callable with keyword arguments
>   you should first compute `get_bound_args(f, *args, **kwargs)` and pass the result.
> - `tokamax.autotuning.bound_args_from_json("/tmp/bound_args.json")`:
>   [`bound_args_from_json`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/autotuning/api.py#L248 "tokamax/_src/autotuning/api#bound_args_from_json") calls `json.loads` on its argument, so
>   a path fails. Use `tokamax.autotuning.bound_args_from_json_file(path)`.
> - `tokamax.PallasMosaicTpuRaggedDot.bind(...)` / `tokamax.PallasMosaicTpuAttention.bind(...)`:
>   neither name is exported from the top-level `tokamax` package. The TPU attention backend is
>   [`PallasMosaicTpuFlashAttention`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L63 "tokamax/_src/ops/attention/pallas_mosaic_tpu#PallasMosaicTpuFlashAttention"), and `bind`
>   is an instance method, so you need an instance such as
>   `pallas_mosaic_tpu.PallasMosaicTpuRaggedDot().bind(...)`.

## Why it matters / when it applies
- The callable form is the simplest way to tune a model function where it is defined. Pass
  `jax.grad(f)` so that the backward kernels, which are separate VJP ops with their own search
  spaces, get tuned too.
- The `BoundArguments` form suits kernel developers and microbenchmarks: you tune exact shapes
  without building a model.
- The xplane form is the production workflow on TPU. You profile the real training or serving
  job once with XProf, extract every tokamax op and shape it actually ran, and tune them offline.
  No model code needs to run in the tuning job.

## Connections
- Code concepts: [autotuning](../concepts/autotuning.md) (`get_bound_args`, the per-op loop),
  [op-framework](../concepts/op-framework.md) (`bind`, `BoundArguments`, op specs in HLO),
  [benchmarking-and-serialization](../concepts/benchmarking-and-serialization.md) (reading ops
  back from HLO).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [autotuning-search-space](autotuning-search-space.md),
  [autotuning-result-reuse](autotuning-result-reuse.md),
  [stablehlo-export-and-cross-compilation](stablehlo-export-and-cross-compilation.md).

## Source
Extracted from `docs/autotuning.md` ("Autotuner Input Formats"), kept in place.
