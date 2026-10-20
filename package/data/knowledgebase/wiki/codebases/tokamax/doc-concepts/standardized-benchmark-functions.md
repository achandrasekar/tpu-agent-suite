---
title: "standardize_function: benchmark-ready functions, data distribution and benchmark modes"
type: doc-concept
provenance: doc
source: docs/benchmarking.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# standardize_function: benchmark-ready functions, data distribution and benchmark modes

## Definition
`docs/benchmarking.md` describes `standardize_function` as the step that "simplifies
complicated functions, with non-array arguments for example". It rewrites `f(*args, **kwargs)`
into a standard form with one argument, a list of arrays. It randomly initializes every abstract
tensor (`jax.ShapeDtypeStruct`) and returns `f_std(args)` with only concrete array arguments,
so the result "can be cleanly jitted without worrying about static arguments such as strings".
The doc gives two further reasons for it. **Data distribution**: performance "can vary
significantly based on data distributions, due to complex hardware-level power and thermal
interactions", so inputs are initialized "in a manner representative of actual training jobs".
**Benchmark mode**: `mode=forward`, `forward_res` (forward plus residuals), `vjp` (VJP only) or
`forward_and_vjp`. The doc warns that VJP-only "could result in OOMs, because the forward pass
is computed outside the returned standardized function, with all intermediates baked into the
HLO which remains resident in HBM memory".

## In tokamax (grounded)
- [`standardize_function`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L379 "tokamax/_src/benchmarking#standardize_function") binds `f`'s signature and splits the
  flattened arguments into arrays and everything else. It closes over the non-array leaves
  (strings, ints, configs) in the inner [`standardize_function.forward`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L445 "tokamax/_src/benchmarking#standardize_function.forward").
  Abstract leaves are filled by [`random_initialize`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/numerics.py#L178 "tokamax/_src/numerics#random_initialize") using the given
  `seed` (default 0). With `seed=None` nothing is initialized; `BatchedShapeDtype` placeholders
  are only converted, so the function stays idempotent.
- Modes: `'forward'` returns `forward`, and `'forward_res'` returns `jax.vjp(forward, ...)[0]`.
  `'forward_and_vjp'` ([`standardize_function.vjp_full`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L481 "tokamax/_src/benchmarking#standardize_function.vjp_full")) runs the
  forward behind [`_optimization_barrier`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L66 "tokamax/_src/benchmarking#_optimization_barrier") and then applies the
  pullback to the output. `'vjp'` runs `jax.vjp` eagerly at standardization time and returns
  only the pullback, which is why its residuals become HLO constants.
- `BatchedShapeDtype` arguments (all or none of them) wrap `forward` in the matching
  `jax.vmap`s, so vmapped kernels can be benchmarked as they are.
- The output `(f_std, args)` is what [`benchmark`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L654 "tokamax/_src/benchmarking#benchmark") and
  [`compile_benchmark`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/benchmarking.py#L565 "tokamax/_src/benchmarking#compile_benchmark") expect. The autotuner standardizes each
  candidate the same way before timing it.

## Why it matters / when it applies
- Tokamax ops take many static keyword arguments (implementation names, masks, configs).
  Standardizing turns them into closure constants, so the timed program contains only the
  kernel and its array inputs.
- The prior work the doc cites shows that power and clock behaviour depend on the input data.
  Benchmarking all-zero or constant inputs can therefore overstate throughput, which is why
  random initialization is the default.
- To time a custom VJP kernel (for example a Mosaic TPU backward), prefer `'forward_and_vjp'`
  and subtract a `'forward'` run when HBM is tight. Use `'vjp'` only when the baked-in
  residuals fit in HBM.

## Connections
- Code concepts: [benchmarking-and-serialization](../concepts/benchmarking-and-serialization.md)
  (`standardize_function` and the runner pipeline), [autotuning](../concepts/autotuning.md),
  [op-framework](../concepts/op-framework.md) (custom VJP, residuals).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [benchmark-timing-methods](benchmark-timing-methods.md),
  [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md).

## Source
Extracted from `docs/benchmarking.md` ("Benchmarking Basics", "Data Distribution", "Benchmarking
Mode"), `README.md` and `docs/basic_usage.md` ("Benchmarking"), kept in place.
