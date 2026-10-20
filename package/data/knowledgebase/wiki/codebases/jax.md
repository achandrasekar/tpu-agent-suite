---
title: "JAX (parallelism APIs)"
type: codebase
tags: [framework, parallelism, compiler]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["jax.jit", "shard_map", "Shardy", "GSPMD"]
---

JAX is the NumPy-style array language whose `jax.jit` traces to StableHLO for XLA, and
whose sharding APIs are the reference implementation of the
[sharding notation](../concepts/sharding-notation.md) used throughout this wiki. Official
docs: <https://docs.jax.dev/> (not yet ingested); the *Distributed arrays and automatic
parallelization* guide is the book's recommended next read
([hub](../sources/2025-scaling-book.md)). Everything below is as of the book's mid-2026
snapshot and may drift.

## Three parallelism modes

| Mode | View | Who propagates sharding | Collectives | API |
|---|---|---|---|---|
| Auto | global | XLA's Shardy partitioner; inserts AllGather/ReduceScatter/AllReduce as needed | implicit | `jax.jit` with `AxisType.Auto` mesh axes; steer with `jax.lax.with_sharding_constraint` |
| Explicit | global | JAX, at trace time — sharding is part of the type (`jax.typeof(x)` → `float32[8@X,2@Y]`); ambiguous ops raise and demand `out_sharding=` | implicit | `jax.jit` with `AxisType.Explicit`; compose via `jax.sharding.auto_axes` / `explicit_axes` |
| Manual | per-device | you | explicit: `jax.lax.all_gather`, `psum` (AllReduce), `psum_scatter` (ReduceScatter), `all_to_all`, `ppermute`, `pmean` | `jax.shard_map(f, in_specs=, out_specs=)` |

([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md).) "Compiler tickling" with
`with_sharding_constraint` is ~60% of Auto-mode parallel programming; Explicit mode exists
because that is not a good programming model, and it errors on the ambiguous
contracting-dim case (`ReduceScatter` vs `AllReduce` output) instead of guessing
([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).

## Vocabulary

- `jax.make_mesh(axis_shapes, axis_names, axis_types)` + `jax.set_mesh(mesh)`; mesh axes
  need not be hardware axes ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- `jax.NamedSharding(mesh, jax.P('X', 'Y'))`; `P(None, 'Y')` = replicate first axis;
  `P(('x','y'))` = one array axis over two mesh axes. Create sharded arrays with
  `device=` / `out_sharding=` at construction, `jax.device_put(x, P(...))`, or a jitted
  producer with `out_shardings=` ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- `x.shape` is global; `x.addressable_shards` is local
  ([scaling-book, Part 3](../sources/2025-scaling-book-sharding.md)).
- Inside `shard_map`: `jax.lax.axis_index`, `axis_size`, `dynamic_slice_in_dim`,
  `fori_loop`, `pcast(..., to='varying')` — the pieces of a
  [collective matmul](../concepts/collective-matmul.md)
  ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- `jax.lax.ragged_dot` for sorted MoE tokens without padding
  ([expert-parallelism](../concepts/expert-parallelism.md)).
- `jax.checkpoint` / `jax.remat` for [gradient checkpointing](../concepts/gradient-checkpointing.md).
- `jax.profiler.trace(dir)` to capture a profile; `jax.jit(f).lower(...).compile().as_text()`
  for [HLO](../concepts/hlo.md) ([scaling-book, Part 9](../sources/2025-scaling-book-profiling.md)).
- CPU emulation of a mesh: `jax.config.update("jax_num_cpu_devices", 8)` (semantics only,
  not performance) ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).

## Pitfalls the book names

- Auto mode can insert a giant unnecessary AllGather; look for it in the profile and
  constrain the intermediate ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- A `jax.jit` MoE gathers all activations locally unless written with `shard_map`
  ([scaling-book, Part 10](../sources/2025-scaling-book-jax-stuff.md)).
- Sharding-mode defaults were changing upstream in 2026 (`f737080`); check the version.

## See also

- [Sharding notation](../concepts/sharding-notation.md), [HLO](../concepts/hlo.md),
  [xprof](xprof.md), [Pallas](pallas.md), [TorchTPU](torch-tpu.md) (the PyTorch path — mapping not yet ingested)

## Sources

- [How To Scale Your Model, Part 10: Programming TPUs in JAX (scaling-book)](../sources/2025-scaling-book-jax-stuff.md)
- [How To Scale Your Model, Part 3: Sharded Matrices (scaling-book)](../sources/2025-scaling-book-sharding.md)
- [How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book)](../sources/2025-scaling-book-profiling.md)
