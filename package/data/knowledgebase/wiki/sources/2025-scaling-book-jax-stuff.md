---
title: "Programming TPUs in JAX (scaling-book, Part 10)"
type: source
tags: [framework, parallelism, compiler]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 10 of [*How To Scale Your Model*](2025-scaling-book.md): JAX's three parallelism modes —
Auto (`jax.jit` + Shardy), Explicit (sharding in types, trace-time ambiguity errors) and
Manual (`jax.shard_map` with explicit collectives) — plus a worked collective matmul that
overlaps a `ppermute` ring with chunked matmuls (311 µs → 244 µs on a v5e-8, vs 224 µs
unsharded).

## Identity & authority

- <https://jax-ml.github.io/scaling-book/jax-stuff> — fetched 2026-09-04; chapter last
  changed upstream 2026-09-03 (`773f2db`, worked-problem hints); 2026-07-05 update of
  sharding-mode code for the upcoming JAX default change (`f737080`).
- Authority: authorial; APIs shown (`jax.make_mesh`, `AxisType`, `jax.P`,
  `jax.set_mesh`, `jax.typeof`, `out_sharding=`) track a JAX version around mid-2026 and
  will drift — verify against the [jax](../codebases/jax.md) canonical docs before use.

## Gaps & caveats

- No torch_tpu / PyTorch equivalents; nothing here should be read as a TorchTPU API.
- Worked problems (MoE with ragged `all_to_all`, ReduceScatter collective matmul,
  bidirectional variants) are sketched, not fully solved.

## Residue (not extracted)

Four worked problems; the local MoE `scan`-over-experts snippet and the chunked
`all_to_all` pseudocode are summarized on
[expert-parallelism](../concepts/expert-parallelism.md).

## Connections

Created: [jax](../codebases/jax.md) (codebase page).
Updated: [collective-matmul](../concepts/collective-matmul.md),
[sharding-notation](../concepts/sharding-notation.md) (code mapping),
[expert-parallelism](../concepts/expert-parallelism.md).

## Sources

- `raw/docs/2025-scaling-book-jax-stuff.html`
