---
title: "Sharded Matrices and How to Multiply Them (scaling-book, Part 3)"
type: source
tags: [parallelism, interconnect, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 3 of [*How To Scale Your Model*](2025-scaling-book.md): the named-axis sharding notation
(`A[I_X, J_Y]`, mesh axes, local vs global shape), the four cases of sharded matmul and the
collective each requires, the four TPU communication primitives with ring-based cost
formulas, latency-bound regime, the AllGather/ReduceScatter duality under backprop, and the
collective-matmul overlap idea.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/sharding> — fetched 2026-09-04; chapter last
  changed upstream 2026-07-08 (`3735e35`, quiz-answer fix; 2026-07 figure redraws with X as
  row axis).
- Authority: authorial; cost formulas are first-principles ring models validated by one
  empirical v5e AllGather plot (~95% of claimed link bandwidth at ~10 MB).

## Gaps & caveats

- Costs assume uniform ICI rings; they ignore optical-switch congestion, partial
  wraparounds (see [ici](../concepts/ici.md)) and XLA's actual scheduling.
- ND AllToAll formula is stated with a hand-wavy bisection derivation.

## Residue (not extracted)

Ten worked problems (latency-bound gathers, matmul strategy comparison, memory-limited
sharding of an FFW block). One conclusion lifted: gather-then-matmul vs
matmul-then-reduce-scatter comms ratio is `M/K`, so the reduce path wins when the
contracting dim is small relative to the output.

## Connections

Created: [sharding-notation](../concepts/sharding-notation.md),
[collective-costs](../concepts/collective-costs.md),
[collective-matmul](../concepts/collective-matmul.md).
Updated: [ici](../concepts/ici.md),
[compute-communication-overlap](../concepts/compute-communication-overlap.md).

## Sources

- `raw/docs/2025-scaling-book-sharding.html`
