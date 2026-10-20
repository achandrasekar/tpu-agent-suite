---
title: "HLO and the TPU compilation stack"
type: concept
tags: [compiler, performance]
created: 2026-09-04
updated: 2026-09-04
sources: 2
human_reviewed: true
aliases: ["StableHLO", "LLO", "XLA IR", "HLO layout", "tiling notation"]
---

HLO is XLA's graph IR: every matmul, fusion, collective and copy the TPU executes is an
HLO op, and its one-line text form encodes shape, dtype, physical layout/tiling and memory
space. Reading it is how a profile is tied back to the roofline
(scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## The stack

`jax.jit` traces Python → **StableHLO** (platform-agnostic) → XLA lowers to **HLO** and runs
its passes (fusion, layout assignment, sharding propagation) → **LLO** (low-level
optimizer) schedules copies between memories, pushes buffers into the systolic array and
issues DMAs → machine code loaded into TPU IMEM. Performance work happens at the JAX
level with HLO semantics in mind; when that fails, the escape hatch is a Pallas kernel
(scaling-book, Part 9 (`index.py get scaling-book-profiling`)). Dump HLO with
`jax.jit(f).lower(*args).compile().as_text()`
(scaling-book, Part 9 (`index.py get scaling-book-profiling`)); the torch_tpu path to the same dump is
*not yet ingested*.

## Anatomy of an op line

```
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{...} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

| Piece | Meaning |
|---|---|
| `fusion.3` | op name; a `dot`/`fusion` holds at most one matmul plus related pointwise VPU ops |
| `bf16[32,32,4096]` | output dtype and logical shape |
| `{2,1,0:T(8,128)(2,1)}` | layout: minor-to-major axis order, then tiling — outer `(8,128)` tiles, inner `(2,1)` so bf16 loads are 4-byte multiples |
| `S(1)` | memory space: `S(1)` = VMEM (`index.py get concept-vmem`), `S(0)` (often omitted) = HBM, `S(2)`/`S(3)` other |
| arguments | producers of this op's inputs (`%fusion.32`) |
| `kind=kCustom, calls=…` | fusion kind and the fused computation (here a reduce-scatter) |

(scaling-book, Part 9 (`index.py get scaling-book-profiling`).) Tiling pads arrays: `f32[3,5]` with
`T(2,2)` occupies `[4,6]` (~1.6× bytes) — the same rule as MXU padding (`index.py get padding`).
Layout mismatches make XLA insert "retile" copies at non-trivial cost; `jax.jit` can be
asked for its preferred input layout (`AUTO`) so inputs arrive pre-tiled
(scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## Reading sharding from HLO

Local shapes reveal the sharding: a `[2,1024]` operand for a global `[8,2048]` array means
4-way batch × 2-way model sharding; a collective's `replica_groups={{0,16,32,…}}` reveals the
group size and hence the model-parallel degree
(scaling-book, Part 10 (`index.py get scaling-book-jax`),
scaling-book, Part 9 (`index.py get scaling-book-profiling`)). Compound ops like
`all-reduce-scatter` are easiest to parse in xprof's Graph Viewer
(xprof (`index.py get codebase-xprof`)).

## See also

- Profile a TPU program (`index.py get playbook-profile-tpu-program`), Padding (`index.py get padding`),
  VMEM (`index.py get concept-vmem`), jax (`index.py get codebase-jax`), Pallas (`index.py get codebase-pallas`)

## Sources

- How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book) (`index.py get scaling-book-profiling`)
- How To Scale Your Model, Part 10: Programming TPUs in JAX (scaling-book) (`index.py get scaling-book-jax`)
