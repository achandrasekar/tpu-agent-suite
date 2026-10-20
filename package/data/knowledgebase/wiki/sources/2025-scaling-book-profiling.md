---
title: "How to Profile TPU Programs (scaling-book, Part 9)"
type: source
tags: [performance, compiler]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 9 of [*How To Scale Your Model*](2025-scaling-book.md): the TPU software stack (JAX →
StableHLO → HLO → LLO → machine code), the JAX profiler / TensorBoard (xprof) Trace Viewer,
Graph Viewer and Memory Profile, how to read an HLO op line (shape, layout/tiling, memory
space `S(0)`/`S(1)`), and a worked profile where a sharded FFW matmul and a ReduceScatter
match roofline predictions within a few percent.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/profiling> — fetched 2026-09-04; chapter last
  changed upstream 2026-07-12 (`3980377`, errata).
- Authority: authorial; the worked profile ran on TPU v2-8 (no longer offered on Colab —
  the book points to Kaggle or an 8-core GCP slice).

## Gaps & caveats

- Tool-level walkthrough only; no coverage of xprof's op-profile, roofline, or memory
  viewer tables, nor of PyTorch/XLA capture. The [xprof](../codebases/xprof.md) page marks
  the repo as a fetch target.
- Layout `AUTO` for `jax.jit` inputs is mentioned as experimental.

## Residue (not extracted)

Two worked problems (diagnose a suspicious profile; fix GSPMD sharding of a mock
Transformer — 184 → 67 ms/layer originally, ~90 → 80 ms/layer with newer XLA).

## Connections

Created: [hlo](../concepts/hlo.md), [xprof](../codebases/xprof.md),
[profile-tpu-program](../playbooks/profile-tpu-program.md).
Updated: [vmem](../concepts/vmem.md) (memory-space notation),
[padding](../concepts/padding.md) (see-also), [jax](../codebases/jax.md).

## Sources

- `raw/docs/2025-scaling-book-profiling.html`
