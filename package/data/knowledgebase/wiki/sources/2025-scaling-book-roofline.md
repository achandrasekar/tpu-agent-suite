---
title: "All About Rooflines (scaling-book, Part 1)"
type: source
tags: [performance, memory, interconnect]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 1 of [*How To Scale Your Model*](2025-scaling-book.md): the three constraints on any
algorithm (compute, bandwidth, capacity), the lower/upper time bounds, arithmetic intensity,
the matmul roofline and the ~240-token rule on v5e, and the first network roofline for a
matmul sharded across two chips.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/roofline> — fetched 2026-09-04; chapter last
  changed upstream 2026-08-20 (`cf7e74d`, text updates).
- Authority: authorial (Google DeepMind); the model is standard, the constants are
  book-reported v5e/H100 figures.

## Gaps & caveats

- The 240 figure is the v5e MXU-from-HBM case only; VMEM-resident weights and VPU work
  have different critical intensities (the book says so in footnotes). Per-generation
  values live on [critical-batch-size](../concepts/critical-batch-size.md).
- Tile-size effects on intensity are a footnote only (`bm*bn/(bm+bn)`).

## Residue (not extracted)

Five worked problems (int8 matmul, int8-weight/bf16-FLOPs, roofline plot code, batched
per-example weights, H100 memory roofline). Their conclusions that matter were lifted
(int8 weights halve the critical batch; per-example weight matmuls have intensity ~2 and
are always comms-bound).

## Connections

Updated: [roofline-analysis](../concepts/roofline-analysis.md) (formal model, network
rooflines), [quantization](../concepts/quantization.md), [ici](../concepts/ici.md).
Created: [critical-batch-size](../concepts/critical-batch-size.md).

## Sources

- `raw/docs/2025-scaling-book-roofline.html`
