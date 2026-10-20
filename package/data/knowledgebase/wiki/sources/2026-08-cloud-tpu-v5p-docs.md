---
title: "Google Cloud TPU v5p documentation"
type: source
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: false
---

Official Cloud TPU v5p architecture/configuration page. **Authority: official vendor
docs.** Snapshot 2026-08-22; page last updated 2026-08-11. Content fully extracted —
specs, cube/wraparound rules, twisted torus, and ICI resiliency live on the wiki pages
below.

## Connections

Created [TPU v5p](../hardware/tpu-v5p.md),
[SparseCore](../concepts/sparsecore.md) (stub); updated
[ICI](../concepts/ici.md) (twisted torus, resiliency),
[Quantization](../concepts/quantization.md) (FP8 = 1× — no speedup),
[MXU](../concepts/mxu.md) (4/TensorCore).

## Sources

- `raw/docs/2026-08-cloud-tpu-v5p-docs.html` — snapshot of
  https://docs.cloud.google.com/tpu/docs/v5p (fetched 2026-08-22)
