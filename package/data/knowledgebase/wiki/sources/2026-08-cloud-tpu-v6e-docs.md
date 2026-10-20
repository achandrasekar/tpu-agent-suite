---
title: "Google Cloud TPU v6e (Trillium) documentation"
type: source
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: false
---

Official Cloud TPU v6e architecture/configuration page ("Trillium" is v6e on all
technical surfaces). **Authority: official vendor docs.** Snapshot 2026-08-22; page
last updated 2026-08-11. Content fully extracted — specs and configs live on the wiki
pages below.

## Gaps & caveats

- States **1 TensorCore per chip**, contradicting the scaling-book's "v6 is megacore"
  claim — contradiction recorded with `[!warning]` on
  [Megacore](../concepts/megacore.md).

## Connections

Created [TPU v6e](../hardware/tpu-v6e.md); updated
[Megacore](../concepts/megacore.md) (contradiction),
[MXU](../concepts/mxu.md) (2/TensorCore),
[Quantization](../concepts/quantization.md) (int8 2×),
[SparseCore](../concepts/sparsecore.md) (2/chip).

## Sources

- `raw/docs/2026-08-cloud-tpu-v6e-docs.html` — snapshot of
  https://docs.cloud.google.com/tpu/docs/v6e (fetched 2026-08-22)
