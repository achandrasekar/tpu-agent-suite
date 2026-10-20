---
title: "Google Cloud TPU v4 documentation"
type: source
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: false
---

Official Cloud TPU v4 architecture/configuration page. **Authority: official vendor
docs.** Snapshot 2026-08-22; page last updated 2026-08-11. Content fully extracted —
specs and topology rules live on the wiki pages below.

## Gaps & caveats

- v4 is legacy tier: Cloud TPU API in maintenance mode, GKE recommended — configs may
  stop evolving.

## Connections

Created [TPU v4](../hardware/tpu-v4.md); updated [ICI](../concepts/ici.md) (twisted
torus, 6 links/chip), [Megacore](../concepts/megacore.md) (unified-HBM datapoint),
[Quantization](../concepts/quantization.md) (int8 = 1×),
[MXU](../concepts/mxu.md) (4/TensorCore).

## Sources

- `raw/docs/2026-08-cloud-tpu-v4-docs.html` — snapshot of
  https://docs.cloud.google.com/tpu/docs/v4 (fetched 2026-08-22)
