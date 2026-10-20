---
title: "Google blog: eighth-generation TPUs — TPU 8t and TPU 8i (announcement)"
type: source
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: false
---

Google blog announcement (Amin Vahdat, Google Cloud Next, April 2026) of the
8th-generation TPUs: TPU 8t (training) and TPU 8i (inference) — the first generation
split into two purpose-built chips. **Authority: vendor announcement — marketing-grade
claims, not a datasheet.** Snapshot 2026-08-22. Claims extracted (with warnings) to
the hardware pages below.

## Gaps & caveats

- GA "later this year" (2026); no Cloud docs spec sheet exists yet.
- No FLOPs precision stated (121 EF/pod for 8t), no HBM bandwidth, no latency numbers,
  no per-chip ICI for 8t; all comparatives ("3×", "80%") lack baselines/conditions.
- **Supersession watch**: when the Cloud docs pages for 8t/8i appear, ingest them and
  re-derive both hardware pages from datasheet numbers.

## Connections

Created [TPU 8t](../hardware/tpu-8t.md), [TPU 8i](../hardware/tpu-8i.md); updated
[ICI](../concepts/ici.md)-adjacent context (8i's Boardfly departs from pure torus —
noted on the 8i page).

## Sources

- `raw/docs/2026-google-blog-tpu-8th-gen.html` — snapshot of
  https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/
  (fetched 2026-08-22)
