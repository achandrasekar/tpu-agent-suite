---
title: "PCIe (host ↔ TPU link)"
type: concept
tags: [hardware, interconnect]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

PCIe connects a CPU host to its tray(s) of TPU chips — the path for feeding data,
host offload, and the host leg of every [DCN](dcn.md) transfer. Chips sit 4 to a tray
(2 trays/host on v5e) ([source](../sources/2025-scaling-book-tpus.md)).

## Behavior

- ~1.6e10 B/s per TPU each way (3.2e10 on v6e) — **~100× slower than
  [HBM](hbm.md)** ([source](../sources/2025-scaling-book-tpus.md)).
- Host-RAM offload is therefore possible but slow: streaming weights from host DRAM
  needs batch ≳ 57,500 to stay FLOPs-bound on v6e — effectively never for serving —
  so parameters must live in HBM ([source](../sources/2025-scaling-book-tpus.md)).
- When moving a host-resident sharded array to chips, prefer many PCIe links in
  parallel (each chip loads its shard, then gather over [ICI](ici.md)) over funneling
  through one host's links ([source](../sources/2025-scaling-book-tpus.md)).

## See also

- [Roofline analysis](roofline-analysis.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
