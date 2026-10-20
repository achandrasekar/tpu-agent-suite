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
host offload, and the host leg of every DCN (`index.py get concept-dcn`) transfer. Chips sit 4 to a tray
(2 trays/host on v5e) (source (`index.py get scaling-book-tpus`)).

## Behavior

- ~1.6e10 B/s per TPU each way (3.2e10 on v6e) — **~100× slower than
  HBM (`index.py get concept-hbm`)** (source (`index.py get scaling-book-tpus`)).
- Host-RAM offload is therefore possible but slow: streaming weights from host DRAM
  needs batch ≳ 57,500 to stay FLOPs-bound on v6e — effectively never for serving —
  so parameters must live in HBM (source (`index.py get scaling-book-tpus`)).
- When moving a host-resident sharded array to chips, prefer many PCIe links in
  parallel (each chip loads its shard, then gather over ICI (`index.py get concept-ici`)) over funneling
  through one host's links (source (`index.py get scaling-book-tpus`)).

## See also

- Roofline analysis (`index.py get concept-roofline-analysis`)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
