---
title: "Kernel class: cross-chip / collective (ring attention, in-kernel collectives)"
type: concept
tags: [tpu-kernel-development, kernel-class, collective, ici, rdma, pallas, briefs-tier]
created: 2026-07-22
updated: 2026-07-22
sources: 0
human_reviewed: false
---

**Route here when** the op spans chips: ring attention, in-kernel all-gather/reduce-scatter, sharded contractions. (No local benchmark op-points yet — this page is a stub carrying the transferable strategy; extend when the first family lands.)

- **Strategy**: chunk the contraction/sequence axis and overlap — compute chunk i while chunk i−1's collective is in flight over ICI (the collective-matmul pattern); implement with the RDMA/semaphore primitives ([pallas-tpu-distributed](pallas-core-mental-model.md)).
- **Plan against the topology, not the marketing number**: ICI is nearest-neighbor (distant chips hop), wraparound exists only on full-cube slices, stated bandwidth is bidirectional — a one-direction pattern gets half (ici-roofline (`index.py get concept-ici`)).
- Overlap wins only when collective latency fits inside the adjacent compute window — verify with the profile, don't assume.
