---
title: "Roofline analysis"
type: concept
tags: [performance]
created: 2026-08-22
updated: 2026-09-04
sources: 2
human_reviewed: true
---

Roofline analysis predicts whether a workload is compute-bound or bandwidth-bound:
time ≈ max(FLOPs / peak FLOPs/s, bytes moved / bandwidth), assuming compute and
transfers overlap (`index.py get concept-compute-communication-overlap`). The first tool to reach for in
any TPU performance question (source (`index.py get scaling-book-tpus`)).

## The model

- `T_math = FLOPs / peak FLOPs/s`; `T_comms = bytes / bandwidth` for whichever link the
  bytes cross (HBM, ICI, PCIe, DCN). With overlap, `T_lower = max(T_math, T_comms)`;
  without, `T_upper = T_math + T_comms`. The two differ by at most 2×, so optimizing
  against the max is the practical target, refined by profiling "overlap regions"
  (scaling-book, Part 1 (`index.py get scaling-book-roofline`)).
- **Arithmetic intensity** = FLOPs / bytes. Compute-bound iff intensity exceeds the
  hardware's own ratio, peak FLOPs/s ÷ bandwidth — ~240 FLOPs/byte for the v5e MXU fed
  from HBM (scaling-book, Part 1 (`index.py get scaling-book-roofline`)). A dot product has
  intensity ~½ and is always bandwidth-bound; a `[B,D]·[D,F]` matmul has intensity ≈ `B`
  when `B ≪ D, F`, hence the per-replica **token** batch rule on
  critical-batch-size (`index.py get concept-critical-batch-size`).
- Roofline plot: achievable FLOPs/s rises linearly with intensity until the hardware's
  critical intensity, then is flat; raising intensity (bigger batch, VMEM-resident
  weights, fusion) or bandwidth are the two moves
  (scaling-book, Part 1 (`index.py get scaling-book-roofline`)).
- **Network rooflines** use the same math with ICI in the denominator: a matmul split
  over two chips on the contracting axis is compute-bound when `D/2 > C/W_ici`, i.e.
  `D > 8,755` on v5e — the threshold depends on `D`, not `B`
  (scaling-book, Part 1 (`index.py get scaling-book-roofline`)). This is the form every
  parallelism (`index.py get concept-parallelism-strategies`) threshold takes.
- Tiling caveat: a large matmul is decomposed into `bm×bn×bk` tiles that fit VMEM, so
  the effective intensity is `bm·bn/(bm+bn)`
  (scaling-book, Part 1 (`index.py get scaling-book-roofline`)).

## The TPU bandwidth hierarchy

Each channel has its own roofline; communication across each must stay proportional to
its speed (source (`index.py get scaling-book-tpus`)):

| Channel | Speed class |
|---|---|
| VMEM (`index.py get concept-vmem`) ↔ MXU | ~22× HBM |
| HBM (`index.py get concept-hbm`) ↔ TensorCore | 1–2 TB/s |
| ICI (`index.py get concept-ici`) per link | ~4.5–9e10 B/s |
| PCIe (`index.py get concept-pcie`) per TPU | ~1.6e10 B/s |
| DCN (`index.py get concept-dcn`) per TPU | ~6.25e9 B/s |

## Worked patterns (from the source)

- **Decode latency floor**: every step loads all params from HBM → params ÷ aggregate
  HBM BW lower-bounds step time (200B bf16 on 32 v4p ≈ 10 ms)
  (source (`index.py get scaling-book-tpus`)).
- **Compute-bound thresholds** for a bf16 matmul: batch ≳ 240 from HBM, ~11–20 from
  VMEM, ~57,500 over PCIe — the same math, different denominators
  (source (`index.py get scaling-book-tpus`)).
- **ICI transfers**: time ≈ hops × per-hop latency (~1 µs) + bytes / (usable ports ×
  link BW); latency and bandwidth terms are additive
  (source (`index.py get scaling-book-tpus`)).
- **Multi-stage pipelines** (PCIe load → ICI gather → HBM→MXU → FLOPs): with good
  overlap, total ≈ the slowest stage, not the sum
  (source (`index.py get scaling-book-tpus`)).

**Arithmetic intensity** — FLOPs per byte moved — is the pivot: intensity above the
hardware ratio (peak FLOPs/s ÷ BW) means compute-bound. Raising it (bigger batches,
VMEM (`index.py get concept-vmem`)-resident weights, fusion) is the essence of most TPU
optimization (source (`index.py get scaling-book-tpus`)).

## See also

- Compute–communication overlap (`index.py get concept-compute-communication-overlap`) — the assumption
  behind the max()
- Critical batch size (`index.py get concept-critical-batch-size`) — the thresholds, per generation
- MFU (`index.py get concept-mfu`), Profile a TPU program (`index.py get playbook-profile-tpu-program`)

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- How To Scale Your Model, Part 1: Rooflines (scaling-book) (`index.py get scaling-book-roofline`)
