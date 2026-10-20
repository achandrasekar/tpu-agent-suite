---
title: "How to Think About TPUs (scaling-book, Part 2)"
type: source
tags: [hardware, performance, interconnect, memory]
created: 2026-08-22
updated: 2026-09-04
sources: 1
human_reviewed: false
---

Part 2 of Google DeepMind's [*How To Scale Your Model*](2025-scaling-book.md) (Austin et al., Feb 2025): how a TPU
chip is built (TensorCore = MXU + VPU + VMEM, attached to HBM), how chips are networked
(ICI tori, PCIe to host, DCN between hosts), and how the resulting bandwidth hierarchy
sets performance limits. The book's canonical hardware-mental-model chapter.

## Key claims

- A TPU is "a compute core that specializes in matrix multiplication (TensorCore)
  attached to a stack of fast memory (HBM)". All operations are pipelined and overlapped;
  HBM↔VMEM and VMEM↔systolic-array bandwidths set the fundamental efficiency limits.
- The MXU performs one `bf16[8,128] @ bf16[128,128] -> f32[8,128]` multiply every
  8 cycles via a 128×128 systolic array (256×256 on TPU v6e/Trillium); TensorCores have
  2–4 MXUs.
- VMEM is a programmer-controlled on-chip scratchpad (128 MiB on v5e) with ~22× the
  HBM bandwidth to the MXU: VMEM-resident weights become FLOPs-bound at arithmetic
  intensity ~10–20 (batch ~11–20) vs ~240 from HBM.
- Bandwidth hierarchy, fastest to slowest: HBM (~1–2 TB/s) → ICI (~90 GB/s per axis,
  3 axes on v5p) → PCIe (~16 GB/s) → DCN (~6.25 GB/s per TPU). Communication across
  each channel must be proportional to its speed to avoid starving the MXU.
- ICI is nearest-neighbor only: 2D torus on v2/v3/v5e/v6e (max 16×16), 3D torus on
  v4/v5p; superpods built from 4×4×4 cubes with reconfigurable optical wraparound links.
  Topologies smaller than a full cube (or a full 16-axis on v5e/v6e) lose wraparounds,
  roughly doubling communication time. Contrast with GPUs' switched point-to-point
  fabrics: TPUs are cheaper and scale further, but distant chips need multi-hop routes.
- Multi-slice: a slice is an ICI-connected set; slices connect over DCN, and a
  TPU-to-TPU DCN transfer traverses PCIe → host → DCN → host → PCIe.
- Weight matrices must be padded to ≥128 per dimension (256 on v6e) to fill the MXU.
- Lower precision is faster where supported: int8/int4 ≈ 2×/4× bf16 throughput; the VPU
  still computes in fp32.
- Megacore: v4–v6 chips pair two cores sharing memory into one logical accelerator;
  TPU v7 drops megacore for a high-bandwidth inter-core link; v5e is single-core.
- Appendix A details the VPU (8×128 SIMD, 4 ALUs per lane/sublane, ~7e12 FLOPs/s on
  v5p ≈ 30× below the MXU), VREGs, the single-threaded scalar core (one DMA request
  per cycle), and the slow cross-lane XLU. Appendix B animates systolic-array matmul.

## Key data points

| Link | Bandwidth (order) | Notes |
|---|---|---|
| VMEM → MXU | ~22× HBM | arithmetic intensity ~10–20 suffices for peak FLOPs |
| HBM → TensorCore | 1–2 TB/s (2.8 TB/s v5p) | the roofline denominator for most workloads |
| ICI per link | 4.5–10e10 B/s one-way | ×2 bidi with a wraparound ring; per-hop latency ~1 µs (v5e) |
| PCIe per TPU | ~1.6e10 B/s (3.2e10 v6e) | ~100× slower than HBM |
| DCN per TPU | ~6.25e9 B/s (12.5e9 v6e/7x, 3.125e9 v5e) | host-to-host only |

The book's per-generation spec tables remain in the raw snapshot but are **superseded
for spec lookups** by the official Cloud docs ingestion (2026-08-22) — see the
`wiki/hardware/` generation pages; note the book carries v3 numbers the docs set lacks.

Six worked roofline problems (sampling-latency floor from HBM BW, PCIe-bound batch
threshold ≈ 57.5k, HBM-vs-VMEM matmul crossover B>267 vs B>11–20, ICI transfer
latency+bandwidth math) — playbook fodder once a second roofline source lands.

## Gaps & caveats

- Spec numbers are book-reported, not vendor datasheets; ICI figures "differ slightly
  depending on operation".
- No Pallas/kernel-level guidance and no framework specifics — pure hardware model.
- GPU comparisons summarized here; the book's Section 12 (GPUs) is a separate source.

## Connections

Created: [MXU](../concepts/mxu.md), [VPU](../concepts/vpu.md), [VMEM](../concepts/vmem.md),
[HBM](../concepts/hbm.md), [ICI](../concepts/ici.md), [DCN](../concepts/dcn.md),
[PCIe](../concepts/pcie.md), [TensorCore](../concepts/tensorcore.md) (hub),
[Systolic array](../concepts/systolic-array.md),
[Roofline analysis](../concepts/roofline-analysis.md),
[Padding](../concepts/padding.md),
[Compute–communication overlap](../concepts/compute-communication-overlap.md),
[Quantization](../concepts/quantization.md), [Megacore](../concepts/megacore.md).

## Sources

- `raw/docs/2025-scaling-book-tpus.html` — snapshot of
  https://jax-ml.github.io/scaling-book/tpus/ (fetched 2026-08-22). Re-fetched
  2026-09-04 during the full-book ingest: text identical; upstream pin `ac72f02`
  (2026-09-03), chapter last changed 2026-06-01 (`62060ee`). Book-level record:
  [hub](2025-scaling-book.md).
