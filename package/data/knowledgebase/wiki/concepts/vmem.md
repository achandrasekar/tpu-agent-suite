---
title: "VMEM (Vector Memory)"
type: concept
tags: [hardware, memory, performance]
created: 2026-08-22
updated: 2026-09-04
sources: 5
human_reviewed: true
---

VMEM is the on-chip, **programmer-controlled** scratchpad inside the
[TensorCore](tensorcore.md) — like an L1/L2 cache but larger and explicitly managed.
All data must pass HBM → VMEM before the [MXU](mxu.md)/[VPU](vpu.md) can touch it
([source](../sources/2025-scaling-book-tpus.md)).

## Numbers that matter

- Capacity: small — 128 MiB on v5e vs 16 GiB HBM
  ([source](../sources/2025-scaling-book-tpus.md)).
- Bandwidth to the MXU: **~22× HBM bandwidth**. Consequence: an MXU op fed from VMEM
  needs arithmetic intensity of only ~10–20 for peak FLOPs, vs ~240 from
  [HBM](hbm.md) — VMEM-resident weights make matmuls FLOPs-bound at batch ~11–20
  ([source](../sources/2025-scaling-book-tpus.md)). See
  [Roofline analysis](roofline-analysis.md).

## VMEM prefetching

Loading weights into VMEM ahead of need masks weight-load cost: e.g. load the
feed-forward weights during attention, hiding the transfer when memory-bandwidth
bound. Requires the layer's weights (possibly sharded) to fit in VMEM with room to
spare ([source](../sources/2025-scaling-book-tpus.md)). A core lever for
memory-bound inference; see
[Compute–communication overlap](compute-communication-overlap.md).

## Optimization implications

- Algorithms with inherently low arithmetic intensity can still hit peak FLOPs if
  their working set fits in VMEM — the central idea behind kernel tiling strategies.
- VMEM's small size is usually the binding constraint; fitting a single layer is
  already a win.
- The VMEM buffer size is a **tunable parameter**, and
  [Pallas](../codebases/pallas.md) kernel block sizes are constrained by the available
  VMEM — buffer sizing is an explicit kernel-tuning knob. Stated in the
  [TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md); likely
  generation-general as a Pallas/XLA-stack property *(unverified — agent inference)*.

## In practice

- In [HLO](hlo.md) an array's memory space shows as `S(1)` for VMEM (`S(0)` = HBM);
  XLA's decision to keep a fusion operand in VMEM is visible on the op line
  ([scaling-book, Part 9](../sources/2025-scaling-book-profiling.md)).
- Flash Attention keeps the `Q` chunk and the running max/sum/output in VMEM and streams
  KV chunks, which is what raises attention's arithmetic intensity
  ([scaling-book, Part 4](../sources/2025-scaling-book-transformers.md)).
- Versus GPU: an H100 has 256 kB SMEM + 256 kB registers per SM (~33 MB each in total)
  plus a 50 MB L2 at ~5.5 TB/s that is not programmer-controlled; TPU VMEM is ~2× larger
  and ~40 TB/s, so weights/activations can be pinned on-chip in ways SMEM cannot
  ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## See also

- [HBM](hbm.md) — the level above; [MXU](mxu.md) — the consumer

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [Cloud TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md)
- [How To Scale Your Model, Part 4: Transformer Math (scaling-book)](../sources/2025-scaling-book-transformers.md),
  [How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book)](../sources/2025-scaling-book-profiling.md),
  [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
