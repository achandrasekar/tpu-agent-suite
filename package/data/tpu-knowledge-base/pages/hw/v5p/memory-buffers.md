---
title: TPU v5p Memory and Buffer Sizing
type: concept
tags:
- hardware
- memory
- profiling
created: '2026-10-03'
human_reviewed: false
source_ids:
- xla-scoped-vmem
- xprof-memory-viewer
- xprof-memory-profile
- openxla-tiled-layout
- openxla-sparsecore
- xprof-utilization-viewer
- xprof-custom-call
generation_recipe: memory-buffers.recipe.json
---

# TPU v5p Memory and Buffer Sizing

## VMEM Limits and Allocation Scopes

TensorCore Vector Memory (VMEM) serves as a private on-chip scratchpad [^xla-scoped-vmem]. XLA manages VMEM across instruction-scoped allocations (transient operands, double buffers, and register spills during a single HLO instruction) and program-scoped allocations (temporaries and intermediate values surviving across instructions) [^xla-scoped-vmem]. Compile-time scoped VMEM out-of-memory errors occur when instruction-scoped needs exceed the configured allocation limit, which can be tuned globally using `--xla_tpu_scoped_vmem_limit_kib` or per custom kernel via `vmem_limit_bytes` [^xla-scoped-vmem]. Usable scoped limits differ from physical SRAM capacity because hardware reserves scratchpad space for system activities [^xla-scoped-vmem].

The compiler-defined usable limits for TPU v5p are listed below [^xla-scoped-vmem]:

| Generation | Human-Friendly Limit | Limit (KiB) | Limit (Bytes) | Derived Limit (MiB) |
| :--- | :--- | :--- | :--- | :--- |
| TPU v5p | 63.9MB | 65472 | 67043328 | 63.9375 MiB (derived) |

## Tile Sizing and Padding

XLA structures TPU data around 32-bit `8x128` vector registers [^openxla-tiled-layout]. Shapes not evenly divisible by tile dimensions are padded up to full tile boundaries with unconstrained padding values, which inflates memory consumption [^openxla-tiled-layout]. For 16-bit BF16 data, TPU uses two-level repeated tiling denoted as `(8,128)(2,1)` [^openxla-tiled-layout]. The initial `8x128` tile covers 1024 elements, and the secondary `(2,1)` tiling pairs two 16-bit values from adjacent rows in the same column into a 32-bit container [^openxla-tiled-layout]. A single complete BF16 resident tile of `(8,128)` elements consumes $8 \times 128 \times 2\text{ bytes} = 2048\text{ bytes}$ (derived: 2 KiB) [^openxla-tiled-layout].

## SparseCore Architecture and Sizing

SparseCore accelerates dynamic and sparse memory workloads using dedicated compute tiles [^openxla-sparsecore]. The hardware parameters established for TPU v5p are detailed below [^openxla-sparsecore]:

| Parameter | Specification |
| :--- | :--- |
| SparseCores per chip | 4 |
| Compute tiles per SparseCore | 16 |
| SIMD width | 8 |

XLA statically sizes scratchpad memory (SPMEM) buffers using user-defined partition limits [^openxla-sparsecore]. High Bandwidth Memory (HBM) stores embedding tables and hosts temporary stack allocations [^openxla-sparsecore]. For a single table, the HBM stack usage per pass is estimated as [^openxla-sparsecore]:
* Forward pass: $\approx (2 \times \text{feature\_width} + 1) \times \text{max\_unique\_nz\_per\_row} \times \text{logical\_replica\_count} \times 4\text{ bytes}$
* Backward pass: $\approx 3 \times \text{feature\_width} \times \text{max\_unique\_nz\_per\_row} \times \text{logical\_replica\_count} \times 4\text{ bytes}$

## Memory Viewer

The XProf Memory Viewer provides a purely static view of allocations derived from the XLA compiler across program order rather than wall-clock execution time [^xprof-memory-viewer]. It inspects peak module memory and displays TPU-specific on-chip memories, including VMEM, SMEM, CMEM, Sync Flags, and SparseCore [^xprof-memory-viewer]. Text overviews quantify total argument versus temporary requirements alongside memory overheads caused by tensor shape padding [^xprof-memory-viewer]. At peak points, buffer breakdown charts sort allocations by program schedule, raw allocation footprint, or padding inefficiency, assisting developers in pinpointing oversized buffers and excessive alignment waste [^xprof-memory-viewer].

## Memory Profile

In contrast to static views, the XProf Memory Profile captures dynamic device allocations across time during a profiling window [^xprof-memory-profile]. Managed by XLA's runtime allocator, it visualizes HBM timelines, stack and heap usage, and memory fragmentation percentages [^xprof-memory-profile]. High fragmentation indicates that available free memory lacks contiguity, risking allocation failures [^xprof-memory-profile]. On TPUs, the profile often appears as a flat horizontal line because the compiler performs large upfront allocations rather than dynamic allocations throughout execution [^xprof-memory-profile]. A memory breakdown table isolates framework ops contributing most heavily to device memory consumption at peak runtime intervals [^xprof-memory-profile].

## DMA Utilization

The XProf Utilization Viewer reports hardware execution unit activity and DMA bandwidth usage for TPU tensor nodes [^xprof-utilization-viewer]. DMA path utilization is calculated as the fraction of achieved bytes per cycle against theoretical peak bandwidth [^xprof-utilization-viewer]. A tensor node routes DMA transfers across seven memory nodes: HIB (host interface), HBM, ICI (inter-chip interconnect), and on-chip memories (IMem, SMem, BMem, VMem) [^xprof-utilization-viewer]. Transfers targeting ICI move data to remote HBM or VMEM across 14 internal pathways [^xprof-utilization-viewer].

## Trace-Buffer Overflow

Profiling custom calls with the flag `--xla_xprof_enable_custom_call_tracing=true` records fine-grained Low-Level Optimizer (LLO) bundle traces, which requires `xprof >= 2.24.0` and compatible toolchains [^xprof-custom-call]. These detailed trace points expand trace sizes significantly and can overflow the hardware trace buffer, causing event drops or truncating outer HLO timeline boundaries [^xprof-custom-call]. To mitigate overflows, trace insertion frequency can be tuned via `xla_tpu_bundle_instrumentation_options` [^xprof-custom-call]. Increasing `trace_best_effort_frequency` (default: 10 bundles) and `trace_guaranteed_frequency` (default: 10 bundles) samples instruction bundles less frequently, reducing trace generation while preserving critical profile metadata [^xprof-custom-call].

The supplied GitHub documents are the scope of the gaps below. For generation specifications such as HBM capacity, see the [hardware overview](overview.md).

LLO timestamps between trace points are interpolated from hardware-cost estimates, rather than individually measured instruction timestamps.[^xprof-custom-call]

## Unknown Capacities

The supplied sources do not establish the physical SRAM capacities of VMEM or the hardware capacities of SMEM, CMEM, BMEM, IMEM, and SPMEM for TPU v5p [^xla-scoped-vmem][^xprof-memory-viewer][^openxla-sparsecore][^xprof-utilization-viewer]. Only usable scoped VMEM limits and total HBM capacity are documented [^xla-scoped-vmem][^openxla-sparsecore].

<!-- sources:generated:start -->

## References

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^xprof-memory-viewer]: [XProf Memory Viewer](https://github.com/openxla/xprof/blob/master/docs/memory_viewer.md). profiler-reference. Static compiler allocation plan per HLO module and program order. URL checked 2026-10-03.

[^xprof-memory-profile]: [XProf Memory Profile](https://github.com/openxla/xprof/blob/master/docs/memory_profile.md). profiler-reference. Runtime memory usage, capacity, fragmentation and capture/lifetime peaks. URL checked 2026-10-03.

[^openxla-tiled-layout]: [OpenXLA tiled layout](https://github.com/openxla/xla/blob/main/docs/tiled_layout.md). compiler-reference. Tiling and padding; not generation-specific physical capacities. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

[^xprof-utilization-viewer]: [XProf Utilization Viewer](https://github.com/openxla/xprof/blob/master/docs/utilization_viewer.md). profiler-reference. Execution-unit and DMA-path utilization, not buffer capacity. URL checked 2026-10-16.

[^xprof-custom-call]: [XProf custom-call profiling](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md). profiler-reference. Kernel tracing, trace-buffer overflow and interpolated timings. URL checked 2026-10-16.

<!-- sources:generated:end -->
