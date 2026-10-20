---
title: TPU v5e Memory and Buffer Sizing
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

# TPU v5e Memory and Buffer Sizing

## VMEM Limits and Allocation Scopes

Vector Memory (VMEM) is an on-chip scratchpad memory utilized exclusively by the TensorCore (TC)[^xla-scoped-vmem][^openxla-sparsecore]. Physical SRAM capacity is not established by the provided sources; the usable limits represent compiler-enforced thresholds after reserving hardware overhead[^xla-scoped-vmem]. Scoped VMEM allocations fall into instruction-scoped memory (temporary storage such as operand span buffers or register spills) and program-scoped memory (temporaries across HLO instructions)[^xla-scoped-vmem]. Compile-time limits can be adjusted globally via `--xla_tpu_scoped_vmem_limit_kib` or configured per custom kernel via the `vmem_limit_bytes` parameter[^xla-scoped-vmem].

The compiler-defined usable scoped VMEM limits for TPU v5e are detailed below[^xla-scoped-vmem]:

| Hardware Generation | Human-Friendly Limit | Limit (KiB) | Limit (Bytes) | Derived Limit (MiB) |
| :--- | :--- | :--- | :--- | :--- |
| TPU v5e | 128MB | 131072 | 134217728 | 128.0 MiB |

*(Derived arithmetic: 131,072 KiB / 1,024 = 128.0 MiB, or 134,217,728 bytes / 1,048,576 = 128.0 MiB.)*

## Tile Sizing and Padding

XLA structures TPU layouts into tiles to match the hardware's 2D vector registers[^openxla-tiled-layout]. Native execution operates on 32-bit values[^openxla-tiled-layout]. Standard operations utilize an `8x128` tile format[^openxla-tiled-layout]. For BF16 arrays, a two-level nested tiling scheme of `(8,128)(2,1)` is used to pack pairs of 16-bit elements along column dimensions into single 32-bit units[^openxla-tiled-layout]. Array bounds are padded to full tile multiples: $\lceil d_i / t_i \rceil \cdot t_i$[^openxla-tiled-layout].

For a single resident BF16 tile laid out under `(8,128)(2,1)`:
- Elements per tile: $8 \times 128 = 1024$ elements[^openxla-tiled-layout].
- Memory footprint: $1024 \text{ elements} \times 2 \text{ bytes} = 2048 \text{ bytes}$ (derived).

## SparseCore

SparseCore is an accelerator for irregular memory workloads using local scratchpad memory (SPMEM) and HBM[^openxla-sparsecore]. Physical specifications for TPU v5e are omitted from the supplied architecture tables[^openxla-sparsecore]:

| Parameter | TPU v5e Specification |
| :--- | :--- |
| SparseCores per chip | Not established[^openxla-sparsecore] |
| Tiles per SparseCore | Not established[^openxla-sparsecore] |
| SIMD Width | Not established[^openxla-sparsecore] |
| HBM Capacity | Not established[^openxla-sparsecore] |

The following is generic SparseCore guidance and does not establish that this chip has SparseCores. SPMEM buffers and HBM stack buffers are statically sized at compile time using operational limits such as `max_ids_per_partition` and `max_unique_ids_per_partition`[^openxla-sparsecore]. Generic HBM stack usage per single table is estimated as:
- Forward Pass HBM Stack $\approx (2 \times \text{feature\_width} + 1) \times \text{max\_unique\_nz\_per\_row} \times \text{logical\_replica\_count} \times 4 \text{ bytes}$[^openxla-sparsecore].
- Backward Pass HBM Stack $\approx 3 \times \text{feature\_width} \times \text{max\_unique\_nz\_per\_row} \times \text{logical\_replica\_count} \times 4 \text{ bytes}$[^openxla-sparsecore].

Embedding tables require 32-byte alignment along the feature dimension[^openxla-sparsecore].

## Memory Viewer

XProf's Memory Viewer displays purely static allocation data determined by the XLA compiler; dynamic runtime allocation is omitted[^xprof-memory-viewer]. It highlights memory usage plotted against program order (scheduled HLO sequence) rather than time[^xprof-memory-viewer]. Visualizations evaluate usage on a per-module basis without tracking co-resident modules executed simultaneously on the same accelerator[^xprof-memory-viewer]. At peak usage points, the tool breaks down allocations by program order, buffer size, and padding overhead[^xprof-memory-viewer]. For TPUs, it inspects on-chip memories, including VMEM, SMEM, CMEM, Sparsecore, and SFlag, alongside Host Memory and HBM[^xprof-memory-viewer].

## Memory Profile

The Memory Profile tool tracks dynamic memory usage across time during profiling sessions[^xprof-memory-profile]. It focuses on the accelerator runtime allocator, which manages HBM[^xprof-memory-profile]. Across time, allocations are categorized into stack, heap, and free space[^xprof-memory-profile]. On TPUs, memory traces frequently appear as flat horizontal lines due to large upfront static allocations compiled prior to model execution[^xprof-memory-profile]. Fragmentation describes whether free memory is split into non-contiguous regions[^xprof-memory-profile]. Unlike Memory Viewer's per-module view, Memory Profile provides a global runtime view across all participating XLA modules[^xprof-memory-profile].

## DMA Utilization

Visualizing DMA bandwidth utilization requires the Utilization Viewer[^xprof-utilization-viewer]. It displays data paths across two Tensor Nodes in a TPU chip[^xprof-utilization-viewer]. The tool tracks 14 DMA paths linking units such as HBM, HIB (host interface), ICI (inter-chip interconnect), and internal memories[^xprof-utilization-viewer]. The path labeled "BMem to ICI/VMem" consolidates transfers to remote targets or local VMEM under a single counter[^xprof-utilization-viewer]. Bandwidth utilization is reported as the fraction of achieved bytes per cycle against peak theoretical throughput[^xprof-utilization-viewer].

## Trace-Buffer Overflow

Fine-grained custom call profiling relies on Low-Level Optimizer (LLO) tracing, which requires `xprof >= 2.24.0`[^xprof-custom-call]. Activating runtime custom call tracing via `--xla_xprof_enable_custom_call_tracing=true` (or the legacy `--xla_enable_custom_call_region_trace=true`) inserts instruction bundle instrumentation[^xprof-custom-call]. This can expand trace sizes dramatically—between 6× and 12×—and trigger hardware trace-buffer overflows[^xprof-custom-call]. Such overflows drop or truncate outer HLO lifecycle events, resulting in `NO_DATA` or `IDLE` markers[^xprof-custom-call].

Trace generation is tuned inside `xla_tpu_bundle_instrumentation_options` via:
- `trace_best_effort_frequency` (default: 10 bundles)[^xprof-custom-call]
- `trace_guaranteed_frequency` (default: 10 bundles)[^xprof-custom-call]

To alleviate event drops and hardware buffer overflows, these frequency values should be increased (e.g., to 50 or 100)[^xprof-custom-call].

The supplied GitHub documents are the scope of the gaps below. For generation specifications such as HBM capacity, see the [hardware overview](overview.md).

LLO timestamps between trace points are interpolated from hardware-cost estimates, rather than individually measured instruction timestamps.[^xprof-custom-call]

## Unknown Capacities

Physical hardware capacities for TPU v5e remain undocumented in the provided texts. Exact physical sizes are not established for:
- Physical VMEM SRAM capacity[^xla-scoped-vmem]
- SparseCore SPMEM scratchpad capacity[^openxla-sparsecore]
- HBM capacity[^openxla-sparsecore]
- Specialized memory units: SMEM, CMEM, BMEM, and IMEM[^xprof-memory-viewer][^xprof-utilization-viewer]

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
