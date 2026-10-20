---
title: Memory and Buffer Sizing for TPU v6e
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

# Memory and Buffer Sizing for TPU v6e

## VMEM Limits and Allocation Scopes

TensorCore uses Vector Memory (VMEM) as a local scratchpad [^xla-scoped-vmem]. XLA manages instruction-scoped allocations (temporary operands, double buffers, register spills) and program-scoped allocations (HLO temporaries, inter-instruction results) [^xla-scoped-vmem]. Exceeding an instruction's limit causes a compile-time scoped VMEM out-of-memory (OOM) error [^xla-scoped-vmem]. Limits are configured globally via `--xla_tpu_scoped_vmem_limit_kib` or per custom kernel via `vmem_limit_bytes` [^xla-scoped-vmem].

The typical usable scoped VMEM limits for TPU v6e are provided below [^xla-scoped-vmem]:

| Hardware | Limit (Human-Friendly) | Limit (KiB) | Limit (Bytes) | Derived Limit (MiB) |
| :--- | :--- | :--- | :--- | :--- |
| TPU v6e | 127.9MB | 131,008 | 134,152,192 | 127.9375 MiB |

*(Derived arithmetic: $131008 \text{ KiB} / 1024 = 127.9375 \text{ MiB}$, or $134152192 \text{ bytes} / 1048576 = 127.9375 \text{ MiB}$)* [^xla-scoped-vmem].

These typical usable limits reflect reserved VMEM set aside by hardware for internal activities and differ from raw physical SRAM capacity, which is not established in the sources [^xla-scoped-vmem].

## Tile Sizing and Padding

XLA structures TPU data into major-to-minor tiled layouts matching hardware vector registers [^openxla-tiled-layout]. The standard layout utilizes $8 \times 128$ tiling [^openxla-tiled-layout]. Small tile formats apply $2 \times 128$ (for second minor dimension sizes of 1–2) or $4 \times 128$ (for sizes 3–4) [^openxla-tiled-layout].

BF16 arrays use repeated tiling: $(8, 128)(2, 1)$ [^openxla-tiled-layout]. The $(2, 1)$ minor packing combines two 16-bit values along a column into a 32-bit word [^openxla-tiled-layout].

*Resident-tile calculation (derived):*
A single $(8, 128)(2, 1)$ BF16 tile contains $8 \times 128 = 1024$ elements [^openxla-tiled-layout]. At 2 bytes per BF16 element, memory consumption is:
$$1024 \times 2 \text{ bytes} = 2048 \text{ bytes} = 2 \text{ KiB}$$
Dimensions non-divisible by tile bounds are padded up to complete tile boundaries ($\lceil d_i / t_i \rceil \cdot t_i$), creating memory overhead [^openxla-tiled-layout].

## SparseCore

The source names TPU v6e as Trillium: 2 SparseCores per chip, 16 tiles per SparseCore, and SIMD width 8 for F32 or 16 for BF16. It does not establish physical SPMEM capacity [^openxla-sparsecore]. Generic SparseCore architecture uses SPMEM for local staging and High Bandwidth Memory (HBM) for tables and stacks [^openxla-sparsecore].

SparseCore HBM stack consumption per table is modeled as:
* Forward Pass: $\approx (2 \cdot \text{feature\_width} + 1) \cdot \text{max\_unique\_nz\_per\_row} \cdot \text{logical\_replica\_count} \cdot 4\text{ bytes}$ [^openxla-sparsecore].
* Backward Pass: $\approx 3 \cdot \text{feature\_width} \cdot \text{max\_unique\_nz\_per\_row} \cdot \text{logical\_replica\_count} \cdot 4\text{ bytes}$ [^openxla-sparsecore].

Embedding tables enforce 32-byte alignment (8 floats) across feature dimensions; smaller feature widths produce padding waste [^openxla-sparsecore].

## Memory Viewer

XProf Memory Viewer displays static compiler memory layouts across scheduled program points (HLO sequence order, not runtime time) [^xprof-memory-viewer]. It inspects peak memory utilization per chosen module and details argument versus temporary splits [^xprof-memory-viewer]. On TPUs, supported memory types include VMEM, SMEM, CMEM, Sync Flags (SFlag), SparseCore, HBM, and Host memory [^xprof-memory-viewer]. Peak buffer charts categorize allocations by program order, size, and padding overhead [^xprof-memory-viewer]. The tool does not capture dynamic co-residence of future modules compiled independently [^xprof-memory-viewer].

## Memory Profile

The Memory Profile tool tracks dynamic accelerator memory usage over time during active profiling windows [^xprof-memory-profile]. It graphs HBM timelines across stack, heap, and free space while computing memory fragmentation percentages [^xprof-memory-profile]. Runtime allocations are driven by XLA’s allocator [^xprof-memory-profile]. Because TPUs typically perform large upfront allocations at initialization, the dynamic trace often appears as a flat horizontal line [^xprof-memory-profile]. Unlike Memory Viewer's single-module view, Memory Profile yields a global, multi-module view [^xprof-memory-profile].

## DMA Utilization

Utilization Viewer provides bird's-eye monitoring of TPU performance [^xprof-utilization-viewer]. It tracks DMA bandwidth utilization across 14 paths in a Tensor Node using NF_CTRL counters [^xprof-utilization-viewer]. DMA paths connect HIB (host interface), HBM, ICI (inter-chip interconnect), and local units including VMEM, BMem, SMem, and IMem [^xprof-utilization-viewer]. Bandwidth utilization reflects achieved bytes per cycle divided by theoretical peak bytes per cycle [^xprof-utilization-viewer].

## Trace-Buffer Overflow

Profiling custom calls with Low-Level Optimizer (LLO) support requires `xprof >= 2.24.0` and Python 3.11+ [^xprof-custom-call]. Standard captures should use `--xla_xprof_register_llo_debug_info=true` [^xprof-custom-call]. Enabling runtime bundle tracing via `--xla_xprof_enable_custom_call_tracing=true` instruments instruction bundles, which on TPU v6e can increase trace sizes by 6.4× (e.g., from 19.8 MB to 127 MB) [^xprof-custom-call].

This surge can overflow hardware trace buffers, dropping HLO boundary events into `NO_DATA` or `IDLE` states [^xprof-custom-call]. To reduce the risk of buffer drops, tune bundle frequencies via `xla_tpu_bundle_instrumentation_options` by increasing `trace_best_effort_frequency` and `trace_guaranteed_frequency` above their default of 10 (e.g., to 50 or 100) [^xprof-custom-call].

The supplied GitHub documents are the scope of the gaps below. For generation specifications such as HBM capacity, see the [hardware overview](overview.md).

LLO timestamps between trace points are interpolated from hardware-cost estimates, rather than individually measured instruction timestamps.[^xprof-custom-call]

## Unknown Capacities

The provided documentation does not establish the physical SRAM capacity of TPU v6e VMEM [^xla-scoped-vmem]. Furthermore, physical memory capacities for TPU v6e SMEM, CMEM, BMEM, IMEM, and SPMEM are not established in the sources [^xprof-memory-viewer][^openxla-sparsecore][^xprof-utilization-viewer].

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
