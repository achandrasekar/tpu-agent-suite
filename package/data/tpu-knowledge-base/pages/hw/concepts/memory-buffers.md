---
title: 'TPU Memory and Buffer Sizing: v6e and v7x Reference'
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

# TPU Memory and Buffer Sizing: v6e and v7x Reference

## VMEM Limits and Allocation Scopes

TensorCore Vector Memory (VMEM) is an on-chip local scratchpad memory used exclusively by the TensorCore (TC)[^xla-scoped-vmem][^openxla-sparsecore]. The XLA compiler divides VMEM into two allocation lifetimes: instruction-scoped allocations, which serve temporary needs during a single HLO instruction (such as operand span double buffers or register spills), and program-scoped allocations, which persist across instructions for HLO temporaries and intermediate results[^xla-scoped-vmem].

A compile-time scoped VMEM OOM occurs when instruction-scoped requirements exceed the configured instruction limit[^xla-scoped-vmem]. Users adjust limits globally via `--xla_tpu_scoped_vmem_limit_kib` or per custom call via the `vmem_limit_bytes` parameter[^xla-scoped-vmem].

The typical usable scoped limits for TPU v6e and TPU 7x are listed below[^xla-scoped-vmem]:

| Generation | Human-Friendly Limit | Limit (KiB) | Limit (bytes) | Limit (MiB, derived) |
| :--- | :--- | :--- | :--- | :--- |
| TPU v6e | 127.9MB | 131008 | 134152192 | 127.9375 MiB |
| TPU 7x | 63.9MB | 65472 | 67043328 | 63.9375 MiB |

Hardware reserves portions of total capacity for system functions, restricting the usable scoped ceiling[^xla-scoped-vmem].

## Tile Sizing and Padding

XLA structures memory into recursive major-to-minor tiles to map to the 32-bit $8 \times 128$ vector registers of the TPU[^openxla-tiled-layout]. Arrays are padded to full tile multiples: $\lceil d_i / t_i \rceil \cdot t_i$[^openxla-tiled-layout]. Sub-byte and multi-byte datatypes use repeated tiling to pack values into native 32-bit slots[^openxla-tiled-layout].

BF16 utilizes an initial tile followed by sub-element packing denoted as $(8,128)(2,1)$, interleaving one element from an even row and one from an odd row within the same column to optimize data movement across the second most minor dimension[^openxla-tiled-layout].

For a resident BF16 tile laid out as $(8, 128)(2, 1)$, the tile holds $8 \times 128 = 1024$ logical elements[^openxla-tiled-layout]. At 2 bytes per BF16 element, this consumes exactly 2048 bytes (derived: $1024 \times 2 = 2048$ bytes; derived: 2 KiB)[^openxla-tiled-layout].

## SparseCore Architecture and Sizing

SparseCore accelerates dynamic and irregular sparse memory access patterns targeting High Bandwidth Memory (HBM)[^openxla-sparsecore].

SparseCore physical organization for Trillium (TPU v6e)[^openxla-sparsecore]:

| Parameter | Trillium (TPU v6e) Specification |
| :--- | :--- |
| SparseCores per chip | 2 |
| Compute tiles per SparseCore | 16 |
| SIMD width | 8 (F32), 16 (BF16) |

Scratchpad memory (SPMEM) is an on-device local SRAM managed statically by XLA; dynamic SPMEM allocation is unsupported[^openxla-sparsecore]. Buffer sizing is statically bounded by `max_ids_per_partition` and `max_unique_ids_per_partition`[^openxla-sparsecore].

SparseCore intermediate results spill into temporary HBM stack space, estimated per table as[^openxla-sparsecore]:
* Forward pass: $\approx (2 \cdot \text{feature\_width} + 1) \cdot \text{max\_unique\_nz\_per\_row} \cdot \text{logical\_replica\_count} \cdot 4\text{ bytes}$
* Backward pass: $\approx 3 \cdot \text{feature\_width} \cdot \text{max\_unique\_nz\_per\_row} \cdot \text{logical\_replica\_count} \cdot 4\text{ bytes}$

Serializing tables using `xla_sc_num_serialized_tables_to_optimize_hbm` decreases concurrent live HBM stack memory at the expense of parallelism[^openxla-sparsecore]. Small feature widths under 8 floats incur significant HBM padding because dimensions are padded to 32-byte alignment[^openxla-sparsecore].

## Static Memory Inspection: Memory Viewer

The XProf Memory Viewer provides a purely static view of allocations derived from the XLA compiler across program order (HLO sequence), without plotting real time[^xprof-memory-viewer]. It exposes memory usage on a per-XLA-module basis (such as top-level execution modules like `jit_train_step`), meaning it cannot track runtime co-residence across separate modules compiled independently[^xprof-memory-viewer].

On TPUs, the tool visualizes multiple memory spaces: HBM, Host memory, and on-chip scratchpads including VMEM, SMEM, CMEM, SparseCore, and Sync Flags (SFlag)[^xprof-memory-viewer]. It isolates peak module consumption, distinguishing shapes, frameworks ops, parameter buffers, outputs, and compiler temporaries, while highlighting padding overhead from dimension alignment[^xprof-memory-viewer].

## Dynamic Memory Profiling: Memory Profile

The XProf Memory Profile tool monitors dynamic runtime allocations and deallocations over time across a profiling interval[^xprof-memory-profile]. In contrast to Memory Viewer, it provides a global view of memory across multi-module workloads and analyzes allocator fragmentation percentages[^xprof-memory-profile].

XLA’s runtime allocator manages the entire physical HBM space[^xprof-memory-profile]. The interface separates device utilization into stack, heap, and free space[^xprof-memory-profile]. Because TPU compilation typically relies on large upfront allocations prior to execution rather than dynamic runtime churn, Memory Profile timeline charts frequently manifest as flat horizontal lines, displaying static initial footprints rather than active step-by-step variations[^xprof-memory-profile].

## DMA Bandwidth and Utilization

TPU utilization analysis is provided by the Utilization Viewer tool[^xprof-utilization-viewer]. Hardware utilization is tracked across the two Tensor Nodes inside a chip using execution unit utilization (busy cycles divided by total cycles) and 14 DMA paths monitored by `NF_CTRL` byte counters[^xprof-utilization-viewer].

Reported DMA paths include transfers between HBM, VMEM, SMEM, BMem, IMem, HIB (host memory transfers), and ICI (Inter-Chip Interconnect)[^xprof-utilization-viewer]. DMAs routed to ICI access remote VMEM or HBM on peer chips[^xprof-utilization-viewer]. Notably, the "BMem to VMem" and "BMem to ICI" routes share physical hardware monitoring and are accumulated under a single counter labeled "BMem to ICI/VMem"[^xprof-utilization-viewer].

## Trace-Buffer Overflow and LLO Instrumentation

Low-Level Optimizer (LLO) analysis and custom call profiling require `xprof >= 2.24.0`[^xprof-custom-call]. Compiling custom calls with `--xla_xprof_enable_custom_call_tracing=true` (requiring `libtpu >= 0.0.44` and `jax >= 0.11.0`)[^xprof-custom-call] records fine-grained runtime bundle trace points inside kernels[^xprof-custom-call].

This fine-grained instrumentation expands trace output sizes by 6× to 12× (for example, growing a 10-iteration Pallas FlashAttention trace from 19.8 MB to 127 MB on TPU v6e-1, and from 115 MB to 1.42 GB on TPU v7x)[^xprof-custom-call]. This volume can overflow the fixed hardware trace buffer, dropping HLO `Begin`/`End` markers (manifesting as `NO_DATA` or `IDLE` in summaries) or discarding custom call records[^xprof-custom-call].

To reduce trace-buffer event drops, users can tune `xla_tpu_bundle_instrumentation_options`[^xprof-custom-call]:
* `trace_best_effort_frequency`: target bundle interval for opportunistic insertion (default: 10)[^xprof-custom-call].
* `trace_guaranteed_frequency`: maximum allowed bundle gap forcing insertion of a standalone trace bundle (default: 10)[^xprof-custom-call].

Increasing these intervals (e.g., to 50 or 100) reduces data volume and the risk of buffer exhaustion[^xprof-custom-call]. Intermediate LLO timestamps are interpolated using hardware-cost estimates, rather than individually measured instruction timings[^xprof-custom-call]. Periodic runtime performance counter sampling (`tpu_enable_periodic_counter_sampling`) is restricted to TPU v7x+[^xprof-custom-call].

## Established vs. Unknown Capacities

Physical hardware capacities for several on-chip TPU memories are not established by the available sources:
* Total physical SRAM VMEM capacities for v6e and v7x are not established (only usable compiler-scoped limits are provided: 134,152,192 bytes for v6e and 67,043,328 bytes for v7x)[^xla-scoped-vmem].
* Physical SRAM capacities for SMEM, CMEM, BMEM, IMEM, and SparseCore SPMEM are not established by the sources[^xla-scoped-vmem][^xprof-memory-viewer][^openxla-sparsecore][^xprof-utilization-viewer].
* SparseCore count, tile breakdown, and HBM capacity for TPU 7x are not established by the sources[^openxla-sparsecore].

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
