---
title: Cloud TPU v5p Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-v5p
- cloud-tpu-7x
- xla-scoped-vmem
- openxla-tiled-layout
- openxla-sparsecore
generation_recipe: overview.recipe.json
---

# Cloud TPU v5p Hardware Overview

## Chip Compute and Architecture

Cloud TPU v5p is a domain-specific accelerator designed for large-scale machine learning workloads [^cloud-tpu-v5p], [^cloud-tpu-7x]. The chip architecture features:

* **TensorCores:** Each TPU v5p chip integrates 2 TensorCores [^cloud-tpu-v5p]. Each TensorCore contains 4 Matrix Multiply Units (MXUs), a vector unit, and a scalar unit [^cloud-tpu-v5p].
* **Peak Compute:** Google reports vendor-claimed peak compute of 459 TFLOPs in BF16 and 459 TFLOPs in FP8 per chip [^cloud-tpu-v5p].
* **SparseCores:** Each chip integrates 4 SparseCores dedicated to irregular and dynamic memory operations, such as large embedding lookups [^cloud-tpu-v5p], [^openxla-sparsecore]. Each SparseCore is structured as a tiled architecture containing 16 compute tiles governed by a centralized sequencer [^openxla-sparsecore]. SparseCore execution operates with a SIMD width of 8 elements and supports data-dependent execution and cross-lane summarization routines (e.g., sorting, filtering, and prefix sums) [^openxla-sparsecore].
* **Vector Execution & Register Tiling:** The generic OpenXLA layout reference describes vector execution over 32-bit 8 x 128 vector registers [^openxla-tiled-layout]. XLA aligns operand layouts to hardware registers through multi-dimensional tiled layouts; for example, native 16-bit BF16 operands employ repeated tiling `(8,128)(2,1)`, packing two 16-bit values from adjacent rows into one 32-bit register element [^openxla-tiled-layout].

## Memory Hierarchy and Buffer Limits

TPU v5p employs a tiered memory hierarchy consisting of host system memory, on-package High-Bandwidth Memory (HBM), and on-chip scratchpad SRAMs [^cloud-tpu-v5p], [^cloud-tpu-7x], [^openxla-sparsecore]:

* **High-Bandwidth Memory (HBM):** Google Cloud specifications state an HBM capacity of 95 GiB per chip with 2765 GBps of bandwidth [^cloud-tpu-v5p], [^cloud-tpu-7x] (though OpenXLA documentation lists HBM capacity as 96 GiB [^openxla-sparsecore]). HBM stores model parameters, optimizer states, SparseCore embedding tables, and temporary execution stacks [^openxla-sparsecore].
* **Vector Memory (VMEM):** VMEM is an on-chip SRAM local to each TensorCore used for stack allocations, intermediate instruction-scoped scratchpads (e.g., double-buffering operand spans), and program-scoped HLO temporaries [^xla-scoped-vmem]. While total physical VMEM SRAM capacity is not stated, XLA enforces a usable scoped VMEM limit of 65,472 KiB / 67,043,328 bytes (63.9375 MiB, derived) [^xla-scoped-vmem]. Exceeding this quota triggers compile-time error `E1001` [^xla-scoped-vmem].
* **SparseCore Scratchpad Memory (SPMEM):** Each SparseCore incorporates local SPMEM (SRAM) across its 16 tiles to opportunistically stage embedding inputs and outputs [^openxla-sparsecore]. SPMEM does not function as a hardware cache; XLA statically allocates SPMEM buffers at compile time based on configured partition limits [^openxla-sparsecore]. Dynamic SPMEM allocation is unsupported [^openxla-sparsecore].

| Memory Tier | Allocation / Capacity | Bandwidth / Scope |
| :--- | :--- | :--- |
| **HBM** | 95 GiB per chip [^cloud-tpu-v5p] (96 GiB in [^openxla-sparsecore]) | 2765 GBps per chip [^cloud-tpu-v5p] |
| **Usable Scoped VMEM** | 65,472 KiB usable scoped limit (63.9375 MiB, derived) [^xla-scoped-vmem] | On-chip TensorCore scratchpad SRAM [^xla-scoped-vmem] |
| **SPMEM** | Statically allocated per tile [^openxla-sparsecore] | On-chip SparseCore scratchpad SRAM [^openxla-sparsecore] |
| **Host System RAM** | 448 GB per 4-chip VM [^cloud-tpu-v5p] | Host system memory [^cloud-tpu-v5p] |

## Interconnect and Pod Scale

TPU v5p utilizes two primary interconnect interfaces:

* **Inter-Chip Interconnect (ICI):** Each chip provides 1200 GBps of bidirectional ICI bandwidth [^cloud-tpu-v5p]. Intra-cube connections use direct copper links, while inter-cube routing traverses optical links mediated by Optical Circuit Switches (OCS) [^cloud-tpu-v5p].
* **Data Center Network (DCN):** Each chip has 50 Gbps of dedicated DCN bandwidth, backed by a 200 Gbps network interface controller (NIC) per 4-chip host [^cloud-tpu-v5p].
* **Pod Scale & Slicing:** A complete TPU v5p Pod comprises 8960 chips (2240 hosts, 140 cubes, and a derived 17,920 TensorCores) [^cloud-tpu-v5p]. The maximum single-slice training job schedules up to 96 cubes (6144 chips / 1536 hosts / 12,288 TensorCores) arranged in a 16 x 16 x 24 topology [^cloud-tpu-v5p]. Multislice scaling supports up to 18,432 chips over DCN [^cloud-tpu-v5p].
* **Topology Topography:** Slices smaller than a full 64-chip cube (4 x 4 x 4) are 3D-mesh connected without wrap-around links [^cloud-tpu-v5p]. Slices of 4 x 4 x 4 and larger possess full 3D torus connectivity [^cloud-tpu-v5p].
* **Twisted Torus Configurations:** Slices configured where each dimension is equal to or double the smallest dimension can configure twisted wrap-around links via the Cloud TPU API (e.g., 4 x 4 x 8, 4 x 8 x 8, 8 x 8 x 16, 8 x 16 x 16, 12 x 12 x 24) [^cloud-tpu-v5p]. In a twisted layout, wrap-around edges shift along an adjacent dimension (e.g., connecting coordinate X to (X + 2) \pmod 4), creating network symmetry, higher bisection bandwidth, and shorter packet paths [^cloud-tpu-v5p]. Vendor claims state theoretical bisection bandwidth gains of ~40% to ~70% [^cloud-tpu-v5p]. MaxText FSDP training experiments demonstrated a 1–2 percentage point model FLOPs utilization (MFU) improvement using twisted topologies [^cloud-tpu-v5p].
* **ICI Resiliency:** Enabled by default on slices of 1 cube or larger, ICI resiliency re-routes traffic around faulted optical links or OCS nodes, maintaining job availability at the cost of transient bandwidth degradation [^cloud-tpu-v5p].

## Slice and Host Configurations

All TPU v5p topologies utilize the `ct5p-hightpu-4t` virtual machine type [^cloud-tpu-v5p]. Each VM maps directly to 1 physical host controlling 4 v5p chips [^cloud-tpu-v5p].

* **Host VM Specs:** 208 vCPUs, 448 GB RAM, 2 NUMA nodes, and 200 Gbps NIC throughput [^cloud-tpu-v5p].
* **NUMA Constraint:** To prevent cross-NUMA latency penalties, workloads binding execution to a single NUMA node have access to exactly half the VM resources: derived as 104 vCPUs and 224 GB RAM [^cloud-tpu-v5p].

| Topology | Cubes | Chips | Cores | Hosts / VMs | Machine Type | Twisted Supported |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2x2x1** | N/A | 4 | 8 | 1 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **2x2x2** | N/A | 8 | 16 | 2 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **2x4x4** | N/A | 32 | 64 | 8 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **4x4x4** | 1 | 64 | 128 | 16 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **4x4x8** | 2 | 128 | 256 | 32 | `ct5p-hightpu-4t` | Yes (~70% bisection gain) [^cloud-tpu-v5p] |
| **4x8x8** | 4 | 256 | 512 | 64 | `ct5p-hightpu-4t` | Yes (~40% bisection gain) [^cloud-tpu-v5p] |
| **8x8x8** | 8 | 512 | 1024 | 128 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **8x8x16** | 16 | 1024 | 2048 | 256 | `ct5p-hightpu-4t` | Yes (~70% bisection gain) [^cloud-tpu-v5p] |
| **8x16x16** | 32 | 2048 | 4096 | 512 | `ct5p-hightpu-4t` | Yes (~40% bisection gain) [^cloud-tpu-v5p] |
| **16x16x16**| 64 | 4096 | 8192 | 1024 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |
| **16x16x24**| 96 | 6144 | 12288| 1536 | `ct5p-hightpu-4t` | No [^cloud-tpu-v5p] |

## Software and Operational Constraints

* **Frameworks & Interfaces:** SparseCore embedding layers are driven through framework APIs including `jax-tpu-embedding`, TensorFlow (`tfrs.layers.embedding.TPUEmbedding`), and Keras (`keras_rs`) [^openxla-sparsecore]. MaxText supports large-scale LLM training on v5p [^cloud-tpu-v5p].
* **SparseCore Pipeline & Limits:** Embedding tables are mod-sharded across all SparseCores: \text{target\_sc\_id} = \text{col\_id} \pmod{\text{num\_total\_sparse\_cores}} [^openxla-sparsecore]. Host preprocessing converts inputs to Coordinate (COO) format (`row_ids`, `col_ids`, `values`) [^openxla-sparsecore]. Sizing of internal buffers depends on `max_ids_per_partition` and `max_unique_ids_per_partition` [^openxla-sparsecore]. If runtime data exceeds compiled partition limits, the job aborts with compile or runtime errors unless `allow_id_dropping=True` is explicitly specified [^openxla-sparsecore].
* **TC/SC Pipelining:** Simultaneous execution across TensorCore dense computations and SparseCore sparse lookups can be toggled via `tf_xla_disable_full_embedding_pipelining` [^openxla-sparsecore]. When active, pipelining can yield up to a claimed 2x step-time improvement, though SparseCores may calculate against stale gradients from preceding steps [^openxla-sparsecore].
* **Quantization:** SparseCore execution units (`SparseDenseMatmulOp`) support FP32 alongside integer quantization (e.g., 256 buckets for 8-bit quantization) and simulated quantization during training [^openxla-sparsecore].
* **Tuning VMEM:** Custom Pallas kernels or complex HLO graphs must respect the 63.9 MB Scoped VMEM barrier [^xla-scoped-vmem]. Developers can adjust the kernel footprint via `vmem_limit_bytes` or globally using `--xla_tpu_scoped_vmem_limit_kib` [^xla-scoped-vmem].

## Gaps

The supplied evidence does not provide details on the following areas:

* **Physical Silicon Parameters:** Clock frequency (MHz/GHz), process node generation, thermal design power (TDP), total physical die area, and physical SRAM capacity for VMEM and SPMEM (distinguished from usable scoped limits) are not disclosed [^cloud-tpu-v5p], [^xla-scoped-vmem], [^openxla-sparsecore].
* **HBM Discrepancy:** Cloud TPU product documentation lists 95 GiB of HBM capacity per chip [^cloud-tpu-v5p], [^cloud-tpu-7x], while OpenXLA documentation lists 96 GiB [^openxla-sparsecore].
* **SparseCore Implementation Internals:** SparseCore low-level compiler lowerings and kernel implementations remain proprietary and unreleased, distributed only within the binary `libtpu.so` driver [^openxla-sparsecore].

## Detailed memory and profiling

See [the generation-specific memory and buffer reference](memory-buffers.md) for tile sizing, padding and XProf inspection.

<!-- sources:generated:start -->

## References

[^cloud-tpu-v5p]: [Google Cloud TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p). specification. Generation specifications, topology and host configurations. URL checked 2026-10-03.

[^cloud-tpu-7x]: [Google Cloud TPU 7x](https://docs.cloud.google.com/tpu/docs/tpu7x). specification. Generation specifications, topology and host configurations. Includes cross-generation comparison figures. URL checked 2026-10-03.

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^openxla-tiled-layout]: [OpenXLA tiled layout](https://github.com/openxla/xla/blob/main/docs/tiled_layout.md). compiler-reference. Tiling and padding; not generation-specific physical capacities. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

<!-- sources:generated:end -->
