---
title: Google TPU 8t Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- tpu-8th-gen
generation_recipe: overview.recipe.json
---

# Google TPU 8t Hardware Overview

Google TPU 8t is an announced eighth-generation Tensor Processing Unit custom-designed for massive, compute-intensive AI model training [^tpu-8th-gen]. TPU 8t is positioned to reduce frontier model development cycles by combining high compute throughput, shared memory pools, scale-up interconnect, and hardware reliability enhancements [^tpu-8th-gen].

> **Announcement Notice**: TPU 8t is an announced product scheduled for general availability "later this year" [^tpu-8th-gen]. Detailed architectural specifications not present in vendor disclosure materials remain unknown [^tpu-8th-gen].

---

## Chip Compute and Architecture

TPU 8t co-locates network connectivity and compute logic on the same chip to reduce data transfer power within pods [^tpu-8th-gen]. Silicon-level specifications—such as Matrix Multiply Units (MXUs), TensorCore counts, vector units, core arrangements, physical clock frequencies, and custom numeric precision formats—have not been disclosed [^tpu-8th-gen]. SparseCore components are not mentioned for this generation [^tpu-8th-gen].

At system scale, Google advertises the compute capacity of a single TPU 8t superpod at 121 ExaFlops, claiming nearly 3× the compute performance per pod over the prior generation [^tpu-8th-gen]. The precise numerical precision (e.g., FP8, BF16, or FP16) used to calculate the 121 ExaFlops metric is unspecified [^tpu-8th-gen]. Dynamic power management integrated across the stack modulates power draw according to real-time workload demand, yielding up to 2× higher performance-per-watt over the preceding Ironwood generation [^tpu-8th-gen]. Heat dissipation is sustained by Google's fourth-generation liquid cooling distribution technology [^tpu-8th-gen].

---

## Memory Hierarchy and Buffer Limits

Detailed per-chip on-chip memory limits (such as VMEM, physical SRAM capacities, and scratchpad buffers) are unstated for TPU 8t [^tpu-8th-gen].

At the pod boundary, Google reports that a single 9,600-chip TPU 8t superpod shares an aggregated high-bandwidth memory (HBM) pool of two petabytes [^tpu-8th-gen]:

| Memory Metric | Announced Superpod Value | Derived Average per Chip |
| :--- | :--- | :--- |
| High Bandwidth Memory (HBM) | 2 PB [^tpu-8th-gen] | ~208.33 GB *(derived)* |
| On-Chip SRAM / Scoped VMEM | Not specified for 8t [^tpu-8th-gen] | Unannounced [^tpu-8th-gen] |
| HBM Memory Bandwidth | Not specified for 8t [^tpu-8th-gen] | Unannounced [^tpu-8th-gen] |

To alleviate external ingestion bottlenecks, TPU 8t integrates 10× faster storage access coupled with TPUDirect, streaming training data directly into the TPU [^tpu-8th-gen].

---

## Interconnect, Pod Scale, and Topology

TPU 8t systems scale across dual networking layers: Inter-Chip Interconnect (ICI) fabrics within the pod and the multi-pod Virgo Network [^tpu-8th-gen].

* **Inter-Chip Interconnect (ICI)**: TPU 8t provides double the interchip bandwidth of the prior generation [^tpu-8th-gen]. Specific per-chip physical lane counts, signaling rates, and physical multidimensional mesh/torus dimensions are unstated [^tpu-8th-gen].
* **Virgo Network**: A new network architecture whose bandwidth targets were designed to meet trillion-parameter model parallelism requirements [^tpu-8th-gen]. Google claims near-linear scaling up to 1,000,000 chips inside a single logical cluster when combining the Virgo Network with JAX and Pathways [^tpu-8th-gen].
* **Pod Capacity**: A single TPU 8t superpod aggregates 9,600 chips into a unified pool [^tpu-8th-gen].
* **Reliability, Availability, and Serviceability (RAS)**: The system targets >97% productive compute time ("goodput") [^tpu-8th-gen]. RAS capabilities include:
  * Fleet-wide real-time telemetry tracking tens of thousands of processors [^tpu-8th-gen].
  * Automated fault detection and real-time rerouting around degraded or broken ICI links without job disruption [^tpu-8th-gen].
  * Optical Circuit Switching (OCS) infrastructure that automatically reconfigures optical paths around failed hardware without manual intervention [^tpu-8th-gen].

---

## Slice and Host Configurations

The announcement does not establish 8t-specific CPU architecture, NUMA configuration or host resource counts [^tpu-8th-gen].

| Attribute | Specification Status / Value |
| :--- | :--- |
| Host CPU Architecture | Not specified for 8t [^tpu-8th-gen] |
| Host Memory Architecture | Not specified for 8t [^tpu-8th-gen] |
| Host Machine Types / VM SKUs | Not specified [^tpu-8th-gen] |
| Host vCPU & DRAM Allocations | Not specified [^tpu-8th-gen] |
| Host-to-Accelerator Ratio | Not specified for 8t [^tpu-8th-gen] |
| Hardware Access Models | Bare metal access (bypasses virtualization overhead) [^tpu-8th-gen] |
| Supported Slice Geometries | Pod scaling at 9,600 chips; intermediate slice steps unlisted [^tpu-8th-gen] |

---

## Software and Operational Constraints

TPU 8t runs within Google's AI Hypercomputer ecosystem, spanning orchestration, delivery, and hardware-software co-design [^tpu-8th-gen].

* **Frameworks and Engines**: Native execution support is available for JAX, PyTorch, and MaxText [^tpu-8th-gen]. Serving runtimes such as SGLang and vLLM are also supported across eighth-generation architectures [^tpu-8th-gen].
* **Distributed Runtime and Optimizers**: Incorporates Pathways software alongside JAX for wide-scale distribution [^tpu-8th-gen]. MaxText provides reference implementations, while Tunix provides reinforcement learning pipelines [^tpu-8th-gen].
* **Execution Boundary**: Workloads can deploy natively on bare metal instances to avoid hypervisor virtualization overhead [^tpu-8th-gen].

---

## Architecture Gaps and Unannounced Specifications

Because the TPU 8t is an announced, upcoming hardware architecture, multiple key engineering specifications remain undisclosed [^tpu-8th-gen]:

* **Compute Subsystem**: Die size, process node, MXU/TensorCore configuration, vector engine ALU layout, SparseCore presence, supported numeric formats (such as FP8, FP4, or INT8 capabilities), and base/boost operational clock frequencies [^tpu-8th-gen].
* **Memory Specifications**: Physical on-chip SRAM capacity, usable scoped VMEM, HBM generation (e.g., HBM3, HBM3e), per-chip memory bus interface width, and per-chip memory bandwidth [^tpu-8th-gen].
* **Interconnect Topography**: Exact per-chip ICI bidirectional throughput in Tb/s (noted only as 2× the prior generation), ICI dimensional topology (e.g., 3D torus, Boardfly, or crossbar), and optical-to-electrical conversion boundaries [^tpu-8th-gen].
* **Host Platform**: Physical CPU socket count per host tray, CPU architecture and core/vCPU count, physical host DDR5 memory sizing, and host PCIe/interconnect bus generations [^tpu-8th-gen].
* **Commercialization**: Exact dates for General Availability (GA), Google Cloud Platform instance names/SKUs, pricing models, and regional data center rollouts [^tpu-8th-gen].

<!-- sources:generated:start -->

## References

[^tpu-8th-gen]: [Google eighth-generation TPU announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/). announcement. Vendor claims for TPU 8t and 8i; not a specification sheet. URL checked 2026-10-03.

<!-- sources:generated:end -->
