---
title: TPU 8i Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- tpu-8th-gen
generation_recipe: overview.recipe.json
---

# TPU 8i Hardware Overview

Google's TPU 8i is a custom-engineered, eighth-generation Tensor Processing Unit built specifically for low-latency, high-throughput AI inference and collaborative agentic workloads [^tpu-8th-gen]. Co-designed alongside Google DeepMind, the chip pairs high-capacity memory with specialized network routing to target frontier reasoning models, continuous execution loops, and Mixture of Experts (MoE) architectures [^tpu-8th-gen].

## Chip Compute and Architecture

TPU 8i integrates compute and network connectivity directly onto the same silicon die to reduce inter-chip data movement power overheads [^tpu-8th-gen].

* **Collectives Acceleration Engine (CAE):** TPU 8i incorporates a newly designed on-chip Collectives Acceleration Engine that offloads global collective communications, cutting on-chip latency by up to 5x to eliminate processing stalls and lag [^tpu-8th-gen].
* **Workload Specialization:** While architected to run diverse models, the hardware logic is tuned to optimize memory bandwidth and reduce inference latency [^tpu-8th-gen].
* **Efficiency and Power Management:** The platform implements real-time dynamic power management to modulate power draw based on active workload demands [^tpu-8th-gen]. Combined with Google's fourth-generation liquid cooling distribution units, the system yields up to a 2x performance-per-watt increase and an 80% performance-per-dollar improvement over previous-generation systems [^tpu-8th-gen].

| Metric | Specification | Notes / Qualification |
| :--- | :--- | :--- |
| **Primary Workload Target** | High-speed, latency-sensitive inference | Vendor-targeted for agentic reasoning flows [^tpu-8th-gen] |
| **On-Chip Collectives Offload** | Collectives Acceleration Engine (CAE) | Reduces on-chip latency by up to 5x [^tpu-8th-gen] |
| **Cooling Technology** | 4th-generation liquid cooling | Employs liquid distribution units [^tpu-8th-gen] |
| **Vendor Efficiency Metrics** | 80% better perf/, up to 2x perf/W | Vendor claims relative to preceding hardware [^tpu-8th-gen] |

## Memory Hierarchy and Buffer Limits

To avoid processor stalls and mitigate the memory wall, TPU 8i pairs external High Bandwidth Memory (HBM) with on-chip static RAM (SRAM) sized specifically to hold the active working sets and key-value (KV) caches of production reasoning models [^tpu-8th-gen].

* **Physical SRAM:** 384 MB of on-chip physical SRAM per chip (a 3x increase over the preceding generation) [^tpu-8th-gen]. The source material designates this strictly as physical on-chip SRAM; it does not disclose software-addressable Vector Memory (VMEM) partition boundaries or software-scoped buffer capacities.
* **High-Bandwidth Memory (HBM):** 288 GB per chip [^tpu-8th-gen]. The vendor claims this capacity keeps active reasoning model working sets directly on-chip and in attached accelerator memory without swapping bottlenecks [^tpu-8th-gen].

## Interconnect and Pod Scale

TPU 8i utilizes custom Interconnect (ICI) links arranged in a multi-tier hierarchical "Boardfly" network topology [^tpu-8th-gen].

* **Interconnect Bandwidth:** Delivers 19.2 Tb/s of ICI bandwidth per chip to support distributed all-to-all communications in MoE models [^tpu-8th-gen].
* **Boardfly Architecture:** Designed to minimize network hops, Boardfly lowers the maximum network diameter by more than 50% compared to earlier generation layouts [^tpu-8th-gen].
* **Hierarchical Topology Structure:**
  1. *Building Block:* 4 fully connected TPU 8i chips [^tpu-8th-gen].
  2. *Group Tier:* 8 boards connected to form a fully connected group [^tpu-8th-gen].
  3. *Pod Scale:* 36 of these fully connected groups are linked together into a full TPU 8i pod [^tpu-8th-gen].
* **Derived Pod Sizing:** At 4 chips per building block (board) and 8 boards per group, each group aggregates 32 chips (derived: 4 \times 8). A full pod of 36 fully connected groups aggregates 1,152 TPU 8i chips (derived: 36 \times 32$).

## Slice and Host Configurations

The host infrastructure uses Google's custom Axion Arm-based server processors [^tpu-8th-gen].

* **Host Integration:** The physical server architecture doubles the number of physical CPU hosts per server using custom Axion Arm CPUs [^tpu-8th-gen].
* **Memory Architecture:** Server memory utilizes a Non-Uniform Memory Architecture (NUMA) to isolate CPU-accelerator domains and optimize memory performance across host processing units [^tpu-8th-gen].
* **Bare-Metal Support:** Deployments offer native bare-metal hardware access, eliminating the virtualization overhead of standard hypervisors [^tpu-8th-gen].

## Software and Operational Constraints

TPU 8i is deployed through Google's AI Hypercomputer software stack [^tpu-8th-gen].

* **Supported Frameworks:** Provides native integration for JAX, PyTorch, and MaxText [^tpu-8th-gen].
* **Serving Engines:** Supports distributed inference serving engines including vLLM and SGLang [^tpu-8th-gen].
* **Tooling and Libraries:** Integrates MaxText reference models and the Tunix library for reinforcement learning optimization workflows [^tpu-8th-gen].
* **Availability Status:** Announced as an upcoming offering, slated for general availability "later this year" [^tpu-8th-gen]. Interested customers must request access directly through Google Cloud channels [^tpu-8th-gen].

## Gaps and Unspecified Specifications

Because the TPU 8i hardware platform is documented via initial announcement disclosures, several detailed architectural parameters have not yet been made public:

* **Compute Pipeline:** Specific clock frequencies, the total number and microarchitecture of Matrix Multiply Units (MXUs) or TensorCores, SparseCore integration details, and raw arithmetic compute capacities (such as TFLOPS/PFLOPS across FP8, BF16, or INT8) are unstated [^tpu-8th-gen].
* **Memory Specifications:** Physical HBM generation standards (e.g., HBM3 or HBM3e), absolute memory throughput in TB/s, channel widths, and usable scoped VMEM vs. physical SRAM partitioning figures are omitted [^tpu-8th-gen].
* **Host and Virtual Machine Topology:** The precise virtual machine (VM) machine type identifiers, exact vCPU counts, host system RAM allocations, and specific sliced sub-pod topology coordinates/dimensions supported by Google Cloud have not been published [^tpu-8th-gen].
* **Data Center Interconnect:** External Data Center Network (DCN) interface bandwidth, NIC provisioning per host, and cluster-scale optical switching capabilities outside of the ICI fabric remain uncharacterized [^tpu-8th-gen].

<!-- sources:generated:start -->

## References

[^tpu-8th-gen]: [Google eighth-generation TPU announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/). announcement. Vendor claims for TPU 8t and 8i; not a specification sheet. URL checked 2026-10-03.

<!-- sources:generated:end -->
