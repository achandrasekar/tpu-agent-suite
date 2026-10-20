# TPU v7x (Ironwood / GhostFish) Specification

This document details the hardware specifications and architectural highlights of Google's seventh-generation Tensor Processing Unit, codenamed **Ironwood** (system vendor codename **GhostFish**).

---

## 1. Overview & Architecture

TPU v7x is Google's flagship seventh-generation accelerator, designed for large-scale dense and Mixture-of-Experts (MoE) foundation model training and serving.

### Terminology Definitions
*   **TPU Chip (Physical Package):** The physical silicon package. It consists of **two compute chiplets** packaged together on a silicon interposer, sharing a total of 192 GB HBM.
*   **TPU Chiplet (Silicon Die / Logical Core):** The independent physical compute block inside the chip. Each chiplet contains 1 TensorCore, 2 SparseCores, and has local access to 96 GB of HBM. **A TPU Chiplet corresponds directly to a single logical device (e.g., in JAX, 1 Chiplet = 1 Device).**
*   **Logical Core (JAX Device):** JAX/XLA exposes each chiplet as a separate logical device. Thus, **1 physical TPU Chip = 2 Chiplets = 2 Logical Cores (JAX Devices)**. For compilation, execution, and data-sharding purposes, a single Chiplet is treated as a **single, discrete device** (e.g., `jax.devices()[i]`).

### Dual-Chiplet Design
*   **Reticle Limit Breakthrough:** Rather than a single massive monolithic die which suffers from low yield due to defects at advanced nodes, TPU7x splits the package into **two smaller compute chiplets**.
*   **Intra-Package Interconnect:** The two chiplets inside the same package communicate via an ultra-high-density, high-speed Die-to-Die (D2D) interface.
*   **Software Exposure:** Previous TPU generations fused cores to expose them as a single device. TPU7x exposes the two chiplets as **two separate logical devices** to JAX/XLA, allowing finer-grained parallelism. Each chiplet behaves as a **single device** from the compiler and framework's perspective.
*   **NUMA Locality:** On the host CPU, the architecture is split into a dual NUMA node topology. CPU, memory, and NICs are divided equally between the two NUMA nodes, requiring multi-container setups to prevent cross-NUMA latency overhead.

---

## 2. GhostFish (P3) Derated Specifications

This section defines the **derated (P3)** specification, which is the baseline configuration currently used in this project's simulator and validation benchmarks.

*   **Peak Compute:**
    *   **Per Chiplet (Core, Single Device):** 1,785.6 TFLOPS (FP4/FP8), 892.8 TFLOPS (BF16)
*   **HBM Memory Bandwidth:**
    *   **Per Chiplet (Core, Single Device):** 3,364.9 GB/s
*   **HBM Capacity:**
    *   **Per Chiplet (Core, Single Device):** 96.0 GB
*   **Interconnect Bandwidth:**
    *   **Intra-Chip (Chiplet-to-Chiplet / D2D):** 600.0 GB/s (unidirectional)
    *   **Inter-Chip (Chiplet-to-ICI Network):** 80.0 GB/s (unidirectional)
    *   **Chiplet-to-Chiplet Latency (Inter-Chip):** 0.0013 ms
*   **MXU Shape:** 2.0 TensorCores per chip (1.0 TensorCore per chiplet), Min MXU Contracting Dim: 256
*   **SparseCore:** 4.0 SparseCores per chip (2.0 SparseCores per chiplet) — see [§4](#4-sparsecore)
*   **Data Center Network (DCN):** 100 Gbps bandwidth per chip
*   **Host VM Specs (Standard 4-chip VM / 8-chiplet VM):**
    *   **vCPUs:** 224
    *   **Host RAM:** 960 GB
    *   **NUMA Nodes:** 2 NUMA nodes per VM (CPU, memory, and NICs are equally split between nodes; multi-container setup is recommended to avoid cross-NUMA latency)

---

## 3. Interconnect & Communication Link Calculation (`# links` / `N_links`)

When calculating collective communication execution times (as specified in `COLLECTIVE.md`), the accessible number of independent communication links (`N_links`) in TPU v7x is determined by its hierarchical, dual-chiplet package architecture:

### 1. Physical Link Availability (Package Level)
Each **physical TPU v7x Package (Chip)** has exactly **6 independent bidirectional ICI links** connecting it to the external 3D torus network (typically mapping to physical network directions $X+, X-, Y+, Y-, Z+, Z-$).

### 2. Logical Link Mapping (JAX Device / Chiplet Level)
Since each physical package is partitioned into **2 discrete logical JAX devices** (Chiplets), they exhibit specialized networking behaviors depending on your sharding mapping:

*   **Intra-Package Transfers (Between Chiplet 0 and Chiplet 1)**: Communicates entirely via the highly advanced internal Die-to-Die (D2D) interface, completely bypassing external ICI links. For performance simulation, this is modeled as **1 ultra-wide internal link** ($N_{\text{links}} = 1$) with an exceptional dedicated unidirectional bandwidth of **600 GB/s**. Because each core has its own dedicated send and receive links, between Core-to-Core bi-directional BW can be used simultaneously.
*   **Inter-Package Transfers (Between completely separate physical packages)**: For traffic exiting the physical silicon package, the 2 logical JAX devices share the 6 external physical communication links via a unified on-chip routing layer (InterChip Router / CMN).
    *   **Ring-based Algorithms (e.g., Ring All-Reduce)**: When constructing a communication ring across independent physical packages, a participating JAX device effectively utilizes **2 external physical links** (1 external send link + 1 external receive link in the active dimension).
    *   **Bandwidth Contention Derating**: If both logical JAX devices within the exact same physical package execute simultaneous, non-overlapping external communications that route across the identical external physical link, the effective multi-link bandwidth per device must be derated by $50\%$ to account for physical wire sharing.

---

## 4. SparseCore

| Metric | Symbol | Value | Unit / Note |
| --- | --- | --- | --- |
| SparseCores per Chiplet | $N_{\text{SC}}$ | 2 | 4 per chip |
| Vector Subcores per SparseCore | $N_{\text{sub}}$ | 16 | 32 per chiplet; one wave runs all 32 in lockstep |
| Lanes per Vector Subcore | $N_{\text{lane}}^{\text{SC}}$ | 16 | |
| VMEM per Vector Subcore | $\text{VMEM}_{\text{sub}}$ | 512 KiB | per subcore, **not poolable** |
| Pooled VMEM per SparseCore | $\text{VMEM}_{\text{SC}}$ | 8 MiB | 2,097,152 words; the working-set denominator |
| VMEM Banks | $N_{\text{bank}}$ | 32 | a per-lane stride sharing a factor with this serializes |
| DMA Granule | $G_{\text{DMA}}$ | 32 | Bytes; sets the gather cost floor |
| Peak BW, one SparseCore, unidirectional | $\text{BW}_{\text{SC}}$ | $1.70 \times 10^{12}$ | Bytes/s (1,700 GB/s); a CMN **command** rate |
| Peak BW, one SparseCore, read + write | $\text{BW}_{\text{SC}}^{\text{rw}}$ | $2.626 \times 10^{12}$ | Bytes/s (2,626 GB/s); budget the **sum** of both streams |
| Achieved BW, streaming | | $3.38 \times 10^{11}$ | Bytes/s (338 GB/s) per chiplet, **aggregate over all 32 subcores** |
| Achieved BW, 640 B indirect rows | | $5.09$–$7.09 \times 10^{11}$ | Bytes/s (509–709 GB/s) |
| Gather Saturation Row Width | $w_{\text{sat}}$ | 64 | Bytes (16 f32); ~230 GB/s beyond it |
| Gather Crossover Row Width | $w_{\text{xover}}$ | 1,024 | Bytes (256 f32); above this XLA wins |
| Per-call Intercept | $t_0^{\text{SC}}$ | 15.22 | µs, fitted; ~half the TensorCore figure |
| Wave Cost Factor | | $\tfrac{32}{33} \approx 0.97$ | $\mathbb{E}[\max]$ over 32 lockstep rows, $r \sim U(0,n)$ |

