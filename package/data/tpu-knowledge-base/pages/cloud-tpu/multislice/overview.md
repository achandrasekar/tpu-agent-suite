---
title: Cloud TPU Multislice Overview
type: concept
tags:
- cloud-tpu
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-multislice
generation_recipe: overview.recipe.json
---

# Cloud TPU Multislice Overview

Cloud TPU Multislice is a performance-scaling technology that links multiple TPU slices within a single pod or across multiple pods, scaling ML training beyond single-slice physical constraints [^cloud-tpu-multislice]. Multislice is supported only on Cloud TPU v4 and newer generations [^cloud-tpu-multislice].

## Slices, Pods, and Interconnects

### Slices vs. Pods
A TPU Pod is a large physical installation of interconnected TPU hardware [^cloud-tpu-multislice]. A slice is a contiguous set of TPU chips within a Pod that communicate directly [^cloud-tpu-multislice]. In the Cloud TPU API Queued Resource (QR) context, each slice is assigned a node ID [^cloud-tpu-multislice]. Multislice scales beyond one slice or combines smaller slices when that improves provisioning opportunities.[^cloud-tpu-multislice] Slices within a Multislice configuration must be homogeneous; heterogeneous slice shapes are not supported [^cloud-tpu-multislice].

### ICI vs. DCN
Communication within and between slices uses two distinct networking layers:

* **Inter-Chip Interconnect (ICI):** High-speed, low-latency direct physical links that connect chips inside the same TPU slice [^cloud-tpu-multislice].
* **Data Center Networking (DCN):** A lower-throughput, higher-latency network that links distinct slices [^cloud-tpu-multislice]. Cross-slice TPU communication requires chips to send data to their host CPUs, which then transmit payloads across standard DCN infrastructure [^cloud-tpu-multislice]. The XLA compiler automatically generates inter-slice DCN communication routines and overlaps this transfer with computation [^cloud-tpu-multislice].

## Parallelism Placement

Workload performance depends on mapping parallel execution strategies to the appropriate network tiers:

* **Intra-Slice (ICI):** Best suited for fine-grained, high-bandwidth techniques such as Fully Sharded Data Parallelism (FSDP) and tensor parallelism [^cloud-tpu-multislice]. ICI is fast enough that pure data parallelism is rarely needed within a single slice [^cloud-tpu-multislice].
* **Inter-Slice (DCN):** Mapped across DCN via hybrid device meshes (such as JAX hybrid device mesh utilities) [^cloud-tpu-multislice].
  * **Data Parallelism (DP):** The primary recommended cross-slice strategy [^cloud-tpu-multislice]. DCN all-reduce durations must not exceed backward-pass compute times to maintain scaling efficiency [^cloud-tpu-multislice].
  * **FSDP:** Reduces per-slice memory pressure by sharding weights and optimizer states across slices, but increases DCN communication [^cloud-tpu-multislice].
  * **Pipeline Parallelism:** Used when global batch size constraints restrict pure DP or FSDP scaling [^cloud-tpu-multislice]. Because it incurs pipeline bubbles and requires micro-batching, it should typically be combined with DP or FSDP [^cloud-tpu-multislice].
  * **Tensor Parallelism:** Excluded from inter-slice placement (`dcn_tensor_parallelism` should remain 1) because DCN bandwidth is insufficient for model activation sharding [^cloud-tpu-multislice].

## Coordinated Training and Failure Recovery

Multislice training uses gang scheduling to allocate all requested slices at the same moment; the provisioning either succeeds completely or not at all [^cloud-tpu-multislice].

Because large distributed training runs must survive interruptions, Cloud TPU offers automated failure recovery:
1. **Auto-Repair:** If a slice suffers preemption, maintenance, or hardware failure, Cloud TPU recreates the damaged slice (or waits until hardware is available) [^cloud-tpu-multislice].
2. **Coordinated Resets:** Once the failed slice is replaced, all remaining healthy slices in the Multislice deployment are automatically restarted or reset to synchronize execution [^cloud-tpu-multislice].
3. **Resumption:** Workloads resume through a VM startup script configured in the Queued Resource payload, which must relaunch the application and restore state from a checkpoint (for example, in Cloud Storage via Orbax) [^cloud-tpu-multislice].

## Resource Management Semantics

Google recommends orchestrating TPU Multislices with Google Kubernetes Engine (GKE) [^cloud-tpu-multislice]. The legacy Cloud TPU API manages Multislice allocations via Queued Resources (`queued-resources`), provisioning instances across reserved, spot, or on-demand capacity [^cloud-tpu-multislice].

Queued Resource startup scripts can relaunch training after coordinated recovery; the application must restore its checkpoint.[^cloud-tpu-multislice]

For up-to-date regional availability, pricing models, supported-version matrices, and GKE version requirements, consult the official Google Cloud documentation [^cloud-tpu-multislice].

<!-- sources:generated:start -->

## References

[^cloud-tpu-multislice]: [Google Cloud: Multislice introduction](https://docs.cloud.google.com/tpu/docs/multislice-introduction). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
