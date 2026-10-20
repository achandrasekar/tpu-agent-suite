---
title: TPU Concepts, Topologies, and Planning on GKE
type: concept
tags:
- cloud-tpu
- gke
created: '2026-10-03'
human_reviewed: false
source_ids:
- gke-tpu-concepts
- gke-plan-tpus
generation_recipe: overview.recipe.json
---

# TPU Concepts, Topologies, and Planning on GKE

## Core Architecture and Resource Semantics

* **TPU Chips and Hosts**: A TPU host is a single virtual machine (VM) represented in Kubernetes as a TPU slice node [^gke-tpu-concepts]. Each TPU VM can host up to 8 TPU chips, depending on the machine type [^gke-tpu-concepts][^gke-plan-tpus].
* **TPU Slice**: A collection of chips within the same TPU Pod linked by high-speed inter-chip interconnects (ICI) [^gke-tpu-concepts]. Slices are categorized by chip or TensorCore counts depending on the generation [^gke-tpu-concepts].
* **TPU Slice Node Pool**: A group of Kubernetes nodes sharing identical TPU configurations [^gke-tpu-concepts].
* **Resource Semantics**: TPU resources differ between Kubernetes manifests and Compute Engine machine selections. In Kubernetes Pod specs, TPUs are requested as an integer count of physical chips using the key `google.com/tpu` [^gke-tpu-concepts]. The concept guide describes whole-node allocation; verify hardware-specific chip-allocation and container-sharing rules in the deployment documentation [^gke-tpu-concepts].

```yaml
# Short resource request example
resources:
  requests:
    google.com/tpu: NUMBER_OF_CHIPS # Represents integer physical chips on the host
  limits:
    google.com/tpu: NUMBER_OF_CHIPS
```
[^gke-tpu-concepts]

## Single-Host vs. Multi-Host Deployments

The choice between single-host and multi-host deployment is governed by model scale and total required chips [^gke-tpu-concepts][^gke-plan-tpus].

| Attribute | Single-Host TPU Slice | Multi-Host TPU Slice |
| :--- | :--- | :--- |
| **Host VM Count** | 1 host VM per slice; node pools contain 1 or more independent VMs [^gke-tpu-concepts][^gke-plan-tpus]. | 2 or more interconnected host VMs [^gke-tpu-concepts]. |
| **Interconnects** | VMs communicate via Data Center Network (DCN); TPU chips are not cross-connected across hosts [^gke-tpu-concepts]. | All TPU chips are connected across hosts via dedicated optical ICI links [^gke-tpu-concepts]. |
| **Failure Domain** | Independent per host VM [^gke-tpu-concepts]. | **Atomic**: If any VM fails or cannot be provisioned, the entire slice is torn down and rescheduled as a whole [^gke-tpu-concepts]. |
| **Autoscaling** | Incremental: Cluster Autoscaler adds or removes individual nodes [^gke-tpu-concepts][^gke-plan-tpus]. | Atomic: Node pools scale strictly between 0 and the exact total nodes required by the topology [^gke-tpu-concepts][^gke-plan-tpus]. |
| **Recommended Use** | Small-scale models and lightweight inference [^gke-tpu-concepts]. | Large-scale training and high-throughput inference [^gke-tpu-concepts]. |

Multi-host slice node pools are immutable; adding individual nodes manually after creation is unsupported [^gke-tpu-concepts].

## Topologies and Hardware Generations

Topology describes the physical dimensional arrangement of TPU chips in a slice [^gke-tpu-concepts]. If a chosen topology requires more chips than can fit on a single host VM, GKE automatically forms a multi-host slice [^gke-plan-tpus].

* **2D Topologies (`{A}x{B}`)**: Utilized by TPU v5e and Trillium (v6e) [^gke-tpu-concepts][^gke-plan-tpus].
* **3D Topologies (`{A}x{B}x{C}`)**: Utilized by TPU v4, v5p, and Ironwood (TPU7x) [^gke-tpu-concepts][^gke-plan-tpus]. A `4x4x4` arrangement forms a 64-chip building block known as a TPU cube [^gke-tpu-concepts]. For topologies exceeding 64 chips, dimensions must be multiples of 4 and follow the non-decreasing pattern $A \le B \le C$ [^gke-tpu-concepts][^gke-plan-tpus].
* **Generation-Specific Features**:
  * *Collection Scheduling*: TPU Trillium exclusively supports collection scheduling to group single-host or multi-host slice node pools into resilient sets for inference workloads [^gke-tpu-concepts][^gke-plan-tpus].
  * *Interconnect Resiliency*: Select hardware uses ICI resiliency features to handle optical switch and link faults between TPU cubes [^gke-tpu-concepts].

## Planning and Operational Decision Criteria

1. **GKE Operating Mode**: GKE Autopilot manages node provisioning, security baselines, and topology placement from workload requests [^gke-plan-tpus]. GKE Standard exposes node-pool configuration and machine-type choices, plus at least one non-TPU pool for cluster-critical system pods [^gke-tpu-concepts][^gke-plan-tpus].
2. **Consumption Models**: Select from Flex-start, preemption-prone Spot VMs, calendar-mode future reservations, long-term reservations, or default on-demand capacity [^gke-tpu-concepts][^gke-plan-tpus].
3. **Zonal Placement**: Check current availability for the selected hardware and deployment mode before provisioning.[^gke-plan-tpus]
4. **Disruption Mitigation**: Multi-host training workloads face total job restart if any node is disrupted [^gke-tpu-concepts][^gke-plan-tpus]. Mitigate interruptions by setting high Kubernetes PriorityClasses, utilizing maintenance exclusions, queuing workloads with Kueue, and maintaining persistent checkpoints [^gke-tpu-concepts][^gke-plan-tpus].

Refer to current official Google Cloud documentation for exact quota limits, supported versions, pricing, and regional zone availability [^gke-tpu-concepts][^gke-plan-tpus].

<!-- sources:generated:start -->

## References

[^gke-tpu-concepts]: [Google Cloud: About TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpus). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

[^gke-plan-tpus]: [Google Cloud: Plan TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
