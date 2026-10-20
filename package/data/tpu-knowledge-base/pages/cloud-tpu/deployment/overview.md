---
title: TPU Workload Scheduling on GKE
type: concept
tags:
- cloud-tpu
- gke
created: '2026-10-03'
human_reviewed: false
source_ids:
- tpu-on-gke
generation_recipe: overview.recipe.json
---

# TPU Workload Scheduling on GKE

## Provisioning and Node Pools

TPU resources in GKE are provisioned through single-host or multi-host TPU slice node pools via manual creation, cluster-level node auto-provisioning, or custom `ComputeClasses` [^tpu-on-gke].

* **Single-host node pools:** Each VM operates independently. Node autoscaling adjusts dynamically between configured minimum and maximum limits [^tpu-on-gke]. Unhealthy nodes undergo individual auto-repair [^tpu-on-gke].
* **Multi-host node pools:** Multiple interconnected VMs form a physical accelerator slice [^tpu-on-gke]. Node pool sizing and autoscaling operate atomically: the pool scales directly between zero and its full capacity based on the topology's total chip count divided by chips per VM [^tpu-on-gke]. Upgrades and repairs recreate the entire multi-host slice as a single unit [^tpu-on-gke].

### Quota and Resource Semantics

Creating on-demand or Spot TPU slice node pools in GKE consumes Compute Engine API quota (`compute.googleapis.com`), which is distinct from Cloud TPU API quota (`tpu.googleapis.com`) [^tpu-on-gke]. Reserved instances draw from committed reservation capacity rather than unreserved on-demand quotas [^tpu-on-gke]. For specific quotas, pricing, and regional availability, consult current official documentation.

## Accelerator and Topology Selectors

Workloads bind to specific hardware configurations using node selectors in the Pod specification [^tpu-on-gke]:

* `cloud.google.com/gke-tpu-accelerator`: Identifies the TPU generation and machine slice type (e.g., `tpu-v4-podslice`, `tpu-v5-lite-podslice`, or `tpu7x-standard-4t`) [^tpu-on-gke].
* `cloud.google.com/gke-tpu-topology`: Defines the physical arrangement and count of TPU chips in the slice (e.g., `2x4` or `2x2x2`) [^tpu-on-gke].

## Chip Resource Requests and Limits

Containers request TPU access via the `google.com/tpu` resource in physical chip units [^tpu-on-gke]. The requested value must equal the resource limit [^tpu-on-gke]. Slices can be consumed by single containers or, on supported configurations like Ironwood, partitioned across multiple containers within a Pod [^tpu-on-gke]. Check current GKE version requirements before relying on a particular security-context behavior.[^tpu-on-gke]

## Node Taints and Isolation

When GKE provisions TPU nodes via automated mechanisms like `ComputeClasses`, it applies node taints alongside workload tolerations to isolate accelerator hardware from standard workloads [^tpu-on-gke]. To avoid scheduling deadlocks between multiple TPU Jobs contending for the same topology, platforms use Pod anti-affinity or the `jobset.sigs.k8s.io/exclusive-topology` annotation to ensure a single workload exclusively secures the necessary node pool [^tpu-on-gke].

## Workload Structure: Single-Host vs. Multi-Host

* **Single-Host Workloads:** Implemented as individual Pods or standard Deployments [^tpu-on-gke]. The container communicates exclusively with the local chips installed on that host [^tpu-on-gke].
* **Multi-Host Workloads:** Deployed using multi-replica indexed Jobs or JobSets paired with a headless Kubernetes Service [^tpu-on-gke]. Pods communicate across TPU VMs using inter-host network ports (defaulting to container port `8471`), while port `8431` can be exposed to export runtime metrics [^tpu-on-gke].

## Sourced Pod Fragment

The following fragment illustrates a container requesting TPU chips via topology selectors and resource limits [^tpu-on-gke]:

```yaml
spec:
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: ACCELERATOR_TYPE
    cloud.google.com/gke-tpu-topology: TOPOLOGY
  containers:
  - name: tpu-container
    image: YOUR_WORKLOAD_IMAGE
    ports:
    - containerPort: 8471 # Inter-host communication port
    - containerPort: 8431 # Runtime metrics port
    resources:
      requests:
        google.com/tpu: CHIP_COUNT # Count represents physical TPU chips; requests must equal limits
      limits:
        google.com/tpu: CHIP_COUNT
```
[^tpu-on-gke]

<!-- sources:generated:start -->

## References

[^tpu-on-gke]: [Google Cloud: Deploy TPU workloads in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
