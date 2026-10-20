---
title: Cloud TPU Capacity Models and Queued Resource Management
type: concept
tags:
- cloud-tpu
- gke
created: '2026-10-03'
human_reviewed: false
source_ids:
- gke-tpu-concepts
- cloud-tpu-queued-resources
- cloud-tpu-spot
generation_recipe: overview.recipe.json
---

# Cloud TPU Capacity Models and Queued Resource Management

Cloud TPU computing capacity can be provisioned through different purchasing models and managed via the Cloud TPU API or Google Kubernetes Engine (GKE). For current quotas, pricing, regional availability, and version support matrices, consult the official Google Cloud documentation.[^gke-tpu-concepts][^cloud-tpu-queued-resources][^cloud-tpu-spot]

## TPU Capacity Options

Workload requirements dictate the capacity model chosen:

- **On-demand**: The default consumption model in GKE that provisions capacity without prior reservation.[^gke-tpu-concepts] Allocation relies on having sufficient on-demand quota, but Google Cloud does not guarantee that physical resources are immediately available.[^gke-tpu-concepts][^cloud-tpu-queued-resources] Once provisioned, on-demand TPUs are not subject to preemption.[^cloud-tpu-queued-resources]
- **Reservations**: Capacity arranged in advance for predictable access.[^gke-tpu-concepts][^cloud-tpu-queued-resources] GKE distinguishes calendar-mode future reservations from long-term reservations.[^gke-tpu-concepts] The Cloud TPU API allows queued resources to allocate against established reservations.[^cloud-tpu-queued-resources]
- **Spot VMs**: Highly discounted, ephemeral instances allocated from excess capacity requiring preemptible quota.[^cloud-tpu-spot] They are eligible for preemption at any time when compute demand shifts.[^cloud-tpu-queued-resources][^cloud-tpu-spot] Warning and shutdown behavior depends on the provisioning interface.[^gke-tpu-concepts] In the Cloud TPU API, Spot VMs cannot be restarted after preemption (they must be recreated),.[^cloud-tpu-spot] Check current documentation for restrictions on combining Spot with scheduling features.[^gke-tpu-concepts]
- **Flex-start**: A GKE provisioning mode allowing hardware allocation on a best-effort basis for a bounded run window.[^gke-tpu-concepts]

## Cloud TPU API Queued Resource Lifecycle

The Cloud TPU API enables queued resource requests to acquire capacity as it becomes free, holding quota for the duration of the request regardless of state.[^cloud-tpu-queued-resources] Requests can include time bounds or durations specifying when allocation may begin or when an unfulfilled request expires.[^cloud-tpu-queued-resources]

A queued resource transitions through the following lifecycle states:

- `WAITING_FOR_RESOURCES`: The request passed validation and sits in the queue awaiting sufficient physical hardware or the expiration of its allocation interval.[^cloud-tpu-queued-resources]
- `PROVISIONING`: Hardware has been chosen from the queue and resources are actively allocating.[^cloud-tpu-queued-resources]
- `ACTIVE`: The allocation succeeded, and the provisioned TPU VM is ready for management.[^cloud-tpu-queued-resources]
- `FAILED`: Allocation failed due to specification errors or capacity could not be secured before the specified allocation interval elapsed. The record stays in the queue until deleted.[^cloud-tpu-queued-resources]
- `SUSPENDING`: The underlying TPU resources are being deleted.[^cloud-tpu-queued-resources]
- `SUSPENDED`: The TPU hardware is deleted and can no longer receive allocations, permitting safe deletion of the queued resource record.[^cloud-tpu-queued-resources]

## GKE vs. Cloud TPU API Mechanisms

Management approaches and operational units differ significantly across interfaces:

The Cloud TPU API is maintained solely for bug and security fixes and does not support Ironwood (TPU7x) or later generations.[^cloud-tpu-queued-resources][^cloud-tpu-spot] Modern workloads instead use Compute Engine or GKE.[^cloud-tpu-queued-resources]

In GKE, Kubernetes manages TPUs using node pools.[^gke-tpu-concepts] Single-host slice node pools contain independent VMs, whereas multi-host slice node pools connect TPU VMs over high-speed Inter-Chip Interconnects (ICI).[^gke-tpu-concepts] Multi-host slices operate atomically: if any single node terminates, fails, or requires repair, GKE evicts all workload Pods and shuts down or recreates the entire slice together.[^gke-tpu-concepts]

TPU scheduling in GKE uses `google.com/tpu` requests counted in chips. The concept guide describes whole-node allocation; verify hardware-specific allocation and container-sharing rules in the deployment documentation rather than treating this as a universal restriction.[^gke-tpu-concepts] See [GKE deployment](../deployment/overview.md).

<!-- sources:generated:start -->

## References

[^gke-tpu-concepts]: [Google Cloud: About TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpus). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

[^cloud-tpu-queued-resources]: [Google Cloud: TPU queued resources](https://docs.cloud.google.com/tpu/docs/queued-resources). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

[^cloud-tpu-spot]: [Google Cloud: Spot TPUs](https://docs.cloud.google.com/tpu/docs/spot). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
