---
title: Storage Options for TPU Workloads
type: concept
tags:
- cloud-tpu
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-storage-options
generation_recipe: overview.recipe.json
---

# Storage Options for TPU Workloads

TPU VM workloads require data storage for preprocessing, host input pipelines, training inputs and outputs, checkpoints, model weights, and key-value (KV) cache offloading. Available storage choices include durable block storage, Google Cloud Storage, and Google Cloud Managed Lustre.[^cloud-tpu-storage-options]

## Durable Block Storage: Boot and Attached Disks

Attached durable block storage is created independently of a VM; retention on resource deletion depends on the disk and deletion policy. Keep recoverable checkpoints and backups separately from the TPU boot disk.[^cloud-tpu-storage-options]

* **Boot Disks:** Each TPU VM includes a boot disk containing the operating system, drivers, and libraries. The boot disk can be resized or customized to hold temporary datasets and model input/output data, provided capacity limits are not exceeded. However, boot disks do not support disk snapshots, and recovering lost or deleted data from them is difficult, necessitating external backups such as Cloud Storage buckets.[^cloud-tpu-storage-options]
* **Attached Disks:** Persistent Disk and Hyperdisk provide durable network-attached block storage created independently of the VM lifecycle. Hyperdisk provides higher IOPS, higher throughput limits, and customizable performance compared to Persistent Disk. Unlike boot disks, attached disks support incremental disk snapshots. Specific disk type support varies by TPU generation; refer to current official documentation for generation-specific compatibility matrices.[^cloud-tpu-storage-options]

## Shared Datasets and Multi-Host Considerations

Workloads utilizing multi-host TPU VM slices have specific concurrency and attachment constraints:[^cloud-tpu-storage-options]

* **Multi-Host Managed Instance Groups (MIGs):** When attaching a disk to a multi-host TPU slice within a MIG, the storage device is attached to each VM in the slice. To prevent data corruption from concurrent writes across hosts, attached disks on multi-host slices must be configured as read-only. This configuration makes them suitable for shared training datasets across hosts.[^cloud-tpu-storage-options]
* **Hyperdisk Restrictions:** Because Hyperdisk Balanced does not support read-only mode, it cannot be shared across multiple hosts in a slice and can only attach to a single TPU VM instance. Hyperdisk ML can be used when high-speed read access to read-only data is required on supported TPU generations.[^cloud-tpu-storage-options]

## Cloud Storage and Filesystem Access

Cloud Storage provides scalable, cost-efficient object storage designed for massive petabyte-scale training datasets (such as video, images, or text). Unlike block storage, it is not bound to a single zone and natively supports simultaneous multi-host and multi-zone reads and writes with built-in data redundancy and operation checksums.[^cloud-tpu-storage-options]

* **Rapid Bucket:** A zonal Cloud Storage class offering sub-millisecond open latency and high throughput, making it suitable for asynchronous checkpoints, training datasets, and model weights.[^cloud-tpu-storage-options]
* **Regional Buckets and Rapid Cache:** Regional buckets provide object storage within a selected region. For read-intensive workloads, such as dataset loading or inference weight loading, regional buckets can be paired with Rapid Cache—a zonal SSD-backed read cache that lowers latency and boosts throughput.[^cloud-tpu-storage-options]
* **Filesystem Integration:** Cloud Storage buckets can be mounted as local filesystems using Cloud Storage FUSE. In Google Kubernetes Engine (GKE) environments, workloads should utilize the Cloud Storage FUSE Container Storage Interface (CSI) driver and Cloud Storage FUSE profiles, distinguishing GKE storage orchestration semantics from direct TPU VM file operations.[^cloud-tpu-storage-options]

## Managed Lustre

Google Cloud Managed Lustre is a fully managed, POSIX-compliant parallel file system. It is optimized for sub-millisecond latency (under 1 ms) and high-concurrency metadata operations. Managed Lustre is best suited for workloads such as synchronous checkpointing, home directories, reinforcement learning high-speed weight propagation, KV cache offloading, and training datasets comprising many small files (under 1 MB). It also includes a Dynamic tier that automatically tunes performance based on access patterns.[^cloud-tpu-storage-options]

<!-- sources:generated:start -->

## References

[^cloud-tpu-storage-options]: [Google Cloud: TPU storage options](https://docs.cloud.google.com/tpu/docs/storage-options). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
