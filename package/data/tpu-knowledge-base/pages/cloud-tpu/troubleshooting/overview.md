---
title: Troubleshooting Cloud TPU Workloads
type: concept
tags:
- cloud-tpu
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-troubleshooting
generation_recipe: overview.recipe.json
---

# Troubleshooting Cloud TPU Workloads

When training or inference workloads stall, slow down, or fail on Cloud TPU resources, follow a structured diagnostic workflow to identify outlier workers, isolate bottlenecks, profile execution, and recover from boot failures.[^cloud-tpu-troubleshooting]

## Management Context and Support Scope

The Cloud TPU API is in maintenance mode and receives only bug fixes and security updates. It is not supported on TPU v7x or newer hardware generations. For ongoing feature updates and current TPU generations, manage TPU resources through Compute Engine or Google Kubernetes Engine (GKE). For regional availability, supported version matrices, quota limits, and minimum GKE versions, consult current official documentation.[^cloud-tpu-troubleshooting]

## Core Troubleshooting Workflow

Troubleshooting follows a three-stage progression:

1. **Evaluate metrics** to spot outlier TPU VMs behaving abnormally compared to the rest of the fleet.[^cloud-tpu-troubleshooting]
2. **Examine logs** specifically targeting those identified outlier VMs.[^cloud-tpu-troubleshooting]
3. **Profile the workload** to evaluate hardware utilization and model execution.[^cloud-tpu-troubleshooting]

Unified dashboards deployed via Terraform scripts can consolidate these metrics and logs into a single Google Cloud console interface.[^cloud-tpu-troubleshooting]

### 1. Collect Metrics and Identify Outliers

Cloud Monitoring automatically tracks time-series data from TPU chips and host Compute Engine VMs. Compare workers across the job to isolate bottlenecks:

* **Host and memory indicators**: Track worker-level resources, such as available system memory (`system_available_memory_GiB`) and host CPU utilization, to detect host bottlenecks.[^cloud-tpu-troubleshooting]
* **Communication and network indicators**: Monitor network usage metrics to detect data transfer stalls across distributed workers.[^cloud-tpu-troubleshooting]
* **Device activity**: Review TensorCore idle duration to identify whether the TPU device is starving for work or underutilized.[^cloud-tpu-troubleshooting]
* **Health and hardware status**: Cloud TPU health monitoring delivers real-time health information for individual VMs and multi-VM slices through the Cloud Monitoring console or the LibTPU Monitoring Library. Hardware failures detected through health monitoring allow for resource reallocation before total job failure occurs.[^cloud-tpu-troubleshooting]

### 2. Inspect Cloud Logging

Cloud Logging captures events automatically from the TPUs and their host VMs, accessible through the Logs Explorer:

* **TPU Worker Logs**: Detail operational events on individual workers within a specific zone, including system memory availability.[^cloud-tpu-troubleshooting]
* **Audited Resource Logs**: Record administrative API calls, noting when operations like `CreateNode`, `UpdateNode`, or `DeleteNode` occurred and who initiated them.[^cloud-tpu-troubleshooting]
* **Application Stack Traces**: Workloads can be instrumented with the `cloud-tpu-diagnostics` Python package to output stack traces directly into the logging stream when investigating hung or failing code.[^cloud-tpu-troubleshooting]

### 3. Profile Workloads

To locate model-level bottlenecks and improve training performance, run performance profiling using TensorBoard equipped with the TPU TensorBoard plugin. Both JAX and PyTorch workloads provide performance integration paths with TensorBoard profiling tools.[^cloud-tpu-troubleshooting]

## Handling VM Boot Failures

If a TPU VM fails to boot after applying custom images or modifications:

1. **Retrieve Serial Output**: The serial console log serves as the primary diagnostic source. Access it through the Google Cloud console or with:

   ```bash
   gcloud compute instances get-serial-port-output VM_NAME --zone=ZONE
   ```
   *(Replace `VM_NAME` with the instance identifier and `ZONE` with the Google Cloud zone.)*[^cloud-tpu-troubleshooting]

2. **Remediate and Recreate**: Because TPU boot disks cannot be repaired offline, you must correct the root cause in your custom image source or startup script, delete the affected resource, and recreate the TPU VM.[^cloud-tpu-troubleshooting]

<!-- sources:generated:start -->

## References

[^cloud-tpu-troubleshooting]: [Google Cloud: Troubleshoot TPUs](https://docs.cloud.google.com/tpu/docs/troubleshooting/troubleshooting). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
