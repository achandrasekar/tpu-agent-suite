---
title: Cloud TPU VM Lifecycle and Management
type: concept
tags:
- cloud-tpu
created: '2026-10-03'
human_reviewed: false
source_ids:
- gcp-tpu-vm-guide
generation_recipe: overview.recipe.json
---

# Cloud TPU VM Lifecycle and Management

Cloud TPU VMs are managed either through the standalone Cloud TPU API, Compute Engine, or Google Kubernetes Engine (GKE) [^gcp-tpu-vm-guide]. The Cloud TPU API is limited to bug fixes and security updates, is not supported on TPU7x or newer hardware generations, and does not receive new feature development; Compute Engine or GKE is recommended instead [^gcp-tpu-vm-guide]. For current pricing, quota limits, regional availability, and supported engine versions, consult official documentation [^gcp-tpu-vm-guide].

## SSH Access and Key Persistence

Connecting to a Cloud TPU VM over SSH requires network reachability, such as an assigned external IP address or an internal IP connection method [^gcp-tpu-vm-guide].

Key management depends on the tool used to initiate the session:
- **CLI (`gcloud`)**: Automatically provisions a persistent SSH key on the host [^gcp-tpu-vm-guide].
- **Google Cloud Console**: Generates an ephemeral SSH key for the browser-based session [^gcp-tpu-vm-guide].

## Multi-Host Coordination

When provisioning a TPU slice that spans more than a single host, a TPU VM is instantiated for each host in the slice [^gcp-tpu-vm-guide]. Multi-host interactions depend on addressing specific workers:

- **Targeted Host Access**: Connect to an individual host within a slice by supplying a 0-based worker index [^gcp-tpu-vm-guide].
- **Parallel Execution Across Workers**: Execute commands (such as dependency installations or setup routines) simultaneously across all worker VMs in the slice [^gcp-tpu-vm-guide].
- **Multislice Allocations**: Queued-resource multislice environments can target commands across all nodes and all workers simultaneously, optionally breaking execution into batch sizes [^gcp-tpu-vm-guide].

## Resource Inspection

Inspecting resources verifies their operational status (`READY`, `STOPPED`, etc.) and configuration parameters [^gcp-tpu-vm-guide].

Inspection is supported at two scopes:
- **List Scope**: Aggregates all TPU instances provisioned within a specified project and zone [^gcp-tpu-vm-guide].
- **Describe Scope**: Retrieves detailed metadata for a single identified instance, including its accelerator type, runtime version, and lifecycle state [^gcp-tpu-vm-guide].

Both actions are accessible via the Google Cloud console, the CLI, or Cloud Client Libraries (Python, Java, Node.js) using Application Default Credentials [^gcp-tpu-vm-guide].

## Stopping vs. Deleting Resources

Understanding the distinction between stopping and deleting a TPU VM is essential for resource lifecycle management:

| Dimension | Stopping a TPU VM | Deleting a TPU VM |
| :--- | :--- | :--- |
| **VM Configuration** | Retained [^gcp-tpu-vm-guide] | Destroyed permanently [^gcp-tpu-vm-guide] |
| **Installed Software** | Preserved on host disks [^gcp-tpu-vm-guide] | Removed with the VM [^gcp-tpu-vm-guide] |
| **Restartability** | Can be returned to `READY` via a start request [^gcp-tpu-vm-guide] | Cannot be restarted; requires new provisioning [^gcp-tpu-vm-guide] |
| **Queued Resources API** | **Not supported**; allocations cannot be stopped [^gcp-tpu-vm-guide] | Supported (required to halt queued charges) [^gcp-tpu-vm-guide] |

### State Transitions

- **Start**: Moves a stopped TPU instance back into a runnable (`READY`) state [^gcp-tpu-vm-guide]. Start/stop operations are not supported for TPU Pods or TPUs managed via queued resources; do not apply standalone VM lifecycle commands to those allocations [^gcp-tpu-vm-guide].
- **Stop**: Transitions an active instance into a `STOPPED` state to prevent further runtime charges while safeguarding installed dependencies and environment state [^gcp-tpu-vm-guide].
- **Delete**: Terminates and tears down the allocated TPU slice permanently once work sessions are completed [^gcp-tpu-vm-guide].

### Resource Definition Example

When constructing programmatic requests using client libraries, resources are identified by their canonical hierarchical URI:

```
projects/<PROJECT_ID>/locations/<ZONE>/nodes/<TPU_NAME>
```

- `<PROJECT_ID>`: User-assigned Google Cloud project identifier or number [^gcp-tpu-vm-guide].
- `<ZONE>`: Target deployment zone (e.g., `europe-west4-a`, `us-central1-f`) [^gcp-tpu-vm-guide].
- `<TPU_NAME>`: Unique identifier assigned to the TPU node resource [^gcp-tpu-vm-guide].

<!-- sources:generated:start -->

## References

[^gcp-tpu-vm-guide]: [Google Cloud: Manage TPU VMs](https://docs.cloud.google.com/tpu/docs/managing-tpus-tpu-vm). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
