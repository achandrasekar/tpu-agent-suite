---
title: "Running TorchTPU on GKE: Kueue-admitted Jobs and JobSets"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_3_get_started_on_gke.md
sources:
  - docs/markdown/1_basics_3_get_started_on_gke.md
  - docs/markdown/1_basics_5_distributed_on_gke.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Running TorchTPU on GKE: Kueue-admitted Jobs and JobSets

## Definition
"Recent TPUs from v7 onwards are only supported on GKE." The GKE guide runs the VM example
(`examples/simple_pytorch/simple_pytorch.py`) on a TPU v7 node pool. You can use the pre-built
nightly TorchTPU image or build your own with `docker/build_image_multistage.sh` and push it to
Artifact Registry. Workloads are submitted as standard Kubernetes `Job` (one host) or `JobSet`
(several hosts) manifests, and Kueue admits them. Kueue is recommended "because it maps well to
the multi-slice structure of TPU deployments". The one-time setup is gcloud and kubectl
credentials, Docker auth to the registry, and applying a Kueue cohort manifest that you have
patched with your node pool name. You then `sed` the image and node pool into a job manifest,
`kubectl apply` it, and read output with `kubectl logs -l job-name=...`.

The cohort setup provides a universal `LocalQueue lq`, compatible with XPK tooling. It feeds
`ClusterQueue tpu-cluster-queue` in the cohort `tpu-shared-reservation`. Kueue matches the
workload's `cloud.google.com/gke-tpu-topology` node selector to a `ResourceFlavor` (`rf-4chip`
through `rf-64chip`). It submits the workload suspended and unsuspends it once cohort quota is
reserved.

## In torch_tpu (grounded)
Each manifest describes the GKE slice in GKE terms (`gke-tpu-topology: 2x2x1` is one host with 4
chips). It separately tells TorchTPU the device mesh through `TORCH_TPU_TOPOLOGY`
([`kTorchTpuTopologyEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L208 "csrc/common/env_vars.h#kTorchTpuTopologyEnvVar")). The two differ
because each v7 chip exposes two devices
([`get_devices_per_chip`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L216 "src/torch_tpu/_internal/utils/hardware#get_devices_per_chip")), so the
TorchTPU topology for v7 has a fourth "devices per chip" component. The auto-detected table
[`_V7_TOPOLOGY`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L72 "src/torch_tpu/_internal/utils/hardware#_V7_TOPOLOGY._V7_TOPOLOGY") maps 8 devices
to `"2,2,1,2"`.

| Manifest | GKE topology | `TORCH_TPU_TOPOLOGY` | Processes |
|---|---|---|---|
| `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` (Job) | `1,1,1,1` | 1, the other devices on the host are ignored |
| `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` (Job) | `2,2,1,2` | 8, one per device |
| `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` (JobSet, 16 pods) | `4,4,4,2` | 128, 8 per host |

The single-device value `1,1,1,1` has no row in the auto-detected v7 table (the smallest entry
is 2 devices), so the manifest has to pin it explicitly. In the multi-host case the externally
supplied slice-wide topology is what lets
[`set_tpu_launch_env`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L328 "src/torch_tpu/_internal/distributed/distributed#set_tpu_launch_env") derive
each host's `NODE_RANK` from physical coordinates. The worker then validates the full variable
set in [`GetDistributedWorkerConfiguration`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/slicebuilder/discovery.cc#L65 "csrc/distributed/slicebuilder/discovery.cc#GetDistributedWorkerConfiguration").
[distributed-launch-environment](distributed-launch-environment.md) covers that step.

## Why it matters / when it applies
GKE is the only route to v7 and to multi-host slices. Kueue's cohort quota lets many users share
one reservation without a cluster each. Things to check when a job misbehaves:
- The GKE topology label, the `google.com/tpu` request (4 per v7 host), and `TORCH_TPU_TOPOLOGY`
  must describe the same slice. The TorchTPU value counts devices, not chips.
- Some file names in the guide are out of date. The single-device job and the Kueue cohort
  manifest live under `examples/simple_pytorch/`, not `examples/tensor_parallel/`. The worker
  script is `tp_worker.py`. The example manifests label workloads for `multislice-queue`, which
  the cohort manifest also defines, and not `lq`.
- The manifests export `TPU_LAUNCH_BLOCKING=1`. TorchTPU reads `TORCH_TPU_LAUNCH_BLOCKING`, so
  as written they do not enable the Debug eager mode (see
  [eager-execution-modes](eager-execution-modes.md)).
- Mount a shared bucket and set a tier-3 cache root so that pods and restarts reuse compiles
  (see [multi-tier-compilation-cache](multi-tier-compilation-cache.md)).

## Connections
- Code concepts: [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md),
  [hardware-introspection](../concepts/hardware-introspection.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [pjrt-client](../concepts/pjrt-client.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/distributed`,
  `src/torch_tpu/_internal/utils`, `csrc/distributed`).
- Related doc-concepts: [distributed-launch-environment](distributed-launch-environment.md),
  [docker-multistage-images](docker-multistage-images.md),
  [tpu-device-onboarding](tpu-device-onboarding.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [tpu-hardware-model](tpu-hardware-model.md).

## Source
Extracted from `docs/markdown/1_basics_3_get_started_on_gke.md` and
`docs/markdown/1_basics_5_distributed_on_gke.md` (kept in place).
