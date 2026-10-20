---
title: 'Cloud TPU gcloud Invocation Patterns: TPU VMs, Queued Resources and GKE Node Pools'
type: concept
tags:
- cloud-tpu
- gcloud
created: '2026-10-03'
human_reviewed: false
source_ids:
- gcloud-tpu-vm
- gcloud-tpu-vm-create
- gcloud-tpu-vm-ssh
- gcloud-tpu-vm-scp
- gcloud-queued-resources-create
- tpu-on-gke
generation_recipe: overview.recipe.json
---

# Cloud TPU gcloud Invocation Patterns: TPU VMs, Queued Resources and GKE Node Pools

Unlike the concepts on neighbouring pages, CLI flags and defaults change between
releases. Confirm anything load-bearing against the current reference.

## The `tpu-vm` command surface

Cloud TPU VM management is anchored in `gcloud compute tpus tpu-vm`[^gcloud-tpu-vm]:

- **Lifecycle and inspection**: `create`, `delete`, `describe`, `list`, `start`, `stop`, `update` and `get-guest-attributes`[^gcloud-tpu-vm].
- **Connectivity**: `ssh` and `scp` reach individual host workers[^gcloud-tpu-vm].
- **Subgroups**: `accelerator-types`, `versions` and `service-identity`[^gcloud-tpu-vm].

## Direct creation versus queued resources

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
    --zone=ZONE --version=VERSION --accelerator-type=ACCELERATOR_TYPE
```

`--accelerator-type` defaults to `v2-8` when omitted, which is rarely what a
caller wants; the chip-based alternative is `--topology` with `--type`, and the
two configurations are mutually exclusive[^gcloud-tpu-vm-create].

```bash
gcloud compute tpus queued-resources create QUEUED_RESOURCE \
    --zone=ZONE --runtime-version=RUNTIME_VERSION \
    --accelerator-type=ACCELERATOR_TYPE \
    --node-count=NODE_COUNT --node-prefix=NODE_PREFIX
```

| Aspect | Direct `tpu-vm create` | `queued-resources create` |
| :--- | :--- | :--- |
| Runtime version flag | `--version`[^gcloud-tpu-vm-create] | `--runtime-version`[^gcloud-queued-resources-create] |
| Node naming | Single positional TPU name[^gcloud-tpu-vm-create] | `--node-id` for one node, or `--node-count` with optional `--node-prefix`; these are mutually exclusive[^gcloud-queued-resources-create] |
| Capacity flags | `--preemptible`, `--spot`, `--reserved`[^gcloud-tpu-vm-create] | `--spot`, `--guaranteed`, `--reserved`, plus `--valid-after-*` and `--valid-until-*` scheduling windows[^gcloud-queued-resources-create] |
| Reservation scope | Reservations in the current project, used in aggregate[^gcloud-tpu-vm-create] | Also `--reservation-host-project`, `--reservation-host-folder` or `--reservation-host-organization`[^gcloud-queued-resources-create] |

The same flag name means different things across the two paths, so the runtime
version flag is a common transcription error.

For multislice, `--node-count` must be greater than 1, and `--node-prefix`
defaults to the queued resource ID rather than to the accelerator name;
supplying `--node-prefix` requires `--node-count`[^gcloud-queued-resources-create].

Capacity flags change what lifecycle commands apply. Preemptible TPUs are
time-limited and may be preempted to free resources for standard TPUs. Spot TPUs
have no runtime limit unless preempted. Neither can be restarted, so `stop` and
`start` are not a recovery path for them[^gcloud-tpu-vm-create].

## Worker addressing

`--worker` defaults to `0`, so a bare `ssh` into a multi-host slice silently
reaches only the first host[^gcloud-tpu-vm-ssh][^gcloud-tpu-vm-scp].

For `ssh`, the index accepts richer forms **only when `--command` is also
supplied**: a comma-separated list such as `1,4,6`, a range such as `1-3`, or
the keyword `all`, which runs the command concurrently on each selected
worker[^gcloud-tpu-vm-ssh]. Interactive `ssh` therefore cannot fan out.

When targeting multiple workers, cache the private key first, by default with
`ssh-add ~/.ssh/google_compute_engine`[^gcloud-tpu-vm-ssh].

```bash
gcloud compute tpus tpu-vm ssh TPU_NAME \
    --zone=ZONE --worker=all --command="uname -a" \
    --output-directory=OUTPUT_DIRECTORY
```

Without `--output-directory`, every worker's output interleaves on standard
output. With it, each worker is written to its own `{WORKER_ID}.log` file, for
example `2.log`. The directory must already exist[^gcloud-tpu-vm-ssh].

Note a documented inconsistency: `scp` repeats the same `--worker` description,
including its reference to `--command`, but the `scp` synopsis defines no
`--command` flag[^gcloud-tpu-vm-scp]. Treat multi-worker `scp` as unconfirmed by
this reference and verify it against the installed CLI before relying on it.

```bash
gcloud compute tpus tpu-vm scp LOCAL_PATH TPU_NAME:REMOTE_PATH \
    --zone=ZONE --recurse
```

`--recurse` is required for directories and `--compress` enables
compression[^gcloud-tpu-vm-scp].

## SSH reachability

- `--internal-ips` at creation time suppresses external IPs; `--internal-ip` at connection time selects the internal address, for reaching a TPU VM from a Compute Engine VM on the same or a peered VPC network. The singular and plural spellings are different flags on different commands[^gcloud-tpu-vm-create][^gcloud-tpu-vm-ssh].
- `--ssh-flag` substitutes `%USER%` and `%TPU%` with resolved values, where `%TPU%` becomes the external IP when connecting externally and the internal IP otherwise. This makes port forwarding such as `--ssh-flag="-L 80:%TPU%:80"` portable across both modes[^gcloud-tpu-vm-ssh].
- Raw arguments for the underlying `ssh` go after a `--` separator[^gcloud-tpu-vm-ssh].
- `--dry-run` prints the equivalent `ssh`/`scp` command instead of running it, which is the quickest way to see what gcloud actually invokes[^gcloud-tpu-vm-ssh][^gcloud-tpu-vm-scp].
- `--plain` suppresses gcloud's automatic flag additions; `--ssh-key-file` overrides the default `~/.ssh/google_compute_engine`[^gcloud-tpu-vm-ssh].

## GKE node pools are a separate surface

GKE TPU node pools are created with `gcloud container node-pools create`, not
with `gcloud compute tpus`, and the flags do not transfer. On-demand and Spot
TPU node pools in GKE draw on Compute Engine API quota
(`compute.googleapis.com`), which is not the same as the Cloud TPU API quota
(`tpu.googleapis.com`) used by `gcloud compute tpus`. A node pool consuming a
reservation needs no TPU quota at all[^tpu-on-gke].

```bash
gcloud container node-pools create NODE_POOL_NAME \
    --cluster=CLUSTER_NAME --location=CONTROL_PLANE_LOCATION \
    --node-locations=NODE_ZONE --machine-type=MACHINE_TYPE \
    --tpu-topology=TPU_TOPOLOGY --placement-type=COMPACT
```

Topology is expressed differently by generation. Earlier generations pass
`--tpu-topology` directly, while Ironwood (TPU7x) instead binds a regional
workload policy created with `gcloud compute resource-policies create
workload-policy --type=HIGH_THROUGHPUT --accelerator-topology=TPU_TOPOLOGY`,
referenced by `--placement-policy`. The policy is reusable across node pools
sharing a topology[^tpu-on-gke].

With `--enable-autoscaling`, set `--num-nodes=0` so the autoscaler provisions
nodes on demand. Multi-host slice node pools scale atomically from zero to the
maximum, and GKE atomically recreates the whole node pool on upgrade rather than
following the configured node upgrade strategy[^tpu-on-gke].

<!-- sources:generated:start -->

## References

[^gcloud-tpu-vm]: [gcloud CLI: gcloud compute tpus tpu-vm](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm). cli-reference. gcloud CLI command surface and flag semantics for Cloud TPU; flags and defaults change between releases and require live verification. URL checked 2026-10-03.

[^gcloud-tpu-vm-create]: [gcloud CLI: gcloud compute tpus tpu-vm create](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/create). cli-reference. gcloud CLI command surface and flag semantics for Cloud TPU; flags and defaults change between releases and require live verification. URL checked 2026-10-03.

[^gcloud-tpu-vm-ssh]: [gcloud CLI: gcloud compute tpus tpu-vm ssh](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/ssh). cli-reference. gcloud CLI command surface and flag semantics for Cloud TPU; flags and defaults change between releases and require live verification. URL checked 2026-10-03.

[^gcloud-tpu-vm-scp]: [gcloud CLI: gcloud compute tpus tpu-vm scp](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/scp). cli-reference. gcloud CLI command surface and flag semantics for Cloud TPU; flags and defaults change between releases and require live verification. URL checked 2026-10-03.

[^gcloud-queued-resources-create]: [gcloud CLI: gcloud compute tpus queued-resources create](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/queued-resources/create). cli-reference. gcloud CLI command surface and flag semantics for Cloud TPU; flags and defaults change between releases and require live verification. URL checked 2026-10-03.

[^tpu-on-gke]: [Google Cloud: Deploy TPU workloads in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus). vendor-docs. Cloud TPU/GKE concepts and operational guidance; support, availability and API details require live verification. URL checked 2026-10-03.

<!-- sources:generated:end -->
