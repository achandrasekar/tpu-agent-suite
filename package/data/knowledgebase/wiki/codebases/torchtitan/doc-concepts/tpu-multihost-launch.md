---
title: "Multi-host TorchTitan training on GKE with Cluster Toolkit"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Multi-host TorchTitan training on GKE with Cluster Toolkit

## Definition
For models larger than one host can hold, the README moves to GKE, provisioned through Google
Cloud's open-source Cluster Toolkit and its `gcluster` CLI. The README lists three provisioning
options: plain GKE, GKE with Cluster Toolkit, and GKE with TPU Cluster Director ("All Capacity
mode"). It uses Cluster Toolkit. (Earlier revisions of the README used XPK for the same flow.) The
flow has five steps:
1. Install `gcluster` (check with `gcluster --version`) and configure Docker
   (`gcloud auth configure-docker`).
2. Run `gcloud auth application-default login` so Terraform can provision, then create a
   **single-slice** cluster from the GKE TPU v6e blueprint:
   `gcluster deploy examples/gke-tpu-v6e/gke-tpu-v6e.yaml --vars project_id=...,deployment_name=${CLUSTER_NAME} --vars region=...,zone=... --vars machine_type=${COMPUTE_TYPE},tpu_topology=${TOPOLOGY},num_slices=1 --vars authorized_cidr=...,enable_gcsfuse_csi=true`.
   The slice is described by a machine type and a topology, for example `ct6e-standard-4t` and
   `4x8` for a `v6e-32`, rather than by an accelerator name.
3. Build and push the TorchTitan TPU image (see
   [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md)).
4. Optionally mount a GCS bucket. With Cluster Toolkit there is no separate attach step: the job
   itself passes `--mount "$BUCKET_NAME;/data;rw"` and `--service-account workload-identity-k8s-sa`.
5. Submit `gcluster job submit --name ... --cluster ... --location $REGION --compute-type ... --topology ... --num-slices 1 --image ...`,
   whose `--command` is an ordinary multi-node `torchrun`.

The worked example is Llama 3 8B with FSDP on `v6e-32`:
`torchrun --nnodes=8 --nproc_per_node=4 --rdzv_backend=static --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 --node_rank=$TPU_WORKER_ID -m torchtitan.experiments.tpu.train --module=torchtitan.experiments.tpu.llama3 --config=llama3_8b --training.max_context_length=512 ...`.
The README says this configuration "was tested on `v6e-16` and `v6e-32`".

## In torchtitan (grounded)
- **Topology arithmetic.** `v6e-32` is 8 hosts x 4 chips. So `--nnodes` equals the host count,
  `--nproc_per_node` equals the chips per host, and `WORLD_SIZE` is 32, one process per chip.
  `--node_rank=$TPU_WORKER_ID` uses the worker index that GKE injects, and the static rendezvous
  points at worker 0's pod DNS name. The workload passes that name as
  `--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-main-job-0-0.$WORKLOAD_NAME"`. The `main-job` segment
  follows the pod naming of jobs submitted with `gcluster job submit` (the XPK version used
  `slice-job`). The blueprint installs Kueue and JobSet, which `gcluster job submit` uses to run
  workloads.
- **Per-worker SliceBuilder bootstrap.** Every rank runs
  [`_start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/gmain.py#L23 "torchtitan/experiments/tpu/gmain#_start_trainer"),
  which calls [`maybe_init_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/distributed_utils.py#L9 "torchtitan/experiments/tpu/distributed_utils#maybe_init_distributed")
  and then [`_maybe_init_distributed_on_gke`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/cloud_distributed_utils.py#L41 "torchtitan/experiments/tpu/cloud_distributed_utils#_maybe_init_distributed_on_gke").
  That function needs the GKE-provided `TPU_WORKER_HOSTNAMES`, `TPU_CHIPS_PER_HOST_BOUNDS` and
  `TPU_HOST_BOUNDS`, plus torchrun's `WORLD_SIZE`, `LOCAL_RANK` and `RANK`. It rewrites the libtpu
  view into one chip per process: per-host bounds become all ones, host bounds are multiplied by
  the old per-host bounds, `TPU_VISIBLE_CHIPS=LOCAL_RANK`, and ports start at 10000 per host. The
  full `host:port` list is exported as both `TPU_PROCESS_ADDRESSES` and
  `TORCH_TPU_SLICEBUILDER_ADDRESSES`, with `TORCH_TPU_TOPOLOGY` set to the new host bounds. On
  TPU7x (`TPU_ACCELERATOR_TYPE` starting with `tpu7x`) the per-host device count doubles and a
  trailing `,2` is appended to the bounds, because each chip exposes two devices
  ([`TpuChip`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/local_tpu_device.py#L29 "torchtitan/experiments/tpu/local_tpu_device#TpuChip")
  records `devices_per_chip=2` for `V7X`).
- **The recipe behind `--config=llama3_8b`.**
  [`llama3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L178 "torchtitan/experiments/tpu/llama3/config_registry#llama3_8b")
  sets SimpleFSDP, full activation checkpointing, `eager_mode="DEFER_AND_FUSE"`, per-block
  `torch.compile(backend="tpu")`, splash attention and the Pallas fused loss. Its training block
  uses token budgets:
  [`max_context_length`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L51 "torchtitan/config/configs#TrainingConfig.max_context_length")`=8192`
  and
  [`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank")`=1 * 8192`.
  The README's command line overrides only `max_context_length` (formerly `seq_len`), down to 512.
  The token budget stays at 8192. So
  [`Trainer.__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L344 "torchtitan/trainer#Trainer.__init__")
  still builds the dataloader with 8192 token slots per rank per microbatch, and derives
  8192 // 512 = 16 sequences as the per-rank batch. The old `seq_len=512` override kept
  `local_batch_size=1`, which was 512 tokens per rank, so the same command now does 16x the work
  per step. This is a code observation; the README does not mention it. Before training,
  [`_fit_rope_to_context`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L192 "torchtitan/experiments/tpu/train#_fit_rope_to_context")
  shrinks the RoPE caches to the context length. Its docstring notes that full-length caches ran
  llama3-70b out of program HBM.
- **Output location.** Traces, TensorBoard logs and checkpoints go under
  [`dump_folder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L96 "torchtitan/trainer#Trainer.Config.dump_folder").
  The README's commented-out GCSFUSE block sets it to a `/data/...` path so outputs land in the bucket.

## Why it matters / when it applies
- **Network and cluster shape come from the blueprint.** The blueprint creates a VPC with an MTU of
  8,896 "for optimal performance", plus a firewall rule for traffic between nodes. It creates a
  regional cluster in `${REGION}` with the TPU node pool in `${ZONE}`, and provisions on-demand TPUs.
  For a reservation, Spot VMs or flex-start, change the consumption options in the blueprint's
  deployment file. `enable_gcsfuse_csi=true` is required if you want the `/data` mount later.
  `authorized_cidr` must contain the IP address of the machine that runs `gcluster`.
- **Silent bootstrap failure on GKE.** `_maybe_init_distributed_on_gke` returns False without
  raising when the Kubernetes variables are inconsistent: the host-bounds product differs from the
  hostname count, or chips x hosts differs from `WORLD_SIZE`. Control then falls through to the
  single-host VM path. A wrong `--nnodes`/`--nproc_per_node` therefore surfaces later, typically as
  the VM path's chip-count `ValueError` or a hang, not as a clear topology error.
- **`WORKERS_0_HOSTNAME` is only a rendezvous address on GKE.** Among the bootstrap code, only the
  single-host VM path reads it. On GKE it serves the `torchrun` static rendezvous, and SliceBuilder
  addresses come from `TPU_WORKER_HOSTNAMES`.
- **Env flags win.** The example exports
  `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536"` through `--env`. Per
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md), an exported variable overrides the
  recipe's own flag set, so prefer `--extra-libtpu-init-flags` when you mean to tweak a recipe.
- **Without the bucket, drop two flags.** If you do not use GCSFUSE, remove `--service-account` and
  `--mount` from the submit command, as the README's comment says. Relative paths then resolve
  inside the image.
- **Clean up.** Cancel the workload with `gcluster job cancel <name> --cluster ... --location
  ${REGION} --project ...`. Then delete the cluster and its VPC with `gcluster destroy
  ${CLUSTER_NAME}`. Run that command from the directory where you ran `gcluster deploy`, because
  `deploy` writes a deployment directory with the Terraform configuration and state, and you need
  it to tear the cluster down.

> [!warning] Doc vs code
> - The optional bucket step runs `gcloud storage buckets create ${REPOSITORY}` where the bucket
>   variable is meant.
> - `BUCKET_NAME` is defined with a `gs://` prefix but then used as `gs://${BUCKET_NAME}` in the
>   tokenizer upload. The TensorBoard command uses `gcs://`.
> - The README still says only `llama3_tpu` is supported and caps `max_context_length` at 512 for
>   this path. The registries have moved past both (see
>   [tpu-support-status-and-limits](tpu-support-status-and-limits.md)).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (topology bootstrap and lane
  selection), [fsdp-sharding](../concepts/fsdp-sharding.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [checkpointing](../concepts/checkpointing.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/cloud_distributed_utils`,
  `torchtitan/experiments/tpu/distributed_utils`, `torchtitan/experiments/tpu/llama3/config_registry`,
  `torchtitan/config/configs`).
- Related doc-concepts: [tpu-vm-provisioning](tpu-vm-provisioning.md),
  [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md),
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md),
  [tpu-profiling-tensorboard](tpu-profiling-tensorboard.md),
  [tpu-rl-kuberay-deployment](tpu-rl-kuberay-deployment.md), [tpu-training-recipes](tpu-training-recipes.md)
  (tuned launch commands per topology), [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md).
- Runtime side: torch_tpu [gke-xpk-workloads](../../torch_tpu/doc-concepts/gke-xpk-workloads.md)
  (the XPK version of the same GKE launch),
  [distributed-launch-environment](../../torch_tpu/doc-concepts/distributed-launch-environment.md),
  [distributed-launchers-and-spmd](../../torch_tpu/concepts/distributed-launchers-and-spmd.md);
  hardware [TPU v6e](../../../hardware/tpu-v6e.md), [TPU7x](../../../hardware/tpu-7x.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (section "Multi-host model training on GKE",
including "Install Cluster Toolkit and its dependencies", "Create a single-slice cluster", "Mount a
Cloud Storage bucket with Cloud Storage FUSE", "Deploy the training workload" and "Clean up"; kept
in place).
