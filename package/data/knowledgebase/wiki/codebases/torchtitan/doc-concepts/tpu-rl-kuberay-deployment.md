---
title: "Deploying TPU RL on a GKE KubeRay cluster"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/rl_nc_ray/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Deploying TPU RL on a GKE KubeRay cluster

## Definition
The RL README's "GKE KubeRay Deployment Walkthrough" is the production path for
[tpu-rl-with-ray](tpu-rl-with-ray.md). It targets a `v6e-8` GKE cluster and has six steps:
- **A.** Build and push a Ray+vLLM image from `scripts/Dockerfile.ray.vllm`, and point the cluster
  manifest at it.
- **B.** Grant bucket access through Workload Identity. The manifest mounts a GCS
  bucket at `/data` in every pod with the GCS Fuse CSI driver. The Kubernetes SA `ray-ksa` is bound
  to a Google SA that holds Storage Object Admin.
- **C.** Enable the GKE `RayOperator` add-on and `kubectl apply -f scripts/ray-v6e8-2.yaml`.
- **D.** Port-forward the head service (`ray-tpu-v6e-cluster-head-svc`) on 8265 (dashboard) and
  10001 (client).
- **E.** Submit `scripts/e2e_recipe_gke.sh <0.6b|1.7b|4b>`, which picks the Qwen3 size.
- **F.** Monitor with `ray list jobs` and `ray job logs <raysubmit_id>`. Plot reward, loss and
  generation/training time with `scripts/plot_metrics_from_log.py`.

For quick iteration without a cluster, `smoke_test_vm_tp1.sh` / `smoke_test_vm_tp4.sh` run on one
TPU VM with `LOCAL_VM_RUN=1`: the trainer runs on CPU and the generator on the VM's TPU.

## In torchtitan (grounded)
- **Cluster shape to layout.** `ray-v6e8-2.yaml` declares one TPU worker group with `replicas: 2`
  ("use 2 slices") and `numOfHosts: 2`. Each pod requests `google.com/tpu: "4"` under
  `gke-tpu-topology: 2x4`. That gives two 8-chip v6e slices.
  [`discover_tpu_cluster_layout`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L37 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#discover_tpu_cluster_layout")
  groups the Ray nodes by the `ray.io/tpu-slice-name` label that KubeRay sets. It assigns slice 0
  to the trainer and slice 1 to the generator, and reads `ray.io/tpu-pod-type` (`v6e-8`) to choose
  the host bounds in
  [`build_trainer_env_vars`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L101 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#build_trainer_env_vars").
- **Job submission.** `e2e_recipe_gke.sh` runs `ray job submit --working-dir .` with
  `PYTHONPATH=torchtitan/experiments/tpu/rl_nc_ray/dummy_modules:.` and then
  `python3 -m torchtitan.experiments.tpu.rl_nc_ray.grpo --module alphabet_sort --config rl_grpo_qwen3_<size>_varlen`.
  That module's entry point is [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L247 "torchtitan/experiments/tpu/rl_nc_ray/grpo#main").
  The script sets generator TP 8 (`spmd_backend spmd_types`) and trainer TP 8 with
  `data_parallel_shard_degree 1`, turns compile and wandb off, runs 10 async-loop training steps,
  and reads weights from `/data/assets/assets/hf/Qwen3-<size>`. Extra arguments are forwarded to
  `grpo.py`. Of the three configs it can select, only [`rl_grpo_qwen3_0_6b_varlen`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py#L86 "torchtitan/experiments/rl/examples/alphabet_sort/config_registry#rl_grpo_qwen3_0_6b_varlen") exists in the `alphabet_sort` registry
  at this commit (see the warning below).
- **Metrics plot.** [`parse_log`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log.py#L5 "torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log#parse_log")
  and [`plot_metrics`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log.py#L63 "torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log#plot_metrics")
  turn the saved `ray job logs` output into a PNG.

## Why it matters / when it applies
- **The image is on a different stack from the main TPU image.** `Dockerfile.ray.vllm` starts from a
  Ray nightly `py312-tpu` base. It pins `jax[tpu]==0.9.1`, `libtpu==0.0.42.1` and CPU
  `torch==2.12.0`, and installs the **pre-built** `torch_tpu` wheel from the virtual registry
  (`--pre --no-deps`). It then builds vLLM `v0.26.1rc0` with TPU support,
  which needs `GITHUB_USER`/`GITHUB_TOKEN` build args. The main TorchTitan image pins `jax 0.11.1`
  / `libtpu 0.0.50` and a source-built wheel (see
  [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md)). Do not assume
  flags or kernels validated on one image behave the same on the other.
- **The manifest and README disagree on the image tag.** The README says the YAML should point at
  `torchtitan-ray:vllm-0612`, but the checked-in `ray-v6e8-2.yaml` uses
  `torchtitan-ray:rebase-0820` for both head and workers. Set it to whatever you pushed in step A.
- **Project-specific values to replace.** The manifest and commands hard-code a GCP project, a
  Google service account, a GCS bucket and a us-west2 registry; substitute your own.
  `e2e_recipe_gke.sh` also prepends a personal `uv_venv` path to `PATH`. The manifest binds
  `ray-ksa` to `cluster-admin`, which its own comment calls "Only for testing". Use a narrower role
  in shared clusters.
- **Local smoke tests clean up hard.** The smoke scripts kill every process holding `/dev/vfio/*`
  and `ray stop --force` "to prevent driver lock deadlocks". Only run them on a dedicated VM.

> [!warning] Doc vs code
> - **Only the 0.6B size runs.** The README offers `0.6b`, `1.7b` and `4b`. `e2e_recipe_gke.sh` maps
>   them to `rl_grpo_qwen3_0_6b_varlen`, `rl_grpo_qwen3_1_7b_varlen` and `rl_grpo_qwen3_4b_varlen`,
>   but the `alphabet_sort` registry defines only [`rl_grpo_qwen3_0_6b_varlen`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py#L86 "torchtitan/experiments/rl/examples/alphabet_sort/config_registry#rl_grpo_qwen3_0_6b_varlen"). The nearest 1.7B config is [`rl_grpo_qwen3_1_7b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py#L456 "torchtitan/experiments/rl/examples/alphabet_sort/config_registry#rl_grpo_qwen3_1_7b"), which is
>   not a varlen config, and there is no 4B config. `1.7b` and `4b` fail at config lookup.
> - **Trainer sharding is TP, not FSDP x DP.** The README claims 2D FSDP x DP for the trainer, but
>   the shipped script runs TP=8 with DP shard 1. See the full mismatch list on
>   [tpu-rl-with-ray](tpu-rl-with-ray.md).
> - **Image tag.** The README's `torchtitan-ray:vllm-0612` does not match the `rebase-0820` tag in
>   the checked-in manifest (see above).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [tensor-parallel](../concepts/tensor-parallel.md), [fsdp-sharding](../concepts/fsdp-sharding.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md) (HF assets under `/data`).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils`,
  `torchtitan/experiments/tpu/rl_nc_ray/grpo`, `torchtitan/experiments/tpu/rl_nc_ray/scripts/plot_metrics_from_log`,
  `torchtitan/experiments/rl/examples/alphabet_sort/config_registry`).
- Related doc-concepts: [tpu-rl-with-ray](tpu-rl-with-ray.md), [rl-post-training](rl-post-training.md),
  [tpu-multihost-launch](tpu-multihost-launch.md) (the Cluster Toolkit path for supervised training),
  [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md).
- Runtime side: torch_tpu [gke-xpk-workloads](../../torch_tpu/doc-concepts/gke-xpk-workloads.md);
  hardware [TPU v6e](../../../hardware/tpu-v6e.md).

## Source
Extracted from `torchtitan/experiments/tpu/rl_nc_ray/README.md` (sections "GKE KubeRay Deployment
Walkthrough" A-F and `plot_metrics_from_log.py`; kept in place).
