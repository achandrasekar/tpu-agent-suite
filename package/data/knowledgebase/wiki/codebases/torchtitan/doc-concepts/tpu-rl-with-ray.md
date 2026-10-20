---
title: "Non-colocated GRPO on TPU with Ray (trainer and vLLM on separate slices)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/rl_nc_ray/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Non-colocated GRPO on TPU with Ray (trainer and vLLM on separate slices)

## Definition
`rl_nc_ray` runs TorchTitan's RL loop (GRPO) in a **non-colocated** layout. The policy trainer
(PyTorch FSDP) and the rollout generator (vLLM) "run on entirely separate slices of TPU hardware".
The README's rationale is memory: separating them "avoids out-of-memory errors by ensuring the
training weights/gradients and the KV cache do not compete for the same HBM". The orchestrator
`grpo.py` runs on the Ray head node. It launches remote `RayTPUPolicyTrainer` and
`RayVLLMGenerator` actors on their slices and moves the updated policy weights to the generator
"via Ray's object store".

## In torchtitan (grounded)
- **Orchestrator.** [`RayController`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L123 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayController")
  subclasses the upstream RL [`Controller`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L260 "torchtitan/experiments/rl/controller#Controller")
  and overrides only setup, in [`setup_async`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L131 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayController.setup_async").
  The GRPO loop itself is unchanged. Proxy classes such as
  [`RayTrainerProxy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L66 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayTrainerProxy")
  and [`RayInterGeneratorRouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L89 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayInterGeneratorRouter")
  translate the upstream Monarch-style `.call()` RPCs into Ray `.remote()` futures. Monarch and
  torchstore are replaced by stubs under `dummy_modules/`, which must be on `PYTHONPATH`.
  [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L247 "torchtitan/experiments/tpu/rl_nc_ray/grpo#main")
  parses a normal TorchTitan RL config (`--module alphabet_sort --config rl_grpo_qwen3_*_varlen`).
- **Slice discovery.** [`discover_tpu_cluster_layout`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L37 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#discover_tpu_cluster_layout")
  groups live Ray TPU nodes by the `ray.io/tpu-slice-name` label. The **first slice becomes the
  trainer** and the second becomes the sampler (generator). With only one slice, both share it.
  The trainer world size is the TPU count of its slice. With `LOCAL_VM_RUN=1` the trainer runs on
  CPU with world size 1.
- **Per-actor TPU environment.** Each actor gets its own PJRT/SliceBuilder environment.
  [`build_trainer_env_vars`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L101 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#build_trainer_env_vars")
  gives each rank `TORCH_TPU_SLICEBUILDER_ADDRESSES` on ports from
  [`TRAINER_BASE_PORT`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L25 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#TRAINER_BASE_PORT")
  (8471), host bounds for `v6e-8`/`v6e-32`, and a `LIBTPU_INIT_ARGS` that disables the enhanced
  launch barrier and pins the slice-builder port.
  [`build_generator_env_vars`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L183 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#build_generator_env_vars")
  uses ports from [`GENERATOR_BASE_PORT`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L26 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#GENERATOR_BASE_PORT")
  (8070), sets `VLLM_PLUGINS=torchtpu,torchtpu_layers`, and sets `TPU_MULTIHOST_BACKEND=ray` when
  the generator's TP > 1. The generator actor is pinned to the sampler slice's first node with a
  hard `NodeAffinitySchedulingStrategy`.
- **HBM budgeting.** The module comment explains that PJRT "preallocates 90% of TPU HBM upfront by
  default". Both actors therefore set
  [`XLA_PYTHON_CLIENT_PREALLOCATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L30 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#XLA_PYTHON_CLIENT_PREALLOCATE")
  to `false` and [`XLA_PYTHON_CLIENT_MEM_FRACTION`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L31 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#XLA_PYTHON_CLIENT_MEM_FRACTION")
  and [`JAX_MEM_FRACTION`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils.py#L28 "torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils#JAX_MEM_FRACTION")
  to `0.45`, "to prevent initialization OOMs/deadlocks".
- **Trainer actor.** [`RayTPUPolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/trainer.py#L120 "torchtitan/experiments/tpu/rl_nc_ray/trainer#RayTPUPolicyTrainer")
  extends the upstream [`PolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L48 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer").
  It fills in torchrun-style variables, forces the `foreach` AdamW implementation ("TPU does not
  support fused AdamW natively"), and on CPU starts a gloo group so XLA does not take over the
  process group. Its
  [`_build_model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/trainer.py#L162 "torchtitan/experiments/tpu/rl_nc_ray/trainer#RayTPUPolicyTrainer._build_model")
  reads the context length from `training.max_context_length` (renamed upstream from
  `training.seq_len`). It raises if that exceeds the model's RoPE maximum, and otherwise rebuilds
  each layer's RoPE config with `max_context_length` set to it. Before that it monkey-patches
  upstream `VarlenInnerAttention` (formerly `VarlenAttention`) with a padded SDPA path, because
  `aten::_flash_attention_forward` is not implemented on TPU (`padding_packing_utils.py`).
- **Weight sync.** [`get_model_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/trainer.py#L233 "torchtitan/experiments/tpu/rl_nc_ray/trainer#RayTPUPolicyTrainer.get_model_state_dict")
  calls [`prepare_trainer_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils.py#L103 "torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils#prepare_trainer_state_dict").
  That function replicates DTensor shards over ICI, concatenates by dtype on TPU to cut PCIe and
  object-store overhead, and offloads to CPU. By default it does this one layer group at a time
  ([`SYNC_LAYER_BY_LAYER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils.py#L32 "torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils#SYNC_LAYER_BY_LAYER")),
  because whole-model sync "leads to a massive TPU memory peak and instant OOM". On the generator,
  [`load_model_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/generator.py#L331 "torchtitan/experiments/tpu/rl_nc_ray/generator#RayVLLMGenerator.load_model_state_dict")
  queues the object refs into the engine loop. A patched pull then fans them out to the vLLM workers
  ([`load_weights_on_worker`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils.py#L311 "torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils#load_weights_on_worker"))
  in chunks of [`TPU_COPY_CHUNK_SIZE_PARAMETERS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils.py#L37 "torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils#TPU_COPY_CHUNK_SIZE_PARAMETERS")
  parameters. The comment explains that free HBM after the KV cache is "extremely tight (~455MB)".
  vLLM itself is adapted through `patch_vllm_for_tpu` (`rl_nc_ray/generator.py`).

## Why it matters / when it applies
- **Pick non-colocated when HBM is the constraint.** If training state and the KV cache do not fit
  together, give each its own slice. You pay for it with a CPU round trip per policy update (TPU ->
  host -> Ray object store -> host -> TPU), which the layer-by-layer, dtype-concatenated transfer
  tries to keep bounded.
- **Two PJRT runtimes on shared hosts need care.** The disabled enhanced launch barrier, distinct
  port bases and memory fractions are all there so that separate distributed runtimes do not hang
  or OOM each other at init. Keep those settings if you change the topology.
- **Topology is hard-coded.** Host bounds are special-cased for `v6e-8` and `v6e-32` (the v6e-32
  branch carries "TODO: confirm this actually works"). Generator topology is a self-described
  "HACK" keyed on TP degree 4, 8 or 32. Other slice shapes need code changes.

> [!warning] Doc vs code
> - The README says `e2e_recipe_gke.sh` configures "2D sharding (FSDP x DP) for the Policy
>   Trainer". The script actually passes `--trainer.parallelism.tensor-parallel-degree 8
>   --trainer.parallelism.data_parallel_shard_degree 1`, so the trainer runs TP-only across the
>   8-chip slice.
> - The README says the script supports Qwen3-0.6B, 1.7B and 4B. Only `rl_grpo_qwen3_0_6b_varlen`
>   exists; the `rl_grpo_qwen3_1_7b_varlen` and `rl_grpo_qwen3_4b_varlen` configs the script
>   selects are not in the registry (see
>   [tpu-rl-kuberay-deployment](tpu-rl-kuberay-deployment.md)).
> - The README presents `smoke_test_vm_tp1.sh` and `smoke_test_vm_tp4.sh` as different generator TP
>   settings. Both scripts set trainer TP 1 and generator TP 4 and differ only in `--log_samples`
>   and the log file name.
> - The `grpo.py` module docstring describes a `RayRLTrainer` and a `RayGeneratorProxy` and says it
>   "Pins the trainer to CPU". The code defines `RayController`, and it pins the trainer to CPU only
>   when `LOCAL_VM_RUN=1`.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (SliceBuilder environment,
  `tpu_dist`), [fsdp-sharding](../concepts/fsdp-sharding.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md) (the `Replicate`
  redistribution before transfer), [state-dict-adapters](../concepts/state-dict-adapters.md)
  (initial HF weight loading in the Ray trainer),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/rl_nc_ray/grpo`,
  `torchtitan/experiments/tpu/rl_nc_ray/tpu_env_utils`, `torchtitan/experiments/tpu/rl_nc_ray/weight_transfer_utils`,
  `torchtitan/experiments/rl/controller`).
- Related doc-concepts: [rl-post-training](rl-post-training.md) (the upstream GPU RL loop this
  subclasses), [tpu-rl-kuberay-deployment](tpu-rl-kuberay-deployment.md),
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md), [tpu-multihost-launch](tpu-multihost-launch.md).
- Runtime side: torch_tpu [synchronization](../../torch_tpu/concepts/synchronization.md),
  [host-device-transfer](../../torch_tpu/concepts/host-device-transfer.md),
  [pjrt-client](../../torch_tpu/concepts/pjrt-client.md); hardware [TPU v6e](../../../hardware/tpu-v6e.md).

## Source
Extracted from `torchtitan/experiments/tpu/rl_nc_ray/README.md` (intro and "Scripts" sections; kept
in place).
