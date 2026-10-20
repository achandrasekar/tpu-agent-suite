---
title: "TorchFT fault tolerance: replica groups, quorum and semi-sync (DiLoCo / LocalSGD)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/torchft/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# TorchFT fault tolerance: replica groups, quorum and semi-sync (DiLoCo / LocalSGD)

## Definition
TorchFT gives fault tolerance "when training with replicated weights, such as in DDP or HSDP".
Instead of one job with `data_parallel_replicate_degree=N`, you launch N separate TorchTitan
instances, called *replica groups*. Each one shards the weights internally with FSDP. The groups
talk through TorchFT, and a separate `torchft_lighthouse` process runs quorum. If a group dies,
the surviving groups keep training rather than the whole job restarting from the last
checkpoint. On top of that synchronous mode, TorchFT offers *semi-synchronous* algorithms
(LocalSGD, DiLoCo) in which groups synchronize weights only every `sync_steps` steps.

## In torchtitan (grounded)
- **Trainer.** [`FaultTolerantTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/trainer.py#L37 "torchtitan/experiments/torchft/trainer#FaultTolerantTrainer") is a Trainer
  variant whose [`init_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/trainer.py#L426 "torchtitan/experiments/torchft/trainer#FaultTolerantTrainer.init_distributed")
  derives this group's global ranks from `replica_id * group_size`. It then builds the manager.
  Its data-parallel info is widened through
  [`get_dp_info`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L137 "torchtitan/experiments/torchft/manager#TorchFTManager.get_dp_info") to
  `(dp_degree * group_size, dp_degree * replica_id + dp_rank)`, which lets the dataloader split
  data across groups.
- **Manager.** [`TorchFTManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L36 "torchtitan/experiments/torchft/manager#TorchFTManager") wraps
  `torchft.Manager` over a `gloo`, `nccl` or `mccl` (torchcomms) process group
  ([`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L38 "torchtitan/experiments/torchft/manager#TorchFTManager.Config"): `enable`, `replica_id`,
  `group_size`, `min_replica_size`, `process_group`, `process_group_timeout_ms`,
  `semi_sync_method`). In synchronous mode (`semi_sync_method=None`) it uses an async quorum and
  a managed `dp_replicate` process group.
  [`maybe_set_all_reduce_hook`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L143 "torchtitan/experiments/torchft/manager#TorchFTManager.maybe_set_all_reduce_hook") installs
  an FSDP all-reduce hook that averages gradients across replica groups, and
  [`loss_sync_pg`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L157 "torchtitan/experiments/torchft/manager#TorchFTManager.loss_sync_pg") syncs the loss.
- **Optimizer.** [`TorchFTOptimizersContainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/optimizer.py#L25 "torchtitan/experiments/torchft/optimizer#TorchFTOptimizersContainer")
  drives quorum itself in synchronous mode:
  [`zero_grad`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/optimizer.py#L83 "torchtitan/experiments/torchft/optimizer#TorchFTOptimizersContainer.zero_grad")
  calls `manager.start_quorum()`, and
  [`step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/optimizer.py#L71 "torchtitan/experiments/torchft/optimizer#TorchFTOptimizersContainer.step")
  skips the update unless `manager.should_commit()` is true, then steps the inner optimizers
  directly (it no longer wraps them in `torchft.Optimizer`). Under semi-sync both hooks are off,
  because the semi-sync algorithm runs quorum at its own synchronization points. Optimizer state is
  initialized eagerly and cached for live recovery; before each save the cache's scalar metadata
  is refreshed in place
  ([`_refresh_cached_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/optimizer.py#L55 "torchtitan/experiments/torchft/optimizer#TorchFTOptimizersContainer._refresh_cached_state_dict")).
- **Semi-sync.** [`maybe_semi_sync_training`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/manager.py#L167 "torchtitan/experiments/torchft/manager#maybe_semi_sync_training") returns a
  `local_sgd.DiLoCo` or `local_sgd.LocalSGD` context. For DiLoCo it splits the model into
  fragments ([`fragment_llm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/diloco/utils.py#L112 "torchtitan/experiments/torchft/diloco/utils#fragment_llm"),
  [`FaultTolerantModelSpec`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/config/job_config.py#L81 "torchtitan/experiments/torchft/config/job_config#FaultTolerantModelSpec")) and gives each
  fragment an outer Nesterov SGD optimizer (lr 0.7, momentum 0.9). The extra knobs live in
  [`FaultTolerance`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/config/job_config.py#L15 "torchtitan/experiments/torchft/config/job_config#FaultTolerance"): `sync_steps` (default 5),
  `should_quantize`, `fragment_sync_delay` (Streaming DiLoCo "tau"), `fragment_update_alpha`,
  `num_fragments` and `module_fqns_per_model_fragment`.
- **Checkpointing.** [`TorchFTCheckpointManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/checkpoint.py#L45 "torchtitan/experiments/torchft/checkpoint#TorchFTCheckpointManager")
  saves a per-replica dataloader checkpoint every step, asynchronously, under
  `ft-replicat-<replica_id>` (sic). It does this so a recovering group does not replay data.
  Only the participating rank 0 writes the full checkpoint. This is the code behind the README's
  "the alive replica group with the smallest replica ID will perform checkpointing".
- **Recipe.** [`llama3_torchft_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/llama3/config_registry.py#L28 "torchtitan/experiments/torchft/llama3/config_registry#llama3_torchft_debugmodel").

## Why it matters / when it applies
It matters for long multi-node runs where a node failure would otherwise stop the job and
trigger a checkpoint reload. The README's DiLoCo note is now in tokens:
`--training.num_tokens_per_train_step` is the global budget across all islands, the per-island
"inner" budget is that divided by the number of replica groups, and the number of accumulated
forward/backward iterations per optimizer step is derived from it.
[`FaultTolerantTrainer.__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchft/trainer.py#L46 "torchtitan/experiments/torchft/trainer#FaultTolerantTrainer.__init__")
does exactly that: it computes `gradient_accumulation_steps = num_tokens_per_train_step //
(num_tokens_per_microbatch_per_dp_rank x num_pp_microbatches x batch_degree)`, where
`batch_degree` is the DP degree already widened by `group_size` through `get_dp_info`, and
raises if the budget is not divisible.

**TPU applicability.** This experiment is GPU-only as written. The process groups are gloo or
CUDA (`nccl`, `mccl` on `torch.device("cuda")`), the launch recipe uses `CUDA_VISIBLE_DEVICES`,
and nothing in `experiments/tpu` imports it. The torch_tpu lane
([torchtpu-backend](../concepts/torchtpu-backend.md)) has no equivalent. On TPU, resilience
comes from checkpoint/restart ([checkpointing](../concepts/checkpointing.md)).

> [!warning] Doc vs code
> - The README presents Example 1 (synchronous HSDP) and Example 2 (semi-synchronous) with the
>   *same* command lines. The config they both load, `llama3_torchft_debugmodel`, already sets
>   `semi_sync_method="diloco"`, `process_group="nccl"`, `sync_steps=10` and `num_fragments=2`.
>   So Example 1 also runs DiLoCo unless you override `--fault_tolerance.semi_sync_method`.
> - `TorchFTManager.Config.process_group` defaults to `gloo`. The README does not mention the
>   `mccl` option or `min_replica_size`.
> - The README spells the HSDP replicate flag `--data_parallel_replica_degree` and
>   `--parallelism.data_parallel_replica_degree`; neither exists. The field is
>   `--parallelism.data_parallel_replicate_degree`.

## Connections
- Code concepts: [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [training-loop](../concepts/training-loop.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [checkpointing](../concepts/checkpointing.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [job-configuration](../concepts/job-configuration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/torchft/manager`, `trainer`, `checkpoint`, `optimizer`).
- Related doc-concepts: [testing-strategy](testing-strategy.md),
  [experiments-policy](experiments-policy.md).

## Source
Extracted from `torchtitan/experiments/torchft/README.md` (kept in place).
