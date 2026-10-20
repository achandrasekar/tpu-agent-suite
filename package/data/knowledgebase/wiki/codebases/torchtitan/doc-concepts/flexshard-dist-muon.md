---
title: "FlexShard: optimizer compute layouts that differ from parameter storage (DistMuon)"
type: doc-concept
provenance: doc
source: torchtitan/distributed/flex_shard/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# FlexShard: optimizer compute layouts that differ from parameter storage (DistMuon)

## Definition
The FlexShard README describes "PyTorch-native building blocks for running optimizer compute
with layouts that differ from persistent DTensor parameter storage layouts". A parameter stays
stored however FSDP/EP shard it, but for the optimizer step FlexShard plans a packed
**storage-to-compute redistribution** into a temporary compute layout, runs the optimizer
there, and overlaps that communication with optimizer work. Its first consumer is
**DistMuon**, a distributed Muon optimizer, which needs whole matrices (or whole fixed-size
blocks of rows) on one rank to orthogonalize them, something a row-sharded FSDP parameter does
not provide.

## In torchtitan (grounded)
- **Public API** (`torchtitan.distributed.flex_shard`):
  - [`ComputeLayout`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L111 "torchtitan/distributed/flex_shard/optimizer_reshard#ComputeLayout") maps named `DeviceMesh` axes to a compute sharding via
    [`shardings_by_mesh_axis`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L165 "torchtitan/distributed/flex_shard/optimizer_reshard#ComputeLayout.shardings_by_mesh_axis"), and optionally orders several axes that shard
    the same tensor dimension via [`shard_order_by_tensor_dim`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L166 "torchtitan/distributed/flex_shard/optimizer_reshard#ComputeLayout.shard_order_by_tensor_dim") (outermost
    first; default is storage-mesh order). It is frozen and hashable so it survives config deep
    copies.
  - Each axis uses `Replicate`, `Shard`, [`BlockShard`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L35 "torchtitan/distributed/flex_shard/optimizer_reshard#BlockShard") or [`Owned`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L22 "torchtitan/distributed/flex_shard/optimizer_reshard#Owned"). `BlockShard`
    "partitions fixed-size blocks without changing the tensor shape" and never splits a block;
    `Owned` gives the complete subgroup-local tensor to one dynamically selected owner rank for
    the compute phase only ("this is not persistent DTensor storage ownership").
  - [`BucketConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/optimizer_reshard.py#L252 "torchtitan/distributed/flex_shard/optimizer_reshard#BucketConfig") (FQN `patterns` + `name`) groups parameters for scheduling: "Compute
    layouts determine communication topology. A bucket controls only scheduling order and
    overlap".
  - [`build_dist_muon`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L58 "torchtitan/distributed/flex_shard/dist_muon#build_dist_muon") constructs [`DistMuon`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L252 "torchtitan/distributed/flex_shard/dist_muon#DistMuon") from param groups plus
    `compute_sharding_by_fqn` and `bucket_configs`.
- **DistMuon's matrix rules.** A `BlockShard` parameter must be 2D `[M * R, C]` with
  contiguous local storage, target dim 0 with `block_size=R`, and have a nonzero leading
  dimension divisible by `R`; every `R` rows is an independent `[R, C]` matrix, viewed locally
  as `[M_local, R, C]` just before Muon compute (see [`_MatrixBatchView`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L651 "torchtitan/distributed/flex_shard/dist_muon#_MatrixBatchView")). A batch-first 3D
  `[M, R, C]` parameter (stacked experts) uses `Shard(0)`; a single 2D matrix uses whole-matrix
  compute such as `Owned`, with owners load-balanced by
  [`_assign_balanced_owner_ranks`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L1006 "torchtitan/distributed/flex_shard/dist_muon#_assign_balanced_owner_ranks"). Matrix-batch compute supports `BlockShard` on at most
  one non-unit mesh axis, whose storage may be exact `Shard(0)` or `Replicate`; every other
  non-unit storage axis must be replicated.
- **Multi-axis sharding of one dim.** The README's EP example keeps an EP-axis `Shard(0)` while
  repartitioning the local expert domain over a preceding EFSDP axis with
  `shard_order_by_tensor_dim={0: ("ep", "efsdp")}`; FlexShard derives each axis's split
  factor from the bound mesh and lowers the EFSDP placement to subgroup-local `Shard(0)`
  ([`_lower_shard_order_to_strided_shards`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/flex_shard/dist_muon.py#L1291 "torchtitan/distributed/flex_shard/dist_muon#_lower_shard_order_to_strided_shards")).
- **Optimizer registry.** The optimizer component registers the name `"DistMuon"` to
  `build_dist_muon`, and per-optimizer factory arguments flow through
  [`optimizer_factory_kwargs_by_name`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/optimizer/optimizer.py#L132 "torchtitan/components/optimizer/optimizer#OptimizersContainer.Config.optimizer_factory_kwargs_by_name").
- **Kimi integration.** [`_dist_muon_optimizer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L294 "torchtitan/models/kimi_k2_7/config_registry#_dist_muon_optimizer") is the shared optimizer config of the
  Kimi-family, Kimi-VL and Moonlight recipes ([`kimi_k2_5`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L222 "torchtitan/models/kimi_k2_7/config_registry#kimi_k2_5"), [`kimi_vl_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L176 "torchtitan/models/kimi_k2_7/config_registry#kimi_vl_a3b"),
  [`moonlight_16b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L133 "torchtitan/models/kimi_k2_7/config_registry#moonlight_16b_a3b")). A regex param group sends attention projections, dense MLP
  `w1/w2/w3`, routed-expert stacks, shared experts and the router gate to DistMuon and
  everything else (embeddings, norms, biases, LM head, vision tower) to AdamW. Layouts:
  `Owned` on the `dp_shard` axis for most matrices, per-head `BlockShard` for the query and
  key/value up-projections (block size = head dim), and a per-expert layout for routed experts.
  Buckets keep layer 0 (large dense MLP) alone and pair subsequent MoE layers, with routed
  experts in their own bucket. [`_KimiTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/kimi_k2_7/config_registry.py#L519 "torchtitan/models/kimi_k2_7/config_registry#_KimiTrainerConfig") rejects
  `tensor_parallel_degree > 1` because TP can produce unsupported `_StridedShard` layouts.

## Why it matters / when it applies
Muon-style optimizers operate on whole matrices, which conflicts with FSDP's flat row shards;
FlexShard makes the optimizer layout a per-parameter construction-time choice instead of
forcing parameters to be stored replicated. Two operational rules from the README: compute
sharding "is validated and frozen when the optimizer is built, but is not stored in its state
dict", so a checkpoint restore must rebuild the optimizer with matching layouts; and the
package is meant to stay self-contained and PyTorch-only (dependencies flow
"TorchTitan components and models -> FlexShard -> PyTorch") because it is intended to be split
into a standalone package. At this pin the `flex_shard` modules import only PyTorch and the
standard library. Nothing in the repo ties FlexShard to TPU; it is plain DTensor/`DeviceMesh`
collectives.

## Connections
- Code concepts: [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md) (the
  `DistMuon` registration and param groups),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [checkpointing](../concepts/checkpointing.md),
  [model-definitions](../concepts/model-definitions.md).
- Module catalogs: [module map](../catalog/index.md) (`distributed/flex_shard/*`,
  `models/kimi_k2_7/config_registry`).
- Related doc-concepts: [distributed-conventions](distributed-conventions.md),
  [deepseek-v3-model](deepseek-v3-model.md).

## Source
Extracted from `torchtitan/distributed/flex_shard/README.md` (kept in place).
