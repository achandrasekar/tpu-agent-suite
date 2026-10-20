---
title: "Distributed-code conventions: explicit meshes, placements and gradient placements"
type: doc-concept
provenance: doc
source: .claude/rules/distributed.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Distributed-code conventions

## Definition
`.claude/rules/distributed.md` (scoped to `torchtitan/distributed/**`) says distributed code must
never assume a 1D mesh (assert on mesh axes before use), must validate `Replicate` / `Shard` /
`Partial` placements explicitly, and must state `grad_placements` on every `DTensor.to_local`,
including the indirect forms: `in_grad_placements` of `local_map` and `grad_placements` of
`full_tensor`. It asks contributors to reason about every parallelism combination, keeps
model-agnostic helpers (`maybe_enable_async_tp`, `NoParallel`, TP utilities) in
`torchtitan/distributed/`, and asks for conservative, numerics-verified changes because silent
errors (wrong gradient placements, identity ops that break DCP) are the typical failure.

## In torchtitan (grounded)
- **Named-axis mesh access.** Axes are looked up by name on
  [`ParallelDims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L105 "torchtitan/distributed/parallel_dims#ParallelDims") through
  [`get_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L435 "torchtitan/distributed/parallel_dims#ParallelDims.get_mesh") and
  [`get_optional_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L367 "torchtitan/distributed/parallel_dims#ParallelDims.get_optional_mesh"); model
  parallelize functions such as [`parallelize_deepseekv3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/parallelize.py#L21 "torchtitan/models/deepseek_v3/parallelize#parallelize_deepseekv3")
  ask for `["dp_replicate", "fsdp"]` or `["dp_replicate", "efsdp"]` (or just `["fsdp"]` /
  `["efsdp"]` when replication is off) instead of indexing a mesh; `get_mesh` raises `ValueError`
  for an invalid or disabled axis combination. Under `spmd_backend="spmd_types"` the typechecked
  regions run on named global meshes,
  [`spmd_dense_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L461 "torchtitan/distributed/parallel_dims#ParallelDims.spmd_dense_mesh") and
  [`spmd_sparse_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L467 "torchtitan/distributed/parallel_dims#ParallelDims.spmd_sparse_mesh")
  (expert dispatch), and [`spmd_axes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/spmd_types.py#L57 "torchtitan/distributed/spmd_types#spmd_axes")
  rejects any layout whose mesh axes are not named ("TorchTitan SPMD layouts require named mesh
  axes").
- **Declared gradient placements.** Placements are data on each module config:
  [`ShardingConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/sharding.py#L60 "torchtitan/protocols/sharding#ShardingConfig") and
  [`LocalMapConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/sharding.py#L34 "torchtitan/protocols/sharding#LocalMapConfig"). The shared GQA helper
  [`set_gqa_inner_attention_local_map`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder_sharding.py#L291 "torchtitan/models/common/decoder_sharding#set_gqa_inner_attention_local_map") passes explicit
  `in_grad_placements` for q, k and v, equal to their input placements over every SPMD axis (CP
  collectives now run inside the CP kernels, so no `Partial` gradient leaves the `local_map`). That
  is the rule applied: the gradient placement is stated, not inferred.
- **Model-agnostic helpers live here.** [`NoParallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/tensor_parallel.py#L19 "torchtitan/distributed/tensor_parallel#NoParallel"),
  [`_maybe_enable_async_tp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/compile.py#L75 "torchtitan/distributed/compile#_maybe_enable_async_tp"),
  [`apply_fsdp_to_decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L189 "torchtitan/distributed/fsdp#apply_fsdp_to_decoder") and
  [`prepare_context_parallel_input`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/context_parallel/api.py#L39 "torchtitan/distributed/context_parallel/api#prepare_context_parallel_input")
  are shared by every model rather than copied per model.

## Why it matters / when it applies
On TPU the same DTensor / `local_map` code is traced into one SPMD program, so a missing or wrong
`grad_placements` is not caught by a crash: it becomes a wrong collective (or a missing reduction)
inside the compiled graph and shows up only as a loss mismatch. Two TPU-specific cases follow from
the rule:
- **Mesh order is not rank order on TPU.** With `enable_tpu_topo_mesh`,
  [`init_device_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L24 "torchtitan/distributed/parallel_dims#init_device_mesh") builds the mesh from a topology-aware
  rank permutation and keeps it (`preserve_rank_order=True`); its comment says sorted sub-group
  ranks would make XLA fall back from the N-D SparseCore collective strategy to a 1-D ring. Code
  that assumes rank `i` sits at mesh coordinate `i` is exactly the "assume a 1D mesh" bug.
- **Test every combination you enable.** The TPU recipes use FSDP-only, HSDP and FSDP+EP layouts
  (see [tpu-training-recipes](tpu-training-recipes.md)); a TP or CP change validated only on GPU
  meshes has not been validated for them.

The rule's own list of helpers is slightly out of date with the SPMD rework: much TP/SP work is no
longer a `parallelize_module` plan in `torchtitan/distributed/` but declarative `sharding_config`
set by per-model `sharding.py` helpers and applied by
[`parallelize`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L279 "torchtitan/protocols/module#Module.parallelize") (see [model-registry](model-registry.md)).
Context parallelism is now installed as a model-config rewrite
([model-config-transforms](model-config-transforms.md)) and requires `spmd_backend="spmd_types"`.

> [!warning] Doc vs code
> The rule names `maybe_enable_async_tp` as a `torchtitan/distributed/` helper. At this pin the
> function is the private `_maybe_enable_async_tp` in `torchtitan/distributed/compile.py`; there is
> no public `maybe_enable_async_tp` and nothing async-TP-related in `tensor_parallel.py`.

## Connections
- Code concepts: [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [context-parallel](../concepts/context-parallel.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [checkpointing](../concepts/checkpointing.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [repo-conventions](repo-conventions.md),
  [model-authoring-conventions](model-authoring-conventions.md),
  [tpu-multihost-launch](tpu-multihost-launch.md),
  [model-config-transforms](model-config-transforms.md).

## Source
Extracted from `.claude/rules/distributed.md` (kept in place).
