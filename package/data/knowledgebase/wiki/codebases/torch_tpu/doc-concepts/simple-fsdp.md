---
title: "SimpleFSDP: compile-first FSDP over DTensor"
type: doc-concept
provenance: doc
source: examples/fsdp/fsdp_to_simple_fsdp/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# SimpleFSDP: compile-first FSDP over DTensor

## Definition
SimpleFSDP is a compile-first Fully Sharded Data Parallel implementation, vendored from
TorchTitan into `examples/fsdp/simple_fsdp.py`, and "one of the recommended approaches to
implement FSDP (and HSDP) on TorchTPU". Parameters live as sharded `DTensor`s; accessing a
parameter in forward runs a parametrization that `redistribute`s it (all-gather), and the
autograd backward of that `redistribute` performs the gradient reduce-scatter. Because there are
no Python runtime hooks, forward, loss and backward compile into a single XLA program with no
graph breaks. The whole model is wrapped with one call,
`simple_fsdp.data_parallel(model, device_mesh=mesh, mode=..., mp_policy=...)`, with modes
`"replicate"`, `"fully_shard"` and `"hybrid_shard"` (HSDP on a 2-D `DeviceMesh`); no per-layer
`fully_shard` calls or prefetch bucket tuning. It composes with activation checkpointing and
`torch.distributed.checkpoint` (weights saved in sharded DTensor form).

## In torch_tpu (grounded)
- `dist.init_process_group(backend="tpu_dist")` creates the TorchTPU process group
  ([`MakeProcessGroup`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/distributed/tpu_distributed.cc#L49 "csrc/internal/distributed/tpu_distributed.cc#MakeProcessGroup") ->
  [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu")), and
  `init_device_mesh("tpu", (world_size,), mesh_dim_names=("fsdp",))` builds the mesh over it.
  For multi-axis meshes, the experimental `torch.tpu.distributed.topology_aware_mesh`
  ([`topology_aware_mesh`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/_distributed_module.py#L111 "src/torch_tpu/_internal/distributed/_distributed_module#topology_aware_mesh")) maps
  logically adjacent ranks to physically adjacent chips.
- The all-gather and reduce-scatter that `redistribute` emits correspond to
  [`_allgather_base`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.cc#L2053 "csrc/distributed/process_group_tpu.cc#ProcessGroupTpu._allgather_base") and
  [`_reduce_scatter_base`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.cc#L2385 "csrc/distributed/process_group_tpu.cc#ProcessGroupTpu._reduce_scatter_base")
  in eager mode; under `torch.compile(model, backend="tpu")` they are traced into the single
  program by [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend").
- On import, TorchTPU patches `DTensor.from_local`
  ([`_patch_dtensor`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L192 "src/torch_tpu/_loader#_patch_dtensor") ->
  [`from_local`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/from_local.py#L43 "src/torch_tpu/_internal/distributed/from_local#from_local")) to warn when shape and stride
  are omitted, a guardrail relevant when building sharded parameters by hand.

## Why it matters / when it applies
- **Gradient scaling differs from FSDPv2.** SimpleFSDP reduces gradients with
  `Partial(reduce_op="sum")`, while FSDPv2 averages by default; scale the loss by
  `1.0 / world_size` before `.backward()` (or scale the learning rate) to match.
- **Activation checkpointing per parameterized layer** frees each layer's unsharded weights after
  forward and re-all-gathers them during backward recomputation; the README recommends it.
- **Options:** `shard_dim` (int or per-parameter dict; parameters smaller than the mesh dimension
  along it are replicated), `ignored_params` (frozen or tied weights), `MixedPrecisionPolicy`
  (`param_dtype`, `reduce_dtype`, for example bf16 reductions), and
  `disable_active_parametrization()` to inspect the underlying sharded DTensor or its `.grad`
  without triggering an all-gather.
- The example (`run_fsdp_to_simple_fsdp.sh`, 8 TPU cores) compares compiled FSDPv2 and compiled
  SimpleFSDP step time and numerical parity; `fsdp_to_simple_fsdp.md` is the side-by-side guide.

## Connections
- Code concepts: [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [dtype-and-precision](../concepts/dtype-and-precision.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/distributed`,
  `src/torch_tpu/_internal/distributed`).
- Related doc-concepts: [tpu-dist-process-group](tpu-dist-process-group.md) (DDP, FSDP2 and
  DTensor on `tpu_dist`), [whole-block-compile-fusion](whole-block-compile-fusion.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md),
  [user-guide-notebooks](user-guide-notebooks.md).

## Source
Extracted from `examples/fsdp/fsdp_to_simple_fsdp/README.md` (kept in place).
