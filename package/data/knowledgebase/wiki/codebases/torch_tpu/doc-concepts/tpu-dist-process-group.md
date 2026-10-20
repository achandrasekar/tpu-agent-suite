---
title: "The tpu_dist process group: DDP, FSDP2, DTensor and topology-aware meshes"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md, docs/markdown/distributed_training_101.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# The tpu_dist process group: DDP, FSDP2, DTensor and topology-aware meshes

## Definition
TorchTPU "implements TPU collectives at the `torch.distributed` Process Group (`c10d::Backend`)
level via `backend="tpu_dist"`". That backend supports DistributedDataParallel,
FullyShardedDataParallel (FSDP2 and SimpleFSDP) and DTensor. Onboarding a GPU job means changing
`backend="nccl"` to `backend="tpu_dist"`, with one process per TPU device, as in PyTorch's own
model. `torch.tpu.distributed.topology_aware_mesh` aligns logical parallelism dimensions with
the physical TPU interconnect.

## In torch_tpu (grounded)
- **Backend registration.** During device initialization,
  [`_init_device_impl`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L206 "src/torch_tpu/_loader#_init_device_impl") calls
  `torch.distributed.Backend.register_backend` for both `"tpu_dist"` and `"tpu"`, with
  `devices=["tpu"]`. Registration is unconditional, so single-host runs that call
  `init_process_group(backend="tpu_dist")` also work. It also registers the torchcomms backend
  ([`register_torchcomms_tpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/torchcomm_tpu.py#L838 "src/torch_tpu/_internal/distributed/torchcomm_tpu#register_torchcomms_tpu")).
- **The c10d backend.** `init_process_group` reaches
  [`MakeProcessGroup`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/distributed/tpu_distributed.cc#L49 "csrc/internal/distributed/tpu_distributed.cc#MakeProcessGroup"), which
  constructs a [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu") from the
  store, rank and size. Collectives are lowered to StableHLO collective ops over the ICI
  network that the launcher variables describe (see
  [distributed-launch-environment](distributed-launch-environment.md)).
- **DTensor.** At load time the loader patches `DTensor.from_local`
  ([`_patch_dtensor`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L192 "src/torch_tpu/_loader#_patch_dtensor")) so that it warns when `shape` and
  `stride` are omitted.
- **Topology-aware meshes.** [`topology_aware_mesh`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/device_mesh.py#L224 "src/torch_tpu/_internal/distributed/device_mesh#topology_aware_mesh")
  returns an int64 tensor of global ranks with shape `mesh_shape`, ready for
  `DeviceMesh("tpu", mesh=...)`. It all-gathers each rank's device coordinates, so every rank
  derives the same layout, and it needs an initialized process group. `mesh_shape` is ordered by
  increasing network intensity. On multi-slice jobs, one axis is split across slices over DCN,
  either inferred or set with `dcn_mesh_shape`, and the remainder is laid out on ICI. The public
  wrapper is [`torch.tpu.distributed.topology_aware_mesh`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/_distributed_module.py#L111 "src/torch_tpu/_internal/distributed/_distributed_module#topology_aware_mesh").

## Why it matters / when it applies
Standard PyTorch parallelism code (DDP, FSDP2, TorchTitan-style DTensor plans) runs unchanged once
the backend string and device are switched. Performance then depends on placement. If ranks are
mapped to a mesh naively, a tensor-parallel group can end up spread across distant chips or
across slices. Build the mesh with `topology_aware_mesh`, and put the most communication-heavy
axis last in `mesh_shape`. That keeps the axis on neighbouring ICI links and leaves DCN to the
lightest axis, usually data parallel.

## DDP worker recipe and the golden rules
The "Distributed Training 101" page gives the smallest data-parallel worker on an SPMD model:
1. **Launch.** Export the launcher environment, then run stock `torchrun`:
   `eval $(python -m torch.tpu.distributed | sed 's/^/export /')` followed by
   `torchrun --nproc_per_node=$WORLD_SIZE your_script.py`. The first command prints
   assignments from [`_print_launch_env`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L479 "src/torch_tpu/_internal/distributed/distributed#_print_launch_env")
   ([distributed-launch-environment](distributed-launch-environment.md)).
2. **Worker.** `torch.device("tpu")` discovers the hardware and registers `tpu_dist`
   ([`_init_device_impl`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L206 "src/torch_tpu/_loader#_init_device_impl")).
   Then call `dist.init_process_group(backend="tpu_dist")`. Each process sees only its own core,
   as `tpu:0`, so DDP is built as `DDP(model)` **without `device_ids`**, unlike CUDA. The example
   casts model and data to bf16 so their dtypes match.
3. **Step and materialize.** After `optimizer.step()`, `loss.item()` is the "mandatory
   materialization trigger" that keeps all ranks in lock-step.

Its golden rules:
- **Never diverge.** Do not wrap ops that materialize (`.item()`) in `if rank == 0`. "If one
  rank runs ahead, the mesh deadlocks". The code shows why. Every rank has to run the same
  collectives in the same order, so by default TorchTPU runs each eager collective in a graph
  of its own, so that ranks that fuse different surrounding ops cannot deadlock. Functions
  that really are identical on all ranks can opt into fusing collectives with their neighbours
  through [`spmd_safe`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/spmd_util.py#L45 "src/torch_tpu/_internal/distributed/spmd_util#spmd_safe")
  (`torch.tpu.distributed.spmd_safe`). Never decorate rank-dependent code with it.
- **Shared seeds.** Start every rank from the same weights by seeding identically
  (`torch.tpu.manual_seed_all`; see [cuda-to-tpu-api-mapping](cuda-to-tpu-api-mapping.md)).
- **Flush at exit.** The page says to call `torch.tpu.manual_seed_all(0)` after the loop "to
  ensure the final weight update is executed".

> [!warning] Not supported by the code at this pin: `manual_seed_all` only sets each device
> generator's seed ([`SetManualSeedAll`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/device_gen_impl.cc#L753 "csrc/eager/device_gen_impl.cc#SetManualSeedAll"))
> and does not call `Materialize`. To make sure deferred work has run before exit, use an explicit
> trigger such as `torch.accelerator.synchronize()` or a host read of a result
> ([eager-execution-modes](eager-execution-modes.md)).

## Connections
- Code concepts: [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md),
  [device-backend-registration](../concepts/device-backend-registration.md),
  [hardware-introspection](../concepts/hardware-introspection.md).
- Module catalog: [catalog index](../catalog/index.md) (`csrc/distributed`,
  `csrc/internal/distributed`, `src/torch_tpu/_internal/distributed`).
- Related doc-concepts: [distributed-launch-environment](distributed-launch-environment.md),
  [gke-xpk-workloads](gke-xpk-workloads.md),
  [tpu-hardware-model](tpu-hardware-model.md),
  [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [tpu-device-onboarding](tpu-device-onboarding.md),
  [cuda-to-tpu-api-mapping](cuda-to-tpu-api-mapping.md),
  [eager-execution-modes](eager-execution-modes.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place).
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
The "DDP worker recipe and the golden rules" section is from `docs/markdown/distributed_training_101.md` (kept in place).
