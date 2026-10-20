---
title: "Distributed launch environment (python -m torch.tpu.distributed + torchrun)"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_5_distributed_on_gke.md
sources: [docs/markdown/1_basics_5_distributed_on_gke.md, docs/markdown/distributed_training_101.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Distributed launch environment (python -m torch.tpu.distributed + torchrun)

## Definition
The distributed GKE guide launches one process per TPU device with stock `torchrun`. Before
that, it runs one TorchTPU step that works out the TPU-specific environment:

```bash
eval $(python3 -m torch.tpu.distributed)
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
python3 -m torch.distributed.run --nproc_per_node=$WORLD_SIZE ./worker.py
```

"Distributed Training 101" uses a single-line form on a TPU VM:
`eval $(python -m torch.tpu.distributed | sed 's/^/export /')`, then
`torchrun --nproc_per_node=$WORLD_SIZE your_script.py`. The worker side of that recipe is on
[tpu-dist-process-group](tpu-dist-process-group.md).

The same variables drive the VM and the GKE paths. On a 4-chip v7 host, `TORCH_TPU_TOPOLOGY` is
`2,2,1,2` "because it will launch with a process per TPU core". The multi-host JobSet runs the
same pattern on each of its 16 hosts, with `--nnodes`, `--node_rank` and the master address
filled in from the printed values.

## In torch_tpu (grounded)
- **The bootstrap module.** `torch.tpu.distributed` is a public namespace declared with a
  [`ModuleSpec`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/public_api.py#L55 "src/torch_tpu/_internal/public_api#ModuleSpec") whose `main` is the internal
  `distributed` module. Running it with `python -m` calls
  [`_print_launch_env`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L479 "src/torch_tpu/_internal/distributed/distributed#_print_launch_env"), which
  calls [`set_tpu_launch_env`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L328 "src/torch_tpu/_internal/distributed/distributed#set_tpu_launch_env")
  and prints shell assignments for `TORCH_TPU_TOPOLOGY`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`,
  `TORCH_TPU_XPROF_SESSION_ID`, `WORLD_SIZE` and `NODE_RANK`, plus `NNODES`, `MASTER_ADDR` and
  `MASTER_PORT` when known. The same function is callable from Python, so it can be used with
  `torch.multiprocessing.spawn` or any other launcher. It only edits `os.environ`, never
  overwrites a variable that is already set, and does not spawn processes.
- **Single host** (fewer than two `TPU_WORKER_HOSTNAMES` and no `JOB_COMPLETION_INDEX`). The
  topology is read from the local PCI devices by
  [`get_tpu_topology`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L332 "src/torch_tpu/_internal/utils/hardware#get_tpu_topology") (for example
  `2,2,1,2` for 8 v7 devices). The slicebuilder addresses are `localhost:<unused port>`, one per
  device.
- **Multi host.** The addresses are `<host>:<TORCH_TPU_BASE_PORT + i>` across all worker hosts
  (base port 8070, `TORCH_TPU_DEVICES_PER_HOST` devices per host). `MASTER_ADDR` is the first
  host. `NODE_RANK` comes from
  [`_calculate_topological_node_rank`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/distributed/distributed.py#L191 "src/torch_tpu/_internal/distributed/distributed#_calculate_topological_node_rank"):
  hosts are sorted Z-major by physical coordinates when a slice-wide `TORCH_TPU_TOPOLOGY` was
  supplied from outside, and ranked by `JOB_COMPLETION_INDEX` otherwise. The coordinate query
  runs in a short-lived child process, so the parent does not claim the TPU devices that the
  workers need.
- **Consumption.** Each worker checks that `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
  `MASTER_PORT`, `TORCH_TPU_SLICEBUILDER_ADDRESSES` and `TORCH_TPU_TOPOLOGY` are all present, in
  [`ValidateRequiredDistributedEnvVars`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/slicebuilder/discovery.cc#L32 "csrc/distributed/slicebuilder/discovery.cc#ValidateRequiredDistributedEnvVars").
  [`GetDistributedWorkerConfiguration`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/slicebuilder/discovery.cc#L65 "csrc/distributed/slicebuilder/discovery.cc#GetDistributedWorkerConfiguration")
  then picks this rank's slicebuilder port and builds the ICI mesh configuration that the PjRt
  client and the `tpu_dist` process group use.

## Why it matters / when it applies
All ranks must agree on the address list and the topology. That is why the values are computed
once in the parent, before the launcher forks, and inherited by every worker. If a multi-chip job
hangs or fails at start-up, check these first: a missing variable fails fast with "missing
required environment variables for distributed training". A topology that does not match the
process count, such as a v7 value without the fourth devices-per-chip component, or a
`TORCH_TPU_TOPOLOGY` that disagrees with the GKE slice, shows up later as a mesh error. Pin a
variable from outside (as the GKE manifests do with `TORCH_TPU_TOPOLOGY`) when auto-detection
cannot see the whole slice.

## Connections
- Code concepts: [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [pjrt-client](../concepts/pjrt-client.md),
  [hardware-introspection](../concepts/hardware-introspection.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/distributed`,
  `src/torch_tpu/_internal/utils`, `csrc/distributed/slicebuilder`).
- Related doc-concepts: [gke-xpk-workloads](gke-xpk-workloads.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [tpu-hardware-model](tpu-hardware-model.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md).

## Source
Extracted from `docs/markdown/1_basics_5_distributed_on_gke.md` (kept in place).
The single-line VM variant is from `docs/markdown/distributed_training_101.md` (kept in place).
