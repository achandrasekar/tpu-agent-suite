---
title: 'Area: csrc'
type: area
area: 'csrc'
description: 'Area csrc of torch_tpu: the C++ runtime layer under the op code - PjRt client and device state, the tpu_dist collectives backend, and the MLIR optimization passes run on every StableHLO module before compilation.'
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Area: `csrc`

## Purpose
`csrc/` is the C++ core of TorchTPU; this area page covers the three units under it that sit
between the op layer and the TPU runtime: the PjRt runtime state, the `tpu_dist` collectives
backend, and the StableHLO optimization pipeline. Their entry points are the process singleton
[`PjrtBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.h#L66 "csrc/pjrt/pjrt_state.h#PjrtBackend"),
reached through [`GetInstance`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.cc#L188 "csrc/pjrt/pjrt_state.cc#PjrtBackend.GetInstance")
and [`GetClient`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.cc#L337 "csrc/pjrt/pjrt_state.cc#PjrtBackend.GetClient");
the `c10d::Backend` subclass [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu")
that `init_process_group(backend="tpu_dist")` creates; and
[`RunTorchTpuOptimizationPasses`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/pipeline.cc#L61 "csrc/mlir/pipeline.cc#RunTorchTpuOptimizationPasses"),
which every built StableHLO module goes through before XLA compiles it.

What the area hands to the rest of the repo is a lazily initialized client and one addressable
device per process (consumed by the eager runtime, the compilation cache, transfers and
executions), collectives that are just more deferred StableHLO ops in the same program, and an
optimized module ready for compilation. The load-bearing mechanisms each have their own page:
[pjrt-client](../concepts/pjrt-client.md) for initialization, device discovery, fork guards and
the execute/transfer helpers; [process-group-and-collectives](../concepts/process-group-and-collectives.md)
for rank-to-device mapping, replica groups and graph splitting around collectives; and
[mlir-optimization-passes](../concepts/mlir-optimization-passes.md) for the pass order, the
norm-weight reassociation rewrite and the IR-dump switches.

## How the units connect
- `csrc/pjrt` is the base the other two stand on. `PjrtBackend` initialization reads the worker
  configuration from `csrc/distributed/slicebuilder/discovery` when `WORLD_SIZE` is set, and
  `ProcessGroupTpu` in turn asks `PjrtBackend` for its client, device and device attributes to
  build the shared rank-to-device mapping.
- Inside `csrc/distributed`, `process_group_tpu` owns the collectives and dispatches builders
  from `allreduce`, `allgather`, `reduce_scatter` and `alltoall`; those share `utils` (the
  `replica_groups` and channel attributes for StableHLO collectives) and `types`, whose
  [`DeviceGroupList`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/types.h#L49 "csrc/distributed/types.h#DeviceGroupList")
  is "a normalized list of TPU device groups" whose fingerprint goes into cache keys and whose
  [`operator<<`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/types.cc#L54 "csrc/distributed/types.cc#operator")
  prints it for logs. `window_tpu` (one-sided DMA/RMA transfers) and `handshake`
  (`TORCH_TPU_SPMD_HANDSHAKE_STAGE`) are small side modules covered on the collectives and
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md) pages.
- `csrc/mlir` has a single production caller, the module-construction step in
  `csrc/common/compilation.cc`, so it runs after the op builders (including the collective
  builders above) and before compilation on the PjRt client.
  [`AddTorchTpuOptimizationPasses`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/pipeline.cc#L54 "csrc/mlir/pipeline.cc#AddTorchTpuOptimizationPasses")
  fills the pass manager, [`createReassociateNormWeightsPass`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/passes/reassociate_norm_weights.cc#L187 "csrc/mlir/passes/reassociate_norm_weights.cc#createReassociateNormWeightsPass")
  is the one torch_tpu rewrite, `instrumentation` attaches optional per-pass IR dumps and
  timing, and `registration` exposes the custom passes to MLIR's global pass registry for
  tooling.

<!-- area:auto:begin -->
## Units
3 planned unit(s) in `csrc`; deep units have their own mechanism page, the rest are sections below.

| Unit | Modules | Symbols | Callers outside | Entry points | Page |
|---|---|---|---|---|---|
| `csrc/distributed` | 15 | 192 | 32 | [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu"), [`operator<<`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/types.cc#L54 "csrc/distributed/types.cc#operator"), [`DeviceGroupList`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/types.h#L49 "csrc/distributed/types.h#DeviceGroupList") | [process-group-and-collectives](../concepts/process-group-and-collectives.md) |
| `csrc/pjrt` | 8 | 99 | 59 | [`GetInstance`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.cc#L188 "csrc/pjrt/pjrt_state.cc#PjrtBackend.GetInstance"), [`PjrtBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.h#L66 "csrc/pjrt/pjrt_state.h#PjrtBackend"), [`GetClient`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/pjrt/pjrt_state.cc#L337 "csrc/pjrt/pjrt_state.cc#PjrtBackend.GetClient") | [pjrt-client](../concepts/pjrt-client.md) |
| `csrc/mlir` | 6 | 37 | 2 | [`RunTorchTpuOptimizationPasses`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/pipeline.cc#L61 "csrc/mlir/pipeline.cc#RunTorchTpuOptimizationPasses"), [`AddTorchTpuOptimizationPasses`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/pipeline.cc#L54 "csrc/mlir/pipeline.cc#AddTorchTpuOptimizationPasses"), [`createReassociateNormWeightsPass`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/mlir/passes/reassociate_norm_weights.cc#L187 "csrc/mlir/passes/reassociate_norm_weights.cc#createReassociateNormWeightsPass") | [mlir-optimization-passes](../concepts/mlir-optimization-passes.md) |

<!-- area:auto:end -->
