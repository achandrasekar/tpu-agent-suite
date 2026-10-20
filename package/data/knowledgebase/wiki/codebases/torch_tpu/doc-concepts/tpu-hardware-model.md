---
title: "TPU hardware model as seen from TorchTPU"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources: [docs/markdown/1_basics_1_intro_to_torchtpu.md, docs/markdown/intro_to_torchtpu.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TPU hardware model as seen from TorchTPU

## Definition
The guide sketches the hardware that TorchTPU programs target:
- **TensorCores.** Each has scalar and vector (VPU) units plus Matrix Multiply Units (MXUs).
  The MXUs are systolic arrays that "natively multiply `bfloat16` inputs and accumulate in
  `float32`".
- **SparseCores.** Dedicated processors for sparse work (embedding lookups, gather, scatter) and
  for collective communication.
- **HBM.** High-capacity on-package memory shared by the operations on a chip.
- **Interconnects.** Inter-Chip Interconnect (ICI) links join the chips of a slice in a 3D torus
  or twisted torus. Data Center Network (DCN) links join slices across pods.

The guide's one hardware-specific API is stream pinning: `torch.tpu.Stream(pinned="sparse")`
puts work on a SparseCore "to overlap collective communication with dense TensorCore math".

## In torch_tpu (grounded)
- **Generation facts.** [`TpuVersion`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L109 "src/torch_tpu/_internal/utils/hardware#TpuVersion") identifies
  the attached chip from its PCI device id. Per-generation tables give HBM per chip
  ([`_HBM_BYTES_PER_CHIP`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L165 "src/torch_tpu/_internal/utils/hardware#_HBM_BYTES_PER_CHIP._HBM_BYTES_PER_CHIP"),
  192 GiB on v7) and peak bf16 FLOP/s
  ([`get_bf16_flops_per_chip`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L232 "src/torch_tpu/_internal/utils/hardware#get_bf16_flops_per_chip")).
  [`get_devices_per_chip`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L216 "src/torch_tpu/_internal/utils/hardware#get_devices_per_chip") records
  how many PyTorch devices one chip exposes. Megacore chips (v4, v5p) fuse their two cores into
  one device. v7 exposes each of its two cores as a separate device, which is why a 4-chip v7 host
  is 8 devices (see [distributed-launch-environment](distributed-launch-environment.md)).
- **MXU precision.** Because the MXU is a bf16 engine, a float32 matmul's accuracy depends on how
  many bf16 passes XLA uses. The global default is the `medium` precision noted on
  [`PrecisionContextState`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/context_states.h#L70 "csrc/common/context_states.h#PrecisionContextState"), and
  [dtype-and-precision](../concepts/dtype-and-precision.md) explains how to raise it.
- **SparseCore pinning.** [`TpuStream`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/stream/streams.py#L86 "src/torch_tpu/_internal/stream/streams#TpuStream") accepts a
  `pinned=` keyword. [`_parse_pinned`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/stream/streams.py#L59 "src/torch_tpu/_internal/stream/streams#_parse_pinned") maps
  strings such as `"sparse"`, `"sc"` or `"dense"` onto
  [`CorePinningMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/cache_key.h#L968 "csrc/common/cache_key.h#CorePinningMode") (`kUnpinned`, `kTensorCore`,
  `kSparseCore`). The pinning mode is part of the graph signature hashed by
  [`Traversal.BuildGraphKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/traversal.cc#L336 "csrc/eager/traversal.cc#Traversal.BuildGraphKey"), so the same ops
  compile to separate executables per core. A pinned module is tagged with an XLA compute-type
  frontend attribute: sparse offload for SparseCore, dense for TensorCore.

## Why it matters / when it applies
- Keep matmul inputs in bfloat16 to match the MXU. If float32 results look like bf16, check the
  precision setting before you suspect a bug.
- HBM is per chip, and on v7 the two devices of a chip share it. Use the per-chip tables when you
  size batches or shard parameters.
- ICI is much faster than DCN. Mesh axes with heavy traffic (tensor parallel) should stay inside a
  slice, and data parallel can span slices. `topology_aware_mesh` builds a mesh that follows
  this layout (see [tpu-dist-process-group](tpu-dist-process-group.md)).
- Pinning collectives to a SparseCore stream lets them overlap with TensorCore compute. Each
  pinning mode is a separate cache entry, so expect extra compiles the first time.

## Connections
- Code concepts: [hardware-introspection](../concepts/hardware-introspection.md),
  [dtype-and-precision](../concepts/dtype-and-precision.md),
  [compilation-cache-key](../concepts/compilation-cache-key.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/utils`,
  `src/torch_tpu/_internal/stream`, `csrc/common`).
- Related doc-concepts: [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [distributed-launch-environment](distributed-launch-environment.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` (kept in place).
`docs/markdown/intro_to_torchtpu.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
