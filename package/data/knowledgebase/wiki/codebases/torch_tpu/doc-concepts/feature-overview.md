---
title: "TorchTPU feature overview (README)"
type: doc-concept
provenance: doc
source: README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# TorchTPU feature overview (README)

## Definition
The README frames TorchTPU as "a PyTorch backend for Google's Tensor Processing Units": custom
ATen kernels for PyTorch operations, a compilation cache "for optimized execution of XLA
computations", distributed training (data parallelism and tensor parallelism), integration with
`torch.compile()` for graph mode, utilities for debugging and benchmarking, and examples for
models such as Llama, Qwen, ResNet, DLRM, minGPT and UNet. This page maps each headline feature
to the mechanism behind it, for a reader onboarding PyTorch code to TPU.

## In torch_tpu (grounded)
- **ATen kernels (eager mode).** Every kernel funnels through
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp"), which records a deferred op instead of running it;
  work executes when a value is needed ([`Materialize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/materialize.cc#L812 "csrc/eager/materialize.cc#Materialize")). See
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)
  and [aten-kernel-authoring](../concepts/aten-kernel-authoring.md).
- **Compilation cache.** [`CompilationCache`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.h#L201 "csrc/common/compilation_cache.h#CompilationCache") and its
  [`GetOrCompile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation_cache.cc#L1015 "csrc/common/compilation_cache.cc#CompilationCache.GetOrCompile") implement the
  in-memory, host-local and persistent tiers; see
  [compilation-cache](../concepts/compilation-cache.md) and
  [compilation-cache-key](../concepts/compilation-cache-key.md).
- **Distributed training.** The `tpu_dist` process-group backend is created by
  [`MakeProcessGroup`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/distributed/tpu_distributed.cc#L49 "csrc/internal/distributed/tpu_distributed.cc#MakeProcessGroup") and implemented by
  [`ProcessGroupTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/distributed/process_group_tpu.h#L134 "csrc/distributed/process_group_tpu.h#ProcessGroupTpu"); DTensor-based sharding such as
  [simple-fsdp](simple-fsdp.md) sits on top. See
  [process-group-and-collectives](../concepts/process-group-and-collectives.md) and
  [distributed-launchers-and-spmd](../concepts/distributed-launchers-and-spmd.md).
- **torch.compile integration.** [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend") is the
  backend. The loader replaces `torch.compile` with
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile"), which picks the TPU backend when no backend
  is given and defaults `dynamic=False` for it; requesting `backend="inductor"` on TPU raises
  [`_INDUCTOR_TPU_ERROR`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L65 "src/torch_tpu/_loader#_INDUCTOR_TPU_ERROR._INDUCTOR_TPU_ERROR") unless
  explicitly bypassed. See [torch-compile-backend](../concepts/torch-compile-backend.md).
- **Debugging and benchmarking utilities.** [`OpTracer`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/utils.py#L327 "src/torch_tpu/_internal/utils/utils#OpTracer")
  and [`ActivationTracer`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/utils.py#L400 "src/torch_tpu/_internal/utils/utils#ActivationTracer") log what a model executes,
  [`synchronize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L762 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.synchronize") is the
  timing barrier, and [`profile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/profiler/_api.py#L119 "src/torch_tpu/_internal/profiler/_api#profile") captures XProf traces;
  see [synchronization](../concepts/synchronization.md) and [profiler](../concepts/profiler.md).

## Why it matters / when it applies
The README's feature list is the vocabulary a newcomer arrives with; the mechanisms are spread
over eager dispatch, deferred execution, compilation and distributed subsystems, so this mapping
is the shortest route from "the README says X" to "here is how X works". The README's "Getting
Started" pointers (tutorials, `examples/`, the public API list) are covered by
[examples-and-accelerator-api](examples-and-accelerator-api.md) and
[user-guide-notebooks](user-guide-notebooks.md).

## Connections
- Code concepts: [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md),
  [profiler](../concepts/profiler.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/common`,
  `csrc/distributed`, `src/torch_tpu/_internal/compile`).
- Related doc-concepts: [tpu-device-onboarding](tpu-device-onboarding.md),
  [eager-execution-modes](eager-execution-modes.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [building-and-installing](building-and-installing.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md),
  [step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md).

## Source
Extracted from `README.md` (kept in place).
