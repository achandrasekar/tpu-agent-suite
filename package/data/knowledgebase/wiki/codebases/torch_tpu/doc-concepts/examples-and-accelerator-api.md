---
title: "Examples layout and the accelerator device API"
type: doc-concept
provenance: doc
source: examples/README.md
sources: [examples/README.md, examples/tutorials/large_language_model/README.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Examples layout and the accelerator device API

## Definition
`examples/` collects "high-quality examples for using TorchTPU" of increasing complexity, from a
hello-world tensor op to scaling workloads "with eager and compile modes"; the README warns they
"demonstrate functionality and have not been optimized for TPU performance". Its canonical
snippet gets the device with `torch.accelerator.current_accelerator()` (not a TPU-specific
call), creates a tensor on it and adds it to itself. The model list names `simple_nn` and
[classical_ml](tpu-training-loop-hygiene.md) (MLP training with TensorBoard plus XProf,
eager versus `torch.compile`).

## In torch_tpu (grounded)
- `torch.accelerator.current_accelerator()` resolves to the TPU because import-time registration
  ([`load`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L379 "src/torch_tpu/_loader#load") calling [`_init_device_impl`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L206 "src/torch_tpu/_loader#_init_device_impl")) renames
  the PrivateUse1 backend to `tpu` and installs the device module from
  [`get_device_module`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L1123 "src/torch_tpu/_internal/device/_device_module#get_device_module").
- `x + x` dispatches through [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp") and executes only when
  `print` materializes the result
  ([deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)).
- `torch.compile(model)` with no backend goes to
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend") via the loader's
  [`_default_tpu_compile`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L153 "src/torch_tpu/_loader#_default_tpu_compile") wrapper (which also defaults `dynamic=False`).
- Distributed examples use the `tpu_dist` group created by
  [`MakeProcessGroup`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/distributed/tpu_distributed.cc#L49 "csrc/internal/distributed/tpu_distributed.cc#MakeProcessGroup").

## Why it matters / when it applies
- The examples show the intended API shape: accelerator-generic device selection, unchanged
  model code, `torch.compile` with the default backend. Because they are not performance
  references, use the concept pages on caching, dynamic shapes and collectives when tuning, and
  the tuned walkthroughs ([FLUX tutorial](whole-block-compile-fusion.md),
  [classical ML recipe](tpu-training-loop-hygiene.md)).
- Stale pointers at this pin: the README's `compile/example/simple.py` and
  `examples/distributed/` do not exist. The actual layout is `simple_pytorch/` (single-device
  script plus Kubernetes job manifests), `tensor_parallel/` (`dtensor_wrapped/` and `manual/`),
  `fsdp/` (`simple_fsdp.py`, `manual/`, and the [SimpleFSDP guide](simple-fsdp.md)),
  `classical_ml/`, and `tutorials/` (`FLUX_optimization/`, a `distributed/` hello world, and
  `large_language_model/`, whose README is only an "Open in Colab" link to
  `hello_world_llm.ipynb`, a Hugging Face LLM run on TPU).

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [process-group-and-collectives](../concepts/process-group-and-collectives.md).
- Module catalogs: [module map](../catalog/index.md) (`src/torch_tpu/_loader.py`,
  `src/torch_tpu/_internal/device`).
- Related doc-concepts: [feature-overview](feature-overview.md),
  [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md),
  [async-micro-batching-and-deferred-loss](async-micro-batching-and-deferred-loss.md),
  [simple-fsdp](simple-fsdp.md), [user-guide-notebooks](user-guide-notebooks.md).

## Source
Extracted from `examples/README.md` and `examples/tutorials/large_language_model/README.md`
(kept in place).
