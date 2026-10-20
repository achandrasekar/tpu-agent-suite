---
title: "SingleTraceTrainer: the whole training step as one compiled graph"
type: doc-concept
provenance: doc
source: docs/markdown/advanced_compile.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# SingleTraceTrainer: the whole training step as one compiled graph

## Definition
The `torch.compile` guide shows that compiling a standard training step (`zero_grad`, forward,
loss, `loss.backward()`, `optimizer.step()`) produces graph breaks, because Dynamo "cannot inline
the eager autograd C++ engine or mutable optimizer operations into a single static graph". To
reach "maximum TPU utilization" the whole step (forward, loss, backward and optimizer update)
should be one static graph. `torch.tpu.optimizers.SingleTraceTrainer` builds that graph in three
moves: reverse-mode autodiff with `torch.func.grad_and_value`, a functional optimizer update
(`AdamW` or `Sgd`), and one `make_fx` capture of the whole pipeline. The result compiles with
`torch.compile(..., backend="tpu", dynamic=False, fullgraph=True)` with zero graph breaks and runs
as a single offload.

## In torch_tpu (grounded)
- **Public surface.** `torch.tpu.optimizers` is declared in the public-API table and exports
  `SingleTraceTrainer`, `AdamW`, `Sgd`, `Muon` and the base `Optimizer`.
  [`SingleTraceTrainer`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L265 "src/torch_tpu/_internal/optimizers/single_trace_trainer#SingleTraceTrainer")
  is marked experimental. Its docstring describes the goal as "a single offload containing the
  optimizer step", "reducing interpreter overhead and enabling better fusion opportunities by the
  compiler".
- **State lives outside the module.**
  [`SingleTraceTrainer.__init__`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L277 "src/torch_tpu/_internal/optimizers/single_trace_trainer#SingleTraceTrainer.__init__")
  detaches the model's parameters and buffers into flat dicts (dots in names become `_`), calls
  `tie_weights()` if the model has it, and asks the optimizer for a fresh `ParamGroup` through
  `init_param_group`. With `optimizer=None` the step only computes gradients and writes them to
  `param.grad`.
- **Functional optimizers.** The optimizers are stateless:
  [`Optimizer`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/optim.py#L33 "src/torch_tpu/_internal/optimizers/optim#Optimizer")
  maps `(param_group, grads)` to a new `ParamGroup`, and its hyperparameters "are captured as
  constants when a train step is traced, so they can't be changed afterwards". Changing the
  learning rate means building a new optimizer and a new step. The
  [`AdamW`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/adamw.py#L192 "src/torch_tpu/_internal/optimizers/adamw#AdamW")
  here defaults to `use_bfloat16_moments=True` and `capturable=True`; the guide's example turns
  bf16 moments off. [`Sgd`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/sgd.py#L175 "src/torch_tpu/_internal/optimizers/sgd#Sgd")
  and [`Muon`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/muon.py#L236 "src/torch_tpu/_internal/optimizers/muon#Muon")
  follow the same contract.
- **Tracing.** [`make_compiled_train_step`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L374 "src/torch_tpu/_internal/optimizers/single_trace_trainer#SingleTraceTrainer.make_compiled_train_step")
  flattens `(param_group, buffers, inputs[, targets])` and traces an inner
  `flattened_stateless_train_step` with `make_fx(..., tracing_mode="fake")`. Inside, the loss
  comes from [`_compute_loss`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L76 "src/torch_tpu/_internal/optimizers/single_trace_trainer#_compute_loss"):
  a user `compute_loss_fn` if given, otherwise MSE against the targets, otherwise the output's
  `loss` field (Hugging Face style), otherwise a mean of the logits or first output tensor.
  Gradients come from `func.grad_and_value`, and the optimizer update is traced into the same
  graph. Embedding backward ops are decomposed during tracing, and
  [`_declare_tracing_to_transformers`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L50 "src/torch_tpu/_internal/optimizers/single_trace_trainer#_declare_tracing_to_transformers")
  makes Hugging Face mask creation take its trace-safe path.
- **Compiling.** `compile_fn` is mandatory (on the constructor or on `make_compiled_train_step`)
  and is called by
  [`_invoke_compile_fn`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L139 "src/torch_tpu/_internal/optimizers/single_trace_trainer#_invoke_compile_fn")
  with `(graph_module, flat_inputs)` or just `(graph_module)`, depending on its signature. The
  guide's `compile_and_verify_breaks` uses that hook to run `torch._dynamo.explain` on the traced
  graph before returning `torch.compile(graph_module, backend="tpu", dynamic=False,
  fullgraph=True)`, which lands in
  [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend").
- **Running.** The returned
  [`train_step`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optimizers/single_trace_trainer.py#L515 "src/torch_tpu/_internal/optimizers/single_trace_trainer#SingleTraceTrainer.train_step")
  flattens the current state plus the new batch, runs the compiled graph, unflattens
  `(loss, new_param_group, updated_buffers)`, stores them back on the trainer and loads the
  buffers into the user's module. Passing targets is allowed only if example targets were given
  at trace time.

## Why it matters / when it applies
Every graph break splits a step into separate XLA executables: intermediates go back to HBM
instead of staying fused, and the host has to dispatch each piece
([torch-compile-graph-breaks](torch-compile-graph-breaks.md)). Capturing forward, backward and
optimizer together gives the compiler one program to fuse and schedule. The trade-offs come
from the functional design:
- The step is traced once for the example shapes. Keep batch shapes fixed, as for any
  `dynamic=False` compile.
- Hyperparameters are baked in. An LR schedule needs a new optimizer and a new trace, not a
  mutation.
- Read updated weights through `trainer.params` (original names), since the trainer, not the
  module, owns the parameter tensors.
- Without the trainer, the guide's lighter-weight advice for a hand-written compiled step is to
  build stateful optimizers with `capturable=True`. TorchTPU also ships a forked
  [`AdamW`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/optim/adamw.py#L26 "src/torch_tpu/_internal/optim/adamw#AdamW")
  that pre-allocates its state on the device to avoid graph breaks; see
  [fused-optimizers](../concepts/fused-optimizers.md).

## Connections
- Code concepts: [torch-compile-backend](../concepts/torch-compile-backend.md),
  [fused-optimizers](../concepts/fused-optimizers.md),
  [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache](../concepts/compilation-cache.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/optimizers`,
  `src/torch_tpu/_internal/optim`, `src/torch_tpu/_internal/compile`).
- Related doc-concepts: [torch-compile-graph-breaks](torch-compile-graph-breaks.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md),
  [simple-fsdp](simple-fsdp.md).

## Source
Extracted from `docs/markdown/advanced_compile.md` (kept in place).
