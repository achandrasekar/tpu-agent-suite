---
title: "GraphTrainer: full train-step graph capture and graph passes"
type: doc-concept
provenance: doc
source: torchtitan/experiments/graph_trainer/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# GraphTrainer: full train-step graph capture and graph passes

## Definition
The GraphTrainer README frames this experiment as "graph-based distributed training ... through
toolkit-style usage of PyTorch's compiler technologies": instead of compiling one
TransformerBlock at a time, it traces forward + loss + backward (and optionally
`optimizer.step`) into a single FX graph with a `make_fx`-based tracer, "without AOTAutograd
partitioning", and then runs a structured pipeline of graph passes over that one graph. Sharding
(SimpleFSDP), per-tensor activation memory policy, bucketing/overlap, Inductor compilation and
CUDA graphs are all expressed as passes on that graph. The default mode is `aot_fx_trace`.

## In torchtitan (grounded)
- **Trainer.** [`GraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/trainer.py#L142 "torchtitan/experiments/graph_trainer/trainer#GraphTrainer")
  subclasses the core trainer and swaps its `compile` field for
  [`GraphTrainerCompileConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/configs.py#L69 "torchtitan/experiments/graph_trainer/configs#GraphTrainerCompileConfig");
  eager recipes are converted with
  [`to_graph_trainer_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/configs.py#L256 "torchtitan/experiments/graph_trainer/configs#to_graph_trainer_config").
- **Capture.** [`minimal_fx_tracer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/make_fx_tracer.py#L405 "torchtitan/experiments/graph_trainer/make_fx_tracer#minimal_fx_tracer")
  traces the step once, threading module parameters/buffers (and optimizer state when present)
  through the graph as static inputs, and returns a
  [`TracedResult`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/make_fx_tracer.py#L328 "torchtitan/experiments/graph_trainer/make_fx_tracer#TracedResult");
  every later step replays it with
  [`run_traced`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/make_fx_tracer.py#L658 "torchtitan/experiments/graph_trainer/make_fx_tracer#run_traced").
- **Pass pipeline.** The trainer looks up `compile.pass_pipeline` in a registry and falls back to
  [`construct_default_graph_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L420 "torchtitan/experiments/graph_trainer/passes#construct_default_graph_passes"),
  which combines [`compile_time_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L150 "torchtitan/experiments/graph_trainer/passes#compile_time_passes")
  (graph cleanup; memory-policy tagging, CPU offload and remat; EP-overlap chunking;
  transformer-block bucketing for comm/compute overlap; async TP) with the terminal
  [`final_inductor_compile_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L362 "torchtitan/experiments/graph_trainer/passes#final_inductor_compile_passes")
  ([`regional_inductor_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/inductor_passes.py#L170 "torchtitan/experiments/graph_trainer/inductor_passes#regional_inductor_pass")
  by default, [`full_inductor_compilation_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/inductor_passes.py#L329 "torchtitan/experiments/graph_trainer/inductor_passes#full_inductor_compilation_pass")
  with `--compile.inductor_compilation full`) and [`cudagraph_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/cudagraph.py#L261 "torchtitan/experiments/graph_trainer/cudagraph#cudagraph_pass").
  [`apply_graph_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L487 "torchtitan/experiments/graph_trainer/passes#apply_graph_passes")
  runs the list; `--compile.disable_passes` removes passes by exact function name
  ([`_filter_disabled_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L469 "torchtitan/experiments/graph_trainer/passes#_filter_disabled_passes")).
  `--compile.numerics_changing_optim` adds
  [`annotate_rmsnorm_for_regional_inductor_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/performance_passes.py#L19 "torchtitan/experiments/graph_trainer/performance_passes#annotate_rmsnorm_for_regional_inductor_pass");
  async TP is [`async_tensor_parallel_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L100 "torchtitan/experiments/graph_trainer/passes#async_tensor_parallel_pass").
- **SimpleFSDP.** [`data_parallel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/simple_fsdp.py#L414 "torchtitan/experiments/graph_trainer/simple_fsdp#data_parallel")
  and [`ReplicateComputation`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/simple_fsdp.py#L281 "torchtitan/experiments/graph_trainer/simple_fsdp#ReplicateComputation")
  express sharding as parametrized collectives, so all-gathers and reduce-scatters appear as
  ordinary nodes in the traced graph.
- **Memory policy.** [`tag_with_memory_policy_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/memory_policy.py#L594 "torchtitan/experiments/graph_trainer/memory_policy#tag_with_memory_policy_pass")
  tags each activation, [`apply_cpu_offload_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/cpu_offload.py#L440 "torchtitan/experiments/graph_trainer/cpu_offload#apply_cpu_offload_pass")
  inserts offload/reload pairs, and [`selective_activation_remat_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/selective_activation_remat.py#L101 "torchtitan/experiments/graph_trainer/selective_activation_remat#selective_activation_remat_pass")
  duplicates recomputed regions into the backward. Under `--compile.memory_policy full`
  (full recompute that keeps layer outputs), `--compile.full_recompute_save_ops` adds
  `MODULE_FQN_PATTERN::OP` selectors joined by `|` (e.g.
  `layers.*.moe.router.gate::aten.mm.dtype`) whose matching ops are saved instead of recomputed;
  [`validate_memory_policy_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/memory_policy.py#L129 "torchtitan/experiments/graph_trainer/memory_policy#validate_memory_policy_config") rejects the flag under any other policy, and
  [`_make_full_memory_policy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/memory_policy.py#L165 "torchtitan/experiments/graph_trainer/memory_policy#_make_full_memory_policy") applies the selectors.
- **AutoParallel.** `--compile.enable_autoparallel` routes placement through
  [`AutoParallelGraph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/autoparallel_api.py#L75 "torchtitan/experiments/graph_trainer/autoparallel_api#AutoParallelGraph")
  (an ILP-based SPMD placement solver), checked by
  [`validate_autoparallel_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/configs.py#L199 "torchtitan/experiments/graph_trainer/configs#validate_autoparallel_config");
  the placed model then goes through the same `aot_fx_trace` pipeline.

> [!warning] Doc vs code
> - The README's collapsed "legacy modes" section still lists an AOT mode (`--compile.mode aot`).
>   The code does not have it: `GraphTrainerCompileConfig.mode` is `Literal["jit", "aot_fx_trace"]`,
>   and `jit` is marked deprecated.
> - The README's CPU-offload example uses `--compile.memory_policy cpu_offload_all`. The code's
>   policy type is `Literal["default", "full", "eager", "sac_and_offload"]`, so that flag value is
>   rejected. Use `sac_and_offload` (SAC first, then offload what is still saved, within
>   `cpu_offload_budget_gb`).
> - The policy registry also has a `min_cut` policy (documented in the `memory_policy` docstring),
>   but it is not in that `Literal`, so it cannot be selected from the command line.

## Why it matters / when it applies
The README's case is that one graph gives "full visibility and control" over numerics and
performance: passes can be switched off one at a time for debugging (`--compile.no-enable_passes`,
`--compile.disable_passes ...`), and the default passes preserve numerics exactly. The
integration suite compares against eager bit for bit. Numerics-changing passes (full Inductor,
RMSNorm fusion) have to be turned on explicitly.

**On TPU.** This is the capture path that the TPU backend builds on, but the CUDA pass list does
not run there. [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer")
picks [`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer")
when `tpu_config.compile_mode == "aot_fx_trace"` (or `use_graph_trainer` is set).
[`to_tpu_graph_trainer_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/config.py#L110 "torchtitan/experiments/tpu/graph_trainer/config#to_tpu_graph_trainer_config")
copies `tpu_config.graph_trainer_memory_policy` into `compile.memory_policy`. That trainer
applies [`construct_tpu_pass_pipeline`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass.py#L647 "torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass#construct_tpu_pass_pipeline"),
which reuses the generic memory-policy, CPU-offload and remat passes and FSDP scheduling. It
does not use Inductor or CUDA graphs. Its last pass is
[`tpu_backend_compile_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass.py#L294 "torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass#tpu_backend_compile_pass"),
which lowers the whole graph to one static XLA executable. Flags that name CUDA passes
(`inductor_compilation`, `cudagraph_pass`, `numerics_changing_optim`) only affect the default
pipeline, so they do nothing on TPU. `--compile.full_recompute_save_ops` does not reach the TPU
trainer either (the conversion copies only the policy and the offload budget); the TPU way to say
"full recompute but keep X" is `graph_trainer_memory_policy=tpu_selective` with
`graph_trainer_save_regions`. The README's composability table marks Float8/MXFP8 in
`aot_fx_trace` as work in progress. Those formats cannot run on TPU in any case (see
[float8-training](float8-training.md)).

## Connections
- Code concepts: [torch-compile-integration](../concepts/torch-compile-integration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [training-loop](../concepts/training-loop.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [job-configuration](../concepts/job-configuration.md).
- Module catalogs: [module map](../catalog/index.md) (see `experiments/graph_trainer/*` and
  `experiments/tpu/graph_trainer/*`).
- Related doc-concepts: [graph-trainer-precompile](graph-trainer-precompile.md),
  [graph-pp-pipeline-parallel](graph-pp-pipeline-parallel.md),
  [ep-overlap-graph-chunking](ep-overlap-graph-chunking.md),
  [graph-trainer-autoresearch](graph-trainer-autoresearch.md),
  [kernel-overrides](kernel-overrides.md).

## Source
Extracted from `torchtitan/experiments/graph_trainer/README.md` (kept in place).
