---
title: "GraphPP: graph-based pipeline parallelism for GraphTrainer"
type: doc-concept
provenance: doc
source: torchtitan/experiments/graph_trainer/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# GraphPP: graph-based pipeline parallelism for GraphTrainer

## Definition
The README defines GraphPP as "the `aot_fx_trace` pipeline-parallel path for GraphTrainer
models". Model splitting reuses TorchTitan's eager pipeline-parallel (PP) module splitting, and
scheduling reuses PyTorch's PP schedules. The difference is execution. GraphPP traces "one
representative microbatch per local stage" with `minimal_fx_tracer`, then reuses the resulting
per-stage graph bundles for every later microbatch. The runtime "only executes the prebuilt
callable for each PP schedule action". The per-stage graphs cover forward and backward,
optional FSDP `UNSHARD` / `REDUCE_GRAD`, optional split dI/dW backward, and multiplexed graphs
for `OVERLAP_F_B` actions.

## In torchtitan (grounded)
- **Entry.** [`graph_pipeline_llm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/pipeline.py#L66 "torchtitan/experiments/graph_trainer/graph_pp/pipeline#graph_pipeline_llm")
  builds the runtime, the model parts and the first/last-stage flags. Before that,
  [`_validate_graph_pp_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/pipeline.py#L40 "torchtitan/experiments/graph_trainer/graph_pp/pipeline#_validate_graph_pp_config")
  checks four things. Mode must be `aot_fx_trace`. No `precompile_artifact_dir` may be set. FSDP
  must not use `fsdp_reshard_after_forward=always`, because GraphPP assumes ZeRO-2. The schedule
  must be a runtime schedule class (`Interleaved1F1B`, `ZBVZeroBubble`, `DualPipeV`).
- **Runtime.** [`GraphPipelineRuntime`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/runner.py#L306 "torchtitan/experiments/graph_trainer/graph_pp/runner#GraphPipelineRuntime")
  registers action handlers on the upstream `_PipelineScheduleRuntime`. Upstream PP still owns
  microbatch splitting, action order and P2P. GraphPP owns graph dispatch, gradient scaling and
  final gradient accumulation.
  [`register_graph_pp_schedule`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/runner.py#L709 "torchtitan/experiments/graph_trainer/graph_pp/runner#register_graph_pp_schedule")
  wires it into a schedule.
- **Stage graphs.** [`GraphPipelineStage`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/stage.py#L310 "torchtitan/experiments/graph_trainer/graph_pp/stage#GraphPipelineStage")
  holds each stage's [`GraphPPStageGraphs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/stage.py#L21 "torchtitan/experiments/graph_trainer/graph_pp/stage#GraphPPStageGraphs").
  The joint graph is cut by
  [`partition_joint_graph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/partition.py#L439 "torchtitan/experiments/graph_trainer/graph_pp/partition#partition_joint_graph").
  FSDP collectives are split out by
  [`split_forward_fsdp_collectives`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/split_fsdp_collectives.py#L88 "torchtitan/experiments/graph_trainer/graph_pp/split_fsdp_collectives#split_forward_fsdp_collectives")
  and [`split_backward_fsdp_collectives`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/split_fsdp_collectives.py#L246 "torchtitan/experiments/graph_trainer/graph_pp/split_fsdp_collectives#split_backward_fsdp_collectives").
  Backward is split into dI and dW by
  [`split_di_dw_graph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/split_di_dw.py#L151 "torchtitan/experiments/graph_trainer/graph_pp/split_di_dw#split_di_dw_graph"),
  and [`multiplex_fw_bw_graph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/graph_pp/graph_multiplex.py#L44 "torchtitan/experiments/graph_trainer/graph_pp/graph_multiplex#multiplex_fw_bw_graph")
  builds the overlapped forward/backward graphs.
- **Compilation.** Regional and full Inductor are applied to the extracted callables through
  the shared [`final_inductor_compile_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L362 "torchtitan/experiments/graph_trainer/passes#final_inductor_compile_passes").
  GraphPP first calls [`compile_time_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L150 "torchtitan/experiments/graph_trainer/passes#compile_time_passes")
  with `include_inductor=False` so that it can partition before compiling.
- **Launch flags.** The README's example passes `--parallelism.num_pp_microbatches 8`
  ([`num_pp_microbatches`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L259 "torchtitan/config/configs#ParallelismConfig.num_pp_microbatches")
  defaults to 1, microbatches per DP rank and accumulation step) and
  `--training.disable_cuda_graphs`. The core trainer captures CUDA graphs by default
  ([`disable_cuda_graphs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L77 "torchtitan/config/configs#TrainingConfig.disable_cuda_graphs")
  is `False`), and
  [`_validate_cuda_graphs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L186 "torchtitan/trainer#Trainer.Config._validate_cuda_graphs")
  rejects looped schedules such as `Interleaved1F1B` while CUDA graphs are on, so the flag is
  required for every GraphPP runtime schedule. GraphPP itself compiles its stage graphs with
  `use_cudagraph=False` and only warns when `cudagraph_pass` is left enabled.

## Why it matters / when it applies
In eager PP, every microbatch dispatches its operators again. GraphPP traces each stage once and
replays it. It keeps the README's "subclass boundary": graphs run on flat plain-tensor leaves,
and only values that leave GraphPP (stage outputs, input gradients, parameter gradients) are
rewrapped as DTensors. Limitations the README lists: no precompile artifacts yet, CUDA-graph
capture of the steady state is future work, and EP-overlap annotations are not yet composed
with GraphPP. The README's composability table marks "Graph-based Pipeline Parallelism" and
"Expert Parallelism + Pipeline Parallelism" as work in progress.

**On TPU.** No TPU recipe or test in the repo runs GraphPP. TPU scan-layer mode refuses
`pipeline_parallel_degree > 1`, because PP would split the layer range the scan covers. The TPU
graph-split mode borrows a few GraphPP partition helpers (the mutation materialization and
mutation-node detection). However, it splits the joint graph into separately compiled
forward and backward XLA executables, not into pipeline stages (see
[torchtpu-backend](../concepts/torchtpu-backend.md)).

## Connections
- Code concepts: [pipeline-parallel](../concepts/pipeline-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`experiments/graph_trainer/graph_pp/*`).
- Related doc-concepts: [graph-trainer](graph-trainer.md),
  [graph-trainer-precompile](graph-trainer-precompile.md),
  [ep-overlap-graph-chunking](ep-overlap-graph-chunking.md).

## Source
Extracted from `torchtitan/experiments/graph_trainer/README.md` (kept in place).
