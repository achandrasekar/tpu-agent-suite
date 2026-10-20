---
title: "Expert-parallel overlap via graph chunking (GraphTrainer)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/graph_trainer/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Expert-parallel overlap via graph chunking (GraphTrainer)

## Definition
The README calls EP overlap "an experimental graph-trainer optimization for MoE models with real
expert-parallel collectives". The traced train step splits each selected region (every
transformer block `layers.*`, or every MoE block `layers.*.moe`) into two chunks along the batch
or sequence dimension. It then schedules them so that one chunk's expert-parallel all-to-all
token exchange overlaps the other chunk's compute. The README describes the tracer contract
behind it. A trace-input preparer marks the token-grid dimensions with Dynamo symbolic-shape
metadata before `minimal_fx_tracer` fakeifies the inputs. The chunk pass then treats those
symbols as "the source of truth" for which live-ins and live-outs must be split.

## In torchtitan (grounded)
- **Config.** [`EpOverlapConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/configs.py#L32 "torchtitan/experiments/graph_trainer/configs#EpOverlapConfig")
  (`enabled`, `chunk_dim`, `strategy`, `module_fqn`, `disable_early_grad_accumulation`) is
  validated by [`validate_ep_overlap_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/configs.py#L209 "torchtitan/experiments/graph_trainer/configs#validate_ep_overlap_config").
  `chunk_dim` must be `batch` or `seq`, and `strategy` must be `eager` or `graph`. `module_fqn`
  must be exactly `layers.*` or `layers.*.moe`, and `seq` is allowed only with `layers.*.moe`.
  These are model-relative FQN patterns, unlike the full-path globs used by
  [kernel-overrides](kernel-overrides.md).
- **Trace-input contract.** [`prepare_ep_overlap_trace_inputs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/ep_chunk_pass.py#L627 "torchtitan/experiments/graph_trainer/ep_chunk_pass#prepare_ep_overlap_trace_inputs")
  and [`mark_chunk_dynamic_dims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/ep_chunk_pass.py#L590 "torchtitan/experiments/graph_trainer/ep_chunk_pass#mark_chunk_dynamic_dims")
  mark the dimensions to chunk. The trainer hooks this preparer into
  [`minimal_fx_tracer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/make_fx_tracer.py#L405 "torchtitan/experiments/graph_trainer/make_fx_tracer#minimal_fx_tracer")
  through its `prepare_inputs` / `prepare_call_inputs` arguments.
- **Passes.** [`ep_overlap_chunk_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/ep_chunk_pass.py#L2426 "torchtitan/experiments/graph_trainer/ep_chunk_pass#ep_overlap_chunk_pass")
  duplicates each selected region per chunk.
  [`ep_overlap_validate_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/ep_overlap_pass.py#L956 "torchtitan/experiments/graph_trainer/ep_overlap_pass#ep_overlap_validate_pass")
  and [`ep_overlap_schedule_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/ep_overlap_pass.py#L981 "torchtitan/experiments/graph_trainer/ep_overlap_pass#ep_overlap_schedule_pass")
  then reorder the token-exchange launches and waits around the other chunk's compute. All of
  these are added by [`compile_time_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L150 "torchtitan/experiments/graph_trainer/passes#compile_time_passes").

## Why it matters / when it applies
In MoE training with EP, the dispatch and combine all-to-alls sit on the critical path. Chunking
gives the scheduler independent work to hide them behind. The README gives its conditions:
turn it on only with `--compile.mode aot_fx_trace` and `expert_parallel_degree > 1`. Sequence
chunking cannot include attention, because attention needs the full K/V. The code adds one the
README omits: `compile_time_passes` raises for the `graph` strategy when
`tensor_parallel_degree > 1`, because splitting TP-local tensors after DTensor lowering is not
proven equivalent to eager chunking (use `strategy=eager` or TP 1). Combining it with other
overlap or sharding modes should be checked with a loss comparison first. The config docstring
adds that `enable_fsdp_dense_region_overlap` composes with EP overlap only when the chunk root
is `layers.*.moe`. The README says GraphPP does not yet compose with EP-overlap annotations.

**On TPU.** The TPU pass pipeline
([`construct_tpu_transform_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass.py#L519 "torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass#construct_tpu_transform_passes"))
does not include the EP chunk or EP schedule passes, so `--compile.ep_overlap.*` does nothing
under [`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer").
On TPU, collective scheduling inside the one lowered program is left to XLA and to the TPU
remat/FSDP schedule passes. TPU scan-layer mode also refuses `expert_parallel_degree > 1`. See
[expert-parallel](../concepts/expert-parallel.md) and [torchtpu-backend](../concepts/torchtpu-backend.md).

## Connections
- Code concepts: [expert-parallel](../concepts/expert-parallel.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`experiments/graph_trainer/ep_chunk_pass`,
  `experiments/graph_trainer/ep_overlap_pass`).
- Related doc-concepts: [graph-trainer](graph-trainer.md),
  [graph-pp-pipeline-parallel](graph-pp-pipeline-parallel.md).

## Source
Extracted from `torchtitan/experiments/graph_trainer/README.md` (kept in place).
