---
title: "Scaling patterns across the TPU recipes (small slice -> full pod)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/recipes/ironwood/qwen3-32b/tpu7x-4x8x8/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Scaling patterns across the TPU recipes (small slice -> full pod)

## Definition
Several models have recipes at two or three slice sizes on the same TPU generation: Llama3-8B,
Qwen3-8B and FLUX on v6e-8 and v6e-256, and Llama3-70B (32, 64, 128 devices), Qwen3-8B (16, 128)
and Qwen3-32B (128, 512) on Ironwood. Comparing matched pairs shows what the maintainers actually
change when they scale out - and what they deliberately leave alone. The short answer: the
parallelism strategy never changes (FSDP over every device, degree = world size), local batch is
held roughly constant, and the tuning that changes is libtpu collective flags - plus, at 512
devices, a fix for shards that no longer divide evenly.

## In torchtitan (grounded)
**The mesh follows the world size.** Every scaled recipe is pure data-parallel sharding.
The models get FSDP from SimpleFSDP flags or registry defaults, and none sets tensor, pipeline, context or expert
degrees (Conformer is the one DDP recipe, `data_parallel_replicate_degree=-1`, single slice only). With
`-1`, [`ParallelismConfig.data_parallel_shard_degree`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L160 "torchtitan/config/configs#ParallelismConfig.data_parallel_shard_degree")
absorbs the whole world in
[`ParallelDims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L105 "torchtitan/distributed/parallel_dims#ParallelDims"),
so scaling out is `--topology` plus `--nnodes` and nothing else in the mesh. The Qwen3, Llama3-70B
and Ironwood flux recipes add `--parallelism.enable_tpu_topo_mesh` (or inherit it from the registry)
([`ParallelismConfig.enable_tpu_topo_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L127 "torchtitan/config/configs#ParallelismConfig.enable_tpu_topo_mesh")),
which makes [`init_device_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L24 "torchtitan/distributed/parallel_dims#init_device_mesh") order ranks by physical
topology through `torch.tpu.topology_aware_mesh`; the code comment notes that sorted sub-group
ranks would scramble the physical order and make "XLA fall back from the fast N-D SparseCore
collective strategy to a slow 1-D ring".
The sharding itself is SimpleFSDP in almost every recipe
([`ParallelismConfig.use_simple_fsdp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L124 "torchtitan/config/configs#ParallelismConfig.use_simple_fsdp")
/ [`TPUConfig.use_simple_fsdp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L63 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.use_simple_fsdp")).
Batch is a per-rank token budget
([`TrainingConfig.num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank")),
so holding it fixed keeps per-device work fixed while global batch grows with the world.

**Matched pairs.** Throughput per chip (v6e) or per device / TensorCore (TPU7x), as reported.

| Model (lane) | Small slice | Large slice | Seqs per rank small -> large | Global batch small -> large | Per-chip/device TPS small -> large | MFU small -> large | What else changed in the command |
|---|---|---|---|---|---|---|---|
| Llama3-8B (TPUTrainer, compile) | v6e-8 | v6e-256 | 2 -> 4 | 16 -> 1024 | 5,383 -> 6,177 | 33.96% -> 38.97% | batch; v6e-8 adds `tpu_program_stack_shrink_after_unload=false` (HBM nearly full) |
| Qwen3-8B (regular trainer) | v6e-8 | v6e-256 | 6 -> 6 | 48 -> 1536 | 5,753 -> 6,377 | 29.59% -> 32.80% | nothing but `--nnodes`, `--topology`; runs 10/4 vs 9/30/2026, averaged over steps 2-15 vs 2-30 |
| FLUX fp32 + MP bf16 | v6e-8 | v6e-256 | 2 -> 2 | - | step 2.82s -> 2.88s | - | vmem flag replaced by SparseCore offload bundle |
| FLUX bf16 | v6e-8 | v6e-256 | 3 -> 3 | - | step 3.59s -> 4.05s | - | same |
| Llama3-70B (TPUTrainer) | 32 devices | 64 devices | 4 -> 4 | 128 -> 256 | 1,138 -> 1,157 | 47.51% -> 48.29% | nothing but `--nnodes`, `--topology` |
| Llama3-70B (TPUTrainer, bf16) | 64 devices | 128 devices | 4 -> 2 | 256 -> 256 | 1,157 -> 1,235 | 48.29% -> 51.5% | halved per-device batch; `--extra-libtpu-init-flags` with p-state 7, async all-gather and SparseCore all-gather offload; runs 9/28 vs 10/5/26 |
| Qwen3-8B (TpuGraphTrainer) | 2x2x2 (16 devices) | 4x4x4 (128 devices) | 6 -> 6 | - | ~11,650 -> ~11,850 | ~47.69% -> ~48.51% | added hierarchical n-D reduce-scatter flags; both runs 9/28/2026 |
| Qwen3-32B (TpuGraphTrainer) | 4x4x4 (128 devices) | 4x8x8 (512 devices) | 8 -> 8 | 1024 -> 4096 | ~2,742 -> ~2,656 | ~48.7% -> ~47.2% | `interleave` instead of `prefetch` FSDP schedule, all-reduce + slice for w1/w3 gradients (both streamed C4, 9/25/2026) |

Global batch is shown only where the README states it. Per-chip / per-device MFU
stays within about five points between the small and large slice in every pair, and rises with scale
in all of them except Qwen3-32B at 512 devices (-1.5 points). That suggests these FSDP-only
configurations are not dominated by collectives at 256 v6e chips or 128 TPU7x cores
*(unverified - agent inference)*. Several pairs were measured on different dates, data or launch
tooling (the Qwen3-32B 4x8x8 number predates its `gcluster` command), so a small delta is not a
clean scaling measurement.

**Where FSDP-only stops being free: shard divisibility.** The Qwen3-32B 4x8x8 README is the first
recipe that needed a scale-specific code path. At world 512 each w1/w3 gradient shard is "25600 /
512 = 50 rows. That is not a multiple of 8, so their reduce-scatters run on the TensorCore instead of
SparseCore (~2.5 s per step)". [`qwen3_32b_4x8x8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L526 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b_4x8x8")
reduces those two gradients with an all-reduce plus a local slice
([`TPUConfig.fsdp_allreduce_slice_parameter_patterns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L91 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.fsdp_allreduce_slice_parameter_patterns")),
which stays on SparseCore; the registry comment quotes 14.33 -> 12.34 s per step "for +49 GiB peak
HBM", and the README notes it "requires FP32 reductions". The schedule choice also flips with scale:
`prefetch` wins at 128 devices, while "before the w1/w3 change, interleave beat prefetch at this
scale by 2.3% step time and 56 GiB peak HBM"
([`TPUConfig.graph_trainer_fsdp_schedule`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L132 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.graph_trainer_fsdp_schedule")).

**Collective flags are what grow with scale.** The libtpu flags travel in the config
([`BaseTPUTrainerConfig.libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L245 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.libtpu_init_flags"),
extended from the CLI by
[`BaseTPUTrainerConfig.extra_libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L251 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.extra_libtpu_init_flags"),
exported by [`apply_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L109 "torchtitan/experiments/tpu/tpu_flags#apply_init_flags"))
and match named bundles in `tpu_flags.py`:

| Bundle | Flags | Where the recipes use it |
|---|---|---|
| [`SPARSECORE_COLLECTIVE_OFFLOAD`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L200 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_COLLECTIVE_OFFLOAD.SPARSECORE_COLLECTIVE_OFFLOAD") | aggregator, offload queuing in LHS, 2d/3d all-gather offload, reduce-scatter offload | FLUX v6e-256 (v6e-8 uses the vmem flag instead) |
| [`SPARSECORE_LATENCY_HIDING_SCHEDULER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L150 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_LATENCY_HIDING_SCHEDULER.SPARSECORE_LATENCY_HIDING_SCHEDULER") | aggregator + latency-hiding layer scheduler | Qwen3-8B on both TPU7x slices, Qwen3-30B-A3B; config-carried for Qwen3-32B and Muse Glimmer |
| [`DUAL_SPARSECORE_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L160 "torchtitan/experiments/tpu/tpu_flags#DUAL_SPARSECORE_REDUCE_SCATTER.DUAL_SPARSECORE_REDUCE_SCATTER") | `xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false` | Qwen3-8B both slices, Qwen3-30B-A3B (and inside the hierarchical bundle) |
| [`HIERARCHICAL_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L169 "torchtitan/experiments/tpu/tpu_flags#HIERARCHICAL_REDUCE_SCATTER.HIERARCHICAL_REDUCE_SCATTER") | dual SC + n-D reduce-scatter offload + hierarchical reduce-scatter | every 128+ device Qwen3 / Muse Glimmer recipe, not the 16-device Qwen3-8B |
| [`TPU7X_THROUGHPUT_P_STATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L211 "torchtitan/experiments/tpu/tpu_flags#TPU7X_THROUGHPUT_P_STATE.TPU7X_THROUGHPUT_P_STATE") | `xla_tpu_dvfs_p_state=7` | all TPU7x Qwen and Muse Glimmer recipes, Llama3-70B at 128 devices |

The Llama3-70B 128-device set also turns on flags with no named bundle
(`xla_enable_async_all_gather`, `xla_tpu_prefer_async_allgather_to_allreduce`,
`xla_tpu_enable_sparse_core_collective_offload_all_gather`, copy-insertion region analysis) and
turns `xla_tpu_enable_offloading_copy_to_sparsecore` off.

**Compile vs eager.** Where a v6e README gives both lanes:

| Recipe | Compile TPS/chip, MFU | Eager `DEFER_AND_FUSE` TPS/chip, MFU |
|---|---|---|
| Conformer, v6e-8 (9/30/26) | 87,181, 6.37% | 88,664, 6.48% |

Compile here means `--compile.enable` on the minimal-loop path
([`CompileConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L338 "torchtitan/config/configs#CompileConfig"),
per-layer with [`TPUConfig.compile_mode`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L64 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.compile_mode")
`"layer"`); eager is [`TPUConfig.eager_mode`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L145 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.eager_mode")
`DEFER_AND_FUSE`. The two are now within ~2%, with eager slightly ahead; the previous measurement
(5/21/26) had compile at ~1.5x eager. The full-graph lane,
[`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer"),
is used only on TPU7x (Qwen3-8B, Qwen3-32B, Muse Glimmer); on v6e the Qwen3 READMEs say it still hits
OOM.

## Why it matters / when it applies
- **To scale a recipe out, change the slice, not the strategy.** Keep the FSDP-only config, change
  `--topology` and `--nnodes`, keep the per-rank token budget, then add SparseCore collective-offload
  flags when moving to a large slice. None of the recipes needed TP or PP to reach 256 v6e chips or
  512 TPU7x cores with models up to 70B.
- **Check shard divisibility at large worlds.** When a weight dimension divided by the world size
  stops being a multiple of 8, its reduce-scatter can fall off SparseCore; the Qwen3-32B 512-device
  recipe's all-reduce + slice is the in-tree fix.
- **Flag coupling.** The qwen3-30b-a3b README: the aggregator and the latency-hiding scheduler must
  be "enabled or disabled together", and the dual-SparseCore reduce-scatter choice "may need to be
  disabled for schedules that overlap all-gather and reduce-scatter". The `tpu_flags.py` comment on
  `HIERARCHICAL_REDUCE_SCATTER` adds that the lowering "is available in libtpu 0.0.47 and newer".
- **Compile vs eager on v6e is model-dependent**: for Conformer the lanes are now equivalent, so
  measure both rather than assuming compile wins *(derived from the table)*.

## Connections
- Code concepts: [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [tensor-parallel](../concepts/tensor-parallel.md) and
  [pipeline-parallel](../concepts/pipeline-parallel.md) (not used by any recipe),
  [profiling-and-tooling](../concepts/profiling-and-tooling.md).
- Module catalogs: [module map](../catalog/index.md).
- Hardware: [TPU v6e (Trillium)](../../../hardware/tpu-v6e.md), [TPU7x (Ironwood)](../../../hardware/tpu-7x.md).
- Related doc-concepts: [tpu-training-recipes](tpu-training-recipes.md),
  [trillium-v6e-recipes](trillium-v6e-recipes.md),
  [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md),
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md).

## Source
Extracted from (kept in place):
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/conformer/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-2x2x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-64/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-128/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-8b/tpu7x-2x2x2/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-8b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-30b-a3b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-32b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-32b/tpu7x-4x8x8/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/muse-glimmer-30b/tpu7x-4x4x4/README.md`
