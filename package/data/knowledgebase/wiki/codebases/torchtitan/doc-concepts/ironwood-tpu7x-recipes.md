---
title: "Ironwood (TPU7x) training recipes compared"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/recipes/ironwood/qwen3-8b/tpu7x-4x4x4/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Ironwood (TPU7x) training recipes compared

## Definition
The `recipes/ironwood/` tree holds tested TorchTitan launch recipes for TPU7x: FLUX.1-dev (12B),
Llama3-70B on three slice sizes (the largest also in float8), Muse Glimmer 30B, Qwen3-30B-A3B
(MoE), Qwen3-32B on two slice sizes and Qwen3-8B on two slice sizes. Two long-context SSM recipes,
Mamba2-2.7B and Mamba3-2.7B at 256K, live next to their model code under
`experiments/tpu/examples/<model>/recipes/ironwood/` (see
[long-context-training-recipe](long-context-training-recipe.md)). All are single-slice, pure
data-parallel (FSDP over every TensorCore) jobs; what distinguishes them is the trainer lane, the
memory knobs and the libtpu flag set. Ironwood runs one process per TensorCore (two per chip), so
the READMEs count "devices" as cores and report per-device throughput with MFU against "the
per-core bf16 peak of 1,153.5 TFLOP/s (half of the 2,307 TFLOP/s chip peak)".

## In torchtitan (grounded)
### Configuration table
Values are quoted from each README's command line and notes. "devices" = TensorCores. All recipes
except Mamba3 launch with `gcluster job submit --compute-type tpu7x --topology <t> --num-slices 1`.
Batch is a token budget per data-parallel rank
([`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank")),
shown here as tokens (sequences x length).

| Recipe dir | `--config` | `--topology` | hosts x procs (devices / chips) | trainer lane | batch tokens per rank | precision | parallelism |
|---|---|---|---|---|---|---|---|
| `flux1/tpu7x-4x4x4` | `flux_dev` | `4x4x4` | 16 x 8 (README title: "64 chip v7x") | FluxTrainer, `--compile.enable --compile.backend=tpu` | 13,824 (3 x 4608, fp32 variant) / 23,040 (5 x 4608, bf16 variant), `img_size=1024` | `--training.dtype=float32` + MixedPrecision bf16, or `--training.dtype=bfloat16` | `use_simple_fsdp`, `enable_tpu_topo_mesh` |
| `llama3-70b/tpu7x-2x2x4` | `llama3_70b` | `2x2x4` | 4 x 8 (32 / 16) | TPUTrainer, layerwise compile (`compile_mode="layer"`) | 32,768 (4 x 8192), GBS 128 | fp32 master, bf16 compute | SimpleFSDP (registry default), `enable_tpu_topo_mesh` |
| `llama3-70b/tpu7x-64` | `llama3_70b` | `2x4x4` | 8 x 8 (64 / 32) | TPUTrainer, layerwise compile | 32,768 (4 x 8192), GBS 256 | fp32 master, bf16 compute | same |
| `llama3-70b/tpu7x-128` | `llama3_70b` or `llama3_70b_float8` | `4x4x4` | 16 x 8 (128 / 64) | TPUTrainer, layerwise compile | 16,384 (2 x 8192), GBS 256 | bf16 compute, or torchao float8 rowwise linears | same |
| `muse-glimmer-30b/tpu7x-4x4x4` | `muse_glimmer_30b_full_logits` | `4x4x4` | 16 x 8 (128 / 64) | TpuGraphTrainer (set by registry config) | 1 x 16384 (registry), GBS 128 | fp32 weights and optimizer state, bf16 compute | SimpleFSDP Zero-3, `prefetch` FSDP schedule |
| `qwen3-30b-a3b/tpu7x-4x4x4` | `qwen3_moe_30b_fsdp_only_full_ac_benchmark` | `4x4x4` | 16 x 8 (128 cores) | set by registry config | not stated in README | "retained FP32 FSDP reductions" | FSDP-only |
| `qwen3-32b/tpu7x-4x4x4` | `qwen3_32b` | `4x4x4` | 16 x 8 (128 / 64) | TpuGraphTrainer (from `qwen3_8b_zero3`) | 8 x 4096, GBS 1024 | FP32 reductions | SimpleFSDP Zero-3, packed QKV, `prefetch` schedule |
| `qwen3-32b/tpu7x-4x8x8` | `qwen3_32b_4x8x8` | `4x8x8` | 64 x 8 (512 / 256) | same as 4x4x4 | 8 x 4096, GBS 4096 | FP32 reductions (required) | as 4x4x4, `interleave` schedule, all-reduce + slice for w1/w3 grads |
| `qwen3-8b/tpu7x-2x2x2` | `qwen3_8b` | `2x2x2` | 2 x 8 (16 / 8) | TpuGraphTrainer, `compile_mode=aot_fx_trace`, `compile_optimizer`, `optimizer_in_graph` | 12,288 (6 x 2048) | `mixed_precision_reduce=float32` | `use_simple_fsdp`, `enable_tpu_topo_mesh` |
| `qwen3-8b/tpu7x-4x4x4` | `qwen3_8b` | `4x4x4` | 16 x 8 (128 / 64) | TpuGraphTrainer, same as above | 12,288 (6 x 2048) | `mixed_precision_reduce=float32` | `use_simple_fsdp`, `enable_tpu_topo_mesh` |
| `examples/mamba2/.../2x2x1` | `mamba2_2_7b_256k` | `2x2x1` | 1 x 8 (8 cores) | TPUTrainer (no compile flags passed) | 2,097,152 (8 x 262144) or 262,144 (1 x 262144) | `mixed_precision_param=bfloat16` | FSDP over 8 cores ("1D Data Parallel mesh") |
| `examples/mamba3/.../2x2x1` | `mamba3_2_7b_256k` | `xpk --tpu-type=tpu7x-2x2x1` | 1 x 8 (8 cores) | TPUTrainer | README passes `local_batch_size` 8 or 1 (see warning) | `mixed_precision_param=bfloat16` | FSDP over 8 cores |

### Kernel and libtpu flags

| Recipe | Attention / loss | Activation checkpointing | libtpu flags |
|---|---|---|---|
| flux1 4x4x4 | Splash attention | `activation-checkpoint:full`, `--activation-checkpoint.no-preserve-rng-state` | `--libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=65536` |
| llama3-70b 32 / 64 devices | Splash, block size 2048; loss kernel enabled | Full (registry default) | `--libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=131072` |
| llama3-70b 128 devices | same | same | `--extra-libtpu-init-flags='xla_tpu_dvfs_p_state=7 xla_tpu_scoped_vmem_limit_kib=131072 xla_tpu_enable_offloading_copy_to_sparsecore=false xla_enable_async_all_gather=true xla_tpu_prefer_async_allgather_to_allreduce=true xla_tpu_enable_sparse_core_collective_offload_all_gather=true xla_tpu_copy_insertion_use_region_analysis=true xla_tpu_copy_insertion_use_region_analysis_limit=1000'` |
| muse-glimmer-30b | JAX Pallas Splash, 1024 tiles, local mask for the sliding-window layers; plain cross entropy over full logits | full graph rematerialization | none on the CLI: carried by the config (p-state 7, scoped VMEM, SparseCore latency-hiding scheduler, hierarchical reduce-scatter) |
| qwen3-30b-a3b | set by registry config | "full_ac" in config name | `--libtpu-init-flags='xla_tpu_enable_sparse_core_collective_aggregator=true xla_tpu_enable_latency_hiding_layer_scheduler=true xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true xla_tpu_dvfs_p_state=7'` |
| qwen3-32b (both) | tuned Splash tiles (from `qwen3_8b_zero3`) | full graph rematerialization | none on the CLI: carried by the config (same set as the Qwen3-8B 4x4x4 command) |
| qwen3-8b 2x2x2 | Splash, all nine `sa_block_*` = 1024 | graph remat policy from registry | `xla_tpu_dvfs_p_state=7 xla_tpu_scoped_vmem_limit_kib=65472 xla_tpu_enable_sparse_core_collective_aggregator=true xla_tpu_enable_latency_hiding_layer_scheduler=true xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false` |
| qwen3-8b 4x4x4 | Splash, all nine `sa_block_*` = 1024 | graph remat policy from registry | the 2x2x2 set plus `xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true` |
| mamba2 / mamba3 256K | n/a (SSM, Pallas scan) | registry sets `FullAC` (see [long-context-training-recipe](long-context-training-recipe.md)) | none |

### Reported metrics

| Recipe | Date | Per-device TPS | Total TPS | MFU | Other |
|---|---|---|---|---|---|
| flux1 4x4x4, fp32 + MP bf16, 3 samples | 9/28/26 | - | - | 20.47% | step 2.57s, step / PDB 0.86s |
| flux1 4x4x4, bf16, 5 samples | 9/30/26 | - | - | 21.64% | step 4.04s, step / PDB 0.81s |
| llama3-70b, 32 devices | 9/28/26 | 1,138 | 36,426 | 47.51% | |
| llama3-70b, 64 devices | 9/28/26 | 1,157 | 74,049 | 48.29% | |
| llama3-70b, 128 devices, bf16 | 10/5/26 | 1,235 | 158,080 | 51.5% | 594 TFLOP/s, step 13.27s, peak 7.37 GiB |
| llama3-70b, 128 devices, float8 | 10/5/26 | 1,636 | 209,408 | N/A | 787 TFLOP/s, step 10.02s, peak 8.67 GiB |
| muse-glimmer-30b 4x4x4 | 10/7/2026 | ~2,913 | - | ~43.8% | ~505.3 TFLOPS, step ~5.62s, peak HBM 46.18GiB |
| qwen3-30b-a3b 4x4x4 | 9/28/2026 | 7,490 (TPS) | - | 13.42% | 154.78 TFLOPS, 65.48 GiB (68.21%), step 1.700s |
| qwen3-32b 4x4x4, `c4_test` | 10/4/2026 | ~2,618 | - | ~46.5% | ~536.1 TFLOPS, step ~12.52s, peak HBM 35.66GiB (37.14%) |
| qwen3-32b 4x4x4, streamed C4 | 9/25/2026 | ~2,742 | - | ~48.7% | ~561.5 TFLOPS, step ~11.95s, 36.46GiB |
| qwen3-32b 4x8x8, streamed C4 (xpk launch) | 9/25/2026 | ~2,656 | - | ~47.2% | ~544.0 TFLOPS, step ~12.34s, peak HBM 83.07GiB |
| qwen3-8b 2x2x2 (16 devices) | 9/28/2026 | ~11,650 (TPS) | - | ~47.69% | ~550.12 TFLOPS, step ~1.05s, 80.48GiB (83.83%) |
| qwen3-8b 4x4x4 (128 devices) | 9/28/2026 | ~11,850 (TPS) | - | ~48.51% | ~559.58 TFLOPS, step ~1.037s, peak HBM 79.26GiB (82.56%) |
| mamba2 256k, 8 seq / core | estimate | - | ~79,900 - 104,800 (cluster) | 17.0% - 22.4% ("Expected") | ~160 - 210s step, 31.80 GiB / core |
| mamba3 256k, 8 seq / core | estimate | - | ~78,000 - 102,000 (cluster) | 16.6% - 21.7% ("Expected") | ~165 - 215s step, ~34.00 GiB / core |

The Qwen3 READMEs label TPS without "/device"; their TFLOPS / MFU ratio matches the per-core peak,
so read them as per-device; the 32B and Muse Glimmer READMEs say "per-device" explicitly and average
steps 10-30. The float8 MFU is "N/A" because, per the README, "with a mix of float8 and bf16 matmuls
there is no single peak to divide by": compare step time and TFLOP/s instead. The Qwen3-32B and Muse
Glimmer recipes log `grad_norm` as 0.0 "because the optimizer runs inside the compiled graph;
gradient clipping is still applied there". The SSM figures are ranges labelled "Expected MFU" and
"Est. Step Latency", not single measured runs.

### How the lanes map to code
- **TpuGraphTrainer lane (Qwen3-8B, Qwen3-32B, Muse Glimmer).** `--tpu_config.use_graph_trainer
  --tpu_config.compile_mode=aot_fx_trace` routes
  [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer")
  to [`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer");
  [`compile_optimizer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L83 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.compile_optimizer")
  and [`optimizer_in_graph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L96 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.optimizer_in_graph")
  put "the optimizer and backward pass ... into a single XLA graph". The base
  [`qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L375 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b")
  already sets these; the 8B recipe restates them on the CLI. The 32B and Muse Glimmer recipes
  pass only `--config` and `--training.steps`: [`qwen3_32b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L499 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b")
  derives from [`qwen3_8b_zero3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L462 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b_zero3")
  (parameters resharded after forward via
  [`fsdp_reshard_after_forward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L172 "torchtitan/config/configs#ParallelismConfig.fsdp_reshard_after_forward")
  - "Zero-2 would keep all ~61 GiB of gathered BF16 weights resident"), packs Q/K/V into one matmul
  with [`TpuPackedQKVConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qkv.py#L198 "torchtitan/experiments/tpu/qkv#TpuPackedQKVConverter") (checkpoint keys stay
  `wq`/`wk`/`wv`; optimizer state is layout-specific), and picks the FSDP schedule through
  [`graph_trainer_fsdp_schedule`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L132 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.graph_trainer_fsdp_schedule").
  [`muse_glimmer_30b_full_logits`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L205 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_full_logits")
  builds the same Zero-3 / prefetch / full-remat GraphTrainer setup for Muse Glimmer on top of the
  upstream [`muse_glimmer_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/muse_glimmer/config_registry.py#L205 "torchtitan/models/muse_glimmer/config_registry#muse_glimmer_30b") settings.
- **Scaling Qwen3-32B to 512 devices.** [`qwen3_32b_4x8x8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L526 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b_4x8x8")
  is `qwen3_32b` plus two changes. At world 512 each w1/w3 gradient shard is "25600 / 512 = 50
  rows. That is not a multiple of 8, so their reduce-scatters run on the TensorCore instead of
  SparseCore (~2.5 s per step)"; the recipe reduces those two gradients with an all-reduce plus a
  local slice, configured through
  [`fsdp_allreduce_slice_parameter_patterns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L91 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.fsdp_allreduce_slice_parameter_patterns"),
  which stays on SparseCore but "requires FP32 reductions, so there is no MaxText-parity (BF16
  reduction) variant at this scale yet". It also switches to the `interleave` schedule.
- **TPUTrainer lane (Llama3-70B).** [`llama3_70b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L244 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b")
  uses `compile.components=["model"]` with the default `tpu_config.compile_mode="layer"`, which
  "compiles each TransformerBlock separately (`fullgraph=True`) rather than the model as one graph";
  beyond dataset, tokenizer and step count the recipes only override the token budget,
  `enable_tpu_topo_mesh` and libtpu flags.
- **Float8 (Llama3-70B, 128 devices).** [`llama3_70b_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L510 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b_float8")
  swaps the linears for torchao `Float8Linear` with the `rowwise` recipe through
  [`TpuFloat8LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/float8_converter.py#L25 "torchtitan/experiments/tpu/float8_converter#TpuFloat8LinearConverter"); the output projection
  (`output` / `lm_head`) stays bf16, and unlike the CUDA converter it turns off `use_fast_accum`
  because "TPU matmuls accumulate in float32". The image needs torchao. Float8 cut step time from
  13.27s to 10.02s (~1.32x TPS) in the 10/5/26 measurement.
- **Topology-aware mesh.** `--parallelism.enable_tpu_topo_mesh`
  ([`ParallelismConfig.enable_tpu_topo_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L127 "torchtitan/config/configs#ParallelismConfig.enable_tpu_topo_mesh"))
  makes [`init_device_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/parallel_dims.py#L24 "torchtitan/distributed/parallel_dims#init_device_mesh") build the mesh from
  `torch.tpu.topology_aware_mesh` with `preserve_rank_order=True`, so ranks follow physical
  coordinates. The Qwen3-8B, Llama3-70B and flux recipes set it on the CLI; `qwen3_8b` (and so
  Qwen3-32B), `llama3_70b` and the Muse Glimmer base config already default it to `True`.
- **libtpu flag bundles.** The inline flag strings correspond to named bundles in `tpu_flags.py`:
  [`SPARSECORE_LATENCY_HIDING_SCHEDULER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L150 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_LATENCY_HIDING_SCHEDULER.SPARSECORE_LATENCY_HIDING_SCHEDULER"),
  [`DUAL_SPARSECORE_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L160 "torchtitan/experiments/tpu/tpu_flags#DUAL_SPARSECORE_REDUCE_SCATTER.DUAL_SPARSECORE_REDUCE_SCATTER"),
  [`HIERARCHICAL_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L169 "torchtitan/experiments/tpu/tpu_flags#HIERARCHICAL_REDUCE_SCATTER.HIERARCHICAL_REDUCE_SCATTER")
  and [`TPU7X_THROUGHPUT_P_STATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L211 "torchtitan/experiments/tpu/tpu_flags#TPU7X_THROUGHPUT_P_STATE.TPU7X_THROUGHPUT_P_STATE").
  The config-carried sets are the same bundles merged in the registry:
  [`_QWEN3_DENSE_4X4X4_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L491 "torchtitan/experiments/tpu/qwen3/config_registry#_QWEN3_DENSE_4X4X4_FLAGS") for Qwen3-32B and
  [`_MUSE_GLIMMER_4X4X4_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L42 "torchtitan/experiments/tpu/muse_glimmer/config_registry#_MUSE_GLIMMER_4X4X4_FLAGS") for Muse
  Glimmer (p-state 7, VMEM 65472, latency-hiding scheduler, hierarchical reduce-scatter). The
  Llama3-70B 128-device recipe adds its set with `--extra-libtpu-init-flags`
  ([`BaseTPUTrainerConfig.extra_libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L251 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.extra_libtpu_init_flags")),
  which merges over the config instead of replacing it the way `--libtpu-init-flags` does (see
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md)).

### Recurring patterns and why
- **FSDP-only everywhere.** No Ironwood recipe sets TP, PP, CP or EP; even the 30B MoE is
  "FSDP-only" on 128 cores, and Qwen3-32B runs "pure FSDP over all 512 devices". Scaling a recipe is
  a world-size change, not a mesh redesign - until shard shapes stop dividing evenly (the
  Qwen3-32B w1/w3 case above).
- **SparseCore collective offload is the big lever at scale.** The qwen3-30b-a3b README reports the
  aggregator + latency-hiding scheduler pair "improved throughput by about 5% at both GBS768/2K and
  GBS1024/2K", and hierarchical n-D reduce-scatter "another 1.09%", raising ring-equivalent
  reduce-scatter bus bandwidth "from 351.29 to 364.57 GB/s". The aggregator and scheduler must be
  "enabled or disabled together"; the dual-SparseCore reduce-scatter flag "may need to be disabled
  for schedules that overlap all-gather and reduce-scatter".
- **`xla_tpu_dvfs_p_state=7`** appears in all Qwen, Muse Glimmer and the 128-device Llama3-70B
  recipes; the README says it "trades higher device power for faster compute rather than changing
  the graph".
- **Scoped VMEM limit is model-specific**: 65536 KiB for flux, 65472 for Qwen3 and Muse Glimmer,
  131072 for Llama3-70B.
- **Zero-3 + prefetch for 30B-class dense models.** Qwen3-32B and Muse Glimmer both reshard
  parameters after forward and use the `prefetch` schedule on 128 devices; the 32B registry comment
  says prefetch "beats interleave by 0.3-0.6% here for ~4 GiB more peak HBM".
- **Full logits vs fused cross entropy (Muse Glimmer).** `muse_glimmer_30b_full_logits` uses plain
  cross entropy; [`muse_glimmer_30b_fused_ce`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L248 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_fused_ce")
  "is ~2% slower here but saves memory when raising tokens per microbatch".
- **Precision policy is explicit.** fp32 master weights with bf16 compute (Llama3, Muse Glimmer),
  and fp32 reductions (`mixed_precision_reduce=float32`) for Qwen3. The Qwen3 READMEs warn to "Keep
  canonical FP32-reduction results separate from MaxText-policy results" produced by
  [`qwen3_8b_maxtext_parity`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L474 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b_maxtext_parity")
  and [`qwen3_32b_maxtext_parity`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L548 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b_maxtext_parity")
  (BF16 FSDP reductions and BF16 RoPE; the 32B variant measured ~49.2% MFU and ~11.83s per step on
  streamed C4).

> [!warning] Doc vs code
> - The Mamba3 recipe still uses `xpk` and passes `--training.local_batch_size` /
>   `--training.seq_len`, which `TrainingConfig` no longer has at this pin (see
>   [long-context-training-recipe](long-context-training-recipe.md)).

## Why it matters / when it applies
Use this table to pick a starting point on TPU7x: dense decoder at 8B -> the Qwen3-8B graph-trainer
recipe (~48% MFU on 16 or 128 devices); dense at ~30B -> the Qwen3-32B or Muse Glimmer Zero-3
GraphTrainer recipes (44-49% MFU); dense at 70B -> the Llama3-70B layerwise-compile recipe (47.5-51.5%
MFU, highest at 128 devices) and its float8 variant for ~1.3x throughput; MoE -> the qwen3-30b-a3b
FSDP-only recipe; diffusion -> flux1; very long context -> mamba2 / mamba3. Remember that
per-device on TPU7x means per TensorCore, so compare with v6e per-chip numbers only after doubling
(and see [trillium-v6e-recipes](trillium-v6e-recipes.md)).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [expert-parallel](../concepts/expert-parallel.md) (not used: the MoE recipe is FSDP-only),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md),
  [quantization](../concepts/quantization.md) (float8 linears),
  [loss-and-validation](../concepts/loss-and-validation.md) (fused cross entropy),
  [model-definitions](../concepts/model-definitions.md).
- Module catalogs: [module map](../catalog/index.md).
- Hardware: [TPU7x (Ironwood)](../../../hardware/tpu-7x.md).
- Related doc-concepts: [tpu-training-recipes](tpu-training-recipes.md),
  [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md),
  [long-context-training-recipe](long-context-training-recipe.md),
  [trillium-v6e-recipes](trillium-v6e-recipes.md),
  [float8-training](float8-training.md), [qwen3-model](qwen3-model.md),
  [flux-model](flux-model.md), [graph-trainer](graph-trainer.md),
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md).

## Source
Extracted from (kept in place):
- `torchtitan/experiments/tpu/recipes/ironwood/flux1/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-2x2x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-64/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-128/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/muse-glimmer-30b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-30b-a3b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-32b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-32b/tpu7x-4x8x8/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-8b/tpu7x-2x2x2/README.md`
- `torchtitan/experiments/tpu/recipes/ironwood/qwen3-8b/tpu7x-4x4x4/README.md`
- `torchtitan/experiments/tpu/examples/mamba2/recipes/ironwood/mamba2-2.7b-256k/2x2x1/README.md`
- `torchtitan/experiments/tpu/examples/mamba3/recipes/ironwood/mamba3-2.7b-256k/2x2x1/README.md`
