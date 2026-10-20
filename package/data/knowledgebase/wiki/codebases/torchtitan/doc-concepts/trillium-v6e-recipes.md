---
title: "Trillium (TPU v6e) training recipes compared"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-8/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Trillium (TPU v6e) training recipes compared

## Definition
The `recipes/trillium/` tree holds tested launch recipes for TPU v6e: Conformer
(v6e-8), FLUX.1-dev 12B (v6e-8, v6e-256), Llama3-8B (v6e-8, v6e-256) and Qwen3-8B (v6e-8,
v6e-256). Every recipe is launched with `gcluster job submit --compute-type ct6e-standard-4t
--topology 2x4` (v6e-8) or `--topology 16x16` (v6e-256), runs one process per chip
(`--nproc_per_node=4` per host) and shards data-parallel across all chips. The Conformer README
gives two variants - `torch.compile` and eager `DEFER_AND_FUSE` - so the compile effect can be read
off directly. Throughput is reported per chip.

## In torchtitan (grounded)
### Configuration table
Values quoted from each README command. "SFSDP" = `--tpu_config.use_simple_fsdp` or
`--parallelism.use_simple_fsdp`. Batch is the token budget
`--training.num_tokens_per_microbatch_per_dp_rank`, shown as tokens (sequences x length).

| Recipe | Entry / `--config` | hosts x 4 | Variant | batch tokens per rank | dtype / precision | Parallelism | Other key flags |
|---|---|---|---|---|---|---|---|
| conformer v6e-8 | `conformer.train_minimal` / `conformer_test` | 2 | compile (`compile_mode=layer`) / eager | 262,144 (512 x 512) | `dtype=float32`, `mixed_precision_param=bfloat16` | shard=1, replicate=-1 (DDP), plus `--tpu_config.use_simple_fsdp` | `activation-checkpoint:full`, Splash, `--conformer.use-ctc-loss`, random data |
| flux1 v6e-8 | `train` / `flux_dev` | 2 | fp32 + MP bf16 / bf16 | 9,216 (2 x 4608) / 13,824 (3 x 4608), `--flux.img_size=1024` | `--training.dtype=float32` or `bfloat16` | `parallelism.use_simple_fsdp` | Splash, compile backend `tpu`, full AC `no-preserve-rng-state`, vmem 65536 |
| flux1 v6e-256 | same | 64 | fp32 + MP bf16 / bf16 | same | same | same | SparseCore offload bundle instead of vmem flag (below) |
| llama3-8b v6e-8 | `train` / `llama3_8b` | 2 | compile | 16,384 (2 x 8192), GBS 16 | fp32 master, bf16 compute | `parallelism.use_simple_fsdp` | Splash `sa_block_kv_compute=2048`, `eager_mode=DEFER_AND_FUSE`, `xla_tpu_scoped_vmem_limit_kib=131072 tpu_program_stack_shrink_after_unload=false` |
| llama3-8b v6e-256 | same | 64 | compile | 32,768 (4 x 8192), GBS 1024 | same | same | same, but only the vmem flag |
| qwen3-8b v6e-8 | `train` / `reg_trainer_qwen3_8b` | 2 | compile, `compile_mode=layer`, `use_graph_split` | 12,288 (6 x 2048), GBS 48 | `mixed_precision_reduce=float32` | SFSDP + `enable_tpu_topo_mesh` ("mapped to the 2x4 ICI physical layout") | FullAC, loss kernel, all `sa_block_*`=1024, `DEFER_AND_FUSE`, `xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536` |
| qwen3-8b v6e-256 | same | 64 | same | 12,288 (6 x 2048), GBS 1536 | same | same | same |

The flux READMEs explain the token budget: `--dataloader.img_size` and
`--training.local_batch_size` "no longer exist", so `--flux.img_size=1024`
([`FluxConfig.img_size`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L175 "torchtitan/experiments/tpu/tpu_job_config#FluxConfig.img_size")) sets the
processor image size and `max_context_length` ("64 x 64 image patches + 512 T5 tokens = 4608"), and
`num_tokens_per_microbatch_per_dp_rank` = samples x 4608.

The flux1 v6e-256 libtpu flags are `xla_tpu_enable_sparse_core_collective_aggregator=true
xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true`, which is exactly the
[`SPARSECORE_COLLECTIVE_OFFLOAD`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L200 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_COLLECTIVE_OFFLOAD.SPARSECORE_COLLECTIVE_OFFLOAD")
bundle. The Qwen3 `xla_tpu_use_enhanced_launch_barrier=false` flag is
[`DISABLE_ENHANCED_LAUNCH_BARRIER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L218 "torchtitan/experiments/tpu/tpu_flags#DISABLE_ENHANCED_LAUNCH_BARRIER.DISABLE_ENHANCED_LAUNCH_BARRIER").
The Llama3-8B v6e-8 README explains its extra flag: `tpu_program_stack_shrink_after_unload=false`
"keeps the TPU program stack at its high-water mark. HBM is nearly full in this recipe, so letting
the stack shrink makes later program loads trigger HBM defragmentation stalls (~0.6% slower step
time on libtpu 0.0.50)".

### Reported metrics (per chip unless noted)

| Recipe | Variant | Date | TPS / chip | Total TPS | MFU | Other |
|---|---|---|---|---|---|---|
| conformer v6e-8 | compile | 9/30/26 | 87,181 (approx.) | 697,448 | 6.37% | |
| conformer v6e-8 | eager | 9/30/26 | 88,664 (approx.) | 709,312 | 6.48% | |
| flux1 v6e-8 | fp32 + MP bf16, 2 samples | 9/29/26 | - | - | - | step 2.82s, step / PDB 1.41s |
| flux1 v6e-8 | bf16, 3 samples | 9/29/26 | - | - | - | step 3.59s, step / PDB 1.20s |
| flux1 v6e-256 | fp32 + MP bf16, 2 samples | 9/29/26 | - | - | - | step 2.88s, step / PDB 1.44s |
| flux1 v6e-256 | bf16, 3 samples | 9/29/26 | - | - | - | step 4.05s, step / PDB 1.35s |
| llama3-8b v6e-8 | compile | 9/29/26 | 5,383 | 43,068 | 33.96% | |
| llama3-8b v6e-256 | compile | 9/29/26 | 6,177 | 1,581,235 | 38.97% | |
| qwen3-8b v6e-8 | regular trainer | 10/4/2026, avg steps 2-15 | 5,753 | - | 29.59% | 271.65 TFLOPS, step 2.14s, memory 3.67GiB (11.45%) |
| qwen3-8b v6e-256 | regular trainer | 9/30/2026, avg steps 2-30 | 6,377 | - | 32.80% | 301.12 TFLOPS, step 1.93s, memory 3.67GiB (11.45%) |

The flux
"step / PDB" value equals step time divided by the samples per rank in each row; the READMEs do not
report MFU for flux on v6e.

### How the recipes map to code
- **Trainer lanes.** Llama3, Qwen3 and FLUX go through
  [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer"):
  Llama3 and Qwen3 on [`TPUTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L42 "torchtitan/experiments/tpu/train#TPUTrainer"),
  FLUX on [`FluxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/trainer.py#L38 "torchtitan/models/flux/trainer#FluxTrainer").
  Conformer uses a minimal training loop
  ([`conformer train_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/conformer/train_minimal.py#L47 "torchtitan/experiments/tpu/conformer/train_minimal#train_step")).
- **Qwen3-8B uses the regular trainer on v6e on purpose.** Both Qwen3 READMEs open with "The Graph
  Trainer currently still encounters Out Of Memory (OOM) errors on v6e", so they use
  [`reg_trainer_qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L289 "torchtitan/experiments/tpu/qwen3/config_registry#reg_trainer_qwen3_8b")
  (FullAC, `compile_mode="layer"`,
  [`use_graph_split`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L78 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.use_graph_split"))
  instead of the graph-trainer [`qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L375 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b")
  used on Ironwood. They also note "`SelectiveAC` causes OOM on v6e due to XLA activation limits".
- **Kernels.** Splash block sizes go to
  [`SplashAttentionKernelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L196 "torchtitan/experiments/tpu/tpu_job_config#SplashAttentionKernelConfig")
  (`sa_block_kv_compute` = 2048 for Llama3-8B, all blocks 1024 for Qwen3-8B); the
  fused loss block sizes go to
  [`LossKernelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L226 "torchtitan/experiments/tpu/tpu_job_config#LossKernelConfig").

### Recurring patterns and why
- **Compile vs eager `DEFER_AND_FUSE` is now a wash for Conformer**: the 9/30/26 measurement has
  eager slightly ahead (88,664 vs 87,181 TPS/chip, 6.48% vs 6.37% MFU), where the earlier 5/21/26
  numbers had compile well ahead (89,000 vs 59,000). The README does not explain the change.
- **Scoped VMEM limit is 65536** for FLUX and Qwen3; Llama3-8B still uses 131072.
- **Reported memory is low for Qwen3-8B on v6e** (3.67GiB, 11.45%, on both slices); the READMEs
  state it without explanation, so treat it as whatever the metric logs rather than a headroom
  estimate.

## Why it matters / when it applies
Pick the v6e recipe for your model family. For Conformer, compile and eager now deliver the same
throughput, so either lane is a valid baseline. Qwen3-style models on v6e should stay on the regular
trainer with FullAC until the graph-trainer OOM is fixed. When moving between v6e-8 and v6e-256,
the recipes change `--topology`, `--nnodes`, sometimes the token budget, and for FLUX and Llama3-8B
the libtpu flags (see [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md)). Per-chip v6e
numbers are not directly comparable with per-TensorCore Ironwood numbers (see
[ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md)).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [job-configuration](../concepts/job-configuration.md),
  [model-definitions](../concepts/model-definitions.md).
- Module catalogs: [module map](../catalog/index.md).
- Hardware: [TPU v6e (Trillium)](../../../hardware/tpu-v6e.md).
- Related doc-concepts: [tpu-training-recipes](tpu-training-recipes.md),
  [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md),
  [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md),
  [flux-model](flux-model.md), [qwen3-model](qwen3-model.md).

## Source
Extracted from (kept in place):
- `torchtitan/experiments/tpu/recipes/trillium/conformer/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-256/README.md`
