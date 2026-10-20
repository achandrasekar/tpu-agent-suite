---
title: "TPU training recipes: layout, anatomy and how to adapt one"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/recipes/ironwood/llama3-70b/tpu7x-2x2x4/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# TPU training recipes: layout, anatomy and how to adapt one

## Definition
`torchtitan/experiments/tpu/recipes` is a tree of tested launch recipes, one README per
(TPU generation, model, topology). The tree is `recipes/<generation>/<model>/<topology>/README.md`:
`ironwood/` holds the TPU7x recipes (flux1, llama3-70b, muse-glimmer-30b, qwen3-30b-a3b, qwen3-32b,
qwen3-8b) and `trillium/` holds the v6e recipes (conformer, flux1, llama3-8b, qwen3-8b). Example
models that live under `experiments/tpu/examples/` keep their recipes next to the code with the
same layout: `examples/mamba2/recipes/ironwood/mamba2-2.7b-256k/2x2x1/` and
`examples/mamba3/recipes/ironwood/mamba3-2.7b-256k/2x2x1/`. Each README is a complete
`gcluster job submit ... --command "torchrun ..."` invocation plus the metrics "you should observe"
with that exact command, usually dated. The recipe directories contain only the README: every tuned
knob lives either in a named config-registry function (`--config=...`) or on the command line of
the README itself - and the newer recipes (Qwen3-32B, Muse Glimmer) put everything, libtpu flags
included, in the config, so the command is just `--config` plus `--training.steps`.

## In torchtitan (grounded)
**Anatomy of a recipe command.** Every recipe has the same four layers:

1. *Cluster and container* - Cluster Toolkit `gcluster job submit` with `--compute-type`
   (`tpu7x` on Ironwood, `ct6e-standard-4t` on v6e), `--topology` (e.g. `4x4x4`, `16x16`),
   `--num-slices 1`, `--image "$WORKLOAD_IMAGE"`, `--gke-disable-parallel-containers` on Ironwood,
   and `WORKERS_0_HOSTNAME` pointed at `<workload>-main-job-0-0`. Setup (gcluster, cluster, image
   build, optional GCS bucket for traces) is delegated to the main `experiments/tpu/README.md`. The
   Mamba3 recipe is the one still written for `xpk workload create --tpu-type=...`.
2. *Process layout* - `torchrun --nnodes=<hosts> --nproc_per_node=<procs> --rdzv_backend=static
   --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 --node_rank=$TPU_WORKER_ID`. Trillium recipes use
   `--nproc_per_node=4` (one process per v6e chip, so v6e-8 is `--nnodes=2`, v6e-256 is
   `--nnodes=64`). Ironwood recipes use `--nproc_per_node=8` because, in the READMEs' words, "an
   ironwood chip exposes two TensorCores, and torchtitan runs one process per core" (so
   `tpu7x-4x4x4` is "128 devices on 64 chips, run as 16 hosts x 8 processes"). The mamba3 recipe
   is the one exception on the rendezvous: `--nnodes=1` with `--rdzv_backend=c10d` (mamba2 now uses
   the static rendezvous like the rest).
3. *Entry point and base config* - `-m torchtitan.experiments.tpu.train` with
   `--module=<model package>` and `--config=<registry function>`. The entry is
   [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer"),
   which picks the trainer class: `flux` always goes to
   [`FluxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/trainer.py#L38 "torchtitan/models/flux/trainer#FluxTrainer"),
   `use_graph_trainer` or `compile_mode == "aot_fx_trace"` selects
   [`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer"),
   and everything else runs on
   [`TPUTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L42 "torchtitan/experiments/tpu/train#TPUTrainer").
   Conformer uses its own minimal loop instead
   ([`conformer start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/conformer/train_minimal.py#L144 "torchtitan/experiments/tpu/conformer/train_minimal#start_trainer")).
   The named base configs are registry functions such as
   [`llama3_70b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L244 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b"),
   [`qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L375 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b"),
   [`reg_trainer_qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L289 "torchtitan/experiments/tpu/qwen3/config_registry#reg_trainer_qwen3_8b"),
   [`qwen3_32b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L499 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b"),
   [`muse_glimmer_30b_full_logits`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L205 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_full_logits")
   and [`mamba2_2_7b_256k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/config_registry.py#L308 "torchtitan/experiments/tpu/examples/mamba2/config_registry#mamba2_2_7b_256k"),
   all returning a
   [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig").
4. *Overrides* - CLI flags that land in config dataclasses. The recurring groups are:

| Flag group (as spelled in the recipes) | Config field it sets |
|---|---|
| `--training.num_tokens_per_microbatch_per_dp_rank`, `--training.max_context_length`, `--training.dtype`, `--training.mixed_precision_param`, `--training.mixed_precision_reduce` | [`TrainingConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L36 "torchtitan/config/configs#TrainingConfig") (e.g. [`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank"), [`max_context_length`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L51 "torchtitan/config/configs#TrainingConfig.max_context_length"), [`mixed_precision_reduce`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L104 "torchtitan/config/configs#TrainingConfig.mixed_precision_reduce")) |
| `--parallelism.use_simple_fsdp`, `--parallelism.enable_tpu_topo_mesh`, `--parallelism.data_parallel_shard_degree=-1`, `--parallelism.data_parallel_replicate_degree` | [`ParallelismConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L123 "torchtitan/config/configs#ParallelismConfig"): [`use_simple_fsdp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L124 "torchtitan/config/configs#ParallelismConfig.use_simple_fsdp"), [`enable_tpu_topo_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L127 "torchtitan/config/configs#ParallelismConfig.enable_tpu_topo_mesh"), [`data_parallel_shard_degree`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L160 "torchtitan/config/configs#ParallelismConfig.data_parallel_shard_degree") |
| `--compile.enable`, `--compile.backend=tpu` | [`CompileConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L338 "torchtitan/config/configs#CompileConfig") |
| `--tpu_config.use_simple_fsdp`, `--tpu_config.compile_mode`, `--tpu_config.eager_mode=DEFER_AND_FUSE`, `--tpu_config.use_graph_trainer`, `--tpu_config.compile_optimizer`, `--tpu_config.optimizer_in_graph`, `--tpu_config.use_graph_split`, `--tpu_config.enable_amp` | [`TPUConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L62 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig") (e.g. [`compile_mode`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L64 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.compile_mode"), [`eager_mode`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L145 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.eager_mode"), [`optimizer_in_graph`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L96 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.optimizer_in_graph")) |
| `--splash_attention_kernel.use_splash_attention_kernel`, `--splash_attention_kernel.sa_block_*` | [`SplashAttentionKernelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L196 "torchtitan/experiments/tpu/tpu_job_config#SplashAttentionKernelConfig") |
| `--loss_kernel.use_loss_kernel`, `--loss_kernel.loss_b_block_size`, `--loss_kernel.loss_h_block_size` | [`LossKernelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L226 "torchtitan/experiments/tpu/tpu_job_config#LossKernelConfig") |
| `--extra-libtpu-init-flags='...'` | [`BaseTPUTrainerConfig.extra_libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L251 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.extra_libtpu_init_flags") (merged over the config's set) |
| `--libtpu-init-flags='...'` | [`BaseTPUTrainerConfig.libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L245 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.libtpu_init_flags"), exported by [`apply_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L109 "torchtitan/experiments/tpu/tpu_flags#apply_init_flags") |

**Batch is a token budget.** `TrainingConfig` no longer has `local_batch_size` or `seq_len`: the
flux READMEs note that "`--dataloader.img_size` and `--training.local_batch_size` no longer exist".
Recipes pass `--training.num_tokens_per_microbatch_per_dp_rank` = sequences per rank x context
length, plus `--training.max_context_length` (e.g. Qwen3-8B: 6 x 2048 = 12288; Llama3-70B: 4 x 8192
= 32768; FLUX: samples x 4608, where `--flux.img_size=1024` fixes the 4608-token context). Global
batch defaults to that budget x data-parallel degree
([`num_tokens_per_train_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L43 "torchtitan/config/configs#TrainingConfig.num_tokens_per_train_step")
= -1); Qwen3-32B sets it explicitly in the registry.

The READMEs repeatedly describe `--libtpu-init-flags` as "a libtpu init flag the config exports
before PJRT initialization": the flags travel inside the training config rather than the
launcher's environment. The qwen3-30b-a3b README adds that "Exporting `LIBTPU_INIT_ARGS` from the
launcher still works and wins over the config for any flag it names". `--libtpu-init-flags`
replaces the config's set, while `--extra-libtpu-init-flags` (used by the Llama3-70B 128-device
recipe) adds to it. The named flag bundles the
recipes spell out inline are also defined in `tpu_flags.py`, e.g.
[`SPARSECORE_LATENCY_HIDING_SCHEDULER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L150 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_LATENCY_HIDING_SCHEDULER.SPARSECORE_LATENCY_HIDING_SCHEDULER")
and [`TPU7X_THROUGHPUT_P_STATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L211 "torchtitan/experiments/tpu/tpu_flags#TPU7X_THROUGHPUT_P_STATE.TPU7X_THROUGHPUT_P_STATE").

**Reading the metrics.** Units differ by generation. Ironwood READMEs report per device (per
TensorCore) and measure MFU "against the per-core bf16 peak of 1,153.5 TFLOP/s (half of the 2,307
TFLOP/s chip peak)". Trillium READMEs report TPS per chip. Some recipes average over a step window
("steps 10-30", "steps 2-15"), others read the value "around step 30" or call it "steady-state"; the
flux recipes report only step time and step time / PDB, and only the Ironwood flux recipe reports
MFU. With float8 enabled torchtitan logs `mfu: N/A`, and graph-trainer recipes with the optimizer in
the graph log `grad_norm` as 0.0. Most runs use synthetic or
test data (`c4_test`, `--flux.random_dataset`, `--dataloader.dataset=random`), so loss values are
not convergence evidence.

**How to adapt a recipe.** Start from the closest recipe on the same generation, keep its
`--config` and flag set, and change one axis at a time:
- *Topology*: change `--topology` and `--nnodes` together (hosts = chips / 4 on v6e, devices / 8 on
  TPU7x). With `data_parallel_shard_degree=-1` the FSDP degree follows the world size, so no mesh
  flag needs editing; global batch = per-rank token budget x world size.
- *Memory*: the knobs the recipes actually turn are the per-rank token budget, activation checkpointing
  (`activation-checkpoint:full`, FullAC in the registry), `xla_tpu_scoped_vmem_limit_kib`, and
  master-weight dtype (`--training.dtype`); on the GraphTrainer lane also Zero-3 resharding and the
  FSDP schedule (`prefetch` vs `interleave`), and fused cross entropy when logits dominate HBM.
- *Speed*: compile over eager, SimpleFSDP, topology-aware mesh, Splash attention block sizes and
  SparseCore collective-offload libtpu flags (see
  [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md)).

## Why it matters / when it applies
These are the only configurations in the repo with measured throughput attached to an exact
command line, so they are the right baseline before tuning a new model on TPU: reproduce the
recipe's number first, then diverge. Caveats the READMEs themselves expose:
- Metrics are pinned to a date and sometimes a torch_tpu commit; re-measure after upgrading.
- The gcluster rewrite fixed the earlier topology inconsistencies (the llama3-70b `tpu7x-2x2x4`
  recipe now passes `--topology 2x2x4`, the qwen3-8b `tpu7x-2x2x2` recipe `--topology 2x2x2`, and
  the v6e-256 flux title says v6e-256). Some numbers still predate the new launcher: the Qwen3-32B
  4x8x8 metrics were "measured with the original `xpk` launch ... not yet re-measured".
- Dated numbers move: the Conformer compile-vs-eager gap measured on 5/21/26 is gone in the
  9/30/26 run (see [trillium-v6e-recipes](trillium-v6e-recipes.md)).

> [!warning] Doc vs code
> The Mamba3 README still passes `--training.local_batch_size` and `--training.seq_len`, which
> `TrainingConfig` no longer defines at this pin; use the token-budget flags as in the Mamba2
> README (see [long-context-training-recipe](long-context-training-recipe.md)).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [job-configuration](../concepts/job-configuration.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [quantization](../concepts/quantization.md).
- Module catalogs: [module map](../catalog/index.md).
- Hardware: [TPU7x (Ironwood)](../../../hardware/tpu-7x.md), [TPU v6e (Trillium)](../../../hardware/tpu-v6e.md).
- Related doc-concepts: [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md),
  [trillium-v6e-recipes](trillium-v6e-recipes.md),
  [tpu-recipe-scaling-patterns](tpu-recipe-scaling-patterns.md),
  [long-context-training-recipe](long-context-training-recipe.md);
  launch and setup from other docs: [tpu-multihost-launch](tpu-multihost-launch.md),
  [tpu-vm-provisioning](tpu-vm-provisioning.md),
  [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md),
  [tpu-init-flags-precedence](tpu-init-flags-precedence.md) (how `--libtpu-init-flags` and
  `LIBTPU_INIT_ARGS` combine), [graph-trainer](graph-trainer.md),
  [float8-training](float8-training.md), [model-registry](model-registry.md).

## Source
Extracted from the recipe READMEs under `torchtitan/experiments/tpu/recipes` and
`torchtitan/experiments/tpu/examples/*/recipes` (kept in place):
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
- `torchtitan/experiments/tpu/recipes/trillium/conformer/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/flux1/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/llama3-8b/v6e-256/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-8/README.md`
- `torchtitan/experiments/tpu/recipes/trillium/qwen3-8b/v6e-256/README.md`
- `torchtitan/experiments/tpu/examples/mamba2/recipes/ironwood/mamba2-2.7b-256k/2x2x1/README.md`
- `torchtitan/experiments/tpu/examples/mamba3/recipes/ironwood/mamba3-2.7b-256k/2x2x1/README.md`
