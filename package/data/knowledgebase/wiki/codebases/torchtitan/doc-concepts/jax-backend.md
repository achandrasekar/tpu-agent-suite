---
title: "Pure-JAX training lane (experiments/jax, Flax NNX)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/jax/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Pure-JAX training lane (experiments/jax, Flax NNX)

## Definition
The JAX experiment is "LLM training on TPU using pure JAX with Flax NNX - no PyTorch dependency in
the training path". It is a sibling of the torchax lane: the same config system and dataloader, but
the model is re-written in Flax NNX, the optimizer is optax (through `nnx.Optimizer`), and the step
is compiled with `nnx.jit` + `nnx.value_and_grad`. It runs as a single process per host that sees
every local TPU chip (no `torchrun`), shards parameters over a 2-D JAX named mesh `(fsdp, tp)`,
stacks transformer blocks under `jax.lax.scan` to keep the XLA graph small, and uses a Pallas
splash-attention kernel on TPU. The README lists Llama3 (8B, 70B) as the only model.

## In torchtitan (grounded)
- **Entry point.** `python -m torchtitan.experiments.jax.train_minimal` runs
  [`main_train_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/train_minimal.py#L65 "torchtitan/experiments/jax/train_minimal#main_train_loop"). It first applies the shared TPU
  XLA/libtpu flags (`experiments/tpu/tpu_flags`), then calls
  [`_maybe_init_jax_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/train_minimal.py#L33 "torchtitan/experiments/jax/train_minimal#_maybe_init_jax_distributed"), which initializes
  `jax.distributed` only when `TPU_WORKER_HOSTNAMES` lists more than one host (coordinator
  `<host0>:8476`, rank from `TPU_WORKER_ID`). It builds the mesh with `fsdp = devices // tp` via
  `jax.make_mesh`, or `create_hybrid_device_mesh` when `tpu_num_slices > 1`, and hands it to
  [`JaxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/trainer.py#L91 "torchtitan/experiments/jax/trainer#JaxTrainer").
- **Config.** [`JaxJobConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/jax_job_config.py#L27 "torchtitan/experiments/jax/jax_job_config#JaxJobConfig") extends the shared
  [`BaseTPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L241 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig") (the same base the torch_tpu
  lane uses) with a [`JaxConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/jax_job_config.py#L9 "torchtitan/experiments/jax/jax_job_config#JaxConfig") block: `use_scan`
  (default True), `tpu_megacore`, `tpu_num_slices`, `model_layer_override`, and
  `adamw_bf16_state`, a custom Adam that keeps both moments in bf16
  ([`_scale_by_adam_bf16_state`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/trainer.py#L37 "torchtitan/experiments/jax/trainer#_scale_by_adam_bf16_state")) for memory-tight runs such
  as Llama3 8B on v6e-4. Per-model config registries derive from the torch_tpu registries,
  e.g. [`llama3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/llama3/config_registry.py#L27 "torchtitan/experiments/jax/llama3/config_registry#llama3_debugmodel") calls
  `JaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())` and swaps in a JAX `ModelSpec`
  whose `parallelize_fn` is a no-op.
- **Model and scan.** [`Transformer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/llama3/model.py#L443 "torchtitan/experiments/jax/llama3/model#Transformer") with
  [`ScannedTransformerBlocks`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/llama3/model.py#L341 "torchtitan/experiments/jax/llama3/model#ScannedTransformerBlocks") stacks every layer's
  parameters on a leading axis and names them after their NNX path (`attention_wq_kernel`, ...).
  [`setup_model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/trainer.py#L165 "torchtitan/experiments/jax/trainer#JaxTrainer.setup_model") builds the model.
- **Sharding.** [`apply_sharding_to_state`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/distributed.py#L103 "torchtitan/experiments/jax/distributed#apply_sharding_to_state") matches NNX
  state paths against the maps [`sharding_map_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/llama3/sharding.py#L53 "torchtitan/experiments/jax/llama3/sharding#sharding_map_scan")
  and [`sharding_map_original`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/llama3/sharding.py#L20 "torchtitan/experiments/jax/llama3/sharding#sharding_map_original");
  [`shard_input`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/distributed.py#L165 "torchtitan/experiments/jax/distributed#shard_input") places the batch on the `fsdp` axis.
- **Batch sizing.** The README's quick start now uses the token-denominated
  `--training.max_context_length=2048 --training.num_tokens_per_train_step=8192` instead of
  `seq_len` / `global_batch_size`. [`setup_dataloader`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/trainer.py#L111 "torchtitan/experiments/jax/trainer#JaxTrainer.setup_dataloader")
  still reasons in samples: the global batch is `num_tokens_per_train_step // max_context_length`
  (8192 / 2048 = 4 sequences, the README's "batch 4"), falling back to
  `num_tokens_per_microbatch_per_dp_rank // max_context_length` when the step budget is unset.
  A dataset name of `fake` (or none) switches to the synthetic loader.
- **Attention, AC, metrics.** [`make_splash_attention_fn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/splash_attn.py#L101 "torchtitan/experiments/jax/splash_attn#make_splash_attention_fn")
  wraps the Pallas kernel when the platform is `tpu` and
  `splash_attention_kernel.use_splash_attention_kernel` is set (default off); on any build error
  it logs and falls back to SDPA. At this pin the kernel is imported from tokamax's splash
  attention (`tokamax==0.0.14` in `experiments/tpu/requirements.txt`), no longer from
  `jax.experimental.pallas`, so the two lanes no longer share a kernel source: torchax keeps its
  own `splash_attn.py` on the JAX Pallas kernel. The activation-checkpoint config
  class maps to a JAX remat policy: `FullAC` -> `nothing_saveable`, `SelectiveAC` ->
  `dots_saveable`. [`JaxMetricsProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/metrics.py#L52 "torchtitan/experiments/jax/metrics#JaxMetricsProcessor") logs TPS, TFLOPs
  and MFU.

## Why it matters / when it applies
This lane shows what the same model reaches on TPU with no PyTorch in the loop. That makes it a
reference point for the torch_tpu path in [torchtpu-backend](../concepts/torchtpu-backend.md). The
README reports ~20% MFU for Llama3 8B (scan, full AC, seq 2048, batch 4) on v6e-4. The ideas that
carry over are scan-over-layers to cut compile time (the torch_tpu lane has `use_scan_layers`),
FSDP over a named mesh, and splash attention. Use it when you want a JAX-native baseline or HLO
to compare against. It has no state-dict adapter (`state_dict_adapter=None`), and `trainer.py` has no checkpoint
save or load path at all. It does not help with converting or importing weights.

> [!warning] Doc vs code
> - The quick start passes `--model.name=llama3 --model.flavor=8B`. The code selects models with
>   `--module ... --config ...` (see the `train_minimal` docstring and the per-model
>   `config_registry`); `JaxTrainer.setup_model` reads `job_config.model_spec.name/flavor`.
> - The quick start's `--training.dataset_path` and `--model.hf_assets_path`, and the
>   `--training.dataset=fake` tip, are not config fields: the asset path is the top-level
>   `--hf_assets_path` and the trainer reads the dataset name from `dataloader.dataset`.
> - "The embedding table is replicated" holds only for the non-scan map. The default scan map
>   shards `tok_embeddings/embedding` on the hidden dim `(None, 'fsdp')`. It also shards
>   `output/kernel` on the vocab dim so that the fp32 logits of the cross-entropy are split
>   (~4 GiB -> ~1 GiB per chip at b=16, s=2048, per the comment in `llama3/sharding.py`).
> - `use_scan` defaults to True in `JaxConfig`, but `llama3_debugmodel` turns it off.
> - `--activation_checkpoint.mode=full` is not what the trainer reads. It dispatches on the AC
>   config class (`FullAC` / `SelectiveAC`).
> - The file layout omits `jax_profiling.py` and `utils.py`.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (the torch_tpu lane whose
  base config and registries this lane derives from),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [job-configuration](../concepts/job-configuration.md),
  [model-definitions](../concepts/model-definitions.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/jax/trainer`, `experiments/jax/distributed`, `experiments/jax/llama3/*`).
- Related doc-concepts: [torchax-backend](torchax-backend.md) (the PyTorch-on-JAX sibling),
  [tpu-multihost-launch](tpu-multihost-launch.md), [tpu-training-recipes](tpu-training-recipes.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md),
  [experiments-policy](experiments-policy.md).

## Source
Extracted from `torchtitan/experiments/jax/README.md` (kept in place).
