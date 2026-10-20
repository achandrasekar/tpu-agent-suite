---
title: "torchax lane: PyTorch models executed as JAX on TPU (experiments/torchax)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/torchax/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# torchax lane: PyTorch models executed as JAX on TPU (experiments/torchax)

## Definition
The torchax experiment is "LLM training on TPU using torchax - a PyTorch-on-JAX backend that
lets you run PyTorch models on JAX devices with full XLA compilation". It keeps TorchTitan's
config system, dataloader and PyTorch model definitions, and swaps the execution backend. The
`nn.Module` is wrapped so that its ATen ops are carried out by JAX. The step is traced and jitted by
JAX, the optimizer is optax, and parameters are sharded over a JAX named mesh `(fsdp, tp)`. Like
the pure-JAX lane, it runs as one process per host, scans over stacked layers and uses Pallas
splash attention. It adds optional CPU offload of decoder-layer inputs and a Pallas grouped
matmul for MoE. The README's model list is Llama3, Qwen3, DeepSeek-V3 and Conformer.

## In torchtitan (grounded)
- **Entry and mesh.** `python -m torchtitan.experiments.torchax.train_minimal` runs
  [`main_train_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/train_minimal.py#L58 "torchtitan/experiments/torchax/train_minimal#main_train_loop"). This function applies the
  shared TPU flags, calls
  [`_maybe_init_jax_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/train_minimal.py#L36 "torchtitan/experiments/torchax/train_minimal#_maybe_init_jax_distributed") for multi-host
  runs, builds the `(fsdp, tp)` mesh, and constructs
  [`TorchaxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/trainer.py#L74 "torchtitan/experiments/torchax/trainer#TorchaxTrainer"). For MFU,
  [`TorchaxParallelDims`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/distributed.py#L269 "torchtitan/experiments/torchax/distributed#TorchaxParallelDims") forces
  `non_data_parallel_size=1`, because torchax uses one global (single-controller) dataloader.
- **Config.** [`TorchaxJobConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/torchax_job_config.py#L32 "torchtitan/experiments/torchax/torchax_job_config#TorchaxJobConfig") extends
  the same [`BaseTPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L241 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig") as the torch_tpu
  lane. [`TorchaxConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/torchax_job_config.py#L10 "torchtitan/experiments/torchax/torchax_job_config#TorchaxConfig") *subclasses*
  [`JaxConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/jax/jax_job_config.py#L9 "torchtitan/experiments/jax/jax_job_config#JaxConfig"), which adds `offload_keys` (default
  `decoder_layer_input`), `use_ddp_sharding` (replicate the LoRA base model to skip the per-layer
  all-gather; LoRA-only, see below) and `split_compile`. With `split_compile`, fwd+bwd and the optimizer update become
  two XLA programs, "mirroring torchtpu's default"
  ([`make_split_train_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/jit_utils.py#L14 "torchtitan/experiments/torchax/jit_utils#make_split_train_step")).
- **Model reuse.** [`setup_model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/trainer.py#L171 "torchtitan/experiments/torchax/trainer#TorchaxTrainer.setup_model") looks
  the model up in
  [`_TORCHAX_MODEL_MODULES`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/trainer.py#L29 "torchtitan/experiments/torchax/trainer#_TORCHAX_MODEL_MODULES") (exactly llama3,
  qwen3, deepseek_v3 and conformer). Each per-model package re-exports the *upstream*
  PyTorch model (for example, llama3 exports `torchtitan.models.llama3.Llama3Model`) plus a
  sharding map. [`train`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/trainer.py#L414 "torchtitan/experiments/torchax/trainer#TorchaxTrainer.train") wraps it in
  `torchax.interop.JittableModule` and builds the step with `torchax.train.make_train_step`.
  [`setup_optimizer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/trainer.py#L327 "torchtitan/experiments/torchax/trainer#TorchaxTrainer.setup_optimizer") translates the
  TorchTitan optimizer config into optax.
- **Sharding and scan.** [`create_sharded_weights`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/distributed.py#L67 "torchtitan/experiments/torchax/distributed#create_sharded_weights")
  applies the map. [`ModelWithScan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/distributed.py#L191 "torchtitan/experiments/torchax/distributed#ModelWithScan") and
  its Conformer counterparts
  [`ConformerModelWithScan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/distributed.py#L250 "torchtitan/experiments/torchax/distributed#ConformerModelWithScan") /
  [`ConformerScannedModule`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/distributed.py#L108 "torchtitan/experiments/torchax/distributed#ConformerScannedModule")
  provide the scanned layer stacks.
- **LoRA.** `setup_model` still picks `sharding_map_scan_lora` (or `sharding_map_scan_lora_ddp`
  with `use_ddp_sharding`) for a flavor whose name contains `lora`, but no in-tree model package
  exports such a map at this pin, so the branch, and `use_ddp_sharding` with it, is dormant.
- **Kernels.** [`declare_splash_attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/splash_attn.py#L89 "torchtitan/experiments/torchax/splash_attn#declare_splash_attention")
  overrides attention with [`tpu_splash_attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/splash_attn.py#L15 "torchtitan/experiments/torchax/splash_attn#tpu_splash_attention").
  [`declare_gmm_kernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/torchax/gmm.py#L125 "torchtitan/experiments/torchax/gmm#declare_gmm_kernel") registers a Pallas grouped matmul
  for MoE experts, with FSDP and TP column/row runners. torchax's splash kernel comes from
  `jax.experimental.pallas.ops.tpu.splash_attention`; the pure-JAX lane now builds its splash
  callable from tokamax instead, so the two lanes no longer share a kernel source.
- **Memory.** With `training.enable_cpu_offload` and `offload_keys` set, the remat policy
  becomes `save_and_offload_only_these_names(... offload_dst='pinned_host')`. Otherwise
  `FullAC` -> `nothing_saveable` and `SelectiveAC` -> `dots_saveable`.

## Why it matters / when it applies
There are three ways to reach TPU from this repo:
1. The torch_tpu lane ([torchtpu-backend](../concepts/torchtpu-backend.md)):
   `torch.device("tpu")`, `torch.compile(backend="tpu")`, DTensor/FSDP2 parallelism.
2. torchax: PyTorch model code, JAX runtime, JAX sharding, optax.
3. Pure JAX ([jax-backend](jax-backend.md)): Flax model code, no PyTorch.

torchax keeps the model definitions (and thus the onboarding surface) closest to upstream
TorchTitan. It does not use TorchTitan's parallelizers (`parallelize_fn=None` in its
`ModelSpec`), or its DTensor sharding, `state_dict_adapter` is `None`, and `trainer.py` has no checkpoint
save or load path. So
the TorchTitan checkpointing, HF import and FSDP2/TP code paths do not apply. The per-model
sharding maps are what count. It is the natural A/B partner for the torch_tpu lane on the same
PyTorch model: `split_compile` exists to compare HLO module sizes between the two frameworks.

> [!warning] Doc vs code
> - The README says `--torchax_config.use_scan` defaults to False. `TorchaxConfig` inherits
>   `use_scan=True` from `JaxConfig`. The per-model registries set it explicitly (False for
>   `llama3_debugmodel`, qwen3 and deepseek_v3; True for Llama3 1B/8B/70B/405B and Conformer).
> - The file layout lists `moe.py` (the file is `moe_utils.py`) and `metrics.py` (there is none;
>   the trainer imports `JaxMetricsProcessor` from `experiments/jax/metrics.py`). It describes
>   `distributed.py` as holding `apply_sharding_to_state, shard_input`; the torchax module has
>   `create_sharded_weights` and the scan wrappers instead. It omits `conformer/` (although
>   Conformer is in the model list) and `xpk/`.
> - The quick start uses `--model.name/--model.flavor`, `--model.hf_assets_path` and
>   `--training.dataset_path` (and `--training.dataset=fake`). None of these are config fields:
>   models are selected through `--module/--config` registries, the asset path is the top-level
>   `--hf_assets_path`, and the trainer reads the dataset from `dataloader.dataset`.
> - `trainer.py` imports `torch_xla2` (the earlier name of torchax) at module import time, so it
>   must be installed alongside `torchax`.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md),
  [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [lora](../concepts/lora.md),
  [job-configuration](../concepts/job-configuration.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/torchax/trainer`, `experiments/torchax/distributed`, `experiments/torchax/gmm`).
- Related doc-concepts: [jax-backend](jax-backend.md),
  [tpu-multihost-launch](tpu-multihost-launch.md), [tpu-training-recipes](tpu-training-recipes.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md),
  [experiments-policy](experiments-policy.md).

## Source
Extracted from `torchtitan/experiments/torchax/README.md` (kept in place).
