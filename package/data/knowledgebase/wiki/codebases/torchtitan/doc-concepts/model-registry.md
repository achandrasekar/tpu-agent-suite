---
title: "Model registry: how a model is added to TorchTitan (folder layout, ModelSpec, config_registry)"
type: doc-concept
provenance: doc
source: torchtitan/models/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Model registry: adding a model

## Definition
`torchtitan/models/README.md` is the recipe for adding a model. New models should normally start
under `torchtitan/experiments` ([experiments-policy](experiments-policy.md)). A model folder holds
single-device model code (a `Model` class with a nested `Config` that owns all hyperparameters,
`get_nparams_and_flops()` and `update_from_config()`), a `state_dict_adapter.py`, a `sharding.py`
that fills declarative `sharding_config` on each module config, a `parallelize.py` that applies
training techniques in a fixed order, an `__init__.py` with the flavor table and a
`model_registry(flavor)` returning a `ModelSpec`, and a `config_registry.py` with one function per
training configuration, selected at run time with `--module <model_name> --config <function_name>`.

## In torchtitan (grounded)
- **Protocol.** The model config subclasses [`BaseModel.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model.py#L109 "torchtitan/protocols/model#BaseModel.Config"),
  with hooks [`update_from_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model.py#L118 "torchtitan/protocols/model#BaseModel.Config.update_from_config") and
  [`get_nparams_and_flops`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model.py#L127 "torchtitan/protocols/model#BaseModel.Config.get_nparams_and_flops"); modules
  subclass [`Module`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L43 "torchtitan/protocols/module#Module"), whose
  [`init_states`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L109 "torchtitan/protocols/module#Module.init_states") applies every sub-config's
  `param_init` and whose [`_init_self_buffers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L242 "torchtitan/protocols/module#Module._init_self_buffers")
  is the override point for device-aware buffers. Language models subclass
  [`Decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L67 "torchtitan/models/common/decoder#Decoder").
- **The bundle.** `model_registry(flavor, attn_backend=..., converters=...)` (for example
  [`llama3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L387 "torchtitan/models/llama3/__init__#model_registry")) picks a flavor
  function from a table such as [`llama3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L377 "torchtitan/models/llama3/__init__#llama3_configs"), runs
  [`validate_converter_order`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/utils.py#L27 "torchtitan/models/utils#validate_converter_order") and each converter over the config, and
  returns a [`ModelSpec`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/model_spec.py#L33 "torchtitan/protocols/model_spec#ModelSpec").
- **Sharding.** Each model has a `set_<model>_sharding_config` helper
  ([`set_llama3_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/sharding.py#L25 "torchtitan/models/llama3/sharding#set_llama3_sharding_config"),
  [`set_qwen3_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/sharding.py#L38 "torchtitan/models/qwen3/sharding#set_qwen3_sharding_config"),
  [`set_deepseek_v3_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/sharding.py#L44 "torchtitan/models/deepseek_v3/sharding#set_deepseek_v3_sharding_config"), ...) built on the shared
  [`set_decoder_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder_sharding.py#L365 "torchtitan/models/common/decoder_sharding#set_decoder_sharding_config"),
  [`set_gqa_attention_sharding`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder_sharding.py#L228 "torchtitan/models/common/decoder_sharding#set_gqa_attention_sharding"),
  [`set_dense_ffn_sharding`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder_sharding.py#L329 "torchtitan/models/common/decoder_sharding#set_dense_ffn_sharding") and, for MoE,
  [`set_moe_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe_sharding.py#L305 "torchtitan/models/common/moe_sharding#set_moe_sharding_config"). Placements are applied by
  [`parallelize`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/module.py#L279 "torchtitan/protocols/module#Module.parallelize").
- **Parallelize order.** [`parallelize_llama`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/parallelize.py#L23 "torchtitan/models/llama3/parallelize#parallelize_llama") is the reference:
  `model.parallelize(parallel_dims)` (TP/SP/`local_map`; always under
  `--parallelism.spmd_backend spmd_types`, otherwise only when TP is on), then activation
  checkpointing, then per-block `torch.compile`, then FSDP/HSDP via
  [`apply_fsdp_to_decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L189 "torchtitan/distributed/fsdp#apply_fsdp_to_decoder").
  For MoE models the same call takes `ep_degree` and `edp_mesh`, so routed experts are sharded over
  the expert-FSDP mesh; multimodal models first shard the vision encoder as one unit with
  [`apply_fsdp_to_vision_encoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L150 "torchtitan/distributed/fsdp#apply_fsdp_to_vision_encoder").
- **Mesh axes.** The README asks sharding configs to declare axes outer-to-inner: `(dp, cp, tp)` for
  dense modules and `(dp_replicate, efsdp, ep)` for routed experts.
  [`spmd_backend`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L199 "torchtitan/config/configs#ParallelismConfig.spmd_backend") is now either
  `partial_dtensor` (default; only `tp` and `ep` are kept, `dp_shard` and `cp` fold into `fsdp`) or
  `spmd_types`; the `full_dtensor` backend is gone.
- **CP.** Language-model CP is input-side: `Decoder.preprocess_inputs` calls
  [`_cp_shard_inputs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L351 "torchtitan/models/common/decoder#Decoder._cp_shard_inputs"), which hands the
  batch to the attention class's [`cp_shard`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/cp_attention.py#L44 "torchtitan/models/common/cp_attention#CPInnerAttention.cp_shard") and from there to
  [`prepare_context_parallel_input`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/context_parallel/api.py#L39 "torchtitan/distributed/context_parallel/api#prepare_context_parallel_input") (multimodal models such as Kimi K2.7
  call that function directly). The README's one-line `preprocess_inputs -> prepare_context_parallel_input`
  description skips the middle step but is otherwise accurate.
- **Run-time selection.** `config_registry.py` functions return a
  [`Trainer.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L76 "torchtitan/trainer#Trainer.Config"), e.g.
  [`llama3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L63 "torchtitan/models/llama3/config_registry#llama3_debugmodel") and
  [`llama3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L257 "torchtitan/models/llama3/config_registry#llama3_8b"); they are found by
  [`ConfigManager._load_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L58 "torchtitan/config/manager#ConfigManager._load_config").
- **Checkpoints.** Adapters subclass [`BaseStateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/state_dict_adapter.py#L19 "torchtitan/protocols/state_dict_adapter#BaseStateDictAdapter")
  ([`to_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/state_dict_adapter.py#L41 "torchtitan/protocols/state_dict_adapter#BaseStateDictAdapter.to_hf"),
  [`from_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/state_dict_adapter.py#L53 "torchtitan/protocols/state_dict_adapter#BaseStateDictAdapter.from_hf")) and are used both
  offline and by [`CheckpointManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/dcp.py#L89 "torchtitan/components/checkpointer/dcp#CheckpointManager") (now in
  `components/checkpointer/dcp.py`) on sharded DTensors.

## Where the README and the code disagree
> [!warning] Doc vs code
> - **No loss builder in `ModelSpec`.** The README lists "loss function builder" as a `ModelSpec`
>   field; the dataclass has `name`, `flavor`, `model`, `max_context_length`, `parallelize_fn`,
>   `pipelining_fn`, `post_optimizer_build_fn` and `state_dict_adapter`, and carries a
>   `TODO: deprecate ModelSpec, move fields to model config or trainer config`. The loss is chosen on
>   the trainer config (the TPU Qwen3 recipes set it there).
> - **The flavor table holds `(builder, max_context_length)` tuples, not configs.** The README says a
>   dict of `[str: Model.Config]`; `model_registry` unpacks the tuple, rejects a `seq_len` above the
>   flavor's maximum, and calls the builder with `attn_backend` (and MoE comm backend) arguments.
> - **No `pipeline.py`.** No core model has one. Decoder LMs pass the shared
>   [`pipeline_llm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L65 "torchtitan/distributed/pipeline_parallel#pipeline_llm"); multimodal models
>   (Qwen3.5/3.6/3.8, Kimi K2.7, Muse Glimmer) pass
>   [`pipeline_with_first_stage_modules`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L144 "torchtitan/distributed/pipeline_parallel#pipeline_with_first_stage_modules") with their vision
>   modules pinned to the first stage; Flux and Kimi K3 pass `None`.
> - **Dead links.** `docs/extension.md` and `docs/converging.md` are not present: the checkout has no
>   top-level `docs/` directory.

## Model families (README status vs code)
Per-family pages: [llama3-model](llama3-model.md), [qwen3-model](qwen3-model.md),
[qwen3-5-model](qwen3-5-model.md) (also Qwen3.6 and Qwen3.8), [deepseek-v3-model](deepseek-v3-model.md),
[deepseek-v4-model](deepseek-v4-model.md), [kimi-k2-7-model](kimi-k2-7-model.md),
[kimi-k3-model](kimi-k3-model.md), [muse-glimmer-model](muse-glimmer-model.md),
[flux-model](flux-model.md). GPT-OSS is covered here because its README is short.

**GPT-OSS** (`torchtitan/models/gpt_oss/README.md`): lists FSDP/HSDP, TP, EP, CP and PP, plus grouped
matmul. CI runs CP and PP on the debug model (`gpt_oss_pp+fsdp+cp+ep+sacop` on recipe
`gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac`, and `gpt_oss_pp+fsdp+ep+sacop`, in
`tests/integration_tests/models.py`) with Interleaved1F1B. Zero-bubble / split-backward PP tests are
disabled in `tests/integration_tests/features.py` because the FlexInnerAttention `BlockMask` is not a
Tensor; full-backward schedules (1F1B, GPipe, Interleaved1F1B) are unaffected. In code,
[`parallelize_gptoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/parallelize.py#L58 "torchtitan/models/gpt_oss/parallelize#parallelize_gptoss") rejects Ulysses CP
([`UlyssesCPFlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/cp_attention.py#L113 "torchtitan/models/common/cp_attention#UlyssesCPFlexInnerAttention")) because the per-head sinks are not
sharded over CP, which the README does not say, and the
[`gpt_oss model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/__init__.py#L395 "torchtitan/models/gpt_oss/__init__#model_registry") sets
`pipelining_fn=pipeline_llm`. The model uses learned
attention sinks ([`apply_attention_sink_rescale`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/model.py#L37 "torchtitan/models/gpt_oss/model#apply_attention_sink_rescale")), alternating
sliding-window layers ([`_build_gptoss_layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/__init__.py#L185 "torchtitan/models/gpt_oss/__init__#_build_gptoss_layers")), biased experts
([`GptOssGroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/gpt_oss/moe.py#L74 "torchtitan/models/gpt_oss/moe#GptOssGroupedExperts")) and defaults to `attn_backend="varlen"`,
which needs CUDA FlashAttention. For TPU there is no `experiments/tpu/gpt_oss`; a port would need
`attn_backend="sdpa"` plus a Splash path for sinks (the TPU Splash patch reads a per-layer
`sliding_window_size`, but sink support is not established), and `dim=2880` is not a multiple of 128
for the MXU (see [model-definitions](../concepts/model-definitions.md)).

## Why it matters / when it applies
This is the checklist for bringing a new architecture to TPU: write the model once against the
shared components, express TP/SP/EP as `sharding_config`, and reuse the core `parallelize.py`
order. The TPU experiment mirrors the layout (the TPU Qwen3 folder, for example, has `__init__.py` with its own
`model_registry`, `config_registry.py` and an `infra/parallelize.py`) and typically *wraps* the core
registry instead of copying it, as [`the TPU Qwen3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/__init__.py#L144 "torchtitan/experiments/tpu/qwen3/__init__#model_registry")
does. The README's numerics method (load the same HF checkpoint into TorchTitan and HF and compare
outputs) is the cheapest correctness check for a TPU port, because it tests the model and the
state-dict adapter together.

## Connections
- Code concepts: [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [model-definitions](../concepts/model-definitions.md),
  [job-configuration](../concepts/job-configuration.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [context-parallel](../concepts/context-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md), [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-authoring-conventions](model-authoring-conventions.md),
  [experiments-policy](experiments-policy.md), [llama3-model](llama3-model.md),
  [qwen3-model](qwen3-model.md), [qwen3-5-model](qwen3-5-model.md),
  [deepseek-v3-model](deepseek-v3-model.md), [deepseek-v4-model](deepseek-v4-model.md),
  [kimi-k2-7-model](kimi-k2-7-model.md), [kimi-k3-model](kimi-k3-model.md),
  [muse-glimmer-model](muse-glimmer-model.md), [flux-model](flux-model.md),
  [checkpoint-conversion](checkpoint-conversion.md), [testing-strategy](testing-strategy.md).

## Source
Extracted from `torchtitan/models/README.md` (kept in place); the GPT-OSS section is from
`torchtitan/models/gpt_oss/README.md`.
