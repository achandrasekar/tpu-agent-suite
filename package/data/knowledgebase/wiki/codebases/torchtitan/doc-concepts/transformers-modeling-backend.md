---
title: "HF Transformers modeling backend: training HF model classes with TorchTitan parallelism"
type: doc-concept
provenance: doc
source: torchtitan/experiments/transformers_modeling_backend/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# HF Transformers modeling backend: training HF model classes with TorchTitan parallelism

## Definition
This experiment lets "HF transformers models be trained with 4D parallelism + torch.compile".
It does not port an architecture into `torchtitan/models/`. Instead, it instantiates the
Hugging Face `transformers` model class (the README pins `transformers==5.9.0`) for a given HF
model ID and plugs it into the standard TorchTitan trainer, parallelizers, pipeline splitter and
checkpointer. It "requires `--parallelism.spmd_backend spmd_types`". The README reports dense
models tested with FSDP/CP/TP/PP and `torch.compile` (Llama-3.2-1B, Qwen2.5-7B, Qwen3-4B,
Mistral-7B, phi-2, OLMo-7B, and others) and, new at this pin, MoE models (Qwen3-30B-A3B,
Mixtral-8x7B, OLMoE-1B-7B, DeepSeek-V2-Lite/V3, GLM-4.7, GLM-5, Gemma-4-26B-A4B) whose MoE block
is swapped for TorchTitan's grouped-experts MoE.

## In torchtitan (grounded)
- **Model wrapper.** [`HFTransformerModel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L196 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel")
  holds the HF `ForCausalLM` as `self.model`. Its config
  ([`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L201 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.Config"))
  loads the HF config with `AutoConfig.from_pretrained(hf_model, ...)` (with `trust_remote_code`
  on unless the model type is on a small denylist, and the `flex_torchtitan` attention
  implementation) and
  maps TorchTitan names onto HF keys through
  [`_TT_TO_HF_MAPPINGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L125 "torchtitan/experiments/transformers_modeling_backend/model#_TT_TO_HF_MAPPINGS") (`dim` ->
  `hidden_size`, `n_layers` -> `num_hidden_layers`, ...). It exposes TorchTitan-shaped
  accessors ([`tok_embeddings`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L1063 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.tok_embeddings"),
  [`layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L1086 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.layers"),
  [`norm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L1110 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.norm"),
  [`lm_head`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L1141 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.lm_head"))
  so that the generic parallelizers work. It patches HF weight init for meta-device
  initialization.
- **Flavors.** [`TitanModelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/__init__.py#L22 "torchtitan/experiments/transformers_modeling_backend/__init__#TitanModelConfig") sets every HF-mirrored field to `None`, so the values from `AutoConfig` win
  unless you override them (as `debugmodel` does with dim=256, 2 layers); it also carries
  `attn_mask_type`. [`TitanMoeModelConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/__init__.py#L57 "torchtitan/experiments/transformers_modeling_backend/__init__#TitanMoeModelConfig") adds the routed-expert fields (`num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`, ...), `experts_implementation` (`grouped_mm` by default),
  `load_balance_coeff` and the EP `comm_backend` (`standard`, `deepep`, `hybridep`).
  [`flavors`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/__init__.py#L108 "torchtitan/experiments/transformers_modeling_backend/__init__#flavors") defines
  `debugmodel`, `full`, `sft_debugmodel`, `sft_full`, `debugmodel_moe` and `full_moe`.
  [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/__init__.py#L163 "torchtitan/experiments/transformers_modeling_backend/__init__#model_registry") ties them
  to the parallelize, pipeline and adapter functions, and registers the MoE load-balancing hook
  as `post_optimizer_build_fn`
  ([`register_moe_load_balancing_hook`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/optimizer/optimizer.py#L449 "torchtitan/components/optimizer/optimizer#register_moe_load_balancing_hook");
  this closes the README's old "missing `build_optimizers_with_moe_load_balancing`" gap).
  The HF model ID is the trainer config field
  [`hf_model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/config_registry.py#L32 "torchtitan/experiments/transformers_modeling_backend/config_registry#TransformersBackendConfig.hf_model")
  on [`TransformersBackendConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/config_registry.py#L31 "torchtitan/experiments/transformers_modeling_backend/config_registry#TransformersBackendConfig").
- **Parallelism.** [`parallelize_hf_transformers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/parallelize.py#L87 "torchtitan/experiments/transformers_modeling_backend/parallelize#parallelize_hf_transformers") first refuses any `spmd_backend` other than `spmd_types`. It then un-ties
  a shared embedding/`lm_head` weight (Gemma-4) for FSDP, swaps each HF MoE block for a Titan MoE
  ([`build_and_swap_native_moe`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/moe_replacement.py#L92 "torchtitan/experiments/transformers_modeling_backend/moe_replacement#build_and_swap_native_moe")), converts the remaining HF modules to the TorchTitan `Module` protocol
  ([`convert_hf_to_module`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/module_conversion.py#L42 "torchtitan/experiments/transformers_modeling_backend/module_conversion#convert_hf_to_module")), and, when TP, EP or CP is on, attaches a declarative sharding config to every module
  ([`set_hf_sharding_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/hf_sharding.py#L104 "torchtitan/experiments/transformers_modeling_backend/hf_sharding#set_hf_sharding_configs")). One `model.parallelize(parallel_dims)` call then applies TP/EP/CP; under CP
  [`_wrap_flex_kernel_cp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/parallelize.py#L39 "torchtitan/experiments/transformers_modeling_backend/parallelize#_wrap_flex_kernel_cp") wraps each flex kernel to all-gather K/V over the CP axis. AC, `torch.compile` of whole
  blocks (MoE included) and FSDP
  ([`apply_fsdp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/parallelize.py#L240 "torchtitan/experiments/transformers_modeling_backend/parallelize#apply_fsdp")) follow.
  [`pipeline_hf_transformers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/pipeline.py#L290 "torchtitan/experiments/transformers_modeling_backend/pipeline#pipeline_hf_transformers")
  splits the model into PP stages.
- **Attention.** Every flavor now runs attention through flex: the HF attention implementation
  is [`_ATTN_IMPLEMENTATION`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L176 "torchtitan/experiments/transformers_modeling_backend/model#_ATTN_IMPLEMENTATION") = `flex_torchtitan`, registered in HF's `AttentionInterface` and routed to
  [`_flex_attention_torchtitan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L72 "torchtitan/experiments/transformers_modeling_backend/model#_flex_attention_torchtitan") (which uses a per-module [`HFFlexKernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L39 "torchtitan/experiments/transformers_modeling_backend/model#HFFlexKernel") under TP so the head sharding is expressed declaratively).
  [`get_attention_masks`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L1232 "torchtitan/experiments/transformers_modeling_backend/model#HFTransformerModel.get_attention_masks") builds the mask from `attn_mask_type`: `causal`, or `block_causal` (causal and
  same-document, for packed/SFT sequences). DSA models such as GLM-5
  ([`_uses_dsa`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/model.py#L179 "torchtitan/experiments/transformers_modeling_backend/model#_uses_dsa")) get a dense 4-D additive mask instead, because their indexer cannot consume a
  `BlockMask`. The earlier `sdpa_torchtitan` default, the `HFFlexAttention` class and the
  separate `SFTTrainer` are gone.
- **HF weights.** [`HFTransformerStateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py#L52 "torchtitan/experiments/transformers_modeling_backend/state_dict_adapter#HFTransformerStateDictAdapter")
  is almost the identity. [`to_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py#L71 "torchtitan/experiments/transformers_modeling_backend/state_dict_adapter#HFTransformerStateDictAdapter.to_hf")
  / [`from_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py#L83 "torchtitan/experiments/transformers_modeling_backend/state_dict_adapter#HFTransformerStateDictAdapter.from_hf")
  only strip or add the `model.` prefix and handle tied `lm_head` / `embed_tokens`.
  [`transformers_modeling_backend_sft_full`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/config_registry.py#L183 "torchtitan/experiments/transformers_modeling_backend/config_registry#transformers_modeling_backend_sft_full")
  loads real Qwen3-0.6B weights with `initial_load_in_hf=True`. MoE layouts differ (HF fuses
  `gate_up_proj`; Titan splits gate/up and renames router keys):
  [`hf_to_titan_moe_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/state_dict_adapter.py#L171 "torchtitan/experiments/transformers_modeling_backend/state_dict_adapter#hf_to_titan_moe_state_dict") and its inverse convert them, but per the module docstring they are used by the
  numerical-equivalence and round-trip tests, not by the checkpoint adapter.
- **MoE parallelism matrix.** The README's table, validated at debug scale (2-step loss
  decreasing), with FSDP composing with every axis:

  | Model(s) | attn | TP | EP | CP |
  |---|---|---|---|---|
  | Qwen3-30B-A3B | GQA | yes | yes | yes |
  | Mixtral-8x7B, OLMoE-1B-7B | GQA | yes | yes | yes |
  | DeepSeek-V2-Lite, V3, GLM-4.7 | MLA | yes | yes | yes (flex+CP verified on V2-Lite) |
  | GLM-5 | MLA + DSA | no | yes | no |
  | Gemma-4-26B-A4B | GQA | no | yes | no |

  The README's reasons for the gaps: GLM-5's DSA indexer "fails loud under TP" and CP is not
  wired for it, and Gemma-4 attention TP is not viable with `num_global_key_value_heads=2`.
  PP is not yet wired for the MoE path. `MODEL_COMPATIBILITY.md` in the same folder records the
  per-model numerical comparison of the Titan MoE swap against unmodified HF (Gemma-4-26B is the
  one FAIL there). The recipe [`transformers_modeling_backend_debugmodel_moe`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/config_registry.py#L74 "torchtitan/experiments/transformers_modeling_backend/config_registry#transformers_modeling_backend_debugmodel_moe") uses Qwen3-30B-A3B.

## Why it matters / when it applies
This is the fastest route to *training* an architecture TorchTitan has not ported. You get
FSDP/TP/PP/CP (and, for MoE models, EP) and compile without writing a model file or a
key-permuting state-dict adapter, because the module tree *is* the HF one, apart from the MoE
blocks that are swapped for Titan's grouped-experts MoE. The costs the README names are lower MFU than native
TorchTitan modeling, and loss/grad-norm that is not bitwise identical between `FSDP=2` and
`FSDP=2 + PP=2` (converging, but not bitwise; the suspected cause is `register_buffer` with seed
checkpoints).

**TPU applicability.** Nothing in this experiment targets TPU, and `experiments/tpu` does not
import it. Parallelism goes through the upstream `spmd_types` Module protocol and FSDP2, and
attention goes through FlexAttention only. In principle it could run on the torch_tpu lane
([torchtpu-backend](../concepts/torchtpu-backend.md)), which also uses DTensor, but no TPU config,
recipe or test here shows that. For TPU onboarding, prefer a native TorchTitan model with a TPU
registry entry.

> [!warning] Doc vs code
> - The README's "Further work" still lists "Load HF weights" as missing. The code ships
>   `HFTransformerStateDictAdapter` and an SFT config that loads HF safetensors at step 0 (dense
>   models; MoE checkpoints would need the expert-layout conversion that only the tests use).
> - The README does not mention how the HF model is chosen: `TransformersBackendConfig.hf_model`
>   (`--hf_model`).
> - The README's `MODEL=` env var in the train command should be `MODULE=` (that is the variable
>   `run_train.sh` reads), and its command does not pass the required
>   `--parallelism.spmd_backend spmd_types`; none of the registry configs set it either, so
>   without the flag `parallelize_hf_transformers` raises.
> - The README pins `transformers==5.9.0`; the CI requirements file for this backend
>   (`.ci/docker/requirements-transformers-modeling-backend.txt`) pins `transformers==5.10.1`.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [context-parallel](../concepts/context-parallel.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [checkpointing](../concepts/checkpointing.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/transformers_modeling_backend/model`, `parallelize`, `pipeline`,
  `state_dict_adapter`).
- Related doc-concepts: [checkpoint-conversion](checkpoint-conversion.md),
  [testing-strategy](testing-strategy.md), [experiments-policy](experiments-policy.md).

## Source
Extracted from `torchtitan/experiments/transformers_modeling_backend/README.md` (kept in place).
