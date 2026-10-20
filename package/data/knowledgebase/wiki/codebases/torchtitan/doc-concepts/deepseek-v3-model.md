---
title: "DeepSeek-V3 in TorchTitan: MLA + fine-grained MoE, tokenizer caveats, fused-kernel option, HF checkpoint loading"
type: doc-concept
provenance: doc
source: torchtitan/models/deepseek_v3/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# DeepSeek-V3 model

## Definition
The DeepSeek-V3 README covers three things. **Tokenizers:** the 671B model uses the
`deepseek-ai/DeepSeek-V3.1-Base` tokenizer; the 16B model borrows the `deepseek-moe-16b-base`
tokenizer only so it can be run, and the README warns that DeepSeek-V3 "has a different architecture
from the deepseek-moe models (different attention implementation, MoE router implementation, etc.)",
so deepseek-moe-16b *weights* cannot be loaded into DeepSeek-V3-16B. **Training:** recipes
`deepseek_v3_debugmodel`, `deepseek_v3_16b` and `deepseek_v3_671b`. **Performance option:** fused
Triton kernels for MLA Q/KV assembly, ComplexRoPE and SwiGLU, opted into with
`--override.imports torchtitan.overrides.fused_mla.fused_mla,torchtitan.overrides.fused_swiglu.fused_swiglu`;
the README says they keep the model parameters and checkpoint layout. **Checkpoints:** HF safetensors
-> DCP conversion through the model's StateDictAdapter, "currently" only offline on CPU plain
tensors, via `scripts/checkpoint_conversion/convert_from_hf.py`.

## In torchtitan (grounded)
- **Model.** [`DeepSeekV3Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L237 "torchtitan/models/deepseek_v3/model#DeepSeekV3Model") with
  [`DeepSeekV3TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L162 "torchtitan/models/deepseek_v3/model#DeepSeekV3TransformerBlock"); attention is Multi-head Latent
  Attention, [`deepseek_v3 Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L32 "torchtitan/models/deepseek_v3/model#Attention"), configured by
  [`q_lora_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L49 "torchtitan/models/deepseek_v3/model#Attention.Config.q_lora_rank"),
  [`kv_lora_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L50 "torchtitan/models/deepseek_v3/model#Attention.Config.kv_lora_rank"),
  [`qk_nope_head_dim`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L53 "torchtitan/models/deepseek_v3/model#Attention.Config.qk_nope_head_dim"),
  [`qk_rope_head_dim`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L54 "torchtitan/models/deepseek_v3/model#Attention.Config.qk_rope_head_dim") and
  [`v_head_dim`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/model.py#L55 "torchtitan/models/deepseek_v3/model#Attention.Config.v_head_dim") (built by
  [`make_mla_attention_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L85 "torchtitan/models/deepseek_v3/__init__#make_mla_attention_config"), called from
  [`build_mla_moe_layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L175 "torchtitan/models/deepseek_v3/__init__#build_mla_moe_layers"), which the Kimi K2.7 family reuses;
  see [kimi-k2-7-model](kimi-k2-7-model.md)). MoE layers use the shared
  [`TokenChoiceTopKRouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L586 "torchtitan/models/common/moe#TokenChoiceTopKRouter") (sigmoid scores, node-limited group routing)
  and [`GroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L440 "torchtitan/models/common/moe#GroupedExperts").
- **Flavors and recipes.** [`deepseekv3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L708 "torchtitan/models/deepseek_v3/__init__#deepseekv3_configs") has
  debugmodel, 16B, 236B and 671B ([`_16b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L406 "torchtitan/models/deepseek_v3/__init__#_16b"),
  [`_671b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L557 "torchtitan/models/deepseek_v3/__init__#_671b")); the recipes are
  [`deepseek_v3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L66 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_debugmodel"),
  [`deepseek_v3_16b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L182 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_16b") and
  [`deepseek_v3_671b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L252 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_671b"), plus GPU EP-backend variants
  the README does not list ([`deepseek_v3_16b_hybridep`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L217 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_16b_hybridep"),
  [`deepseek_v3_16b_minimal_async_ep`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L230 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_16b_minimal_async_ep")), an MTP debug recipe
  ([`deepseek_v3_debugmodel_mtp`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L105 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_debugmodel_mtp")) and low-precision
  variants ([`deepseek_v3_debugmodel_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L118 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_debugmodel_mxfp8"),
  [`deepseek_v3_671b_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L292 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_671b_float8")).
- **Fused-kernel option.** `fused_mla` swaps the attention config for
  [`FusedMLAAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_mla.py#L803 "torchtitan/overrides/fused_mla#FusedMLAAttention") via
  [`fused_mla`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_mla.py#L899 "torchtitan/overrides/fused_mla#fused_mla"), which keeps the stock `Attention` parameters
  and replaces only the Q/KV layout boundary around ComplexRoPE (Q RoPE in place, K RoPE plus head
  expansion in one Triton kernel, V left as a view of the packed KV projection). Its module docstring
  limits it to DeepSeek-V3's packed MLA layout with `ComplexRoPE` (no `CosSinRoPE` or `MRoPE`).
  [`fused_swiglu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L585 "torchtitan/overrides/fused_swiglu#fused_swiglu") covers dense FFNs only; the GPU EP
  recipes call [`enable_fused_swiglu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L55 "torchtitan/models/deepseek_v3/config_registry#enable_fused_swiglu"), which also adds
  [`fused_grouped_experts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L710 "torchtitan/overrides/fused_swiglu#fused_grouped_experts") for the routed experts. The
  README's command line does not include that MoE override, so on its own it fuses only the dense
  layers and shared experts. Overrides are applied through
  [`OverrideConfig.imports`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/override.py#L62 "torchtitan/config/override#OverrideConfig.imports") (see [kernel-overrides](kernel-overrides.md),
  [fused-swiglu-and-helion-rope](fused-swiglu-and-helion-rope.md)).
- **Registry.** [`deepseek_v3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/__init__.py#L723 "torchtitan/models/deepseek_v3/__init__#model_registry")
  takes `attn_backend`, `moe_comm_backend` and `non_blocking_capacity_factor`, and wires
  [`parallelize_deepseekv3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/parallelize.py#L21 "torchtitan/models/deepseek_v3/parallelize#parallelize_deepseekv3") and
  [`register_moe_load_balancing_hook`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/optimizer/optimizer.py#L449 "torchtitan/components/optimizer/optimizer#register_moe_load_balancing_hook").
- **Checkpoints: the README understates the code.**
  [`DeepSeekV3StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/state_dict_adapter.py#L20 "torchtitan/models/deepseek_v3/state_dict_adapter#DeepSeekV3StateDictAdapter") overrides
  [`get_hf_storage_reader`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/state_dict_adapter.py#L129 "torchtitan/models/deepseek_v3/state_dict_adapter#DeepSeekV3StateDictAdapter.get_hf_storage_reader")
  to return PyTorch's `QuantizedHuggingFaceStorageReader` (128-block dequantization to fp32) when
  `from_quantized` is set, "to read DeepSeek-V3 671B model", and implements
  [`to_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/state_dict_adapter.py#L152 "torchtitan/models/deepseek_v3/state_dict_adapter#DeepSeekV3StateDictAdapter.to_hf") (splitting
  grouped expert weights per expert) and
  [`from_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/state_dict_adapter.py#L211 "torchtitan/models/deepseek_v3/state_dict_adapter#DeepSeekV3StateDictAdapter.from_hf"). That
  is the hook [`CheckpointManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/dcp.py#L89 "torchtitan/components/checkpointer/dcp#CheckpointManager") uses to load HF weights into
  sharded DTensors during training, so "offline only" reads as stale; treat the offline script as one
  supported path, not the only one.

## Why it matters / when it applies
For TPU, DeepSeek-V3 is covered by the TPU experiment and the graph trainer, but only at small
scale in this checkout. The TPU table ([`TPU deepseekv3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/deepseek_v3/__init__.py#L156 "torchtitan/experiments/tpu/deepseek_v3/__init__#deepseekv3_configs"))
builds `debugmodel` with `attn_backend="sdpa"` and the large flavors with `"flex"`, and the only TPU
recipe is [`deepseek_v3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/deepseek_v3/config_registry.py#L28 "torchtitan/experiments/tpu/deepseek_v3/config_registry#deepseek_v3_debugmodel") (plus the
tiny [`_testmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/deepseek_v3/__init__.py#L17 "torchtitan/experiments/tpu/deepseek_v3/__init__#_testmodel") and
[`_testmodel_moe`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/deepseek_v3/__init__.py#L77 "torchtitan/experiments/tpu/deepseek_v3/__init__#_testmodel_moe")). The TPU configs use
`moe_comm_backend="standard"`: the HybridEP and minimal-async EP recipes above use GPU-specific
communication backends (see [expert-parallel](../concepts/expert-parallel.md)). Shape risks for the MXU: MLA's q/k head dim is 128 + 64 = 192 (not a multiple of 128) and
the value head dim is 128, and fine-grained MoE (256 experts, top-8) yields small per-expert token
groups. The tokenizer caveat also matters for TPU smoke tests: a 16B run uses a stand-in tokenizer,
so its loss is not comparable with an official DeepSeek-V3 evaluation. The README's
performance-optimized option does not carry to TPU: `fused_mla` and `fused_swiglu` launch Triton
kernels, so a TPU run keeps the stock modules (or needs a Pallas equivalent).

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [checkpointing](../concepts/checkpointing.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md), [qwen3-model](qwen3-model.md),
  [deepseek-v4-model](deepseek-v4-model.md), [kimi-k2-7-model](kimi-k2-7-model.md),
  [kernel-overrides](kernel-overrides.md), [fused-swiglu-and-helion-rope](fused-swiglu-and-helion-rope.md),
  [checkpoint-conversion](checkpoint-conversion.md), [tpu-training-recipes](tpu-training-recipes.md).

## Source
Extracted from `torchtitan/models/deepseek_v3/README.md` (kept in place).
