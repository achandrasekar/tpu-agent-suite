---
title: "Qwen3.5 in TorchTitan (and the Qwen3.6 / Qwen3.8 packages): hybrid GatedDeltaNet attention, vision scatter, MRoPE"
type: doc-concept
provenance: doc
source: torchtitan/models/qwen3_5/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Qwen3.5 model

## Definition
The Qwen3.5 README describes a multimodal model with a **hybrid decoder** ("75% GatedDeltaNet
(linear attention) + 25% full attention with output gating and partial RoPE"), a ViT vision
encoder with 2D RoPE and bilinear-interpolated learned position embeddings, a patch merger that
folds `merge_size^2` patches into one token, **MRoPE** interleaving temporal/height/width positions,
and an MoE variant with routed experts plus a sigmoid-gated shared expert. Visual tokens are
**scattered** into placeholder positions of the text embedding sequence. Dense flavors run from
0.8B to 27B (head dim 256), MoE flavors are 35B-A3B, 122B-A10B (256 experts, top-8) and 397B-A17B
(512 experts, top-10). Parallelisms: FSDP/HSDP (vision encoder sharded as one unit), TP with SP
(head-sharded on GatedDeltaNet projections), EP, PP (vision encoder on the first stage) and sample
packing, now opt-in via `MMSamplePackingConfig`; CP and video training are TODO. Extra dependencies
are installed with `pip install -r .ci/docker/requirements-vlm.txt`. It reports ~3e-7 average KL against HF Transformers on 4B
and bit-identical logits across no-parallel, FSDP, FSDP+EP and FSDP+EP+TP.

## In torchtitan (grounded)
- **Model.** [`Qwen35Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/model.py#L248 "torchtitan/models/qwen3_5/model#Qwen35Model") (a [`Decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L67 "torchtitan/models/common/decoder#Decoder")),
  layers [`Qwen35TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/model.py#L185 "torchtitan/models/qwen3_5/model#Qwen35TransformerBlock") holding either
  [`Qwen35Attention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/model.py#L87 "torchtitan/models/qwen3_5/model#Qwen35Attention") or [`GatedDeltaNet`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L349 "torchtitan/models/qwen3_5/gdn#GatedDeltaNet");
  the recurrence kernel is [`GatedDeltaKernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L207 "torchtitan/models/qwen3_5/gdn#GatedDeltaKernel"), both now in `gdn.py`; the gated norm
  [`RMSNormGated`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L56 "torchtitan/models/qwen3_5/gdn#RMSNormGated"), the shared expert is the common
  [`SigmoidGatedFeedForward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/feed_forward.py#L57 "torchtitan/models/common/feed_forward#SigmoidGatedFeedForward") (built by [`_shared_experts_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/__init__.py#L130 "torchtitan/models/qwen3_5/__init__#_shared_experts_config")), positions [`MRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/rope.py#L15 "torchtitan/models/qwen3_5/rope#MRoPE").
- **Vision.** [`Qwen35VisionEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/vision_encoder.py#L308 "torchtitan/models/qwen3_5/vision_encoder#Qwen35VisionEncoder") with
  [`PatchMerger`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/vision_encoder.py#L265 "torchtitan/models/qwen3_5/vision_encoder#PatchMerger") and
  [`VisionRotaryEmbedding`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/vision_encoder.py#L225 "torchtitan/models/qwen3_5/vision_encoder#VisionRotaryEmbedding"); the scatter is
  the shared
  [`scatter_vision_embeds`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/multimodal.py#L115 "torchtitan/models/common/multimodal#scatter_vision_embeds") at runs found by
  [`get_vision_positions`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/multimodal.py#L21 "torchtitan/models/common/multimodal#get_vision_positions"), driven by [`_prepare_multimodal_embeds`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/model.py#L538 "torchtitan/models/qwen3_5/model#Qwen35Model._prepare_multimodal_embeds").
- **Parallelism.** [`parallelize_qwen3_5`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/parallelize.py#L34 "torchtitan/models/qwen3_5/parallelize#parallelize_qwen3_5"),
  [`apply_fsdp_to_vision_encoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L150 "torchtitan/distributed/fsdp#apply_fsdp_to_vision_encoder") ("One all-gather for all
  vision params is more efficient than per-layer sharding"), PP through
  [`pipeline_with_first_stage_modules`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L144 "torchtitan/distributed/pipeline_parallel#pipeline_with_first_stage_modules") with
  `first_stage_module_fqns=("vision_encoder",)`, sharding by
  [`set_qwen35_sharding_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/sharding.py#L115 "torchtitan/models/qwen3_5/sharding#set_qwen35_sharding_config"). CP raises
  `NotImplementedError` ("GatedDeltaNet (75% of layers) requires full-sequence allgather, and
  multimodal CP needs vision scatter before CP sharding"). The parallelizer calls
  `model.parallelize` under `spmd_types` or when TP or EP is on.
- **Registry and checkpoints.** [`qwen3_5 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/__init__.py#L1096 "torchtitan/models/qwen3_5/__init__#model_registry")
  over [`qwen3_5_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/__init__.py#L1082 "torchtitan/models/qwen3_5/__init__#qwen3_5_configs"), recipes such as
  [`qwen35_4b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/config_registry.py#L207 "torchtitan/models/qwen3_5/config_registry#qwen35_4b") and
  [`qwen35_35b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/config_registry.py#L310 "torchtitan/models/qwen3_5/config_registry#qwen35_35b_a3b"), adapter
  [`Qwen35StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/state_dict_adapter.py#L36 "torchtitan/models/qwen3_5/state_dict_adapter#Qwen35StateDictAdapter").

## Why it matters / when it applies
Qwen3.5 is the hardest core model to bring to TPU, and nothing under `experiments/tpu` targets it
yet:
- **GatedDeltaNet depends on a GPU kernel library.** `gdn.py` imports `chunk_gdn`, `causal_conv1d`,
  `l2norm` and `recurrent_gdn` from `attn_gym.linear` at module top level (pinned as
  `attn-gym[linear]==0.0.8` in `.ci/docker/requirements.txt`, replacing the README's earlier
  flash-linear-attention dependency). [`GatedDeltaKernel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L207 "torchtitan/models/qwen3_5/gdn#GatedDeltaKernel")
  has no backend switch any more: it runs the fused chunked kernel, or
  [`_recurrent_gdn_fwd`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L86 "torchtitan/models/qwen3_5/gdn#_recurrent_gdn_fwd") in batch-invariant mode, and
  the varlen causal conv ([`_causal_conv1d_varlen`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/gdn.py#L36 "torchtitan/models/qwen3_5/gdn#_causal_conv1d_varlen")) is
  documented as "CUDA-only". The pure-torch reference now lives only in a GPU unit test. A TPU port
  needs a chunked gated-delta kernel (for example in Pallas) and must also avoid the hard `attn_gym`
  import.
- **Full-attention layers** use head dim 256 with output gating and partial (64-dim) MRoPE; they would
  run through SDPA/Splash on TPU, but gated attention support in the TPU Splash path is not
  established (see [model-definitions](../concepts/model-definitions.md)).
- **Dynamic vision shapes.** The scatter is a Python loop over host-side `(item_idx, vision_start,
  n_tokens)` tuples (`get_vision_positions` converts counts with `int()`), so each distinct image layout gives a different program; under a whole-graph
  TPU compile that likely means recompilation per layout unless inputs are bucketed or padded.
  (Inference from the code; not measured.)
- **Upside:** the reported bit-identical logits across FSDP, EP and TP are a ready-made parity
  target for a TPU port, and the MoE widths (256/512 experts) exercise the same grouped-matmul path
  as Qwen3 MoE.

## Qwen3.6 and Qwen3.8: version packages over the Qwen3.5 code
`torchtitan/models/qwen3_6/README.md` and `torchtitan/models/qwen3_8/README.md` describe "lightweight
version-specific" packages: the architecture stays in `qwen3_5/` "because Qwen3.6 uses the same
Qwen3.5 Hugging Face architecture", and each directory owns only a model registry, training recipes
with HF asset paths, and checkpoint notes. The shared model, Gated DeltaNet, RoPE, vision encoder,
sharding, parallelization and state-dict adapter keep their Qwen3.5 names.
- **Qwen3.6.** [`qwen3_6 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_6/__init__.py#L43 "torchtitan/models/qwen3_6/__init__#model_registry")
  over [`qwen3_6_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_6/__init__.py#L35 "torchtitan/models/qwen3_6/__init__#qwen3_6_configs") reuses the Qwen3.5 `27B` (dense) and
  `35B-A3B` (MoE, 256 experts top-8) builders, both with the vision encoder; recipes include
  [`qwen36_27b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_6/config_registry.py#L129 "torchtitan/models/qwen3_6/config_registry#qwen36_27b") (`MODULE=qwen3_6 CONFIG=qwen36_27b`) and
  [`qwen36_35b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_6/config_registry.py#L164 "torchtitan/models/qwen3_6/config_registry#qwen36_35b_a3b").
- **Qwen3.8.** [`qwen3_8 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_8/__init__.py#L103 "torchtitan/models/qwen3_8/__init__#model_registry")
  over [`qwen3_8_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_8/__init__.py#L95 "torchtitan/models/qwen3_8/__init__#qwen3_8_configs"): `27B` is the Qwen3.5 `_27b`
  builder ("architecturally identical to Qwen3.5-27B"), and
  [`_qwen3_8_2_4t_a95b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_8/__init__.py#L41 "torchtitan/models/qwen3_8/__init__#_qwen3_8_2_4t_a95b") is a text-only MoE (`dim=8192`, 92
  layers, 64/4 heads, 512 experts top-10, no vision encoder). Because that config has no vision
  encoder, [`Qwen35StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_5/state_dict_adapter.py#L36 "torchtitan/models/qwen3_5/state_dict_adapter#Qwen35StateDictAdapter") maps keys under
  `model.` instead of `model.language_model.`, which is the README's text-only prefix note. The HF
  configs' `output_gate_type: "swish"` matches the SiLU gate already in the GatedDeltaNet gated norm;
  full-attention output gating stays sigmoid. Recipes:
  [`qwen38_27b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_8/config_registry.py#L142 "torchtitan/models/qwen3_8/config_registry#qwen38_27b"),
  [`qwen38_2_4t_a95b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3_8/config_registry.py#L177 "torchtitan/models/qwen3_8/config_registry#qwen38_2_4t_a95b").
- **Shared limits (both READMEs).** Released MTP weights are neither imported nor exported (only the
  primary next-token decoder is trained), and pre-quantized FP8 HF repositories are not supported by
  the adapter: convert from the non-FP8 checkpoints (see [checkpoint-conversion](checkpoint-conversion.md)).
  Everything above about TPU applies unchanged, since both packages run the Qwen3.5 modules.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [context-parallel](../concepts/context-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [qwen3-model](qwen3-model.md), [model-registry](model-registry.md),
  [kimi-k2-7-model](kimi-k2-7-model.md), [kimi-k3-model](kimi-k3-model.md),
  [checkpoint-conversion](checkpoint-conversion.md),
  [model-authoring-conventions](model-authoring-conventions.md).

## Source
Extracted from `torchtitan/models/qwen3_5/README.md` (kept in place); the Qwen3.6 / Qwen3.8 section
is from `torchtitan/models/qwen3_6/README.md` and `torchtitan/models/qwen3_8/README.md`.
