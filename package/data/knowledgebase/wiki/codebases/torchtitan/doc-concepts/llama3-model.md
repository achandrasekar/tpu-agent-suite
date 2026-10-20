---
title: "Llama 3 in TorchTitan: the reference decoder, its recipes and parallelism coverage"
type: doc-concept
provenance: doc
source: torchtitan/models/llama3/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Llama 3 model

## Definition
The Llama 3 README calls it "the reference decoder model in torchtitan": the model the models README
tells new models to copy ([model-registry](model-registry.md)) and "the baseline used to validate
distributed training techniques". Recipes cover Llama 3.1 8B, 70B and 405B plus a small debug model
used by CI; the 8B/70B/405B recipes expect tokenizer assets under `./assets/hf/Llama-3.1-8B` (and the
matching 70B / 405B paths), downloaded with `scripts/download_hf_assets.py`. Its parallelism table,
which it says matches `parallelize.py` and the Llama 3 jobs in `tests/integration_tests/features.py`
(plus Float8 jobs in `tests/integration_tests/h100.py`), lists FSDP/HSDP, TP with sequence parallel
(async TP on H100), CP (composing with FSDP, HSDP, DDP and TP), PP (1F1B, Interleaved1F1B, GPipe),
DDP (including DDP+CP), selective and full activation checkpointing, `torch.compile`, Float8 and
MXFP8. It warns against treating "any single published KL or MFU number as a parity claim".

## In torchtitan (grounded)
- **Model.** [`Llama3Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/model.py#L59 "torchtitan/models/llama3/model#Llama3Model") with
  [`Llama3TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/model.py#L22 "torchtitan/models/llama3/model#Llama3TransformerBlock"): GQA with fused QKV, SwiGLU, `ComplexRoPE`
  (see [model-definitions](../concepts/model-definitions.md)). Flavors `debugmodel`, `1B`, `3B`, `8B`,
  `70B`, `405B` live in [`llama3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L377 "torchtitan/models/llama3/__init__#llama3_configs") as
  `(builder, max_context_length)` pairs (131072 for all), built by
  [`_build_llama3_layers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L69 "torchtitan/models/llama3/__init__#_build_llama3_layers").
- **Registry.** [`llama3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L387 "torchtitan/models/llama3/__init__#model_registry") defaults to
  `attn_backend="flex"`, also takes `tp_gemm_backend`, and wires
  [`parallelize_llama`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/parallelize.py#L23 "torchtitan/models/llama3/parallelize#parallelize_llama"), the shared
  [`pipeline_llm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L65 "torchtitan/distributed/pipeline_parallel#pipeline_llm") and
  [`Llama3StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/state_dict_adapter.py#L18 "torchtitan/models/llama3/state_dict_adapter#Llama3StateDictAdapter").
- **Parallelize order.** `parallelize_llama` calls `model.parallelize` under `spmd_types` or when TP
  is on, then activation checkpointing, per-block compile, and always
  [`apply_fsdp_to_decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L189 "torchtitan/distributed/fsdp#apply_fsdp_to_decoder") ("with shard_degree=1 it is a no-op for the
  all-gather but still installs the MixedPrecisionPolicy"). CP is not applied here; it is input-side
  through `Decoder.preprocess_inputs` (see [context-parallel](../concepts/context-parallel.md)).
- **Recipes.** [`llama3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L63 "torchtitan/models/llama3/config_registry#llama3_debugmodel") (test tokenizer),
  [`llama3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L257 "torchtitan/models/llama3/config_registry#llama3_8b"),
  [`llama3_70b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L332 "torchtitan/models/llama3/config_registry#llama3_70b"),
  [`llama3_405b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L370 "torchtitan/models/llama3/config_registry#llama3_405b"), the Float8 debug recipe
  [`llama3_debugmodel_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L142 "torchtitan/models/llama3/config_registry#llama3_debugmodel_float8") (imported directly by
  `h100.py`) and [`llama3_8b_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L316 "torchtitan/models/llama3/config_registry#llama3_8b_mxfp8"). The registry also has
  variants the README does not list: varlen attention
  ([`llama3_debugmodel_varlen_attn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L110 "torchtitan/models/llama3/config_registry#llama3_debugmodel_varlen_attn")), dist-GEMM TP
  ([`llama3_debugmodel_dist_gemm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L122 "torchtitan/models/llama3/config_registry#llama3_debugmodel_dist_gemm")), NVFP4
  ([`llama3_debugmodel_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L174 "torchtitan/models/llama3/config_registry#llama3_debugmodel_nvfp4"),
  [`llama3_8b_first_85_pct_layers_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L292 "torchtitan/models/llama3/config_registry#llama3_8b_first_85_pct_layers_nvfp4")), a float8 LoRA
  emulation recipe and an SFT debug recipe ([`sft_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L425 "torchtitan/models/llama3/config_registry#sft_debugmodel")).
- **CI coverage.** The feature jobs use recipes from `torchtitan_recipes/tests/`, for example
  [`llama3_debugmodel_ddp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L363 "torchtitan_recipes/tests/features#llama3_debugmodel_ddp4"),
  [`llama3_debugmodel_ddp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L419 "torchtitan_recipes/tests/features#llama3_debugmodel_ddp2_cp2") and
  [`llama3_debugmodel_tp2_asynctp_compile`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/h100.py#L25 "torchtitan_recipes/tests/h100#llama3_debugmodel_tp2_asynctp_compile"). The zero-bubble recipe
  [`llama3_debugmodel_pp4_zero_bubble`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L324 "torchtitan_recipes/tests/features#llama3_debugmodel_pp4_zero_bubble") exists but its test is
  disabled in `features.py`: split-backward schedules call `requires_grad` on every stage input,
  including the forwarded FlexInnerAttention `BlockMask`, which is not a Tensor. The same limit is
  recorded for GPT-OSS ([model-registry](model-registry.md)).

> [!warning] Doc vs code
> The README's "Numerical checks" section links `docs/converging.md`; the checkout has no top-level
> `docs/` directory, so the link is dead.

## Why it matters / when it applies
Llama 3 is also the reference model of the TPU lane. `experiments/tpu/llama3` rebuilds the upstream
flavors with `attn_backend="sdpa"` through
[`_upstream_flavor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/__init__.py#L48 "torchtitan/experiments/tpu/llama3/__init__#_upstream_flavor") into its own
[`TPU llama3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/__init__.py#L63 "torchtitan/experiments/tpu/llama3/__init__#llama3_configs") (FlexAttention
compiles through inductor/Triton and is not the TPU path), and ships TPU recipes such as
[`TPU llama3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L178 "torchtitan/experiments/tpu/llama3/config_registry#llama3_8b"),
[`TPU llama3_70b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L244 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b"), scan-lane variants
([`llama3_70b_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L472 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b_scan")), Float8
([`llama3_70b_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L510 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b_float8")) and SFT
([`sft_llama3_8b_metamath`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L376 "torchtitan/experiments/tpu/llama3/config_registry#sft_llama3_8b_metamath")). The README's GPU
coverage table therefore does not describe TPU support: on TPU, the per-feature status is on
[tpu-support-status-and-limits](tpu-support-status-and-limits.md), and sized recipes are on
[tpu-training-recipes](tpu-training-recipes.md) and [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md).
Shapes are MXU-friendly: head dim 128 in the 8B/70B/405B flavors and hidden sizes that are multiples
of 1024 (see [model-definitions](../concepts/model-definitions.md)).

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md), [tensor-parallel](../concepts/tensor-parallel.md),
  [context-parallel](../concepts/context-parallel.md),
  [pipeline-parallel](../concepts/pipeline-parallel.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [quantization](../concepts/quantization.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md), [testing-strategy](testing-strategy.md),
  [float8-training](float8-training.md), [mxfp8-training](mxfp8-training.md),
  [nvfp4-training](nvfp4-training.md),
  [tpu-training-recipes](tpu-training-recipes.md), [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md).

## Source
Extracted from `torchtitan/models/llama3/README.md` (kept in place).
