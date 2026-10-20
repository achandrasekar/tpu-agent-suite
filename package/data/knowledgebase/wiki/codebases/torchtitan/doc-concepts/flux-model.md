---
title: "FLUX in TorchTitan: text-to-image diffusion training, per-block compile, and its TPU wiring"
type: doc-concept
provenance: doc
source: torchtitan/models/flux/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# FLUX model

## Definition
The FLUX README says TorchTitan implements pre-training of the text-to-image part of FLUX (the
autoencoder is downloaded from `black-forest-labs/FLUX.1-dev`). Supported: FSDP, HSDP and CP;
activation checkpointing; `torch.compile` applied **per block** to the repeated `DoubleStreamBlock`
and `SingleStreamBlock` layers with `fullgraph=True` (model and loss both compiled by default,
selectable with `--compile.components`); MXFP8 linear quantization on SM100+ (Blackwell) GPUs; and
distributed checkpointing, with weights re-sharded at the end of evaluation so the checkpoint layout
matches training. The remaining TODO is more parallelism (TP and PP); the earlier TODO for FLOPs
accounting in `get_nparams_and_flops()` has been dropped from the README.

## In torchtitan (grounded)
- **Model.** [`FluxModel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/model.py#L30 "torchtitan/models/flux/model/model#FluxModel") built from
  [`DoubleStreamBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/layers.py#L221 "torchtitan/models/flux/model/layers#DoubleStreamBlock") and
  [`SingleStreamBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/layers.py#L340 "torchtitan/models/flux/model/layers#SingleStreamBlock"), with
  [`AutoEncoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/autoencoder.py#L385 "torchtitan/models/flux/model/autoencoder#AutoEncoder") and text encoders
  ([`FluxEmbedder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/hf_embedder.py#L18 "torchtitan/models/flux/model/hf_embedder#FluxEmbedder")). It is not a decoder LM, so it has its
  own trainer, [`FluxTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/trainer.py#L38 "torchtitan/models/flux/trainer#FluxTrainer").
- **Registry.** [`flux_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/__init__.py#L670 "torchtitan/models/flux/__init__#flux_configs") (`flux-dev`, `flux-schnell`,
  `flux-debug`) and [`flux model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/__init__.py#L678 "torchtitan/models/flux/__init__#model_registry"), which sets
  `pipelining_fn=None` (no PP, as the README says) and uses
  [`FluxStateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/model/state_dict_adapter.py#L22 "torchtitan/models/flux/model/state_dict_adapter#FluxStateDictAdapter").
- **Compile.** [`apply_compile`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/parallelize.py#L184 "torchtitan/models/flux/parallelize#apply_compile") compiles each double and single
  block with `fullgraph=True`, as documented.
- **MXFP8.** The README still calls the converter `MXFP8Converter`; the recipes
  [`flux_schnell_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L244 "torchtitan/models/flux/config_registry#flux_schnell_mxfp8") and
  [`flux_dev_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L272 "torchtitan/models/flux/config_registry#flux_dev_mxfp8") actually use
  [`MXFP8LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L58 "torchtitan/components/quantization/mxfp8/converter#MXFP8LinearConverter") (now under
  `components/quantization/mxfp8/`, exported from `torchtitan.components.quantization`). Use that name
  in a custom config.
- **FLOPs accounting is implemented**, matching the README's removal of that TODO:
  `FluxModel.Config.get_nparams_and_flops` computes `num_flops_per_token` (6 x params with
  corrections for the two-stream blocks and per-sample modulation).

> [!warning] Doc vs code
> The README names `MXFP8Converter`; no such class exists at this pin, the converter is
> `MXFP8LinearConverter`.

## Why it matters / when it applies
In this fork, Flux is wired for TPU **inside core**, not only under `experiments/tpu`:
- the Flux recipes ([`flux_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L55 "torchtitan/models/flux/config_registry#flux_debugmodel"),
  [`flux_dev`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L128 "torchtitan/models/flux/config_registry#flux_dev"), [`flux_schnell`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L186 "torchtitan/models/flux/config_registry#flux_schnell"))
  set a [`TPUConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L62 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig") with `eager_mode="DEFER_AND_FUSE"`,
  and `FluxTrainer.Config` subclasses [`TPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L308 "torchtitan/experiments/tpu/tpu_job_config#TPUTrainerConfig");
- [`parallelize_flux`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/parallelize.py#L52 "torchtitan/models/flux/parallelize#parallelize_flux") applies
  [`use_splash_attention_patch`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/workarounds.py#L33 "torchtitan/experiments/tpu/workarounds#use_splash_attention_patch") when the TPU config enables the
  Splash kernel, supports a SimpleFSDP path (`use_simple_fsdp`), and `apply_compile` calls
  [`apply_spmd_safe`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/spmd_utils.py#L17 "torchtitan/experiments/tpu/spmd_utils#apply_spmd_safe") on every block before compiling.
The TPU launcher [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer") routes `model == "flux"` to this
core `FluxTrainer` (never the graph trainer, and `compile_grad_norm` is rejected), with the TPU table
[`TPU flux_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/flux/__init__.py#L27 "torchtitan/experiments/tpu/flux/__init__#flux_configs") reusing the core flavors
and its own [`TPU parallelize_flux`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/flux/infra/parallelize.py#L10 "torchtitan/experiments/tpu/flux/infra/parallelize#parallelize_flux").
This is the clearest case of experiment code leaking into core, contrary to the "don't leak
experiments into core" principle ([repo-conventions](repo-conventions.md),
[experiments-policy](experiments-policy.md)).

What carries to TPU from the README: per-block `fullgraph=True` compile is the TPU-friendly shape
(one compiled program reused across 19 + 38 identical blocks), and FSDP/HSDP/CP are the available
parallelisms. What does not: MXFP8 is Blackwell-only (it needs `torchao` MX kernels on SM100+), so on
TPU it is not an option in this checkout.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [quantization](../concepts/quantization.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [context-parallel](../concepts/context-parallel.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [checkpointing](../concepts/checkpointing.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md), [repo-conventions](repo-conventions.md),
  [experiments-policy](experiments-policy.md), [mxfp8-training](mxfp8-training.md).

## Source
Extracted from `torchtitan/models/flux/README.md` (kept in place).
