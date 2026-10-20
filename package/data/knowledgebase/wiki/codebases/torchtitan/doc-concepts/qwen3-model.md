---
title: "Qwen3 in TorchTitan: dense and MoE flavors, MoE load balancing, and the TPU lane"
type: doc-concept
provenance: doc
source: torchtitan/models/qwen3/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Qwen3 model

## Definition
The Qwen3 README lists a dense model (FSDP/HSDP, TP, CP, DDP, activation checkpointing,
`torch.compile`) and an MoE model (the same plus EP) with Token Choice routing whose load balancing
"follows the original Qwen3 training recipe (auxiliary loss)". The "still under development" banner
is gone: dense and MoE debug models are covered by `tests/integration_tests/models.py` (FSDP+TP+CP,
and FSDP+TP+CP+EP for MoE). Architectures exist for 4B, 8B and 235B-A22B without pretrain
`config_registry` recipes; 8B is available as SFT via `sft_qwen3_8b_math`. Remaining work: those
recipes, verifying the learning rate and schedule on longer runs, and comparing against established
performance benchmarks. Tokenizers come from the HF repo (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, ...).

## In torchtitan (grounded)
- **Model.** [`Qwen3Model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/model.py#L75 "torchtitan/models/qwen3/model#Qwen3Model") with
  [`Qwen3TransformerBlock`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/model.py#L26 "torchtitan/models/qwen3/model#Qwen3TransformerBlock"), a [`Decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/decoder.py#L67 "torchtitan/models/common/decoder#Decoder")
  with per-head QK-norm and `CosSinRoPE` (theta 1e6). Flavors are in
  [`qwen3_configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L669 "torchtitan/models/qwen3/__init__#qwen3_configs") and bundled by
  [`qwen3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L686 "torchtitan/models/qwen3/__init__#model_registry"), which uses
  [`parallelize_qwen3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/parallelize.py#L28 "torchtitan/models/qwen3/parallelize#parallelize_qwen3"), the shared
  [`pipeline_llm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/pipeline_parallel.py#L65 "torchtitan/distributed/pipeline_parallel#pipeline_llm"),
  [`register_moe_load_balancing_hook`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/optimizer/optimizer.py#L449 "torchtitan/components/optimizer/optimizer#register_moe_load_balancing_hook") and
  [`Qwen3StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/state_dict_adapter.py#L26 "torchtitan/models/qwen3/state_dict_adapter#Qwen3StateDictAdapter").
- **"Sizes without pretrain recipes" matches the code.** `qwen3_configs` has 4B, 8B and 235B-A22B flavors
  ([`_4b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L340 "torchtitan/models/qwen3/__init__#_4b"), [`_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L378 "torchtitan/models/qwen3/__init__#_8b"),
  [`_235b_a22b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L591 "torchtitan/models/qwen3/__init__#_235b_a22b")), but the core `config_registry.py` only ships
  recipes such as [`qwen3_0_6b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L203 "torchtitan/models/qwen3/config_registry#qwen3_0_6b"),
  [`qwen3_1_7b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L233 "torchtitan/models/qwen3/config_registry#qwen3_1_7b"),
  [`qwen3_14b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L288 "torchtitan/models/qwen3/config_registry#qwen3_14b"),
  [`qwen3_32b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L382 "torchtitan/models/qwen3/config_registry#qwen3_32b") and
  [`qwen3_30b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L323 "torchtitan/models/qwen3/config_registry#qwen3_30b_a3b").
  8B appears only as SFT, [`sft_qwen3_8b_math`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L512 "torchtitan/models/qwen3/config_registry#sft_qwen3_8b_math"), and its NVFP4 derivative
  [`qwen3_8b_first_85_pct_layers_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L262 "torchtitan/models/qwen3/config_registry#qwen3_8b_first_85_pct_layers_nvfp4"). The CI matrix the README
  cites includes [`qwen3_debugmodel_fsdp2_tp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L232 "torchtitan_recipes/tests/models#qwen3_debugmodel_fsdp2_tp2_cp2") and
  [`qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/models.py#L216 "torchtitan_recipes/tests/models#qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8") (recipes live in
  `torchtitan_recipes/tests/models.py`).
- **Load balancing: bias by default, aux loss opt-in.** [`_30b_a3b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L529 "torchtitan/models/qwen3/__init__#_30b_a3b") (128 experts, top-8,
  expert hidden 768) defaults to `load_balance_coeff=1e-3`
  ([`MoE.Config.load_balance_coeff`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L958 "torchtitan/models/common/moe#MoE.Config.load_balance_coeff")): a
  per-expert bias steers routing and is updated by the optimizer hook, not by a loss term, and the
  router's `aux_loss` defaults to `None`; [`_235b_a22b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L591 "torchtitan/models/qwen3/__init__#_235b_a22b") uses the same default. The
  alternative [`_30b_a3b_global_aux`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/__init__.py#L576 "torchtitan/models/qwen3/__init__#_30b_a3b_global_aux") and recipe
  [`qwen3_30b_a3b_global_aux`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L358 "torchtitan/models/qwen3/config_registry#qwen3_30b_a3b_global_aux") use Qwen's global aux loss
  ([`MoELoadBalanceAuxLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/moe.py#L177 "torchtitan/models/common/moe#MoELoadBalanceAuxLoss")).

> [!warning] Doc vs code
> The README now says MoE load balancing "follows the original Qwen3 training recipe (auxiliary
> loss)". In code the shipped MoE flavors (`30B-A3B`, `235B-A22B`) default to the auxiliary-loss-free
> per-expert bias (`load_balance_coeff=1e-3`, no router aux loss); Qwen's global-batch aux loss is
> used only by the `30B-A3B-global-aux` flavor and its recipe (and by the TPU 235B recipes).

## Why it matters / when it applies
Qwen3 is the model the TPU lane is built around. `experiments/tpu/qwen3` wraps the core registry
([`TPU qwen3 model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/__init__.py#L144 "torchtitan/experiments/tpu/qwen3/__init__#model_registry")) and changes only
FLOP accounting ([`MaxTextQwen3Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/model.py#L19 "torchtitan/experiments/tpu/qwen3/model#MaxTextQwen3Config"), MaxText-style FLOPs
so MFU is comparable with MaxText), except for 8B with SDPA, where
[`_8b_tpu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/__init__.py#L72 "torchtitan/experiments/tpu/qwen3/__init__#_8b_tpu") sets `fuse_qkv=False` "to avoid running attention
projection GEMMs as a single fused block on TPU". The TPU recipes (see
[tpu-training-recipes](tpu-training-recipes.md)) include
[`qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L375 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b"),
[`qwen3_moe_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L557 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_30b") and the 235B family, and add TPU
choices the README does not mention:
- the Pallas fused cross-entropy ([`_enable_pallas_cross_entropy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L121 "torchtitan/experiments/tpu/qwen3/config_registry#_enable_pallas_cross_entropy"),
  setting [`Trainer.Config.loss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L130 "torchtitan/trainer#Trainer.Config.loss") to
  [`PallasCrossEntropyLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/loss.py#L103 "torchtitan/experiments/tpu/loss#PallasCrossEntropyLoss"));
- for 235B, a switch *away* from the core bias-based default:
  [`_use_global_batch_aux_loss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L128 "torchtitan/experiments/tpu/qwen3/config_registry#_use_global_batch_aux_loss") sets
  `load_balance_coeff=None` and a batch-wise aux loss (coeff 1e-3) on every MoE layer, used by
  [`qwen3_moe_235b_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L950 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_235b_scan") among others. Loss curves
  from those runs are therefore not directly comparable to core bias-balanced runs.

Shapes are TPU-friendly: head dim 128, and the MoE expert hidden size 768 is 6 x 128 (see the MXU
note on [model-definitions](../concepts/model-definitions.md)). For the README's open benchmark TODO,
the TPU lane's reference point is MaxText rather than a GPU baseline (for example
[`qwen3_8b_maxtext_parity`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L474 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b_maxtext_parity")); the README's LR-schedule
TODO has no dedicated recipe in this checkout.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [model-registry](model-registry.md), [qwen3-5-model](qwen3-5-model.md),
  [deepseek-v3-model](deepseek-v3-model.md), [tpu-training-recipes](tpu-training-recipes.md).

## Source
Extracted from `torchtitan/models/qwen3/README.md` (kept in place).
