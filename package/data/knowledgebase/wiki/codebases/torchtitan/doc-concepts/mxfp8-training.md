---
title: "MXFP8 training (block-scaled FP8) on Blackwell GPUs"
type: doc-concept
provenance: doc
source: torchtitan/components/quantization/mxfp8/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# MXFP8 training (block-scaled FP8) on Blackwell GPUs

## Definition
The MXFP8 guide describes MXFP8 as "a microscaling format from the MX OCP spec that uses
block-based scaling". Every 1x32 block of `float8_e4m3fn` values shares one `float8_e8m0fnu`
scale, which is finer-grained than per-tensor or per-row scaling. Activations are quantized
dynamically before each GEMM or grouped GEMM, and the GEMMs run on B200 tensor cores via
`scaled_mm` / `torch._scaled_grouped_mm`. At this pin the dense linear is no longer a thin
torchao wrapper: TorchTitan owns the MXFP8 linear autograd and an FSDP-managed weight lifecycle,
and weights use square **32x32** scale tiles ("orientation-symmetric: FPROP and DGRAD use the
same quantized values and share one cached qdata allocation") while activations keep 1x32
tiles. The doc reports up to 28% speedup over BF16 on dense models and +20% on a GB200 MoE run,
a Llama 3 8B 32x32-weight convergence run that "closely tracks the BF16 baseline", and says
MXFP8 composes with `torch.compile`, FSDP2/TP/EP/PP and full activation checkpointing, with
communication in high precision.

## In torchtitan (grounded)
- **Division of labour.** The doc draws a narrow boundary: torchao supplies only kernels
  (`mxfp8_quantize_cuda` for activations, a Triton 32x32 swizzled weight cast with one shared
  qdata allocation and both scale layouts, and a scale-layout rearrange), while TorchTitan owns
  the autograd, the FSDP unsharded-tensor lifecycle, quantized-weight storage, and the per-model
  input-activation storage policy. The code lives in the `components/quantization/mxfp8/`
  package (`converter.py`, `linear.py`, `tensor.py`) plus the generic
  `components/quantization/_fsdp_tensor.py`.
- **Linear converter.** [`MXFP8LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L58 "torchtitan/components/quantization/mxfp8/converter#MXFP8LinearConverter")
  swaps each `Linear.Config` whose FQN contains an entry of
  [`fqns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L63 "torchtitan/components/quantization/mxfp8/converter#MXFP8LinearConverter.Config.fqns")
  (every Linear when `fqns` is empty) to the config of
  [`MXFP8Linear`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L303 "torchtitan/components/quantization/mxfp8/linear#MXFP8Linear").
  Selecting a router gate raises ("exclude ... with fqns"). Unlike float8, this converter does
  not apply the divisible-by-16 filter from
  [`module_filter_fn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/utils.py#L16 "torchtitan/components/quantization/utils#module_filter_fn");
  instead [`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L307 "torchtitan/components/quantization/mxfp8/linear#MXFP8Linear.Config") raises unless both
  `in_features` and `out_features` are divisible by 32 (the doc: "both local weight dimensions
  must be divisible by 32").
- **FSDP-managed weights.** `MXFP8Linear.__init__` wraps its weight in
  [`_LinearShardedTensorWithMXFP8Compute`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/tensor.py#L83 "torchtitan/components/quantization/mxfp8/tensor#_LinearShardedTensorWithMXFP8Compute").
  After FSDP all-gathers the BF16 weight,
  [`fsdp_post_all_gather`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/_fsdp_tensor.py#L398 "torchtitan/components/quantization/_fsdp_tensor#_ShardedFSDPTensor.fsdp_post_all_gather")
  quantizes it with
  [`_quantize_mxfp8_weight`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/tensor.py#L54 "torchtitan/components/quantization/mxfp8/tensor#_quantize_mxfp8_weight")
  and returns a storage-free
  [`_UnshardedFSDPTensor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/_fsdp_tensor.py#L446 "torchtitan/components/quantization/_fsdp_tensor#_UnshardedFSDPTensor")
  holding FPROP and DGRAD operands, so the quantized weight follows FSDP's
  reshard-after-forward / reshard-after-backward policy and the BF16 all-gather output can be
  released. With `reshard_after_forward=False` the operands are built on the first unshard and
  reused until FSDP releases them after backward (pipeline microbatches reuse them too); with
  `True`, backward does another BF16 all-gather and quantization. Both operands are built on
  every actual unshard because FSDP does not expose which phase an unshard serves. When no data
  parallel implementation owns the weight (for example GraphTrainer under `spmd_types`),
  [`forward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L368 "torchtitan/components/quantization/mxfp8/linear#MXFP8Linear.forward")
  quantizes the high-precision storage on every call.
- **Input-activation save format.** WGRAD needs the linear input in backward.
  [`input_activation_format_for_backward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L310 "torchtitan/components/quantization/mxfp8/linear#MXFP8Linear.Config.input_activation_format_for_backward")
  is `"bf16"` (save the input, quantize it columnwise in backward; the default) or `"mxfp8"`
  (produce rowwise and columnwise operands in forward and save the columnwise qdata and scales).
  The converter sets it per linear from
  [`linears_saving_inputs_for_backward_in_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L69 "torchtitan/components/quantization/mxfp8/converter#MXFP8LinearConverter.Config.linears_saving_inputs_for_backward_in_mxfp8")
  (FQN substrings; an empty selector, or one matching no converted linear, raises). BF16 is "the
  conservative default" because saving MXFP8 only helps when no other op already keeps the BF16
  input alive (flash attention's output is the doc's example). Under full activation
  checkpointing the configured format is applied to both the original forward and the
  recompute, so an MXFP8-selected linear builds and discards a columnwise operand in the
  original forward; the doc accepts that to avoid a recomputation-aware autograd contract. The
  built-in policies, shared by Trainer and GraphTrainer, are
  [`llama3_mxfp8_linear_converter_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L41 "torchtitan/models/llama3/config_registry#llama3_mxfp8_linear_converter_config")
  (`attention.qkv_linear.wqkv`, `feed_forward.w2`) and
  [`deepseek_v3_mxfp8_linear_converter_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L31 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_mxfp8_linear_converter_config")
  (`attention.wkv_b`, `feed_forward.w2`, `shared_experts.w2`, with `fqns` limited to attention,
  shared experts and dense FFNs); Flux saves BF16 everywhere.
- **SPMD backend.** [`build`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L336 "torchtitan/components/quantization/mxfp8/linear#MXFP8Linear.Config.build")
  rejects `parallelism.spmd_backend='partial_dtensor'`: the MXFP8 matmul
  ([`_MXFP8LinearFunction`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/linear.py#L66 "torchtitan/components/quantization/mxfp8/linear#_MXFP8LinearFunction")) is an opaque autograd
  function with no DTensor sharding rule, so tensor parallelism needs the `spmd_types` backend.
  The doc does not mention this.
- **MoE converter.** [`MXFP8GroupedExpertsConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L236 "torchtitan/components/quantization/mxfp8/converter#MXFP8GroupedExpertsConverter")
  creates an MXFP8 subclass of each `GroupedExperts` class
  ([`_get_mxfp8_grouped_experts_cls`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L188 "torchtitan/components/quantization/mxfp8/converter#_get_mxfp8_grouped_experts_cls"))
  that overrides `_grouped_mm` to call torchao's `_quantize_then_scaled_grouped_mm` with
  `MXFP8TrainingOpConfig.from_recipe(...)`. It pads token groups to
  [`pad_multiple`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L248 "torchtitan/components/quantization/mxfp8/converter#MXFP8GroupedExpertsConverter.Config.pad_multiple")
  (default 32) through [`swap_token_dispatcher`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/utils.py#L33 "torchtitan/components/quantization/utils#swap_token_dispatcher"),
  which takes the routed-experts parent config. That function turns an all-to-all dispatcher into
  [`TorchAOTokenDispatcher`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/token_dispatcher.py#L621 "torchtitan/models/common/token_dispatcher#TorchAOTokenDispatcher"),
  gives [`HybridEPTokenDispatcher`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/token_dispatcher.py#L881 "torchtitan/models/common/token_dispatcher#HybridEPTokenDispatcher")
  a padded config, and raises for any other dispatcher. The docstring of `pad_multiple` notes
  that the CuTeDSL quantization kernel on sm_100 needs multiples of 128.
- **Hardware and install gates.** Both converters raise `ValueError("MXFP8 is only supported on
  SM100 or later architectures")` unless `has_cuda_capability(10, 0)` holds. There is no ROCm
  path and no emulate path. The linear converter also raises `ImportError` when torchao lacks the
  32x32 swizzled cast kernel, printing a nightly install command from
  [`_torchao_nightly_install_command`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/mxfp8/converter.py#L37 "torchtitan/components/quantization/mxfp8/converter#_torchao_nightly_install_command").
- **Recipes.** [`llama3_8b_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L316 "torchtitan/models/llama3/config_registry#llama3_8b_mxfp8"),
  [`flux_schnell_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L244 "torchtitan/models/flux/config_registry#flux_schnell_mxfp8"),
  [`flux_dev_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/flux/config_registry.py#L272 "torchtitan/models/flux/config_registry#flux_dev_mxfp8")
  and the GraphTrainer variant
  [`graph_trainer_llama3_8b_mxfp8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/llama3/config_registry.py#L137 "torchtitan/experiments/graph_trainer/llama3/config_registry#graph_trainer_llama3_8b_mxfp8")
  ship with MXFP8 (plus debug-model variants).

> [!warning] Doc vs code
> - **Argument name.** Every example in the doc calls `model_registry(..., quantization=[...])`.
>   The real parameter is `converters=[...]`, and the MXFP8 recipes use it.
> - **torchao version.** The doc's requirements say "TorchAO 0.18.0 or later". The converter's
>   import error says the 32x32 swizzled cast kernels "are not in any release up to v0.18.0, so a
>   nightly is required".
> - **Who installs the FSDP wrapper.** The doc says "FSDP setup installs the unsharded-tensor
>   wrapper immediately before sharding" and that the module "keeps an ordinary `nn.Parameter`
>   when FSDP is not used". In the code `MXFP8Linear.__init__` installs the wrapper up front ("so
>   no caller has to remember to do it"); without FSDP it stays inert and `forward` quantizes its
>   storage per call.
> - **Grouped-experts recipe.** `MXFP8GroupedExpertsConverter.Config.recipe_name` exists, but it
>   is `Literal["mxfp8_rceil"]` (its only value).
> - **Dispatcher name.** The doc names `DeepEPTokenDispatcher` as the padded dispatcher. The code
>   handles `HybridEPTokenDispatcher`.
> - **EP requirement.** The doc says "Expert parallelism (EP) must be enabled". The code does not
>   check the EP degree. It only requires a dispatcher that can pad.
> - **Example FQNs.** The combined and "671B" examples reuse Flux's
>   `fqns=["double_blocks", "single_blocks"]`; on a DeepSeek model those match no linear, so the
>   linear converter would convert nothing.

## Why it matters / when it applies
MXFP8 is aimed at NVIDIA B200/GB200 (SM100) with a PyTorch nightly and a torchao build that has
the CUTLASS/Triton kernels. The doc says it is a prototype with no backward-compatibility
guarantees. The doc's GB200 MoE example leaves `output`, `router.gate`, `attention.wk` and
`attention.wv` in BF16, because they are too small to benefit. The save-format choice is a memory
knob: the right setting depends on which ops already keep a linear's input alive and on the
activation-checkpointing policy, so the shipped policies opt in only audited single-consumer
inputs.

**On TPU this cannot run at all.** The SM100 capability check fails on any host without a CUDA
device, there is no emulate flag, and the kernels are CUDA/Triton. Both converters therefore
raise inside `model_registry(...)`, while the config is still being built, and the TPU
experiment ships no MXFP8 subclass (its only low-precision converter is the float8 one described
in [float8-training](float8-training.md)). For how low precision is handled on the TPU path, see
[quantization](../concepts/quantization.md) and [torchtpu-backend](../concepts/torchtpu-backend.md).
The README's composability table in [graph-trainer](graph-trainer.md) also lists Float8/MXFP8
under `aot_fx_trace` as work in progress.

## Connections
- Code concepts: [quantization](../concepts/quantization.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [model-definitions](../concepts/model-definitions.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`components/quantization/mxfp8/*`,
  `components/quantization/_fsdp_tensor`, `components/quantization/utils`,
  `models/common/token_dispatcher`).
- Related doc-concepts: [float8-training](float8-training.md), [nvfp4-training](nvfp4-training.md),
  [kernel-overrides](kernel-overrides.md), [graph-trainer](graph-trainer.md).

## Source
Extracted from `torchtitan/components/quantization/mxfp8/README.md` (kept in place; moved from
`torchtitan/components/quantization/mxfp8.md`).
