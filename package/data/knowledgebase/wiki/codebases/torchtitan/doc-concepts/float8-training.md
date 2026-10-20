---
title: "Float8 training (rowwise) via torchao converters"
type: doc-concept
provenance: doc
source: torchtitan/components/quantization/float8.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Float8 training (rowwise) via torchao converters

## Definition
The float8 guide ("Float8 Training on H100s") says float8 helps "models where the majority of
GEMMs are sufficiently large enough that the speedup from using float8 tensorcores outweighs the
overhead of dynamic quantization". TorchTitan turns it on at config-construction time. A
converter walks the model config tree and replaces each eligible `Linear.Config` with a torchao
`Float8Linear` config, so the model is built already quantized. For MoE models a second
converter quantizes the grouped-expert GEMMs. The only scaling supported is rowwise dynamic
scaling, with recipes `rowwise` (the default) and `rowwise_with_gw_hp`. Distributed
communication stays in high precision.

## In torchtitan (grounded)
- **Linear converter.** [`Float8LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L53 "torchtitan/components/quantization/float8#Float8LinearConverter")
  takes a config with [`recipe_name`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L58 "torchtitan/components/quantization/float8#Float8LinearConverter.Config.recipe_name"),
  [`filter_fqns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L61 "torchtitan/components/quantization/float8#Float8LinearConverter.Config.filter_fqns"),
  [`emulate`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L68 "torchtitan/components/quantization/float8#Float8LinearConverter.Config.emulate")
  and the inherited [`model_compile_enabled`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/__init__.py#L30 "torchtitan/components/quantization/__init__#QuantizationConverter.Config.model_compile_enabled").
  [`convert`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L151 "torchtitan/components/quantization/float8#Float8LinearConverter.convert")
  swaps matching nodes to [`Float8Linear`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L27 "torchtitan/components/quantization/float8#Float8Linear").
  Eligibility is decided by [`module_filter_fn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/utils.py#L16 "torchtitan/components/quantization/utils#module_filter_fn"):
  both dimensions must be divisible by 16, and the FQN must not contain any `filter_fqns`
  entry. This is a **substring** match. Putting `auto_filter_small_kn` in `filter_fqns` switches
  to torchao's `_auto_filter_for_recipe`, whose size thresholds come from H100 measurements.
  With the `rowwise` recipe the converter also sets
  `torch._inductor.config.emulate_precision_casts = True`. A selected linear whose config is a
  [`RouterGateLinear`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/linear.py#L101 "torchtitan/models/common/linear#RouterGateLinear")
  config raises `ValueError` ("exclude it with filter_fqns"), so MoE recipes must list
  `router.gate` in `filter_fqns`, as the doc's MoE example does.
- **MoE converter.** [`Float8GroupedExpertsConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L224 "torchtitan/components/quantization/float8#Float8GroupedExpertsConverter")
  subclasses each `GroupedExperts` class. The subclass keeps a torchao `Float8TrainingOpConfig`
  and overrides
  [`_grouped_mm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L206 "torchtitan/components/quantization/float8#_get_float8_grouped_experts_cls.Float8GroupedExperts._grouped_mm")
  to call torchao's quantize-then-scaled grouped GEMM (it no longer runs `quantize_` on the
  module). Through
  [`swap_token_dispatcher`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/utils.py#L33 "torchtitan/components/quantization/utils#swap_token_dispatcher")
  it also pads token groups (the function now takes the routed-experts parent config, which owns
  the `token_dispatcher` child) to
  [`PAD_MULTIPLE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/float8.py#L228 "torchtitan/components/quantization/float8#Float8GroupedExpertsConverter.PAD_MULTIPLE")
  = 16. That function raises unless the dispatcher is an all-to-all/TorchAO dispatcher or a
  HybridEP dispatcher.
- **Hardware gate.** The linear converter only constructs if one of these holds:
  - NVIDIA SM89 or newer (`has_cuda_capability(8, 9)`);
  - AMD gfx942/MI300 (`has_rocm_capability(9, 4)`);
  - `emulate=True` and `model_compile_enabled=False`.

  Otherwise it raises `ValueError`. The grouped-experts converter has no emulate escape, and it
  also raises without SM89+ or MI300.
- **Recipes.** [`llama3_405b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L370 "torchtitan/models/llama3/config_registry#llama3_405b")
  and [`deepseek_v3_671b_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L292 "torchtitan/models/deepseek_v3/config_registry#deepseek_v3_671b_float8")
  are the recipes that ship with float8 (the 671B float8 recipe uses
  `filter_fqns=["lm_head", "router.gate"]` plus the grouped-experts converter; the base
  `deepseek_v3_671b` recipe is no longer quantized). [`llama3_debugmodel_float8_emulate_lora`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L225 "torchtitan/models/llama3/config_registry#llama3_debugmodel_float8_emulate_lora")
  is the emulate example.

> [!warning] Doc vs code
> - The doc says quantization goes in through a `quantization=[...]` argument of
>   `model_registry()`. There is no such parameter. The real parameter is `converters=[...]`
>   (for example, llama3
>   [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/__init__.py#L387 "torchtitan/models/llama3/__init__#model_registry")`(flavor, attn_backend, converters)`), and a
>   `quantization=` keyword raises `TypeError`.
> - The doc's examples filter `"output"`; the final projection is named `lm_head` at this pin
>   and the shipped recipes filter `"lm_head"`. Because `filter_fqns` is a substring match,
>   `"output"` does not exclude the final projection.
> - Converter order is checked:
>   [`validate_converter_order`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/utils.py#L27 "torchtitan/models/utils#validate_converter_order")
>   raises if a quantization converter comes after a `LoRAConverter`.
> - The doc never mentions `emulate`, which is the only test path that works without a GPU.
> - The error message still points to a legacy `float8.emulate` flag. The real knob is
>   `Float8LinearConverter.Config(emulate=True)`.

## Why it matters / when it applies
Float8 pays off only when GEMMs are large enough, which is why the doc points to torchao's
microbenchmarks and to `filter_fqns` / `auto_filter_small_kn` to skip small layers. The doc and
`QuantizationConverter` both assume `torch.compile` is on, because compile fuses the
scale/cast kernels.

**On TPU: use the TPU subclass, not the upstream converter.** On a Cloud TPU host there is no
CUDA device, so both upstream converters raise `ValueError` inside `model_registry(...)`, before
training starts (the only exception is `Float8LinearConverter` with `emulate=True` in eager
mode, which the code calls "for test purpose only"). The TPU experiment now ships
[`TpuFloat8LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/float8_converter.py#L25 "torchtitan/experiments/tpu/float8_converter#TpuFloat8LinearConverter"),
a subclass that skips the CUDA/ROCm capability check, builds the torchao config from
`recipe_name` itself with `use_fast_accum=False` on the forward GEMM (its docstring: "TPU v7x has
native float8 matmuls (TorchTPU lowers `torch._scaled_mm`)", and TorchTPU's `_scaled_mm` rejects
fast accumulation because TPU matmuls accumulate in FP32), rejects `auto_filter_small_kn`
("uses CUDA-tuned thresholds; list the FQNs to skip explicitly on TPU"), and inherits `convert`,
so the resulting `Float8Linear` layers are identical to upstream. The Ironwood recipe
[`llama3_70b_float8`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L510 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b_float8")
uses it with the `rowwise` recipe (output projection kept in bf16); its tpu7x-128 recipe README
reports 10.02 s vs 13.27 s step time against bf16. There is no TPU path for the grouped-experts
converter. See [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md) for that recipe, and the TPU
reading in the [quantization](../concepts/quantization.md) code concept for MXU dtype support.
The FP8-with-FSDP walkthrough in `experiments/tpu/torchtitan-gke-xpk.md` targets GPU nodes (it
sources NCCL and uses legacy `quantize.linear.float8.*` flags), not TPU.

## Connections
- Code concepts: [quantization](../concepts/quantization.md),
  [model-definitions](../concepts/model-definitions.md),
  [lora](../concepts/lora.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`components/quantization/float8`,
  `components/quantization/utils`).
- Related doc-concepts: [mxfp8-training](mxfp8-training.md), [nvfp4-training](nvfp4-training.md),
  [kernel-overrides](kernel-overrides.md), [graph-trainer](graph-trainer.md),
  [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md).

## Source
Extracted from `torchtitan/components/quantization/float8.md` (kept in place).
