---
title: "Worked override examples: fused SwiGLU, Helion RoPE and Triton OffsetRMSNorm"
type: doc-concept
provenance: doc
source: torchtitan/overrides/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Worked override examples: fused SwiGLU, Helion RoPE and Triton OffsetRMSNorm

## Definition
The overrides README ships worked examples of the
[@override mechanism](kernel-overrides.md).
- `fused_swiglu.py` is "the parametrization example". It fuses the SwiGLU gate and up
  projections into one `w13` Linear with a `(2*hidden, dim)` weight whose gate/up rows are
  interleaved, computed by a single GEMM, with a `ShardingConfig` that composes with both FSDP
  and TP.
- `helion_rope.py` is "the custom-kernel example". It swaps the stock RoPE for a fused Helion
  kernel (forward and backward), wrapped as a `torch.library.custom_op` with `register_fake` /
  `register_autograd`.
- `offset_rmsnorm.py` (new at this pin) replaces Qwen3.5 `OffsetRMSNorm` with fused Triton
  forward and backward kernels "while preserving the stock zero-centered weight and checkpoint
  layout".

The README uses the first two to teach three things an override author owns: TP-safe weight layout,
checkpoint compatibility through state-dict hooks, and making custom kernels traceable by
`torch.compile`.

## In torchtitan (grounded)
- **Fused SwiGLU.** [`fused_swiglu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L585 "torchtitan/overrides/fused_swiglu#fused_swiglu")
  is now an `exact=True` override on `FeedForward.Config` with no `fqns` (activate it as
  `torchtitan.overrides.fused_swiglu.fused_swiglu`). It builds [`FusedSwiGLU`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L450 "torchtitan/overrides/fused_swiglu#FusedSwiGLU"),
  whose [`forward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L472 "torchtitan/overrides/fused_swiglu#FusedSwiGLU.forward")
  runs the `w13` Linear, unflattens the output to `(hidden, 2)`, and calls the fused
  [`silu_and_mul_op`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L301 "torchtitan/overrides/fused_swiglu#silu_and_mul_op")
  (`torchtitan::silu_and_mul`, a Triton kernel). Because gate/up rows are interleaved, sharding
  row axis 0 gives each TP rank matching slices of both projections (the Megatron
  column-parallel layout), through the ordinary `Linear` `ShardingConfig`; keeping `w13` a
  standard `Linear` also lets config-time linear converters replace it. A second exact override,
  [`dist_gemm_fused_swiglu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L594 "torchtitan/overrides/fused_swiglu#dist_gemm_fused_swiglu"),
  targets [`DistGEMMFeedForward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/dist_gemm.py#L178 "torchtitan/models/common/dist_gemm#DistGEMMFeedForward")
  and builds [`DistGEMMFusedSwiGLU`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L507 "torchtitan/overrides/fused_swiglu#DistGEMMFusedSwiGLU"),
  so the dist-GEMM TP overlap "cannot be replaced accidentally by the regular fused
  implementation" (README; used with
  [`llama3_debugmodel_dist_gemm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L122 "torchtitan/models/llama3/config_registry#llama3_debugmodel_dist_gemm")).
  The same module also registers
  [`fused_grouped_experts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L710 "torchtitan/overrides/fused_swiglu#fused_grouped_experts")
  for `GroupedExperts.Config`, which builds
  [`FusedGroupedExperts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L600 "torchtitan/overrides/fused_swiglu#FusedGroupedExperts")
  with an `(E, F, 2, D)` weight and `torch._grouped_mm` (not `exact`; it returns any non-stock
  config unchanged). The README does not mention this override. Since each override is now
  activated by its own `module.function` target, DeepSeek-V3 recipes list both
  `...fused_swiglu.fused_swiglu` and `...fused_swiglu.fused_grouped_experts` through
  [`enable_fused_swiglu`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/deepseek_v3/config_registry.py#L55 "torchtitan/models/deepseek_v3/config_registry#enable_fused_swiglu").
- **Checkpoint hooks.** [`_split_w13_on_save`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L478 "torchtitan/overrides/fused_swiglu#FusedSwiGLU._split_w13_on_save")
  saves `w13.weight` as the stock `w1.weight` / `w3.weight`, and
  [`_merge_w13_on_load`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/fused_swiglu.py#L490 "torchtitan/overrides/fused_swiglu#FusedSwiGLU._merge_w13_on_load")
  interleaves them back on load, so checkpoints are a drop-in for stock `FeedForward` and for
  the HF adapter. The former native `w13` key is still accepted for back-compat. (The README's
  earlier claim that fused SwiGLU "is not interoperable with stock checkpoints" was corrected at
  this pin.)
- **Helion RoPE.** [`helion_cos_sin_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/helion_rope.py#L1041 "torchtitan/overrides/helion_rope#helion_cos_sin_rope")
  and [`helion_complex_rope`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/helion_rope.py#L1055 "torchtitan/overrides/helion_rope#helion_complex_rope")
  are `exact=True` overrides on `CosSinRoPE.Config` and `ComplexRoPE.Config`. Subclasses such as
  Qwen3.5 `MRoPE` are left alone. Each must be listed by name
  (`torchtitan.overrides.helion_rope.helion_cos_sin_rope` and/or `...helion_complex_rope`). They
  build
  [`HelionCosSinRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/helion_rope.py#L972 "torchtitan/overrides/helion_rope#HelionCosSinRoPE")
  and [`HelionComplexRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/helion_rope.py#L1006 "torchtitan/overrides/helion_rope#HelionComplexRoPE").
  At run time, [`_cossin_eligible`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/helion_rope.py#L736 "torchtitan/overrides/helion_rope#_cossin_eligible")
  requires CUDA tensors on a single device with the expected shapes. Otherwise the module warns
  once and falls back to the PyTorch RoPE.
- **Triton OffsetRMSNorm.** [`triton_offset_rmsnorm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/offset_rmsnorm.py#L343 "torchtitan/overrides/offset_rmsnorm#triton_offset_rmsnorm")
  builds [`TritonOffsetRMSNorm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/overrides/offset_rmsnorm.py#L306 "torchtitan/overrides/offset_rmsnorm#TritonOffsetRMSNorm");
  activate it with `--override.imports torchtitan.overrides.offset_rmsnorm.triton_offset_rmsnorm`.

> [!warning] Doc vs code
> - The README says `helion` is optional and the module "falls back to the PyTorch RoPE when it
>   (or CUDA) is unavailable". The fallback exists only for unsupported inputs, including
>   non-CUDA tensors. If `helion` is not installed, the factory raises `ImportError` while the
>   override is being applied.
> - The README's file table lists only `fused_swiglu.py` and `helion_rope.py` under
>   `torchtitan/overrides/`; the directory also ships `offset_rmsnorm.py`, `fused_mla.py`
>   (DeepSeek-V3 fused MLA Q/KV assembly) and `moe_token_dispatcher.py` (DeepEP / HybridEP
>   dispatcher overrides).

## Why it matters / when it applies
These two files are the templates to copy when writing a new override. Copy the weight-layout
reasoning from `fused_swiglu`, because a fused gate+up weight must order its rows (here,
interleaved) so that a row shard gives every TP rank matching gate and up slices. Copy the custom-op wrapping from `helion_rope`, because a raw kernel call breaks Dynamo
tracing.

**On TPU, neither example is useful.**
- `fused_swiglu.py` imports `triton` when the module loads, and registers `torchtitan::silu_and_mul`
  with `device_types="cuda"` only. `FusedSwiGLU.forward` always calls that op, so it has no TPU
  implementation.
- `offset_rmsnorm.py` is likewise a Triton kernel.
- `helion_rope` would at best fall back to plain PyTorch RoPE on TPU tensors (after a warning),
  and it fails outright if `helion` is not installed.

The TPU-native overrides that follow the same pattern live in `experiments/tpu/overrides/`:
- [`TpuActivationDtypeCosSinRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/bf16_rope.py#L26 "torchtitan/experiments/tpu/overrides/bf16_rope#TpuActivationDtypeCosSinRoPE")
  is the TPU counterpart of a faster RoPE. It removes the FP32 elementwise intermediates instead
  of calling a custom kernel.
- [`TpuRealArithmeticComplexRoPE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/real_rope.py#L30 "torchtitan/experiments/tpu/overrides/real_rope#TpuRealArithmeticComplexRoPE")
  is bit-identical to `ComplexRoPE` but can be traced inside `scan`.
- [`FusedLossLinearWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/overrides/fused_loss.py#L18 "torchtitan/experiments/tpu/overrides/fused_loss#FusedLossLinearWrapper")
  is the fusion example on TPU: it defers the LM-head GEMM into a Pallas fused cross-entropy
  kernel.

The XLA compiler fuses elementwise tails such as SiLU-and-mul by itself, so the Triton kernel has
nothing to port to TPU. No TPU recipe or test in the repo checks whether the fused `w13` GEMM
layout helps on TPU.

## Connections
- Code concepts: [model-definitions](../concepts/model-definitions.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [checkpointing](../concepts/checkpointing.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [expert-parallel](../concepts/expert-parallel.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [module map](../catalog/index.md) (`overrides/fused_swiglu`,
  `overrides/helion_rope`, `overrides/offset_rmsnorm`, `experiments/tpu/overrides/*`).
- Related doc-concepts: [kernel-overrides](kernel-overrides.md),
  [qwen3-5-model](qwen3-5-model.md), [deepseek-v3-model](deepseek-v3-model.md).

## Source
Extracted from `torchtitan/overrides/README.md` (kept in place).
