---
title: "NVFP4 training (experimental FP4 linears) on Blackwell GPUs"
type: doc-concept
provenance: doc
source: torchtitan/components/quantization/nvfp4.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# NVFP4 training (experimental FP4 linears) on Blackwell GPUs

## Definition
The NVFP4 guide describes it as dynamically quantizing "linear GEMM activations, weights, and
gradients to NVFP4 through TorchAO's training prototype", while "the model weights and
distributed collectives remain in bf16". It targets NVIDIA Blackwell (SM100+), is marked
experimental ("not guarded by CI"), and its central recipe is a **mixed-precision layer
policy**: run most decoder blocks in NVFP4 but "leave approximately the final 15% of decoder
blocks in BF16", and keep the LM head in BF16. The doc's evidence is two 200M-token C4 runs:
Llama 3 8B (NVFP4 30,040 tokens/s/GPU vs 21,919 BF16, loss on par, ~111 GiB vs ~174 GiB) and
Qwen3 8B (28% more throughput than BF16 with 30% less peak reserved memory, loss 0.015 above BF16).

## In torchtitan (grounded)
- **Converter.** [`NVFP4LinearConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L267 "torchtitan/components/quantization/nvfp4#NVFP4LinearConverter") swaps every
  `Linear.Config` whose FQN contains an entry of
  [`fqns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L272 "torchtitan/components/quantization/nvfp4#NVFP4LinearConverter.Config.fqns") (all linears when `fqns` is empty, which
  would also quantize the LM head) for the config of [`NVFP4Linear`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L84 "torchtitan/components/quantization/nvfp4#NVFP4Linear").
  [`convert`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L299 "torchtitan/components/quantization/nvfp4#NVFP4LinearConverter.convert") raises on a selected router gate. The
  constructor raises `ImportError` if torchao lacks `nvfp4_training`, raises
  `ValueError("NVFP4 is only supported on SM100 or later architectures")` without
  `has_cuda_capability(10, 0)`, and only warns when `model_compile_enabled` is off.
- **Module.** `NVFP4Linear` reuses torchao's `NVFP4Linear` (bf16 weight, functional forward)
  and adds TorchTitan's Module protocol. Its [`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L95 "torchtitan/components/quantization/nvfp4#NVFP4Linear.Config") rejects
  `in_features`/`out_features` not divisible by
  [`_NVFP4_BLOCK`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L40 "torchtitan/components/quantization/nvfp4#_NVFP4_BLOCK") = 128 ("exclude it from the converter fqns"); the
  per-rank local dims after TP sharding are checked later by the torchao kernel, matching the
  doc's "local GEMM dimensions divisible by 128" requirement.
- **Pure leaf swap under TP.** The module docstring calls the converter "a pure leaf swap": it
  "inherits the model's stock colwise/rowwise sharding and changes only the GEMM", and "NVFP4
  does not move fp4 codes over the wire". [`build`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L113 "torchtitan/components/quantization/nvfp4#NVFP4Linear.Config.build") folds the
  stock weight placement into an `spmd.local_map` region around the opaque NVFP4 GEMM, which is
  registered as a local autograd function for `spmd_types`.
- **Runtime buffers.** The Random Hadamard Transform sign vector is the fixed
  [`_HARDCODED_SIGN_VECTOR`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L48 "torchtitan/components/quantization/nvfp4#_HARDCODED_SIGN_VECTOR"), identical on every TP rank because
  the transform "only cancels between the two operands when both use the same sign vector".
  The stochastic-rounding seed is drawn per rank in
  [`_init_self_buffers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L210 "torchtitan/components/quantization/nvfp4#NVFP4Linear._init_self_buffers"). Both buffers are
  non-persistent, so neither is checkpointed.
- **The 85% policy.** [`nvfp4_bf16_tail_fqns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/nvfp4.py#L247 "torchtitan/components/quantization/nvfp4#nvfp4_bf16_tail_fqns") turns a layer
  count and a BF16 tail fraction into `fqns` entries `layers.<i>.` for the leading layers (the
  trailing dot keeps `layers.1.` from matching `layers.10`), and raises if nothing would be
  converted, since an empty list would convert every linear. With fraction 0.15,
  [`llama3_8b_first_85_pct_layers_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L292 "torchtitan/models/llama3/config_registry#llama3_8b_first_85_pct_layers_nvfp4")
  converts layers 0-26 of 32 and
  [`qwen3_8b_first_85_pct_layers_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/qwen3/config_registry.py#L262 "torchtitan/models/qwen3/config_registry#qwen3_8b_first_85_pct_layers_nvfp4")
  layers 0-29 of 36, as the doc states. The Llama recipe also sets
  `parallelism.spmd_backend = "spmd_types"` and enables model compile, which the doc mentions
  only as "enables model compilation automatically". Debug-model variants such as
  [`llama3_debugmodel_nvfp4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/config_registry.py#L174 "torchtitan/models/llama3/config_registry#llama3_debugmodel_nvfp4") exist for smoke tests.
- **MFU.** [`has_quantization`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/quantization/utils.py#L67 "torchtitan/components/quantization/utils#has_quantization")
  recognizes `NVFP4Linear` configs, so MFU is not reported for NVFP4 runs (see
  [quantization](../concepts/quantization.md)).

## Why it matters / when it applies
NVFP4 is a throughput and memory play for Blackwell pretraining, with a numerical cost that the
doc manages by layer selection rather than by per-tensor tricks. Its recommendations, drawn
from two cited papers: keep the precision-sensitive final ~15% of blocks and the LM head in
BF16; if matching BF16 loss matters, switch to higher precision shortly before learning-rate
decay. The papers find most of the gap comes from forward-pass quantization, but the
TorchTitan integration "does not support independently switching Fprop to BF16 while retaining
NVFP4 for Dgrad and Wgrad", so the practical fallback is a full-BF16 finish: save a checkpoint
at the switch boundary, restart with the NVFP4 converter removed, and restore model, optimizer,
LR scheduler, dataloader and step state (see [checkpointing](../concepts/checkpointing.md)).
Known limits: SM100+ only, no end-to-end GPU CI coverage, GEMMs only (TP collectives stay in
bf16), and results validated only on the two documented 8B configurations.

**On TPU this cannot run.** The SM100 check fails on any host without CUDA, the kernels are
torchao Triton kernels, and there is no emulate path or TPU subclass. FP4 block formats are
designed around Blackwell tensor cores; see the TPU reading in
[quantization](../concepts/quantization.md).

## Connections
- Code concepts: [quantization](../concepts/quantization.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [checkpointing](../concepts/checkpointing.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md).
- Module catalogs: [module map](../catalog/index.md) (`components/quantization/nvfp4`,
  `components/quantization/utils`).
- Related doc-concepts: [float8-training](float8-training.md), [mxfp8-training](mxfp8-training.md),
  [qwen3-model](qwen3-model.md).

## Source
Extracted from `torchtitan/components/quantization/nvfp4.md` (kept in place).
