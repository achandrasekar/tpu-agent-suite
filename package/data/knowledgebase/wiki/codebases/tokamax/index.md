---
slug: tokamax
commit: 3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68
scip_tool: scip-python
updated: 2026-10-02
okf_version: "0.2"
sources:
  - {resource: https://github.com/openxla/tokamax/tree/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68, title: tokamax @ 3a0a3a34a5}
---

# tokamax internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — Tokamax (openxla/tokamax, Apache-2.0) is a JAX library of custom accelerator kernels for TPU and GPU - each op (attention, splash, ragged dot/GMM, MoE dispatch, paged and latent attention, linear attention, fused cross-entropy, top-k) dispatches over Pallas Mosaic TPU, Mosaic GPU, Triton and XLA backends, with autotuning, device-time benchmarking and a PyTorch bridge through torch_tpu.

## Areas
One page per top-level area: what it is for, the units inside it with their entry points, and the small units that have no page of their own.
- [experimental](areas/experimental.md) — The PyTorch bridge (TorchOp wrappers that run tokamax TPU kernels on torch_tpu tensors) and standalone TPU tuning scripts.
- [src](areas/src.md) — The tokamax library proper: the op framework, every op and its backends (TPU kernels with their own pages, GPU-only and XLA-only ops as sections), autotuning and shared utilities.

## Concepts (deep)
Grouped by the source area each page's citations point into.

### `tokamax/_src`
| Concept | Description | Status |
|---|---|---|
| [benchmarking-and-serialization](concepts/benchmarking-and-serialization.md) | How tokamax times a kernel on TPU (standardize_function, compile_benchmark, the hermetic_xprof timer and its Disjoint Interval Union device time), and how every op call serializes its BoundArguments into the lowered HLO so get_kernel_info / get_opspecs / get_bound_args can recover tokamax ops for offline autotuning and export; read before trusting or writing a tokamax benchmark. | fresh |
| [mosaic-tpu-helpers](concepts/mosaic-tpu-helpers.md) | The shared plumbing under tokamax's Pallas Mosaic TPU kernels - custom_buffered_pallas_call (an emit_pipeline-based PrefetchScalarGridSpec call with per-input buffer depth and runtime grid sizes), quant_block_spec / ScalesTilingInfo (making qwix QArray scales addressable by a BlockSpec), poison_tpu_memory, and precision.py (canonicalize_precision, to_dot_algorithm_preset) - plus a note that pallas/block.py and pallas/grid.py are the GPU-side counterparts; read before writing or tuning a TPU kernel that uses them. | fresh |

### `tokamax/_src/ops/attention`
| Concept | Description | Status |
|---|---|---|
| [dot-product-attention](concepts/dot-product-attention.md) | How tokamax.dot_product_attention canonicalizes masks and arguments into a DotProductAttention op, how it picks between mosaic (TPU or GPU), triton, xla, xla_chunked and cudnn with NotImplementedError fallback, and how the mosaic_tpu implementation and its VJP are thin wrappers that vmap the splash attention kernels over batch; read before using, tuning or debugging attention on TPU. | fresh |
| [kernel-catalog](concepts/kernel-catalog.md) | One table of every tokamax op x backend implementation (Mosaic TPU, Mosaic GPU, Triton, CuTe DSL, XLA chunked, jax.nn, XLA reference) with its hardware gate, forward or VJP role and purpose, plus the default order each op's dispatcher tries and what TPU falls back to; read this first to find which kernel runs for an op on TPU or GPU, or to locate a kernel by name. | fresh |

### `tokamax/_src/autotuning`
| Concept | Description | Status |
|---|---|---|
| [autotuning](concepts/autotuning.md) | How tokamax finds and reuses the fastest Op Config: tokamax.autotune recovers every op's BoundArguments from the lowered HLO, the Autotuner compiles each candidate config in parallel and times them serially, AutotuningData picks the lowest median, and the result is applied as a context-manager overlay, serialized to JSON, or shipped in the per-device-kind cache under tokamax/data/autotuning; read this to tune kernels on TPU, to understand which config a call will use, or to add an autotuning search space. | fresh |

### `tokamax/_src/ops`
| Concept | Description | Status |
|---|---|---|
| [op-framework](concepts/op-framework.md) | How every tokamax kernel is packaged as an Op subclass: bind canonicalizes arguments into BoundArguments, supported_on gates the device, default_config resolves a config (explicit, autotuning cache, autotune, heuristics), the call is wrapped in jax.custom_vjp with the op serialized into HLO metadata, and the backward goes to the op's vjp; read this before writing a new kernel backend or debugging why a config or VJP was chosen. | fresh |

### `tokamax/_src/ops/causal_conv1d_gated_delta_rule`
| Concept | Description | Status |
|---|---|---|
| [gated-delta-rule](concepts/gated-delta-rule.md) | How tokamax's CausalConv1dGatedDeltaRule op runs a Qwen-style Gated DeltaNet layer (causal depthwise conv1d, SiLU, gated delta rule) for ragged serving batches in one Pallas Mosaic TPU kernel - the decode/prefill two-pass split, SMEM metadata, custom buffered refs and in-place state caches, VMEM-driven tile sizing and autotuning; read before changing the GDN kernel or tuning its tile sizes. | fresh |

### `tokamax/_src/ops/experimental/batched_rpa/kernel`
| Concept | Description | Status |
|---|---|---|
| [batched-ragged-paged-attention](concepts/batched-ragged-paged-attention.md) | How tokamax's experimental batched ragged paged attention serves mixed prefill/decode batches against a paged KV cache on TPU - the three-phase decode/prefill/mixed launch, the SMEM schedule kernel, the KV page layouts, the emit_pipeline flash-attention body, VMEM-driven block sizing and per-token quantization; read it before tuning, debugging or calling the op from JAX or PyTorch. | fresh |

### `tokamax/_src/ops/experimental/fused_moe`
| Concept | Description | Status |
|---|---|---|
| [fused-moe](concepts/fused-moe.md) | How tokamax's experimental FusedMoe op runs a whole expert-parallel MoE layer - top-k routing, a host-side routing plan, one Pallas Mosaic TPU kernel that gathers tokens, runs each local expert's FFN and pushes result rows to their owner shard by remote DMA, and a combine - plus how its VMEM budget, weight formats and prefetch schedule are chosen; read it before serving or tuning MoE layers with PallasMosaicTpuFusedMoe. | fresh |

### `tokamax/_src/ops/experimental/gmm_v2`
| Concept | Description | Status |
|---|---|---|
| [gmm-v2](concepts/gmm-v2.md) | How the experimental GMM v2 Pallas Mosaic TPU kernels compute grouped matmuls with in-kernel group metadata, bounded-slice DMAs, M-bucketing and megacore meshes; how calculate_tiling / calculate_tgmm_tiling pick tile_m/k/n from a VMEM estimate; and exactly how a K that is not a multiple of tile_k is masked; read it before tuning or debugging ragged_dot implementation="mosaic_tpu_v2". | fresh |

### `tokamax/_src/ops/experimental/kda`
| Concept | Description | Status |
|---|---|---|
| [kimi-delta-attention](concepts/kimi-delta-attention.md) | How tokamax's experimental Kimi Delta Attention op dispatches between the Pallas Mosaic TPU, xla_chunked and dense-reference backends, and how the chunked forward (Stage 1+2, Stage 3+4), the three-kernel custom backward, varlen alignment and context parallelism are built for TPU; read before touching the KDA kernels or debugging a KDA fallback. | fresh |

### `tokamax/_src/ops/experimental/mla`
| Concept | Description | Status |
|---|---|---|
| [multi-head-latent-attention](concepts/multi-head-latent-attention.md) | How tokamax's experimental DeepSeek-style MLA op runs absorbed latent attention over a paged, packed KV cache on TPU - the op and its autotuning config, the v1 manual double-buffered Pallas kernel (batched decode / decode / mixed launches, new-KV packing and cache writeback), and what v2 adds (head-major q_nope transposes, a prefill launch, q_split, tuned-parameter tables); read it before tuning or calling either kernel. | fresh |

### `tokamax/_src/ops/experimental/tpu/splash_attention`
| Concept | Description | Status |
|---|---|---|
| [splash-attention](concepts/splash-attention.md) | How tokamax's experimental TPU splash attention turns an attention mask into block-sparse MaskInfo metadata that drives a scalar-prefetch Pallas grid, how the forward flash kernel and the fused dQ/dK/dV backward kernel are tiled by SplashConfig and QKVLayout, how the tokamax op dispatches and autotunes them, and how ring attention reuses the forward kernel; read before tuning or debugging splash attention. | fresh |

### `tokamax/_src/ops/experimental/tpu/topk`
| Concept | Description | Status |
|---|---|---|
| [tpu-topk](concepts/tpu-topk.md) | How tokamax's experimental top_k op dispatches between the XLA lax.top_k reference and a Pallas SparseCore kernel that runs a digit-by-digit MSB radix select across vector subcores, and what Config knobs, shape constraints and tie semantics come with it; read before using or tuning top_k on TPU v6+. | fresh |

### `tokamax/_src/ops/linear_softmax_cross_entropy_loss`
| Concept | Description | Status |
|---|---|---|
| [linear-softmax-cross-entropy](concepts/linear-softmax-cross-entropy.md) | How tokamax's linear_softmax_cross_entropy_loss fuses the final x@W projection, log-softmax and cross-entropy so the [B, V] logits never reach HBM - the Pallas Mosaic TPU forward (online log-sum-exp over V tiles) and its software-pipelined backward, VMEM-model heuristics and autotuning, the chunked XLA alternative, and the torch_tpu wrapper; read before tuning block sizes or debugging LM-head memory. | fresh |

### `tokamax/_src/ops/ragged_dot`
| Concept | Description | Status |
|---|---|---|
| [ragged-dot-gmm](concepts/ragged-dot-gmm.md) | How tokamax.ragged_dot picks a backend, how the Pallas Mosaic TPU v1 and v2 implementations lower forward / dlhs / drhs onto gmm and tgmm kernels, how the v1 tile Config is chosen by VMEM heuristics or autotuning, and how quantized inputs and the custom VJP are handled; read it before tuning, debugging or calling ragged_dot on TPU. | fresh |

### `tokamax/_src/ops/ragged_gather_reduce`
| Concept | Description | Status |
|---|---|---|
| [ragged-gather-reduce](concepts/ragged-gather-reduce.md) | How tokamax's ragged_gather_reduce op computes out[g] = sum over a group of weighted, gathered rows (the MoE top-k combine) - the XLA reference, the dispatch and fallback rules, and the Pallas SparseCore kernel that partitions rows and columns over vector subcores, compacts valid rows, gathers with indirect BlockSpecs and merges group runs in f32; read it before using, benchmarking or debugging the op. | fresh |

### `tokamax/experimental/torch_tpu/ops`
| Concept | Description | Status |
|---|---|---|
| [torch-tpu-bridge](concepts/torch-tpu-bridge.md) | How tokamax.experimental.torch_tpu exposes JAX tokamax ops (Pallas Mosaic TPU batched ragged paged attention and linear softmax cross-entropy) as PyTorch custom ops through torch_tpu's jax_op - the TorchOp base class, config resolution, autograd wiring and torch.compile handling - plus the tune-jax scripts in experimental/utils/tuning/tpu that find tile sizes for ragged dot and splash attention; read this to call or wrap a tokamax kernel from PyTorch on TPU. | fresh |

## Doc-derived concepts
Concepts extracted from the project's own docs (README / `docs/`), grounded to the symbol catalog. The source docs stay in place.
- [autotuner-input-formats](doc-concepts/autotuner-input-formats.md)
- [autotuning-result-reuse](doc-concepts/autotuning-result-reuse.md)
- [autotuning-search-space](doc-concepts/autotuning-search-space.md)
- [benchmark-timing-methods](doc-concepts/benchmark-timing-methods.md)
- [implementation-selection-and-fallback](doc-concepts/implementation-selection-and-fallback.md)
- [kda-chunked-backward-fusion](doc-concepts/kda-chunked-backward-fusion.md)
- [kda-chunked-forward-pipeline](doc-concepts/kda-chunked-forward-pipeline.md)
- [kda-context-parallel-state-summaries](doc-concepts/kda-context-parallel-state-summaries.md)
- [kda-residuals-and-rematerialization](doc-concepts/kda-residuals-and-rematerialization.md)
- [kda-variable-length-alignment](doc-concepts/kda-variable-length-alignment.md)
- [kda-wy-triangular-solve](doc-concepts/kda-wy-triangular-solve.md)
- [stablehlo-export-and-cross-compilation](doc-concepts/stablehlo-export-and-cross-compilation.md)
- [standardized-benchmark-functions](doc-concepts/standardized-benchmark-functions.md)
- [supported-ops-and-hardware](doc-concepts/supported-ops-and-hardware.md)

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); the **symbol
index** represents the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **5738** across 333 modules
- deep (concept pages): **704** (12.3%)
- index-only: **5034**
- represented total: **5738** (100.0%)
- classes represented: **466/466**

The catalog is the **symbol index**: [`catalog/index.md`](catalog/index.md) maps every module; `catalog/symbols/*.tsv` holds one row per symbol (anchor, path, line, kind, rank, hash, callers, citing pages) and `catalog/edges/*.tsv` the caller edges — grep them by anchor, never read them whole.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
