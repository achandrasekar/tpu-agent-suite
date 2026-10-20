---
slug: tokamax
repo: https://github.com/openxla/tokamax
# Pin (main HEAD, 2026-10-02; the repo has no release tags); also recorded in
# wiki/codebases/tokamax/index.md `commit:`. `prepare`, `finalize` and `verify` all
# re-acquire this ref, so when ingesting from a developer working copy comment it out until
# the run is finished, then restore it.
ref: 3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68
acquire: clone
wiki_subdir: codebases
source_type: code
# Catalog tier: the symbol index (catalog/symbols.tsv, edges.tsv, index.md)
# instead of one page per module; `full` profile keeps signatures and doc lines in the rows.
catalog: index
index_profile: full
index_shard_depth: 0
# Pure Python (JAX + Pallas), Apache-2.0: no bazel_targets and no compile DB.
languages: [python]
# Tests sit next to the code (`foo_test.py` beside `foo.py`), not under a tests/ tree, so
# wikify's default excludes don't catch them; classify them here.
tests: ["**/*_test.py"]
docs: ["README.md", "docs/**/*.md"]
# Citations, OKF `sources` and change pages link GitHub at the pinned commit; the local
# checkout (raw/code/tokamax) is gitignored and excluded from the dist.
source_url: "https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68"
# Tests and benchmarks keep a citeable catalog row but drop the member body.
coverage_collapse: ["**/*_test.py", "tokamax/_src/test_utils.py", "tokamax/benchmarks/**", "tokamax/conftest.py"]
synthesis_focus: >-
  TPU kernels: how each op dispatches and lowers on TPU, the Pallas Mosaic TPU kernel design
  (grid, BlockSpecs, tiling and VMEM budgeting, megacore), the custom VJP / backward
  kernels, autotuning and benchmarking, and using the kernels from PyTorch via torch_tpu.
  GPU backends (Mosaic GPU sm90/sm100, Triton) are context only: name them where dispatch
  chooses between backends, never explain their kernels.
agenda: subsystems
# Every page is pinned under ## Concepts, so the planner proposes no deep pages of its own:
# a ceiling of 1 deep unit (the top-ranked one, which a pinned concept claims anyway)
# leaves every other unplanned unit as a section of its area page. That keeps the
# `tokamax/_src` root grab bag, the GPU-only ops (normalization, gated_linear_unit) and the
# small ops (ragged_gather/scatter, flex_attention, triangle_multiplication, fused_moe_rs,
# pallas) on the area page without mechanism pages. Not `agenda_exclude: [tokamax/_src]`:
# wikify 0.4.0 matches an exclude glob against `<glob>/*` as well, so it would drop every
# unit under `_src` and orphan the `_src` area page.
agenda_max: 1
---

# tokamax - knowledge base ingest config

Tokamax (openxla/tokamax, Apache-2.0), the JAX GPU and TPU kernel library, ingested as the
grounded code silo behind [`wiki/codebases/tokamax.md`](../wiki/codebases/tokamax.md).

Each op under `tokamax/_src/ops/<op>/` follows one layout: `api.py` (public entry and
implementation dispatch), `base.py` (the reference XLA implementation), `arg_specs.py`
(benchmark/autotune argument specs), and one module per backend (`pallas_mosaic_tpu*`,
`pallas_mosaic_gpu*`, `pallas_triton*`, `xla_chunked`, `jax_nn`). Concept pages follow the
TPU path. `kernel-catalog` lists every op and every backend implementation - TPU, GPU
(Mosaic GPU, Triton, CuTe DSL) and XLA - with its purpose, so GPU-only kernels are findable
by name without a mechanism page of their own.

## Concepts
<!-- Discovery-driven: the subsystem planner proposes the units; the entries below pin, rename and add. -->
- **kernel-catalog** - seeds: `PallasMosaicTpuFlashAttention`, `PallasMosaicTpuFlashAttentionVjp`, `PallasMosaicGpuFlashAttention`, `PallasMosaicGpuFlashAttentionVjp`, `PallasTritonFlashAttention`, `PallasTritonFlashAttentionVjp`, `XlaChunkedDotProductAttention`, `JaxNnDotProductAttention`, `PallasMosaicTpuSplashAttention`, `PallasMosaicTpuSplashAttentionVjp`, `WrappedFlexAttention`, `PallasMosaicTpuRaggedDot`, `PallasMosaicTpuV2RaggedDot`, `PallasMosaicGpuRaggedDot`, `TritonRaggedDot`, `PallasTpuRaggedGather`, `PallasTpuRaggedGatherReduce`, `PallasTpuRaggedScatter`, `PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`, `PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp`, `ChunkedXlaLinearSoftmaxCrossEntropyLoss`, `ChunkedXlaLinearSoftmaxCrossEntropyLossVjp`, `PallasTritonNormalization`, `PallasTritonNormalizationVjp`, `PallasMosaicGpuGatedLinearUnit`, `TritonGatedLinearUnit`, `CuteDslGatedLinearUnit`, `PallasMosaicTpuCausalConv1dGatedDeltaRule`, `PallasTpuBatchedRpa`, `PallasMosaicTpuFusedMoe`, `PallasMosaicTpuKimiDeltaAttention`, `PallasMosaicTpuKimiDeltaAttentionVjp`, `XlaChunkedKimiDeltaAttention`, `XlaChunkedKimiDeltaAttentionVjp`, `PallasTpuMultiHeadLatentAttention`, `PallasTpuTopK`, `TriangleMultiplication`, `gmm_v2`, `tgmm_v2`
- **op-framework** - seeds: (subsystem: tokamax/_src/ops/op.py)
- **autotuning** - seeds: (subsystem: tokamax/_src/autotuning)
- **benchmarking-and-serialization** - seeds: `standardize_function`, `compile_benchmark`, `benchmark`, `BenchmarkData`, `XprofProfileSession`, `xprof_timer`, `hermetic_xprof_timer`, `wallclock_timer`, `register_benchmark`, `get_kernel_info`, `get_opspecs`, `get_nested_opspecs`, `get_bound_args`
- **mosaic-tpu-helpers** - seeds: `custom_buffered_pallas_call`, `quant_block_spec`, `ScalesTilingInfo`, `poison_tpu_memory`, `pallas_call`, `get_cheapest_grid_grouping`, `canonicalize_precision`, `to_dot_algorithm_preset`
- **dot-product-attention** - seeds: (subsystem: tokamax/_src/ops/attention)
- **splash-attention** - seeds: (subsystem: tokamax/_src/ops/experimental/tpu/splash_attention)
- **tpu-topk** - seeds: (subsystem: tokamax/_src/ops/experimental/tpu/topk)
- **ragged-dot-gmm** - seeds: (subsystem: tokamax/_src/ops/ragged_dot)
- **gmm-v2** - seeds: (subsystem: tokamax/_src/ops/experimental/gmm_v2)
- **fused-moe** - seeds: (subsystem: tokamax/_src/ops/experimental/fused_moe)
- **ragged-gather-reduce** - seeds: (subsystem: tokamax/_src/ops/ragged_gather_reduce)
- **batched-ragged-paged-attention** - seeds: (subsystem: tokamax/_src/ops/experimental/batched_rpa)
- **multi-head-latent-attention** - seeds: (subsystem: tokamax/_src/ops/experimental/mla)
- **kimi-delta-attention** - seeds: (subsystem: tokamax/_src/ops/experimental/kda)
- **gated-delta-rule** - seeds: (subsystem: tokamax/_src/ops/causal_conv1d_gated_delta_rule)
- **linear-softmax-cross-entropy** - seeds: (subsystem: tokamax/_src/ops/linear_softmax_cross_entropy_loss)
- **torch-tpu-bridge** - seeds: (subsystem: tokamax/experimental)
