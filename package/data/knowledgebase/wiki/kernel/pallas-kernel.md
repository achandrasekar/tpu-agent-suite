---
title: "Pallas Kernel"
type: concept
tags: [stub, compiler, kernels]
created: 2026-04-22
updated: 2026-10-13
sources: 3
human_reviewed: false
---

User-authored custom kernel written in Pallas, JAX's kernel DSL; compiles to Mosaic (TPU) or Triton (GPU).

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Pallas hub](../codebases/pallas.md) — the Mosaic backend a Pallas kernel compiles to on TPU
- [Pallas validation and XProf](pallas-validation-and-xprof.md) — the XLA custom call boundary: a kernel shows up as an opaque `tpu_custom_call`
- [Splash attention](../codebases/tokamax/concepts/splash-attention.md) — block-sparse flash attention, and its block sizes (`SplashConfig`: `block_q`, `block_kv`, `block_kv_compute`, and the `_dkv` backward set)
- [dot_product_attention](../codebases/tokamax/concepts/dot-product-attention.md) — tokamax's dense Mosaic TPU flash attention

## Sources

- xprof Custom Call Profiling — `raw/code/xprof/docs/custom_call_profiling.md`
- tokamax (openxla/tokamax @ `3a0a3a34`), ingested as the [tokamax code wiki](../codebases/tokamax/overview.md); its `docs/basic_usage.md` and `docs/splash_attention.md` are covered there by [implementation selection and fallback](../codebases/tokamax/doc-concepts/implementation-selection-and-fallback.md) and [splash attention](../codebases/tokamax/concepts/splash-attention.md)

## In this wiki's repos
Grounded implementations of **pallas-kernel** across the ingested repos:

- `ejkernel`: `ejkernel/kernels/_pallas/tpu/blocksparse_attention/_info` — the sparse MaskInfo the Splash kernel prefetches
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel` — the Splash attention Pallas kernel (fwd + dq/dkv bwd)
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/blocksparse_attention/_masks` — the composable lazy mask algebra
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/flash_attention/_utils` — BlockSizes, the flash-attention TPU tiling with major/minor validation
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl` — the MoE grouped matmul with fused activation
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_pallas_impl_fwd` — MLA paged attention with explicit async pipelining
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core` — bit-unpacking, dequant, and predecode caching for quantized matmul
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd` — mixed prefill/decode paged attention kernel
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils` — precomputed tuned block sizes + TPU packing helpers
- `ejkernel`: `ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd` — sequence-parallel ring attention over Splash
- `tokamax`: [kernel catalog](../codebases/tokamax/concepts/kernel-catalog.md) — every tokamax op x backend (Mosaic TPU, Mosaic GPU sm80/sm90/sm100, Triton, CuTe DSL, XLA) with hardware gate and purpose; the Mosaic GPU flash-attention fwd/bwd kernels are listed there
- `tokamax`: [splash attention](../codebases/tokamax/concepts/splash-attention.md) — block-sparse flash attention: MaskInfo scalar prefetch, fused dKV backward, ring attention
- `tokamax`: [dot_product_attention](../codebases/tokamax/concepts/dot-product-attention.md) — the Mosaic TPU flash attention, a vmap over the splash kernels
- `tokamax`: [ragged dot / GMM](../codebases/tokamax/concepts/ragged-dot-gmm.md) — gmm/tgmm Pallas TPU kernels behind ragged_dot fwd, dlhs, drhs; VMEM-heuristic tiling
- `tokamax`: [GMM v2](../codebases/tokamax/concepts/gmm-v2.md) — dynamic-DMA grouped matmul, VMEM-aware tiling, K-tail masking (at this pin under `ops/experimental/gmm_v2`)
- `tokamax`: [multi-head latent attention](../codebases/tokamax/concepts/multi-head-latent-attention.md) — MLA ragged+paged attention, MlaCase dispatch, v2 kernel
- `jax`: `jax._src.pallas.fuser.block_spec` — backward BlockSpec propagation for kernel fusion
- `jax`: `jax._src.pallas.mosaic.lowering` — jaxpr-to-Mosaic lowering via per-(kernel_type, primitive) rules
- `jax`: `jax._src.pallas.mosaic_gpu.core` — MemorySpace hierarchy and the kernel() entry point
- `easydel`: `easydel/caching/ragged_page/cache` — paged KV cache with a TPU Pallas update kernel
- `maxtext`: `MegaBlox GMM v2` — pipelined grouped-matmul kernel for MoE
- `pallas-forge`: `pallas_forge/tune/runner` — warmup-disciplined kernel benchmarking
- `recurrentgemma`: `recurrentgemma.jax.pallas` — the TPU Pallas kernel for the RG-LRU scan
- `ringattention`: `Ring Flash Attention` — TPU Pallas kernel
- `sglang-jax`: `sgl_jax.srt.kernels.fused_moe.v1.kernel` — fused expert-parallel MoE Pallas kernel with inline all-to-all
- `simply`: `simply.kernels.ragged_paged_attention` — the Pallas double-buffered flash-attention-over-pages kernel
- `tpu-inference`: `tpu_inference.kernels.mla.v2.kernel` — the MLA (Multi-head Latent Attention) Pallas kernel
