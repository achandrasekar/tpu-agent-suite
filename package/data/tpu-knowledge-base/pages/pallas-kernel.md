---
title: "Pallas Kernel"
type: concept
tags: [tpu-kernel-development, compiler, kernels]
created: 2026-04-22
updated: 2026-10-01
sources: 2
human_reviewed: false
---

User-authored custom kernel written in Pallas, JAX's kernel DSL; compiles to Mosaic (TPU) or Triton (GPU).

How to write one is in the Pallas hub (`index.py get codebase-pallas`), which lists
the official Pallas docs and the authored kernel pages in this index.

## See also

- Pallas hub (`index.py get codebase-pallas`)
- Attention kernels (`index.py get kernel-class-attention`)
- FlashAttention-2 paper (`index.py get flash-attention`)
- Splash attention, and its block sizes (`index.py get splash-attention`)
- Profiling a kernel as an XLA custom call (`index.py get custom-call`)

## Sources

- XProf custom call profiling (`index.py get custom-call-profiling`)
- Tokamax (`index.py get tokamax`)

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
- `tokamax`: `tokamax._src.ops.attention.pallas_mosaic_gpu_kernel_sm100` — Blackwell flash attention, collective 2-CTA MMA, split TMA loads
- `tokamax`: `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp_kernel_sm100` — flash-attention backward, split dQ/dKV configs, residual-in-regs option
- `tokamax`: `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp_kernel_sm90` — Hopper flash-attention backward, compute_wgs warpgroup fan-out
- `tokamax`: `tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_kernel` — MLA ragged+paged attention, MlaCase dispatch
- `tokamax`: `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2_gmm_kernel` — GMM v2, MXU-column-size-gated dequantization timing
- `tokamax`: `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2_tgmm_kernel` — TGMM (lhs.T @ dout) tiling, VMEM-budget-aware, XLU-transpose-cache-aware
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
