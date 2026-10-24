---
title: "Inferact tpu-megakernels repo snapshot (2026-10)"
type: source
tags: [codebase, kernel, compiler, pallas, megakernel, inference]
created: 2026-10-24
updated: 2026-10-24
sources: 1
human_reviewed: false
---

Provenance record for the ingest of Inferact's open-source `tpu-megakernels` repository (`https://github.com/Inferact/tpu-megakernels`), a collection of full-model Pallas TPU decode and speculative-verification megakernels targeting TPU v7 (Ironwood) and multi-host ICI slices.

## Identity

| | |
|---|---|
| Repository | https://github.com/Inferact/tpu-megakernels (public) |
| License | Apache-2.0 |
| Commit | `97dcfa12082e4cd434c016967a7e0a5253f16107` (`main` HEAD) |
| Snapshot date | 2026-10-24 |
| Authority | primary source: full-model Pallas Mosaic TPU kernel implementations, shared VMEM aliasing and multi-host remote DMA collective libraries, checkpoint packing loaders, and benchmark scripts |

## What it covers

1. **Shared Megakernel Runtime Primitives**:
   - `pool_alias.py`: Raw VMEM buffer aliasing via a custom `Reinterpret(types.ReshapeTransform)` dataclass and a lowering hook on `jax._src.pallas.mosaic.lowering._reshape_memref` that emits `#tpu.tiled<({rows},128){inner},[...]>` layouts through `tpu.reinterpret_cast` and `tpu.erase_memref_layout`, with `storage_size(shape, dtype)` and `view(ref, shape, dtype)` helpers.
   - `collectives32.py`: In-kernel TP32 hierarchical reduction (`start_local_piece`, `encode_partial`, `reduce`, `barrier`) across 4 hosts (32 TPU TensorCores) using `tpu.make_async_remote_copy` with `DeviceIdType.MESH`, intra-host 8-rank XOR-butterfly half-packet overlap (`rank ^ offset`), inter-host 4-way exchange (`rank ^ (offset * 8)`), 2-slot ping-pong buffers (`iteration % 2`), and int16 shared-exponent packet compression (`encode_partial`).
2. **Qwen3.8-27B + DFlash2 (`qwen/`)**:
   - `qwen/decode_megakernel.py` and `qwen/load.py`: Single-host TP8 full-model decode (`transformer_stack`, $B=1$) and speculative block-verify (`transformer_stack_block`, $B=8$) megakernels fusing all 64 hybrid layers (3:1 Gated Delta Net linear attention + full GQA with Q/K RMSNorm, partial RoPE, and output gate + dense SwiGLU MLP) and the LM head into one grid-less `pl.pallas_call`. Introduces the **Global Bank-Ring Weight Streaming Engine** (`BANK_COUNT=12`, `MAX_BANK_K=1024`, `MAX_BANK_N=1280`, `tile_hw`, `tile_matrix`, `schedule`, cross-layer `fetch(g)`, `gemv`), double-buffered 128-token KV cache streaming with online softmax, in-VMEM causal conv1d + recurrent state DMA (`sbuf`), XOR-butterfly AllReduce ($B=1$), FP32 ReduceScatter + BF16 AllGather ($B=8$), and in-kernel greedy argmax token exchange.
   - `qwen/dflash.py`: 8-layer DFlash2 block-diffusion speculative draft megakernel (`B=8`) with host-side channel permutation that converts 64-group depthwise dynamic convolution into a reshape-free VPU multiply-add.
3. **Kimi K3 2.8T + DSpark (`kimi/`)**:
   - `kimi/decode_megakernel.py`: Multi-host TP32 full-model decode and 8-token verify megakernel across all 93 layers (KDA linear attention + MLA latent attention + 896-expert MoE with MXFP4 and reblocked FP8 weights + attention-residual mixture over all prior layer states). Demonstrates paired-rank 512-score router exchange, `pool_alias` sharing between attention projections and 4-slot expert streaming buffers, MXFP4 in-VMEM dequantization (`tpu.bitcast(..., jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)` with `exponents << 23` power-of-two scales), and two-pass `hilo` FP32 MXU emulation for MLA probability-value products.
   - `kimi/dspark.py`: Fused DSpark low-rank Markov speculative draft megakernel with int32 bitcast candidate packing (`_pack_candidates`) to prevent TPU FP32 denormal flushing from corrupting token IDs during `all_gather`.
4. **Muse-Spark-Open 816B-A42B + DSpark (`mpk/`)**:
   - `mpk/decode_megakernel.py` and `mpk/load.py`: TP8/TP16/TP32 hybrid megakernel fusing sliding-window and global GQA (with 2-heads-per-row `[nh // 2, 128]` packing for `head_dim=64`, weightless MXU `rms_pair` via `sel128`, and split-half RoPE) with NVFP4 latent pool/unpool routed MoE (`tpu.bitcast(packed, jnp.float4_e2m1fn).astype(jnp.bfloat16) * scale`), `dense_overlay` (reusing idle NVFP4 VMEM slots for BF16 dense-layer experts), and fused router-partial + pool-shard collectives (`exchange_flat`).

## Gaps and caveats
- Benchmark figures reported in the repository README (709 tok/s on 16x TPU v7 for Kimi K3 + DSpark, 1,515 tok/s on 4x TPU v7 for Qwen3.8-27B + DFlash2) are low-concurrency ($B=1$ with 8-token speculative verification) measurements; at high concurrency ($B \ge B_{\text{crit}}$) where layers become compute-bound rather than HBM weight-bandwidth bound, standard per-op paged attention and grouped matmul kernels remain preferred.
- `pool_alias.py` patches private Mosaic lowering internals (`jax._src.pallas.mosaic.lowering._reshape_memref`); verify compatibility when upgrading JAX/libtpu versions.

## Connections
- Created: [Kernel Megakernel](../kernel/kernel-megakernel.md).
- Updated: [Kernel-Optimization Index](../kernel-optimization-index.md), `index.md`, `log.md`.
- Related: [Kernel Cross-Chip Collective](../kernel/kernel-cross-chip-collective.md), [Kernel Grouped Ragged Indirection](../kernel/kernel-grouped-ragged-indirection.md), [Kernel State Carry Scan](../kernel/kernel-state-carry-scan.md), [Pallas](../codebases/pallas.md), [TorchTPU](../codebases/torch-tpu.md).
