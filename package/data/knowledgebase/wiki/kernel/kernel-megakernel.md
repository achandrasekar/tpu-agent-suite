---
title: "Kernel class: full-model & multi-layer megakernels (bank-ring DMA, VMEM pool aliasing, in-kernel ICI collectives)"
type: concept
tags: [kernel-class, megakernel, pallas, vmem, collective, ici, rdma, moe, attention, quantization]
created: 2026-10-24
updated: 2026-10-24
sources: 1
human_reviewed: false
aliases: ["tpu-megakernel", "pallas-megakernel", "decode-megakernel"]
---

**Route here when** low-batch decode ($B=1$) or speculative block verification ($B \le 16$) is bottlenecked by HBM weight-streaming latency, per-op XLA dispatch/barrier overhead, activation HBM round-trips between consecutive layers, or exposed tensor-parallel `all_reduce` latency across a multi-layer Transformer / hybrid SSM / MoE stack ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)).

## 1. When a Megakernel Wins vs. Single-Op Kernels

In autoregressive decode ($B=1..16$), arithmetic intensity is far below the TPU MXU ridge point (~240 FLOPs/byte on BF16, ~120–160 on FP8/FP4). Splitting an $L$-layer model into $10L+$ separate XLA/Pallas ops pays four compounding penalties:
1. **Zero cross-op weight prefetch**: Layer $l+1$'s weights cannot start streaming into VMEM while layer $l$'s final projection or collective is still running.
2. **Repeated HBM activation round-trips**: Residual streams, normalized hidden states, QKV projections, and router logits bounce between VMEM and HBM at every fusion boundary.
3. **Exposed collective launch latency**: Each layer launches 2+ separate XLA `all-reduce` / `all-to-all` ops (~3–8 $\mu\text{s}$ floor per op on ICI).
4. **Per-kernel dispatch floor**: $60\text{--}93$ layers $\times$ 4–8 custom calls/layer adds hundreds of microseconds of host/scalar dispatch overhead per token.

A **TPU Megakernel** collapses the entire $L$-layer decoder stack (plus token embedding, final RMSNorm, LM head, and greedy argmax or speculative verification) into a **single grid-less `pl.pallas_call`** wrapped in `jax.jit(jax.shard_map(..., check_vma=False), donate_argnums=(1,))` ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)).

## 2. Canonical Grid-Less Pallas Megakernel Skeleton

All production TPU megakernels use a grid-less `tpu.PrefetchScalarGridSpec` (`grid=()`, `num_scalar_prefetch=1` or `2` for runtime token `pos` / `length` / `step` in SMEM) with every HBM weight, KV cache, and state tensor bound via full-shape `pl.BlockSpec(memory_space=tpu.HBM)` references so the kernel body issues explicit `tpu.make_async_copy` and `tpu.make_async_remote_copy` descriptors ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)):

```python
pl.pallas_call(
    kernel_body,
    out_shape=out_shape,
    grid_spec=tpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[pl.BlockSpec(memory_space=tpu.HBM) for _ in hbm_inputs],
        out_specs=[pl.BlockSpec(memory_space=tpu.HBM) for _ in hbm_outputs],
        scratch_shapes=scratch_shapes,
    ),
    compiler_params=tpu.CompilerParams(
        collective_id=17,
        vmem_limit_bytes=64 * 1024 * 1024,  # 64 MiB on TPU v7x (32 MiB on v6e)
        disable_bounds_checks=True,
        shape_invariant_numerics=True,
    ),
    input_output_aliases=aliases,  # in-place KV cache and recurrent state mutation
)
```

- **Startup ICI Barrier**: Before any remote DMA (`make_async_remote_copy`), synchronize all TP peers using `barrier = tpu.get_barrier_semaphore()`: signal every peer `device_id=(rank ^ offset,)` with `device_id_type=pl.DeviceIdType.MESH` for `offset in range(1, tp)` and `pl.semaphore_wait(barrier, tp - 1)` ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)).

## 3. Weight Streaming Architectures

Choose between two weight-streaming patterns based on the model's FFN structure ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)):

### Pattern A: Global Bank-Ring DMA Engine (Dense & Hybrid Linear/Full-Attention Models)
Used when every projection in the layer schedule is known statically (e.g., Qwen3.8-27B 64-layer 3:1 Gated Delta Net + GQA + SwiGLU):
1. **Host-Side Pre-Tiling (`tile_hw`, `tile_matrix`)**:
   - Every weight matrix `[..., K, N]` is reshaped and transposed at load time into contiguous tiles `[..., T, bk, bn]` where `bk <= MAX_BANK_K` (e.g., `1024`), `bn <= MAX_BANK_N` (e.g., `1280`), both multiples of 128, and `T = (K // bk) * (N // bn)` with tile index `ti = ni * nk + ki`.
2. **VMEM Ring Buffer**:
   - Allocate `banks = tpu.VMEM((BANK_COUNT, MAX_BANK_K, MAX_BANK_N), jnp.bfloat16)` (typically `BANK_COUNT = 12` or `13`, ~30 MiB in BF16) and `ws = tpu.SemaphoreType.DMA((BANK_COUNT,))`.
3. **Global Linear Tile Schedule (`fetch(g)`)**:
   - Number all weight tiles across the entire model $g \in [0, T_{\text{total}})$.
   - At kernel start, prefill the ring: `for ti in range(BANK_COUNT): fetch(ti)`.
   - Inside `gemv(base, x, y, K, N)`: loop `ti` over `nt` tiles (`lax.fori_loop`), `wait_bank((base + ti) % BANK_COUNT)`, run MXU `jnp.dot` (broadcasting $B=1$ to 8 or 16 rows so the systolic array executes native tile instructions), accumulate into `y`, and immediately call `fetch(base + ti + BANK_COUNT)`.
   - Because `fetch(base + ti + BANK_COUNT)` indexes the global schedule, the last tiles of layer $l$'s down-projection automatically prefetch layer $l+1$'s QKV weights, and the last layer prefetches the LM head—achieving continuous HBM-to-VMEM streaming across all layers with zero pipeline bubbles.

### Pattern B: Pool-Aliased Multi-Slot Streaming (MoE & Quantized Models)
Used when layers combine dense attention with sparse data-dependent routed experts (e.g., Kimi K3 896-expert MXFP4/FP8 MoE, Muse-Spark-Open NVFP4 MoE):
1. **Raw VMEM Pool Aliasing (`pool_alias.py`)**:
   - Attention projections and MoE experts never execute simultaneously within a layer. Allocate one backing `uint8` VMEM pool `pa = tpu.VMEM((pool_pages, 8, 128), jnp.uint8)` and alias typed views (`pool_alias.view(pa, shape, dtype, offset)`) for attention buffers (`qbuf`, `kvbuf`, `obuf`) and expert slot buffers (`gp`, `dp`, `gs`, `ds`).
   - In hybrid models with both dense and MoE layers (`mpk/decode_megakernel.py`), overlay BF16 dense-layer expert tiles directly onto idle NVFP4 packed-weight slots (`dense_overlay`).
2. **Multi-Slot Routed Expert Pipeline (`EXPERT_SLOTS = 2..4`)**:
   - After in-kernel router top-$k$, filter to local experts owned by this rank (`local_id = global_id - rank * experts_per_rank`) or compact shared/singleton routes.
   - Pre-issue async DMAs for the first $S$ (`EXPERT_SLOTS`) local experts before entering the expert loop.
   - On iteration $e$: wait on slot $e \bmod S$, dequantize in VMEM, execute paired `gate` + `up` MXU contractions, apply gated activation in FP32, execute `down` contraction, accumulate weighted output in FP32, and issue async DMA for expert $e + S$ into slot $e \bmod S$.

## 4. In-Kernel TPU ICI Collectives via Remote DMA

Never exit the Pallas kernel to run `lax.psum` or `all_gather`. Issue ICI transfers directly inside the megakernel using `tpu.make_async_remote_copy` ([Inferact tpu-megakernels](../sources/2026-10-inferact-tpu-megakernels.md)):

| Topology & Batch | Collective Pattern | Implementation Mechanics |
|---|---|---|
| **Single-Host Decode (`TP=8`, $B=1$)** | 2-Slot Ping-Pong XOR-Butterfly AllReduce | Alternate `slot = collective_idx % 2` in `recv_buf[2, 8, 8, H]` so peers can overwrite the next slot without a post-collective barrier. Copy local `proj` to `recv_buf[slot, rank]`, issue `make_async_remote_copy` to `device_id=(rank ^ offset,)` for `offset in 1..7`, wait on all 7 `recvs`, and `jnp.sum(recv_buf[slot], axis=0)`. |
| **Batched / Speculative Verify (`TP=8`, $B=8$)** | FP32 Reduce-Scatter + BF16 All-Gather | Partition hidden dim into `sw = H // 8` shards. Scatter each rank's FP32 shard `proj[:, tgt*sw:(tgt+1)*sw]` to peer `tgt`, sum in FP32, round once to BF16 (`rounded`), then All-Gather the BF16 shards—cutting ICI traffic in half while preserving single-rounding BF16 parity. |
| **Multi-Host Decode (`TP=16` / `TP=32`)** | Hierarchical Local-8 + Inter-Host-4 with Int16 Shared-Exponent Encoding | Split O-proj and down-proj outputs into 2 halves (`start_local_piece`). Overlap intra-host 8-rank XOR butterfly (`rank ^ offset`, `offset=1..7`) of half 0 with MXU compute of half 1. Quantize the intra-host sum into int16 + shared max exponent (`encode_partial`), exchange across hosts (`rank ^ (offset * 8)` for `offset=1..3`), dequantize, and sum (`collectives32.py`). |
| **Fused Small Payloads** | Concatenated `combuf` Exchange (`exchange_flat`) | When router logits (`[eshard]`) and latent pool vectors (`[mh_r]`) need an AllGather at the same boundary, pack both into a single `[8, width]` VMEM buffer and issue one remote DMA wave instead of two. |

## 5. Hardware-Critical TPU Megakernel Idioms & Pitfalls

1. **Broadcast $B=1$ to 8 or 16 Rows Before `jnp.dot`**:
   - The TPU MXU native tile height is 8 (and 16 for padded BF16 rows in `kimi/decode_megakernel.py`). Broadcast single-token activations `jnp.broadcast_to(x, (8, K))` (or pad to 16 rows via `_PaddedBf16Rows`) before calling `jnp.dot(..., preferred_element_type=jnp.float32)` and read row `0` from the output.
2. **Two-Heads-Per-Row Packing for `head_dim = 64`**:
   - Because VMEM tiles have a 128-element minor dimension, storing 64-dim heads one per row wastes 50% of VMEM capacity and VPU lanes. Pack pairs of heads into `[nh // 2, 128]`. Compute per-head RMSNorm via a 2-column selector matrix `sel128` (`sel128[:64, 0] = 1/64`, `sel128[64:, 1] = 1/64`) on the MXU: `ms = jnp.dot(x * x, sel128)`, and apply split-half RoPE on slices `:32, 32:64, 64:96, 96:128`.
3. **In-VMEM Low-Bit Weight Dequantization**:
   - **MXFP4 (`kimi/`)**: Store packed `uint32` weights + `uint8` E8M0 exponents. In VMEM, view as `uint8`, convert via `tpu.bitcast(u8, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)`, reconstruct scales via `jax.lax.bitcast_convert_type(exponents.astype(jnp.int32) << 23, jnp.float32)`, and contract on the FP8 MXU.
   - **NVFP4 (`mpk/`)**: Store packed `uint8` weights + `float8_e4m3fn` group scales + `float32` global scale. In VMEM, unpack via `tpu.bitcast(packed, jnp.float4_e2m1fn).astype(jnp.bfloat16) * scale`.
   - **Reblocked FP8 (`kimi/`)**: Reblock FP8 weights so scale groups align to 128-element contraction tiles; feed `float8_e4m3fn` directly to `lax.dot_general` and multiply the FP32 accumulator by the scale vector after each tile.
4. **Two-Pass `hilo` FP32 Emulation on BF16 MXU**:
   - When attention probability-value (`P @ V`) contractions in long-context MLA require higher than BF16 input precision, split FP32 `P` into `p_hi = P.astype(jnp.bfloat16)` and `p_lo = (P - p_hi.astype(jnp.float32)).astype(jnp.bfloat16)` and sum two MXU passes (`p_hi @ V + p_lo @ V`).
5. **Never Bitcast `int32` Token IDs to `float32` for Collectives**:
   - TPUs flush FP32 subnormal/denormal numbers to zero! Small integer token IDs ($1, 2, \dots$) bitcast to `float32` are positive denormals and become `0` after an XLA/Pallas collective. Always bitcast `float32` scores to `int32` (`lax.bitcast_convert_type(scores, jnp.int32)`) and `all_gather` an `int32` tensor (`kimi/dspark.py`).

## 6. Integration into `torch_tpu` and `vllm-torchtpu` Serving

To serve a Pallas megakernel in `vllm-torchtpu`:
1. **Analyze & Scaffold**: Run `scripts/tpu-kernel-development/megakernel_analyzer.py --config <config.json> --tpu-version <v6e|v7x> --tp-size <tp> --emit-scaffold <dir>` to compute the per-rank tile schedule, verify the VMEM budget (`banks` + `pool_alias` + collective scratch $\le$ `vmem_limit_bytes`), and emit the megakernel + `tpu_agent_suite_kernels` plugin op module.
2. **Bridge via `tpu_agent_suite_kernels`**:
   - Register the megakernel through `bridge.make_op` (`torch_tpu`'s `jax_op`) with `OWNED_BUCKETS` set to the decode and speculative-verify token buckets (e.g., `(1, 8, 16)`), falling back to `fallback_fn` on prefill buckets ($\ge 64$).
   - Pre-pack and tile checkpoint weights (`tile_matrix`) once in `process_weights_after_loading` (or on first eligible module initialization) so decode steps incur zero host or HBM relayout overhead.
3. **Verify & Package**:
   - Build the plugin `.tgz` with `scripts/tpu-kernel-integrator/build_plugin.py` and verify op parity + `torch.compile(fullgraph=True)` speedup on TPU with `scripts/tpu-kernel-integrator/inplace_harness.py` before launching `vllm serve` with `TPU_AGENT_SUITE_KERNELS=<op_name>`.

## See also
- [Kernel-Optimization Index](../kernel-optimization-index.md) — routes full-model and multi-layer decode fusion here.
- [Kernel Cross-Chip Collective](kernel-cross-chip-collective.md) — single-op ICI RDMA overlap patterns.
- [Kernel Grouped Ragged Indirection](kernel-grouped-ragged-indirection.md) — MoE and paged KV indirection mechanics.
- [Kernel State Carry Scan](kernel-state-carry-scan.md) — recurrent state-carry scans for SSM and linear attention.
- [VMEM](../concepts/vmem.md) — TPU VMEM scratchpad capacity and bandwidth hierarchy.

## Sources
- [Inferact tpu-megakernels repo snapshot (2026-10)](../sources/2026-10-inferact-tpu-megakernels.md)
