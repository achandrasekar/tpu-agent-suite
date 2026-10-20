# CUDA construct → Pallas TPU mapping

Use this during Step 1 (triage every construct in the CUDA source) and Step 4 (implementation). "Drop" means the construct exists only because of GPU hardware and has no counterpart; do not try to emulate it.

## Contents
1. Execution model
2. Memory and data movement
3. Synchronization and cross-block communication
4. Compute
5. Control flow and indexing
6. Launch-level and host-side patterns
7. Common kernel families: how the CUDA design maps

## 1. Execution model

| CUDA | Pallas TPU | Notes |
|---|---|---|
| `__global__ void k(...)` | `def kernel(*refs)` passed to `pl.pallas_call` | Inputs, outputs, then scratch refs, in that order. With scalar prefetch: prefetch refs come first. |
| `gridDim`, `blockIdx.{x,y,z}` | `grid=(...)`, `pl.program_id(axis)`, `pl.num_programs(axis)` | Grid can have more than 3 axes. Order axes: parallel output axes first, reduction axes last. |
| `blockDim`, `threadIdx` | **Drop** | Replace "each thread handles element i" with array ops over the whole block. |
| Warp (32 threads), lane id | **Drop** | TPU "lanes" are SIMD lanes of a vector register, not threads. |
| Thread coarsening / items per thread | **Drop** | Block shape covers this. |
| Persistent kernel (grid = #SMs, loop over work) | Grid over work items, driven by a scalar-prefetched work list | The grid loop *is* the persistent loop. |
| Cooperative groups, clusters, DSM | **Adapt**: split into separate `pallas_call`s or carry state across `"arbitrary"` steps | No inter-core shared memory within a call (except megacore-specific APIs; avoid unless required). |
| Streams / concurrent kernels | **Drop** at kernel level | XLA schedules; fuse if concurrency was hiding latency. |

## 2. Memory and data movement

| CUDA | Pallas TPU | Notes |
|---|---|---|
| Global memory pointer + offset arithmetic | `BlockSpec(block_shape, index_map)` | `index_map(*grid_idx) -> block indices` (in units of blocks, not elements). |
| Coalesced loads, `float4`/`uint4` vector loads | **Drop** | DMAs move whole blocks; layout (lane-dense) is what matters. |
| `__shared__` tile | VMEM block (from BlockSpec) or `pltpu.VMEM(shape, dtype)` scratch | Scratch persists across grid steps. |
| Shared-memory padding / XOR swizzle for bank conflicts | **Drop** | No banks. |
| Registers / register tiling | **Drop** | Values in the kernel body; compiler allocates vregs. Keep intermediates modest to avoid VMEM spills. |
| `cp.async` + multi-stage pipeline | Default BlockSpec pipelining (2 buffers) or `pipeline_mode=pl.Buffered(n)` | Pallas overlaps next-block DMA with current compute automatically. |
| TMA (tensor memory accelerator) | BlockSpec DMA, or `pltpu.make_async_copy(src.at[slices], dst, sem)` | TPU DMAs are natively "bulk tensor copies". |
| Manual double buffering with `__syncthreads` handoff | `memory_space=pl.ANY` input + VMEM scratch `(2, ...)` + `SemaphoreType.DMA((2,))`, `.start()` / `.wait()` | See kernel-patterns.md §6. |
| Pointer chasing / page tables (`block_table[b][i]`) | `PrefetchScalarGridSpec(num_scalar_prefetch=k)`; use the SMEM table inside `index_map` | See kernel-patterns.md §5. |
| `__ldg`, read-only cache, L2 hints, `__restrict__` | **Drop** | |
| Constant memory (`__constant__`) | Small arrays: SMEM (scalars) or a full-array VMEM BlockSpec with constant `index_map` (fetched once) | A block whose index never changes is fetched once. |
| Texture memory / interpolation | Explicit math on VMEM blocks | |
| In-place update (same pointer in/out) | `input_output_aliases={in_idx: out_idx}` | Needed for KV-cache writes, optimizer steps, etc. Indices count scalar-prefetch operands too. |
| Unified/host memory | **Drop** | |

## 3. Synchronization and cross-block communication

| CUDA | Pallas TPU | Notes |
|---|---|---|
| `__syncthreads()` | **Drop** (single control thread) | Only synchronize DMAs: `copy.wait()`. |
| `__syncwarp`, memory fences | **Drop** | |
| `atomicAdd` to accumulate partial sums across blocks | Reduction axis in the grid, marked `"arbitrary"`, accumulating into scratch or a revisited output block | Revisited output: its block index must not change along the reduction axis, and that axis must be innermost among axes the output depends on. |
| Split-K with atomics or a fixup kernel | Option A: K as innermost `"arbitrary"` axis (no split). Option B (when M×N parallelism is too small to fill megacore): produce partials `(splits, M, N)` with `"parallel"` split axis, then reduce in XLA or a second kernel | |
| `atomicMax`/`atomicMin` | Same as above with max/min | |
| Atomic counters / work stealing / histograms | Precomputed work lists (scalar prefetch); histograms via one-hot matmul or sorting in XLA | |
| Grid-wide sync (`grid.sync()`) | Split into multiple `pallas_call`s | |
| Warp-vote (`__any_sync`, `__ballot_sync`) | `jnp.any` / `jnp.all` on the block, or scalar flags from SMEM | |

## 4. Compute

| CUDA | Pallas TPU | Notes |
|---|---|---|
| `wmma` / `mma.sync` / WGMMA / tcgen05 / CUTLASS MMA | `jnp.dot(a, b, preferred_element_type=jnp.float32)` or `lax.dot_general` | Feeds the MXU. Tiles: multiples of 128 (≤v5) / 256 (v6e+). |
| `ldmatrix`, fragment layouts, register permutations | **Drop** | |
| Warp-shuffle reductions (`__shfl_xor_sync` trees) | `jnp.sum/max/min(x, axis=..., keepdims=True)` | |
| Block-level reduction via shared memory tree | `jnp.sum(...)` on the block | |
| Warp-level prefix scan | `jnp.cumsum` on the block (verify lowering; for long axes, use a matmul with a triangular mask or a log-step `pltpu.roll` scan) | |
| `__expf`, `__fdividef`, fast-math intrinsics | `jnp.exp`, `pltpu.reciprocal(x, approx=True)` | Match the tolerance to the approximation. |
| `half2` / packed math | bf16 arrays (packed natively) | |
| FP8/INT8 dequant in registers | Dequant on the VMEM block (`x.astype(jnp.float32) * scale`) then bf16 matmul — or feed int8 to the MXU directly when both operands are int8 | |
| Tensor-core-friendly padding of dims to 8/16 | Padding to 128/256 (MXU) and (8,128) tiles | |

## 5. Control flow and indexing

| CUDA | Pallas TPU | Notes |
|---|---|---|
| `if (idx < n)` bounds check | Pad in wrapper, or mask: `row = pl.program_id(0)*bm + lax.broadcasted_iota(jnp.int32, (bm, bn), 0)`; `jnp.where(row < n, x, fill)` | Writing garbage into padded output rows is fine if the wrapper slices them off. |
| Early `return` for out-of-range blocks | `@pl.when(cond)` around the body | Point the `index_map` of skipped steps at an already-resident block to avoid wasted DMA. |
| Per-thread `for` loop | Vectorized op over the axis, or `lax.fori_loop` over chunks | |
| Loop with runtime trip count (seq_len) | `lax.fori_loop(0, n_dynamic, ...)` with `n` read from SMEM, or grid over the max and `pl.when` to skip | |
| Causal masking by `col > row` | `iota` compare + `jnp.where(mask, s, -inf)`; skip blocks entirely above the diagonal with `pl.when` | |
| Grid-stride loop | More grid steps, or larger blocks | |
| Index math with `/` and `%` | In `index_map` (scalar, cheap) or precomputed; avoid per-element div/mod in the vector body | |
| Templates / `constexpr` params | `functools.partial(kernel, **static_cfg)`; Python-level specialization | |

## 6. Launch-level and host-side patterns

| CUDA | Pallas TPU |
|---|---|
| `<<<grid, block, smem, stream>>>` | `pl.pallas_call(kernel, out_shape, grid=..., in_specs, out_specs, scratch_shapes, compiler_params)` |
| Launch bounds / occupancy / `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | `pltpu.CompilerParams(vmem_limit_bytes=...)` |
| Autotuning over configs (Triton `@autotune`) | `harness.sweep` offline; bake per-generation defaults into the wrapper |
| Heuristic dispatch between kernels by shape | Same, in the Python wrapper (shapes are static under `jit`) |
| Host-built metadata (cu_seqlens, block tables, schedules) | Build with `jnp` in the wrapper (under `jit`) and pass via scalar prefetch |
| Memory allocation / workspace buffers | Scratch shapes or extra outputs |
| Cost hints for the scheduler | `pl.pallas_call(..., cost_estimate=pl.CostEstimate(flops=..., transcendentals=..., bytes_accessed=...))` — helps XLA overlap the kernel with neighbours |

## 7. Common kernel families

**GEMM / CUTLASS-style.** Keep: tile sizes chosen for reuse, fp32 accumulation, fused epilogue (bias, activation, scaling, residual add). Drop: warp tiling, swizzles, stages count. Adapt: split-K (see §3), stream-K (work list). Grid `(M/bm, N/bn, K/bk)`, `("parallel","parallel","arbitrary")`.

**FlashAttention (v1–v3).** Keep: online softmax (running max `m`, denominator `l`, fp32 accumulator), causal block skipping, GQA head sharing, never materializing the full `S×T` matrix. Drop: warp specialization, ping-pong scheduling, TMA multicast, register-level softmax. Adapt: KV loop → innermost `"arbitrary"` grid axis (or in-kernel `fori_loop` over KV chunks if you manage KV DMA manually). For GQA, put the query heads that share a KV head into the same block so KV is fetched once.

**Paged / ragged attention (vLLM-style).** Keep: page-granular KV fetch, per-sequence lengths, the decode/prefill split idea. Adapt: block tables and lengths → scalar prefetch; per-page fetches → either `index_map` using the page table (one page per step) or manual DMA of several pages per step into a contiguous VMEM buffer (fewer grid steps; usually faster); variable lengths → work list + `pl.when` skip + masking of the tail page. Merge small pages into larger VMEM chunks to keep the MXU busy.

**LayerNorm / RMSNorm / softmax.** Keep: single pass over the row with fp32 statistics, fused scale/shift. Drop: warp/block reduction trees. One or more full rows per block when the row fits VMEM; otherwise two passes (stats, then normalize) or a column-chunked reduction axis.

**Elementwise / activation fusions.** Usually XLA already fuses these well; a Pallas kernel only wins when fused with something XLA won't fuse (a matmul epilogue, a custom reduction, a gather). Check the XLA baseline first.

**Quantized GEMM (W4A16, W8A8, FP8 block-scaled).** Keep: dequant-in-kernel, per-group scales. Adapt: unpack int4 in VMEM with shifts/masks (packed along sublanes-friendly axes), apply group scales on the smaller operand or after the partial matmul per group. Use int8×int8 MXU where both operands are int8.

**MoE grouped GEMM / routing.** Keep: grouped matmul over sorted tokens. Adapt: group offsets → scalar prefetch; tile-to-group mapping as a precomputed work list; boundary tiles masked. The JAX repo's `megablox` kernel is a reference design.

**Reductions to scalars / global statistics.** Per-block partials in a `"parallel"` grid, final reduction in XLA; or one `"arbitrary"` axis accumulating into a resident output block when total work is small.
