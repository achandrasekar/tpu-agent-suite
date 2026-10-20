---
title: "Chapter 5: Data Management on TPU v6e"
type: concept
tags: [kernel, pallas, v6e, data-management, double-buffering]
created: 2026-07-16
updated: 2026-10-01
sources: 0
human_reviewed: false
---

# Chapter 5: Data Management on TPU v6e

Efficient data management is the cornerstone of high-performance kernel
authoring in JAX Pallas for Trillium (TPU v6e). Because the ALU computation
rates vastly outpace memory bandwidth, hiding latency through intelligent DMA
scheduling, maximizing descriptor efficiency, and avoiding unnecessary HBM
materialization are critical to achieving peak performance.

## Manual Pipeline Double Buffering via `make_async_copy`

While Pallas provides transparent double buffering via
`pipeline_mode=pl.Buffered(buffer_count=2)`, some advanced use cases—such as
dynamically paged KV cache generation—require explicit control over asynchronous
fetches.

**Advanced Manual Double Buffering:** In highly optimized memory-bound kernels
(like Paged Attention), memory latency can be completely masked by using
`pltpu.make_async_copy`. By scheduling the DMA fetch command for the *next*
block of data (e.g., `async_copy_next.start()`) exactly when the innermost
compute operations for the *current* block begin, the TPU execution entirely
hides High-Bandwidth Memory (HBM) latency.

**Compiler Context on Buffering:** It is important to note that the Pallas MLIR
compiler actively rejects triple buffering (e.g., `buffer_count=3`), throwing a
`LoweringException`. Because TPU DMA transfers are highly deterministic and
symmetric, double buffering is physically sufficient to achieve 100% overlap
between DMA-in, compute, and DMA-out operations. Furthermore, pipeline double
buffering does not logically multiply the scoped Vector Memory (VMEM) limit; in
the measured configuration a 32 MB sum of block sizes executed successfully even
with `buffer_count=2`. The scoped limit is a compiler default
(`--xla_tpu_scoped_vmem_limit_kib`), not physical capacity, which is 64-128 MiB
per core on v5e to TPU7x.

### TPU Pipeline Stage Depth Constraints (AX-088)

***(Reference: exp084_tpu_pipeline_stage_depth__ax_0.py)***

Unlike GPUs, where triple-buffering is often employed to hide memory latency,
TPU DMA latencies are extremely short compared to vector computation times.
Consequently, double-buffering (`buffer_count=2`) is mathematically sufficient
to completely hide HBM-to-VMEM latency.

Attempting to configure more than two pipeline stages (e.g.,
`pl.Buffered(buffer_count=3)`) will trigger a hard compiler error:

```
LoweringException: Only single (1) and double (2) buffering are supported
```

Because TPU DMA transfers are highly deterministic and symmetric, double
buffering is physically sufficient to achieve 100% overlap between DMA-in,
compute, and DMA-out operations. Furthermore, pipeline double buffering does not
logically multiply the scoped Vector Memory (VMEM) limit; in the measured
configuration a 32 MB sum of block sizes executed successfully even with
`buffer_count=2`.

#### Empirical Performance Analysis

Microbenchmarks on TPU v6e hardware confirm the precise latency-hiding
characteristics of these configurations:

*   **Pure Memory Copy Workloads:** Single-buffering a pure memory copy takes
    roughly twice the time of double-buffering (e.g., 154.2 μs vs 81.4 μs),
    demonstrating perfectly balanced latency hiding when transitioning to
    `buffer_count=2`.
*   **Compute-Bound Workloads:** The difference between single and double
    buffering is minimal (e.g., 7,996 μs for single vs 7,877 μs for double)
    because the DMA transfer time is already fully hidden by the dominant
    compute execution.

## Avoiding Double-Buffering for Scalar Hyperparameters (AX-159)

***(Reference: exp038_avoid_double_buffering_scalar_.py)***

When designing complex pipelined kernels, it is common to pass dynamically
computed hyperparameter vectors (e.g., a vector of shape `(7,)` containing
dynamic scaling factors, thresholds, or block offsets) alongside massive data
tensors. However, applying pipeline double-buffering—such as
`pl.Buffered(buffer_count=2)`—to these small hyperparameter inputs is a severe
performance anti-pattern.

### The DMA Fragmentation Constraint

If a small hyperparameter vector is mapped with a static broadcast index map
(e.g., `index_map=lambda i, j: (0,)`) and configured with `buffer_count=2`, the
compiler attempts to schedule asynchronous double-buffered DMA transfers for it.
Because these transfers compete directly with the massive HBM-to-VMEM
dual-buffers of the primary data streams, they cause severe **DMA pipeline
fragmentation** and descriptor contention. The overhead of managing these tiny,
redundant asynchronous transfers completely eclipses any theoretical
latency-hiding benefits.

### Resolution: SMEM Storage and Scalar Loading

To resolve this, small hyperparameter vectors should be stored directly in
Shared Memory (`memory_space=pltpu.SMEM`) without any explicit pipeline
double-buffering. SMEM loads are extremely low latency and do not require
asynchronous double-buffering. By keeping these parameters in SMEM, you
eliminate descriptor contention and preserve the DMA engine's full bandwidth for
the primary VMEM data streams.

#### The SMEM Scalar Load Constraint

When utilizing SMEM for hyperparameter storage, developers must adhere to a
strict hardware access constraint: **Pallas only supports scalar loads from the
SMEM memory space**.

Unlike VMEM references, you cannot index an SMEM array using slices or load an
entire block at once (e.g., `h_ref[...]` or `h_ref[:]`). Attempting to do so
will trigger a compile-time or runtime validation error:

```
ValueError: Can only load scalars from SMEM
```

Instead, arrays stored in SMEM must be loaded into local registers
element-by-element as scalars (e.g., looping or explicitly indexing `h_ref[0]`,
`h_ref[1]`, etc.) before they can be used in vector computations.

## The Dangers of Micro-fetch Starvation

The TPU DMA engine relies heavily on descriptor efficiency. Breaking up a memory
load into many small fetches (e.g., 64 distinct 64 KB fetches for a 4 MB
parameter load) generates a massive amount of command-dispatch overhead. This
leads to **DMA command-queue starvation**, where bandwidth utilization falls
drastically because the engine spends more time processing descriptors than
transferring data.

**Best Practice:** To maximize throughput, load the maximal contiguous array
bounds allowed by the kernel's scoped VMEM limit. Amortizing this descriptor overhead
allows the memory pipeline to run at maximum capability. Consolidating fetches
can yield multi-fold speedups compared to processing tiny, fragmented slices.

## SparseCore Overlapping Fallacies

A common hardware hallucination is the assumption that SparseCore can be
utilized for indirect streaming of High-Bandwidth Memory within Pallas kernels.
This is false.

**The Reality:** Pallas explicitly restricts direct HBM loads within kernels
(raising a `ValueError` for loads not on VMEM or SMEM references). In complex
operations like PagedAttention, native Low-Level Optimizer (LLO) MLIR dumps
reveal that dense cache blocks are fetched completely via VMEM DMA streams
(`tpu.enqueue_dma` / `tpu.wait_dma`), entirely bypassing the SparseCore
subsystem. The architectural performance gains originate strictly from the
asynchronous overlapping of HBM to VMEM DMA requests, not from SparseCore
offloading.

## HBM Materialization Limits

XLA is strict regarding the physical materialization of tensors expanded on the
host side. Using functions like `jnp.broadcast_to` on the host prior to a
`pallas_call` forces the compiler to physically duplicate the data across HBM,
which can easily double memory reads and destroy bandwidth.

**Best Practice:** For custom Pallas kernels, pass minimally-sized tensors as
inputs and rely strictly on internal slice indexing (e.g., `ref[...][None, :]`)
to perform zero-cost, in-kernel broadcasting via VPU registers. Ensure you avoid
host-side broadcasting.

## Sequential JAX Compute and Pallas Calls (AX-026)

***(Reference: exp113_sequential_jax_compute_and_pal.py)***

Placing native JAX operations (like `jax.lax.dot_general`) and an internal
`pl.pallas_call` sequentially inside a single `@custom_jax_kernel` wrapper does
not result in a fused macro-kernel.

### The Opaque Compiler Barrier

The `pl.pallas_call` acts as an opaque compiler barrier within the StableHLO
graph. Because the XLA compiler has zero visibility into the Pallas MLIR
payload, it cannot perform cross-boundary optimizations or fuse the native JAX
operations with the custom Pallas kernel. This forces XLA to physically
materialize all intermediate tensors (e.g., the outputs of the JAX dense
projections) into High-Bandwidth Memory (HBM) before the Pallas block can fetch
them, destroying memory bandwidth.

### Empirical Verification on TPU v6e

This behavior has been verified empirically on TPU v6e (Trillium) via XLA
compiler dump analysis. Intermediate buffers outputted by JAX dense projections
are assigned to strictly allocated HBM regions
(`after_optimizations-buffer-assignment.txt`) which the opaque `custom-call`
(Pallas) reads, explicitly demonstrating a lack of operation fusion.

### Resolution and Best Practices

To achieve true macro-block fusion and avoid the HBM materialization penalty,
you must use one of the following strategies:

1.  **In-Kernel Computation:** Move the dense projections or elementwise
    operations natively inside the Pallas block using Pallas-native operators
    like `pl.dot` or VPU-level vector instructions.
2.  **Aggressive Hardware Pipelining:** If the operations must remain separate,
    apply aggressive hardware pipelining (e.g., `pl.Buffered` or custom
    `pl.BlockSpec` configurations) to overlap and mask the HBM round-trip
    latencies.

### Element-wise Pallas Kernels and HBM Materialization Overhead (AX-091)

***(Reference: exp114_pallas_element_wise_hbm_materi.py)***

A particularly severe manifestation of this opaque compiler barrier occurs when
attempting to chain a native `jax.lax.dot_general` call with a custom Pallas
element-wise kernel (e.g., an activation function or scale-and-bias operation)
inside a single `@custom_jax_kernel` wrapper.

#### The Fusion Failure and Memory Bandwidth Collapse

Because XLA cannot compile across the Pallas boundary, it is forced to
physically materialize the massive intermediate output tensor of the matrix
multiplication into HBM before invoking the Pallas element-wise kernel. This
destroys memory bandwidth, often degrading performance to less than 1% of peak
memory bandwidth (Mem BW < 1%).

Unlike pure JAX pipelines where element-wise operations are aggressively fused
into the preceding MatMul's epilogue, the presence of the Pallas custom call
acts as an absolute barrier. Tuning Pallas block sizes or grid configurations
will not bypass this performance wall.

#### Comparison and Resolution

To beat native `torch.compile` fusion (which easily fuses element-wise epilogues
with matrix multiplications), you must avoid this sequential pattern entirely.
You must either:

1.  Pull the MatMul computations completely *inside* the Pallas kernel using
    `pl.dot` so that the element-wise operations can be applied directly to the
    registers/VMEM buffers before writing to HBM.
2.  Utilize Advanced Async Pipelining or SparseCore offloading to hide the
    memory latency of the intermediate HBM round-trip.

## Fused GEMM-Activation-GEMM (GLU) and the TPU Fusion Fallacy

***(Reference: exp019_pallas_on_triton_fused_gemm_ac.py)***

A common pattern in modern transformer architectures is the Gated Linear Unit
(GLU) and its variants (SwiGLU, GEGLU, REGLU), mathematically defined as:

$$\text{out} = \text{activation}(X W_{\text{gate}}) \odot (X W_{\text{up}})$$

A naive implementation of this block requires four distinct HBM passes: loading
$X$ and $W_{\text{gate}}$ to compute the gate matrix multiplication, writing the
intermediate pre-activation to HBM, reading it back to apply the activation
function, loading $X$ and $W_{\text{up}}$ to compute the up-projection, and
finally reading both intermediates back to perform the element-wise
multiplication. This results in eight distinct HBM transactions for inputs,
weights, and intermediate states.

### The GPU-Side Fused Kernel Pattern

On GPU architectures, writing a custom Pallas-on-Triton kernel to fuse this
entire sequence is a highly effective, production-grade optimization pattern
(e.g., DeepMind's AlphaFold3 `PallasGatedLinearUnit`). The fused GPU kernel
operates in a single pass:

1.  Load tiles of $X$, $W_{\text{gate}}$, and $W_{\text{up}}$ simultaneously.
2.  Compute the gate projection tile: $\text{gate\_preact} = X_{\text{tile}}
    \otimes W_{\text{gate\_tile}}$.
3.  Apply the activation function (e.g., SiLU, GELU) directly in registers:
    $\text{gate} = \text{activation}(\text{gate\_preact})$.
4.  Compute the up-projection tile: $\text{up} = X_{\text{tile}} \otimes
    W_{\text{up\_tile}}$.
5.  Perform element-wise multiplication: $\text{out\_tile} = \text{gate} \odot
    \text{up}$.
6.  Write the final $\text{out\_tile}$ back to HBM, optionally using destination
    aliasing to avoid allocating an extra HBM buffer.

This approach saves two complete HBM round-trips on the gate and up
intermediates, yielding performance gains directly proportional to the reduction
in HBM bandwidth traffic.

### The TPU Fusion Fallacy: Why Custom GLU Kernels Regress on TPU

While the fused GEMM-Activation-GEMM pattern is highly successful on GPUs,
attempting to implement a custom Pallas GLU kernel on TPU v6e is a performance
anti-pattern. Empirical HLO and XPlane profiling confirms that the XLA compiler
already handles this structure optimally, rendering custom Pallas
implementations strictly unneeded and highly likely to regress.

#### 1. Perfect Epilogue Fusion

Native StableHLO dumps reveal that the XLA compiler automatically and perfectly
fuses the activation function (e.g., `silu`) and the subsequent element-wise
multiplication directly into the epilogue of the second matrix multiplication
(`multiply_multiply_fusion`). XLA does not perform four HBM passes; it executes
exactly two highly optimized dot passes, completely avoiding the final
write/read cycle of the second intermediate tensor.

#### 2. Compute-Bound Efficiency

For large, standard hidden dimensions (e.g., $M=4096, K=4096, N=4096$), the
execution is heavily compute-bound. The TPU's Matrix Multiply Unit (MXU)
completely hides the HBM read/write latency of the first intermediate
pre-activation tensor. Profiling shows that two sequential native JAX/XLA matrix
multiplications complete in approximately **329 µs**, whereas a custom Pallas
fused concatenated matrix multiplication approach takes **410 µs** due to
slicing overheads and register pressure penalties.

#### 3. Memory-Bound Bottleneck Realities

For small sequence lengths (e.g., $M=128$), the layer becomes memory-bound.
However, the memory traffic is overwhelmingly dominated by loading the massive
weight matrices ($W_{\text{gate}}$ and $W_{\text{up}}$ consume ~32 MB for
$K=4096, N=4096$), while the intermediate activation tensors consume less than 1
MB. Eliminating the intermediate HBM round-trip mathematically yields less than
a **3% reduction** in total HBM traffic, which is easily wiped out by the loss
of XLA's highly tuned MXU scheduling.

Consequently, dispatching to native XLA on TPU (while utilizing Pallas-on-Triton
exclusively on GPU) is the optimal architectural strategy.

## The SwiGLU Graph Fracturing and Layout Mismatch Trap (AX-076)

***(Reference: exp026_final_hypothesis_c_plateau__ep.py)***

While single, isolated dense MLPs might occasionally benefit from PyTorch eager
dispatch interleaved with isolated Pallas element-wise kernels (due to 100%
memory bandwidth utilization despite trace scheduling overhead), this approach
is highly detrimental for larger macro-blocks.

### The Danger of Fractured XLA Graphs

Breaking larger macro-blocks (such as SwiGLU) into eager dispatch combined with
isolated Pallas kernels completely destroys HBM fusion, leading to severe
performance regressions against highly optimized compilers like `torch.compile`.

When a SwiGLU macro-block is broken by a Pallas kernel, the XLA graph is
fractured. This forces the outputs of the linear projections (e.g., `F.linear`)
to be physically materialized to HBM. This destroys XLA's native operation
fusion, which would otherwise keep intermediate activations (such as `SiLU` or
element-wise multiplication) entirely within VMEM or VPU registers.

### Opaque Physical Layouts and Crash Risks

Furthermore, the XLA compiler aggressively tiles and transposes intermediate
matrices in physical memory to maximize TPU hardware efficiency. Intercepting
these intermediate buffers with a custom Pallas kernel natively exposes these
opaque physical layouts. Because Pallas expects standard contiguous layouts,
this mismatch often leads to:

1.  **Layout Mismatch Crashes:** The compiler or runtime throws layout-related
    exceptions when the physical memory representation does not match the
    logical representation expected by the Pallas kernel.
2.  **Expensive Contiguous Copies:** To avoid crashes, the compiler may insert
    implicit, highly expensive contiguous memory copies to layout-normalize the
    tensors, completely wiping out any theoretical performance gains from the
    custom kernel.

## Avoiding 1D Memory Fragmentation

When configuring the Pallas grid, avoid artificial 1D grid consolidations of 2D
or multi-dimensional tensors if they require dynamic index mapping (e.g.,
div/mod operations like `i // blocks, i % blocks`).

**The Trap:** Dynamic indexing from a 1D iteration space down to
multi-dimensional coordinates fractures memory descriptors in the Low-Level
Optimizer (LLO). The DMA unit cannot statically determine the continuity of the
memory blocks, resulting in many disjoint, small VMEM HBM transfers rather than
large, efficient block transfers. This fragmentation can cause an extreme memory
bandwidth drop (upwards of a 90% loss in DMA efficiency).

**Best Practice:** Stick to native multi-dimensional grids that maintain a
1-to-1 structural mapping with the underlying data layout for optimal DMA
coalescence. Additionally, do not rely on out-of-place 1D alignment bypasses
(like the debunked "trunk-and-tail" concatenation method), as they trigger full
memory copy overheads and allocate new contiguous blocks in HBM.

## 3D Dimension Squeezing for Scattered Gathers (AX-116)

***(Reference: exp096_3d_dimension_squeezing_for_sca.py)***

When implementing dynamic gathers or pointer-chasing kernels (e.g.,
`PrefetchScalarGridSpec`), developers often attempt to flatten 3D tensors (such
as `pages.reshape(NumPages * PageSize, D)`) to satisfy the 2D grid flattening
rule. However, this is a severe performance and stability hazard.

### The Reshape and Strided DMA Constraint

Explicitly flattening 3D tensors forces the XLA compiler to emit HLO `reshape`
instructions that break physical memory contiguity. This degradation is
particularly severe for dynamic pointer chasing, where memory layout
predictability is paramount.

Furthermore, the Pallas lowering pipeline for flattened 2D arrays frequently
fails to correctly map dynamic scalar indices (e.g., `idx * PageSize`). This
failure causes the compiler to wrongly assume linear strides across the grid,
leading to incorrect strided DMA vectorization. At runtime, this triggers
catastrophic hardware `BoundsCheck` failures on `dma.hbm_to_vmem`, halting the
TPU cores prematurely.

### Resolution: 3D Dimension Squeezing

To bypass this constraint and avoid HLO reshapes entirely, you can rely on **3D
Dimension Squeezing**. Instead of flattening the input tensor, map the native 3D
tensor directly and configure the block shape with a `None` dimension:

```python
block_shape = (None, PageSize, D)
```

The `None` dimension instructs the compiler to completely drop the dynamic block
dimension during DMA routing. This provides a clean, 2D `out_ref` inside the
kernel body without requiring any physical layout changes.

#### Critical Performance Warning

*This technique is only safe for unbuffered descriptors.* If you attempt to use
`None` dimensions in conjunction with pipeline double-buffering (`pl.Buffered`),
the compiler will fail to optimize the descriptor ring, resulting in a severe
performance crash.

#### Empirical Validation

Hardware validation on TPU v6e confirms that the 3D dimension squeezing method
emits exactly zero HLO `reshape` instructions before the `tpu_custom_call`,
preserving physical memory contiguity and ensuring stable, high-throughput DMA
execution.

## DMA Affine Maps vs. In-Flight Routing (AX-028)

***(Reference: exp111_dma_affine_maps_vs__in_flight_.py)***

When designing complex routing kernels—such as Mixture of Experts (MoE) token
dispatchers or dynamic sparse gather/scatter operations—it is tempting to
attempt conditional or dynamic data routing directly during the HBM
$\rightarrow$ VMEM DMA transfer phase. This is typically attempted by embedding
non-linear expressions, dynamic index lookups, or routing tables inside Pallas's
`window_params` or `transform_indices` (e.g., `index_map`).

### The Linearity Constraint of MLIR Affine Maps

The underlying TPU compiler architecture relies on MLIR `affine_map`
representations to configure the hardware DMA engine. These affine maps are
mathematically restricted to **strictly linear expressions**. While they can
natively handle constants, additions, subtractions, and basic modulo operations
to facilitate strided block fetching, they are fundamentally incapable of
expressing:

*   Data-dependent conditions (e.g., `if` conditions based on token values).
*   Dynamic, runtime-evaluated lookups (e.g., indexing an external array to find
    a destination block).
*   Non-linear arithmetic (e.g., quadratic indexing or multiplication of loop
    induction variables).

The `window_params` attribute is designed exclusively to determine the static,
structured origin of a data block in HBM. Consequently, all dynamic
token-routing, gather/scatter logic, and MoE decision networks **must** be
implemented natively inside the kernel's computational body operating on data
that has already been loaded into VMEM (or offloaded via SparseCore `sync_copy`
commands).

### Failure Modes: Trace-Time vs. Runtime

Attempting to bypass these constraints leads to two distinct failure modes
depending on how the non-linear or dynamic mapping is constructed:

1.  **Trace-Time Rejection (JIT Compilation Failure):** If you attempt to
    reference an external array or variable (such as a routing index table
    `routing_indices[i]`) inside a JAX Pallas `index_map` function, the JAX
    tracer will immediately reject the operation during JIT compilation,
    throwing:

    ```
    ValueError: Index map function must not capture constants
    ```

    This occurs because the tracer cannot lower a dynamic, non-local array
    reference into a static MLIR affine map.

2.  **Hardware Halting (TensorCoreSequencer Failure):** If you circumvent the
    tracer by constructing geometrically valid but non-linear arithmetic (for
    example, using expressions like `i * i * 8` inside the block map), the code
    may silently compile. However, at runtime, it will immediately trigger a
    catastrophic hardware halt:

    ```
    XlaRuntimeError: Core halted unexpectedly
    ```

    This happens because the hardware TensorCoreSequencer / DMA sequencer throws
    a physical `BoundsCheck` exception when attempting to parse and execute
    non-strided, non-linear `dma.hbm_to_vmem` commands that violate the
    hardware's linear addressing capabilities.

## Block-Sparse Offset Masks for Paged Attention

***(Reference: exp078_block_sparse_offset_masks.py)***

In advanced attention mechanisms such as sliding-window or block-sparse
attention, the vast majority of `(q_block, kv_block)` pairs are masked out
(i.e., evaluated as zero). Naively, a standard Pallas kernel still loads these
masked KV blocks from HBM to VMEM and performs the matrix multiplication, only
to multiply the results by zero. This represents a massive waste of both DMA
bandwidth and Vector Processing Unit (VPU) FLOPs.

To bypass this bottleneck, highly optimized implementations (such as AxLearn's
`tpu_paged_attention_kernel.py`) employ a **block-sparse offset-mask** approach.
This technique precomputes an `(n_kv_blocks, n_kv_blocks)` table of offsets
pointing exclusively to **unmasked KV blocks**, transforming a sparse attention
computation into a dense-load-over-sparse-offsets pattern.

### Mechanism of Offset-Masking

1.  **Precomputation:** On the host (or during a lightweight pre-pass), generate
    an offset table where entry `(i, j)` contains the physical index of the
    $j$-th unmasked `kv_block` associated with the $i$-th `q_block`.
2.  **Kernel Execution:** Inside the Pallas kernel, instead of iterating
    sequentially over all physical KV blocks, the loop index $j$ runs from `0`
    to `n_unmasked_per_q_block`.
3.  **Indirect Addressing:** The kernel loads the target block index from the
    precomputed offset table (typically stored in Shared Memory, or SMEM) and
    uses it to perform an indirect lookup: `kv_page[offset_table[i, j]]`.
4.  **Dense Computation:** Because every loaded block is guaranteed to contain
    unmasked, mathematically relevant data, the VPU executes dense matrix
    multiplications without any multiply-by-zero overhead.

### Why It Matters for TPU Performance

On long-context sliding-window workloads (such as Gemma local-attention layers
with a 4K window on a 128K sequence), the fraction of unmasked blocks can be as
low as ~3%. Skipping the remaining 97% of blocks converts the attention
operation from a compute-bound dense bottleneck into a highly efficient,
memory-bound sparse operation. The resulting wall-clock speedup is roughly
proportional to the ratio of total KV blocks to unmasked KV blocks:

$$\text{Speedup} \approx \frac{\text{Total KV Blocks}}{\text{Unmasked KV Blocks}}$$

### Hardware Validation Notes

Empirical microbenchmarks on TPU v6e hardware confirm the massive performance
gains of this indirect addressing pattern. Comparing a dense attention loop
iterating over 128 KV blocks against a sparse attention loop iterating over 8
unmasked KV blocks using an offset mask in SMEM yielded the following results:

*   **Dense execution latency:** ~587 µs
*   **Sparse execution latency:** ~48 µs
*   **Speedup:** **~12.2x speedup** for 16x block sparsity.

This hardware validation confirms that indirect addressing (loading the
`offsets` table into SMEM, then indexing `k_ref[kv_idx]`) successfully bypasses
unnecessary DMA transfers and VPU instructions, translating directly into
proportional wall-clock savings.

### Applicability and Limitations

*   **Applies to:** Sliding-window attention, chunked-causal attention,
    block-sparse attention, and any paged attention variant where the mask
    structure is statically known per query block.
*   **Does not apply to:** Dense causal attention (which lacks block-level mask
    sparsity) or dynamic, data-dependent attention masks where the offset table
    cannot be precomputed prior to kernel dispatch.

## Scaling Beyond Single-Device HBM: Ring Attention

***(Reference: exp076_ring_attention.py)***

When sequence lengths scale to extreme limits, the memory footprint of the Key
(K) and Value (V) activation tensors quickly exceeds the physical High-Bandwidth
Memory (HBM) capacity of a single TPU v6e core. For instance, configuring a
massive attention workload with `num_heads=256`, `seq_len=131072`, and
`head_dim=128` using `bfloat16` precision requires initializing arrays that
easily exceed single-device limits, triggering a hard physical allocation error:

```
RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 8.00G. That was not possible. There are 7.25G free; (0x0x0_HBM0)
```

To bypass this single-device HBM ceiling, **Ring Attention** partitions the
sequence dimension across a mesh of $N$ physical devices. Instead of
materializing the entire sequence on each core, each device holds only a $1/N$
slice of the sequence in its local HBM.

### Inter-Chip Interconnect (ICI) Streaming

The core mechanism of Ring Attention relies on streaming K and V chunks around a
logical ring of devices using the TPU's high-speed Inter-Chip Interconnect
(ICI). During the forward pass:

1.  Each device computes local attention queries ($Q$) against its resident $K$
    and $V$ chunks.
2.  The $K$ and $V$ chunks are concurrently rotated to the next adjacent device
    in the ring topology using JAX collective communication primitives like
    `jax.lax.powermute`.
3.  This rotation and local computation cycle repeats $N$ times until every
    device has computed its local query against all global key/value blocks.

By overlapping the `ppermute` communication latency with the local
matrix-multiplication fusions, the TPU execution pipeline remains highly
saturated. XProf telemetry (`analyze_xplane`) confirms that this multi-device
mesh successfully distributes physical execution directly onto local convolution
and matrix-multiplication fusions without stalling on HBM limits.

### Zig-Zag Ring Attention

While Ring Attention solves the physical memory bottleneck, applying a naive
causal mask introduces a severe load-imbalance problem. Because causal attention
only computes the lower-triangular portion of the attention matrix, devices
processing early sequence blocks finish their computations early, leaving up to
50% of the total mesh compute capacity idle.

To rebalance this workload, **Zig-Zag Ring Attention** (Brandon et al. 2023)
reorganizes the sequence partitioning. Instead of contiguous block partitioning,
the sequence is interleaved and paired (e.g., pairing early blocks with late
blocks) such that every step of the ring rotation performs an approximately
equal number of causal-masked matrix multiplications. This rebalancing
eliminates the idle compute overhead, restoring near-linear scaling efficiency
across the TPU mesh.
