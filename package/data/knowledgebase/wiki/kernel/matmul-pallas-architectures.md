---
title: "Chapter 1: Program Instance (Kernel Math & Precision)"
type: concept
tags: [kernel, pallas, matmul, precision]
created: 2026-07-16
updated: 2026-07-16
sources: 0
human_reviewed: false
---

# Chapter 1: Program Instance (Kernel Math & Precision)

Writing performant Pallas kernels for TPU v6e (Trillium) requires a fundamental
shift from mathematical theory to physical hardware reality. Unlike generalized
compute architectures, the TPU execution is divided between tightly coupled,
specialized processing units—primarily the Vector Processing Unit (VPU), the
Matrix Multiplication Unit (MXU), and the Cross-Lane Unit (XLU). A single line
of code can trigger disastrous pipeline stalls or severe layout transformations
if not mapped correctly to these units.

This chapter details the strict constraints and behavioral nuances of kernel
math, data types, and precision within the TPU program instance.

## Matrix Multiplication Unit (MXU) Behavior & Constraints

The MXU is the workhorse of the TPU, capable of immense throughput for matrix
operations. However, it expects rigid conformity regarding datatypes and memory
layouts.

### 32-Bit Accumulator Constraint

When targeting the Mosaic TPU backend, `jax.lax.dot_general` strictly lowers to
the `tpu.matmul` operation. The underlying MLIR lowering expects the accumulator
data type to be explicitly 32-bit (e.g., `vector<32x32xf32>`).

If the inputs are `bfloat16` and the accumulator type is omitted, JAX defaults
the accumulator to `bfloat16`. This leads to a fatal compiler verification
error: `'tpu.matmul' op Expected matmul acc to be 32-bit`

**Best Practice:** Always pass `preferred_element_type=jnp.float32` when
invoking `jax.lax.dot_general`. If subsequent operations require lower
precision, explicitly downcast the result (e.g., `.astype(jnp.bfloat16)`) after
the matmul completes.

### Dtype Strategy: VPU-to-MXU Casting Overhead

***(Reference: exp059_dtype_strategy.py)***

The choice and mixing of data types—such as `bfloat16`, `float32`, `float8`, or
`int8`—across parameters, activations, and compute pipelines dictates whether a
kernel achieves peak hardware utilization or suffers severe execution penalties.
The TPU Matrix Multiplication Unit (MXU) structurally demands native inputs
(such as `bfloat16` for standard multipliers). Mixing data types—such as
multiplying a `bfloat16` activation with an `fp32` weight matrix—prevents native
ingestion.

To resolve the mismatch, the XLA compiler will implicitly inject a casting
operation prior to fusion. This forces the VPU to intervene, casting the `fp32`
values to `bfloat16` upstream of the MXU dot operation. On large-scale
operations (e.g., an 8192 x 8192 dot product), this implicit VPU casting acts as
a critical path bottleneck, empirically bloating execution latency by up to 35%.
Ensure all inputs to the MXU match its expected native precision to avoid VPU
stalling.

#### Empirical Verification of Casting Overhead

An empirical verification conducted on an 8192x8192 `jax.numpy.dot` workload on
TPU validates this hardware bottleneck:

*   **Native Execution:** Using native `bfloat16` inputs for both activations
    and weights resulted in a purely native MXU execution, yielding a latency of
    **~1.53ms** per iteration.
*   **Mixed-Precision Execution:** Mixing `bfloat16` activations with `float32`
    weights forced the XLA compiler to emit an implicit cast fusion, increasing
    hardware execution time to **~2.06ms** per iteration.
*   **Measured Penalty:** This represents a **~35% cost overhead** (with a
    baseline penalty of at least ~17% depending on tensor shapes and memory
    bandwidth bounds). This decisively proves that non-native weight dtypes
    force a runtime cast on the VPU before the MXU can ingest them, violating
    optimal execution strategy.

### VPU-MXU Tiled Layout Compatibility

When VPU element-wise operations feed directly into MXU dot-products, XLA
cross-compatibly assigns the MXU tiled layout backward through the execution
graph. The VPU natively processes these tiled standard block memory structures
without injecting XLU copies, reshapes, or transpositions. Values loaded into
VMEM are seamlessly computed in-place by the VPU and immediately ingested into
the MXU pipeline.

### Native VMEM Dense Projections (MXU Fusion) (AX-019)

***(Reference: exp047_native_vmem_dense_projections_.py)***

Wrapping host-side dense operations (such as `jax.lax.dot_general`) and a
`pallas_call` inside a single `@custom_jax_kernel` creates an opaque compiler
barrier. This barrier forces XLA to physically materialize massive intermediate
tensors into High Bandwidth Memory (HBM), severely limiting performance and
wasting memory bandwidth.

To bypass this limitation, Pallas supports native dense matrix multiplications
(MXU transverse contractions) directly within the VMEM execution block. Moving
the projection math entirely inside the `pallas_call` kernel body allows XLA to
fuse the operations, drastically reducing HBM traffic.

#### Critical Syntax and Hardware Requirements

*   **32-Bit Accumulator Enforcement:** When multiplying `bfloat16` inputs
    inside a native Pallas block, the TPU Matrix Multiplication Unit (MXU)
    strictly requires 32-bit accumulators. You **must** explicitly provide
    `preferred_element_type=jnp.float32` to the internal `dot_general` call.
    Omitting this parameter will cause the compilation to crash with an internal
    error: `tpu.matmul requires 32-bit accumulators for bfloat16 inputs`
*   **VMEM Footprint Overhead:** Developers must account for the physical
    constraint that accumulator buffers take up $4 \times$ the memory footprint
    in VMEM per block output compared to standard 16-bit blocks. This increased
    footprint must be factored into block size planning to avoid VMEM overflow.

## Transverse Contraction

***(Reference: exp021_transverse_contraction__ax_015.py)***

Transpositions (`.T`) in Vector Memory (VMEM) are executed by the Cross-Lane
Unit (XLU) *after* loading. This is a high-latency hardware swizzle operation
that competes for vector lanes. Benchmarks on Trillium confirm that purely
reorganizing vector lanes introduces an unnecessary physical hardware penalty
(~9.4% overhead compared to native baseline copies).

**Resolution:** Instead of transposing in VMEM, rely on **Transverse
Contraction**. By executing `jax.lax.dot_general(A, B, dimension_numbers=(((1,),
(1,)), ((), ())))`, you instruct the MXU to mathematically absorb the transposed
layout during the accumulation phase itself. This completely bypasses the XLU
materialization penalty, guaranteeing optimal layout traversal.

### Hardware Telemetry Verification (AX-015)

Empirical profiling on Trillium TPUs validates this constraint and resolution:

1.  **Isolated Swizzle Penalty:** Benchmarking a pure memory copy vs. a pure
    transpose (`.T`) in VMEM across a 4x4 grid of 512x512 bfloat16 chunks
    confirms that the XLU swizzle adds significant overhead. The VPU baseline
    copy averaged `16.64 us` per macro-step, while the physical transpose took
    `18.21 us`—a strictly physical hardware penalty of ~9.4% just to reorganize
    the vector lanes.
2.  **MXU Absorption Bypass:** Evaluating an explicit `A @ B.T` against a native
    Transverse Contraction via `jax.lax.dot_general(A, B,
    dimension_numbers=(((1,), (1,)), ((), ())))` yielded identical performance
    curves (`24.98 us`).
3.  **Compiler Robustness:** The MXU is physically capable of absorbing the
    transposed memory layout via the `dot_general` contract dimensions without
    materializing the layout shift in the XLU. While the modern Mosaic compiler
    is often smart enough to auto-fuse `.T` directly followed by a matmul,
    relying entirely on Transverse Contractions using `dot_general` is the most
    robust way to guarantee the penalty is bypassed, especially in complex
    pipelines where the XLA fusion heuristics might break.

## Execution Bounds and Memory Realities

### Avoiding VPU Math Optimizations when HBM Bound

Operations possessing exceptionally low arithmetic intensity—such as Rotary
Position Embeddings (RoPE)—are entirely memory-bound by HBM/VMEM load limits.
The VPU requires an identical number of hardware execution cycles to process
math regardless of how the operations are structured (e.g., splitting arrays
into subset sizes `N/2` or performing single vector ops).

Attempting to restructure host math or kernel operations to "optimize" VPU
arithmetic yields absolutely zero latency improvement. When a kernel is
memory-bound, do not waste architectural effort over-optimizing the VPU
instruction layout.

### Physical Cost of Reshape Operations

Pallas developers often flatten array dimensions to bypass strict block
specifications (e.g., reshaping `(5, 512)` to `(2560)`), assuming the operation
has zero cost. However, JAX arrays on TPU are natively stored in tiled physical
layouts in High Bandwidth Memory (HBM).

A rank-2 array of `(5, 512)` defaults to a `T(8, 128)` tile layout. Squeezing
this to a 1D dimension abruptly changes the physical topology requirement to a
`T(1024)` layout. XLA will issue operations that physically reorganize and move
data in HBM, destroying in-place execution guarantees and introducing immense
overhead. Reshaping on TPU is rarely zero-cost.

### Persistent State via Loop Carries

***(Reference: exp080_persistent_state_via_loop_carr.py)***

When writing iterative kernels (such as chunked attention or autoregressive
generation), holding persistent state (like running sums or maximums) across
sequence chunks can be challenging. While developers often resort to fragile
`scratch_shapes` APIs to allocate temporary workspace in Vector Memory (VMEM), a
highly optimized hardware alternative exists.

By relying exclusively on `jax.lax.fori_loop` state tuples (e.g., passing
`(acc_max, acc_sum)`), the compiler perfectly hoists these loop-carried
variables into dedicated Vector Registers (typed natively as `vector<128xf32>`
in Mosaic MLIR). This provides zero-cost state persistence across iterations
without triggering VMEM memory spills.

**Register Allocation and Pressure Constraints:**

*   **Native Vector Mapping:** As long as the tuple elements can map to native
    vector registers (e.g., shapes of `(128,)` or multiples thereof), they will
    be passed purely in registers.
*   **Register Spilling:** Avoid exploding register pressure by carrying too
    many giant tensors in the loop state. Exceeding the hardware's physical
    register capacity forces the compiler to spill these variables back to VMEM,
    negating the zero-cost performance benefit. Keep loop-carried states compact
    and aligned with native vector dimensions.

### 2D Topology Static Mapping (Dynamic Slicing Constraints)

***(Reference: exp083_2d_topology_static_mapping__ax.py)***

Dynamic slicing within the VPU (`KernelType.TC`) is strictly unsupported on TPU
hardware. Attempting to use dynamic indexing or slicing operations natively—such
as `jax.lax.dynamic_slice` or `jax.lax.dynamic_update_slice`—inside a Pallas
kernel will trigger a fatal compiler lowering failure: `NotImplementedError:
Unimplemented primitive in Pallas TPU lowering for KernelType.TC`

To bypass this hardware and compiler limitation, developers must enforce static
2D topology mapping. Tensors must be securely mapped into local SRAM (VMEM)
using static, compile-time offsets and block specifications rather than dynamic
runtime slicing. This static mapping guarantees mathematical parity, prevents
compiler lowering failures, and ensures peak memory bandwidth utilization.

## Precision Drops and Weak Typing

### Pallas Weak Type Promotion (JEP-401)

Pallas intrinsically respects JAX's weak-typing semantics introduced by JEP-401.
Inside a Pallas kernel, raw Python scalars (e.g., `1e-5` or `0.0`) are evaluated
as "weakly typed." If they interact with a strongly-typed VMEM block (such as
`bfloat16`), the scalar seamlessly inherits the target precision without
implicitly promoting the entire array up to `float64`.

**Do not forcibly cast scalars:** Developers may be tempted to explicitly define
scalar types (e.g., `jnp.float32(1e-5)`) as a protective measure. This assigns a
"strong type" to the scalar. If applied inside a `bfloat16` computation graph,
the strong `float32` type forces intermediate states to upcast, resulting in
fatal compiler crashes (`ValueError: Invalid dtype for swap`) when attempting to
rewrite the result back into the strictly allocated `bfloat16` reference space.
Rely entirely on raw Python scalars.
