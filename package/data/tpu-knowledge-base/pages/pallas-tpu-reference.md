---
title: "Pallas TPU API Reference"
type: concept
tags: [tpu-kernel-development, pallas, tpu, api-reference]
created: 2026-07-16
updated: 2026-07-16
sources: 0
human_reviewed: false
---

# Pallas TPU API Reference

This document provides a comprehensive reference of the Pallas API specifically
for TPU. It is intended for use by Gemini agents and developers writing custom
high-performance kernels for Google TPUs.

## Core Pallas APIs (`jax.experimental.pallas`)

The following APIs are the foundational building blocks of the Pallas
programming model.

### Kernel Invocation

-   **`pallas_call`**
    -   **Signature:** `pallas_call(kernel, out_shape, *, grid_spec=None,
        grid=(), in_specs=NoBlockSpec, out_specs=NoBlockSpec, scratch_shapes=(),
        input_output_aliases={}, debug=False, interpret=False, name=None,
        compiler_params=None, cost_estimate=None, backend=None)`
    -   **Description:** Invokes a Pallas kernel on some inputs. Returns a
        function that can be called on array arguments.
    -   **Key Arguments:**
        -   `kernel`: The kernel function, receiving a Ref for each input and
            output.
        -   `out_shape`: A PyTree of `jax.ShapeDtypeStruct` describing the shape
            and dtypes of outputs.
        -   `grid`: The iteration space as a tuple of integers.
        -   `in_specs`, `out_specs`: PyTrees of `BlockSpec` with structure
            matching inputs/outputs.
        -   `scratch_shapes`: PyTree of backend-specific temporary objects
            (e.g., semaphores).
        -   `compiler_params`: Optional backend-specific dataclass (e.g.,
            `pltpu.CompilerParams`).
        -   `interpret`: If True, runs the kernel on CPU (useful for debugging).
-   **`core_map(f, *args, ...)`**: Maps a function over a mesh of devices, often
    used for multi-core TPU kernels.

### Tiling and Memory Mapping

-   **`BlockSpec`**
    -   **Signature:** `BlockSpec(block_shape=None, index_map=None,
        indexing_mode=None, pipeline_mode=None, *, memory_space=None)`
    -   **Description:** Specifies how an array should be sliced for each
        invocation of a kernel.
    -   **Arguments:**
        -   `block_shape`: Sequence of `int | None` or `BlockDim` types (e.g.
            `pl.Element`, `pl.Squeezed`, `pl.Blocked`, `pl.BoundedSlice`).
            Specifies the size of the block dimension. `None` is a squeezed
            dimension.
        -   `index_map`: A function mapping grid indices to block offsets. Must
            return a tuple of the same length as `block_shape`.
        -   `memory_space`: Where the block should reside (e.g., `pltpu.VMEM`
            for TPU SRAM).
        -   `pipeline_mode`: E.g., `pl.Buffered()` for double-buffering
            configurations.
-   **`GridSpec`**
    -   **Signature:** `GridSpec(grid=(), in_specs=NoBlockSpec,
        out_specs=NoBlockSpec, scratch_shapes=())`
    -   **Description:** Encodes the grid parameters for `pallas_call` natively
        into one object.
-   **`Slice(start, size, stride=1)`**: Represents a memory slice.
-   **`ds(start, size)`**: Shorthand for creating a `Slice` (Dynamic Slice).

### Grid Intrinsics

-   **`program_id(axis)`**: Returns the current grid index along the specified
    axis.
-   **`num_programs(axis)`**: Returns the total size of the grid along the
    specified axis.

### Memory Operations

-   **`load(ref_or_view, idx, *, mask=None, other=None, cache_modifier=None,
    eviction_policy=None, volatile=False)`**: Reads an array from the given
    index. `mask` specifies which indices to load; `other` is the value to use
    where mask is False.
-   **`store(ref_or_view, idx, val, *, mask=None, eviction_policy=None)`**:
    Stores a value at the given index. `mask` specifies which indices to write.
-   **`ref[...]`**: Shorthand for `load` and `store`.
-   **`ANY`**: A generic memory space, usually representing HBM.
-   **`atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`,
    `atomic_or`, `atomic_xchg`, `atomic_xor`**: Atomic memory operations
    performed on a `Ref`.
-   **`max_contiguous(x, n)`**: A compiler hint about the maximum number of
    contiguous elements.

### Math and Logic

-   **`dot(a, b, ...)`**: A high-performance matrix multiplication intrinsic. On
    TPU, this maps to the Matrix Unit (MXU).
-   **`broadcast_to(x, shape)`**: Broadcasts a value to a given shape.
-   **`cdiv(a, b)`**: Ceiling division.
-   **`multiple_of(x, n)`**: A compiler hint asserting that `x % n == 0`.
-   **`next_power_of_2(n)`**: Computes the next power of two.
-   **`reciprocal(x)`**: Computes the reciprocal.

### Scoping and State

-   **`run_scoped(f, *args, **kwargs)`**: Creates a new scope for memory
    allocations (using `Ref`).
-   **`run_state(f, *args, **kwargs)`**: Used for kernels that maintain internal
    state.

### Control Flow

-   **`loop(n_iter)`**: A decorator or context manager for hardware-accelerated
    loops.
-   **`when(condition)`**: A decorator or context manager for conditional
    execution.

### Debugging

-   **`debug_print(fmt, *args)`**: Prints values during kernel execution (use
    with caution).
-   **`debug_check(condition, msg)`**: Asserts a condition during execution.

--------------------------------------------------------------------------------

## TPU-Specific APIs (`jax.experimental.pallas.tpu`)

Imported as: `from jax.experimental.pallas import tpu as pltpu`

### Memory Spaces

-   **`pltpu.VMEM`**: Vector Memory (SRAM). The primary on-chip storage for
    array data.
-   **`pltpu.SMEM`**: Scalar Memory (SRAM). For scalar values and control logic.
-   **`pltpu.CMEM`**: Common Memory. Used for shared constants or low-latency
    common storage.
-   **`pltpu.HBM`**: High-Bandwidth Memory (DRAM). Off-chip storage.
-   **`pltpu.SEMAPHORE`**: Dedicated memory space for synchronization
    primitives.

### DMA and Data Movement

-   **`pltpu.async_copy(src_ref, dst_ref)`**: Initiates an asynchronous DMA
    transfer between memory spaces (e.g., HBM to VMEM).
-   **`pltpu.sync_copy(src_ref, dst_ref)`**: Performs a synchronous DMA
    transfer.
-   **`pltpu.make_async_copy(src_ref, dst_ref, sem)`**: Issues a DMA copying
    from `src_ref` to `dst_ref`.
-   **`pltpu.make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem,
    device_id, device_id_type=pl.DeviceIdType.MESH)`**: Creates a description of
    a remote copy operation (RDMA) to `device_id`. Both semaphores should be
    waited on using the descriptor on both source and target devices.

### Synchronization

-   **`pltpu.semaphore_signal(sem_ref, value=1, device_id=None)`**: Increments a
    semaphore.
-   **`pltpu.semaphore_wait(sem_ref, value)`**: Blocks until the semaphore
    reaches the specified value.
-   **`pltpu.semaphore_read(sem_ref)`**: Returns the current value of a
    semaphore.
-   **`pltpu.core_barrier()`**: A barrier across all cores in a TPU device.
-   **`pltpu.get_barrier_semaphore()`**: Retrieves a specialized semaphore for
    inter-core synchronization.
-   **`pltpu.dma_semaphore()`**: A specialized semaphore for DMA operations.

### TPU Intrinsics

-   **`pltpu.bitcast(x, dtype)`**: Reinterprets the bits of a value as a
    different data type.
-   **`pltpu.roll(x, shift, axis)`**: Performs a roll operation on an array.
-   **`pltpu.repeat(x, repeats, axis)`**: Repeats elements of an array.
-   **`pltpu.delay(cycles)`**: Inserts a hardware delay (useful for fine-grained
    timing).
-   **`pltpu.sample_block(...)`**: Used for block-sparse or sampled memory
    access.

### Pipelining

-   **`pltpu.emit_pipeline`**
    -   **Signature:** `emit_pipeline(body, *, grid, in_specs=None,
        out_specs=None, should_accumulate_out=False, core_axis=None,
        core_axis_name=None, dimension_semantics=None, trace_scopes=True,
        no_pipelining=False, use_sreg_for_state=False)`
    -   **Description:** Creates a function to emit a manual pallas pipeline.
        Has the same semantics as `pallas_call` but is meant to be called inside
        `pallas_call` for nesting grids. Useful for separating windowing
        strategies for communication and computation.
    -   **Arguments:**
        -   `should_accumulate_out`: boolean to indicate which outputs should be
            treated as accumulators automatically within/across pipeline
            invocations.
        -   `core_axis`: optional int, indicates whether to partition the grid
            along the core axis.
-   **`pltpu.BufferedRef`**: A helper class to automate VMEM double buffering in
    pallas pipelines. Manages `window_ref`, `current_slot`, `sem_recvs`, and
    `sem_sends`.
-   **`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch, grid=(),
    in_specs=NoBlockSpec, out_specs=NoBlockSpec, scratch_shapes=())`**:
    Optimization for prefetching scalar values into SMEM.
-   **`pl.Buffered(buffer_count, use_lookahead=True)`**: Used in `BlockSpec` to
    enable automatic multi-buffering and prefetching.

### Random Number Generation

-   **`pltpu.prng_seed(seed)`**: Initializes the TPU-native PRNG state.
-   **`pltpu.prng_random_bits(shape, dtype)`**: Generates random bits using the
    TPU hardware generator.

### Compiler Configuration

-   **`pltpu.CompilerParams`**
    -   **Signature:** `CompilerParams(dimension_semantics=None,
        allow_input_fusion=None, vmem_limit_bytes=None, collective_id=None,
        has_side_effects=False, flags=None, internal_scratch_in_bytes=None,
        serialization_format=1, kernel_type=<KernelType.TC: 0>,
        disable_bounds_checks=False)`
    -   **Description:** Mosaic TPU compiler parameters.
    -   **Key Attributes:**
        -   `dimension_semantics`: A list of dimension semantics for each grid
            dimension of the kernel. Either "parallel" (`pltpu.PARALLEL`) or
            "arbitrary" (`pltpu.ARBITRARY`).
        -   `vmem_limit_bytes`: Overrides the default VMEM limit for a kernel
            (use with `--xla_tpu_scoped_vmem_limit_kib`).
        -   `collective_id`: Indicates which barrier semaphore to use for the
            kernel.
-   **`pltpu.PARALLEL`**: Semantic hint that a grid dimension can execute in any
    order.
-   **`pltpu.ARBITRARY`**: Semantic hint for grid dimensions that must execute
    sequentially.

--------------------------------------------------------------------------------

## Common TPU Patterns

### Manual Pipelining with Semaphores

```python
# Allocate semaphores as scratch in pallas_call
# scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]

def kernel(x_ref, y_ref, send_sem, recv_sem):
    copy = pltpu.make_async_copy(x_ref, y_ref, send_sem, recv_sem)
    copy.start()

    # Do work while copying
    ...

    copy.wait_send() # Local resource (x_ref) can be reused
    copy.wait_recv() # Remote resource (y_ref) is ready to read
```

### Megacore Parallelism

```python
# Parallelize across both TensorCores on a TPU v4/v5p chip
compiler_params = pltpu.CompilerParams(
    dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL)
)
pl.pallas_call(..., compiler_params=compiler_params)
```

### Semaphore Types

-   **`pltpu.SemaphoreType.DMA`**: Optimized for DMA tracking.
-   **`pltpu.SemaphoreType.REGULAR`**: General purpose synchronization.

--------------------------------------------------------------------------------

## Implementation Constraints on TPU

-   **SRAM Limits**: VMEM/SMEM are limited (e.g., 16MB per core on v4). Large
    kernels must tile aggressively.
-   **Alignment**: TPU memory accesses are most efficient when aligned to 128
    bytes (or multiples of 128 elements).
-   **Data Types**: `bfloat16` is the native high-performance type. `float32` is
    supported but often slower and uses twice the VMEM.
-   **Grid Order**: The default grid execution order is usually lexicographical,
    but this can be influenced by pipelining.

--------------------------------------------------------------------------------

## Agent Best Practices & TPU Mental Model

When writing Pallas kernels, keep the following foundational concepts in mind to
ensure performance and correctness:

### 1. The TPU Execution Model

-   **Grid vs. Core:** A single `pallas_call` executes over a multi-dimensional
    grid of blocks. *Do not confuse the grid with physical TPU cores.* JAX
    handles mapping the grid iterations to the available hardware (often
    sequentially within a single core, unless `core_axis` or `core_map` is
    used).
-   **Static Sizing:** All shapes and slice sizes inside a Pallas kernel *must*
    be statically known at compile time. You cannot use dynamic tensor values to
    determine the size of a block or loop iteration. (Use `pl.BoundedSlice` or
    `pl.ds` carefully if dynamic boundaries are strictly necessary).

### 2. Memory Hierarchy Management

-   **HBM is Slow, SRAM is Fast:** The primary goal of a Pallas kernel is to
    move data from HBM (DRAM) to VMEM (Vector SRAM), do as much math as
    possible, and move it back.
-   **Dereferencing `Ref`s:** You cannot perform math directly on a `Ref` object
    that points to HBM (`pl.ANY`). You *must* `load` it into VMEM/Registers
    first: `x = x_ref[...]`, perform math: `y = x + 1`, and `store` it back:
    `y_ref[...] = y`.
-   **Double Buffering:** The `pl.Buffered` configuration in `BlockSpec` is
    crucial for hiding memory latency. It automatically allocates two slots in
    VMEM so the TPU can load the next block while computing the current one.

### 3. TPU Hardware Units

-   **VPU (Vector Processing Unit):** Handles element-wise operations and
    standard math on arrays (using VMEM).
-   **MXU (Matrix Multiply Unit):** A highly specialized, massive systolic
    array. You *must* use `pl.dot()` to access it. If you write manual nested
    loops to multiply matrices element-by-element, it will run on the VPU and be
    incredibly slow.
-   **Scu (Scalar Unit):** Handles loop counters, control flow, and scalar math.
    Lives in SMEM.

### 4. Debugging and Development

-   **Start with `interpret=True`:** Always test logic on the CPU first. It runs
    the kernel as a standard JAX scan loop, allowing you to use
    `jax.debug.print` natively and catch index-out-of-bounds errors easily.
-   **`debug_print` Caveats:** On actual TPU hardware, `debug_print` output is
    often buffered, delayed, or out of order due to massive parallelism. Use it
    sparingly.
-   **Avoid Complex Control Flow:** TPUs are SIMD (Single Instruction, Multiple
    Data) machines. Divergent control flow (`if/else` inside loops where
    conditions depend on vector data) will cause severe performance degradation
    (branch masking). Prefer `pl.when` or mathematical masking (`x * mask`).
