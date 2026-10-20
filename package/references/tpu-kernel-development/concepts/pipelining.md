---
name: pipelining
description: "TPU Pipelining and Double Buffering"
---

+++ {"id": "7704d3bb"}

(pallas_tpu_pipelining)=

+++ {"id": "teoJ_fUwlu0l"}

# TPU Pipelining

<!--* freshness: { reviewed: '2024-04-08' } *-->

+++ {"id": "gAJDZh1gBh-h"}

This guide serves as a reference for TPU-specific pipelining concerns.
We'll review the memory hierarchy and compute units on TPUs, and TPU-specific features of the pipelining API. For a more general-purpose overview of pipelining, see the {ref}`pallas_software_pipelining`.

```{code-cell} ipython3
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1744908474512
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: ejAVO6ikUUuF
---
#@title Imports

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

+++ {"id": "0e212a5e"}

(tpu_and_its_memory_spaces)=

## TPU and its memory spaces

+++ {"id": "NnWW9GV4kW6P"}

A TPU and its TensorCore consist of memory spaces (where arrays can reside),
registers (which temporarily store scalar and array values) and compute units
(that do computation with values in registers).
Below is a diagram of a TPU in which `x` and `y` are arrays that live in
high-bandwidth memory (HBM):

(Figure not included: the TPU memory spaces. Its components are described
below.)

Let's talk about the components of this diagram in more detail:

* **Memory spaces**: A TPU has high-bandwidth memory (HBM) which is what we
  often think of as "device memory".
  There is also vector memory (VMEM),
  a cache meant for storing vector and array values, and scalar memory (SMEM),
  a cache designed to store scalar values.
* **Registers**: A TensorCore has two main types of registers: vector
  registers (VREGs) store array values, and scalar registers (SREGs) store
  scalar values.
  Values can be loaded into memory from their respective caches (VMEM for
  VREGs and SMEM for SREGs).
* **Compute units**: A TensorCore has a scalar unit, vector unit (VPU) and
  matrix unit (MXU) that can do numerical computation. Each of these compute units can operate asynchronously, but this is managed by the TPU compiler and thus from the programmer's perspective a TPU program is single-threaded.
  Compute units operate on values that live in SREGs and VREGs and output
  values into those registers as well.

+++ {"id": "8Tl3wt5Wk3Ek"}

## TPU-specific Pipelining Features

Pallas TPU supports the following platform-specific features.

+++ {"id": "1jg5WmExk47l"}

### TPU Memory Spaces

Pallas exposes all levels of the TPU memory hierarchy to users. The following table maps from Pallas TPU memory spaces to their standard memory types (DRAM/SRAM):

| Pallas Enum | TPU Memory Space | Type (DRAM/SRAM) |
| --- | --- | --- |
| `pl.ANY` | HBM (usually) or VMEM | DRAM |
| `pltpu.VMEM` | VMEM | SRAM |
| `pltpu.SMEM` | SMEM | SRAM |
| `pltpu.SEMAPHORE` | Semaphore | SRAM |

- `MemorySpace.VMEM` denotes vector SRAM. It is the default memory space if nothing is specified.
- `MemorySpace.SMEM` denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM.
- `MemorySpace.ANY` is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or `pltpu.async_copy`.
- `MemorySpace.SEMAPHORE` is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see {ref}`pallas_async` for more details.

Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The default behavior for `pallas_call` on TPU is that arguments to `pallas_call` are assumed to live in HBM, and inputs to the user kernel body are stored in VMEM.

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the `memory_space` argument on a `BlockSpec`. Note that pipelining is not allowed unless the `memory_space` is marked as `VMEM`. Memory spaces can also be used to specify scratch arguments to a kernel via the `scratch_shapes` argument on `pallas_call`. Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in `VMEM`, `SMEM`, or `SEMAPHORE`.

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer `x_hbm_ref` into a scratch VMEM buffer `scratch_vmem_ref` before using it for arithmetic and storing the result into an output VMEM buffer:

```{code-cell} ipython3
---
executionInfo:
  elapsed: 65
  status: ok
  timestamp: 1744908591430
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: zcqz1CA_o50a
---
def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
  pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
  out_vmem_ref[...] = scratch_vmem_ref[...] + 1

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
out = pl.pallas_call(hbm_vmem_kernel,
  in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
  out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
  scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),)
)(x)

np.testing.assert_allclose(out, x[0:1] + 1)
```

### Multiple Buffering

Multiple buffering can be specified on a per-argument basis to the pipeline via the `pipeline_mode` option on `pl.BlockSpec`. To do so, pass a `pl.Buffered` object to `pl.BlockSpec` specifying the number of buffers to allocate for this particular argument:

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count)
)
```

The default buffer count is 2 for all inputs and outputs.

+++

(pallas_tpu_emit_pipeline)=

### pltpu.emit_pipeline

`pltpu.emit_pipeline` is a pipelining API implemented in Pallas that allows you to construct pipelines inside of a kernel rather than only on kernel entry. This several use-cases over using `pl.pallas_call`, such as:
- For constructing nested pipelines. For example, an outer pipeline that communicates between chips, and an inner pipeline that performs HBM-VMEM pipelining.
- For using `emit_pipeline` specific features such as lookahead prefetch and dynamic block shapes (covered below).

`pltpu.emit_pipeline` follows a similar signature to `pl.pallas_call` and requires you to specify a body `kernel`, a grid, and block specs for inputs and outputs:

```python
def emit_pipeline(
    kernel: Callable,
    grid: tuple[int],
    in_specs: PyTree[BlockSpec] = None,
    out_specs: PyTree[BlockSpec] = None,
    dimension_semantics: tuple[GridDimensionSemantics] = None,
    core_axis: int | None = None,
) -> Callable:
  ... # Returns a custom pipeline given an inner kernel and BlockSpecs.
```

The `dimension_semantics` and `core_axis` arguments are used for partitioning the kernel grid over Megacore (see below).

+++

### Lookahead Prefetch

Lookahead prefetch is a pipelining feature where the pipeline will attempt to prefetch the next input block as soon as a buffering slot is available, rather than the iteration directly before it would be used. For example, if the kernel had a grid of `(8,)` and the block indices to fetch on each iteration were `0, 0, 0, 0, 1, 1, 1, 1`, then lookahead prefetch will begin fetching both blocks `0` and `1` on iteration 0, whereas the standard pipeline schedule would fetch block `0` on iteration 0 but not begin fetching block `1` until iteration 3. There is a small amount of control flow overhead in performing lookahead so it is disabled by default.

Lookahead is primarily useful when there is a variable amount of compute work in each block, such as when some blocks contain skipped or a reduced amount of work. In these cases, there may not be enough compute work in the iteration immediately preceding the step when the block is needed to fully overlap with the memory transfer. Therefore, we would like to begin fetching blocks earlier in the pipeline.

Lookahead prefetch can be used in conjunction with multiple buffering and can likewise be enabled by passing `pl.Buffered` into the `pipeline_mode` argument:
```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
)
```

+++

### Dynamic Block Shapes

`pltpu.emit_pipeline` supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with `pl.BoundedSlice(max_size)` rather than a static integer size, where `max_size` is the maximum size of the block. In addition, the corresponding index returned by `index_map` should be a dynamic slice constructed via `pl.ds(start, size)` where both `start` and `size` are _element_ indices (not block indices) and can be dynamic.

The following is an example for a block spec with a dynamic first dimension:

```python
pl.BlockSpec(
   block_shape=(pl.BoundedSlice(32), 256),
   index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
)
```

```{code-cell} ipython3
# The following kernel copies `x` to the output in dynamic-sized chunks
# passed in via `slices`.

def dynamic_block_example_kernel(x_hbm, slices_hbm, o_hbm, slices_smem):
    pltpu.sync_copy(slices_hbm, slices_smem)  # Copy slices into SMEM.
    def pipeline_body(x_vmem, o_vmem):
        o_vmem[...] = x_vmem[...]
    def index_map(i):
        start = slices_smem[i, 0]
        size = slices_smem[i, 1] - slices_smem[i, 0]
        return (pl.ds(start, size), 0)
    block_spec = pl.BlockSpec(block_shape=(pl.BoundedSlice(8), 128),
                              index_map=index_map)
    pltpu.emit_pipeline(
        pipeline_body,
        grid=(slices.shape[0],),
        in_specs=[block_spec],
        out_specs=block_spec
    )(x_hbm, o_hbm)

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
slices = jnp.array([[0, 2], [2, 3], [3, 5], [5, 8]], dtype=jnp.int32)

hbm_block_spec = pl.BlockSpec(memory_space=pl.ANY)
out = pl.pallas_call(dynamic_block_example_kernel,
               in_specs=[hbm_block_spec, hbm_block_spec],
               out_specs=hbm_block_spec,
               out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
               scratch_shapes=(pltpu.SMEM(slices.shape, jnp.int32),)
              )(x, slices)

np.testing.assert_allclose(x, out)
```

+++ {"id": "KvPFez9N8cKJ"}

(pallas_tpu_megacore)=

### TPUs in Megacore configuration

+++ {"id": "0f4HAVzQ8n71"}

Some TPU chips have two TensorCores but appear as one device to JAX users.
This is called "megacore".
The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs
and compute units but *share HBM*.

(Figure not included: the TPU memory spaces in megacore, with two
TensorCores.)

Conceptually, TPUs in Megacore behave like very simple GPUs, i.e. they have
only two threads.
How do we modify our kernels to utilize both TensorCores simultaneously?

The basic idea is that if we have embarrassingly parallel dimensions in our
computation, we can split up those dimensions across the TensorCores.
We can indicate which dimensions are parallelizable by providing an
annotation to `pallas_call` called `dimension_semantics`.

```{code-cell} ipython3
---
executionInfo:
  elapsed: 106
  status: ok
  timestamp: 1744910274556
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: nQNa8RaQ-TR1
outputId: 29c0b574-3528-49a5-8a88-b6987efc69ce
---
def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
  z_vmem_ref[:, :] = z_vregs

def add_matrices_pipelined_megacore(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=jax.ShapeDtypeStruct.like(x),
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(2,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )(x, y)

x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices_pipelined_megacore(x, y)
```

+++ {"id": "xG51AiUC-8cl"}

`dimension_semantics` should be a tuple of same length as `grid` where each
entry is either `"parallel"` or `"arbitrary"`. `"parallel"` indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. `"arbitrary"` indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

By specifying `dimension_semantics`, we now execute the kernel
simultaneously on each TensorCore. Pallas will handle splitting up the grid
automatically.

> Note that Megacore is only currently available on TPU `v4` and TPU `v5p`. Supplying `dimension_semantics` annotations is a no-op on other platforms, but *not* specifying it will result in only one TensorCore being used (even if there are more than one available).

When using `pltpu.emit_pipeline`, `core_axis` should be passed into `emit_pipeline`. `core_axis` should be the index of a parallel grid axis to partition the grid on. For example, the following template can be used to partition the kernel over a leading parallel grid dimension:

```python
def kernel_body(...):
  def inner_pipeline_body(...):
    ...
  pltpu.emit_pipeline(inner_pipeline_body,
                      grid=(4, 4), 
                      core_axis=0,
                      dimension_semantics=("parallel", "sequential"))

pl.pallas_call(
      kernel_body,
      grid=(num_cores,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )
```
