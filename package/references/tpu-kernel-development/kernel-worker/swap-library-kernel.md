---
scripts:
- scripts/tpu-kernel-development/tune_library_kernel.py
---

# Use the Tokamax kernel

Read this in Phase 1 when `state.library_check.result` is `"use"`. It replaces
planning and implementing.

The user chose an existing Tokamax kernel, named in
`state.library_check.module` (for example `tokamax.ragged_dot`). Instead of
writing a kernel, you write a small file that calls it. The rest of the
iteration then tests and measures that file like any other kernel.

## Step 1: Read the op's source

Look at the Tokamax source installed on the TPU before writing anything. Print
it with `{{runtime}} compute run`, for example:

```python
import inspect
import tokamax._src.ops.ragged_dot.api as api
print(inspect.getsource(api))
```

Some ops have no `api.py` (for example `gmm_v2` and `fused_moe`). For those,
also print the op's test file, which shows how the kernel is called.

If the import fails, see **If Tokamax or the op is missing** below.

## Step 2: Write the kernel file

Write `{optimized_kernel_path}` with a module-level `computation` function. It
takes the inputs that `{test_file_path}`'s `get_inputs()` produces, returns
the same output as `{base_kernel_path}`, and calls Tokamax:

```python
import jax.numpy as jnp
import tokamax


def computation(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    return tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="mosaic")
```

Guidelines:

-   Use the top-level function (`tokamax.ragged_dot`, `tokamax.layer_norm`,
    and so on) when there is one. Experimental ops don't have one, so import
    them from `tokamax._src` and note that in the plan.
-   Pass the TPU `implementation=` value, as in the example, so the call runs
    the TPU kernel and not plain XLA. If `state.library_check.module` names a
    value, use that one.
-   Match `base.py`'s call exactly, including the output dtype
    (`preferred_element_type`). Only reshape or cast around the call as much
    as `base.py`'s inputs and output require.
-   Don't add a `kernel` function or error handling.

Then write `{kernel_plan_path}` with three lines:

1.  The call you made.
2.  `Optimization Base Choice: base`
3.  A note that the kernel comes from Tokamax.

## Step 3: Compile and test (Phases 2 and 3)

Run Phases 2 and 3 as usual. The test checks the Tokamax kernel against
`base.py` within `atol`/`rtol` and measures its speedup. These are the untuned
numbers.

## Step 4: Tune (replaces the Phase 4 sweep)

Tokamax's default settings are not tuned for these shapes, so the kernel can
run much slower than it should. Tune it with Tokamax's own autotuner.

1.  Run the tuning script on the `<work>/iter<n>/test_run.py` that Phase 3
    assembled:

    ```bash
    {{runtime}} compute run --code {{script:tpu-kernel-development/tune_library_kernel.py}} --timeout 1800 \
      --input <work>/iter<n>/test_run.py:test_run.py \
      --output tokamax_tuning.json:<work>/iter<n>/tokamax_tuning.json \
      -- test_run.py tokamax_tuning.json --budget 1500
    ```

    The script stops after `--budget` seconds and saves whatever it has
    measured so far. Keep the budget about 300 seconds below `--timeout`.

    For each call shape it prints a `BEST` line with the fastest config,
    marked `(from Tokamax's cache)` when Tokamax had already tuned that shape.
    A call whose every config failed gets a `NO_CONFIG` line instead: leave
    that shape on its default config. The last line says whether tuning
    finished, stopped early, or was not needed. Copy those lines into
    `{autotune_summary_path}`. For example:

    ```
    BEST lhs=bfloat16[8192, 2048], rhs=bfloat16[8, 2048, 4096]: tokamax._src.ops.ragged_dot.pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=Config(tile_m=256, tile_k=2048, tile_n=2048, input_buffer_count=3, combine_scopes=False))
    ```

2.  Pin that config in `{optimized_kernel_path}`: create the op from the
    `BEST` line once, and pass it as the implementation. Keep the rest of the
    call unchanged.

    ```python
    import tokamax
    from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as tpu

    _OP = tpu.PallasMosaicTpuRaggedDot(
        config=tpu.Config(tile_m=256, tile_k=2048, tile_n=2048,
                          input_buffer_count=3, combine_scopes=False),
        bypass_device_check=True,
    )


    def computation(lhs, rhs, group_sizes):
        return tokamax.ragged_dot(lhs, rhs, group_sizes, implementation=(_OP,))
    ```

    If the `BEST` lines give different configs for different shapes, create
    one op per shape and pick it inside `computation` with a plain `if` on
    the input shapes.

3.  Re-run Phase 3 steps 1 and 2 on the tuned file. Use its `SPEEDUP` and
    `RESULT_TIME` in Phase 5. Write both the untuned and tuned numbers to
    `{autotune_summary_path}`.

If tuning fails, or the tuned file fails the test, put the untuned file back,
keep the Phase 3 numbers, and write down why in `{autotune_summary_path}`.

Phase 4.5 then captures a trace of whichever file you kept.

## If Tokamax or the op is missing

If the import fails with `No module named 'tokamax'`, or the installed version
doesn't have this op, editing the file won't help. Skip the compile fixer, go
to Phase 5 with `compile_ok` set to false, and return `failed` with
`failure_reason: "compilation_error"`.

In `error_message`, say what is missing and how to fix it. Tokamax is not part
of the package's TPU requirements. Give the user this command for the TPU
environment; it keeps the `xprof` that tpu-diagnostics uses at the package's
floor:

```bash
pip install tokamax==0.0.15 'xprof>=2.24.0'
```

Leave the install to the user.
