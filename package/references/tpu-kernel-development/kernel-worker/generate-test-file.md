---
scripts:
- scripts/tpu-kernel-development/assemble_test_harness.py
- scripts/tpu-kernel-development/capture_torch_golden.py
---

# Generate get_inputs()

The kernel-worker reads this in Phase 0.10 step 3 to write the **input generation
function** `get_inputs()` for testing a Pallas kernel. This runs ONCE per run,
before any optimization has started -- there is no optimized kernel to look
at, because none exists yet.

**This is the ONLY LLM-authored file in the shared test harness.** Everything
else -- inlining the base kernel, copying the fixed correctness/benchmark
logic, keeping the two isolated so a helper function named the same in both
kernels can't collide -- is handled deterministically afterward by
`{{script:tpu-kernel-development/assemble_test_harness.py}}` (plain file I/O and `exec()`-based namespace
isolation, no LLM involved). Do not try to do any of that here; do not
inline `{base_kernel_path}`'s source into your output, and do not read the
shared test-harness template at all -- the job in this step is `get_inputs()`
and nothing else.

## Finding the Base Kernel

**Step 1: Check Session State**

-   Base kernel: `{base_kernel_path}`

If the path is available → proceed to read it with `read` (to learn its
signature and shapes -- not to copy its source).

**Step 2: If Path is Missing**

**STOP immediately.** Phase 0.10 step 2 owed you `base.py`; return
`invalid_input` from the kernel-worker naming the missing path. DO NOT use
list_directory or search for files.

## Tool Usage

1.  `read`: To read the base kernel (to learn its signature/shapes only),
    `<work>/state.json`, and the serving-configuration files it names.
2.  `write`: To write `get_inputs()` to `{get_inputs_path}`, and, when the
    caller recorded the served workload, `<work>/input_provenance.md`.

## Your Task

1.  **Read the base kernel** (`{base_kernel_path}`) with `read` to
    understand:
    -   The function name and signature (the entry point is always named
        `computation`)
    -   Input shapes and types (especially `jax.numpy` arrays)
    -   Any configuration parameters (block_size, tile_size, etc.)

2.  **CRITICAL: Check for an existing input generation function or test
    cases.** If the base kernel already defines a `get_inputs()` (or similar)
    or specific test shapes, you MUST reuse those exact shapes/values --
    adapt them into the required format below rather than inventing new ones.

    Read `state.primary.language` from `<work>/state.json`. When it is
    `"pytorch"` or `"cuda"`, `base.py` is a port, and the canonical shapes live
    in the *original* source, not in it.

    **Prefer recorded shapes over described ones.** When
    `state.primary.golden_meta_path` is set, read that JSON first
    (`<work>/torch_golden.json`). It is not a description of the inputs -- it
    is a record of the tensors that were actually passed when the user's module
    was executed, produced by `{{script:tpu-kernel-development/capture_torch_golden.py}}`. For
    each config it gives, per argument: `argnum`, `shape`, `dtype`, and whether
    the argument is dynamic (a traced array) or static (a `static_argnums`
    value the kernel branches on at trace time). It also names every
    `nn.Parameter` and buffer, which are **dynamic arguments** on the JAX side
    because `computation` has no `self` to read them from.

    Build `get_inputs()` directly from that manifest: one
    `(dynamic_args, static_args)` tuple per config, with the arguments in
    `argnum` order and the same dtypes. This is the one place in the loop where
    the shapes are known rather than inferred.

    Only when the manifest is absent (a degraded golden capture) fall back to
    prose: read `state.primary.context_path` -- Section 2
    lists every tensor's shape and dtype, and Section 3 gives the argument
    contract. Reuse those shapes and dtypes exactly; translate `torch` dtypes
    to their `jnp` equivalents and nothing more. Never substitute your own
    dimensions.

2b. **Build from the served workload first, when the caller recorded it.**
    Read `state.served_calls` from `<work>/state.json`, and the two files
    `state.serving_config` names (`deployments_json`, `launch_recipe`) with
    `read`. When either is set, the first configs describe the calls the
    served model makes, and **every value in them has a source you can name**.
    In order of authority:

    1.  **`state.served_calls`** -- measured. Each of its `cases` is a call the
        server made to this kernel, taken from a profile: `call_shapes` lists
        the dtype and shape of every operand and output, in order. Make one
        `(dynamic_args, static_args)` config per case, **before** any other
        config, with a comment above it naming the case's `label`. A shape
        may end in a layout in braces, e.g. `bf16[8,2048,4,2,128]{4,3,2,1,0}`:
        build the array from the dtype and the dimensions before the braces
        -- `get_inputs()` cannot set the layout -- and copy the layout into
        `input_provenance.md` beside the operand, so the layout the production
        call received stays on record.
    2.  **The serving configuration** -- configured. The deployments manifest
        and the launch recipe give what the shapes do not: the maximum context
        length (with the page size, the pages per sequence, and the longest
        context a sequence can reach), the tensor and data parallelism (and so
        how many heads one device holds), the dtypes, the attention backend,
        and the engine flags that were set. A flag that is in neither file was
        left at the engine's default: say so, and take its value from the
        shapes when they show it (a per-sequence array of length 256 means 256
        sequence slots) rather than guessing the default.
    3.  **The base kernel itself** -- its signature, its defaults, its tuned
        tables -- for anything the caller's artifacts leave open.
    4.  **An assumption** -- only for what none of the above fixes, and always
        written down with the reason.

    Never take a value from the caller's `prompt` prose when one of 1-3 gives
    it, and never invent one silently.

    -   **Match the shapes to the arguments by reading the base kernel**, not
        by position alone. The shapes are of the compiled call, which can
        differ from `computation`'s arguments: a wrapper may reshape or pad
        before the custom call (e.g. merging K and V, or packing heads), and
        may pass extra scalar operands of its own. Work out which operand each
        argument becomes, then set every argument dimension the shapes fix --
        token counts, sequence slots, page or block counts, head counts, cache
        size. When the shapes and the serving configuration disagree (e.g. on
        heads per device), the shapes win, because they were measured: record
        the disagreement. The cases cover every call to the kernel's file, so
        one may be a call `computation` does not make -- another kernel the
        same file defines: build no config for it, and say why in
        `input_provenance.md`.
    -   **Values neither artifact can fix** -- how many of the sequence slots
        are live, each sequence's query and context length, a batch split such
        as a `distribution` array -- come from the case's `runtime_values` when
        it has them. These are what the server's metrics endpoint reported
        over the capture window, keyed by its own metric names, not values per
        call and not per device: take the live sequences from its
        running-request readings, divided across the data-parallel replicas
        when the configuration has them (one config at the typical reading, a
        second of the same shape at the highest when they differ), and context
        lengths from its prompt and generation token counts, and name the key
        you read in `input_provenance.md`. When it does not have them, choose
        them from the case's `load` and the configuration's maximum context,
        e.g. context lengths around the prompt length plus the tokens
        generated so far. When the number of
        live sequences is unknown, do not fill every slot: build the case at a
        light occupancy, with a second config of the same shape at full
        occupancy, and label both. A full call and a mostly padded call of the
        same shape can differ several-fold in time.
    -   **Keep the production sizes**, including the cache, unless they do not
        fit in device memory -- the harness donates in-place buffers, so a
        large cache is not copied on every call. If you shrink a dimension,
        record it in `input_provenance.md` with the reason: that case no
        longer times the call the server makes.
    -   **Write `<work>/input_provenance.md`**: one row per config and per
        argument dimension or value you set, with the value and its source --
        `served_calls` case and operand (with its layout, when it has one), a
        `runtime_values` key, a manifest key, a recipe flag, a base kernel
        default, or "assumption" with the reason. Repeat the source as
        a short comment beside the value in `get_inputs()`.

    The configs from step 2's sources (an existing `get_inputs()`, the golden
    manifest) and edge cases come **after** the served-call configs, never in
    their place. Take the edge cases from the configuration where it gives
    them -- a sequence at the maximum context, the batch sizes the server
    allows -- rather than from round numbers.

3.  **Write `get_inputs()`**:
    -   Import necessary libraries (`jax`, `jax.numpy as jnp`).
    -   Define `def get_inputs():` returning a **list of
        `(dynamic_args, static_args)` tuples**.
        -   `dynamic_args`: arrays/tensors the kernel is JIT-traced over.
        -   `static_args`: scalars/config values (block sizes, etc.) passed as
            `static_argnums` -- values the kernel branches on at trace time,
            not array data.
    -   Cover multiple sizes and edge cases (zeros, ones, random inputs) if no
        existing input generator was found.
    -   Example:
        ```python
        import jax
        import jax.numpy as jnp

        def get_inputs():
            key = jax.random.PRNGKey(0)
            cases = []

            x1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
            y1 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
            cases.append(([x1, y1], []))

            x2 = jnp.zeros((256, 256), dtype=jnp.float32)
            y2 = jnp.zeros((256, 256), dtype=jnp.float32)
            cases.append(([x2, y2], []))

            return cases
        ```
    -   Your output must contain ONLY imports and this one function -- no
        base-kernel code, no harness code, no `opt_computation` stub.

## Output Format

Use the `write` tool to write the snippet above to `{get_inputs_path}`
(NOT `{test_file_path}` -- Phase 0.10 step 4 assembles the final harness at
`{test_file_path}` from this file deterministically, in a separate step).

Then return to Phase 0.10 step 4.
