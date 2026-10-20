---
scripts:
- scripts/tpu-kernel-development/verify_port.py
- scripts/tpu-kernel-development/capture_torch_golden.py
- scripts/tpu-kernel-development/assemble_test_harness.py
- scripts/tpu-knowledge-base/kb_lookup.py
- scripts/tpu-knowledge-base/catalog.py
- scripts/tpu-knowledge-base/kb_summarize.py
- scripts/tpu-knowledge-base/rank.py
references:
- references/tpu-kernel-development/concepts
---

# Analyze the PyTorch source

The kernel-worker reads this **once per run**, in Phase 0.5, at the very front of the
loop, on the run's **primary** source — the thing being converted.

Following it, you produce exactly two artifacts:

1.  **`<work>/torch_context.md`** — your understanding of the source, written
    for the planning step (Phase 1), which reads it every iteration.
2.  **`<work>/base.py`** — a faithful JAX reference implementation exposing a
    module-level `computation(...)`.

In this phase you do **not** write a Pallas kernel and you do **not** optimize
anything.

--------------------------------------------------------------------------------

## Inputs and outputs

Everything this phase produces lives under the kernel-worker's own `<work>`,
`{{runs}}/<run_id>/artifacts/tpu-kernel-development/<slot>`, the `work_dir` its request carried.

*   State file: `<work>/state.json` — read `primary.source_path`,
    `primary.golden_path`, `primary.golden_meta_path`, `atol`, `rtol`
*   Original source: `state.primary.source_path`
*   Golden manifest: `state.primary.golden_meta_path` (`<work>/torch_golden.json`)
*   Context brief output: `<work>/torch_context.md`
*   JAX reference output: `<work>/base.py`

**Do NOT read `<work>/ref/`.** That directory holds the reference kernel,
which is advisory and belongs to a different chain of custody. Your job is to
record what the *primary* computes; letting a reference influence that would
make the run's baseline a blend of two sources and its measurements
meaningless.

Read only `<work>` (excluding `ref/`), `state.primary.source_path`, the
shipped kernel concepts under `{{reference:tpu-kernel-development/concepts}}/` by
explicit path, and tpu-knowledge-base's pages, each through `python3
{{script:tpu-knowledge-base/kb_lookup.py}} <keyword>` and the path it
prints. Never write into the installed package tree (`{{package}}`), which is read-only
shipped content.

Leave `state.json` alone while following this reference. Phase 0.5 records
the artifacts after verifying them on disk.

--------------------------------------------------------------------------------

## Step 1: Read the golden manifest FIRST

Before you read a line of the source, read `state.primary.golden_meta_path`.

`{{script:tpu-kernel-development/capture_torch_golden.py}}` already executed the user's module
on CPU and recorded what it actually does. That manifest is **measured fact**,
and it outranks your reading of the code wherever the two disagree. It gives
you:

*   `entry_point.name` and `entry_point.kind` — what was actually called.
*   `entry_point.parameter_order` — every `nn.Parameter` and buffer, in the
    order they must appear as arguments.
*   `entry_point.module_scalars` — constructor scalars like `eps` that live on
    `self` and must become `static_argnums` arguments.
*   `configs[].dynamic` / `configs[].static` — the exact argument list, with
    `argnum`, shape and dtype for every one.
*   `configs[].outputs` — how many tensors come back, with shapes and dtypes.
*   `configs[].tolerance_recommendation` — the source's own fp32-vs-fp64
    spread. Advisory; the run's tolerances come from the caller.

**This settles the signature question that ruins most PyTorch ports.** A
`nn.Module` reads its weights from `self`; a JAX `computation` has no `self`,
so every parameter becomes an argument. The manifest tells you exactly which,
in exactly what order.

If the manifest is absent (golden capture degraded), say so explicitly in
Section 8 and derive the signature from the source — but flag it as
unverified, because the port then has no independent check at all.

## Step 2: Read the source, completely

Read every file under `state.primary.source_path`. Read the `forward()` body,
the `__init__`, any helper functions, and any `get_inputs()` /
`get_init_inputs()` the file ships.

If `state.primary.source_path` is missing or empty, STOP and return an
`invalid_input` `kernel-worker-result` naming the path. Do not search for a replacement and do not
invent a module.

## Step 3: Write the context brief

Write `<work>/torch_context.md` with the sections below. Two rules govern the
whole document:

*   **Separate *what* from *how*.** The mathematics survives the port to TPU;
    the op-by-op decomposition that XLA will re-fuse does not. Sections 1–3
    must fully determine the output and be readable by someone who has never
    used PyTorch.
*   **Be specific and quote the source.** Name real identifiers, real shapes,
    real constants, real line ranges. "It normalizes" is useless; "RMS over the
    last axis of `(4, 256, 2048)`, accumulating in fp32 from a bf16 input,
    `model.py:18-21`" is what the planner can act on.

```markdown
# PyTorch Source Context: <module name>

## 1. Source Inventory
- Files read, with line counts.
- The entry point: class and method, or function, with its full signature.
- Constructor parameters vs forward arguments — which is which, and why.
- Any `get_inputs()` / `get_init_inputs()` the file ships, verbatim.

## 2. What It Computes (framework-independent specification)
- The mathematics, as equations over named tensors. No torch vocabulary.
- Every tensor: name, role (in / out / parameter / buffer / intermediate),
  logical shape, dtype.
- Boundary and edge-case behaviour: masking, padding, ragged tails, handling
  of non-divisible sizes.
- The exact reduction / accumulation order where it is numerically
  load-bearing.

## 3. Module Contract  ← the section that makes or breaks the port
Reproduce the argument table from the golden manifest, and commit to it:

| argnum | name | origin | kind | shape | dtype |
|---|---|---|---|---|---|
| 0 | x | forward argument | dynamic | (4, 256, 2048) | bfloat16 |
| 1 | weight | nn.Parameter | dynamic | (2048,) | float32 |
| 2 | eps | module attribute | **static** | scalar | float |

- Dynamic arguments are traced arrays. Static arguments are values the kernel
  branches on at trace time and are passed via `static_argnums`.
- Parameters and buffers are **dynamic arguments**, not constants. The JAX
  function has no `self` to read them from.
- State plainly what the module allocates or pre-transforms in `__init__`
  that the JAX side must receive instead.

## 4. Materialization Map
For each op in `forward()`, in order: its output shape, whether the
intermediate is materialized to HBM, and whether XLA would fuse it away.
Then, explicitly:
- **The fusion boundaries.** Where does XLA have to write a full intermediate
  to HBM and read it back? These are the round trips a Pallas kernel can
  collapse, and they are the likeliest place this kernel earns its keep.
- **Total HBM traffic per call**, as a number, and the arithmetic intensity
  (FLOPs / byte) that follows from it.

## 5. Numerics
- Every dtype transition: `.half()`, `.bfloat16()`, `.float()`, autocast
  regions, and where they sit relative to each reduction.
- Accumulator dtype of every reduction vs. the storage dtype.
- Any `torch.backends` flag, `allow_tf32`, or fused-kernel path that changes
  precision.
- The golden manifest's `tolerance_recommendation`, and whether you agree with
  it given what you just read.

## 6. Dynamic-Shape and Control-Flow Hazards
Anything with no clean TPU form, surfaced now rather than discovered at
compile time: `nonzero`, `masked_select`, boolean indexing, `.item()`,
`torch.where` with data-dependent shapes, Python loops over tensor values,
early returns conditioned on tensor contents. For each: what it does, and what
the TPU-shaped alternative would have to be.

## 7. PyTorch → TPU Translation Notes
For each mechanism actually present in this source, its TPU counterpart:

| PyTorch construct in this source | TPU / Pallas counterpart |
| --- | --- |
| A chain of elementwise ops | XLA already fuses these — no Pallas needed |
| A reduction followed by a broadcast multiply | One Pallas program per tile, reduction in VMEM |
| `nn.Linear` / `torch.matmul` | MXU; contracting dims want multiples of 128 |
| A materialized `(B,H,S,S)` attention score | The round trip a Pallas kernel exists to remove |
| `torch.cat` of tiles | Usually an artefact of the op graph; re-tile instead |
| `.contiguous()` / `.transpose()` | Layout is a `BlockSpec` concern, not an op |
| In-place ops (`add_`, `mul_`) | `input_output_aliases`, or nothing at all |

Then: **what is genuinely being computed that XLA handles badly**, and **what
is torch-API bookkeeping that must not be carried across**.

## 8. Baseline Port Notes
- How `<work>/base.py` maps onto the specification in Section 2.
- Deliberate differences and why they are semantically safe.
- Anything you could not port faithfully, stated plainly.
- Risk list: where a correctness failure most likely means the *port* is
  wrong rather than the Pallas kernel.
- Whether the golden manifest was available. If it was not, say so loudly:
  the port has no independent check.

## 9. Optimization Opportunities Visible From the Source
- Intermediates that get materialized to HBM for no reason.
- Fusion boundaries a single Pallas kernel could collapse.
- Work the TPU can restructure that the eager op-by-op model forced.
- Reuse the op graph throws away between calls.
```

## Step 4: Write the JAX reference `base.py`

`<work>/base.py` is the **denominator of every measurement in this run**.
Everything the loop reports is relative to it, so it must be right first and
unremarkable second.

Requirements:

1.  **Pure JAX.** `jax`, `jax.numpy`, `jax.lax`. **No Pallas, no
    `pallas_call`, no hand-tiling, no `torch`, no custom calls.** The baseline
    is what a competent engineer writes without a custom kernel; the loop's
    whole job is to beat it. This is also load-bearing for verification:
    `{{script:tpu-kernel-development/verify_port.py}}` runs `base.py` on the **CPU** backend,
    which only works because it is plain JAX.
2.  **A module-level `def computation(...)`.** Hard contract:
    `{{script:tpu-kernel-development/assemble_test_harness.py}}` binds it as
    `base_computation` and refuses the file without it.
3.  **Exactly the signature in Section 3** — same argument order, same
    `argnum` for every entry, array arguments first and static scalars last.
    Parameters and buffers are arguments, not module state and not constants.
4.  **Same dtypes, in and out**, as the manifest records. If the source stores
    bf16 and accumulates in fp32, the JAX must too
    (`preferred_element_type=jnp.float32`). A baseline that quietly computes
    in higher precision makes every later correctness check meaningless.
5.  **Straightforward and idiomatic.** Do not pessimize it to flatter the
    optimized kernel, and do not pre-optimize it either. No `jax.jit`
    decorator — the harness jits it.
6.  **Shape-commented** on every intermediate: `# Shape: (batch, seq, hidden)`.
7.  **Preserve the user's problem dimensions exactly.** Never substitute a
    convenient size for one the source declares.

## Step 5: Verify your own port before you finish

Syntax and entry point, locally:

```bash
python3 -c "import ast; s=open('<work>/base.py').read(); ast.parse(s); assert 'def computation' in s; print('base.py OK')"
```

Then, when a golden file exists, **check the port against it yourself**. The
check needs JAX, so it runs through the compute runtime like every other
execution in this package; `verify_port.py` pins `JAX_PLATFORMS=cpu` itself, so
it lands on the target's CPU and never touches a chip:

```bash
{{runtime}} compute run --code {{script:tpu-kernel-development/verify_port.py}} --timeout 600 \
  --input <work>/base.py:base.py \
  --input <work>/torch_golden.npz:torch_golden.npz \
  --input <work>/torch_golden.json:torch_golden.json \
  --output port_verification.json:<work>/port_verification.json \
  -- base.py torch_golden.npz torch_golden.json \
     --atol <state.atol> --rtol <state.rtol> --out port_verification.json
```

There is no reason not to: finding the bug here costs one edit, and finding it
in Phase 0.9 costs a repair pass and up to three attempts. If it reports
`PORT_VERIFIED: False`, read the diagnosis and **fix `base.py` now** — the
failing fraction tells you what kind of bug it is:

*   nearly all elements wrong → a wholesale semantic difference (wrong axis,
    wrong op, missing scale), not rounding;
*   a thin sliver wrong → boundary handling: masking, a ragged tail, an
    off-by-one, non-divisible tiling;
*   a broad middle → accumulator dtype or reduction order.

Do not run the test harness and do not profile anything here. Later phases do
that.

## Before you return to Phase 0.5

Both `<work>/base.py` and `<work>/torch_context.md` must exist and be
non-empty. Record in `<work>/tpu_kernel_debug_history.md`, in 2–3 sentences:
the entry point, the one-line specification of what it computes, whether
`verify_port.py` passed, and any porting risk you flagged in Section 8. Then
go back to Phase 0.5 step 3.

If the source was missing, unreadable, or exposed no entry point you could
identify, return an `invalid_input` `kernel-worker-result` naming what was missing.
If you could not produce a faithful port, say what defeated you in the debug
history rather than leaving a port you know is wrong: every number the run
reports would be measured against it.
