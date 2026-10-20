---
title: "Pallas kernels as PyTorch custom ops (jax_op)"
type: doc-concept
provenance: doc
source: docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md
sources: [docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md, docs/markdown/custom_pallas_kernel.md, docs/markdown/quantized_sum.md, docs/markdown/custom_ops_via_pallas.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Pallas kernels as PyTorch custom ops (jax_op)

## Definition
Pallas is the JAX-based kernel language for TPUs. It gives "direct control over the TPU's memory
hierarchy and vector units while remaining in pure Python", and the user guide presents it as
the tool for new research ideas or for "more fine-grained control over the code executed on the
TPU hardware". TorchTPU integrates it through `torch.tpu.pallas.jax_op`: you write a Pallas
kernel, wrap it in a JAX function with `pl.pallas_call`, and `jax_op("ns::name", jax_fn)` hands
back a PyTorch custom op that you call like any other function on `tpu` tensors. JAX must be
installed (`pip install jax`).

The guide's mental model of the kernel body:
- **`_ref` arguments** are references to local VMEM. In the guide's examples Pallas moves the data
  from HBM into VMEM for you.
- **`ref[...]`** loads the whole local tile into registers, and `o_ref[...] = ...` stores it back.
- **Output shapes** come from the `out_shape` of `pl.pallas_call`, given as a
  `jax.ShapeDtypeStruct`. For several outputs, pass a list of them; the op then returns a tuple,
  as in the guide's `add_subtract` example.

## In torch_tpu (grounded)
The public module `torch.tpu.pallas` is declared in the public-API table
(`src/torch_tpu/_internal/public_api.py`) and exports exactly two names: `jax_op` and
`get_pallas_mesh`. Both are implemented in the lazily imported `torch_tpu._internal.pallas`
package. The deprecated `custom_jax_kernel` stays internal.
[`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op")
is marked experimental. It rejects names that lack exactly one `::`. It checks the JAX function's
annotations ([`_verify_signature`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L498 "src/torch_tpu/_internal/pallas/pallas#_verify_signature"))
and treats every non-tensor parameter as static
([`_infer_static_argnums`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L591 "src/torch_tpu/_internal/pallas/pallas#_infer_static_argnums")).
It then wraps the function in a
[`JaxCallable`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L930 "src/torch_tpu/_internal/pallas/pallas#JaxCallable")
through [`custom_jax_kernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1399 "src/torch_tpu/_internal/pallas/pallas#custom_jax_kernel")
and registers the result as a `torch.library.custom_op`, using a schema produced by
[`_generate_torch_schema`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L897 "src/torch_tpu/_internal/pallas/pallas#_generate_torch_schema").
The return value is a
[`_JaxCustomOpDef`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L918 "src/torch_tpu/_internal/pallas/pallas#_JaxCustomOpDef"),
which is the `CustomOpDef` the guide later calls `register_autograd` on.

What happens on a call:
1. **Export once.** The `JaxCallable` constructor runs `jax.export` on the function for the TPU
   platform.
2. **Specialize per call signature.** On a cache miss,
   [`JaxCallable.__call__`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1198 "src/torch_tpu/_internal/pallas/pallas#JaxCallable.__call__")
   lowers the export for that signature. The cache key comes from
   [`_get_kernel_invocation_key`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L290 "src/torch_tpu/_internal/pallas/pallas#_get_kernel_invocation_key")
   and covers shapes and static args.
3. **Register the module.** The StableHLO module is fingerprinted with location info stripped,
   then registered with
   [`PyRegisterCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/pallas/pallas_py.cc#L49 "csrc/internal/pallas/pallas_py.cc#PyRegisterCustomKernel"),
   which stores it in the process-wide registry through
   [`RegisterCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L598 "csrc/ops/custom_kernels.cc#RegisterCustomKernel").
4. **Build output placeholders.** The traced output avals become output placeholders
   ([`torch_placeholder`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L220 "src/torch_tpu/_internal/pallas/pallas#torch_placeholder")).
5. **Call the kernel.** Execution goes through
   [`PyCallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/internal/pallas/pallas_py.cc#L129 "csrc/internal/pallas/pallas_py.cc#PyCallCustomKernel").
   On the C++ side,
   [`CallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L608 "csrc/ops/custom_kernels.cc#CallCustomKernel")
   names the specialization by input dims and dtypes
   ([`GetUniqueCustomKernelName`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L94 "csrc/ops/custom_kernels.cc#GetUniqueCustomKernelName")).
   It then emits a `func.call` to that function inside the program being built, so the kernel is
   deferred and fused with the surrounding ops like any other op.

The `JaxCallable` docstring explains why `out_shape` is enough: output types are "determined via
tracing, instead of requiring they be passed in explicitly". The guide's prose still mentions an
`output_shapes` argument, but `jax_op` has no such parameter at this pin. `output_shapes` exists
only internally, as the placeholder list handed to `call_custom_kernel`.

## Plain `jax.lax` functions (no Pallas)
`jax_op` does not require a Pallas kernel. Any JAX function that exports to StableHLO works. The
`quantized_sum` chapter uses this to reach XLA ops that PyTorch does not spell: "JAX ultimately
compiles down to the same HLO as TorchTPU", and `jax.lax` maps nearly 1:1 onto StableHLO.
- **The example.** `jax.lax.population_count` lowers straight to StableHLO `popcnt`.
  `jax_op("test_example::popcount", jax_popcount)` wraps it, and the caller "has no visibility
  into the fact that the op is implemented in terms of JAX". The resulting op sums one-bit
  packed weights without unpacking them
  ([one-bit-quantization-example](one-bit-quantization-example.md)).
- **Same path as a Pallas kernel.** The plain function goes through the same export, register
  and `CallCustomKernel` path described above. In an HLO dump the JAX function appears as its
  own computation inside a `*tt_jit_custom_kernel*` module, with `jit(jax_popcount)/...`
  `op_name` metadata ([xla-compilation-dumps](xla-compilation-dumps.md)).
- **No automatic gradient.** `population_count` is not differentiable, so this op is for
  inference. The chapter suggests `jax.lax.approx_max_k` as a more realistic candidate.
- **Backward through JAX.** The canonical example of deriving a backward with
  `jax.jvp`/`jax.grad` is `test_jax_dot_grad_for_backwards`, at `tests/pallas/pallas_test.py` at
  this pin (the chapter gives the older `torch_tpu/tests/...` path). See also
  [pallas-kernel-autograd](pallas-kernel-autograd.md).

## Why it matters / when it applies
Use this when an op has no good ATen lowering, or when profiling shows that the compiler's
fusion falls short and you need to control tiling and memory movement yourself. The guide covers
only the TorchTPU integration and sends readers to the Pallas Quickstart for kernel-writing
technique. The guide does not mention three behaviours visible in the code:
- `jax_op` also accepts `donate_argnums`. Donated inputs are left invalid after the call.
- A distributed kernel can pass `mesh` together with input and output partition specs.
- Each new input shape triggers a fresh trace and registration. Later calls with the same
  signature reuse the registry entry.

## Case study: register-level fusion (quantize and pack in one kernel)
"Custom Ops via Pallas" is the advanced follow-up. It is about performance: Pallas, "like the
Triton CUDA kernel", gives control of the TPU memory hierarchy (HBM, VMEM, registers). The
example is inspired by DeepSeek V3's `act_quant` Triton kernel, which quantizes activations
blockwise to fp8 with an f32 scale per block. A composition-based version needs two passes, one
to binarize and one to pack bits. That writes the binarized tensor to HBM "only to read them
back immediately for packing", which "creates a severe memory bottleneck". The Pallas version
loads the activations once, computes the scale (`mean(|x|)`), turns signs into bits, shifts and
sums them into a `uint32` per row "entirely within the TPU registers", and writes once. It has
two outputs (`packed`, `scale`), declared as a tuple of `jax.ShapeDtypeStruct` in
`pl.pallas_call`. Everything after the
[`jax_op`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/pallas/pallas.py#L1547 "src/torch_tpu/_internal/pallas/pallas#jax_op")
wrap (`"testing_op::quant_and_pack"`) is pure PyTorch. For a row `[-7, 7, 1, -1] * 8` the
expected results are `scale == 4.0` and `packed == 0x66666666`.

The method the page recommends, whatever the kernel:
- Write the unoptimized baseline in plain PyTorch first. It sets the performance bar and is the
  numerical reference. Profile it before deciding to write a kernel.
- Check the result in the HLO and LLO dumps (the page sets `XLA_FLAGS` and `LIBTPU_INIT_ARGS`
  as in [xla-compilation-dumps](xla-compilation-dumps.md)).
- Because [`CallCustomKernel`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/custom_kernels.cc#L608 "csrc/ops/custom_kernels.cc#CallCustomKernel")
  emits the kernel as a `func.call` inside the surrounding program, the custom op is deferred
  with its neighbours. The fusion you write inside the kernel adds to whatever XLA does around
  it. It does not replace it.

## Connections
- Code concepts: [pallas-custom-kernels](../concepts/pallas-custom-kernels.md) (the full
  `jax_op` / `JaxCallable` mechanism),
  [scan-and-custom-kernels](../concepts/scan-and-custom-kernels.md) (the C++ registry and the
  inlining of `CallCustomKernel`),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [compilation-cache](../concepts/compilation-cache.md) (where the fused program is cached).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/pallas`,
  `csrc/internal/pallas`, `csrc/ops`).
- Related doc-concepts: [pallas-ops-under-torch-compile](pallas-ops-under-torch-compile.md),
  [pallas-kernel-autograd](pallas-kernel-autograd.md),
  [extending-pytorch-on-tpu](extending-pytorch-on-tpu.md),
  [aten-to-llo-lowering-path](aten-to-llo-lowering-path.md),
  [xla-compilation-dumps](xla-compilation-dumps.md).

## Source
Extracted from `docs/markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md`
(kept in place).
`docs/markdown/custom_pallas_kernel.md` is an older, unnumbered export of the same Pallas chapter with identical content (kept in place).
The "Plain `jax.lax` functions" section is from `docs/markdown/quantized_sum.md` (kept in place).
The "Case study: register-level fusion" section is from `docs/markdown/custom_ops_via_pallas.md` (kept in place).
