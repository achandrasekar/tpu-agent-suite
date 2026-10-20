# Routing map

A kernel-development question → the *one* upstream file or directory that answers it.

**Path convention.** Every path here starts at one of the two clones under `<code-reference>`:
`<jax>` is the `jax` clone and `<tokamax>` is the `tokamax` clone, as in
`<jax>/tests/pallas/tpu_gmm_test.py` and `<tokamax>/tokamax/_src/ops/ragged_dot/`. The doubled
`<tokamax>/tokamax` is real — the package sits inside the repo of the same name. In a citation you
return, the path is relative to `<code-reference>`, so its first segment is `jax` or `tokamax`.
The shell commands run from the workspace root, `$ROOT`, so they name the clones
`code-reference/jax` and `code-reference/tokamax`.

**The contract, restated.** Grep the named subtree, Read a window around the hit, open at most five
files, and return `path:line-line@sha` citations rather than pasted source. Everything `gpu_*`,
`mgpu_*`, `pallas_triton*` or `pallas_mosaic_gpu*` is a different compiler and is not yours.

**Provenance.** The routes below are built from the directory listings of both repos at
`openxla/tokamax@main` and `jax-ml/jax@main`, read 2026-09-10 — the *paths* are verified to exist.
The one-line annotations of what each file contains are derived from its name and its siblings, not
from reading every file. Treat an annotation as a hypothesis worth one grep, and when one turns out
wrong, fix the row — see § Gaps.

## Fast lookup

| Question | jax | tokamax |
| --- | --- | --- |
| A first Pallas TPU kernel, any shape | `<jax>/jax/experimental/pallas/ops/tpu/example_kernel.py`; `<jax>/docs/pallas/quickstart.md` | — |
| Grid, `BlockSpec`, index maps | `<jax>/docs/pallas/grid_blockspec.md`; `<jax>/tests/pallas/pallas_test.py` | — |
| Mosaic TPU semantics — DMA, semaphores, scratch, `pltpu` primitives | `<jax>/tests/pallas/tpu_pallas_test.py` (grep only) | — |
| Manual pipelining, `emit_pipeline`, double buffering | `<jax>/tests/pallas/tpu_pallas_pipeline_test.py`; `<jax>/docs/pallas/pipelining.md` | — |
| Async copies, overlapping a DMA with compute | `<jax>/tests/pallas/tpu_pallas_async_test.py` | — |
| Memory spaces — VMEM, SMEM, HBM, `ANY` | `<jax>/tests/pallas/tpu_pallas_memory_space_test.py` | — |
| Stateful refs, accumulation across grid steps | `<jax>/tests/pallas/tpu_pallas_state_test.py` | — |
| Ref indexing, slicing, strided loads | `<jax>/tests/pallas/indexing_test.py` | — |
| **SparseCore** Pallas kernel | **Start at `<jax>/docs/pallas/tpu/sparsecore.md`** — it carries runnable worked examples, each ending in its own assertion. Then `<jax>/tests/pallas/tpu_sparsecore_pallas_test.py`; API surface is the 24 re-exports in `<jax>/jax/experimental/pallas/tpu_sc.py` | none — tokamax has no SparseCore op |
| SparseCore across devices | `<jax>/tests/pallas/tpu_sparsecore_pallas_distributed_test.py` | — |
| SparseCore debug checks / assertions in-kernel | `<jax>/tests/pallas/tpu_sparsecore_pallas_debug_check_test.py` | — |
| **Grouped GEMM / MoE / ragged dot** | `<jax>/jax/experimental/pallas/ops/tpu/megablox/`; `<jax>/tests/pallas/tpu_gmm_test.py` | `<tokamax>/tokamax/_src/ops/ragged_dot/` |
| Plain matmul kernel, tiling and accumulation | `<jax>/jax/experimental/pallas/ops/tpu/matmul.py` | — |
| Fusing a matmul with its epilogue | `<jax>/tests/pallas/tpu_fusible_matmul_test.py`; `<jax>/jax/experimental/pallas/fuser.py`; `<jax>/tests/pallas/fusion_test.py`, `fuser_block_spec_test.py` | — |
| **Splash / flash attention** | `<jax>/jax/experimental/pallas/ops/tpu/splash_attention/`, `flash_attention.py`; tests `tpu_splash_attention_kernel_test.py`, `tpu_splash_attention_kernel_sharded_test.py` | `<tokamax>/tokamax/_src/ops/attention/`; `<tokamax>/docs/splash_attention.md` |
| Attention **masks** — causal, local, custom | `<jax>/tests/pallas/tpu_splash_attention_mask_test.py` | — |
| **Paged attention** (vLLM decode) | `<jax>/jax/experimental/pallas/ops/tpu/paged_attention/`; `<jax>/tests/pallas/tpu_paged_attention_kernel_test.py` | — |
| **Ragged paged attention** | `<jax>/jax/experimental/pallas/ops/tpu/ragged_paged_attention/`; `<jax>/tests/pallas/tpu_ragged_paged_attention_test.py` | — |
| Arbitrary attention variants / score mods | — | `<tokamax>/tokamax/_src/ops/flex_attention/` — **GPU only**, no TPU implementation |
| In-kernel **collectives**, cross-device Pallas | `<jax>/tests/pallas/tpu_pallas_distributed_test.py`; `<jax>/tests/pallas/tpu_all_gather_test.py`; `<jax>/jax/experimental/pallas/ops/tpu/all_gather.py` | — |
| Multi-program / MPMD | `<jax>/tests/pallas/tpu_pallas_mpmd_test.py` | — |
| Debugging a kernel without a TPU (interpret mode) | `<jax>/tests/pallas/tpu_pallas_interpret_test.py`; `<jax>/tests/pallas/tpu_pallas_interpret_distributed_test.py` | — |
| Reading an error out of a failing kernel; `pl.debug_print` | `<jax>/tests/pallas/pallas_error_handling_test.py`; `<jax>/tests/pallas/tpu_pallas_call_print_test.py` | — |
| "Does this `jnp` op lower on TPU inside a kernel?" | `<jax>/tests/pallas/tpu_ops_test.py`; `<jax>/tests/pallas/ops_test.py` | — |
| Cost model / `CostEstimate` on a `pallas_call` | `<jax>/tests/pallas/pallas_cost_estimate_test.py` | — |
| In-kernel RNG | `<jax>/jax/experimental/pallas/ops/tpu/random/`; `<jax>/tests/pallas/tpu_pallas_random_test.py` | — |
| `vmap` over a kernel | `<jax>/tests/pallas/pallas_vmap_test.py` | — |
| Shape polymorphism / dynamic shapes | `<jax>/tests/pallas/pallas_shape_poly_test.py` | — |
| **Ragged gather / scatter / gather-reduce** | — | `<tokamax>/tokamax/_src/ops/ragged_gather/`, `ragged_scatter/`, `ragged_gather_reduce/` — all three ship a real TPU Pallas kernel |
| RMSNorm / LayerNorm kernel | — | `<tokamax>/tokamax/_src/ops/normalization/` — **GPU only**, no TPU implementation |
| SwiGLU / gated MLP | — | `<tokamax>/tokamax/_src/ops/gated_linear_unit/` — **GPU only**, no TPU implementation |
| Fused linear + softmax cross-entropy | — | `<tokamax>/tokamax/_src/ops/linear_softmax_cross_entropy_loss/` |
| Mamba-style causal conv1d / gated delta rule | — | `<tokamax>/tokamax/_src/ops/causal_conv1d_gated_delta_rule/` |
| Triangle multiplicative update (AlphaFold-shaped) | — | `<tokamax>/tokamax/_src/ops/triangle_multiplication/` — **GPU only**, no TPU implementation |
| A kernel's **backward pass / VJP** on TPU | — | `<tokamax>/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py` — the only TPU VJP in the tree |
| **Autotuning** a kernel's block sizes | — | `<tokamax>/tokamax/_src/autotuning/`; `<tokamax>/tokamax/autotuning.py`; `<tokamax>/docs/autotuning.md` |
| **Benchmarking** a kernel | — | `<tokamax>/tokamax/_src/benchmarking.py`; any op's `bench.py`; `<tokamax>/docs/benchmarking.md` |
| In-kernel quantization, fp8/int8 scales | — | `<tokamax>/tokamax/_src/quantization.py`; `<tokamax>/tokamax/_src/precision.py` |
| Inspecting the HLO a kernel lowers to | — | `<tokamax>/tokamax/_src/hlo_utils.py` |
| Mosaic-TPU helper layer above raw Pallas | — | `<tokamax>/tokamax/_src/mosaic_tpu.py` |
| Numerical tolerance / reference comparison helpers | — | `<tokamax>/tokamax/_src/numerics.py`; `<tokamax>/tokamax/_src/test_utils.py` |
| How to structure a new kernel **and its test** | — | § The tokamax per-op file convention, below |

## `<jax>/tests/pallas` — what each TPU-relevant file is for

Grep these; do not read them. Several are thousands of lines.

| file | what is in it |
| --- | --- |
| `tpu_pallas_test.py` | The main body of Mosaic-TPU semantics — the first place to grep for any `pltpu.*` primitive |
| `tpu_pallas_pipeline_test.py` | `emit_pipeline`, manual multiple-buffering, grid-loop structure |
| `tpu_pallas_async_test.py` | Async DMA start/wait, overlapping copy with compute |
| `tpu_pallas_memory_space_test.py` | VMEM / SMEM / HBM / `ANY` placement and the copies between them |
| `tpu_pallas_state_test.py` | Stateful refs and accumulators carried across grid steps |
| `tpu_pallas_distributed_test.py` | Cross-device kernels, remote DMA, device barriers |
| `tpu_pallas_mpmd_test.py` | Multi-program, heterogeneous per-device programs |
| `tpu_pallas_random_test.py` | RNG inside a kernel |
| `tpu_pallas_interpret_test.py` | Interpret mode — running a kernel with no TPU |
| `tpu_pallas_interpret_distributed_test.py`, `tpu_pallas_interpret_thread_map_test.py` | Interpret mode for the multi-device and thread-mapped cases |
| `tpu_pallas_call_print_test.py` | `pl.debug_print` and what actually reaches stdout |
| `tpu_pallas_state_test.py`, `tpu_side_effects_test.py`, `tpu_trace_value_test.py` | Effect ordering, side effects, traced values |
| `tpu_sparsecore_pallas_test.py` | **SparseCore** kernels — the single most relevant test file for SparseCore kernel work. A SparseCore kernel is `pl.kernel(..., mesh=ScalarSubcoreMesh\|VectorSubcoreMesh)`, **not** `pl.pallas_call`; the file's `PallasSCTest` harness and its `vector_subcore_kernel` decorator are the thing to subclass, and every assertion in it is bit-exact `assert_array_equal`, not a tolerance |
| `tpu_sparsecore_pallas_distributed_test.py`, `tpu_sparsecore_pallas_debug_check_test.py` | SparseCore across devices; in-kernel debug assertions |
| `tpu_gmm_test.py` | Grouped matmul (megablox) — group sizes, ragged boundaries, the reference implementation to diff against |
| `tpu_fusible_matmul_test.py`, `tpu_fusible_matmul_with_stateful_fusions_test.py` | Matmul with a fused epilogue, via the fuser |
| `fusion_test.py`, `fuser_block_spec_test.py`, `fuser_utils_test.py` | The fuser itself |
| `tpu_ragged_paged_attention_test.py` | Ragged paged attention — the closest upstream analogue of vLLM's decode kernel |
| `tpu_paged_attention_kernel_test.py` | Paged attention |
| `tpu_splash_attention_kernel_test.py`, `..._sharded_test.py`, `tpu_splash_attention_mask_test.py` | Splash attention, its sharded form, and the mask abstractions |
| `tpu_all_gather_test.py` | An all-gather written as a Pallas kernel |
| `tpu_ops_test.py`, `ops_test.py` | Which `jnp`/`lax` ops lower inside a kernel, and with what dtypes |
| `indexing_test.py` | Ref indexing and slicing rules |
| `pallas_test.py`, `pallas_vmap_test.py`, `pallas_shape_poly_test.py` | Backend-neutral `pallas_call` behaviour, `vmap`, polymorphic shapes |
| `pallas_cost_estimate_test.py` | The `CostEstimate` a kernel reports |
| `pallas_error_handling_test.py`, `pallas_lowering_determinism_test.py` | Error surfaces; lowering determinism |
| `tpu_info_test.py` | Querying TPU generation and limits from Python |

**Not yours:** every `gpu_*`, `mgpu_*`, `mosaic_gpu_test.py`, `triton_*`, and the `einshape_*` files.

## `<jax>/jax/experimental/pallas/ops/tpu` — the shipped kernels

These are implementations, not tests: read them for structure, read the test for the contract.

| path | what it is |
| --- | --- |
| `example_kernel.py` | The minimal worked example. Start here if you have never written one. |
| `matmul.py` | Blocked matmul — tiling, accumulation, the `BlockSpec` idiom |
| `flash_attention.py` | Flash attention, single file |
| `splash_attention/` | The production sparse-mask attention: kernel, mask abstractions, and its own masking DSL |
| `megablox/` | Grouped GEMM — the MoE workhorse. Pair with `<jax>/tests/pallas/tpu_gmm_test.py` |
| `paged_attention/` | Paged KV attention |
| `ragged_paged_attention/` | Ragged + paged — variable sequence lengths against a paged cache |
| `all_gather.py` | A collective expressed as a kernel; the model for in-kernel remote DMA |
| `random/` | In-kernel PRNG |

API surface lives one level up: `<jax>/jax/experimental/pallas/tpu.py` (`pltpu`),
`<jax>/jax/experimental/pallas/tpu_sc.py` (SparseCore), `<jax>/jax/experimental/pallas/fuser.py`.

## `<tokamax>/tokamax/_src/ops` — the op directories

Every directory follows the same convention (next section). TPU implementations are the
`pallas_mosaic_tpu*` files; the `pallas_mosaic_gpu*` and `pallas_triton*` siblings are not yours.
**TPU coverage is much thinner than GPU here — only 6 of the 11 ops have a TPU kernel at all.**
Enumerated at `6303c14`, 2026-09-10; re-run the loop in § Gaps when the clone moves.

| directory | op | TPU files |
| --- | --- | --- |
| `ragged_dot/` | Grouped/ragged matmul. **The reference for the per-op convention** | `pallas_mosaic_tpu{,_kernel,_test,_v2,_v2_test}.py` — two implementations side by side |
| `attention/` | Attention, incl. the splash-attention integration in `<tokamax>/docs/splash_attention.md` | `pallas_mosaic_tpu{,_common,_test,_vjp}.py` — the only TPU **VJP** in the tree |
| `ragged_gather/`, `ragged_scatter/`, `ragged_gather_reduce/` | The ragged data-movement primitives — closest upstream analogue of SparseCore gather/scatter work | each has `pallas_mosaic_tpu{,_kernel,_test}.py` |
| `linear_softmax_cross_entropy_loss/` | Fused final projection + CE loss | `pallas_mosaic_tpu{,_kernel,_kernel_test,_test}.py` |
| `causal_conv1d_gated_delta_rule/` | Mamba/GDN-style recurrence | `pallas_mosaic_tpu{,_test}.py` |
| `normalization/` | RMSNorm / LayerNorm | **none** — GPU only |
| `gated_linear_unit/` | SwiGLU-family gated MLP | **none** — GPU only |
| `flex_attention/` | Attention with pluggable score/mask modifications | **none** — GPU only |
| `triangle_multiplication/` | AlphaFold-shaped triangle update | **none** — GPU only |
| `experimental/` | Unstable; read for ideas, do not cite as settled | none |
| `op.py`, `op_test.py` | The `Op` framework every directory above plugs into | — |

Supporting modules worth knowing: `<tokamax>/tokamax/_src/test_utils.py` (the shared test helpers),
`numerics.py` (tolerance and reference comparison), `benchmarking.py`, `autotuning/`,
`quantization.py`, `precision.py`, `hlo_utils.py`, `mosaic_tpu.py`.

## The tokamax per-op file convention

This is the answer to "how do I lay out a new kernel and its test". Worked on
`<tokamax>/tokamax/_src/ops/ragged_dot/`, which carries every role:

| upstream file | role | copy it when |
| --- | --- | --- |
| `api.py` / `api_test.py` | Public entry point — dtype and shape validation, backend dispatch | the kernel is called from model code |
| `base.py` / `base_test.py` | The abstract op plus its config dataclass, shared by every backend | you will have more than one implementation |
| **`test_base.py`** | The shared correctness suite that each backend's test **subclasses** | **always.** This is the trick worth stealing: write the correctness suite once, against the mathematical contract, and let every variant inherit it. A variant cannot then be "tested" by a suite that only covers what that variant happens to do |
| `arg_specs.py` | The canonical shape/dtype set that the benchmark and the autotuner both consume | you want the benchmark and the test to agree on shapes instead of drifting apart |
| `pallas_mosaic_tpu.py` | The TPU wrapper — grid, block sizes, config plumbing | always |
| `pallas_mosaic_tpu_kernel.py` | The Pallas kernel body, and nothing else | always. Wrapper and body in separate files is the pattern; it is what lets a second body reuse the first wrapper |
| `pallas_mosaic_tpu_test.py` | Subclasses `test_base`, adds TPU-specific cases | always |
| `pallas_mosaic_tpu_v2.py` / `..._v2_test.py` | A second implementation living *beside* the first, not replacing it | you have a challenger variant |
| `bench.py` | The benchmark entry, over `arg_specs` | always |

Two things to carry away:

- **`test_base.py` is the correctness contract, and it is written before any backend exists.** The
  per-backend test file adds hardware-specific cases and inherits everything else. That is what
  makes "my new kernel passes the same suite the incumbent passes" a checkable claim.
- **`_v2` sits beside `_v1`.** Upstream keeps the old implementation importable, tested and
  benchmarkable rather than deleting it — the same discipline as the write-tpu-kernels
  rule that earlier variants stay beside new ones, and its "the incumbent is a candidate". An edit in place leaves no baseline surviving the
  edit.

## Docs worth reading whole

Short, and cheaper than reconstructing the same thing from tests:

- `<jax>/docs/pallas/quickstart.md` — the mental model
- `<jax>/docs/pallas/grid_blockspec.md` — grids and `BlockSpec`, the part people get wrong
- `<jax>/docs/pallas/pipelining.md` — the pipelining model
- `<jax>/docs/pallas/tpu/sparsecore.md` — the best SparseCore starting point in either repo. Its
  snippets are runnable: each ends in its own `np.testing.assert_array_equal` against a jnp
  reference, so the doc doubles as the minimal test. Also published at
  `https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html`
- the rest of `<jax>/docs/pallas/tpu/` — other TPU-specific pages
- `<tokamax>/docs/basic_usage.md`, `supported_ops.md` — what tokamax offers and how it is called
- `<tokamax>/docs/autotuning.md`, `benchmarking.md` — the two harnesses
- `<tokamax>/docs/splash_attention.md`, `<tokamax>/docs/ops/` — per-op notes

## Verifying a pinned citation from the write-tpu-kernels references

Those area files carry ~76 tags of the form
`` [source: `tokamax/_src/ops/<op>/<file>.py:588-596@9da2369`] `` and JAX tags of the form
`` [source: `jax/_src/pallas/mosaic/tpu_info.py@jax 0.10.2`] ``. Note the tokamax tags omit the outer
repo directory — `tokamax/_src/…` in a tag means `<tokamax>/tokamax/_src/…` on disk.

```bash
# tokamax — works directly, which is why tokamax is cloned with full history
git -C code-reference/tokamax show 9da2369:tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py \
  | sed -n '588,596p'

# jax — fetch the tag first
git -C code-reference/jax fetch --depth 1 origin refs/tags/jax-v0.10.2:refs/tags/jax-v0.10.2
git -C code-reference/jax show jax-v0.10.2:jax/_src/pallas/mosaic/tpu_info.py | sed -n '1,60p'
```

A mismatch between the pinned lines and the verbatim excerpt in the area file is a **real finding**.
Report it; do not quietly fix the area file — citation drift against `9da2369` has been swept once
already, and a second sweep should be a deliberate pass, not a drive-by edit.

## Do not read

- Anything `gpu_*`, `mgpu_*`, `mosaic_gpu*`, `triton*`, `pallas_mosaic_gpu*`, `pallas_triton*`.
  Different compiler, different memory model, different answer.
- `<tokamax>/tokamax/_src/pallas/block.py` — named as if it were generic; it is Triton/GPU code.
  Confirmed at line 27, `from jax.experimental.pallas import triton as plgpu`, and every `load`/
  `store` below it goes through that.
- `<tokamax>/tokamax/_src/ops/experimental/` for anything you intend to cite as settled.
- Any tokamax or jax checkout outside `<code-reference>`. Not this skill's, and not the one the
  citations resolve against.

## Gaps

Things this map does not yet cover. Strike a line when you close it, and add the row above.

- The annotations in § "what each TPU-relevant file is for" are name-derived, not read-derived.
  Every route you actually follow should either confirm its row or correct it. (The tokamax op
  table above is *not* in this category — it was enumerated from disk.)
- No route for **DSA / sparse attention top-k**, which is where a lot of SparseCore kernel work lands.
  Nearest upstream is `<tokamax>/tokamax/_src/ops/ragged_gather/` — which does ship a real TPU Pallas
  kernel — plus `<jax>/tests/pallas/tpu_sparsecore_pallas_test.py`. Whether either is a *structural*
  analogue of a SparseCore top-k kernel is still unread.
- No route for **KV-cache layout** or block-table management — vLLM-specific, likely absent
  upstream. Confirm the absence once, then record it here so nobody looks twice.
- ~~Which tokamax ops ship a `pallas_mosaic_tpu*` file has not been enumerated.~~ **Closed
  2026-09-10 at `6303c14`** — 6 of 11, listed in the op table above. Re-run when the clone moves:

  ```bash
  cd code-reference/tokamax/tokamax/_src/ops && for d in */; do \
    printf '%-42s %s\n' "${d%/}" "$(ls "$d" | grep 'pallas_mosaic_tpu' | tr '\n' ' ')"; done
  ```
- No route for anything in `<jax>/jax/experimental/pallas/ops/gpu/` — correct, but the reason should
  be stated once rather than rediscovered.
