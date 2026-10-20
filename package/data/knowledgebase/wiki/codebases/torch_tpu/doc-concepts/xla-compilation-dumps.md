---
title: "Dumping HLO and LLO: XLA_FLAGS, LIBTPU_INIT_ARGS and tt_jit module names"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md
sources: [docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md, docs/markdown/background_on_torch_compile_on_tpus_via_xla.md, docs/markdown/environment_variables.md, docs/markdown/quant.md, docs/markdown/quantized_sum.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Dumping HLO and LLO: XLA_FLAGS, LIBTPU_INIT_ARGS and tt_jit module names

## Definition
TorchTPU exposes XLA's own dump flags rather than a dump API of its own. HLO is dumped with
`XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"`; the directory then holds, per
compiled module, `*before_optimizations.txt`, `*after_optimizations.txt` (the final
hardware-level graph), `*after_codegen.txt`, a `*.flagfile` listing every active XLA flag
(useful to confirm `xla_optimization_level`) and a `*tpu_comp_env.txt` with the TPU compilation
environment. LLO (TPU assembly) is dumped through libtpu with
`LIBTPU_INIT_ARGS="--xla_jf_dump_to=<dir> --xla_jf_dump_llo_text=true"`; per fusion it writes
files such as `<id>-add_maximum_fusion-01-original.txt`. Both variables must be set before the
backend initializes.

## In torch_tpu (grounded)
- **Module names.** Each module is named by
  [`BuildModuleNameFromPyContext`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/op_builder_utils.cc#L1175 "csrc/ops/op_builder_utils.cc#BuildModuleNameFromPyContext") as
  `tt_jit_<file>_L<line>C<col>_<func>_<opname>`, from the Python frame that triggered the
  compilation (usually the materialization point) and the op that triggered it. The frame comes
  from the captured Python context turned into an MLIR location by
  [`MakeMlirLocation`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/python_context.cc#L239 "csrc/ops/python_context.cc#MakeMlirLocation"); with Python tracebacks disabled the name
  falls back to `tt_jit` plus the op call chain. This is how you find your function among the
  many small modules (copies, RNG seeds) in a dump directory, and the HLO `op_name` metadata
  (`add/add`, `relu/relu`) comes from the same context
  ([python-context-and-op-names](../concepts/python-context-and-op-names.md)).
- **`XLA_FLAGS`** ([`kXlaFlagsEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L250 "csrc/common/env_vars.h#kXlaFlagsEnvVar")) is parsed into the XLA debug
  options once, through the cached build options of
  [`GetDefaultExecutableBuildOptions`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/excess_precision.cc#L43 "csrc/common/excess_precision.cc#GetDefaultExecutableBuildOptions") applied in
  [`UpdateFromXlaFlags`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L218 "csrc/common/compilation.cc#UpdateFromXlaFlags"). TorchTPU's own override variable
  `TORCH_TPU_INTERNAL_XLA_OPTIONS` ([`kTorchTpuInternalXlaOptionsEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L179 "csrc/common/env_vars.h#kTorchTpuInternalXlaOptionsEnvVar"),
  read by [`GetCompilerOptionOverridesFromEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L197 "csrc/common/compilation.cc#GetCompilerOptionOverridesFromEnvVar")) takes precedence
  over `XLA_FLAGS` debug options ([compilation-pipeline](../concepts/compilation-pipeline.md)).
- **`LIBTPU_INIT_ARGS`** ([`kLibtpuInitArgsEnvVar`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L53 "csrc/common/env_vars.h#kLibtpuInitArgsEnvVar")) is read by libtpu.
  For distributed runs TorchTPU rewrites it before the runtime starts:
  [`ConfigureLibtpuInitArgs`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/environment.cc#L150 "csrc/common/environment.cc#ConfigureLibtpuInitArgs") keeps the user's flags and adds the
  launch-barrier and SliceBuilder settings it needs.
- **Cache key.** [`MakeCompileOptionsKey`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/compilation.cc#L441 "csrc/common/compilation.cc#MakeCompileOptionsKey") fingerprints the raw
  `XLA_FLAGS` and `LIBTPU_INIT_ARGS` strings into the compile-options key. Turning dumps on (or
  reordering the same flags) therefore changes the key, misses every cache tier, and recompiles,
  which is what makes the dump files appear even for graphs that were cached before
  ([compilation-cache-key](../concepts/compilation-cache-key.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md)).

- **In the custom-op notebooks.** The `quant` and `quantized_sum` notebooks use the same recipe
  with two additions. They append the libtpu dump flags to `sys.argv` as well as setting
  `LIBTPU_INIT_ARGS`, because "it sometimes needs flags, sometimes envvar". They also call
  `torch._logging.set_logs(aot_graphs=True)` to print the ATen graph next to the HLO. A
  `jax_op` kernel shows up as its own module, for example
  `module_0039.tt_jit_custom_kernel.before_optimizations.txt`, where the JAX function's ops
  carry `jit(<fn>)/...` `op_name` metadata
  ([jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md),
  [one-bit-quantization-example](one-bit-quantization-example.md)).

## Why it matters / when it applies
Dumps answer "what did XLA actually build": whether ops fused, which layouts and tilings were
picked, and which flags were in effect. They are level 4 of the
[debugging-ladder](debugging-ladder.md). Expect hundreds of lines per module and many files per
run; the separate key also means a dump run does not reuse the persistent cache entries of a
normal run.

## Connections
- Code concepts: [compilation-pipeline](../concepts/compilation-pipeline.md),
  [compilation-cache-key](../concepts/compilation-cache-key.md),
  [python-context-and-op-names](../concepts/python-context-and-op-names.md),
  [stablehlo-op-builder](../concepts/stablehlo-op-builder.md),
  [pjrt-client](../concepts/pjrt-client.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/common`, `csrc/ops`).
- Related doc-concepts: [aten-to-llo-lowering-path](aten-to-llo-lowering-path.md),
  [debugging-ladder](debugging-ladder.md),
  [environment-variables](environment-variables.md),
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md`
(kept in place); the dump-file list (`after_optimizations`, `flagfile`, `tpu_comp_env`) is from
`docs/markdown/6_reference_4_environment_variables.md`.
`docs/markdown/background_on_torch_compile_on_tpus_via_xla.md` is an identical copy of the viewing-compilation-artifacts chapter, and `docs/markdown/environment_variables.md` of the environment-variables reference (kept in place).
The custom-op notebook bullet is from `docs/markdown/quant.md` and `docs/markdown/quantized_sum.md` (kept in place).
