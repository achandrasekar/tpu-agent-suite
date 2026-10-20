---
title: "Public API surface and stability stages"
type: doc-concept
provenance: doc
source: docs/developers/api_policy.md
sources: [docs/developers/api_policy.md, docs/developers/api_design_guide.md, docs/developers/python_support_policy.md, docs/developers/README.md, docs/markdown/public_apis.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Public API surface and stability stages

## Definition
TorchTPU's policy is "the best API is no API": where a PyTorch API exists (preferably
`torch.accelerator` over `torch.tpu`), TorchTPU implements it for TPU rather than inventing one.
New APIs pass through stages borrowed from PyTorch plus an explicit **internal** stage:
internal (unstable, may change without notice, undefined behavior if used), **experimental**
(usable, may change or disappear; a one-time warning fires on use), **stable** (backward-
incompatible changes require deprecation), and **deprecated** (kept at least 2 minor releases
before removal, with a run-time warning). Public Python APIs live only under `torch.tpu.*`; the
`torch_tpu` package itself has no public API (implementation is in `torch_tpu._internal`), and
each object has at most one canonical public path. Python support follows PyTorch's release
compatibility matrix for the minimum supported PyTorch: Python 3.11 through 3.14, including
free-threaded 3.14t, at this pin.

## In torch_tpu (grounded)
- **The `torch.tpu` namespace is declarative.** [`install`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/public_api.py#L545 "src/torch_tpu/_internal/public_api#install")
  turns the [`_PUBLIC_MODULES`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/public_api.py#L130 "src/torch_tpu/_internal/public_api#_PUBLIC_MODULES._PUBLIC_MODULES")
  table of [`ModuleSpec`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/public_api.py#L55 "src/torch_tpu/_internal/public_api#ModuleSpec") entries into lazily resolved
  modules: `torch.tpu`, `torch.tpu.compiler`, `torch.tpu.distributed`, `torch.tpu.ops`,
  `torch.tpu.optimizers`, `torch.tpu.pallas` and `torch.tpu.profiler` (the design guide also
  lists `exceptions`, which has no entry at this pin). Global knobs are properties, for example
  `torch.tpu.allow_excess_precision`
  ([`allow_excess_precision`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L283 "src/torch_tpu/_internal/device/_device_module#_DeviceModuleMeta.allow_excess_precision"));
  custom ops are re-exported under `torch.tpu.ops`, for example
  [`align_jagged_to_multiple`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/ops/jagged.py#L62 "src/torch_tpu/_internal/ops/jagged#align_jagged_to_multiple").
- **Python stage markers.** [`experimental`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L339 "src/torch_tpu/_internal/utils/annotations#experimental"),
  [`stable`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L362 "src/torch_tpu/_internal/utils/annotations#stable") and
  [`deprecated`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L380 "src/torch_tpu/_internal/utils/annotations#deprecated") decorate callables
  ([`Stage`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L104 "src/torch_tpu/_internal/utils/annotations#Stage") lists the stages and their warning
  categories); non-callable attributes are declared in a module's `__tt_api_stages__`.
- **Custom-op markers.** C++ ops registered in `torch.ops.tpu` use
  [`ImplInternal`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L259 "csrc/eager/tpu_aten_kernels.cc#ImplInternal") (name starts with `_`),
  [`ImplExperimental`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L327 "csrc/eager/tpu_aten_kernels.cc#ImplExperimental"),
  [`ImplStable`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L249 "csrc/eager/tpu_aten_kernels.cc#ImplStable") or
  [`ImplDeprecated`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/tpu_aten_kernels.cc#L345 "csrc/eager/tpu_aten_kernels.cc#ImplDeprecated").
- **Environment variables.** Every variable TorchTPU reads must be registered in
  [`kEnvVarToStage`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L254 "csrc/common/env_vars.h#kEnvVarToStage") with a [`SymbolStage`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/symbol_stage.h#L40 "csrc/common/symbol_stage.h#SymbolStage");
  [`GetEnvOnce`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/common/env_vars.h#L374 "csrc/common/env_vars.h#GetEnvOnce") enforces this at compile time, reads the variable once
  and memoizes it, and warns once for experimental or deprecated variables. Established
  launcher and runtime variables (`LIBTPU_INIT_ARGS`, `LOCAL_RANK`, `MASTER_ADDR`, ...) are
  marked stable.

## Why it matters / when it applies
- **Reading the names tells you the contract.** `TORCH_TPU_INTERNAL_*` env vars,
  `torch_tpu_internal_*` flags and any `_`-prefixed Python name are internal: do not build on
  them. `TORCH_TPU_*` env vars and `torch.tpu.*` names are experimental or stable; a one-time
  warning on first use means experimental. Command-line flags are never a supported user
  interface.
- **Precedence rules:** a Python API call overrides the matching environment variable; a local
  context manager overrides the matching global knob (for example `enable_tracebacks` over the
  internal traceback flag); the latest change to a global knob wins.
- **Import from `torch.tpu`, not `torch_tpu`.** Code importing `torch_tpu._internal...` paths
  can break on any release.
- The generated public API list (`docs/markdown/public_apis.md`, built from the stage
  annotations) is the authoritative inventory of experimental, stable and deprecated APIs.

## The generated public API list (`public_apis.md`)
`docs/markdown/public_apis.md` is the inventory, regenerated by the repository's API scanner
(`tests/api/scan_tt_apis.py`) rather than written by hand. The scanner imports the public
`torch.tpu` submodules and reflects over them. It probes each `torch.accelerator` API on TPU and
keeps only the ones TPU implements. It reads env-var stages from `kEnvVarToStage` and custom-op
stages from the C++ registration blocks. The page warns that using any TorchTPU API not on it
"is undefined behavior and unsupported". Its shape at this pin:
- **Stable:** the `torch.accelerator` device, stream, memory and RNG calls TPU implements; the
  `torch.tpu.ops.max_pool{1,2,3}d` ops and their backwards; and launcher/runtime env vars
  (`LIBTPU_INIT_ARGS`, `RANK`/`WORLD_SIZE`/`LOCAL_RANK`, `MASTER_ADDR`/`MASTER_PORT`,
  `NNODES`/`NODE_RANK`, `TPU_VISIBLE_DEVICES`, `TPU_PREMAPPED_BUFFER_SIZE`,
  `TORCH_DEVICE_BACKEND_AUTOLOAD`, `TMPDIR`).
- **Experimental:** nearly everything TPU-specific. That covers eager modes (`torch.tpu.EagerMode`,
  `set_eager_mode`), `Stream`/`Event`, `Precision`/`precision`/`allow_excess_precision`,
  `compiler.custom_compiler_options`, the `distributed` helpers (`topology_aware_mesh`,
  `set_tpu_launch_env`, `spmd_safe`), the ragged, sparse and collective ops in `torch.tpu.ops`,
  `torch.tpu.optimizers` (`AdamW`, `Muon`, `Sgd`, `SingleTraceTrainer`), `pallas.jax_op` and
  `get_pallas_mesh`, and `profiler.TpuProfilerConfig`. All `TORCH_TPU_*` env vars plus
  `TORCH_SHOW_CPP_STACKTRACES`, `TORCH_TRACE` and `XLA_FLAGS` are experimental too.
- **Deprecated:** `torch.accelerator.current_device_idx`/`set_device_idx` (deprecated by PyTorch)
  and `torch.tpu.topology_aware_mesh` (moved to `torch.tpu.distributed`).
- **Internal (callable only by the framework):** `torch.tpu.current_device`, `device_count`,
  `synchronize`, `set_device`, `is_available`, `get_rng_state` and similar. They stay under
  `torch.tpu` so PyTorch can find them, but calling them from user code raises. Use the
  `torch.accelerator` equivalent the list names. This is the
  [`internal_or_experimental`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L406 "src/torch_tpu/_internal/utils/annotations#internal_or_experimental")
  decorator: the API is internal when the running PyTorch already provides the replacement
  (2.11 for `torch.accelerator`, 2.13 for `torch.accelerator.random`) and experimental otherwise.

For onboarding code the practical rule is: device, stream, memory and RNG calls go through
`torch.accelerator`, and every `torch.tpu.*` call is either experimental (expect a one-time
warning) or an error.

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md),
  [error-handling](../concepts/error-handling.md),
  [dtype-and-precision](../concepts/dtype-and-precision.md),
  [hardware-introspection](../concepts/hardware-introspection.md).
- Module catalogs: [module map](../catalog/index.md) (`src/torch_tpu/_internal`,
  `csrc/common`, `csrc/eager`).
- Related doc-concepts: [environment-variables](environment-variables.md) (the variables
  themselves), [building-and-installing](building-and-installing.md),
  [feature-overview](feature-overview.md), [user-guide-notebooks](user-guide-notebooks.md).

## Source
Extracted from `docs/developers/api_policy.md`, `docs/developers/api_design_guide.md` and
`docs/developers/python_support_policy.md` (kept in place). `docs/developers/README.md` is an
index of these; `docs/developers/list_ci_tests_guide.md` (CI tag-filter tooling for contributors)
was skipped as out of scope for users.
The "generated public API list" section is from `docs/markdown/public_apis.md` (kept in place).
