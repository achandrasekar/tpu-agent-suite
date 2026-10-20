---
title: "Building, installing and developing TorchTPU"
type: doc-concept
provenance: doc
source: README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Building, installing and developing TorchTPU

## Definition
The README describes two supported ways to get a working `torch_tpu` into a Python virtual
environment (created with `python -m venv` or `uv venv`): install a nightly wheel from the
project's Google Artifact Registry Python index (after installing the
`keyrings.google-artifactregistry-auth` keyring and running `gcloud auth login` plus
`gcloud auth application-default login`), or build the wheel from source with Bazel. Prebuilt
Docker images are a third route ([docker-multistage-images](docker-multistage-images.md)).
Python support follows PyTorch's release compatibility matrix; at this pin that means
**Python 3.11+** (the developer policy lists 3.11 through 3.14, including free-threaded 3.14t;
see [api-stability-policy](api-stability-policy.md)).

## In torch_tpu (grounded)
- **One wheel, several PyTorch ABIs.** The wheel ships one native glue library per supported
  `torch` version. At import, [`install_hook`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_versioned_so_loader.py#L274 "src/torch_tpu/_versioned_so_loader#install_hook") selects the
  glue variant matching the installed `torch` and installs an import hook
  ([`VersionDispatchFinder`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_versioned_so_loader.py#L227 "src/torch_tpu/_versioned_so_loader#VersionDispatchFinder")) that redirects
  `torch_tpu._internal` imports to it. This is why the README pins the torch index URL: a
  mismatched `torch` is the most common install failure.
- **What import does.** [`load`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L379 "src/torch_tpu/_loader#load") detects a TPU with
  [`get_tpu_device_count`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/hardware.py#L322 "src/torch_tpu/_internal/utils/hardware#get_tpu_device_count") and initializes the backend
  through [`_init_device_impl`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L206 "src/torch_tpu/_loader#_init_device_impl"); see
  [device-backend-registration](../concepts/device-backend-registration.md).
- **Allocator check.** [`check_tcmalloc`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_tcmalloc_check.py#L100 "src/torch_tpu/_tcmalloc_check#check_tcmalloc") warns when tcmalloc is
  not the process allocator: eager dispatch and background runtime threads allocate and free
  across threads, which contends on glibc's default allocator. The Docker image preloads
  tcmalloc; bare-metal installs should too.

## Why it matters / when it applies
- **Wheel builds need `--config=wheel_common`** (`bazel build --config=wheel_common
  //ci/wheel:torch_tpu_wheel`); the build fails without it. The config sets
  `--//:wheel_build=True`, which routes the shared XLA/MLIR backend into a single
  `libxla_base.so`. Without it each per-version glue carries its own backend copy and the wheel
  aborts on `import torch` with duplicate static registrations.
- **Optimized by default.** OSS Bazel builds default to `-c opt` via `.bazelrc`. Unoptimized
  builds (`-c dbg`, `-c fastbuild`) degrade performance and require the explicit opt-in
  `--//:allow_unoptimized_wheel=True`; use them only when debugging C++.
- **No remote execution credentials:** append `--config=no_rbe` *after* `--config=wheel_common`;
  the later config wins for the remote-cache and remote-executor flags.
- **CPU torch, not CUDA.** Installing the built wheel with a plain index pulls the CUDA build of
  `torch`; use the nightly index URL (which ranks the CPU build higher) or install CPU `torch`
  separately. The TPU backend does not need CUDA.
- **Developer tooling.** Bazel is installed via Bazelisk (`bazel info` verifies the setup). For
  C++ navigation the README recommends VS Code with clangd fed by a compilation database from
  the Bazel action graph (`tools/setup_clangd.py`, or `nox -s refresh_compile_commands --
  --no-build`); the first run can exceed 30 minutes, later runs are incremental.

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md) (what
  import does), [hardware-introspection](../concepts/hardware-introspection.md) (how the TPU is
  detected without initializing the runtime), [pjrt-client](../concepts/pjrt-client.md) (the
  lazily initialized runtime).
- Module catalogs: [module map](../catalog/index.md) (`src/torch_tpu/_loader.py`,
  `src/torch_tpu/_versioned_so_loader.py`, `src/torch_tpu/_tcmalloc_check.py`).
- Related doc-concepts: [tpu-device-onboarding](tpu-device-onboarding.md) (first steps after
  install), [docker-multistage-images](docker-multistage-images.md),
  [user-guide-notebooks](user-guide-notebooks.md), [feature-overview](feature-overview.md),
  [api-stability-policy](api-stability-policy.md).

## Source
Extracted from `README.md` (kept in place).
