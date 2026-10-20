---
title: "Installing TorchTitan with a source-built torch_tpu wheel (VM and Docker image)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Installing TorchTitan with a source-built torch_tpu wheel (VM and Docker image)

## Definition
The TPU README treats the software stack as a version-pinning problem. A pre-built `torch_tpu`
wheel exists on an Artifact Registry, but "it currently only supports
torch `2.11` and is not forward compatible with later torch versions". TorchTitan needs torch
`>=2.12`, so "we must build torch_tpu from source". The recipe has five steps:
1. Create a Python 3.12 venv.
2. Clone `torch_tpu` and raise its `pyproject.toml` pin to `torch>=2.12.0`.
3. Regenerate the lockfiles with `bash requirements/lock_environments.sh`.
4. Build with `bazel build -c opt //ci/wheel:torch_tpu_wheel`, adding `--config=no_rbe` when you
   have no remote-execution cache.
5. Install in a fixed order: torch nightly CPU, then the TPU requirements, then TorchTitan's
   requirements and the Flux requirements, and last the wheel from `bazel-bin/ci/wheel/`.

For GKE, `torchtitan/experiments/tpu/Dockerfile` bakes in nearly the same order. The difference is
that it installs the wheel right after the TPU requirements, before TorchTitan's own requirements.
You copy the built wheel into the repo root so the Docker build context can see it, build the image,
and push it to Artifact Registry.

## In torchtitan (grounded)
- **Pinned TPU dependencies.** `torchtitan/experiments/tpu/requirements.txt` pins `jax==0.11.1`,
  `jaxlib==0.11.1`, `libtpu==0.0.50`, `tokamax==0.0.14` and `torchao==0.17.0`, plus `portpicker`,
  `absl-py`, `numpy`, `gcsfs`, `frozendict` and `triton`. The libtpu pin moved up from `0.0.48` at
  this commit, `torchao` was added and `tamm` was dropped. The README installs it with
  `-f https://storage.googleapis.com/jax-releases/libtpu_releases.html` so pip can resolve libtpu.
  These are not code symbols. The versions matter because the libtpu flag bundles in
  [`tpu_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L109 "torchtitan/experiments/tpu/tpu_flags#apply_init_flags")
  assume a recent libtpu. For example,
  [`HIERARCHICAL_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L169 "torchtitan/experiments/tpu/tpu_flags#HIERARCHICAL_REDUCE_SCATTER.HIERARCHICAL_REDUCE_SCATTER")
  says it "is available in libtpu 0.0.47 and newer". tokamax provides the Mosaic kernels behind
  the fused loss, [`pallas_cross_entropy_loss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/loss.py#L18 "torchtitan/experiments/tpu/loss#pallas_cross_entropy_loss").
- **CPU torch first.** Both the README and the Dockerfile install `torch` from the nightly CPU
  index before anything else, "so downstream packages do not pull CUDA wheels" (Dockerfile
  comment). The README warns that the bazel wheel build itself "will install the CUDA version of
  `torch`" unless you point it at the CPU index.
- **Image layout.** The Dockerfile copies `torchtitan/`, `assets/` and `tests/` into `/torchtitan`
  and uses it as the working directory. It also installs `requirements-dev.txt` because "CI runs
  with this image". Relative paths in a launch command, such as `--hf_assets_path=assets/hf/...` and
  `--dataloader.dataset_path=tests/assets/c4_test`, therefore resolve inside the image. The
  tokenizer path lands on
  [`hf_assets_path`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L89 "torchtitan/trainer#Trainer.Config.hf_assets_path").
- **tcmalloc at runtime.** The Dockerfile installs `libgoogle-perftools-dev` first and sets
  `LD_PRELOAD` to `libtcmalloc.so.4` as the last step, "after all RUN steps so build steps are
  unaffected". Every process in the container, including `torchrun` and its workers, runs with
  tcmalloc. Unset `LD_PRELOAD` if you need to rule the allocator out while debugging host memory.

## Why it matters / when it applies
- **Version skew is the first failure mode.** If you mix the pre-built 2.11 wheel with a 2.12+
  nightly torch, you get the incompatibility the README warns about. Build the wheel against the same
  torch you install.
- **Tokenizer and test data for GKE.** The README uploads `assets/hf` and `tests` to a bucket. That is
  only needed when the job reads them through a FUSE mount. The image already contains both
  directories. See [tpu-multihost-launch](tpu-multihost-launch.md).

> [!warning] Doc vs code
> - **Python version.** The README tells you to create a Python 3.12 venv, but the Dockerfile
>   defaults to `ARG PYTHON_VERSION=3.14`. A wheel built in a 3.12 venv will not install into the
>   default image unless you override that build arg (or build the wheel under 3.14). The README
>   does not reconcile the two.
> - **Broken in-doc link.** The Docker section links to "Build and install TorchTPU wheel from
>   source", but the actual heading is "Build TorchTPU wheel from source", so the anchor does not
>   resolve.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (what the installed stack
  runs), [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md) (`hf_assets_path`,
  tokenizer loading), [loss-and-validation](../concepts/loss-and-validation.md) (the tokamax-backed
  loss).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/tpu_flags`,
  `torchtitan/trainer`).
- Related doc-concepts: [tpu-vm-provisioning](tpu-vm-provisioning.md),
  [tpu-multihost-launch](tpu-multihost-launch.md), [tpu-init-flags-precedence](tpu-init-flags-precedence.md).
- Runtime side: torch_tpu [building-and-installing](../../torch_tpu/doc-concepts/building-and-installing.md),
  [docker-multistage-images](../../torch_tpu/doc-concepts/docker-multistage-images.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (sections "Build TorchTPU wheel from source",
"Install TorchTitan + TorchTPU and its dependencies", "Build the TorchTitan TPU-enabled Docker
image"; kept in place).
