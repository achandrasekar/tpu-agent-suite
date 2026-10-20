---
title: "Docker multistage images and wheel sharing"
type: doc-concept
provenance: doc
source: docker/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Docker multistage images and wheel sharing

## Definition
The Docker build is "decoupled" into three images so heavy dependencies are installed once and
code changes touch only the wheel: `torch-tpu-base` (Python plus PyTorch, JAX, libtpu and other
stable dependencies, resolved from `pyproject.toml`), `torch-tpu` (the full
`Dockerfile.multistage` build, which installs the `torch_tpu` wheel on top of the base and keeps
it under `/opt/wheels/` for other projects), and `torchtpu-vllm` (built in the separate
`torchtpu-vllm` repository from `torch-tpu-base`, which "siphons the compiled wheel from
`torch-tpu`" instead of rebuilding it).

## In torch_tpu (grounded)
The wheel built inside the image is the same Bazel target described in
[building-and-installing](building-and-installing.md); inside the container, `import torch_tpu`
runs [`load`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L379 "src/torch_tpu/_loader#load") exactly as with a pip install. Details of the Dockerfile
at this pin that differ from, or add to, the README:
- The README says `torch-tpu` builds the wheel from source. The `final` stage actually installs
  a prebuilt wheel placed in the build context's `dist/` (`build_image_multistage.sh` copies one
  there when given a wheel path) and otherwise downloads the latest nightly `torch_tpu` wheel
  from Artifact Registry, authenticated by a BuildKit secret or a `GCLOUD_TOKEN` build argument.
  Build the wheel with Bazel first if you need local changes in the image.
- The image also copies `examples/` to `/workspace/examples`.
- The base stage preloads tcmalloc (`LD_PRELOAD` of `libtcmalloc.so.4`), which silences the
  allocator warning from [`check_tcmalloc`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_tcmalloc_check.py#L100 "src/torch_tpu/_tcmalloc_check#check_tcmalloc"); eager dispatch
  allocates across threads and contends under the default glibc allocator.
- The README says the base carries Python 3.12, but at this pin `Dockerfile.multistage` defaults
  its `BASEIMAGE` build argument to a Python 3.14 slim image; trust the Dockerfile argument.

## Why it matters / when it applies
- Rebuild the base image only when dependency versions change; day-to-day `torch_tpu` changes
  "only require rebuilding the wheel and re-running the last few steps of the vllm build".
- Workflow: build the `base` stage to produce `torch-tpu-base:latest`, build the full file to
  produce `torch-tpu:latest` (wheel at `/opt/wheels/`), then the vLLM Dockerfile copies the
  wheel out of `torch-tpu:latest` (`docker/build_image_multistage.sh` drives the first two).

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md).
- Module catalogs: [module map](../catalog/index.md) (`src/torch_tpu/_loader.py`,
  `src/torch_tpu/_tcmalloc_check.py`).
- Related doc-concepts: [building-and-installing](building-and-installing.md),
  [user-guide-notebooks](user-guide-notebooks.md).

## Source
Extracted from `docker/README.md` (kept in place).
