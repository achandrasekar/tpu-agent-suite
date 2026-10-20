---
title: "User guide and documentation notebooks"
type: doc-concept
provenance: doc
source: docs/README.md
sources: [docs/README.md, docs/source/README.md, docs/notebooks/README.md, README.md, docs/markdown/index.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# User guide and documentation notebooks

## Definition
TorchTPU's user guide is generated from Marimo notebooks. `docs/source/` holds Marimo `.py`
notebooks that "serve as the single source of truth for TorchTPU documentation and tutorials";
`publish_notebook.py` generates Jupyter notebooks (`docs/notebooks/*.ipynb`, ready for
JupyterLab, VS Code or the "Open in Colab" links on the public Cloud documentation) and Markdown
exports from them. The top-level README points to the static guide at
http://google-pytorch.github.io/torch_tpu/ ("a static version of the interactive notebooks").
The sources run directly on a TPU VM (`marimo edit <file>.py` or `marimo run <file>.py`); the
README suggests a v6e GCE instance as the easiest host. `docs/README.md` itself is a one-line
directory header.

## In torch_tpu (grounded)
The numbered notebook series at this pin maps onto this wiki's concept pages:
- **1 Basics** (intro, getting started on a VM and on GKE, running upstream PyTorch examples,
  distributed on GKE): device setup through [`load`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L379 "src/torch_tpu/_loader#load")
  ([device-backend-registration](../concepts/device-backend-registration.md)) and the
  `tpu_dist` group ([process-group-and-collectives](../concepts/process-group-and-collectives.md)).
  Doc-concepts: [tpu-device-onboarding](tpu-device-onboarding.md),
  [tpu-hardware-model](tpu-hardware-model.md), [tpu-dist-process-group](tpu-dist-process-group.md),
  [gke-xpk-workloads](gke-xpk-workloads.md).
- **2 In depth** (how TorchTPU works, known limitations, the TPU memory model, execution modes):
  eager dispatch through [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp") and materialization
  ([deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)),
  graph mode through [`TpuBackend`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/compile/_backend.py#L613 "src/torch_tpu/_internal/compile/_backend#TpuBackend")
  ([torch-compile-backend](../concepts/torch-compile-backend.md)). Doc-concepts:
  [eager-execution-modes](eager-execution-modes.md),
  [torch-compile-tpu-backend](torch-compile-tpu-backend.md),
  [torchtpu-lowering-stack](torchtpu-lowering-stack.md).
- **3 Practical examples** (LLM inference and training end to end, FSDP best practices, Pallas
  custom kernels, distributed Pallas kernels, bounded dynamism): see
  [simple-fsdp](simple-fsdp.md), [pallas-custom-kernels](../concepts/pallas-custom-kernels.md),
  [eager-dynamic-shapes](../concepts/eager-dynamic-shapes.md) and
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md); doc-concepts
  [jax-and-pallas-custom-ops](jax-and-pallas-custom-ops.md).
- **4 Debugging and performance analysis** (numerical precision, XProf, better error
  information, viewing compilation artifacts): [dtype-and-precision](../concepts/dtype-and-precision.md),
  [profiler](../concepts/profiler.md), [error-handling](../concepts/error-handling.md);
  doc-concepts [torch-profiler-on-tpu](torch-profiler-on-tpu.md),
  [debugging-ladder](debugging-ladder.md), [xla-compilation-dumps](xla-compilation-dumps.md).
- **5 Case studies** (FLUX optimization): the same material as
  [zero-copy-views-vs-layout-copies](zero-copy-views-vs-layout-copies.md),
  [fused-algebraic-rope](fused-algebraic-rope.md) and
  [whole-block-compile-fusion](whole-block-compile-fusion.md).
- **6 Reference** (supported feature set, TorchTPU API, XProf reference, environment variables):
  [environment-variables](environment-variables.md), [api-stability-policy](api-stability-policy.md).

Older unnumbered sources (eager mode, compilation cache, precision management, Pallas custom ops,
numerical parity, ViT/MNIST porting) still sit alongside them, with prebuilt pages in
`docs/html/` and Markdown exports in `docs/markdown/`. Several are byte-identical copies of a
numbered chapter (`intro_to_torchtpu`, `error_glossary`, `performance_profiling`); the rest have
their own pages: [mxu-precision-modes](mxu-precision-modes.md) and
[excess-precision-and-out-dtype](excess-precision-and-out-dtype.md) (precision management),
[numerical-parity-checking](numerical-parity-checking.md),
[porting-vit-walkthrough](porting-vit-walkthrough.md),
[extending-pytorch-on-tpu](extending-pytorch-on-tpu.md) and
[one-bit-quantization-example](one-bit-quantization-example.md) (the composition and custom-op
series), and [api-stability-policy](api-stability-policy.md) (the generated public API list).
The landing page `docs/markdown/index.md` lists the numbered series. Some of its chapter links
point at titles that are still stubs at this pin (for example the numerical-precision chapter
has only a heading). This page covers only how the documentation is organized and produced.

## Why it matters / when it applies
Start with the guide for a runnable, narrative introduction; come to this wiki for the mechanism
or exact symbol behind a behavior the guide demonstrates. Edit the Marimo source, not the
generated `.ipynb` or Markdown, when contributing documentation. Interactive runs require TPU
hardware; the static site and Colab previews do not.

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [profiler](../concepts/profiler.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [building-and-installing](building-and-installing.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md),
  [api-stability-policy](api-stability-policy.md).

## Source
Extracted from `docs/README.md`, `docs/source/README.md`, `docs/notebooks/README.md` and the
"User Guide and Interactive Notebooks" section of `README.md` (kept in place).
`docs/markdown/index.md` is the guide's landing page: it lists the six numbered series and one-line summaries of every chapter (kept in place).
