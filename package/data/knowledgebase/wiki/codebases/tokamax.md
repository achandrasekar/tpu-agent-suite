---
title: "Tokamax (JAX TPU and GPU kernel library)"
type: codebase
tags: [codebase, kernel, compiler, tokamax, pallas]
created: 2026-10-02
updated: 2026-10-02
sources: 1
human_reviewed: false
aliases: [tokamax, openxla/tokamax]
---

# Tokamax (JAX TPU and GPU kernel library)

Tokamax is the OpenXLA library of custom accelerator kernels for JAX, "supporting both NVIDIA
GPUs and Google TPUs": attention (flash and splash), ragged dot / grouped matmul for MoE,
MoE dispatch and combine, paged and latent attention for serving, linear attention (KDA,
gated delta rule), a fused LM-head cross-entropy and top-k, with autotuning, device-time
benchmarking and a PyTorch bridge through torch_tpu. This hub page states the ingested pin and
routes into the grounded subwiki; mechanism and symbol-level detail live there, not here.

---
## Pin

| | |
|---|---|
| Upstream | https://github.com/openxla/tokamax (public) |
| License | Apache-2.0 (`LICENSE` at the pin) |
| Ingested commit | `3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68` (main HEAD, 2026-10-02; the repo has no release tags) |
| Ingest config | `config/tokamax.md` (wikify-repo; re-ingest at a new pin with `wikify prepare tokamax --ref <commit>`) |
| Provenance | [tokamax repo snapshot (2026-10)](../sources/2026-10-tokamax-repo.md) |

## Start here (subwiki)
- [Overview](tokamax/overview.md): the system on one page, two architecture diagrams (the op
  framework and the TPU kernels by model layer) and a task table.
- [Kernel catalog](tokamax/concepts/kernel-catalog.md): every op x backend implementation -
  Mosaic TPU, Mosaic GPU (sm80/sm90/sm100), Triton, CuTe DSL, chunked XLA, XLA reference - with
  hardware gate, forward/VJP role, purpose and dispatch order. GPU kernels are listed here and
  in the symbol index; they have no mechanism pages.
- [Concept index](tokamax/index.md): 18 grounded, adversarially verified mechanism pages (op
  framework, autotuning, benchmarking, Mosaic TPU helpers, one page per TPU kernel, the
  torch_tpu bridge), plus 2 area pages.
- `tokamax/doc-concepts/`: 14 pages from the README and `docs/` (autotuning, benchmarking,
  export and cross-compilation, supported ops, the KDA design docs), several with warnings where
  the docs disagree with the code at the pin.
- `tokamax/catalog/`: the symbol index - `symbols.tsv` (5,738 symbols with signatures and doc
  lines), `edges.tsv` (12,497 caller edges) and [`index.md`](tokamax/catalog/index.md) (module
  map, 333 modules). Citations link the source line on GitHub at the pin.

## Using tokamax on TPU

```python
import tokamax

out = tokamax.ragged_dot(lhs, rhs, group_sizes)                 # default: first backend that accepts
out = tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="mosaic_tpu_v2")  # pin a kernel
```

- With no `implementation=`, a kernel that declines the arguments falls back to XLA silently;
  name the implementation to make that an error:
  [op-framework](tokamax/concepts/op-framework.md),
  [kernel-catalog](tokamax/concepts/kernel-catalog.md).
- Tile sizes come from an explicit config, the autotuning cache or heuristics:
  [autotuning](tokamax/concepts/autotuning.md).
- From PyTorch on TPU (batched ragged paged attention, linear softmax cross-entropy):
  [torch-tpu-bridge](tokamax/concepts/torch-tpu-bridge.md).

## See also
- [Pallas Kernel](../kernel/pallas-kernel.md) - authoring mechanics; its repo-link tail points at
  the tokamax kernels.
- [Pallas](pallas.md) - the kernel language tokamax is written in.
- [TorchTPU](torch-tpu.md) - the runtime behind the PyTorch bridge
  ([pallas-custom-kernels](torch_tpu/concepts/pallas-custom-kernels.md) is the torch_tpu side
  of `jax_op`).

## Sources
- [tokamax repo snapshot (2026-10)](../sources/2026-10-tokamax-repo.md): the ingested source
  at the pin; grounds everything in the subwiki.
