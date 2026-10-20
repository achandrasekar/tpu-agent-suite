---
title: "tokamax repo snapshot (2026-10)"
type: source
tags: [codebase, kernel, compiler, tokamax]
created: 2026-10-02
updated: 2026-10-02
sources: 1
human_reviewed: false
---

Provenance record for the ingest of OpenXLA's `tokamax` repository (a JAX kernel library for
TPU and GPU) into the codebase subwiki `wiki/codebases/tokamax/`, produced by wikify-repo from
the source at one pinned commit. Knowledge lives on the subwiki pages; this page records where
it came from and what was left out.

## Identity

| | |
|---|---|
| Repository | https://github.com/openxla/tokamax (public) |
| License | Apache-2.0 |
| Commit | `3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68` (main HEAD; no release tags exist) |
| Snapshot date | 2026-10-02 (first build, wikify 0.4.0) |
| Acquired as | clean `git clone` at the pin under `raw/code/tokamax/` (gitignored, never shipped); `config/tokamax.md` holds the pin and the ingest settings (`catalog: index`, `index_profile: full`) |
| Indexed with | SCIP: scip-python over the whole repo (464 tracked files, pure Python) |
| Authority | primary source: the implementation, its docstrings, tests and in-repo docs |

## What it produced
- `wiki/codebases/tokamax/overview.md`, 18 concept pages, 2 area pages, 14 doc-concept pages
  from 9 docs (README and `docs/`; `contributing.md` and the one-line `splash_attention.md`
  stub yielded none), and the symbol index (`catalog/symbols.tsv`: 5,738 symbols across 333
  modules, 100% of documentable symbols; `catalog/edges.tsv`: 12,497 caller edges).
- Lens (`synthesis_focus`): TPU kernels - dispatch and lowering on TPU, Pallas Mosaic TPU
  kernel design, tiling and VMEM, custom VJP, autotuning and benchmarking, the torch_tpu
  bridge. GPU backends are catalogued, not explained: the
  [kernel catalog](../codebases/tokamax/concepts/kernel-catalog.md) lists every op x backend
  with its purpose, and the GPU-only ops (`layer_norm`, `gated_linear_unit`) are sections of
  the `_src` area page.
- Every concept page passed the citation linter, then adversarial verification claim by claim:
  337 load-bearing claims, 19 refuted and corrected in place (mostly execution-order details in
  the serving kernels); each page carries a `verified:` stamp, verdicts in `.cache/verify/`.
- The hub page [Tokamax](../codebases/tokamax.md); [Pallas Kernel](../kernel/pallas-kernel.md) now
  links the subwiki instead of `raw/code/tokamax/docs/` paths that were never in git, and its
  GMM v2 rows no longer name modules absent at this pin.

## Gaps and caveats
- The pin is a moving `main` commit; tokamax publishes no tags. Experimental ops
  (`ops/experimental/`) change fastest.
- Docs vs code at the pin, recorded as warnings on the doc-concept pages:
  `docs/benchmarking.md` names the timer `xprof_hermetic` (code: `hermetic_xprof`);
  several `docs/autotuning.md` examples do not run as written; `docs/supported_ops.md` omits
  the TPU kernels for attention and linear softmax cross-entropy and every experimental TPU op;
  the KDA forward design doc's float32 cumsum and softplus-gate statements do not match the
  Mosaic backend.
- Possible upstream defects surfaced while reading the source (unverified at runtime, stated
  as open questions on the concept pages): unmasked quantized-scale reads in the GMM v2 K tail
  ([gmm-v2](../codebases/tokamax/concepts/gmm-v2.md)); ring attention merging base-2
  statistics with `exp` ([splash-attention](../codebases/tokamax/concepts/splash-attention.md));
  three batched-RPA arguments dropped by the PyTorch wrappers
  ([torch-tpu-bridge](../codebases/tokamax/concepts/torch-tpu-bridge.md)).
- `tokamax/_src/pallas/` (BlockRef helpers) is GPU-only despite its generic name.

## Connections
- Created: the `wiki/codebases/tokamax/` subwiki, [Tokamax hub](../codebases/tokamax.md).
  Updated: [Pallas Kernel](../kernel/pallas-kernel.md), `index.md`, `log.md`.
- Related: [TorchTPU](../codebases/torch-tpu.md) (the torch_tpu `jax_op` the bridge calls),
  [Pallas](../codebases/pallas.md).
