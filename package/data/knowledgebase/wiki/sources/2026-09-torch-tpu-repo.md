---
title: "torch_tpu repo snapshot (2026-09)"
type: source
tags: [codebase, torch-tpu, pytorch, compiler]
created: 2026-09-05
updated: 2026-09-25
sources: 1
human_reviewed: false
---

Provenance record for the ingest of Google's `torch_tpu` repository (the PyTorch backend for
Cloud TPU) into the codebase subwiki `wiki/codebases/torch_tpu/`, produced by wikify-repo from
the source at one pinned commit. Knowledge lives on the subwiki pages; this page records where
it came from and what was left out.

## Identity

| | |
|---|---|
| Repository | https://github.com/google-pytorch/torch_tpu (private) |
| Commit | `4c0c7c5711368001ed94830a4fe76652d1cb6c31` |
| Snapshot date | 2026-09-05 (first build, wikify 0.2); regenerated 2026-09-16 with wikify 0.3.0 from a clean clone at the same pin, tracked files only |
| Acquired as | clean `git clone` at the pin under `raw/code/torch_tpu/` (gitignored, never shipped); `config/torch_tpu.md` holds the pin and the ingest settings (`catalog: index`, `index_profile: full`) |
| Indexed with | SCIP: scip-python over `torch_tpu/*`, scip-clang over the Bazel compile database for `//torch_tpu/...` |
| Authority | primary source: the implementation itself, its docstrings and comments, and its in-repo README files |

## What it produced
- `wiki/codebases/torch_tpu/overview.md`, 27 concept pages, 15 doc-concept pages extracted
  from 12 README files, and the symbol index (`catalog/symbols.tsv`: 13,328 symbols in 10,334
  rows with signatures and doc lines; `catalog/edges.tsv`: 58,393 caller edges;
  `catalog/index.md`: module map over 556 modules; 100% of documentable symbols represented),
  plus the silo's generated `index.md` and `log.md`. No per-module catalog pages: the index
  tier replaces them (wikify 0.3.0, `docs/catalog-index.md`).
- Regeneration 2026-09-16: the first build had been indexed from a developer working copy, so
  untracked files (Jupyter checkpoints) and an empty-path shard leaked into the catalog, and
  overloaded functions showed their first `@overload` stub. Rebuilt with wikify 0.3.0 from a
  clean clone: 26 concept pages unchanged, 1 re-stamped (`pallas-custom-kernels`, whose cited
  symbols' merged records changed hash; content re-checked against the packet and kept).
- Citation links 2026-09-25 (wikify 0.4.0 `source-links`): the 1,785 citations pointed at
  `catalog/<module>.md` pages that the `index` tier does not render, so none resolved in a
  browser. Each now links the symbol's source line at the pin, titled with its index key;
  path and line come from this build's `catalog/symbols.tsv`, with no re-index and no page
  text changed.
- The hub page [TorchTPU Runtime & OpenXLA Backend](../codebases/torch-tpu.md) now states the
  pin, routes into the subwiki and carries a contradiction note on two of its earlier claims.
- `SCHEMA.md` gained the codebase-subwiki conventions; `validate_kb.py` and the knowledge-base
  tests treat a subwiki as one catalogued entry.

## Gaps and caveats
- Doc extraction covered README files only. The Marimo notebooks under `docs/`,
  `CONTRIBUTING.md` and the `examples/tutorials` Colab were not ingested.
- Only the `torch_tpu/` package is indexed. `examples/` and `tests/` are not in the symbol index, so
  example-level classes (benchmark wrappers, DeepSeek test layers) are described in prose.
- Source links are GitHub permalinks into a private repository; readers without access
  should open the local checkout at the pin instead.
- The synthesis lens (`synthesis_focus` in the config) is customer onboarding and optimization;
  subsystems outside it (build tooling, CI, experimental modules) are index-only.
- The concept pages were written from the source with the citation linter as the gate, then
  adversarially verified claim by claim (416 claims; one misquote found and corrected); the
  verdicts live in `.cache/verify/` (not shipped) and each page carries a `verified:` stamp.
- Comments quote internal `go/` links and bug ids; they are kept as written and are not
  resolvable outside Google.

## Connections
- Created: the `wiki/codebases/torch_tpu/` subwiki. Updated:
  [TorchTPU hub](../codebases/torch-tpu.md), `index.md`, `log.md`, `SCHEMA.md`.
- Related: [Pallas](../codebases/pallas.md) (stub; the Pallas surface inside torch_tpu is
  [pallas-custom-kernels](../codebases/torch_tpu/concepts/pallas-custom-kernels.md)).
