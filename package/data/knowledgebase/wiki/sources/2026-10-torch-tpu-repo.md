---
title: "torch_tpu repo snapshot (2026-10, release/0.1.2)"
type: source
tags: [codebase, torch-tpu, pytorch, compiler]
created: 2026-10-13
updated: 2026-10-13
sources: 1
human_reviewed: false
---

Provenance record for the re-ingest of Google's `torch_tpu` repository (the PyTorch backend for
Cloud TPU) at the head of its `release/0.1.2` branch into the codebase subwiki
`wiki/codebases/torch_tpu/`, produced by wikify-repo. It supersedes
[the 2026-09 snapshot](2026-09-torch-tpu-repo.md) at `4c0c7c57`; the delta between the two pins
is in [`changes/6d2a45754b.md`](../codebases/torch_tpu/changes/6d2a45754b.md). Knowledge lives on
the subwiki pages; this page records where it came from and what was left out.

## Identity

| | |
|---|---|
| Repository | https://github.com/google-pytorch/torch_tpu (private) |
| Commit | `6d2a45754bdb52743883f75fbf753408345f6d1b` (head of `release/0.1.2`, 2026-10-12) |
| Previous pin | `4c0c7c5711368001ed94830a4fe76652d1cb6c31`; the two pins share no git history (no common ancestor), so there is no meaningful commit count between them |
| Acquired as | the existing clean clone under `raw/code/torch_tpu/` (gitignored, never shipped), fetched to the new commit; tracked files only |
| Indexed with | SCIP: scip-python over `src/torch_tpu/*`, scip-clang over the Bazel compile database for `//csrc/...`; wikify 0.4.1 |
| Authority | primary source: the implementation, its docstrings and comments, its README files and the user guide under `docs/markdown/` |

## What changed in the repo, and what the ingest did about it
- **Layout.** Between the two pins the repo moved its C++ from `torch_tpu/` to `csrc/` and its
  Python package to `src/torch_tpu/`. Every citation anchor changed. `config/torch_tpu.md` was
  remapped (Bazel targets, index shards, planner excludes, every `(subsystem: ...)` prefix);
  directories that used to mix C++ and Python are seeded from the half the existing page cited.
- **Reconcile.** 7,738 symbols only moved: wikify rewrote 1,215 citations in place, and two
  pages (`dtype-and-precision`, `error-handling`) needed nothing more. 25 concept pages were
  rebuilt by updating the existing page in place (wording, structure and diagrams kept where
  still accurate), and one page is new: `mlir-optimization-passes` (`csrc/mlir`).
- **Docs.** The doc worklist grew from 12 README files to 63 documents, including the user
  guide under `docs/markdown/`. 45 doc-concept pages now cover them; 14 guide chapters are
  title-only at this pin and produced no page. Five doc-concept pages whose source README no
  longer exists and that no current document covers were removed (DeepSeek decomposition tests,
  DeepSeek-V3 techniques, diffusers smoke tests, dynamic-shape compilation, quality benchmark
  suite); five other old slugs were rewritten from new documents.
- **Symbol index.** `catalog/symbols.tsv`: 19,064 symbols in 15,910 rows with signatures and
  doc lines; `catalog/edges.tsv`: 75,029 caller edges; `catalog/index.md`: module map over 716
  modules; 100% of documentable symbols represented. One area page (`areas/csrc.md`).
- **Verification.** The citation linter gates every page. The concept pages were then
  adversarially verified against the source at the pin (about 485 load-bearing claims; 34
  refuted and corrected in place, among them the error-handling root-op naming, the
  compile-builder path in compilation-pipeline, and a claim that attention uses the custom-kernel
  path). Verdicts live in `.cache/verify/` (not shipped); each page carries a `verified:` stamp.
- **Public-repository hygiene.** Internal bug-tracker IDs, short links, personal usernames in
  `TODO(...)` comments and internal build names were removed from the symbol index's doc column
  (16 rows) and kept out of every page; the generated catalog is scrubbed after each `finalize`.

## Gaps and caveats
- The repo's own docs disagree with the code in places, recorded on the doc-concept pages: GKE
  manifests set `TPU_LAUNCH_BLOCKING` where the code reads `TORCH_TPU_LAUNCH_BLOCKING`; the
  environment-variable table describes `TORCH_TPU_TIER2_COMPILATION_CACHE` as a path where the
  code takes a cache name under `/dev/shm`; the Docker README and `examples/README.md` describe
  build steps and paths that do not exist at this pin; two guide pages disagree on bfloat16 MXU
  accumulation.
- Only `csrc/` and `src/torch_tpu/` are indexed; `examples/` and `tests/` classes are described
  in prose.
- Source links are GitHub permalinks into a private repository; readers without access should
  open the local checkout at the pin instead.
- The synthesis lens is customer onboarding and optimization; build tooling, CI and experimental
  modules are index-only.

## Connections
- Updated: the `wiki/codebases/torch_tpu/` subwiki, the
  [TorchTPU hub](../codebases/torch-tpu.md), `config/torch_tpu.md`, `index.md`, `log.md`.
