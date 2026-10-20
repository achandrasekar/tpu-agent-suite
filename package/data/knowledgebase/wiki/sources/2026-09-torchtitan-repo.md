---
title: "torchtitan repo snapshot (2026-09)"
type: source
tags: [codebase, torchtitan, pytorch, parallelism]
created: 2026-09-25
updated: 2026-09-26
sources: 1
human_reviewed: false
---

Provenance record for the ingest of the `torchtitan` repository (the PyTorch-native
distributed training platform, in the fork that carries the TorchTPU backend) into the codebase
subwiki `wiki/codebases/torchtitan/`, produced by wikify-repo from the source at one pinned
commit. Knowledge lives on the subwiki pages; this page records where it came from and what was
left out.

## Identity

| | |
|---|---|
| Repository | https://github.com/google-pytorch/torchtitan: Google's TorchTitan, the only one with the TorchTPU backend. Not the official https://github.com/pytorch/torchtitan, which has no TPU backend; the two share no git history (history here starts 2026-03-10, upstream changes are ported by hand) and at this pin 301 of 804 files exist at the same path in upstream `main` (2026-09-25), 75 of them identical |
| Commit | `63e334d5ba90602b17b916e9e0c193059a294007` (main HEAD, 2026-09-23) |
| Snapshot date | 2026-09-25, wikify 0.3.4 |
| Acquired as | clean `git clone` of `main` at the pin (a fresh checkout passed to `wikify prepare --repo`; tracked files only). A developer working copy of the same repo was deliberately not used: the 2026-09-05 torch_tpu build showed untracked files leaking into the catalog from one. `config/torchtitan.md` holds the pin and the ingest settings (`catalog: index`, `index_profile: full`) |
| Indexed with | SCIP: scip-python over `torchtitan/*` (15 shards). Pure Python, so no compile database and no scip-clang stage |
| Authority | primary source: the implementation itself, its docstrings and comments, and its in-repo README files |

## What it produced
- `wiki/codebases/torchtitan/overview.md`, 24 concept pages and 45 doc-concept pages extracted
  from the in-repo docs.
- The symbol index (`catalog/symbols.tsv`: 9,889 symbols, one row each, with signatures and
  doc lines; `catalog/edges.tsv`: 29,086 caller edges; `catalog/index.md`: module map over 520
  modules). Coverage: 100% of documentable symbols and all 1,028 classes represented; 744
  symbols (7.5%) are covered by a concept page, the rest by the index.
- Withheld from the wiki: partner model packages, Google-internal launch and test
  infrastructure, and the doc pages derived from them. They are excluded from the graph, so they
  appear in no page, index row or edge; the exclusion list is kept outside the repository.
- Citation links (wikify 0.4.0 `source-links`, after the ingest): the 2,409 citations
  pointed at `catalog/<module>.md` pages that the `index` tier does not render. Each now
  links the symbol's source line at the pin, titled with its index key; path and line come
  from this build's `catalog/symbols.tsv`, with no re-index and no page text changed.
- The concept agenda was curated rather than taken from the planner: the parallelism axes
  (FSDP, TP, PP, CP, EP, activation checkpointing, DTensor layouts, torch.compile) each have
  their own page, and the six per-model directories fold into one `model-definitions` page with
  a section per family.
- The concept pages passed the citation linter on the first `finalize` pass, then were
  adversarially verified claim by claim by reviewers who had not written them: 428
  load-bearing claims checked against the source, 24 wrong statements found and corrected.
  The verdicts live in `.cache/verify/` (not shipped).
- The hub page [TorchTitan](../codebases/torchtitan.md) states the pin and routes into the
  subwiki.

## Gaps and caveats
- **The TPU backend needed a workaround.** wikify's planner classifies every `experiments/`
  path as non-library (`discover.DEFAULT_EXCLUDES`, not configurable), which would hide
  `torchtitan/experiments/tpu` - the TorchTPU integration, the most relevant subsystem for this
  knowledge base. It was seeded with explicit symbols instead (`torchtpu-backend` in the
  config), which `packet.resolve_seeds` matches against the whole symbol graph. Other
  `experiments/` subsystems (graph_trainer, jax, torchax, rl, torchft) have no concept page;
  they are covered by doc-concepts and the symbol index. Worth reporting upstream to wikify.
- Upstream bugs found and confirmed from source during verification are listed on the
  overview under "Known issues at this pin" (e.g. `get_peak_flops` returning `None` for an
  unrecognized TPU name, and the in-training Hugging Face export calling a function whose
  import is commented out). They are properties of the repo at this pin, not of this wiki.
- Many in-repo docs have drifted from the code (wrong flag names, stale support limits,
  classes that no longer exist). Each doc-concept page states the disagreements it found
  rather than repeating the doc.
- Source links are GitHub permalinks at the pin; a reader who cannot open them uses the link
  title (the index key) to find the row in `catalog/symbols.tsv`, or opens a local checkout at
  the pin.
- The synthesis lens (`synthesis_focus` in the config) is training on TPU; subsystems outside
  it (GPU-only kernels, CI, benchmark submission rules) are index-only.

## Connections
- Created: the `wiki/codebases/torchtitan/` subwiki, the hub page
  [TorchTitan](../codebases/torchtitan.md) and this page.
- Related subwiki: [torch_tpu](../codebases/torch-tpu.md) - the runtime TorchTitan's TPU
  backend binds to (Pallas, sync, StaticCompiler, hardware introspection).
