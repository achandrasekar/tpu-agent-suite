# Knowledge Base — Schema & Operating Rules

You are the maintainer of the knowledge base shipped with the **tpu-agent-suite**
distribution: a file-based wiki about TPU hardware, model optimization, and the supported
codebases. Agents build it, cross-link it, and answer from it. Raw sources are ground
truth; the wiki is synthesis; this file is the rules.

**Judge every decision against one question:** *does this help an agent make a correct,
cited TPU optimization or onboarding decision?*

## Use cases (priority order)

1. Grounded questions about **TPU hardware and model optimization**.
2. Grounded questions about the **supported codebases** — torch_tpu, TorchTitan,
   vLLM/tpu-inference, SGLang, tokamax, JAX/Pallas, MaxText, and peers.
3. **Converting customer CUDA/PyTorch code to TorchTPU.**
4. **Optimizing Pallas kernels.**
5. **Optimizing models end-to-end on TPU.**

Reference pages (`hardware/`, `codebases/`, `concepts/`) serve 1–2; task-shaped
`playbooks/` (op mappings, kernel recipes, e2e checklists) serve 3–5.

## Principles

- **Three layers**: `raw/` = immutable ground truth (read, cite, never edit) → `wiki/` =
  agent-owned markdown synthesis → `SCHEMA.md` = the rules.
- **Knowledge compounds**: every ingest updates and cross-links the pages it touches;
  answers that took real work are filed as analysis pages, never re-derived.
- **Trust is explicit**: every actionable claim (spec number, API behavior, perf figure,
  onboarding rule) carries an inline citation to a source page, whose `## Sources` names
  the raw file. Uncited claims are marked `*(unverified — agent inference)*`. Unreviewed
  synthesis is usable but presented as agent-derived, not authoritative.

## Directory layout

```
./                       <- knowledge base root (agent's CWD; all paths relative to it)
  SCHEMA.md              <- this file
  CLAUDE.md              <- @SCHEMA.md pointer for the agent harness
  index.md               <- content catalog
  log.md                 <- append-only event log (newest first)
  config/<slug>.md       <- wikify-repo ingest config per code repo (repo, pin, focus)
  .cache/                <- wikify indexes and packets (gitignored, never shipped)
  raw/
    docs/                <- design docs, guides, curricula, blog posts, talks
    papers/              <- research papers
    code/<slug>/         <- pinned repo checkouts for wikify (gitignored, never shipped)
    assets/              <- images extracted from sources
  wiki/
    kernel-optimization-index.md  ← router for kernel/: read order + per-category load mandate
    hardware/            ← one spec page per TPU generation
    codebases/           ← one hub page per supported repo + <slug>/ wikify subwiki
    concepts/            ← techniques, architectures, terms
    playbooks/           ← task recipes: onboarding, kernel opt, e2e optimization
    kernel/              ← imported Pallas kernel-optimization corpus, entered via the
                           router above; pages are `type: concept` but kept grouped
                           because they are a curated external set, not our synthesis
    sources/             ← one summary page per ingested raw source (provenance layer)
    analyses/            ← saved query outputs, comparisons, syntheses
```

## Page format

```yaml
---
title: "<human-readable>"
type: hardware | codebase | concept | playbook | source | analysis
tags: [<canonical tags, see below>]
created: YYYY-MM-DD          # ISO dates everywhere
updated: YYYY-MM-DD
sources: 0                   # number of raw sources informing this page
human_reviewed: false        # set true only by/with a human. Agent edits that touch
                             # claims reset it to false (note the reset in log.md);
                             # pure housekeeping (metadata counts, link formatting,
                             # section scaffolding) preserves it but is still logged
aliases: []                  # optional alternative names
---
```

Body: a 1–3 sentence **lead** defining the topic; H2 sections; `## See also` links to
related pages; `## Sources` listing the source pages behind it.

- **Links**: relative markdown links with `.md` extensions, spaces URL-encoded as `%20`.
  Never `[[wikilinks]]` — they don't render on GitHub. Codebase pages carry a pointer
  to their canonical docs/repo URL (marked "not yet ingested" until it is) — it's a
  fetch target for future ingests, not a citation.
- **Contradictions**: never silently overwrite a claim. Mark the old one
  `> [!warning] Contradicted by [source page](path)`, add the new claim below it, and
  leave adjudication to a human.
- **Stubs**: same frontmatter with `tags: [stub]`, `sources: 0`; body is
  `*Stub — expand when more sources are available.*` plus `## See also`.

## Page types

| Type | Directory | Holds |
|---|---|---|
| `hardware` | `wiki/hardware/` | One spec page per TPU generation: cited tables of HBM, FLOPs, ICI, pod/host topology. Volatile per-chip numbers live here |
| `codebase` | `wiki/codebases/` | One hub page per supported repo (`<repo>.md`: what it is, the pin, routing); once the repo is ingested, a generated subwiki `<slug>/` built by wikify-repo (below) |
| `concept` | `wiki/concepts/` | One mechanism/technique/term per page — hardware blocks included (MXU, VMEM, ICI), parallelism axes, precision, compilation, profiling. Timeless mechanism lives here |
| `playbook` | `wiki/playbooks/` | Task recipes (template below) |
| `source` | `wiki/sources/` | Provenance record per ingested raw source: what it is, its authority, what it changed, what wasn't extracted yet |
| `analysis` | `wiki/analyses/` | Filed answers: comparisons, investigations, root-cause writeups |

### Codebase subwikis

A supported repo is ingested by the `wikify-ingest-repo` skill, never by hand. `config/<slug>.md`
names the repo, the pin (`ref:`), the languages and the synthesis focus; the skill indexes the
pinned checkout (`raw/code/<slug>/`, gitignored), writes one cited page per subsystem and gates
the result with a citation linter. The output is `wiki/codebases/<slug>/`:

- `overview.md` is the entry page (what the system is, a task-to-page map, core diagrams);
  `concepts/` holds one mechanism per page with every claim citing a symbol; `doc-concepts/`
  covers the repo's own docs; `changes/` records what moved between pins; the subwiki has its
  own `index.md` and `log.md`.
- `catalog/` is the symbol index, not pages: `symbols.tsv` (one row per symbol: anchor, path,
  line, kind, rank, hash, callers, citing pages, signature, doc line; sharded into
  `symbols/<dir>.tsv` for large repos), `edges.tsv` (every caller edge) and `index.md` (the
  module map). A citation links the symbol's source line at the pinned commit and is titled
  with its index key (`<path>#<Symbol>`, column 1 of `symbols.tsv`); in a private repo the
  link needs access, the key does not. Links into a private `https://github.com/google-pytorch/`
  repo (torch_tpu) return 404 to a plain fetch; with access, read them through `gh`
  (`gh api -H "Accept: application/vnd.github.raw"
  "repos/google-pytorch/<repo>/contents/<path>?ref=<pin>"`). `google-pytorch/torchtitan` is
  Google's TorchTitan with the TorchTPU backend, not the official `pytorch/torchtitan`, which
  has none and has diverged. How to query it: "Code wikis (wikify)" below.
- Front matter is wikify's (OKF 0.2: `generated`, `verified`, `sources`), not the KB page
  schema; `verified: [{by: human:<id>}]` plays the role of `human_reviewed`. The KB schema
  applies to the hub page `wiki/codebases/<repo>.md` (pin and routing) and to the provenance
  record in `wiki/sources/`.
- `index.md` lists a subwiki once, through its `overview.md`; the validator and the tests count
  it as one entry. Source links are GitHub permalinks at the pin (`source_url` in the config),
  so pages read the same in a clone and in the shipped distribution, which never carries
  `raw/code/`, `.cache/` or `.wikify/`.
- Update: bump `ref:` in `config/<slug>.md` and run the skill again ("wikify torch_tpu"); only
  pages whose cited symbols changed are rebuilt. Never hand-edit `catalog/`, `index.md`,
  `log.md` or `changes/` inside a subwiki.

### Playbook template

Written for an agent to execute, not a human to browse:

- **Applies when** — checkable preconditions (framework, hardware, symptom).
- **Steps** — numbered; each gives the command/edit and the signal that it worked.
- **Mappings** (onboarding playbooks) — `CUDA/PyTorch construct → TorchTPU equivalent →
  caveats` tables, every row cited.
- **Pitfalls** — known failure modes and their signatures.
- **Verification** — benchmark / profile / numerics check.

## Concept extraction rules (during ingest)

For each named thing the source discusses non-trivially (technique, hardware block, tool,
API, config knob, failure mode), decide in order:

1. **Already paged?** Check `index.md` under its name *and* synonyms/codenames
   ("Trillium" = "TPU v6e") → update that page, add the new name to `aliases:`.
   Never create a duplicate.
2. **TPU optimization concept** (sharding strategy, double buffering, fusion,
   rematerialization, roofline analysis, layout/padding, collective cost model, kernel
   pattern) → **always page it**, as a stub if the source is thin. This is the wiki's
   core inventory; err toward paging.
3. **ML concept** (attention variants, MoE, KV cache, quantization, optimizers, norms)
   → page it when it has a TPU-relevant performance/memory/implementation angle — most
   load-bearing ones do — and **write it from that angle**: 1–2 cited sentences of
   definition, then the TPU mapping and costs (attention → MXU/VMEM tiling; quantization
   → natively executed formats). No textbook ML. No TPU angle at all ("gradient descent",
   "tokenizer") → inline mention, nothing created.
4. **Variant or sub-mechanism of an existing concept** (a new PP schedule, an API flag)
   → section on that page; split out only past ~500 lines.
5. **Else route by type**: TPU generation → `wiki/hardware/` (spec page), repo/tool →
   `wiki/codebases/`, anything else page-worthy — hardware blocks and mechanisms
   included → `wiki/concepts/`. Papers stay source pages — never double-paged.

Full page if the source supports a lead + one cited section; else stub. A stub graduates
when a second source touches it or a playbook/query needs it.

**Granularity** — a concept page is a **decision-sized technique**: something an agent
adopts, tunes, or rules out as a whole. Litmus: can you state when to use it and its
cost/benefit without leaning on a parent technique?

| | Rule | Examples |
|---|---|---|
| Too coarse | Umbrella bundling independent decisions — split | "parallelism" → FSDP, TP, PP, EP, CP pages; "memory optimization" → remat, activation checkpointing, quantization |
| Right | One named technique with its own tradeoffs and cost | FSDP; double buffering; scan-layers; roofline analysis; KV cache; GSPMD |
| Too fine | A knob, variant, or detail of one technique — section | DualPipe-V → in `pipeline-parallelism.md`; a `jax.jit` kwarg → in `jax-jit.md`; a tile-size heuristic → in its kernel pattern's page |

Umbrella terms agents genuinely search for ("distributed training", "quantization") may
get a short **hub page**: lead + comparison table + links down to the decision-sized
pages, no deep content of its own.

Link every new page from its source page and from one related page's `## See also`. When
two concepts interact, state *how* in a cited sentence on both pages — a bare link
carries no mechanism. Unsure between page and section → section; prefer candidates that
feed playbooks over encyclopedic completeness.

## Canonical tags

| Kind | Tag | Examples |
|---|---|---|
| Hardware generation | `#hardware` | TPU v4, v5p, Trillium (v6e), Ironwood (v7) |
| Compiler / runtime | `#compiler` | XLA, MLIR, Pallas, Mosaic, PJRT |
| Framework | `#framework` | torch_tpu, JAX, TorchTitan, MaxText, vLLM, SGLang |
| Kernel | `#kernel` | Pallas kernels, tokamax, FlashAttention |
| Onboarding | `#onboarding` | CUDA→TorchTPU mappings and rules |
| Parallelism | `#parallelism` | FSDP, TP, PP, EP, CP, GSPMD |
| Precision | `#precision` | bf16, quantization, AQT |
| Performance | `#performance` | rooflines, profiling, Xprof, autotuning |
| Interconnect / memory | `#interconnect` `#memory` | ICI, DCN, HBM, VMEM |
| Paper | `#paper` | any published/arXiv paper |
| Workload | `#workload` | training, inference/serving, MoE, diffusion |

## Operations

### INGEST — a file lands in `raw/` (or text/URL is provided) with an instruction to ingest

1. **File the raw source** under the right `raw/` subdir (naming below); for URLs, fetch
   and snapshot first. Immutable from this moment.
2. **Read it fully.**
3. Human present → **discuss takeaways** and emphasis before writing. Autonomous →
   record editorial decisions in `log.md` instead.
4. **Write the source page** `wiki/sources/<date-prefix>-<slug>.md` (prefix per the
   Naming table: publication year, or snapshot month for living sources) — a **provenance record,
   not a second knowledge store**: identity + snapshot date, authority level (official
   docs / peer-reviewed / book / vendor claim), Gaps & caveats, Connections (the wiki
   pages this source created or updated). Knowledge lives once, on wiki pages.
   - **Residue rule**: a source page carries content only when it was deliberately not
     extracted yet (tables held for a later ingest, playbook fodder awaiting a second
     source) — and says so, so a later pass can file it.
   - Rich many-page sources (books, handbooks, repo snapshots) may add a short claims
     digest, since no single wiki page covers them. Code repos are ingested by wikify-repo
     into a subwiki (see "Codebase subwikis"); their source page is the provenance record
     of the pin, and a re-ingest at a new pin gets a new dated source page while the
     subwiki's `changes/` holds the delta.
   - A 1:1 source (spec sheet → one hardware page) gets metadata only, ~15 lines.
5. **Update connected pages**: integrate claims with inline citations, append to
   `## Sources`, bump `sources:`/`updated:`, reset `human_reviewed`.
6. **Extract concepts** — run the decision list above on every candidate.
7. **Update `index.md`; append to `log.md`.**

### QUERY — someone asks a question

1. Read `index.md` first — never guess which pages exist.
2. Read the relevant pages, following links as needed.
3. Answer with citations to wiki pages (they carry the raw-source chain); distinguish
   `human_reviewed` claims from unreviewed synthesis when it matters.
4. Real synthesis work → file it at `wiki/analyses/<yyyy-mm-dd>-<slug>.md`, update
   `index.md` + `log.md`.

### LINT — "lint" / "health check", or before cutting a release

Report: unresolved `[!warning]` contradictions; orphan pages (no inbound links); broken
links; stale stubs fillable from existing sources; uncited claims; playbooks referencing
APIs changed in a newer code snapshot; suggested next sources based on gaps. Fix
mechanical issues; flag the rest.

<!-- wikify:begin -->
## Code wikis (wikify)
Grounded internals wikis for ingested repositories live under `wiki/codebases/<slug>/` (`wiki/codebases/index.md` lists
them); every claim on a concept page cites a real symbol, gated by a linter at build time.
- **Mechanism:** start at `wiki/codebases/<slug>/overview.md` (questions and tasks to pages), then
  `areas/<area>.md` (what an area is for, its units) and `concepts/<unit>.md` (how a subsystem works,
  cited). Grep `index.md` descriptions and pages' `aliases:` to pick a page; read only that page.
- **Source:** a citation links the symbol's source line at the pinned commit, and its title is the
  symbol's index key `<path>#<Symbol>`.
- **Symbols:** the index `catalog/symbols.tsv` (`symbols/*.tsv` when sharded; columns anchor, path,
  line, kind, rank, hash, callers, citing pages) and `catalog/edges.tsv` (`callee<TAB>caller`), both
  headed by their recipes; `catalog/index.md` maps modules. Grep by anchor, never read a file whole,
  and mind the search tool's line cap: `grep -c` first, narrow a hub's callers by directory
  (`cut -f2 | cut -d/ -f1-2 | sort | uniq -c`), end every listing with `| head -20`.
  `grep -P '^<path>#<Symbol>\t'` gives one row or, on edges, its callers;
  `grep -P '\t<path>#<Symbol>$'` on edges gives what it calls;
  `grep -P '[#.]<Name>\t' | sort -t$'\t' -k5 -nr | head -20` finds a bare name,
  `grep -P '^<dir>/.*#<Name>\t'` a name within an area. Source links need repository access (a private
  repo returns 404): never WebFetch them; `grep -P '^<title>\t'` gives the row (path, line, signature,
  doc). For a body, read a local checkout; else try once `gh api -H "Accept: application/vnd.github.raw"
  "repos/<owner>/<repo>/contents/<path>?ref=<pin>"` and on failure answer from the row.
- **Trust:** `verified:` in the front matter says who checked a page; without it the page is
  agent-generated, say so when you rely on it. Changes between pins: `changes/<ref>.md` and `log.md`;
  the pin is `commit:` in the silo's `index.md`.
- Never bulk-read pages or hand-edit `catalog/`, `index.md`, `log.md`, `changes/` or `area:auto`
  blocks (regenerated by `wikify finalize`). To ingest or update a repository, invoke the
  `wikify-ingest-repo` skill: "wikify <repo url or path>".
<!-- wikify:end -->

## index.md / log.md

`index.md` is a catalog, not a memo — one line per page:
`- [Title](wiki/<dir>/<slug>.md) — one-line summary.` Grouped:

```markdown
# Knowledge Base Index
## Hardware
## Codebases
## Concepts         → subsections: Architecture / Parallelism / Precision & Numerics /
                      Performance & Profiling / Compilation
## Playbooks        → subsections: Onboarding (CUDA → TorchTPU) / Pallas kernels /
                      End-to-end optimization
## Sources
## Analyses
```

No page count and no last-updated line: `scripts/validate_kb.py` (pre-commit and CI)
already checks that every page under `wiki/` is cataloged and every link resolves, which
implies the count, and `log.md` carries the dates. Section headings carry **no counts**
either — nothing maintains them, and a hand-kept number is a merge conflict on every rebase.

`log.md` — append-only, newest first:

```markdown
## [YYYY-MM-DD] <operation> | <title>
**Operation**: ingest | query | lint | manual      ← the only tokens the validator accepts
**Pages created**: list
**Pages updated**: list
**Notes**: decisions, guidance received, contradictions found, human_reviewed resets
```

## Naming

Slugs: lowercase, hyphens only.

| Item | Convention | Example |
|---|---|---|
| Hardware (generation) pages | `wiki/hardware/tpu-<version>.md` — official version tag, lowercased; codenames (Trillium, Ironwood) only in `title`/`aliases` | `wiki/hardware/tpu-v6e.md`, `wiki/hardware/tpu-7x.md` |
| Codebase / concept pages | `wiki/<dir>/<slug>.md` | `wiki/concepts/systolic-array.md` |
| Codebase subwikis | `wiki/codebases/<slug>/` (wikify-repo silo; slug as in `config/<slug>.md`; entry page `overview.md`) | `wiki/codebases/torch_tpu/overview.md` |
| Playbooks | `wiki/playbooks/<task-slug>.md` | `wiki/playbooks/convert-cuda-streams.md` |
| Source pages | date prefix at the source's **natural granularity**: `<yyyy>` = publication year for one-off works (papers, books, posts); `<yyyy-mm>` = snapshot month for living sources (docs sites, repos — re-ingests get new files, no collisions) | `wiki/sources/2025-scaling-book-tpus.md`, `wiki/sources/2026-08-cloud-tpu-v6e-docs.md` |
| Analyses | `wiki/analyses/<yyyy-mm-dd>-<slug>.md` | `wiki/analyses/2026-08-22-v6e-fsdp-hang.md` |
| Raw docs / papers | same prefix as their source page | `raw/papers/2022-flashattention.pdf`, `raw/docs/2026-08-cloud-tpu-v6e-docs.html` |
| Raw code | `raw/code/<slug>/` (pinned checkout managed by wikify: `ref:` in `config/<slug>.md`, `commit:` in the subwiki index; gitignored, never shipped) | `raw/code/torch_tpu/` |
| Assets | `raw/assets/<source-slug>-<n>.<ext>` | `raw/assets/pallas-handbook-fig1.png` |

## Rules

1. Never modify `raw/` once ingested.
2. Read `index.md` before answering; update `index.md` + `log.md` after any operation
   that creates or changes pages. `python3 scripts/validate_kb.py --check-git` verifies
   both; pre-commit runs it, and CI (`make verify`, plus a log-entry check against the PR
   base) gates on it.
3. Every actionable claim: cite it or mark it unverified. Never silently overwrite —
   use the `[!warning]` block.
4. Prefer tables for numbers (specs, benchmarks, bandwidths) and mappings
   (CUDA→TorchTPU); prose for mechanisms.
5. One topic per page; split into a subwiki past ~500 lines.
6. Playbooks must be executable by an agent — checkable preconditions, steps with
   success signals, a verification section. If an agent can't execute it, it isn't done.
