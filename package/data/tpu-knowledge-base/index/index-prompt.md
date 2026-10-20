<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Writing an index row

A prompt for whoever — person or model — adds or changes rows in `index/`.
It is for maintainers and is never sent at lookup; the lookup-time
instructions are `../prompts/`. Fill in the three inputs and hand it over.

---

You are writing index rows for tpu-knowledge-base. A row is a routing record, not
a summary for humans: its only jobs are to be found by the questions this
source answers and to *not* be found by questions other sources answer better.

## Input

- Source: {URL, repo path to a file, or authored page name}
- Target index file: {e.g. index/pallas.jsonl}
- Today: {YYYY-MM-DD}

## 1. Read the source, and only the source

Fetch or open it. If a URL does not return 200 with the real content — watch
for sign-in pages, JS shells, "older version" banners, redirects to a landing
page — stop and report: a dead row is worse than no row. Everything below must
come from what you just read, not from memory.

## 2. Choose the kind (exactly one)

- `url` — a public, fetchable, stable page. An entry is generated from it at
  lookup.
- `file` — a repo file used as evidence; an entry is generated from it.
  Prefer this for long documents: `page` rows are ranked on their full text,
  and long pages crowd out better entries.
- `page` — authored knowledge with no citable source, served as-is from
  `pages/`. No prompt, no template. It says which claims are documented and
  which observed, and carries `authority: first-party-verified`.

## 3. Pick prompt and template from what exists

`prompt` is an id in `../prompts/prompt_table.jsonl` (`howto`, `reference`,
`concept`, `tpu-spec`, `mldiagnostics`, `gcloud-reference`, `repo-overview`,
`paper`). `template` is a file in `../templates/` (`sw-doc`, `sw-design`,
`hw-spec`, `paper`, `repo-overview`). Pick by what the page *is*, not by its
topic: tutorial → `howto`/`sw-doc`; API or flag list → `reference`/`sw-doc`;
mechanism → `concept`/`sw-design`; chip numbers → `tpu-spec`/`hw-spec`. Never
invent an id.

## 4. Keywords

Keywords are what a caller types; resolution is exact match first, then ranked
search. Write 4–20, lowercase-kebab unless the real token has case (API names,
error strings and env vars keep their exact spelling).

Include, in priority order:

1. The row's own id and one or two natural names for the page.
2. Tokens this source is authoritative for: exact API symbols, flags, env
   vars, error strings, config keys, distinctive technique names
   (`CompileTimeScopedVmemOom`, `vmem_limit_bytes`, `TORCH_TPU_DEFER_AND_FUSE`).
3. Symptom phrases a user would say when this page is the answer
   (`eager-too-slow-many-small-ops`).

Every keyword must be grounded in the exact text this row serves:

- A keyword that names a symbol, flag or error must appear **verbatim** in
  that text. `grep -F` it.
- A phrase keyword must name something the text actually covers, not
  something it mentions in passing.
- When the row's source changes — `url` to `file`, a different copy, a new
  version — re-check every keyword and the summary against the new text. Two
  renderings of "the same document" are often not the same text: a notebook's
  HTML can carry code and sections its markdown export does not, and a
  keyword grounded in one sends callers to a page that cannot answer them in
  the other.

Leave out:

- A general topic word plus a `-tpu` or `-pallas` suffix
  (`speculative-decoding-tpu`, `ici-vs-dcn`, `tpu-v7-vmem`), unless this source
  is the best answer in the whole index to that general question. If a
  canonical source (vendor docs, the scaling book, a hardware spec row) covers
  it, leave the question to that source.
- Any keyword another row already claims — the loader rejects duplicates.
  Check with `grep -hF '"<keyword>"' index/*.jsonl`.

## 5. Summary

One sentence, 25–60 words, in the words a caller would ask with, naming the
concrete things the text covers (APIs, numbers, procedures) in the order it
weights them. It is ranked, never served, so make it dense and specific. Every
name in it passes the same verbatim check as the keywords. If the text covers
several subjects, name them; do not pad with subjects it only touches.

## 6. Provenance, honestly

- `authority`: reuse a value already in the index (`vendor-docs`,
  `first-party-verified`, `third-party-verified`, `paper`, `oss-repo`,
  `vendor-blog`).
- `verified`: today only if you read this exact source today. Changing the
  source does not carry the old date or state over — re-check, or downgrade.
- `verification_state`: `human-verified`, `machine-verified` or `unverified`.
  Use `unverified` when figures or claims were not re-checked. Nothing at
  lookup reads this field, so it does not warn the caller: a page that
  contradicts pages already shipped should not be indexed until reconciled.
- `verified_from` (optional): exactly what the keywords were checked against,
  including the upstream commit where there is one.
- `license`: the source's, if known.
- `note` (optional, never sent to a model): where the content came from
  (upstream repo and commit or version), what the row replaced, and why this
  kind. Say only what you checked; "synced from X" must be true of X.

## 7. Check before handing it back

1. The row is one line of valid JSON with exactly one of `url`/`file`/`page`,
   and `python3 -m unittest tests.test_provenance_table` passes (run from
   `package`).
2. Three questions the source should answer: write the keyword a caller
   would actually type for each, and check the row lists it. A lookup is
   exact, so a keyword nobody would guess reaches nothing.
3. Three to five *general* questions near its topic that a canonical source
   should answer: none of their keywords belongs on this row. A keyword on
   two rows is a silent shadow -- the second never answers.
4. Any hard numbers the source states — capacities, bandwidths, limits — agree
   with rows and pages already shipped for the same hardware. Report a
   conflict instead of indexing over it.
5. `python3 tests/search_eval.py` does not regress, and the queries from 2 and
   3 are added to `tests/search_eval.jsonl`.

## Output

- The JSONL row or rows.
- A short report: the kind and why; keywords dropped as ungrounded or too
  general; the searches from step 7 with their scores; any contradictions;
  what was not verified.
