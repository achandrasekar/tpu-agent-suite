# `raw/` — the sources the entries were rewritten from

**Nothing in here is ever edited.** These are verbatim campaign artifacts, copied out of a
campaign workspace before it was thrown away. One exception was made when this wiki moved into this
package: paths in the original repository were rewritten (`<campaign>/` for the campaign workspace,
no `references/` prefix for wiki files), and mentions of tools that are not installed were
reworded. No number and no job id was changed. They exist so that every rewritten entry in
`strategies/` has an original a reader can check it against.

Entries in `strategies/` are deliberately *not* copies — promotion restates a finding at the highest
altitude its evidence supports, which means dropping shapes, renaming constants and sometimes
splitting one campaign note into two rules (see `../PROTOCOL.md`, "Promotion is a rewrite, not a
copy"). That rewrite is where an entry becomes reusable, and also where it can quietly drift from
what was measured. `raw/` is the control.

## Layout

Flat, one uid prefix per campaign — `<YYYY-MM>-<short-slug>-<artifact>.md`. Flat for the same reason
`strategies/` is: a uid prefix greps as well as a directory does, and it keeps a two-file campaign
from costing a folder.

```
2026-09-sctopk-STRATEGY.md            the campaign's append-only rule catalog, written during the work
2026-09-sctopk-final_report.md        the shipped result: levers, rejections, correctness, carried items
2026-09-sctopk-WHITEBOARD-excerpt.md  §7h/§7h-old/§7j of an 891-line log — the arbitration record only
```

**Copy what the entries cite, not the whole workspace.** `STRATEGY.md` is not sufficient by itself:
it is a summary written mid-campaign, three of this campaign's ten entries have no line in it at
all, and two more take their job ids from `final_report.md`. `final_report.md` is the cheap,
load-bearing one — copy it always.

A long running log is the exception. Keep it as a **labelled section excerpt**: verbatim sections,
a header saying what was dropped and why, and a table naming which entry each section serves. The
sections worth keeping are the ones holding an *arbitration* — a measurement that retracted an
earlier number, and the reasoning as it stood before it — because those are the parts nobody would
reconstruct and the parts a future reader is most likely to disbelieve.

The uid is the whole `<YYYY-MM>-<short-slug>` stem. Pick it once, keep it stable, and never reuse
it — `source:` fields across `strategies/` point at it.

## Rules

- **Copy, never link.** The campaign workspace is disposable by design; a symlink into it is a dangling
  citation the day the tree is cleaned.
- **Never edit a file here**, including to fix an error. If a raw source is wrong, that fact belongs
  in the entry that cites it — the entry is where corrections live, and the disagreement between the
  two is itself information.
- **Every entry cites its source.** The `source:` field in an entry's frontmatter points at a file
  here and, where it helps, a section. An entry with no `source:` was not promoted from a campaign
  and should say so in `origin:` instead.
- **Not everything with a source is promotable.** Most of what is here is block-specific and stays
  here. `raw/` is an archive, not a queue.
- **Copy by citation, not by category.** The check that closes the question is mechanical: every job
  id and every number an entry quotes must resolve inside `raw/`. Run it before you delete the
  workspace. If it passes, the probe directories, roofline and design docs stay behind — they are
  the campaign, and copying them makes `raw/` the campaign too.
- **One exception: a null needs its pre-registration.** A measured null or a confound is only worth
  trusting if you can see the prediction was frozen *before* the run. If that prediction lives only
  in a probe's `README.md`, copy it or excerpt its hypothesis block; if the final report already
  records it, you do not need the probe.

## Contents

| uid | block | hardware | pins |
| --- | --- | --- | --- |
| `2026-09-sctopk` | `vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` @ `486af819` — GLM-5.2-FP8 DSA indexer top-k, 128k/256k context, k=2048 | TPU v7x (Ironwood), SparseCore | jax/jaxlib 0.10.2, libtpu 0.0.44.1 |

Promoted from this campaign: `STRAT-SC-01` … `STRAT-SC-10`, plus the SparseCore section of
`../hardware-facts.md`.
