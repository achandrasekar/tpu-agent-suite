---
references:
- references/write-tpu-kernels/
---
# `references/` — the kernel wiki

This directory is the only part of `write-tpu-kernels` that survives a campaign. A kernel ships and
its campaign workspace goes stale; what stays true is the handful of things the hardware taught you
on the way. Those go here, and the next campaign reads them before it spends a job re-learning them.

**Read this file before reading or writing anything else in this directory.**

The installed copy of this directory is read-only. A campaign writes its promotions under
`<campaign>/wiki/`, with the same layout, and the package maintainer merges them here. `<campaign>`
is the campaign workspace, `{{runs}}/<run_id>/artifacts/write-tpu-kernels/<slug>/`.

---

## What belongs here, and what does not

An entry earns its place by being **true of the platform rather than of your block**. The test is a
single question:

> If someone optimized a completely different kernel next month, would this change what they do?

| belongs | does not belong |
| --- | --- |
| a hardware behaviour with a number on it | a result about your kernel |
| an API that does not do what its name says | a bug you fixed in your own code |
| a cost model that a measurement contradicted | a lever that won, without the mechanism that made it win |
| a method rule that stopped a wrong number reaching a report | anything you have not measured or read out of source |

A win with no mechanism is a benchmark result, and benchmark results go in the campaign's
`final_report.md` and the run's artifacts directory, `{{runs}}/<run_id>/artifacts/`, not here. "H16 was worth −15%" is a result. "A per-lane stride that is a multiple of the bank count
serializes 16-way, and 32 is the bank count" is an entry.

---

## Layout

```
PROTOCOL.md        this file — the protocol
INDEX.md           one row per entry: id, hardware, unit, stage, claim. Read this first.
hardware-facts.md  the chip's constants and the compiler's hard limits. Read whole, at step 1.
strategies/        one file per rule, STRAT-<AREA>-<nn>-<slug>.md
raw/               the cited campaign sources, <uid>-<artifact>.md. Verbatim, never edited. See raw/README.md.
```

**`raw/` is the control on `strategies/`.** Promotion rewrites a finding (below), and a rewrite is
where an entry becomes reusable *and* where it can quietly drift from what was measured. So the
campaign artifacts the entries cite are copied in verbatim under a uid prefix, and every entry's
`source:` field points back at one. Read `raw/` when you doubt an entry, when you need a number the
rewrite dropped, or before superseding something. Do not read it in the normal flow — it is
campaign-shaped, not reader-shaped, and that is exactly why the entries exist.

**Copy what your entries cite, not the whole workspace.** `STRATEGY.md` alone is rarely enough — it
is itself a summary, and its evidence lines routinely omit the job ids that live in
`final_report.md`. A long running log may be kept as a **labelled section excerpt** rather than
whole, provided the excerpt is verbatim, says what was dropped, and names the sections it keeps.
Keep the sections that hold an *arbitration* — a measurement that retracted an earlier number, and
the reasoning as it stood before — since those are the parts no one would reconstruct.

`hardware-facts.md` is deliberately not split into entries. It is the file you read *before* you
have a question, because most of what it contains invalidates a design silently rather than
answering something you thought to ask. Everything in `strategies/` is the opposite: read on demand,
at a named stage, because a question raised it.

Its §9, "Instruments that lie", is the one part read at step 5 rather than step 1, and it stays in
this file because it came from a workspace with no `raw/` copy in this tree, so it cannot become `strategies/` entries: there is no `source:`
for them to point at.

---

## Labels

Every entry opens with a fenced `yaml` label block, as below. It is not file frontmatter: the
package build strips frontmatter from references, and the labels must ship so that the greps below
work. **`hardware:` is mandatory and has no default.** An entry
without it is unusable — a number measured on one generation's silicon is not a fact about TPUs.

````
```yaml
id: STRAT-SC-06                     # stable. Never reuse, never renumber.
title: <the rule in one line>
hardware: tpu-v7x                   # tpu-v7x | tpu-v6e | tpu-v5p | tpu-v5e | tpu-v4 | any
unit: sparsecore                    # sparsecore | tensorcore | any
software: any                       # or a pin: "jax 0.10.2 / jaxlib 0.10.2 / libtpu 0.0.44.1"
stage: [3, 4]                       # which SKILL.md steps read it
lever: layout                       # algorithm | layout | pipelining | fusion | precision | method
generality: unit-wide               # narrow | unit-wide | general
status: measured                    # measured | source | modelled | unverified | superseded | retracted
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-06   # the verbatim original this rewrites
recheck: the bank count on the target part            # what to re-measure before reusing elsewhere
origin: campaign sparsecore-topk, 2026-09
```
````

Three of these are easy to get wrong.

**`hardware:` is where it was measured. `generality:` is how far you may carry it.** They are not
the same field and one does not imply the other. `hardware: tpu-v7x` + `generality: general` means
"measured on v7x, and the mechanism is not specific to v7x" — a claim you must be able to defend
from the mechanism, not from the fact that you only had one chip. When they disagree, the reader
trusts `generality:` and re-verifies on the new part.

**`software:` is not optional decoration for API entries.** A lowering bug is a fact about a
`jaxlib` build, and it expires. Pin it, and the next reader knows to re-check rather than assume.

**`recheck:` is the field that makes an entry portable.** One line naming the shortest measurement
that earns the right to reuse this somewhere else — the bank count, the lockstep width, the two
source ranges to re-read after a jax upgrade. If you cannot name one, either the entry is genuinely
`generality: general`, or you do not yet understand the mechanism well enough to write it down.

**The id prefix is provenance, not scope.** `STRAT-SC-*` means "found during the SparseCore
campaign". Several of those entries are not about SparseCore at all. Read `hardware:`/`unit:`/
`generality:`, never the prefix. Ids are stable so that the campaign workspace, `final_report.md` and
this wiki can cite the same rule; that stability is worth more than a tidy taxonomy.

---

## Reading: which stage reads what

`SKILL.md` names five steps. Do not read the wiki whole at step 1 — read the rows your current step
owns, and read the body of an entry only when its INDEX row is relevant to the block in front of you.

| step | read |
| --- | --- |
| 1 — shapes, dtypes, hardware limits | `hardware-facts.md` **whole**, plus `stage: 1` entries. These set the roofline and the gate. |
| 2 — dataflow, buffers, reference mining | `stage: 2` — pass costs, transfer behaviour, what the unit is actually good at. |
| 3 — algebra and the freeze gate | `stage: 3`, and re-read any `lever: layout` entry before locking a layout. |
| 4 — mechanical lowering | `stage: 4` — the APIs that lie. Cheapest possible read; each one is a bug you do not write. |
| 5 — probes and profiling reads | `hardware-facts.md` **§9**, plus `stage: 5` entries — probe method and diagnostic order. Read **before** dispatching the probe, not while reading its output. |

The query is a grep, not a directory walk:

```bash
cd {{reference:write-tpu-kernels/}}
grep -l "stage:.*\b4\b"       strategies/*.md            # what step 4 reads
grep -l "hardware: tpu-v7x"   strategies/*.md            # what was measured on this part
grep -L "generality: narrow"  strategies/*.md            # what is safe to carry to a new part
```

Filing is flat on purpose. Directories per generation would force the rules that are about method
rather than silicon — over half of what is here — into a hardware bucket they do not belong in, and
would hide an entry the moment it turned out to generalize.

**A `status: retracted` or `superseded` entry stays in the tree.** It is deleted from `INDEX.md`'s
live table and moved to the footer, with the entry that replaced it named. The whole value of
recording a wrong cost model is that the next person recognises the reasoning as theirs.

---

## Writing: promoting a rule out of a campaign

`SKILL.md` has kernel work write `STRATEGY.md` in the campaign workspace, `<campaign>/`, as it goes —
a compact entry each time a mutation wins or a hardware rule falls out of a probe. That file is a
campaign artifact and it is **not** the wiki. Promotion is a separate, deliberate step at the end of
the work.

**`STRATEGY.md` is where you start, not where you stop.** It holds what the campaign *noticed it was
learning*, which systematically excludes everything learned by being wrong. Sweep the final report
too, and specifically these four places, none of which write to `STRATEGY.md`:

| where | what hides there |
| --- | --- |
| the **rejections** table | a measured null with its mechanism — the most reusable thing a campaign produces, and the one nobody writes up |
| the **correctness** section | how the gate was shown to be able to fail, and which cell classes were blind |
| **anything the campaign says it predicted backwards** | a cost model that a measurement contradicted, which is on the belongs-here list |
| the **carried** items and harness notes | method rules that stopped a wrong number reaching a report |

Re-ingesting this campaign from `raw/` found six such entries against seven in its `STRATEGY.md`.
Expect roughly half your entries to come from outside it.

Promote when all five hold:

1. **The mechanism is named.** Not "this was faster" — *why the silicon made it faster*, in terms
   that predict where else it applies.
2. **The evidence is a `raw/` source or a pinned source path**, and the number quoted is the number
   that source records. Entries carry no job ids: the `source:` field points at the `raw/` copy, which
   keeps them. Tag a pinned-source figure `[source: path@sha]`. A modelled figure is tagged `[modelled]`
   and its entry is `status: modelled`. A figure tallied *across* a campaign rather than printed by
   one job — "61/124 cells tripped the gate" — is promotable, but say in the body that the tally is campaign-wide. Do not let it masquerade as a
   single-job number.
3. **It is not already here.** Grep first. If it sharpens an existing entry, edit that entry and
   add the new `source:` — do not open a second one that half-contradicts the first.
4. **The scope survives contact.** Write down what it does *not* apply to. An entry with no
   negative scope has not been thought about.
5. **The hardware label is real.** If it was measured on one part and you believe it generalizes,
   say so in `generality:` and say why in the body.

### Promotion is a rewrite, not a copy

**A `STRATEGY.md` line is written by someone holding one kernel. A wiki entry is read by someone
holding a different one.** Those are different sentences, and copying the first into the second is
the main way this directory fills up with things nobody can use. Before an entry lands, put it
through four moves.

**1. State the invariant, not the instance.** The **Rule** line is what the mechanism guarantees.
Every number you measured belongs in **Evidence**. If the Rule contains a shape, a bucket count, a
speedup, or one of your kernel's identifiers, you have written the instance.

> ✗ `A per-lane table addressed lane * NUM_BUCKETS + bucket serializes whenever NUM_BUCKETS % 32 == 0.`
> ✓ `A per-lane stride sharing a factor with the bank count collapses all lanes onto one bank.`
> — 256, 32 and the 16-way ceiling move down into Evidence, where they are a demonstration rather
> than a precondition.

**2. Name every constant.** `32` becomes "the bank count, 32 on v7x". `0.97` becomes
"`E[max over W]`, ≈0.97n for W=32". A named constant can be re-measured on a new part; a bare number
can only turn out to be wrong. This is what `recheck:` is derived from.

**3. Lead with the most transferable sentence.** If the reusable half of an entry is sitting in
*Does not apply when* while the Rule line is about your block, the entry is upside down. A confound
you fell into, or a diagnostic that narrowed the search, is often worth more to the next reader than
the lever that eventually won — and if it is a rule in its own right, **give it its own file** so a
`stage:`/`lever:` grep can find it.

**4. Write `recheck:`.** The shortest measurement that lets someone reuse this on other hardware or
a later toolchain.

### When to reject

After those four moves, apply the test: **can the Rule still be stated without naming your block's
shapes, dtypes or files?** If not, reject it. What you have is a result, and results go to
the run's artifacts directory and `final_report.md`, which is where someone comparing kernels will look for
them. Specifically, do not promote:

| looks like a rule | actually is | goes to |
| --- | --- | --- |
| "H16 was worth −15.08%" | a delta with no mechanism | the run's artifacts directory |
| "fusing the cvt pass into L0 saved a full-width pass" | a description of your patch | `final_report.md` |
| "this kernel is compute-bound" | a property of your block | `WHITEBOARD.md` |
| "`emit_pipeline` was fine here" | one observation, no ceiling and no mechanism | nowhere — measure why, or drop it |

Rejecting is the normal outcome for a *lever*, and the reject table above is mostly full of them.
It is **not** the normal outcome for a `STRATEGY.md` line: those were written as candidate rules
already, and most survive the rewrite intact. Judge yourself on the rewrite, not on a promotion
count — an entry whose Rule line still names your block has failed even if you promoted only one,
and thirteen invariant-led entries off a campaign this size is not over-promotion.

Then, in this order, under `<campaign>/wiki/`:

```
raw/<uid>-STRATEGY.md, raw/<uid>-final_report.md, …   # copy the cited sources in FIRST, verbatim
strategies/STRAT-<AREA>-<nn>-<slug>.md                # next free nn in that area, never reuse
INDEX.md                                              # the new rows only; keep them sorted by id
```

**Copy `raw/` in before writing a single entry.** The rewrite is checkable only against an original
that already exists, and the campaign workspace is disposable — a campaign cleaned up before its sources
were copied leaves entries nobody can audit. Pick the uid once (`<YYYY-MM>-<short-slug>`), never
reuse it, and point every `source:` at it.

"Copy what the entries cite" is circular if read strictly, since you do not know the citations until
the entries exist. Resolve it in two passes: **`STRATEGY.md` and `final_report.md` go in first,
unconditionally** — they are small and you will cite both — and the **log excerpt goes in last**,
once the drafted entries tell you which sections they lean on.

The body is four headings and nothing else — **Rule**, **Mechanism**, **Evidence**, **Scope** —
followed by an optional **Does not apply when**. Keep an entry under ~25 lines. If it needs more,
what you have is a reference file, and it needs its own name and its own INDEX row.

### What a rejection is worth

A measured null is one of the most valuable things this wiki can hold, and the one most likely to be
dropped on the way out of a campaign because nobody wants to write up a lever that did not work.
Promote it with the number that rejected it. The next campaign will propose the same lever — it will
sound just as plausible then — and the entry is what stops it spending another job.

### Superseding

Never edit a measured number to a better one. Add the new figure and its `source:` alongside, and
say which conditions separate them. If an entry is now wrong, set `status: superseded`, add
`superseded-by: STRAT-…`, and leave the reasoning intact.
