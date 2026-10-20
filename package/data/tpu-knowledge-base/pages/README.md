# Grounded knowledge sources

A page carries its own evidence. Every source it cites is written out in
full at the bottom of the page, so a reader needs nothing else open and
the package's file reader needs no include expansion or extra request.

## Footnote citations

Pages declare the source IDs they use in YAML frontmatter:

```yaml
source_ids:
  - xla-scoped-vmem
  - xprof-memory-viewer
```

Place a citation beside the claim using standard Markdown footnote syntax:

```markdown
The typical scoped limit is distinct from physical capacity.[^xla-scoped-vmem]
```

and define it at the bottom of the page, with the title, the url, what
kind of document it is, what it supports, and when the url was last
checked:

```markdown
[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.
```

A source ID identifies an external document independently of page location
or lookup keywords, so reuse the same ID where several pages cite the same
document. Every declared ID must be cited, and every citation must be
declared. Evidence for a whole specification table can be cited immediately
above it. Keep derived arithmetic, secondary-source claims and unknown
values explicitly identified. Ordinary links between knowledge pages remain
Markdown links.

Definitions were generated from a shared `sources.json` catalog, which went
out with the page generator: a second copy of a url and a date, in a file
nothing read, drifting from the page that cites it. Edit a footer by hand,
and check the url when you do.

## Replay queries

The query that produced a page lives in its sibling `.recipe.json`, not in
the page: the question asked is authoring metadata, and the page itself
should carry only what a reader needs. A recipe holds that query, the
authoring request behind it, and the source IDs it may draw on -- what a
later regeneration needs. The page keeps `created`, `source_ids` for its
footnotes, and a `generation_recipe` pointer.

What the recipes used to carry alongside it -- provider, project, model id,
sampling config, source hashes, token usage -- went out with the generator
that wrote them, along with the `.generation.json` records. A detailed
account of a model call no tool in this tree can repeat is a claim, not
provenance.

Until then a page is edited by hand, and `human_reviewed: false` in its frontmatter remains until a person reviews it.
