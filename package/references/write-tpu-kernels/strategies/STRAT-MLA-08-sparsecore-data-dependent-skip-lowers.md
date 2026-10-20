```yaml
id: STRAT-MLA-08
title: On SparseCore, pl.when may wrap emit_pipeline and its grid length may be traced
hardware: tpu-v7x
unit: sparsecore
software: "jax 0.10.2 / jaxlib 0.10.2 / libtpu 0.0.44.1"
stage: [4]
lever: pipelining
generality: unit-wide
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-LOWER-01
recheck: re-run a two-line lowering probe of both constructs after any jax/jaxlib/libtpu bump
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-08 — Data-dependent skipping does lower on SparseCore

**Rule.** Two constructs widely assumed unsupported on SparseCore both lower: `pl.when` wrapping
`pltpu.emit_pipeline` on a `VectorSubcoreMesh`, and a **traced** (runtime-valued) grid length for
`emit_pipeline`. Together they are the enabling primitives for any data-dependent skip on the unit,
so the assumption that they are unavailable is expensive — it rules out a class of design before it
is costed.

**Mechanism.** None to state: this is a lowering capability established by probe, not a behaviour
derived from the hardware. It is recorded because a *negative* assumption about an API is
self-confirming — nobody measures what they believe will not compile — and because the cheapest way
to find out is a probe that takes minutes.

**Evidence.** Both constructs compile and are byte-exact
against the unmodified kernel. The two spellings of the same skip are also performance-equivalent,
landing within **0.6 µs** of each other at two geometries (**12.47 vs 12.35 µs**), so the choice can follow readability. Both re-verified as
explicit lowering arms in the shipped correctness suite, PASS.

**Scope.** v7x SparseCore at the pinned toolchain. A lowering capability is a fact about a build and
expires with it, which is why `software:` is pinned and `recheck:` is a probe rather than a
measurement.

**Does not apply as a performance claim.** That a predicate *lowers* does not mean the skipped work
*costs nothing*. On this unit the predicated-off path still pays window traversal — see
[[STRAT-MLA-07]], where ~10 µs/layer survived a skip that was working exactly as written. Lowering
buys the design; it does not buy the saving.
