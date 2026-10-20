```yaml
id: STRAT-SC-04
title: An ignored keyword argument is a silent no-op, and emit_pipeline(scratches=) is one
hardware: any
unit: any
software: "jax/jaxlib 0.10.2"
stage: [4]
lever: pipelining
generality: general
status: source
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-04
recheck: whether the primitive branch still swallows unknown kwargs after a jax upgrade — re-read both source ranges
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-04 — `emit_pipeline(..., scratches=)` is silently dropped on the primitive branch

**Rule.** Pass scratch buffers into an `emit_pipeline` body by **closure capture**, never via
`scratches=`. More generally: where an API has two lowering branches selected by a config flag, a
kwarg honoured on one and absorbed by a `**kwargs` on the other **fails without an error**. Zero
upstream call sites is the cheapest signal that a documented kwarg is not the supported path.

**Mechanism.** The kwarg is real on the legacy branch and forwarded through `**kwargs`;
`config.use_emit_pipeline_primitive` routes to a branch that drops unknown kwargs with no
diagnostic. Nothing fails — the scratch is simply never wired, so the kernel reads whatever was
already in that memory.

**Evidence.** [source: jax/_src/pallas/mosaic/pipeline.py:1768-1790, 2061-2078;
jax/_src/pallas/config.py:1906-1912 @ jax 0.10.2] Closure-capture form confirmed working in
`tpu_sparsecore_pallas_test.py:2995-3025`. No upstream call site passes `scratches=`.

**Scope.** All Pallas TPU pipelines at this jax pin. Related and worth checking at the same time:
the 2-D `Tiling.SPARSE_CORE` path is unproven at 16 lanes — the upstream test skips unless
`num_lanes == 8` (open upstream bug 478865387) — so prefer the 1-D `BlockSpec` form.

**Does not apply when** you have pinned the legacy branch explicitly. Confirm which branch the flag
selects rather than assuming the default.
