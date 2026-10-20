```yaml
id: STRAT-MLA-02
title: A knob that sets grid-truncation granularity has an interior optimum, not a direction
hardware: tpu-v7x
unit: any
software: any
stage: [3, 4]
lever: pipelining
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-GRID-01; raw/2026-09-dsamla-WHITEBOARD-excerpt.md §22
recheck: the chunk extent your grid walks, and the per-grid-step cost on the target unit
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-02 — Read what a "batch size" actually does before you sweep it

**Rule.** A knob named like a tile size may instead set the **granularity at which a data-dependent
grid stops early**. Such a knob does not trade off against occupancy or VMEM; it trades *padding
truncated* against *grid steps walked*. Two opposing terms means an interior optimum — find it by
sweeping, and never assume the knob has a direction.

**Mechanism.** For a chunk of extent `C` walked in steps of `B`, a grid that stops at the last step
holding live work runs `ceil(live/B)` steps of width `B`, out of `C/B` possible. Cost is
`steps × per_step + truncation_waste × per_row`. Shrinking `B` cuts the waste linearly but raises
the step count as `C/B`. The first term dominates while padding is most of the chunk (decode); the
second dominates once the chunk is full (prefill). The optimum sits strictly inside the range, and
the two ends are each optimal for a different serving regime — so a single value is a **compromise
that must be priced at both ends**, not a tuning to be maximized at one.

**Evidence.** At `C = 64`, going `B: 16 → 4` improved decode
*and* a full prefill chunk — the unambiguous move. Going `4 → 1` improved decode a further ~20% but
**regressed prefill ~11%**, because a full chunk then costs 64 grid steps instead of 4. `B = 4` is
the turning point and was shipped. The earlier `B = 1` reading that looked like a win at both ends
came from a shortened axis and was retracted — see [[STRAT-MLA-03]].

**The two-term model is incomplete, and this entry's own evidence says so.** Both readings above
have `B: 16 → 4` winning on a chunk with *no truncation waste to recover* — the full prefill chunk,
and later a fully-live `C = 16` decode bucket at **+11% and +7%** on two runs, where the composed candidate's
other lever is structurally inert. With the waste term at zero the model predicts a loss (4 grid
steps against 1) and a gain was measured instead. So a **third term favours smaller `B`
independently of truncation** — plausibly overlap across narrower steps, not isolated here. Note the
same fully-live cell at `C = 64` reads 1.02× / 1.00×, i.e. nothing: the third term is visible only
where the step count is small enough for it to matter. The interior optimum stands; its left-hand
term is not only truncation waste.

**In-program, this knob is the whole decode win, and the ratio is the knob's ratio.** An xprof
diff of the two shipped arms (rank0, GLM-5.2 decode) reads the attention kernel at **36.08 → 8.56 µs/layer median, 4.21×**, against
`B: 16 → 4`, i.e. **4.00×** — the grid was walking 16 rows to use 1, and the truncation term alone
accounts for it. Core-blocking attention per step 2.766 → 0.668 ms (4.14%→ 1.62% of the step). Two
things to carry: the effect is *larger* in the program than in the chained microbenchmark that found
it (1.88× there, because that harness serializes a gather the real program hides — see
[[STRAT-MLA-01]]), and this knob, not the padding skip it shipped beside, is what moved decode. When
two levers ship together, an in-program trace is the only thing that says which one paid.

**Scope.** Any grid whose trip count is truncated against a runtime extent — ragged attention,
bucketed serving, variable-length reductions. The mechanism is arithmetic, not silicon; what is
hardware-specific is only the per-step cost that sets where the turning point lands.

**Does not apply when** the grid always runs to completion — then the knob is a tile size and the
usual occupancy and working-set reasoning governs. But do **not** infer from a zero truncation term
that the knob is therefore neutral: measured twice, it was not. Sweep it there too.
