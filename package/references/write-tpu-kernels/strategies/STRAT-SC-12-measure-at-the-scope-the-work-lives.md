```yaml
id: STRAT-SC-12
title: A lever that acts between components reads as zero on either component alone
hardware: any
unit: any
software: any
stage: [1, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-final_report.md §2 — not in that campaign's STRATEGY.md
recheck: nothing — but re-derive the dilution factor from where the work lives, not from the call-size ratio
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-12 — Measure at the scope where the work actually lives

**Rule.** Measure a lever on the **whole call**, and derive its dilution from *where its work sits*,
not from the ratio of call sizes. A lever that deletes work **between** two components is invisible
when either component is timed alone. A lever that is **global** to the call does not dilute the way
a component-local one does. Scaling a component-scoped measurement by a size ratio gets both cases
wrong, and in opposite directions.

**Mechanism.** The size-ratio heuristic assumes a lever's savings are proportional to the fraction
of the call its component occupies. That holds only for work confined to that component. Inter-
component work belongs to neither, so it is excluded from both measurements and appears only in the
total; call-wide state is present in every component, so it saves everywhere and dilutes less than
the ratio predicts.

**Evidence.** A lever
deleting an inter-stage gather measured **+0.04% on stage 1** and **−8.28% on the whole call** — the
stage-scoped number had the wrong sign and would have rejected it. A kernel-wide stride change saved
**15.0 µs** whole against **12.0 µs** measured directly, an effective dilution of **1.25**, not the
**1.56** predicted from the call-size ratio. The campaign recorded predicting this backwards.

**Scope.** General to any multi-stage or multi-kernel op. The direction of the error is predictable
from where the work sits, so this is usable as a forecast, not only as a postmortem.

**Does not apply when** the lever's work is genuinely confined to one component and nothing crosses
the boundary — which is a claim about the dataflow that costs nothing to check and is wrong more
often than it looks.
