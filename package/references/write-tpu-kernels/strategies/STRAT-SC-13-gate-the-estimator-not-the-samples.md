```yaml
id: STRAT-SC-13
title: A stability gate must bound the estimator you report, not the spread of the samples
hardware: any
unit: any
software: any
stage: [5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-final_report.md §7.3 — not in that campaign's STRATEGY.md
recheck: which statistic your harness actually reports, and what its quarantine rule actually bounds
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-13 — Gate the uncertainty of the estimator, not the dispersion of the samples

**Rule.** A benchmark's quarantine or stability criterion must bound the uncertainty of **the
statistic you report**. Gate the dispersion of the individual rounds instead and the rule is
backwards: **collecting more rounds makes it fire more often**, while the estimate it is guarding
gets better. Read the harness's criterion before trusting the cells it did not quarantine.

**Mechanism.** Sample dispersion is a property of the distribution and does not shrink with `n`; the
standard error of a median does. A conjunct written on a p10–p90 spread therefore tightens nothing
as evidence accumulates — it just samples the tail more times. The failure is silent in the useful
direction too: the gate looks strict while bounding a quantity nobody reports.

**Evidence.** A harness gating the p10–p90 dispersion of its rounds while reporting their **median**
tripped on **61/124 cells (49.2%)** on that conjunct alone, and on **none** via its ratio conjunct.
Nothing was actually dropped; two full replicates were substituted as the error bar instead and
agreed to **0.16% median, 0.08 pp on deltas** over 212 matched units — an error bar two orders tighter than the spread the
gate was objecting to. The cell tally is a campaign-wide count from the final report, not a single
job.

**Scope.** General to any measurement harness. The replacement is a bootstrap CI on the reported
statistic, or agreement between independent replicates; a ratio-based conjunct on the same rounds is
sound and worth keeping.

**Does not apply when** the reported statistic *is* a dispersion — a tail-latency target, for
instance. Then gating spread is gating the estimator, and this rule is satisfied already.
