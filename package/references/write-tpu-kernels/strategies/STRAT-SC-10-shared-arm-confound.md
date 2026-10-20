```yaml
id: STRAT-SC-10
title: Arms that all move by the same amount are measuring what they share
hardware: tpu-v7x
unit: any
software: any
stage: [5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-final_report.md §5; raw/2026-09-sctopk-WHITEBOARD-excerpt.md §7j
recheck: nothing — this is an experiment-design rule
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-10 — Arms that agree are probably measuring what they share

**Rule.** When every arm of a probe moves by about the same amount, the default reading is **not**
"the lever works." It is that the arms share something the control does not, and that shared thing
is what moved. Before crediting a lever, name what its arms have in common and put an arm in that
isolates it.

**Mechanism.** Implementing a lever usually requires a restructuring that is not the lever — a fused
pass, a changed guard, a different buffer. If that restructuring is present in every treatment arm
and absent from the control, it is perfectly confounded with the lever, and the probe cannot tell
them apart no matter how many arms it has or how tight the error bars are. Agreement between arms
is the signature, and it is easy to misread as robustness.

**Evidence.** Three structurally different DMA treatments came
in at **−6.21%, −6.17%, −6.15%** — agreement that read as a robust win. All three had been built on
top of the same `cvt` fusion, which the control lacked. Isolated, the fusion was the entire **−6.2%**
and the DMA lever was a null at **≤0.5%**. The convergence was the tell: three unrelated mechanisms
do not land within 0.06 pp of each other.

**Scope.** General experiment design, any hardware. Cheap to apply — it costs one extra arm holding
the shared change with the lever removed.

**Does not apply when** the arms are genuinely minimal edits off a common base and you can point at
the diff proving it. That proof is the work; arm agreement is not a substitute for it.
