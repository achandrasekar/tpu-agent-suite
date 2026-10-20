```yaml
id: STRAT-SC-08
title: Every transfer lever measures the same on a compute-bound kernel
hardware: tpu-v7x
unit: sparsecore
software: "jax/jaxlib 0.10.2"
stage: [2, 4]
lever: pipelining
generality: general
status: measured
source: raw/2026-09-sctopk-final_report.md §5, §6; raw/2026-09-sctopk-WHITEBOARD-excerpt.md §7j — not in that campaign's STRATEGY.md
recheck: the aggregate streaming bandwidth of the target unit, and which side of the roofline your kernel sits on
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-08 — Prove the load is the bottleneck before restructuring it

**Rule.** Every transfer lever — `pl.emit_pipeline`, a hand-rolled double buffer, a traced DMA
extent — changes **when** bytes arrive. On a compute-bound kernel they therefore change nothing and
measure the same as each other. Do not spend a job ranking them; establish that the roofline binds
on transfer first, and if it does not, pick whichever is shortest to read. **Measured null.**

**Mechanism.** These levers hide latency under compute. Where there is no exposed latency to hide,
their delta is the noise floor. The ceiling they must beat is an **aggregate** bandwidth figure, not
a per-worker one — dividing a chip's streaming bandwidth by its worker count invents headroom that
does not exist, and that invented headroom is usually what made the lever look worth trying.

**Evidence.** On a compute-bound v7x SparseCore kernel, full |
ragged: `emit_pipeline` **−6.21% / −4.97%**, hand-rolled double buffer **−6.17% / −5.17%**, traced
DMA extent **−6.15% / −4.74%**. `emit_pipeline` versus hand-rolled is **−0.04% / +0.19%** —
indistinguishable, closing that long-open SparseCore question as a null in its own right. Isolated
from its confound (see [[STRAT-SC-10]]) the DMA lever is a null at **≤0.5%**. The figure it was
tested against: SparseCore streams **338 GB/s per chiplet, aggregate over all 32 subcores**
— there is no per-subcore escape hatch. **Two caveats, neither of which
touches the gating rule, only the number it was tested against.** *Width:* 338 GB/s is a figure for
narrow reads, and a later campaign measured 509–709 GB/s on 640-byte indirect rows — see
[[STRAT-MLA-06]]. *Kind:* it is an **achieved** bandwidth, not a ceiling; the ceiling is 1700 GB/s per SparseCore
unidirectional, so **3400 per chiplet** (`hardware-facts.md` §2) — and since 338 is itself a
per-chiplet aggregate over both cores, 3400 is the denominator that matches it: **9.9%**, not the
19.9% you get by pricing an aggregate against one core. Do not reuse either as a roofline.

**Scope.** General as a gating rule. v7x SparseCore-specific: the 338 GB/s, and the finding that
`emit_pipeline` carries no measurable advantage over hand-rolled buffering on this unit at this pin.

**Does not apply when** the kernel is bandwidth-bound — on such a pipeline buffering depth is the
whole point, and these levers separate immediately.
