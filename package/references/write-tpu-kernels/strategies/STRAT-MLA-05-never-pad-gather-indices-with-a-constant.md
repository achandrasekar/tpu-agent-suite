```yaml
id: STRAT-MLA-05
title: Pad unused gather-index slots with distinct addresses; a repeated index is the degenerate bank conflict
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [3, 4]
lever: layout
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-DMA-01; raw/2026-09-dsamla-WHITEBOARD-excerpt.md §16
recheck: whether the target unit coalesces identical addresses, and its bank count (see STRAT-SC-06)
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-05 — Scatter the pad indices of an indirect gather

**Rule.** Unused slots in an indirect-gather index array must be filled with **distinct** addresses,
not a repeated one. The two idioms everyone reaches for — pad with zero, or pad with the last valid
index — concentrate every lane's read on a single address. On banked memory that is the worst case
available, not a neutral one. Padding with something that varies per slot costs nothing and avoids
it entirely.

**Mechanism.** This is [[STRAT-SC-06]]'s mechanism reached without a stride. Bank index is the
address modulo the bank count, so *identical addresses are identical banks by definition* — no
unlucky stride required, and no padding of a stride can fix it. SC-06 arrives here from a per-lane
write stride sharing a factor with the bank count; this entry is the read-side, stride-free
instance, and it is worth its own row only because the **trigger** is invisible from SC-06: nothing
about "choose a fill value for an index array" reads as a layout decision.

**Evidence.** Concentrated reads cost **40.1 ns/read** against
**0.97 ns/read** scattered — **41×**, `r² = 1.0000`. The ratio is consistent with v7x SparseCore's
**32 banks** ([[STRAT-SC-06]]) plus per-descriptor issue overhead, and that consistency is the
evidence that these are one mechanism rather than two findings. This ratio is measured *within* a
single instrument, so unlike [[STRAT-MLA-06]] it does not depend on the absolute byte accounting
being right — which makes it the more transferable of that campaign's two bandwidth results.

**Scope.** Any indirect gather or scatter into banked memory. The fix is free and local: fill pad
slot `i` with something that varies in `i` (`i % N` suffices) rather than a constant.

**Does not apply when** the unit coalesces or broadcasts identical addresses — some do, and then a
repeated index is genuinely the cheap case. v7x SparseCore does not, so on this part the default
assumption should be that a constant fill is a performance bug.
