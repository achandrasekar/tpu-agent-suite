```yaml
id: STRAT-SC-11
title: A correctness oracle that has never failed is not known to be able to fail
hardware: any
unit: any
software: any
stage: [4, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-final_report.md §3, §7.4 — not in that campaign's STRATEGY.md
recheck: which of your cell classes can discriminate the specific fault you are guarding against
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-11 — Ship the oracle's positive control, not just its PASS

**Rule.** A green oracle is evidence only if you have shown that oracle can go red. **Inject the
fault you are guarding against and report which cells catch it**, per cell class. A gate that is
blind on the cell class matching production is not a gate, however many cells it passes — and the
blindness is invisible from the PASS.

**Mechanism.** Two things make an oracle blind and neither shows up in its output. A transform that
is an **involution** turns a double-application fault into the *original bits* rather than garbage.
And a distribution can place every element in the region where the wrong computation and the right
one happen to agree — for an order-preserving bit trick, a strictly-positive input is exactly that
region. Production distributions are the ones most likely to sit in it, because the kernel was
tuned for them.

**Evidence.** A `monotone_key` involution meant a
double-conversion fault yielded raw bits, and on full rows of a zero-mean distribution the top-k is
entirely positive, where raw int32 order agrees with float order. The fault-injection control:
full rows **0/4 cells caught — BLIND**; `U(0,n_p)` **0/4 — BLIND**; `U(0,4096)` **4/4**. A live
double-conversion arm confirmed it: discriminating cells **24/24 caught**, non-discriminating
**16/20 missed**. 616 real-arm units passed with `rows_differing` 0 — a number that would have been
equally reassuring on a broken kernel. Cell-design correction from the same campaign: `U(k,3k)`
straddles zero on only **8/16** rows on device; **`U(k,1.5k)` on 16/16**.

**Second instance, a different blindness — the oracle compared the kernel against itself.** A
store-exactness probe stayed green on a mutant producing residuals at **46–65× tolerance**, and
stayed green *non-vacuously*: its `body_max` moved with the mutation while `dirty_rows` held at
**0/22** (GDN conv1d workspace). The cause was an import — it took `entry` from the variant's own package, so
every mutation moved both sides together. This is the same rule from the other direction: the first
instance is an oracle that *cannot* discriminate, this one is an oracle with **no independent
reference to discriminate against**. Both are invisible from the PASS, and both are caught by the
same control. Add one check to the ritual: **before quoting a probe, read what it imports its
reference from.**

**Scope.** General. The mandatory set it argues for: short rows, an in-range sentinel rather than an
out-of-range one, a cell class that straddles the discriminating boundary, and one injected fault
per invariant the kernel claims.

**Does not apply when** the oracle is an exact bit-comparison against an independent implementation
on the same inputs — but confirm the reference is genuinely independent, since a shared helper
reintroduces exactly this failure.
