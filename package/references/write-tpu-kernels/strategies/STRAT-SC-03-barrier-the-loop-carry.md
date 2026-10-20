```yaml
id: STRAT-SC-03
title: A chained-invocation clock must barrier the loop carry, or LICM hoists the kernel out of its own loop
hardware: tpu-v7x
unit: any
software: any
stage: [4, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-sctopk-STRATEGY.md § STRAT-SC-03; raw/2026-09-sctopk-final_report.md §8; raw/2026-09-sctopk-WHITEBOARD-excerpt.md §7h, §7h-old
recheck: nothing on the mechanism — re-derive the healthy `T_hi/T_1` band for your own launch overhead
origin: campaign sparsecore-topk, 2026-09
```

# STRAT-SC-03 — Barrier the loop carry, never the captured invariants

**Rule.** A microbenchmark that amortizes launch overhead by chaining M invocations must route its
data dependency **through the loop carry**. Barrier the invariants instead and the compiler is free
to prove the inputs unchanging, hoist the call out of the loop, and run M nominal invocations
**once**. Then assert the call count in the HLO equals the expected multiple of M, and derive
per-call cost from an **M-sweep slope** rather than dividing by M at a single M.

**Mechanism.** Loop-invariant code motion is a correctness-preserving transform, so **no correctness
guard can catch this**: a hoisted loop is idempotent and still returns the right answer. The
symptom is a physically impossible number — a per-call time below a hardware floor — and the tell in
the HLO is the async-start taking top-level operands instead of a `while`-carry
`%get-tuple-element`.

**Evidence.** A probe barriering `(scores, lengths, dep)` —
all loop-invariant — reported **26.40 µs/call** against an xprof device-time arbiter at
**68.26 µs**, a 2.8× phantom speedup and a sub-HBM-floor reading. Its M=20 chain contained **one**
kernel event; the correctly-carried probe's contained **20 at 70.06 µs each**. Five reported numbers
were retracted. The defence used by every later probe: HLO `tpu_custom_call` count == 2M,
**496/496** units, plus `T(M) = a + sM` with `a ≈ 250 µs`, where a healthy `T_hi/T_1` is 13–50 and a
hoist reads **~1.0–1.6**.

**Scope.** Any chained-invocation clock on any accelerator whose compiler does LICM. The 250 µs and
the 13–50 band are this host and this launch path.

**Does not apply when** the harness times single unchained invocations — but then launch overhead is
in every number, and subtracting a fitted constant reintroduces a different way to be wrong.

**Refinement (2026-09-dsanew, source: raw/2026-09-dsanew-STRATEGY.md § Job #1b, § instruments v3, § critical-review B4/B5).**
Gate the "scales with M" check on the **kernels** (`tpu_custom_call`) only: loop-invariant non-kernel ops (an XLA
`AssumeGatherIndicesInBound` over the page table) hoist legally and made every attention arm read FAIL. But a kernel-only gate has its own blind spot: scalar prep outside the kernel (per-chunk live-count
math) can hoist out of the chain, inflating the M=1 region and the intercept (+300 µs) while the slope looks clean —
cross-check the slope against the single-call and unchained clocks and report the range.
