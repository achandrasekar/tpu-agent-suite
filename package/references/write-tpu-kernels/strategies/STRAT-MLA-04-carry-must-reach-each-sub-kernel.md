```yaml
id: STRAT-MLA-04
title: In a multi-kernel op the loop carry must reach each kernel's own inputs, not just the op's
hardware: tpu-v7x
unit: any
software: any
stage: [4, 5]
lever: method
generality: general
status: measured
source: raw/2026-09-dsamla-STRATEGY.md § STRAT-MEASURE-01; raw/2026-09-dsamla-WHITEBOARD-excerpt.md §15
recheck: the static custom-call count of the op you are timing, so the per-iteration assertion has a target
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-04 — A per-op carry is not a per-kernel carry

**Rule.** [[STRAT-SC-03]] requires a chained-invocation clock to route its dependency through the
loop carry. That is necessary but **not sufficient** when the op lowers to more than one kernel.
Each kernel is eliminated on its own operand set, so the carry must reach the inputs that *the stage
you are timing* depends on. For a gather, perturb the **indices**; perturbing the values it is later
combined with leaves the gather invariant.

**Mechanism.** Common-subexpression elimination works per call, on operands. A multi-kernel op has
kernels with disjoint operand sets: a gather depends on `(cache, indices, page_table)`, while the
consumer depends on the gather's result and on the query values. A carry folded into the query
values makes the *consumer* vary across iterations — so the consumer chains correctly and the loop
looks healthy — while the gather subgraph stays bit-identical and `K` gathers collapse to one. The
clock then prices the consumer plus a single gather. Because CSE is correctness-preserving, **the
answer is right and the question was wrong**, so no oracle, no bit-exactness check and no NaN guard
can fire. The tell is a physically impossible *relation* rather than an impossible absolute: here,
"the attention is cheaper than the gather it contains".

**Evidence.** The value-perturbed chain reported **31–35 µs**
and the impossible relation above. Rotating the indices instead — `tk = (topk + off + carry) % SLOTS`,
which changes *which* rows are fetched without changing how many — read **64.16 µs**. What caught it was exactly the defence [[STRAT-SC-03]]
prescribes, applied per kernel: `tpu_custom_call` count **per chained iteration** must equal the op's
static kernel count — **3.00** here, on 12/12 configurations; the collapsed version read below 2.5.
Two instruments were voided.

**Scope.** Any op lowering to more than one kernel, on any compiler that CSEs calls — which is all
of them. The self-check generalizes without modification: count calls per iteration, compare against
the static kernel count, and require equality rather than "greater than one".

**Does not apply when** the op is a single kernel. Then [[STRAT-SC-03]] alone covers it, since there
is only one operand set and the carry either reaches it or does not.
