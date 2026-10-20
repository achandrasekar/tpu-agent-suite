# Wiki index

One row per entry. **Read this table, then open only the bodies whose row is about the block in
front of you.** The protocol — labels, what may be written here, how to promote — is `PROTOCOL.md`.

`hardware` is where the entry was measured. `carry` is how far you may take it without re-measuring:
**narrow** = this unit, this shape class; **unit-wide** = any kernel on that unit, same generation;
**general** = the mechanism is not specific to the silicon it was found on.

## Read whole, at step 1

| file | hardware | covers |
| --- | --- | --- |
| `hardware-facts.md` | tpu-v7x, with v4/v5e/v5p/v6e in the denominator table | chip constants, compiler hard limits, the chiplet memory system and its bandwidth ceilings, sublane tiling, relayout costs, what fits, what a runtime value may be used for, instruments that lie, SparseCore banks and waves |

## Strategies

| id | hardware / unit | stage | lever | carry | claim |
| --- | --- | --- | --- | --- | --- |
| [STRAT-SC-01](strategies/STRAT-SC-01-compaction-pass-cost.md) | v7x / sparsecore | 2, 3 | algorithm | general | A pass with cross-lane serializing ops in its body is priced by those ops per chunk, not by its trip count. |
| [STRAT-SC-02](strategies/STRAT-SC-02-absolute-savings-compose.md) | v7x / any | 1, 5 | method | general | Predict a stacked optimization from absolute time saved; percentages against a moving baseline compose wrong. |
| [STRAT-SC-03](strategies/STRAT-SC-03-barrier-the-loop-carry.md) | v7x / any | 4, 5 | method | general | A chained-invocation clock must barrier the loop carry; LICM hoists an invariant-barriered kernel out of the loop and no correctness guard catches it. |
| [STRAT-SC-04](strategies/STRAT-SC-04-emit-pipeline-scratches-noop.md) | any / any · **jax 0.10.2** | 4 | pipelining | general | `emit_pipeline(..., scratches=)` is silently dropped on the primitive branch — pass scratch by closure capture. |
| [STRAT-SC-05](strategies/STRAT-SC-05-wave-longest-row.md) | v7x / sparsecore | 1, 2 | algorithm | general | A lockstep group costs its longest member — model ragged work as `E[max over W]`, not the mean. `W = 32` on v7x SC. |
| [STRAT-SC-06](strategies/STRAT-SC-06-pad-scatter-stride-off-bank-count.md) | v7x / sparsecore | 3, 4 | layout | general | A per-lane stride sharing a factor with the bank count collapses every lane onto one bank; pad it coprime. 32 banks on v7x SC. |
| [STRAT-SC-07](strategies/STRAT-SC-07-distribution-sensitivity-is-a-conflict.md) | v7x / any | 5 | method | general | Latency that moves with the input distribution at constant op count is a bank conflict; the fix's signature is the spread collapsing. |
| [STRAT-SC-08](strategies/STRAT-SC-08-prove-the-load-binds-first.md) | v7x / sparsecore · **jax 0.10.2** | 2, 4 | pipelining | general | **Measured null.** Prove the roofline binds on transfer first — on a compute-bound kernel every transfer lever measures the same. |
| [STRAT-SC-09](strategies/STRAT-SC-09-unit-choice-follows-what-binds.md) | v7x / sparsecore | 1, 2 | algorithm | unit-wide | Unit choice follows what binds the algorithm, not pipe width; large-`k` top-k is `k`-bound and SparseCore wins 8.70–22.97×. |
| [STRAT-SC-10](strategies/STRAT-SC-10-shared-arm-confound.md) | v7x / any | 5 | method | general | Arms that all move by the same amount are probably measuring what they share, not the lever under test. |
| [STRAT-SC-11](strategies/STRAT-SC-11-oracle-needs-a-positive-control.md) | any / any | 4, 5 | method | general | A green oracle proves nothing until an injected fault has made it red — report which cell classes can discriminate. |
| [STRAT-SC-12](strategies/STRAT-SC-12-measure-at-the-scope-the-work-lives.md) | any / any | 1, 5 | method | general | A lever acting *between* components reads as zero on either alone; a call-wide lever dilutes less than the size ratio predicts. |
| [STRAT-SC-13](strategies/STRAT-SC-13-gate-the-estimator-not-the-samples.md) | any / any | 5 | method | general | A stability gate must bound the reported statistic's uncertainty; gating sample spread fires *more* as evidence accumulates. |
| [STRAT-MLA-01](strategies/STRAT-MLA-01-bucket-padding-is-not-free-upstream.md) | v7x / sparsecore | 2, 3 | algorithm | general | Bucket padding is free only *downstream* of the mask; a gather upstream of it pays full price for rows already known to be zero. |
| [STRAT-MLA-02](strategies/STRAT-MLA-02-truncation-granularity-has-an-interior-optimum.md) | v7x / any | 3, 4 | pipelining | general | A knob that sets grid-*truncation* granularity trades padding against grid steps, so it has an interior optimum, not a direction. |
| [STRAT-MLA-03](strategies/STRAT-MLA-03-shortened-axis-biases-the-sweep.md) | any / any | 1, 5 | method | general | Sweeping a knob on a shortened axis is biased when that axis carries the knob's cost; the trend does not extrapolate, not even in sign. |
| [STRAT-MLA-04](strategies/STRAT-MLA-04-carry-must-reach-each-sub-kernel.md) | v7x / any | 4, 5 | method | general | Extends SC-03: in a multi-kernel op the carry must reach *each kernel's own* inputs — for a gather, the indices, not the values. |
| [STRAT-MLA-05](strategies/STRAT-MLA-05-never-pad-gather-indices-with-a-constant.md) | v7x / sparsecore | 3, 4 | layout | general | Corollary of SC-06: a repeated pad index concentrates every lane on one address — the degenerate bank conflict. 41×. |
| [STRAT-MLA-06](strategies/STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md) | v7x / sparsecore | 1, 2 | layout | unit-wide | SparseCore streaming bandwidth is access-width dependent; 338 GB/s is not a ceiling for wide-row indirect reads. Qualifies SC-08. |
| [STRAT-MLA-07](strategies/STRAT-MLA-07-widening-roofline-ratio-means-fixed-cost-bound.md) | v7x / any | 1, 5 | method | general | A roofline ratio that *widens* as you remove work proves the floor models the wrong resource — attack fixed cost next. |
| [STRAT-MLA-08](strategies/STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md) | v7x / sparsecore · **jax 0.10.2** | 4 | pipelining | unit-wide | `pl.when` around `emit_pipeline` *and* a traced grid length both lower on SparseCore — data-dependent skipping is available. |
| [STRAT-MLA-09](strategies/STRAT-MLA-09-sparsecore-rank0-smem-xor.md) | v7x / sparsecore · **jax 0.10.2** | 4 | layout | unit-wide | SparseCore rank-0 loads are legal only from SMEM and SMEM loads only at rank 0; a scalar predicate must arrive as a vector. |
| [STRAT-MLA-10](strategies/STRAT-MLA-10-a-global-constant-must-pay-on-shapes-you-did-not-measure.md) | any / any | 3, 5 | method | general | Evidence is owed by blast radius: a module-level constant must pay on every shape it reaches. Tune through an argument. |
| [STRAT-MLA-11](strategies/STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md) | v7x / sparsecore | 1, 5 | method | general | "Fixed cost binds" is a diagnosis, not a target. Sweep the grid at constant live work: the slope is your lever's whole ceiling. Corrects MLA-07's localization. |
| [STRAT-MLA-12](strategies/STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md) | v7x / sparsecore · **jax 0.10.2** | 4 | pipelining | unit-wide | A traced-offset, traced-extent SC DMA lowers into a 1-D ring slot with `multiple_of` hints; a 2-D ring row slice is rejected. |
| [STRAT-MLA-13](strategies/STRAT-MLA-13-a-compute-bound-is-not-a-dma-bound.md) | v7x / any | 2, 3 | algorithm | general | Bounding loops by the live length does not bound a static-extent DMA; index-assigned work keeps live units at full cost — rebalance with a dynamic width. |
| [STRAT-MLA-14](strategies/STRAT-MLA-14-xla-scatter-into-subword-array-can-relayout-the-whole-operand.md) | v7x / tensorcore | 2, 3 | layout | general | An XLA scatter into a sub-32-bit tiled array can relayout the whole operand at large update counts; cost scales with the array. |
| [STRAT-MLA-15](strategies/STRAT-MLA-15-spill-removal-cannot-move-a-valu-bound-kernel.md) | v7x / tensorcore | 3, 5 | method | general | **Measured null.** Removing spills cannot move a VALU-bound kernel — spills co-issue in free vst/vld slots. |

## By stage

Cheaper than a grep when you just want the list for the step you are on.

| step | entries |
| --- | --- |
| 1 — shapes, dtypes, hardware limits | `hardware-facts.md` whole · SC-02 · SC-05 · SC-09 · SC-12 · MLA-03 · MLA-06 · MLA-07 · MLA-11 |
| 2 — dataflow, buffers, reference mining | SC-01 · SC-05 · SC-08 · SC-09 · MLA-01 · MLA-06 · MLA-13 · MLA-14 |
| 3 — algebra and the freeze gate | SC-01 · SC-06 · MLA-01 · MLA-02 · MLA-05 · MLA-10 · MLA-13 · MLA-14 · MLA-15 |
| 4 — mechanical lowering | SC-03 · SC-04 · SC-06 · SC-08 · SC-11 · MLA-02 · MLA-04 · MLA-05 · MLA-08 · MLA-09 · MLA-12 |
| 5 — probes and profiling reads | `hardware-facts.md` §9 · SC-02 · SC-03 · SC-07 · SC-10 · SC-11 · SC-12 · SC-13 · MLA-03 · MLA-04 · MLA-07 · MLA-10 · MLA-11 · MLA-15 |

## Superseded and retracted

None yet. Entries that stop being true move here rather than out of the tree — see `PROTOCOL.md`,
"Superseding".

---

**Provenance — `2026-09-dsanew` (MLA-12 … MLA-15; sharpens MLA-01, SC-03).** A fresh-context campaign on the GLM-5.2-FP8 DSA
indexer + sparse MLA at 256k/1k c16 (vllm-torchtpu `dsa_new` from main `22286cfcb`), TPU v7x `2x2x2`, jax/jaxlib 0.10.2,
libtpu 0.0.44.1. Delivered +16.7% e2e output throughput. Sources: `raw/2026-09-dsanew-*`.

**Provenance — `2026-09-sctraversal` (MLA-11).** One entry from a one-job campaign against the same
block, opened to attack the residual `2026-09-dsamla` carried and **closed by its first measurement**.
It keeps the `MLA` prefix because it is the same lineage; ids are provenance, not scope. Its lasting
effect is a **correction to [STRAT-MLA-07](strategies/STRAT-MLA-07-widening-roofline-ratio-means-fixed-cost-bound.md)**,
whose Rule it confirms and whose attribution of the residual to *window traversal* it refutes — the
cost is per-call, and the two name different levers. Sources: `raw/2026-09-sctraversal-*`.

**Provenance — `2026-09-dsamla` (MLA-01 … MLA-10).** Promoted from the DSA/MLA decode campaign
against `vllm-torchtpu/src/vllm_torchtpu/kernels/mla/sparse/` @ `61b1c0b8`, GLM-5.2-FP8 at 128k/1k
on TPU v7x `2x2x2`, jax/jaxlib 0.10.2, libtpu 0.0.44.1. Sources: `raw/2026-09-dsamla-*`.

Two of that campaign's findings are **not** here as entries, because they were already covered and
the protocol says to sharpen rather than duplicate: its TensorCore-gather null became a second
instance inside [STRAT-SC-09](strategies/STRAT-SC-09-unit-choice-follows-what-binds.md), and its
bandwidth refinement is cross-referenced from
[STRAT-SC-08](strategies/STRAT-SC-08-prove-the-load-binds-first.md), whose 338 GB/s ceiling it
qualifies. Two more entries are explicit extensions that cite their parent rather than restating it
— MLA-04 extends SC-03, MLA-05 is a corollary of SC-06.

**Provenance — `2026-09-sctopk` (SC-01 … SC-13).** All thirteen entries were promoted from campaign `2026-09-sctopk` against
`vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` @ `486af819`, on TPU v7x
(Ironwood) at jax/jaxlib 0.10.2, libtpu 0.0.44.1. Its verbatim sources are in `raw/` and every
entry's `source:` field points at one — read those before superseding anything, since the entries
are **rewrites, not copies** (`PROTOCOL.md`, "Promotion is a rewrite, not a copy").

The `SC` prefix records that campaign and is **not** a scope claim — SC-02, SC-03, SC-04, SC-07 and
SC-10 through SC-13 are not about SparseCore. Twelve of the thirteen carry `generality: general`,
meaning the Rule line is stated as an invariant and the v7x numbers under it are the demonstration;
each one's `recheck:` field names what to re-measure before relying on it elsewhere.

Six entries have no line in the campaign's own `STRATEGY.md` — SC-08 through SC-13 were promoted
from `final_report.md`, which is why `raw/README.md` requires copying it.
