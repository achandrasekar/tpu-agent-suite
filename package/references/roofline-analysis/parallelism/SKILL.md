---
name: parallelism-collective-inventory
description: Per-parallelism collective inventory for roofline analysis — which collective a sharding scheme issues, how many per layer, over which group, at which dtype. One directory per scheme (ep, tp, dcp, pcp, pp, dp).
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/NOTATION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/parallelism/dcp/SKILL.md
- references/roofline-analysis/parallelism/dp/SKILL.md
- references/roofline-analysis/parallelism/ep/SKILL.md
- references/roofline-analysis/parallelism/pcp/SKILL.md
- references/roofline-analysis/parallelism/pp/SKILL.md
- references/roofline-analysis/parallelism/sp/SKILL.md
- references/roofline-analysis/parallelism/tp/SKILL.md
---

# Parallelism Collective Inventory

`model/common/<op>/` decomposes a model by **operation**. This tree decomposes it by
**parallelism**: for each scheme, which collectives it issues, how many per layer, over which
group, and at which dtype.

A scheme changes two things, and a file that covers only one of them is misleading:

* **HBM** — what it divides (weights, KV, activations), what it *multiplies*, and what it leaves
  replicated. This is usually why the scheme exists, and under long context it decides the
  binding admission gate.
* **Communication** — what it costs to buy that division.

`COLLECTIVE.md` answers *"what does an all-gather of B bytes over these axes cost"*. This tree
answers *"which collective, and how big"*. Keep the two separate — **a variant file never writes
its own closed-form ring math.** It produces `(collective, elems, dtype, group)` and
`cost_estimator/collective.py` prices it. A file that inlines its own formula has forked the
physics and thrown away the axis-aware D2D/ICI model.

## Status of this tree

| scheme | file | state |
| --- | --- | --- |
| TP | [`tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) | written |
| EP | [`ep/SKILL.md`]({{reference:roofline-analysis/parallelism/ep/SKILL.md}}) | written |
| SP | [`sp/SKILL.md`]({{reference:roofline-analysis/parallelism/sp/SKILL.md}}) | written — re-forms TP's collective, adds none |
| DP | [`dp/SKILL.md`]({{reference:roofline-analysis/parallelism/dp/SKILL.md}}) | written — no collective of its own |
| DCP | [`dcp/SKILL.md`]({{reference:roofline-analysis/parallelism/dcp/SKILL.md}}) | written — cited at vllm-torchtpu `fde6c3994` |
| PCP | [`pcp/SKILL.md`]({{reference:roofline-analysis/parallelism/pcp/SKILL.md}}) | written — cited at vllm-torchtpu `fde6c3994` |
| PP | [`pp/SKILL.md`]({{reference:roofline-analysis/parallelism/pp/SKILL.md}}) | written — cited at vllm-torchtpu `fde6c3994`; one collective-permute per forward |

All seven are written. **Only `dcp/`, `pcp/` and `pp/` are cited at a pinned sha** (vllm-torchtpu
`origin/main` `fde6c3994`, 2026-09-23); every other `source:` field is `unread` unless it names a
job id — and an `unread` variant must not reach a shippable recommendation. One variant has a
profile behind it: EP's collectives are confirmed against `j-484cd4c0`, which shows all-gather +
reduce-scatter and **no all-to-all in the trace at all**.

What the tree still owes, per scheme:

| scheme | owed |
| --- | --- |
| TP | which variant is live, and whether the reduction is overlapped — it is the most frequent collective in the model and every cost model here adds it serially |
| EP | a `file:line` for the AG+RS choice. Priced 2026-09-19 it beats a2a from `topk=2` up, and by 4.5x at GLM-5.2's `topk=8` |
| DCP | a measurement of the balanced-ownership assumption behind the sparse core's `K_idx / DCP`, and of overlap. The combine is read and priced — `dcp/` §6, §7 |
| PCP | a GDN term — Qwen3.5-class PCP is priced on its full-attention layers only; a step-level row for the once-per-step hidden gather; a measurement of how much of the KV ring the kernel hides |
| PP | time-budget chunk sizing (`TPU_PP_DYNAMIC_CHUNKS`) and stage placement are not modelled; the partition is now a knob (`Sharding.pp_layers`) with a search (`best_pp_partition`), not yet driven by `sharding-strategy` |
| DP | the straggler and dummy-forward terms are named and not priced — both need an arrival model this roofline does not have |
| SP | nothing beyond the sha |

Cross-cutting, and larger than any row above: **nothing anywhere models compute/collective
overlap.** A profile puts exposed collective wait at 21.9% of a decode step and 37.8% of prefill,
while every model here charges 100%. On a comm-bound ranking that is a bigger error bar than the
choice of variant.

## The boundary with `model/common/`

`model/common/<op>/` owns **operations** — shapes, FLOPs, memory transfer. It does not own
collectives. Where an op file still documents one, that is content awaiting a move here.

**Move a collective section out of `model/common/` only once its receiving
`parallelism/<scheme>/` file exists.** Deleting it earlier loses the content; leaving it after
creates two authorities that drift. EP and TP are moved: the op files now describe the
dataflow ("this projection's partial sums need a reduction") and point here for the payload, dtype
site, per-layer count and variant. DCP, PCP and PP have nothing to move yet.

## Variant schema

A scheme has more than one implementation, and they do not cost the same. Each variant carries:

**Memory** — one row per tensor the scheme touches:

| field | meaning |
| --- | --- |
| **tensor** | Weights, KV, or a named activation buffer |
| **per-device elems** | In `NOTATION.md` symbols, with the divisor or multiplier shown |
| **dtype site** | The numerics-sheet row that sets its width — never a hardcoded number of bytes |
| **effect** | `÷EP`, `xEP`, `replicated`, `untouched` |

State the multipliers as loudly as the divisors. A scheme is adopted for what it divides, and the
thing that bites is what it silently multiplies.

**Local data movement is a memory row too** — but only when the *scheme* causes it. A sharding
that forces a gather, a scatter or a re-layout pays HBM traffic that appears in no collective call
and is invisible to anyone reading only the communication half. A re-layout the operation would
perform at degree 1 anyway is the op's cost, not the scheme's, and belongs in `model/common/`.

**Communication:**

| field | meaning |
| --- | --- |
| **collectives** | The ordered sequence issued per layer, as `collective.py` calls |
| **payload** | `elems` per call, in `NOTATION.md` symbols |
| **group** | Which ranks participate, as a mesh-axis subset |
| **dtype site** | The numerics-sheet row that sets each call's dtype |
| **overlap** | `serial` \| `overlapped` \| `partial`, and the evidence |

**Both halves:**

| field | meaning |
| --- | --- |
| **source** | Where the numbers come from — see below |
| **interacts** | Which other schemes change this one's group, payload or divisor |

### The `source` field

Every variant carries where its numbers come from, stated plainly rather than as a label:

* a `file:line` in the checkout, with the sha it was read at;
* a job id, for something seen in a profile or a benchmark;
* or **`unread`** — nobody has checked.

**A variant whose source is `unread` must not reach a shippable recommendation.** It is priceable
so the space can be explored; it is not evidence. Say which of the three it is on every row, and
never leave the field off.

## Constraints and interactions

**This section is the only place constraints live.** A scheme file describes what that scheme
divides and what it issues; it does not carry a pairwise interaction table, because a pairwise fact
written in two files becomes two facts. If a scheme file needs an interaction, it links here.

### Degrees, booleans, and the world-size product

```
world = TP x PCP x PP x DP
```

Four factors. Everything else reuses ranks that product already allocated:

| scheme | knob | consumes ranks? |
| --- | --- | --- |
| TP | `--tensor-parallel-size` | yes |
| PCP | `--prefill-context-parallel-size` | yes |
| PP | `--pipeline-parallel-size` | yes |
| DP | `--data-parallel-size` | yes |
| DCP | `--decode-context-parallel-size` | **no** — subdivides TP |
| EP | `--enable-expert-parallel` | **no** — a boolean; spans `TP x PCP x DP` |
| SP | `use_sequence_parallel_moe` | **no** — a boolean; runs on the TP group, so `SP = TP` |

`tp16-dp1-dcp4` is still sixteen devices. There is no `--expert-parallel-size` and no
`--sequence-parallel-size`.

### Legality

| rule | condition |
| --- | --- |
| PCP does not support data parallelism | `pcp > 1 and dp > 1` is refused (`vllm/config/parallel.py:527`) |
| PCP and PP together | refused on TPU (`platforms/pcp_validation.py:114`) |
| PP needs TP = 1 | `pp > 1 and tp > 1`: TP inside a stage refused on TPU (`platforms/pp_validation.py:48-53`) |
| PP needs sync scheduling | `--no-async-scheduling`; no speculative decoding; KV connector only `TPURaidenConnector`, and then every stage needs a full-attention layer (`pp_validation.py:54-83`) |
| PP with DP | **no code refuses it; unmeasured.** `sharding-strategy` carries it as a hard `INFRA` refusal |
| `pp_layers` | when set: `pp` entries, each `>= 1`, summing to `L` (`vllm/distributed/utils.py:151-154`) |
| PCP on an MoE model needs EP | `pcp > 1`, MoE, no `--enable-expert-parallel`: refused on TPU (`pcp_validation.py:115-122`) |
| PCP on a decode / KV-consumer worker | refused on TPU (`pcp_validation.py:108-111`) |
| PCP across hosts | **allowed** since vllm-torchtpu `656a5f5ed` (#1083). Refused only together with multihost DP (`tpu_platform.py:889-894`) |
| MLA has no PCP path | MLA and `pcp > 1`: **not refused, not implemented** — the MLA forward never reads the PCP size. Priced for evaluation per `Model.pcp_kv_strategy` (`pcp/` §5) |
| DCP reuses the TP ranks | when `pcp == 1`, requires `tp % dcp == 0` (`vllm/config/parallel.py:529-532`) |
| GQA DCP needs replicated KV heads | not MLA and `dcp > 1` requires `tp > H_kv`, `dcp <= tp // H_kv`, and `(H_q / H_kv) % dcp == 0` (`vllm/config/model.py:1335-1362`) |
| PCP and DCP together | refused on TPU: `PCP runner path does not support DCP yet` (vllm-torchtpu `platforms/pcp_validation.py:112-113`) |
| Dense MLA has no DCP path | MLA without an indexer and `dcp > 1`: **not refused, not implemented** — the non-sparse branch never reads `dcp_size` (`layers/adapter/attention.py:2279-2295`). Treat as illegal |
| SP's MoE path requires no pipelining | `use_sequence_parallel_moe` applies at `PP = 1` |

### Group membership

| collective | runs on |
| --- | --- |
| TP reduction | the TP group |
| SP reduce-scatter / all-gather | the TP group — the same ranks, the same collective re-formed |
| EP dispatch / collect | `TP x PCP x DP` — **not** `DP` alone |
| PCP KV ring (in the attention kernel) | the PCP group |
| DCP query gather, LSE merge, indexer candidate exchange | the DCP group: `dcp` consecutive TP ranks inside one engine (`vllm/distributed/parallel_state.py:1851-1854`) |
| PP hand-off | a collective-permute over the whole PP group, every stage, every forward (`distributed/pp_shift.py`) |

### Pairwise effects

| pair | effect |
| --- | --- |
| **TP ↔ DP** | Trade on a fixed slice: `TP = world / (PP * DP)`. Raising DP lowers TP and multiplies everything TP divides |
| **TP ↔ EP** | EP spans the TP ranks. Routed expert weights are therefore indifferent to the TP/DP split |
| **TP ↔ SP** | SP runs on TP's group and re-forms its collective; `SP = TP` |
| **TP ↔ DCP** | DCP re-spends `dcp` of TP's ways on context positions: the attention core runs `H_q / (tp / dcp)` heads over `1 / dcp` of the keys. Projections, `o_proj` and the TP reduction are unchanged |
| **DP ↔ DCP** | Independent: the DCP group lives inside one engine. On a fixed slice they compete for ranks, `dp * dcp <= world` at `pcp = pp = 1`. Legal; **never run** |
| **TP ↔ PCP** | **Orthogonal, not folded.** PCP ranks replicate TP's dense, attention and embedding shards; PCP divides tokens, TP divides weights. Only the experts see both, through EP's `TP x PCP x DP` group |
| **TP ↔ `H_kv`** | Not a scheme, but it caps TP's KV benefit: sharding stops at `H_kv`, and under MLA at 1 |
| **DP ↔ EP** | EP spans DP, which puts the DP ranks in one collective group and forces them to step together |
| **EP ↔ SP** | Under SP's MoE path the attention exit's reduce-scatter *is* the MoE block's entry |
| **EP ↔ DCP** | None — DCP shards context, EP shards experts |
| **PCP ↔ DP** | Illegal together — see Legality |
| **PP ↔ everything** | Consumes from the same device budget. On TPU today it takes the whole slice — `TP = DP = PCP = 1` |
| **PP ↔ concurrency** | Decode needs `pp` disjoint request groups in flight; one long prompt pipelines by itself (`pp/` §6) |

### Do not double-count

Three collectives are claimed by two files each. The owner is listed first:

| collective | owned by | the other file that describes it |
| --- | --- | --- |
| MoE layer's trailing reduction, EP on | `ep/` | `tp/` — TP's FFN reduction does not fire on an MoE layer |
| Attention-exit reduce-scatter, SP on | `sp/` | `tp/` — it is TP's reduction re-formed, not an extra one |
| Token permute, dispatch-based EP | the dispatch | `model/common/moe/` — fused into the send buffers |

### Combination collectives — traffic no single scheme owns

Each scheme file describes its own block boundary **assuming the next block wants the layout it
produces**. Where that assumption fails, the mismatch is paid in a collective that appears in
neither file. These are per-*combination*, which is why they cannot live in a scheme file.

The rule: **if block N's output layout differs from block N+1's input layout, there is a reshard,
and someone is paying for it.** Most of the common pairs turn out to be already-counted — but
"already counted" and "free" are different answers, and the table says which.

| combination | extra traffic? | |
| --- | --- | --- |
| TP attention → EP MoE | **no, already counted** | The layout change *is* EP's all-gather / all-to-all. Counting a TP reduction as well double-charges — see the table above |
| SP → EP MoE | **no, already counted** | SP's reduce-scatter is the MoE block's entry collective, not a separate one |
| DP attention → EP experts | **no, already counted** | Getting each rank's tokens to the right experts is exactly what EP's collective does |
| PCP prefill → DCP decode | **no, and free** — but **refused on TPU** (see Legality) | PCP leaves the cache replicated; DCP needs a slice of it. Going from replicated to sharded is a **local discard**, not a collective. Only relevant where PCP and DCP may coexist, which on TPU today is across a disaggregated pair |
| **TP + vocab-parallel LM head** | **YES — and it is counted nowhere** | See below |
| **DCP + TP row-parallel output** | **YES — an extra collective, resolved** | `o_proj` does not absorb DCP's partials: they are LSE-merged and reduce-scattered back to TP-local heads before `W_UV`, and the TP reduction then runs unchanged. Priced in `dcp/` §6, §7 |
| TP + PCP | **no — resolved** | PCP only splits tokens. The TP reduction stays inside the TP group and carries `T / PCP` tokens; no weight is sharded jointly across `TP x PCP` except the experts, whose collective is EP's |
| PP + EP | **no** | Each stage owns its own EP group, so the group shrinks with `TP`. A smaller group, not an extra collective |

#### The sampling reduction — **measured at zero, resolved**

Checked against the `j-484cd4c0` profile (GLM-5.2-FP8, 2x2x2, TP16, decode and prefill op tables
plus the collective table). Result: **it is not on the TPU critical path and does not need
modelling.**

* The collective table contains **exactly three** collectives per step, none vocab-sized: the EP
  token all-gather, the EP reduce-scatter, and the router-logit all-gather. There is no
  `V`-dimension collective at all.
* The only vocab-touching op on device is a `gather_fusion` over `bf16[154880,6144]` with `s32`
  indices — the **input embedding lookup**, not the output head — at **0.015 ms, 0.0066% of the
  step**.

So the output projection and sampling either run on host or are too small to appear. Either way
the earlier concern that this could be 24% of the step's collective budget is **disproved for this
model and cell**. Left unmodelled deliberately, with the evidence rather than by omission.

### Still unestablished

**Do not compose two schemes whose interaction is not listed** — read the code and add the row.
Nothing here is cited at a pinned sha.

Known gap in the table above: whether the MoE dispatch bound sees the microbatch or the whole running set under PP.

## Required reading

* [`../COLLECTIVE.md`]({{reference:roofline-analysis/COLLECTIVE.md}}) — the estimator API, and the reason all-to-all is not a
  reduce-scatter variant.
* [`../NOTATION.md`]({{reference:roofline-analysis/NOTATION.md}}) — `EP`, `TP`, `DP`, `E`, `K`, `D`, `B_local`, `S`. Note `G` is
  already the GQA group size; the EP group size is written `EP`.
* [`../model/common/REQUIREMENTs.md`]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}) — Python for all arithmetic,
  ms / GFLOPs / MB units.
