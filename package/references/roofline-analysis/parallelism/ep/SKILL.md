---
name: expert-parallelism-roofline
description: Expert Parallelism (EP) for roofline analysis — what it divides in HBM, what it multiplies, and which collectives each implementation variant issues, with payloads, dtype sites and per-layer counts.
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/parallelism/SKILL.md
---

# Expert Parallelism (EP)

EP shards the **routed experts** across devices: each device holds `E / EP` of them and no device
holds the whole MoE block. It exists to divide expert weights, which on a large MoE checkpoint are
most of the parameters.

See [`../SKILL.md`]({{reference:roofline-analysis/parallelism/SKILL.md}}) for the variant schema and the `source` field, and
[`../../COLLECTIVE.md`]({{reference:roofline-analysis/COLLECTIVE.md}}) for the estimator API.

## 1. Notation

| symbol | meaning |
| --- | --- |
| `EP` | Expert-parallel degree — the size of the group the collectives run on |
| `E` | Routed experts |
| `K` | Experts selected per token (`top_k`) |
| `D` | Hidden dimension |
| `F_expert` | Per-expert intermediate dimension |
| `n_mat` | Weight matrices per expert — `3` for a gated expert (gate, up, down), `2` otherwise |
| `T_dev` | Tokens **this device** contributes to the step: prefill chunk tokens plus one per decoding request |
| `T_ep` | Tokens visible to the whole EP group = `EP * T_dev` |
| `L_moe` | Number of **MoE** layers — see §2 |

## 2. Per layer, and which layers

Everything in §4 and §5 is **per MoE layer**, and fires on **every** MoE layer of the forward
pass. A step's EP cost is the per-layer cost times `L_moe`.

**`L_moe` is not `L`.** A checkpoint that interleaves dense FFN layers with MoE layers runs no EP
collective on the dense ones, and their expert weights do not exist to divide. Take `L_moe` from
the model's own config — the count of layers whose FFN is a routed MoE — and never assume it
equals the layer count. Getting this wrong scales the entire EP term.

## 3. What EP does to HBM

Per device, per MoE layer. Dtype-site names set each row's width; never hardcode a number of
bytes.

| tensor | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| Routed expert weights | `(E / EP) * n_mat * D * F_expert` | `moe` | **÷EP** — the reason EP exists |
| Router / gate weights | `D * E` | `moe` | replicated — small, not zero |
| Shared expert weights | `n_mat * D * F_shared / TP` | `moe` | untouched — the shared expert is TP-sharded even when EP is on |
| KV cache | — | `kv_cache` | **untouched** |
| Expert intermediate buffer | `T_dev * K * F_expert` | `moe` (activation leg) | untouched, and identical in every variant |
| MoE hidden-state buffers | variant-dependent — §4, §5 | `expert_dispatch` / `expert_collect` | **the multiplier to watch** |

Three rows are worth stating outright, because each is easy to get wrong in the optimistic
direction:

**The intermediate buffer does not scale with `EP`.** Across the group there are `T_ep * K`
token-expert pairs; a device holding `E / EP` experts receives `T_ep * K / EP = T_dev * K` of them
under uniform routing. So the expert compute and its intermediate activation are **identical in
every variant** — the variants differ in how the tokens arrive, not in how many arrive. Anything
that divides Option A's token count by `EP` has counted the dispatch twice.

**EP does not move the KV gate.** On a long-context cell the binding admission gate is usually KV
capacity, and EP does nothing for it. It frees HBM that a larger KV pool could then use, which is
a second-order effect and only materialises if the pool is actually resized.

**Weight bytes per emitted token are unchanged by EP alone.** The checkpoint is fixed; `÷EP` on
each device is `xEP` devices reading in parallel. EP pays off through what the freed HBM lets you
do next, not through the division by itself.

## 4. Variant B — Replicated-Input EP — **DEFAULT**

All-gather the tokens so every rank sees the whole group's activations, run local experts, then
reduce-scatter the summed outputs back. Static shapes end to end, which is why it is the default
on an XLA backend: a dispatch whose per-rank token count depends on routing does not lower well.

**Memory** — adds to §3:

| buffer | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| Gathered input | `T_ep * D` = `EP * T_dev * D` | `expert_dispatch` | **xEP** |
| Pre-scatter output | `T_ep * D` | `expert_collect` | **xEP** |

This is the variant's real cost and the thing its name hides: it divides the weights by `EP` and
multiplies the hidden-state buffers by `EP`. At a large `max_num_batched_tokens` those buffers are
what the activation reserve has to cover.

**Communication** — **three** collectives, once per MoE layer:

```python
collective.all_gather(m,     elems_full=T_ep * D, dtype=<expert_dispatch>, axes=EP_AXES)  # tokens
collective.reduce_scatter(m, elems=T_ep * D,      dtype=<expert_collect>,  axes=EP_AXES)  # outputs
collective.all_gather(m,     elems_full=T_ep * E, dtype="fp32",            axes=EP_AXES)  # router logits
```

**The router-logit all-gather is the one that gets forgotten.** Every rank needs every token's
routing decision, so the router's `[tokens, E]` output is gathered alongside the tokens. Its width
is `E`, the expert count — not `D` — which is why it does not look like the others and why a model
built from the token flow alone omits it.

* `all_gather` takes the **gathered** element count, so it is `T_ep * D`, not `T_dev * D`.
* The trailing collective is a `reduce_scatter` when the next layer is sequence-sharded
  (DP / CP / SP). When the next layer is TP-sharded it is an `all_reduce` on the same payload
  instead, because every rank then needs the full result.
* Volume is **independent of `K`** — every token is gathered whatever its routing.
* The two legs routinely differ in dtype (fp8 out, bf16 back); issue them as two calls.

| | |
| --- | --- |
| **group** | EP group — §6 |
| **overlap** | `partial` — issued async on the SparseCore thread, but heavily exposed. **Measured** below |
| **source** | `j-484cd4c0` — seen running in that profile, §4b. The checkout itself is **unread** |

### 4b. What the profile shows

From the `j-484cd4c0` collective table (GLM-5.2-FP8, 2x2x2, TP16, `L_moe` = 75):

| collective | shape | occ/step | decode ms/step | prefill ms/step |
| --- | --- | ---: | ---: | ---: |
| reduce-scatter (outputs) | `bf16[256,6144] -> bf16[16,6144]` | 75 | 6.60 | 145.1 |
| all-gather (tokens) | `bf16[16,6144] -> bf16[256,6144]` | 75 | 2.23 | 58.5 |
| all-gather (router logits) | `f32[16,256] -> f32[256,256]` | 75 | 1.11 | 27.7 |
| | | | **9.93** | **231.3** |

Five things this settles:

1. **The variant is B.** All-gather then reduce-scatter, and **no all-to-all appears anywhere in
   the profile.** Variant A is not what runs on this backend.
2. **`occ_per_step` is 75, which is `L_moe`, not `L` = 78.** The per-layer rule in §2 is confirmed
   by measurement.
3. **The gathered dimension is 16x the per-rank one** (`16 -> 256` tokens, `1024 -> 16384` at
   prefill), so the EP group is all 16 dies — it spans the whole world, as §6 says.
4. **Both legs are `bf16`, not fp8.** Whatever a numerics sheet says about `expert_dispatch`, this
   build sends bf16 in both directions. Do not price an fp8 dispatch leg without re-checking.
5. **The collectives are async but badly exposed.** They are issued on
   `async_execution_thread="sparsecore"`, yet exposed collective wait is **21.9% of the decode step
   and 37.8% of prefill**. Async does not mean free, and `overlap: serial` would have been closer
   to the truth than `overlap: overlapped`.

The router-logit gather is **11-12% of EP's collective time** in both phases — small next to the
other two, and far too large to be the zero it was before this table was read.

### 4c. Scheduling B's all-gather — flat vs hierarchical

§4's all-gather is one collective on paper and a choice of schedule in practice, whenever the slice
has **two link classes** — an on-package die-to-die link and an off-package ICI. Which phase rides
which link then matters in proportion to the ratio between them, which is a property of the
hardware pin, not of any model: read it from `tpu/<gen>.md`.

| schedule | `Model.moe_dispatch_strategy` | what it does |
| --- | --- | --- |
| flat | `all_gather` | one call over every axis; `cost_estimator` searches the axis order |
| hierarchical | `hierarchical_ag` | two named phases: gather across chips over ICI, **then** the chip's own cores over d2d |

**Which model uses which is a config field, not a fact about EP.** Set
`model.moe_dispatch_strategy` in the model's config; nothing here should name a checkpoint.

**The ordering rule, and it is the opposite of the intuitive one.** An all-gather's payload *grows*
every phase — after gathering over a group of `n` you carry `n` times what you started with — so
the fastest link belongs on the **last, largest** phase, not the first. Put d2d first and it spends
the fast link on the smallest message while the full gathered chunk crosses the slow one.

**Reduce-scatter is the mirror image.** Its payload *shrinks*, so the largest phase is the first
and d2d belongs at the front. `MoE: Collect` must derive its own order; copying this one inverts it.

**Expect flat to price at or below hierarchical here, and do not read that as a verdict.**
`cost_estimator` already searches the axis order and can schedule every axis together, which a
forced two-phase split cannot, so a hierarchical schedule tends to come out slightly *more*
expensive on this model. Its real advantage is one the roofline never credits: a hierarchical
implementation **overlaps the two phases except the final one**, while every collective here is
charged fully exposed (README §3). Read `hierarchical_ag` as "the schedule we intend to run", not
"the cheaper number", and price both before concluding anything about a specific cell.

## 5. Variant A — Dispatch-based EP

Send each token only to the ranks owning its `K` selected experts, compute, send the results back.

**Memory** — adds to §3:

| buffer | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| Received tokens | `T_dev * K * D` | `expert_dispatch` | scales with `K`, not `EP` |
| Pre-combine output | `T_dev * K * D` | `expert_collect` | scales with `K`, not `EP` |

**Communication** — two all-to-alls, once per MoE layer:

```python
collective.all_to_all(m, elems=T_dev * K * D, dtype=<expert_dispatch>,
                      axes=EP_AXES, algo="sequential")
collective.all_to_all(m, elems=T_dev * K * D, dtype=<expert_collect>,
                      axes=EP_AXES, algo="sequential")
```

* `elems` is the **full** per-device payload, including the shard bound for this device itself.
* Volume scales linearly with `K`.
* `algo` must be named. `concurrent` is the bisection floor, not a prediction, and `auto` is
  refused — see [`../../COLLECTIVE.md`]({{reference:roofline-analysis/COLLECTIVE.md}}) §2.4.

**Sub-variant — capacity-padded.** A static-shape implementation pads each expert to a capacity
`C`, replacing the payload with `(E / EP) * C * D` and making it independent of the actual
routing. Tokens beyond `C` are dropped. Price this form whenever the implementation pads; the
padded volume can be several times the ragged one.

**Sub-variant — latent dispatch.** Where the experts operate in a latent space
(`D_latent < D`), both payloads become `T_dev * K * D_latent`. At `D_latent = D / 2` that halves
the exchange. Only the width changes — collective, group and dtype sites are the parent's. Applies
to this variant only: variant B gathers before the down-projection, so it moves `D` either way.

| | |
| --- | --- |
| **group** | EP group — §6 |
| **overlap** | `serial`. **No measurement** |
| **source** | **unread** |

## 6. Constraints and interactions

In [`../SKILL.md`]({{reference:roofline-analysis/parallelism/SKILL.md}}) — the world-size product, which schemes are degrees and which are booleans,
the legality rules, group membership, the pairwise effects table, and the do-not-double-count list.
**Not repeated here**: a pairwise fact written in two files becomes two facts.

## 7. Open

* **Variant A is unread and unobserved** — the profile shows it is not what runs here, so nothing
  constrains its numbers. Keep them for costing the alternative, not for recommending it.
* **§4b is one model on one cell.** `L_moe`, the bf16 legs and the exposure fractions are
  GLM-5.2-FP8 on 2x2x2/TP16 at `j-484cd4c0`; none of them transfer to another checkpoint unread.
* **Uniform routing is assumed** throughout. Real expert load is skewed, which makes the
  intermediate buffer a per-device maximum rather than a mean and adds a straggler this model has
  no term for.
* **Router / top-k compute is not counted.** Small next to the expert GEMMs, but not verified
  small on a large `E`.
