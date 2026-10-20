---
name: data-parallelism-roofline
description: Data Parallelism (DP) for roofline analysis — what it replicates in HBM, what it divides, why it issues no collective of its own, and the lockstep coupling it inherits when expert parallelism spans it.
---

# Data Parallelism (DP)

`DP` independent engines over one slice, each with its own scheduler, KV pool and copy of the
weights. Requests are spread across them.

**DP issues no collective.** That is the whole of its communication section, and the reason this
file is short. The one coupling in §4 is inherited from EP, not caused by DP.

See [`../parallelism.md`](../parallelism.md) for the schema, [`../../COLLECTIVE.md`](../../COLLECTIVE.md) for
the estimator API.

## 1. Notation

| symbol | meaning |
| --- | --- |
| `DP` | Data-parallel degree — number of engines |
| `TP` | Tensor-parallel degree within one engine |
| `W` | Devices in the slice. With no PP or PCP, `TP * DP = W` |
| `N_fleet` | Requests in flight across the deployment; per engine `N_fleet / DP` |

## 2. What DP does to HBM

On a **fixed slice** `DP` is bought from `TP` (`TP = W / DP`), so anything `TP` shards gets `DP`
times larger per device. DP does not replicate weights per *device* — it replicates them per
*replica*, which on a fixed slice is the same thing. State any DP claim against a fixed `W`.

| tensor | per-device | dtype site | effect |
| --- | --- | --- | --- |
| Attention + dense weights | `/ TP` = `* DP / W` | `attn_proj`, `dense_ffn` | **xDP** — the cost |
| Shared expert weights | `/ TP` | `moe` | **xDP** |
| Routed expert weights | `/ EP`, and `EP` spans `TP x PCP x DP` | `moe` | **untouched** |
| KV cache | requests held `/ DP`; bytes per token unchanged | `kv_cache` | **÷DP in requests held** |
| Activation buffers | per engine, sized by that engine's `max_num_batched_tokens` | — | untouched |

Two rows carry the whole trade:

**Routed expert weights are indifferent to the TP/DP split.** The EP group spans `TP x PCP x DP`,
so with no PP or PCP it is the whole world `W` whatever the split, and expert weights land at
`E_total / W` either way. What DP makes worse is the attention and dense weights — a small
fraction of a large number on a big MoE checkpoint, which is why DP-heavy shardings do well there.
Check the ratio; it inverts on a dense model.

**On MLA, DP is the only thing that divides per-device KV.** Per-device KV is
`(requests held) * (bytes per token per device) * context`. DP divides the first factor. TP
divides the second by `min(TP, H_kv)` — but MLA has no KV heads to shard, so under MLA TP divides
it by nothing. At long context, where the KV gate usually binds, that makes DP the lever and TP a
non-lever.

## 3. Communication

None. DP replicas do not exchange activations, gradients or KV.

## 4. Lockstep — inherited from EP, not caused by DP

When EP spans the DP ranks they are in one collective group and **must enter the MoE collective
together**. Not a configuration choice; it follows from EP's group, so it applies whenever both
are on. Two costs that do not exist without EP:

* **Dummy forwards.** A rank with no work still enters the collective and runs a forward on
  padding. Weights are read whether or not there are tokens to apply them to, so it costs a real
  step and returns nothing. Worst when the per-rank batch is thin — the low-concurrency regime
  where DP is otherwise most attractive.
* **Step alignment.** The group moves at its slowest rank. `prefill_schedule_interval` damps this
  by admitting prefills only every `N` steps so per-step forward times stay comparable
  (`vllm/config/scheduler.py:143` — **not re-read at a pinned sha**). Model it as an additive
  `(N-1)/2 * T_dispatch` on TTFT, the mean wait for the next admitting step.

The EP collectives themselves are priced in [`../ep/ep.md`](../ep/ep.md). Do not count them
again here.

## 5. Per engine, not per fleet

`max_num_seqs`, `max_num_batched_tokens` and the KV pool are all **per engine**. Fleet-wide is
`x DP`, and reading one as the other is the standard way to be wrong about DP by a factor of `DP`.
`max_num_running_reqs` in the vLLM source is an alias of `max_num_seqs` — not a separate knob, and
not "concurrency".

## 6. Constraints and interactions

In [`../parallelism.md`](../parallelism.md) — the world-size product, which schemes are degrees and which are booleans,
the legality rules, group membership, the pairwise effects table, and the do-not-double-count list.
**Not repeated here**: a pairwise fact written in two files becomes two facts.

## 7. Open

* **No straggler term.** Ranks are assumed balanced at `N_fleet / DP` each. Real imbalance is
  first-order and is the reason `prefill_schedule_interval` exists — so this models the remedy
  without the disease.
* **Dummy-forward frequency is not estimated.** §4 names the cost; quantifying it needs an arrival
  model this roofline does not have.
* **Nothing here has been read at a pinned sha**, including §4's `scheduler.py` line.
