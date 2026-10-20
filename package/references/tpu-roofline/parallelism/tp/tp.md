---
name: tensor-parallelism-roofline
description: Tensor Parallelism (TP) for roofline analysis — what it divides in HBM, where its KV sharding stops, and which collectives each implementation variant issues, with payloads, dtype sites and per-layer counts.
---

# Tensor Parallelism (TP)

TP splits every weight matrix across `TP` devices: attention heads by head, FFN by intermediate
dimension. Each rank computes a partial result and the ranks reduce.

It is the densest of the schemes — it divides almost everything, and it pays for that on **every
layer**, not on a subset.

See [`../parallelism.md`](../parallelism.md) for the variant schema and the `source` field, and
[`../../COLLECTIVE.md`](../../COLLECTIVE.md) for the estimator API.

## 1. Notation

| symbol | meaning |
| --- | --- |
| `TP` | Tensor-parallel degree — the size of the group the collectives run on |
| `D` | Hidden dimension |
| `H_q`, `H_kv` | Query and key/value head counts |
| `D_h` | Head dimension |
| `F_dense` | Intermediate dimension of a dense FFN layer |
| `T_dev` | Tokens entering the block on this device: prefill chunk tokens plus one per decoding request |
| `L` | Layers |
| `L_moe` | Layers whose FFN is a routed MoE |
| `W` | Devices in the slice |

## 2. Per layer, and which layers

The standard layout puts one reduction after each RowParallel projection: one after attention's
output projection, one after the FFN's down projection. So **two per layer, on every layer** —
there is no dense/sparse split of the kind EP has.

With one exception, which is the thing to get right:

```
attention reduction :  every layer                 ->  L
FFN reduction       :  dense-FFN layers only       ->  L - L_moe
```

**On an MoE layer with EP on, the FFN's trailing reduction is EP's, not TP's.** The EP group spans
the TP ranks, so its reduce-scatter (or all-reduce) already reduces across them. Counting a TP
all-reduce there as well double-charges the largest collective in the layer. Price the MoE
layer's exit in `../ep/`; price only the attention reduction here.

## 3. What TP does to HBM

Per device, per layer.

| tensor | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| GQA QKV + O projection weights | `/ TP` (head-sharded) | `attn_proj` | **÷TP** |
| MLA `q_b_proj`, `kv_b_proj`, `o_proj` | `/ TP` (head-sharded) | `attn_proj` | **÷TP** |
| MLA `q_a_proj`, `kv_a_proj_with_mqa` | full — **replicated** | `attn_proj` | **untouched**, i.e. ×TP across the slice |
| Dense FFN weights | `/ TP` (intermediate-sharded) | `dense_ffn` | **÷TP** |
| Shared expert weights | `/ TP` | `moe` | **÷TP** |
| Embedding / LM head | `/ TP` (vocab-sharded) | `lm_head` | **÷TP** |
| Routed expert weights | `/ EP`, and `EP` spans `TP` | `moe` | untouched by `TP` alone |
| KV cache | `/ min(TP, H_kv)` — **and `/ 1` under MLA** | `kv_cache` | see below |
| Block activations | `T_dev * D`, replicated — unless SP is on, §5 | `tp_allreduce` | plain AR does **not** divide them |

Four rows are worth stating outright:

**MLA's two down-projections are replicated, not sharded.** `q_a_proj` `[D, q_lora]` and
`kv_a_proj_with_mqa` `[D, kv_lora + rope]` project into the shared latent, which has no head
dimension to split. vLLM builds them as one `DeepSeekV2FusedQkvAProjLinear`
(`MergedColumnParallelLinear(..., disable_tp=True)`, `vllm/model_executor/models/deepseek_v2.py:912-927,
1010-1015`), or as `ReplicatedLinear` without `q_lora` (`:1017`). On GLM-5.2 they are 9.8% of an
attention layer's parameters — and because the rest divides, their share grows with TP: at TP=8
dividing them by TP under-counts ~1 GiB per core. vllm-torchtpu's `TPU_LATENT_PROJ_INTRA_CHIP_TP`
exists because of exactly this replication (`layers/adapter/latent_proj_intra_chip.py`), and
splits them across one chip's cores; that is not modelled here.

**TP's KV sharding stops at `H_kv`.** KV is sharded by head, so beyond `TP = H_kv` the heads run
out and further ranks hold replicas. On a model with `H_kv = 8`, `TP = 16` divides KV by 8, not
16 — and the second 8 ranks buy nothing for KV while still paying full collective cost.

**Under MLA, TP divides KV by nothing at all.** MLA compresses KV to one latent vector per token
with no head dimension to split, so every TP rank holds the whole thing. On an MLA checkpoint at
long context — where the KV admission gate usually binds — TP is not a KV lever. This is the
single most common way a TP-heavy sharding is over-credited.

**Plain TP does not divide activations.** Every rank materialises the full `T_dev * D` block
input and output. Closing that gap is sequence parallelism's job, not TP's — §5.

## 3b. Per-device shapes — the definitions op files use

TP's partition is expressed to the rest of the tree as these symbols. **This table is their
definition**; an op file uses them and does not restate how they are derived.

| symbol | per-device value | sharding |
| --- | --- | --- |
| `H_q_local` | `H_q / TP` | column-parallel, by query head. The attention *core* runs `H_q_local * DCP` under DCP — `../dcp/` §4 |
| `H_kv_local` | `max(1, H_kv / TP)` | column-parallel, by KV head |
| `F_local` | `F / TP` | column-parallel on gate/up, row-parallel on down |
| `D_conv_local` | `D_conv / TP` | channel-sharded; depthwise conv stays local |
| `F_expert_local` | `F_expert / TP` | only when the MoE is TP-sharded rather than EP-sharded |

Which projections take which form: **QKV, gate and up are column-parallel; output and down are
row-parallel.** The row-parallel ones are where the reduction in §2 comes from.

**A fused projection is cut per segment, not as one flat dimension.** QKV's output is
`[D, (H_q + 2*H_kv) * D_h]` flattened, but a contiguous split would hand one rank all of Q.
`QKVParallelLinear` slices Q, K and V separately by head and concatenates the pieces, so each rank
holds `[Q: H_q/TP | K: H_kv/TP | V: H_kv/TP]` heads; past `TP = H_kv`, `tp_rank // replicas`
makes neighbouring ranks load the same KV head (`vllm/model_executor/layers/linear.py:1075-1081,
1318-1380`). Gate/up is cut the same way (`MergedColumnParallelLinear`, `:831-872`).

**`max(1, ...)` is the replication rule, and it is not free.** When `TP > H_kv` the KV heads run
out and are replicated across the surplus ranks. Those ranks add no KV division and no attention
FLOP division while still paying full collective cost — the cap described in §3. Under MLA there
is no KV head dimension at all, so the cap is 1 for every `TP`.

## 4. Variant A — Plain all-reduce — **DEFAULT**

ColumnParallel then RowParallel, with an all-reduce after each RowParallel. The classic Megatron
layout and the assumption behind almost every TP number in this repo.

**Memory** — adds to §3:

| buffer | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| Block input / output | `T_dev * D` | `tp_allreduce` | replicated — **not** divided |

**Communication** — per layer, per §2's counts:

```python
collective.all_reduce(m, elems=T_dev * D, dtype=<tp_allreduce>, axes=TP_AXES)
```

* One call after the attention output projection, on every layer.
* One call after the FFN down projection, on dense-FFN layers only.
* If the implementation accumulates in a wider type than the operand, pass `wire_dtype=` — a
  reducing collective is the one place that is legal, and an fp8 activation summed in bf16 puts
  twice the bytes on the wire.

| | |
| --- | --- |
| **group** | TP group |
| **overlap** | `serial`. **No measurement** |
| **source** | **unread** |

## 5. Sequence parallelism — see [`../sp/`](../sp/sp.md)

SP runs on the TP group and re-forms the reduction above into a reduce-scatter leaving the block
and an all-gather entering the next. **Same bytes, different shape** — `AR = RS + AG`, so it moves
no more and no less traffic than variant A.

It is described in [`../sp/sp.md`](../sp/sp.md) and **not duplicated here**. Two things to
carry when SP is on:

* The reduction is counted **once**, in `sp/`. It is this file's collective re-formed, not an
  additional one.
* The block activations in §3 stop being replicated and become `T_dev * D / TP`. That memory row
  is SP's product and is priced there.

## 6. Variant B — Overlapped / collective matmul

The GEMM is decomposed into chunks so each chunk's collective runs against the next chunk's
arithmetic. Wire volume is unchanged; the point is that it stops being additive.

This is the **only variant whose `overlap` field is not `serial`**, and the only one that can
change a comm-bound row into a compute-bound one without moving a byte. Since every cost model in
this tree charges `max(t_mem, t_comp) + t_comm`, a deployment running this and a model assuming
variant A differ by the whole of `t_comm`.

| | |
| --- | --- |
| **collectives** | As variant A or B, same payloads |
| **group** | TP group |
| **overlap** | `overlapped`, partially — the fraction is the unknown |
| **source** | **unread**. Whether XLA's latency-hiding scheduler achieves this on TPU is neither read nor measured |

## 7. Constraints and interactions

In [`../parallelism.md`](../parallelism.md) — the world-size product, which schemes are degrees and which are booleans,
the legality rules, group membership, the pairwise effects table, and the do-not-double-count list.
**Not repeated here**: a pairwise fact written in two files becomes two facts.

## 8. Open

* **Which variant is live is UNREAD.** Whether sequence parallelism is enabled in the bench recipe
  decides the activation-memory rows, though not the collective volume. Whether anything overlaps
  decides whether `t_comm` should be charged at all.
* **The all-reduce dtype is unverified.** Whether the reduction upcasts is now expressible
  (`wire_dtype=`) and still unread; an fp8-operand model that reduces in bf16 is understated 2x on
  its most frequent collective.
* **Overlap is unmeasured** and charged serially, which on the most frequent collective in the
  model is the largest single error bar in this file.
* **TP beyond `H_kv` is not flagged anywhere.** Nothing warns that the extra ranks buy no KV
  division while paying full collective cost.
