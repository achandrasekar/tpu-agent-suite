---
name: decode-context-parallelism-roofline
description: Decode Context Parallelism (DCP) for roofline analysis — how it shards the KV cache along context and re-spends TP's head sharding on positions, why its write path needs no collective, which attention paths implement it, and the exact collectives each path issues, with payloads, dtype sites and per-layer counts.
---

# Decode Context Parallelism (DCP)

DCP shards the **KV cache along the context dimension**: each rank owns a disjoint, interleaved
slice of every sequence's cached tokens. It reuses the TP ranks rather than adding any, so it costs
no devices.

The mechanism is a **re-spend of TP**, and everything below follows from it:

> DCP doesn't add a new parallel dimension—it borrows from TP. It reallocates dcp ways from heads
> to sequence positions, reducing head sharding from tp to tp // dcp.
> — `vllm-torchtpu:layers/adapter/cp_mla_attention.py:46`

Its product is KV capacity. Its cost is a gather of query heads before the attention core and a
log-sum-exp merge after it, on **every** attention layer.

See [`../parallelism.md`](../parallelism.md) for the variant schema, the `source` field, and all constraints
and cross-scheme interactions.

**Source for this file:** vllm-torchtpu `origin/main` at `fde6c3994`, read 2026-09-23 — what runs
today — plus the upstream vLLM installed in `vllm-torchtpu/uv_venv` (`0.27.0+empty`). Two open PRs
are carried as flags: #1230 (§7.2) and #1218 (§7.3).

## 1. Notation

| symbol | meaning |
| --- | --- |
| `DCP` | Decode-context-parallel degree. Reuses TP ranks — `TP % DCP == 0` |
| `T` | Tokens entering attention on this rank this pass. Queries are **replicated** across the DCP group |
| `H_local` | Query heads this rank owns under TP: `H_q / TP` (`tp/` §3b) |
| `H_attn` | Query heads the attention core runs on this rank: `H_local * DCP` |
| `D_h` | Head dimension (GQA) |
| `d_c`, `d_r` | MLA latent (`kv_lora_rank`) and RoPE (`qk_rope_head_dim`) widths |
| `S` | Context length |
| `K_idx` | Tokens a sparse indexer selects per query |
| `L`, `L_idx` | Layers; layers that run the indexer |

## 2. Which attention paths implement it

DCP is not a property of the model; it is a property of the attention kernel the layer runs. On
main:

| path | DCP | where |
| --- | --- | --- |
| **GQA / MHA**, batched RPA long-context kernel | **yes** — variant A, §6 | `layers/adapter/cp_attention.py`; dispatched at `layers/adapter/attention.py:1425` |
| **Sparse MLA** (MLA + top-k indexer, e.g. GLM-5.2) | **yes** — variant B, §7 | `layers/adapter/attention.py:2248-2278`, `layers/adapter/cp_mla_attention.py`, `kernels/deepseek_v4/streamindex_topk` |
| **Dense MLA** (no indexer) | **no path** | the non-sparse branch (`attention.py:2279-2295`) never reads `dcp_size`. **Nothing refuses the config** — treat as unimplemented, never as priced |
| Kimi K3 attention | refused | `supports_dcp = False`, `models/vllm/kimi_k3/attention.py:442` |
| attention sinks, bundled KV cache | refused | `attention.py:1426-1430` |
| GQA with the HND KV layout | CUSTOM backend and `head_size != 64` only | `attention.py:1268-1276` |

The model-shaped legality rules (GQA head counts, PCP) are in [`../parallelism.md`](../parallelism.md) §Legality.

## 3. What DCP does to HBM

Per device.

| tensor | per-device | dtype site | effect |
| --- | --- | --- | --- |
| KV cache | `/ DCP` along context | `kv_cache` | **÷DCP** — the entire point. vLLM sizes it as `cdiv(max_model_len, DCP)` (`vllm/v1/kv_cache_interface.py:268-270`) |
| Indexer K cache | `/ DCP` along context | `indexer_cache` | **÷DCP** — same interleave |
| Weights — all of them | unchanged | — | **untouched** |
| Gathered queries | `T * H_attn * (width)` | `attn_qk` | ×DCP over TP alone — transient |
| Partial outputs + LSEs | §6, §7 | `attn_qk`, `dcp_lse`, `dcp_combine` | transient |

**DCP shards no parameter.** A report crediting it with weight savings has attributed TP's work.
Upstream's `VLLM_DCP_Q_REPLICATE` would trade the query gather for `DCP`-way replicated `q_b_proj`
weights, but it is **not wired on TPU** (§8), so no weight row changes.

**It is the KV lever that works under MLA.** TP shards KV by head and MLA has no KV heads, so TP
divides MLA's KV by nothing. DCP shards by *context* instead. On an MLA checkpoint at long context,
DCP and DP are the two levers and TP is not one.

**On GQA it only reclaims TP's replicas.** vLLM requires `DCP <= TP / H_kv` (`../parallelism.md`
§Legality), so the DCP group always shares one KV head that TP had replicated. The per-device
divisor `min(TP, H_kv) * DCP` therefore never exceeds `TP`: DCP turns wasted replicas into context
shards, and buys nothing on a model where TP had not run out of KV heads.

**Diminishing returns are structural.** DCP buys KV capacity, and KV capacity stops being worth
anything once some other gate binds first. Past that point additional `DCP` adds §6/§7 collective
cost for no admission benefit.

## 4. What DCP does to the attention core

Every rank runs the attention kernel with the **whole DCP group's heads** against **its own
context shard**.

| quantity | TP alone | TP + DCP | why |
| --- | --- | --- | --- |
| heads in the core | `H_local` | `H_attn = H_local * DCP` | queries are all-gathered by head first |
| keys scanned per query — dense | `S` | `S / DCP` | only this rank's positions are resident |
| keys scanned per query — sparse | `K_idx` | `~K_idx / DCP` | each rank attends only to the global winners it **owns** (`-1` elsewhere); balanced ownership assumed, §9 |
| attention FLOPs | `2·T·H_local·S·d` | **unchanged** | `DCP`× the heads over `1/DCP` of the keys |
| KV read from HBM per step | `fetch` | **`fetch / DCP`** | each rank reads only its own shard |
| KV resident | `B·S` | `B·S / DCP` | §3 |

The **indexer** (sparse MLA only) scores every replicated query against this rank's shard:

| quantity | TP + DCP |
| --- | --- |
| indexer QK FLOPs, indexer-cache read, `[T, S]` score matrix | **÷DCP** |
| local top-k | over `S / DCP` candidates, emits `K_idx` |
| merge top-k | over `DCP * K_idx` gathered candidates, identical on every rank (`streamindex_topk.py` stage 3) |

## 5. The write path is free

Queries are replicated; each rank writes only the rows it owns, and rows belonging to another rank
are aimed at an out-of-bounds page and discarded. **No collective.** GQA's new-token pass
(§6) is also local: new tokens are not DCP-sharded, so every rank attends them with its own heads.

## 6. Variant A — GQA two-pass (batched RPA long-context)

`layers/adapter/cp_attention.py:300-383`, `forward_with_dcp`:

1. **CACHE_ONLY pass.** All-gather the queries by head (`:350`), attend against this rank's cache
   shard, returning a partial output `[T, H_attn, D_h]` in the query dtype
   (`batched_rpa_longctx/wrapper.py:609-610`) and a 32-bit LSE `[T, H_attn]`.
2. All-gather partials **and** LSEs along the token axis (`:100-102`), fold them, and slice this
   rank's own `H_local` heads back out (`:370-376`).
3. **NEW_TOKENS_ONLY pass.** This rank's own heads over this step's new tokens. Local.
4. LSE-combine the two results. Local.

```python
collective.all_gather(m, elems_full=T * H_local * DCP * D_h,        dtype=<attn_qk>,  axes=DCP_AXES)  # 1. query heads
collective.all_gather(m, elems_full=T * H_local * DCP * D_h * DCP,  dtype=<attn_qk>,  axes=DCP_AXES)  # 2. partials
collective.all_gather(m, elems_full=T * H_local * DCP * DCP,        dtype=<dcp_lse>,  axes=DCP_AXES)  # 2. LSEs
```

* All three, **every** attention layer.
* **The partial gather is `DCP²`, not `DCP`.** Each rank's partial already carries `H_attn =
  H_local * DCP` heads, and the gather stacks `DCP` of them. Writing `T * H * D_h * DCP` with `H`
  the TP-local head count under-charges it `DCP`-fold.
* The partial gather is what makes variant A expensive at high degree: at `DCP = 8` it moves 8×
  the query gather.

| | |
| --- | --- |
| **group** | The DCP group: `DCP` consecutive TP ranks inside one engine |
| **overlap** | `serial`. **No measurement** |
| **source** | `vllm-torchtpu@fde6c3994:layers/adapter/cp_attention.py:83-110, 300-383` |

## 7. Variant B — sparse MLA (MLA + top-k indexer)

### 7.1 The attention combine — every sparse-attention layer

`layers/adapter/attention.py:2248-2278`, then `cp_mla_attention.py`:

1. All-gather the absorbed query heads, `ql_nope` `[T, H_local, d_c]` and `q_pe` `[T, H_local,
   d_r]`, to `H_attn` (`all_gather_heads`, `cp_mla_attention.py:43-60`). Two collectives, in the
   attention input dtype.
2. The sparse kernel attends against this rank's shard over the winners it owns, returning a
   partial `[T, H_attn, d_c]` and its LSE `[T, H_attn]`.
3. All-gather the LSEs, cast to **f32** (`_weight_by_lse`, `:96-110`).
4. **Reduce-scatter** the LSE-weighted partials, in **f32**, along the head axis — which both sums
   the ranks' contributions and narrows the heads back to `H_local` (`:93`).

```python
collective.all_gather(m,     elems_full=T * H_local * DCP * d_c,  dtype=<attn_qk>,     axes=DCP_AXES)  # 1. ql_nope
collective.all_gather(m,     elems_full=T * H_local * DCP * d_r,  dtype=<attn_qk>,     axes=DCP_AXES)  # 1. q_pe
collective.all_gather(m,     elems_full=T * H_local * DCP * DCP,  dtype=<dcp_lse>,     axes=DCP_AXES)  # 3. LSEs
collective.reduce_scatter(m, elems=T * H_local * DCP * d_c,       dtype=<dcp_combine>, axes=DCP_AXES)  # 4. weighted partials
```

* All four fire on **every layer that runs sparse attention** — gated on `topk_indices`, not on
  whether that layer ran its own indexer. On GLM-5.2 that is all `L = 78`, not `L_idx = 21`.
* **This merge is not absorbed by `o_proj`.** The partials "must go through the log-sum-exp merge
  above before `W_UV`" (`layers/core/attention_interface.py`, `sparse_mla_attention_dcp`), and the
  reduce-scatter's output is TP-local again, so `o_proj`'s TP reduction then runs as it would
  without DCP. The merge is an extra collective, not a re-formed one.
* `dcp_combine` is f32 because the weighting runs in f32 and the result is cast back only after
  the reduce-scatter.

| | |
| --- | --- |
| **group** | The DCP group |
| **overlap** | `serial`. **No measurement** |
| **source** | `vllm-torchtpu@fde6c3994:layers/adapter/attention.py:2248-2278`, `layers/adapter/cp_mla_attention.py:43-110` |

### 7.2 The indexer candidate exchange — indexer layers only

Each rank takes a local top-`K_idx` over its shard, the candidates are exchanged, and every rank
merges them into the exact global top-`K_idx`, keeping the winners it owns
(`kernels/deepseek_v4/streamindex_topk`, `streamindex_topk_dcp`). Two implementations, selected by
**`Model.dcp_indexer_gather`**:

| value | what | collectives | bytes per candidate |
| --- | --- | --- | --- |
| **`"two_ag"`** — **main today, the default** | indices and scores gathered separately, each `[T, K_idx] -> [T, DCP*K_idx]` | 2 × all-gather | 8 |
| `"single_ag"` — **PR #1230, open** | scores plus one trailing source-rank column, `[T, K_idx+1] -> [T, DCP*(K_idx+1)]`. Indices never leave: rank `r` scores only its own shard, so every candidate it emits is one it owns | 1 × all-gather | `4 + 4/K_idx` ≈ 4 |

```python
# "two_ag" (main)
collective.all_gather(m, elems_full=T * K_idx * DCP,       dtype=<dcp_gather>, axes=DCP_AXES)  # indices, int32
collective.all_gather(m, elems_full=T * K_idx * DCP,       dtype=<dcp_gather>, axes=DCP_AXES)  # scores, int32 (raw f32 bits)
# "single_ag" (#1230)
collective.all_gather(m, elems_full=T * (K_idx + 1) * DCP, dtype=<dcp_gather>, axes=DCP_AXES)
```

* On `L_idx` layers only — the layers that run the indexer. The payload scales with `K_idx`, not
  with `H * D_h`, which is why it does not look like an attention collective.
* `dcp_gather` is a 4-byte site: indices are int32 and scores travel as int32 raw f32 bits
  (SparseCore's top-k compares bits).
* Flip the default to `"single_ag"` when #1230 merges. Its commit reports MMLU-Pro
  0.7471 → 0.7457 on GLM-5.2-NVFP4 at `TP=8/DCP=8`, 1400 samples — neutral, though both arms carry
  ~236 null completions.

| | |
| --- | --- |
| **group** | The DCP group |
| **overlap** | `serial`. **No measurement** |
| **source** | `vllm-torchtpu@fde6c3994:kernels/deepseek_v4/streamindex_topk` stage 2; `"single_ag"`: PR #1230 head `de1b7990` |

### 7.3 Query-token chunking — PR #1218, open: an overlap, not a cost

`TPU_STREAMIDX_CHUNK_TOKENS` cuts the indexer's replicated query-token axis into pieces; each
piece scores into its own buffer and runs its own SparseCore top-k, and consecutive pieces share an
XLA scheduling group so chunk *i*'s top-k overlaps chunk *i+1*'s TensorCore scoring
(`kernels/deepseek_v4/streamindex_topk/streamindex_topk.py:1068-1110` on main). The knob already
exists on main for the non-DCP indexer (`mla_attention_op.py:190`); #1218 extends it to
`streamindex_topk_dcp`. It is orthogonal to the KV axis DCP shards.

**What it changes is time, not traffic.** Bytes and FLOPs are unchanged; the SparseCore top-k is
hidden behind scoring instead of following it. `roofline.py` sums every op serially by design, so
it **cannot show this gain** and carries no knob for it. Ceiling on what is being left out: the
whole `DSA Indexer: TopK` row — on GLM-5.2 256k/1k, 442 ms of a 45.4 s prefill at `tp16`
(~1%), 40 ms of 60.9 s at `tp16-dcp16`. Two modelling notes for whoever adds overlap: the row is
charged as a TensorCore VPU op while main runs it on the SparseCore, and under DCP it also carries
the `DCP * K_idx` merge.

#1218 measured MMLU-Pro 0.7579 at `TP=16, DCP=16` on GLM-5.2-FP8 (`j-7f518b28-3690-4810-8e5d`),
which is also the evidence that multi-host DCP runs.

## 8. Not variants on TPU

Named so nobody prices them:

* **`VLLM_DCP_Q_REPLICATE`** (upstream, `vllm/model_executor/models/deepseek_v2.py:1026-1032`)
  swaps `q_b_proj` for `DCPGroupColumnParallelLinear` so each rank computes the group's heads
  itself. vllm-torchtpu's sparse DCP branch calls `all_gather_heads` unconditionally; the flag is
  mentioned on main only in a comment (`mla_attention_op.py:596`).
* **`dcp_comm_backend = "a2a"`** (upstream `vllm/config/parallel.py:354`) is a GPU backend choice;
  it does not appear in vllm-torchtpu.

## 9. Open

* **Balanced ownership is assumed** for the sparse core's `K_idx / DCP`. The interleave
  (`cp_kv_cache_interleave_size`, a multiple of 4 on sparse MLA) spreads winners, but a contiguous
  run of winners can land on fewer ranks, and the merge waits on the busiest.
* **Uniform context distribution is assumed** across a ragged batch; no straggler term.
* **Overlap is unmeasured** and every collective above is charged serially.
* **DCP + DP has never been run.** Legal by reading (the DCP group is carved inside each engine's
  TP group), unexercised by any test or job.
