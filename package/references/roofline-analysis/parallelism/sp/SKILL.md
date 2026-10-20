---
name: sequence-parallelism-roofline
description: Sequence Parallelism (SP) for roofline analysis — what it divides in HBM, why it changes no communication volume, and which implementation variants exist, with payloads, dtype sites and per-layer counts.
references:
- references/roofline-analysis/parallelism/SKILL.md
---

# Sequence Parallelism (SP)

SP shards **activations along the sequence dimension** in the regions a tensor-parallel layer
leaves replicated — norm, residual, the block boundaries — so that no rank materialises the full
token batch at full width.

Its product is activation memory. It is the one scheme here whose communication column is a
change of *form* rather than of volume.

See [`../SKILL.md`]({{reference:roofline-analysis/parallelism/SKILL.md}}) for the variant schema, the `source` field, and **all
constraints and cross-scheme interactions** — including why SP has no degree of its own and runs
on the TP group.

## 1. Notation

| symbol | meaning |
| --- | --- |
| `SP` | Sequence-parallel width. **Equals `TP`** — SP reuses the TP ranks and has no separate degree |
| `D` | Hidden dimension |
| `T_dev` | Tokens entering the block on this device before any sequence sharding |
| `L` | Layers |
| `L_moe` | Layers whose FFN is a routed MoE |

## 2. Per layer, and which layers

SP does not add collectives. It changes the form of the reductions TP already issues, at the same
per-layer counts — one at the attention exit on every layer, one at the FFN exit on dense-FFN
layers. Under vLLM's variant (§5) the change is applied at the MoE block boundary specifically, so
it touches `L_moe` layers rather than all of them.

## 3. What SP does to HBM

Per device, per layer. One row, and it is the entire reason to turn SP on.

| tensor | per-device elems | dtype site | effect |
| --- | --- | --- | --- |
| Inter-block activations (norm, residual, block I/O) | `T_dev * D / SP` | `tp_allreduce` | **÷SP** |
| Weights — all of them | unchanged | — | **untouched** |
| KV cache | unchanged | `kv_cache` | **untouched** |
| Expert intermediate buffers | unchanged | `moe` | **untouched** |

**SP shards no parameter.** Every weight division on a sharding row comes from TP, EP or PP; SP
contributes none. A report that credits SP with weight savings has attributed TP's work to it.

**The saving is real but bounded by what was replicated in the first place.** Plain TP replicates
only the inter-block activations — the block interiors are already sharded. So SP's ceiling is the
`T_dev * D` boundary buffers, not the layer's whole activation footprint. That ceiling grows with
`max_num_batched_tokens`, which is where SP earns its keep: at a large prefill chunk those buffers
are what the activation reserve has to cover.

## 4. Communication — same bytes, different shape

> [!IMPORTANT]
> **SP does not reduce communication.** `AR = RS + AG` by construction. Replacing one all-reduce
> with one reduce-scatter plus one all-gather moves exactly the same bytes: per layer, TP alone is
> `2 x AR` and TP+SP is `2 x RS + 2 x AG`. Identical. Anyone expecting a 2x collective saving from
> turning SP on has mis-derived it — the saving is the memory row in §3.

With SP off, the boundary collective is TP's:

```python
collective.all_reduce(m, elems=T_dev * D, dtype=<tp_allreduce>, axes=TP_AXES)
```

With SP on, the same traffic is split across the boundary:

```python
collective.reduce_scatter(m, elems=T_dev * D, dtype=<tp_allreduce>, axes=TP_AXES)   # leaving
collective.all_gather(m, elems_full=T_dev * D, dtype=<tp_allreduce>, axes=TP_AXES)  # entering next
```

**Count these once.** They are the same collectives `tp/` lists, re-formed — not additional ones.
Whichever file you price them in, do not price them in both.

## 5. Variant A — MoE-block SP — the one that exists in vLLM

`use_sequence_parallel_moe` with `PP = 1` disables `reduce_results` on `out_proj`, so the attention
exit emits a **reduce-scatter into a sequence-parallel MoE block** instead of an all-reduce
(`Qwen3_5DecoderLayer.use_attn_reduce_scatter_for_moe`).

Two consequences the generic description does not give you:

* **The flag is per model, not global.** It is a decoder-layer property, so a model that does not
  implement it gets variant B's behaviour no matter what is configured.
* **The reduce-scatter feeds EP's block.** On this path the attention exit and the MoE entry are
  the same collective. Charging a TP all-reduce at the attention exit *and* an EP collective at the
  MoE entry double-counts it.

| | |
| --- | --- |
| **collectives** | RS at the attention exit; the MoE block's own entry is EP's — §4 and `../ep/` |
| **group** | TP group |
| **overlap** | `serial`. **No measurement** |
| **source** | `Qwen3_5DecoderLayer.use_attn_reduce_scatter_for_moe` — a named symbol, **not read** at a pinned sha |

## 6. Variant B — Off — **DEFAULT**

Boundary activations replicated across the TP ranks, boundary collective is a plain all-reduce.
This is what every TP number in this repo assumes.

| | |
| --- | --- |
| **collectives** | TP's all-reduce, unchanged — `../tp/` |
| **group** | TP group |
| **overlap** | `serial` |
| **source** | **unread** |

## 7. Variant C — Full Megatron-style SP

Norm, residual and dropout regions sequence-sharded throughout the layer, not only at the MoE
boundary. Strictly more activation saving than variant A at identical communication volume.

| | |
| --- | --- |
| **collectives** | RS + AG at both block boundaries, `L` and `L - L_moe` per §2 |
| **group** | TP group |
| **overlap** | `serial` |
| **source** | **unread**. No vLLM flag found that enables it independently of the MoE path — a negative result, weaker than a confirmed absence |

## 8. Open

* **Which variant is live is UNREAD**, and it decides only the memory rows — the collective volume
  is the same either way, which is the useful half of this file's answer.
* **No measurement of the activation reserve** against the `÷SP` claim. The saving is derived, not
  observed.
* **Variant C rests on a negative result** — "no flag found" is weaker than "not supported".
  Someone should confirm it rather than inherit the absence.
