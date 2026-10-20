---
name: prefill-context-parallelism-roofline
description: Prefill Context Parallelism (PCP) for roofline analysis — how it splits every step's tokens across ranks of its own, what that divides and what it replicates across attention, dense, MoE and the KV cache, the in-kernel KV ring that pays for it, which attention paths implement it, and the replicated-cache variant priced for evaluation.
---

# Prefill Context Parallelism (PCP)

PCP splits the **tokens of each step** across ranks, and every rank runs the whole model on its
share. Almost every op in a transformer is per-token, so projections, FFN and experts simply run on
`1/PCP` of the rows. Attention is the exception — a query needs every earlier key — and PCP's
implementation is almost entirely about that one op.

It consumes ranks of its own: `world = TP x PCP x PP x DP`.

See [`../parallelism.md`](../parallelism.md) for the variant schema, the `source` field, and all constraints
and cross-scheme interactions.

**Source for this file:** vllm-torchtpu `origin/main` at `fde6c3994`, read 2026-09-23, plus the
upstream vLLM installed in `vllm-torchtpu/uv_venv` (`0.27.0+empty`).

## 1. Notation

| symbol | meaning |
| --- | --- |
| `PCP` | Prefill-context-parallel degree. Consumes ranks |
| `T` | Tokens the **engine** schedules this step (prefill chunk + decode tokens) |
| `T_local` | Tokens on this rank: `T / PCP` |
| `B` | Sequences in the step |
| `S` | Context length |
| `I` | `cp_kv_cache_interleave_size` |
| `w_kv` | KV width per token per layer: `d_c + d_r` (MLA), `2 * H_kv_local * D_h` (GQA) |

## 2. How the tokens are split, and when

Global positions are cut into chunks of `I` and dealt round-robin, rank-major
(`kernels/experimental/pcp_streaming_rpa/README.md`, "Token and Cache Layout"):

```text
positions: [0,I) [I,2I) [2I,3I) [3I,4I) [4I,5I) ...
rank:        0     1      2      3      0     ...
```

Each rank packs its chunks into compact local rows; the kernel reconstructs global positions for
the causal mask. Interleaving rather than contiguous quarters balances causal work — late
positions attend to more keys.

**Every step with scheduled tokens runs this layout, decode included** (`layers/core/
pcp_sequence_layout.py`, `evaluate_runner_chunk`). The name says where it pays, not where it runs:
in unified serving a decode step is PCP-split and pays the attention ring below. The only refusal
is a request span crossing the prompt/decode boundary. Ranks pad to an equal local token count
(`global_num_tokens = local * PCP`), so an uneven split is paid as padding.

## 3. What PCP does, per component

Per device.

| component | under PCP | effect | source |
| --- | --- | --- | --- |
| Attention / MLA / dense-FFN **weights** | TP-sharded as usual; the PCP ranks are **replicas** — vLLM linear layers use the TP group only | ÷TP, **×PCP replicated** | vLLM `ColumnParallelLinear`/`RowParallelLinear` |
| Embedding / LM head | TP only | ÷TP, ×PCP replicated | |
| **Routed experts** | the MoE group flattens over `DP x PCP x TP` (`vllm/model_executor/layers/fused_moe/config.py:1113-1120`). **TPU requires EP with PCP** (§6) | ÷(TP·PCP·DP), whole experts per rank | |
| Shared experts, router | TP-sharded / replicated, as without PCP | ÷TP / ×1 | |
| **Every per-token op** — projections, FFN, experts, norms, RoPE | runs on `T_local` rows | FLOPs and activation traffic **÷PCP** | |
| TP reduction | inside the TP group, on `T_local` tokens | payload **÷PCP** | |
| MoE dispatch/collect | over the EP group, which includes the PCP ranks | per-rank payload on `T_local` tokens | `../ep/` |
| **Attention FLOPs** | local queries × **all** keys | ÷PCP | |
| **KV cache** (variant A) | **striped by page**: global page `p` on rank `p % PCP` | **÷PCP** — a capacity lever | README |
| KV **read from HBM** per step (variant A) | each rank reads its own shard; the rest arrives over ICI | ÷PCP | |

**The weight rows are what folding PCP into TP gets wrong.** PCP ranks are not more TP ranks: a
dense weight is replicated across them, and only the experts — through EP — divide. What PCP
divides is **tokens**, and it divides them for every op, not just attention.

## 4. Variant A — Ring attention over a striped cache — **DEFAULT, what main runs for GQA**

`kernels/experimental/pcp_streaming_rpa`; dispatched at `layers/adapter/attention.py:931-1027`.
Each rank keeps a local shard of the paged KV cache and streams KV pages around the PCP ring while
computing attention for its local query tiles, with online softmax. No rank gathers the whole
cache. Two Pallas passes: current-chunk causal, then history no-causal, reusing each ring transfer
for all local KV heads. New K/V rows are written to their owner — directly, or captured during the
ring and written back before the history pass. **No separate collective.**

The exchange is point-to-point remote DMA inside the kernel. Its volume per layer is that of a
ring all-gather of every KV page the step's sequences read, so it is priced as one:

```python
collective.all_gather(m, elems_full=fetch * w_kv, dtype=<kv_cache>, axes=PCP_AXES)   # every attention layer
collective.all_gather(m, elems_full=B * S * D_idx, dtype=<indexer_cache>, axes=PCP_AXES)  # indexer layers, MLA only
```

* `fetch` is the distinct KV tokens the step reads (`kv_fetched`) — for dense attention `B * S`.
  **It scales with the context, not with the new tokens.** At long context a decode step streams
  the whole cache across ICI every layer, which is the regime DCP exists for.
* The ring overlaps the attention compute by construction. The roofline charges it serially, like
  every other collective here, so this is an upper bound on its exposed cost.
* The indexer line is part of the MLA evaluation (§5): the indexer scores every key, so its K cache
  streams in full.

| | |
| --- | --- |
| **group** | The PCP group |
| **overlap** | `overlapped` in the kernel; charged `serial` |
| **source** | `vllm-torchtpu@fde6c3994:kernels/experimental/pcp_streaming_rpa/README.md`, `layers/adapter/attention.py:931-1095` |

## 5. Which attention paths implement it

| path | PCP on main | priced as |
| --- | --- | --- |
| **GQA / MHA full attention** (batched RPA) | **yes** — variant A. Refuses attention sinks, logits soft-cap, `skip_kv_update` (`attention.py:1073-1095`), sliding window and `head_dim == 64` (README) | variant A |
| **GDN** (Qwen3.5 linear attention) | yes — its own kernel pushing token chunks between ranks per head shard (`kernels/gdn/v3/pcp_wrapper.py`) | **not priced**: the roofline has no GDN ops |
| **MLA**, dense or sparse (GLM-5.2) | **no path.** The MLA forward never reads `prefill_context_parallel_size`. One docstring describes a replicated-cache design (`kernels/mla/kv_cache_utils.py:434`); no code implements it | **priced for evaluation**, per `Model.pcp_kv_strategy`, and flagged by `Sharding.check` |

`Model.pcp_kv_strategy` selects what an MLA row — or a what-if GQA row — is priced as:

| value | what | KV per device | collective per layer |
| --- | --- | --- | --- |
| **`"ring"`** — default | variant A: striped cache, KV ring | ÷PCP | ring of `fetch * w_kv` |
| `"allgather"` | variant B: replicated cache; each rank all-gathers the step's **new** rows | **×1** | all-gather of `T * w_kv` |

## 6. Variant B — Replicated cache with a new-rows all-gather — evaluation only

Every rank holds the whole cache and, after writing its own new rows, all-gathers everyone
else's. Cheaper on the wire than the ring whenever `T << fetch` — always at decode, and in any
prefill chunk short against its context — but it buys **no KV capacity**. This is the design the
`kv_cache_utils.py:434` docstring describes for MLA. **Nothing on main implements it.**

```python
collective.all_gather(m, elems_full=T * w_kv, dtype=<kv_cache>, axes=PCP_AXES)          # every attention layer
collective.all_gather(m, elems_full=T * D_idx, dtype=<indexer_cache>, axes=PCP_AXES)    # indexer layers
```

| | |
| --- | --- |
| **group** | The PCP group |
| **overlap** | `serial`. **No measurement** |
| **source** | **unread** as code — a design in a docstring |

## 7. Once per step, not priced

The PCP layout ends the forward with an all-gather of hidden states and an all-reduce that picks
the logits rows each rank owns (`pcp_sequence_layout.py:479-510`). Both are once per **step**, not
per layer, and the roofline has no step-level row to hang them on. At GLM-5.2's 78 layers they are
small against the per-layer collectives; they are named here so nobody counts them as zero by
accident.

## 8. Open

* **GDN is unpriced**, so PCP on Qwen3.5-class models is costed on its full-attention layers only.
* **MLA has no path**; every MLA PCP number is an evaluation of a kernel nobody has written.
* **Overlap is unmeasured.** The ring is designed to hide behind attention compute and is charged
  in full.
* **Padding to equal local counts is not modelled**; a ragged split is paid as idle rows.
