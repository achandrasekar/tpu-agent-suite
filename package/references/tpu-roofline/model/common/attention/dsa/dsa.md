---
name: dsa-roofline
description: >-
  Calculates FLOPS and memory transfer for DeepSeek Sparse Attention (DSA).
  Use when estimating DSA performance, focusing on the Lightning Indexer overhead and the sparse MLA execution using top-k tokens.
scripts:
- scripts/tpu-roofline/roofline.py
---

# DeepSeek Sparse Attention (DSA) Roofline Estimation

Estimate the roofline performance of DeepSeek Sparse Attention (DSA) by
calculating FLOPS and memory transfer. Report per-operation level FLOPS and
memory transfer. DSA is a sparse variant of Multi-Head Latent Attention (MLA).
It uses a lightweight **Lightning Indexer** to select the top-$K_{idx}$ most
relevant tokens, and then performs standard MLA attention *only* on those
selected tokens.

Carefully distinguish between the **Prefill** and **Decode** phases.

## Requirements

You must read and follow:

*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)

## Workflow

1.  **Identify DSA Dimensions:** Determine the following dimensions:
    *   `B`: Batch Size
    *   `S`: Sequence Length (for Prefill) or current KV cache length (for
        Decode)
    *   `D`: Hidden Dimension (model dimension)
    *   `H_q`: Number of Query heads (for main MLA)
    *   `D_h`: Head dimension (for main MLA)
    *   `D_c`: KV compression latent dimension (for main MLA)
    *   `D_c_q`: Query compression latent dimension (for main MLA)
    *   `D_r`: RoPE head dimension (for main MLA)
    *   `H_idx`: Number of indexer query heads (default 64)
    *   `D_idx`: Head dimension for indexer query and key (default 128)
    *   `K_idx`: Number of tokens selected by the indexer (default 2048)
2.  **Determine Data Types & Apply Type Promotion:** Read
    [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md).
3.  **Identify the Phase:** Prefill or Decode.
4.  **Write and Run Python Script for Calculations:** Do all math in Python.
5.  **Output Format:** Output a markdown table and JSON block as required.

--------------------------------------------------------------------------------

## Calculations

**Note:** Assume FlashAttention-like sparse kernels are used. Intermediate
attention maps (scores/logits) HBM write/read costs can be ignored if fused.

> **"If fused" is a capacity test, not an assumption.**
>
> The per-head logits `[B, S_q, S, H_idx]` are genuinely never material — a flash kernel tiles over
> keys. The **aggregated score matrix `[B, S_q, S]` is different**: top-K needs all `S` scores for a
> query before it can rank them, so unless the kernel streams its top-K it materialises. Test it:
> `B * S_q * S * bytes` against VMEM (8 MiB on v7x).
>
> | shape | score matrix | |
> | --- | --- | --- |
> | prefill, `S_q` 1024, `S` 128K, bf16 | 256 MiB | **32x over VMEM — spills** |
> | decode, `S_q` 1, `S` 128K, bf16 | 0.25 MiB | fits |
>
> So the sections below hold for **decode** and not for **prefill at long context**. When it spills,
> Score Aggregation *writes* `B*S_q*S*bytes` and Top-K *reads* it back. `{{script:tpu-roofline/roofline.py}}` models
> both via `Model.indexer_fused_scores`; charging the read without the write, or the reverse, is the
> error this note exists to stop.

### 1. Prefill Phase (S tokens)

During Prefill, we process a block of $S$ tokens. We run the indexer to select
top-$K_{idx}$ tokens for each query, and then run sparse MLA.

#### A. Lightning Indexer (Overhead)

The indexer runs on all $S$ input tokens to select the top-$K_{idx}$ keys for
each query.

*   **Query Projection (`wq_b`)**: Projects query from latent space to indexer
    space.
    *   Input: `low_rank_q` shape `[B, S, D_c_q]`
    *   FLOPS: `2 * B * S * D_c_q * H_idx * D_idx`
    *   Memory: Read `low_rank_q` (`B*S*D_c_q`), Write `q_idx`
        (`B*S*H_idx*D_idx`)
*   **Key Projection (`wk`)**: Projects input to shared indexer key.
    *   Input: `inputs_kv` shape `[B, S, D]`
    *   FLOPS: `2 * B * S * D * D_idx`
    *   Memory: Read `inputs_kv` (`B*S*D`), Write `k_idx` (`B*S*D_idx`) to
        Indexer KV Cache.
*   **Head Weights Projection (`weights_proj`)**: Projects input to head
    weights.
    *   Input: `inputs_q` shape `[B, S, D]`
    *   FLOPS: `2 * B * S * D * H_idx`
    *   Memory: Read `inputs_q` (`B*S*D`), Write `weights` (`B*S*H_idx`)
*   **QK Logits**: pairwise similarity using ReLU.
    *   $q_{idx}$ shape `[B, S, H_idx, D_idx]`, $k_{idx}$ shape `[B, S, D_idx]`
    *   Formula: `ReLU(q_idx @ k_idx.T)`
    *   FLOPS: `2 * B * S^2 * H_idx * D_idx`
    *   Memory: Read `q_idx`, Read `k_idx`.
*   **Score Aggregation**: Weighted sum over heads.
    *   Logits shape `[B, S, S, H_idx]`, Weights shape `[B, S, H_idx]`
    *   Formula: `Logits @ Weights`
    *   FLOPS: `2 * B * S^2 * H_idx`
    *   Memory: Read Weights (`B*S*H_idx * bytes`), **once** -- and Write Score
        (`B*S*S*bytes`) unless the kernel streams top-K. Logits are read from
        VMEM. The Score is 256 MiB at S=128K against 8 MiB of VMEM, so at prefill
        it does **not** stay resident -- see the capacity test above.
*   **Top-K Selection**:
    *   Select top $K_{idx}$ tokens from Score.
    *   Memory: Write Indices (`B*S*K_idx * 4` -- **int32, not the activation
        dtype**; an index into `S` needs `ceil(log2(S))` = 17 bits at S=128K). A
        *mask* instead of indices would be `B*S*S` of int8, 64x larger here -- say
        which you mean. Plus Read Score (`B*S*S*bytes`) whenever Score Aggregation
        had to write it. **The two must agree**: read it back if and only if it was
        written.

#### B. Sparse MLA (Core Attention)

The main attention is computed only for the selected $K_{idx}$ tokens per query.
We assume weight absorption is used.

*   **Query Projections (Dense)**: All queries are projected.
    *   Generate $c_q$: `2 * B * S * D * D_c_q` FLOPS.
    *   Generate $Q_c$: `2 * B * S * D_c_q * (H_q * D_h)` FLOPS.
    *   Generate $Q_r$: `2 * B * S * D * (H_q * D_r)` FLOPS.
*   **KV Projections (Dense)**: All keys/values are compressed and cached.
    *   Generate $c_{kv}$ (cache): `2 * B * S * D * D_c` FLOPS. (Written to main
        KV cache).
    *   Generate $K_r$ (cache): `2 * B * S * D * D_r` FLOPS. (Written to main KV
        cache).
*   **Q Proj to K (Weight Absorption for Q)**:
    *   Project $Q_c$ to $Q'_c$ shape `[B, S, H_q, D_c]`.
    *   FLOPS: `2 * B * S * H_q * D_h * D_c`
*   **QK Phase (Sparse)**: Only compute for selected $K_{idx}$ keys.
    *   **Transfer and occupancy differ by `S / K_idx` here** -- 64x at S=128K. The
        memory TRANSFER is the `B * K_idx` selected tokens below; the cache
        OCCUPANCY is the whole `B * S`. See REQUIREMENTs.md.
    *   Nope QK ($Q'_c @ c_{kv}^T$): `2 * B * S * K_idx * H_q * D_c` FLOPS.
    *   Rope QK ($Q_r @ K_r^T$): `2 * B * S * K_idx * H_q * D_r` FLOPS.
    *   Memory: Read $Q'_c$, $Q_r$. Read selected $c_{kv}$ and $K_r$ from cache.
*   **AV Phase (Sparse)**:
    *   $O_{latent} = Score \times c_{kv}$ (only for selected keys).
    *   FLOPS: `2 * B * S * K_idx * H_q * D_c`
    *   Memory: Write $O_{latent}$ (`B*S*H_q*D_c`). **No $c_{kv}$ read** -- the QK
        Phase above already fetched it. Under weight absorption the latent is *both*
        K and V, so the flash kernel fetches it once and uses it twice; charging it
        in both ops reads 136 MiB/layer from a 72 MiB cache at S=128K.
        Contrast plain GQA/MHA, where K and V are distinct tensors and the AV read
        is real -- see `../global_attention`.
*   **V Up Proj (Deferred V Projection)**:
    *   Project $O_{latent}$ back to $D_h$.
    *   FLOPS: `2 * B * S * D_c * (H_q * D_h)`
*   **Output Projection**:
    *   FLOPS: `2 * B * S * (H_q * D_h) * D`

--------------------------------------------------------------------------------

### 2. Decode Phase (1 token)

During decode, we process 1 new token ($S_q = 1$) attending to $S$ cached
tokens.

#### A. Lightning Indexer (Overhead)

The indexer scans all $S$ cached keys to select the top-$K_{idx}$ tokens for the
single new query.

*   **Query Projection (`wq_b`)**:
    *   FLOPS: `2 * B * 1 * D_c_q * H_idx * D_idx`
*   **Key Projection (`wk`)**: For the new token.
    *   FLOPS: `2 * B * 1 * D * D_idx` (Written to indexer KV cache).
*   **Head Weights Projection (`weights_proj`)**:
    *   FLOPS: `2 * B * 1 * D * H_idx`
*   **QK Logits (Dense scan)**:
    *   $q_{idx}$ (size 1) @ $k_{idx}$ (size $S$, read from indexer cache).
    *   FLOPS: `2 * B * 1 * S * H_idx * D_idx`
    *   Memory: Read Indexer Key Cache: `B * S * D_idx * bytes`.
*   **Score Aggregation**:
    *   FLOPS: `2 * B * 1 * S * H_idx`
    *   Memory: Read Weights (`B * 1 * H_idx * bytes`). Score is `B*1*S` -- 0.25
        MiB at S=128K, inside VMEM -- so here the fused case really does hold and
        its HBM transfers are ignored.
*   **Top-K Selection**: Select $K_{idx}$ tokens from $S$ scores.
    *   Memory: Write Indices (`B * 1 * K_idx * 4`, int32 -- see prefill).
        (Score is read
        from VMEM, so its HBM read cost is ignored).

#### B. Sparse MLA (Core Attention with Weight Absorption)

We only perform MLA attention computation and cache reads for the selected
$K_{idx}$ tokens.

*   **Query & KV Latents Generation (1 Token)**:
    *   Generate $c_q$: `2 * B * 1 * D * D_c_q` FLOPS.
    *   Generate $Q'$ (Absorbed Query): `2 * B * 1 * D_c_q * (H_q * D_c)` FLOPS.
    *   Generate $Q_r$ (RoPE Query): `2 * B * 1 * D * (H_q * D_r)` FLOPS.
    *   Generate $c_{kv}$: `2 * B * 1 * D * D_c` FLOPS. (Written to main KV
        cache).
    *   Generate $K_r$ (RoPE Key): `2 * B * 1 * D * D_r` FLOPS. (Written to main
        KV cache).
*   **QK Phase (Sparse)**:
    *   Nope QK ($Q' @ c_{kv}^T$): `2 * B * 1 * K_idx * H_q * D_c` FLOPS.
    *   Rope QK ($Q_r @ K_r^T$): `2 * B * 1 * K_idx * H_q * D_r` FLOPS.
    *   Memory: **Only read selected keys from cache**.
        *   Read selected $c_{kv}$ Cache: `B * K_idx * D_c * bytes`
        *   Read selected $K_r$ Cache: `B * K_idx * D_r * bytes`
*   **AV Phase (Sparse)**:
    *   $O_{latent} = Score \times c_{kv}$ (only for selected keys).
    *   FLOPS: `2 * B * 1 * K_idx * H_q * D_c`
    *   Memory: Write $O_{latent}$ only. **No $c_{kv}$ read** -- the QK Phase already
        fetched those same selected entries. See prefill.
*   **Output Projection**:
    *   $O_{proj} = O_{latent} W_O'$ (where $W_O'$ has shape `[H_q * D_c, D]`)
    *   FLOPS: `2 * B * 1 * (H_q * D_c) * D`

--------------------------------------------------------------------------------

## Tensor Parallelism (TP)

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and how the per-device `*_local` symbols are derived, is defined in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b.
> This section states only the resulting per-device FLOPs and memory.

*   Replace $H_q$ with `H_q_local` and $H_idx$ with `H_idx_local` in all equations above for
    per-device metrics.
*   **DSA-specific:** the caches ($c_{kv}$, $K_r$, and the indexer key cache $k_{idx}$) are
    replicated rather than sharded, so their read bandwidth does **not** scale down with TP even
    though the FLOPs do. This asymmetry is the op's, not TP's — it is why a DSA layer stays
    memory-bound as TP rises.
*   The output projection's partial sums require a reduction across the TP group; it is priced in
    [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md), not here.

--------------------------------------------------------------------------------

## References

*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Type Promotion](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Collective Cost](../../../../COLLECTIVE.md)
