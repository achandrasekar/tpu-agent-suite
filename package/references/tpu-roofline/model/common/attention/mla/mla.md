---
name: mla-roofline
description: >-
  Calculates FLOPS and memory transfer for Multi-Head Latent Attention (MLA).
  Use when estimating MLA performance, particularly focusing on weight absorption in decode phase, latent KV caching, and decoupled RoPE.
---
# Multi-Head Latent Attention (MLA) Roofline Estimation
Estimate the roofline performance of Multi-Head Latent Attention (MLA) by calculating FLOPS and memory transfer.
Report per-operation level FLOPS and memory transfer rather than for the attention layer as a whole.
For MLA, carefully distinguish between the **Prefill** (Causal) and **Decode** (Weight Absorption) phases, as their arithmetic intensity and memory footprints differ significantly.

## Requirements
You must read and follow:
*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)

## Workflow
1.  **Identify MLA Dimensions:** Determine the following dimensions from the notation:
*   `B_local`: Local Batch Size per device (`B / DP`)
*   `TP`: Tensor Parallelism degree
*   `H_q`: Total Number of Query heads
*   `H_q_local`: Local query heads per device — derived in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b
*   `H_kv`: Total Number of Key/Value heads (after up-projection)
*   `H_kv_local`: Local KV heads per device — derived in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b. **Under MLA the KV latent has no head dimension, so it is replicated across TP regardless**
*   `S`: Sequence Length (for Training and Prefill) or KV cache length (for Decode)
*   `D`: Hidden Dimension
*   `D_c`: KV compression latent dimension
*   `D_c_q`: Query compression latent dimension
*   `D_v`: Value head dimension
*   `D_q_nope`: Query NOPE (Non-Positional) head dimension
*   `D_q_rope`: Query RoPE (Positional) head dimension

2.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md).
*   Unless a global precision (e.g., `bf16`) is explicitly requested, matrix multiplication outputs are written to memory in the accumulation dtype (`float32` / 4 bytes per element) by default.
3.  **Identify the Phase:**
*   **Prefill:** Processes a sequence of length `S`. Keys/Values are typically computed directly without matrix associativity tricks because materializing full matrices is efficient for large `S`.
*   **Decode:** Processes 1 token (`S=1`) attending to past `S` tokens in the KV cache. Leverages **Matrix Associativity (so-called "Weight Absorption")** to avoid projecting KV latent cache `c_{kv}` into `K_c` and `V_c`.
*   **Training (Forward Pass):** Processes batches of sequences of length `S`. Identical to Standard Prefill, as full activations are required for the backward pass.

4.  **Write and Run Python Script for Calculations:** Do all math in Python.

5.  **Output Format:** Output a markdown table and JSON block as required.

## Calculations
**Note:** Assume FlashAttention is used by default. Therefore, the HBM read/write costs for intermediate attention maps (scores/logits) can be ignored. 

### 1. Prefill Phase (S tokens)
During Prefill, we process `S` tokens. We can compute it with or without weight
absorption (matrix associativity). You MUST check if the workload uses weight
absorption.
-   **DeepSeek-V3 prefill** uses **Flow B (With Weight Absorption)** because it
    is designed to use latent cache directly and defer V projection to save
    FLOPS/Memory.
-   Standard MLA without these optimizations uses **Flow A (Without Weight
    Absorption)**.

#### Flow A: Prefill Standard without Weight Absorption
Use this flow if Key and Value are projected to head dimensions before
attention.
*   **Query Latent Down-Projection ($c_q$; fully replicated):**
*   FLOPS: `2 * B_local * S * D * D_c_q`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * D_c_q * bytes_per_element`), Write $c_q$ (`B_local * S * D_c_q * bytes_per_element`)
*   **Query Up-Projection for both RoPE and NoPE ($Q_c$):**
*   FLOPS: `2 * B_local * S * D_c_q * H_q_local * (D_q_nope + D_q_rope)`
*   Memory: Read $c_q$ (`B_local * S * D_c_q * bytes_per_element`), Read Weight (`D_c_q * H_q_local * (D_q_nope + D_q_rope) * bytes_per_element`), Write $Q_c$ (`B_local * S * H_q_local * (D_q_nope + D_q_rope) * bytes_per_element`)
*   **KV Down-Projection ($c_{kv}$ and $K_r$ combined and fully replicated):**
*   FLOPS: `2 * B_local * S * D * (D_c + D_q_rope)`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * (D_c + D_q_rope) * bytes_per_element`), Write Output (`B_local * S * (D_c + D_q_rope) * bytes_per_element`)
*   **Key/Value Up-Projection ($K_c, V_c$):**
*   FLOPS: ` 2 * B_local * S * D_c * (D_q_nope + D_v) * H_kv_local`
*   Memory: Read $c_{kv}$ (`B_local * S * D_c * bytes_per_element`), Read Weight (` D_c * H_kv_local * (D_q_nope +D_v) * bytes_per_element`), Write $K_c, V_c$ (` B_local * S * H_kv_local * (D_q_nope + D_v) * bytes_per_element`)
*   **QK Phase:**
*   $Q = [Q_c, Q_r]$ shape: `[B_local, S, H_q_local, D_q_nope + D_q_rope]`
*   $K = [K_c, K_r]$ shape: `[B_local, S, H_kv_local, D_q_nope + D_q_rope]`
*   FLOPS: `2 * B_local * S^2 * H_q_local * (D_q_nope + D_q_rope)` (Note: Apply MXU padding if applicable).
*   Causal FLOPS: ` B_local * S^2 * H_q_local * (D_q_nope + D_q_rope)`
*   Memory Transfer: Read $Q$: ` B_local * S * H_q_local * (D_q_nope + D_q_rope) * bytes_per_element `
*   Memory Transfer: Read $K$: ` B_local * S * (H_kv_local *D_q_nope + D_q_rope) * bytes_per_element `
*   **AV Phase:**
*   $V_c$ shape: `[B_local, S, H_kv_local, D_v]`
*   Bidirectional FLOPS: `2 * B_local * S^2 * H_q_local * D_v`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * D_v`
*   Memory: Read Activation (`B_local * S * H_q_local * D_v * bytes_per_element`)
*   Memory: Read Weight (`H_q_local * D_v * D * bytes_per_element`)
*   Memory: Write Output When Output Is Not Sharded  (`B_local * S * D * bytes_per_element`)

*   **Output Projection:**
*   FLOPS: `2 * B_local * S * H_q_local * D_v * D`
*   Memory: Read Activation (`B_local * S * H_q_local * D_v * bytes_per_element`) 
*   Memory: Read Weight (`H_q_local * D_v * D * bytes_per_element`)
*   Memory: Write Output When Output Is Not Sharded  (`B_local * S * D * bytes_per_element`)

#### Flow B: Prefill with Weight Absorption (Matrix Associativity)
Use this flow if attention is computed on compressed latent states directly, and
projection is deferred (e.g., DeepSeek-V3). *Note: Key/Value Up-Projection
($K_c, V_c$) is NOT computed in this flow.*
*   **Query Latent Down-Projection ($c_q$; fully replicated):**
*   FLOPS: `2 * B_local * S * D * D_c_q`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * D_c_q * bytes_per_element`), Write $c_q$ (`B_local * S * D_c_q * bytes_per_element`)
*   **Query Up-Projection for both RoPE and NoPE ($Q_c$):**
*   FLOPS: `2 * B_local * S * D_c_q * H_q_local * (D_q_nope + D_q_rope)`
*   Memory: Read $c_q$ (`B_local * S * D_c_q * bytes_per_element`), Read Weight (`D_c_q * H_q_local * (D_q_nope + D_q_rope) * bytes_per_element`), Write $Q_c$ (`B_local * S * H_q_local *(D_q_nope + D_q_rope) * bytes_per_element`)
*   **KV Down-Projection ($c_{kv}$ and $K_r$ combined and fully replicated):**
*   FLOPS: `2 * B_local * S * D * (D_c + D_q_rope)`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * (D_c + D_q_rope) * bytes_per_element`), Write Output (`B_local * S * (D_c + D_q_rope) * bytes_per_element`)
*   **Q Proj to K (Equivalence for Q):**
*   Multiplies Query $Q_c$ with Key Up-Projection weight $W_{UK}$ to produce $Q_c'$ with shape `[B_local, S, H_q_local, D_c]`.
*   FLOPS: `2 * B_local * S * H_q_local * D_q_nope * D_c` (if applied dynamically) or `0` (if pre-absorbed).
*   Memory: Read $Q_c$ (`B_local * S * H_q_local * D_q_nope * bytes_per_element`), Read Weight $W_{UK}$ (`H_q_local * D_q_nope * D_c * bytes_per_element`), Write $Q_c'$ (`B_local*S*H_q_local*D_c* bytes_per_element`)
*   **QK Phase (nope) with flash attention:**
*   $Q_c'$ shape: `[B_local, S, H_q_local, D_c]`
*   $c_{kv}$ shape: `[B_local, S, D_c]`
*   FLOPS: `2 * B_local * S^2 * H_q_local * D_c`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * D_c`
*   Memory Transfer: Read $Q_c'$: `B_local * S * H_q_local * D_c * bytes_per_element`
*   Memory Transfer: Read $c_{kv}$: `B_local * S * D_c * bytes_per_element`
*   Memory Transfer: Write 0
*   **QK Phase (rope) with flash attention:**
*   $Q_r$ shape: `[B_local, S, H_q_local, D_q_rope]`
*   $K_r$ shape: `[B_local, S, D_q_rope]`
*   FLOPS: `2 * B_local * S^2 * H_q_local * D_q_rope`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * D_q_rope`
*   Memory Transfer: Read $Q_r$: `B_local * S * H_q_local * D_q_rope * bytes_per_element`
*   Memory Transfer: Read $K_r$: `B_local * S * D_q_rope * bytes_per_element`
*   Memory Transfer: Write 0
*   **AV Phase with full weight absorption and flash attention:**
*   Attention Scores shape: `[B_local, S, S, H_q_local]`
*   $c_{kv}$ shape: `[B_local, S, D_c]`
*   Bidirectional FLOPS: `2 * B_local * S^2 * H_q_local * D_c`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * D_c`
*   Memory Read $c_{kv}$: **0** -- the QK Phase (nope) above already fetched it.
    Weight absorption makes the latent *both* K and V, so the flash kernel reads it
    once and uses it twice. Charging it in both phases reads more KV than the layer's
    cache holds (136 MiB against 72 MiB on GLM-5.2 at S=128K).
*   Write Output: (`B_local * S * H_q_local * D_c * bytes_per_element`).
*   **V Up Proj (Deferred V Projection):**
*   FLOPS: `2 * B_local * S * D_c * H_q_local * D_v`
*   Memory: Read AV Output (`B_local * S * H_q_local * D_c * bytes_per_element`), 
*   Memory: Read Weight $W_{UV}$ (`D_c * H_q_local * D_v * bytes_per_element`)
*   Memory: Write Output (`B_local * S * H_q_local * D_v * bytes_per_element`)
*   **Output Projection:**
*   FLOPS: `2 * B_local * S * H_q_local * D_v * D`
*   Memory: Read Activation (`B_local * S * H_q_local * D_v * bytes_per_element`) 
*   Memory: Read Weight (`H_q_local * D_v * D * bytes_per_element`)
*   Memory: Write Output When Output Is Not Sharded  (`B_local * S * D * bytes_per_element`)

> [!IMPORTANT] **Flow B Output:** Report all 10 sub-operations (Projections, QK/AV phases) separately. Do not combine.

### 2. Decode Phase (Matrix Associativity / 'Weight Absorption', 1 token)
During decode (autoregressive generation), MLA leverages matrix associativity (often called "Weight Absorption") to reduce the cache size. The KV cache only stores $c_{kv}$ (size $D_c$) and $K_r$ (size $D_q_rope$) per token, drastically reducing memory bandwidth.

#### Generation (1 Token, fully fused)
*   **Query & KV Latents:**
*   Generate $c_q$: `2 * B_local * 1 * D * D_c_q` FLOPS.
*   Generate $Q_{nope}$: `2 * B_local * 1 * D_c_q * H_q_local * D_c` FLOPS.
*   Generate $Q_r$ (RoPE Query): `2 * B_local * 1 * D_c_q * H_q_local * D_q_rope` FLOPS.
*   Generate $c_{kv}$: `2 * B_local * 1 * D * D_c` FLOPS. (Written to cache).
*   Generate $K_r$ (RoPE Key): `2 * B_local * 1 * D * D_q_rope` FLOPS. (Written to cache)
*   Activation read: `B * D * bytes_per_element`
*   Weight read: `(D * D_c_q + D_c_q * H_q_local *D_c + D_c_q * H_q_local * D_q_rope + D * (D_c + D_q_rope))* bytes_per_element`
*   KV-cache write: `B * (D_c + D_q_rope) * bytes_per_element`


#### QK Phase
*   **Compute Scores:**
*   Read $c_{kv}$ Cache: `B_local * S * D_c * bytes_per_element`
*   Read $K_r$ Cache: `B_local * S * D_q_rope * bytes_per_element`
*   Write Memory (flash attention): `0` 
*   Latent QK FLOPS (Nope): `2 * B_local * 1 * S * H_q_local * D_c`
*   RoPE QK FLOPS: `2 * B_local * 1 * S * H_q_local * D_q_rope`
*   Total QK FLOPS: `2 * B_local * S * H_q_local * (D_c + D_q_rope)`

#### AV Phase
*   **Compute Absorbed Output:**
*   Note: $c_{kv}$ is broadcasted across $H_q_local$ heads.
*   FLOPS (AV + Up Projection): `2 * B_local * S * H_q_local * D_c + 2 * B_local * S * H_q_local * D_v * D_c`
*   Memory: Read $c_{kv}$: **0** -- the QK Phase already fetched the same cache
    entries. See the prefill AV Phase note.
*   Memory: Write Output (fused with output projection): 0
*   Output shape: `[B_local, 1, H_q_local, D_v]`


#### Output Projection
*   **Absorbed Output Proj:**
*   FLOPS: `2 * B_local * 1 * (H_q_local * D_v) * D`
*   Memory Transfer: Read $W_O'$ (`H_q_local * D_v * D * bytes_per_element`)
*   Memory Transfer: Write $O_{proj}$ (`B_local * 1 * D * bytes_per_element`)

### 3. Training Phase (Forward Pass)
During the training forward pass, the model processes batches of sequences of length `S`. The computation is mathematically identical to the Prefill Standard phase without weight absorption (Flow A). Weight absorption is an inference-time optimization to reduce KV cache size, which is not applicable during training since full intermediate activations are required for the backward pass computations.

*   **Query Latent Down-Projection ($c_q$ and fully replicated):** 
*   FLOPS: `2 * B_local * S * D * D_c_q`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * D_c_q * bytes_per_element`), Write $c_q$ (`B_local * S * D_c_q * bytes_per_element`)
*   **Query Up-Projection for both RoPE and NoPE ($Q_c$):**
*   FLOPS: `2 * B_local * S * D_c_q * H_q_local * ( D_q_nope + D_q_rope )`
*   Memory: Read $c_q$ (`B_local * S * D_c_q * bytes_per_element`), Read Weight (`D_c_q * H_q_local * (D_q_nope+D_q_rope) * bytes_per_element`), Write $Q_c$ (`B_local * S* H_q_local* ( D_q_nope + D_q_rope ) * bytes_per_elements`)
*   **KV Down-Projection ($c_{kv}$ and $K_r$ combined and fully replicated):**
*   FLOPS: `2 * B_local * S * D * (D_c + D_q_rope)`
*   Memory: Read X (`B_local * S * D * bytes_per_element`), Read Weight (`D * (D_c + D_q_rope) * bytes_per_element`), Write Output (`B_local * S * (D_c + D_q_rope) * bytes_per_element`)
*   **Key/Value Up-Projection ($K_c, V_c$):**
*   FLOPS: `2 * B_local * S * D_c * H_kv_local * (D_q_nope + D_v)`
*   Memory: Read $c_{kv}$ (`B_local * S * D_c * bytes_per_element`), Read Weight (`D_c * H_kv_local * (D_q_nope + D_v) * bytes_per_element`), Write $K_c, V_c$ (`B_local * S * H_kv_local * (D_q_nope + D_v) * bytes_per_element`)
*   **QK Phase (flash attention):**
*   $Q = [Q_c, Q_r]$ shape: `[B_local, S, H_q_local, D_q_nope + D_q_rope]`
*   $K = [K_c, K_r]$ shape: `[B_local, S, H_kv_local, D_q_nope + D_q_rope]`
*   Bidirectional FLOPS: `2 * B_local * S^2 * H_q_local * (D_q_nope + D_q_rope)`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * (D_q_nope + D_q_rope)` 
*   Memory: Read $Q$ ` B_local * S * H_q_local * (D_q_nope + D_q_rope) * bytes_per_element `
    Memory: Read $K$ ` B_local * S * (H_kv_local *D_q_nope + D_q_rope) * bytes_per_element `
    Memory: Write 0
*   **AV Phase:**
*   $V_c$ shape: `[B_local, S, H_kv_local, D_v]`
*   Bidirectional FLOPS: `2 * B_local * S^2 * H_q_local * D_v`
*   Causal FLOPS: ` B_local * S^2 * H_q_local * D_v`
*   Memory: Read $V_c$: ` B_local * S * H_kv_local * D_v * bytes_per_element `
*   Write Output (`B_local * S * H_kv_local * D_v`).
*   **Output Projection:**
*   FLOPS: `2 * B_local * S * H_q_local * D_v * D`
*   Memory: Read Activation (`B_local * S * H_q_local * D_v * bytes_per_element`) 
*   Memory: Read Weight (`H_q_local * D_v * D * bytes_per_element`)
*   Memory: Write Output When Output Is Not Sharded  (`B_local * S * D * bytes_per_element`)

## Tensor Parallelism (TP)
The formulas above are already per-device: substitute `H_q_local` and `H_kv_local`, derived in
[`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b.

**MLA-specific:** the latent cache ($c_{kv}$ and $K_r$) is broadcast to all local query heads
rather than sharded, so **TP divides MLA's KV bytes per token by nothing**. The output
projection's partial sums need a TP reduction, priced in the same file.

## References
*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Type Promotion](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Collective Cost](../../../../COLLECTIVE.md)
