---
name: qkv-proj-roofline
description: >-
  Calculates FLOPS and memory transfer for Query, Key, and Value (QKV) projections in Transformer attention layers.
  Use when estimating QKV projection performance, analyzing weight/activation shapes, or calculating FLOPs/bytes for MHA, MQA, and GQA variants under Tensor Parallelism. Don't use for Attention (QK/AV) operations or FFN layers.
---

# QKV Projection Roofline Estimation

Estimate the roofline performance of Query, Key, and Value (QKV) Projections by calculating FLOPS and memory transfer.
Report per-operation level FLOPS and memory transfer rather than for the attention layer as a whole, because roofline bottlenecks depend on the specific tensor shapes of each sub-operation.

## Requirements

You must read and follow:

*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)

## Workflow

1.  **Identify the Attention Variant:** Determine if the workload uses Multi-Head Attention (MHA), Multi-Query Attention (MQA), or Grouped-Query Attention (GQA). This defines `H_q` and `H_kv`.
2.  **Determine Dimensions:** Identify the following dimensions from the notation:
    *   `B`: Global Batch Size
    *   `B_local`: Local Batch Size (per device)
    *   `S`: Sequence Length
    *   `D`: Hidden Dimension
    *   `H_q`: Number of Query heads
    *   `H_kv`: Number of Key/Value heads
    *   `D_h`: Head dimension (typically `D / H_q`)
3.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md).
    *   Unless a global precision (e.g., `bf16`) is explicitly requested, matrix multiplication outputs are written to memory in the accumulation dtype (`float32` / 4 bytes per element) by default.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory bytes, latencies, collective times). Do not perform roofline arithmetic mentally or in the chat because large arithmetic calculations are prone to hallucination.
5.  **Calculate Roofline & Total Latency:** Combine these values with hardware specs to estimate execution time for each sub-operation.
6.  **Output Format:** Present the results in a clearly formatted markdown table and a JSON block using standard units (GFLOPS, MB, ms) as described in [Common Requirements](../../REQUIREMENTs.md).

## Calculations

### Unsharded QKV Projections (`X @ W_qkv`)

For an unsharded projection, the operations are standard Linear transformations.
In many implementations, Q, K, and V projections can be computed separately or packed into a single combined matrix multiplication. 

*   **Weight Shape:** `[D, (H_q + 2 * H_kv) * D_h ]`
*   **Activation Shape:** Input `[B_local, S, D]` -> Output `[B_local, S, (H_q +
    2 * H_kv) * D_h]`
*   **Q Projection FLOPS:** `2 * B_local * S * D * (H_q * D_h)`
*   **K Projection FLOPS:** `2 * B_local * S * D * (H_kv * D_h)`
*   **V Projection FLOPS:** `2 * B_local * S * D * (H_kv * D_h)`
*   **Activation Read (QKV projection operations are fused):** `B_local * S *
    D * bytes_per_element`
*   **Query Weight Read:** `D * (H_q * D_h) * bytes_per_element`
*   **Key Weight Read:** `D * (H_kv * D_h) * bytes_per_element`
*   **Value Weight Read:** `D * (H_kv * D_h) * bytes_per_element`
*   **Query Activation Write:** `B_local * S * (H_q * D_h) * bytes_per_element`
*   **Key Activation Write:** `B_local * S * (H_kv * D_h) * bytes_per_element`
*   **Value Activation Write:** `B_local * S * (H_kv * D_h) * bytes_per_element`

## Parallelism Strategies

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and the per-device symbols used below, are defined in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b.
> This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

In a fused implementation the packed Q, K and V matrices are sharded together as a single linear
layer, so the per-device output width is `(H_q_local + 2 * H_kv_local) * D_h`.

#### Fused QKV Projection Sharding

*   **Weight Shape:** `[D, (H_q_local + 2 * H_kv_local) * D_h]`
*   **Activation Shape:** Input `[B_local, S, D]` -> Output `[B_local, S,
    (H_q_local + 2 * H_kv_local) * D_h]`
*   **Q Projection FLOPS:** `2 * B_local * S * D * (H_q_local * D_h)`
*   **K Projection FLOPS:** `2 * B_local * S * D * (H_kv_local * D_h)`
*   **V Projection FLOPS:** `2 * B_local * S * D * (H_kv_local * D_h)`
*   **Activation Read (QKV projection operations are fused):** `B_local * S *
    D * bytes_per_element`
*   **Query Weight Read:** `D * (H_q_local * D_h) * bytes_per_element`
*   **Key Weight Read:** `D * (H_kv_local * D_h) * bytes_per_element`
*   **Value Weight Read:** `D * (H_kv_local * D_h) * bytes_per_element`
*   **Query Activation Write:** `B_local * S * H_q_local * D_h *
    bytes_per_element`
*   **Key Activation Write:** `B_local * S * H_kv_local * D_h *
    bytes_per_element`
*   **Value Activation Write:** `B_local * S * H_kv_local * D_h *
    bytes_per_element`

*   **Communication:** None.

## References

*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Type Promotion](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Collective Cost](../../../../COLLECTIVE.md)
