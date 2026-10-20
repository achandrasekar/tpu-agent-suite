---
name: local-attention-roofline
description: >-
  Calculates FLOPS and memory transfer for Local Attention (Sliding Window Attention) in Transformer layers.
  Use when estimating Local Attention performance, analyzing weight/activation shapes, or calculating FLOPs/bytes for MHA, MQA, and GQA variants under Tensor Parallelism. Don't use for Global Attention, QKV/Out projections, or FFN layers.
---

# Local Attention Roofline Estimation

Estimate the roofline performance of Local Attention (also known as Sliding Window Attention) by calculating FLOPS and memory transfer.
Report per-operation level FLOPS and memory transfer rather than for the attention layer as a whole, because roofline bottlenecks depend on the specific tensor shapes of each sub-operation.

## Requirements

You must read and follow:

*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)

## Workflow

1.  **Identify the Attention Variant:** Determine if the workload uses Multi-Head Attention (MHA), Multi-Query Attention (MQA), or Grouped-Query Attention (GQA). This defines `H_q` and `H_kv`.
2.  **Determine Dimensions:** Identify the following dimensions from the notation:
    *   `B`: Batch Size
    *   `S`: Sequence Length
    *   `W`: Window Size (local attention context size)
    *   `D`: Hidden Dimension
    *   `H_q`: Number of Query heads
    *   `H_kv`: Number of Key/Value heads
    *   `D_h`: Head dimension (typically `D / H_q`)
3.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md).
    *   Unless a global precision (e.g., `bf16`) is explicitly requested, matrix multiplication outputs are written to memory in the accumulation dtype (`float32` / 4 bytes per element) by default.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory bytes, latencies, collective times). Do not perform roofline arithmetic mentally or in the chat because large arithmetic calculations are prone to hallucination.
5.  **Calculate Roofline & Total Latency:** Combine these values with hardware specs to estimate execution time for each sub-operation. **CRITICAL:** The roofline metrics (memory time, compute time, and bottleneck latency) for the QK phase and the AV phase must be calculated separately.
6.  **Output Format:** Present the results in a clearly formatted markdown table and a JSON block using standard units (GFLOPS, MB, ms) as described in [Common Requirements](../../REQUIREMENTs.md).

## Calculations

Assuming FlashAttention-like fusion, the two primary matrix multiplications (QK^T and Score * V) are fused. Intermediate score and probability matrices (logits map) are NOT materialized to memory. For roofline estimation, the total time is often modeled as the sum of two phases: the QK phase and the AV phase.

Note: The FLOPS calculations depend on whether the attention mask is bidirectional or causal (typically used in Prefill). For causal sliding window attention, the operations are reduced because the mask clips the backward-looking window for early tokens. If the window size `W` covers the entire sequence (`W >= S`), causal FLOPS are exactly halved (`B * S^2 * H_q * D_h`). For general `W`, the causal FLOPS are `2 * B * (S * W - W^2 / 2) * H_q * D_h`.

### Unsharded Local Attention (Fused)

#### 1. QK Phase

*   **Query Shape:** `[B, S, H_q, D_h]`
*   **Key Shape:** `[B, S, H_kv, D_h]`
*   **QK phase Bidirectional FLOPS:** `2 * B * S * W * H_q * D_h`
*   **QK phase Causal (Prefill) FLOPS:** `2 * B * (S * W - W^2 / 2) * H_q * D_h`
*   **QK phase Query Read:** `B * S * H_q * D_h * bytes_per_element`
*   **QK phase Key Read (ideal reuse model):** `B * S * H_kv * D_h *
    bytes_per_element` (assuming perfect caching across sliding windows)

#### 2. AV Phase

*   **Value Shape:** `[B, S, H_kv, D_h]`
*   **Output Shape:** `[B, S, H_q, D_h]`
*   **AV phase Bidirectional FLOPS:** `2 * B * S * W * H_q * D_h`
*   **AV phase Causal (Prefill) FLOPS:** `2 * B * (S * W - W^2 / 2) * H_q * D_h`
*   **AV phase Value Read (ideal reuse model):** `B * S * H_kv * D_h *
    bytes_per_element` (assuming perfect caching across sliding windows)
*   **AV phase Output Write:** `B * S * H_q * D_h * bytes_per_element`

### Value Cache Transfer Overlapping (Optional)

**Only apply this rule if the user explicitly specifies that value cache transfer can be overlapped with the QK operation.**

If specified:

*   The `Value Read` memory transfer in the AV Phase is assumed to be fully overlapped with the QK phase execution.
*   Therefore, the effective `Value Read` bytes for the AV Phase calculation are treated as `0` for bottleneck analysis.

## Parallelism Strategies

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and the per-device symbols used below, are defined in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b.
> This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

Under standard Tensor Parallelism, attention heads are divided across devices. The sequence length `S` and window size `W` remain unsharded.

#### 1. QK Phase Sharding

*   **QK phase Bidirectional FLOPS:** `2 * B * S * W * H_q_local * D_h`
*   **QK phase Causal (Prefill) FLOPS:** `2 * B * (S * W - W^2 / 2) *
    H_q_local * D_h`
*   **QK phase Query Read:** `B * S * H_q_local * D_h * bytes_per_element`
*   **QK phase Key Read (ideal reuse model):** `B * S * H_kv_local * D_h *
    bytes_per_element`
*   **Communication:** None.

#### 2. AV Phase Sharding

*   **Output Shape:** `[B, S, H_q_local, D_h]`
*   **AV phase Bidirectional FLOPS:** `2 * B * S * W * H_q_local * D_h`
*   **AV phase Causal (Prefill) FLOPS:** `2 * B * (S * W - W^2 / 2) *
    H_q_local * D_h`
*   **AV phase Value Read (ideal reuse model):** `B * S * H_kv_local * D_h *
    bytes_per_element`
*   **AV phase Output Write:** `B * S * H_q_local * D_h * bytes_per_element`
*   **Communication:** None (Assuming the output projection matrix handles the all-reduce).

## References

*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Type Promotion](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Collective Cost](../../../../COLLECTIVE.md)
