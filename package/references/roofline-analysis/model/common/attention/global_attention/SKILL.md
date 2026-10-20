---
name: global-attention-roofline
description: >-
  Calculates FLOPS and memory transfer for Global Attention in Transformer layers.
  Use when estimating Global Attention performance, analyzing weight/activation shapes, or calculating FLOPs/bytes for MHA, MQA, and GQA variants under Tensor Parallelism. Don't use for QKV or Output projections, FFN layers, or MLA (Multi-Head Latent Attention) layers (use mla-roofline instead).
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/NOTATION.md
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/parallelism/tp/SKILL.md
- references/roofline-analysis/tpu/V7X.md
---

# Global Attention Roofline Estimation

Estimate the roofline performance of Global Attention by calculating FLOPS and memory transfer.
Report per-operation level FLOPS and memory transfer rather than for the attention layer as a whole, because roofline bottlenecks depend on the specific tensor shapes of each sub-operation.

## Requirements

You must read and follow:

*   [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}})
*   [Type Promotion Rules]({{reference:roofline-analysis/TYPE_PROMOTION.md}})

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
3.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md]({{reference:roofline-analysis/TYPE_PROMOTION.md}}).
    *   Unless a global precision (e.g., `bf16`) is explicitly requested, matrix multiplication outputs are written to memory in the accumulation dtype (`float32` / 4 bytes per element) by default.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory bytes, latencies, collective times). Do not perform roofline arithmetic mentally or in the chat because large arithmetic calculations are prone to hallucination.
5.  **Calculate Roofline & Total Latency:** Combine these values with hardware specs to estimate execution time for each sub-operation. **CRITICAL:** The roofline metrics (memory time, compute time, and bottleneck latency) for the QK phase and the AV phase must be calculated separately.
6.  **Output Format:** Present the results in a clearly formatted markdown table and a JSON block using standard units (GFLOPS, MB, ms) as described in [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}).

## Calculations

Assuming FlashAttention-like fusion, the two primary matrix multiplications (QK^T and Score * V) are fused. Intermediate score and probability matrices (logits map) are NOT materialized to memory. For roofline estimation, the total time is often modeled as the sum of two phases: the QK phase and the AV phase.

Note: The FLOPS calculations depend on whether the attention mask is bidirectional (all tokens attend to all tokens) or causal (tokens only attend to past tokens, typically used in Prefill). For causal attention, the total operations are exactly halved because the attention mask forms a lower triangular matrix.

### Unsharded Global Attention (Fused)

#### 1. QK Phase

*   **Query Shape:** `[B_local, S, H_q, D_h]`
*   **Key Shape:** `[B_local, S, H_kv, D_h]`
*   **QK phase Bidirectional FLOPS:** `2 * B_local * S^2 * H_q * D_h`
*   **QK phase Causal (Prefill) FLOPS:** `B_local * S^2 * H_q * D_h`
*   **QK phase Query Read:** `B_local * S * H_q * D_h * bytes_per_element`
*   **QK phase Key Read:** `B_local * S * H_kv * D_h * bytes_per_element`

#### 2. AV Phase

*   **Value Shape:** `[B_local, S, H_kv, D_h]`
*   **Output Shape:** `[B_local, S, H_q, D_h]`
*   **AV phase Bidirectional FLOPS:** `2 * B_local * S^2 * H_q * D_h`
*   **AV phase Causal (Prefill) FLOPS:** `B_local * S^2 * H_q * D_h`
*   **AV phase Value Read:** `B_local * S * H_kv * D_h * bytes_per_element`
*   **AV phase Output Write:** `B_local * S * H_q * D_h * bytes_per_element`

### Value Cache Transfer Overlapping (Optional)

**Only apply this rule if the user explicitly specifies that value cache transfer can be overlapped with the QK operation.**

If specified:

*   The `Value Read` memory transfer in the AV Phase is assumed to be fully overlapped with the QK phase execution.
*   Therefore, the effective `Value Read` bytes for the AV Phase calculation are treated as `0` for bottleneck analysis.

## Parallelism Strategies

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and the per-device symbols used below, are defined in [`../../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) §3b.
> This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

Under standard Tensor Parallelism, attention heads are divided across devices. The sequence length `S` remains unsharded.

#### 1. QK Phase Sharding

*   **QK phsae Bidirectional FLOPS:** `2 * B_local * S^2 * H_q_local * D_h`
*   **QK phsae Causal (Prefill) FLOPS:** `B_local * S^2 * H_q_local * D_h`
*   **QK phsae Query Read:** `B_local * S * H_q_local * D_h * bytes_per_element`
*   **QK phsae Key Read:** `B_local * S * H_kv_local * D_h * bytes_per_element`
*   **QK phsae Communication:** None.

#### 2. AV Phase Sharding

*   **Output Shape:** `[B_local, S, H_q_local, D_h]`
*   **AV phase Bidirectional FLOPS:** `2 * B_local * S^2 * H_q_local * D_h`
*   **AV phase Causal (Prefill) FLOPS:** `B_local * S^2 * H_q_local * D_h`
*   **AV phase Value Read:** `B_local * S * H_kv_local * D_h *
    bytes_per_element`
*   **AV phase Output Write:** `B_local * S * H_q_local * D_h *
    bytes_per_element`
*   **Communication:** None (Assuming the output projection matrix handles the all-reduce).

## References

*   [TPU Specifications]({{reference:roofline-analysis/tpu/V7X.md}})
*   [Type Promotion]({{reference:roofline-analysis/TYPE_PROMOTION.md}})
*   [Tensor Dimension Notation]({{reference:roofline-analysis/NOTATION.md}})
*   [Collective Cost]({{reference:roofline-analysis/COLLECTIVE.md}})
