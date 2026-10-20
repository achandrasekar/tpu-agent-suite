---
name: out-proj-roofline
description: >-
  Calculates FLOPS and memory transfer for the Output Projection in Transformer attention layers.
  Use when estimating Output Projection performance, analyzing weight/activation shapes, or calculating FLOPs/bytes/communication for attention output under Tensor Parallelism. Don't use for Attention (QK/AV) operations or FFN layers.
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/NOTATION.md
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/parallelism/tp/SKILL.md
- references/roofline-analysis/tpu/V7X.md
---

# Output Projection Roofline Estimation

Estimate the roofline performance of the Output Projection by calculating FLOPS, memory transfer, and communication cost.
Report per-operation level FLOPS and memory transfer rather than for the attention layer as a whole, because roofline bottlenecks depend on the specific tensor shapes of each sub-operation.

## Requirements

You must read and follow:

*   [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}})
*   [Type Promotion Rules]({{reference:roofline-analysis/TYPE_PROMOTION.md}})

## Workflow

1.  **Identify the Attention Variant:** Determine if the workload uses Multi-Head Attention (MHA), Multi-Query Attention (MQA), or Grouped-Query Attention (GQA). This affects `H_q`.
2.  **Determine Dimensions:** Identify the following dimensions from the notation:
    *   `B`: Global Batch Size
    *   `B_local`: Local Batch Size (per device)
    *   `S`: Sequence Length
    *   `D`: Hidden Dimension
    *   `H_q`: Number of Query heads
    *   `D_h`: Head dimension (typically `D / H_q`)
3.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md]({{reference:roofline-analysis/TYPE_PROMOTION.md}}).
    *   Unless a global precision (e.g., `bf16`) is explicitly requested, matrix multiplication outputs are written to memory in the accumulation dtype (`float32` / 4 bytes per element) by default.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory bytes, latencies, collective times). Do not perform roofline arithmetic mentally or in the chat because large arithmetic calculations are prone to hallucination.
5.  **Calculate Roofline & Total Latency:** Combine these values with hardware specs to estimate execution time for each sub-operation.
6.  **Output Format:** Present the results in a clearly formatted markdown table and a JSON block using standard units (GFLOPS, MB, ms) as described in [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}).

## Calculations

### Unsharded Output Projection

For an unsharded projection, the operation is a standard Linear transformation mapping the concatenated attention heads back to the hidden dimension.

#### 1. Output Projection (`X @ W_o`)

*   **Weight Shape:** `[H_q * D_h, D]`
*   **Activation Shape:** Input `[B_local, S, H_q * D_h]` -> Output `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (H_q * D_h) * D`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * (H_q * D_h) * bytes_per_element`
    *   Weight Read: `(H_q * D_h) * D * bytes_per_element`
    *   Activation Write: `B_local * S * D * bytes_per_element`

## Parallelism Strategies

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and the per-device symbols used below, are defined in [`../../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) §3b.
> This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

The input activation arrives already partitioned across heads from the preceding attention phase; the projection consumes that partition and emits a partial sum.

#### Output Projection Sharding

*   The input activation is already partitioned because the preceding attention phase computes local query heads.
*   **Weight Shape:** `[H_q_local * D_h, D]`
*   **Local Input Activation Shape:** `[B_local, S, H_q_local * D_h]`
*   **Local Output Shape (Before All-Reduce):** `[B_local, S, D]`
*   **Final Output Shape (After All-Reduce):** `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (H_q_local * D_h) * D`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * (H_q_local * D_h) * bytes_per_element`
    *   Weight Read: `(H_q_local * D_h) * D * bytes_per_element`
    *   Activation Write (Partial sum): `B_local * S * D * bytes_per_element`
*   **Communication:** The partial sums require a reduction across the TP group; it is priced in
    [`../../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}), not here.

## References

*   [TPU Specifications]({{reference:roofline-analysis/tpu/V7X.md}})
*   [Type Promotion]({{reference:roofline-analysis/TYPE_PROMOTION.md}})
*   [Tensor Dimension Notation]({{reference:roofline-analysis/NOTATION.md}})
*   [Collective Cost]({{reference:roofline-analysis/COLLECTIVE.md}})
