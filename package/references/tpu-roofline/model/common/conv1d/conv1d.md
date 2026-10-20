---
name: rooflining-conv1d
description: Calculates FLOPS and memory transfer for Causal Depthwise 1D Convolution layers in Transformer or Recurrent sequence models. Use when estimating Conv1D performance, analyzing weight/activation shapes, or calculating FLOPs/bytes for causal 1D depthwise convolutions under Tensor Parallelism. Don't use for Attention (QK/AV) operations or standard 2D convolutions.
---

# Conv1D Roofline Estimation

Estimate the roofline performance of Causal Depthwise 1D Convolution layers by calculating FLOPS and memory transfer.
Report per-operation level FLOPS and memory transfer rather than for the block as a whole, because roofline bottlenecks depend on the specific tensor shapes, precision, and fusion configurations of each sub-operation.

## Requirements

You must read and follow:
*   [Common Requirements](../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../TYPE_PROMOTION.md)

## Workflow

1.  **Identify the Execution Regime:** Determine if the workload is in **Prefill Mode** (multi-token sequence forward) or **Decode Mode** (single-token autoregressive decoding).
2.  **Determine Dimensions:** Identify the following dimensions from the notation:
    *   `B`: Batch Size
    *   `S`: Sequence Length (input sequence length `S` for Prefill, or sequence context length `S` for Decode)
    *   `D_conv`: Convolution channel dimension (e.g., `key_dim * 2 + value_dim` in Qwen3.5 Gated DeltaNet)
    *   `K_conv`: Convolution kernel size (typically `k = 4`)
3.  **Determine Data Types & Apply Type Promotion:** Read [TYPE_PROMOTION.md](../../../TYPE_PROMOTION.md).
    *   Identify precision for activations, weights, and the convolution state (cache) if applicable.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory bytes, latencies). Do not perform roofline arithmetic mentally or in the chat because large arithmetic calculations are prone to hallucination.
5.  **Calculate Roofline & Total Latency:** Combine these values with hardware specs to estimate execution time.
6.  **Output Format:** Present the results in a clearly formatted markdown table and a JSON block using standard units (GFLOPS, MB, ms) as described in [Common Requirements](../REQUIREMENTs.md).


## Calculations

### 1. Prefill Mode (Multi-Token Sequence Forward)

In Prefill mode, the convolution is computed over the entire input sequence. Causal masking is implemented via padding or triangular constraint, but all elements in `S` are computed.

*   **Input Activation Shape:** `[B, S, D_conv]`
*   **Weight Shape:** `[D_conv, 1, K_conv]` (Depthwise)
*   **Output Activation Shape:** `[B, S, D_conv]`
*   **FLOPS:** `2 * B * S * D_conv * K_conv`
*   **Memory Transfer (Bytes):**
    *   *Standalone/Unfused*:
        *   Activation Read: `B * S * D_conv * bytes_per_element`
        *   Weight Read: `D_conv * K_conv * bytes_per_element` (often negligible)
        *   Activation Write: `B * S * D_conv * bytes_per_element`
    *   *Fused (e.g., with QKV Projection and SiLU)*:
        *   If the input activation is computed in-place in VMEM (reusing the output of QKV projection) and/or the output is consumed in-place by a subsequent activation (like SiLU), the intermediate HBM reads/writes are avoided.
        *   In a fully fused QKV Projection + Conv1D + SiLU block:
            *   Read input `x` of QKV Projection: `B * S * D_input * bytes_per_element`
            *   Read weight `W_qkv` of QKV Projection: `D_input * D_conv * bytes_per_element`
            *   Write final output of SiLU: `B * S * D_conv * bytes_per_element`
            *   *Total HBM Transfer:* Sum of the above three (intermediate QKV projection output and Conv1D output are kept in VMEM, transferring 0 bytes to HBM).

### 2. Decode Mode (Single-Token Autoregressive Generation)

In Decode mode, the sequence length `S` is 1 (representing the current token). Conv1D reads the current token and the cached history of the past `K_conv - 1` tokens to compute the output, then updates the cache.

*   **Input Token Shape:** `[B, 1, D_conv]`
*   **Weight Shape:** `[D_conv, 1, K_conv]`
*   **Conv State (Cache) Shape:** `[B, K_conv - 1, D_conv]`
*   **Output Token Shape:** `[B, 1, D_conv]`
*   **FLOPS:** `2 * B * D_conv * K_conv`
*   **Memory Transfer (Bytes):**
    *   *State Transferred to/from HBM*:
        *   Input Read: `B * 1 * D_conv * bytes_per_element`
        *   State Read (History): `B * (K_conv - 1) * D_conv * bytes_per_element`
        *   Weight Read: `D_conv * K_conv * bytes_per_element` (negligible)
        *   Output Write: `B * 1 * D_conv * bytes_per_element`
        *   State Write (Updated History): `B * (K_conv - 1) * D_conv * bytes_per_element`
        *   *Total HBM Transfer:* `2 * B * D_conv * K_conv * bytes_per_element`
    *   *VMEM-Cached State*:
        *   If the `conv_state` size (`B * (K_conv - 1) * D_conv * bytes_per_element`) is small enough to fit within local SRAM/VMEM across steps, HBM transfers for reading and writing the state cache can be avoided (effectively treating the state transfer bytes as `0`).
        *   *Total HBM Transfer:* Only input read and output write (`2 * B * D_conv * bytes_per_element`).


## Parallelism Strategies

> **Partitioning is not defined here.** Which projections are column- or row-parallel,
> and the per-device symbols used below, are defined in [`../../../parallelism/tp/tp.md`](../../../parallelism/tp/tp.md) §3b.
> This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)


Because the convolution is depthwise (`groups = D_conv`), the operations for each channel are computed completely independently. Each device computes its sharded channels locally using local inputs, weights, and states.

*   **Local Input Token Shape:** `[B, 1, D_conv_local]`
*   **Local Weight Shape:** `[D_conv_local, 1, K_conv]`
*   **Local Conv State Shape:** `[B, K_conv - 1, D_conv_local]`
*   **Local Output Shape:** `[B, 1, D_conv_local]`
*   **FLOPS per device (Prefill):** `2 * B_local * S * D_conv_local * K_conv`
*   **FLOPS per device (Decode):** `2 * B_local * D_conv_local * K_conv`
*   **HBM Transfer per device (Decode, VMEM-Cached):** `2 * B_local * D_conv_local * bytes_per_element`
*   **Communication:** **None.** No All-Reduce, All-Gather, or other interconnect communications are required.


## References

*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Type Promotion](../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../NOTATION.md)
*   [Collective Cost](../../../COLLECTIVE.md)
