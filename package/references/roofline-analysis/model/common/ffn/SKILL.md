---
name: ffn-roofline
description: >-
  Calculates FLOPS and memory transfer for Feed-Forward Network (FFN) layers in Transformer models.
  Use when estimating FFN performance, analyzing weight/activation shapes, or calculating FLOPs/bytes for standard or gated FFN variants. Don't use for attention layers or collective operations.
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/NOTATION.md
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/parallelism/tp/SKILL.md
- references/roofline-analysis/tpu/V7X.md
---

# Feed-Forward Network (FFN) Roofline Estimation

Estimate the roofline performance of Feed-Forward Network (FFN) layers by
calculating FLOPS and memory transfer. The TP reduction this layer triggers is
priced in [`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}).

## Requirements

You must read and follow:

*   [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}})
*   [Type Promotion Rules]({{reference:roofline-analysis/TYPE_PROMOTION.md}})


## Workflow

1.  **Identify the FFN Variant:** Determine if the workload uses a Standard FFN or a Gated FFN (e.g., GLU, SwiGLU).
2.  **Determine Dimensions:** Identify tensor dimensions:
    *   `B_local`: Local Batch Size (per device)
    *   `S`: Sequence Length
    *   `D`: Hidden Dimension (Input/Output features)
    *   `F`: Intermediate Dimension (Filter/Feed-forward features)
    *   `F_local`: Local intermediate dimension — derived in [`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) §3b
3.  **Determine Data Types & Byte Multipliers:** Read
    [TYPE_PROMOTION.md]({{reference:roofline-analysis/TYPE_PROMOTION.md}}):
    *   `bytes_weight`: Precision of weight matrices (e.g., `bf16` = 2 bytes)
    *   `bytes_act`: Precision of activation inputs/reads (e.g., `bf16` = 2
        bytes)
    *   `bytes_acc`: Precision of matmul outputs/accumulation writes
        (`float32` = 4 bytes by default)
    *   `bytes_comm`: Precision of TP collective operations (`bf16` = 2 bytes by
        default). Passed through to the TP collective; the dtype site that owns
        it is `tp_allreduce` in `parallelism/tp/`.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python
    script to calculate all numerical values (FLOPS, bytes, latencies,
    collective times). Do not perform arithmetic mentally.
5.  **Calculate Roofline & Total Latency:** Combine sub-operation FLOPS, memory
    transfer, and collective times with hardware specifications to calculate
    total FFN layer latency.
6.  **Output Format:** Present results in a markdown table and JSON block using
    standard units (GFLOPS, MB, ms) per
    [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}).

--------------------------------------------------------------------------------

## Calculations (Unsharded / Single-Device)

### Standard FFN

*   **Flow:** `Output = Activation(X @ W_1) @ W_2`
*   **Shapes:** `W_1`: `[D, F]`, `W_2`: `[F, D]`, `Activation`: `[B_local, S,
    F]`, `Output`: `[B_local, S, D]`
*   **FLOPS (Fused Matmuls):** `4 * B_local * S * D * F`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_act`
    *   Weight Read: `2 * D * F * bytes_weight`
    *   Activation Write: `B_local * S * D * bytes_acc`

---

### Gated FFN (e.g., GLU, SwiGLU)

*   **Flow:** `Output = (Activation(X @ W_gate) * (X @ W_up)) @ W_down`
*   **Shapes:** `W_gate`, `W_up`: `[D, F]`, `W_down`: `[F, D]`, `Output`:
    `[B_local, S, D]`
*   **FLOPS:** `6 * B_local * S * D * F`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_act`
    *   Weight Read: `3 * D * F * bytes_weight`
    *   Activation Write: `B_local * S * D * bytes_acc`
*   **The `F`-wide intermediate, which the line above does not cover.** `W_gate` and
    `W_up` produce two `[B_local, S, F]` tensors and the activation collapses them to
    one. Whether they reach HBM is the fused question, and it is worth 2x the
    intermediate plus a whole pass:
    *   *fused epilogue (assume this):* write `B_local * S * F * bytes_acc` once. No
        separate activation op.
    *   *unfused:* write `2 * B_local * S * F * bytes_acc`, then a SwiGLU op reads
        `2 *` and writes `1 *` that again.
    Charging the unfused write **and** a separate SwiGLU double-counts; charging the
    fused write and a separate SwiGLU counts a pass over data nobody stored.

--------------------------------------------------------------------------------

## Parallelism Strategies (1D Tensor Parallelism)

> **Partitioning is not defined here.** How TP shards this op, and the per-device symbols below (`H_q_local`, `H_kv_local`, `F_local`, `D_conv_local`), are defined in
> [`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) §3b. This section states only the resulting per-device FLOPs and memory.

When 1D Megatron-style Tensor Parallelism (`TP`) is applied, the intermediate
dimension is sharded per device as `F_local`.

### Standard FFN (TP)

*   **Sharding:** `W_1` is Column Parallel (`[D, F_local]`), `W_2` is Row
    Parallel (`[F_local, D]`).
*   **FLOPS:** `4 * B_local * S * D * F_local`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_act`
    *   Weight Read: `2 * D * F_local * bytes_weight`
    *   Activation Write: `B_local * S * D * bytes_acc`
*   **Communication:** A reduction across the TP group after the Row-Parallel `W_down`. Priced in
    [`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) — do not use a
    closed-form ring formula here; it is blind to lane count and to the D2D/ICI split.

--------------------------------------------------------------------------------

### Gated FFN (TP)

*   **Sharding:** `W_gate` & `W_up` are Column Parallel (`[D, F_local]`),
    `W_down` is Row Parallel (`[F_local, D]`).
*   **FLOPS:** `6 * B_local * S * D * F_local`
*   **Memory Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_act`
    *   Weight Read: `3 * D * F_local * bytes_weight`
    *   Activation Write: `B_local * S * D * bytes_acc`
*   **Communication:** A reduction across the TP group after the Row-Parallel `W_down`. Priced in
    [`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) — do not use a
    closed-form ring formula here; it is blind to lane count and to the D2D/ICI split.

## References

*   [TPU Specifications]({{reference:roofline-analysis/tpu/V7X.md}})
*   [Type Promotion]({{reference:roofline-analysis/TYPE_PROMOTION.md}})
*   [Tensor Dimension Notation]({{reference:roofline-analysis/NOTATION.md}})
*   [Collective Cost]({{reference:roofline-analysis/COLLECTIVE.md}})
