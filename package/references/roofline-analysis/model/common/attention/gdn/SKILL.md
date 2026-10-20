---
name: rooflining-gdn-attention
description: Calculates FLOPS and memory transfer for Gated Delta Network (GDN) linear attention layers.
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/NOTATION.md
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/model/common/conv1d/SKILL.md
- references/roofline-analysis/parallelism/tp/SKILL.md
- references/roofline-analysis/tpu/V7X.md
---

# Gated Delta Network (GDN) Performance Estimation

Estimate the roofline performance of Gated Delta Network (GDN) linear attention
layers (such as the 45 linear layers in Qwen3.5-397B-A17B) by calculating FLOPS
and memory transfer for each sub-operation.

Report per-operation level metrics because different sub-operations (e.g., ZBA
projection vs. core recurrent update kernel) exhibit different bottlenecks
(compute-bound vs. memory-bound).

## Requirements

You must read and follow:

*   [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}})
*   [Type Promotion Rules]({{reference:roofline-analysis/TYPE_PROMOTION.md}})

## Workflow

1.  **Identify Dimensions:** Identify the following dimensions from the
    notation:
    *   `B`: Global Batch Size
    *   `B_local`: Local Batch Size (per device)
    *   `S`: Sequence Length (context length)
    *   `D`: Hidden Dimension (model dimension)
    *   `H_qk`: Number of Query/Key heads (e.g., 16 in Qwen3.5)
    *   `H_v`: Number of Value heads (e.g., 64 in Qwen3.5)
    *   `D_h_qk`: Head dimension of Query/Key (e.g., 128)
    *   `D_h_v`: Head dimension of Value (e.g., 128)
    *   `C`: Chunk Size (typically 64 for chunk-wise Prefill updates)
2.  **Determine Precision:** Read
    [TYPE_PROMOTION.md]({{reference:roofline-analysis/TYPE_PROMOTION.md}}) to calculate
    byte-size configurations for activations, weights, and caches.
3.  **Write and Run Python Script for Calculations:** Write and execute a Python
    script to perform all numerical calculations (FLOPS, memory bytes,
    latencies). Do not perform roofline arithmetic mentally.
4.  **Calculate Roofline & Total Latency:** Combine calculated values with
    TPU/GPU hardware specs to estimate execution times.
5.  **Output Format:** Output execution results in standard JSON format as
    described in [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}).

--------------------------------------------------------------------------------

## Operations & Calculations

A GDN linear attention block consists of 5 main operational steps:

```
[Input X] ---> (1. QKV Proj) ---> (2. Conv1D) ----\
         ---> (1.5 ZBA Proj) -------------------> [3. Gated Delta Rule Kernel] ---> (4. Gated RMSNorm) ---> [Out Proj]
```

### 1. QKV Projection (`in_proj_qkv`)

Projects hidden states into Q, K, and V spaces prior to convolution.

*   **Weight Shape:** `[D, 2 * H_qk * D_h_qk + H_v * D_h_qk]`
*   **Activation Shape:** Input `[B_local, S, D]` -> Output `[B_local, S, 2 * H_qk * D_h_qk + H_v * D_h_qk]`
*   **FLOPS:** `2 * B_local * S * D * (2 * H_qk * D_h_qk + H_v * D_h_qk)`
*   **HBM Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_per_element`
    *   Weight Read: `D * (2 * H_qk * D_h_qk + H_v * D_h_qk) *
        bytes_per_element`
    *   Activation Write: `B_local * S * (2 * H_qk * D_h_qk + H_v * D_h_qk) *
        bytes_per_element`

### 2. ZBA Projection (`in_proj_z`, `in_proj_b`, `in_proj_a` - Fused)

Projects hidden states to the gate parameters: **Z** (RMSNorm gate), **B** (Beta
update gate), and **A** (Alpha decay/forget gate).

*   **Weight Shape:** `[D, H_v * D_h_v + 2 * H_v]`
*   **Activation Shape:** Input `[B_local, S, D]` -> Output `[B_local, S, H_v * D_h_v + 2 * H_v]`
*   **FLOPS:** `2 * B_local * S * D * (H_v * D_h_v + 2 * H_v)`
*   **HBM Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_per_element`
    *   Weight Read: `D * (H_v * D_h_v + 2 * H_v) * bytes_per_element`
    *   Activation Write: `B_local * S * (H_v * D_h_v + 2 * H_v) * bytes_per_element`

### 3. Causal Depthwise 1D Convolution

Refer to the [Conv1D Roofline Skill]({{reference:roofline-analysis/model/common/conv1d/SKILL.md}}) instructions for
calculating FLOPS, Cache State shapes, and sharding.

*   `D_conv` here maps to `2 * H_qk * D_h_qk + H_v * D_h_qk`.
*   Note: In Decode mode, Conv1D requires reading and writing the convolution
    history cache from HBM unless states are kept in VMEM.

### 4. Gated Delta Rule Kernel (Attention Engine)

Calculates recurrent updates to memory state matrix $S$ and yields core
attention output. Evaluated differently depending on the execution regime:

#### A. Prefill Mode (Chunk-wise Forward)

Operates chunk-by-chunk using chunk size `C` (typically `C = 64`).

*   **FLOPS**:
    1.  *Gating Activations & Norms*: `O(B_local * S * H_v * D_h_v)`
    2.  *Intra-chunk Attention Matrix*: `2 * S * C * H_v * D_h_qk`
    3.  *Intra-chunk Recurrent Scan Loop*: `2 * S * C * H_v`
    4.  *Chunk Value Update*: `2 * S * C * H_v * D_h_v`
    5.  *Key Cumulative Decay*: `2 * S * C * H_v * D_h_qk`
    6.  *Inter-chunk State Propagation (Matrix Multiplies over S/C chunks)*:
        *   Query-Key Matrix: `2 * B_local * H_v * (S/C) * C^2 * D_h_qk`
        *   State Retrieval: `4 * B_local * H_v * (S/C) * C * D_h_qk * D_h_v`
        *   Attention Output: `2 * B_local * H_v * (S/C) * C^2 * D_h_v`
        *   State Matrix update: `2 * B_local * H_v * (S/C) * D_h_qk * C * D_h_v`
*   **HBM Transfer (Bytes)** (Fused kernel on VMEM):
    *   Reads (Q, K, V, g, beta): `B_local * S * (H_qk * D_h_qk * 2 + H_v *
        D_h_v) * bytes_per_element` + gates.
    *   Writes (core_attn_out): `B_local * S * H_v * D_h_v * bytes_per_element`.
    *   If intermediate states are stored to HBM for validation or training
        backward pass, include State cache read/write: `B_local * H_v * D_h_qk *
        D_h_v * bytes_per_element`.

#### B. Decode Mode (Autoregressive Token Generation)

Operates sequentially with context $S = 1$.

*   **FLOPS** (Per step):
    *   State Decay: `B_local * H_v * D_h_qk * D_h_v`
    *   Retrieve Memory (`S^T @ k`): `2 * B_local * H_v * D_h_qk * D_h_v`
    *   Compute Delta: `B_local * H_v * D_h_v`
    *   State Update (`S + k @ delta^T`): `2 * B_local * H_v * D_h_qk * D_h_v`
    *   Attention Output (`S^T @ q`): `2 * B_local * H_v * D_h_qk * D_h_v`
    *   *Total FLOPS*: `7 * B_local * H_v * D_h_qk * D_h_v`
*   **HBM Transfer (Bytes)** (Fused kernel):
    *   Q, K, V, Gates Read: `B_local * 1 * (2 * D_h_qk + D_h_v) *
        bytes_per_element` + gates (negligible).
    *   Recurrent State Matrix Read: `B_local * H_v * D_h_qk * D_h_v *
        bytes_per_element`
    *   Recurrent State Matrix Write: `B_local * H_v * D_h_qk * D_h_v *
        bytes_per_element`
    *   Output Write: `B_local * 1 * H_v * D_h_v * bytes_per_element`

### 5. Gated RMSNorm

Scales the attention output with the normalizer tensor $Z$.

*   **Activation Shapes:** Input `core_attn_out` shape `[B_local, S, H_v * D_h_v]`, normalizer `Z` shape `[B_local, S, H_v * D_h_v]`
*   **FLOPS:** `B_local * S * H_v * D_h_v` (for variance, division, SiLU
    activation, and element-wise multiplication).
*   **HBM Transfer (Bytes):**
    *   *If unfused*: Read `core_attn_out` + Read `Z` + Write `Output` = `3 *
        B_local * S * H_v * D_h_v * bytes_per_element`.
    *   *If fused (kept in VMEM)*: Intermediate reads/writes are avoided. Only
        writes final output: `B_local * S * H_v * D_h_v * bytes_per_element`.

--------------------------------------------------------------------------------

## Parallelism Strategies

> **Partitioning is not defined here.** How TP shards this op, and the per-device symbols below (`H_q_local`, `H_kv_local`, `F_local`, `D_conv_local`), are defined in
> [`../../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}) §3b. This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

projections (`in_proj_qkv`, `in_proj_zba`) and activations are split across
devices.

*   **ZBA gate width:** `zba_local = H_v_local * D_h_v + 2 * H_v_local`.
*   **Gated Delta Kernel & RMSNorm**: Device execution scales with the local
    number of heads ($H\_v\_local$). No cross-device communications are required
    inside the attention kernel.

--------------------------------------------------------------------------------

## References

*   [TPU Specifications]({{reference:roofline-analysis/tpu/V7X.md}})
*   [Type Promotion]({{reference:roofline-analysis/TYPE_PROMOTION.md}})
*   [Tensor Dimension Notation]({{reference:roofline-analysis/NOTATION.md}})
*   [Collective Cost]({{reference:roofline-analysis/COLLECTIVE.md}})
