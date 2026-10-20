---
name: gated-mla-roofline-model
description: Calculates FLOPS, memory transfer, KV cache footprint, and collective communication for Gated Multi-Head Latent Attention (Gated NoPE MLA) layers, such as in Kimi K3. Covers query/KV latent compression, weight absorption attention, multiplicative output gating, gated RMSNorm, decode KV caching, and Tensor Parallelism.
---

# Gated Multi-Head Latent Attention (Gated MLA) Performance Estimation

This document details the operational formulas, mathematical shapes, memory transfers, and collective communication costs for **Gated Multi-Head Latent Attention (Gated NoPE MLA)** layers, as used in the 24 global attention layers of **Kimi K3**.

---

## 1. Overview & Architectural Principles

In hybrid linear-attention models like **Kimi K3**, global attention is selectively applied in an interleaved 3:1 pattern (3 Kimi Delta Attention layers to 1 Gated MLA layer across 93 total layers). 

Gated MLA combines the extreme KV-cache memory efficiency of Multi-Head Latent Attention (MLA) with NoPE (No Positional Embeddings) and a multiplicative output gating projection ($W_g$):

```text
[Input X: B_local x S x D]
       |
       +------------------------------------+------------------------------------+
       |                                    |                                    |
       v                                    v                                    v
[Query Down-Proj W_DQ: D -> D_c,q]   [KV Down-Proj W_DKV: D -> D_c + D_r]   [Output Gate W_g: D -> H_q*D_v]
       |                                    |                                    |
       v                                    v                                    v
[Query Up-Proj W_UQ: D_c,q -> H_q*D_h,qk]  [Write to KV Cache: (D_c + D_r)]   [Sigmoid Gate g: (0, 1)]
       |                                    |                                    |
       +-----------------+------------------+                                    |
                         |                                                       |
                         v                                                       |
        [Core Latent Attention Engine: QK -> Softmax -> AV]                      |
                         | (Weight Absorption / Matrix Associativity)            |
                         v                                                       |
         [Deferred Value Up-Proj W_UV: D_c -> D_v]                               |
                         |                                                       |
                         v                                                       |
           [Raw Attention Output: H_q x D_v]                                     |
                         |                                                       |
                         +-------------------------------------------------------+
                         |
                         v
           [Gated RMSNorm: o = sigmoid(g) * RMSNorm(u)]
                         |
                         v
           [Output Proj W_o: (H_q*D_v) -> D]
                         |
                         v (TP Row-Parallel All-Reduce)
             [Output Y: B_local x S x D]
```

### Key Architectural Characteristics:

1. **NoPE (No Positional Embeddings):**
   * Unlike standard DeepSeek-style MLA, Rotary Positional Embeddings (RoPE) are completely omitted from the attention key/query projections in Kimi K3's MLA layers.
   * Positional and temporal context is preserved and propagated through the recurrent state and 1D causal convolutions of the 69 interleaved KDA layers.
   * Removing RoPE eliminates the separate uncompressed rotary key cache stream and enables pure matrix absorption.
2. **Decoupled Shared Dimension ($D_r = 64$):**
   * In addition to the compressed latent vector $c_{kv} \in \mathbb{R}^{D_c}$ ($512$), a decoupled shared key dimension $k_r \in \mathbb{R}^{D_r}$ ($64$) is preserved for fine-grained query-key matching.
   * Total Query/Key head dimension is $D_{h,qk} = 192$ (where $128 \text{ latent head dimension} + 64 \text{ decoupled shared dimension} = 192$).
3. **Multiplicative Output Gating ($W_g$):**
   * Hidden states are projected to generate a multiplicative output gate $g = \text{sigmoid}(W_g x) \in \mathbb{R}^{H_q \times D_v}$ ($12,288$ channels).
   * The core attention output is normalized via headwise RMSNorm and modulated by $g$: $o = \text{sigmoid}(g) \odot \text{RMSNorm}(u)$.
4. **Extreme KV Cache Compression:**
   * Only the compressed latent vector $c_{kv}$ ($512$) and decoupled key $k_r$ ($64$) are stored per token ($576$ elements = $1,152\text{ Bytes}$ in `bfloat16`).
   * Across 24 MLA layers, the KV cache footprint is only $\mathbf{27.65\text{ KB} \times S}$ per sequence (over $7\times$ smaller than standard MHA).

---

## 2. Required References and Notation

You **MUST** use the standard tensor dimension naming conventions defined in `../../NOTATION.md`:

| Notation | Description | Example (Kimi K3) |
| :--- | :--- | :--- |
| `B` | Global Batch size | Configurable |
| `B_local` | Local Batch size per device (`B / DP`) | Configurable |
| `S` | Sequence length (Tokens per sequence in Prefill) | Configurable |
| `S_kv` | Past KV cache length (Tokens in historical cache during Decode) | Configurable |
| `D` | Model hidden dimension | 7,168 |
| `H_q` | Number of Query attention heads | 96 |
| `H_q_local` | Local query heads per device — derived in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b | $96 / \text{TP}$ |
| `H_kv` / `H_s` | Number of Key/Value heads | 96 |
| `D_c_q` | Query compression latent dimension | 1,536 |
| `D_c` | KV compression latent dimension | 512 |
| `D_r` | Decoupled shared dimension for Query/Key matching | 64 |
| `D_h,qk` | Total Query/Key head dimension ($128 \text{ NoPE} + 64 \text{ shared}$) | 192 |
| `D_v` | Value head dimension | 128 |
| `L_mla` | Number of Gated MLA layers in model | 24 |
| `P` / `TP` | Number of participating devices / Tensor Parallelism degree | Configurable |

You must read and follow:
*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Collective Cost Model](../../../../COLLECTIVE.md)
*   TPU v7x Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base

---

## 3. Precision and Data Types

Read [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md) to set byte multipliers:

*   **Activations (`bytes_act`):** `bfloat16` = 2 Bytes/elem (or `fp8` = 1 Byte/elem).
*   **Weights (`bytes_weight`):** `bfloat16` = 2 Bytes/param (or `fp8` = 1 Byte/param).
*   **KV Cache Storage (`bytes_kv`):** `bfloat16` = 2 Bytes/elem (stores $D_c + D_r = 576$ elements per token).
*   **Matmul Accumulation (`bytes_acc`):** `float32` = 4 Bytes/elem (TPU MXU accumulation).
*   **Collective Communication (`bytes_comm`):** `bfloat16` = 2 Bytes/elem.

---

## 4. Component Operations and Mathematical Breakdown

A Gated MLA layer executes 6 distinct sub-operations:

```text
[Input X] ---> (1) Query Compression (W_DQ -> W_UQ) -------------\
          ---> (2) KV Latent Compression (W_DKV -> KV Cache) ------> [3. Latent Attention Engine] ---> (5. Gated RMSNorm) ---> [6. Out Proj]
          ---> (4) Output Gate Projection (W_g) -----------------/
```

---

### 4.1. Query Compression & Projection

Transforms the model hidden state into compressed multi-head queries.

#### A. Query Down-Projection ($W_{DQ}$)
Compresses input $X$ from $D \to D_{c,q}$ ($7,168 \to 1,536$).
*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, D_c_q]` (Replicated across TP)
*   **Output Shape:** `[B_local, S, D_c_q]`
*   **FLOPS:** `2 * B_local * S * D * D_c_q`
*   **Memory Reads (Bytes):** `(B_local * S * D) * bytes_act + (D * D_c_q) * bytes_weight`
*   **Memory Writes (Bytes):** `(B_local * S * D_c_q) * bytes_act`

#### B. Query Up-Projection ($W_{UQ}$)
Projects compressed query $c_q$ to multi-head queries $Q \in \mathbb{R}^{H_q \times D_{h,qk}}$ ($96 \times 192 = 18,432$).
*   **Input Shape:** `[B_local, S, D_c_q]`
*   **Weight Shape:** `[D_c_q, H_q_local * D_h_qk]`
*   **Output Shape:** `[B_local, S, H_q_local, D_h_qk]`
*   **FLOPS:** `2 * B_local * S * D_c_q * (H_q_local * D_h_qk)`
*   **Memory Reads (Bytes):** `(B_local * S * D_c_q) * bytes_act + (D_c_q * H_q_local * D_h_qk) * bytes_weight`
*   **Memory Writes (Bytes):** `(B_local * S * H_q_local * D_h_qk) * bytes_act`

---

### 4.2. Key-Value Latent Compression ($W_{DKV}$)

Compresses input $X$ into the combined latent vector $c_{kv} \in \mathbb{R}^{D_c}$ and decoupled shared key $k_r \in \mathbb{R}^{D_r}$ ($512 + 64 = 576$).

*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, D_c + D_r]` (Replicated across TP)
*   **Output Shape:** `[B_local, S, D_c + D_r]`
*   **FLOPS:** `2 * B_local * S * D * (D_c + D_r)`
*   **Memory Reads (Bytes):** `(B_local * S * D) * bytes_act + (D * (D_c + D_r)) * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * (D_c + D_r) * bytes_kv` (Written to persistent KV Cache in HBM)

---

### 4.3. Multi-Head Latent Attention Core

Depending on the execution phase:

#### Phase A: Prefill Mode (Sequence Length $S > 1$, Weight Absorption Flow)

In prefill with matrix associativity (weight absorption), queries are projected into the latent Key space, attention scores are computed directly against latent $c_{kv}$ and decoupled $k_r$, and Value projection is deferred until after attention score reduction.

1.  **Query Projection to Key Space ($Q_{\text{absorbed}}$):**
    *   $Q' = Q_{:, :, :, :D_c} W_{UK}^T \in [B_{\text{local}}, S, H_{q,\text{local}}, D_c]$
    *   **FLOPS:** `2 * B_local * S * H_q_local * D_h_qk * D_c` (if applied dynamically)
    *   **Memory Reads:** `(B_local * S * H_q_local * D_h_qk) * bytes_act + (H_q_local * D_h_qk * D_c) * bytes_weight`
    *   **Memory Writes:** `(B_local * S * H_q_local * D_c) * bytes_act`
2.  **QK Phase (FlashAttention Causal Scores):**
    *   Latent QK: $Q' \cdot c_{kv}^T \implies B_{\text{local}} \cdot S^2 \cdot H_{q,\text{local}} \cdot D_c$
    *   Decoupled Shared QK: $Q_r \cdot k_r^T \implies B_{\text{local}} \cdot S^2 \cdot H_{q,\text{local}} \cdot D_r$
    *   **Causal QK FLOPS:** `B_local * S^2 * H_q_local * (D_c + D_r)`
    *   **Memory Reads:** Read $Q'$ (`B_local * S * H_q_local * D_c * bytes_act`), Read $c_{kv, \text{full}}$ (`B_local * S * (D_c + D_r) * bytes_kv`)
    *   **Memory Writes:** 0 (Fused inside FlashAttention SRAM/VMEM)
3.  **AV Phase (Attention over Latent KV):**
    *   Multiplies causal attention probabilities $A$ by latent $c_{kv}$: $u_{\text{latent}} = A \cdot c_{kv} \in [B_{\text{local}}, S, H_{q,\text{local}}, D_c]$.
    *   **Causal AV FLOPS:** `B_local * S^2 * H_q_local * D_c`
    *   **Memory Reads:** Read $c_{kv}$ (`B_local * S * D_c * bytes_kv`)
    *   **Memory Writes:** `(B_local * S * H_q_local * D_c) * bytes_act`
4.  **Deferred Value Up-Projection ($W_{UV}$):**
    *   Projects $u_{\text{latent}}$ from latent space $D_c \to D_v$ ($512 \to 128$):
        $$u = u_{\text{latent}} W_{UV} \in [B_{\text{local}}, S, H_{q,\text{local}}, D_v]$$
    *   **Weight Shape:** `[D_c, H_q_local * D_v]`
    *   **FLOPS:** `2 * B_local * S * D_c * (H_q_local * D_v)`
    *   **Memory Reads:** `(B_local * S * H_q_local * D_c) * bytes_act + (D_c * H_q_local * D_v) * bytes_weight`
    *   **Memory Writes:** `(B_local * S * H_q_local * D_v) * bytes_act`

#### Phase B: Decode Mode (Single-Token Autoregressive Generation, $S=1$ against KV Cache $S_{\text{kv}}$)

In decode mode, single token latents are generated, appended to the KV cache, and attended against historical tokens using absorbed matrices.

1.  **Step Latents Generation ($S = 1$):**
    *   Generate $c_q$: `2 * B_local * 1 * D * D_c_q` FLOPS.
    *   Generate $Q$: `2 * B_local * 1 * D_c_q * (H_q_local * D_h_qk)` FLOPS.
    *   Generate new KV cache entry $[c_{kv}, k_r]$: `2 * B_local * 1 * D * (D_c + D_r)` FLOPS.
    *   **KV Cache Append Write:** `B_local * 1 * (D_c + D_r) * bytes_kv`.
2.  **Decode QK Phase against Historical Cache ($S_{\text{kv}}$):**
    *   Attends 1 query token against $S_{\text{kv}}$ cached tokens:
    *   **QK FLOPS:** `2 * B_local * 1 * S_kv * H_q_local * (D_c + D_r)`
    *   **Memory Reads:** Read $c_{kv}$ Cache (`B_local * S_kv * D_c * bytes_kv`), Read $k_r$ Cache (`B_local * S_kv * D_r * bytes_kv`).
    *   **Memory Writes:** 0 (Fused in VMEM).
3.  **Decode AV Phase & Value Up-Projection:**
    *   Latent AV combination: $u_{\text{latent}} = A \cdot c_{kv} \in [B_{\text{local}}, 1, H_{q,\text{local}}, D_c]$.
        *   **AV FLOPS:** `2 * B_local * 1 * S_kv * H_q_local * D_c`
    *   Value Up-Projection ($D_c \to D_v$):
        *   **Value Proj FLOPS:** `2 * B_local * 1 * D_c * (H_q_local * D_v)`
    *   **Memory Reads:** Read $c_{kv}$ Cache (`B_local * S_kv * D_c * bytes_kv`), Read $W_{UV}$ (`D_c * (H_q_local * D_v) * bytes_weight`).
    *   **Memory Writes:** `(B_local * 1 * H_q_local * D_v) * bytes_act`.

---

### 4.4. Output Gate Projection ($W_g$)

Computes the multiplicative output gate $g$ from model dimension $D$:

$$g = \text{sigmoid}(W_g X) \in \mathbb{R}^{B_{\text{local}} \times S \times (H_{q,\text{local}} \times D_v)}$$

*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, H_q_local * D_v]` ($7,168 \times 12,288 / \text{TP}$)
*   **Output Shape:** `[B_local, S, H_q_local * D_v]`
*   **FLOPS:** `2 * B_local * S * D * (H_q_local * D_v)`
*   **Memory Reads (Bytes):** `(B_local * S * D) * bytes_act + (D * H_q_local * D_v) * bytes_weight`
*   **Memory Writes (Bytes):** `(B_local * S * H_q_local * D_v) * bytes_act`

---

### 4.5. Gated RMSNorm Modulation

Applies headwise RMSNorm on the core attention output $u$ and modulates by the sigmoid output gate $g$:

$$o = \text{sigmoid}(g) \odot \text{RMSNorm}(u) \in \mathbb{R}^{B_{\text{local}} \times S \times (H_{q,\text{local}} \times D_v)}$$

*   **Input Shapes:** $u \in [B_{\text{local}}, S, H_{q,\text{local}} \times D_v]$, $g \in [B_{\text{local}}, S, H_{q,\text{local}} \times D_v]$
*   **Output Shape:** $o \in [B_{\text{local}}, S, H_{q,\text{local}} \times D_v]$
*   **FLOPS:** `3 * B_local * S * (H_q_local * D_v)` (RMS variance reduction, scaling, and sigmoid gate Hadamard multiplication).
*   **Memory Transfer:** Fused in VMEM with attention output and Output Projection (0 extra HBM access).

---

### 4.6. Output Projection (`out_proj`) & Collective Communication

Projects the gated multi-head output back to the model hidden dimension $D$:

$$Y = o \cdot W_o \in \mathbb{R}^{B_{\text{local}} \times S \times D}$$

*   **Input Shape:** `[B_local, S, H_q_local * D_v]`
*   **Weight Shape:** `[H_q_local * D_v, D]` ($12,288 / \text{TP} \times 7,168$)
*   **Output Shape:** `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (H_q_local * D_v) * D`
*   **Memory Reads (Bytes):** `(B_local * S * H_q_local * D_v) * bytes_act + (H_q_local * D_v * D) * bytes_weight`
*   **Memory Writes (Bytes):** `(B_local * S * D) * bytes_act`
*   **Collective Communication:** The row-parallel output projection's partial sums require a
    reduction across the TP group. Payload, dtype site, per-layer count and variant are priced in
    [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md).

---

## 5. Distributed Parallelism Strategies

### Tensor Parallelism (TP)

> **Partitioning is not defined here.** How TP shards this op, and the per-device symbols below (`H_q_local`, `H_kv_local`, `F_local`, `D_conv_local`), are defined in
> [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b. This section states only the resulting per-device FLOPs and memory.
When Tensor Parallelism of degree `TP` is applied:
*   **Query Heads Sharding:** $H_{q,\text{local}} = H_q / \text{TP} = 96 / \text{TP}$.
*   **Gate Channels Sharding:** $(H_q \times D_v) / \text{TP} = 12,288 / \text{TP}$.
*   **KV Compression ($W_{DKV}$):** Fully replicated across devices (all devices receive the full $576$-dim compressed KV representation).
*   **Projections Sharding:**
    *   $W_o$ (`out_proj`) emits a partial sum needing a TP reduction — priced in
        [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md).

---

## 6. Python Calculation Template

Always write and execute a Python script to compute roofline metrics. Do not perform arithmetic mentally.

```python
from cost_estimator import collective

def compute_gated_mla_roofline(
    b_local: int,
    s: int,
    is_decode: bool = False,
    s_kv: int = 0,
    d: int = 7168,
    h_q: int = 96,
    d_c_q: int = 1536,
    d_c: int = 512,
    d_r: int = 64,
    d_h_qk: int = 192,
    d_v: int = 128,
    tp: int = 1,
    bytes_act: float = 2.0,
    bytes_weight: float = 2.0,
    bytes_kv: float = 2.0,
    bw_hbm: float = 3.69e12,
    peak_tflops_bf16: float = 1153.5e12,
):
    results = {}
    h_q_local = h_q // tp
    cur_s = 1 if is_decode else s
    ctx_len = s_kv if is_decode else s

    # 1. Query Latent Down-Projection (D -> D_c_q)
    flops_q_down = 2 * b_local * cur_s * d * d_c_q
    bytes_q_down = (b_local * cur_s * d + b_local * cur_s * d_c_q) * bytes_act + (d * d_c_q) * bytes_weight
    t_comp_q_down = flops_q_down / peak_tflops_bf16 * 1000.0
    t_mem_q_down = bytes_q_down / bw_hbm * 1000.0
    results["q_latent_down_proj"] = {
        "GFlops": flops_q_down / 1e9,
        "memory_transfer (MB)": bytes_q_down / 1e6,
        "execution_time (ms)": max(t_comp_q_down, t_mem_q_down),
        "bound": "Compute" if t_comp_q_down >= t_mem_q_down else "Memory",
    }

    # 2. Query Up-Projection (D_c_q -> H_q_local * D_h_qk)
    flops_q_up = 2 * b_local * cur_s * d_c_q * (h_q_local * d_h_qk)
    bytes_q_up = (b_local * cur_s * d_c_q + b_local * cur_s * h_q_local * d_h_qk) * bytes_act + \
                 (d_c_q * h_q_local * d_h_qk) * bytes_weight
    t_comp_q_up = flops_q_up / peak_tflops_bf16 * 1000.0
    t_mem_q_up = bytes_q_up / bw_hbm * 1000.0
    results["q_up_proj"] = {
        "GFlops": flops_q_up / 1e9,
        "memory_transfer (MB)": bytes_q_up / 1e6,
        "execution_time (ms)": max(t_comp_q_up, t_mem_q_up),
        "bound": "Compute" if t_comp_q_up >= t_mem_q_up else "Memory",
    }

    # 3. KV Latent Down-Projection (D -> D_c + D_r)
    flops_kv_down = 2 * b_local * cur_s * d * (d_c + d_r)
    bytes_kv_down = (b_local * cur_s * d) * bytes_act + (d * (d_c + d_r)) * bytes_weight + \
                    (b_local * cur_s * (d_c + d_r)) * bytes_kv
    t_comp_kv_down = flops_kv_down / peak_tflops_bf16 * 1000.0
    t_mem_kv_down = bytes_kv_down / bw_hbm * 1000.0
    results["kv_latent_down_proj"] = {
        "GFlops": flops_kv_down / 1e9,
        "memory_transfer (MB)": bytes_kv_down / 1e6,
        "execution_time (ms)": max(t_comp_kv_down, t_mem_kv_down),
        "bound": "Compute" if t_comp_kv_down >= t_mem_kv_down else "Memory",
    }

    # 4. Core Latent Attention Engine (QK + AV + Value Up Proj)
    if not is_decode:
        # Prefill Flow B (Weight Absorption)
        flops_qk = b_local * (s ** 2) * h_q_local * (d_c + d_r)
        flops_av = b_local * (s ** 2) * h_q_local * d_c
        flops_v_up = 2 * b_local * s * d_c * (h_q_local * d_v)
        flops_attn = flops_qk + flops_av + flops_v_up
        bytes_attn = (b_local * s * h_q_local * d_c + b_local * s * (d_c + d_r)) * bytes_kv + \
                     (d_c * h_q_local * d_v) * bytes_weight + (b_local * s * h_q_local * d_v) * bytes_act
    else:
        # Decode Mode (1 token attending to s_kv)
        flops_qk = 2 * b_local * 1 * ctx_len * h_q_local * (d_c + d_r)
        flops_av = 2 * b_local * 1 * ctx_len * h_q_local * d_c
        flops_v_up = 2 * b_local * 1 * d_c * (h_q_local * d_v)
        flops_attn = flops_qk + flops_av + flops_v_up
        bytes_attn = (b_local * ctx_len * (d_c + d_r)) * bytes_kv + \
                     (d_c * h_q_local * d_v) * bytes_weight + (b_local * 1 * h_q_local * d_v) * bytes_act

    t_comp_attn = flops_attn / peak_tflops_bf16 * 1000.0
    t_mem_attn = bytes_attn / bw_hbm * 1000.0
    results["latent_attention_core"] = {
        "GFlops": flops_attn / 1e9,
        "memory_transfer (MB)": bytes_attn / 1e6,
        "execution_time (ms)": max(t_comp_attn, t_mem_attn),
        "bound": "Compute" if t_comp_attn >= t_mem_attn else "Memory",
    }

    # 5. Output Gate Projection (W_g: D -> H_q_local * D_v)
    flops_gate = 2 * b_local * cur_s * d * (h_q_local * d_v)
    bytes_gate = (b_local * cur_s * d + b_local * cur_s * h_q_local * d_v) * bytes_act + \
                 (d * h_q_local * d_v) * bytes_weight
    t_comp_gate = flops_gate / peak_tflops_bf16 * 1000.0
    t_mem_gate = bytes_gate / bw_hbm * 1000.0
    results["output_gate_proj"] = {
        "GFlops": flops_gate / 1e9,
        "memory_transfer (MB)": bytes_gate / 1e6,
        "execution_time (ms)": max(t_comp_gate, t_mem_gate),
        "bound": "Compute" if t_comp_gate >= t_mem_gate else "Memory",
    }

    # 6. Gated RMSNorm Modulation (Fused)
    flops_norm = 3 * b_local * cur_s * (h_q_local * d_v)
    bytes_norm = (b_local * cur_s * h_q_local * d_v * 2) * bytes_act
    t_comp_norm = flops_norm / peak_tflops_bf16 * 1000.0
    t_mem_norm = bytes_norm / bw_hbm * 1000.0
    results["gated_rmsnorm"] = {
        "GFlops": flops_norm / 1e9,
        "memory_transfer (MB)": bytes_norm / 1e6,
        "execution_time (ms)": max(t_comp_norm, t_mem_norm),
        "bound": "Compute" if t_comp_norm >= t_mem_norm else "Memory",
    }

    # 7. Output Projection (W_o: H_q_local * D_v -> D)
    flops_out = 2 * b_local * cur_s * (h_q_local * d_v) * d
    bytes_out = (b_local * cur_s * h_q_local * d_v + b_local * cur_s * d) * bytes_act + \
                (h_q_local * d_v * d) * bytes_weight
    t_comp_out = flops_out / peak_tflops_bf16 * 1000.0
    t_mem_out = bytes_out / bw_hbm * 1000.0
    results["out_proj"] = {
        "GFlops": flops_out / 1e9,
        "memory_transfer (MB)": bytes_out / 1e6,
        "execution_time (ms)": max(t_comp_out, t_mem_out),
        "bound": "Compute" if t_comp_out >= t_mem_out else "Memory",
    }

    return results
```

---

## 7. Concrete Parameter Specification: Kimi K3 Gated MLA Layer

For Kimi K3 (24 MLA Layers):

| Parameter | Notation | Numerical Value | Description |
| :--- | :--- | :--- | :--- |
| **Model Hidden Dim** | $D$ | 7,168 | Model feature dimension |
| **Query Heads** | $H_q$ | 96 | Attention query heads |
| **State / KV Heads** | $H_{kv}$ / $H_s$ | 96 | Attention key/value heads |
| **Query Latent Dimension** | $D_{c,q}$ | 1,536 | Low-rank query compression dimension |
| **KV Latent Dimension** | $D_c$ | 512 | Compressed KV representation dimension |
| **Decoupled Shared Dim** | $D_r$ | 64 | Decoupled query/key matching dimension |
| **Query/Key Head Dim** | $D_{h,qk}$ | 192 | Total query/key head dimension ($128 \text{ NoPE} + 64 \text{ shared}$) |
| **Value Head Dim** | $D_v$ | 128 | Value head dimension |
| **Output Gate Output Dim** | $H_q \times D_v$ | 12,288 | Multiplicative sigmoid gate channels ($96 \times 128$) |
| **Output Projection Dim** | $H_q D_v \to D$ | $12,288 \to 7,168$ | Linear output projection to hidden state |
| **KV Cache Footprint** | — | **1,152 Bytes / token** | $576 \text{ elements} \times 2\text{ Bytes}$ (`bf16`) per layer |

---

## References

*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)
*   [Standard MLA Roofline Skill](../mla/mla.md)
*   [KDA Roofline Skill](../kda/kda.md)
*   [Kimi K3 Architecture Specification](../../../architecture/kimi/K3.md)
*   [Collective Cost Model](../../../../COLLECTIVE.md)
*   TPU v7x Specification: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
