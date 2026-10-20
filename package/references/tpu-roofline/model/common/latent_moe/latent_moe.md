---
name: latent-moe-roofline-model
description: Guidelines and mathematical formulas for calculating Roofline performance estimation for Latent Mixture of Experts (LatentMoE) and Stable LatentMoE components (such as in Kimi K3). Covers latent down-projection, Quantile Balancing routing, routed expert MLP in latent space with SiTU-GLU/SwiGLU, MXFP4 quantization, latent RMSNorm pre-projection, latent up-projection, and shared experts. EP collectives live in parallelism/ep/.
---

# Latent Mixture of Experts (LatentMoE) Component

This document describes the operations, mathematical formulas, tensor shapes, memory transfers, and collective communication overheads involved in a **Latent Mixture of Experts (LatentMoE)** layer for roofline performance estimation. It covers both the general LatentMoE design and the **Stable LatentMoE** architecture used in frontier models like **Kimi K3**.

---

## 1. Overview & Architectural Principles

In standard Mixture of Experts (MoE) models (e.g., Mixtral, DeepSeek-V3), tokens are routed and processed by experts directly in the full hidden dimension $D$. As models scale to hundreds or thousands of fine-grained experts, the communication volume for expert token dispatching and the memory bandwidth required to fetch expert weights become severe performance bottlenecks.

**Latent MoE** addresses this by performing token routing, expert parallel communication, and expert MLP computation inside a compressed **latent bottleneck dimension** $D_{\text{latent}} < D$ (e.g., $D = 7,168 \to D_{\text{latent}} = 3,584$ in Kimi K3).

```text
[Input X: B_local x S x D]
       |
       +-------------------------------------------------------------+
       | (Parallel Shared Expert Branch - Full Dimension D)           | (Parallel Routed Latent MoE Branch)
       v                                                             v
[Shared W1 (Gate/Up): D -> 2*F_shared]                       [Latent Down-Proj: D -> D_latent]
       |                                                             |
[Activation: SiTU-GLU / SwiGLU]                                      v
       |                                                     [Quantile Router: Top K of E Experts]
[Shared W2 (Down): F_shared -> D]                                    |
       |                                                             v
       |                                             [Token Dispatch (All-to-All in D_latent or All-Gather)]
       |                                                             |
       |                                                             v
       |                                                     [Routed W1 (Gate/Up): D_latent -> 2*F_expert]
       |                                                             | (MXFP4 Weights: 0.5 Bytes/param)
       |                                                             v
       |                                                     [Activation: SiTU-GLU Soft-Clamping]
       |                                                             |
       |                                                             v
       |                                                     [Routed W2 (Down): F_expert -> D_latent]
       |                                                             | (MXFP4 Weights: 0.5 Bytes/param)
       |                                                             v
       |                                                     [Expert Output Combine / Weighted Sum]
       |                                                             |
       |                                                             v
       |                                                     [Latent RMSNorm Pre-Projection]
       |                                                             |
       |                                                             v
       |                                                     [Token Combine (All-to-All in D_latent)]
       |                                                             |
       |                                                             v
       |                                                     [Latent Up-Proj: D_latent -> D]
       |                                                             |
       +----------------------------> (+) <--------------------------+
                                      |
                           [Output Y: B_local x S x D]
```

### Key Architectural Advantages of Latent MoE:
1. **50% Communication Volume Reduction (Dispatch-based EP):** Token dispatch and token combine collectives (All-to-All) operate on vectors of size $D_{\text{latent}}$ rather than $D$. For $D_{\text{latent}} = D / 2$, communication volume across accelerator nodes is halved.
2. **50% Weight Parameter & Memory Bandwidth Reduction per Expert:** Expert matrices ($W_1, W_2, W_3$) have dimensions $D_{\text{latent}} \times F_{\text{expert}}$ instead of $D \times F_{\text{expert}}$. This drastically cuts HBM footprint and weight fetch time during memory-bound decoding.
3. **Extreme Expert Sparsity Scaling:** Allows scaling to large expert counts (e.g., $E = 896$ routed experts with $K = 16$ active in Kimi K3) while keeping activation and weight footprints manageable.

### Kimi K3 Stable LatentMoE Interventions:
*   **Quantile Balancing (QB):** An auxiliary-loss-free, hyperparameter-free routing mechanism where routing biases are deterministically calibrated based on router-score quantiles to eliminate routing collapse across 896 experts.
*   **SiTU-GLU (Sigmoid Tanh Unit Gated Linear Unit):** Replaces standard unbounded SwiGLU with scaled, soft-capped $\tanh$ functions to prevent coordinate blow-up and numeric overflows during low-precision (MXFP4) computation:
    $$h(g, u) = \left[ \beta_1 \tanh\left(\frac{g}{\beta_1}\right) \cdot \sigma(g) \right] \odot \left[ \beta_2 \tanh\left(\frac{u}{\beta_2}\right) \right]$$
    where $\beta_1 = 4.0, \beta_2 = 25.0$, bounding the output range to $[-100, 100]$.
*   **Latent RMSNorm Pre-Projection:** Applies an RMSNorm to the aggregated latent expert output $z_{\text{moe}} \in \mathbb{R}^{D_{\text{latent}}}$ before the linear up-projection $D_{\text{latent}} \to D$, stabilizing gradient and representation variance across deep (93-layer) networks.
*   **Microscopic FP4 Quantization (MXFP4):** Routed expert weights are quantized to 4-bit microscopic scaling format (`mxfp4`, 0.5 Bytes per parameter).
*   **Full-Rank Shared Experts ($E_{\text{shared}} = 2$):** A parallel, uncompressed path operating directly on hidden dimension $D$ in `bfloat16` to preserve dense contextual representations.

---

## 2. Required References and Notation

You **MUST** use the standard tensor dimension naming conventions defined in `../../NOTATION.md`:

| Notation | Description | Example (Kimi K3) |
| :--- | :--- | :--- |
| `B` | Global Batch size | Configurable |
| `B_local` | Local Batch size per device (`B / DP`) | Configurable |
| `S` | Sequence length (Tokens per sequence) | Prefill: $S \ge 1$, Decode: $S = 1$ |
| `D` | Hidden dimension (Model dimension) | 7,168 |
| `D_latent` | Latent bottleneck dimension for MoE down-projection | 3,584 |
| `F_expert` | Intermediate hidden dimension per routed expert | 3,072 |
| `F_shared` | Total intermediate dimension for shared experts | 6,144 ($2 \times 3,072$) |
| `E` / `E_routed` | Total number of routed experts | 896 |
| `K` / `E_active` | Top-K routed experts activated per token | 16 |
| `E_shared` | Total number of shared experts | 2 |
| `P` | Number of participating devices | Configurable |
| `TP` | Tensor Parallelism degree | Configurable |
| `EP` | Expert Parallelism degree | Configurable |
| `DP` | Data Parallelism degree | Configurable |

You must read and follow:
*   [Common Requirements](../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../NOTATION.md)
*   [Collective Cost Model](../../../COLLECTIVE.md)
*   TPU v7x Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base

---

## 3. Precision and Data Type Multipliers

Read [TYPE_PROMOTION.md](../../../TYPE_PROMOTION.md) to set byte multipliers:

*   **Activations (`bytes_act`):** `bfloat16` = 2 Bytes/elem (or `fp8` = 1 Byte/elem).
*   **Routed Expert Weights (`bytes_weight_routed`):** **`mxfp4`** = 0.5 Bytes/param (4-bit microscopic format).
*   **Shared Expert & Projection Weights (`bytes_weight`):** `bfloat16` = 2 Bytes/param.
*   **Matmul Accumulation (`bytes_acc`):** `float32` = 4 Bytes/elem (TPU MXU accumulation).
*   **Collective Communication (`bytes_comm`):** `bfloat16` = 2 Bytes/elem.

---

## 4. Component Operations and Mathematical Breakdown

A Latent MoE layer consists of 7 sub-operations split across two parallel branches:

```text
Branch A (Routed Latent MoE):  (1) Down-Proj -> (2) Router -> (3) Routed W1 -> (4) SiTU -> (5) Routed W2 -> (6) RMSNorm -> (7) Up-Proj
Branch B (Shared Experts):     (1B) Shared W1 -> (2B) Activation -> (3B) Shared W2
Merge:                         Output = Y_routed + Y_shared (+ Residual)
```

---

### 4.1. Latent Down-Projection (`W_down_lat`)

Compresses the input tokens from model dimension $D$ down to the latent space $D_{\text{latent}}$.

#### Single-Device / Unsharded / Replicated TP
*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, D_latent]`
*   **Output Shape:** `[B_local, S, D_latent]`
*   **FLOPS:** `2 * B_local * S * D * D_latent`
*   **Memory Reads (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_act`
    *   Weight Read: `D * D_latent * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * D_latent * bytes_act`
*   **Collective Communication:** None

#### Tensor Parallelism (TP Column-Parallel)
*   **Weight Shape:** `[D, D_latent / TP]`
*   **Output Shape:** `[B_local, S, D_latent / TP]`
*   **FLOPS:** `2 * B_local * S * D * (D_latent / TP)`
*   **Memory Reads:** `(B_local * S * D) * bytes_act + D * (D_latent / TP) * bytes_weight`
*   **Memory Writes:** `B_local * S * (D_latent / TP) * bytes_act`

---

### 4.2. Quantile Router (Gating)

Computes gating logits across all $E$ experts for each token. In Stable LatentMoE, Quantile Balancing determines the top-$K$ active experts without auxiliary loss.

*   **Input Shape:** `[B_local, S, D]` (or `[B_local, S, D_latent]` if router projects from latent space)
*   **Weight Shape:** `[D, E]` (or `[D_latent, E]`) (Replicated across devices)
*   **Output Shape:** Logits `[B_local, S, E]` $\to$ Top-$K$ weights & indices `[B_local, S, K]`
*   **FLOPS:** `2 * B_local * S * D * E` (or `2 * B_local * S * D_latent * E`)
*   **Memory Reads (Bytes):** `B_local * S * D * bytes_act + D * E * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * K * bytes_act` (Top-$K$ gating weights)
*   **Ignored Overhead:** As per standard practice, top-$k$ sorting, quantile thresholding, and token permutation indexing FLOPs/memory transfers are negligible and ignored.

---

### 4.3. Routed MoE W1 (Gate/Up Projection with SiTU-GLU)

Each token routes its compressed $D_{\text{latent}}$ representation to $K$ active experts. The total number of tokens processed across all experts is $B_{\text{local}} \cdot S \cdot K$.

Because Kimi K3 uses gated activations (SiTU-GLU), each expert has two projection matrices: $W_{\text{gate}}$ (or $W_3$) and $W_{\text{up}}$ (or $W_1$), each projecting $D_{\text{latent}} \to F_{\text{expert}}$.

#### Tensor Parallelism (TP) Only
*   **Input Shape:** `[B_local * S * K, D_latent]`
*   **Weight Shape per Device:** $E$ experts, each `[D_latent, 2 * (F_expert / TP)]` (stored in **MXFP4**)
    *   Total weights per device: `E * D_latent * 2 * (F_expert / TP)`
*   **Output Shape:** `[B_local * S * K, 2 * (F_expert / TP)]`
*   **FLOPS (TP):** `4 * B_local * S * K * D_latent * (F_expert / TP)`
*   **Memory Reads (Bytes):**
    *   Activation Read: `(B_local * S * K * D_latent) * bytes_act`
    *   Weight Read: `E * D_latent * 2 * (F_expert / TP) * bytes_weight_routed`  *(where `bytes_weight_routed = 0.5` for MXFP4)*
*   **Memory Writes (Bytes):** `B_local * S * K * 2 * (F_expert / TP) * bytes_act`

#### Expert Parallelism (EP - Default: Option B Replicated-Input EP)
Under Replicated-Input EP along the data parallel axis:
*   **Weights per Device:** `E / EP` experts, each `[D_latent, 2 * F_expert]`
    *   Total weights per device: `(E / EP) * D_latent * 2 * F_expert`
*   **Tokens Processed per Device:** `B_local * S * K` tokens
*   **FLOPS (EP):** `4 * (B_local * S * K) * D_latent * F_expert`
*   **Memory Reads (Bytes):**
    *   Activation Read: `(B_local * S * K * D_latent) * bytes_act`
    *   Weight Read: `(E / EP) * D_latent * 2 * F_expert * bytes_weight_routed`  *(0.5 Bytes/param for MXFP4)*
*   **Memory Writes (Bytes):** `(B_local * S * K) * 2 * F_expert * bytes_act`

#### Expert Parallelism (EP - Option A Dispatch-based EP, Training Only)
*   **Tokens Processed per Device:** `(B_local * S * K) / EP`
*   **FLOPS (EP Dispatch):** `4 * ((B_local * S * K) / EP) * D_latent * F_expert`
*   **Memory Reads:** `((B_local * S * K) / EP) * D_latent * bytes_act + (E / EP) * D_latent * 2 * F_expert * bytes_weight_routed`
*   **Memory Writes:** `((B_local * S * K) / EP) * 2 * F_expert * bytes_act`

---

### 4.4. SiTU-GLU Activation Function

Applies the Sigmoid Tanh Unit Gated Linear Unit with soft-clamping:

$$h(g, u) = \left[ \beta_1 \tanh\left(\frac{g}{\beta_1}\right) \cdot \sigma(g) \right] \odot \left[ \beta_2 \tanh\left(\frac{u}{\beta_2}\right) \right]$$

*   **Constants:** $\beta_1 = 4.0$, $\beta_2 = 25.0$ (output bounded to $[-100, 100]$).
*   **Input Shapes:** Gate $g \in [N_{\text{tokens}}, F_{\text{local}}]$, Up $u \in [N_{\text{tokens}}, F_{\text{local}}]$ where $F_{\text{local}} = F_{\text{expert}} / \text{TP}$.
*   **Output Shape:** $h \in [N_{\text{tokens}}, F_{\text{local}}]$
*   **FLOPS:** $\approx 8 \times N_{\text{tokens}} \times F_{\text{local}}$ (tanh, sigmoid, scaling, and elementwise multiplication).
*   **Memory Transfer:** When fused in VMEM / SRAM with W1 and W2, intermediate HBM read/write is 0 Bytes.

---

### 4.5. Routed MoE W2 (Down Projection in Latent Space)

Projects tokens from expert intermediate dimension $F_{\text{expert}}$ back to the **latent space** $D_{\text{latent}}$.

#### Tensor Parallelism (TP) Only
*   **Input Shape:** `[B_local * S * K, F_expert / TP]`
*   **Weight Shape per Device:** $E$ experts, each `[F_expert / TP, D_latent]` (stored in **MXFP4**)
    *   Total weights per device: `E * (F_expert / TP) * D_latent`
*   **Output Shape:** `[B_local * S * K, D_latent]`
*   **FLOPS (TP):** `2 * B_local * S * K * (F_expert / TP) * D_latent`
*   **Memory Reads (Bytes):**
    *   Activation Read: `(B_local * S * K * (F_expert / TP)) * bytes_act`
    *   Weight Read: `E * (F_expert / TP) * D_latent * bytes_weight_routed`  *(0.5 Bytes/param for MXFP4)*
*   **Memory Writes (Bytes):** `B_local * S * K * D_latent * bytes_act`

#### Expert Parallelism (EP - Default: Option B Replicated-Input EP)
*   **Weights per Device:** `(E / EP) * F_expert * D_latent`
*   **FLOPS (EP):** `2 * (B_local * S * K) * F_expert * D_latent`
*   **Memory Reads (Bytes):**
    *   Activation Read: `((B_local * S * K) * F_expert) * bytes_act`
    *   Weight Read: `(E / EP) * F_expert * D_latent * bytes_weight_routed`  *(0.5 Bytes/param for MXFP4)*
*   **Memory Writes (Bytes):** `(B_local * S * K) * D_latent * bytes_act`

---

### 4.6. Expert Output Aggregation & Latent RMSNorm Pre-Projection

1.  **Weighted Sum Aggregation:**
    Combines the $K$ expert outputs for each token in the latent space using router softmax weights $s_k$:
    $$z_{\text{moe}} = \sum_{k=1}^K s_k \cdot \text{Expert}_k(z) \in \mathbb{R}^{B_{\text{local}} \times S \times D_{\text{latent}}}$$
    *   **FLOPS:** `2 * B_local * S * K * D_latent`
2.  **Latent RMSNorm:**
    Normalizes the combined latent representation:
    $$\hat{z}_{\text{moe}} = \text{RMSNorm}(z_{\text{moe}}) = \frac{z_{\text{moe}}}{\sqrt{\frac{1}{D_{\text{latent}}} \sum_{i=1}^{D_{\text{latent}}} z_{\text{moe}, i}^2 + \epsilon}} \odot \gamma$$
    *   **Weight Shape:** `[D_latent]` ($\gamma$ scale parameter in `bf16`)
    *   **FLOPS:** `3 * B_local * S * D_latent`
    *   **Memory Transfer:** Fused in VMEM with aggregation and Latent Up-Projection (0 extra HBM access).

---

### 4.7. Latent Up-Projection (`W_up_lat`)

Decompresses the aggregated, normalized latent representation back to the full model hidden dimension $D$.

#### Single-Device / Unsharded / Replicated TP
*   **Input Shape:** `[B_local, S, D_latent]`
*   **Weight Shape:** `[D_latent, D]` (stored in `bfloat16`)
*   **Output Shape:** `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * D_latent * D`
*   **Memory Reads (Bytes):**
    *   Activation Read: `B_local * S * D_latent * bytes_act`
    *   Weight Read: `D_latent * D * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * D * bytes_act`

#### Tensor Parallelism (TP Row-Parallel)
*   **Weight Shape:** `[D_latent / TP, D]`
*   **Output Shape:** `[B_local, S, D]` (Partial sum requiring **All-Reduce** across TP group)
*   **FLOPS:** `2 * B_local * S * (D_latent / TP) * D`
*   **Memory Reads:** `(B_local * S * (D_latent / TP)) * bytes_act + (D_latent / TP) * D * bytes_weight`
*   **Memory Writes:** `B_local * S * D * bytes_act`

---

### 4.8. Shared Experts (Full-Rank Parallel Path)

The Shared Expert path operates unconditionally on all tokens in the **uncompressed hidden dimension** $D$.

> [!NOTE]
> **Shared Expert Parallelism:** The Shared Expert always uses **Tensor Parallelism (TP)** across devices, even when Expert Parallelism (EP) is enabled for the Routed MoE.

#### A. Shared Expert W1 (Gate/Up Projection)
*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, 2 * (F_shared / TP)]` (stored in `bfloat16`)
*   **Output Shape:** `[B_local, S, 2 * (F_shared / TP)]`
*   **FLOPS:** `4 * B_local * S * D * (F_shared / TP)`
*   **Memory Reads (Bytes):** `(B_local * S * D) * bytes_act + D * 2 * (F_shared / TP) * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * 2 * (F_shared / TP) * bytes_act`

#### B. Shared Expert Activation (SiTU-GLU / SwiGLU)
*   **FLOPS:** `8 * B_local * S * (F_shared / TP)` (Fused in VMEM).

#### C. Shared Expert W2 (Down Projection)
*   **Input Shape:** `[B_local, S, F_shared / TP]`
*   **Weight Shape:** `[F_shared / TP, D]` (stored in `bfloat16`)
*   **Output Shape:** `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (F_shared / TP) * D`
*   **Memory Reads (Bytes):** `(B_local * S * (F_shared / TP)) * bytes_act + (F_shared / TP) * D * bytes_weight`
*   **Memory Writes (Bytes):** `B_local * S * D * bytes_act`

---

## 5. Distributed Parallelism & Collective Communications

**EP moved.** The expert-parallel collective inventory — Replicated-Input (the default),
Dispatch-based, and their payloads, groups, dtype sites and HBM effects — lives in
[`../../../parallelism/ep/ep.md`](../../../parallelism/ep/ep.md). The `D_latent` dispatch
width that makes this architecture's exchange half the size of a standard MoE's is recorded there
as the **latent dispatch** sub-variant of Option A.

**TP moved.** The partial sums from the Latent Up-Projection ($W_{\text{up\_lat}}$) and the Shared
Expert Down-Projection ($W_{2,\text{shared}}$) merge into a **single** reduction across the TP
group rather than two — that merge is this architecture's property and is worth knowing here, but
the collective's payload, dtype site, per-layer count and variant are priced in
[`../../../parallelism/tp/tp.md`](../../../parallelism/tp/tp.md).

---

## 6. Python Calculation Template

Always write and execute a Python script to compute roofline metrics. Do not perform arithmetic mentally.

```python
from cost_estimator import collective

def compute_latent_moe_roofline(
    b_local: int,
    s: int,
    d: int = 7168,
    d_latent: int = 3584,
    f_expert: int = 3072,
    f_shared: int = 6144,
    e_routed: int = 896,
    k_active: int = 16,
    tp: int = 1,
    ep: int = 1,
    bytes_act: float = 2.0,       # bfloat16
    bytes_weight: float = 2.0,    # bfloat16
    bytes_weight_mxfp4: float = 0.5, # mxfp4
    bw_hbm: float = 3.69e12,     # TPU v7x HBM BW per chiplet (Bytes/s)
    peak_tflops_mxfp4: float = 2307e12,   # TPU v7x peak FP8 TFLOPS per chiplet; MXFP4 not stated
    peak_tflops_bf16: float = 1153.5e12,  # TPU v7x peak BF16 TFLOPS per chiplet
):
    results = {}

    # 1. Latent Down-Projection (D -> D_latent)
    flops_down = 2 * b_local * s * d * d_latent
    bytes_down = (b_local * s * d + b_local * s * d_latent) * bytes_act + (d * d_latent) * bytes_weight
    t_comp_down = flops_down / peak_tflops_bf16 * 1000.0
    t_mem_down = bytes_down / bw_hbm * 1000.0
    results["latent_down_proj"] = {
        "GFlops": flops_down / 1e9,
        "memory_transfer (MB)": bytes_down / 1e6,
        "execution_time (ms)": max(t_comp_down, t_mem_down),
        "bound": "Compute" if t_comp_down >= t_mem_down else "Memory",
    }

    # 2. Router Gating (D -> E)
    flops_router = 2 * b_local * s * d * e_routed
    bytes_router = (b_local * s * d + b_local * s * e_routed) * bytes_act + (d * e_routed) * bytes_weight
    t_comp_router = flops_router / peak_tflops_bf16 * 1000.0
    t_mem_router = bytes_router / bw_hbm * 1000.0
    results["router_gating"] = {
        "GFlops": flops_router / 1e9,
        "memory_transfer (MB)": bytes_router / 1e6,
        "execution_time (ms)": max(t_comp_router, t_mem_router),
        "bound": "Compute" if t_comp_router >= t_mem_router else "Memory",
    }

    # 3. Routed Experts W1 (Gate/Up with SiTU-GLU in Latent Space)
    # Total tokens = b_local * s * k_active
    tokens_routed = b_local * s * k_active
    f_exp_local = f_expert // tp
    e_local = e_routed // ep
    flops_w1 = 4 * tokens_routed * d_latent * f_exp_local
    bytes_w1 = (tokens_routed * d_latent + 2 * tokens_routed * f_exp_local) * bytes_act + \
               (e_local * d_latent * 2 * f_exp_local) * bytes_weight_mxfp4
    t_comp_w1 = flops_w1 / peak_tflops_mxfp4 * 1000.0
    t_mem_w1 = bytes_w1 / bw_hbm * 1000.0
    results["routed_w1_gate_up"] = {
        "GFlops": flops_w1 / 1e9,
        "memory_transfer (MB)": bytes_w1 / 1e6,
        "execution_time (ms)": max(t_comp_w1, t_mem_w1),
        "bound": "Compute" if t_comp_w1 >= t_mem_w1 else "Memory",
    }

    # 4. Routed Experts W2 (Down Projection: F_expert -> D_latent)
    flops_w2 = 2 * tokens_routed * f_exp_local * d_latent
    bytes_w2 = (tokens_routed * f_exp_local + tokens_routed * d_latent) * bytes_act + \
               (e_local * f_exp_local * d_latent) * bytes_weight_mxfp4
    t_comp_w2 = flops_w2 / peak_tflops_mxfp4 * 1000.0
    t_mem_w2 = bytes_w2 / bw_hbm * 1000.0
    results["routed_w2_down"] = {
        "GFlops": flops_w2 / 1e9,
        "memory_transfer (MB)": bytes_w2 / 1e6,
        "execution_time (ms)": max(t_comp_w2, t_mem_w2),
        "bound": "Compute" if t_comp_w2 >= t_mem_w2 else "Memory",
    }

    # 5. Latent RMSNorm Pre-Projection & Aggregation (Fused)
    flops_norm = 2 * tokens_routed * d_latent + 3 * b_local * s * d_latent
    bytes_norm = (b_local * s * d_latent * 2) * bytes_act
    t_comp_norm = flops_norm / peak_tflops_bf16 * 1000.0
    t_mem_norm = bytes_norm / bw_hbm * 1000.0
    results["latent_rmsnorm"] = {
        "GFlops": flops_norm / 1e9,
        "memory_transfer (MB)": bytes_norm / 1e6,
        "execution_time (ms)": max(t_comp_norm, t_mem_norm),
        "bound": "Compute" if t_comp_norm >= t_mem_norm else "Memory",
    }

    # 6. Latent Up-Projection (D_latent -> D)
    flops_up = 2 * b_local * s * d_latent * d
    bytes_up = (b_local * s * d_latent + b_local * s * d) * bytes_act + (d_latent * d) * bytes_weight
    t_comp_up = flops_up / peak_tflops_bf16 * 1000.0
    t_mem_up = bytes_up / bw_hbm * 1000.0
    results["latent_up_proj"] = {
        "GFlops": flops_up / 1e9,
        "memory_transfer (MB)": bytes_up / 1e6,
        "execution_time (ms)": max(t_comp_up, t_mem_up),
        "bound": "Compute" if t_comp_up >= t_mem_up else "Memory",
    }

    # 7. Shared Expert (D -> 2*F_shared -> D)
    f_sh_local = f_shared // tp
    flops_sh_w1 = 4 * b_local * s * d * f_sh_local
    bytes_sh_w1 = (b_local * s * d + 2 * b_local * s * f_sh_local) * bytes_act + (d * 2 * f_sh_local) * bytes_weight
    flops_sh_w2 = 2 * b_local * s * f_sh_local * d
    bytes_sh_w2 = (b_local * s * f_sh_local + b_local * s * d) * bytes_act + (f_sh_local * d) * bytes_weight
    flops_sh = flops_sh_w1 + flops_sh_w2
    bytes_sh = bytes_sh_w1 + bytes_sh_w2
    t_comp_sh = flops_sh / peak_tflops_bf16 * 1000.0
    t_mem_sh = bytes_sh / bw_hbm * 1000.0
    results["shared_expert"] = {
        "GFlops": flops_sh / 1e9,
        "memory_transfer (MB)": bytes_sh / 1e6,
        "execution_time (ms)": max(t_comp_sh, t_mem_sh),
        "bound": "Compute" if t_comp_sh >= t_mem_sh else "Memory",
    }

    return results
```

---

## 7. Concrete Parameter Specification: Kimi K3 MoE Layer

For Kimi K3 (Layers 1 to 92):

| Parameter | Notation | Numerical Value | Description |
| :--- | :--- | :--- | :--- |
| **Model Hidden Dim** | $D$ | 7,168 | Model hidden representation dimension |
| **Latent Bottleneck Dim** | $D_{\text{latent}}$ | 3,584 | Compressed latent space dimension ($D / 2$) |
| **Routed Experts** | $E_{\text{routed}}$ | 896 | Total pool of routed experts |
| **Active Experts / Token** | $K$ | 16 | Experts activated per token via Quantile Balancing |
| **Expert Intermediate Dim** | $F_{\text{expert}}$ | 3,072 | Intermediate dimension per routed expert |
| **Expert Weight Precision** | `dtype` | `MXFP4` (0.5 B) | 4-bit microscopically scaled expert parameters |
| **Shared Experts** | $E_{\text{shared}}$ | 2 | Full-rank parallel shared experts ($D=7,168$) |
| **Shared Intermediate Dim** | $F_{\text{shared}}$ | 6,144 | Combined intermediate dimension ($2 \times 3,072$) |
| **Shared Weight Precision** | `dtype` | `bfloat16` (2.0 B) | 16-bit uncompressed shared expert weights |
| **Activation Function** | — | `SiTU-GLU` | Sigmoid Tanh Unit GLU ($\beta_1=4.0, \beta_2=25.0$) |
| **Latent Normalization** | — | `Latent RMSNorm` | RMSNorm on $D_{\text{latent}}=3,584$ before up-projection |

---

## References

*   [Tensor Dimension Notation](../../../NOTATION.md)
*   [Common Requirements](../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../TYPE_PROMOTION.md)
*   [Standard MoE Roofline Skill](../moe/moe.md)
*   [FFN Roofline Skill](../ffn/ffn.md)
*   [Kimi K3 Architecture Specification](../../architecture/kimi/K3.md)
*   [Collective Cost Model](../../../COLLECTIVE.md)
*   TPU v7x Specification: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
