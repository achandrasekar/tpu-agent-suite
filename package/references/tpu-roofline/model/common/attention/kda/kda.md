---
name: rooflining-kda-attention
description: Calculates FLOPS and memory transfer for Kimi Delta Attention (KDA) linear attention layers. Use when estimating KDA performance, analyzing recurrent state updates, chunkwise prefill solves, or decode token generation under Tensor Parallelism.
---

# Kimi Delta Attention (KDA) Performance Estimation

Estimate the roofline performance of Kimi Delta Attention (KDA) linear attention layers (such as the 69 KDA layers in Kimi K3) by calculating FLOPS, HBM memory transfer, and collective communication latency for each sub-operation.

Report per-operation level metrics because different sub-operations (e.g., GEMM projections vs. depthwise 1D convolution vs. core recurrent delta update kernel vs. output projection All-Reduce) exhibit vastly different computational bottlenecks (compute-bound vs. memory-bandwidth-bound vs. collective-bound).

## Requirements

You must read and follow:

*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Collective Cost](../../../../COLLECTIVE.md)

## Workflow

1.  **Identify Dimensions:** Determine the following dimensions from the notation:
    *   `B`: Global Batch Size
    *   `B_local`: Local Batch Size per device (`B / DP`)
    *   `S`: Sequence Length (context length `S` for Prefill, or sequence context `S` for Decode)
    *   `D`: Hidden Dimension (model dimension, e.g., 7168 in Kimi K3)
    *   `H_q`: Number of Query heads (e.g., 96 in Kimi K3)
    *   `H_s`: Number of Key/Value State heads (e.g., 96 in Kimi K3, or fewer under GHS / GQA)
    *   `D_k`: Head dimension of Query and Key (e.g., 128)
    *   `D_v`: Head dimension of Value (e.g., 128)
    *   `D_conv`: Combined convolution channel dimension (`H_q * D_k + H_s * D_k + H_s * D_v`, e.g., 36864)
    *   `K_conv`: Convolution kernel size (typically `K_conv = 4`)
    *   `D_decay_lora`: Low-rank latent dimension for decay generation (e.g., 128 in Kimi K3)
    *   `C`: Chunk Size for chunk-wise parallel Prefill updates (typically `C = 64`)
    *   `TP`: Tensor Parallelism degree
2.  **Determine Precision & Apply Type Promotion:** Read [TYPE_PROMOTION.md](../../../../TYPE_PROMOTION.md) to calculate byte-size configurations:
    *   **Activations:** `bfloat16` (2 bytes per element) by default.
    *   **Weights:** `bfloat16` (2 bytes per element) or `fp8` (1 byte per element).
    *   **Recurrent State Matrix ($S$):** Stored in **`float32`** (4 bytes per element) in HBM / VMEM for numerical stability of the linear delta recurrence.
    *   **Convolution History Cache:** `bfloat16` (2 bytes per element).
3.  **Identify Execution Regime:**
    *   **Prefill Mode (Chunk-wise Parallel Forward):** Sequence length `S > 1` divided into $S / C$ chunks. Employs intra-chunk strictly lower-triangular causal solves and inter-chunk boundary state propagation.
    *   **Decode Mode (Autoregressive Generation):** Single-token step ($S = 1$). Performs step-by-step $O(1)$ recurrent state update and Group-Head State readout against the persistent $H_s \times D_k \times D_v$ state cache.
    *   **Training Forward Pass:** Mathematically identical to Prefill mode, with full activation storage for backward gradient propagation.
4.  **Write and Run Python Script for Calculations:** Write and execute a Python script to perform all numerical calculations (FLOPS, memory transfer bytes, latencies). Do not perform roofline arithmetic mentally.
5.  **Calculate Roofline & Total Latency:** Combine calculated values with TPU/GPU hardware specs (peak TFLOPS, HBM bandwidth, and collective cost estimator) to compute execution times.
6.  **Output Format:** Output execution results in standard markdown table and JSON format as described in [Common Requirements](../../REQUIREMENTs.md).

--------------------------------------------------------------------------------

## Operations & Calculations

A KDA linear attention block consists of 6 main operational steps:

```
[Input X] ---> (1. QKV Proj) ---------> (2. Causal Conv1D + SiLU) -\
          ---> (1.5 Decay & Gate Proj) -----------------------------> [3. KDA Gated Delta Kernel] ---> (4. Gated RMSNorm) ---> [5. Out Proj]
```

### 1. QKV Projections (`in_proj_qkv` / `q_proj`, `k_proj`, `v_proj`)

Projects hidden states into Query, Key, and Value spaces prior to 1D convolution.

*   **Weight Shapes:**
    *   $W_q$: `[D, H_q * D_k]`
    *   $W_k$: `[D, H_s * D_k]`
    *   $W_v$: `[D, H_s * D_v]`
    *   Combined / Fused Weight: `[D, H_q * D_k + H_s * D_k + H_s * D_v]`
*   **Activation Shapes:**
    *   Input: `[B_local, S, D]`
    *   $Q$ Output: `[B_local, S, H_q * D_k]`
    *   $K$ Output: `[B_local, S, H_s * D_k]`
    *   $V$ Output: `[B_local, S, H_s * D_v]`
*   **FLOPS:** `2 * B_local * S * D * (H_q * D_k + H_s * D_k + H_s * D_v)`
*   **HBM Transfer (Bytes):**
    *   Activation Read: `B_local * S * D * bytes_per_activation`
    *   Weight Read: `D * (H_q * D_k + H_s * D_k + H_s * D_v) * bytes_per_weight`
    *   Activation Write: `B_local * S * (H_q * D_k + H_s * D_k + H_s * D_v) * bytes_per_activation`

### 2. Decay, Beta, and Output Gate Projections

Projects hidden states to generate the channel-wise retention decay $\alpha$, write strength $\beta$, and multiplicative output gate $g$.

#### A. Channel-Wise Decay Projection ($\alpha$ / $g_{\text{raw}}$ - Low-Rank Bottleneck)
In frontier architectures like Kimi K3, the decay gate $g_{\text{raw}}$ is parameterized via a low-rank bottleneck $D \to D_{\text{decay\_lora}} \to H_s \times D_k$ (e.g., $7168 \to 128 \to 12288$):
*   **Down-Projection:**
    *   Weight Shape: `[D, D_decay_lora]`
    *   FLOPS: `2 * B_local * S * D * D_decay_lora`
    *   HBM Transfer: Read $X$ (`B_local * S * D * bytes`), Read $W_{\text{down}}$ (`D * D_decay_lora * bytes`), Write $c_{\text{decay}}$ (`B_local * S * D_decay_lora * bytes`)
*   **Up-Projection:**
    *   Weight Shape: `[D_decay_lora, H_s * D_k]`
    *   FLOPS: `2 * B_local * S * D_decay_lora * (H_s * D_k)`
    *   HBM Transfer: Read $c_{\text{decay}}$ (`B_local * S * D_decay_lora * bytes`), Read $W_{\text{up}}$ (`D_decay_lora * H_s * D_k * bytes`), Write $g_{\text{raw}}$ (`B_local * S * H_s * D_k * bytes`)
*   *Note: If computed via full-rank direct projection, FLOPS is `2 * B_local * S * D * (H_s * D_k)`.*

#### B. Beta Write Rate Projection ($\beta$)
Projects hidden states to per-head write strength $\beta_t = \text{sigmoid}(W_\beta x_t) \in (0, 1)$:
*   **Weight Shape:** `[D, H_s]`
*   **Activation Shape:** `[B_local, S, H_s]`
*   **FLOPS:** `2 * B_local * S * D * H_s`
*   **HBM Transfer:** Read $X$ (`B_local * S * D * bytes`), Read $W_\beta$ (`D * H_s * bytes`), Write $\beta$ (`B_local * S * H_s * bytes`)

#### C. Output Gate Projection ($g$)
Projects hidden states to full-rank multiplicative output gate $g_t = \text{sigmoid}(W_g x_t)$:
*   **Weight Shape:** `[D, H_q * D_v]`
*   **Activation Shape:** `[B_local, S, H_q * D_v]`
*   **FLOPS:** `2 * B_local * S * D * (H_q * D_v)`
*   **HBM Transfer:** Read $X$ (`B_local * S * D * bytes`), Read $W_g$ (`D * H_q * D_v * bytes`), Write $g$ (`B_local * S * H_q * D_v * bytes`)

### 3. Causal Depthwise 1D Convolution

Applied across the projected Q, K, and V channels ($D_{\text{conv}} = H_q \cdot D_k + H_s \cdot D_k + H_s \cdot D_v$) with kernel size $K_{\text{conv}} = 4$, followed by a SiLU activation.

#### A. Prefill Mode (Multi-Token Sequence Forward)
*   **Input / Output Shape:** `[B_local, S, D_conv]`
*   **Weight Shape:** `[D_conv, 1, K_conv]` (Depthwise)
*   **FLOPS:** `2 * B_local * S * D_conv * K_conv`
*   **HBM Transfer (Bytes):**
    *   *Standalone / Unfused:* Read $X$ (`B_local * S * D_conv * bytes`), Read $W_{\text{conv}}$ (`D_conv * K_conv * bytes`), Write Output (`B_local * S * D_conv * bytes`).
    *   *Fused (kept in VMEM with QKV Proj & SiLU):* Intermediate HBM transfers are bypassed (0 bytes).

#### B. Decode Mode (Single-Token Autoregressive Generation, $S = 1$)
*   **Input Token Shape:** `[B_local, 1, D_conv]`
*   **Conv State (Cache) Shape:** `[B_local, K_conv - 1, D_conv]`
*   **FLOPS:** `2 * B_local * D_conv * K_conv`
*   **HBM Transfer (Bytes):**
    *   *State in HBM:* Read Input (`B_local * D_conv * bytes`), Read Conv Cache (`B_local * (K_conv - 1) * D_conv * bytes`), Write Conv Cache (`B_local * (K_conv - 1) * D_conv * bytes`), Write Output (`B_local * D_conv * bytes`). Total: `2 * B_local * D_conv * K_conv * bytes`.
    *   *State in VMEM / SRAM:* Only transfer input read and output write (`2 * B_local * D_conv * bytes`).

### 4. Q/K Normalization & Key L2 Normalization

*   **Q/K RMSNorm:** Per-head RMSNorm on Q and K:
    *   FLOPS: `B_local * S * (H_q * D_k + H_s * D_k)`
*   **Key L2 Normalization:** Normalizes key vector $k_t / \|k_t\|_2$ along the head dimension $D_k$ to enforce delta-rule contractive stability:
    *   FLOPS: `2 * B_local * S * H_s * D_k`
*   *Memory Transfer:* Fused in VMEM with the attention kernel (0 extra HBM transfer).

### 5. KDA Core Gated Delta Recurrent Kernel (Attention Engine)

The core recurrent update and retrieval kernel executes the error-correcting delta rule:

$$\hat{v}_t = k_t^T (\alpha_t \odot S_{t-1})$$
$$e_t = v_t - \hat{v}_t$$
$$S_t = \alpha_t \odot S_{t-1} + \beta_t k_t e_t^T$$
$$u_t^{(h)} = q_t^{(h)} S_t^{(h // G)}$$

Evaluated depending on the execution regime:

#### A. Prefill Mode (Chunk-wise Parallel Forward, Chunk Size $C = 64$)

Divides sequence length $S$ into $N_c = S / C$ non-overlapping chunks.

*   **Intra-Chunk & Inter-Chunk Calculations (per chunk of size $C$):**
    1.  **Log-Decay Prefix Sum & Cumulative Decay ($\Gamma$):**
        *   $\Gamma_{n, t} = \sum_{\tau=1}^t \log \alpha_{n, \tau}$
        *   FLOPS: `B_local * S * H_s * D_k`
    2.  **Intra-Chunk Causal Gram Interaction Matrix ($G \in \mathbb{R}^{C \times C}$):**
        *   $G_{t, s} = \beta_s (k_t^T k_s) \odot \exp(\Gamma_t - \Gamma_s)$ for $t > s$.
        *   FLOPS: `2 * B_local * H_s * (S / C) * (C * (C - 1) / 2) * D_k` $\approx$ `B_local * S * C * H_s * D_k`
    3.  **Intra-Chunk Triangular Solve $(I + G) E = V - U_{\text{prev}} S_{n-1}$:**
        *   State prediction from incoming boundary state $U_{\text{prev}} S_{n-1}$ ($U_{\text{prev}} = K \odot \exp(\Gamma - \Gamma_0) \in [C, D_k]$):
            *   FLOPS: `2 * B_local * H_s * (S / C) * C * D_k * D_v` = `2 * B_local * S * H_s * D_k * D_v`
        *   Strictly lower-triangular substitution / solve for error matrix $E \in [C, D_v]$:
            *   FLOPS: `B_local * H_s * (S / C) * C^2 * D_v` = `B_local * S * C * H_s * D_v`
        *   Effective value scaling $V_{\text{eff}} = \text{diag}(\beta) E$: `B_local * S * H_s * D_v`
    4.  **Intra-Chunk Attention Output ($O_{\text{intra}}$):**
        *   Intra-chunk score $A_{\text{intra}} = \text{tril}((Q K^T) \odot \exp(\Gamma_Q - \Gamma_K))$:
            *   FLOPS: `2 * B_local * H_q * (S / C) * (C^2 / 2) * D_k` = `B_local * S * C * H_q * D_k`
        *   Intra-chunk accumulation $O_{\text{intra}} = A_{\text{intra}} V_{\text{eff}}$:
            *   FLOPS: `2 * B_local * H_q * (S / C) * (C^2 / 2) * D_v` = `B_local * S * C * H_q * D_v`
    5.  **Inter-Chunk Retrieval Output ($O_{\text{inter}}$):**
        *   $O_{\text{inter}} = (Q \odot \exp(\Gamma - \Gamma_0)) S_{n-1}$:
            *   FLOPS: `2 * B_local * H_q * (S / C) * C * D_k * D_v` = `2 * B_local * S * H_q * D_k * D_v`
    6.  **Inter-Chunk Boundary State Update ($S_n$):**
        *   $S_n = \exp(\Gamma_C - \Gamma_0) \odot S_{n-1} + \sum_{t=1}^C (\exp(\Gamma_C - \Gamma_t) \odot k_t) v_{\text{eff}, t}^T$:
            *   FLOPS: `2 * B_local * H_s * (S / C) * C * D_k * D_v` = `2 * B_local * S * H_s * D_k * D_v`

*   **Total Prefill Kernel FLOPS:**
    $$\text{FLOPS}_{\text{prefill}} = B_{\text{local}} \cdot S \cdot \left[ C \cdot (H_s D_k + H_s D_v + H_q D_k + H_q D_v) + 4 H_s D_k D_v + 2 H_q D_k D_v \right]$$

*   **Prefill HBM Transfer (Bytes - Fused Kernel):**
    *   Activation Read: `B_local * S * (H_q * D_k + H_s * D_k + H_s * D_v) * bytes_per_activation` + gates.
    *   Activation Write: `B_local * S * H_q * D_v * bytes_per_activation`.
    *   Boundary State Read / Write (if saved across chunks to HBM): `2 * B_local * (S / C) * H_s * D_k * D_v * 4` (fp32).

#### B. Decode Mode (Autoregressive Generation, $S = 1$)

Operates sequentially token-by-token with constant $O(1)$ step time.

*   **FLOPS per Step:**
    1.  Channel-wise Decay: $S_{\text{decayed}} = \alpha_t \odot S_{t-1} \implies$ `B_local * H_s * D_k * D_v`
    2.  Retrieve Prediction: $\hat{v}_t = k_t^T S_{\text{decayed}} \implies$ `2 * B_local * H_s * D_k * D_v`
    3.  Compute Error: $e_t = v_t - \hat{v}_t \implies$ `B_local * H_s * D_v`
    4.  State Matrix Update: $S_t = S_{\text{decayed}} + \beta_t k_t e_t^T \implies$ `2 * B_local * H_s * D_k * D_v`
    5.  Group-Head State Readout: $u_t = q_t S_t \implies$ `2 * B_local * H_q * D_k * D_v`
    *   **Total Decode Kernel FLOPS:**
        $$\text{FLOPS}_{\text{decode}} = (5 H_s + 2 H_q) \cdot B_{\text{local}} \cdot D_k \cdot D_v + B_{\text{local}} \cdot H_s \cdot D_v$$
        *(When $H_q = H_s$, exactly $7 \cdot B_{\text{local}} \cdot H \cdot D_k \cdot D_v$).*

*   **Decode HBM Transfer (Bytes - Fused Kernel):**
    *   Input Tokens Read ($q, k, v, \alpha, \beta$): `B_local * (H_q * D_k + H_s * D_k + H_s * D_v) * bytes_per_activation` (negligible).
    *   Recurrent State Read ($S_{t-1}$): `B_local * H_s * D_k * D_v * 4` (fp32, 4 bytes/element).
    *   Recurrent State Write ($S_t$): `B_local * H_s * D_k * D_v * 4` (fp32, 4 bytes/element).
    *   Output Write ($u_t$): `B_local * 1 * H_q * D_v * bytes_per_activation`.
    *   *Total Recurrent State HBM Transfer:* $2 \cdot B_{\text{local}} \cdot H_s \cdot D_k \cdot D_v \times 4 \text{ Bytes}$.

### 6. Gated RMSNorm

Applies headwise RMSNorm on the core attention output and modulates it with the multiplicative output gate $g$:

$$o_t = \text{sigmoid}(g_t) \odot \text{RMSNorm}(u_t)$$

*   **Activation Shapes:** Input `[B_local, S, H_q * D_v]`, Gate `[B_local, S, H_q * D_v]`, Output `[B_local, S, H_q * D_v]`
*   **FLOPS:** `B_local * S * H_q * D_v` (variance reduction, normalization scale, sigmoid gate multiplication).
*   **HBM Transfer (Bytes):**
    *   *Standalone:* Read $u$ (`B_local * S * H_q * D_v * bytes`), Read $g$ (`B_local * S * H_q * D_v * bytes`), Write $o$ (`B_local * S * H_q * D_v * bytes`).
    *   *Fused in VMEM:* Intermediate $u$ transfer is eliminated.

### 7. Output Projection (`out_proj` / `o_proj`) & Collective Communication

Projects the concatenated gated attention output back to the model hidden dimension $D$.

*   **Weight Shape:** `[H_q * D_v, D]` (or `[H_q_local * D_v, D]` under TP)
*   **Activation Shapes:** Input `[B_local, S, H_q * D_v]` -> Output `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (H_q * D_v) * D`
*   **HBM Transfer (Bytes):**
    *   Activation Read: `B_local * S * (H_q * D_v) * bytes_per_activation`
    *   Weight Read: `(H_q * D_v) * D * bytes_per_weight`
    *   Activation Write: `B_local * S * D * bytes_per_activation`
*   **Collective Communication:** The row-parallel output projection's partial sums require a
    reduction across the TP group. Payload, dtype site, per-layer count and variant are priced in
    [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md).

--------------------------------------------------------------------------------

## Parallelism Strategies

> **Partitioning is not defined here.** How TP shards this op, and the per-device symbols below (`H_q_local`, `H_kv_local`, `F_local`, `D_conv_local`), are defined in
> [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md) §3b. This section states only the resulting per-device FLOPs and memory.

### Tensor Parallelism (TP)

Under standard Tensor Parallelism:

*   **Query Heads Sharding:** $H_{q,\text{local}} = H_q / \text{TP}$
*   **State Heads Sharding:** $H_{s,\text{local}} = \max(1, H_s / \text{TP})$
*   **Convolution Channel Sharding:** $D_{\text{conv},\text{local}} = D_{\text{conv}} / \text{TP} = (H_{q,\text{local}} D_k + H_{s,\text{local}} D_k + H_{s,\text{local}} D_v)$
*   **Projections:**
    *   Per-device weight shape for `out_proj` is `[H_q_local * D_v, D]`; its output is a partial
        sum needing a TP reduction, priced in [`../../../../parallelism/tp/tp.md`](../../../../parallelism/tp/tp.md).
*   **Core KDA Kernel & Gated RMSNorm:** Compute purely locally per device on local heads ($H_{q,\text{local}}, H_{s,\text{local}}$). **No cross-device communication** is required inside the attention recurrence.
*   **Recurrent State Footprint per Device:**
    $$\text{State Size per Device} = B_{\text{local}} \cdot H_{s,\text{local}} \cdot D_k \cdot D_v \times 4 \text{ Bytes}$$

--------------------------------------------------------------------------------

## References

*   [Common Requirements](../../REQUIREMENTs.md)
*   [Type Promotion Rules](../../../../TYPE_PROMOTION.md)
*   [Tensor Dimension Notation](../../../../NOTATION.md)
*   TPU Specifications: the TPU7x knowledge-base page `hw-tpu-7x`, obtained through tpu-knowledge-base
*   [Collective Cost](../../../../COLLECTIVE.md)
*   [Conv1D Roofline Skill](../../conv1d/conv1d.md)
*   [Kimi K3 Architecture Specification](../../../architecture/kimi/K3.md)
