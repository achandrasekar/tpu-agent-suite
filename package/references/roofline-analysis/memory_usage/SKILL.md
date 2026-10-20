---
name: memory-usage-roofline
description: >-
  Calculates LLM memory usage including per-device weight footprint summation, KV cache size per request,
  and maximum concurrent requests (local and global batch capacity) based on hardware HBM memory limits
  and distributed parallelism strategies (TP, PP, EP, DP).
references:
- references/roofline-analysis/NOTATION.md
---

# LLM Memory Usage & Capacity Calculation

This skill defines standard formulas and workflows to calculate memory consumption and serving capacity for Transformer model inference across distributed hardware (TPUs/GPUs).

Specifically, it computes:
1. **Model Weight Footprint per Device** ($M_{\text{weight\_device}}$)
2. **KV Cache Size per Request** ($M_{\text{kv\_request\_device}}$ and $M_{\text{kv\_request\_global}}$)
3. **Maximum Concurrent Requests (Serving Capacity)** ($B_{\text{local, max}}$ and $B_{\text{global, max}}$)

## Requirements & Notations

You **MUST** follow the standard tensor dimension notations defined in [NOTATION.md]({{reference:roofline-analysis/NOTATION.md}}).

For layer internal parameter shapes and weight details (Attention, FFN, MoE), refer to the respective module skills (`../model/common/...`).


---

## 1. Weight Usage per Device

The weight footprint on a single device is computed by summing the local parameter counts of non-layer modules (embeddings and LM head) and layer modules sharded across Pipeline Parallelism ($PP$), Tensor Parallelism ($TP$), and Expert Parallelism ($EP$).

### 1.1 Token Embedding & LM Output Head

*   **Token Embedding ($W_{\text{emb}}$)**: Shape `[V, D]`. If sharded by Vocabulary/Tensor Parallelism ($TP$):
    $$\text{Params}_{\text{emb, local}} = \frac{V \times D}{TP}$$
*   **LM Output Head ($W_{\text{head}}$)**: Shape `[V, D]`. If untied from embedding and sharded by $TP$:
    $$\text{Params}_{\text{head, local}} = \frac{V \times D}{TP}$$
    *(If tied with token embedding, $\text{Params}_{\text{head, local}} = 0$).*

### 1.2 Layer Weight Parameter Summation

Refer to the respective module skills under `../model/common/` to get local per-device parameter counts per layer:
*   $\text{Params}_{\text{attn, local}}$: Local parameters per layer for Attention (e.g., from `attention/qkv_proj` and `attention/out_proj`).
*   $\text{Params}_{\text{mlp, local}}$: Local parameters per layer for FFN / MLP or MoE (e.g., from `ffn` or `moe`).

Each Pipeline Parallelism stage holds $L_{\text{local}} = \frac{L}{PP}$ layers.

### 1.3 Total Weight Memory per Device Formula

Summing local parameter counts across embeddings, output head, and local layers:

$$\text{Params}_{\text{total, device}} = \text{Params}_{\text{emb, local}} + \text{Params}_{\text{head, local}} + \frac{L}{PP} \times \left( \text{Params}_{\text{attn, local}} + \text{Params}_{\text{mlp, local}} \right)$$

$$M_{\text{weight\_device}} = b_w \times \text{Params}_{\text{total, device}}$$

*(Convert Bytes to Gigabytes: $1 \text{ GB} = 10^9 \text{ Bytes}$ or $1 \text{ GiB} = 2^{30} \text{ Bytes}$).*

---

## 2. KV Cache Size per Request ($M_{\text{kv\_request}}$)

The Key-Value (KV) cache stores historical key and value activations across all transformer layers for a single sequence of length $S$ during autoregressive decoding.

### 2.1 Standard Attention (MHA, GQA, MQA)

For a sequence of length $S$:
*   **Cached Key & Value per layer**: Shape `[S, H_kv, D_h]` each.
*   **Elements per token per layer (global)**: $2 \times H_{kv} \times D_h$

#### Global KV Cache per Request (across all layers and heads)
$$\text{Elements}_{\text{kv\_global}} = 2 \times L \times S \times H_{kv} \times D_h$$
$$M_{\text{kv\_request\_global}} = b_{kv} \times 2 \times L \times S \times H_{kv} \times D_h$$

#### Local KV Cache per Request (per Device)
Sharded by Pipeline Parallelism ($L_{\text{local}} = \frac{L}{PP}$) and Tensor Parallelism ($H_{kv,\text{local}} = \max\left(1, \left\lceil \frac{H_{kv}}{TP} \right\rceil\right)$):

$$M_{\text{kv\_request\_device}} = b_{kv} \times 2 \times \frac{L}{PP} \times S \times H_{kv,\text{local}} \times D_h$$

*(Note: If $TP > H_{kv}$, KV heads are replicated across $TP / H_{kv}$ devices, so $H_{kv,\text{local}} = 1$).*

### 2.2 Multi-Head Latent Attention (MLA - DeepSeek)

MLA compresses the KV cache into a low-rank latent vector $D_c$ and a decoupled RoPE key vector $D_r$:
*   **Elements per token per layer (global)**: $D_c + D_r$

#### Global KV Cache per Request (MLA)
$$M_{\text{kv\_request\_global, MLA}} = b_{kv} \times L \times S \times (D_c + D_r)$$

#### Local KV Cache per Request per Device (MLA)
$$M_{\text{kv\_request\_device, MLA}} = b_{kv} \times \frac{L}{PP} \times S \times (D_c + D_r)$$

---

## 3. Maximum Concurrent Requests ($B_{\text{max}}$)

Given the total physical device HBM capacity $M_{\text{total}}$, compute how many concurrent requests (batch size) can be served simultaneously.

### 3.1 Device Memory Budget & Reserve

Total HBM is partitioned into three main components:
$$M_{\text{total}} = M_{\text{weight\_device}} + M_{\text{reserve\_device}} + M_{\text{kv\_total\_device}}$$

where $M_{\text{reserve\_device}}$ accounts for:
1.  **Framework & Runtime Overhead**: CUDA / XLA context, compiled kernel executables (~1–3 GB).
2.  **Collective Communication Buffers**: Workspace memory for AllReduce, AllGather, and AllToAll transfers (~0.5–2 GB).
3.  **Dynamic Activation Memory**: Temporary intermediate activations during forward pass execution.

$M_{\text{reserve\_device}}$ is modeled as a reserved fraction of HBM (typically 10%–20% of $M_{\text{total}}$) or estimated explicitly:
$$M_{\text{reserve\_device}} = \text{Overhead\_Fraction} \times M_{\text{total}} + M_{\text{activation\_temp}}$$

### 3.2 Available KV Cache Memory per Device

$$M_{\text{available\_device}} = M_{\text{total}} - M_{\text{weight\_device}} - M_{\text{reserve\_device}}$$

### 3.3 Maximum Concurrent Requests per Device ($B_{\text{local, max}}$)

The maximum number of concurrent sequences a single device can store in its HBM is:

$$B_{\text{local, max}} = \left\lfloor \frac{M_{\text{available\_device}}}{M_{\text{kv\_request\_device}}} \right\rfloor$$

### 3.4 Maximum Concurrent Requests across Cluster ($B_{\text{global, max}}$)

With Data Parallelism $DP$ (or $DP$ independent serving instances of the model):

$$B_{\text{global, max}} = B_{\text{local, max}} \times DP$$

---

## 4. Python Implementation Workflow

Always write and execute a Python script to compute exact memory footprints and capacity:
