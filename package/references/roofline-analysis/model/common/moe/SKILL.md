---
name: moe-roofline-model
description: Guidelines for calculating Roofline performance estimation for a Mixture of Experts (MoE) component.
references:
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/model/common/REQUIREMENTs.md
- references/roofline-analysis/parallelism/ep/SKILL.md
- references/roofline-analysis/parallelism/tp/SKILL.md
---

# Mixture of Experts (MoE) Component

This document describes the operations involved in a Mixture of Experts (MoE) layer for roofline performance estimation. It details both a Shared Expert and a Routed MoE component.

## 1. Required References and Notation

You **MUST** use the standard tensor dimension naming conventions defined in `../../NOTATION.md`.

*   `B`: Batch size
*   `S`: Sequence length
*   `D`: Hidden dimension
*   `F_shared`: Intermediate dimension of the Shared Expert
*   `F_expert`: Intermediate dimension of a routed expert (MoE)
*   `E`: Number of experts
*   `K`: Top-K experts selected per token
*   `P`: Number of participating devices

You must read and follow:

*   [Common Requirements]({{reference:roofline-analysis/model/common/REQUIREMENTs.md}})
*   [Type Promotion Rules]({{reference:roofline-analysis/TYPE_PROMOTION.md}})


## 2. Component Specifications and Assumptions

*   **Parallel Branches:** The Shared Expert is a parallel branch with the RoutedMoE sharing the same input. We need to consider that during visualization. However, note that the actual computation for these branches does not proceed in parallel.
*   **Token distribution:** For the Routed MoE, assume tokens are distributed evenly across all `E` experts. Each expert processes `C = (B * S * K) / E` tokens. The total number of tokens processed across all experts is `B * S * K`.
*   **Tensor Parallelism (TP):** When TP is enabled (`P > 1`):
    *   **RoutedMoE** is sharded along its intermediate dimension `F_expert` of
        *each* expert is sharded across `P` devices (`F_expert / P`).
    *   **SharedExpert** is sharded along its intermediate dimension `F_shared` (`F_shared / P`) by default, though it can be replicated for simplicity.
    *   **Router (Gating)** is replicated across devices by default unless specified otherwise.
    *   The **All-reduce** operation at the end of the RoutedMoE and the SharedExpert (if sharded) can be merged into a single All-reduce.
*   **Expert Parallelism (EP):** Each device holds `E / EP` routed experts. The shapes and FLOPs
    that follow are what this file owns; **which collectives EP issues, their payloads and dtype
    sites, and what EP does to HBM are owned by
    [`../../../parallelism/ep/SKILL.md`]({{reference:roofline-analysis/parallelism/ep/SKILL.md}}).**
*   **Ignored Costs:** Ignore the computational cost of `topk`. **Do not ignore `token permute`
    and `token unpermute`** — they were on this list until 2026-09-19 and at prefill they are the
    largest memory term in the layer. See §3.6.

## 3. Component Operations

### 3.1 Router (Gating)
The Router computes the gating logits for all tokens.

*   **Input Shape:** `[B, S, D]`
*   **Weight Shape:** `[D, E]` (Replicated across devices)
*   **Output Shape:** `[B, S, E]`
*   **FLOPS:** `2 * B * S * D * E`
*   **Memory Activation Read:** `B * S * D * bytes_per_elem`
*   **Memory Weights Read:** `D * E * bytes_per_elem`
*   **Memory Writes:** `B * S * E * bytes_per_elem`
*   **Collective Communication:** None (Replicated)

### 3.2 RoutedMoE W1 (Gate/Up Projection)
Each token is routed to `K` experts. Total tokens processed is `B * S * K`.

#### Tensor Parallelism (TP)

*   **Input Shape:** `C` tokens per expert -> effectively `[B_local * S * K, D]` across all experts.
*   **Weight Shape:** `E` experts, each `[D, F_expert / TP]` (or `[D, 2 *
    F_expert / TP]` if gated) -> Total weights per device: `E * D * (F_expert /
    TP)` (Multiply by 2 if gated).
*   **Output Shape:** `[B_local * S * K, F_expert / TP]` across all experts.
*   **FLOPS (TP):** `2 * B_local * S * K * D * (F_expert / TP)` (Multiply by 2
    if it's a gated activation like SwiGLU which has two W1 matrices).
*   **Memory Activation Read (fused operations two weight in gated cases):**
    `B_local * S * K * D * bytes_per_elem`.
*   **Memory Weights Read:** `E * D * (F_expert / TP) * (2 if gated else 1) *
    bytes_per_elem`
*   **Memory Writes:** `B_local * S * K * (F_expert / TP) * bytes_per_elem`.
    (Note: Gated intermediate output is written in activation precision, e.g.,
    `bf16` / 2B, to preserve precision for activation function).
    **One `F`-wide tensor, not two** -- this assumes the activation is a fused
    epilogue: gate and up stay in VMEM, `SiLU(gate)*up` is applied there, and no
    separate SwiGLU op follows. Unfused it is `2 *` this, plus a SwiGLU pass reading
    both halves back. `roofline.py` models both (`Model.ffn_fused_activation`) and
    got this wrong -- writing `2 *` **and** charging the SwiGLU -- until 2026-09-22.

#### Expert Parallelism (EP)

*   **Input Shape:** effectively `[(B_local * S * K) / EP, D]` tokens processed per device.
*   **Weight Shape:** `E / EP` experts per device, each `[D, F_expert]` (or `[D,
    2 * F_expert]` if gated) -> Total weights per device: `(E / EP) * D *
    F_expert` (Multiply by 2 if gated).
*   **Output Shape:** `[(B_local * S * K) / EP, F_expert]` local output on one
    device.
*   **FLOPS (EP along the same axis as DP):** `2 * ((B_local * S * K)) * D *
    F_expert` (Multiply by 2 if it's a gated activation like SwiGLU which has
    two W1 matrices).
*   **Memory Activation Read (EP along the same axis as DP):** `B_local * S *
    K * D * bytes_per_elem`.
*   **Memory Weights Read:** `(E / EP) * D * F_expert * bytes_per_elem`.
    (Multiply by 2 if it's a gated activation like SwiGLU which has two W1
    matrices).
*   **Memory Writes (EP along the same axis as DP):** `Intermediate` =
    `B_local * S * K * F_expert * bytes_per_elem`. (Note: Gated intermediate
    output is written in activation precision, e.g., `bf16` / 2B, to preserve
    precision for activation function).

### 3.3 RoutedMoE W2 (Down Projection)

#### Tensor Parallelism (TP) only

*   **Input Shape:** `[B_local * S * K, F_expert / TP]` across all experts.
*   **Weight Shape:** `E` experts, each `[F_expert / TP, D]` -> Total weights
    per device: `E * (F_expert / TP) * D`.
*   **Output Shape:** `[B_local * S * K, D]` (Pre-unpermute).
*   **FLOPS (TP):** `2 * B_local * S * K * (F_expert / TP) * D`.
*   **Memory Activation Read:** `B_local * S * K * (F_expert / TP) *
    bytes_per_elem`.
*   **Memory Weights Read:** `E * (F_expert / TP) * D * bytes_per_elem`.
*   **Memory Writes:** `B_local * S * K * D * bytes_per_elem`.

#### Expert Parallelism (EP)

*   **Input Shape:** `[(B_local * S * K) / EP, F_expert]` tokens processed per
    device.
*   **Weight Shape:** `E / EP` experts per device, each `[F_expert, D]` -> Total
    weights per device: `(E / EP) * F_expert * D`.
*   **Output Shape:** `[(B_local * S * K) / EP, D]` local output on one device (Pre-unpermute).
*   **FLOPS (EP along the same axis as DP):** `2 * (B_local * S * K) *
    F_expert * D`.
*   **Memory Activation Reads:** `(B_local * S * K) * F_expert *
    bytes_per_elem`.
*   **Memory Weights Reads:** `(E / EP) * F_expert * D * bytes_per_elem`.
*   **Memory Writes:** `(B_local * S * K) * D * bytes_per_elem`.

### 3.4 SharedExpert W1 (Gate/Up Projection)
The shared expert processes all `B * S` tokens. **Note: The Shared Expert always uses Tensor Parallelism (TP), even when Expert Parallelism (EP) is enabled for the Routed MoE.**

*   **Input Shape:** `[B_local, S, D]`
*   **Weight Shape:** `[D, F_shared / TP]`
*   **Output Shape:** `[B_local, S, F_shared / TP]`
*   **FLOPS:** `2 * B_local * S * D * (F_shared / TP)` (Multiply by 2 if gated).
*   **Memory Activation Reads:** `B_local * S * D * bytes_per_elem`.
*   **Memory Weights Reads:** `D * (F_shared / TP) * bytes_per_elem`.
*   **Memory Writes:** `B_local * S * (F_shared / TP) * bytes_per_elem`.

### 3.5 SharedExpert W2 (Down Projection)

*   **Input Shape:** `[B_local, S, F_shared / TP]`
*   **Weight Shape:** `[F_shared / TP, D]`
*   **Output Shape:** `[B_local, S, D]`
*   **FLOPS:** `2 * B_local * S * (F_shared / TP) * D`.
*   **Memory Activation Reads:** `B_local * S * (F_shared / TP) *
    bytes_per_elem`.
*   **Memory Weights Reads:** `(F_shared / TP) * D * bytes_per_elem`.
*   **Memory Writes:** `B_local * S * D * bytes_per_elem`.

### 3.6 Token Permute / Unpermute

The grouped expert GEMM needs the tokens for one expert to be contiguous, so each MoE layer
gathers rows into expert-major order before it and scatters them back after. This is **local HBM
traffic** — it is not a collective and appears in no `collective.py` call, which is why it is easy
to leave out of a roofline entirely.

It is not small. At prefill it is the largest single memory term in the layer:

| per MoE layer, per device | formula | `B_local * S` = 256 | `B_local * S` = 4096 |
| --- | --- | ---: | ---: |
| Expert weights read | `(E / EP) * n_mat * D * F_expert * bytes_weight` | 604 MB | 604 MB |
| **Permute + unpermute** | `4 * B_local * S * K * D * bytes_act` | 101 MB (17%) | **1,611 MB (267%)** |

*(`E`=256, `EP`=16, `K`=8, `D`=6144, `F_expert`=2048, `n_mat`=3, fp8 weights, bf16 activations.)*

*   **Memory Transfer:** `4 * B_local * S * K * D * bytes_per_elem` — read plus write for the
    gather, read plus write for the scatter-add.
*   **FLOPS:** none. Pure data movement.
*   **Collective Communication:** None.

**The coefficient is the uncertainty, not the existence.** If the grouped GEMM gathers its rows
directly instead of materialising a permuted buffer, `4` falls toward `2` or to zero; if the
scatter-add target is a wider buffer it rises. Treat `4` as a modelled upper-middle estimate and
the term as **present**. Dropping it is the one choice the table above rules out.

There is evidence the permute path has more than one implementation and switches between them by
size: the Qwen3.5-397B benchmark configs set `USE_MOE_SPARSE_CORE=1` and
`ONEHOT_MOE_PERMUTE_THRESHOLD=32768`. Reading what those select is the way to replace the modelled
`4` with a measured one, and would likely give two coefficients rather than one.

> [!IMPORTANT]
> **Do not double-charge under dispatch-based EP.** When the MoE runs dispatch-based EP, the
> permutation is normally fused into building the all-to-all send buffers, so this row is already
> paid inside the dispatch. It stands alone under replicated-input EP and at `EP = 1`. See
> [`../../../parallelism/ep/SKILL.md`]({{reference:roofline-analysis/parallelism/ep/SKILL.md}}) for which variant is in
> play.

### 3.7 MoE Collectives

**Moved.** The EP collective inventory — Replicated-Input (the default), Dispatch-based, and
their payloads, groups, dtype sites and HBM effects — lives in
[`../../../parallelism/ep/SKILL.md`]({{reference:roofline-analysis/parallelism/ep/SKILL.md}}). This file covers the
operations; that one covers what the sharding costs to communicate.

The `TP`-only case (no EP) is priced in
[`../../../parallelism/tp/SKILL.md`]({{reference:roofline-analysis/parallelism/tp/SKILL.md}}), which also states the rule
that matters when both are on: **when EP spans the TP ranks, the MoE layer's trailing reduction is
EP's, not TP's** — counting both double-charges the largest collective in the layer.
