<!--
This document is programmatically parsed by Python scripts. Please follow these formatting rules when making edits:

- Table column layout: | Notation | Alternative Names | Description |
- Enclose all symbols and alternative names in backticks (`symbol`)
-->

# Tensor Dimension Notation for Transformer Models

This document defines standard notations for tensor dimensions used in various transformer architectures, including Dense, Mixture-of-Experts (MoE), and different attention variants (MHA, MQA, GQA).

## Base Dimensions

| Notation   | Alternative Names                 | Description          |
| :--------- | :-------------------------------- | :------------------- |
| `B`        | `batch_size`, `N`                 | Global Batch size    |
| `B_local`  | `local_batch_size`                | Local Batch size     |
:            :                                   : (per device)         :
| `S`        | `seq_len`, `T`, `L_seq`           | Sequence length      |
:            :                                   : (input sequence      :
:            :                                   : length)              :
| `T`        | `tgt_len`                         | Target sequence      |
:            :                                   : length (during       :
:            :                                   : training/generation) :
| `D`        | `d_model`, `hidden_size`, `H`     | Hidden dimension     |
:            :                                   : (model dimension)    :
| `F`        | `d_ff`, `intermediate_size`,      | Intermediate         |
:            : `H_ffn`                           : dimension in FFN /   :
:            :                                   : MLP                  :
| `F_shared` | `shared_expert_intermediate_size` | Intermediate         |
:            :                                   : dimension of the     :
:            :                                   : Shared Expert (MoE)  :
| `F_expert` | `moe_intermediate_size`,          | Intermediate         |
:            : `expert_intermediate_size`        : dimension of a       :
:            :                                   : routed expert (MoE)  :
| `V`        | `vocab_size`                      | Vocabulary size      |
| `L`        | `num_layers`,                     | Number of layers     |
:            : `num_hidden_layers`, `N_layers`   :                      :

## Attention Dimensions (MHA, MQA, GQA)

| Notation | Alternative Names | Description |
| `H_q` | `h_q`, `num_attention_heads` | Number of Query heads |
| `H_kv` | `h_kv`, `num_key_value_heads` | Number of Key/Value heads |
| `D_h` | `d`, `d_kv`, `size_per_head` | Head dimension (dimension per head; formula: `D` / `H_q`) |
| `G` | — | Group size (number of query heads per KV head; formula: `H_q` / `H_kv`) |
| `D_rope` | `rotary_dim`, `partial_rotary_dim` | Rotary (RoPE) sub-dimension of each head under partial RoPE (formula: `partial_rotary_factor` * `D_h`). The remaining `D_h` - `D_rope` channels carry no positional rotation. Equals `D_h` when RoPE is applied to the full head |
| `A_gate` | `attn_output_gate` | Attention output-gate multiplier on the query side of the fused QKV projection. `1` when the layer has no output gate; `2` when a gate the same width as Q is emitted alongside it, making the fused QKV output width (`A_gate` * `H_q` + 2 * `H_kv`) * `D_h` |
| `H_idx` | `indexer_n_heads` | Number of indexer query heads (for DSA) |
| `D_idx` | `indexer_head_dim` | Head dimension for indexer query and key (for DSA) |
| `K_idx` | `indexer_topk` | Number of tokens selected by the indexer (for DSA) |

### Attention Variants Relationship

*   **Multi-Head Attention (MHA)**: `H_q` = `H_kv` (or `G` = 1)
*   **Multi-Query Attention (MQA)**: `H_kv` = 1 (or `G` = `H_q`)
*   **Grouped-Query Attention (GQA)**: 1 < `H_kv` < `H_q` (typically `H_kv` is a divisor of `H_q`)

## Multi-Head Latent Attention (MLA) Dimensions

| Notation | Alternative Names | Description |
| `H_q` | `h_q`, `num_attention_heads` | Number of Query heads |
| `H_kv` | `h_kv`, `num_key_value_heads` | Number of Key/Value heads |
| `D_c` | `kv_lora_rank` | KV compression latent dimension |
| `D_c_q` | `q_lora_rank` | Query compression latent dimension |
| `D_v` | `v_head_dim` | Value head dimension |
| `D_q_nope` | `q_nope_dim`, `qk_nope_head_dim` | Query NOPE (Non-Positional) head dimension |
| `D_q_rope` | `q_rope_dim`, `qk_rope_head_dim` | Query RoPE (Positional) head dimension |

## Linear & Delta Attention Dimensions (GDN, KDA)

| Notation | Alternative Names | Description |
| `H_s` | `num_state_heads`, `h_s`, `H_kv` | Number of Key/Value state heads |
| `D_k` | `key_dim`, `d_k`, `D_h_qk` | Key head dimension |
| `D_v` | `value_dim`, `d_v`, `D_h_v` | Value head dimension |
| `C` | `chunk_size` | Chunk size for chunkwise parallel linear recurrence |
| `D_decay_lora` | `decay_rank`, `lora_rank_decay` | Low-rank latent dimension for decay generation |

## Mixture of Experts (MoE) Dimensions

| Notation | Alternative Names | Description |
| `E` | `num_experts` | Number of experts |
| `K` | `top_k` | Top-K experts selected per token |
| `C` | `expert_capacity` | Capacity factor / Expert capacity |
| `D_latent` | `latent_dim`, `d_latent` | Latent bottleneck dimension for MoE down-projection |
| `F_expert` | `expert_intermediate_size` | Intermediate hidden dimension per routed expert |
| `E_routed` | `num_routed_experts` | Total number of routed experts |
| `E_shared` | `num_shared_experts` | Total number of shared experts |
| `F_dense` | `dense_intermediate_size` | Intermediate dimension for dense FFN layers |
| `L_kda` | `num_kda_layers` | Number of Kimi Delta Attention layers |
| `L_mla` | `num_mla_layers` | Number of Multi-Head Latent Attention layers |

## Convolution Dimensions

| Notation | Alternative Names | Description |
| `D_conv` | `conv_dim`, `in_channels`, `out_channels` | Convolution channel dimension |
| `K_conv` | `kernel_size`, `k`, `conv_kernel_size` | Convolution kernel size |

## Weight Matrices

| Notation | Alternative Names | Shape | Description |
| `W_q` | — | `[D, H_q * D_h]` (or `[D, D]`) | Query projection weight |
| `W_k` | — | `[D, H_kv * D_h]` | Key projection weight |
| `W_v` | — | `[D, H_kv * D_h]` | Value projection weight |
| `W_o` | — | `[H_q * D_h, D]` (or `[D, D]`) | Attention output projection weight |
| `W_alpha` | `W_g`, `W_decay` | `[D, H_s * D_k]` (or `[D, D_decay_lora]`) | Channel-wise decay / retention gate weight |
| `W_beta` | `W_b` | `[D, H_s]` (or `[D, 1]`) | Write strength / learning rate projection weight |
| `W_gate` | `W_g_out` | `[D, H_q * D_v]` (or `[D, F]`) | Output gate projection / FFN gate projection weight |
| `W_down_lat` | `W_down_latent` | `[D, D_latent]` | Latent MoE down-projection weight |
| `W_up_lat` | `W_up_latent` | `[D_latent, D]` | Latent MoE up-projection weight |
| `W_1` | `W_up` | `[D, F]` (or `[D_latent, F_expert]`) | FFN / Expert up-projection weight |
| `W_2` | `W_down` | `[F, D]` (or `[F_expert, D_latent]`) | FFN / Expert down-projection weight |
| `W_conv` | — | `[D_conv, 1, K_conv]` | Convolution weight matrix (for depthwise 1D conv) |

## Common Tensor Shapes

### Standard Input/Output

*   Input Tokens: `[B, S]`
*   Embeddings / Hidden States: `[B, S, D]`

### Attention Projections (Dense)

*   Query (Q): `[B, S, H_q, D_h]` (or flattened: `[B, S, H_q * D_h]` which is `[B, S, D]`)
*   Key (K): `[B, S, H_kv, D_h]`
*   Value (V): `[B, S, H_kv, D_h]`
*   Attention Scores: `[B, H_q, S, S]` (or reshaped: `[B, H_kv, G, S, S]`)
*   Attention Output: `[B, S, H_q, D_h]` -> projected to `[B, S, D]`

### KV Cache (for autoregressive decoding)

*   Cached K: `[B, H_kv, S, D_h]` (or `[B, S, H_kv, D_h]`)
*   Cached V: `[B, H_kv, S, D_h]` (or `[B, S, H_kv, D_h]`)

### Linear & Delta Attention Tensors (GDN, KDA)

*   Recurrent State Matrix $S$: `[B, H_s, D_k, D_v]` (stored in fp32)
*   Decay Gate $\alpha$: `[B, S, H_s, D_k]` (or `[B, H_s, S, D_k]`)
*   Write Rate $\beta$: `[B, S, H_s, 1]` (or `[B, H_s, S, 1]`)
*   Output Gate $g$: `[B, S, H_q, D_v]`

### FFN (Feed-Forward Network) / MLP

*   Dense MLP Gate/Up projection input: `[B, S, D]`
*   Dense MLP Intermediate state: `[B, S, F]`
*   Dense MLP Output: `[B, S, D]`

### Mixture of Experts (MoE)

*   Gate logits: `[B * S, E]` (or `[B, S, E]`)
*   Routing weights: `[B * S, K]`
*   Selected Expert Indices: `[B * S, K]`
*   **Expert Input (with capacity)**: `[E, C, D]` (where `C` is expert capacity, often computed as `(B * S * K) / E * capacity_factor`)
*   **Expert Weights (MLP)**:
    *   Gate/Up W1: `[E, D, F]`
    *   Down W2: `[E, F, D]`
*   **Expert Output**: `[E, C, D]`

### Convolution Tensors

*   Input / Output Activations: `[B, S, D_conv]`
*   Conv State (cache for decoding): `[B, K_conv - 1, D_conv]`

## Hardware and Distributed Communication Dimensions

| Notation | Alternative Names | Description |
| `P` | `world_size` | Number of participating devices (parallelism degree) |
| `TP` | `n_dev`, `tensor_parallel_size` | Tensor Parallelism degree / number of devices |
| `DP` | `data_parallel_size` | Data Parallelism degree |
| `FSDP` | `fsdp_size`, `fully_sharded_data_parallel_size` | Fully Sharded Data Parallelism degree |
| `EP` | `expert_parallel_size` | Expert Parallelism degree |
| `CP` | `context_parallel_size` | Context Parallelism degree |
| `SP` | `use_sequence_parallel_moe` | Sequence Parallelism. **A boolean, not a degree** — it consumes no ranks (`world = TP x PCP x PP x DP`) and reuses the TP group. Modelled as a variant of TP, see `parallelism/tp/tp.md` §5. Not to be confused with Ulysses-style sequence sharding through attention, which is context parallelism (`PCP`/`DCP`) |
| `M` | `payload_size`, `Local Device Memory` | Total message size or local target tensor size to be reduced/gathered (in Bytes) |
| `N_links` | `#link` | Number of accessible independent communication links per device |
| `BW` | `bandwidth` | Effective interconnect bandwidth per link |
| `BW_bi` | `bidirectional BW` | Bidirectional bandwidth per link (used when a Ring algorithm is applicable) |
| `BW_uni` | `unidirectional BW` | Unidirectional bandwidth per link (used when Ring is not applicable) |
| `BW_ici` | `BW_ICI`, `Beta`, `ici_bandwidth` | Interconnect (ICI) single-link bidirectional bandwidth |
| `BW_hbm` | `BW_HBM`, `hbm_bandwidth` | HBM memory bandwidth per chip |
| `t_net_lat` | `alpha`, `L_net` | Interconnect network link latency (startup cost/delay per hop/link) |
| `T_allreduce` | `t_comm`, `allreduce_time` | Total execution time for an All-Reduce collective operation |

