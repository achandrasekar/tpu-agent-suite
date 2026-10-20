---
title: "KDA variable-length input: per-segment chunk alignment under static TPU shapes"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_fwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA variable-length input: per-segment chunk alignment under static TPU shapes

## Definition
Packed variable-length batches are described by 1-indexed `segment_ids [B, T]`, where `0`
marks padding. Both KDA design docs handle them the same way. "TPU kernels require static
shapes. Variable-length sequences do not trigger a separate ragged kernel, but still use fixed
`[B, T]` tensors and a fixed `(H // MB, B, NT)` grid" (backward doc). Each segment is padded on
its own to a multiple of the chunk size, `L^_n = C * ceil(L_n / C)`, so "one chunk never crosses
two independent sequences". The total tail padding is below `N (C - 1)` for `N` segments. For
`B > 1`, each batch element is aligned separately and then padded to one static total, rather
than padding every segment to the longest sequence. The forward doc calls this a
"single-alignment contract". Alignment happens exactly once, before the kernels, and outputs
return to the original token order only at the end. As a result, forward outputs, residuals,
chunk indices and the backward pass all share one token mapping.

## In tokamax (grounded)
- [`PallasMosaicTpuKimiDeltaAttention._preprocess_inputs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L224 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#PallasMosaicTpuKimiDeltaAttention._preprocess_inputs")
  is the single alignment point. It derives `cu_seqlens` (with
  [`segment_ids_to_cu_seqlens`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L113 "tokamax/_src/ops/experimental/kda/utils#segment_ids_to_cu_seqlens"), or
  [`derive_context_parallel_metadata`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L69 "tokamax/_src/ops/experimental/kda/utils#derive_context_parallel_metadata") under CP),
  repacks `q / k / v / g / beta` with [`_align_seqs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L161 "tokamax/_src/ops/experimental/kda/utils#_align_seqs"),
  and builds the chunk -> segment map with [`prepare_chunk_indices`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L409 "tokamax/_src/ops/experimental/kda/utils#prepare_chunk_indices")
  and the aligned IDs with [`align_segment_ids`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L316 "tokamax/_src/ops/experimental/kda/utils#align_segment_ids"). It
  also applies the optional q/k L2 normalization ([`l2norm_fwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L55 "tokamax/_src/ops/experimental/kda/utils#l2norm_fwd"))
  to the *aligned* tensors and keeps `q_rstd / k_rstd`.
- **Padding semantics.** Padded data is zero, and a pre-activated gate padded with `0` is the
  identity decay (`alpha = 1`). With in-kernel gate activation, the aligned raw-gate padding is
  overwritten with `-1e4`, so the activation drives it towards the same identity behaviour.
- **Static bound.** `max_num_segments` fixes the compile-time shape of `cu_seqlens` and of the
  chunk map. It is required for variable-length input without an `initial_state`, and always
  required under CP. A tight value avoids work on empty segment slots.
- **Kernel use.** [`chunk_kda_fwd_h_o_varlen`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L1152 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#chunk_kda_fwd_h_o_varlen")
  uses the map to load the initial state at a segment's first chunk and write the final state
  at its last. Fixed-length input is the special case of one `[0, T]` segment per batch element.
  In backward, the last chunk of each real segment seeds the recurrence with `dht`, the first
  chunk emits `dh0`, and chunks with segment id 0 are padding.
  [`_unalign_output`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L272 "tokamax/_src/ops/experimental/kda/utils#_unalign_output") restores `o` and the token
  gradients to the original `[B, T]` layout.

## Why it matters / when it applies
- Per-segment alignment gives every chunk a regular `64 x K` tile with no ragged indexing in the
  kernel, at a padding cost of fewer than 64 tokens per segment. That is much less than padding
  every sequence to the batch maximum.
- Causality, segment boundaries and padding are "fixed components of the KDA recurrence rather
  than three independently supplied arbitrary masks". Changing one layer (for example the
  padding value) without the others breaks correctness.
- The aligned length is what the kernels actually process. In the forward doc's workload,
  logical `T = 8192` becomes physical `T = 9792` across 25 segments. Benchmarks and roofline
  numbers must report both lengths.

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (preprocessing
  and segment metadata in code), [batched-ragged-paged-attention](../concepts/batched-ragged-paged-attention.md)
  and [gated-delta-rule](../concepts/gated-delta-rule.md) (other ways of handling ragged batches
  on TPU).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-chunked-forward-pipeline](kda-chunked-forward-pipeline.md),
  [kda-context-parallel-state-summaries](kda-context-parallel-state-summaries.md),
  [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md).

## Source
Extracted from `docs/ops/kda/chunk_kda_fwd_design_doc.md` (Sections 2.2, 3.5, 5.2, 9.1.7) and
`docs/ops/kda/chunk_kda_bwd_design_doc.md` (Section 3.8), kept in place.
