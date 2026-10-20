---
title: "KDA context parallelism: affine state summaries instead of token transfer"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_fwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA context parallelism: affine state summaries instead of token transfer

## Definition
With context parallelism (CP), one logical sequence is split across ranks along time. Stage 1+2
can run on rank-local tokens, but Stage 3 needs the true state entering the rank. The forward
design doc's key observation is that the per-chunk update is affine in the chunk-start state:
`S_{c+1} = Phi_c S_c + Psi_c`, with `Phi_c = Diag(gamma_C) - kg^T W` and `Psi_c = kg^T U`.
Affine maps compose (`Phi_ba = Phi_b Phi_a`, `Psi_ba = Phi_b Psi_a + Psi_b`), so a whole rank's
tokens fold into a fixed-size pair `(M_local, S_ext_local)`, a `K x K` and a `K x V` matrix. The
ranks all-gather these pairs, and each rank composes its upstream ranks' pairs in order to
recover its own initial state. "Communication volume is independent of sequence length": the
payload is `O(KV + K^2)` per batch, head and rank, not `O(T)`. The backward doc mirrors this in
the opposite direction. Each rank summarizes its first real local segment into `(dS_ext, dM)`,
and those are merged from downstream ranks to upstream.

## In tokamax (grounded)
- **Forward.** [`_prepare_cp_initial_state`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L352 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_prepare_cp_initial_state")
  is the whole bridge between Stage 2 and Stage 3.
  1. [`chunk_gated_delta_rule_fwd_h_pre_process`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L300 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#chunk_gated_delta_rule_fwd_h_pre_process")
     scans only the final local segment, accumulating `S_ext <- Phi S_ext + Psi` and
     `M <- Phi M` in float32 VMEM scratch. Only the trailing segment can continue onto the next
     rank.
  2. Two [`all_gather_into_tensor`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/cp_utils.py#L155 "tokamax/_src/ops/experimental/kda/cp_utils#all_gather_into_tensor") calls run along
     the CP mesh axis, one for `S_ext` and one for `M`.
  3. [`_merge_initial_state`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/cp_utils.py#L330 "tokamax/_src/ops/experimental/kda/cp_utils#_merge_initial_state") applies the upstream
     chain for each batch element. The result is written into slot 0 of a zero
     `[B, N, H, K, V]` float32 initial state.
- **Backward.** [`chunk_gated_delta_rule_bwd_dhu_pre_process`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L260 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#chunk_gated_delta_rule_bwd_dhu_pre_process")
  produces `(dS_ext, dM)`. These are packed into a single all-gather and merged in
  [`_merge_dht`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/cp_utils.py#L418 "tokamax/_src/ops/experimental/kda/cp_utils#_merge_dht") using the `post_num_ranks` /
  `is_last_rank` metadata that the forward pass kept. The merged `dht` goes into the last real
  local segment slot before the fused backward kernel runs. Its kernel,
  [`_chunk_gated_delta_rule_bwd_dhu_pre_process_kernel`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L61 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#_chunk_gated_delta_rule_bwd_dhu_pre_process_kernel"),
  is the one KDA kernel with hand-written DMA: `pltpu.make_async_copy`, semaphores and
  double-buffered VMEM inputs.
- **Contract.** [`ContextParallelMetadata`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/cp_utils.py#L60 "tokamax/_src/ops/experimental/kda/cp_utils#ContextParallelMetadata") carries
  the mesh, the axis name and the rank-chain fields. Rank boundaries are derived from rank-local
  `segment_ids`. Active CP forbids an external `initial_state` and `output_final_state=True`,
  requires `max_num_segments`, and requires `K` and `V` to be multiples of 128. These are
  checked in [`check_inputs_support`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L112 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#check_inputs_support").

## Why it matters / when it applies
- CP is what lets KDA scale to sequences longer than one chip's HBM or time budget, without
  sending token-level state between chips. All summaries and merges stay in float32, so error
  does not grow through low-precision products across many ranks.
- The bottleneck is not the collective. In the historical profiles (forward on v7 with
  `cp_size=4`, backward on v6e and v7x), the all-gather is small and latency-bound. The
  rank-local summary scan is sequential, so it caps scaling: the forward pass measured 1.54x on
  4 ranks (38% efficiency). Both docs point optimization at that pre-process: fewer re-reads, a
  more parallel chain product, or fusing it into an adjacent stage.
- The pinned tests cover only float32 CP with `B = 1`. bfloat16 CP and CP with `B > 1` are
  listed as validation gaps, not as unsupported features.

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (CP path in code),
  [splash-attention](../concepts/splash-attention.md) (ring attention, the softmax-attention
  approach to sequence parallelism on TPU), [fused-moe](../concepts/fused-moe.md) (another
  communication-aware TPU kernel).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-chunked-forward-pipeline](kda-chunked-forward-pipeline.md),
  [kda-chunked-backward-fusion](kda-chunked-backward-fusion.md),
  [kda-variable-length-alignment](kda-variable-length-alignment.md).

## Source
Extracted from `docs/ops/kda/chunk_kda_fwd_design_doc.md` (Sections 4.2, 4.5, 7.1.3-7.1.5,
9.1.8) and `docs/ops/kda/chunk_kda_bwd_design_doc.md` (Sections 2.4, 3.5, 4.3, 5.2), kept in
place.
