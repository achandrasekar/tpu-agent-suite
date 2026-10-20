---
title: "KDA residual contract and manual rematerialization (KdaResiduals, rematerialize_for_backward)"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_bwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA residual contract and manual rematerialization (KdaResiduals, rematerialize_for_backward)

## Definition
The KDA custom VJP decides explicitly which forward intermediates go to HBM and which the
backward pass recomputes. The backward design doc sorts them into three groups. **Saved:** the
intra-chunk query-key matrix `Aqk`, the WY inverse `Akk`, and (by default) the chunk-start
hidden states `h [H, B, NT, K, V]`. **Recomputed in backward:** `w`, `qg`, `kg`, `v_new`.
**Optionally recomputed:** the post-cumsum gate (when the gate is activated in the kernel) and,
under `rematerialize_for_backward=True`, `h` and `v_new` themselves. The doc stresses that this
is "manual rematerialization within the Mosaic custom VJP, not a call to `jax.checkpoint` or
`jax.remat`". The forward doc states the invariant that makes this safe: both policies evaluate
the same forward map `F_fwd(Q, K, V, G, D_beta, S_0) = (O, S_NT)`. They differ "only in whether
selected matrices ... are retained ... or reconstructed".

## In tokamax (grounded)
- [`KdaResiduals`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_types.py#L37 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_types#KdaResiduals") is the typed
  contract. It holds the prepared (aligned, possibly L2-normalized) `q / k`, `v`, `beta`, the
  chosen gate form (`g_cumsum`, or `g_org` with `a_log` / `delta_time_bias`), `aqk`, `akk`,
  optional `h`, `initial_state`, `q_rstd / k_rstd`, and the alignment and CP metadata. `w`,
  `u`, `kg`, `qg` and `v_new` are deliberately absent.
- The knob is [`Config.rematerialize_for_backward`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L64 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#Config.rematerialize_for_backward")
  (default `False`), a field of the backend's `Config` rather than of the public API. In
  [`chunk_kda_fwd_custom`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L1425 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#chunk_kda_fwd_custom"),
  `save_for_backward = return_residuals and disable_recompute`, where `disable_recompute` is
  `not rematerialize_for_backward`. This matches the doc's `return_residuals and not
  config.rematerialize_for_backward`. `V_new` is never stored (`store_v_new=False`).
- **Backward, saved-`h` path.** [`fused_recompute_w_u_vnew_from_h_pallas`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L655 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#fused_recompute_w_u_vnew_from_h_pallas")
  rebuilds `u = Akk (v * beta)`, `w = Akk (k * beta * 2^g)`, `v_new = u - w h`, `qg` and `kg`
  chunk by chunk. Because `h` is given, there is no cross-chunk dependency, and `u` never leaves
  VMEM.
- **Backward, low-memory path.** [`_recompute_w_u_fwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L510 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#_recompute_w_u_fwd")
  followed by [`chunk_gated_delta_rule_fwd_h`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/common.py#L820 "tokamax/_src/ops/experimental/kda/common#chunk_gated_delta_rule_fwd_h") reruns
  the state recurrence to recover `h` and `v_new`. With in-kernel gate activation, both paths
  recompute the cumulative gate through [`kda_gate_chunk_cumsum`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L493 "tokamax/_src/ops/experimental/kda/utils#kda_gate_chunk_cumsum"),
  even when `g_cumsum` was saved.
- The orchestrator is [`chunk_kda_bwd_custom`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L1481 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#chunk_kda_bwd_custom"),
  reached from [`PallasMosaicTpuKimiDeltaAttentionVjp._fwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L487 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#PallasMosaicTpuKimiDeltaAttentionVjp._fwd").
  It consumes the residual copies rather than the original tensors that Tokamax's generic VJP
  replays. That way the backward sees exactly the aligned, normalized inputs that produced `o`.

## Why it matters / when it applies
- `h` is the largest residual: one `K x V` state per chunk, head and batch element.
  For long sequences with many heads, `rematerialize_for_backward=True` trades a serial
  recurrence in backward for a much smaller HBM footprint between forward and backward.
- With the default saved-`h` path, every backward kernel is chunk-parallel except the reverse
  state-gradient recurrence. The historical v6e profile shows the recompute kernel at about 92%
  of HBM roofline.
- One documented wiring quirk: with `return_residuals=False` and saving enabled, Stage 3+4
  still writes `h` and then discards it. Stage 1+2 also materializes `qg` when saving, but `qg`
  is not part of `KdaResiduals`.

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (residuals and VJP
  in code), [op-framework](../concepts/op-framework.md) (`return_residuals`, custom VJP
  plumbing), [linear-softmax-cross-entropy](../concepts/linear-softmax-cross-entropy.md)
  (another TPU kernel that saves a minimal residual and recomputes in backward).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-chunked-backward-fusion](kda-chunked-backward-fusion.md),
  [kda-chunked-forward-pipeline](kda-chunked-forward-pipeline.md),
  [standardized-benchmark-functions](standardized-benchmark-functions.md) (benchmarking
  `forward_res` and `vjp` modes).

## Source
Extracted from `docs/ops/kda/chunk_kda_bwd_design_doc.md` (Sections 1.1, 1.3, 2.1) and
`docs/ops/kda/chunk_kda_fwd_design_doc.md` (Section 4.3.4), kept in place.
