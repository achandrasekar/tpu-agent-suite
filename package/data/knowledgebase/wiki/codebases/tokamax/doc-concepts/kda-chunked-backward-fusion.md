---
title: "KDA chunked backward: six stages fused into three Pallas TPU kernels"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_bwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA chunked backward: six stages fused into three Pallas TPU kernels

## Definition
Mathematically, the KDA backward pass has six stages: (1) recompute forward intermediates,
(2) differentiate the intra term `Aqk @ v_new`, (3) run the state gradient `dh` backward along
the chunk axis, (4) backpropagate the WY representation, (5) backpropagate the intra-chunk
attention, and (6) apply a reverse cumsum to the gate gradient. The design doc's central claim
is that "most stages of the KDA backward have low arithmetic intensity and, when split apart
individually, are easily HBM-bandwidth bound". So on the saved-`h` path the six stages run as
three kernels: **recompute** (`w, qg, kg, v_new`), **dAv** (`dAqk, dv`) and
**dhu/WY/intra/cumsum** (`dq, dk, dv, db, dg, dh0`). "The main benefit is not reducing the
mathematical operations, but reducing the reads and writes at stage boundaries."

## In tokamax (grounded)
- [`chunk_kda_bwd_custom`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L1481 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#chunk_kda_bwd_custom")
  orchestrates the pass. It aligns `do`, restores CP metadata, recovers the intermediates (see
  [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md)), and then calls:
  1. [`fused_recompute_w_u_vnew_from_h_pallas`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L655 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#fused_recompute_w_u_vnew_from_h_pallas"),
     the chunk-parallel recompute.
  2. [`chunk_kda_bwd_dAv_kernel`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L1357 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#chunk_kda_bwd_dAv_kernel"):
     `dAqk = s * tril(do v_new^T)` and `dv_new = tril(Aqk)^T do`. The scale is applied once,
     here, and the stored `Aqk` already includes it.
  3. Optionally, the CP state-gradient merge (see
     [kda-context-parallel-state-summaries](kda-context-parallel-state-summaries.md)).
  4. [`_fused_dhu_wy_intra_cumsum_pallas_jit`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L1080 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#_fused_dhu_wy_intra_cumsum_pallas_jit")
     runs in reverse chunk order (the chunk axis is `arbitrary`; heads and batch are parallel).
     It keeps `dh` in a `[MB, K, V]` VMEM scratch and fuses the `dhu` recurrence, the WY
     backward ([`compute_wy_backward`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L570 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#compute_wy_backward")),
     the intra backward ([`compute_intra_backward`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L810 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#compute_intra_backward"))
     and the reverse cumsum ([`compute_reverse_cumsum_dg`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L954 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#compute_reverse_cumsum_dg")).
     No separate cumsum kernel is launched.
  5. Optionally, [`kda_gate_bwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel.py#L387 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_bwd_kernel#kda_gate_bwd")
     maps `dg` back through the in-kernel gate activation to the raw gate, `a_log` and
     `delta_time_bias`. Then [`l2norm_bwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L61 "tokamax/_src/ops/experimental/kda/utils#l2norm_bwd") runs for
     `dq / dk`, and [`_unalign_output`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/utils.py#L272 "tokamax/_src/ops/experimental/kda/utils#_unalign_output") restores the
     variable-length layout.
- **TPU adaptations in code.** The cumulative gate is kept in log2 space, so decays are `exp2`
  and combine by adding exponents. float32 inputs use `jax.lax.Precision.HIGHEST`, bfloat16
  uses the default precision with float32 accumulation, and the reverse-cumsum dot always uses
  `HIGHEST` because `dg` contributions can nearly cancel. Mini-batch `MB` (head tiles per program)
  comes from VMEM estimates: [`estimate_mini_batch`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/common.py#L71 "tokamax/_src/ops/experimental/kda/common#estimate_mini_batch") for
  dAv, and kernel-local heuristics elsewhere. The intra backward normalizes `2^(g_r - g_j)` around
  a per-sub-block reference gate. Data movement is `BlockSpec`-driven, with no hand-written DMA
  except in the CP pre-process.

## Why it matters / when it applies
- **Padding policy.** "Padding itself is not necessarily a bad thing." The `[BT, BT]` `Aqk / Akk`
  tiles accept hardware-alignment padding. A packed layout is used only when the HBM traffic it
  saves exceeds the extra reshape and gather cost. Non-CP backward supports unaligned `K / V`;
  CP requires multiples of 128.
- **Numerics.** `compute_intra_backward` casts `dg_acc + dg_intra` to the input dtype and back
  to float32 before the reverse cumsum. The doc treats this truncation as delivered behaviour:
  keep it in refactors, and choose reference-comparison tolerances with it in mind.
- **Historical performance** (v6e, bf16, 5 packed segments, H=16, T=8192, K=V=128). The recompute
  and dAv kernels reach about 92% and 88% of HBM roofline. The fused kernel reaches 66%, because
  it mixes a recurrence, matmuls and reductions. On v7x all three run faster but at lower
  utilization (63%, 57%, 33%). The doc's first follow-up is to retune tiles, `MB` and VMEM
  pressure for v7. The numbers predate the current adapter and must be rerun before being used
  as a baseline.

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (backward
  kernels in code), [op-framework](../concepts/op-framework.md) (custom VJP op),
  [splash-attention](../concepts/splash-attention.md) and
  [linear-softmax-cross-entropy](../concepts/linear-softmax-cross-entropy.md) (other fused TPU
  backward kernels), [mosaic-tpu-helpers](../concepts/mosaic-tpu-helpers.md) (dot precision).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md),
  [kda-wy-triangular-solve](kda-wy-triangular-solve.md),
  [kda-context-parallel-state-summaries](kda-context-parallel-state-summaries.md),
  [kda-variable-length-alignment](kda-variable-length-alignment.md),
  [benchmark-timing-methods](benchmark-timing-methods.md).

## Source
Extracted from `docs/ops/kda/chunk_kda_bwd_design_doc.md` (Sections 1-5), kept in place.
