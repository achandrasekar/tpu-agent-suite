---
title: "KDA intra-chunk WY solve: finite Neumann inversion on the MXU"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_fwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA intra-chunk WY solve: finite Neumann inversion on the MXU

## Definition
Inside one chunk, the delta rule still has a recurrence over positions: each token's write
depends on the state left by earlier tokens in the same chunk. Following the Kimi Linear paper
(Eq. 6-7), the forward design doc rewrites that recurrence as one unit lower-triangular system.
`L = StrictTril(D_beta K+ (K-)^T)`, where `K+ = Gamma * K` and `K- = K / Gamma`. The doc then
solves `A = (I + L)^-1` and applies it to all right-hand sides at once:
`[U W A] = A [D_beta V, D_beta K+, I]`. `W` and `U` are the compact "WY" factors that later
propagate the state. `Aqk = Tril((Gamma * q~)(K-)^T)` comes out of the same stacked matmul as
`L`. The TPU-specific part is how `(I + L)^-1` is computed for bfloat16. `L` is split into its
8 x 8 block-diagonal part `L_d` and the cross-block remainder `F`. Then
`P = (I + L_d)^-1 = sum_{p=0..7} (-L_d)^p` (exact, because `L_d^8 = 0`), `J = P F`, and
`(I + L)^-1 = (I + J)^-1 P`, where `(I + J)^-1` is another finite Neumann series. Both series
are evaluated by doubling. "In exact arithmetic, this expansion is an equivalent inverse ... it
is not a truncation of an infinite series."

## In tokamax (grounded)
- [`_fused_gate_intra_kernel`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L589 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_fused_gate_intra_kernel")
  builds `Aqk` and `L` in `BC = 16` row sub-blocks. One batched matmul over the stacked
  `[q_eg; k_eg]` operands yields the rows of both matrices. With `safe_gate`, the exponent is
  factored around the sub-block midpoint (`ref_idx = BC // 2`), which narrows the range of
  `exp2` on each side.
- bfloat16 (anything but float32) goes to
  [`_solve_unit_lower_triangular_neumann_batched`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L514 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_solve_unit_lower_triangular_neumann_batched"):
  8-token diagonal blocks, repeated squaring of `-L_d` for the block inverse, and a fused
  `P @ [F | rhs]` matmul. It returns both the solution and the inverse, all with float32
  accumulation.
- float32 goes to [`_solve_unit_lower_triangular_batched`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L456 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_solve_unit_lower_triangular_batched"):
  exact block forward substitution with 16-row blocks. The identity is appended as an extra
  right-hand side, so `Akk` comes out of the same solve.
- The inverse is stored as runtime `Akk`. The doc stresses that it is `(I + L)^-1`, not the
  paper's `M_c = (I + L)^-1 D_beta` and not the CP transition matrix `M_local`. Stage 3+4 does
  not consume it; it is a residual for the backward pass (see
  [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md)).

## Why it matters / when it applies
- Row-by-row forward substitution is a chain of skinny vector-matrix ops that leaves the MXU
  idle. The Neumann and doubling form turns the solve into a few batched `64 x 64` matmuls, "moving
  the dominant solve work onto the MXU". The sub-block sizes (16 for relations, 8 for inversion)
  bound peak VMEM.
- Because the inverse is exact in exact arithmetic, the bfloat16 and float32 paths differ only
  in rounding. That makes float32 the precise reference when comparing numerics.
- Masking discipline matters. Anti-causal exponent differences are set to zero *before*
  `exp2`, so masked positions cannot produce `inf * 0 = NaN`.

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (Stage 1+2 in
  code), [mosaic-tpu-helpers](../concepts/mosaic-tpu-helpers.md) (dot precision on TPU),
  [gated-delta-rule](../concepts/gated-delta-rule.md) (a sibling delta-rule kernel).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-chunked-forward-pipeline](kda-chunked-forward-pipeline.md),
  [kda-chunked-backward-fusion](kda-chunked-backward-fusion.md).

## Source
Extracted from `docs/ops/kda/chunk_kda_fwd_design_doc.md` (Sections 4.1, 9.1.4-9.1.5), kept in
place.
