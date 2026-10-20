---
title: "KDA chunked forward: four semantic stages in two fused TPU kernels"
type: doc-concept
provenance: doc
source: docs/ops/kda/chunk_kda_fwd_design_doc.md
updated: 2026-10-02
generated: {by: wikify/0.4.0, at: 2026-10-02T23:59:32Z}
---
# KDA chunked forward: four semantic stages in two fused TPU kernels

## Definition
The KDA forward design doc describes Kimi Delta Attention as compressing history into a fixed
`K x V` state per batch element and head. "Queries read from that state, keys and values update
it through the delta rule, and a per-channel gate controls the forgetting rate of each key
channel." A token-by-token recurrence `S_0 -> S_1 -> ... -> S_T` has almost no time
parallelism. The delivered design instead cuts the sequence into chunks of `C = 64` and computes
four semantic stages:
1. Gate activation and chunk-local cumulative decay (`g_cumsum`, kept in log2 space).
2. The intra-chunk solve, which removes the delta rule's sequential dependency inside a chunk
   (`w`, `u`, `kg`, `Aqk`, `Akk`).
3. State propagation across chunks.
4. Output: the read of the pre-chunk state plus the causal intra-chunk contribution.

The stages "describe mathematical dependencies, not four mandatory kernel dispatches".
Stages 1+2 run as one Pallas kernel and Stages 3+4 as another. The only HBM boundary left
between them is the `g_cumsum / w / u / kg / Aqk` hand-off, and the context-parallel bridge
can sit there (see [kda-context-parallel-state-summaries](kda-context-parallel-state-summaries.md)).

## In tokamax (grounded)
- **Entry and dispatch.** [`KimiDeltaAttention.bind`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/base.py#L83 "tokamax/_src/ops/experimental/kda/base#KimiDeltaAttention.bind")
  validates the public contract and turns `scale=None` into `K^-1/2`.
  [`PallasMosaicTpuKimiDeltaAttention._fwd`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L372 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#PallasMosaicTpuKimiDeltaAttention._fwd")
  resolves the gate strategy and rejects unsupported calls in
  [`check_inputs_support`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L112 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#check_inputs_support") (for example
  `K > 256`) before tracing any kernel, so dispatch can fall back to XLA. It then runs
  [`PallasMosaicTpuKimiDeltaAttention._preprocess_inputs`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L224 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#PallasMosaicTpuKimiDeltaAttention._preprocess_inputs")
  and [`chunk_kda_fwd_custom`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L1425 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#chunk_kda_fwd_custom").
  The backend's `supported_on` requires TPU v6 or later.
- **Stage 1+2.** [`kda_fwd_intra_fused`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L949 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#kda_fwd_intra_fused")
  asserts `chunk_size == 64` and calls
  [`pallas_kda_fwd_intra_fused`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L808 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#pallas_kda_fwd_intra_fused").
  That kernel, [`_fused_gate_intra_kernel`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L589 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_fused_gate_intra_kernel"),
  activates the gate if needed and forms the prefix sum as a lower-triangular ones matmul scaled
  by `1/ln 2`. It builds `Aqk` and `L` in 16-row sub-blocks and solves the unit lower-triangular
  system (see [kda-wy-triangular-solve](kda-wy-triangular-solve.md)). Several heads are processed
  per grid point; [`_compute_intra_fused_mini_batch`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L783 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_compute_intra_fused_mini_batch")
  sizes this mini-batch from a VMEM budget, capped at 16.
- **Stage 3+4.** [`chunk_kda_fwd_h_o_varlen`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L1152 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#chunk_kda_fwd_h_o_varlen")
  serves both fixed-length input (one `[0, T]` segment per batch element) and variable-length
  input. Per chunk it computes `V_new = U - W S`, the inter output `(Gamma * q~) S`, the intra
  output `Aqk V_new`, and the update `S' = Diag(gamma_C) S + kg^T V_new`: four MXU matmuls. The
  running state lives in a float32 VMEM scratch. The chunk grid axis is `arbitrary` (ordered)
  and the head-group and batch axes are `parallel`. The launcher pads `K` and `V` to the TPU
  block alignment and trims the outputs afterwards.
- **Numerics.** All decays use `exp2` of log2 cumulative gates. In Stage 2, anti-causal exponent
  differences are zeroed before `exp2` so they cannot overflow. In Stage 3+4 the inter read is
  factored around the chunk's first gate row, and exponents are clamped at `-126`. Critical
  accumulators and the state are float32, and `o` is cast back to the query dtype.

> [!warning] Two support-matrix statements that differ from the code at the pin
> - **float32 does not use a separate gate cumsum.** The doc's support table and Section 5.1 say
>   float32 uses "separate gate cumsum" (bfloat16 is "fused Stage 1+2"). In the code,
>   [`kda_fwd_intra_fused`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L949 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#kda_fwd_intra_fused") always
>   calls the fused kernel, and
>   [`pallas_kda_fwd_intra_fused`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L808 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#pallas_kda_fwd_intra_fused")
>   always passes `fuse_cumsum=True`. Inside
>   [`_fused_gate_intra_kernel`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel.py#L589 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_fwd_kernel#_fused_gate_intra_kernel"),
>   dtype selects only the solve (`use_neumann = dtype != jnp.float32`). Both dtypes fuse the
>   gate cumsum. float32 differs only in using exact forward substitution.
> - **Softplus raw-gate activation does not run on the Mosaic backend.** The doc lists softplus
>   decay (`lower_bound=None`) as supported raw-gate activation. However,
>   [`_check_gate_support`](https://github.com/openxla/tokamax/blob/3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68/tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu.py#L79 "tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu#_check_gate_support") raises
>   `NotImplementedError` ("`mosaic` requires `lower_bound` to be set.") whenever
>   `use_gate_in_kernel=True` and `lower_bound is None`. Such calls fall back to the next
>   implementation. In-kernel gate activation on TPU is the bounded sigmoid variant only.

## Why it matters / when it applies
- Chunking keeps the linear-in-`T` cost of a fixed-size state, while most of the work becomes
  dense `64 x 64` and `64 x K` matmuls for the MXU. Fusing adjacent stages cuts HBM round trips
  but keeps the ordered Stage 3 recurrence separate from the parallel Stage 1+2.
- The doc's historical v7 profile (B=1, T=8192, H=32, K=V=128, bf16, 25 packed sequences) puts
  both fused kernels at roughly 26-28% of HBM roofline. That measurement predates the current
  adapter refactor. The doc requires re-benchmarking before the numbers are used as a gate.
- Runtime invariants to respect: `chunk_size=64`, fixed-length `T` divisible by 64, `K <= 256`,
  and a non-positive pre-activated gate (`alpha` in `(0, 1]`).

## Connections
- Code concepts: [kimi-delta-attention](../concepts/kimi-delta-attention.md) (the full code
  walkthrough), [gated-delta-rule](../concepts/gated-delta-rule.md) (the related GDN serving
  kernel), [op-framework](../concepts/op-framework.md), [kernel-catalog](../concepts/kernel-catalog.md).
- Module catalogs: [module map](../catalog/index.md).
- Related doc-concepts: [kda-wy-triangular-solve](kda-wy-triangular-solve.md),
  [kda-variable-length-alignment](kda-variable-length-alignment.md),
  [kda-context-parallel-state-summaries](kda-context-parallel-state-summaries.md),
  [kda-residuals-and-rematerialization](kda-residuals-and-rematerialization.md),
  [kda-chunked-backward-fusion](kda-chunked-backward-fusion.md).

## Source
Extracted from `docs/ops/kda/chunk_kda_fwd_design_doc.md` (Sections 1-5, 7, 8), kept in place.
