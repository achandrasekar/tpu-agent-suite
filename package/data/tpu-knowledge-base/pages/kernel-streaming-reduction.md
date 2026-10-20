---
title: "Kernel class: streaming reduction (norms, cross-entropy, standalone softmax)"
type: concept
tags: [tpu-kernel-development, kernel-class, reduction, norm, cross-entropy, pallas, briefs-tier]
created: 2026-07-22
updated: 2026-10-14
sources: 0
human_reviewed: false
---

**Route here when** K1 shows a big reduction over a materialized axis without a dominating matmul producer (RMSNorm/LayerNorm, cross-entropy, standalone softmax). If a GEMM/conv produces the reduced tensor, route to [gemm-conv-epilogue](kernel-gemm-conv-epilogue.md) instead — the precondition test there decides.

- **Strategy**: tile the reduction axis, maintain running state in VMEM (running max/sum for softmax; mean/var for norms; logsumexp for CE); accumulate fp32. The intermediate never lands in HBM.
- **Cross-entropy over logits already in HBM: XLA is hard to beat.** On v6e (131,072-token vocabulary, bf16) a two-pass Pallas log-sum-exp ran at 0.63× of XLA's, and a chunked in-place CE gradient at 0.79× (`index.py get v6e-measured-model-ops`). The single-pass online form above was not measured. Write the CE kernel when it removes the logits materialization (fused with the LM-head matmul, see gemm-conv-epilogue), not to re-read logits XLA already reads.
- **Cheap first rung before authoring a norm kernel**: XLA's default mean/variance lowering pays two reads; the `E[x²]−E[x]²` single-pass moment rewrite is near-free, needs no kernel (reassociates ⇒ tolerance-gated).
- **Check the overlap decomposition first** (BRIEFS §2): on compute-adjacent CE-class ops, eliminating GB of traffic that hides under a matmul buys ~6% — naive-wall vs compute-floor decides whether the class lever is worth authoring.
- **Cross-lane (last-dim) reductions route through the XLU** and cost markedly more than sublane reductions — layout choice is part of the design (index Hardware envelope).
- Parity: gate calibration + ULP-at-magnitude + non-degenerate probes (BRIEFS §4) — softmax/exp tails make max_rel meaningless near zero; grade max-abs at magnitude.
