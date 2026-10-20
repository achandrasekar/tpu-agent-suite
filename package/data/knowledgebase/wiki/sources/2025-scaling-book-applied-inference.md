---
title: "Serving LLaMA 3-70B on TPUs (scaling-book, Part 8)"
type: source
tags: [workload, performance, memory, precision]
created: 2026-09-04
updated: 2026-09-04
sources: 1
human_reviewed: true
---

Part 8 of [*How To Scale Your Model*](2025-scaling-book.md): LLaMA 3-70B served on TPU v5e —
FLOPs/$ hardware choice, KV cache per token, memory fit by dtype and the resulting minimum
topology, the weight-loading latency floor, throughput per chip, how much model parallelism
is allowed (FLOPs-bound vs latency-bound), prefill latency, KV eviction rate, the 3:1
prefill:generate server ratio, and the latency/throughput Pareto frontier.

## Identity & authority

- <https://jax-ml.github.io/scaling-book/applied-inference> — fetched 2026-09-04; chapter
  last changed upstream 2026-09-03 (`ac72f02`, added a from-scratch LLaMA inference
  exercise).
- Authority: authorial worked example; pricing is Google Cloud list price as of
  February 2025 and is **not** a spec.

## Gaps & caveats

- v5e-only (16 GB HBM per chip) — every fit-by-dtype table changes on v6e (32 GB) and
  TPU7x (192 GiB). Re-derive with [size-serving-topology](../playbooks/size-serving-topology.md).
- Throughput figures ignore working memory (activations, attention scratch); the book
  says real numbers are about half.

## Residue (not extracted)

Four worked problems (405B FLOPs, 8B at BS 240, 405B on v5e under a 15 ms/token limit,
build-an-inference-engine exercise). The roofline-plot code (param/KV/FLOPs time
decomposition) is reproduced in spirit in the playbook's verification step.

## Connections

Created: [size-serving-topology](../playbooks/size-serving-topology.md).
Updated: [prefill-vs-decode](../concepts/prefill-vs-decode.md),
[kv-cache](../concepts/kv-cache.md), [quantization](../concepts/quantization.md),
[tensor-parallelism](../concepts/tensor-parallelism.md), [tpu-v5e](../hardware/tpu-v5e.md)
(see-also only).

## Sources

- `raw/docs/2025-scaling-book-applied-inference.html`
