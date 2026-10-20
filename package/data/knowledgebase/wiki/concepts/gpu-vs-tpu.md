---
title: "GPU vs TPU: hardware, networking and roofline mapping"
type: concept
tags: [hardware, interconnect, onboarding, performance]
created: 2026-09-04
updated: 2026-10-12
sources: 2
human_reviewed: false
aliases: ["H100 vs TPU", "NVLink vs ICI", "SM vs TensorCore", "GPU mapping"]
---

A modern ML GPU is, like a TPU, matmul cores attached to HBM — but split into a hundred-plus
independent Streaming Multiprocessors with a switched NVLink/InfiniBand network instead of
one or two big TensorCores on a torus. This page maps the blocks, the numbers and the
scaling rooflines one-to-one so a GPU-shaped intuition can be translated before an
onboarding ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## Block-level mapping

| GPU | TPU | H100 count | TPU v5p count | Note |
|---|---|---|---|---|
| SM (streaming multiprocessor) | [TensorCore](tensorcore.md) | 132 | 2 | the "cell" containing the others |
| Warp scheduler (32 CUDA cores) | [VPU](vpu.md) | 528 | 8 slots | SIMT vs SIMD: CUDA threads have their own instruction pointers, at the cost of divergence stalls |
| CUDA core | VPU ALU | 16,896 | 8,192 | GPU has ~2× the vector lanes at similar clock |
| Tensor Core | [MXU](mxu.md) | 528 | 8 | H100 TC ≈ 1024 bf16 FLOPs/cycle (≈ 8×8×8 matmul) vs a 128×128 systolic array |
| SMEM (L1) + registers | [VMEM](vmem.md) + VRegs | 32 MB + 32 MB | 128 MB + 256 kB | VMEM is larger and ~40 TB/s; L2 (50 MB, ~5.5 TB/s) has no TPU analogue and is not programmer-controlled |
| HBM (GMEM) | [HBM](hbm.md) | 80 GB, 3.35 TB/s | 95 GiB, 2.8 TB/s | B200: 192 GB, 8 TB/s |

([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md).) Consequences the book draws: TPUs are
cheaper and simpler and reach roofline with less tuning because the compiler must pipeline
a single-threaded core; GPUs "just work" on new tasks but hide L2 thrashing and warp
divergence behind the hardware; a single H200 has ~2× a v5p's FLOPs and ~2.5× its price
([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## Chip rooflines

| GPU | bf16 FLOPs/s | HBM B/s | critical intensity (H100/B200 stated by the book; A100/H200 derived) |
|---|---|---|---|
| A100 | 3.1e14 | 2.0e12 | ~155 |
| H100 / H200 | 9.9e14 | 3.4e12 / 4.8e12 | ~295 / ~206 |
| B200 | 2.3e15 | 8.0e12 | ~281 |

([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md).) Same order as TPU (v5e 240, TPU7x ~313).
Low precision doubles peak on both, but not by the same type: fp8 on H100/B200
([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)), int8 on TPU v5e/v6e
([TPU v5e](../hardware/tpu-v5e.md), [TPU v6e](../hardware/tpu-v6e.md)), and fp8 only from TPU7x
([TPU7x](../hardware/tpu-7x.md)). GPUs typically reach 80–85% of claimed peak, TPUs closer to 95%
([scaling-book, Part 1](../sources/2025-scaling-book-roofline.md)). Vector-op intensity is ~10 on both
H100 and B200 ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## Networking

- **Node (NVLink domain)**: 8 GPUs (72 for GB200 NVL72) all-to-all through NVSwitches;
  GPU egress 450 GB/s on H100, 900 on B200 ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).
- **Scale-out**: InfiniBand fat tree (leaf = 32-node SU of 256 GPUs, spine above) with
  full bisection bandwidth of 400 GB/s per node; GB200 NVL72 raises node egress to
  3.6 TB/s ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).
- Versus [ICI](ici.md): a v5p has ~540 GB/s egress across its three axes but only to
  neighbors; TPUs scale to 8,960-chip tori without switches, GPUs add switch layers and
  latency ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).
- Collectives: same ring formulas in-node; cross-node AllGather/ReduceScatter ≈
  `bytes / W_node`; AllToAll cheaper in-node but >4× worse across nodes; SHARP in-network
  reduction gives ~30% not 2×; measured AllReduce ~370 of 450 GB/s and only with ~10 GB
  messages, whereas TPUs hit peak at ~10 MB
  ([collective-costs](collective-costs.md#on-gpus-for-comparison)).

## LLM scaling rooflines on GPUs

| Strategy | GPU condition (H100) | TPU counterpart |
|---|---|---|
| DP / ZeRO | ~2,200 tokens/GPU in-node, ~2,475 across nodes; ×`E/k` for MoE | `α_ici/M_X` = 850 on v5p (3 axes); see [critical-batch-size](critical-batch-size.md) |
| TP | `Y < F/2475` → ~8-way, one node (2 nodes at most) | up to `M_Y·F/α`, 8–16 way on v5p |
| PP | comms ~free; enables ZeRO-1 DP below the floor; incompatible with ZeRO-3 | rarely needed inside a pod |
| EP | 1–2 nodes if `F < 8C/W_node`, else up to `E` nodes | AllToAll ¼ AllGather on a ring |

([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md).) Book recipes: small dense → aggressive
FSDP; large dense → 1–2 node TP + many-node PP + DP; MoE → prefer EP over TP.

## What changes when moving a workload to TPU *(synthesis — agent inference)*

- Kernel-level thread control (warps, SMEM tiling, streams) has no counterpart; the
  compiler schedules the single-threaded core, and custom work goes through
  [Pallas](../codebases/pallas.md) — the existing
  Triton (tpu-knowledge-base: `triton-to-pytorch-ops`) and
  FlashAttention (tpu-knowledge-base: `flash-attention-to-sdpa`) playbooks are the practical
  form of this.
- Multi-node TP/EP limits relax inside a pod, and cross-pod DP over [DCN](dcn.md) replaces
  the InfiniBand layer; pipelining is usually unnecessary.
- Large, fast [VMEM](vmem.md) rewards weight/activation residency that SMEM cannot hold.
- Collective bandwidth is reached at far smaller messages, so small-batch decode
  collectives behave differently ([collective-costs](collective-costs.md)).

## See also

- [Collective costs](collective-costs.md), [Parallelism strategies](parallelism-strategies.md),
  [TensorCore](tensorcore.md), [ICI](ici.md), CUDA API stripping (tpu-knowledge-base: `cuda-api-stripping-and-runtime`)

## Sources

- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
- [How To Scale Your Model, Part 1: Rooflines (scaling-book)](../sources/2025-scaling-book-roofline.md)
