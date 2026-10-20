---
title: "TPU 8t (training) — announced"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-08-22
sources: 1
human_reviewed: true
---

Eighth-generation **training** chip, announced April 2026, GA "later this year" —
the first generation split into separate training (8t) and inference
([TPU 8i](tpu-8i.md)) chips. **All numbers below are vendor announcement claims**
([blog](../sources/2026-google-blog-tpu-8th-gen.md)); no Cloud docs spec sheet yet —
do not use for roofline math until datasheet figures land.

## Announced claims

| Claim | Value |
|---|---|
| Superpod | 9,600 chips; 2 PB shared HBM (~208 GB/chip implied) |
| Compute / pod | 121 ExaFlops — **precision unstated** (implies ~1.26e16/chip, likely low-precision) |
| vs previous gen | ~3× compute per pod; 2× interchip bandwidth; up to 2× perf/W vs Ironwood |
| Scaling | "Virgo Network" + Pathways: near-linear to 1M chips in one logical cluster |
| Goodput | >97% target via RAS: real-time telemetry, automatic ICI fault rerouting, OCS reconfiguration |
| I/O | 10× faster storage access; TPUDirect data path into the TPU |
| Host | Axion ARM-based CPUs (2× hosts/server, NUMA isolation); bare-metal access |
| Frameworks | JAX, MaxText, PyTorch, SGLang, vLLM; Tunix (RL) |

## See also

- [TPU7x](tpu-7x.md) — current generation; [TPU 8i](tpu-8i.md) — inference sibling

## Sources

- [Google blog: 8th-gen TPU announcement](../sources/2026-google-blog-tpu-8th-gen.md)
