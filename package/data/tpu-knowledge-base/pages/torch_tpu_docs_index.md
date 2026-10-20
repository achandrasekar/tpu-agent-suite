---
title: "TorchTPU docs: which page answers what"
type: navigation
tags: [torch-tpu, navigation]
created: 2026-10-06
updated: 2026-10-19
human_reviewed: false
---

Which TorchTPU page to read, for a reader who knows what they want to do but
not where it is written down. The keyword in backticks is what reaches it, and
asking for that keyword reads the page.

The upstream docs live in the public repository at
`github.com/google-pytorch/torch_tpu/tree/main/docs`, and the rows marked
*upstream* read them there at `blob/main`. TorchTPU installs from PyPI with
`pip install torch_tpu`; no Artifact Registry or gcloud step is needed. The
rows marked *authored* are pages this package ships, written for it rather
than copied from upstream.

## Start here

| Page | | For |
|---|---|---|
| `torch-tpu-intro` | upstream | What TorchTPU is, a quickstart, key features and current limits |
| `torch-tpu-get-started` | upstream | First TPU VM: `pip install torch_tpu` on Python 3.12, the first tensor |
| `torch-tpu-get-started-gke` | upstream | Running on GKE with the public release container, single-device and distributed |
| `torch-tpu-api-mapping` | upstream | `torch.cuda` to `torch.accelerator` / `torch.tpu`, and every environment variable |
| `torch-tpu-faq` | upstream | TorchTPU vs PyTorch/XLA and TorchAX, supported generations, support channels |
| `torch-tpu-feature-set` | upstream | What is supported today and what is planned |

## How it works

| Page | | For |
|---|---|---|
| `torch-tpu-compilation-cache` | upstream | How TorchTPU works: the backend, graph capture, the tiered compilation cache, `torch.compile` |
| `torch-tpu-eager-mode` | upstream | The execution model: strict, debug and defer-and-fuse eager modes |
| `torch-tpu-memory-model` | upstream | Storage, tensors, view decomposition, in-place mutation and buffer donation |
| `torch-tpu-anti-patterns` | authored | Where the speedup actually comes from |
| `torch-tpu-precision-ladder` | authored | FP32 emulation through BF16 to FP8 on the MXU |
| `torch-tpu-hardware-alignment-vmem` | authored | Alignment cliffs and VMEM, measured on v7 |

## How do I ...

| Page | | For |
|---|---|---|
| `torch-tpu-dist-training-101` | upstream | Distributed support: `tpu_dist`, launchers, topology-aware device meshes |
| `torch-tpu-fsdp-best-practices` | upstream | Choosing between SimpleFSDP and manual FSDP |
| `torch-tpu-fsdp-to-simple-fsdp` | upstream | Migrating `fully_shard` code to SimpleFSDP |
| `torch-tpu-fsdp2` | authored | FSDP2 and 2D data x tensor parallelism |
| `torch-tpu-parallelism-decision-tree` | authored | Choosing between DDP, FSDP2 and tensor parallelism |
| `torch-tpu-perf-profiling` | upstream | Profiling with XProf and `torch.profiler` |
| `torch-tpu-precision-management` | upstream | Controlling matmul precision, and checking TPU output against CPU |
| `torch-tpu-nan-debugging` | authored | Where BFloat16's 7-bit mantissa bites |
| `torch-tpu-dynamic-shapes` | upstream | Bounded dynamism and growing tensors in decode loops |
| `torch-tpu-avoiding-recompilation` | authored | Keeping Dynamo from recompiling every step |
| `torch-tpu-llm-model-optimizations` | authored | LLaMA-3, Qwen-2.5 and DeepSeek specifics |
| `torch-tpu-moe-serving` | authored | Serving: prefill vs decode, paged KV cache |
| `torch-tpu-xla-flags` | authored | XLA flags and the lowering pipeline |

## Custom kernels

| Page | | For |
|---|---|---|
| `torch-tpu-custom-pallas-kernel` | upstream | Tokamax kernels, Pallas kernels via `torch.tpu.pallas`, Helion |
| `torch-tpu-ops-custom-kernels` | upstream | The `torch.tpu.ops` source: Splash attention, ragged dot, collectives |

## Case studies

| Page | | For |
|---|---|---|
| `torch-tpu-case-study-flux` | upstream | Optimizing FLUX step time, milestone by milestone |
| `torch-tpu-case-study-muse-glimmer-inference` | upstream | Autoregressive decoding of Muse Glimmer 30B with static and dynamic KV caches |
| `torch-tpu-case-study-muse-glimmer-training` | upstream | Training Muse Glimmer 30B: per-layer compile, checkpointing, FSDP |

## Reference and contributing

| Page | | For |
|---|---|---|
| `torch-tpu-public-apis` | upstream | Every public API and env var, by stability stage |
| `torch-tpu-api-policy` | upstream | API stages and the deprecation rule |
| `torch-tpu-api-design-guide` | upstream | How new TorchTPU APIs, flags and env vars are designed |
| `torch-tpu-python-support-policy` | upstream | Which Python versions are supported |
| `torch-tpu-repo` | upstream | The repository README: install from PyPI, build from source |
| `torch-tpu-codebase-overview` | upstream | The top-level source layout |
