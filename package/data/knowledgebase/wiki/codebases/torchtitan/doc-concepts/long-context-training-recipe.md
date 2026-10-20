---
title: "Long-context training recipes: Mamba2 and Mamba3 2.7B at 256K tokens on TPU7x"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/examples/mamba2/recipes/ironwood/mamba2-2.7b-256k/2x2x1/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Long-context training recipes: Mamba2 and Mamba3 2.7B at 256K tokens on TPU7x

## Definition
Two recipes train a ~3.3B-parameter state-space model at a sequence length of 262,144 tokens
("256K") on the smallest Ironwood slice, `2x2x1` (8 TensorCores, one host): Mamba-2 2.7B
(3,280,906,240 parameters) and Mamba-3 2.7B (3,266,507,264 untied / 2,938,171,904 tied). Both live
with their model code under `experiments/tpu/examples/<model>/recipes/ironwood/` rather than in
the shared `recipes/` tree. They are the only long-context recipes in the repo, and they get there
without context parallelism: the model is plain FSDP over 8 cores, and the context length is
affordable because the SSM "has linear complexity O(B * S)", avoiding "the quadratic memory
blowup of full self-attention".

## In torchtitan (grounded)
### Mamba-2 (primary recipe)
**Launch.** `gcluster job submit --compute-type tpu7x --topology 2x2x1 --num-slices 1`, then one
host running `torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=static`, module
`torchtitan.experiments.tpu.examples.mamba2`, config
[`mamba2_2_7b_256k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/config_registry.py#L308 "torchtitan/experiments/tpu/examples/mamba2/config_registry#mamba2_2_7b_256k"),
with `--training.max_context_length=262144 --training.mixed_precision_param=bfloat16`. Batch size
is expressed in tokens, so the two options pass
`--training.num_tokens_per_microbatch_per_dp_rank=2097152` (Option A, 8 sequences x 262,144 per
core) or `=262144` (Option B, 1 sequence per core); these land in
[`TrainingConfig.num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank")
and [`TrainingConfig.max_context_length`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L51 "torchtitan/config/configs#TrainingConfig.max_context_length").
The registry function already defaults to `8 * 262144` tokens and a 262,144 context, so Option A
restates the default. No compile, kernel or libtpu flags are passed. The README pins the software
as torch `2.12.0`, torch_tpu `0.1.1`, libtpu `0.0.41`, jax/jaxlib `0.11.0`.

**Model shape (README).**

| Field | Mamba-2 2.7B | Mamba-3 2.7B |
|---|---|---|
| Parameters | 3,280,906,240 | 3,266,507,264 untied / 2,938,171,904 tied |
| d_model | 2,560 | 2,560 |
| Layers | 64 | 64 |
| n_heads | 80 | 80 |
| Head dim | 64 (d_inner // n_heads = 5120 // 80) | 64 |
| d_state | 16 | 128 |
| SSM chunk size | 64 (4,096 chunks at S = 262,144) | 256 (1,024 chunks) |

The Mamba-2 values match the `2.7B` flavor that
[`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/config_registry.py#L126 "torchtitan/experiments/tpu/examples/mamba2/config_registry#model_registry")
builds on [`Mamba2Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/model.py#L34 "torchtitan/experiments/tpu/examples/mamba2/model#Mamba2Config")
(chunk size field [`Mamba2Config.chunk_size`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/model.py#L45 "torchtitan/experiments/tpu/examples/mamba2/model#Mamba2Config.chunk_size")),
whose default SSM backend is `"pallas"`; the kernel entry is
[`ssd_scan_pallas_jax`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/kernels/mamba2_pallas.py#L84 "torchtitan/experiments/tpu/examples/mamba2/kernels/mamba2_pallas#ssd_scan_pallas_jax").

**Parallelism.** [`parallelize_mamba2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba2/parallelize.py#L21 "torchtitan/experiments/tpu/examples/mamba2/parallelize#parallelize_mamba2")
applies activation checkpointing, optional compile, then
[`apply_fsdp_to_decoder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/fsdp.py#L189 "torchtitan/distributed/fsdp#apply_fsdp_to_decoder")
with `param_dtype` / `reduce_dtype` from `mixed_precision_param` / `mixed_precision_reduce`. It
raises `NotImplementedError` if tensor or context parallelism is enabled, so FSDP over the 8-core
data-parallel mesh is the only option - which is what the README describes ("FSDP2 sharding across
the 8 TPU cores (1D Data Parallel mesh)").

**Memory budget (README, 96 GiB per core).**

| Component | GiB per core |
|---|---|
| Parameters (bf16, sharded) | 0.82 |
| Gradients (bf16, sharded) | 0.82 |
| AdamW states (fp32, sharded) | 3.28 |
| Main weights (fp32, sharded) | 1.64 |
| Total static state | ~6.56 |
| Activations, local batch 1 | ~3.8 -> peak ~10.4 (10.8%) |
| Activations, local batch 8 | ~25.2 -> peak ~31.8 (33.1%) |

**Throughput (README "Scaling Across Batch Configurations").**

| Configuration | Global batch | Tokens / step | Memory / core | Cluster TPS | Est. step latency | Attained TFLOPS | Expected MFU |
|---|---|---|---|---|---|---|---|
| Baseline, S=128, B_local=2 | 16 | 2,048 | 2.91 GiB | 109 | 2.35s | 2.15 | 0.023% |
| 256K, B_local=1 | 8 | 2,097,152 | 10.40 GiB | ~65,500 - 87,400 | ~24 - 32s | ~1,290 - 1,720 | 14.0% - 18.6% |
| 256K, B_local=8 | 64 | 16,777,216 | 31.80 GiB | ~79,900 - 104,800 | ~160 - 210s | ~1,570 - 2,060 | 17.0% - 22.4% |

MFU is computed against 9,228 TFLOPS ("8 cores @ 1,153.5 TFLOPS/core") with model FLOPs per token
taken as 6 x N_params (~19.685 GFLOPs/token). The 256K rows are ranges labelled "Expected" and
"Est.", not a single measured run.

### Mamba-3 (sibling recipe)
**Launch.** Same slice, process layout and dataset, but still written for `xpk workload create
--tpu-type="tpu7x-2x2x1"` with `--rdzv_backend=c10d`, module
`torchtitan.experiments.tpu.examples.mamba3` and config
[`mamba3_2_7b_256k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba3/config_registry.py#L60 "torchtitan/experiments/tpu/examples/mamba3/config_registry#mamba3_2_7b_256k")
(model built by [`model_registry`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba3/config_registry.py#L41 "torchtitan/experiments/tpu/examples/mamba3/config_registry#model_registry")
on [`Mamba3Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba3/model.py#L31 "torchtitan/experiments/tpu/examples/mamba3/model#Mamba3Config"); same token-budget defaults
and `FullAC` as the Mamba-2 256K function).
[`parallelize_mamba3`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba3/parallelize.py#L17 "torchtitan/experiments/tpu/examples/mamba3/parallelize#parallelize_mamba3") is the same
FSDP-only wrapper as Mamba-2 and rejects TP and CP the same way.

**What Mamba-3 adds (README).** Continuous-time 2D RoPE rotation over `d_state // 2 = 64` pairs per
head, exponential-trapezoidal discretization, a data-dependent decay A computed per token through
a softplus projection, and a strict causal diagonal correction in the intra-chunk term. The fused
Pallas kernel ([`ssd3_chunk_scan_pallas`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/examples/mamba3/kernels/mamba3_pallas.py#L1060 "torchtitan/experiments/tpu/examples/mamba3/kernels/mamba3_pallas#ssd3_chunk_scan_pallas"))
runs with `chunk_size = 256` and `head_group = 8`.

**Throughput and memory (README, estimates).**

| Configuration | Global batch | Memory / core | Cluster TPS | Est. step latency | Attained TFLOPS | Expected MFU |
|---|---|---|---|---|---|---|
| Baseline, S=128, B_local=2 | 16 | ~2.95 GiB | ~110 | ~2.30s | ~2.15 | ~0.023% |
| 256K, B_local=1 | 8 | ~10.70 GiB | ~64,000 - 85,000 | ~25 - 33s | ~1,250 - 1,670 | 13.5% - 18.1% |
| 256K, B_local=8 | 64 | ~34.00 GiB | ~78,000 - 102,000 | ~165 - 215s | ~1,530 - 2,000 | 16.6% - 21.7% |

The README uses 6 x 3,266,507,264 = ~19.60 GFLOPs/token (untied) and frames Mamba-3 as "within
0.4% - 0.7%" of Mamba-2's MFU at batch 8 despite `d_state` = 128 vs 16. It also reports measured
micro-benchmarks: the Pallas scan alone at 256K and batch 1 takes 781.84 ms forward + backward
(~335,290 tok/s), and one full `Mamba3Block` on a single TensorCore reaches only ~2.2% - 2.7% MFU at
batch 1,
which the README attributes to the narrow M = 1 x S matmul dimension and missing collective overlap.

> [!warning] Doc vs code
> - The Mamba-3 README passes `--training.local_batch_size=8|1` and `--training.seq_len=262144`.
>   `TrainingConfig` at this pin has neither field (batch is
>   `num_tokens_per_microbatch_per_dp_rank`, context is `max_context_length`), so those commands
>   need the Mamba-2 spelling: `--training.num_tokens_per_microbatch_per_dp_rank=2097152` (or
>   `262144`) with `--training.max_context_length=262144`.
> - The Mamba-2 README memory section says "Selective Activation Checkpointing", but
>   `mamba2_2_7b_256k` sets `activation_checkpoint=FullAC.Config()` (the non-256K `mamba2_2_7b`
>   uses `SelectiveAC`), and the recipe does not override it. The Mamba-3 README says Full AC,
>   which matches its config.
> - `Mamba3Config` defaults to `tie_embeddings=True` and its `get_nparams_and_flops` counts
>   `model.parameters()`, so the logged FLOPs per token presumably use the tied 2.94B count, not
>   the untied figure behind the README's MFU table *(unverified - agent inference)*.

## Why it matters / when it applies
- **Why it works.** Linear-in-S compute and memory: going from S=128 to 256K raises Mamba-2 per-core
  memory only from 2.91 GiB to 10.4 GiB (batch 1) or 31.8 GiB (batch 8), "maintaining over 60 GiB
  of safety headroom" (Mamba-3: ~34.0 GiB, "over 62 GiB" headroom). At S=128 MFU is "~0.02%"
  because the step is "dominated by host launch overhead, distributed step synchronization, and
  memory bus latency"; at 256K "the TPU MXU systolic arrays are fully saturated". Long sequences are
  what make these small models efficient.
- **Throughput vs batch.** Option A (batch 8) is the "High Throughput & High Memory Utilization"
  choice; Option B (batch 1) keeps global batch at 8 for a lower-memory run.
- **Mamba-2 vs Mamba-3.** Per the README, Mamba-3's larger state and RoPE / trapezoidal terms cost
  under one MFU point at 256K, so the choice is about model quality, not TPU efficiency.
- **Transfer to attention models.** These recipes do not show how to reach 256K on a Transformer:
  no recipe in the tree exercises context parallelism, and both parallelize functions reject it. An
  attention model at this length would likely need it *(unverified - agent inference)*.

## Connections
- Code concepts: [context-parallel](../concepts/context-parallel.md) (not used here; the SSM avoids
  it), [fsdp-sharding](../concepts/fsdp-sharding.md),
  [activation-checkpointing](../concepts/activation-checkpointing.md),
  [model-definitions](../concepts/model-definitions.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [job-configuration](../concepts/job-configuration.md) (token-budget batch fields),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md).
- Module catalogs: [module map](../catalog/index.md).
- Hardware: [TPU7x (Ironwood)](../../../hardware/tpu-7x.md).
- Related doc-concepts: [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md),
  [tpu-training-recipes](tpu-training-recipes.md),
  [model-registry](model-registry.md).

## Source
Extracted from (kept in place):
- `torchtitan/experiments/tpu/examples/mamba2/recipes/ironwood/mamba2-2.7b-256k/2x2x1/README.md`
- `torchtitan/experiments/tpu/examples/mamba3/recipes/ironwood/mamba3-2.7b-256k/2x2x1/README.md`
