---
title: "vLLM engine arguments: the KV cache and scheduler knobs"
type: reference
tags: [vllm, serving, kv-cache]
created: 2026-10-06
human_reviewed: false
source_ids:
  - vllm-config-cache
  - vllm-config-scheduler
  - vllm-docs-engine-args
---

Authored, not fetched. `docs.vllm.ai` sits behind a Cloudflare bot
challenge: every page under it answers `429` with `cf-mitigated: challenge`
to any client that is not a browser, so no lookup can read it. The
rendered page is also not where the values live — its markdown source is
the stub `--8<-- "gen:engine-args"`, and the table is generated at build
time from the config classes. Those classes are public and readable, and
are what this page is written from. vLLM's own documentation says to read
them: they *"are the source of truth for types, defaults and
docstrings"*.[^vllm-docs-engine-args]

Defaults below are from `main` as checked on 2026-10-06. They move; check
the class before relying on one.

## KV cache

| argument | default | what it is |
|---|---|---|
| `--block-size` | **16** tokens[^vllm-config-cache] | tokens per contiguous cache block |
| `--gpu-memory-utilization` | **0.92**[^vllm-config-cache] | fraction of device memory for the model executor |
| `--kv-cache-dtype` | `auto` [^vllm-config-cache] | `auto` uses the model dtype |
| `--enable-prefix-caching` | **on**[^vllm-config-cache] | reuse blocks across requests sharing a prefix |

`block_size` is *"size of a contiguous cache block in number of
tokens"*.[^vllm-config-cache] It is an allocation unit, not a hardware
tile — see below.

For what the cache holds and why it dominates decode bandwidth — per-token
size, GQA/MQA, quantization, paged attention, prefix caching and eviction —
see `index.py get concept-kv-cache`. This page is only about the arguments.

`prefix_match_unit` is separate and finer: the token boundary a prefix-cache
hit may land on. It can be set below the physical block size, provided every
KV cache group's `block_size` divides by it, which lets a hit land inside a
physical block. It changes matching granularity only, not how often state is
stored.[^vllm-config-cache]

## Scheduler

| argument | default | what it is |
|---|---|---|
| `--max-num-seqs` | **128**[^vllm-config-scheduler] | sequences batched per iteration |
| `--max-num-batched-tokens` | **2048**[^vllm-config-scheduler] | tokens batched per iteration |

## Why a smaller block does not lower HBM use

The KV cache is sized by a *budget*, not by demand. vLLM profiles memory,
takes the fraction `--gpu-memory-utilization` allows, and allocates the whole
remainder as cache blocks. Halving `--block-size` divides that same budget
into four times as many blocks; it does not hand any memory back. HBM use is
flat by construction.

The saving is in **internal fragmentation**. A sequence occupies whole
blocks, so its last block is partly waste — up to `block_size - 1` tokens of
it. At 256 a request of 80 to 180 tokens wastes most of one block; at 64 it
wastes at most 63 tokens. The same budget therefore holds more concurrent
sequences, which is the effect actually observed: more blocks, more short
requests resident, identical HBM.

Decode latency going the other way is the trade. More, smaller blocks mean
more block lookups and a longer block table per sequence, and attention
kernels read KV in block-sized runs — shorter runs are less efficient per
step. Fit and step time pull against each other, so the block size is chosen
against the request length distribution, not against the hardware.

## Is there a hardware block size to match on v6e?

No, not for this argument. The KV cache block size is a scheduler and
memory-accounting unit. The TPU shapes that do matter are register and
systolic-array shapes, and they constrain *tensor dimensions*, not how many
tokens share a cache block: the VPU's vector registers are (8, 128) — 8
sublanes by 128 lanes — so a last dimension pads to 128 and a
second-to-last to 8 (`index.py get torch-tpu-hardware-alignment-vmem`).
Those apply to head dimension and hidden size, which the model fixes.

Choose `--block-size` from the request length distribution: large enough
that the block table and per-step lookups stay cheap, small enough that the
tail block of a typical request is not mostly padding.

vLLM's TPU backend no longer lives in the vLLM tree —
`vllm/platforms/tpu.py` is a stub, and the kernels and worker are in
`vllm-project/tpu-inference` (`index.py get tpu-inference-kernels`). Any
TPU-specific constraint on block size is that repository's to state, and
none of the above asserts one.

## Sources

[^vllm-config-cache]: [vLLM `CacheConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py). configuration-source. `DEFAULT_BLOCK_SIZE`, `gpu_memory_utilization`, `cache_dtype`, `enable_prefix_caching`, `prefix_match_unit` and their docstrings. Read 2026-10-06.
[^vllm-config-scheduler]: [vLLM `SchedulerConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py). configuration-source. `DEFAULT_MAX_NUM_SEQS`, `DEFAULT_MAX_NUM_BATCHED_TOKENS`. Read 2026-10-06.
[^vllm-docs-engine-args]: [vLLM Engine Arguments](https://github.com/vllm-project/vllm/blob/main/docs/configuration/engine_args.md). vendor-docs. The page that names the config classes as the source of truth; its own body is generated at build time. Read 2026-10-06.
