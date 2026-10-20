---
title: "KV cache"
type: concept
tags: [workload, memory, performance]
created: 2026-09-04
updated: 2026-10-14
sources: 3
human_reviewed: true
aliases: ["key-value cache", "KV caching", "paged attention", "prefix caching"]
---

The KV cache stores every past token's key and value projections so generation never
reprocesses the prefix: `[2, S, L, K, H]` per sequence, `2 · bytes · H · K · L` per token.
It turns sampling from `O(n²)`/`O(n³)` into `O(n)`/`O(n²)` and becomes the dominant memory
and bandwidth consumer at long context (scaling-book, Part 4 (`index.py get scaling-book-transformers`),
scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## Size, and why it dominates

| Model | KV per token | Per sequence |
|---|---|---|
| LLaMA 3-70B (`K=8, H=128, L=80`), int8 | 160 kB | 5.3 GB at 32k; 1.3 TB for BS 240 (scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)) |
| LLaMA 2-13B (`K=40, H=128, L=40`), bf16 | 820 kB (computed from the formula) | 6.7 GB at 8k — four sequences exceed the 26 GB of weights (scaling-book, Part 7 (`index.py get scaling-book-inference`)) |
| "moderate" `KH = 8192, L = 64`, int8 | 1 MB | 8 GiB at 8k (scaling-book, Part 4 (`index.py get scaling-book-transformers`)) |

At decode the step time is `B·KV_size / W_hbm` for attention plus the weight term, so once
`B·KV` is comparable to the parameters, throughput gains from batching flatten; the
crossover is roughly `2DF / (S·H·K)`, i.e. `F` versus sequence length
(scaling-book, Part 7 (`index.py get scaling-book-inference`)). For LLaMA 3-70B above 2k context KV
loading exceeds FLOPs time at every batch size the book plots
(scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).

## Shrinking it (architecture and runtime)

| Technique | Effect | Note |
|---|---|---|
| GQA / MQA (fewer KV heads) | `N/K×` smaller; raises attention intensity | LLaMA 2-13B with 8 KV heads: 5× smaller cache, max batch 16 → 64 fits, peak throughput 964 → 4,529 tokens/s (~4.7×) (scaling-book, Part 7 (`index.py get scaling-book-inference`)) |
| Local / sliding attention layers | caps cache length for those layers | mixed with global layers (scaling-book, Part 7 (`index.py get scaling-book-inference`)) |
| KV sharing across layers | smaller cache | may re-read from HBM, so step time need not improve (scaling-book, Part 7 (`index.py get scaling-book-inference`)) |
| Quantization (int8, int4, fp8) | halves/quarters bytes and bandwidth; often post-training. The step-time gain is smaller where dequantizing runs on the VPU: an int4 KV cache measured 1.19–1.27× on v6e (`index.py get v6e-measured-decode-tricks`) | (scaling-book, Part 7 (`index.py get scaling-book-inference`); see quantization (`index.py get concept-quantization`)) |
| Ragged reads / paged attention | read only the non-padding part; page tables avoid padding | runtime-only, adds complexity (scaling-book, Part 7 (`index.py get scaling-book-inference`)) |

Together these can shrink the cache by over an order of magnitude versus MHA
(scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## Sharding the cache

Never replicate it. Megatron-shard over KV heads first (limited to `K` ways), then over
the batch: `KV[2, B_Z, S, K_Y, H]`, which costs two AllToAlls per attention layer on the
small `Q` and output activations (`Z→B` before attention, `Z→M` after). Sequence-shard it
when batch is small or context long, paying a Q AllGather and a Flash-style accumulation
(scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## Prefix caching and eviction

Autoregression means shared prefixes (chat history, few-shot/system prompts) share KV
entries, so prefill can be skipped for all but new tokens — stored as an LRU trie in spare
HBM or in host DRAM (fast enough to read, not to decode from), with affinity routing so
follow-ups land on the replica holding the cache
(scaling-book, Part 7 (`index.py get scaling-book-inference`)). Steady-state eviction is
`B · (P + G) / G` tokens per step for prefill length `P` and decode length `G`
(scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).

## See also

- Prefill vs decode (`index.py get concept-prefill-vs-decode`), HBM (`index.py get concept-hbm`),
  Transformer FLOPs accounting (`index.py get concept-transformer-flops-accounting`),
  Size a serving topology (`index.py get playbook-size-serving-topology`)

## Sources

- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
- How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-inference`)
