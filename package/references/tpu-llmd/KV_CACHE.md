---
references:
  - references/tpu-llmd/ROUTER.md
---
# KV cache reuse: approximate routing, precise routing, tiered host KV

Three arms of the study raise the share of prompt tokens served from cache
instead of recomputed. They differ in where the index lives and what it
costs the fleet, which is why they are separate arms and not one knob.
Parameter names and defaults are from the llm-d router plugin READMEs at
the pinned release and the llm-d guides at the pinned guides release;
the TPU numbers are from llm-d's published benchmark page, not from this
package's measurements.

## 1. Approximate prefix routing (every router profile but `load-only`)

The EPP hashes each request's prompt in blocks and keeps an LRU per pod of
the block hashes it has sent there (`approx-prefix-cache-producer`:
`blockSizeTokens` 16, clamped to 64; `lruCapacityPerServer` 31250;
`maxPrefixTokensToMatch` 131072;
https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/README.md).
It never asks the engine what it holds, so it is wrong whenever the engine
evicted a block or a pod restarted. Its cost is nothing on the fleet. This
is the mechanism behind `router.profile` arms, tuned by the hot knobs in
`{{reference:tpu-llmd/ROUTER.md}}`. `prefix_hit_rate` in the fleet profile
(`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`) is the number
that says how well it works; a shared-prefix workload under kube-proxy
spreading sits near `1/replicas` of what one pod could reach.

## 2. Precise prefix routing (`kv.precise_routing: true`)

vLLM publishes block-stored / block-removed events over ZMQ and the EPP's
`precise-prefix-cache-producer` indexes them, so the router knows exactly
which pod holds which block
(https://github.com/llm-d/llm-d/tree/v0.10.0/guides/precise-prefix-cache-routing;
https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache/README.md).
What the renderer adds for it, from the guide's TPU patch:

| side | setting | value |
| :--- | :--- | :--- |
| vLLM | `--kv-events-config` | `{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5556","replay_endpoint":"tcp://*:5559","topic":"kv@$(POD_IP):8000@<model>"}` |
| vLLM | `--block-size` | `kv.block_size` (64 in the guide) |
| vLLM | `--enable-scale-out` | exposes the `/render` endpoints the EPP's `token-producer` (`vllm` backend) needs for real token ids |
| EPP | `precise-prefix-cache-producer.kvEventsConfig.podDiscoveryConfig` | `socketPort 5556`, `replaySocketPort 5559` |
| EPP | `tokenProcessorConfig.blockSizeTokens` | equals `kv.block_size` |
| EPP | `token-producer.vllm.url` | the release's render Service |

The EPP recomputes block keys from the tokenised prompt, so the engine's
hash seed need not match; it does need real tokens, which is why the
`estimate` tokenizer backend silently degrades to approximate behaviour
(precise README above). The guide pins one EPP replica with leader
election off for this producer. Cost: ZMQ publishing on every pod and a
tokenizer call per request at the router. It is a **warm** change to the
model servers (new flags), so on an adopted fleet it is
`blocked_on_adopted_fleet`.

Prerequisite: the serving image's vLLM has `vllm.distributed.kv_events`
(`kv_events` in `{{schema:engine-capabilities.json}}`).

## 3. Tiered host KV (`kv.tiered_host`)

A KV connector in `kv_both` role spills evicted blocks to host memory and
reloads them on a hit, so the effective cache is HBM plus a slice of the
host's RAM
(https://github.com/llm-d/llm-d/tree/v0.10.0/guides/tiered-prefix-cache).
From the guide's single-host TPU patch:

| setting | guide value | config key |
| :--- | :--- | :--- |
| `--kv-transfer-config` | `{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}` | derived from `engine_flavor` in the recipe (`tpu_inference` or `torchtpu`); `kv.tiered_host.module` / `.connector` override |
| `TPU_OFFLOAD_NUM_CPU_CHUNKS` | 25000 | `kv.tiered_host.cpu_chunks` |
| `TPU_OFFLOAD_NUM_STAGING_BLOCKS` | 1000 | `kv.tiered_host.staging_blocks` |

The guide also runs a privileged init container raising `vm.max_map_count`
and `dma_entry_limit`; the renderer does not, and a pod that fails to
allocate the host tier will say so in `vllm:kv_offload_allocation_failure_total`
(the guide names it). llm-d's published TPU v7 result for Qwen3-32B is mean
TTFT 0.98 s to 0.56 s and +8.9% output tokens/s with 25000 chunks
(https://github.com/llm-d/llm-d/blob/v0.10.0/guides/tiered-prefix-cache/benchmark-results/vllm-qwen3-32b-tpuv7.md);
treat it as the ceiling's provenance, not as a prediction for another
model. The router side of the guide adds a second
`approx-prefix-cache-producer` sized for the host tier
(`lruCapacityPerServer` hand-sized, because vLLM does not export host-tier
occupancy); the renderer keeps the chosen profile and leaves that producer
to the hot loop. Also a **warm** change, so `blocked_on_adopted_fleet` on
an adopted fleet; needs `offload_connector` in the capabilities.

The host tier and P/D are not rendered together: no TPU guide ships
vLLM's MultiConnector, and the renderer refuses the pair with
`NOT_IMPLEMENTED` so the two stay separate arms.

## What the fleet profile tells the planner

| `metrics.json` field | reads as |
| :--- | :--- |
| `prefix_hit_rate` low with a shared-prefix workload | routing is not finding the holder: raise precise routing's headroom |
| `prefix_hit_rate` already high | little left in reuse; the control arm (`load-only`) decides whether affinity costs balance |
| `kv_util_mean` near 1.0 and multi-turn | HBM is the cache ceiling: the host tier's headroom |
| `kv_util_pod_stddev` high | one pod is hoarding: affinity too sticky, lower `maxTTFTPenaltyMs` or raise `peakPrefillThroughput` in the hot loop |
