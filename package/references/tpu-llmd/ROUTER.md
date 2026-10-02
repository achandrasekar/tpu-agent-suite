---
references:
  - references/tpu-llmd/KV_CACHE.md
---
# The router: profiles, plugins and what the hot loop may touch

The study's router is llm-d's Endpoint Picker (EPP), installed by the
standalone chart with an Envoy in front of it, configured by one
`EndpointPickerConfig` document the renderer writes from
`router.profile` and `router.plugins` in `llmd-config.json`
(https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md).
Four named profiles are rendered; a profile is a fixed plugin set, and the
hot loop moves weights and parameters within it.

## The profiles

| profile | plugins | what it optimises | from |
| :--- | :--- | :--- | :--- |
| `sticky-until-saturated` (baseline) | `approx-prefix-cache-producer`, `inflight-load-producer`, `prefix-cache-affinity-filter`, `token-load-scorer` | send to the pod that already holds the prefix until its estimated TTFT penalty says it is saturated, then to the pod with the fewest in-flight tokens | llm-d's optimized-baseline router values at the pinned release (https://github.com/llm-d/llm-d/blob/main/guides/optimized-baseline/README.md) |
| `prefix-weighted` | `approx-prefix-cache-producer`, `prefix-cache-scorer`, `kv-cache-utilization-scorer`, `queue-scorer`, `max-score-picker` | a weighted sum of prefix overlap, KV-cache headroom and queue depth, highest score wins | llm-d's documented scorer set (https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/scheduling.md) |
| `load-only` | `kv-cache-utilization-scorer`, `queue-scorer`, `max-score-picker` | balance only; no prefix signal at all | the control arm |
| `session-sticky` | `session-affinity-scorer`, `queue-scorer`, `max-score-picker` | pin a conversation to the pod that served its earlier turns | multi-turn workloads |

Why one profile is an arm and not a knob: a profile swap changes which data
producers run and what the scorers see, so a weight tuned under one profile
means nothing under another. `deploy reconfigure` refuses `router.profile`
as a cold path.

## The hot knobs, by plugin

`router.plugins.<plugin type>.weight` sets the scorer's weight in the
scheduling profile; any other key under `router.plugins.<plugin type>.`
goes into the plugin's `parameters`. The renderer rejects a plugin the
profile does not contain. Parameter names and defaults below are from the
plugin READMEs under the llm-d-router repository
(https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins); the
affinity filter's at
https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity/README.md.

| plugin | parameter | default | effect |
| :--- | :--- | :--- | :--- |
| `prefix-cache-affinity-filter` | `affinityThreshold` | 0.80 | prefix-match score at or above which a pod counts as sticky |
| | `peakPrefillThroughput` | 15928 tok/s | the per-pod prefill rate used to turn in-flight tokens into an estimated TTFT penalty; the default is calibrated for a 32B model on a GPU and **must be set from the recipe's measured prefill throughput on TPU** |
| | `maxTTFTPenaltyMs` | 18000 | estimated TTFT penalty at which stickiness breaks; `0` means always stick |
| | `explorationProbability` | 0 | chance of skipping the gate so other pods warm their caches |
| `token-load-scorer` | `queueThresholdTokens` | 4194304 | in-flight tokens at which a pod scores zero |
| `prefix-cache-scorer` | `matchLengthWeight` | 0.0 | mixes absolute matched length into the ratio score |
| | `matchLengthScaleTokens` | 8192 | normaliser for the length term |
| `approx-prefix-cache-producer` | `blockSizeTokens` | 16 (clamped to 64) | hash block size; `autoTune: true` infers it from endpoint metrics |
| | `lruCapacityPerServer` | 31250 | per-pod index size |
| | `maxPrefixTokensToMatch` | 131072 | cap on prefix length hashed per request |
| `queue-scorer`, `kv-cache-utilization-scorer` | -- | -- | no parameters; only `weight` |
| `session-affinity-scorer` | `sessionIdConfig` | header-based | where the session id is read from |

The data the scorers read comes from the vLLM metrics endpoint of each pod
(`vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, ...), which is also
what `deploy collect` reads back from Managed Prometheus; a routing change
that works shows up in `kv_util_pod_stddev` falling or `prefix_hit_rate`
rising in `metrics.json`, and a weight that is too high shows up as one pod's
queue growing while the others idle.

## Standalone mode

In standalone mode the chart deploys EPP with an Envoy proxy that intercepts
traffic and routes it through EPP, and no `InferencePool`, `HTTPRoute` or
Gateway API class is needed on the cluster
(https://github.com/llm-d/llm-d-router/blob/main/config/charts/README.md).
The release's router Service is what the bench Jobs and, after promotion,
the user's clients call. `gke_gateway` mode renders the same EPP config
behind the GKE Inference Gateway instead; the config and the knobs are the
same, only the front door changes.

## Flow control and priorities (`router.flow_control`)

With `featureGates: [flowControl]` in the EPP config, a saturated pool
queues requests instead of sending them to a full pod; without it only
sheddable (negative-priority) requests are rejected with 429
(https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/flow-control.md).
The renderer emits the gate, a `flowControl` block and the detector from
`router.flow_control`:

| config key | EPP key | default |
| :--- | :--- | :--- |
| `saturation_detector.type` | `saturationDetector.pluginRef` | `utilization-detector` |
| `saturation_detector.queueDepthThreshold` | detector parameter | 5 |
| `saturation_detector.kvCacheUtilThreshold` | detector parameter | 0.8 |
| `max_requests`, `max_bytes`, `default_request_ttl`, `enable_eviction` | `flowControl.maxRequests` ... | 0 (unlimited), 0, client deadline, false |
| `priority_bands[].priority` | `flowControl.priorityBands[]` and one `InferenceObjective` per tenant priority (`router.inferenceObjectives` in the chart) | -- |

Clients select a band with the `x-llm-d-inference-objective` header. The
objective is named `<release>-<tenant>` (an `InferenceObjective` is a
namespaced object, and a tenant name alone collided when a study arm and
the production router shared a namespace); the bench Job sets that name
per tenant, and the HANDOFF runbook tells production clients the same.
The two detector thresholds are hot paths.

## Predicted latency (`router.predicted_latency: true`)

Enables the chart's `router.latencyPredictor` (a training server and
prediction server beside the EPP) and the alpha plugins
`predicted-latency-producer`, `latency-scorer`, `slo-headroom-tier-filter`
and `latency-slo-admitter`, which need `--allow-experimental-plugins`
(https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/README.md).
Clients may send `x-llm-d-slo-ttft-ms`; the renderer does not set it. The
sidecars cost CPU on the control-plane nodes (`llmd.sizing` in the
environment), which is why it is an arm with its own release.

## HA and the front door (`router.epp_replicas`, `epp_ha_mode`, `proxy_*`)

`router.epp.replicas > 1` makes the chart add
`--ha-enable-leader-election` (active-passive); `epp_ha_mode:
active-active` turns that flag off so every replica picks
(https://github.com/llm-d/llm-d-router/blob/v0.11.0/config/charts/README.md).
`proxy_mode: sidecar` (default) runs Envoy beside the EPP; `service` runs
it as its own Deployment with `proxy_replicas`. None of these change a
routing decision, so they are promotion settings first and arms only when
the EPP itself is the bottleneck (`llm_d_epp_*` latency in the profile).

## Precise prefix routing

`kv.precise_routing` swaps the approximate producer for the KV-event
driven one and needs model-server flags; it is described with the other
reuse mechanisms in `{{reference:tpu-llmd/KV_CACHE.md}}`.
