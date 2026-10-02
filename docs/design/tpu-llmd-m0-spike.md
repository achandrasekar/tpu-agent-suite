# tpu-llmd M0 spike: architectural findings for the deploy runtime

The design ([tpu-llmd.md](tpu-llmd.md) §14) scoped M0 as: probe the engine
image, deploy the llm-d router, and measure a fleet baseline. Five findings
from the spike shaped the runtime.

## 1. Adopted fleets vs. managed fleets

The initial design assumed the study always renders and owns the model servers
(`managed`). On clusters where model servers are already running and hold the
available TPU slices, a study that re-deploys them cannot run. M0 added a
second fleet mode to the serving recipe:

```json
"fleet": {"mode": "adopted", "selector": {"app": "model-server"},
          "service": "model-server", "port": 8000, "replicas": 2}
```

An adopted release renders no model-server manifests, selects the running
pods by label for the router's `InferencePool`, benchmarks against either
the fleet Service (`router.profile: passthrough`, the control) or the
router, and refuses any config path that would change the servers
(`ADOPTED_FLEET_IMMUTABLE`: `topology.mode`, `topology.prefill`, `kv.*`,
`engine.*`, replica count). The study's baseline on an adopted fleet is
kube-proxy spreading over the Service, with no router. The seeder and planner
write the engine-touching arms as `blocked_on_adopted_fleet` rather than
dropping them, so the report still states what the workload would want from a
managed fleet.

## 2. In-pod capability probing and `engine_flavor: torchtpu`

When no spare TPU chip is free for a standalone probe Job, `deploy probe`
supports an `exec` mode (`--selector app=<label>`) that imports the connector
modules inside a running serving pod with JAX and PyTorch pinned to CPU:

| capability | module |
| :--- | :--- |
| `pd_connector` | `vllm_torchtpu.distributed.kv_transfer.tpu_connector` (classes `TPUConnector`, `TPURaidenConnector`) |
| `offload_connector` | `vllm_torchtpu.offload.raiden_connector` (`TPURaidenOffloadingConnector`) |
| `kv_events` | `vllm.distributed.kv_events` |

The renderer supports `engine_flavor: torchtpu` with those module and class
names alongside `tpu_inference`.

## 3. Control-plane pod sizing

CPU node pools on shared clusters often cannot schedule the upstream guide's
default bench Job and router sizes. The environment schema includes
`llmd.sizing.{bench,epp,proxy,aiperf}`, and the renderer applies those
requests and limits to the bench Job, the EPP, and the Envoy proxy.

## 4. Load-generator sizing and `load.client_limited`

**In-flight ceiling.** `inference-perf` caps requests in flight at
`num_workers x worker_max_concurrency` (1 x 100 by default). When open-loop
conversational load requires more concurrent requests than that ceiling, the
load generator throttles itself and reports a high schedule delay. The
renderer sizes worker count and per-worker concurrency from the target rate
(`workload.loadgen_sizing`), and the bench report carries a
`load.client_limited` gate (achieved rate below 90% of requested, or mean
schedule delay over 1 s) that marks a client-limited run inadmissible instead
of recording a false tie.

**Connection pooling and retries.** Worker count is kept low (one per ~10 rps)
with higher per-worker concurrency to avoid stale keep-alive disconnects, and
the renderer sets `load.request_retries: 2`, `load.request_timeout: 120`, and
`stage_teardown_grace_seconds: 60`.

## 5. Operating near the knee and `sweep` stage rate ladders

`goodput_at_slo_per_chip` at the `slo` stage is bounded by
`target_qps / chips`. When the baseline already serves every request inside
the SLO at the target rate, every arm ties at the ceiling; the discriminating
regime is near the knee. The `sweep` stage renders an explicit `rate.sweep`
open-loop rate ladder so the knee can be located before locking `target_qps`
for the study.

## 6. Router chart CRDs, release naming, HA readiness, and Prometheus labels

- **Standalone router CRDs:** `llm-d-router-standalone` renders each
  `router.inferenceObjectives` entry as an `InferenceObjective` in the
  `llm-d.ai` API group. `preflight` checks `crd.inferenceobjective` and
  `--fix` applies the pinned `llm-d-router` CRD manifest alongside the GAIE
  CRDs.
- **Co-existence in one namespace:** The standalone chart's Envoy preset
  defaults its ConfigMap name to `envoy`. The renderer sets
  `router.proxy.configMap.names.envoy` to `<release>-epp-envoy` and prefixes
  `InferenceObjective` names with `<release>-` so a study release and a
  promoted production release can share a namespace.
- **EPP active-passive HA readiness:** With `router.epp.replicas: 2` and
  leader election enabled, the standby replica deliberately reports unready
  until it acquires the lease (`1/2` Ready). `wait` honours `min_ready: 1`
  for active-passive EPP deployments.
- **Managed Prometheus queries:** Samples carry `job="<release>-modelserver"`,
  and TPOT queries accept either `vllm:inter_token_latency_seconds` or
  `vllm:time_per_output_token_seconds`.
