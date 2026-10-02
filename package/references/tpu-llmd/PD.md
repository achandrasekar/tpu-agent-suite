# Prefill/decode disaggregation on TPU

`topology.mode: disaggregated` renders two model-server pools from one
recipe -- a prefill pool (`topology.prefill`) and a decode pool
(`topology.decode`) -- joined by a KV connector and routed by the EPP's
disaggregation profile handler. It is an arm, never a knob: it changes the
chip count, the pods, and what the scorers see. Everything below is from
llm-d's P/D guide at the pinned guides release and the router's
disaggregation docs at the pinned router release
(https://github.com/llm-d/llm-d/tree/v0.10.0/guides/pd-disaggregation;
https://github.com/llm-d/llm-d-router/blob/v0.11.0/docs/disaggregation.md).

## How a request flows

1. The EPP's `disagg-profile-handler` runs a `prefill` scheduling profile
   (filtered to pods labelled `llm-d.ai/role=prefill` by `prefill-filter`)
   and a `decode` profile (`decode-filter`) when its decider says the
   request should disaggregate.
2. The chosen prefill pod's address goes to the decode pod in the
   `x-prefiller-host-port` header.
3. The decode pod's **routing-proxy sidecar** (listening on 8000) forwards
   the prompt to the prefiller, which computes and transfers KV through the
   connector, then streams decode from its own vLLM on 8200
   (`ghcr.io/llm-d/llm-d-router-disagg-sidecar`, args `--port=8000
   --kv-connector=nixlv2 --zap-log-level=1 --secure-proxy=false`, run as
   a native sidecar init container in the guide).

The older `pd-profile-handler` / `prefill-header-handler` names are not
registered in the pinned router release and the renderer does not emit
them.

## What the renderer emits

| where | setting | value |
| :--- | :--- | :--- |
| prefill vLLM | `--kv-transfer-config` | `{"kv_connector":<flavor>,"kv_connector_module_path":<flavor>,"kv_role":"kv_producer","kv_ip":"$(POD_IP)"}` |
| decode vLLM | `--kv-transfer-config` | same with `"kv_role":"kv_consumer"`; `--port 8200` |
| both | env | `POD_IP`, `TPU_SIDE_CHANNEL_PORT=9600`, `TPU_KV_TRANSFER_PORT=9100`; ports `tpu-kv-transfer` 9100, `tpu-pd-coord` 9600 |
| decode pod | sidecar | the routing proxy above, on the pod's 8000 |
| Service | target | the decode pods' 8000 (the sidecar) |
| EPP | profile handler | `disagg-profile-handler` with `profiles.prefill/decode`, `deciders.prefill` = `topology.pd_decider` |
| EPP | filters | `prefill-filter` in the prefill profile, `decode-filter` in the decode profile; both keep the chosen `router.profile`'s scorers |

`<flavor>` comes from the recipe's `engine_flavor`: `tpu_inference` uses
`TPUConnectorHMA` in `tpu_inference.distributed.tpu_connector_hma` (the
guide's tpu7x overlay); `torchtpu` uses `TPUConnector` in
`vllm_torchtpu.distributed.kv_transfer.tpu_connector`, read from the
image. `kv.pd_connector.module` / `.connector` override either.

## The decider: when to disaggregate

`topology.pd_decider` picks the plugin:

| decider | behaviour | hot parameter |
| :--- | :--- | :--- |
| `always-disagg-pd-decider` | every request prefills remotely | none |
| `prefix-based-pd-decider` | disaggregate only when the uncached prompt is long enough | `nonCachedTokens` (0 = disabled), `promptTokens`, `prefixMatchInfoProducerName` |

The second is the one worth sweeping: short, mostly-cached turns of a
chat stay on the decode pod; long cold prompts go to the prefill pool.
`topology.pd_decider.parameters.nonCachedTokens` is a hot path in
`llmd-config.json`.

## When it is worth a slice

P/D costs one more slice (the prefill pool) before it returns anything.
The ceiling is real when decode TPOT is being disturbed by long prefills:
`tpot_p99` fails its gate while `ttft` passes, or `kv_util_mean` is low
while `requests_waiting_mean` is high at target rate. It is not the lever
for a workload whose prompts are short and mostly cached, which is why the
planner ranks it below reuse for `interactive-chat` and above it for
prefill-heavy catalog entries whose metadata lists disaggregation.

The guide notes its router values' `peakPrefillThroughput` is a GPU
number to be re-measured on TPU; the renderer leaves it to the hot loop.
The guide also pins an older image for its tpu7x example because of a
tpu-inference issue in later releases; a managed P/D arm is where that is
checked for the recipe's image, and `wait` failing is the arm's `invalid`,
not a reason to retry.

## Limits

- P/D and `kv.tiered_host` together are refused (`NOT_IMPLEMENTED`): no
  TPU guide ships vLLM's MultiConnector.
- On an adopted fleet P/D is `blocked_on_adopted_fleet`: it needs new
  pods and new flags on the ones that exist.
- Needs `pd_connector` in `{{schema:engine-capabilities.json}}`.
- Multi-host slices (LeaderWorkerSet) are not rendered.
