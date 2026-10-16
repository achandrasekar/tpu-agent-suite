# tpu-llmd M6 — InferenceX AgentX: hill-climbing the leaderboard with llm-d xPyD on tpu7x

Addendum to [tpu-llmd.md](tpu-llmd.md) and [tpu-llmd-m5.md](tpu-llmd-m5.md).
Status: **M6a–M6e complete.**

## 1. Goal

Run the SemiAnalysis **InferenceX AgentX** benchmark (the agentic-coding
trace replay) through this agent suite for **Qwen3.5-397B-A17B-FP8 on
tpu7x behind llm-d**, and hill-climb the leaderboard objective across
**concurrency points**, **xP:yD shapes**, **engine knobs** and **routing**.

## 2. The contract to reproduce (from the InferenceX harness)

**Harness.** aiperf at submodule pin `754356e9`, run as
`aiperf profile --scenario inferencex-agentx-mvp --url <front door>
--endpoint /v1/chat/completions --endpoint-type chat --streaming --model
<served name> --tokenizer <HF id> --concurrency <C> --benchmark-duration
<D> --stats-interval 30 --random-seed 42 --failed-request-threshold 0.25
--trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
--warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds <cap>
--warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry
--tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration
1.0 [--server-metrics <prefill /metrics urls>] --output-artifact-dir <dir>
[--unsafe-override when D < 900] --public-dataset
semianalysis_cc_traces_weka_062126_256k`
(`benchmarks/benchmark_lib.sh: build_replay_cmd`, `resolve_trace_source`).
The dataset is the public, ungated HF dataset
`semianalysisai/cc-traces-weka-062126-256k`, downloaded with `hf download
--repo-type dataset`. Fast mode (`AIPERF_EXPERIMENTAL_FAST=1`) is 1200 s
with 1 warm-up request per lane. The client sets
`AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true` (session
headers the EPP's affinity plugins read) and
`AIPERF_HTTP_TCP_USER_TIMEOUT=900000`
(`benchmarks/multi_node/agentic/qwen3.5_fp8_tpuv7_vllm-disagg-llmd.sh`).
Between concurrency points the client waits until every engine reports
`vllm:num_requests_running + waiting == 0`, and the driver resets prefix
caches (`POST /reset_prefix_cache`, falling back from
`?reset_external=true`).

**Metrics.** From `profile_export.jsonl`, keeping records whose
`metadata.benchmark_phase == "profiling"` and that carry no `error`:
`metrics.time_to_first_token` (ms), `request_latency`,
`inter_token_latency` (TPOT is the ITL distribution),
`input_sequence_length`, `output_sequence_length`; duration =
`max(metadata.request_end_ns) − min(request_start_ns)`; `total_tput_tps =
(Σ in + Σ out) / duration`, `output_tput_tps = Σ out / duration`;
per-chip = `/ num physical chips` (TP8 on a 4-chip tpu7x host counts 4;
`LOGICAL_DEVICES_PER_CHIP`); interactivity = 1 / ITL
(`utils/agentic/aggregation/request_metrics.py`,
`process_agentic_result.py`). Prefix-cache hit rates come from aiperf's
`--server-metrics` scrape of the prefill engines (GPU/HBM and CPU tiers).

**Objective.** Leaderboard ranking on **total tok/s/chip** with
**TTFT p50 ≤ 5 s** as the qualifying gate; output tok/s/chip reported
alongside. Points are concurrency values; a submission lists the
concurrencies it qualifies at.

**Engines.** One whole `tpu7x-standard-4t` node (2x2x1, 4 chips, TP8/EP8)
per engine; `vllm serve` is driven by a role script
(`qwen3.5_fp8_tpuv7_vllm-disagg_server.sh`, `ROLE=prefill|decode`),
prefill MTP k=1 / decode MTP k=3 with synthetic acceptance,
`TPURaidenConnector` KV transfer (`kv_producer` on 8000, `kv_consumer` on
7200), optional host-DRAM prefix cache (`PREFILL_KV_CPU_OFFLOAD_BYTES` →
`TPUMultiConnector` + `TPURaidenOffloadingConnector`),
`MAX_NUM_SEQS_DECODE`, `TPU_HMA_MAX_INFLIGHT_PULLS`,
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900`, weights from a ReadOnlyMany PVC at
`/mnt/models-ro`, compile caches on a hostPath or PVC, privileged
containers with a sysctl init step (`runners/launch_tpuv7-llmd-gke.sh`).
The decode pod carries the llm-d P/D sidecar `--port 8000
--model-server-port 7200 --kv-connector nixlv2`.

**Router.** llm-d-router *main* images (EPP `sha256:713f62c0…`, sidecar
`sha256:98d3020e…`; v0.11.0 does not accept the `llm-d.ai/v1` config),
`--allow-experimental-plugins=true`, one EPP replica, and one of two
configs: `turn-priority-config.yaml` (agent-identity, program-aware
fairness, flow control, always-disagg P/D, per-profile session affinity,
active-request scorer) or `nofc` (the same without `flowControl` and the
fairness plugin), or the hybrid `agentx-prefix-token-session` profile.

**Sizing priors:** default 96 sessions per prefill engine and 112 sessions
per decode engine (`recipe.measured.sessions_per_{prefill,decode}_engine`).

## 3. Architecture and design for each component

| gap | design |
| :--- | :--- |
| Engines are `vllm serve <args>`; AgentX engines are a role script with env | **Recipe `launch`**: `command` (container entrypoint), `role_env.{prefill,decode}`, `ports.{prefill,decode}`, and a **knob map** `knobs.<name>: {env: X} \| {arg: --x}` that tells the renderer how each whitelisted engine knob reaches *this* engine. `topology.<role>.engine.*` overlays resolve through it, so `max_num_seqs` on decode becomes `MAX_NUM_SEQS_DECODE=48` here and `--max-num-seqs=48` on a plain image. |
| Pod needs PVCs, hostPath caches, init containers, privileged | **Recipe `pod`**: `volumes`, `volume_mounts`, `init_containers`, `privileged`, `security_context`, `tolerations`, `annotations` — rendered verbatim into the model-server pod after schema validation. A `source.repo/ref/token_secret` block renders the clone init container the InferenceX pods use, so the role script comes from the InferenceX checkout the user names. |
| Router images and config are pinned to v0.11.0 | **Recipe `router_images.{epp,sidecar}`** override the pins; **router profiles `agentx-turn-priority`, `agentx-session-affinity`, and `agentx-prefix-token-session`** carry the AgentX EPP configs (apiVersion `llm-d.ai/v1`, `featureGates`, EPP flag `--allow-experimental-plugins=true`). |
| Workload is open-loop (rps) through inference-perf | **`workload.agentx`** source kind: scenario, dataset loader, `concurrency` list or `{start, step, max}`, `duration_s`, fast mode, served name, tokenizer. **aiperf bench driver**: a Job that installs aiperf at the pin, downloads the dataset, runs the exact command, computes the InferenceX aggregate in-Job (the formulas above, as a script shipped in the ConfigMap), and echoes it. The bench runner drains engines and resets prefix caches between points through the API-server pod proxy. |
| Metrics and gates are goodput/SLO-percentile | **Primary metrics** `total_tokens_per_s_per_chip` (ranking) and `output_tokens_per_s_per_chip`; **gates** `agentx.ttft_p50_ms ≤ 5000`, `agentx.error_rate`, `budget.max_chips`; the report gains an `agentx` block (TTFT p50/p90/p99, TPOT, interactivity, E2E, ok/total, GPU/CPU hit rates, duration, chips). |
| Concurrency is fixed by the workload | **`workload.concurrency` is a hot knob in `llmd-config`** (no redeploy): the arm is (config, concurrency). Planner arm kind `operating_point.concurrency` climbs by the spec's step while the gate holds and total/chip rises past the noise floor, and backs off on a failed gate. |
| Shape sizing uses token rates | For AgentX the opening shape and neighbours use **sessions per engine**: `P ≥ ⌈C / sessions_per_prefill⌉`, `D ≥ ⌈C / sessions_per_decode⌉`, from `recipe.measured.sessions_per_prefill_engine / sessions_per_decode_engine` (defaults 96 / 112, labelled priors). |
| Attribution reads inference-perf gates | **Cliff signatures** from role metrics sampled during the run through the pod proxy (`vllm:num_requests_running/waiting`, `kv_cache_usage_perc`, prefix hit/query counters): prefill cliff = hit rate < 0.6 and prefill waiting high → prefill-bound; decode cliff = running ≈ `max_num_seqs` and KV > 0.85 → decode-bound; both → lower concurrency first. Works on kind (no Prometheus). |
| No leaderboard view | **`agentx_leaderboard.py`**: summary table over every measured point, the best qualifying point per configuration, and the two charts the dashboard draws -- total tok/s per chip and total tokens per dollar against **P90 interactivity** -- with the leaderboard series on the same axes. Axes, prices and frontier rule are the dashboard's own (`InferenceX-app`): x = 1 / p90 inter-token latency (full-response ITL), y per *physical* chip (`tp`), tokens/$ = tok/s/chip x 3600 / price, prices per TCO basis (tpu7x $1.03 internal / $1.21 external / $2.00 rental; `pins.AGENTX_PRICES`), per-series `upper_left` Pareto frontier. The leaderboard reference file (`references/tpu-llmd/agentx/leaderboard.json`) refreshes with `fetch-live`. |

## 4. Search space and the loop

Hot (rebench only): `workload.concurrency`, `router.profile`,
`router.flow_control`, `router.plugins.*`. Warm (roll one pool):
`topology.decode.engine.max_num_seqs`, `topology.prefill.engine.kv_cpu_offload_bytes`,
`topology.decode.engine.kv_cpu_offload_bytes`, `engine.hma_max_inflight_pulls`,
MTP k per role. Cold (redeploy): `topology.pd_shape`. Order per round:
climb concurrency on the incumbent shape → router A/B → per-role engine
knobs → shape neighbour by attribution; the plateau rule and the M5
engine escalation apply unchanged (objective unmet = TTFT p50 gate failing
at the best admissible point, or `objective.target` not reached), and
the escalation's operating point is the AgentX concurrency and the
measured ISL/OSL.

## 5. Milestones

| milestone | contents | exit |
| :--- | :--- | :--- |
| **M6a** engine launch + pod | recipe `launch`, `pod`, `source`, `router_images`; knob map; AgentX router profiles; tests | the Qwen3.5 recipe renders pods equivalent to the InferenceX launcher's (same env, mounts, sidecar flags, EPP config) |
| **M6b** AgentX workload + aiperf | `workload.agentx`, aiperf bench Job + in-Job aggregate, report block, gates, primary metrics, concurrency as a config knob, drain + prefix reset | a point benches on the kind sim fleet and the report carries total tok/s/chip and TTFT p50 |
| **M6c** search | concurrency climb arm, sessions-per-engine sizing, cliff attribution from pod-proxy samples, planner ordering, escalation operating point | the planner walks c → shape → knobs on kind and records a study |
| **M6d** leaderboard view | `agentx_leaderboard.py` (`build`, `fetch-live`, `import-inferencex`), `svgchart.py`, dashboard prices in `pins`, P90 interactivity in the reducer and report, docs | `leaderboard.md` and the two charts from a study |
| **M6e** validation & pre-work | kind validation; `deploy stage-weights`; `source.patches[]`; per-role `recipe_bump` | unit-tested and kind-verified |

## 6. Key implementation details validated on kind

| area | implementation detail |
| :--- | :--- |
| aiperf `/metrics` scraping | reducer counts only the endpoints the runner gave the Job (`AGENTX_SERVER_METRICS_URLS`) so the router proxy Service is not double-counted |
| pod discovery during rollout | `engine_pods` skips `deletionTimestamp` and non-Running pods |
| container log rotation | Job echoes a trimmed `server_metrics_summary.json` (timeslices stripped) and echoes `agentx_point.json` last; a lost point is marked `telemetry_valid: false` |
| pod-proxy timeouts | `pod_proxy_get/post` treat a transient timeout as a missed sample rather than aborting the benchmark |
| shape arm operating point | a shape arm proposed after a cliff carries the failed rung (`workload.concurrency`) so shapes are evaluated at the concurrency rung they were sized to sustain |
| protocol rendering | `render_protocol.py` supports AgentX primary metrics, closed-loop cells, and the AgentX block in `llmd-study.json` |
| weights staging | `deploy stage-weights` (`framework/deploy/weights.py`) stages model weights into a RWO PVC or snapshots and clones a ReadOnlyMany PVC (`--ro-via-snapshot`, `--from-snapshot`) |
| source patches | `source.patches[]` inlines unified diffs into a `<release>-source-patches` ConfigMap applied by the clone init container via `git apply` |
| launch-recipe bumps | `recipe_bump.py` applies env deltas to `launch.role_env.<escalated role>` and maps vLLM flags through `launch.knobs` |
