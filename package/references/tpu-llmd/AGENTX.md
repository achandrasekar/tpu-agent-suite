---
scripts:
  - scripts/tpu-llmd/agentx_leaderboard.py
references:
  - references/tpu-llmd/agentx/leaderboard.json
  - references/tpu-llmd/agentx/qwen35-397b-tpu7x-llmd.serving-recipe.json
---
# AgentX: hill-climbing the InferenceX agentic-coding leaderboard

How the suite runs the SemiAnalysis InferenceX **AgentX** benchmark against
an llm-d xP:yD fleet, searches the operating point, the P/D shape, the
router and the per-role engine knobs, and draws the result the way the
leaderboard draws it. Everything here is re-runnable; nothing is tied to a
particular study.

Sources: the InferenceX harness (`SemiAnalysisAI/InferenceX`:
`benchmarks/multi_node/agentic/`, `utils/agentic/`, `runners/`), and the
dashboard application `SemiAnalysisAI/InferenceX-app` (`gpu-keys.ts`, the
views API).

## 1. What the benchmark measures

| term | definition (InferenceX) | where the suite computes it |
|---|---|---|
| point | one aiperf run at a fixed number of concurrent agentic sessions (`--concurrency C`) replaying `semianalysisai/cc-traces-weka-062126-256k` with aiperf pinned at `754356e9…`; the harness drains the engines and `POST /reset_prefix_cache` on every engine before each point | `bench.run_agentx_bench` -> one Job per point (`render.agentx_bench_job`) |
| total tok/s per chip | (input + output tokens) / s over the profiling window, divided by **physical** chips (tpu7x: 4 per host; the dashboard's `tp` field) | `agentx_aggregate.py` in the Job; `metrics.total_tokens_per_s_per_chip` in the report |
| interactivity (dashboard x axis) | **P90 interactivity** = 1 / p90 of the per-request inter-token latency (aiperf `full_response_itl` when present, else `itl`), in tok/s/user | `interactivity_p90`, `interactivity_source` in each point's `tenant.replicates[].agentx` |
| qualifying gate | TTFT p50 <= 5 s, plus the harness's error gate (<= 10 % failed requests); the leaderboard itself publishes every point | tenant `agentx` SLO `ttft_p50_ms: 5000` (override in the workload spec) and `availability.error_rate_max: 0.10` |
| submission-valid | a point of >= 900 s profiling after warm-up on the full ladder | `agentx.submission_valid` in the report; shorter runs are allowed for search with `duration_s` + `unsafe` override and are flagged `(short run)` |
| tokens per dollar | tok/s/chip x 3600 / price per chip-hour; the dashboard's price table (`pins.AGENTX_PRICES`, from InferenceX-app `HW_REGISTRY`): tpu7x **$1.03** internal (dashboard default), $1.21 external, $2.00 rental; gb300 2.31, b300 2.26, b200 1.73, mi355x 1.50, h200 1.22, h100 1.17, mi325x 1.10, mi300x 0.95 | `agentx_leaderboard.py build --tco-basis` |
| frontier | per series, the dashboard's `upper_left` Pareto set (maximise interactivity and throughput); dominated points are shown but not on the line | `agentx_leaderboard.pareto` |

## 2. Inputs

### 2.1 Serving recipe

Start from `{{reference:tpu-llmd/agentx/qwen35-397b-tpu7x-llmd.serving-recipe.json}}`
and fill in what is yours (image digest, `source.ref`, `source.token_secret`,
the weights PVC). The fields that matter for AgentX:

| field | purpose |
|---|---|
| `launch.command` | the role script the engines run (the InferenceX `*_server.sh`), checked out by the `source` init container; the suite never edits it |
| `launch.role_env.{prefill,decode}` | per-role env the script reads (`MAX_NUM_SEQS_DECODE`, MTP tokens, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900`, …) |
| `launch.knobs.<name>` | how a config knob reaches the script: `{env|arg, roles, unit}`; only mapped knobs become arms (`ENGINE_KNOB_UNMAPPED` otherwise). Mapped today: `max_num_seqs`, `kv_cpu_offload_bytes` (GiB), `hma_max_inflight_pulls`, `mtp_num_speculative_tokens` |
| `launch.ports`, `launch.sidecar_args`, `launch.readiness_path`, `launch.startup_timeout_s` | engine ports per role, the disagg sidecar's flags, readiness |
| `source.patches[]` | unified diffs applied with `git apply` after the clone, for what the pinned ref lacks: the reference recipe carries `patches/qwen35-disagg-server-kv-cpu-offload.patch`, which wires `PREFILL_/DECODE_KV_CPU_OFFLOAD_BYTES` into vLLM (`TPUMultiConnector` + `TPURaidenOffloadingConnector`); the release stores the diff and its sha256 |
| `router_images.{epp,sidecar}` | the router images **by digest** the submission used; folded into the chart values |
| `pod.*` | privileged, volumes/mounts (weights PVC, compile cache), init containers (`vm.max_map_count`), tolerations, grace period |
| `measured.sessions_per_{prefill,decode}_engine` | sizing priors for the opening shape (default priors: 96 / 112 per tpu7x host, labelled PRIOR until the suite measures them) |

### 2.2 Workload spec

```json
{
  "name": "qwen35-agentx",
  "workload": {"agentx": {
      "concurrency": [96, 128, 160, 192, 224, 256],
      "duration_s": 900,
      "warmup_requests_per_lane": 1,
      "tokenizer": "Qwen/Qwen3.5-397B-A17B-FP8"}},
  "tenancy": [{"name": "agentx", "slo": {"ttft_p50_ms": 5000}}],
  "availability": {"error_rate_max": 0.10},
  "budget": {"max_chips": 48, "accelerator": "tpu7x"},
  "search": {"engine_loop": {"enabled": true, "max_escalations": 2}}
}
```

`workload.agentx` resolves (`workload._resolve_agentx`) to a profile with
no target rate, primary metric `total_tokens_per_s_per_chip`, one tenant
`agentx`, and the ladder in `concurrency`. A config's operating point is
`workload.concurrency`; `deploy bench --stage slo` runs that one point,
`--concurrency C` another, `--stage sweep` the whole ladder (the
submission's full curve).

## 3. Running one point by hand

Prerequisites on the cluster, once: the HF token Secret
(`llmd.hf_token_secret`), the repo token Secret the recipe's `source`
names, and the weights claim the recipe mounts. `preflight --recipe`
reports all three; the claim is staged by the suite:

```bash
RT="python3 framework/runtime.py deploy --package torchtpu-agents --workspace $WS"
$RT preflight --fix --recipe recipe.json
# read-only Hyperdisk ML layout (populate <pvc>-rw, snapshot, clone ReadOnlyMany):
$RT stage-weights --recipe recipe.json --pvc qwen-cache-ro --size 500Gi \
    --storage-class hyperdisk-ml-sc --ro-via-snapshot pd-snapshot-class --tolerate-tpu
```

Then per point:

```bash
$RT render --name ax --recipe recipe.json --workload spec.json --replicas 2 --out $REL
$RT apply  --name ax --release $REL
$RT wait   --name ax --release $REL --timeout 3600   # repeat; weights + compile take longer
$RT bench  --name ax --release $REL --stage slo --concurrency 128 --out $OUT/c128
$RT bench  --name ax --release $REL --stage sweep --out $OUT/ladder   # full curve
```

The bench Job (`ghcr.io/astral-sh/uv:python3.11-bookworm`, sizing
`llmd.sizing.aiperf`, default 12Gi/32Gi: aiperf reconstructs the whole
trace in memory) runs the harness's exact aiperf command, then the
in-Job reducer; the runner samples every engine's `/metrics` through the
pod proxy during the window (prefill/decode queue, KV usage, prefix hits),
and writes `llmd-bench-report.json` with `agentx.points[]`. A sampling
miss is a missed sample, never a failed point.

## 4. The search

`seed_arms.py` on an AgentX spec opens on the sessions-per-engine shape
(sessions / per-engine capacity, rounded up, under `budget.max_chips`),
router `agentx-turn-priority`, and queues:

| arm kind | sets | why |
|---|---|---|
| `operating_point.concurrency` | next rung of the ladder | total/chip rises with sessions until the gate breaks |
| `router.profile` | `agentx-session-affinity` (the InferenceX "nofc" config) | A/B flow control |
| `engine.role_knob` | one mapped knob on one role | `max_num_seqs` decode, offload bytes prefill, … |
| `pd_shape` | +1P or +1D | where attribution puts the cliff |

Every later round `plan_arms.py` calls `agentx_search.propose`, which reads
the last point's role samples and hit rates and attributes the cliff
(`agentx_search.attribute`): prefill cache thrash (GPU hit < 0.6 with
prefill queueing) -> add a prefill; decode KV saturation (running ~=
`max_num_seqs` x pods, KV > 0.85) -> add a decode or raise `max_num_seqs`;
gate broken with neither -> prefill; gate holding with both idle -> climb
concurrency. When the fleet search plateaus below the objective the usual
engine escalation applies (`ESCALATION.md`): the bound role's recipe goes to
`tpu-performance-optimization` and the improved recipe re-enters the study.

## 5. The leaderboard view

```bash
python3 {{script:tpu-llmd/agentx_leaderboard.py}} build \
    --study-dir $STUDY --leaderboard {{reference:tpu-llmd/agentx/leaderboard.json}} \
    --tco-basis internal --interactivity intvty_p90 --png --out $STUDY/leaderboard
```

writes `leaderboard.md` (this study's points, best qualifying point per
configuration, the leaderboard frontiers, peak per series),
`throughput_vs_interactivity.svg` (total tok/s/chip vs P90 interactivity),
`tokens_per_dollar_vs_interactivity.svg` (at the chosen price basis),
`points.csv`/`points.json`, and PNGs when a headless Chrome is installed
(ImageMagick's own SVG renderer draws these charts wrong; browsers and
Chrome are right). Our series are solid lines with concurrency labels,
leaderboard series dashed; both follow their Pareto frontier with dominated
points faded; a hollow marker failed the gate.

`{{reference:tpu-llmd/agentx/leaderboard.json}}` is the leaderboard reference
file (`entries: []` by default). Populate or refresh it with

```bash
python3 {{script:tpu-llmd/agentx_leaderboard.py}} fetch-live \
    --merge {{reference:tpu-llmd/agentx/leaderboard.json}} \
    --out {{reference:tpu-llmd/agentx/leaderboard.json}}
```

(`GET https://inferencex.semianalysis.com/api/v1/views/inference?model=…&sequence=agentic-traces&metric=tpPerGpu&percentile=p90&xmode=interactivity&allPoints=true`),
or add a run from InferenceX agg JSONs with `import-inferencex`.

## 6. Caveats the reader should know

* Interactivity on the dashboard is **P90**, from the full-response ITL.
  The suite's reducer emits it from aiperf's per-record ITL when the
  export carries it (`interactivity_source` says which); a run whose
  export lacks per-record ITL falls back to the aggregate percentiles.
* Qualifying is TTFT p50 <= 5 s; the public
  leaderboard shows every point. Charts draw non-qualifying points hollow.
* Simulated fleets (`engine_flavor: sim`) exercise the whole pipeline but
  their numbers say nothing about a TPU.
* The reference recipe carries `source.patches` so `PREFILL_KV_CPU_OFFLOAD_BYTES`
  reaches the role script when the upstream script does not yet wire it.
* An engine-study win comes back through `recipe_bump.py`: on a launch
  recipe, env deltas land on `launch.role_env.<escalated role>` (every role
  for an aggregated escalation) and a vLLM flag reaches the script only
  through a `launch.knobs` entry that maps it; anything else is reported
  `unapplied` for a person to carry into the script.
