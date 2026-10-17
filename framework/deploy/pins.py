# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned upstream coordinates for llm-d deployments.

One file, so upgrading llm-d is a reviewed change to a handful of constants
plus a re-run of the golden-file tests, and so no prompt or reference ever
spells an OCI ref or a release URL. Every value here is recorded into a
release's `release.json` at render time, which is how a study proves what it
deployed.

Sources (all read at the pinned tag):
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/env.sh
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/recipes/router/base.values.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/optimized-baseline/router/optimized-baseline.values.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/recipes/modelserver/base/single-host/default/decode-deployment.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/optimized-baseline/modelserver/tpu/v7/vllm/patch-vllm.yaml
"""

from __future__ import annotations

# The llm-d release whose guides the renderer mirrors. The guide overlays set
# image, args, env, nodeSelector, resources and labels on one base Deployment;
# `render.py` sets the same fields from the recipe and the incumbent config.
LLMD_REF = "v0.10.0"

# guides/env.sh at LLMD_REF.
GATEWAY_API_VERSION = "v1.5.1"
GAIE_VERSION = "v1.5.0"
GAIE_CRD_URL = ("https://github.com/kubernetes-sigs/"
                "gateway-api-inference-extension/releases/download/"
                f"{GAIE_VERSION}/v1-manifests.yaml")
ROUTER_RELEASE_VERSION = "v0.11.0"
ROUTER_CHART_VERSION = "v0.11.0"
ROUTER_STANDALONE_CHART = "oci://ghcr.io/llm-d/charts/llm-d-router-standalone"
ROUTER_GATEWAY_CHART = "oci://ghcr.io/llm-d/charts/llm-d-router-gateway"
ROUTER_EPP_IMAGE = "ghcr.io/llm-d/llm-d-router-endpoint-picker"
ROUTER_EPP_VERSION = "v0.11.0"

# The EPP config document the router reads. A file in a ConfigMap, not a CRD.
EPP_CONFIG_API_VERSION = "llm-d.ai/v1alpha1"
EPP_CONFIG_KIND = "EndpointPickerConfig"

# The InferencePool API the standalone chart creates and the EPP watches.
INFERENCE_POOL_CRD = "inferencepools.inference.networking.k8s.io"

# The router's own CRDs. The v0.11.0 chart renders every
# `router.inferenceObjectives` entry as an `InferenceObjective` in
# `llm-d.ai/v1alpha2` -- the llm-d-router fork of the API, not GAIE's
# `inference.networking.x-k8s.io` one -- so a cluster with only the GAIE
# bundle fails the install with "no matches for kind InferenceObjective".
# The release publishes the two llm-d.ai CRDs as `manifests.yaml`
# (`make artifacts` in the repo: `kubectl kustomize config/crd` filtered to
# group llm-d.ai).
INFERENCE_OBJECTIVE_CRD = "inferenceobjectives.llm-d.ai"
ROUTER_CRD_URL = ("https://github.com/llm-d/llm-d-router/releases/download/"
                  f"{ROUTER_RELEASE_VERSION}/manifests.yaml")

# inference-perf: the benchmark harness and the workload catalog vendored in
# `workload_catalog/`. The catalog ref is also in CATALOG.json; the two are
# compared by a test so they cannot drift.
INFERENCE_PERF_REF = "45af1d3c5f57de7cf43866539fd9d4d7a3591ad9"
INFERENCE_PERF_IMAGE = "quay.io/inference-perf/inference-perf:latest"

# The one-shot in-cluster HTTP probe `deploy drill` runs against a release's
# front door after a fault; a pinned minor so the drill row is reproducible.
PROBE_CURL_IMAGE = "curlimages/curl:8.11.1"

# Labels this runtime puts on everything it creates, so status, teardown and
# the reaper find a release by one selector and never by name pattern.
RELEASE_LABEL = "tpu-coworker.llmd/release"
# On the weights PVC / snapshot / download Job `deploy stage-weights`
# creates; they outlive releases and the reaper leaves them alone.
WEIGHTS_LABEL = "tpu-coworker.llmd/weights"
PRODUCTION_LABEL = "tpu-coworker.llmd/production"
MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
MANAGED_BY_VALUE = "tpu-coworker"

# llm-d's own labels, kept so llm-d dashboards and the EPP's pod discovery
# recognise the pods.
LLMD_GUIDE_LABEL = "llm-d.ai/guide"
LLMD_ROLE_LABEL = "llm-d.ai/role"
LLMD_SERVING_LABEL = "llm-d.ai/inference-serving"
LLMD_ENGINE_LABEL = "llm-d.ai/engine-type"

# GKE TPU scheduling, as in compute/drivers/gke_job.py.
ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU_RESOURCE = "google.com/tpu"
# TPU7x multi-host node pools are placed by a workload policy rather than
# a topology flag; the pod names it with this selector
# (https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus).
PLACEMENT_POLICY_LABEL = "cloud.google.com/placement-policy-name"
# Chips one host (one node) exposes as `google.com/tpu`, per generation.
# A multi-host replica is `hosts x chips_per_host`; every pod of the group
# must request all of its node's chips. tpu7x: 4 chips per host, machine
# type tpu7x-standard-4t for every topology
# (https://docs.cloud.google.com/tpu/docs/tpu7x). Other generations are
# not pinned here: the recipe's `accelerator.chips_per_host` says.
CHIPS_PER_HOST: dict[str, int] = {"tpu7x": 4}

# Multi-host pools render a LeaderWorkerSet (one pod per host, the leader
# serving). The CRD is installed cluster-wide from the pinned release
# (https://github.com/kubernetes-sigs/lws/releases).
LWS_API_VERSION = "leaderworkerset.x-k8s.io/v1"
LWS_CRD = "leaderworkersets.leaderworkerset.x-k8s.io"
LWS_VERSION = "v0.11.1"
LWS_MANIFEST_URL = ("https://github.com/kubernetes-sigs/lws/releases/download/"
                    f"{LWS_VERSION}/manifests.yaml")
# Which member of a group a pod is; workers carry no serving/role label so
# every selector (Service, InferencePool, PodMonitoring) sees leaders only.
MEMBER_LABEL = "tpu-coworker.llmd/member"

# llm-d-inference-sim: the engine stand-in for a simulated fleet. Real
# router, real pools, no accelerator; unknown vLLM flags abort it, so the
# renderer emits only its own flags
# (https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/configuration.md).
SIM_IMAGE = "ghcr.io/llm-d/llm-d-inference-sim"
SIM_VERSION = "v1.0.0"

# Ports. The standalone chart's Envoy sidecar listens on 8081 and the chart's
# `extraServicePorts` maps Service port 80 to it; the EPP serves metrics on
# 9090; vLLM serves on 8000.
MODEL_SERVER_PORT = 8000
EPP_METRICS_PORT = 9090
ROUTER_SERVICE_PORT = 80
ROUTER_SIDECAR_PORT = 8081

# TPU P/D side channels used by llm-d's tpu-inference connectors
# (guides/pd-disaggregation/modelserver/tpu/base/vllm/patch-*.yaml).
TPU_SIDE_CHANNEL_PORT = 9600
TPU_KV_TRANSFER_PORT = 9100
# Header the EPP's disaggregation profile handler sets for the decode-side
# routing sidecar (llm-d-router/docs/disaggregation.md at the router tag).
PD_PREFILL_HEADER = "x-prefiller-host-port"
PD_SIDECAR_IMAGE = "ghcr.io/llm-d/llm-d-router-disagg-sidecar"
PD_SIDECAR_VERSION = "v0.11.0"
PD_DECODE_PORT = 8200  # vLLM behind the sidecar; the sidecar takes 8000
# The v0.11.0 sidecar's flags. A recipe whose router images are a newer
# main (the InferenceX AgentX stack names `--model-server-port` and
# `--secure-serving`) overrides these through `launch.sidecar_args`.
PD_SIDECAR_ARGS = ["--port={port}", "--kv-connector={connector}",
                   "--zap-log-level=1", "--secure-proxy=false"]

# The InferenceX AgentX llm-d stack for Qwen3.5-397B on tpu7x pins the
# llm-d-router *main* builds it was validated against, by digest, because
# v0.11.0 does not accept the `llm-d.ai/v1` EndpointPickerConfig its
# turn-priority config uses (SemiAnalysisAI/InferenceX
# runners/launch_tpuv7-llmd-gke.sh). Recipes name them in
# `router_images`; nothing here selects them by default.
AGENTX_EPP_IMAGE = ("ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:"
                    "713f62c0f2db1b0805c23e04c6fedea9680fa6d82185d749cb4a8e"
                    "416531b889")
AGENTX_SIDECAR_IMAGE = ("ghcr.io/llm-d/llm-d-router-disagg-sidecar@sha256:"
                        "98d3020eebe3a5605d52ef3a0cc0a0565ab7a5486f0205a33a"
                        "3397a62e1ef1fe")
AGENTX_SIDECAR_ARGS = ["--port={port}", "--model-server-port={decode_port}",
                       "--kv-connector={connector}", "--secure-serving=false"]
EPP_CONFIG_API_VERSION_V1 = "llm-d.ai/v1"

# The InferenceX AgentX harness: aiperf at the submodule pin of
# SemiAnalysisAI/InferenceX (utils/aiperf), run as
# `aiperf profile --scenario inferencex-agentx-mvp ...` exactly as
# benchmarks/benchmark_lib.sh build_replay_cmd composes it; traces are the
# public HF dataset the scenario's `--public-dataset` loader names. The
# bench image is a uv-equipped Python so the Job installs aiperf at the pin
# into its own venv (InferenceX does the same inside the server image).
AIPERF_REPO = "https://github.com/SemiAnalysisAI/aiperf.git"
AIPERF_REF = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"
AIPERF_BENCH_IMAGE = "ghcr.io/astral-sh/uv:python3.11-bookworm"
AGENTX_SCENARIO = "inferencex-agentx-mvp"
AGENTX_DATASET_LOADER = "semianalysis_cc_traces_weka_062126_256k"
AGENTX_DATASETS = {
    "semianalysis_cc_traces_weka_062126_256k":
        "semianalysisai/cc-traces-weka-062126-256k",
    "semianalysis_cc_traces_weka_062126":
        "semianalysisai/cc-traces-weka-062126",
    "semianalysis_cc_traces_weka_061526_256k":
        "semianalysisai/cc-traces-weka-061526-256k",
    "semianalysis_cc_traces_weka_061526":
        "semianalysisai/cc-traces-weka-061526",
}
AGENTX_DURATION_S = 3600
AGENTX_FAST_DURATION_S = 1200
AGENTX_MIN_VALID_DURATION_S = 900   # shorter runs need --unsafe-override
AGENTX_WARMUP_PER_LANE = 10
AGENTX_FAST_WARMUP_PER_LANE = 1
AGENTX_WARMUP_GRACE_S = 1800
AGENTX_IDLE_GAP_CAP_S = 300
AGENTX_LIVE_FAILED_THRESHOLD = 0.25   # aiperf aborts the run past this
AGENTX_FAILED_THRESHOLD = 0.10        # InferenceX's post-run validity gate
AGENTX_TTFT_P50_GATE_MS = 5000        # default TTFT p50 qualifying gate
AGENTX_NUM_DATASET_ENTRIES = 393

# The InferenceX dashboard's price per accelerator-hour, from the app's
# HW_REGISTRY (SemiAnalysisAI/InferenceX-app packages/constants/src/
# gpu-keys.ts, citing "SemiAnalysis Market July 2026 Pricing Surveys & AI
# Cloud TCO Model", https://semianalysis.com/ai-cloud-tco-model/) as read
# on 2026-10-16. Three bases: `internal` is the dashboard default (the
# owning tier, with tpuv7 repriced to Google's internal owner cost, 1.03,
# in packages/app/src/lib/catalog/constants.ts getGpuSpecs); `external`
# is the owning tier as an external buyer pays (costh); `rental` is the
# retail tier (costr; for tpuv7 "GCP 3-year commit rental rate per
# chip-hour"). tokens per $ = tok/s per chip x 3600 / price. TPU prices
# are per physical chip-hour; no tpuv6e key exists on the dashboard.
AGENTX_PRICE_SOURCE = ("InferenceX-app HW_REGISTRY (gpu-keys.ts) @ bbcf1128,"
                       " SemiAnalysis AI Cloud TCO Model, July 2026")
AGENTX_PRICES: dict[str, dict[str, float]] = {
    "tpuv7": {"internal": 1.03, "external": 1.21, "rental": 2.00},
    "tpu7x": {"internal": 1.03, "external": 1.21, "rental": 2.00},
    "gb300": {"internal": 2.31, "external": 2.31, "rental": 5.00},
    "gb200": {"internal": 1.86, "external": 1.86, "rental": 4.00},
    "b300": {"internal": 2.26, "external": 2.26, "rental": 4.25},
    "b200": {"internal": 1.73, "external": 1.73, "rental": 3.70},
    "h200": {"internal": 1.22, "external": 1.22, "rental": 2.90},
    "h100": {"internal": 1.17, "external": 1.17, "rental": 2.00},
    "mi355x": {"internal": 1.50, "external": 1.50, "rental": 2.90},
    "mi325x": {"internal": 1.10, "external": 1.10, "rental": 1.60},
    "mi300x": {"internal": 0.95, "external": 0.95, "rental": 1.30},
    "vr200": {"internal": 3.61, "external": 3.61, "rental": 8.50},
    "rtx6000pro": {"internal": 0.68, "external": 0.68, "rental": 0.52},
}
AGENTX_TCO_BASES = ("internal", "external", "rental")
# The dashboard's public, unauthenticated views API (docs at
# https://inferencex.semianalysis.com/api): chart-ready points per
# hardware/framework series for one model and the agentic sequence.
AGENTX_LEADERBOARD_API = "https://inferencex.semianalysis.com/api/v1/views/inference"
AGENTX_LEADERBOARD_MODEL = "Qwen-3.5-397B-A17B"
AGENTX_LEADERBOARD_SEQUENCE = "agentic-traces"

# vLLM KV-event publishing for precise prefix routing
# (guides/precise-prefix-cache-routing/modelserver/*/patch-vllm.yaml and
# router/precise-prefix-cache-routing.values.yaml at LLMD_REF). The guide
# uses 5556/5559; the EPP's code default is 5557, so both sides are set
# explicitly from these pins.
KV_EVENTS_PORT = 5556
KV_EVENTS_REPLAY_PORT = 5559
KV_EVENTS_BLOCK_SIZE_DEFAULT = 64

# Host-tier KV offload on TPU. tpu-inference's connector and env knobs
# (guides/tiered-prefix-cache/modelserver/tpu/base/vllm/single-host/
# patch-vllm.yaml at LLMD_REF); the torchtpu image carries its own
# connector module instead. The torchtpu module and class names were read
# from the image by `deploy probe` and `inspect.getmembers` on the pod
# (vllm_torchtpu 0.0.0, vllm 0.29.0); whether that connector honours the
# TPU_OFFLOAD_* env knobs is UNVERIFIED and is what a tiered-host arm on a
# managed torchtpu fleet would establish.
TPU_OFFLOAD_MODULE_TPU_INFERENCE = "tpu_inference.offload.tpu_offload_connector"
TPU_OFFLOAD_CONNECTOR_TPU_INFERENCE = "TPUOffloadConnector"
TPU_OFFLOAD_MODULE_TORCHTPU = "vllm_torchtpu.offload.raiden_connector"
TPU_OFFLOAD_CONNECTOR_TORCHTPU = "TPURaidenOffloadingConnector"
TPU_OFFLOAD_CPU_CHUNKS_ENV = "TPU_OFFLOAD_NUM_CPU_CHUNKS"
TPU_OFFLOAD_STAGING_BLOCKS_ENV = "TPU_OFFLOAD_NUM_STAGING_BLOCKS"

# P/D connectors (guides/pd-disaggregation/modelserver/tpu/v7/vllm at
# LLMD_REF for tpu-inference; the torchtpu image's module and class from
# the same introspection as above, which also lists TPURaidenConnector).
TPU_PD_MODULE_TPU_INFERENCE = "tpu_inference.distributed.tpu_connector_hma"
TPU_PD_CONNECTOR_TPU_INFERENCE = "TPUConnectorHMA"
TPU_PD_MODULE_TORCHTPU = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
TPU_PD_CONNECTOR_TORCHTPU = "TPUConnector"
