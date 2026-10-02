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

# inference-perf: the benchmark harness and the workload catalog vendored in
# `workload_catalog/`. The catalog ref is also in CATALOG.json; the two are
# compared by a test so they cannot drift.
INFERENCE_PERF_REF = "45af1d3c5f57de7cf43866539fd9d4d7a3591ad9"
INFERENCE_PERF_IMAGE = "quay.io/inference-perf/inference-perf:latest"

# Labels this runtime puts on everything it creates, so status, teardown and
# the reaper find a release by one selector and never by name pattern.
RELEASE_LABEL = "tpu-coworker.llmd/release"
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

# Ports. The standalone chart's Envoy sidecar listens on 8081 and the chart's
# `extraServicePorts` maps Service port 80 to it; the EPP serves metrics on
# 9090; vLLM serves on 8000.
MODEL_SERVER_PORT = 8000
EPP_METRICS_PORT = 9090
ROUTER_SERVICE_PORT = 80
ROUTER_SIDECAR_PORT = 8081

# TPU P/D side channels used by llm-d's tpu-inference connectors.
TPU_SIDE_CHANNEL_PORT = 9600
TPU_KV_TRANSFER_PORT = 9100
