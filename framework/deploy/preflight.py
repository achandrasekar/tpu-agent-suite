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
"""Is this cluster ready for an llm-d release, and if not, what is missing.

Every check names its fix. `--fix` performs the two that are safe to do
without a person: creating the namespace and installing the Gateway API
Inference Extension CRDs the router chart needs (an InferencePool is a CRD,
and helm will not create the CRD for it). Secrets, TPU node pools and
Managed Prometheus are reported, never created: each is an account- or
capacity-level decision.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import pins
from .config import DeployEnv
from .kube import DeployError, Kube, tools_present

GMP_CRD = "podmonitorings.monitoring.googleapis.com"
GATEWAY_CRD = "gateways.gateway.networking.k8s.io"


@dataclasses.dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _crd_present(kube: Kube, crd: str) -> bool:
    return kube.exists("crd", crd, namespaced=False)


def _tpu_capacity(kube: Kube, accelerator: str) -> tuple[int, int, int]:
    """Nodes with the accelerator label, chips they hold, chips unallocated.

    Allocation is read from `kubectl describe`-free sources only: capacity
    from the node, requests from the pods on it.
    """
    selector = f"{pins.ACCELERATOR_LABEL}={accelerator}" if accelerator else (
        pins.ACCELERATOR_LABEL)
    nodes = kube.kubectl_json(["get", "nodes", "-l", selector],
                              namespaced=False,
                              check=False)
    items = nodes.get("items", []) if nodes else []
    capacity = 0
    for node in items:
        capacity += int(
            ((node.get("status") or {}).get("allocatable") or {}).get(
                pins.TPU_RESOURCE) or 0)
    pods = kube.kubectl_json(
        ["get", "pods", "--all-namespaces", "--field-selector",
         "status.phase!=Succeeded,status.phase!=Failed"],
        namespaced=False,
        check=False)
    requested = 0
    node_names = {n["metadata"]["name"] for n in items}
    for pod in (pods.get("items", []) if pods else []):
        if (pod.get("spec") or {}).get("nodeName") not in node_names:
            continue
        for c in (pod.get("spec") or {}).get("containers", []):
            requested += int(((c.get("resources") or {}).get("requests")
                              or {}).get(pins.TPU_RESOURCE) or 0)
    return len(items), capacity, max(capacity - requested, 0)


def run_preflight(env: DeployEnv,
                  *,
                  fix: bool = False,
                  runner: Any = None) -> dict[str, Any]:
    checks: list[Check] = []
    tools = tools_present()
    for tool in ("kubectl", "helm"):
        checks.append(
            Check(f"tool.{tool}", "ok" if tools[tool] else "fail",
                  tools[tool] or f"{tool} not on PATH",
                  f"install {tool}"))
    checks.append(
        Check("tool.gcloud", "ok" if tools["gcloud"] else "warn",
              tools["gcloud"] or "gcloud not on PATH; `collect` needs it",
              "install the Google Cloud CLI"))
    if not tools["kubectl"]:
        return _result(env, checks, fixed=[])

    kube = env.kube(runner=runner)
    fixed: list[str] = []

    proc = kube.kubectl(["get", "namespaces", "-o", "name"],
                        namespaced=False,
                        check=False)
    if proc.returncode != 0:
        checks.append(
            Check("cluster.reachable", "fail",
                  (proc.stderr or "").strip()[:300],
                  "check kubeconfig/context and cluster credentials"))
        return _result(env, checks, fixed=fixed)
    checks.append(Check("cluster.reachable", "ok", "API server answered"))

    for label, ns in (("namespace", env.namespace),
                      ("production_namespace", env.production_namespace)):
        if not ns:
            continue
        if kube.exists("namespace", ns, namespaced=False):
            checks.append(Check(f"{label}.{ns}", "ok", "exists"))
        elif fix:
            kube.kubectl(["create", "namespace", ns], namespaced=False)
            fixed.append(f"created namespace {ns}")
            checks.append(Check(f"{label}.{ns}", "ok", "created"))
        else:
            checks.append(
                Check(f"{label}.{ns}", "fail", "missing",
                      "re-run with --fix to create it"))

    if _crd_present(kube, pins.INFERENCE_POOL_CRD):
        checks.append(Check("crd.inferencepool", "ok",
                            pins.INFERENCE_POOL_CRD))
    elif fix:
        kube.kubectl(["apply", "-f", pins.GAIE_CRD_URL], namespaced=False)
        fixed.append(f"applied GAIE CRDs {pins.GAIE_VERSION}")
        checks.append(
            Check("crd.inferencepool", "ok",
                  f"installed from {pins.GAIE_CRD_URL}"))
    else:
        checks.append(
            Check("crd.inferencepool", "fail",
                  f"{pins.INFERENCE_POOL_CRD} missing",
                  f"re-run with --fix to apply {pins.GAIE_CRD_URL}"))

    if env.gateway_mode == "gke":
        checks.append(
            Check("crd.gateway", "ok" if _crd_present(kube, GATEWAY_CRD) else
                  "fail", GATEWAY_CRD,
                  "enable the Gateway API on the cluster"))

    secret_ok = kube.exists("secret", env.hf_token_secret)
    checks.append(
        Check("secret.hf_token", "ok" if secret_ok else "fail",
              f"{env.hf_token_secret} in {env.namespace}",
              f"create secret {env.hf_token_secret} with key HF_TOKEN in"
              f" namespace {env.namespace}"))

    nodes, capacity, free = _tpu_capacity(kube, env.tpu_accelerator)
    checks.append(
        Check("tpu.nodes", "ok" if nodes else "fail",
              f"{nodes} node(s) with {pins.ACCELERATOR_LABEL}="
              f"{env.tpu_accelerator or '*'}; {capacity} chips allocatable,"
              f" {free} unrequested",
              "create a TPU node pool matching compute.params.tpu_accelerator"))

    checks.append(
        Check("crd.podmonitoring",
              "ok" if _crd_present(kube, GMP_CRD) else "warn", GMP_CRD,
              "enable Managed Service for Prometheus; `collect` needs it"))

    if tools["helm"]:
        proc = kube.helm([
            "show", "chart", pins.ROUTER_STANDALONE_CHART, "--version",
            pins.ROUTER_CHART_VERSION
        ], check=False)
        checks.append(
            Check("helm.router_chart", "ok" if proc.returncode == 0 else
                  "warn",
                  f"{pins.ROUTER_STANDALONE_CHART}:{pins.ROUTER_CHART_VERSION}"
                  + ("" if proc.returncode == 0 else
                     f" -- {(proc.stderr or '').strip()[:200]}"),
                  "check registry access to ghcr.io"))
    return _result(env, checks, fixed=fixed, tpu={
        "nodes": nodes,
        "chips_allocatable": capacity,
        "chips_free": free
    })


def _result(env: DeployEnv, checks: list[Check], *, fixed: list[str],
            **extra: Any) -> dict[str, Any]:
    failed = [c.name for c in checks if c.status == "fail"]
    return {
        "ok": not failed,
        "failed": failed,
        "fixed": fixed,
        "checks": [c.to_dict() for c in checks],
        "env": {
            "namespace": env.namespace,
            "production_namespace": env.production_namespace,
            "gateway_mode": env.gateway_mode,
            "tpu_accelerator": env.tpu_accelerator,
            "tpu_topology": env.tpu_topology,
        },
        **extra,
    }


def require_ok(result: dict[str, Any]) -> None:
    if not result["ok"]:
        raise DeployError(
            f"preflight failed: {', '.join(result['failed'])}",
            error_code="PREFLIGHT_FAILED")
