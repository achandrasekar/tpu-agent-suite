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
"""Render one llm-d release from an incumbent config, a recipe and a workload.

Pure: given the same inputs it writes the same files, which is what lets a
golden test pin the manifests and what lets `reconfigure` re-render a
release in place and apply only the part that changed. Nothing here talks
to a cluster.

The output mirrors llm-d's optimized-baseline guide on TPU at `pins.LLMD_REF`
-- one base Deployment with the TPU overlay's nodeSelector, resources, probes
and cache mounts, and the standalone router chart with its guide values --
but as self-contained documents, so a release carries everything it deployed
and no study depends on a clone of the guides.

Sources, all at the pinned tag:
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/recipes/modelserver/base/single-host/default/decode-deployment.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/optimized-baseline/modelserver/tpu/v7/vllm/patch-vllm.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/recipes/router/base.values.yaml
  https://github.com/llm-d/llm-d/blob/v0.10.0/guides/optimized-baseline/router/optimized-baseline.values.yaml
  https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import pathlib
import re
from typing import Any

from . import pins, workload
from .kube import DeployError

MANIFEST_DIR = "manifests"
MODELSERVER_FILE = f"{MANIFEST_DIR}/10-modelserver.json"
MONITORING_FILE = f"{MANIFEST_DIR}/40-monitoring.json"
ROUTER_VALUES_FILE = "router.values.json"
EPP_CONFIG_FILE = "epp-config.json"
EPP_CONFIG_KEY = "tpu-coworker-plugins.yaml"
RELEASE_FILE = "release.json"
CONFIG_FILE = "llmd-config.json"
RECIPE_FILE = "serving-recipe.json"
PROFILE_FILE = "workload_profile.json"
BENCH_DIR = "bench"
BENCH_STAGES = ("sweep", "slo")

DEFAULT_MEMORY = "64Gi"
DEFAULT_CPU = "16"
DEFAULT_SHM = "64Gi"
MODELSERVER_CONTAINER = "modelserver"
BENCH_CONTAINER = "inference-perf"
BENCH_REPORT_DIR = "/reports"
BENCH_CONFIG_DIR = "/config"
BENCH_REPORT_MARK = "===REPORT"
BENCH_REPORT_END = "===END==="

# What the incumbent may ask for in this milestone. Anything listed here
# that a config turns on is refused at render time rather than dropped, so
# a planner that proposes an unsupported arm finds out before a pod exists.
UNSUPPORTED_IN_THIS_MILESTONE = (
    ("topology.mode", lambda c: c["topology"].get("mode", "aggregated") !=
     "aggregated", "prefill/decode disaggregation"),
    ("topology.prefill", lambda c: bool(c["topology"].get("prefill")),
     "prefill pool"),
    ("kv.precise_routing", lambda c: bool(c["kv"].get("precise_routing")),
     "precise prefix-cache routing"),
    ("kv.tiered_host", lambda c: bool(c["kv"].get("tiered_host")),
     "tiered host KV offload"),
    ("kv.pd_connector", lambda c: bool(c["kv"].get("pd_connector")),
     "P/D KV connector"),
    ("router.flow_control", lambda c: bool(c["router"].get("flow_control")),
     "flow control"),
    ("router.predicted_latency",
     lambda c: bool(c["router"].get("predicted_latency")),
     "predicted-latency routing"),
)

# The router profiles the planner may name. Each is an EndpointPickerConfig
# body; `router.plugins` in the config overrides any plugin's parameters or
# a profile entry's weight, so the inner sweep can tune a profile without
# the renderer knowing every knob.
#   sticky-until-saturated: llm-d optimized-baseline (affinity filter gated
#     by peakPrefillThroughput, then least in-flight tokens).
#   prefix-weighted: the scorer set llm-d documents as recommended.
#   load-only: no prefix affinity; the control a reuse claim is measured
#     against.
#   session-sticky: multi-turn conversations pinned to one pod.
ROUTER_PROFILES: dict[str, dict[str, Any]] = {
    "sticky-until-saturated": {
        "plugins": [
            {"type": "approx-prefix-cache-producer"},
            {"type": "inflight-load-producer"},
            {"type": "prefix-cache-affinity-filter"},
            {"type": "token-load-scorer"},
        ],
        "profile": [
            {"pluginRef": "prefix-cache-affinity-filter"},
            {"pluginRef": "token-load-scorer"},
        ],
    },
    "prefix-weighted": {
        "plugins": [
            {"type": "approx-prefix-cache-producer"},
            {"type": "prefix-cache-scorer"},
            {"type": "kv-cache-utilization-scorer"},
            {"type": "queue-scorer"},
            {"type": "max-score-picker"},
        ],
        "profile": [
            {"pluginRef": "prefix-cache-scorer", "weight": 3.0},
            {"pluginRef": "kv-cache-utilization-scorer", "weight": 2.0},
            {"pluginRef": "queue-scorer", "weight": 2.0},
            {"pluginRef": "max-score-picker"},
        ],
    },
    "load-only": {
        "plugins": [
            {"type": "kv-cache-utilization-scorer"},
            {"type": "queue-scorer"},
            {"type": "max-score-picker"},
        ],
        "profile": [
            {"pluginRef": "kv-cache-utilization-scorer", "weight": 1.0},
            {"pluginRef": "queue-scorer", "weight": 1.0},
            {"pluginRef": "max-score-picker"},
        ],
    },
    "session-sticky": {
        "plugins": [
            {"type": "session-affinity-scorer"},
            {"type": "queue-scorer"},
            {"type": "max-score-picker"},
        ],
        "profile": [
            {"pluginRef": "session-affinity-scorer", "weight": 2.0},
            {"pluginRef": "queue-scorer", "weight": 1.0},
            {"pluginRef": "max-score-picker"},
        ],
    },
}

# Engine knobs the fleet may tune, and the vllm flag each sets. The
# whitelist is the boundary with the engine-optimization domains: a knob
# not listed here is a recommendation for tpu-performance, not an arm.
ENGINE_FLAGS: dict[str, str] = {
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "kv_cache_dtype": "--kv-cache-dtype",
    "max_model_len": "--max-model-len",
}
ENGINE_SWITCHES: dict[str, str] = {"async_scheduling": "--async-scheduling"}

NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$")


# -- defaults and validation -------------------------------------------------


def default_config(recipe: dict[str, Any], replicas: int = 1) -> dict[str, Any]:
    """The incumbent a study starts from: the recipe, N times, as llm-d's
    optimized-baseline deploys it."""
    accel = recipe.get("accelerator") or {}
    return {
        "topology": {
            "mode": "aggregated",
            "decode": {
                "replicas": int(replicas),
                "tp": int(recipe.get("tensor_parallel") or 1),
                "slice": accel.get("tpu_topology") or "",
            },
            "prefill": None,
        },
        "kv": {
            "precise_routing": False,
            "block_size": None,
            "tiered_host": None,
            "pd_connector": None,
        },
        "router": {
            "profile": "sticky-until-saturated",
            "plugins": {},
            "flow_control": None,
            "predicted_latency": False,
            "epp_replicas": 1,
        },
        "engine": {},
        "provenance": {"arm": "baseline", "execution_id": "", "metric": None},
    }


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill the sections a config may omit so every reader sees one shape."""
    out = copy.deepcopy(config)
    out.setdefault("topology", {})
    out["topology"].setdefault("mode", "aggregated")
    out["topology"].setdefault("decode", {})
    out["topology"].setdefault("prefill", None)
    out.setdefault("kv", {})
    for key in ("precise_routing", "block_size", "tiered_host",
                "pd_connector"):
        out["kv"].setdefault(key, None if key != "precise_routing" else False)
    out.setdefault("router", {})
    out["router"].setdefault("profile", "sticky-until-saturated")
    out["router"].setdefault("plugins", {})
    out["router"].setdefault("flow_control", None)
    out["router"].setdefault("predicted_latency", False)
    out["router"].setdefault("epp_replicas", 1)
    out.setdefault("engine", {})
    out.setdefault("provenance", {})
    return out


def validate_config(config: dict[str, Any]) -> None:
    for path, is_on, what in UNSUPPORTED_IN_THIS_MILESTONE:
        if is_on(config):
            raise DeployError(
                f"{what} ({path}) is not supported by this runtime yet",
                error_code="NOT_IMPLEMENTED")
    decode = config["topology"]["decode"]
    replicas = decode.get("replicas")
    if not isinstance(replicas, int) or replicas < 1:
        raise DeployError("topology.decode.replicas must be an integer >= 1",
                          error_code="INVALID_FIELD")
    profile = config["router"]["profile"]
    if profile not in ROUTER_PROFILES:
        raise DeployError(
            f"router.profile {profile!r} is not one of"
            f" {sorted(ROUTER_PROFILES)}",
            error_code="INVALID_FIELD")
    unknown = set(config["engine"]) - set(ENGINE_FLAGS) - set(ENGINE_SWITCHES)
    if unknown:
        raise DeployError(
            f"engine knobs {sorted(unknown)} are outside the fleet whitelist"
            f" {sorted(ENGINE_FLAGS) + sorted(ENGINE_SWITCHES)}; hand them to"
            " the engine-optimization domain instead",
            error_code="INVALID_FIELD")


def validate_recipe(recipe: dict[str, Any]) -> None:
    for key in ("model", "image"):
        if not recipe.get(key):
            raise DeployError(f"serving recipe needs {key!r}",
                              error_code="INVALID_FIELD")
    accel = recipe.get("accelerator") or {}
    if not accel.get("tpu_accelerator") or not accel.get("tpu_topology"):
        raise DeployError(
            "serving recipe needs accelerator.tpu_accelerator and"
            " accelerator.tpu_topology",
            error_code="INVALID_FIELD")


def config_digest(config: dict[str, Any]) -> str:
    """Digest of what the config deploys, ignoring where it came from."""
    body = {k: v for k, v in config.items() if k != "provenance"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chips_for_topology(topology: str, default: int = 4) -> int:
    """Chips on one single-host slice, e.g. `2x2x1` -> 4, `2x4` -> 8."""
    try:
        dims = [int(d) for d in str(topology).lower().split("x")]
    except ValueError:
        return default
    product = 1
    for d in dims:
        product *= d
    return product or default


def chips_per_replica(recipe: dict[str, Any], config: dict[str, Any]) -> int:
    accel = recipe.get("accelerator") or {}
    explicit = accel.get("chips_per_replica")
    if explicit:
        return int(explicit)
    slice_shape = config["topology"]["decode"].get("slice") or accel.get(
        "tpu_topology", "")
    return chips_for_topology(slice_shape)


def label_value(text: str) -> str:
    """A Kubernetes label value: 63 chars of [A-Za-z0-9._-], edges alnum."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", str(text))[:63]
    return cleaned.strip("-._") or "model"


# -- vllm args -------------------------------------------------------------


def _flag_name(arg: str) -> str | None:
    if not arg.startswith("--"):
        return None
    return arg.split("=", 1)[0]


def merge_args(recipe_args: list[str], overrides: dict[str, Any],
               switches: dict[str, bool]) -> list[str]:
    """Recipe args with fleet knobs applied; later wins, one form per flag.

    Both `--flag value` and `--flag=value` forms are recognised in the
    recipe so a knob the recipe already sets is replaced, not duplicated.
    """
    out: list[str] = []
    skip_next = False
    override_flags = set(overrides) | set(switches)
    for i, arg in enumerate(recipe_args):
        if skip_next:
            skip_next = False
            continue
        flag = _flag_name(arg)
        if flag in override_flags:
            if "=" not in arg and i + 1 < len(recipe_args) and not str(
                    recipe_args[i + 1]).startswith("--"):
                skip_next = True
            continue
        out.append(str(arg))
    for flag, value in overrides.items():
        out.append(f"{flag}={value}")
    for flag, enabled in switches.items():
        if enabled:
            out.append(flag)
    return out


def vllm_args(recipe: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """`vllm serve` arguments, as the TPU overlay orders them."""
    tp = config["topology"]["decode"].get("tp") or recipe.get(
        "tensor_parallel") or 1
    base = [
        str(recipe["model"]),
        f"--port={pins.MODEL_SERVER_PORT}",
        "--disable-access-log-for-endpoints=/health,/metrics,/v1/models",
        f"--tensor-parallel-size={int(tp)}",
    ]
    overrides: dict[str, Any] = {}
    for key, flag in ENGINE_FLAGS.items():
        if config["engine"].get(key) is not None:
            overrides[flag] = config["engine"][key]
    if config["kv"].get("block_size"):
        overrides["--block-size"] = int(config["kv"]["block_size"])
    switches = {
        flag: bool(config["engine"].get(key))
        for key, flag in ENGINE_SWITCHES.items()
        if key in config["engine"]
    }
    recipe_args = [
        a for a in list(recipe.get("args") or [])
        if _flag_name(a) not in ("--port", "--tensor-parallel-size")
    ]
    return base + merge_args(recipe_args, overrides, switches)


# -- model server manifests ----------------------------------------------------


def release_labels(name: str, production: bool) -> dict[str, str]:
    labels = {
        pins.RELEASE_LABEL: name,
        pins.MANAGED_BY_LABEL: pins.MANAGED_BY_VALUE,
    }
    if production:
        labels[pins.PRODUCTION_LABEL] = "true"
    return labels


def modelserver_manifests(name: str, recipe: dict[str, Any],
                          config: dict[str, Any], env: dict[str, Any],
                          production: bool) -> dict[str, Any]:
    """ServiceAccount, decode Deployment and Service for one release."""
    accel = recipe["accelerator"]
    decode = config["topology"]["decode"]
    chips = chips_per_replica(recipe, config)
    slice_shape = decode.get("slice") or accel["tpu_topology"]
    resources = dict(recipe.get("resources") or {})
    limits = {
        "memory": resources.get("memory", DEFAULT_MEMORY),
        "cpu": str(resources.get("cpu", DEFAULT_CPU)),
        pins.TPU_RESOURCE: chips,
    }
    pod_labels = {
        **release_labels(name, production),
        pins.LLMD_ROLE_LABEL: "decode",
        pins.LLMD_SERVING_LABEL: "true",
        pins.LLMD_ENGINE_LABEL: "vllm",
        pins.LLMD_GUIDE_LABEL: "tpu-coworker",
        "llm-d.ai/model": label_value(recipe["model"]),
    }
    selector = {pins.RELEASE_LABEL: name, pins.LLMD_ROLE_LABEL: "decode"}

    env_list = [{
        "name": "HF_TOKEN",
        "valueFrom": {
            "secretKeyRef": {
                "name": env["hf_token_secret"],
                "key": "HF_TOKEN"
            }
        },
    }]
    for key in sorted(recipe.get("env") or {}):
        env_list.append({"name": key, "value": str(recipe["env"][key])})

    container = {
        "name": MODELSERVER_CONTAINER,
        "image": recipe["image"],
        "command": ["vllm", "serve"],
        "args": vllm_args(recipe, config),
        "env": env_list,
        "ports": [{
            "name": MODELSERVER_CONTAINER,
            "containerPort": pins.MODEL_SERVER_PORT,
            "protocol": "TCP",
        }],
        "resources": {"limits": limits, "requests": dict(limits)},
        "volumeMounts": [
            {"mountPath": "/.cache", "name": "torch-compile-cache"},
            {"mountPath": "/.triton", "name": "triton-cache"},
            {"mountPath": "/.config", "name": "vllm-config"},
            {"mountPath": "/dev/shm", "name": "dshm"},
        ],
        "startupProbe": {
            "httpGet": {"path": "/v1/models", "port": MODELSERVER_CONTAINER},
            "initialDelaySeconds": 15,
            "periodSeconds": 30,
            "timeoutSeconds": 5,
            "failureThreshold": 120,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": MODELSERVER_CONTAINER},
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {"path": "/v1/models", "port": MODELSERVER_CONTAINER},
            "periodSeconds": 5,
            "timeoutSeconds": 2,
            "failureThreshold": 3,
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{name}-decode",
            "labels": pod_labels,
            "annotations": {
                "tpu-coworker.llmd/config-digest": config_digest(config)
            },
        },
        "spec": {
            "replicas": int(decode["replicas"]),
            "selector": {"matchLabels": selector},
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": {
                    "serviceAccountName": f"{name}-modelserver",
                    "nodeSelector": {
                        pins.TOPOLOGY_LABEL: slice_shape,
                        pins.ACCELERATOR_LABEL: accel["tpu_accelerator"],
                    },
                    "containers": [container],
                    "volumes": [
                        {"name": "torch-compile-cache", "emptyDir": {}},
                        {"name": "triton-cache", "emptyDir": {}},
                        {"name": "vllm-config", "emptyDir": {}},
                        {
                            "name": "dshm",
                            "emptyDir": {
                                "medium": "Memory",
                                "sizeLimit": resources.get("shm", DEFAULT_SHM)
                            }
                        },
                    ],
                },
            },
        },
    }
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": f"{name}-modelserver",
            "labels": release_labels(name, production)
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-modelserver",
            "labels": release_labels(name, production)
        },
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": [{
                "name": "http",
                "port": pins.MODEL_SERVER_PORT,
                "targetPort": MODELSERVER_CONTAINER,
                "protocol": "TCP",
            }],
        },
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [service_account, deployment, service]
    }


# -- router ------------------------------------------------------------------


def epp_config(config: dict[str, Any]) -> dict[str, Any]:
    """The EndpointPickerConfig for the config's router profile and knobs."""
    profile = ROUTER_PROFILES[config["router"]["profile"]]
    overrides = config["router"].get("plugins") or {}
    plugins = []
    for entry in profile["plugins"]:
        item: dict[str, Any] = {"type": entry["type"]}
        params = dict(entry.get("parameters") or {})
        extra = dict(overrides.get(entry["type"]) or {})
        extra.pop("weight", None)
        params.update(extra)
        if params:
            item["parameters"] = params
        plugins.append(item)
    refs = []
    for entry in profile["profile"]:
        ref = dict(entry)
        weight = (overrides.get(entry["pluginRef"]) or {}).get("weight")
        if weight is not None:
            ref["weight"] = float(weight)
        refs.append(ref)
    unknown = set(overrides) - {p["type"] for p in profile["plugins"]}
    if unknown:
        raise DeployError(
            f"router.plugins names {sorted(unknown)}, which profile"
            f" {config['router']['profile']!r} does not instantiate",
            error_code="INVALID_FIELD")
    return {
        "apiVersion": pins.EPP_CONFIG_API_VERSION,
        "kind": pins.EPP_CONFIG_KIND,
        "plugins": plugins,
        "schedulingProfiles": [{"name": "default", "plugins": refs}],
    }


def router_values(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Values for the standalone router chart: base + guide overrides."""
    epp_doc = json.dumps(epp_config(config), indent=2) + "\n"
    return {
        "router": {
            "extraServicePorts": [{
                "name": "http",
                "port": pins.ROUTER_SERVICE_PORT,
                "protocol": "TCP",
                "targetPort": pins.ROUTER_SIDECAR_PORT,
            }],
            "epp": {
                "replicas": int(config["router"].get("epp_replicas") or 1),
                "flags": {"v": 2},
                "pluginsConfigFile": EPP_CONFIG_KEY,
                "pluginsCustomConfig": {EPP_CONFIG_KEY: epp_doc},
                "resources": {
                    "requests": {"cpu": "4", "memory": "8Gi"},
                    "limits": {"memory": "16Gi"},
                },
            },
            "inferencePool": {"failureMode": "FailOpen"},
            "modelServers": {
                "protocol": "http",
                "matchLabels": {pins.RELEASE_LABEL: name},
            },
            "proxy": {
                "proxyType": "envoy",
                "args": [
                    "--service-node", "envoy-sidecar", "--log-level", "warn",
                    "--concurrency", "8", "--drain-strategy", "immediate",
                    "--drain-time-s", "60", "-c", "/etc/envoy/envoy.yaml"
                ],
                "resources": {
                    "requests": {"cpu": "4", "memory": "8Gi"},
                    "limits": {"memory": "16Gi"},
                },
            },
        }
    }


def router_service(name: str, namespace: str) -> str:
    """The in-cluster host benchmarks send to: the standalone chart's EPP
    Service, with `extraServicePorts` exposing the Envoy sidecar on 80."""
    return f"{name}-epp.{namespace}.svc.cluster.local"


# -- monitoring ----------------------------------------------------------------


def monitoring_manifests(name: str, production: bool) -> dict[str, Any]:
    """Managed Prometheus PodMonitoring for vLLM and the EPP."""
    labels = release_labels(name, production)
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "monitoring.googleapis.com/v1",
                "kind": "PodMonitoring",
                "metadata": {"name": f"{name}-modelserver", "labels": labels},
                "spec": {
                    "selector": {
                        "matchLabels": {
                            pins.RELEASE_LABEL: name,
                            pins.LLMD_ROLE_LABEL: "decode"
                        }
                    },
                    "endpoints": [{
                        "port": MODELSERVER_CONTAINER,
                        "path": "/metrics",
                        "interval": "15s"
                    }],
                },
            },
            {
                "apiVersion": "monitoring.googleapis.com/v1",
                "kind": "PodMonitoring",
                "metadata": {"name": f"{name}-epp", "labels": labels},
                "spec": {
                    "selector": {
                        "matchLabels": {"app.kubernetes.io/instance": name}
                    },
                    "endpoints": [{
                        "port": pins.EPP_METRICS_PORT,
                        "path": "/metrics",
                        "interval": "15s"
                    }],
                },
            },
        ],
    }


# -- benchmark Jobs ------------------------------------------------------------


def bench_job(name: str, tenant: str, stage: str, env: dict[str, Any],
              production: bool) -> dict[str, Any]:
    """A Job that runs inference-perf and prints every report it wrote.

    Reports are echoed between markers rather than copied out, so `bench`
    needs only the Job's logs -- no exec, no cp, no shared volume -- and the
    Job stays a plain Kubernetes object the reaper can delete.
    """
    job_name = f"{name}-bench-{tenant}-{stage}"
    script = (f"set -e; inference-perf --config_file"
              f" {BENCH_CONFIG_DIR}/config.json;"
              f" for f in $(find {BENCH_REPORT_DIR} -name '*.json' | sort);"
              f" do echo '{BENCH_REPORT_MARK}' \"$f\"; cat \"$f\"; echo;"
              f" echo '{BENCH_REPORT_END}'; done")
    labels = {
        **release_labels(name, production),
        "tpu-coworker.llmd/bench-tenant": tenant,
        "tpu-coworker.llmd/bench-stage": stage,
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": BENCH_CONTAINER,
                        "image": env["inference_perf_image"],
                        "command": ["sh", "-c", script],
                        "env": [{
                            "name": "HF_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": env["hf_token_secret"],
                                    "key": "HF_TOKEN"
                                }
                            },
                        }],
                        "volumeMounts": [
                            {"name": "config", "mountPath": BENCH_CONFIG_DIR},
                            {"name": "reports", "mountPath": BENCH_REPORT_DIR},
                        ],
                        "resources": {
                            "requests": {"cpu": "4", "memory": "8Gi"},
                            "limits": {"memory": "16Gi"},
                        },
                    }],
                    "volumes": [
                        {"name": "config", "configMap": {"name": job_name}},
                        {"name": "reports", "emptyDir": {}},
                    ],
                },
            },
        },
    }


def bench_configmap(name: str, tenant: str, stage: str,
                    config: dict[str, Any], production: bool) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{name}-bench-{tenant}-{stage}",
            "labels": release_labels(name, production)
        },
        "data": {"config.json": json.dumps(config, indent=2) + "\n"},
    }


# -- the release ---------------------------------------------------------------


def _write(path: pathlib.Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")


def render_release(*,
                   name: str,
                   config: dict[str, Any],
                   recipe: dict[str, Any],
                   profile: dict[str, Any],
                   env: dict[str, Any],
                   out_dir: pathlib.Path,
                   production: bool = False,
                   now: str | None = None) -> dict[str, Any]:
    """Write every file a release needs under `out_dir`; return release.json.

    `env` is the subset of `DeployEnv` the manifests depend on (namespace,
    secret, image, accelerator). It is recorded in release.json so a later
    `reconfigure` re-renders with exactly what `render` saw.
    """
    if not NAME_RE.match(name):
        raise DeployError(
            f"release name {name!r} must be a DNS label of at most 40"
            " characters", error_code="INVALID_FIELD")
    config = normalize_config(config)
    validate_recipe(recipe)
    validate_config(config)
    namespace = env["namespace"]
    chips = chips_per_replica(recipe, config)
    replicas = int(config["topology"]["decode"]["replicas"])

    out_dir = pathlib.Path(out_dir)
    _write(out_dir / MODELSERVER_FILE,
           modelserver_manifests(name, recipe, config, env, production))
    _write(out_dir / MONITORING_FILE, monitoring_manifests(name, production))
    _write(out_dir / ROUTER_VALUES_FILE, router_values(name, config))
    _write(out_dir / EPP_CONFIG_FILE, epp_config(config))

    bench_files: dict[str, dict[str, str]] = {}
    base_url = (f"http://{router_service(name, namespace)}"
                f":{pins.ROUTER_SERVICE_PORT}")
    for tenant in profile["tenants"]:
        for stage in BENCH_STAGES:
            ip_config = workload.render_inference_perf(
                profile,
                tenant,
                stage=stage,
                base_url=base_url,
                model_name=recipe["model"],
                duration_s=int(env.get("slo_stage_duration_s")
                               or workload.DEFAULT_SLO_STAGE_DURATION_S),
                report_dir=BENCH_REPORT_DIR)
            key = f"{tenant['name']}.{stage}"
            cm_path = f"{BENCH_DIR}/{key}.configmap.json"
            job_path = f"{BENCH_DIR}/{key}.job.json"
            _write(out_dir / cm_path,
                   bench_configmap(name, tenant["name"], stage, ip_config,
                                   production))
            _write(out_dir / job_path,
                   bench_job(name, tenant["name"], stage, env, production))
            bench_files[key] = {"configmap": cm_path, "job": job_path}

    _write(out_dir / CONFIG_FILE, config)
    _write(out_dir / RECIPE_FILE, recipe)
    _write(out_dir / PROFILE_FILE, profile)

    release = {
        "name": name,
        "rendered_at": now or datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "namespace": namespace,
        "production": bool(production),
        "model": recipe["model"],
        "image": recipe["image"],
        "accelerator": recipe["accelerator"],
        "replicas": replicas,
        "chips": {"per_replica": chips, "total": chips * replicas},
        "config_digest": config_digest(config),
        "router": {
            "profile": config["router"]["profile"],
            "helm_release": name,
            "chart": pins.ROUTER_STANDALONE_CHART,
            "chart_version": pins.ROUTER_CHART_VERSION,
            "epp_image": f"{pins.ROUTER_EPP_IMAGE}:{pins.ROUTER_EPP_VERSION}",
            "service": router_service(name, namespace),
            "port": pins.ROUTER_SERVICE_PORT,
        },
        "pins": {
            "llmd_ref": pins.LLMD_REF,
            "gaie_version": pins.GAIE_VERSION,
            "inference_perf_ref": pins.INFERENCE_PERF_REF,
            "inference_perf_image": env["inference_perf_image"],
        },
        "workload": {
            "name": profile["name"],
            "source": profile["source"],
            "tenants": [t["name"] for t in profile["tenants"]],
            "target_qps": profile["rate"]["target_qps"],
            "primary_metric": profile["objective"]["primary_metric"],
        },
        "env": env,
        "files": {
            "modelserver": MODELSERVER_FILE,
            "monitoring": MONITORING_FILE,
            "router_values": ROUTER_VALUES_FILE,
            "epp_config": EPP_CONFIG_FILE,
            "config": CONFIG_FILE,
            "recipe": RECIPE_FILE,
            "profile": PROFILE_FILE,
            "bench": bench_files,
        },
        "deployments": [f"{name}-decode"],
        "history": [],
    }
    _write(out_dir / RELEASE_FILE, release)
    return release


def rerender(release_dir: pathlib.Path,
             config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-render a release in place, optionally with a new config.

    Keeps `history` and the deployed state fields from the existing
    release.json so re-rendering for `reconfigure` does not forget what was
    applied.
    """
    release_dir = pathlib.Path(release_dir)
    old = json.loads((release_dir / RELEASE_FILE).read_text(encoding="utf-8"))
    recipe = json.loads((release_dir / RECIPE_FILE).read_text(encoding="utf-8"))
    profile = json.loads(
        (release_dir / PROFILE_FILE).read_text(encoding="utf-8"))
    if config is None:
        config = json.loads(
            (release_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    new = render_release(name=old["name"],
                         config=config,
                         recipe=recipe,
                         profile=profile,
                         env=old["env"],
                         out_dir=release_dir,
                         production=old.get("production", False),
                         now=old.get("rendered_at"))
    for key in ("history", "applied_at", "state", "job_id", "rerendered_at"):
        if key in old:
            new[key] = old[key]
    new["rerendered_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    _write(release_dir / RELEASE_FILE, new)
    return new


def load_release(release_dir: pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(release_dir) / RELEASE_FILE
    if not path.is_file():
        raise DeployError(f"{path} does not exist",
                          error_code="RELEASE_NOT_FOUND")
    return json.loads(path.read_text(encoding="utf-8"))


def save_release(release_dir: pathlib.Path, release: dict[str, Any]) -> None:
    _write(pathlib.Path(release_dir) / RELEASE_FILE, release)
