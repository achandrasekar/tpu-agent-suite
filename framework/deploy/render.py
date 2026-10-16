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
EXPOSURE_FILE = f"{MANIFEST_DIR}/50-exposure.json"
ROUTER_VALUES_FILE = "router.values.json"
EPP_CONFIG_FILE = "epp-config.json"
EPP_CONFIG_KEY = "tpu-coworker-plugins.yaml"
RELEASE_FILE = "release.json"
CONFIG_FILE = "llmd-config.json"
RECIPE_FILE = "serving-recipe.json"
PROFILE_FILE = "workload_profile.json"
BENCH_DIR = "bench"
BENCH_STAGES = ("sweep", "slo")
# How a production release's front door is reached. `cluster-ip` is the
# default because it adds nothing a shared cluster has to approve; the
# internal passthrough LB is the GKE-native next step
# (https://cloud.google.com/kubernetes-engine/docs/concepts/service-load-balancer).
EXPOSURE_CLUSTER_IP = "cluster-ip"
EXPOSURE_INTERNAL_LB = "internal-lb"
EXPOSURE_MODES = (EXPOSURE_CLUSTER_IP, EXPOSURE_INTERNAL_LB)
INTERNAL_LB_ANNOTATION = "networking.gke.io/load-balancer-type"

DEFAULT_MEMORY = "64Gi"
DEFAULT_CPU = "16"
DEFAULT_SHM = "64Gi"
MODELSERVER_CONTAINER = "modelserver"
# A simulated engine is a small Go process; these keep a sim fleet of many
# pools inside a CPU node pool.
SIM_MEMORY = "1Gi"
SIM_CPU = 1
SIM_SHM = "256Mi"
BENCH_CONTAINER = "inference-perf"
BENCH_REPORT_DIR = "/reports"
BENCH_CONFIG_DIR = "/config"
BENCH_REPORT_MARK = "===REPORT"
BENCH_REPORT_END = "===END==="

# Config paths the renderer refuses outright (none at present: every arm
# family in SEARCH.md renders). Kept as the place a future llm-d feature is
# parked until its manifests exist, so a planner that proposes it finds out
# at render time rather than after a pod exists. Entries are
# (path, predicate(config) -> bool, human name).
UNSUPPORTED_IN_THIS_MILESTONE: tuple = ()

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

# The control every router arm is measured against: no EPP at all, the
# benchmark sends straight to the model-server Service and kube-proxy
# spreads requests across pods. Not an EndpointPickerConfig, so it is not
# in ROUTER_PROFILES; the renderer skips the chart when it is chosen.
PASSTHROUGH_PROFILE = "passthrough"

# How a release relates to its model servers.
#   managed: the release renders, applies and tears down the Deployment.
#   adopted: the fleet is already running and owned by someone else; the
#     release only adds the router and benchmarks around it, selects the
#     pods by the labels the recipe names, and never applies, scales or
#     deletes them. This is the production case -- the customer's servers
#     are up -- and the only case on a cluster whose chips are already
#     held by the deployment under study.
FLEET_MANAGED = "managed"
FLEET_ADOPTED = "adopted"

# Incumbent paths that would change the model servers themselves. On an
# adopted fleet they are refused at render time with ADOPTED_FLEET_IMMUTABLE
# so the executor records `blocked_on_adopted_fleet` instead of restarting
# someone else's pods.
ADOPTED_IMMUTABLE_PATHS = (
    ("topology.mode", lambda c: c["topology"].get("mode", "aggregated") !=
     "aggregated"),
    ("topology.prefill", lambda c: bool(c["topology"].get("prefill"))),
    ("kv.precise_routing", lambda c: bool(c["kv"].get("precise_routing"))),
    ("kv.tiered_host", lambda c: bool(c["kv"].get("tiered_host"))),
    ("kv.pd_connector", lambda c: bool(c["kv"].get("pd_connector"))),
    ("kv.block_size", lambda c: c["kv"].get("block_size") is not None),
    ("engine", lambda c: bool(c["engine"])),
    ("topology.decode.engine",
     lambda c: bool((c["topology"].get("decode") or {}).get("engine"))),
)


def fleet_spec(recipe: dict[str, Any]) -> dict[str, Any]:
    """The recipe's fleet block, normalised; `managed` when it names none."""
    fleet = dict(recipe.get("fleet") or {})
    mode = str(fleet.get("mode") or FLEET_MANAGED)
    if mode not in (FLEET_MANAGED, FLEET_ADOPTED):
        raise DeployError(
            f"fleet.mode must be {FLEET_MANAGED!r} or {FLEET_ADOPTED!r},"
            f" got {mode!r}",
            error_code="INVALID_FIELD")
    if mode == FLEET_MANAGED:
        return {"mode": FLEET_MANAGED}
    selector = dict(fleet.get("selector") or {})
    if not selector or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in selector.items()):
        raise DeployError(
            "an adopted fleet needs fleet.selector, a non-empty map of pod"
            " labels", error_code="INVALID_FIELD")
    replicas = fleet.get("replicas")
    if not isinstance(replicas, int) or replicas < 1:
        raise DeployError("an adopted fleet needs fleet.replicas >= 1, the"
                          " number of pods the selector matches",
                          error_code="INVALID_FIELD")
    service = str(fleet.get("service") or "")
    if not service:
        raise DeployError(
            "an adopted fleet needs fleet.service, the ClusterIP Service in"
            " front of its pods (the passthrough control sends there)",
            error_code="INVALID_FIELD")
    return {
        "mode": FLEET_ADOPTED,
        "selector": selector,
        "replicas": int(replicas),
        "service": service,
        "port": int(fleet.get("port") or pins.MODEL_SERVER_PORT),
        "role": str(fleet.get("role") or "decode"),
    }


def epp_deployment(name: str) -> str:
    """The EPP Deployment the standalone chart creates for release `name`.

    From the chart's `llm-d-router.name` helper at the pinned version: the
    release name, lower-cased and truncated to 40, plus `-epp`.
    """
    return f"{name.lower()[:40]}-epp"


def epp_selector(name: str) -> str:
    """The pod label selector the chart puts on the EPP pods.

    The chart labels pods `llm-d-router-gateway: <name>-epp` whenever it
    creates the InferencePool (its default), which is the mode used here;
    `app.kubernetes.io/name` is on the Deployment only, not the pods.
    """
    return f"llm-d-router-gateway={epp_deployment(name)}"


# -- defaults and validation -------------------------------------------------


def default_config(recipe: dict[str, Any], replicas: int = 1) -> dict[str, Any]:
    """The incumbent a study starts from.

    A managed fleet starts as the recipe, N times, as llm-d's
    optimized-baseline deploys it. An adopted fleet starts as what is
    running today: its own replica count and no router (`passthrough`), so
    round 1 measures the state the customer already has and every router
    arm is a change from it.
    """
    accel = recipe.get("accelerator") or {}
    fleet = fleet_spec(recipe)
    adopted = fleet["mode"] == FLEET_ADOPTED
    if adopted:
        replicas = fleet["replicas"]
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
            "profile": PASSTHROUGH_PROFILE if adopted else
                       "sticky-until-saturated",
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
    out["topology"].setdefault("pd_decider", None)
    out.setdefault("router", {})
    out["router"].setdefault("profile", "sticky-until-saturated")
    out["router"].setdefault("plugins", {})
    out["router"].setdefault("flow_control", None)
    out["router"].setdefault("predicted_latency", False)
    out["router"].setdefault("epp_replicas", 1)
    out["router"].setdefault("epp_ha_mode", None)
    out["router"].setdefault("proxy_mode", None)
    out["router"].setdefault("proxy_replicas", None)
    out.setdefault("engine", {})
    out.setdefault("provenance", {})
    return out


def validate_config(config: dict[str, Any],
                    fleet: dict[str, Any] | None = None) -> None:
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
    mode = config["topology"].get("mode", "aggregated")
    if mode not in ("aggregated", "disaggregated"):
        raise DeployError("topology.mode must be aggregated or disaggregated",
                          error_code="INVALID_FIELD")
    prefill = config["topology"].get("prefill")
    if mode == "disaggregated":
        if not prefill or not isinstance(prefill.get("replicas"), int) or \
                prefill["replicas"] < 1:
            raise DeployError(
                "a disaggregated topology needs topology.prefill.replicas"
                " >= 1", error_code="INVALID_FIELD")
    elif prefill:
        raise DeployError(
            "topology.prefill is set but topology.mode is aggregated",
            error_code="INVALID_FIELD")
    profile = config["router"]["profile"]
    if profile == PASSTHROUGH_PROFILE:
        hot = [k for k in ("flow_control", "predicted_latency")
               if config["router"].get(k)]
        if config["kv"].get("precise_routing"):
            hot.append("kv.precise_routing")
        if mode == "disaggregated":
            hot.append("topology.mode=disaggregated")
        if hot:
            raise DeployError(
                f"{hot} need an EPP; the passthrough profile has none",
                error_code="INVALID_FIELD")
    for role in (ROLE_PREFILL, ROLE_DECODE):
        hosts = (config["topology"].get(role) or {}).get("hosts")
        if hosts is not None and (not isinstance(hosts, int) or hosts < 1):
            raise DeployError(f"topology.{role}.hosts must be an integer"
                              " >= 1", error_code="INVALID_FIELD")
    ha = config["router"].get("epp_ha_mode")
    if ha not in (None, "active-passive", "active-active"):
        raise DeployError("router.epp_ha_mode must be active-passive or"
                          " active-active", error_code="INVALID_FIELD")
    proxy_mode = config["router"].get("proxy_mode")
    if proxy_mode not in (None, "sidecar", "service"):
        raise DeployError("router.proxy_mode must be sidecar or service",
                          error_code="INVALID_FIELD")
    offload = config["kv"].get("tiered_host")
    if offload is not None and not isinstance(offload, dict):
        raise DeployError("kv.tiered_host must be an object",
                          error_code="INVALID_FIELD")
    if profile != PASSTHROUGH_PROFILE and profile not in ROUTER_PROFILES:
        raise DeployError(
            f"router.profile {profile!r} is not one of"
            f" {sorted(ROUTER_PROFILES) + [PASSTHROUGH_PROFILE]}",
            error_code="INVALID_FIELD")
    if profile == PASSTHROUGH_PROFILE and config["router"].get("plugins"):
        raise DeployError(
            "router.plugins has no effect under the passthrough profile;"
            " there is no EPP to configure",
            error_code="INVALID_FIELD")
    allowed = set(ENGINE_FLAGS) | set(ENGINE_SWITCHES)
    scopes = [("engine", config["engine"])] + [
        (f"topology.{role}.engine",
         (config["topology"].get(role) or {}).get("engine") or {})
        for role in (ROLE_PREFILL, ROLE_DECODE)]
    for scope, knobs in scopes:
        unknown = set(knobs) - allowed
        if unknown:
            raise DeployError(
                f"{scope} knobs {sorted(unknown)} are outside the fleet"
                f" whitelist {sorted(allowed)}; hand them to the"
                " engine-optimization domain instead",
                error_code="INVALID_FIELD")
    if (config["topology"].get("prefill") or {}).get("engine") and \
            mode != "disaggregated":
        raise DeployError(
            "topology.prefill.engine is set but there is no prefill pool",
            error_code="INVALID_FIELD")
    if fleet and fleet["mode"] == FLEET_ADOPTED:
        touched = [path for path, is_on in ADOPTED_IMMUTABLE_PATHS
                   if is_on(config)]
        if replicas != fleet["replicas"]:
            touched.append("topology.decode.replicas")
        if touched:
            raise DeployError(
                f"{touched} would change the adopted fleet's model servers,"
                " which this release does not own; only router paths may"
                " change on an adopted fleet",
                error_code="ADOPTED_FLEET_IMMUTABLE")


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
    fleet_spec(recipe)


def validate_recipe_config(recipe: dict[str, Any],
                           config: dict[str, Any]) -> None:
    """Refusals that need both the recipe and the config.

    A simulated engine has no KV connector and publishes no events the EPP
    can index without a tokenizer service, so precise routing and host
    tiering are `NOT_IMPLEMENTED` on it rather than silently inert: an arm
    the sim cannot exercise must say so, not tie.
    """
    engine_flavor(recipe)
    if is_sim(recipe):
        if config["kv"].get("precise_routing"):
            raise DeployError(
                "kv.precise_routing has no simulated counterpart"
                " (engine_flavor sim)", error_code="NOT_IMPLEMENTED")
        if config["kv"].get("tiered_host"):
            raise DeployError(
                "kv.tiered_host has no simulated counterpart"
                " (engine_flavor sim)", error_code="NOT_IMPLEMENTED")
    for role in (ROLE_PREFILL, ROLE_DECODE):
        if pool_hosts(config, role) > 1:
            chips_per_replica(recipe, config, role)  # refuses without a host size
            if not (recipe.get("multihost") or {}).get("worker_command"):
                raise DeployError(
                    f"topology.{role}.hosts > 1 needs recipe.multihost."
                    "worker_command", error_code="INVALID_FIELD")


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


def pool_hosts(config: dict[str, Any], role: str = "decode") -> int:
    """Hosts per replica of a pool: 1 is a Deployment, more an LWS group."""
    pool = config["topology"].get(role) or config["topology"]["decode"]
    return max(1, int(pool.get("hosts") or 1))


def chips_per_host(recipe: dict[str, Any]) -> int | None:
    """Chips one node exposes, from the recipe or the pinned generation
    table; None when neither says (a multi-host render then refuses)."""
    accel = recipe.get("accelerator") or {}
    explicit = accel.get("chips_per_host")
    if explicit:
        return int(explicit)
    return pins.CHIPS_PER_HOST.get(str(accel.get("tpu_accelerator") or ""))


def chips_per_replica(recipe: dict[str, Any], config: dict[str, Any],
                      role: str = "decode") -> int:
    """Chips one replica of a pool holds.

    Single-host: the recipe's explicit count or the slice's product.
    Multi-host (`hosts > 1`): hosts x chips per host, because each pod of
    the group requests all the chips on its node.
    """
    accel = recipe.get("accelerator") or {}
    hosts = pool_hosts(config, role)
    if hosts > 1:
        per_host = chips_per_host(recipe)
        if not per_host:
            raise DeployError(
                f"topology.{role}.hosts={hosts} needs accelerator."
                "chips_per_host (chips one node exposes) in the recipe",
                error_code="INVALID_FIELD")
        return hosts * per_host
    explicit = accel.get("chips_per_replica")
    if explicit:
        return int(explicit)
    pool = config["topology"].get(role) or config["topology"]["decode"]
    slice_shape = pool.get("slice") or config["topology"]["decode"].get(
        "slice") or accel.get("tpu_topology", "")
    return chips_for_topology(slice_shape)


def total_chips(recipe: dict[str, Any], config: dict[str, Any]) -> int:
    """Chips the whole fleet holds: decode, plus prefill under P/D."""
    decode = config["topology"]["decode"]
    total = chips_per_replica(recipe, config, "decode") * int(
        decode.get("replicas") or 1)
    prefill = config["topology"].get("prefill")
    if config["topology"].get("mode") == "disaggregated" and prefill:
        total += chips_per_replica(recipe, config, "prefill") * int(
            prefill.get("replicas") or 1)
    return total


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


# Engine flavours: which connector modules the serving image carries. The
# probe (`deploy probe`) reports the modules; `recipe.engine_flavor` or the
# config's connector blocks pick the names rendered into the manifests.
ENGINE_FLAVORS: dict[str, dict[str, str]] = {
    "tpu_inference": {
        "pd_module": pins.TPU_PD_MODULE_TPU_INFERENCE,
        "pd_connector": pins.TPU_PD_CONNECTOR_TPU_INFERENCE,
        "offload_module": pins.TPU_OFFLOAD_MODULE_TPU_INFERENCE,
        "offload_connector": pins.TPU_OFFLOAD_CONNECTOR_TPU_INFERENCE,
    },
    "torchtpu": {
        "pd_module": pins.TPU_PD_MODULE_TORCHTPU,
        "pd_connector": pins.TPU_PD_CONNECTOR_TORCHTPU,
        "offload_module": pins.TPU_OFFLOAD_MODULE_TORCHTPU,
        "offload_connector": pins.TPU_OFFLOAD_CONNECTOR_TORCHTPU,
    },
    # llm-d-inference-sim in place of vLLM: a simulated fleet that runs the
    # whole orchestration (router, pools, LWS groups, P/D through the
    # sidecar, benchmarks, collect) on CPU nodes. It has no KV connector
    # (the sidecar's nixlv2 path is generic enough for the sim's
    # do_remote_prefill), no KV events the EPP can consume without a
    # tokenizer service, and no host tier.
    "sim": {
        "pd_module": "",
        "pd_connector": "",
        "offload_module": "",
        "offload_connector": "",
    },
}
SIM_FLAVOR = "sim"
ROLE_DECODE = "decode"
ROLE_PREFILL = "prefill"


def is_sim(recipe: dict[str, Any]) -> bool:
    return str(recipe.get("engine_flavor") or "") == SIM_FLAVOR


def engine_flavor(recipe: dict[str, Any]) -> dict[str, str]:
    flavor = str(recipe.get("engine_flavor") or "tpu_inference")
    if flavor not in ENGINE_FLAVORS:
        raise DeployError(
            f"engine_flavor {flavor!r} is not one of {sorted(ENGINE_FLAVORS)}",
            error_code="INVALID_FIELD")
    return ENGINE_FLAVORS[flavor]


def kv_transfer_config(recipe: dict[str, Any], config: dict[str, Any],
                       role: str) -> dict[str, Any] | None:
    """The `--kv-transfer-config` JSON for one role, or None.

    P/D follows guides/pd-disaggregation/modelserver/tpu/v7: the prefill
    pool is `kv_producer`, decode is `kv_consumer`, both keyed by POD_IP.
    Host offload follows guides/tiered-prefix-cache (`kv_both`). When a
    config asks for both, vLLM's MultiConnector is not rendered -- the
    executor records the arm blocked -- because no TPU guide ships that
    combination.
    """
    flavor = engine_flavor(recipe)
    kv = config["kv"]
    pd = config["topology"].get("mode") == "disaggregated"
    offload = kv.get("tiered_host") or None
    if pd and offload:
        raise DeployError(
            "P/D and host offload together are not a documented TPU path;"
            " run them as separate arms", error_code="NOT_IMPLEMENTED")
    if pd:
        conn = dict(kv.get("pd_connector") or {})
        return {
            "kv_connector": conn.get("connector") or flavor["pd_connector"],
            "kv_connector_module_path": conn.get("module") or
                                        flavor["pd_module"],
            "kv_role": "kv_producer" if role == ROLE_PREFILL else
                       "kv_consumer",
            "kv_ip": "$(POD_IP)",
            **({"kv_connector_extra_config": conn["extra_config"]}
               if conn.get("extra_config") else {}),
        }
    if offload:
        doc = {
            "kv_connector": offload.get("connector") or
                            flavor["offload_connector"],
            "kv_connector_module_path": offload.get("module") or
                                        flavor["offload_module"],
            "kv_role": "kv_both",
        }
        if offload.get("extra_config"):
            doc["kv_connector_extra_config"] = offload["extra_config"]
        return doc
    return None


def kv_events_config(model: str, port: int) -> dict[str, Any]:
    """vLLM's `--kv-events-config`, as guides/precise-prefix-cache-routing
    sets it: ZMQ publisher on the pod IP, topic keyed by pod and model so
    the EPP's indexer can attribute blocks to endpoints."""
    return {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{pins.KV_EVENTS_PORT}",
        "replay_endpoint": f"tcp://*:{pins.KV_EVENTS_REPLAY_PORT}",
        "topic": f"kv@$(POD_IP):{port}@{model}",
    }


def role_engine(config: dict[str, Any], role: str) -> dict[str, Any]:
    """The engine knobs one pool runs: the fleet default under the pool's
    own overlay.

    Prefill and decode want different engines (prefill: large batched-token
    budgets; decode: many sequences), so `topology.<role>.engine` overrides
    `engine` key by key. The whitelist is the same; an overlay never adds
    a knob the fleet may not tune.
    """
    merged = dict(config.get("engine") or {})
    pool = (config.get("topology") or {}).get(role) or {}
    for key, value in (pool.get("engine") or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def vllm_args(recipe: dict[str, Any], config: dict[str, Any],
              role: str = ROLE_DECODE) -> list[str]:
    """`vllm serve` arguments for one role, as the TPU overlays order them.

    Under P/D the decode engine listens on PD_DECODE_PORT behind the routing
    sidecar, which takes the model-server port; prefill keeps the default.
    """
    pd = config["topology"].get("mode") == "disaggregated"
    pool = config["topology"].get(role) or config["topology"]["decode"]
    tp = pool.get("tp") or config["topology"]["decode"].get("tp") or \
        recipe.get("tensor_parallel") or 1
    port = pins.PD_DECODE_PORT if (pd and role == ROLE_DECODE) else \
        pins.MODEL_SERVER_PORT
    base = [
        str(recipe["model"]),
        f"--port={port}",
        "--disable-access-log-for-endpoints=/health,/metrics,/v1/models",
        f"--tensor-parallel-size={int(tp)}",
    ]
    engine = role_engine(config, role)
    overrides: dict[str, Any] = {}
    for key, flag in ENGINE_FLAGS.items():
        if engine.get(key) is not None:
            overrides[flag] = engine[key]
    if config["kv"].get("block_size"):
        overrides["--block-size"] = int(config["kv"]["block_size"])
    switches = {
        flag: bool(engine.get(key))
        for key, flag in ENGINE_SWITCHES.items()
        if key in engine
    }
    transfer = kv_transfer_config(recipe, config, role)
    if transfer:
        overrides["--kv-transfer-config"] = json.dumps(
            transfer, separators=(",", ":"))
    if config["kv"].get("precise_routing"):
        overrides["--kv-events-config"] = json.dumps(
            kv_events_config(str(recipe["model"]), port),
            separators=(",", ":"))
        # The EPP's vLLM token backend tokenizes through the engines'
        # /v1/*/render endpoints, which this flag enables.
        switches["--enable-scale-out"] = True
    recipe_args = [
        a for a in list(recipe.get("args") or [])
        if _flag_name(a) not in ("--port", "--tensor-parallel-size")
    ]
    return base + merge_args(recipe_args, overrides, switches)


# Fleet knobs the simulator understands, and its flag for each. The rest
# of the whitelist (batched tokens, memory utilisation, KV dtype) has no
# simulated counterpart and is dropped from a sim render.
SIM_ENGINE_FLAGS: dict[str, str] = {
    "max_num_seqs": "--max-num-seqs",
    "max_model_len": "--max-model-len",
}
SIM_PASSTHROUGH_DROP = ("--port", "--model", "--tensor-parallel-size",
                        "--kv-transfer-config", "--kv-events-config",
                        "--enable-scale-out", "--render-url")


def sim_args(recipe: dict[str, Any], config: dict[str, Any],
             role: str = ROLE_DECODE) -> list[str]:
    """llm-d-inference-sim arguments for one pool.

    The simulator aborts on any flag it does not register, so nothing of
    vLLM's is passed through except what the recipe's `args` carry (the
    recipe author chose the sim's latency model there). `--render-url=`
    empty selects the simulator's own tokenizer, so no render service is
    needed; the model name is whatever the recipe says.
    """
    pd = config["topology"].get("mode") == "disaggregated"
    port = pins.PD_DECODE_PORT if (pd and role == ROLE_DECODE) else \
        pins.MODEL_SERVER_PORT
    base = [f"--model={recipe['model']}", f"--port={port}",
            "--render-url="]
    engine = role_engine(config, role)
    overrides: dict[str, Any] = {}
    for key, flag in SIM_ENGINE_FLAGS.items():
        if engine.get(key) is not None:
            overrides[flag] = engine[key]
    if config["kv"].get("block_size"):
        overrides["--block-size"] = int(config["kv"]["block_size"])
    recipe_args = [a for a in list(recipe.get("args") or [])
                   if _flag_name(a) not in SIM_PASSTHROUGH_DROP]
    return base + merge_args(recipe_args, overrides, {})


# -- model server manifests ----------------------------------------------------


def release_labels(name: str, production: bool) -> dict[str, str]:
    labels = {
        pins.RELEASE_LABEL: name,
        pins.MANAGED_BY_LABEL: pins.MANAGED_BY_VALUE,
    }
    if production:
        labels[pins.PRODUCTION_LABEL] = "true"
    return labels


def pool_names(name: str, config: dict[str, Any]) -> list[str]:
    """Workload names a managed release owns: decode, plus prefill under P/D."""
    out = [f"{name}-{ROLE_DECODE}"]
    if config["topology"].get("mode") == "disaggregated":
        out.append(f"{name}-{ROLE_PREFILL}")
    return out


def pool_records(name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """What each pool is on the cluster, so `wait`/`status`/`drill` read
    the right kind: a Deployment for single-host replicas, a
    LeaderWorkerSet for multi-host ones."""
    roles = [ROLE_DECODE]
    if config["topology"].get("mode") == "disaggregated":
        roles.append(ROLE_PREFILL)
    out = []
    for role in roles:
        hosts = pool_hosts(config, role)
        out.append({
            "name": f"{name}-{role}",
            "role": role,
            "kind": "LeaderWorkerSet" if hosts > 1 else "Deployment",
            "replicas": int((config["topology"].get(role) or {}).get(
                "replicas") or 1),
            "hosts": hosts,
        })
    return out


def offload_env(config: dict[str, Any]) -> list[dict[str, str]]:
    """Host-offload sizing env, per guides/tiered-prefix-cache."""
    offload = config["kv"].get("tiered_host") or {}
    out = []
    if offload.get("cpu_chunks") is not None:
        out.append({"name": pins.TPU_OFFLOAD_CPU_CHUNKS_ENV,
                    "value": str(int(offload["cpu_chunks"]))})
    if offload.get("staging_blocks") is not None:
        out.append({"name": pins.TPU_OFFLOAD_STAGING_BLOCKS_ENV,
                    "value": str(int(offload["staging_blocks"]))})
    for key in sorted(offload.get("env") or {}):
        out.append({"name": key, "value": str(offload["env"][key])})
    return out


def _server_deployment(name: str, role: str, recipe: dict[str, Any],
                       config: dict[str, Any], env: dict[str, Any],
                       production: bool) -> dict[str, Any]:
    accel = recipe["accelerator"]
    pd = config["topology"].get("mode") == "disaggregated"
    pool = config["topology"].get(role) or config["topology"]["decode"]
    replicas = int(pool.get("replicas") or 1)
    sim = is_sim(recipe)
    hosts = pool_hosts(config, role)
    chips = chips_per_replica(recipe, config, role)
    # Every pod of a multi-host group requests all of its node's chips.
    chips_per_pod = chips // hosts if hosts > 1 else chips
    slice_shape = pool.get("slice") or accel.get("tpu_topology") or ""
    resources = dict(recipe.get("resources") or {})
    if sim:
        limits = {"memory": resources.get("memory", SIM_MEMORY),
                  "cpu": str(resources.get("cpu", SIM_CPU))}
    else:
        limits = {
            "memory": resources.get("memory", DEFAULT_MEMORY),
            "cpu": str(resources.get("cpu", DEFAULT_CPU)),
            pins.TPU_RESOURCE: chips_per_pod,
        }
    pod_labels = {
        **release_labels(name, production),
        pins.LLMD_ROLE_LABEL: role,
        pins.LLMD_SERVING_LABEL: "true",
        pins.LLMD_ENGINE_LABEL: "vllm",
        pins.LLMD_GUIDE_LABEL: "tpu-coworker",
        "llm-d.ai/model": label_value(recipe["model"]),
    }
    if hosts > 1:
        pod_labels[pins.MEMBER_LABEL] = "leader"
    selector = {pins.RELEASE_LABEL: name, pins.LLMD_ROLE_LABEL: role}

    env_list: list[dict[str, Any]] = [{
        "name": "HF_TOKEN",
        "valueFrom": {
            "secretKeyRef": {
                "name": env["hf_token_secret"],
                "key": "HF_TOKEN"
            }
        },
    }]
    # The simulator refuses to start its KV cache without POD_IP (found
    # live: "IP should be defined in the environment (POD_IP)"), so a sim
    # pod always carries it.
    needs_pod_ip = pd or bool(config["kv"].get("precise_routing")) or sim
    if needs_pod_ip:
        env_list.append({"name": "POD_IP", "valueFrom": {
            "fieldRef": {"fieldPath": "status.podIP"}}})
    if pd and not sim:
        env_list += [
            {"name": "TPU_SIDE_CHANNEL_PORT",
             "value": str(pins.TPU_SIDE_CHANNEL_PORT)},
            {"name": "TPU_KV_TRANSFER_PORT",
             "value": str(pins.TPU_KV_TRANSFER_PORT)},
        ]
        if role == ROLE_PREFILL:
            env_list.append({"name": "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
                             "value": "120"})
    if not sim:
        env_list += offload_env(config)
    for key in sorted(recipe.get("env") or {}):
        env_list.append({"name": key, "value": str(recipe["env"][key])})
    multihost = dict(recipe.get("multihost") or {})
    if hosts > 1:
        for key in sorted(multihost.get("env") or {}):
            env_list.append({"name": key,
                             "value": str(multihost["env"][key])})

    engine_port = pins.PD_DECODE_PORT if (pd and role == ROLE_DECODE) else \
        pins.MODEL_SERVER_PORT
    ports = [{"name": MODELSERVER_CONTAINER, "containerPort": engine_port,
              "protocol": "TCP"}]
    if pd and not sim:
        ports += [
            {"name": "tpu-kv-transfer",
             "containerPort": pins.TPU_KV_TRANSFER_PORT, "protocol": "TCP"},
            {"name": "tpu-pd-coord",
             "containerPort": pins.TPU_SIDE_CHANNEL_PORT, "protocol": "TCP"},
        ]
    if config["kv"].get("precise_routing"):
        ports += [
            {"name": "kv-events", "containerPort": pins.KV_EVENTS_PORT,
             "protocol": "TCP"},
            {"name": "kv-replay", "containerPort": pins.KV_EVENTS_REPLAY_PORT,
             "protocol": "TCP"},
        ]

    container = {
        "name": MODELSERVER_CONTAINER,
        "image": recipe["image"],
        **({} if sim else {"command": ["vllm", "serve"]}),
        "args": sim_args(recipe, config, role) if sim else
                vllm_args(recipe, config, role),
        "env": env_list,
        "ports": ports,
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
    node_selector: dict[str, str] = {}
    if sim:
        node_selector = dict(env.get("cpu_node_selector") or {})
    else:
        node_selector = {
            pins.TOPOLOGY_LABEL: slice_shape,
            pins.ACCELERATOR_LABEL: accel["tpu_accelerator"],
        }
        if hosts > 1 and accel.get("placement_policy"):
            node_selector[pins.PLACEMENT_POLICY_LABEL] = str(
                accel["placement_policy"])
    pod_spec: dict[str, Any] = {
        "serviceAccountName": f"{name}-modelserver",
        **({"nodeSelector": node_selector} if node_selector else {}),
        "containers": [container],
        "volumes": [
            {"name": "torch-compile-cache", "emptyDir": {}},
            {"name": "triton-cache", "emptyDir": {}},
            {"name": "vllm-config", "emptyDir": {}},
            {
                "name": "dshm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": resources.get(
                        "shm", SIM_SHM if sim else DEFAULT_SHM)
                }
            },
        ],
    }
    if pd and role == ROLE_DECODE:
        # The routing sidecar fronts decode on the model-server port and
        # forwards prefill to the pod the EPP named in PD_PREFILL_HEADER.
        # A native sidecar (restartPolicy Always init container), as the
        # guide's patch-sidecar.yaml adds it.
        sidecar_conn = (config["kv"].get("pd_connector") or {}).get(
            "sidecar_connector") or "nixlv2"
        pod_spec["initContainers"] = [{
            "name": "routing-proxy",
            "image": f"{pins.PD_SIDECAR_IMAGE}:{pins.PD_SIDECAR_VERSION}",
            "restartPolicy": "Always",
            "args": [
                f"--port={pins.MODEL_SERVER_PORT}",
                f"--kv-connector={sidecar_conn}",
                "--zap-log-level=1",
                "--secure-proxy=false",
            ],
            "ports": [{"name": "proxy",
                       "containerPort": pins.MODEL_SERVER_PORT,
                       "protocol": "TCP"}],
            "resources": {"requests": {"cpu": "1", "memory": "1Gi"},
                          "limits": {"memory": "4Gi"}},
        }]
    metadata = {
        "name": f"{name}-{role}",
        "labels": pod_labels,
        "annotations": {
            "tpu-coworker.llmd/config-digest": config_digest(config)
        },
    }
    if hosts == 1:
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": selector},
                "template": {
                    "metadata": {"labels": pod_labels},
                    "spec": pod_spec,
                },
            },
        }
    # A multi-host replica is a LeaderWorkerSet group: the leader runs the
    # engine and is the only member the Service, the InferencePool and the
    # PodMonitoring select (workers carry no role/serving label). Workers
    # run the recipe's `multihost.worker_command`; how an image joins its
    # hosts is the recipe's knowledge, not the renderer's.
    worker_command = list(multihost.get("worker_command") or [])
    if not worker_command:
        raise DeployError(
            f"topology.{role}.hosts={hosts} needs recipe.multihost."
            "worker_command (what the non-leader hosts run)",
            error_code="INVALID_FIELD")
    worker_labels = {**release_labels(name, production),
                     pins.MEMBER_LABEL: "worker",
                     "llm-d.ai/model": label_value(recipe["model"])}
    worker_container = {
        "name": "worker",
        "image": recipe["image"],
        "command": worker_command,
        "env": [e for e in env_list if e["name"] != "POD_IP"] + [
            {"name": "POD_IP", "valueFrom": {
                "fieldRef": {"fieldPath": "status.podIP"}}}],
        "resources": {"limits": limits, "requests": dict(limits)},
        "volumeMounts": list(container["volumeMounts"]),
    }
    worker_spec = {
        key: value for key, value in pod_spec.items()
        if key not in ("containers", "initContainers")}
    worker_spec["containers"] = [worker_container]
    return {
        "apiVersion": pins.LWS_API_VERSION,
        "kind": "LeaderWorkerSet",
        "metadata": metadata,
        "spec": {
            "replicas": replicas,
            "startupPolicy": "LeaderCreated",
            "leaderWorkerTemplate": {
                "size": hosts,
                "restartPolicy": "RecreateGroupOnPodRestart",
                "leaderTemplate": {
                    "metadata": {"labels": pod_labels},
                    "spec": pod_spec,
                },
                "workerTemplate": {
                    "metadata": {"labels": worker_labels},
                    "spec": worker_spec,
                },
            },
        },
    }


def modelserver_manifests(name: str, recipe: dict[str, Any],
                          config: dict[str, Any], env: dict[str, Any],
                          production: bool) -> dict[str, Any]:
    """ServiceAccount, Deployments (decode; prefill under P/D) and Service.

    The Service always fronts decode on the model-server port: under P/D
    that is the routing sidecar, so the passthrough control still works
    against a disaggregated fleet (every request runs both stages locally
    when no prefill header arrives).
    """
    pd = config["topology"].get("mode") == "disaggregated"
    items: list[dict[str, Any]] = [{
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": f"{name}-modelserver",
            "labels": release_labels(name, production)
        },
    }]
    items.append(_server_deployment(name, ROLE_DECODE, recipe, config, env,
                                    production))
    if pd:
        items.append(_server_deployment(name, ROLE_PREFILL, recipe, config,
                                        env, production))
    items.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-modelserver",
            "labels": release_labels(name, production)
        },
        "spec": {
            "type": "ClusterIP",
            "selector": {pins.RELEASE_LABEL: name,
                         pins.LLMD_ROLE_LABEL: ROLE_DECODE},
            "ports": [{
                "name": "http",
                "port": pins.MODEL_SERVER_PORT,
                "targetPort": pins.MODEL_SERVER_PORT,
                "protocol": "TCP",
            }],
        },
    })
    return {"apiVersion": "v1", "kind": "List", "items": items}


# -- router ------------------------------------------------------------------


def _precise_plugins(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Producers precise prefix routing needs instead of the approx one.

    Per the v0.11.0 precise-prefix-cache-producer README, the EPP recomputes
    block keys from real tokens, so it needs the vLLM `token-producer`
    backend (served by the engines' `/v1/.../render` endpoints) and the
    kv-events listener; `tokenProcessorConfig.blockSizeTokens` follows the
    engine's `--block-size` as the guide does.
    """
    kv = config["kv"]
    block = int(kv.get("block_size") or pins.KV_EVENTS_BLOCK_SIZE_DEFAULT)
    return [
        {"type": "token-producer",
         "parameters": {"backend": "vllm", "modelName": "${MODEL}",
                        "vllm": {"url": "${RENDER_URL}"}}},
        {"type": "precise-prefix-cache-producer",
         "parameters": {
             "tokenProcessorConfig": {"blockSizeTokens": block},
             "kvEventsConfig": {
                 "topicFilter": "kv@",
                 "concurrency": 8,
                 "discoverPods": True,
                 "podDiscoveryConfig": {
                     "socketPort": pins.KV_EVENTS_PORT,
                     "replaySocketPort": pins.KV_EVENTS_REPLAY_PORT,
                 },
             },
             "speculativeIndexing": True,
         }},
    ]


def _pd_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Scheduling profiles for a disaggregated fleet.

    v0.11.0 registers `disagg-profile-handler` with pluggable deciders; the
    older `pd-profile-handler` / `prefill-header-handler` names are gone.
    `always-disagg-pd-decider` sends every request through prefill, which
    is what the llm-d P/D guide ships; `prefix-based-pd-decider` is the
    knob the inner sweep tunes (`nonCachedTokens`) once the arm is live.
    """
    decider = (config["topology"].get("pd_decider") or
               {"type": "always-disagg-pd-decider"})
    return {
        "plugins": [
            {"type": "disagg-profile-handler",
             "parameters": {"deciders": {"prefill": decider["type"]}}},
            {"type": decider["type"],
             **({"parameters": decider["parameters"]}
                if decider.get("parameters") else {})},
            {"type": "prefill-filter"},
            {"type": "decode-filter"},
            {"type": "token-load-scorer"},
            {"type": "max-score-picker"},
        ],
        "profiles": [
            {"name": "prefill", "plugins": [
                {"pluginRef": "prefill-filter"},
                {"pluginRef": "token-load-scorer"},
                {"pluginRef": "max-score-picker"},
            ]},
            {"name": "decode", "plugins": [
                {"pluginRef": "decode-filter"},
                # decode scoring is the chosen profile's scorers
            ]},
        ],
    }


# Plugins that predicted-latency routing adds, per the v0.11.0 chart's
# latencyPredictor-enabled default config. Alpha: needs
# `--allow-experimental-plugins` and the predictor sidecars.
PREDICTED_LATENCY_PLUGINS = [
    {"type": "predicted-latency-producer"},
    {"type": "latency-scorer"},
    {"type": "slo-headroom-tier-filter"},
    {"type": "latency-slo-admitter"},
]
PREDICTED_LATENCY_PROFILE = [
    {"pluginRef": "slo-headroom-tier-filter"},
    {"pluginRef": "latency-scorer", "weight": 1.0},
]
EXPERIMENTAL_PLUGINS = {p["type"] for p in PREDICTED_LATENCY_PLUGINS} | {
    "session-affinity-scorer"}


def epp_config(config: dict[str, Any]) -> dict[str, Any]:
    """The EndpointPickerConfig for the config's router profile and knobs.

    The profile gives the plugin set and the default scheduling profile;
    `kv.precise_routing`, `router.flow_control`, `router.predicted_latency`
    and a disaggregated topology each add their documented block. The
    router profile's own scorers always stay in the decode-side profile so
    an advanced arm is "profile + one more thing", not a different router.
    """
    router = config["router"]
    profile = ROUTER_PROFILES[router["profile"]]
    overrides = router.get("plugins") or {}
    plugins: list[dict[str, Any]] = []
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
            f" {router['profile']!r} does not instantiate",
            error_code="INVALID_FIELD")

    if config["kv"].get("precise_routing"):
        # Swap the approximate producer for the precise one and point every
        # prefix consumer at it.
        plugins = [p for p in plugins
                   if p["type"] != "approx-prefix-cache-producer"]
        plugins = _precise_plugins(config) + plugins
        for p in plugins:
            if p["type"] in ("prefix-cache-affinity-filter",
                             "prefix-cache-scorer",
                             "inflight-load-producer"):
                p.setdefault("parameters", {})[
                    "prefixMatchInfoProducerName"] = \
                    "precise-prefix-cache-producer"

    if router.get("predicted_latency"):
        plugins += copy.deepcopy(PREDICTED_LATENCY_PLUGINS)
        refs = copy.deepcopy(PREDICTED_LATENCY_PROFILE) + refs

    doc: dict[str, Any] = {
        "apiVersion": pins.EPP_CONFIG_API_VERSION,
        "kind": pins.EPP_CONFIG_KIND,
    }
    flow = router.get("flow_control")
    if flow:
        doc["featureGates"] = ["flowControl"]
        doc["flowControl"] = flow_control_block(flow)
        plugins = plugins + flow_control_plugins(flow)
    doc["plugins"] = plugins
    if config["topology"].get("mode") == "disaggregated":
        pd = _pd_profile(config)
        doc["plugins"] = pd["plugins"] + [
            p for p in plugins if p["type"] not in
            {q["type"] for q in pd["plugins"]}]
        decode_profile = dict(pd["profiles"][1])
        decode_profile["plugins"] = decode_profile["plugins"] + refs
        doc["schedulingProfiles"] = [pd["profiles"][0], decode_profile]
    else:
        doc["schedulingProfiles"] = [{"name": "default", "plugins": refs}]
    return doc


def flow_control_block(flow: dict[str, Any]) -> dict[str, Any]:
    """The `flowControl:` block, from the config's compact form.

    Keys follow docs/architecture/core/router/epp/flow-control.md: a
    saturation detector (utilization by default), optional queue bounds and
    TTL, and one priority band per tenant priority the workload names.
    """
    detector = dict(flow.get("saturation_detector") or {})
    det_type = detector.pop("type", "utilization-detector")
    out: dict[str, Any] = {
        "saturationDetector": {"pluginRef": det_type},
    }
    if flow.get("max_requests") is not None:
        out["maxRequests"] = int(flow["max_requests"])
    if flow.get("max_bytes") is not None:
        out["maxBytes"] = int(flow["max_bytes"])
    if flow.get("default_request_ttl"):
        out["defaultRequestTTL"] = str(flow["default_request_ttl"])
    if flow.get("enable_eviction") is not None:
        out["enableEviction"] = bool(flow["enable_eviction"])
    bands = []
    for band in flow.get("priority_bands") or []:
        item = {"priority": int(band["priority"])}
        for src, dst in (("max_requests", "maxRequests"),
                         ("max_bytes", "maxBytes"),
                         ("ordering_policy", "orderingPolicyRef"),
                         ("fairness_policy", "fairnessPolicyRef")):
            if band.get(src) is not None:
                item[dst] = band[src]
        bands.append(item)
    if bands:
        out["priorityBands"] = bands
    return out


def flow_control_plugins(flow: dict[str, Any]) -> list[dict[str, Any]]:
    """Detector plugin entry when the config tunes its thresholds."""
    detector = dict(flow.get("saturation_detector") or {})
    det_type = detector.pop("type", "utilization-detector")
    params = {k: v for k, v in detector.items()}
    if not params:
        return []
    return [{"type": det_type, "parameters": params}]


def modelserver_selector(name: str, fleet: dict[str, Any]) -> dict[str, str]:
    """The pod labels the router and monitoring select model servers by."""
    if fleet["mode"] == FLEET_ADOPTED:
        return dict(fleet["selector"])
    return {pins.RELEASE_LABEL: name, pins.LLMD_ROLE_LABEL: "decode"}


def modelserver_port(fleet: dict[str, Any]) -> int:
    return int(fleet.get("port") or pins.MODEL_SERVER_PORT)


def render_service_url(name: str, namespace: str, fleet: dict[str, Any]
                       ) -> str:
    """Where the EPP's vLLM token-producer renders prompts: the model
    servers' own Service (any replica can tokenize)."""
    host, port = modelserver_service(name, namespace, fleet)
    return f"http://{host}:{port}"


def router_values(name: str, config: dict[str, Any],
                  fleet: dict[str, Any] | None = None,
                  env: dict[str, Any] | None = None,
                  model: str = "", tenants: list[dict[str, Any]] | None = None
                  ) -> dict[str, Any]:
    """Values for the standalone router chart: base + guide overrides.

    `modelServers.targetPorts` is declared explicitly because the chart
    passes it both to the InferencePool and to the EPP's endpoint flags;
    the adopted fleet may serve on a port other than the recipe default.
    EPP replicas above 1 make the chart add `--ha-enable-leader-election`
    (active-passive) by itself; `router.epp_ha_mode: active-active` turns
    that off. Tenant priorities become InferenceObjectives so flow control
    has something to order by.
    """
    fleet = fleet or {"mode": FLEET_MANAGED}
    env = env or {}
    router = config["router"]
    epp_doc = epp_config(config)
    namespace = env.get("namespace", "default")
    text = json.dumps(epp_doc, indent=2) + "\n"
    if config["kv"].get("precise_routing"):
        text = text.replace("${MODEL}", model or "").replace(
            "${RENDER_URL}", render_service_url(name, namespace, fleet))
    flags: dict[str, Any] = {"v": 2}
    if any(p["type"] in EXPERIMENTAL_PLUGINS for p in epp_doc["plugins"]):
        flags["allow-experimental-plugins"] = True
    if int(router.get("epp_replicas") or 1) > 1 and \
            router.get("epp_ha_mode") == "active-active":
        flags["ha-enable-leader-election"] = False
    epp = {
        "replicas": int(router.get("epp_replicas") or 1),
        "flags": flags,
        "pluginsConfigFile": EPP_CONFIG_KEY,
        "pluginsCustomConfig": {EPP_CONFIG_KEY: text},
        "resources": sizing_for("epp", env),
    }
    if env.get("control_plane_affinity"):
        epp["affinity"] = env["control_plane_affinity"]
    if env.get("control_plane_tolerations"):
        epp["tolerations"] = env["control_plane_tolerations"]
    proxy: dict[str, Any] = {
        "proxyType": "envoy",
        "mode": router.get("proxy_mode") or "sidecar",
        "args": [
            "--service-node", "envoy-sidecar", "--log-level", "warn",
            "--concurrency", "8", "--drain-strategy", "immediate",
            "--drain-time-s", "60", "-c", "/etc/envoy/envoy.yaml"
        ],
        # The chart's envoy preset names its ConfigMap a bare `envoy`, so a
        # second release in the namespace (a production router beside a
        # study arm, as found live) fails Helm's ownership check. Both the
        # ConfigMap template and the preset's volume mount read
        # `presets.envoy.configMap.name` (the volume hard-codes that path),
        # so the override must go there: naming `router.proxy.configMap`
        # alone created a second ConfigMap while the pod still mounted
        # `envoy`, which was found live when the pod was rescheduled after
        # the study release that owned `envoy` was gone.
        "presets": {"envoy": {"configMap": {
            "name": proxy_configmap_name(name)}}},
        "resources": sizing_for("proxy", env),
    }
    if router.get("proxy_replicas"):
        proxy["replicas"] = int(router["proxy_replicas"])
    values: dict[str, Any] = {
        "router": {
            "extraServicePorts": [{
                "name": "http",
                "port": pins.ROUTER_SERVICE_PORT,
                "protocol": "TCP",
                "targetPort": pins.ROUTER_SIDECAR_PORT,
            }],
            "epp": epp,
            "inferencePool": {"failureMode": "FailOpen"},
            "modelServers": {
                "protocol": "http",
                "matchLabels": modelserver_selector(name, fleet),
                "targetPorts": [{"number": modelserver_port(fleet)}],
            },
            "proxy": proxy,
        }
    }
    if config["topology"].get("mode") == "disaggregated":
        # Both pools share the release label; the role label tells the
        # prefill/decode filters apart.
        values["router"]["modelServers"]["matchLabels"] = {
            pins.RELEASE_LABEL: name, pins.LLMD_SERVING_LABEL: "true"}
    if router.get("predicted_latency"):
        values["router"]["latencyPredictor"] = {"enabled": True}
    # An InferenceObjective is a namespaced object named here, so a tenant
    # name alone collides as soon as two releases share a namespace (a
    # study arm beside the production router, found live); the bench Job
    # sends the same release-scoped name in the objective header.
    objectives = [{"name": objective_name(name, t["name"]),
                   "priority": int(t.get("priority") or 0)}
                  for t in (tenants or []) if t.get("priority") is not None]
    if objectives:
        values["router"]["inferenceObjectives"] = objectives
    return values


def objective_name(name: str, tenant: str) -> str:
    """The release-scoped InferenceObjective for a tenant."""
    return f"{name.lower()[:40]}-{str(tenant).lower()}"


def router_service(name: str, namespace: str) -> str:
    """The in-cluster host benchmarks send to: the standalone chart's EPP
    Service, with `extraServicePorts` exposing the Envoy sidecar on 80."""
    return f"{epp_deployment(name)}.{namespace}.svc.cluster.local"


def proxy_configmap_name(name: str) -> str:
    """The release's Envoy ConfigMap; per-release so two routers (a study
    arm and the production release) can share a namespace."""
    return f"{epp_deployment(name)}-envoy"


def exposure_service(name: str) -> str:
    """The Service a production release adds in front of the router."""
    return f"{name.lower()[:40]}-ingress"


def exposure_manifests(name: str, namespace: str, mode: str,
                       production: bool = True) -> dict[str, Any]:
    """A front-door Service for the router, in one of `EXPOSURE_MODES`.

    It selects the chart's EPP pods -- Envoy is their sidecar, listening on
    `ROUTER_SIDECAR_PORT` -- so it inherits the chart's readiness: under
    leader election only the leader is Ready, and the Service follows it.
    The internal passthrough LB is GKE's annotation-driven Service type;
    nothing here needs the Gateway API.
    """
    if mode not in EXPOSURE_MODES:
        raise DeployError(
            f"exposure must be one of {', '.join(EXPOSURE_MODES)}",
            error_code="INVALID_FIELD")
    metadata: dict[str, Any] = {
        "name": exposure_service(name),
        "namespace": namespace,
        "labels": {
            **release_labels(name, production),
            "tpu-coworker.llmd/exposure": mode,
        },
    }
    spec: dict[str, Any] = {
        "selector": {"llm-d-router-gateway": epp_deployment(name)},
        "ports": [{
            "name": "http",
            "port": pins.ROUTER_SERVICE_PORT,
            "targetPort": pins.ROUTER_SIDECAR_PORT,
            "protocol": "TCP",
        }],
        "type": "ClusterIP",
    }
    if mode == EXPOSURE_INTERNAL_LB:
        metadata["annotations"] = {INTERNAL_LB_ANNOTATION: "Internal"}
        spec["type"] = "LoadBalancer"
        spec["externalTrafficPolicy"] = "Cluster"
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [{
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": spec,
        }],
    }


def epp_ha(config: dict[str, Any]) -> dict[str, Any]:
    """How many EPP pods the chart runs and how many must be Ready.

    With `epp_replicas > 1` the chart enables leader election unless told
    `active-active`; under leader election the standby deliberately fails
    its readiness probe so the Service only routes to the leader, which is
    why `min_ready` is 1 there rather than the replica count
    (https://github.com/llm-d/llm-d-router/tree/v0.11.0/config/charts).
    """
    router = config["router"]
    replicas = int(router.get("epp_replicas") or 1)
    mode = router.get("epp_ha_mode")
    if replicas > 1 and mode is None:
        mode = "active-passive"
    if replicas <= 1:
        mode = None
    min_ready = replicas if mode in (None, "active-active") else 1
    return {"replicas": replicas, "mode": mode, "min_ready": min_ready}


def modelserver_service(name: str, namespace: str,
                        fleet: dict[str, Any]) -> tuple[str, int]:
    """Host and port of the Service in front of the model servers."""
    if fleet["mode"] == FLEET_ADOPTED:
        return (f"{fleet['service']}.{namespace}.svc.cluster.local",
                modelserver_port(fleet))
    return (f"{name}-modelserver.{namespace}.svc.cluster.local",
            pins.MODEL_SERVER_PORT)


def bench_base_url(name: str, namespace: str, config: dict[str, Any],
                   fleet: dict[str, Any]) -> str:
    """Where the benchmark sends: the router, or the servers' own Service
    under the passthrough control."""
    if config["router"]["profile"] == PASSTHROUGH_PROFILE:
        host, port = modelserver_service(name, namespace, fleet)
        return f"http://{host}:{port}"
    return f"http://{router_service(name, namespace)}:{pins.ROUTER_SERVICE_PORT}"


# -- monitoring ----------------------------------------------------------------


# The target label the model-server PodMonitoring copies the pod's
# `llm-d.ai/role` into; the collector selects on it per pool.
ROLE_TARGET_LABEL = "llmd_role"


def modelserver_monitoring_name(name: str) -> str:
    """Name of the release's model-server PodMonitoring.

    Managed Prometheus labels every scraped sample with `job=<PodMonitoring
    name>`, so the collector filters on this exact string; defining it once
    keeps the renderer and the collector from drifting apart.
    """
    return f"{name}-modelserver"


def monitoring_manifests(name: str, production: bool,
                         fleet: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
    """Managed Prometheus PodMonitoring for vLLM and the EPP.

    Reading an adopted fleet's metrics is the one thing a release does to
    it, and a PodMonitoring is a read: it changes nothing in the pods.
    """
    fleet = fleet or {"mode": FLEET_MANAGED}
    labels = release_labels(name, production)
    epp_label_key, epp_label_value = epp_selector(name).split("=", 1)
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "monitoring.googleapis.com/v1",
                "kind": "PodMonitoring",
                "metadata": {"name": modelserver_monitoring_name(name),
                             "labels": labels},
                "spec": {
                    "selector": {
                        "matchLabels": modelserver_selector(name, fleet)
                    },
                    "endpoints": [{
                        "port": modelserver_port(fleet),
                        "path": "/metrics",
                        "interval": "15s"
                    }],
                    # Carry the pod's role label into every sample so the
                    # collector can attribute prefill and decode pools
                    # separately (the xPyD search moves the axis that is
                    # bound). Adopted pods without the label simply get
                    # an empty `role`.
                    "targetLabels": {
                        "fromPod": [{"from": pins.LLMD_ROLE_LABEL,
                                     "to": ROLE_TARGET_LABEL}],
                    },
                },
            },
            {
                "apiVersion": "monitoring.googleapis.com/v1",
                "kind": "PodMonitoring",
                "metadata": {"name": f"{name}-epp", "labels": labels},
                "spec": {
                    "selector": {
                        "matchLabels": {epp_label_key: epp_label_value}
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


# Guide-sized requests for the CPU parts of a release; `llmd.sizing` in the
# environment overrides any of them per part.
DEFAULT_SIZING: dict[str, dict[str, dict[str, str]]] = {
    "bench": {"requests": {"cpu": "4", "memory": "8Gi"},
              "limits": {"memory": "16Gi"}},
    "epp": {"requests": {"cpu": "4", "memory": "8Gi"},
            "limits": {"memory": "16Gi"}},
    "proxy": {"requests": {"cpu": "4", "memory": "8Gi"},
              "limits": {"memory": "16Gi"}},
}


def sizing_for(part: str, env: dict[str, Any] | None) -> dict[str, Any]:
    """Resource block for `part`, the environment's override winning."""
    base = copy.deepcopy(DEFAULT_SIZING[part])
    override = ((env or {}).get("sizing") or {}).get(part) or {}
    for key in ("requests", "limits"):
        if key in override:
            base[key] = {k: str(v) for k, v in dict(override[key]).items()}
    return base


def bench_job(name: str, tenant: str, stage: str, env: dict[str, Any],
              production: bool, suffix: str = "") -> dict[str, Any]:
    """A Job that runs inference-perf and prints every report it wrote.

    Reports are echoed between markers rather than copied out, so `bench`
    needs only the Job's logs -- no exec, no cp, no shared volume -- and the
    Job stays a plain Kubernetes object the reaper can delete.
    """
    job_name = f"{name}-bench-{tenant}-{stage}{suffix}"
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
                    **({"nodeSelector": dict(env["cpu_node_selector"])}
                       if env.get("cpu_node_selector") else {}),
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
                        "resources": sizing_for("bench", env),
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
                    config: dict[str, Any], production: bool,
                    suffix: str = "") -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{name}-bench-{tenant}-{stage}{suffix}",
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

    What is written depends on the fleet and the router profile: an
    adopted fleet gets no model-server manifests (its pods are not the
    release's to apply), and the passthrough profile gets no router values
    (there is no EPP). Files a previous render of this directory wrote for
    a part that is now absent are removed, so release.json and the
    directory never disagree.
    """
    if not NAME_RE.match(name):
        raise DeployError(
            f"release name {name!r} must be a DNS label of at most 40"
            " characters", error_code="INVALID_FIELD")
    config = normalize_config(config)
    validate_recipe(recipe)
    fleet = fleet_spec(recipe)
    validate_config(config, fleet)
    validate_recipe_config(recipe, config)
    namespace = env["namespace"]
    chips = chips_per_replica(recipe, config)
    replicas = int(config["topology"]["decode"]["replicas"])
    adopted = fleet["mode"] == FLEET_ADOPTED
    router_installed = config["router"]["profile"] != PASSTHROUGH_PROFILE

    out_dir = pathlib.Path(out_dir)
    files: dict[str, Any] = {}
    if adopted:
        (out_dir / MODELSERVER_FILE).unlink(missing_ok=True)
        files["modelserver"] = None
    else:
        _write(out_dir / MODELSERVER_FILE,
               modelserver_manifests(name, recipe, config, env, production))
        files["modelserver"] = MODELSERVER_FILE
    _write(out_dir / MONITORING_FILE,
           monitoring_manifests(name, production, fleet))
    files["monitoring"] = MONITORING_FILE
    if router_installed:
        _write(out_dir / ROUTER_VALUES_FILE,
               router_values(name, config, fleet, env=env,
                             model=str(recipe["model"]),
                             tenants=profile["tenants"]))
        _write(out_dir / EPP_CONFIG_FILE, epp_config(config))
        files["router_values"] = ROUTER_VALUES_FILE
        files["epp_config"] = EPP_CONFIG_FILE
    else:
        (out_dir / ROUTER_VALUES_FILE).unlink(missing_ok=True)
        (out_dir / EPP_CONFIG_FILE).unlink(missing_ok=True)
        files["router_values"] = None
        files["epp_config"] = None
    exposure_mode = str(env.get("exposure") or EXPOSURE_CLUSTER_IP)
    if production and router_installed:
        _write(out_dir / EXPOSURE_FILE,
               exposure_manifests(name, namespace, exposure_mode, production))
        files["exposure"] = EXPOSURE_FILE
    else:
        (out_dir / EXPOSURE_FILE).unlink(missing_ok=True)
        files["exposure"] = None

    bench_files: dict[str, dict[str, str]] = {}
    base_url = bench_base_url(name, namespace, config, fleet)
    for tenant in profile["tenants"]:
        for stage in BENCH_STAGES:
            ip_config = workload.render_inference_perf(
                profile,
                tenant,
                stage=stage,
                base_url=base_url,
                model_name=recipe["model"],
                tokenizer=recipe.get("tokenizer"),
                duration_s=int(env.get("slo_stage_duration_s")
                               or workload.DEFAULT_SLO_STAGE_DURATION_S),
                report_dir=BENCH_REPORT_DIR,
                objective=(objective_name(name, tenant["name"])
                           if router_installed else None),
                sweep_stage_s=int(env.get("sweep_stage_duration_s")
                                  or workload.DEFAULT_SWEEP_STAGE_DURATION_S))
            key = f"{tenant['name']}.{stage}"
            # A rate one load-generator pod cannot carry is split across
            # shards that `bench` runs together and pools; one shard is
            # the plain case and keeps the old file names.
            shards = workload.loadgen_shards(workload.peak_rate(ip_config))
            pieces = workload.shard_load(ip_config, shards)
            entries = []
            for idx, piece in enumerate(pieces):
                # Shard 0 keeps the plain name so a one-shard release and
                # the first shard of a split read the same way.
                suffix = "" if idx == 0 else f"-s{idx}"
                cm_path = f"{BENCH_DIR}/{key}{suffix}.configmap.json"
                job_path = f"{BENCH_DIR}/{key}{suffix}.job.json"
                _write(out_dir / cm_path,
                       bench_configmap(name, tenant["name"], stage, piece,
                                       production, suffix=suffix))
                _write(out_dir / job_path,
                       bench_job(name, tenant["name"], stage, env,
                                 production, suffix=suffix))
                entries.append({"configmap": cm_path, "job": job_path})
            bench_files[key] = {"configmap": entries[0]["configmap"],
                                "job": entries[0]["job"],
                                "shards": entries}

    _write(out_dir / CONFIG_FILE, config)
    _write(out_dir / RECIPE_FILE, recipe)
    _write(out_dir / PROFILE_FILE, profile)

    ms_host, ms_port = modelserver_service(name, namespace, fleet)
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
        "chips": {"per_replica": chips, "total": total_chips(recipe, config)},
        "fleet": {
            **fleet,
            "service_host": ms_host,
            "service_port": ms_port,
        },
        "config_digest": config_digest(config),
        "router": {
            "installed": router_installed,
            "profile": config["router"]["profile"],
            "helm_release": name,
            "chart": pins.ROUTER_STANDALONE_CHART,
            "chart_version": pins.ROUTER_CHART_VERSION,
            "epp_image": f"{pins.ROUTER_EPP_IMAGE}:{pins.ROUTER_EPP_VERSION}",
            "epp_deployment": epp_deployment(name),
            "epp_selector": epp_selector(name),
            "service": router_service(name, namespace),
            "port": pins.ROUTER_SERVICE_PORT,
            "epp_ha": epp_ha(config),
        },
        "exposure": ({
            "mode": exposure_mode,
            "service": exposure_service(name),
            "host": f"{exposure_service(name)}.{namespace}.svc.cluster.local",
            "port": pins.ROUTER_SERVICE_PORT,
        } if files["exposure"] else None),
        "bench": {"base_url": base_url},
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
            **files,
            "config": CONFIG_FILE,
            "recipe": RECIPE_FILE,
            "profile": PROFILE_FILE,
            "bench": bench_files,
        },
        "deployments": [] if adopted else pool_names(name, config),
        "pools": [] if adopted else pool_records(name, config),
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
