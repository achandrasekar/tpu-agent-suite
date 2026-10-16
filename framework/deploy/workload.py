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
"""From a workload spec to one workload profile and inference-perf configs.

The spec does not describe traffic; it names an inference-perf workload
catalog entry (or a user's own inference-perf config) and adds what the
catalog cannot know: rate, tenants with SLOs, budget and constraints. This
module resolves that into a `workload_profile.json` the planner reasons
about, and renders the per-tenant inference-perf config each benchmark Job
runs. Keeping both derivations here means the planner and the harness see
the same workload no matter which source it came from.

Catalog source: https://github.com/kubernetes-sigs/inference-perf/tree/main/workload-catalog
inference-perf config reference: https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/config.md
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import re
from typing import Any

from . import pins
from .kube import DeployError

CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "workload_catalog"

PRIMARY_METRICS = ("goodput_at_slo_per_chip", "cost_per_1M_tokens_at_slo",
                   "total_tokens_per_s_per_chip",
                   "output_tokens_per_s_per_chip")
DEFAULT_PRIMARY_METRIC = "goodput_at_slo_per_chip"
# The InferenceX AgentX leaderboard ranks on total tok/s per chip with
# output tok/s per chip alongside; a closed-loop concurrency workload has
# no goodput-at-rate to rank on.
AGENTX_PRIMARY_METRICS = ("total_tokens_per_s_per_chip",
                          "output_tokens_per_s_per_chip")
AGENTX_DEFAULT_PRIMARY_METRIC = "total_tokens_per_s_per_chip"
AGENTX_SHAPE_KIND = "agentx"
AGENTX_TENANT = "agentx"

# SLO keys the spec may carry per tenant. inference-perf reports these
# latencies per stage; `bench` reads the matching percentile back out.
SLO_KEY = re.compile(r"^(ttft|tpot|itl|e2e)_p(50|90|99)_ms$")
# Which inference-perf latency field each SLO metric reads.
SLO_METRIC_FIELD = {
    "ttft": "time_to_first_token",
    "tpot": "time_per_output_token",
    "itl": "inter_token_latency",
    "e2e": "request_latency",
}

DEFAULT_SWEEP_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_SLO_STAGE_DURATION_S = 300
# A rate-ladder stage: long enough for a multi-turn replay to reach steady
# state at the new rate, short enough that a five-rung sweep fits an arm.
DEFAULT_SWEEP_STAGE_DURATION_S = 120
DEFAULT_ERROR_RATE_MAX = 0.001


# -- catalog ---------------------------------------------------------------


def catalog_index() -> dict[str, Any]:
    with open(CATALOG_DIR / "CATALOG.json", encoding="utf-8") as handle:
        return json.load(handle)


def catalog_entries() -> list[str]:
    return list(catalog_index().get("entries", []))


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Parse a YAML document; pyyaml is a declared host dependency."""
    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - host install guarantees it
        raise DeployError("pyyaml is needed to read inference-perf configs",
                          error_code="TOOL_MISSING",
                          detail="pip install pyyaml") from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise DeployError(f"{path} is not a mapping",
                          error_code="INVALID_FIELD")
    return data


def load_catalog_entry(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The entry's harness-agnostic config.json and its inference-perf.yaml."""
    if name not in catalog_entries():
        raise DeployError(
            f"unknown workload catalog entry {name!r}; one of"
            f" {', '.join(catalog_entries())}",
            error_code="UNKNOWN_CATALOG_ENTRY")
    folder = CATALOG_DIR / name
    with open(folder / "config.json", encoding="utf-8") as handle:
        shape = json.load(handle)
    return shape, _load_yaml(folder / "inference-perf.yaml")


def catalog_summary(name: str) -> str:
    """One line the coworker echoes so the workload choice is visible."""
    shape, _ = load_catalog_entry(name)
    meta = shape.get("metadata", {})
    isl = shape.get("input_sequence_length", {}).get("mean")
    osl = shape.get("output_sequence_length", {}).get("mean")
    turns = shape.get("number_of_turns", {}).get("mean")
    opts = ", ".join(meta.get("suitable_optimizations", [])) or "none listed"
    return (f"{name}: ISL~{isl} OSL~{osl} turns~{turns}"
            f" prefill:decode={meta.get('prefill_decode_ratio')}"
            f"; suits {opts}")


# -- spec -> profile ---------------------------------------------------------


def _shape_from_custom(config: dict[str, Any]) -> dict[str, Any]:
    """The catalog's statistics, as far as a custom config states them.

    A user config may be synthetic, shared-prefix or trace-replay; only what
    it says is recorded, and `metadata` is left empty so the planner treats
    reuse as unknown rather than assumed.
    """
    data = config.get("data", {}) or {}
    shape: dict[str, Any] = {"data_type": data.get("type")}
    for key in ("input_distribution", "output_distribution"):
        if key in data:
            shape[key] = data[key]
    if "shared_prefix" in data:
        shape["shared_prefix"] = data["shared_prefix"]
    if "conversation_replay" in data:
        shape["conversation_replay"] = data["conversation_replay"]
        shape["multi_turn"] = True
    shape["metadata"] = {}
    return shape


def _validate_tenants(tenants: list[dict[str, Any]],
                      default_slo: dict[str, Any]) -> list[dict[str, Any]]:
    if not tenants:
        tenants = [{"name": "default", "share": 1.0, "priority": 0}]
    out = []
    total = 0.0
    names: set[str] = set()
    for tenant in tenants:
        name = str(tenant.get("name") or "").strip()
        if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
            raise DeployError(
                f"tenant name {name!r} must be a lowercase slug",
                error_code="INVALID_FIELD")
        if name in names:
            raise DeployError(f"duplicate tenant {name!r}",
                              error_code="INVALID_FIELD")
        names.add(name)
        share = float(tenant.get("share", 1.0 if len(tenants) == 1 else 0))
        if share <= 0:
            raise DeployError(f"tenant {name!r} needs share > 0",
                              error_code="INVALID_FIELD")
        total += share
        # The user's SLO is the SLO. The catalog's `metadata.slos` stand in
        # only for a tenant that names none; merging them underneath would
        # add gates nobody pre-registered, and every one of them is a way
        # for an arm to fail.
        slo = dict(tenant.get("slo") or {}) or dict(default_slo)
        for key in slo:
            if not SLO_KEY.match(key):
                raise DeployError(
                    f"tenant {name!r}: unknown SLO key {key!r}; expected"
                    " <ttft|tpot|itl|e2e>_p<50|90|99>_ms",
                    error_code="INVALID_FIELD")
        if not slo:
            raise DeployError(
                f"tenant {name!r} has no SLO and the workload source names"
                " none; set tenancy[].slo",
                error_code="INVALID_FIELD")
        out.append({
            "name": name,
            "share": share,
            "priority": int(tenant.get("priority", 0)),
            "slo": slo,
        })
    if abs(total - 1.0) > 1e-6:
        raise DeployError(f"tenant shares sum to {total:g}, not 1.0",
                          error_code="INVALID_FIELD")
    return out


def _catalog_default_slo(shape: dict[str, Any]) -> dict[str, Any]:
    slos = (shape.get("metadata") or {}).get("slos") or {}
    return {k: v for k, v in slos.items() if SLO_KEY.match(k)}


def is_agentx(profile: dict[str, Any]) -> bool:
    """Whether a resolved profile is the closed-loop AgentX replay."""
    return (profile.get("shape") or {}).get("kind") == AGENTX_SHAPE_KIND


def concurrency_points(agentx: dict[str, Any]) -> list[int]:
    """The concurrency ladder of an AgentX spec: an explicit list, or
    `{start, step, max}` (start, start+step, ... <= max)."""
    conc = agentx.get("concurrency")
    if isinstance(conc, list):
        points = sorted({int(c) for c in conc})
    elif isinstance(conc, dict):
        start = int(conc.get("start") or 0)
        step = int(conc.get("step") or 0)
        top = int(conc.get("max") or 0)
        if start < 1 or step < 1 or top < start:
            raise DeployError(
                "workload.agentx.concurrency {start, step, max} needs"
                " start >= 1, step >= 1, max >= start",
                error_code="INVALID_FIELD")
        points = list(range(start, top + 1, step))
    else:
        raise DeployError(
            "workload.agentx.concurrency must be a list of integers or"
            " {start, step, max}", error_code="INVALID_FIELD")
    if not points or min(points) < 1:
        raise DeployError("workload.agentx.concurrency needs values >= 1",
                          error_code="INVALID_FIELD")
    return points


def _resolve_agentx(spec: dict[str, Any]) -> dict[str, Any]:
    """The profile of an AgentX spec: one tenant, a concurrency ladder,
    aiperf's scenario parameters, the leaderboard objective and gates."""
    agentx = dict((spec.get("workload") or {}).get("agentx") or {})
    points = concurrency_points(agentx)
    loader = str(agentx.get("dataset_loader") or pins.AGENTX_DATASET_LOADER)
    dataset = agentx.get("dataset") or pins.AGENTX_DATASETS.get(loader)
    if not dataset:
        raise DeployError(
            f"workload.agentx.dataset_loader {loader!r} is not one the"
            " pins know; set workload.agentx.dataset to its HF id",
            error_code="INVALID_FIELD")
    fast = bool(agentx.get("fast"))
    duration = int(agentx.get("duration_s") or (
        pins.AGENTX_FAST_DURATION_S if fast else pins.AGENTX_DURATION_S))
    warmup = int(agentx.get("warmup_requests_per_lane") or (
        pins.AGENTX_FAST_WARMUP_PER_LANE if fast else
        pins.AGENTX_WARMUP_PER_LANE))
    objective = dict(spec.get("objective") or {})
    primary = objective.get("primary_metric") or AGENTX_DEFAULT_PRIMARY_METRIC
    if primary not in AGENTX_PRIMARY_METRICS:
        raise DeployError(
            f"objective.primary_metric for an AgentX workload must be one"
            f" of {AGENTX_PRIMARY_METRICS}", error_code="INVALID_FIELD")
    objective["primary_metric"] = primary
    tenancy = list(spec.get("tenancy") or [])
    if len(tenancy) > 1:
        raise DeployError("an AgentX workload has one tenant (the replay)",
                          error_code="INVALID_FIELD")
    tenant = dict(tenancy[0]) if tenancy else {}
    tenant.setdefault("name", AGENTX_TENANT)
    tenant.setdefault("share", 1.0)
    tenant.setdefault("priority", 0)
    tenants = _validate_tenants(
        [tenant], {"ttft_p50_ms": pins.AGENTX_TTFT_P50_GATE_MS})
    availability = dict(spec.get("availability") or {})
    # InferenceX's post-run validity gate, not this suite's 0.1 % default:
    # the replay's own error accounting drops failed records first.
    availability.setdefault("error_rate_max", pins.AGENTX_FAILED_THRESHOLD)
    return {
        "name": str(spec.get("name") or "agentx"),
        "source": {"kind": AGENTX_SHAPE_KIND,
                   "scenario": str(agentx.get("scenario") or
                                   pins.AGENTX_SCENARIO),
                   "dataset_loader": loader, "dataset": str(dataset),
                   "aiperf_ref": str(agentx.get("aiperf_ref") or
                                     pins.AIPERF_REF),
                   "summary": f"InferenceX AgentX replay ({loader}),"
                              f" concurrency {points}, {duration} s/point"},
        "shape": {"kind": AGENTX_SHAPE_KIND, "data_type": "trace_replay",
                  "multi_turn": True,
                  "metadata": {"prefill_decode_ratio": "agentic",
                               "suitable_optimizations": [
                                   "pd_disaggregation", "session_affinity",
                                   "prefix_caching", "kv_offload"]}},
        "rate": {"target_qps": None, "peak_qps": None, "sweep": [],
                 "sweep_explicit": [], "mode": "closed-loop"},
        "agentx": {
            "scenario": str(agentx.get("scenario") or pins.AGENTX_SCENARIO),
            "dataset_loader": loader,
            "dataset": str(dataset),
            "aiperf_ref": str(agentx.get("aiperf_ref") or pins.AIPERF_REF),
            "concurrency": points,
            "duration_s": duration,
            "fast": fast,
            "warmup_requests_per_lane": warmup,
            "warmup_grace_s": int(agentx.get("warmup_grace_s") or
                                  pins.AGENTX_WARMUP_GRACE_S),
            "trace_idle_gap_cap_s": int(agentx.get("trace_idle_gap_cap_s")
                                        or pins.AGENTX_IDLE_GAP_CAP_S),
            "live_failed_request_threshold": float(
                agentx.get("live_failed_request_threshold") or
                pins.AGENTX_LIVE_FAILED_THRESHOLD),
            "num_dataset_entries": int(agentx.get("num_dataset_entries") or
                                       pins.AGENTX_NUM_DATASET_ENTRIES),
            "served_model_name": agentx.get("served_model_name"),
            "tokenizer": agentx.get("tokenizer"),
            "max_context_length": agentx.get("max_context_length"),
            "extra_args": [str(a) for a in (agentx.get("extra_args") or [])],
            "unsafe_override": duration < pins.AGENTX_MIN_VALID_DURATION_S
            or bool(agentx.get("unsafe_override")),
            "submission_valid": duration >= pins.AGENTX_MIN_VALID_DURATION_S
            and not agentx.get("unsafe_override"),
        },
        "tenants": tenants,
        "objective": objective,
        "availability": availability,
        "budget": dict(spec.get("budget") or {}),
        "constraints": dict(spec.get("constraints") or {}),
        "search": dict(spec.get("search") or {}),
        "overrides": {},
        "base_config": {},
    }


def resolve_workload(spec: dict[str, Any],
                     spec_dir: pathlib.Path | None = None) -> dict[str, Any]:
    """Resolve a workload spec into the profile every later step reads.

    Exactly one of `workload.catalog_entry`, `workload.inference_perf_config`
    and `workload.agentx` must be set. `workload.trace` is reserved: turning
    a request log into a replay config is a later milestone, and refusing
    it here is better than silently benchmarking something else.
    """
    workload = spec.get("workload") or {}
    sources = [
        k for k in ("catalog_entry", "inference_perf_config", "trace",
                    "agentx")
        if workload.get(k)
    ]
    if len(sources) != 1:
        raise DeployError(
            "workload must set exactly one of catalog_entry,"
            " inference_perf_config, trace, agentx",
            error_code="INVALID_FIELD")
    source_kind = sources[0]
    if source_kind == "agentx":
        return _resolve_agentx(spec)
    if source_kind == "trace":
        raise DeployError(
            "workload.trace is not supported yet; convert the trace to an"
            " inference-perf trace_replay config and pass it as"
            " workload.inference_perf_config",
            error_code="NOT_IMPLEMENTED")

    if source_kind == "catalog_entry":
        entry = str(workload["catalog_entry"])
        shape, base_config = load_catalog_entry(entry)
        source = {
            "kind": "catalog",
            "catalog_entry": entry,
            "inference_perf_ref": pins.INFERENCE_PERF_REF,
            "summary": catalog_summary(entry),
        }
    else:
        path = pathlib.Path(workload["inference_perf_config"])
        if not path.is_absolute() and spec_dir is not None:
            path = spec_dir / path
        if not path.is_file():
            raise DeployError(f"inference_perf_config {path} does not exist",
                              error_code="INVALID_FIELD")
        base_config = _load_yaml(path)
        shape = _shape_from_custom(base_config)
        source = {"kind": "custom", "path": str(path)}

    overrides = workload.get("overrides") or {}
    rate = dict(overrides.get("rate") or {})
    target_qps = rate.get("target_qps")
    if target_qps is None or float(target_qps) <= 0:
        raise DeployError(
            "workload.overrides.rate.target_qps is required and must be > 0",
            error_code="INVALID_FIELD")
    target_qps = float(target_qps)
    peak_qps = float(rate.get("peak_qps") or target_qps)
    sweep_explicit = [float(x) for x in (rate.get("sweep") or [])]
    sweep = sweep_explicit or [
        round(target_qps * f, 3) for f in DEFAULT_SWEEP_FRACTIONS
    ]

    objective = dict(spec.get("objective") or {})
    primary = objective.get("primary_metric") or DEFAULT_PRIMARY_METRIC
    if primary not in PRIMARY_METRICS:
        raise DeployError(
            f"objective.primary_metric must be one of {PRIMARY_METRICS}",
            error_code="INVALID_FIELD")
    budget = dict(spec.get("budget") or {})
    if primary == "cost_per_1M_tokens_at_slo" and not budget.get(
            "chip_hour_cost"):
        raise DeployError(
            "objective.primary_metric cost_per_1M_tokens_at_slo needs"
            " budget.chip_hour_cost",
            error_code="INVALID_FIELD")
    objective["primary_metric"] = primary

    availability = dict(spec.get("availability") or {})
    availability.setdefault("error_rate_max", DEFAULT_ERROR_RATE_MAX)

    tenants = _validate_tenants(list(spec.get("tenancy") or []),
                                _catalog_default_slo(shape))
    name = str(spec.get("name") or "workload")

    return {
        "name": name,
        "source": source,
        "shape": shape,
        "rate": {
            "target_qps": target_qps,
            "peak_qps": peak_qps,
            "sweep": [float(x) for x in sweep],
            # Only an explicit list turns the sweep stage into a rate
            # ladder; the derived one is informational and the catalog's
            # own stages are kept.
            "sweep_explicit": sweep_explicit,
        },
        "tenants": tenants,
        "objective": objective,
        "availability": availability,
        "budget": budget,
        "constraints": dict(spec.get("constraints") or {}),
        # How far past the fleet the search may go (engine escalation).
        "search": dict(spec.get("search") or {}),
        "overrides": {
            k: v for k, v in overrides.items() if k != "rate"
        },
        "base_config": base_config,
    }


# -- profile -> inference-perf config ---------------------------------------


def _goodput_constraints(slo: dict[str, Any]) -> dict[str, float]:
    """inference-perf's goodput block: p-agnostic per-request bounds in s.

    The harness judges each request against one bound per metric, so the
    tightest percentile named for a metric is the bound used.
    """
    bounds: dict[str, float] = {}
    for key, value in slo.items():
        match = SLO_KEY.match(key)
        if not match:
            continue
        metric = match.group(1)
        field = {
            "ttft": "ttft",
            "tpot": "tpot",
            "itl": "itl",
            "e2e": "request_latency",
        }[metric]
        seconds = float(value) / 1000.0
        bounds[field] = min(bounds.get(field, seconds), seconds)
    return bounds


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """`extra` over `base`, recursing into mappings; lists are replaced."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def render_inference_perf(profile: dict[str, Any],
                          tenant: dict[str, Any],
                          *,
                          stage: str,
                          base_url: str,
                          model_name: str,
                          duration_s: int = DEFAULT_SLO_STAGE_DURATION_S,
                          report_dir: str = "/reports",
                          objective: str | None = None,
                          sweep_stage_s: int = DEFAULT_SWEEP_STAGE_DURATION_S,
                          tokenizer: str | None = None,
                          ) -> dict[str, Any]:
    """The inference-perf config one tenant's benchmark Job runs.

    `sweep` keeps the source's own load stages (the catalog's are concurrency
    ladders) so the knee is found the way the catalog authors measured it.
    `slo` replaces them with an open-loop Poisson arrival at this tenant's
    share of `target_qps`, which is the load the SLO is a promise about.

    The load generator must never be the bottleneck it is measuring. With
    the pinned inference-perf, requests in flight are capped at
    `num_workers x worker_max_concurrency` (1 x 100 by default), which an
    open-loop rate of 20 rps with 5 s streaming responses already exceeds;
    the symptom is a growing `load_summary.schedule_delay` and an achieved
    rate below the requested one. Both are sized from the rate here, and a
    source config may still raise them.

    `objective` sends `x-llm-d-inference-objective: <objective>` -- the
    release-scoped InferenceObjective the router created for this tenant --
    so the router's flow control places the tenant in the band its priority
    maps to; without a router the header is inert, so it is only set when
    given.
    """
    if stage not in ("sweep", "slo"):
        raise DeployError("stage must be sweep or slo",
                          error_code="INVALID_FIELD")
    config = copy.deepcopy(profile["base_config"])
    server = dict(config.get("server") or {})
    server.update({
        "type": "vllm",
        "model_name": model_name,
        "base_url": base_url
    })
    config["server"] = server
    # inference-perf counts tokens with a Hugging Face tokenizer named
    # independently of the served model; a recipe whose model name is
    # not a hub id (a simulated fleet, a private mirror) names one.
    if tokenizer:
        tok = dict(config.get("tokenizer") or {})
        tok["pretrained_model_name_or_path"] = str(tokenizer)
        config["tokenizer"] = tok

    api = dict(config.get("api") or {})
    if "streaming" in profile.get("overrides", {}):
        api["streaming"] = bool(profile["overrides"]["streaming"])
    api.setdefault("type", "completion")
    if objective:
        headers = dict(api.get("headers") or {})
        headers.setdefault(OBJECTIVE_HEADER, str(objective))
        api["headers"] = headers
    config["api"] = api

    data_overrides = (profile.get("overrides") or {}).get("data")
    if data_overrides:
        config["data"] = _deep_merge(dict(config.get("data") or {}),
                                     data_overrides)

    if stage == "slo":
        rate = profile["rate"]["target_qps"] * float(tenant["share"])
        source_load = dict(config.get("load") or {})
        workers, concurrency = loadgen_sizing(rate)
        config["load"] = {
            "type": "poisson",
            "stages": [{
                "rate": round(rate, 4),
                "duration": int(duration_s)
            }],
            "num_workers": max(workers, int(source_load.get("num_workers")
                                            or 1)),
            "worker_max_concurrency": max(
                concurrency, int(source_load.get("worker_max_concurrency")
                                 or 0)),
        }
    else:
        load = dict(config.get("load") or {})
        explicit = (profile.get("rate") or {}).get("sweep_explicit")
        if explicit:
            # The spec named the rates: an open-loop ladder, each stage long
            # enough for the multi-turn replay to settle, sized for its top
            # rate so the client never shapes the knee.
            top = max(float(r) for r in explicit) * float(tenant["share"])
            workers, concurrency = loadgen_sizing(top)
            load = {
                "type": "poisson",
                "stages": [{
                    "rate": round(float(r) * float(tenant["share"]), 4),
                    "duration": int(sweep_stage_s),
                } for r in explicit],
                "num_workers": max(workers, int(load.get("num_workers")
                                                or 1)),
                "worker_max_concurrency": max(
                    concurrency, int(load.get("worker_max_concurrency")
                                     or 0)),
            }
        if not load.get("stages"):
            raise DeployError(
                "workload source has no load.stages to sweep",
                error_code="INVALID_FIELD")
        config["load"] = load
    config["load"] = harden_load(config["load"])

    report = dict(config.get("report") or {})
    report["request_lifecycle"] = {
        "summary": True,
        "per_stage": True,
        "per_request": False
    }
    constraints = _goodput_constraints(tenant["slo"])
    if constraints:
        report["goodput"] = {"constraints": constraints}
    config["report"] = report
    config["storage"] = {
        "local_storage": {
            "path": report_dir,
            "report_file_prefix": f"{tenant['name']}-{stage}",
        }
    }
    config.pop("metrics", None)  # fleet metrics come from `deploy collect`
    return config


# Header the llm-d router reads to pick a tenant's InferenceObjective
# (https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/flow-control.md).
OBJECTIVE_HEADER = "x-llm-d-inference-objective"

# Load-generator sizing. The in-flight ceiling (`num_workers x
# worker_max_concurrency`) is what throttles an open-loop rate; worker
# count matters less and is kept low on purpose. Each worker is an aiohttp
# process with its own connection pool, and a pool whose connections idle
# past the server's keep-alive timeout reuses stale sockets: eight workers
# at 2 rps each produced ~0.5% `ServerDisconnected` against the same
# fleet that one worker at 20 rps drove with zero failures. So one worker
# per ~10 rps, and the concurrency sized to hold 30 s of arrivals open.
LOADGEN_RPS_PER_WORKER = 10.0
LOADGEN_MAX_WORKERS = 8
LOADGEN_INFLIGHT_SECONDS = 30.0
LOADGEN_MIN_CONCURRENCY = 100


def loadgen_sizing(rate: float) -> tuple[int, int]:
    """(num_workers, worker_max_concurrency) that keep `rate` client-free.

    Workers scale with the rate; the per-worker concurrency is sized so the
    whole pool can hold `rate x LOADGEN_INFLIGHT_SECONDS` requests open.
    """
    workers = max(1, min(LOADGEN_MAX_WORKERS,
                         math.ceil(rate / LOADGEN_RPS_PER_WORKER)))
    inflight = math.ceil(rate * LOADGEN_INFLIGHT_SECONDS)
    concurrency = max(LOADGEN_MIN_CONCURRENCY,
                      math.ceil(inflight / workers))
    return workers, concurrency


# One load-generator pod carries at most LOADGEN_MAX_WORKERS workers at
# LOADGEN_RPS_PER_WORKER each; a rate above that is split across pods so
# the client is never the knee it is measuring.
LOADGEN_RPS_PER_POD = LOADGEN_RPS_PER_WORKER * LOADGEN_MAX_WORKERS


def loadgen_shards(rate: float) -> int:
    """How many load-generator pods an open-loop `rate` needs."""
    return max(1, math.ceil(float(rate) / LOADGEN_RPS_PER_POD))


def peak_rate(config: dict[str, Any]) -> float:
    """The highest open-loop rate in a config's stages (0 for closed loop)."""
    load = config.get("load") or {}
    rates = [float(s.get("rate") or 0) for s in (load.get("stages") or [])]
    return max(rates) if rates else 0.0


def shard_load(config: dict[str, Any], shards: int) -> list[dict[str, Any]]:
    """Split one inference-perf config into `shards` equal-rate configs.

    Open-loop stages get `rate / shards` each and a worker pool sized for
    that share; closed-loop stages get `concurrency_level / shards`
    (ceiling, never below one). Every other field is shared, so the shards
    are one measurement taken from several pods, and `bench` pools them.
    """
    shards = max(1, int(shards))
    if shards == 1:
        return [copy.deepcopy(config)]
    out: list[dict[str, Any]] = []
    for _ in range(shards):
        piece = copy.deepcopy(config)
        load = dict(piece.get("load") or {})
        stages = []
        for stage in load.get("stages") or []:
            st = dict(stage)
            if st.get("rate") is not None:
                st["rate"] = round(float(st["rate"]) / shards, 4)
            if st.get("concurrency_level") is not None:
                st["concurrency_level"] = max(
                    1, math.ceil(int(st["concurrency_level"]) / shards))
            stages.append(st)
        load["stages"] = stages
        top = max([float(s.get("rate") or 0) for s in stages] or [0.0])
        if top > 0:
            workers, concurrency = loadgen_sizing(top)
            load["num_workers"] = workers
            load["worker_max_concurrency"] = concurrency
        piece["load"] = load
        out.append(piece)
    return out


# Request-level hardening. inference-perf retries only faults raised before
# response headers arrive (a connection refused, reset or dropped from the
# pool), never anything after first byte, so a retry cannot mix two
# measurements; that is exactly the stale keep-alive case above. The timeout
# bounds a half-open socket that would otherwise hold a stage open past its
# drain, and the grace bounds the drain itself (default 120 s in the pinned
# release) so report generation always runs.
LOADGEN_REQUEST_TIMEOUT_S = 120.0
LOADGEN_REQUEST_RETRIES = 2
LOADGEN_TEARDOWN_GRACE_S = 60.0


def harden_load(load: dict[str, Any]) -> dict[str, Any]:
    """Fill the request-level safety fields a source config left unset."""
    out = dict(load)
    out.setdefault("request_timeout", LOADGEN_REQUEST_TIMEOUT_S)
    out.setdefault("request_retries", LOADGEN_REQUEST_RETRIES)
    out.setdefault("stage_teardown_grace_seconds", LOADGEN_TEARDOWN_GRACE_S)
    return out


def slo_targets(tenant: dict[str, Any]) -> list[dict[str, Any]]:
    """Each SLO as (metric field, percentile, bound ms) for `bench` to check."""
    out = []
    for key, value in tenant["slo"].items():
        match = SLO_KEY.match(key)
        if not match:
            continue
        out.append({
            "key": key,
            "field": SLO_METRIC_FIELD[match.group(1)],
            "percentile": f"p{match.group(2)}",
            "bound_ms": float(value),
        })
    return out
