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
import pathlib
import re
from typing import Any

from . import pins
from .kube import DeployError

CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "workload_catalog"

PRIMARY_METRICS = ("goodput_at_slo_per_chip", "cost_per_1M_tokens_at_slo")
DEFAULT_PRIMARY_METRIC = "goodput_at_slo_per_chip"

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


def resolve_workload(spec: dict[str, Any],
                     spec_dir: pathlib.Path | None = None) -> dict[str, Any]:
    """Resolve a workload spec into the profile every later step reads.

    Exactly one of `workload.catalog_entry` and
    `workload.inference_perf_config` must be set. `workload.trace` is
    reserved: turning a request log into a replay config is a later
    milestone, and refusing it here is better than silently benchmarking
    something else.
    """
    workload = spec.get("workload") or {}
    sources = [
        k for k in ("catalog_entry", "inference_perf_config", "trace")
        if workload.get(k)
    ]
    if len(sources) != 1:
        raise DeployError(
            "workload must set exactly one of catalog_entry,"
            " inference_perf_config, trace",
            error_code="INVALID_FIELD")
    source_kind = sources[0]
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
    sweep = rate.get("sweep") or [
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
        },
        "tenants": tenants,
        "objective": objective,
        "availability": availability,
        "budget": budget,
        "constraints": dict(spec.get("constraints") or {}),
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


def render_inference_perf(profile: dict[str, Any],
                          tenant: dict[str, Any],
                          *,
                          stage: str,
                          base_url: str,
                          model_name: str,
                          duration_s: int = DEFAULT_SLO_STAGE_DURATION_S,
                          report_dir: str = "/reports") -> dict[str, Any]:
    """The inference-perf config one tenant's benchmark Job runs.

    `sweep` keeps the source's own load stages (the catalog's are concurrency
    ladders) so the knee is found the way the catalog authors measured it.
    `slo` replaces them with an open-loop Poisson arrival at this tenant's
    share of `target_qps`, which is the load the SLO is a promise about.
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

    api = dict(config.get("api") or {})
    if "streaming" in profile.get("overrides", {}):
        api["streaming"] = bool(profile["overrides"]["streaming"])
    api.setdefault("type", "completion")
    config["api"] = api

    if stage == "slo":
        rate = profile["rate"]["target_qps"] * float(tenant["share"])
        config["load"] = {
            "type": "poisson",
            "stages": [{
                "rate": round(rate, 4),
                "duration": int(duration_s)
            }],
            "num_workers": max(1, int(config.get("load", {}).get(
                "num_workers", 1))),
        }
    else:
        load = dict(config.get("load") or {})
        if not load.get("stages"):
            raise DeployError(
                "workload source has no load.stages to sweep",
                error_code="INVALID_FIELD")
        config["load"] = load

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
