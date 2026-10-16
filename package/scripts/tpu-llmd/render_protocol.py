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
"""Turn a workload spec and a serving recipe into a study protocol.

The protocol is pre-registered at `study start` and never edited by a
round, so everything a verdict depends on -- the primary metric and its
direction, the per-tenant SLO gates, replicates, noise floor and stopping
rule -- is derived here, once, from the spec the user approved. What the
framework's protocol schema has no field for (inner sweep budget, chip-hour
bound, timeouts) goes in a companion `llmd-study.json` beside it.

Usage:
    render_protocol.py --workload-spec SPEC --recipe RECIPE --out DIR
                       [--mode approve|auto] [--replicates 3]
                       [--noise-floor-pct 5] [--max-rounds 12]
                       [--plateau-rounds 3] [--inner-sweep-budget 8]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve()
_SHIPPED_RUNTIME = _HERE.parents[2] / "runtime"
_REPO_NEXUS = _HERE.parents[3]
for _candidate in (_SHIPPED_RUNTIME, _REPO_NEXUS):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    from framework.deploy import render, workload  # noqa: E402
except ImportError:  # shipped runtime: `deploy` is top-level
    from deploy import render, workload  # type: ignore  # noqa: E402

METRIC_UNITS = {
    "goodput_at_slo_per_chip": ("req/s/chip", "maximize"),
    "cost_per_1M_tokens_at_slo": ("usd/1M-tokens", "minimize"),
    # AgentX (InferenceX): the leaderboard ranks total tokens/s per
    # physical chip; output tok/s/chip is the secondary.
    "total_tokens_per_s_per_chip": ("tok/s/chip", "maximize"),
    "output_tokens_per_s_per_chip": ("tok/s/chip", "maximize"),
}

DEFAULTS = {
    "replicates": 3,
    "noise_floor_pct": 5.0,
    "max_rounds": 12,
    "plateau_rounds": 3,
    "inner_sweep_budget": 8,
    "startup_timeout_s": 1800,
    "benchmark_duration_s": 300,
    "bench_timeout_s": 3600,
}


def build_protocol(profile: dict[str, Any], recipe: dict[str, Any], *,
                   mode: str, replicates: int, noise_floor_pct: float,
                   max_rounds: int, plateau_rounds: int) -> dict[str, Any]:
    """A study-protocol.json document for this workload and recipe."""
    primary = profile["objective"]["primary_metric"]
    unit, direction = METRIC_UNITS[primary]
    accel = recipe["accelerator"]
    gates: list[dict[str, Any]] = []
    for tenant in profile["tenants"]:
        for target in workload.slo_targets(tenant):
            gates.append({
                "name": f"{tenant['name']}.{target['key']}",
                "metric": target["key"],
                "comparator": "lte",
                "bound": target["bound_ms"],
                "unit": "ms",
                "same_execution": True,
            })
        gates.append({
            "name": f"{tenant['name']}.error_rate",
            "metric": "error_rate",
            "comparator": "lte",
            "bound": float(profile["availability"]["error_rate_max"]),
            "unit": "ratio",
            "same_execution": True,
        })
    max_chips = (profile.get("budget") or {}).get("max_chips")
    if max_chips:
        gates.append({
            "name": "budget.max_chips",
            "metric": "chips",
            "comparator": "lte",
            "bound": int(max_chips),
            "unit": "chips",
            "same_execution": True,
        })
    # The load generator is part of the measurement: a replicate whose
    # client fell behind its own schedule measured the client, and the
    # bench report marks it. Pre-registering the gate makes such a run
    # inadmissible rather than a quiet tie.
    gates.append({
        "name": "load.client_limited",
        "metric": "client_limited_replicates",
        "comparator": "lte",
        "bound": 0,
        "unit": "replicates",
        "same_execution": True,
    })
    source = profile["source"]
    shape = source.get("catalog_entry") or "custom"
    if profile.get("agentx"):
        # Closed loop: the cell is the ladder, not a rate.
        ladder = profile["agentx"]["concurrency"]
        cell_shape = (f"{source.get('scenario') or 'agentx'}"
                      f"@c{ladder[0]}-{ladder[-1]}")
    else:
        cell_shape = f"{shape}@{profile['rate']['target_qps']:g}qps"
    return {
        "primary_metric": {
            "name": primary,
            "unit": unit,
            "direction": direction
        },
        "measurement_cell": {
            "topology": f"{accel['tpu_accelerator']}-{accel['tpu_topology']}",
            "shape": cell_shape,
        },
        "replicate_count": int(replicates),
        "noise_floor_pct": float(noise_floor_pct),
        "stopping_rule": {
            "max_rounds": int(max_rounds),
            "plateau_rounds": int(plateau_rounds),
        },
        "admissibility_gates": gates,
        "mode": mode,
    }


def _secondary_metric(primary: str) -> str:
    if primary == "goodput_at_slo_per_chip":
        return "cost_per_1M_tokens_at_slo"
    if primary.startswith("total_tokens_per_s"):
        return "output_tokens_per_s_per_chip"
    if primary.startswith("output_tokens_per_s"):
        return "total_tokens_per_s_per_chip"
    return "goodput_at_slo_per_chip"


def build_study_extras(profile: dict[str, Any], recipe: dict[str, Any], *,
                       inner_sweep_budget: int) -> dict[str, Any]:
    """What the protocol schema cannot hold but every round must respect."""
    budget = profile.get("budget") or {}
    config = render.default_config(recipe)
    return {
        "schema": "llmd-study",
        "primary_metric": profile["objective"]["primary_metric"],
        "secondary_metric": _secondary_metric(
            profile["objective"]["primary_metric"]),
        "inner_sweep_budget": int(inner_sweep_budget),
        "max_chip_hours": budget.get("max_chip_hours"),
        "max_chips": budget.get("max_chips"),
        "chip_hour_cost": budget.get("chip_hour_cost"),
        "chips_per_replica": render.chips_per_replica(recipe, config),
        # The fleet <-> engine loop: when the fleet search plateaus with
        # the objective unmet, the bound role goes to tpu-performance and
        # the improved recipe comes back (engine_escalation.py). On by
        # default; `search.engine_loop` in the workload spec bounds it.
        "engine_loop": {
            "enabled": bool(((profile.get("search") or {}).get("engine_loop")
                             or {}).get("enabled", True)),
            "max_escalations": int(((profile.get("search") or {}).get(
                "engine_loop") or {}).get("max_escalations") or 3),
            "escalations": [],
            "exhausted_roles": [],
        },
        "objective_target": dict(
            (profile["objective"].get("target") or {})) or None,
        "startup_timeout_s": DEFAULTS["startup_timeout_s"],
        "benchmark_duration_s": (profile["agentx"]["duration_s"]
                                 if profile.get("agentx")
                                 else DEFAULTS["benchmark_duration_s"]),
        "bench_timeout_s": DEFAULTS["bench_timeout_s"],
        "workload": {
            "name": profile["name"],
            "source": profile["source"],
            "target_qps": profile["rate"]["target_qps"],
            **({"agentx": {"concurrency": profile["agentx"]["concurrency"],
                           "duration_s": profile["agentx"]["duration_s"],
                           "submission_valid": profile["agentx"][
                               "submission_valid"]}}
               if profile.get("agentx") else {}),
            "tenants": [{
                "name": t["name"],
                "share": t["share"],
                "priority": t["priority"],
                "slo": t["slo"]
            } for t in profile["tenants"]],
        },
        "recipe": {
            "model": recipe["model"],
            "image": recipe["image"],
            "accelerator": recipe["accelerator"],
            "knee_rps": (recipe.get("measured") or {}).get("knee_rps"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-spec", required=True, type=pathlib.Path)
    parser.add_argument("--recipe", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--mode", choices=["approve", "auto"],
                        default="approve")
    parser.add_argument("--replicates", type=int,
                        default=DEFAULTS["replicates"])
    parser.add_argument("--noise-floor-pct", type=float,
                        default=DEFAULTS["noise_floor_pct"])
    parser.add_argument("--max-rounds", type=int,
                        default=DEFAULTS["max_rounds"])
    parser.add_argument("--plateau-rounds", type=int,
                        default=DEFAULTS["plateau_rounds"])
    parser.add_argument("--inner-sweep-budget", type=int,
                        default=DEFAULTS["inner_sweep_budget"])
    args = parser.parse_args(argv)

    spec = json.loads(args.workload_spec.read_text(encoding="utf-8"))
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    render.validate_recipe(recipe)
    profile = workload.resolve_workload(spec, spec_dir=args.workload_spec.parent)
    protocol = build_protocol(profile,
                              recipe,
                              mode=args.mode,
                              replicates=args.replicates,
                              noise_floor_pct=args.noise_floor_pct,
                              max_rounds=args.max_rounds,
                              plateau_rounds=args.plateau_rounds)
    extras = build_study_extras(profile,
                                recipe,
                                inner_sweep_budget=args.inner_sweep_budget)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (args.out / "llmd-study.json").write_text(
        json.dumps(extras, indent=2) + "\n", encoding="utf-8")
    shown = dict(profile)
    shown.pop("base_config", None)
    (args.out / render.PROFILE_FILE).write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": str(args.out / "protocol.json"),
        "llmd_study": str(args.out / "llmd-study.json"),
        "workload_profile": str(args.out / render.PROFILE_FILE),
        "primary_metric": protocol["primary_metric"],
        "gates": len(protocol["admissibility_gates"]),
        "workload_summary": profile["source"].get("summary", "custom config"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
