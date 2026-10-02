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
"""Seed the study's arms from the workload profile and the recipe.

The outer loop is greedy coordinate ascent over cold dimensions in a staged
order: capacity first, then KV reuse, then routing. This script writes the
opening queue in that order with a modelled ceiling for each arm, so round
one has something ranked to pick from and every later round can trace why
an arm exists. Arms the runtime cannot deploy yet are written `blocked`
with the reason, so the plan is visible even where it cannot yet run.

Two files: `arms.json` in study-round shape (what `record-round` takes) and
`arm_plans.json` keyed by arm id with the `arm_kind`, the cold `sets` and
the hot/warm `sweep_knobs` the executor may move.

Usage:
    seed_arms.py --workload-spec SPEC --recipe RECIPE --out DIR
                 [--max-chips N]
    seed_arms.py apply-sets --config IN --set path=value ... --out OUT
"""

from __future__ import annotations

import argparse
import json
import math
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
    from framework.deploy import release, render, workload  # noqa: E402
except ImportError:  # shipped runtime: `deploy` is top-level
    from deploy import release, render, workload  # type: ignore  # noqa: E402

# Hot/warm paths each arm kind may sweep in the inner loop. The executor
# chooses values; this bounds where it may look.
SWEEP_KNOBS: dict[str, list[str]] = {
    "baseline": [
        "engine.max_num_seqs", "engine.max_num_batched_tokens",
        "router.plugins.prefix-cache-affinity-filter.peakPrefillThroughput",
    ],
    "topology.replicas": [
        "engine.max_num_seqs", "engine.max_num_batched_tokens",
        "engine.gpu_memory_utilization",
    ],
    "router.profile": [
        "router.plugins.prefix-cache-scorer.weight",
        "router.plugins.kv-cache-utilization-scorer.weight",
        "router.plugins.queue-scorer.weight",
        "router.plugins.prefix-cache-affinity-filter.affinityThreshold",
        "router.plugins.prefix-cache-affinity-filter.peakPrefillThroughput",
        "router.plugins.session-affinity-scorer.weight",
    ],
    "engine.fleet_knob": [
        "engine.max_num_seqs", "engine.max_num_batched_tokens",
        "engine.gpu_memory_utilization", "engine.kv_cache_dtype",
        "kv.block_size", "engine.async_scheduling",
    ],
}

# Arm kinds the renderer refuses today, and why. Seeded as blocked arms when
# the workload would benefit, so the plan shows what a later milestone opens.
NOT_YET = {
    "kv.precise_routing": "runtime does not render precise prefix-cache"
                          " routing yet (kv-events, block-size 64, EPP"
                          " precise producer)",
    "kv.tiered_host": "runtime does not render tiered host KV offload yet",
    "topology.pd": "runtime does not render prefill/decode disaggregation"
                   " yet",
    "router.flow_control": "runtime does not render flow control and"
                           " InferenceObjective priorities yet",
}

COARSE_KIND = {
    "baseline": "fleet-topology",
    "topology.replicas": "fleet-topology",
    "topology.pd": "fleet-topology",
    "kv.precise_routing": "kv-reuse",
    "kv.tiered_host": "kv-reuse",
    "router.profile": "routing-policy",
    "router.flow_control": "routing-policy",
    "router.predicted_latency": "routing-policy",
    "engine.fleet_knob": "engine-fleet-knob",
    "platform.capacity": "platform-capacity",
}


def capacity_floor(profile: dict[str, Any], recipe: dict[str, Any]) -> tuple[int, str]:
    """Replicas needed at target_qps from the recipe's knee, and how known."""
    knee = (recipe.get("measured") or {}).get("knee_rps")
    target = float(profile["rate"]["target_qps"])
    if knee and float(knee) > 0:
        return max(1, math.ceil(target / float(knee))), "modeled"
    return 1, "unknown"


def _reuse_signals(profile: dict[str, Any]) -> dict[str, bool]:
    shape = profile.get("shape") or {}
    meta = shape.get("metadata") or {}
    opts = " ".join(meta.get("suitable_optimizations") or []).lower()
    multi_turn = bool(shape.get("multi_turn"))
    shared_prefix = bool((shape.get("system_prompt") or {}).get("mean")) or (
        "shared_prefix" in shape) or ("prefix" in opts)
    return {
        "multi_turn": multi_turn,
        "shared_prefix": shared_prefix,
        "pd_suggested": "p/d" in opts or "disaggregation" in opts,
        "offload_suggested": "offload" in opts or "tier" in opts
                             or (multi_turn and shared_prefix),
    }


def seed(profile: dict[str, Any], recipe: dict[str, Any], *,
         max_chips: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    n0, n0_provenance = capacity_floor(profile, recipe)
    chips = render.chips_per_replica(recipe, render.default_config(recipe))
    cap = max_chips or (profile.get("budget") or {}).get("max_chips")
    if cap:
        n_max = max(1, int(cap) // chips)
        if n0 > n_max:
            n0 = n_max
    signals = _reuse_signals(profile)
    arms: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}

    def add(arm_id: str, arm_kind: str, claim: str, *, share: float,
            headroom: float, sets: dict[str, Any], status: str = "open",
            blocked_on: str = "", rationale: str = "") -> None:
        entry: dict[str, Any] = {
            "arm": arm_id,
            "kind": COARSE_KIND[arm_kind],
            "status": status,
            "claim": claim,
            "share": round(share, 4),
            "headroom": round(headroom, 4),
            "since_round": 1,
        }
        if blocked_on:
            entry["blocked_on"] = blocked_on
        arms.append(entry)
        plans[arm_id] = {
            "arm_kind": arm_kind,
            "sets": sets,
            "sweep_knobs": SWEEP_KNOBS.get(arm_kind, []),
            "rationale": rationale,
            "ceiling_provenance": "modeled",
        }

    # Stage 1: capacity.
    add("baseline", "baseline",
        f"{n0} aggregated replica(s) of the recipe behind the standalone"
        " router with sticky-until-saturated routing meet the SLO at"
        f" {profile['rate']['target_qps']:g} qps",
        share=1.0, headroom=0.0, sets={"topology.decode.replicas": n0},
        rationale=f"capacity floor ceil(target_qps / knee_rps) = {n0}"
                  f" ({n0_provenance}); inner sweep tunes the knee flags")
    if n0 > 1:
        add("a1", "topology.replicas",
            f"{n0 - 1} replicas still meet the SLO once max_num_seqs and"
            " batched tokens are re-tuned, saving one slice",
            share=1.0, headroom=1.0 / n0,
            sets={"topology.decode.replicas": n0 - 1},
            rationale="goodput per chip rises by N/(N-1) if attainment holds")
    if cap is None or (n0 + 1) * chips <= int(cap):
        add("a2", "topology.replicas",
            f"{n0 + 1} replicas recover SLO attainment the baseline misses"
            " at target rate",
            share=1.0, headroom=0.0,
            sets={"topology.decode.replicas": n0 + 1},
            rationale="only a win if the baseline fails a gate; headroom"
                      " zero by construction, ranked last among capacity")

    # Stage 2: KV reuse -- blocked until the runtime renders it, but seeded
    # so the plan says what the workload wants.
    if signals["shared_prefix"] or signals["multi_turn"]:
        add("a3", "kv.precise_routing",
            "exact KV-event-driven prefix routing raises prefix hit rate"
            " over the approximate index",
            share=0.3, headroom=0.3, sets={"kv.precise_routing": True,
                                           "kv.block_size": 64},
            status="blocked", blocked_on=NOT_YET["kv.precise_routing"],
            rationale="workload has shared prefixes or multi-turn reuse")
    if signals["offload_suggested"]:
        add("a4", "kv.tiered_host",
            "host-memory KV tier keeps multi-turn context beyond HBM"
            " capacity",
            share=0.2, headroom=0.3,
            sets={"kv.tiered_host": {"num_cpu_chunks": 4096}},
            status="blocked", blocked_on=NOT_YET["kv.tiered_host"],
            rationale="multi-turn with long shared prefixes")

    # Stage 3: routing policy, all hot; the control arm measures what
    # prefix affinity is worth at all.
    add("a5", "router.profile",
        "weighted prefix + load scoring spreads load better than sticky"
        " affinity at this rate",
        share=0.2, headroom=0.25, sets={"router.profile": "prefix-weighted"},
        rationale="llm-d's recommended scorer set; sweeps weights")
    add("a6", "router.profile",
        "load-only routing (no prefix affinity) is no worse: the control"
        " for every reuse claim",
        share=0.2, headroom=0.05, sets={"router.profile": "load-only"},
        rationale="control arm; a win here says prefix reuse is not the"
                  " lever for this workload")
    if signals["multi_turn"]:
        add("a7", "router.profile",
            "session affinity pins conversations to one pod and raises"
            " effective prefix reuse",
            share=0.2, headroom=0.15, sets={"router.profile": "session-sticky"},
            rationale="multi-turn workload")

    # Stage 4: P/D, when the workload is prefill heavy.
    if signals["pd_suggested"]:
        add("a8", "topology.pd",
            "a prefill pool isolates long prefills from decode so TPOT"
            " holds at target rate",
            share=0.4, headroom=0.3,
            sets={"topology.mode": "pd", "kv.pd_connector": "tpu_connector"},
            status="blocked", blocked_on=NOT_YET["topology.pd"],
            rationale="catalog metadata lists P/D disaggregation")

    return {"arms": arms}, {
        "capacity_floor": {"replicas": n0, "provenance": n0_provenance,
                           "chips_per_replica": chips, "max_chips": cap},
        "signals": signals,
        "plans": plans,
    }


def apply_sets(config: dict[str, Any], sets: dict[str, Any]) -> dict[str, Any]:
    """The incumbent plus an arm's cold assignments, normalised."""
    out = render.normalize_config(config)
    for path, value in sets.items():
        release.set_path(out, path, value)
    out["provenance"] = {"arm": "", "execution_id": "", "metric": None}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "apply-sets":
        p = argparse.ArgumentParser(prog="seed_arms.py apply-sets")
        p.add_argument("--config", required=True, type=pathlib.Path)
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--sets-json", default=None,
                       help="JSON object of path -> value, as in arm_plans")
        p.add_argument("--out", required=True, type=pathlib.Path)
        a = p.parse_args(argv[1:])
        sets: dict[str, Any] = json.loads(a.sets_json) if a.sets_json else {}
        for item in a.set:
            path, _, value = item.partition("=")
            sets[path.strip()] = release.parse_set_value(value.strip())
        config = apply_sets(json.loads(a.config.read_text("utf-8")), sets)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(a.out), "sets": sets,
                          "config_digest": render.config_digest(config)}))
        return 0

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-spec", required=True, type=pathlib.Path)
    parser.add_argument("--recipe", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--max-chips", type=int, default=None)
    args = parser.parse_args(argv)
    spec = json.loads(args.workload_spec.read_text(encoding="utf-8"))
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    render.validate_recipe(recipe)
    profile = workload.resolve_workload(spec, spec_dir=args.workload_spec.parent)
    arms, plans = seed(profile, recipe, max_chips=args.max_chips)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arms.json").write_text(json.dumps(arms, indent=2) + "\n",
                                        encoding="utf-8")
    (args.out / "arm_plans.json").write_text(
        json.dumps(plans, indent=2) + "\n", encoding="utf-8")
    baseline = apply_sets(render.default_config(recipe),
                          plans["plans"]["baseline"]["sets"])
    baseline["provenance"]["arm"] = "baseline"
    (args.out / "baseline-config.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "arms": [(a["arm"], a["kind"], a["status"]) for a in arms["arms"]],
        "capacity_floor": plans["capacity_floor"],
        "signals": plans["signals"],
        "files": {
            "arms": str(args.out / "arms.json"),
            "arm_plans": str(args.out / "arm_plans.json"),
            "baseline_config": str(args.out / "baseline-config.json"),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
