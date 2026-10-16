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
an arm exists.

An arm is **gated, never pruned**. One that needs a connector the serving
image lacks is written `blocked` with `blocked_on_engine_capability`; one
that would change model servers an adopted fleet does not own is written
`blocked` with `blocked_on_adopted_fleet`. Both stay in the plan so the
study reports what the workload wanted and what would open it (a package
change, or taking the fleet under management). The same `gate()` is what
`plan_arms.py` applies to the arms it proposes later.

Two files: `arms.json` in study-round shape (what `record-round` takes) and
`arm_plans.json` keyed by arm id with the `arm_kind`, the cold `sets` and
the hot/warm `sweep_knobs` the executor may move.

Usage:
    seed_arms.py --workload-spec SPEC --recipe RECIPE --out DIR
                 [--max-chips N] [--capabilities engine-capabilities.json]
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

if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import size_pd  # noqa: E402  (same directory)

BLOCKED_CAPABILITY = "blocked_on_engine_capability"
BLOCKED_ADOPTED = "blocked_on_adopted_fleet"

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
    "kv.precise_routing": [
        "router.plugins.prefix-cache-affinity-filter.affinityThreshold",
        "router.plugins.prefix-cache-affinity-filter.peakPrefillThroughput",
        "router.plugins.token-load-scorer.weight",
    ],
    "kv.tiered_host": [
        "kv.tiered_host.cpu_chunks", "kv.tiered_host.staging_blocks",
        "router.plugins.prefix-cache-scorer.weight",
    ],
    "topology.pd": [
        "topology.pd_decider.parameters.nonCachedTokens",
        "router.plugins.token-load-scorer.weight",
        "topology.prefill.engine.max_num_batched_tokens",
        "topology.decode.engine.max_num_seqs",
        "topology.decode.engine.max_num_batched_tokens",
    ],
    # A shape arm moves the xP:yD lattice; its inner sweep is the P/D one.
    "topology.pd_shape": [
        "topology.pd_decider.parameters.nonCachedTokens",
        "router.plugins.token-load-scorer.weight",
        "topology.prefill.engine.max_num_batched_tokens",
        "topology.decode.engine.max_num_seqs",
        "topology.decode.engine.max_num_batched_tokens",
    ],
    "router.flow_control": [
        "router.flow_control.saturation_detector.queueDepthThreshold",
        "router.flow_control.saturation_detector.kvCacheUtilThreshold",
    ],
    "router.predicted_latency": [
        "router.plugins.latency-scorer.weight",
    ],
    "router.epp_replicas": [],
}

# Which probe capability each arm family needs. A `false` in
# engine-capabilities.json blocks the arm; a missing probe leaves it open,
# because the vllm-torchtpu image is assumed to carry what llm-d's guides
# name (tpu-llmd/references/PLATFORM.md) until a probe says otherwise.
NEEDS_CAPABILITY: dict[str, str] = {
    "kv.precise_routing": "kv_events",
    "kv.tiered_host": "offload_connector",
    "topology.pd": "pd_connector",
    "topology.pd_shape": "pd_connector",
}

COARSE_KIND = {
    "baseline": "fleet-topology",
    "topology.replicas": "fleet-topology",
    "topology.pd": "fleet-topology",
    "topology.pd_shape": "fleet-topology",
    "kv.precise_routing": "kv-reuse",
    "kv.tiered_host": "kv-reuse",
    "router.profile": "routing-policy",
    "router.flow_control": "routing-policy",
    "router.predicted_latency": "routing-policy",
    "router.epp_replicas": "routing-policy",
    "engine.fleet_knob": "engine-fleet-knob",
    "platform.capacity": "platform-capacity",
}

# The guide's single-host tiered-prefix-cache overlay sizes; the inner sweep
# moves them (https://github.com/llm-d/llm-d/tree/v0.10.0/guides/tiered-prefix-cache).
TIERED_HOST_DEFAULT = {"cpu_chunks": 25000, "staging_blocks": 1000}


def load_capabilities(path: pathlib.Path | str | None) -> dict[str, Any]:
    """The `capabilities` map of a probe document, or {} when none."""
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return dict(doc.get("capabilities") or {})


def gate(arm_kind: str, sets: dict[str, Any], *, incumbent: dict[str, Any],
         recipe: dict[str, Any], capabilities: dict[str, Any]) -> str:
    """Why this arm cannot run now, or '' when it can.

    Capability first: an image without the connector cannot run the arm
    anywhere. Then the fleet: on an adopted fleet any path in
    `render.ADOPTED_IMMUTABLE_PATHS`, or a replica count other than the one
    running, would restart servers this release does not own.
    """
    cap = NEEDS_CAPABILITY.get(arm_kind)
    if cap and capabilities and capabilities.get(cap) is False:
        return (f"{BLOCKED_CAPABILITY}: image lacks {cap}"
                " (engine-capabilities.json)")
    fleet = render.fleet_spec(recipe)
    if fleet["mode"] != render.FLEET_ADOPTED:
        return ""
    trial = apply_sets(incumbent, sets)
    touched = [path for path, is_on in render.ADOPTED_IMMUTABLE_PATHS
               if is_on(trial)]
    if int(trial["topology"]["decode"]["replicas"]) != fleet["replicas"]:
        touched.append("topology.decode.replicas")
    if touched:
        return (f"{BLOCKED_ADOPTED}: {touched} would change model servers"
                " this release does not own")
    return ""


def capacity_floor(profile: dict[str, Any], recipe: dict[str, Any]) -> tuple[int, str]:
    """Replicas needed at target_qps from the recipe's knee, and how known.

    An adopted fleet's floor is what is running: the study cannot scale it.
    """
    fleet = render.fleet_spec(recipe)
    if fleet["mode"] == render.FLEET_ADOPTED:
        return fleet["replicas"], "adopted"
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
         max_chips: int | None,
         capabilities: dict[str, Any] | None = None
         ) -> tuple[dict[str, Any], dict[str, Any]]:
    caps = capabilities or {}
    n0, n0_provenance = capacity_floor(profile, recipe)
    base = render.default_config(recipe)
    adopted = render.fleet_spec(recipe)["mode"] == render.FLEET_ADOPTED
    chips = render.chips_per_replica(recipe, base)
    cap = max_chips or (profile.get("budget") or {}).get("max_chips")
    if cap and not adopted:
        n_max = max(1, int(cap) // chips)
        if n0 > n_max:
            n0 = n_max
    signals = _reuse_signals(profile)
    tenants = profile.get("tenants") or []
    priorities = {int(t.get("priority") or 0) for t in tenants}
    tp = base["topology"]["decode"].get("tp")
    arms: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}

    def add(arm_id: str, arm_kind: str, claim: str, *, share: float,
            headroom: float, sets: dict[str, Any], rationale: str = "",
            prerequisites: dict[str, Any] | None = None) -> None:
        blocked_on = "" if arm_kind == "baseline" else gate(
            arm_kind, sets, incumbent=base, recipe=recipe, capabilities=caps)
        entry: dict[str, Any] = {
            "arm": arm_id,
            "kind": COARSE_KIND[arm_kind],
            "status": "blocked" if blocked_on else "open",
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
            "prerequisites": prerequisites or {},
        }

    # Stage 1: capacity. The baseline is what round one measures: for a
    # managed fleet, llm-d's optimized baseline at the capacity floor; for
    # an adopted fleet, the servers as they run today with no router.
    rate = profile["rate"]["target_qps"]
    if adopted:
        add("baseline", "baseline",
            f"the {n0} running replica(s) reached through the fleet Service"
            f" (passthrough, no router) meet the SLO at {rate:g} qps",
            share=1.0, headroom=0.0, sets={},
            rationale="adopted fleet: the customer's current state is the"
                      " control every router arm is judged against")
    else:
        add("baseline", "baseline",
            f"{n0} aggregated replica(s) of the recipe behind the standalone"
            " router with sticky-until-saturated routing meet the SLO at"
            f" {rate:g} qps",
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

    # Stage 2: KV reuse.
    if signals["shared_prefix"] or signals["multi_turn"]:
        add("a3", "kv.precise_routing",
            "exact KV-event-driven prefix routing raises prefix hit rate"
            " over the approximate index",
            share=0.3, headroom=0.3,
            sets={"kv.precise_routing": True, "kv.block_size": 64},
            rationale="workload has shared prefixes or multi-turn reuse",
            prerequisites={"capabilities": ["kv_events"],
                           "fleet_mode": render.FLEET_MANAGED})
    if signals["offload_suggested"]:
        add("a4", "kv.tiered_host",
            "host-memory KV tier keeps multi-turn context beyond HBM"
            " capacity",
            share=0.2, headroom=0.3,
            sets={"kv.tiered_host": dict(TIERED_HOST_DEFAULT)},
            rationale="multi-turn with long shared prefixes; guide sizes"
                      " from the tiered-prefix-cache single-host overlay",
            prerequisites={"capabilities": ["offload_connector"],
                           "fleet_mode": render.FLEET_MANAGED})

    # Stage 3: routing policy, all hot; the control arm measures what
    # prefix affinity is worth at all. An adopted fleet starts without a
    # router, so llm-d's optimized-baseline profile is its first arm.
    if adopted:
        add("a0", "router.profile",
            "sticky-until-saturated affinity routing beats kube-proxy"
            " spreading at this rate",
            share=1.0, headroom=0.3,
            sets={"router.profile": "sticky-until-saturated"},
            rationale="llm-d optimized-baseline profile; sweeps"
                      " peakPrefillThroughput and affinityThreshold")
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
    if len(priorities) > 1:
        bands = [{"priority": p} for p in sorted(priorities, reverse=True)]
        add("a9", "router.flow_control",
            "queueing at the router with per-priority bands keeps the"
            " high-priority tenant inside its SLO under burst",
            share=0.3, headroom=0.1,
            sets={"router.profile": "prefix-weighted",
                  "router.flow_control": {
                      "saturation_detector": {"type": "utilization-detector",
                                              "queueDepthThreshold": 5,
                                              "kvCacheUtilThreshold": 0.8},
                      "priority_bands": bands}},
            rationale=f"tenant priorities {sorted(priorities)} map to"
                      " InferenceObjective bands")

    # Stage 4: P/D, when the workload is prefill heavy. The opening shape
    # is sized (size_pd), not fixed at one prefill slice.
    if signals["pd_suggested"]:
        baseline_cfg = dict(base)
        baseline_cfg["topology"] = dict(base["topology"])
        baseline_cfg["topology"]["decode"] = dict(base["topology"]["decode"],
                                                  replicas=n0)
        opening = size_pd.opening_shape(profile, recipe, baseline_cfg,
                                        hit_rate=None, max_chips=cap)
        p_n, d_n = opening["prefill_replicas"], opening["decode_replicas"]
        add("a8", "topology.pd",
            f"a {p_n}P:{d_n}D split isolates long prefills from decode so"
            " TPOT holds at target rate",
            share=0.4, headroom=0.3,
            sets={"topology.mode": "disaggregated",
                  "topology.prefill": {"replicas": p_n, "tp": tp},
                  "topology.decode.replicas": d_n},
            rationale="catalog metadata lists P/D disaggregation;"
                      f" {opening['note']}",
            prerequisites={"capabilities": ["pd_connector"],
                           "chips": opening["chips"],
                           "fleet_mode": render.FLEET_MANAGED,
                           "sizing_provenance": opening["provenance"]})

    return {"arms": arms}, {
        "capacity_floor": {"replicas": n0, "provenance": n0_provenance,
                           "chips_per_replica": chips, "max_chips": cap,
                           "fleet_mode": render.FLEET_ADOPTED if adopted
                           else render.FLEET_MANAGED},
        "signals": signals,
        "capabilities": caps,
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
    parser.add_argument("--capabilities", type=pathlib.Path, default=None,
                        help="engine-capabilities.json from the platform"
                             " probe; gates connector arms")
    args = parser.parse_args(argv)
    spec = json.loads(args.workload_spec.read_text(encoding="utf-8"))
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    render.validate_recipe(recipe)
    profile = workload.resolve_workload(spec, spec_dir=args.workload_spec.parent)
    arms, plans = seed(profile, recipe, max_chips=args.max_chips,
                       capabilities=load_capabilities(args.capabilities))
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
