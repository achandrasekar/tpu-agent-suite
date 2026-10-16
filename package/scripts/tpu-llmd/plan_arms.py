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
"""Propose the next arms of an llm-d study as deltas against the incumbent.

`seed_arms.py` opens a study with a staged queue from the workload's shape
alone. This script is what the planner runs on every later call: it reads
the incumbent, what the baseline's fleet profile measured, which arms are
decided and what the image can do, and writes an `llmd-plan-result` -- a
ranked bank of arms, each with its cold `sets`, the hot/warm knobs the
executor may sweep, a modelled ceiling, and the evidence it rests on.

The model is deterministic and small on purpose: the planner agent reads
its output, applies judgement to the ranking and the rationale, and returns
it. What the script decides, and why:

* **Gating, never pruning.** An arm whose connector the image lacks, or
  whose change the adopted fleet cannot take, is written `blocked` with
  `blocked_on_engine_capability` / `blocked_on_adopted_fleet`. The study
  keeps the finding; a package or fleet change opens it.
* **Evidence over hints.** The catalog says a workload "suits prefix-aware
  routing"; the baseline's `prefix_hit_rate` says whether reuse is already
  happening. Low measured hit rate with a shared-prefix workload raises the
  ceiling of precise routing; a decided `load-only` win lowers it.
* **Incumbent-relative.** Every `sets` is applied over the incumbent, so an
  arm proposed after a replica-count win inherits it.
* **Dedup.** Arms whose `sets` match a decided or open arm are not
  re-proposed.

Usage:
    plan_arms.py --request REQUEST.json --out DIR
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

if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import seed_arms  # noqa: E402  (same directory)
import size_pd  # noqa: E402  (same directory)

BLOCKED_CAPABILITY = seed_arms.BLOCKED_CAPABILITY
BLOCKED_ADOPTED = seed_arms.BLOCKED_ADOPTED
SWEEP_KNOBS = seed_arms.SWEEP_KNOBS

EVIDENCE = {
    "kv.precise_routing": [
        "https://github.com/llm-d/llm-d/tree/v0.10.0/guides/precise-prefix-cache-routing",
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache/README.md",
    ],
    "kv.tiered_host": [
        "https://github.com/llm-d/llm-d/tree/v0.10.0/guides/tiered-prefix-cache",
        "https://github.com/llm-d/llm-d/blob/v0.10.0/guides/tiered-prefix-cache/benchmark-results/vllm-qwen3-32b-tpuv7.md",
    ],
    "topology.pd": [
        "https://github.com/llm-d/llm-d/tree/v0.10.0/guides/pd-disaggregation",
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/docs/disaggregation.md",
    ],
    "topology.pd_shape": [
        "https://github.com/llm-d/llm-d/tree/v0.10.0/guides/pd-disaggregation",
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/docs/disaggregation.md",
    ],
    "router.flow_control": [
        "https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/flow-control.md",
    ],
    "router.predicted_latency": [
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/README.md",
    ],
    "router.profile": [
        "https://github.com/llm-d/llm-d/blob/v0.10.0/guides/optimized-baseline/router/optimized-baseline.values.yaml",
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity/README.md",
    ],
    "engine.fleet_knob": [],
    "topology.replicas": [],
    "router.epp_replicas": [
        "https://github.com/llm-d/llm-d-router/blob/v0.11.0/config/charts/README.md",
    ],
}


def _read(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _metric(profile_doc: dict[str, Any] | None, name: str) -> float | None:
    if not profile_doc:
        return None
    entry = (profile_doc.get("metrics") or {}).get(name) or {}
    value = entry.get("value")
    return float(value) if value is not None else None


def _sets_key(sets: dict[str, Any]) -> str:
    return json.dumps(sets, sort_keys=True, separators=(",", ":"))


def _already(sets: dict[str, Any], decided: list[dict[str, Any]],
             plans_open: dict[str, Any]) -> bool:
    key = _sets_key(sets)
    for d in decided:
        if _sets_key(d.get("sets") or {}) == key:
            return True
    for plan in plans_open.values():
        if _sets_key(plan.get("sets") or {}) == key:
            return True
    return False


def propose(req: dict[str, Any]) -> dict[str, Any]:
    spec_path = pathlib.Path(req["workload_spec_path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    recipe = json.loads(pathlib.Path(req["recipe_path"]).read_text("utf-8"))
    render.validate_recipe(recipe)
    profile = workload.resolve_workload(spec, spec_dir=spec_path.parent)
    incumbent = render.normalize_config(json.loads(
        pathlib.Path(req["incumbent_config_path"]).read_text("utf-8")))
    caps_doc = _read(req.get("capabilities_path")) or {}
    caps = caps_doc.get("capabilities") or {}
    fleet = render.fleet_spec(recipe)
    fleet_mode = req.get("fleet_mode") or fleet["mode"]
    adopted = fleet_mode == render.FLEET_ADOPTED
    baseline_profile = _read(req.get("baseline_profile_path"))
    decided = list(req.get("decided_arms") or [])
    open_ids = set(req.get("open_arms") or [])
    max_new = int(req.get("max_new_arms") or 4)
    max_chips = req.get("max_chips") or (profile.get("budget") or {}).get(
        "max_chips")

    signals = seed_arms._reuse_signals(profile)
    hit_rate = _metric(baseline_profile, "prefix_hit_rate")
    kv_util = _metric(baseline_profile, "kv_util_mean")
    waiting = _metric(baseline_profile, "requests_waiting_mean")
    kv_stddev = _metric(baseline_profile, "kv_util_pod_stddev")
    tenants = profile.get("tenants") or []
    priorities = {int(t.get("priority") or 0) for t in tenants}
    decided_kinds = {d["arm_kind"]: d for d in decided}
    load_only_won = (decided_kinds.get("router.profile") or {}).get(
        "verdict") == "win" and "load-only" in json.dumps(
        decided_kinds.get("router.profile", {}).get("sets", {}))
    chips_per = render.chips_per_replica(recipe, incumbent)
    replicas = int(incumbent["topology"]["decode"]["replicas"])
    used = render.total_chips(recipe, incumbent)

    # Which arm ids are taken: decided + open.
    taken = {d["arm"] for d in decided} | open_ids
    counter = [0]

    def next_id() -> str:
        while True:
            counter[0] += 1
            cand = f"p{counter[0]}"
            if cand not in taken:
                taken.add(cand)
                return cand

    arms: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}
    closed: list[str] = []

    def gate(arm_kind: str, sets: dict[str, Any]) -> str:
        """Why this arm cannot run now, or '' when it can (seed_arms.gate)."""
        return seed_arms.gate(arm_kind, sets, incumbent=incumbent,
                              recipe=recipe, capabilities=caps)

    def add(arm_kind: str, claim: str, *, share: float, headroom: float,
            sets: dict[str, Any], rationale: str,
            prerequisites: dict[str, Any] | None = None) -> None:
        if _already(sets, decided, plans):
            closed.append(f"{arm_kind}: already proposed {sets}")
            return
        blocked_on = gate(arm_kind, sets)
        arm_id = next_id()
        entry: dict[str, Any] = {
            "arm": arm_id,
            "kind": seed_arms.COARSE_KIND.get(arm_kind, "routing-policy"),
            "status": "blocked" if blocked_on else "open",
            "claim": claim,
            "share": round(share, 4),
            "headroom": round(headroom, 4),
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
            "evidence": EVIDENCE.get(arm_kind, []),
        }

    # -- 1. KV reuse ---------------------------------------------------------
    reuse_headroom = 0.0
    if signals["shared_prefix"] or signals["multi_turn"]:
        # Every uncached prompt token is prefill work; the share of tokens a
        # better router could have served from cache bounds the gain.
        reuse_headroom = 0.3 if hit_rate is None else max(
            0.05, min(0.5, 0.8 - hit_rate))
        if load_only_won:
            reuse_headroom *= 0.3
    if reuse_headroom > 0 and incumbent["router"]["profile"] != \
            render.PASSTHROUGH_PROFILE:
        if not incumbent["kv"].get("precise_routing"):
            add("kv.precise_routing",
                "exact KV-event-driven prefix routing raises the prefix hit"
                " rate over the approximate index",
                share=0.3, headroom=reuse_headroom,
                sets={"kv.precise_routing": True, "kv.block_size":
                      pins_block(incumbent)},
                rationale=(f"measured prefix_hit_rate={hit_rate}" if hit_rate
                           is not None else "prefix hit rate not measured")
                + "; workload has shared prefixes or multi-turn reuse",
                prerequisites={"capabilities": ["kv_events"],
                               "fleet_mode": "managed"})
    if signals["offload_suggested"] and not incumbent["kv"].get("tiered_host"):
        add("kv.tiered_host",
            "a host-memory KV tier keeps multi-turn context past HBM"
            " capacity, cutting repeated prefill",
            share=0.2, headroom=max(0.1, reuse_headroom * 0.8),
            sets={"kv.tiered_host": dict(seed_arms.TIERED_HOST_DEFAULT)},
            rationale="multi-turn with long shared prefixes; guide sizes"
                      " from tiered-prefix-cache single-host overlay"
            + (f"; kv_util_mean={kv_util}" if kv_util is not None else ""),
            prerequisites={"capabilities": ["offload_connector"],
                           "fleet_mode": "managed"})

    # -- 2. Routing policy (hot inside, cold between) -----------------------
    current = incumbent["router"]["profile"]
    if current == render.PASSTHROUGH_PROFILE:
        # Nothing routes yet: the optimized-baseline profile is the first
        # thing to try, then the control.
        add("router.profile",
            "sticky-until-saturated affinity routing beats kube-proxy"
            " spreading at this rate",
            share=1.0, headroom=max(0.1, reuse_headroom),
            sets={"router.profile": "sticky-until-saturated"},
            rationale="llm-d optimized-baseline profile; sweeps"
                      " peakPrefillThroughput and affinityThreshold")
        add("router.profile",
            "load-only routing (no prefix affinity) is no worse: the control"
            " for every reuse claim",
            share=0.2, headroom=0.05, sets={"router.profile": "load-only"},
            rationale="control arm")
    for prof, head, why in (
            ("prefix-weighted", 0.25, "llm-d's recommended scorer set"),
            ("session-sticky", 0.15 if signals["multi_turn"] else 0.0,
             "multi-turn workload"),
            ("sticky-until-saturated", 0.2, "llm-d optimized-baseline"),
            ("load-only", 0.05, "control arm")):
        if prof == current or head <= 0:
            continue
        add("router.profile", f"{prof} routing beats {current} at this rate",
            share=0.2, headroom=head, sets={"router.profile": prof},
            rationale=why)

    # -- 3. Flow control, when tenancy or saturation asks for it ------------
    if not incumbent["router"].get("flow_control") and current != \
            render.PASSTHROUGH_PROFILE:
        saturated = (waiting is not None and waiting > 1.0) or (
            kv_util is not None and kv_util > 0.8)
        if len(priorities) > 1 or saturated:
            bands = [{"priority": p} for p in sorted(priorities, reverse=True)]
            add("router.flow_control",
                "queueing at the router with per-priority bands keeps the"
                " high-priority tenant inside its SLO under burst",
                share=0.3, headroom=0.15 if saturated else 0.1,
                sets={"router.flow_control": {
                    "saturation_detector": {"type": "utilization-detector",
                                            "queueDepthThreshold": 5,
                                            "kvCacheUtilThreshold": 0.8},
                    "priority_bands": bands}},
                rationale=f"tenant priorities {sorted(priorities)};"
                          f" requests_waiting_mean={waiting},"
                          f" kv_util_mean={kv_util}")

    # -- 4. Predicted latency, when request lengths are heterogeneous -------
    shape = profile.get("shape") or {}
    isl = shape.get("input_sequence_length") or {}
    heterogeneous = bool(isl.get("standard_deviation")) and float(
        isl.get("standard_deviation") or 0) > 0.5 * float(
        isl.get("mean") or 1)
    if heterogeneous and not incumbent["router"].get("predicted_latency") \
            and current != render.PASSTHROUGH_PROFILE:
        add("router.predicted_latency",
            "SLO-aware routing on predicted TTFT/TPOT improves attainment"
            " for mixed-length requests",
            share=0.2, headroom=0.1,
            sets={"router.predicted_latency": True},
            rationale="input length std-dev > 0.5 x mean; alpha plugins in"
                      " router v0.11.0, needs the predictor sidecars")

    # -- 5. P/D: the opening shape, then the lattice around it -------------
    incumbent_profile = _read(req.get("incumbent_profile_path")) or \
        baseline_profile
    incumbent_report = _read(req.get("incumbent_report_path"))
    attribution = size_pd.attribute(incumbent_profile, incumbent_report)
    pd_now = incumbent["topology"]["mode"] == "disaggregated"
    tp = incumbent["topology"]["decode"].get("tp")
    if signals["pd_suggested"] and not pd_now:
        # The first split is sized, not guessed: the ratio model picks
        # (P, D) from token demand and the recipe's per-role capacity,
        # or the token ratio over the incumbent's own slices.
        opening = size_pd.opening_shape(profile, recipe, incumbent,
                                        hit_rate=hit_rate,
                                        max_chips=max_chips)
        p_n, d_n = opening["prefill_replicas"], opening["decode_replicas"]
        need = opening["chips"]
        fits = max_chips is None or need <= int(max_chips)
        add("topology.pd",
            f"a {p_n}P:{d_n}D split isolates long prefills from decode so"
            " TPOT holds at target rate",
            share=0.4, headroom=0.3 if fits else 0.0,
            sets={"topology.mode": "disaggregated",
                  "topology.prefill": {"replicas": p_n, "tp": tp},
                  "topology.decode.replicas": d_n},
            rationale=f"{opening['note']}; needs {need} chips"
                      f" (budget {max_chips})" + ("" if fits else
                                                 "; over budget"),
            prerequisites={"capabilities": ["pd_connector"],
                           "chips": need, "fleet_mode": "managed",
                           "sizing_provenance": opening["provenance"]})
    elif pd_now:
        # Coordinate ascent on the shape: move the bound axis first, at
        # constant chips when possible; shrink when neither pool is bound.
        p_cur = int(incumbent["topology"]["prefill"]["replicas"])
        d_cur = int(incumbent["topology"]["decode"]["replicas"])
        for i, nb in enumerate(size_pd.neighbours(
                p_cur, d_cur, attribution["role"], chips_per=chips_per,
                max_chips=max_chips)):
            np_, nd = nb["prefill_replicas"], nb["decode_replicas"]
            add("topology.pd_shape",
                f"{np_}P:{nd}D beats {p_cur}P:{d_cur}D on goodput per chip"
                f" ({nb['why']})",
                share=0.5, headroom=max(0.05, 0.25 - 0.05 * i),
                sets={"topology.prefill.replicas": np_,
                      "topology.decode.replicas": nd},
                rationale="; ".join([nb["why"]] + attribution["evidence"]),
                prerequisites={"capabilities": ["pd_connector"],
                               "chips": nb["chips"], "fleet_mode": "managed",
                               "attribution": attribution})

    # -- 6. Capacity around the incumbent (aggregated fleets; a split
    #       moves through the shape lattice above) ------------------------
    if replicas > 1 and not pd_now:
        add("topology.replicas",
            f"{replicas - 1} replicas still meet the SLO once the engine"
            " knobs are re-tuned, saving one slice",
            share=1.0, headroom=1.0 / replicas,
            sets={"topology.decode.replicas": replicas - 1},
            rationale="goodput per chip rises by N/(N-1) if attainment holds")
    if not pd_now and (max_chips is None or used + chips_per <= int(max_chips)):
        add("topology.replicas",
            f"{replicas + 1} replicas recover attainment the incumbent"
            " misses at target rate",
            share=1.0, headroom=0.0,
            sets={"topology.decode.replicas": replicas + 1},
            rationale="only a win if the incumbent fails a gate")

    # -- 7. Engine fleet knobs, a warm arm for managed fleets ---------------
    if not incumbent["engine"]:
        add("engine.fleet_knob",
            "fleet-level max_num_seqs / batched-tokens retuning beats the"
            " single-replica recipe values behind a router",
            share=0.3, headroom=0.1,
            sets={"engine.max_num_seqs": 256},
            rationale="recipe knobs were tuned for one replica without a"
                      " router; the inner sweep walks the ladder")

    # Rank: open before blocked, then headroom x share.
    def rank(a: dict[str, Any]) -> tuple[int, float]:
        return (0 if a["status"] == "open" else 1,
                -(a["headroom"] * max(a["share"], 0.05)))

    arms.sort(key=rank)
    kept = arms[:max_new] + [a for a in arms[max_new:]
                             if a["status"] == "blocked"]
    kept_ids = {a["arm"] for a in kept}
    plans = {k: v for k, v in plans.items() if k in kept_ids}
    status = "planned" if any(a["status"] == "open" for a in kept) else (
        "planned" if kept else "nothing_to_propose")
    return {
        "status": status,
        "arms": kept,
        "plans": plans,
        "model": {
            "incumbent_profile": current,
            "incumbent_replicas": replicas,
            "chips_used": used,
            "chips_per_replica": chips_per,
            "max_chips": max_chips,
            "fleet_mode": fleet_mode,
            "signals": signals,
            "observed": {"prefix_hit_rate": hit_rate, "kv_util_mean": kv_util,
                         "requests_waiting_mean": waiting,
                         "kv_util_pod_stddev": kv_stddev},
            "attribution": attribution,
            "reuse_headroom": round(reuse_headroom, 4),
            "load_only_won": load_only_won,
            "capabilities": caps,
            "closed": closed,
        },
        "summary": f"{sum(a['status'] == 'open' for a in kept)} open,"
                   f" {sum(a['status'] == 'blocked' for a in kept)} blocked"
                   f" arm(s) proposed against incumbent {current!r} x"
                   f" {replicas}",
    }


def pins_block(incumbent: dict[str, Any]) -> int:
    """Block size precise routing uses: keep the incumbent's, else 64."""
    return int(incumbent["kv"].get("block_size") or 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    req = json.loads(args.request.read_text(encoding="utf-8"))
    for key in ("run_id", "study_id", "workload_spec_path", "recipe_path",
                "incumbent_config_path"):
        if not req.get(key):
            print(json.dumps({"status": "invalid_input",
                              "summary": f"request lacks {key}"}))
            return 2
    result = propose(req)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "llmd-plan-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["artifacts"] = [str(args.out / "llmd-plan-result.json")]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
