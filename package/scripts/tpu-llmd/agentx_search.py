#!/usr/bin/env python3
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
"""The AgentX hill-climb: what to try next on a closed-loop xP:yD fleet.

The InferenceX AgentX benchmark ranks total tokens/s per chip, measured
in closed loop at a fixed number of concurrent sessions, qualifying at
TTFT p50 <= 5 s. The search therefore has one axis the rate workloads do
not -- the **operating point** (`workload.concurrency`) -- and its
shape sizing runs on **sessions per engine**, not tokens per second:

* a prefill engine holds the prefix caches of so many agentic sessions
  before it thrashes (default prior: 96 on one tpu7x host);
* a decode engine keeps KV headroom for so many (default prior: 112).

Both come from `recipe.measured.sessions_per_{prefill,decode}_engine`;
without them the default priors stand in, labelled as priors.

Round order on a point: climb concurrency on the incumbent shape while
the gate holds and total/chip rises; A/B the router (the InferenceX
turn-priority config against the same without flow control); the
per-role engine knobs the recipe maps; then the shape neighbour the
cliff attribution names. Attribution reads the role metrics the bench
sampled through the pod proxy and the prefix-hit rates aiperf scraped:

| signature | bound |
| prefill GPU prefix hit < 0.6, prefill waiting high | prefill (cache thrash) |
| decode running ~= max_num_seqs x pods, decode KV > 0.85 | decode (KV saturation) |
| TTFT gate failed, neither | prefill |
| gate holds, both idle | neither: climb concurrency |

Used by seed_arms.py (round one) and plan_arms.py (every later round).
"""

from __future__ import annotations

import math
import sys
import pathlib
from typing import Any

_NEXUS = pathlib.Path(__file__).resolve().parents[3]
if str(_NEXUS) not in sys.path:
    sys.path.insert(0, str(_NEXUS))

from framework.deploy import render  # noqa: E402

# Default sizing priors for Qwen3.5-397B-A17B-FP8 on one tpu7x 2x2x1 host
# per engine. Replaced by the recipe's `measured` block as soon as it
# carries them.
PRIOR_SESSIONS_PER_PREFILL = 96.0
PRIOR_SESSIONS_PER_DECODE = 112.0
PREFILL_HIT_CLIFF = 0.60       # GPU prefix hit below this: cache thrash
DECODE_KV_CLIFF = 0.85         # decode KV util above this: saturation
DECODE_RUNNING_FRACTION = 0.9  # running >= this x max_num_seqs x pods
AGENTX_PROFILES = ("agentx-turn-priority", "agentx-session-affinity")
# Per-role engine knobs to explore: each is an arm when the recipe maps it.
ENGINE_ARMS = (
    ("decode", "max_num_seqs", 48,
     "decode admits 48 sequences instead of 32 without flow control"),
    ("prefill", "kv_cpu_offload_bytes", 256,
     "a 256 GiB host prefix tier behind each prefill engine lifts the"
     " prefill cliff"),
    ("decode", "hma_max_inflight_pulls", 16,
     "more concurrent KV pulls per decode engine cut the time finished"
     " prefill KV sits on the prefill pods"),
)


def sessions_per_engine(recipe: dict[str, Any]) -> dict[str, Any]:
    measured = recipe.get("measured") or {}
    p = measured.get("sessions_per_prefill_engine")
    d = measured.get("sessions_per_decode_engine")
    return {
        "prefill": float(p) if p else PRIOR_SESSIONS_PER_PREFILL,
        "decode": float(d) if d else PRIOR_SESSIONS_PER_DECODE,
        "provenance": "measured" if (p and d) else "prior",
    }


def opening_shape(concurrency: int, recipe: dict[str, Any],
                  chips_per: int, max_chips: int | None) -> dict[str, Any]:
    """P and D for `concurrency` sessions from the per-engine capacities,
    clipped to the chip budget (decode first when something must give:
    an overloaded decode pool wrecks prefill caching too)."""
    cap = sessions_per_engine(recipe)
    p = max(1, math.ceil(concurrency / cap["prefill"]))
    d = max(1, math.ceil(concurrency / cap["decode"]))
    budget = (int(max_chips) // chips_per) if max_chips else None
    note = (f"{concurrency} sessions / {cap['prefill']:g} per prefill engine"
            f" -> {p}P; / {cap['decode']:g} per decode engine -> {d}D"
            f" ({cap['provenance']})")
    if budget is not None and p + d > budget:
        while p + d > budget and p > 1:
            p -= 1
        while p + d > budget and d > 1:
            d -= 1
        note += f"; clipped to {budget} slices"
    return {"prefill_replicas": p, "decode_replicas": d,
            "chips": (p + d) * chips_per, "note": note,
            "provenance": cap["provenance"]}


def attribute(report: dict[str, Any] | None, incumbent: dict[str, Any]
              ) -> dict[str, Any]:
    """Which pool bounds the measured point, from the sampled role metrics
    and the gate, with the evidence."""
    if not report:
        return {"role": "none", "kind": "none", "evidence": [],
                "failed_gates": []}
    tenants = report.get("tenants") or {}
    tenant = next(iter(tenants.values()), {}) if tenants else {}
    reps = tenant.get("replicates") or []
    rep = reps[-1] if reps else {}
    roles = rep.get("roles") or {}
    ax = rep.get("agentx") or {}
    failed = [f"{name}.{key}" for name, block in tenants.items()
              for key, entry in (block.get("slo") or {}).items()
              if entry and entry.get("pass") is False]
    evidence: list[str] = []
    role, kind = "none", "none"
    pf, dec = roles.get("prefill") or {}, roles.get("decode") or {}
    hit = ax.get("gpu_prefix_hit_rate")
    if hit is None:
        hit = pf.get("prefix_hit_rate")
    max_seqs = int(render.role_engine(incumbent, "decode").get(
        "max_num_seqs") or 0)
    dec_pods = int(dec.get("pods") or 1)
    running = dec.get("requests_running_max") or 0.0
    kv = dec.get("kv_util_max") or 0.0
    if (kv >= DECODE_KV_CLIFF) or (
            max_seqs and running >= DECODE_RUNNING_FRACTION * max_seqs *
            dec_pods):
        role, kind = "decode", "memory-capacity" if kv >= DECODE_KV_CLIFF \
            else "scheduler-batching"
        evidence.append(f"decode: running_max={running:g} of"
                        f" {max_seqs or '?'} x {dec_pods} pods,"
                        f" kv_util_max={kv:.2f}")
    if hit is not None and hit < PREFILL_HIT_CLIFF:
        evidence.append(f"prefill: GPU prefix hit {hit:.2f} <"
                        f" {PREFILL_HIT_CLIFF}")
        if role == "none":
            role, kind = "prefill", "memory-capacity"
    waiting = pf.get("requests_waiting_max") or 0.0
    if waiting > 2 * max(1, int(pf.get("pods") or 1)):
        evidence.append(f"prefill: requests_waiting_max={waiting:g}")
        if role == "none":
            role, kind = "prefill", "scheduler-batching"
    if role == "none" and any(".ttft_" in g for g in failed):
        role, kind = "prefill", "scheduler-batching"
        evidence.append(f"TTFT gate failed ({', '.join(failed)}) with no"
                        " saturation signature: prefill queueing")
    return {"role": role, "kind": kind, "evidence": evidence,
            "failed_gates": failed,
            "observed": {"gpu_prefix_hit_rate": hit,
                         "decode_running_max": running,
                         "decode_kv_util_max": kv,
                         "prefill_waiting_max": waiting}}


def next_point(ladder: list[int], current: int | None, *,
               direction: str = "up") -> int | None:
    """The next concurrency on the spec's ladder, or None at its end."""
    if not ladder:
        return None
    if current is None:
        return ladder[0]
    if direction == "up":
        higher = [c for c in ladder if c > current]
        return higher[0] if higher else None
    lower = [c for c in ladder if c < current]
    return lower[-1] if lower else None


def _mapped(recipe: dict[str, Any], knob: str, role: str) -> bool:
    """Whether the recipe can set this knob on this role at all."""
    launch = recipe.get("launch") or {}
    spec = (launch.get("knobs") or {}).get(knob)
    if spec is None:
        return knob in render.ENGINE_FLAGS or knob in render.ENGINE_SWITCHES
    roles = spec.get("roles")
    return not roles or role in roles


def router_profiles_for(recipe: dict[str, Any]) -> tuple[str, str | None]:
    """(baseline profile, A/B profile): the InferenceX configs when the
    recipe names the router images they need, else the v0.11.0 P/D set."""
    if (recipe.get("router_images") or {}).get("epp"):
        return "agentx-turn-priority", "agentx-session-affinity"
    return "sticky-until-saturated", None


def seed(profile: dict[str, Any], recipe: dict[str, Any], *,
         max_chips: int | None, capabilities: dict[str, Any] | None,
         add) -> dict[str, Any]:
    """Round one of an AgentX study, through the caller's `add`.

    The baseline is the submission's shape at the ladder's first point:
    P/D sized by sessions per engine, the recipe's router. Then the
    operating-point, router A/B, and mapped engine-knob arms.
    """
    base = render.default_config(recipe)
    chips_per = render.chips_per_replica(recipe, base)
    ladder = list(profile["agentx"]["concurrency"])
    c0 = ladder[0]
    opening = opening_shape(c0, recipe, chips_per, max_chips)
    p_n, d_n = opening["prefill_replicas"], opening["decode_replicas"]
    tp = base["topology"]["decode"].get("tp")
    baseline_profile, ab_profile = router_profiles_for(recipe)
    base_sets = {"topology.mode": "disaggregated",
                 "topology.prefill": {"replicas": p_n, "tp": tp},
                 "topology.decode.replicas": d_n,
                 "router.profile": baseline_profile,
                 "workload.concurrency": c0}
    add("baseline", "baseline",
        f"{p_n}P:{d_n}D behind {baseline_profile} at {c0} sessions"
        " qualifies (TTFT p50 gate) and sets the total tok/s/chip to beat",
        share=1.0, headroom=0.0, sets=base_sets,
        rationale=opening["note"],
        prerequisites={"capabilities": ["pd_connector"],
                       "chips": opening["chips"], "fleet_mode": "managed",
                       "sizing_provenance": opening["provenance"]})
    nxt = next_point(ladder, c0)
    if nxt is not None:
        add("c1", "operating_point.concurrency",
            f"{nxt} sessions raise total tok/s/chip over {c0} while TTFT"
            " p50 stays under the gate",
            share=1.0, headroom=min(0.5, (nxt - c0) / max(c0, 1)),
            sets={"workload.concurrency": nxt},
            rationale="closed-loop throughput grows with sessions until an"
                      " engine saturates; the ladder is the spec's")
    if ab_profile:
        add("r1", "router.profile",
            f"{ab_profile} (no flow control) beats {baseline_profile}:"
            " the EPP's saturation detector parks agentic requests",
            share=0.4, headroom=0.35,
            sets={"router.profile": ab_profile},
            rationale="A/B flowControl on the AgentX router profile")
    for i, (role, knob, value, claim) in enumerate(ENGINE_ARMS):
        if not _mapped(recipe, knob, role):
            continue
        add(f"e{i + 1}", "engine.role_knob", claim,
            share=0.3, headroom=0.15 - 0.03 * i,
            sets={f"topology.{role}.engine.{knob}": value},
            rationale=f"reaches the engine through recipe.launch.knobs.{knob}",
            prerequisites={"fleet_mode": "managed"})
    return {"opening": opening, "ladder": ladder,
            "router": {"baseline": baseline_profile, "ab": ab_profile}}


def propose(profile: dict[str, Any], recipe: dict[str, Any],
            incumbent: dict[str, Any], *, decided: list[dict[str, Any]],
            incumbent_report: dict[str, Any] | None,
            max_chips: int | None, chips_per: int, add) -> dict[str, Any]:
    """Later rounds: climb or back off on concurrency by the incumbent's
    verdict and attribution, A/B the router once, map the engine knobs,
    then move the shape along the bound axis."""
    ladder = list(profile["agentx"]["concurrency"])
    cur = (incumbent.get("workload") or {}).get("concurrency") or ladder[0]
    attribution = attribute(incumbent_report, incumbent)
    admissible = bool((incumbent_report or {}).get("admissible"))
    tried = {str(d.get("sets", {}).get("workload.concurrency"))
             for d in decided if "workload.concurrency" in (d.get("sets") or {})}
    # The rung the incumbent shape could not hold (an operating-point arm
    # above `cur` that failed its gate): the shape and bound-role arms
    # claim *that* point, since the move exists to sustain it. Sets on a
    # cold path cost nothing extra; on a warm path the operating point is
    # a bench-only path anyway.
    cliff = max((int(d["sets"]["workload.concurrency"]) for d in decided
                 if str(d.get("arm_kind")) == "operating_point.concurrency"
                 and d.get("verdict") in ("invalid", "loss")
                 and int((d.get("sets") or {}).get("workload.concurrency")
                         or 0) > cur), default=None)
    up = next_point(ladder, cur, direction="up")
    down = next_point(ladder, cur, direction="down")
    if admissible and up is not None and str(up) not in tried:
        add("operating_point.concurrency",
            f"{up} sessions raise total tok/s/chip over {cur} while TTFT"
            " p50 stays under the gate",
            share=1.0, headroom=min(0.5, (up - cur) / max(cur, 1)),
            sets={"workload.concurrency": up},
            rationale="the incumbent qualifies; the next ladder point")
    if not admissible and attribution["role"] == "none" and down is not None \
            and str(down) not in tried:
        add("operating_point.concurrency",
            f"{down} sessions bring TTFT p50 back under the gate",
            share=1.0, headroom=0.1, sets={"workload.concurrency": down},
            rationale="gate failed with no pool saturated: the point sits"
                      " past the fleet's knee")
    baseline_profile, ab_profile = router_profiles_for(recipe)
    cur_profile = incumbent["router"]["profile"]
    for prof in (ab_profile, baseline_profile):
        if prof and prof != cur_profile and not any(
                (d.get("sets") or {}).get("router.profile") == prof
                for d in decided):
            add("router.profile", f"{prof} beats {cur_profile} at {cur}"
                                  " sessions",
                share=0.4, headroom=0.3 if prof == ab_profile else 0.1,
                sets={"router.profile": prof},
                rationale="flow-control A/B on the AgentX router profile")
    for i, (role, knob, value, claim) in enumerate(ENGINE_ARMS):
        if not _mapped(recipe, knob, role):
            continue
        if render.role_engine(incumbent, role).get(knob) is not None:
            continue
        bound = attribution["role"]
        head = 0.2 if bound == role else 0.1 - 0.02 * i
        knob_sets = {f"topology.{role}.engine.{knob}": value}
        if cliff and bound == role:
            knob_sets["workload.concurrency"] = cliff
        add("engine.role_knob", claim, share=0.3, headroom=max(0.02, head),
            sets=knob_sets,
            rationale=f"{role} is {'the bound pool' if bound == role else 'not the bound pool'};"
                      f" reaches the engine through recipe.launch.knobs.{knob}",
            prerequisites={"fleet_mode": "managed"})
    if incumbent["topology"].get("mode") == "disaggregated":
        p_cur = int(incumbent["topology"]["prefill"]["replicas"])
        d_cur = int(incumbent["topology"]["decode"]["replicas"])
        bound = attribution["role"] if attribution["role"] in (
            "prefill", "decode") else "none"
        if bound == "none" and admissible:
            # Nothing is saturated and the gate holds: a leaner shape earns
            # more per chip before more sessions do.
            neighbours = [n for n in render_neighbours(
                p_cur, d_cur, "none", chips_per, max_chips)]
        else:
            neighbours = render_neighbours(p_cur, d_cur, bound, chips_per,
                                           max_chips)
        for i, nb in enumerate(neighbours):
            np_, nd = nb["prefill_replicas"], nb["decode_replicas"]
            shape_sets = {"topology.prefill.replicas": np_,
                          "topology.decode.replicas": nd}
            at = cur
            if cliff and bound != "none":
                shape_sets["workload.concurrency"] = cliff
                at = cliff
            add("topology.pd_shape",
                f"{np_}P:{nd}D beats {p_cur}P:{d_cur}D on total tok/s/chip"
                f" at {at} sessions ({nb['why']})"
                + (f"; {p_cur}P:{d_cur}D broke the gate there" if at != cur
                   else ""),
                share=0.5, headroom=max(0.05, 0.25 - 0.05 * i),
                sets=shape_sets,
                rationale="; ".join([nb["why"]] + attribution["evidence"]),
                prerequisites={"capabilities": ["pd_connector"],
                               "chips": nb["chips"], "fleet_mode": "managed",
                               "attribution": attribution})
    return {"attribution": attribution, "concurrency": cur, "cliff": cliff,
            "ladder": ladder, "admissible": admissible}


def render_neighbours(p: int, d: int, bound: str, chips_per: int,
                      max_chips: int | None) -> list[dict[str, Any]]:
    """size_pd.neighbours without importing it circularly."""
    from size_pd import neighbours  # noqa: PLC0415  (sibling script)
    return neighbours(p, d, bound, chips_per=chips_per, max_chips=max_chips)
