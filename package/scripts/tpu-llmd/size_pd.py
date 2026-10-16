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
"""Size a prefill/decode split, and say which pool is bound.

The xPyD search is a walk on a lattice of shapes `(P prefill replicas,
D decode replicas)`. Walking it blind costs a fleet per step, so this
module does two cheaper things first:

* **Opening point.** From the workload's token demand at `target_qps`
  (prefill tokens/s = qps x ISL x (1 - prefix hit rate); decode tokens/s =
  qps x OSL) and the per-slice capacities the recipe measured
  (`measured.peak_prefill_throughput`, `measured.peak_decode_throughput`),
  the smallest `(P, D)` that carries the demand with 20 % headroom. When
  the recipe has no per-role capacities the opening point is the token
  ratio applied to the incumbent's chip count -- a heuristic, labelled as
  one, that the first measured shape corrects.

* **Attribution.** From a measured shape's per-role fleet metrics and its
  bench report: which pool is bound (queueing, KV pressure, or the gate
  that failed), so the planner moves that axis rather than both. The same
  reading is what the engine-escalation path names as its role.

Everything here is a model; `ceiling_provenance` says so on every arm it
feeds, and nothing is written to the study. Usage:

    size_pd.py --workload <llmd-workload-spec.json> --recipe <serving-recipe.json>
               --incumbent <llmd-config.json> [--profile <metrics.json>]
               [--report <llmd-bench-report.json>] [--max-chips N] --out <dir>
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
    from framework.deploy import render, workload  # noqa: E402
except ImportError:  # shipped runtime: `deploy` is top-level
    from deploy import render, workload  # type: ignore  # noqa: E402

HEADROOM = 0.8          # size for 80 % utilisation of each pool
QUEUE_BOUND = 0.5       # mean requests waiting that counts as queueing
KV_BOUND = 0.85         # max KV occupancy that counts as capacity pressure
SIZING_FILE = "pd-sizing.json"


def _read(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _metric(doc: dict[str, Any] | None, name: str,
            role: str | None = None) -> float | None:
    if not doc:
        return None
    block = doc.get("metrics") or {}
    if role:
        block = (doc.get("roles") or {}).get(role) or {}
    value = (block.get(name) or {}).get("value")
    return float(value) if value is not None else None


# -- demand -------------------------------------------------------------------


def demand(profile: dict[str, Any], hit_rate: float | None) -> dict[str, Any]:
    """Prefill and decode tokens/s the workload asks for at target_qps.

    Prefill demand counts only uncached prompt tokens: the hit rate is the
    measured one when a fleet has run, else the catalog's shared-prefix
    hint (0.5 for a shared-prefix workload, 0 otherwise).
    """
    shape = profile.get("shape") or {}
    isl = float((shape.get("input_sequence_length") or {}).get("mean") or 0)
    osl = float((shape.get("output_sequence_length") or {}).get("mean") or 0)
    qps = float(profile["rate"]["target_qps"])
    if hit_rate is None:
        signals = _signals(profile)
        hit_rate = 0.5 if signals["shared_prefix"] else 0.0
        hit_source = "assumed"
    else:
        hit_source = "measured"
    hit_rate = max(0.0, min(0.95, float(hit_rate)))
    return {
        "target_qps": qps,
        "isl_mean": isl,
        "osl_mean": osl,
        "prefix_hit_rate": hit_rate,
        "prefix_hit_source": hit_source,
        "prefill_tokens_per_s": qps * isl * (1.0 - hit_rate),
        "decode_tokens_per_s": qps * osl,
    }


def _signals(profile: dict[str, Any]) -> dict[str, bool]:
    shape = profile.get("shape") or {}
    meta = shape.get("metadata") or {}
    opts = " ".join(meta.get("suitable_optimizations") or []).lower()
    shared = bool((shape.get("system_prompt") or {}).get("mean")) or (
        "shared_prefix" in shape) or ("prefix" in opts)
    return {"shared_prefix": shared,
            "pd_suggested": "p/d" in opts or "disaggregation" in opts}


# -- capacity -----------------------------------------------------------------


def capacity(recipe: dict[str, Any]) -> dict[str, Any] | None:
    """Per-slice prefill and decode tokens/s the recipe measured, or None.

    Both numbers are single-replica facts `tpu-onboarding` (or an engine
    study's final E2E) records; the planner never invents them.
    """
    measured = recipe.get("measured") or {}
    prefill = measured.get("peak_prefill_throughput")
    decode = measured.get("peak_decode_throughput")
    if not prefill or not decode:
        return None
    return {"prefill_tokens_per_s_per_slice": float(prefill),
            "decode_tokens_per_s_per_slice": float(decode),
            "source": "recipe.measured"}


# -- opening point --------------------------------------------------------------


def opening_shape(profile: dict[str, Any], recipe: dict[str, Any],
                  incumbent: dict[str, Any], *,
                  hit_rate: float | None,
                  max_chips: int | None) -> dict[str, Any]:
    """The `(P, D)` the search opens with, and how it was derived."""
    dem = demand(profile, hit_rate)
    cap = capacity(recipe)
    chips_per = render.chips_per_replica(recipe, incumbent)
    budget_slices = (int(max_chips) // chips_per) if max_chips else None
    if cap:
        p = max(1, math.ceil(dem["prefill_tokens_per_s"] /
                             (cap["prefill_tokens_per_s_per_slice"] * HEADROOM)))
        d = max(1, math.ceil(dem["decode_tokens_per_s"] /
                             (cap["decode_tokens_per_s_per_slice"] * HEADROOM)))
        provenance = "modeled"
        note = ("sized from the recipe's measured per-role throughput at"
                f" {int(HEADROOM * 100)} % utilisation")
    else:
        # Token-ratio heuristic over the slices the incumbent already
        # holds: prefill's share of compute is its share of tokens, decode
        # tokens weighted equally. The first measured shape corrects it.
        total = max(2, int(incumbent["topology"]["decode"]["replicas"]) +
                    int((incumbent["topology"].get("prefill") or {})
                        .get("replicas") or 0))
        tokens = dem["prefill_tokens_per_s"] + dem["decode_tokens_per_s"]
        share = (dem["prefill_tokens_per_s"] / tokens) if tokens else 0.5
        p = max(1, min(total - 1, round(total * share)))
        d = max(1, total - p)
        provenance = "heuristic"
        note = ("recipe has no measured.peak_prefill_throughput /"
                " peak_decode_throughput; opened on the token ratio over"
                f" the incumbent's {total} slice(s)")
    bound_by = None
    if budget_slices is not None and p + d > budget_slices:
        # Shrink the pool with more slack first, never below one.
        while p + d > budget_slices and (p > 1 or d > 1):
            if p >= d and p > 1:
                p -= 1
            elif d > 1:
                d -= 1
            else:
                break
        bound_by = "max_chips"
    return {
        "prefill_replicas": p,
        "decode_replicas": d,
        "chips": (p + d) * chips_per,
        "chips_per_replica": chips_per,
        "provenance": provenance,
        "note": note,
        "bound_by": bound_by,
        "demand": dem,
        "capacity": cap,
    }


# -- attribution ----------------------------------------------------------------


def attribute(profile_doc: dict[str, Any] | None,
              report: dict[str, Any] | None) -> dict[str, Any]:
    """Which pool is bound on the measured shape, with the evidence.

    Reads per-role fleet metrics (`roles.prefill` / `roles.decode` from
    `deploy collect`) and the bench report's gates. Order of evidence:
    a failed gate names the role its latency belongs to (TTFT -> prefill,
    TPOT -> decode); then queueing; then KV pressure. `none` means both
    pools have slack, which is the signal to try a leaner shape.
    """
    evidence: list[str] = []
    role = "none"
    gates = _failed_gates(report)
    if any(g.endswith("ttft_p99_ms") or g.endswith("ttft_p90_ms")
           for g in gates):
        role = "prefill"
        evidence.append(f"TTFT gate failed ({', '.join(gates)})")
    elif any(g.endswith("tpot_p99_ms") or g.endswith("tpot_p90_ms")
             for g in gates):
        role = "decode"
        evidence.append(f"TPOT gate failed ({', '.join(gates)})")
    for cand in ("prefill", "decode"):
        waiting = _metric(profile_doc, "requests_waiting_mean", cand)
        kv_max = _metric(profile_doc, "kv_util_max", cand)
        preempt = _metric(profile_doc, "preempted_per_s", cand)
        if waiting is not None and waiting > QUEUE_BOUND:
            evidence.append(f"{cand}: requests_waiting_mean={waiting:.2f}")
            role = role if role != "none" else cand
        if kv_max is not None and kv_max >= KV_BOUND:
            evidence.append(f"{cand}: kv_util_max={kv_max:.2f}")
            role = role if role != "none" else cand
        if preempt:
            evidence.append(f"{cand}: preempted_per_s={preempt:.3f}")
            role = role if role != "none" else cand
    kind = "none"
    if role != "none":
        kv_max = _metric(profile_doc, "kv_util_max", role)
        preempt = _metric(profile_doc, "preempted_per_s", role)
        kind = ("memory-capacity" if (kv_max or 0) >= KV_BOUND or preempt
                else "scheduler-batching")
    return {"role": role, "kind": kind, "evidence": evidence,
            "failed_gates": gates}


def _failed_gates(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    out: list[str] = []
    for tenant, block in (report.get("tenants") or {}).items():
        for gate, entry in (block.get("slo") or {}).items():
            if entry and entry.get("pass") is False:
                out.append(f"{tenant}.{gate}")
    return out


# -- neighbours -----------------------------------------------------------------


def neighbours(p: int, d: int, bound: str, *, chips_per: int,
               max_chips: int | None) -> list[dict[str, Any]]:
    """Shapes to try next around `(p, d)`, most promising first.

    Bound pool first: move a slice toward it at constant chips (the
    goodput-per-chip-neutral move), then grow it if the budget allows.
    Neither bound: shrink each pool by one; a leaner shape that still
    attains wins on goodput per chip.
    """
    out: list[dict[str, Any]] = []
    budget = (int(max_chips) // chips_per) if max_chips else None

    def add(np_: int, nd: int, why: str) -> None:
        if np_ < 1 or nd < 1 or (np_, nd) == (p, d):
            return
        if budget is not None and np_ + nd > budget:
            return
        if any(n["prefill_replicas"] == np_ and n["decode_replicas"] == nd
               for n in out):
            return
        out.append({"prefill_replicas": np_, "decode_replicas": nd,
                    "chips": (np_ + nd) * chips_per, "why": why})

    if bound == "prefill":
        add(p + 1, d - 1, "prefill bound: move one slice from decode")
        add(p + 1, d, "prefill bound: one more prefill slice")
        add(p, d - 1, "prefill bound: decode has slack")
    elif bound == "decode":
        add(p - 1, d + 1, "decode bound: move one slice from prefill")
        add(p, d + 1, "decode bound: one more decode slice")
        add(p - 1, d, "decode bound: prefill has slack")
    else:
        add(p - 1, d, "both have slack: fewer prefill slices")
        add(p, d - 1, "both have slack: fewer decode slices")
        add(p - 1, d - 1, "both have slack: one fewer of each")
    return out


# -- cli ------------------------------------------------------------------------


def size(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = pathlib.Path(args.workload)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    profile = workload.resolve_workload(spec, spec_dir=spec_path.parent)
    recipe = json.loads(pathlib.Path(args.recipe).read_text(encoding="utf-8"))
    incumbent = render.normalize_config(json.loads(
        pathlib.Path(args.incumbent).read_text(encoding="utf-8")))
    profile_doc = _read(args.profile)
    report = _read(args.report)
    max_chips = args.max_chips or (profile.get("budget") or {}).get("max_chips")
    hit_rate = _metric(profile_doc, "prefix_hit_rate")
    opening = opening_shape(profile, recipe, incumbent, hit_rate=hit_rate,
                            max_chips=max_chips)
    attribution = attribute(profile_doc, report)
    topo = incumbent["topology"]
    current = None
    if topo.get("mode") == "disaggregated":
        current = {"prefill_replicas": int(topo["prefill"]["replicas"]),
                   "decode_replicas": int(topo["decode"]["replicas"])}
        nxt = neighbours(current["prefill_replicas"],
                         current["decode_replicas"], attribution["role"],
                         chips_per=opening["chips_per_replica"],
                         max_chips=max_chips)
    else:
        nxt = []
    return {
        "schema": "llmd-pd-sizing",
        "incumbent_shape": current,
        "opening": opening,
        "attribution": attribution,
        "neighbours": nxt,
        "max_chips": max_chips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--profile", help="fleet metrics.json of the"
                        " incumbent shape (deploy collect)")
    parser.add_argument("--report", help="llmd-bench-report.json of the"
                        " incumbent shape")
    parser.add_argument("--max-chips", type=int)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    doc = size(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / SIZING_FILE).write_text(json.dumps(doc, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
