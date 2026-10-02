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
"""PreToolUse hook on `delegate`: an llm-d arm may not outspend the chip budget.

An `llmd-arm-request` names the arm's `sets` (the deltas against the
incumbent), the incumbent config, the recipe and the workload spec. From
those four the chips the arm would hold are arithmetic: replicas per pool
times the recipe's chips per replica (a prefill pool of its own under P/D).
The budget is the workload spec's `budget.max_chips`, less what production
releases in the job store already hold. An arm that would exceed it is
denied before any executor renders it, with the numbers in the reason.

The rule is deliberately narrow:

- only delegations to `llmd-arm-executor` are examined;
- an adopted fleet costs the study nothing (`chips_held` is 0 for it, the
  servers are the customer's), so adopted arms always pass;
- any file the hook cannot read or parse means allow -- a guard that fails
  closed on a missing path would break every delegation the moment a
  workspace is laid out differently, and the executor itself refuses what
  it cannot render.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

EXECUTOR = "llmd-arm-executor"


def json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object embedded in `text`, in order."""
    found: list[dict[str, Any]] = []
    if not isinstance(text, str):
        return found
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        doc = json.loads(text[start:i + 1])
                    except ValueError:
                        doc = None
                    if isinstance(doc, dict):
                        found.append(doc)
                    start = -1
    return found


def chips_per_replica(recipe: dict[str, Any]) -> int:
    """Chips one replica holds, from the recipe's topology (product of the
    `AxBxC` slice) or its explicit `chips_per_replica`."""
    accel = recipe.get("accelerator") or {}
    explicit = accel.get("chips_per_replica") or recipe.get("chips_per_replica")
    if explicit:
        return int(explicit)
    topo = str(accel.get("tpu_topology") or "")
    chips = 1
    for part in topo.lower().split("x"):
        if part.strip().isdigit():
            chips *= int(part)
    return max(1, chips)


def arm_chips(request: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """Chips the arm would hold, or None when it cannot be computed."""
    try:
        recipe = json.loads(pathlib.Path(os.path.expanduser(
            request["recipe_path"])).read_text(encoding="utf-8"))
        config = json.loads(pathlib.Path(os.path.expanduser(
            request["incumbent_config_path"])).read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, TypeError):
        return None, {}
    fleet = recipe.get("fleet") or {}
    if fleet.get("mode") == "adopted":
        return 0, {"fleet_mode": "adopted"}
    topology = dict(config.get("topology") or {})
    decode = dict(topology.get("decode") or {})
    prefill = dict(topology.get("prefill") or {})
    mode = topology.get("mode")
    for path, value in (request.get("sets") or {}).items():
        parts = str(path).split(".")
        if parts[:2] == ["topology", "decode"] and parts[-1] == "replicas":
            decode["replicas"] = value
        elif parts[:2] == ["topology", "prefill"] and parts[-1] == "replicas":
            prefill["replicas"] = value
        elif parts == ["topology", "mode"]:
            mode = value
        elif parts == ["topology", "prefill"] and isinstance(value, dict):
            prefill.update(value)
    per = chips_per_replica(recipe)
    decode_n = int(decode.get("replicas") or 1)
    total = decode_n * per
    if mode == "disaggregated":
        total += int(prefill.get("replicas") or 0) * per
    return total, {"chips_per_replica": per, "decode_replicas": decode_n,
                   "prefill_replicas": int(prefill.get("replicas") or 0),
                   "mode": mode}


def budget_chips(request: dict[str, Any]) -> int | None:
    try:
        spec = json.loads(pathlib.Path(os.path.expanduser(
            request["workload_spec_path"])).read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, TypeError):
        return None
    budget = (spec.get("budget") or {}).get("max_chips")
    return int(budget) if budget else None


def production_chips() -> int:
    """Chips production releases in the job store hold right now.

    Records live at `<TPU_AGENT_SUITE_HOME>/runs/jobs/<id>/job.json`; a
    release record has backend `gke_release` and `extra.production`.
    """
    home = pathlib.Path(os.environ.get("TPU_AGENT_SUITE_HOME",
                                       pathlib.Path.home() / ".tpu-agent-suite"))
    jobs = home / "runs" / "jobs"
    total = 0
    for path in (jobs.glob("*/job.json") if jobs.is_dir() else []):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        extra = doc.get("extra") or {}
        if doc.get("backend") != "gke_release" or not extra.get("production"):
            continue
        if doc.get("status") in ("CANCELLED", "FAILED", "COMPLETED"):
            continue
        total += int(extra.get("chips_held") or 0)
    return total


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("tool") or "") != "delegate":
        return {"decision": "allow"}
    subagent = str(payload.get("subagent") or "")
    if subagent.split(":")[-1] != EXECUTOR:
        return {"decision": "allow"}
    args = payload.get("args") or {}
    candidates = json_objects(args.get("prompt") or "")
    request = next((c for c in reversed(candidates)
                    if "incumbent_config_path" in c and "recipe_path" in c),
                   None)
    if request is None:
        return {"decision": "allow"}
    chips, detail = arm_chips(request)
    budget = budget_chips(request)
    if chips is None or budget is None or chips == 0:
        return {"decision": "allow"}
    held = production_chips()
    if chips + held <= budget:
        return {"decision": "allow"}
    return {
        "decision": "deny",
        "reason": (
            f"Blocked: arm {request.get('arm', '?')} would hold {chips} chips"
            f" ({detail.get('decode_replicas')} decode"
            + (f" + {detail.get('prefill_replicas')} prefill"
               if detail.get("mode") == "disaggregated" else "")
            + f" x {detail.get('chips_per_replica')} per replica) while"
            f" production holds {held}; the workload spec's budget.max_chips"
            f" is {budget}.\n\nThe arm is not pruned: record it as blocked"
            " with `blocked_on: chip budget` and the numbers above, so the"
            " final table says what the workload would want. Raising the"
            " budget is the user's decision, in the workload spec."),
    }


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        print(json.dumps({"decision": "allow"}))
        return
    print(json.dumps(decide(payload)))


if __name__ == "__main__":
    main()
