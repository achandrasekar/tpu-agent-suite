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
"""Bring an engine study's win back into the serving recipe.

The second half of the fleet <-> engine seam. When the engine study
`engine_escalation.py` opened has closed, its wins are serving-config
deltas recorded as `target_parameters.serving` (`args`, `env`, `image`)
on its arms -- in the hypothesis bank sidecar `hypothesis_bank.py` keeps
beside the study, which is the one place an arm's parameters survive the
round record's closed schema. This script:

* picks the engine study's incumbent: the winning arm with the largest
  `delta_pct` (ties and losses leave the recipe alone);
* applies its serving delta to the recipe -- args merged flag by flag
  (a `null` removes a flag), env merged, image replaced -- and refreshes
  `measured.peak_prefill_throughput` / `peak_decode_throughput` from the
  arm's metric when it is that role's tokens/s;
* writes `serving-recipe.v<N+1>.json` with `source.engine_study_id`, the
  winning arm and its execution id, and an amendment for the llm-d study:
  the runtime refuses to replace a baseline that arms were measured
  against ("amend the protocol and start a new study"), so the amendment
  names the **successor** fleet study (`<study_id>-v<N+1>`) and the
  baseline `fingerprint.commit_sha` (= the new recipe's digest) its first
  measurement must carry. The next fleet round opens that study and
  re-baselines the incumbent shape on the new engine instead of comparing
  across recipes.

A study with no win writes `recipe-bump.json` saying so and leaves the
recipe untouched; the orchestrator then marks the role exhausted. Usage:

    recipe_bump.py --recipe <serving-recipe.json> \\
        --engine-study-state <status.json> [--hypothesis-bank <bank.json>] \\
        --escalation <engine-escalation.json> --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

BUMP_FILE = "recipe-bump.json"
AMENDMENT_FILE = "llmd-protocol-amendment.json"
# Role -> the recipe's measured field the engine metric refreshes.
MEASURED_FIELD = {"prefill": ("prompt_tokens_per_s", "peak_prefill_throughput"),
                  "decode": ("generation_tokens_per_s",
                             "peak_decode_throughput")}
_VERSION_RE = re.compile(r"\.v(\d+)\.json$")


def _digest(doc: Any) -> str:
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _flag_name(arg: str) -> str:
    return str(arg).split("=", 1)[0]


def winning_arm(state: dict[str, Any]) -> dict[str, Any] | None:
    """The engine study's incumbent: largest measured win, or None."""
    best = None
    for arm in state.get("arms") or []:
        result = arm.get("result") or {}
        if arm.get("status") != "decided" or result.get("verdict") != "win":
            continue
        delta = float(((result.get("metric") or {}).get("delta_pct")) or 0)
        if best is None or delta > best[0]:
            best = (delta, arm)
    return best[1] if best else None


def serving_delta(arm_id: str, bank: dict[str, Any] | None
                  ) -> dict[str, Any]:
    """`target_parameters.serving` for an arm, from the bank sidecar.

    The bank merges a hypothesis's parameters with its variant's, so the
    arm entry is the complete delta; a missing entry is an empty delta,
    which `apply` reports rather than guessing at.
    """
    arms = (bank or {}).get("arms") or {}
    entry = arms.get(arm_id) or {}
    params = entry.get("target_parameters") or {}
    serving = params.get("serving") or {}
    return {"args": dict(serving.get("args") or {}),
            "env": dict(serving.get("env") or {}),
            "image": serving.get("image") or ""}


def apply_delta(recipe: dict[str, Any], delta: dict[str, Any]
                ) -> tuple[dict[str, Any], list[str]]:
    """The recipe with the delta applied, and a line per change."""
    out = copy.deepcopy(recipe)
    changes: list[str] = []
    args = list(out.get("args") or [])
    for flag, value in (delta.get("args") or {}).items():
        before = [a for a in args if _flag_name(a) == flag]
        args = [a for a in args if _flag_name(a) != flag]
        if value is None:
            if before:
                changes.append(f"args: removed {flag}")
            continue
        if value is True:
            args.append(flag)
            rendered = flag
        elif value is False:
            rendered = f"{flag}=false"
            args.append(rendered)
        else:
            rendered = f"{flag}={value}"
            args.append(rendered)
        changes.append(f"args: {before[0] if before else '(absent)'} ->"
                       f" {rendered}")
    out["args"] = args
    if delta.get("env"):
        env = dict(out.get("env") or {})
        for key, value in delta["env"].items():
            if value is None:
                if key in env:
                    env.pop(key)
                    changes.append(f"env: removed {key}")
            else:
                changes.append(f"env: {key}={env.get(key, '(absent)')} ->"
                               f" {value}")
                env[key] = value
        out["env"] = env
    if delta.get("image") and delta["image"] != out.get("image"):
        changes.append(f"image: {out.get('image')} -> {delta['image']}")
        out["image"] = delta["image"]
    return out, changes


def refresh_measured(recipe: dict[str, Any], role: str,
                     metric: dict[str, Any] | None) -> list[str]:
    """Carry the engine study's measured tokens/s into `measured` when it
    is this role's metric; the next fleet sizing reads it."""
    if not metric or role not in MEASURED_FIELD:
        return []
    name, field = MEASURED_FIELD[role]
    if metric.get("name") != name or metric.get("value") is None:
        return []
    measured = dict(recipe.get("measured") or {})
    before = measured.get(field)
    measured[field] = float(metric["value"])
    recipe["measured"] = measured
    return [f"measured.{field}: {before} -> {measured[field]}"]


def next_version_path(recipe_path: pathlib.Path) -> pathlib.Path:
    """`serving-recipe.json` -> `serving-recipe.v2.json`, `.v2` -> `.v3`."""
    name = recipe_path.name
    m = _VERSION_RE.search(name)
    if m:
        n = int(m.group(1)) + 1
        return recipe_path.with_name(_VERSION_RE.sub(f".v{n}.json", name))
    stem = name[:-5] if name.endswith(".json") else name
    return recipe_path.with_name(f"{stem}.v2.json")


_SUCCESSOR_RE = re.compile(r"-v\d+$")


def successor_study_id(fleet_study_id: str, recipe_path: pathlib.Path
                       ) -> str:
    """`acme-chat` + `serving-recipe.v2.json` -> `acme-chat-v2`.

    The fleet study that measured arms against the old recipe is closed,
    not amended: the runtime refuses a new baseline under measured arms,
    and the deltas in the old ledger stay comparable to each other. The
    successor carries the recipe version so a reader can line the two up.
    """
    m = _VERSION_RE.search(recipe_path.name)
    version = int(m.group(1)) if m else 2
    base = _SUCCESSOR_RE.sub("", fleet_study_id)
    suffix = f"-v{version}"
    return base[:64 - len(suffix)] + suffix


def bump(args: argparse.Namespace) -> dict[str, Any]:
    recipe_path = pathlib.Path(args.recipe)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    state = json.loads(pathlib.Path(args.engine_study_state).read_text(
        encoding="utf-8"))
    escalation = json.loads(pathlib.Path(args.escalation).read_text(
        encoding="utf-8")) if args.escalation else {}
    bank = None
    if args.hypothesis_bank and pathlib.Path(args.hypothesis_bank).is_file():
        bank = json.loads(pathlib.Path(args.hypothesis_bank).read_text(
            encoding="utf-8"))
    role = str(escalation.get("role") or "aggregated")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    win = winning_arm(state)
    doc: dict[str, Any] = {
        "schema": "recipe-bump",
        "engine_study_id": state.get("study_id"),
        "engine_study_status": state.get("status"),
        "role": role,
        "recipe_before": str(recipe_path.resolve()),
        "recipe_digest_before": _digest(recipe),
    }
    if win is None:
        doc.update({"status": "no_win", "changes": [],
                    "summary": f"engine study {state.get('study_id')} closed"
                               f" without a win; recipe unchanged, role"
                               f" {role} exhausted"})
        (out / BUMP_FILE).write_text(json.dumps(doc, indent=2) + "\n",
                                     encoding="utf-8")
        return doc
    delta = serving_delta(win["arm"], bank)
    new_recipe, changes = apply_delta(recipe, delta)
    if changes:
        # Only a reproducible win refreshes the measured block: a number
        # without the change that produced it would mis-size the next
        # fleet.
        changes += refresh_measured(new_recipe, role,
                                    (win.get("result") or {}).get("metric"))
    if not changes:
        doc.update({"status": "no_delta", "winning_arm": win["arm"],
                    "changes": [],
                    "summary": f"arm {win['arm']} won but recorded no"
                               " serving delta (target_parameters.serving);"
                               " the win is reported, not bumped"})
        (out / BUMP_FILE).write_text(json.dumps(doc, indent=2) + "\n",
                                     encoding="utf-8")
        return doc
    source = dict(new_recipe.get("source") or {})
    source.update({
        "engine_study_id": state.get("study_id"),
        "engine_arm": win["arm"],
        "engine_execution_id": (win.get("result") or {}).get("execution_id"),
        "bumped_from": str(recipe_path.resolve()),
        "escalation": escalation.get("escalation"),
    })
    new_recipe["source"] = source
    new_path = next_version_path(recipe_path)
    new_path.write_text(json.dumps(new_recipe, indent=2) + "\n",
                        encoding="utf-8")
    fleet_study = str(escalation.get("study_id") or "")
    successor = successor_study_id(fleet_study, new_path) if fleet_study else ""
    amendment = {
        "reason": f"recipe bumped by engine study {state.get('study_id')}"
                  f" (arm {win['arm']}); every earlier measurement is on"
                  " the old engine and the incumbent shape is re-baselined",
        "apply": "successor_study",
        "predecessor_study_id": fleet_study,
        "successor_study_id": successor,
        "fingerprint": {"commit_sha": _digest(new_recipe)},
        "recipe_path": str(new_path.resolve()),
    }
    (out / AMENDMENT_FILE).write_text(json.dumps(amendment, indent=2) + "\n",
                                      encoding="utf-8")
    doc.update({
        "status": "bumped",
        "winning_arm": win["arm"],
        "delta_pct": ((win.get("result") or {}).get("metric") or {}).get(
            "delta_pct"),
        "serving_delta": delta,
        "changes": changes,
        "recipe_after": str(new_path.resolve()),
        "recipe_digest_after": _digest(new_recipe),
        "amendment_path": str(out / AMENDMENT_FILE),
        "summary": f"recipe -> {new_path.name}: " + "; ".join(changes),
    })
    (out / BUMP_FILE).write_text(json.dumps(doc, indent=2) + "\n",
                                 encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--engine-study-state", required=True,
                        help="`study status` JSON of the engine study")
    parser.add_argument("--hypothesis-bank",
                        help="the engine study's hypothesis_bank.json")
    parser.add_argument("--escalation", help="engine-escalation.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    doc = bump(args)
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
