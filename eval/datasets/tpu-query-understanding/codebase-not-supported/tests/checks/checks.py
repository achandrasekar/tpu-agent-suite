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
"""Trajectory checks for tpu-agent's query-understanding reference.

Every task in this dataset carries a copy of this file. Everything above the
TASK CRITERIA marker is shared and must stay identical across tasks; below it,
each task declares the `@criterion` functions it scores.

The checks read tool calls only, never prose, so they hold across harnesses:
tool names and argument keys are matched for Claude Code, Antigravity and Codex
alike. Prose -- whether the plan and its questions are good -- is the judge's
to score.

Query understanding is a skill, so it runs as tpu-agent and every call it
makes is in tpu-agent's own trajectory. Its first turn plans and asks: the
one delegation it may make before the user answers is to tpu-knowledge-base.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rewardkit import criterion

_REFERENCE = "tpu-agent/query-understanding.md"
# What the reference may consult before the user has confirmed the plan.
_BEFORE_CONFIRMATION = {"tpu-knowledge-base"}

_SUBAGENT_TOOLS = {
    "agent", "task", "invoke_subagent", "subagent", "delegate", "spawn_agent"
}
# Where each harness names the agent a delegation starts.
_TARGET_KEYS = ("subagent_type", "subagent", "agent_name", "type_name",
                "typename", "role")
# Every harness reads a reference as a file; these are their read tools.
_READ_TOOLS = {"read", "view_file", "read_file", "view", "open"}
_PATH_KEYS = ("file_path", "path", "absolutepath", "targetfile", "filename")


def load_trajectory(workspace: Path) -> dict[str, Any]:
    """The run's trajectory, from wherever the harness left it."""
    candidates = [
        os.environ.get("HARBOR_TRAJECTORY_PATH", ""),
        "/logs/agent/trajectory.json", "/logs/trajectory.json",
        str(workspace / "agent" / "trajectory.json"),
        str(workspace / "trajectory.json")
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


def tool_calls(trajectory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(lower-cased tool name, arguments) for every call, in order."""
    raw: list[dict[str, Any]] = []
    for step in trajectory.get("steps") or []:
        if isinstance(step, dict):
            raw.extend(tc for tc in step.get("tool_calls") or []
                       if isinstance(tc, dict))
    raw.extend(tc for tc in trajectory.get("tool_calls") or []
               if isinstance(tc, dict))
    calls = []
    for tc in raw:
        name = str(tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or tc.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


def delegation_targets(trajectory: dict[str, Any]) -> list[str]:
    """The agent each delegation started, in order; "" when none is named."""
    targets = []
    for name, args in tool_calls(trajectory):
        if name not in _SUBAGENT_TOOLS:
            continue
        named = [
            str(v) for k, v in args.items()
            if k.lower() in _TARGET_KEYS and isinstance(v, str) and v
        ]
        # Antigravity starts several at once, as a list of subagents.
        for sub in args.get("Subagents") or args.get("subagents") or []:
            if isinstance(sub, dict):
                named += [
                    str(v) for k, v in sub.items()
                    if k.lower() in _TARGET_KEYS and isinstance(v, str) and v
                ]
        targets += [n.lower().replace("_", "-") for n in named] or [""]
    return targets


def read_reference(trajectory: dict[str, Any]) -> bool:
    """The query-understanding reference was read."""
    return any(
        isinstance(v, str) and _REFERENCE in v
        for name, args in tool_calls(trajectory) if name in _READ_TOOLS
        for k, v in args.items() if k.lower() in _PATH_KEYS)


def delegated_before_confirmation(trajectory: dict[str, Any]) -> bool:
    """Any delegation but to tpu-knowledge-base.

    A delegation whose target cannot be read counts: the gate is that no
    specialist does work before the user has confirmed the plan.
    """
    return any(t not in _BEFORE_CONFIRMATION
               for t in delegation_targets(trajectory))


def consulted_knowledge_base(trajectory: dict[str, Any]) -> bool:
    """tpu-knowledge-base was asked what is already known."""
    return "tpu-knowledge-base" in delegation_targets(trajectory)


# ---- TASK CRITERIA: codebase-not-supported ----------------------------------


@criterion(description="The query-understanding reference was followed")
def reference_followed(workspace: Path) -> bool:
    return read_reference(load_trajectory(workspace))


@criterion(description=
           "No specialist but tpu-knowledge-base ran before the user answered")
def stopped_before_model_onboarding(workspace: Path) -> bool:
    return not delegated_before_confirmation(load_trajectory(workspace))
