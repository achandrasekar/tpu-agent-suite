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
"""Evaluation utility functions."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    """Identify repository root using relative hierarchy and project sentinels.

    Does not depend on Git or subprocess execution. Checks in order:
    1. REPO_ROOT environment variable override.
    2. Parent directory walk looking for project sentinels (`package/package.release.json` or `AGENTS.md`).
    3. Direct file-relative hierarchy (one level above `eval`).
    """
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if candidate.is_dir():
            return candidate

    cur = (start or Path(__file__)).resolve()
    # Walk parents checking for unambiguous project sentinels
    for parent in [cur] + list(cur.parents):
        if (parent / "package" / "package.release.json").is_file() or (
                parent / "AGENTS.md").is_file():
            return parent

    # Fallback to direct relative hierarchy from eval/
    eval_dir = Path(__file__).resolve().parent
    return eval_dir.parent


def is_control_arm(config_path: Path | str) -> bool:
    """Whether a config is a control arm, which installs nothing.

    Decided by name alone: a stem with a `baseline` part, split on `_` or `-`,
    as in `claude-code/baseline.yaml`. Every other config installs the whole
    package.
    """
    return "baseline" in re.split(r"[-_]", Path(config_path).stem.lower())


def harness_for_config(config: dict[str, Any]) -> str:
    """The compiler harness named by the config's first agent."""
    agents = config.get("agents") or []
    name = str(agents[0].get("name", "")).lower() if agents else ""
    for token, harness in (("claude", "claude-code"), ("codex", "codex"),
                           ("cursor", "cursor")):
        if token in name:
            return harness
    return "antigravity"


def released_skills(repo_root: Path | None = None) -> list[str]:
    """The skills package.release.json releases: what the whole package installs."""
    root = repo_root or find_repo_root()
    manifest = root / "package" / "package.release.json"
    return list(json.loads(manifest.read_text(encoding="utf-8"))["skills"])


def skill_subset(names: Iterable[str], released: Iterable[str]) -> list[str]:
    """The skills named, comma- or space-separated, checked against `released`.

    The subset mode: a debugging run that installs only some of the package.
    """
    chosen = sorted(
        {n.strip()
         for value in names
         for n in value.split(",") if n.strip()})
    known = set(released)
    unknown = [n for n in chosen if n not in known]
    if unknown or not chosen:
        raise ValueError(
            f"--skills: unknown or empty: {', '.join(unknown) or '(none)'}. "
            f"The package releases: {', '.join(sorted(known))}")
    return chosen
