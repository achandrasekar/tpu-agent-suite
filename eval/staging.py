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
"""Staging pipeline for evaluation packages.

Implements single-translation, single-installation composite workspace staging
with multi-factor SHA-256 cache invalidation and empty-set bypass.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.utils import find_repo_root


@dataclass(frozen=True)
class StagingResult:
    """Artifacts produced by staging a package selection."""
    staged_root: Path
    staged_workspace: Path
    staged_dist: Path
    central_pkg_dir: Path
    is_empty: bool


def get_git_commit_hash(repo_root: Path) -> str:
    """Identify current Git commit hash, including dirty state for cache safety."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        commit = res.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        if status.stdout.strip():
            dirty_hash = hashlib.sha256(
                status.stdout.encode("utf-8")).hexdigest()[:8]
            return f"{commit[:12]}-dirty-{dirty_hash}"
        return commit[:16]
    except Exception:
        return "unknown"


def compute_staging_digest(
    repo_root: Path,
    harness: str,
    compute_answers: dict[str, Any],
    skills: list[str] | None = None,
) -> str:
    """Generate fast multi-factor cache digest.

    Incorporates:
      - Git commit hash of the repository
      - Target harness
      - Resolved compute answers
      - The skill subset, when one was chosen
    """
    commit_hash = get_git_commit_hash(repo_root)
    answers_str = json.dumps(compute_answers, sort_keys=True)
    subset = ",".join(skills) if skills else "ALL"
    raw_key = f"{repo_root}:{commit_hash}:{harness}:{answers_str}:{subset}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def stage_composite_package(
    harness: str,
    *,
    control_arm: bool = False,
    skills: list[str] | None = None,
    compute_answers: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    pkg_name: str = "tpu-agent-suite",
    pkg_version: str = "0.1.0",
) -> StagingResult:
    """Compile and install the composite package distribution into a staged workspace.

    Executes a single `compiler.py translate` pass and a single `installer.py install` pass
    of the whole package, or of `skills` when a debugging run chose a subset. A
    control arm installs nothing, so it stages nothing.
    """
    repo = repo_root or find_repo_root()
    compute = compute_answers or {"backend": "local"}

    if control_arm:
        dummy_path = Path(tempfile.gettempdir()) / "eval_empty_stage"
        return StagingResult(
            staged_root=dummy_path,
            staged_workspace=dummy_path,
            staged_dist=dummy_path,
            central_pkg_dir=dummy_path,
            is_empty=True,
        )

    # Compute cache digest
    digest = compute_staging_digest(
        repo_root=repo,
        harness=harness,
        compute_answers=compute,
        skills=skills,
    )
    user = os.environ.get("USER", "user")
    staged_root = Path(tempfile.gettempdir()) / f"eval_staged_{user}_{digest}"
    staged_dist = staged_root / harness / pkg_name
    staged_workspace = staged_root / "workspace"
    central_pkg_dir = staged_root / "packages" / f"{pkg_name}@{pkg_version}"

    # Export paths into os.environ for legacy configs and substitution
    os.environ["STAGED_WORKSPACE"] = str(staged_workspace)
    os.environ["HOST_CENTRAL_PKG"] = str(central_pkg_dir)
    os.environ["HOST_CENTRAL_PKG_TARGET"] = (
        f"/root/.tpu-agent-suite/packages/{pkg_name}@{pkg_version}")

    def _clean_run_ids(ws: Path) -> None:
        try:
            (ws / ".tpu-agent-suite" /
             "current_run_id").unlink(missing_ok=True)
        except Exception:
            pass

    packages_dir = staged_root / "packages"
    if packages_dir.is_dir():
        matches = sorted(packages_dir.glob(f"{pkg_name}@*"))
        if matches:
            central_pkg_dir = matches[-1]
            os.environ["HOST_CENTRAL_PKG"] = str(central_pkg_dir)
            os.environ["HOST_CENTRAL_PKG_TARGET"] = (
                f"/root/.tpu-agent-suite/packages/{central_pkg_dir.name}")

    # Check cache freshness via sentinel file and directory structure
    sentinel = staged_root / ".staging_complete"
    fresh = (sentinel.is_file() and (staged_dist / "skills").is_dir()
             and ((staged_workspace / ".agents").is_dir() or
                  (staged_workspace / ".claude").is_dir())
             and central_pkg_dir.is_dir()
             and (central_pkg_dir / "framework" / "runtime.py").is_file())

    if fresh:
        _clean_run_ids(staged_workspace)
        ws_env = staged_workspace / ".tpu-agent-suite" / "environment.json"
        if (ws_env.is_file()
                and not (staged_root / "environment.json").is_file()):
            shutil.copy2(ws_env, staged_root / "environment.json")
        print(f"==> staged package cache hit: {staged_workspace}")
        return StagingResult(
            staged_root=staged_root,
            staged_workspace=staged_workspace,
            staged_dist=staged_dist,
            central_pkg_dir=central_pkg_dir,
            is_empty=False,
        )

    # Cache miss: Clean leftover run IDs
    staged_workspace.mkdir(parents=True, exist_ok=True)
    _clean_run_ids(staged_workspace)

    # 1. SINGLE Translation Pass
    compiler_script = repo / "framework" / "compiler.py"
    translate_cmd = [
        sys.executable,
        str(compiler_script),
        "translate",
        str(repo / "package"),
        "--harness",
        harness,
        "--output",
        str(staged_dist),
    ]
    if skills:
        translate_cmd.extend(["--skills", ",".join(skills)])
        print(f"==> compiling skills {', '.join(skills)} for '{harness}'...")
    else:
        print(f"==> compiling the whole package for '{harness}'...")
    subprocess.run(translate_cmd, cwd=repo, check=True)

    # 2. SINGLE Installation Pass, then the `default` profile from the answers
    flat_answers: dict[str, Any] = dict(compute)

    # Keep only the profile fields questions.json defines for this backend
    questions_file = staged_dist / "environment" / "questions.json"
    if not questions_file.is_file():
        questions_file = repo / "package" / "environment" / "questions.json"
    if questions_file.is_file():
        try:
            q_data = json.loads(questions_file.read_text(encoding="utf-8"))
            backend = flat_answers.get("backend")
            valid_q_names = {
                q["name"]
                for q in q_data.get("questions", [])
                if str(q.get("path") or q["name"]).startswith("compute.")
                and backend in q.get("when", {}).get("backend", [backend])
            }
            flat_answers = {
                k: v
                for k, v in flat_answers.items() if k in valid_q_names
            }
        except Exception:
            pass

    answers_file = staged_root / "answers.json"
    answers_file.write_text(json.dumps(flat_answers, indent=2),
                            encoding="utf-8")
    installer_script = repo / "framework" / "installer.py"

    print(f"==> installing staged package to {staged_workspace}...")
    staged_workspace.parent.mkdir(parents=True, exist_ok=True)
    installer_env = {**os.environ, "TPU_AGENT_SUITE_HOME": str(staged_root)}
    subprocess.run(
        [
            sys.executable,
            str(installer_script),
            "install",
            str(staged_dist),
            "--destination",
            str(staged_workspace),
            "--non-interactive",
            "--skip-harness-check",
            # Hooks find the store because the container mounts
            # staged_root's package at /root/.tpu-agent-suite.
            "--relocated-store",
        ],
        cwd=repo,
        env=installer_env,
        check=True,
    )

    packages_dir = staged_root / "packages"
    if packages_dir.is_dir():
        matches = sorted(packages_dir.glob(f"{pkg_name}@*"))
        if matches:
            central_pkg_dir = matches[-1]
            os.environ["HOST_CENTRAL_PKG"] = str(central_pkg_dir)
            os.environ["HOST_CENTRAL_PKG_TARGET"] = (
                f"/root/.tpu-agent-suite/packages/{central_pkg_dir.name}")

    # Install sets up no compute; the runtime saves the profile, with the
    # same checks as a session's setup.
    subprocess.run(
        [
            sys.executable,
            str(central_pkg_dir / "framework" / "runtime.py"),
            "compute",
            "--package",
            central_pkg_dir.name,
            "setup",
            "apply",
            "--profile",
            "default",
            "--answers",
            str(answers_file),
            "--workspace",
            str(staged_workspace),
        ],
        cwd=staged_workspace,
        env=installer_env,
        check=True,
    )

    # Propagate environment.json to the central store (staged_root is the
    # installer's TPU_AGENT_SUITE_HOME), so a command run outside the workspace
    # still resolves it, as a user-scope install would.
    ws_env = staged_workspace / ".tpu-agent-suite" / "environment.json"
    if ws_env.is_file():
        shutil.copy2(ws_env, staged_root / "environment.json")

    # Write sentinel indicating complete and successful staging
    sentinel.write_text("ok\n", encoding="utf-8")

    return StagingResult(
        staged_root=staged_root,
        staged_workspace=staged_workspace,
        staged_dist=staged_dist,
        central_pkg_dir=central_pkg_dir,
        is_empty=False,
    )
