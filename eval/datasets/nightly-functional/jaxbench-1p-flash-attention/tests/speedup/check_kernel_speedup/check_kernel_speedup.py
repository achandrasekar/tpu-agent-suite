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
"""Speedup verifier evaluating optimization multiplier and execution latency. (Trajectory)"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extract_report import extract_report
from rewardkit import criterion


def load_verifier_params(section: str | None = None,
                         workspace: Path | None = None) -> dict[str, Any]:
    """Finds and parses verifier_params.json for this checker."""
    candidates = [Path("/tests/verifier_params.json")]
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        candidates.append(curr / "verifier_params.json")
        if curr.parent == curr:
            break
        curr = curr.parent
    if workspace:
        candidates.extend([
            workspace / "verifier_params.json",
            workspace / "tests" / "verifier_params.json",
        ])
    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data.get(section, {}) if section else data
            except Exception:
                pass
    return {}


def evaluate_kernel_speedup(
    workspace: Path,
    target_speedup: float | None = None,
    custom_speedup_patterns: list[str] | None = None,
) -> float:
    """Evaluates whether candidate execution speedup meets or exceeds target_speedup."""
    params = load_verifier_params("check_kernel_speedup", workspace)
    eff_target_speedup = (
        target_speedup if target_speedup is not None else float(
            params.get("target_speedup", params.get("min_speedup_factor",
                                                    1.0))))

    eff_patterns = custom_speedup_patterns or params.get(
        "custom_speedup_patterns")

    report = extract_report(
        workspace,
        target_speedup=eff_target_speedup,
        custom_speedup_patterns=eff_patterns,
    )

    # 1. Multiplicative Correctness Gate: If correctness was broken, score 0.0
    corr = report.get("correctness", {})
    if not corr.get("passed", False):
        print(
            "[speedup] FAIL: Numerical correctness failed or unverified. Multiplicative gate applied (score=0.0)."
        )
        return 0.0

    # 2. Speedup Extraction
    sp = report.get("speedup", {})
    factor = sp.get("speedup_factor")
    cand_ms = sp.get("candidate_latency_ms")
    base_ms = sp.get("baseline_latency_ms")
    source = sp.get("source", "unknown")

    if factor is None and cand_ms is not None and base_ms is not None and cand_ms > 0:
        factor = base_ms / cand_ms

    if factor is None:
        print(
            "[speedup] FAIL: No benchmark timing or speedup metrics found in trajectory or artifacts."
        )
        return 0.0

    factor = float(factor)
    factor_str = f"{factor:.2f}x"
    target_str = f"{eff_target_speedup:.2f}x"

    if factor >= eff_target_speedup:
        print(
            f"[speedup] PASS: Speedup {factor_str} >= Target {target_str} (Source={source})"
        )
        return 1.0
    else:
        print(
            f"[speedup] FAIL: Speedup {factor_str} < Target {target_str} (Source={source})"
        )
        return 0.0


@criterion(
    description=
    "Verify kernel execution speedup meets or exceeds target multiplier")
def check_kernel_speedup(workspace: Path) -> float:
    """RewardKit criterion verifying kernel speedup."""
    return evaluate_kernel_speedup(workspace)
