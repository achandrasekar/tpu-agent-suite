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
"""Numerical correctness gate verifying tensor parity against baseline. (Trajectory)"""

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


def evaluate_kernel_correctness(
    workspace: Path,
    custom_parity_patterns: list[str] | None = None,
) -> float:
    """Verifies that the agent tested correctness and confirmed numerical parity."""
    params = load_verifier_params("check_kernel_correctness", workspace)
    eff_patterns = custom_parity_patterns or params.get(
        "custom_parity_patterns")

    report = extract_report(
        workspace,
        custom_parity_patterns=eff_patterns,
    )
    corr = report.get("correctness", {})
    passed = bool(corr.get("passed", False))
    reason = corr.get("reason", "No verification found")
    max_diff = corr.get("max_diff")

    if not passed:
        print(f"[numerical_accuracy] FAIL: {reason}")
        return 0.0

    diff_str = f" (max_diff={max_diff:.6f})" if max_diff is not None else ""
    print(f"[numerical_accuracy] PASS: {reason}{diff_str}")
    return 1.0


@criterion(
    description=
    "Verify kernel numerical correctness and parity against reference implementation"
)
def check_kernel_correctness(workspace: Path) -> float:
    """RewardKit criterion verifying numerical correctness and parity."""
    return evaluate_kernel_correctness(workspace)
