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
"""Deterministic Autotuning Sweep Runner for tpu-kernel-development.

Iterates over parameter combinations in an autotune specification/payload,
executes each candidate kernel via Python, measures performance, and
outputs the best configuration.
"""

import argparse
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# The assembled test harness (`test_harness_template.py`) prints exactly one
# `RESULT_TIME: <ms> ms` line for the optimized kernel. Trials are ranked on
# that line and nothing else: a looser pattern would pick up a stray
# "compile time: ..." log, and falling back to process wall clock would rank
# on JAX compilation rather than kernel latency.
_RESULT_TIME_RE = re.compile(r"^RESULT_TIME:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$",
                             re.MULTILINE)


def _extract_time_ms(output: str) -> float | None:
    """Return the harness-reported kernel time in ms, or None if absent."""
    match = _RESULT_TIME_RE.search(output)
    return float(match.group(1)) if match else None


def _failure_reason(all_results: list[dict[str, Any]]) -> str:
    """Explain a sweep in which no trial succeeded.

    Distinguishes a sweep whose trials all died the same way -- which is what
    hardware contention on a shared target looks like, since every trial then
    hits the same "device busy" error regardless of its configuration -- from
    one whose configurations genuinely failed on their own terms. The caller
    that summarises the sweep should not read the former as "every config is
    bad".
    """
    if not all_results:
        return ("no trials ran (empty search space or total timeout hit "
                "before the first trial)")

    def signature(result: dict[str, Any]) -> tuple[int, str]:
        lines = (result.get("stderr") or "").strip().splitlines()
        return result["exit_code"], lines[-1] if lines else ""

    signatures = {signature(r) for r in all_results}
    if len(all_results) > 1 and len(signatures) == 1:
        exit_code, last_line = next(iter(signatures))
        return (
            f"all {len(all_results)} trials failed identically (exit code "
            f"{exit_code}: {last_line!r}) -- likely a device or environment "
            "problem such as hardware contention, not a configuration problem")
    return f"no trial succeeded ({len(all_results)} tried)"


def run_autotune(
    payload_path: str,
    output_path: str | None = None,
    per_trial_timeout: int = 120,
    total_timeout: int = 1800,
) -> dict[str, Any]:
    """Runs autotuning sweep across search space and writes best configuration."""
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    code_template = payload.get("code_template", "")
    search_space = payload.get("search_space", {})
    dependencies = payload.get("dependencies", {})

    if not code_template:
        raise ValueError(f"Payload in {payload_path} missing 'code_template'")
    if not search_space:
        raise ValueError(f"Payload in {payload_path} missing 'search_space'")

    keys = list(search_space.keys())
    values = [v if isinstance(v, list) else [v] for v in search_space.values()]
    combinations = list(itertools.product(*values))

    logging.info(
        f"Starting autotuning sweep: {len(combinations)} total combinations to evaluate."
    )

    all_results: list[dict[str, Any]] = []
    start_sweep_time = time.time()
    best_config: dict[str, Any] | None = None
    best_time_ms: float = float("inf")

    temp_dir = tempfile.mkdtemp(prefix="tpu-kernel-development_autotune_")
    try:
        # Save any bundled dependencies
        for rel_path, content in dependencies.items():
            dep_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dep_path), exist_ok=True)
            with open(dep_path, "w", encoding="utf-8") as df:
                df.write(content)

        for idx, combo in enumerate(combinations, 1):
            if (time.time() - start_sweep_time) > total_timeout:
                logging.warning(
                    f"Total autotune timeout ({total_timeout}s) exceeded. Terminating early."
                )
                break

            cfg = dict(zip(keys, combo))
            trial_code = code_template
            for k, v in cfg.items():
                trial_code = trial_code.replace(f"{{{k}}}", str(v))

            trial_script = os.path.join(temp_dir, "run_code.py")
            with open(trial_script, "w", encoding="utf-8") as f:
                f.write(trial_code)

            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, trial_script],
                    capture_output=True,
                    text=True,
                    timeout=per_trial_timeout,
                    cwd=temp_dir,
                )
                wall_ms = (time.time() - t0) * 1000.0
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired as exc:
                wall_ms = (time.time() - t0) * 1000.0
                exit_code = 124
                stdout = exc.stdout or "" if isinstance(exc.stdout,
                                                        str) else ""
                stderr = f"Trial timed out after {per_trial_timeout} seconds."
            except Exception as exc:  # noqa: BLE001 - recorded per trial
                wall_ms = (time.time() - t0) * 1000.0
                exit_code = 1
                stdout = ""
                stderr = f"{type(exc).__name__}: {exc}"
            measured_ms = _extract_time_ms(stdout)
            if exit_code == 0 and measured_ms is None:
                # A clean exit without the marker is still a failed trial:
                # there is no kernel time to rank it on, and substituting
                # another number would let the sweep "succeed" on the wrong
                # configuration.
                exit_code = 1
                stderr = (stderr + "\n" if stderr else "") + (
                    "no `RESULT_TIME: <ms> ms` line in stdout; the harness "
                    "must print it for the trial to be ranked")

            result_entry = {
                "trial": idx,
                "config": cfg,
                "exit_code": exit_code,
                "time_ms": measured_ms,
                "wall_ms": wall_ms,
                "stdout": stdout,
                "stderr": stderr,
            }
            all_results.append(result_entry)

            if exit_code == 0:
                logging.info(
                    f"Trial {idx}/{len(combinations)} succeeded: {cfg} -> {measured_ms:.2f}ms"
                )
                if measured_ms < best_time_ms:
                    best_time_ms = measured_ms
                    best_config = cfg
            else:
                logging.info(
                    f"Trial {idx}/{len(combinations)} failed with exit code {exit_code}: {cfg}"
                )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # A sweep with no successful trial has no best_config. It must not fall
    # back to some arbitrary combination: apply_best_config.py would then
    # write a configuration that never ran into the kernel, and the profile
    # and state record downstream would describe that untested kernel.
    status = "failed" if best_config is None else "success"
    failure_reason = _failure_reason(
        all_results) if best_config is None else None
    if failure_reason:
        logging.warning(failure_reason)

    summary = {
        "status": status,
        "best_config": best_config,
        "best_time_ms": best_time_ms if best_time_ms != float("inf") else None,
        "total_trials": len(all_results),
        "all_results": all_results,
    }
    if failure_reason:
        summary["failure_reason"] = failure_reason

    if output_path:
        out_file = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Saved autotune results to {out_file}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=
        "Run tpu-kernel-development autotuning sweep across search space.")
    parser.add_argument("payload", help="Path to autotune_payload.json")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional path to output autotune_results.json",
    )
    parser.add_argument(
        "--trial-timeout",
        "--timeout",
        dest="trial_timeout",
        type=int,
        default=120,
        help="Timeout per trial in seconds (default 120)",
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=1800,
        help="Total sweep timeout in seconds (default 1800)",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_autotune(
            payload_path=args.payload,
            output_path=args.output,
            per_trial_timeout=args.trial_timeout,
            total_timeout=args.total_timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("status") == "success" else 1
    except Exception as exc:
        logging.error(f"Autotuning failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
