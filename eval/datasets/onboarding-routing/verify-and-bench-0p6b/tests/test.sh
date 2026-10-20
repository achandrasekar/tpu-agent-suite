#!/bin/bash
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
#
# Did onboarding hold one node from verify through the benchmark, reapably,
# book a node once for its preflight checks, and get the number asked for?
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
python3 - "$LOGS" "${TPU_AGENT_SUITE_HOME:-/root/.tpu-agent-suite}" <<'PYEOF'
import json, pathlib, re, sys

# Judged by what the run left behind (compute's job records and the files the
# steps wrote), not by how the agent spelled its commands.
logs, home = (pathlib.Path(p) for p in sys.argv[1:3])
runs = home / "runs"


def load(path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def recipe(record):
    return pathlib.Path(record.get("recipe_path") or "").name


records = [r for r in map(load, (runs / "jobs").glob("*/job.json"))
           if isinstance(r, dict)]
preflight_jobs = [r for r in records if recipe(r) == "onboarding_preflight.py"]
servers = [r for r in records
           if re.match(r"(launch_vllm|fastapi_server)", recipe(r))]
reports = [r for r in map(load, runs.glob("*/artifacts/**/preflight.json"))
           if isinstance(r, dict)]
cells = [c for s in map(load, runs.glob("*/artifacts/**/summary.json"))
         if isinstance(s, dict) for c in s.get("cells") or []]
answer = load(pathlib.Path("/logs/artifacts/answer.json"))
checks = {
    "one_preflight_job": len(preflight_jobs) == 1,
    "no_separate_probe_check_or_flags_job": not any(
        recipe(r) in ("image_probe.py", "preflight_check.py",
                      "engine_compatibility_check.py",
                      "discover_vllm_flags.sh") for r in records),
    # On a CPU node, holding no TPU, with torch_tpu told the profile's TPU.
    "preflight_on_a_cpu_node": any(
        (report.get("chipless") or {}).get("exit_code") == 0
        for report in reports),
    "preflight_steps_all_passed": any(
        all((report.get(step) or {}).get("exit_code") == 0
            for step in ("image_probe", "full_check",
                         "engine_compatibility_check", "discover_vllm_flags"))
        for report in reports),
    "one_server_job": len(servers) == 1,
    "server_reapable_and_cancelled": bool(servers) and all(
        not r.get("serve") and r.get("status") == "CANCELLED"
        for r in servers),
    "benchmark_cell_measured": any(
        c.get("cell") == "isl1024_osl128_c1" and c.get("status") == "ok"
        for c in cells),
    "answer_written": isinstance(answer, dict) and bool(answer.get("answer")),
}
score = 1.0 if all(checks.values()) else 0.0
(logs / "reward.txt").write_text(f"{score}\n")
(logs / "reward.json").write_text(json.dumps({"reward": score}))
(logs / "reward-details.json").write_text(json.dumps(
    {"checks": {k: int(v) for k, v in checks.items()},
     "jobs": [{k: r.get(k) for k in ("job_id", "recipe_path", "serve",
                                     "status", "exit_code")}
              for r in records]}, indent=1))
for k, v in checks.items():
    print(f"  {k:<42} {'PASS' if v else 'FAIL'}")
PYEOF
