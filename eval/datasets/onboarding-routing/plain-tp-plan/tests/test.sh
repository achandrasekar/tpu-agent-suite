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

# Did onboarding ask tpu-knowledge-base for the TP/DP rule, rather than
# only reading the reference that names it?
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
python3 - "$LOGS" "${AGENT_LOGS_DIR:-/logs/agent}" <<'PYEOF'
import json, pathlib, re, sys

logs, agent = (pathlib.Path(p) for p in sys.argv[1:3])
# A delegation counts once it started and its result was not an error; one
# the run ended on before it returned counts, since it started.
calls, failed = {}, set()
for path in agent.rglob("*.jsonl"):
    for line in path.read_text("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        content = (row.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if (block.get("type") == "tool_use"
                    and block.get("name") in ("Agent", "Task")):
                args = block.get("input") or {}
                if args.get("subagent_type", "").endswith("tpu-knowledge-base"):
                    calls[block.get("id")] = json.dumps(args)[:400]
            elif block.get("type") == "tool_result" and block.get("is_error"):
                failed.add(block.get("tool_use_id"))
delegations = [args for call, args in calls.items() if call not in failed]
named = any("vllm-tp-dp-plan" in d for d in delegations)
about_plan = [d for d in delegations if "vllm-tp-dp-plan" in d or re.search(
    r"\bTP\b|tensor.parallel|tp_plan|data.parallel", d, re.I)]
deployments = pathlib.Path("/workspace/inference/deployments.json").is_file()
checks = {"asked_knowledge_base": bool(delegations),
          "asked_about_tp_plan": bool(about_plan),
          "named_vllm_tp_dp_plan": named,
          "wrote_deployments": deployments}
score = 1.0 if about_plan else 0.0
(logs / "reward.txt").write_text(f"{score}\n")
(logs / "reward.json").write_text(json.dumps({"reward": score}))
(logs / "reward-details.json").write_text(json.dumps(
    {"checks": {k: int(v) for k, v in checks.items()},
     "delegations": delegations}, indent=1))
for k, v in checks.items():
    print(f"  {k:<22} {'PASS' if v else 'FAIL'}")
PYEOF
