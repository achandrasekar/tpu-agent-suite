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

import json
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion
from rewardkit.criteria._trajectory import collect_tool_calls, load_trajectory

_VIEW_TOOLS = {
    "view_file",
    "read_file",
    "cat",
    "open",
    "view",
    "activate_skill",
    "load_skill",
    "read",
}
_CMD_TOOLS = {
    "run_shell_command",
    "run_command",
    "bash",
    "execute_command",
    "sh",
    "terminal",
    "Bash",
    "exec",
}
_SUBAGENT_TOOLS = {
    "invoke_subagent",
    "subagent",
    "call_subagent",
    "delegate",
    "task",
    "Task",
}


def _collect_all_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    # 1. ATIF trajectory files
    for traj in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        data = load_trajectory(traj)
        if data:
            calls.extend(collect_tool_calls(data))

    # 2. Antigravity stream log (stream-json events)
    stream_logs = [
        Path("/logs/agent/stream.log"),
        Path("/logs/agent/stream.jsonl"),
        *Path("/logs/agent").glob("*.log"),
    ]
    for sl in stream_logs:
        if not sl.exists():
            continue
        try:
            with open(sl, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    su = obj.get("step_update", {})
                    ti = su.get("tool_info")
                    if isinstance(ti, dict):
                        calls.append({
                            "function_name":
                            ti.get("name") or ti.get("function_name"),
                            "arguments":
                            ti.get("args") or ti.get("arguments") or {},
                        })
                    tc = su.get("tool_call")
                    if isinstance(tc, dict):
                        calls.append({
                            "function_name":
                            tc.get("name") or tc.get("function_name"),
                            "arguments":
                            tc.get("args") or tc.get("arguments") or {},
                        })
        except Exception:
            pass

    # 3. Agent output text files for embedded JSON / tool invocations
    for p in Path("/logs/agent").glob("**/*"):
        if p.is_file() and p.suffix in (".log", ".json", ".jsonl", ".txt"):
            try:
                for line in p.read_text(encoding="utf-8",
                                        errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        if "tool_name" in obj or "function_name" in obj or "name" in obj:
                            calls.append({
                                "function_name":
                                obj.get("tool_name")
                                or obj.get("function_name") or obj.get("name"),
                                "arguments":
                                obj.get("arguments") or obj.get("args")
                                or obj.get("parameters") or {},
                            })
                        call_field = obj.get("tool_call") or obj.get("call")
                        if isinstance(call_field, dict):
                            calls.append({
                                "function_name":
                                call_field.get("name")
                                or call_field.get("function_name"),
                                "arguments":
                                call_field.get("args")
                                or call_field.get("arguments") or {},
                            })
            except Exception:
                pass

    return calls


def _read_all_log_text() -> str:
    texts: list[str] = []
    for ld in [Path("/logs/agent"), Path("/logs")]:
        if not ld.is_dir():
            continue
        for p in ld.glob("**/*"):
            if p.is_file() and p.suffix in (".log", ".json", ".jsonl", ".txt"):
                try:
                    texts.append(p.read_text(encoding="utf-8",
                                             errors="ignore"))
                except Exception:
                    pass
    return "\n".join(texts)


def _is_target_routed(calls: list[dict[str, Any]], all_text: str,
                      target_names: set[str]) -> bool:
    for tc in calls:
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if tool_name in _SUBAGENT_TOOLS:
            subagents = args.get("Subagents") or args.get("subagents") or []
            if isinstance(subagents, list):
                for sub in subagents:
                    if isinstance(sub, dict):
                        t_name = str(
                            sub.get("TypeName") or sub.get("type_name")
                            or sub.get("Role") or sub.get("role")
                            or "").lower()
                        if any(n in t_name for n in target_names):
                            return True
                        prompt = str(
                            sub.get("Prompt") or sub.get("prompt")
                            or "").lower()
                        if any(n in prompt for n in target_names):
                            return True
            t_name = str(
                args.get("TypeName") or args.get("type_name")
                or args.get("subagent_type") or args.get("agent_name")
                or args.get("Role") or args.get("role") or "").lower()
            if any(n in t_name for n in target_names):
                return True
            args_str = str(args).lower()
            if any(n in args_str for n in target_names):
                return True

        if tool_name in _VIEW_TOOLS:
            path = str(
                args.get("AbsolutePath") or args.get("file_path")
                or args.get("path") or args.get("TargetFile")
                or args.get("filename") or args.get("name")
                or args.get("skill_name") or "").lower()
            if any(n in path for n in target_names):
                return True

    delegation_re = re.compile(
        r'\\?"(?:type_name|TypeName|subagent|subagent_type|agent_name)'
        r'\\?"\s*:\s*\\?"([a-z0-9][a-z0-9_-]{2,40})\\?"',
        re.IGNORECASE,
    )
    for match in delegation_re.finditer(all_text):
        matched_name = match.group(1).lower().replace("_", "-")
        if any(n.replace("_", "-") in matched_name for n in target_names):
            return True

    return False


@criterion(description="Verify routing from tpu-agent to tpu-model-onboarding")
def check_routing_hierarchy(workspace: Path) -> bool:
    calls = _collect_all_calls()
    all_text = _read_all_log_text()

    onboarding_names = {
        "tpu-model-onboarding", "tpu_model_onboarding", "onboarding"
    }

    return _is_target_routed(calls, all_text, onboarding_names)
