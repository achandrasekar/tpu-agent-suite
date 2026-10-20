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

import re
from pathlib import Path

from rewardkit import criterion
from rewardkit.criteria._trajectory import collect_tool_calls, load_trajectory

_SKILL_PATTERN = re.compile(r"skills/([^/]+)/SKILL\.md")
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


@criterion(description="Verify skill 'smart_hf_search' was loaded")
def check_skill_loaded(workspace: Path) -> bool:
    traj_path = "/logs/agent/trajectory.json"
    if not Path(traj_path).exists():
        traj_path = "/logs/trajectory.json"
    calls = collect_tool_calls(load_trajectory(traj_path) or {})
    skill_name = "smart_hf_search"
    skill_hyphen = "smart-hf-search"
    for tc in calls:
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if tool_name in _VIEW_TOOLS:
            path = str(
                args.get("AbsolutePath") or args.get("file_path")
                or args.get("path") or args.get("TargetFile")
                or args.get("filename") or args.get("name")
                or args.get("skill_name") or "")
            match = _SKILL_PATTERN.search(path)
            if match and match.group(1) == skill_name:
                return True
            if skill_name in path or skill_hyphen in path:
                return True
        if tool_name in _CMD_TOOLS:
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if f"skills/{skill_name}" in cmd or f"{skill_name}/SKILL.md" in cmd:
                return True
    return False
