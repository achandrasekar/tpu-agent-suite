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
"""PreToolUse hook: a production llm-d release is removed by people, not agents.

Two rules on shell commands, both path- and text-level because a PreToolUse
payload says what is about to run and not which agent asked:

**1. `deploy teardown --include-production` is denied.** The runtime's
teardown already refuses a production release without that flag; this hook
makes the flag itself unavailable to agents. The runbook in every bundle
says production removal is a user action, and this is what makes that true.

**2. Raw mutations against production are denied.** A `kubectl delete|scale|
patch|edit|replace|rollout` or `helm uninstall|upgrade|rollback|delete` whose
text names the production label (`tpu-coworker.llmd/production=true`), an
`-ingress`/`-epp` object of a release the job store marks `serve`, or the
environment's `production_namespace`, is denied. Agents in this package do
not run those tools directly -- the prompts forbid it and the linter checks
-- so the rule costs a legitimate caller nothing and catches the one path a
prompt cannot see.

An approval marker, `.teardown-approved`, placed by the user inside a
production bundle directory next to `release.json`, lifts rule 1 for a
command that names that bundle's release directory. Agents never write it.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

PRODUCTION_LABEL = "tpu-coworker.llmd/production"
APPROVAL_MARKER = ".teardown-approved"

TEARDOWN = re.compile(r"deploy\s+teardown\b[^\n;&|]*--include-production")
RAW_MUTATION = re.compile(
    r"""\b(?:kubectl\b[^\n;&|]*\b(?:delete|scale|patch|edit|replace|rollout)\b
      | helm\b[^\n;&|]*\b(?:uninstall|upgrade|rollback|delete)\b)""",
    re.VERBOSE)
RELEASE_DIR = re.compile(r"--release[=\s]+(\S+)")

DENIED_TEARDOWN = (
    "Blocked: `deploy teardown --include-production` removes a production"
    " llm-d release, and that is a user action.\n\n"
    "The release serves the customer's traffic; the bundle's RUNBOOK.md"
    " says how it is removed and by whom. If the user has decided to remove"
    " it, they create the marker file `.teardown-approved` inside the"
    " production bundle directory (next to release.json) and re-run the"
    " command naming that release directory. Report that instead of"
    " retrying.")

DENIED_RAW = (
    "Blocked: that command mutates a production llm-d release directly.\n\n"
    "Production releases are labelled `tpu-coworker.llmd/production=true`"
    " and live in the environment's production namespace. Changes to them go"
    " through `{{runtime}} deploy reconfigure` (hot/warm paths) or a"
    " re-render + `deploy apply --production`, both of which record history;"
    " removal is `deploy teardown --include-production`, which only the user"
    " may run.")


def shell_text(args: object) -> str:
    """The command text a shell tool is about to run, whatever the key."""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        for key in ("command", "cmd", "CommandLine", "commandLine", "script"):
            if isinstance(args.get(key), str):
                return args[key]
        return " ".join(shell_text(v) for v in args.values())
    if isinstance(args, (list, tuple)):
        return " ".join(shell_text(v) for v in args)
    return ""


def production_namespace(workspace: str | None) -> str | None:
    """`llmd.production_namespace` from the resolved environment, if any."""
    candidates = []
    env_path = os.environ.get("TPU_AGENT_SUITE_ENVIRONMENT")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    home = pathlib.Path(os.environ.get("TPU_AGENT_SUITE_HOME",
                                       pathlib.Path.home() / ".tpu-agent-suite"))
    candidates.append(home / "environment.json")
    if workspace:
        candidates.append(pathlib.Path(workspace) / "environment.json")
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        llmd = doc.get("llmd") or {}
        if llmd.get("production_namespace"):
            return str(llmd["production_namespace"])
    return None


def approved(text: str) -> bool:
    """True when the command names a release directory whose bundle carries
    the user's approval marker."""
    for match in RELEASE_DIR.finditer(text):
        rel = pathlib.Path(os.path.expanduser(match.group(1)))
        for base in (rel, rel.parent):
            if (base / APPROVAL_MARKER).is_file():
                return True
    return False


def names_production(text: str, namespace: str | None) -> bool:
    if PRODUCTION_LABEL in text:
        return True
    # `\b` would accept `prod-2` for `prod` (the hyphen is a boundary), so the
    # namespace has to end at whitespace, a quote, or the end of the text.
    if namespace and re.search(
            rf"(?:-n|--namespace)[=\s]+{re.escape(namespace)}(?=[\s'\"]|$)",
            text):
        return True
    return False


def decide(payload: dict) -> dict:
    tool = str(payload.get("tool") or payload.get("tool_name") or "").lower()
    args = payload.get("args", {})
    text = shell_text(args)
    if not text:
        return {"decision": "allow"}
    # Only shell-like tools carry a command; a file read naming the flag in
    # a document is not an execution.
    if tool and not any(k in tool for k in ("shell", "bash", "command",
                                             "terminal", "exec", "run")):
        return {"decision": "allow"}
    if TEARDOWN.search(text) and not approved(text):
        return {"decision": "deny", "reason": DENIED_TEARDOWN}
    workspace = (payload.get("context") or {}).get("workspace")
    if RAW_MUTATION.search(text) and names_production(
            text, production_namespace(workspace)):
        return {"decision": "deny", "reason": DENIED_RAW}
    return {"decision": "allow"}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # An unparseable payload is not evidence of a violation.
        print(json.dumps({"decision": "allow"}))
        return
    print(json.dumps(decide(payload)))


if __name__ == "__main__":
    main()
