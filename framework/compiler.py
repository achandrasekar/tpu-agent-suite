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
"""Compiler module for TPU agent suite framework.

Handles package manifest verification, structural integrity checks, target
harness compatibility validation, document generation, and compilation into
distribution formats (Claude Code and Codex).
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import tarfile
from typing import Any, Callable, Iterable, Sequence

# Run as a script, this file has its own directory on sys.path, not the one
# that holds the framework package: in the repository the root, and the compiler
# only runs there. Add it, so `framework` imports.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework import compute, frontmatter, journal, lint, utils  # noqa: E402

Path = pathlib.Path

TpuAgentSuiteError = utils.TpuAgentSuiteError
DEFAULT_HOOK_TIMEOUT = utils.DEFAULT_HOOK_TIMEOUT
DEFAULT_HOOK_ON_FAILURE = utils.DEFAULT_HOOK_ON_FAILURE
TPU_AGENT_SUITE_VERSION = utils.TPU_AGENT_SUITE_VERSION
DEFAULT_MANIFEST = "package.release.json"
GENERATED_MARKER = utils.GENERATED_MARKER
GOOGLE_HARNESSES = utils.GOOGLE_HARNESSES
HARNESS_CONFIG_DIRS = utils.HARNESS_CONFIG_DIRS
HARNESS_DISPLAY_NAMES = utils.HARNESS_DISPLAY_NAMES
PORTABLE_TOOLS = utils.PORTABLE_TOOLS
get_native_tools_catalog = utils.get_native_tools_catalog
get_package_manifest_schema = utils.get_package_manifest_schema
get_agent_frontmatter_schema = utils.get_agent_frontmatter_schema
get_supported_hook_events = utils.get_supported_hook_events
get_tool_mappings = utils.get_tool_mappings
harness_mapping = utils.harness_mapping
is_google_harness = utils.is_google_harness
subagent_handback_mapping = utils.subagent_handback_mapping
to_harness_event = utils.to_harness_event
load_json = utils.load_json
load_schema_example = utils.load_schema_example
parse_version_range = utils.parse_version_range
parse_frontmatter_document = frontmatter.parse_frontmatter_document
rooted = utils.rooted
safe_relative = utils.safe_relative
validate_schema_definition = utils.validate_schema_definition
write_json = utils.write_json

# =====================================================================
# Package Validation & Graph Checking
# =====================================================================


def get_harness_tool_aliases() -> dict[str, str]:
    """Derive legacy tool alias suggestions dynamically from tool-mappings.json as single source of truth."""
    mappings = get_tool_mappings()
    canonical_tools = {"read", "write", "edit", "shell", "search"}
    aliases: dict[str, str] = {
        "sh": "shell",
        "bash": "shell",
        "grep": "search",
        "glob": "search",
        "edit_file": "edit",
        "apply_patch": "write",
    }
    for canonical, defn in mappings.items():
        if canonical not in canonical_tools or not isinstance(defn, dict):
            continue
        for harness, h_defn in defn.get("harnesses", {}).items():
            for t in h_defn.get("tools", []):
                t_lower = t.lower()
                if t_lower not in canonical_tools and t_lower not in aliases:
                    aliases[t_lower] = canonical
    return aliases


HARNESS_TOOL_ALIASES: dict[str, str] = get_harness_tool_aliases()

AGENT_REFERENCE_PATTERN = r"\{\{agent:([a-z0-9]+(?:-[a-z0-9]+)*)}}"
SKILL_REFERENCE_PATTERN = r"\{\{skill:([a-z0-9]+(?:-[a-z0-9]+)*)}}"
# `{{reference:<path>}}`, `{{script:<path>}}`, `{{data:<path>}}` and
# `{{schema:<path>}}` each name a file by its path under the package's
# references/, scripts/, data/ or schemas/ root, and render to the installed
# location, so prose never spells an install path itself.
RESOURCE_REFERENCE_PATTERN = r"\{\{(reference|script|data|schema):([A-Za-z0-9_][A-Za-z0-9_./-]*)}}"

# Token kind -> the package directory it names.
RESOURCE_ROOTS = {
    "reference": utils.REFERENCES_DIRNAME,
    "script": utils.SCRIPTS_DIRNAME,
    "data": utils.DATA_DIRNAME,
    "schema": utils.SCHEMAS_DIRNAME,
}
# `reap-jobs` cancels the run's leftover compute jobs, which is only right once
# the session is over.
VALID_BUILTIN_HOOK_ACTIONS: dict[str, frozenset[str]] = {
    "start-run":
    frozenset({"SessionStart"}),
    "validate-schema":
    frozenset({"PreToolUse", "SubagentStop"}),
    "reap-jobs":
    frozenset({"SessionEnd"}),
    # Records the active skill (PostToolUse) and blocks tools that skill
    # doesn't allow (PreToolUse).
    "skill-tools":
    frozenset({"PreToolUse", "PostToolUse"}),
    # The other builtins act: `start-run` initializes a run, `reap-jobs`
    # cancels compute. Each is pinned to the events where that action is
    # correct. `record-event` only appends to the study's event log, so it
    # changes nothing the session goes on to do and is legal on every event.
    # A recorder pinned the same way would simply be absent from any event
    # nobody anticipated, and a missing recorder looks identical to nothing
    # happening.
    "record-event":
    frozenset({
        "SessionStart", "PreToolUse", "PostToolUse", "SubagentStop", "Stop",
        "SessionEnd"
    }),
}

# Levels of delegation permitted below a released skill. Must match the harness
# nesting limit the install plan sets, or a package builds here and fails there.
MAX_DELEGATION_DEPTH = 3

# Delegates a single session may run at once. The harnesses count every live
# subagent, and a parent waiting on its child is still live, so a chain of
# nested delegates holds one slot per level. The cap therefore has to cover
# the deepest chain the compiler accepts, or that chain builds and then fails
# to spawn its last level. That is all it covers: delegates fanned out in one
# message each hold their own chain of slots, so no package prompt may do so.
MAX_CONCURRENT_DELEGATES = MAX_DELEGATION_DEPTH

# Longest `compute run --timeout` a package prompt may ask for, in seconds.
# A synchronous run holds the shell call open for its whole duration, so the
# harness's shell timeout has to outlast it.
MAX_SYNC_COMPUTE_TIMEOUT_S = 3600

# Margin over the run's own timeout for staging inputs, pulling outputs, and
# the run reporting its own timeout before the shell kills it.
SHELL_TIMEOUT_HEADROOM_S = 300

# How long a harness shell call must be allowed to stay open.
SHELL_TIMEOUT_MS = (MAX_SYNC_COMPUTE_TIMEOUT_S +
                    SHELL_TIMEOUT_HEADROOM_S) * 1000

# Margin the harness allows a hook over the hook's own timeout. hook-dispatch
# enforces the declared timeout on the action it runs, but finishes a little
# after it: interpreter start and event normalization come first, the verdict
# after. Given the same deadline, the harness kills dispatch before that
# verdict is written -- a hung hook's deny never arrives, and Claude Code runs
# the tool anyway. About 0.1s on an idle host, more on a loaded one; it costs
# nothing on the ordinary path, since the harness waits only for dispatch to
# exit, and bounds dispatch itself should it hang.
HOOK_DISPATCH_HEADROOM_S = 5

# Exit code of the compiled hook command when hook-dispatch itself exits
# non-zero. Dispatch always exits 0 with its verdict in JSON, so a non-zero
# exit is never a verdict, and the policy decides. `deny` is 2, the one code
# Claude Code, Cursor and Codex block on (others fail open); Antigravity
# refuses tool hooks on any non-zero. The policy applies on tool events only,
# where a block is one refusal: on Stop and SubagentStop a block means "keep
# going", with no way out once dispatch cannot run, and session events carry
# no decision, so those exit 0 whatever the policy.
HOOK_FAIL_SAFE_EXIT_CODES = {"allow": 0, "deny": 2}
HOOK_FAIL_SAFE_POLICY_EVENTS = frozenset({"PreToolUse", "PostToolUse"})


def fail_safe_hook_command(dispatch_cmd: str, hook_name: str, on_failure: str,
                           event: str) -> str:
    """Append a shell fallback applying `on_failure` if dispatch exits non-zero.

    Needs nothing installed, so it works when `runtime.py` is gone. The
    stderr line is the only trace of a fail-open: the dispatcher's failure
    log cannot be written by a dispatcher that never started. `event` is
    the canonical event; outside HOOK_FAIL_SAFE_POLICY_EVENTS the fallback
    exits 0 whatever the policy, and the line says so.
    """
    if on_failure not in HOOK_FAIL_SAFE_EXIT_CODES:
        raise TpuAgentSuiteError(
            f"hook '{hook_name}' on_failure must be one of "
            f"{list(HOOK_FAIL_SAFE_EXIT_CODES)}, got {on_failure!r}")
    if event in HOOK_FAIL_SAFE_POLICY_EVENTS:
        exit_code = HOOK_FAIL_SAFE_EXIT_CODES[on_failure]
        after = f"before a verdict; on_failure={on_failure}"
    else:
        exit_code = 0
        after = (f"before a verdict; allowing: on_failure={on_failure} would "
                 f"loop on {event}")
    # `"$?"` is its own argument so it still holds the dispatcher's exit;
    # the literal parts are single-quoted. Hook names are schema-restricted
    # to `[a-zA-Z0-9_.-]`.
    before = shlex.quote(
        f"tpu-agent-suite: hook {hook_name}: hook-dispatch exited")
    return (f'{dispatch_cmd} || {{ echo {before} "$?" {shlex.quote(after)} '
            f">&2; exit {exit_code}; }}")


# Environment the Claude Code install plan writes into the workspace settings.
# The two caps keep the harness in step with what the compiler already
# enforces. The third is what makes a subagent's stop mean it finished:
# without it Claude Code launches delegates asynchronously and backgrounds a
# command that outruns its timeout, so a subagent waiting on either ends its
# turn and fires SubagentStop with nothing to report. A run recorded here was
# blocked four times on stops that were only pauses. The switch runs delegates
# in the foreground and removes `run_in_background` from Bash and the subagent
# tools, leaving one stop per subagent.
#
# With nothing backgrounded, a Bash call that reaches its timeout is killed,
# and the `compute run` inside it with it. Claude Code caps the timeout a call
# may ask for at 10 minutes and defaults to 2, both well short of the runs the
# packages make, and nearly every command a package issues is such a run. So
# the cap and the default are both raised to cover the longest one; the
# default must be set explicitly, since unset it falls back to 2 minutes.
CLAUDE_CODE_SETTINGS_ENV: dict[str, str] = {
    "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": str(MAX_DELEGATION_DEPTH),
    "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": str(MAX_CONCURRENT_DELEGATES),
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "BASH_DEFAULT_TIMEOUT_MS": str(SHELL_TIMEOUT_MS),
    "BASH_MAX_TIMEOUT_MS": str(SHELL_TIMEOUT_MS),
}

# Tokens whose value depends on the harness a distribution is compiled for,
# not on the manifest. `render_agent_references` accepts them and leaves them
# in place; `render_harness_tokens` expands them once the harness is known.
HARNESS_SCOPED_TOKENS = ("{{harness}}", "{{built-in:ask_user_question}}")

# What `{{built-in:ask_user_question}}` becomes, per harness. Two of the four
# have a tool for it under different names; the other two have none, so the
# token renders as the plain instruction and the sentence reads the same
# either way. Compiling is never refused over it: an author should be able to
# say "ask the user" once and ship to all four.
ASK_USER_QUESTION_TOOLS: dict[str, str] = {
    "claude-code": "AskUserQuestion",
    "antigravity": "ask_question",
}
ASK_USER_QUESTION_PROSE = "ask the user"

# Hook events whose payload carries something for `matchers` to select: the
# tool on the tool events, the subagent on SubagentStop. The other events
# name neither, so the runtime's matcher emulation would skip every fire of
# a hook that filtered on them -- a hook that silently never runs.
MATCHABLE_HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SubagentStop")


def check_hook_matchers(hook_name: str, event: str,
                        matchers: list[str] | None) -> None:
    """Reject `matchers` on a hook event that has nothing to match them on."""
    if matchers and event not in MATCHABLE_HOOK_EVENTS:
        raise TpuAgentSuiteError(
            f"hook '{hook_name}' declares matchers, but {event} has nothing "
            f"to match them against; matchers apply only to "
            f"{', '.join(MATCHABLE_HOOK_EVENTS)}")


def render_harness_tokens(text: str, harness: str) -> str:
    """Expand the tokens whose value depends on the harness being built.

    `{{harness}}` resolves here because a skill that calls the runtime must
    pass the harness it is running under, and the harness is known only at
    translate time.

    `{{built-in:ask_user_question}}` is the same shape: the tool that puts a
    question to the user is named differently on the two harnesses that have
    one, and on the two that do not the token becomes the plain instruction,
    so one authored sentence ships to all four.

    Only harness-dependent tokens live here; `{{runtime}}`, `{{package}}`
    and `{{package_tag}}` are manifest-dependent and resolve in
    `render_agent_references`, which has no harness in scope.
    """
    text = text.replace("{{harness}}", harness)
    return text.replace(
        "{{built-in:ask_user_question}}",
        ASK_USER_QUESTION_TOOLS.get(harness, ASK_USER_QUESTION_PROSE))


# Every kind of `{{kind:name}}` token there is. A kind outside this set is a
# misspelling, and left alone it would reach the agent as literal braces.
TOKEN_KINDS = ("agent", "skill", "built-in", *RESOURCE_ROOTS)
_KINDED_TOKEN = re.compile(r"\{\{([A-Za-z_-]+):([^{}\n]*)}}")


def reject_unknown_token_kinds(text: str) -> None:
    """Fail on a `{{kind:name}}` token whose kind does not exist."""
    unknown = [(number, match.group(0), match.group(1))
               for number, line in enumerate(text.splitlines(), 1)
               for match in _KINDED_TOKEN.finditer(line)
               if match.group(1) not in TOKEN_KINDS]
    if not unknown:
        return
    lines = []
    for number, token, kind in unknown:
        close = difflib.get_close_matches(kind.lower(), TOKEN_KINDS, n=1)
        hint = f" (did you mean {{{{{close[0]}:...}}}}?)" if close else ""
        lines.append(f"  line {number}: {token}{hint}")
    raise TpuAgentSuiteError(
        "instructions use a token of a kind that does not"
        f" exist; the kinds are {', '.join(TOKEN_KINDS)}:\n" +
        "\n".join(lines))


def render_markdown(text: str, manifest: dict[str, Any], harness: str) -> str:
    """Render a markdown file an agent reads: every token a prompt may use."""
    return render_harness_tokens(render_agent_references(text, manifest),
                                 harness)


def render_agent_references(text: str,
                            manifest: dict[str, Any],
                            prefix: str = "") -> str:
    """Replace {{agent:NAME}} template tokens with prefixed agent identifiers.

    A reference to an agent or skill the build does not hold is an error. A
    build holds everything its skills reach, so a name that fails here is one
    the document never declared.
    """
    reject_unknown_token_kinds(text)
    known_agents = {agent["name"] for agent in manifest["agents"]}
    known_skills = {
        skill["name"]
        for skill in manifest.get("skills", []) if isinstance(skill, dict)
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in known_agents:
            raise TpuAgentSuiteError(
                f"instructions reference unknown agent: {name}")
        return prefix + name

    def replace_skill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in known_skills:
            raise TpuAgentSuiteError(
                f"instructions reference unknown skill: {name}")
        return prefix + name

    rendered = re.sub(AGENT_REFERENCE_PATTERN, replace, text)
    if "{{agent:" in rendered:
        raise TpuAgentSuiteError(
            "instructions contain a malformed agent reference")
    rendered = re.sub(SKILL_REFERENCE_PATTERN, replace_skill, rendered)
    if "{{skill:" in rendered:
        raise TpuAgentSuiteError(
            "instructions contain a malformed skill reference")

    # A reference or script token names a file under the package's
    # references/ or scripts/ root; it renders to the installed path. Whether
    # the file exists is checked against the source tree by
    # `validate_resource_tokens`, which has the package directory in scope.
    _, pkg_path, _ = package_runtime_paths(manifest)

    def replace_resource(match: re.Match[str]) -> str:
        return f"{pkg_path}/{RESOURCE_ROOTS[match.group(1)]}/{match.group(2)}"

    rendered = re.sub(RESOURCE_REFERENCE_PATTERN, replace_resource, rendered)
    if any(f"{{{{{kind}:" in rendered for kind in RESOURCE_ROOTS):
        raise TpuAgentSuiteError(
            "instructions contain a malformed reference, script, data or"
            " schema token")

    # Automatically inject `--package {{package_tag}}` into `{{runtime}} compute`
    # and `{{runtime}} deploy` calls
    rendered = _RUNTIME_COMPUTE_PATTERN.sub(
        r"{{runtime}} \1 --package {{package_tag}}", rendered)

    for token, value in resource_registry(manifest).items():
        rendered = rendered.replace(token, value)

    # Harness-scoped tokens resolve in `render_harness_tokens`, which runs on
    # this function's output at translate time; here they are known, not
    # expanded, since no harness is in scope when a package is only verified.
    unknown = sorted(
        set(re.findall(r"\{\{[a-z_]+(?:-[a-z_]+)*(?::[a-z_]+)?\}\}", rendered))
        - set(HARNESS_SCOPED_TOKENS))
    if unknown:
        known = sorted({*resource_registry(manifest), *HARNESS_SCOPED_TOKENS})
        raise TpuAgentSuiteError(
            "instructions reference unknown resource tokens: "
            f"{', '.join(unknown)} (known: {', '.join(known)})")

    return rendered


# `{{runtime}} compute` or `{{runtime}} deploy` not already followed by
# `--package`. Both resolve the package's environment.json; neither verb
# family should make an author spell the package tag.
_RUNTIME_COMPUTE_PATTERN = re.compile(
    r"\{\{runtime}}\s+(compute|deploy)(?!\s+--package\b)")


def resource_token_targets(text: str) -> list[tuple[str, str]]:
    """Every `{{reference:...}}`, `{{script:...}}`, `{{data:...}}` and
    `{{schema:...}}` token in `text`, as (kind, package-relative path)
    pairs."""
    return [(kind, f"{RESOURCE_ROOTS[kind]}/{rel}")
            for kind, rel in re.findall(RESOURCE_REFERENCE_PATTERN, text)]


# What a plain path must be written as instead. A path typed out in prose is
# read from wherever the agent happens to be running, which is its workspace
# and not the installed package, so it names a file that is not there.
_PLAIN_PACKAGE_PATH = re.compile(
    r"(?<![\w./{}:~-])((?:%s)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)" %
    "|".join(sorted(set(RESOURCE_ROOTS.values()))))
_VIA_PACKAGE_PATH = re.compile(
    r"\{\{package}}/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)")
_ANY_TOKEN = re.compile(r"\{\{[^{}\n]*}}")


def token_for(path: str) -> str:
    """The token that names the package path `path`, or what to do when no
    token can: the path is under a directory no build ships by that name."""
    root, _, rest = path.partition("/")
    kind = next((k for k, v in RESOURCE_ROOTS.items() if v == root), None)
    if kind is None or not rest:
        return ("no token names it; package files live under " +
                ", ".join(f"{v}/" for v in RESOURCE_ROOTS.values()))
    return f"{{{{{kind}:{rest}}}}}"


# A token this compiler once resolved. It named a directory, so a path built
# on it was checked against nothing, and it went on resolving after the files
# it was written for stopped shipping.
_RETIRED_TOKEN = re.compile(r"\{\{knowledgebase}}(/[A-Za-z0-9_.*/-]*)?")


def plain_package_paths(text: str) -> list[tuple[int, str]]:
    """(line, path) for every package file `text` names other than by token.

    Two spellings count: the path typed out (`scripts/fam/tool.py`) and any
    path built on the package token (`{{package}}/scripts/fam/tool.py`). The
    second renders to a place under the package but is checked against
    nothing, so it may name a directory the package no longer has.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in _VIA_PACKAGE_PATH.finditer(line):
            found.append((number, match.group(1).rstrip(".")))
        bare = _ANY_TOKEN.sub(lambda m: " " * len(m.group(0)), line)
        for match in _PLAIN_PACKAGE_PATH.finditer(bare):
            found.append((number, match.group(1).rstrip(".")))
    return found


def body_first_line(document: str, body: str) -> int:
    """The line of `document` that `body`, its text after frontmatter, starts
    on."""
    return document.count("\n", 0, len(document) - len(body)) + 1


def reject_plain_package_paths(text: str,
                               label: str,
                               first_line: int = 1) -> None:
    """Fail when `text` names a package file other than by token."""
    retired = [(number, match.group(0))
               for number, line in enumerate(text.splitlines(), 1)
               for match in _RETIRED_TOKEN.finditer(line)]
    if retired:
        raise TpuAgentSuiteError(
            f"{label} uses the retired {{{{knowledgebase}}}} token; name a"
            " provenance page with {{data:tpu-knowledge-base/pages/<name>.md}}:\n"
            + "\n".join(f"  line {first_line + number - 1}: {found}"
                        for number, found in retired))
    found = plain_package_paths(text)
    if found:
        raise TpuAgentSuiteError(
            f"{label} names package files by path; a token is the only way"
            " to name one:\n" +
            "\n".join(f"  line {first_line + number - 1}: {path}"
                      f" -> {token_for(path)}" for number, path in found) +
            "\nA path that names a file in the workspace, not in the"
            " package, is written as one, with something in front of it:"
            f" ./{found[0][1]} or <work>/{found[0][1]}.")


def covers(declared: Iterable[str], target: str, package_dir: Path) -> bool:
    """Whether `target` is among `declared`, under a declared directory, or a
    directory whose every file is declared."""
    entries = {entry.rstrip("/") for entry in declared}
    if target in entries:
        return True
    if any(target.startswith(entry + "/") for entry in entries):
        return True
    path = package_dir / target
    if path.is_dir():
        files = [
            _relative_to_package(f, package_dir) for f in path.rglob("*")
            if f.is_file() and not dist_excluded(f.relative_to(package_dir))
        ]
        return bool(files) and all(
            covers(entries, name, package_dir) for name in files)
    return False


def validate_resource_tokens(
    text: str,
    package_dir: Path,
    label: str,
    declared: Iterable[str],
    assets: Iterable[str] = ()) -> None:
    """Fail when a token names a file its document does not ship.

    `declared` is what the document that holds `text` ships: what it declared
    in frontmatter, and what the references it reads declared. A file that
    exists but that only some other unit declares does not count, because a
    build that holds this document need not hold that one. A token that
    renders to a path nothing ships is the migration bug this layout exists
    to prevent, so it fails the build rather than the run.

    Data and schemas belong to the package rather than to a unit, so a data
    or schema token is held to `assets`, what package.release.json copies.
    """
    declared, assets = list(declared), list(assets)
    for kind, rel in resource_token_targets(text):
        target = rooted(package_dir, rel, f"{label} {kind} token")
        if not target.exists():
            raise TpuAgentSuiteError(
                f"{label} names {kind} {rel!r} in a token, but the package"
                " has no such file")
        if kind in ("data", "schema"):
            if not covers(assets, rel, package_dir):
                raise TpuAgentSuiteError(
                    f"{label} names {kind} {rel!r} in a token, but no build"
                    " ships it: package.release.json does not list it under `copy`"
                )
            continue
        if not covers(declared, rel, package_dir):
            raise TpuAgentSuiteError(
                f"{label} names {kind} {rel!r} in a token but does not"
                f" declare it: add it to `{kind}s` in its frontmatter, or"
                " to the frontmatter of a reference it reads")


def shipped_by(unit: dict[str, Any]) -> list[str]:
    """Everything a skill or agent ships: its own files and what the
    references it reads brought."""
    return [
        *unit.get("references", []), *unit.get("scripts", []),
        *unit.get("declared_references", []),
        *unit.get("declared_scripts", [])
    ]


def shipped_by_reference(package_dir: Path, rel: str) -> list[str]:
    """Everything a markdown reference ships with it: itself and what its
    frontmatter declares, transitively."""
    shipped: list[str] = [rel]
    pending, seen = [rel], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = package_dir / current
        if path.suffix != ".md" or not path.is_file():
            continue
        frontmatter = load_reference_frontmatter(path)
        shipped.extend(frontmatter.get("scripts", []))
        for entry in frontmatter.get("references", []):
            shipped.append(entry)
            files = reference_files(package_dir, entry,
                                    f"reference {current} reference")
            shipped.extend(files)
            pending.extend(files)
    return _dedupe(shipped)


def resource_registry(manifest: dict[str, Any]) -> dict[str, str]:
    """The single map from authored resource tokens to installed locations.

    Authors name WHAT they want; where it lives is routed here and nowhere
    else -- prose never states a location class, so relocating a resource
    (e.g. the knowledge base moving from workspace to central) is a one-line
    change with zero authored-file churn. `{{package_tag}}` is the exception:
    it expands to the `name@version` string itself, which compiled
    instructions pass to `--package` flags at run time -- an identifier, not
    a filesystem path.

    Expansion happens at compile time, and a distribution is compiled once
    then installed at any scope, so no token can encode scope. Every path
    token expands to a ~-anchored central location, valid from any cwd on
    the machine that installed it.

    Environment config deliberately has NO token: its location depends on
    install scope (workspace slot vs central store), which is unknowable at
    compile time, so a path in prose would be wrong at some scope. Prose
    that needs effective config directs the agent through the CLI
    (`{{runtime}} compute inspect --package {{package_tag}} --json`), and
    utils.resolve_environment applies the ownership-keyed rule at run time.
    """
    pkg_tag, pkg_path, runtime = package_runtime_paths(manifest)
    return {
        "{{package}}": pkg_path,
        "{{package_tag}}": pkg_tag,
        "{{runs}}": utils.RUNS_HOME_SPELLING,
        "{{runtime}}": runtime,
    }


def validate_delegation_graph(
    skills: Sequence[dict[str, Any]],
    agents_by_name: dict[str, dict[str, Any]],
    expected_agents: set[str],
) -> None:
    """Validate the delegation graph: cycles, depth, and reachability.

    Every released skill is a root of the graph. A skill may delegate to an
    agent or to another skill.

    Depth is counted from the skill. The skill is level 0, what it delegates
    to is level 1, and no chain may go deeper than MAX_DELEGATION_DEPTH:
    skill -> L1 -> L2 -> L3.

    The limit must stay in step with the harness. Claude Code enforces its own nesting
    limit via CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH, set by the install plan, and
    a package that builds here but exceeds that there fails at the point of
    delegation rather than at build time.
    """
    visited: set[str] = set()
    roots = list(skills)
    nodes: dict[str, dict[str, Any]] = dict(agents_by_name)
    for root in roots:
        nodes.setdefault(root["name"], root)

    def visit(agent_name: str, depth: int, path: tuple[str, ...]) -> None:
        if agent_name in path:
            cycle = " -> ".join((*path, agent_name))
            raise TpuAgentSuiteError(
                f"delegation graph contains a cycle: {cycle}")
        if depth > MAX_DELEGATION_DEPTH:
            chain = " -> ".join((*path, agent_name))
            raise TpuAgentSuiteError(
                f"delegation exceeds {MAX_DELEGATION_DEPTH} levels below"
                f" skill {path[0]}: {chain}")

        visited.add(agent_name)
        for delegated in nodes[agent_name].get("delegates", []):
            visit(delegated, depth + 1, (*path, agent_name))

    for root in roots:
        for delegated in root.get("delegates", []):
            visit(delegated, 1, (root["name"], ))
        # A skill that is also a selectable agent (its twin) is its own root:
        # nothing delegates to it, so reachability has to count it as reached
        # or the package fails to build for the one agent a user starts from.
        visited.add(root["name"])

    unreachable = expected_agents - visited
    if unreachable:
        raise TpuAgentSuiteError(
            f"agents are unreachable from any skill: {sorted(unreachable)}")


def delegation_warnings(manifest: dict[str, Any]) -> list[str]:
    """Delegation the guide allows and expects to be rare.

    Two cases, neither of which fails a build: a skill that delegates to a
    skill, and a skill that is built only because something delegates to it.
    """
    skills = {skill["name"]: skill for skill in manifest.get("skills", [])}
    warnings: list[str] = []
    for name, skill in skills.items():
        for delegate in skill.get("delegates", []):
            if delegate in skills and delegate != name:
                warnings.append(
                    f"skill '{name}' delegates to skill '{delegate}'; a"
                    " skill should rarely delegate to a skill")
    units = (*manifest.get("skills", []), *manifest.get("agents", []))
    for name in manifest.get("delegated_skills", []):
        by = sorted({
            unit["name"]
            for unit in units if name in unit.get("delegates", [])
        })
        warnings.append(
            f"skill '{name}' is not one of the skills this build names; it"
            f" is built only because {', '.join(by)} delegates to it")
    return warnings


def print_delegation_warnings(manifest: dict[str, Any]) -> None:
    for warning in delegation_warnings(manifest):
        print(f"WARNING: {warning}", file=sys.stderr)


def load_agent_document(doc_path: Path, package_dir: Path) -> dict[str, Any]:
    """Load an agent definition from a Markdown document with YAML frontmatter."""
    if not doc_path.is_file():
        raise TpuAgentSuiteError(f"agent document not found: {doc_path}")
    content = doc_path.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter_document(content, doc_path.name)
    if not fm:
        raise TpuAgentSuiteError(
            f"agent document '{doc_path.name}' must start with '---' YAML frontmatter"
        )
    schema = get_agent_frontmatter_schema()
    schema_errors = utils.validate_instance(fm, schema)
    if schema_errors:
        raise TpuAgentSuiteError(
            f"invalid YAML frontmatter in agent document '{doc_path.name}':\n"
            + "\n".join(schema_errors))

    if doc_path.stem != fm["name"]:
        raise TpuAgentSuiteError(
            f"agent document filename '{doc_path.name}' frontmatter name '{fm['name']}' "
            f"does not match expected agent name '{doc_path.stem}'")

    try:
        rel_instructions = str(doc_path.resolve().relative_to(
            package_dir.resolve()))
    except ValueError:
        rel_instructions = str(doc_path)

    agent = dict(fm)
    agent["instructions"] = rel_instructions
    agent.setdefault("tools", [])
    agent.setdefault("delegates", [])
    agent.setdefault("scripts", [])
    agent.setdefault("references", [])
    return agent


def _relative_to_package(doc_path: Path, package_dir: Path) -> str:
    try:
        return doc_path.resolve().relative_to(package_dir.resolve()).as_posix()
    except ValueError:
        return str(doc_path)


def load_skill_document(doc_path: Path, package_dir: Path) -> dict[str, Any]:
    """Load a skill from its SKILL.md.

    The frontmatter is the whole declaration: description, delegates,
    references, scripts and dependencies. The skill's name is the directory
    that holds SKILL.md, and the frontmatter must agree with it.
    """
    if not doc_path.is_file():
        raise TpuAgentSuiteError(f"skill document not found: {doc_path}")
    content = doc_path.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter_document(content, doc_path.name)
    if not fm:
        raise TpuAgentSuiteError(
            f"skill document '{doc_path.name}' must start with '---' YAML"
            " frontmatter")
    schema_errors = utils.validate_instance(
        fm, utils.get_skill_frontmatter_schema())
    if schema_errors:
        raise TpuAgentSuiteError(
            f"invalid YAML frontmatter in skill document '{doc_path}':\n" +
            "\n".join(schema_errors))
    expected = doc_path.parent.name
    if fm["name"] != expected:
        raise TpuAgentSuiteError(
            f"skill document '{doc_path}' frontmatter name '{fm['name']}' "
            f"does not match expected skill name '{expected}'")

    skill = dict(fm)
    skill["instructions"] = _relative_to_package(doc_path, package_dir)
    for key in ("tools", "delegates", "scripts", "references",
                "host_dependencies", "tpu_dependencies"):
        skill.setdefault(key, [])
    return skill


def load_reference_frontmatter(doc_path: Path) -> dict[str, Any]:
    """The frontmatter of a reference document, or {} when it has none.

    A reference needs no frontmatter at all; when it has one it may bring
    scripts, further references, delegates and dependencies, all of which the
    compiler adds to every unit that reads it.
    """
    if doc_path.suffix != ".md":
        return {}
    content = doc_path.read_text(encoding="utf-8", errors="ignore")
    if not content.lstrip().startswith("---"):
        return {}
    fm, _ = parse_frontmatter_document(content, doc_path.name)
    schema_errors = utils.validate_instance(
        fm, utils.get_reference_frontmatter_schema())
    if schema_errors:
        raise TpuAgentSuiteError(
            f"invalid YAML frontmatter in reference '{doc_path}':\n" +
            "\n".join(schema_errors))
    for key in ("delegates", "scripts", "references", "host_dependencies",
                "tpu_dependencies"):
        fm.setdefault(key, [])
    return fm


def reference_files(package_dir: Path, rel: str, label: str) -> list[str]:
    """The files a `references:` entry brings: itself, or every file under it
    when it names a directory."""
    source = rooted(package_dir, rel, label)
    if source.is_dir():
        return sorted(
            _relative_to_package(path, package_dir)
            for path in source.rglob("*") if path.is_file()
            and not dist_excluded(path.relative_to(package_dir)))
    if source.is_file():
        return [rel]
    raise TpuAgentSuiteError(f"{label} does not exist: {source}")


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def discover_units(package_dir: Path, manifest: dict[str, Any]) -> None:
    """Resolve the package's skills, then everything reachable from them.

    `skills` in the manifest names the roots. From each root the compiler
    follows `delegates` to agents (`agents/<name>.md`) or to other skills,
    and `references` to files under `references/`, reading the
    frontmatter of every markdown reference on the way: what a reference
    declares -- scripts, further references, delegates, dependencies -- is
    added to every unit that reads it, transitively. Directory references
    bring every file under them.

    On return each skill and agent carries flat lists: `references` and
    `scripts` are the files it ships, `delegates` the names it may invoke
    (its own plus those its references named, kept apart in
    `own_delegates`), and `declared_references` what it wrote. The manifest
    gains the union of dependencies, every `schemas/*.json`, and a twin agent
    for each skill that delegates, so a harness that starts from an agent
    (`--agent <skill>`) finds one carrying the same delegation contract.
    """
    skills: list[dict[str, Any]] = []
    for item in manifest.get("skills", []):
        doc = rooted(package_dir,
                     f"{utils.SKILLS_DIRNAME}/{item}/{utils.SKILL_DOCUMENT}",
                     f"skill {item}")
        if not doc.is_file():
            raise TpuAgentSuiteError(
                f"skill {item!r} is named but {utils.SKILLS_DIRNAME}/{item}/"
                f"{utils.SKILL_DOCUMENT} does not exist")
        skills.append(load_skill_document(doc, package_dir))
    if not skills:
        raise TpuAgentSuiteError(
            "package releases no skills: name at least one under `skills`")
    names = [skill["name"] for skill in skills]
    if len(names) != len(set(names)):
        raise TpuAgentSuiteError(f"skill names are not unique: {names}")
    skill_names = set(names)
    delegated_skills: list[str] = []

    agents: dict[str, dict[str, Any]] = {}
    host_deps: list[str] = []
    tpu_deps: list[str] = []
    reference_cache: dict[str, dict[str, Any]] = {}

    def closure(unit: dict[str, Any], label: str) -> None:
        """Expand one unit's references transitively, in place."""
        unit.setdefault("own_delegates", list(unit.get("delegates", [])))
        unit.setdefault("declared_references", list(unit.get("references",
                                                             [])))
        unit.setdefault("declared_scripts", list(unit.get("scripts", [])))
        files: list[str] = []
        scripts: list[str] = list(unit.get("scripts", []))
        delegates: list[str] = list(unit.get("delegates", []))
        pending = list(unit.get("references", []))
        seen: set[str] = set()
        while pending:
            rel = pending.pop(0)
            for path in reference_files(package_dir, rel,
                                        f"{label} reference"):
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
                if path not in reference_cache:
                    reference_cache[path] = load_reference_frontmatter(
                        package_dir / path)
                fm = reference_cache[path]
                scripts.extend(fm.get("scripts", []))
                delegates.extend(fm.get("delegates", []))
                host_deps.extend(fm.get("host_dependencies", []))
                tpu_deps.extend(fm.get("tpu_dependencies", []))
                pending.extend(r for r in fm.get("references", [])
                               if r not in seen)
        unit["references"] = _dedupe(files)
        unit["scripts"] = _dedupe(scripts)
        unit["delegates"] = _dedupe(delegates)
        host_deps.extend(unit.get("host_dependencies", []))
        tpu_deps.extend(unit.get("tpu_dependencies", []))

    worklist: list[tuple[dict[str, Any],
                         str]] = [(skill, f"skill {skill['name']}")
                                  for skill in skills]
    while worklist:
        unit, label = worklist.pop(0)
        closure(unit, label)
        for delegate in unit["delegates"]:
            if delegate in agents or delegate in skill_names:
                continue
            doc = package_dir / utils.AGENTS_DIRNAME / f"{delegate}.md"
            if doc.is_file():
                agent = load_agent_document(doc, package_dir)
                agents[delegate] = agent
                worklist.append((agent, f"agent {delegate}"))
                continue
            # A skill may delegate to a skill. One that is reached is built,
            # whether or not the list named it: a build holds what it reaches.
            doc = package_dir / utils.SKILLS_DIRNAME / delegate / utils.SKILL_DOCUMENT
            if doc.is_file():
                skill = load_skill_document(doc, package_dir)
                skills.append(skill)
                skill_names.add(delegate)
                delegated_skills.append(delegate)
                worklist.append((skill, f"skill {delegate}"))
            # Otherwise not an error here: `load_package` reports an
            # unresolved delegate against the unit that named it.

    # A skill that delegates is also selectable as an agent (`--agent <skill>`)
    # carrying the same delegation contract.
    for skill in skills:
        if skill["delegates"] and skill["name"] not in agents:
            agents[skill["name"]] = dict(skill)

    manifest["skills"] = skills
    manifest["delegated_skills"] = delegated_skills
    manifest["agents"] = list(agents.values())
    manifest["host_dependencies"] = _dedupe(host_deps)
    manifest["tpu_dependencies"] = _dedupe(tpu_deps)
    schemas_dir = package_dir / "schemas"
    manifest["schemas"] = sorted(
        f"schemas/{path.name}" for path in schemas_dir.glob(
            "*.json")) if schemas_dir.is_dir() else []


def get_declared_tools(package_dir: Path,
                       manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index declared tools and assert file existence and global name uniqueness."""
    declared: dict[str, dict[str, Any]] = {}

    # 1. Native tools (from native_tools.json catalog or explicit manifest)
    for tool in manifest.get("native_tools", []):
        name = tool["name"]
        if name in declared:
            raise TpuAgentSuiteError(f"duplicate tool name declared: '{name}'")
        declared[name] = {"category": "native", **tool}

    # 2. Script Tools
    for script in manifest.get("script_tools", []):
        name = script["name"]
        if name in declared:
            raise TpuAgentSuiteError(f"duplicate tool name declared: '{name}'")
        script_path = rooted(package_dir, script["path"],
                             f"script tool '{name}'")
        if not script_path.is_file():
            raise TpuAgentSuiteError(
                f"script tool '{name}' file does not exist: {script_path}")
        declared[name] = {"category": "script", **script}

    # 3. MCP Servers
    for mcp in manifest.get("mcp_servers", []):
        name = mcp["name"]
        if name in declared:
            raise TpuAgentSuiteError(f"duplicate tool name declared: '{name}'")
        declared[name] = {"category": "mcp", **mcp}

    return declared


def validate_agent_tool(
    tool_name: str,
    declared_tools: dict[str, dict[str, Any]],
    agent_name: str,
    kind: str = "agent",
) -> dict[str, Any]:
    """Validate that a tool a skill or agent lists is declared and has
    action_level metadata, suggesting canonical names for legacy aliases.

    `kind` is what `agent_name` names, "agent" or "skill", for the message.
    """
    if tool_name not in declared_tools:
        canonical = HARNESS_TOOL_ALIASES.get(tool_name.lower())
        suggestion = (f" (did you mean canonical tool '{canonical}'?)"
                      if canonical else "")
        raise TpuAgentSuiteError(
            f"{kind} '{agent_name}' references undeclared tool '{tool_name}'{suggestion}. "
            f"{kind.capitalize()}s must declare canonical tools "
            f"({', '.join(repr(t) for t in sorted(PORTABLE_TOOLS))}) "
            "or declared package scripts/mcp_servers.")
    tool = declared_tools[tool_name]
    if "action_level" not in tool:
        raise TpuAgentSuiteError(
            f"tool '{tool_name}' declared for {kind} '{agent_name}' lacks required 'action_level' metadata"
        )
    return tool


# Fields package.release.json once carried. Each is rejected by name, with where its
# content lives now, rather than as an anonymous unknown property.
REMOVED_MANIFEST_FIELDS = {
    "entrypoint":
    "name the skill in `skills`",
    "agents":
    "agents are discovered from the `delegates` of each skill; put each at"
    " agents/<name>.md",
    "include":
    "fragment files are gone; declare metadata in the frontmatter of the"
    " skill, agent or reference it belongs to",
    "schemas":
    "every schemas/*.json is registered automatically",
    "host_dependencies":
    "declare it in the frontmatter of the unit that needs it",
    "tpu_dependencies":
    "declare it in the frontmatter of the unit that needs it",
    "required_capabilities":
    "harness capabilities are uniform; nothing declares them",
}


def reject_removed_fields(manifest: dict[str, Any]) -> None:
    """Fail on a package.release.json field that is no longer read, saying why."""
    found = [key for key in REMOVED_MANIFEST_FIELDS if key in manifest]
    if found:
        raise TpuAgentSuiteError(
            "package.release.json declares fields that are no longer"
            " supported:\n" +
            "\n".join(f"  {key}: {REMOVED_MANIFEST_FIELDS[key]}"
                      for key in found))


def resolve_manifest_extends(package_dir: Path, manifest: dict[str, Any],
                             manifest_name: str) -> dict[str, Any]:
    """Merge a manifest that `extends` another manifest onto that base.

    The base's fields apply first and this manifest's fields replace them.
    `add_skills` appends to the base's `skills`, so a variant package names
    only what it adds. One level only: the base must not extend another.
    """
    if "extends" not in manifest:
        if "add_skills" in manifest:
            raise TpuAgentSuiteError(
                f"{manifest_name}: `add_skills` needs `extends`")
        return manifest
    base_name = manifest["extends"]
    if not isinstance(base_name, str) or base_name == manifest_name:
        raise TpuAgentSuiteError(
            f"{manifest_name}: `extends` must name another manifest file")
    base = load_json(rooted(package_dir, base_name, "extended manifest"))
    if not isinstance(base, dict):
        raise TpuAgentSuiteError(
            f"{base_name}: manifest must be a JSON object")
    if "extends" in base:
        raise TpuAgentSuiteError(
            f"{manifest_name}: extends {base_name}, which extends another"
            " manifest; only one level is supported")
    add_skills = manifest.get("add_skills", [])
    if not isinstance(add_skills, list) or not all(
            isinstance(name, str) for name in add_skills):
        raise TpuAgentSuiteError(
            f"{manifest_name}: `add_skills` must be a list of skill names")
    merged = {
        **base,
        **{
            key: value
            for key, value in manifest.items() if key not in ("extends", "add_skills")
        }
    }
    skills = list(merged.get("skills", []))
    repeated = sorted(set(add_skills) & set(skills))
    if repeated:
        raise TpuAgentSuiteError(
            f"{manifest_name}: `add_skills` repeats skills already listed:"
            f" {repeated}")
    merged["skills"] = skills + add_skills
    return merged


def resolve_manifest_composition(
    package_dir: Path,
    manifest: dict[str, Any],
    skills: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The manifest plus everything its skills reach.

    Returns a copy carrying the default native tool catalog and what
    `discover_units` found: the loaded skills, the agents they reach, the
    union of dependencies and the registered schemas.

    `skills` replaces the manifest's own list (`--skills`). It may name any
    skill the package holds under `skills/`, released or not, and each is
    built whole: nothing is trimmed from what a skill delegates to.
    """
    composed = dict(manifest)
    if skills is not None:
        composed["skills"] = list(skills)

    if "native_tools" not in composed:
        try:
            default_native = get_native_tools_catalog()
            if "native_tools" in default_native:
                composed["native_tools"] = default_native["native_tools"]
        except TpuAgentSuiteError:
            pass

    discover_units(package_dir, composed)
    return composed


def load_package(package_dir: Path,
                 allow_extra_fields: bool = False,
                 skills: Sequence[str] | None = None,
                 manifest: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load, validate, and verify the structural integrity of a package manifest and its graph.

    `manifest` names the manifest file in `package_dir`. Several manifests
    can share one package directory, each releasing its own package from
    the same skills, agents, scripts and references. A manifest that
    `extends` another is merged onto it first.
    """
    manifest_path = rooted(package_dir, manifest, "manifest")
    raw_manifest = resolve_manifest_extends(package_dir,
                                            load_json(manifest_path), manifest)
    reject_removed_fields(raw_manifest)
    manifest_schema = get_package_manifest_schema()

    raw_schema_errors = utils.validate_instance(
        raw_manifest,
        manifest_schema,
        allow_additional_properties=allow_extra_fields,
    )
    if raw_schema_errors:
        raise TpuAgentSuiteError(f"invalid {manifest}:\n" +
                                 "\n".join(raw_schema_errors))

    # 1. Discover what the released skills reach
    manifest = resolve_manifest_composition(package_dir,
                                            raw_manifest,
                                            skills=skills)

    manifest_name = manifest["name"]

    # 2. Tool indexing & name uniqueness
    declared_tools = get_declared_tools(package_dir, manifest)

    # 3. Agent presence & uniqueness
    agent_names = [agent["name"] for agent in manifest["agents"]]
    if len(agent_names) != len(set(agent_names)):
        raise TpuAgentSuiteError(
            f"Agent names are not unique in package: {manifest_name}")
    name_set = set(agent_names)

    # What package.release.json copies: what a data or schema token may name.
    assets = list(manifest.get("copy", []))

    # 4. Skill validation. Every released skill is a delegation root.
    skills = manifest["skills"]
    skill_names = {skill["name"] for skill in skills}
    for skill in skills:
        # Checked here, on the skill, rather than left to the agent emitted
        # for a skill that delegates: a skill that does not delegate has no
        # such agent, and a message about one names a file nobody wrote.
        for tool_name in skill.get("tools", []):
            validate_agent_tool(tool_name,
                                declared_tools,
                                skill["name"],
                                kind="skill")
        unknown = set(skill["delegates"]) - name_set - skill_names
        if unknown:
            raise TpuAgentSuiteError(
                f"skill {skill['name']} delegates to unknown agents:"
                f" {sorted(unknown)}")

    # 5. Agent definitions & files
    agents_by_name = {agent["name"]: agent for agent in manifest["agents"]}
    for agent in manifest["agents"]:
        for tool_name in agent.get("tools", []):
            validate_agent_tool(tool_name, declared_tools, agent["name"])

        unknown_delegates = (set(agent.get("delegates", [])) - name_set -
                             skill_names)
        if unknown_delegates:
            raise TpuAgentSuiteError(
                f"agent {agent['name']} delegates to unknown agents:"
                f" {unknown_delegates}")

        source_inst = rooted(package_dir, agent["instructions"],
                             "agent instructions")
        if not source_inst.is_file():
            raise TpuAgentSuiteError(
                f"missing agent instructions: {source_inst}")
        inst_content = source_inst.read_text(encoding="utf-8")
        if inst_content.lstrip().startswith("---"):
            fm, body = parse_frontmatter_document(inst_content,
                                                  source_inst.name)
            if fm.get("name") == agent["name"]:
                reject_plain_package_paths(body, f"agent {agent['name']}",
                                           body_first_line(inst_content, body))
                render_agent_references(body, manifest)
                validate_resource_tokens(body, package_dir,
                                         f"agent {agent['name']}",
                                         shipped_by(agent), assets)
            else:
                raise TpuAgentSuiteError(
                    f"agent instructions file '{source_inst.name}' "
                    f"frontmatter name '{fm.get('name')}' does not match "
                    f"expected agent name '{agent['name']}'")
        else:
            reject_plain_package_paths(inst_content, f"agent {agent['name']}")
            render_agent_references(inst_content, manifest)
            validate_resource_tokens(inst_content,
                                     package_dir, f"agent {agent['name']}",
                                     shipped_by(agent), assets)

        for schema_key in ("accepts", "produces"):
            val = agent.get(schema_key)
            if not val:
                continue
            schema_path = rooted(package_dir, val, f"agent {schema_key}")
            if not schema_path.is_file():
                raise TpuAgentSuiteError(
                    f"missing agent {schema_key} schema: {schema_path}")

        for ref_file in agent.get("references", []):
            ref_path = rooted(package_dir, ref_file,
                              f"agent {agent['name']} reference file")
            if not ref_path.is_file():
                raise TpuAgentSuiteError(
                    f"agent {agent['name']} reference file does not exist: {ref_path}"
                )

        for script_file in agent.get("scripts", []):
            script_path = rooted(package_dir, script_file,
                                 f"agent {agent['name']} script file")
            if not script_path.is_file():
                raise TpuAgentSuiteError(
                    f"agent {agent['name']} script file does not exist: {script_path}"
                )

    # 6. Delegation graph traversal: cycle detection, max depth <= 3, reachability
    validate_delegation_graph(skills, agents_by_name, name_set)

    # 6b. `ask` belongs to skills alone. A skill is what the user talks to; an
    # agent's caller is blocked waiting on its result, so a question raised by
    # an agent reaches nobody -- granting the tool there would compile a
    # promise the harness cannot keep. Checked here rather than left to review
    # because the failure is silent at run time: the agent calls a tool it
    # holds and no one answers. The agent built from a skill that delegates
    # is that skill's own document, so it keeps what the skill holds.
    skill_names = {skill["name"] for skill in skills}
    for agent in agents_by_name.values():
        if agent["name"] in skill_names:
            continue
        offending = sorted(
            set(agent.get("tools") or []) & utils.SKILL_ONLY_TOOLS)
        if offending:
            raise TpuAgentSuiteError(
                f"agent {agent['name']} declares {', '.join(offending)}, which"
                " only a skill may hold: an agent has no user to ask. Return"
                " status needs_input and let the caller ask instead.")

    # 7. Skill documents and their references, then environment files
    for skill in skills:
        label = f"skill {skill['name']}"
        entry_inst = rooted(package_dir, skill["instructions"],
                            f"{label} instructions")
        if not entry_inst.is_file():
            raise TpuAgentSuiteError(
                f"missing {label} instructions: {entry_inst}")
        entry_content = entry_inst.read_text(encoding="utf-8")
        # load_skill_document has already required frontmatter and checked
        # the name in it, so only the body is left to check.
        _, body = parse_frontmatter_document(entry_content, entry_inst.name)
        reject_plain_package_paths(body, label,
                                   body_first_line(entry_content, body))
        render_agent_references(body, manifest)
        validate_resource_tokens(body, package_dir, label, shipped_by(skill),
                                 assets)
        for ref_file in skill.get("references", []):
            ref_path = rooted(package_dir, ref_file, f"{label} reference file")
            if not ref_path.is_file():
                raise TpuAgentSuiteError(
                    f"{label} reference file does not exist: {ref_path}")
        for script_file in skill.get("scripts", []):
            script_path = rooted(package_dir, script_file,
                                 f"{label} script file")
            if not script_path.is_file():
                raise TpuAgentSuiteError(
                    f"{label} script file does not exist: {script_path}")

    # A markdown reference reaches its reader as a prompt does, so its tokens
    # are held to the same rules as a prompt's. It is checked against its own
    # frontmatter, not its reader's, so once is enough however many read it.
    validated: set[str] = set()
    for unit in (*skills, *manifest["agents"]):
        for ref_file in unit.get("references", []):
            if ref_file in validated:
                continue
            validated.add(ref_file)
            ref_path = package_dir / ref_file
            if ref_path.suffix == ".md" and ref_path.is_file():
                text = ref_path.read_text(encoding="utf-8", errors="ignore")
                _, body = (parse_frontmatter_document(text, ref_path.name)
                           if text.lstrip().startswith("---") else ({}, text))
                reject_plain_package_paths(body, f"reference {ref_file}",
                                           body_first_line(text, body))
                render_agent_references(body, manifest)
                validate_resource_tokens(
                    body, package_dir, f"reference {ref_file}",
                    shipped_by_reference(package_dir, ref_file), assets)

    questions = rooted(package_dir, manifest["environment"]["questions"],
                       "environment questions")
    env_schema = rooted(package_dir, manifest["environment"]["schema"],
                        "environment schema")
    if not questions.is_file() or not env_schema.is_file():
        raise TpuAgentSuiteError("environment questions or schema is missing")

    # 8. Hook definitions & validation
    hook_names: set[str] = set()
    for hook in manifest.get("hooks", []):
        h_name = hook.get("name")
        if not h_name:
            raise TpuAgentSuiteError("hook declaration is missing 'name'")
        if h_name in hook_names:
            raise TpuAgentSuiteError(
                f"duplicate hook name declared: '{h_name}'")
        hook_names.add(h_name)

        event = hook.get("event")
        supported_events = get_supported_hook_events()
        if event not in supported_events:
            raise TpuAgentSuiteError(
                f"hook '{h_name}' has unsupported event '{event}'; "
                f"expected one of {supported_events}")

        matchers = hook.get("matchers")
        if matchers is not None:
            if not isinstance(matchers, list) or not all(
                    isinstance(m, str) for m in matchers):
                raise TpuAgentSuiteError(
                    f"hook '{h_name}' matchers must be a list of strings")
            check_hook_matchers(h_name, event, matchers)

        action = hook.get("action")
        command = hook.get("command")

        if not action and not command:
            raise TpuAgentSuiteError(
                f"hook '{h_name}' must specify either an 'action' or 'command'"
            )
        if action and command:
            raise TpuAgentSuiteError(
                f"hook '{h_name}' cannot specify both 'action' and 'command'")

        if action:
            if not isinstance(action,
                              str) or not (action.startswith("script:")
                                           or action.startswith("builtin:")):
                raise TpuAgentSuiteError(
                    f"hook '{h_name}' action must start with 'script:' or 'builtin:', got '{action}'"
                )
            if action.startswith("builtin:"):
                builtin_target = action[len("builtin:"):]
                allowed_events = VALID_BUILTIN_HOOK_ACTIONS.get(builtin_target)
                if allowed_events is None:
                    raise TpuAgentSuiteError(
                        f"hook '{h_name}' unknown builtin action '{builtin_target}', expected one of {sorted(VALID_BUILTIN_HOOK_ACTIONS)}"
                    )
                if event not in allowed_events:
                    raise TpuAgentSuiteError(
                        f"hook '{h_name}' builtin action '{builtin_target}' cannot be attached to event '{event}', expected one of {sorted(allowed_events)}"
                    )
            elif action.startswith("script:"):
                script_path = rooted(package_dir, action[len("script:"):],
                                     f"hook '{h_name}' script file")
                if not script_path.is_file():
                    raise TpuAgentSuiteError(
                        f"hook '{h_name}' script file does not exist: {script_path}"
                    )
        elif command:
            if not isinstance(command, str) or not command.strip():
                raise TpuAgentSuiteError(
                    f"hook '{h_name}' command must be a non-empty string")

        timeout = hook.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(
                timeout, bool) or timeout <= 0):
            raise TpuAgentSuiteError(
                f"hook '{h_name}' timeout must be a positive integer, got {timeout!r}"
            )

        on_failure = hook.get("on_failure", DEFAULT_HOOK_ON_FAILURE)
        if on_failure not in utils.HOOK_FAILURE_POLICIES:
            raise TpuAgentSuiteError(
                f"hook '{h_name}' on_failure must be one of "
                f"{list(utils.HOOK_FAILURE_POLICIES)}, got {on_failure!r}")

    # 9. Workspace memory block content. A block is prose an agent reads in
    #    every session, so it is held to what a prompt is held to. It belongs
    #    to the package rather than to a unit, so a script or reference token
    #    in one may name anything some unit of this build ships.
    shipped_by_any = [
        path for unit in (*skills, *manifest["agents"])
        for path in shipped_by(unit)
    ]
    for entry in manifest.get("workspace_memory", []):
        content = rooted(package_dir, entry["content"],
                         "workspace memory content")
        if not content.is_file():
            raise TpuAgentSuiteError(
                f"missing workspace memory content file: {content}")
        label = f"workspace memory block {entry['marker']!r}"
        block = content.read_text(encoding="utf-8")
        reject_plain_package_paths(block, label)
        render_agent_references(block, manifest)
        validate_resource_tokens(block, package_dir, label, shipped_by_any,
                                 assets)

    return manifest


def target_for(manifest: dict[str, Any], harness: str) -> dict[str, Any]:
    """Retrieve and validate the compatibility target configuration for a given harness."""
    target = next(
        (t for t in manifest.get("compatibility_targets", [])
         if t.get("harness") == harness),
        None,
    )
    if not target:
        raise TpuAgentSuiteError(
            f"no compatibility target for harness {harness}")

    versions = target.get("versions")
    if versions is None:
        raise TpuAgentSuiteError(
            f"compatibility target for {harness} is missing 'versions'")
    parse_version_range(versions)

    return target


VERIFY_CONTENT_CHECKS = (
    lint.check_instruction_links,
    lint.check_reference_links,
    lint.check_environment_contract,
    lint.check_workspace_config_paths,
)


def validate_requirements_catalog(req_path: Path) -> None:
    """Validate that every package entry in a requirements catalog specifies a version constraint."""
    if not req_path.is_file():
        return
    for line_no, raw_line in enumerate(
            req_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line_no_comment = line.split("#", 1)[0].strip()
        req_part = line_no_comment.split(";", 1)[0].strip()
        if not any(ch in req_part for ch in (">", "=", "<", "~", "@")):
            raise TpuAgentSuiteError(
                f"Unpinned package in '{req_path.name}' at line {line_no}: '{raw_line}'. "
                "All dependencies in central requirements catalogs must specify version constraints (e.g. '>=', '==')."
            )


def verify_package(package_dir: Path,
                   allow_extra_fields: bool = False,
                   skills: Sequence[str] | None = None,
                   manifest_name: str = DEFAULT_MANIFEST) -> None:
    """Verify package manifest, schema definitions, and target compatibility."""
    # Ensure no scattered requirements files exist in sub-directories
    allowed_reqs = {
        (package_dir / "requirements-client.txt").resolve(),
        (package_dir / "requirements-tpu.txt").resolve(),
    }
    for req_file in package_dir.rglob("*requirements*.txt"):
        rel_path = req_file.relative_to(package_dir)
        if dist_excluded(rel_path):
            continue  # never ships (pinned source checkouts under raw/code, tool state)
        if req_file.resolve() not in allowed_reqs:
            raise TpuAgentSuiteError(
                f"Scattered requirements file found at '{rel_path}'. "
                "All Python dependencies must be declared in "
                "'requirements-client.txt' (host) or 'requirements-tpu.txt' "
                "(TPU workloads) at the package root.")

    host_req_path = package_dir / "requirements-client.txt"
    tpu_req_path = package_dir / "requirements-tpu.txt"
    validate_requirements_catalog(host_req_path)
    validate_requirements_catalog(tpu_req_path)

    manifest = load_package(package_dir,
                            allow_extra_fields=allow_extra_fields,
                            skills=skills,
                            manifest=manifest_name)
    print_delegation_warnings(manifest)
    for schema_rel in manifest.get("schemas", []):
        schema_path = rooted(package_dir, schema_rel, "schema path")
        schema = load_json(schema_path)
        errors = validate_schema_definition(schema)
        if errors:
            raise TpuAgentSuiteError(f"invalid schema {schema_rel}:\n" +
                                     "\n".join(errors))

    harnesses = {
        target["harness"]
        for target in manifest["compatibility_targets"]
    }
    for harness in harnesses:
        target_for(manifest, harness)

    for check in VERIFY_CONTENT_CHECKS:
        findings = check(package_dir, manifest)

        if findings:
            raise TpuAgentSuiteError(f"{findings[0].check}:\n" +
                                     "\n".join(f"  {f}" for f in findings))

    # Verify declared dependencies exist in central catalogs
    declared_host = manifest.get("host_dependencies", [])
    if declared_host:
        if not host_req_path.is_file():
            raise TpuAgentSuiteError(
                f"Package declares host_dependencies but '{host_req_path.name}' is missing at the package root."
            )
        known_host = extract_requirement_names(
            host_req_path.read_text(encoding="utf-8"))
        for dep in declared_host:
            if canonical_pkg_name(dep) not in known_host:
                raise TpuAgentSuiteError(
                    f"Undeclared host dependency: '{dep}'. All host dependencies "
                    f"must be declared with versions in '{host_req_path.name}' at the package root."
                )

    declared_tpu = manifest.get("tpu_dependencies", [])
    tpu_req_path = package_dir / "requirements-tpu.txt"
    if declared_tpu:
        if not tpu_req_path.is_file():
            raise TpuAgentSuiteError(
                f"Package declares tpu_dependencies but '{tpu_req_path.name}' is missing at the package root."
            )
        known_tpu = extract_requirement_names(
            tpu_req_path.read_text(encoding="utf-8"))
        for dep in declared_tpu:
            if canonical_pkg_name(dep) not in known_tpu:
                raise TpuAgentSuiteError(
                    f"Undeclared TPU dependency: '{dep}'. All TPU dependencies "
                    f"must be declared with versions in '{tpu_req_path.name}' at the package root."
                )


# =====================================================================
# Document & Prompt Generators
# =====================================================================


def generated_header(manifest: dict[str, Any], target: dict[str, Any]) -> str:
    """Generate standard header (disabled to reduce prompt token usage)."""
    return ""


def compute_contract_section(runtime: str, pkg_tag: str) -> str:
    """Render the accelerator-access contract injected into agent prompts.

    Backends and their flags come from the driver registry rather than a
    hand-written list, so the prompt can never advertise a backend this build
    does not have -- an agent following a stale list gets INVALID_BACKEND.

    Every command carries `--package`, because the package's compute
    profiles (environment.json, written by `compute setup apply`) are found
    per package, and the resolver only looks for them when it is told which
    package to look for. Without the flag a set-up package resolves to
    NO_COMPUTE_BACKEND, which reads as "nothing is configured" rather than
    "you did not say whose configuration to read".
    """
    env_hint = (f"the package's installed environment (view it with"
                f" `{runtime} compute inspect --package {pkg_tag} --json`)")
    backends = compute.describe_backends()
    required = compute.describe_required_flags()
    if backends:
        lines = []
        for name, flags in backends.items():
            needed = required.get(name, set())
            rendered = " ".join(f"`{flag} <value>`" +
                                ("" if flag in needed else " (optional)")
                                for flag in flags)
            lines.append(
                f"- `{name}`: {rendered or '(no backend-specific flags)'}")
        backend_text = (
            f"Configuration inherits from {env_hint}. When "
            "overriding `--backend` via CLI, supply that backend's arguments:\n"
            + "\n".join(lines))
    else:
        backend_text = (
            f"Configuration inherits from {env_hint}. This "
            f"build registers no compute backends; run `{runtime} compute "
            "--help` to see what is available before using these commands.")

    compute_cmd = f"{runtime} compute --package {pkg_tag}"
    # `wait` has its own line: it takes its own flags, and its point -- use it
    # instead of a sleep loop -- is lost in a list of action names.
    job_actions = "|".join(a for a in compute.JOB_ACTIONS if a != "wait")
    cli = compute.cli
    setup_guide = (
        f"{utils.central_home_spelling_for(*utils.parse_package_tag(pkg_tag))}"
        f"/{utils.REFERENCES_DIRNAME}/{compute.SETUP_GUIDE.name}")
    xpk_note = (
        " On `gke_xpk` the recipe instead runs on this machine, with"
        " `CLUSTER_NAME`, `PROJECT_ID` and `ZONE` set, and must itself create"
        " the XPK workload named `$WORKLOAD_NAME`; `--input`/`--output` are"
        " refused there." if "gke_xpk" in backends else "")
    return f"""## Accelerator Compute Access
Whenever you need to interact with TPU or compute, execute via `{compute_cmd}`.
Every job, server, log and port-forward goes through it. Don't run `kubectl`, `ssh`, `scp`, `xpk` or `gcloud compute` yourself, not even to read; a recipe that compute runs may. The cluster and its namespace are shared with other runs and with people: compute only acts on jobs this run submitted, and a raw command can't tell which those are. Anything already there belongs to someone else: don't read, copy, benchmark, delete or reuse it. When a compute command fails, read its error, which names the cause and usually the next command; if it still fails, stop and report the command and its error rather than working around it with raw cluster commands.
{backend_text}

Commands:
- Inspect hardware and health: `{compute_cmd} inspect [--json]`
- Releases of jax, jaxlib, libtpu, torch and torch_tpu on the target, for citing docs written for that release: `{compute_cmd} versions [--json]`
- Run synchronous workloads: `{compute_cmd} run --code <path> [--name <job_name>] [--timeout <s>] [--interpreter <python>] [--input <local>[:<name>]]... [--output <name>[:<local>]]... [-- <script args>]`
- Submit asynchronous batch workloads: `{compute_cmd} submit --recipe <path> [--name <job_name>] [--run-id <run_id>] [--interpreter <python>] [--serve] [--input <local>[:<name>]]... [--output <name>]... [-- <recipe args>]`
- Manage jobs and pull artifacts: `{compute_cmd} job [{job_actions}] [--job-id <id>] [--run-id <id>] [--include-serving]`
- Wait for a job instead of sleeping or polling: `{compute_cmd} job wait --job-id <id> [--until running|done|log:<regex>] [--fail-on-log <regex>]... [--timeout <s>]`. Exit {cli.WAIT_REACHED}: reached; {cli.WAIT_FAILURE_LOGGED}: a failure line was logged; {cli.WAIT_TIMED_OUT}: timed out after `--timeout` (default {cli.WAIT_TIMEOUT_SEC}s), so run it again; {cli.WAIT_JOB_ENDED}: the job ended first; 1: `wait` itself failed (read stderr).
- Forward ports to remote jobs: `{compute_cmd} port-forward --port [LOCAL:]REMOTE [--job-id <id>] [--background]`

For later `job` commands, use the `job_id` that `run` and `submit` print, not your `--name`: on `gke_job` the two can differ. Batch jobs are tagged with the session's run automatically; `--run-id <run_id>` overrides that when your request names a different run to work in. Workloads launched with `--serve` are exempt from automatic session reaping: stop one with `job cancel --job-id <id>`, or clear a run's serving jobs with `job reap --include-serving`. Port forwarding blocks in the foreground by default; pass `--background` (or run with `&` in shell) to detach and print the PID so the command does not hang. `job artifacts` defaults to the run's artifact directory (`{utils.RUNS_HOME_SPELLING}/<run_id>/artifacts/<job_id>/`); pass `--out-dir` to override. Each `--output <name>` of a `submit` lands there as `<dir>/<name>` and nowhere else: unlike `run`, `submit` takes no `:<local>` destination, so when you need a file in your work directory, copy it from the path `job artifacts` prints (or pass `--out-dir <work>`). It exits non-zero and names the files in `missing_outputs` when the workload never produced something it declared with `--output`, however the job's own status reads: report that the workload did not produce its result and stop, rather than continuing from whatever the artifact directory happens to hold. Only `--code` and `--recipe` reach the target by themselves: declare files the workload reads with `--input` and files it writes with `--output`; both name the file relative to the workload's working directory on the target, so pass bare names after `--`.

A recipe (`--recipe`, like `run`'s `--code`) is one local script file, copied to the target and executed there with the arguments after `--`, in the working directory that holds its `--input` files and collects its `--output` files. The runner is picked by extension: a `.sh` file runs under `bash`; any other file is a Python script. There is no recipe schema or config format: the script is the whole workload.{xpk_note} Python scripts run under the interpreter set in compute setup (`python_interpreter` in the compute profile's `params` in the workspace's `environment.json`), or the target's `python3` if none was configured; `--interpreter <python>` overrides it for one call. Nothing activates a virtualenv -- name its `bin/python3` directly. A `.sh` recipe gets that same interpreter as `$TPU_AGENT_SUITE_PYTHON`: call it instead of a bare `python3`, and look for tools installed beside it (such as a virtualenv's commands) in its directory. For example, a `train.sh` holding `exec "$TPU_AGENT_SUITE_PYTHON" train.py "$1"` runs with `{compute_cmd} submit --recipe train.sh --input train.py -- <config>`; then `job wait --job-id <id> --until done`.

Session start says whether compute is set up for this run, and the message is in your context. `COMPUTE_NOT_CONFIGURED` (no usable profile) and the other `*_NOT_CONFIGURED` codes such as `STAGING_NOT_CONFIGURED` (a field asked only when first needed; `missing` names it and `fix` is the command that asks for it) are fixed with the user by following the compute setup guide, `{setup_guide}`. Only the main session asks the user: elsewhere, stop and report the error with its `missing` and `fix`. A compute command checks the target again first if its profile changed; a failed check lists each failure and its fix, which needs a person (credentials, cluster access, configuration), so report it to the user rather than working around it. `CLUSTER_ACCESS_DENIED` means the cluster refused both the gcloud account and application default credentials; report its `fix` to the user.

`run` holds your shell call open until the workload returns, and a shell call that hits its own timeout kills the run with it. Pass `--timeout <s>`, and if your shell tool takes a per-call timeout, set it to at least `<s>` + {SHELL_TIMEOUT_HEADROOM_S} seconds. Keep `--timeout` at or under {MAX_SYNC_COMPUTE_TIMEOUT_S}; longer work goes through `submit`.

A shared cluster may stop any job a fixed time after it starts running (`declared_duration_minutes` in `compute inspect`'s `details` when configured, or a limit the prompt or an eviction message states), counted from when the job started running (`job status` shows its `start_time`):
- Budget: everything a job does must fit inside the limit, a server's startup (often 15-30 minutes) included. Plan on about 75% of it as usable, and before starting a long phase on a server that is already up, check what is left of it: the limit less the time since its `start_time`.
- Waits: wait in `job wait` calls of `--timeout` 600 or less and run the same one again on exit {cli.WAIT_TIMED_OUT}, never in one long blocking wait or a sleep loop, so a job the cluster stops is noticed within minutes.
- Exit {cli.WAIT_JOB_ENDED}: the job ended first; read its `error`. If the job had run and the `error` says the cluster stopped or deactivated it, or names a time limit, it outgrew the limit: split it into shorter jobs or resume from a checkpoint rather than resubmitting it as it was. If it never started (its pod never became Ready, or Kueue would not admit it), fix that cause -- image, topology, a crash, the queue or priority -- before submitting again.
- If `job status` says Kueue preempted and requeued the job, it is waiting to run again, not lost: run the same wait again, or cancel and resubmit if what is left of the limit no longer fits the work."""


def journal_contract_section(runtime: str, agent_name: str) -> str:
    """Return the prompt section that tells an agent to keep the session journal.

    The session journal is one shared log per run (`journal.jsonl`, shown as
    `experiment.md`) where agents record what they tried, why, and what
    happened, so the user and the next agent can follow the work. Writing it
    needs the `runtime.py journal` command, so this section goes into every
    agent and skill that has a shell, with `agent_name` filled in as the
    entry's author. Agents' own instructions don't need to mention it.
    """
    run_dir = f"{utils.RUNS_HOME_SPELLING}/<run_id>"
    journal_cmd = f"{runtime} journal"
    kinds = "|".join(journal.KINDS)
    return f"""## Session journal
Keep the session's experiment journal, so the user, your caller and the next agent can see what was tried, why, and what it showed. One per run, shared by every agent in it: `{run_dir}/{journal.JOURNAL_FILE}`, re-rendered after every entry to `{run_dir}/{journal.RENDERED_FILE}`. Both sit beside `artifacts/`, not in it: they are the one exception to keeping every output under `artifacts/` or your own work directory. Write them only with these commands, never by editing either file:
- `{journal_cmd} show --run-id <run_id> [--open] [--agent <name>] [--kind <kind>]`
- `{journal_cmd} add --run-id <run_id> --agent {agent_name} --kind <{kinds}> --text "<one or two sentences>" [--parent <id>] [--verdict supported|refuted|inconclusive] [--evidence <path or job id>]... [--metric key=value]...`

`add` prints the entry as JSON; keep its `id` to refer to it later.
- `hypothesis`: before an experiment, what will change, by how much and why; give a number when there is one.
- `experiment`: when it launches; `--parent` its hypothesis, `--evidence` the job id or config path.
- `result`: when it is measured; `--parent` its hypothesis, `--verdict`, `--evidence` the file or job the numbers came from, and the numbers as `--metric`. Say whether they are measured or modelled.
- `rejected`: a hypothesis dropped untested; `--parent` it, and why.
- `decision`: a route, an accept or revert, a stop, with the reason.
- `blocked`: you cannot go on at all; say what would unblock it. A failure you or your caller can retry is a `note`.
- `note`: anything else the next reader needs.

Run `show` before you start, and don't re-run a hypothesis this run already refuted unless the new one says what is different. Write as you go, not at the end: the journal is how a crash, a compaction or the next agent picks up the work. It is a record, not a dependency: if a write is refused, fix the call (the message says what is missing); if the command itself fails, say so in your result and carry on. Never put credentials or tokens in it."""


# How each harness handles tool restrictions declared in a skill's frontmatter.
# Claude Code enforces them with the skill-tools hook, which limits the main
# thread only. Other harnesses can't enforce them, so the build warns instead
# of silently dropping the restriction. A skill that delegates also ships as
# an agent of the same name, and its `tools` there are the agent's: a harness
# started with `--agent <skill>` (Antigravity's runner starts tpu-agent
# that way) enforces them as it does any agent's.
#
# Not Claude Code's own `disallowed-tools` skill field: that removes the tools
# from the session's pool for the rest of the turn, and subagents started in
# that turn draw from the same pool. A headless trial is one turn, so every
# specialist tpu-agent delegated to ran without Bash.
SKILL_TOOLS_HOOK = "hook"
SKILL_TOOLS_UNENFORCED = "unenforced"
SKILL_TOOL_RESTRICTION: dict[str, str] = {
    "claude-code": SKILL_TOOLS_HOOK,
    "codex": SKILL_TOOLS_UNENFORCED,
    "antigravity": SKILL_TOOLS_UNENFORCED,
    "cursor": SKILL_TOOLS_UNENFORCED,
}


def skill_tools_enforced(skill: dict[str, Any], harness: str) -> bool:
    """Whether the harness enforces the skill's `tools` limit; warns if it can't."""
    granted = skill.get("tools") or []
    if not granted:
        return False
    mode = SKILL_TOOL_RESTRICTION.get(harness)
    if mode is None:
        raise TpuAgentSuiteError(
            f"Skill '{skill['name']}' declares tools {sorted(granted)}, but harness "
            f"'{harness}' is missing from SKILL_TOOL_RESTRICTION.")
    if mode == SKILL_TOOLS_UNENFORCED:
        print(
            f"WARNING: Skill '{skill['name']}' declares tools {sorted(granted)}, but harness "
            f"'{harness}' cannot enforce tool restrictions on skills. All tools will remain available.",
            file=sys.stderr,
        )
        return False
    return True


# Tools the skill-tools hook always allows, as the hook sees them (portable
# names for mapped tools, native names otherwise). None run commands or edit files.
SKILL_ALWAYS_ALLOWED_TOOLS: dict[str, list[str]] = {
    "claude-code": [
        "delegate",
        "Skill",
        "ToolSearch",
        "TodoWrite",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskUpdate",
        "TaskOutput",
    ],
}


def hook_allowed_skill_tools(manifest: dict[str, Any],
                             harness: str) -> dict[str, list[str] | None]:
    """Map each skill to the tool names the skill-tools hook allows (None = no limit)."""
    enforced = SKILL_TOOL_RESTRICTION.get(harness) == SKILL_TOOLS_HOOK
    always = SKILL_ALWAYS_ALLOWED_TOOLS.get(harness, [])
    return {
        skill["name"]: (sorted({*skill["tools"], *always})
                        if enforced and skill.get("tools") else None)
        for skill in manifest["skills"]
    }


def skill_frontmatter(name: str, description: str) -> str:
    """Generate YAML frontmatter for a skill."""
    escaped = description.replace('"', "'")
    return f'---\nname: {name}\ndescription: "{escaped}"\n---\n\n'


def package_runtime_paths(manifest: dict[str, Any]) -> tuple[str, str, str]:
    """Return (pkg_tag, pkg_path, runtime_command) for manifest."""
    name = manifest["name"]
    version = manifest.get("version", "")
    pkg_tag = utils.format_package_tag(name, version)
    pkg_path = utils.central_home_spelling_for(name, version)
    runtime = f"python3 {pkg_path}/framework/runtime.py"
    return pkg_tag, pkg_path, runtime


def get_manifest_agent_map(
    manifest: dict[str, Any], ) -> dict[str, dict[str, Any]]:
    """Index the manifest's agents by name."""
    return {a["name"]: a for a in manifest.get("agents", []) if "name" in a}


def delegation_contract_section(
    manifest: dict[str, Any],
    delegates: list[str],
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Generate delegation request contract section showing required JSON payload shapes."""
    if not delegates:
        return ""
    all_agents = get_manifest_agent_map(manifest)

    groups: dict[str, list[str]] = {}
    for d in delegates:
        clean_name = (d[len(agent_prefix):]
                      if agent_prefix and d.startswith(agent_prefix) else d)
        agent_obj = all_agents.get(d) or all_agents.get(clean_name)
        if not agent_obj:
            continue
        accepts = agent_obj.get("accepts")
        if accepts:
            groups.setdefault(accepts, []).append(d)

    if not groups:
        return ""

    lines = [
        "### Delegation request payloads",
        "",
        "When delegating, format the request as a structured JSON object in the delegation prompt/message parameter matching the target contract.",
        "Provide task inputs only; do NOT specify output format templates, response schemas, or examples in the delegation prompt.",
        "",
    ]
    for schema_path, targets in groups.items():
        example = load_schema_example(package_dir, schema_path)
        targets_str = ", ".join(f"`{t}`" for t in targets)
        lines.append(f"- Targets: {targets_str} (`{schema_path}`):")
        if example is not None:
            lines.append("```json")
            lines.append(json.dumps(example, indent=2))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()


def response_deliverable_section(
    agent: dict[str, Any],
    package_dir: Path | None = None,
) -> str:
    """Generate response deliverable contract section showing required JSON result shape."""
    produces = agent.get("produces")
    if not produces:
        return ""
    example = load_schema_example(package_dir, produces)
    lines = [
        "### Response deliverable format",
        "",
        f"When completing your task, your final response MUST be a JSON codeblock conforming to `{produces}`. Runtime validation rejects a response without it.",
        "",
    ]
    if example is not None:
        lines.append("```json")
        lines.append(json.dumps(example, indent=2))
        lines.append("```")
    return "\n".join(lines).strip()


def skill_package_contract(
    manifest: dict[str, Any],
    skill: dict[str, Any],
    harness: str,
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Build the runtime execution contract section for an installed skill package."""
    delegates = [agent_prefix + name for name in skill.get("delegates", [])]
    # The journal asks the skill to run commands, so it follows what the skill
    # declares, on every harness. Compute guidance is a guard on commands the
    # skill can run at all, so it also goes where the harness can't remove
    # the shell.
    declares_shell = not skill.get("tools") or "shell" in skill["tools"]
    shell_available = (declares_shell or SKILL_TOOL_RESTRICTION.get(harness)
                       == SKILL_TOOLS_UNENFORCED)
    targets = ", ".join(delegates) or "none"
    pkg_tag, pkg_path, runtime = package_runtime_paths(manifest)

    sections = [
        "## Installed package contract",
        "",
        f"- Read `.tpu-agent-suite/{utils.ENVIRONMENT_FILE}` (workspace installs) or `{utils.TPU_AGENT_SUITE_HOME_SPELLING}/{utils.ENVIRONMENT_FILE}` (user-scope installs) before delegating — whichever exists; never any other location. It records which machine the work runs on and which interpreter to use for what; guessing either wastes the run. Never create, copy, move, or edit an environment.json by hand: change it only through `{runtime} compute setup` and `{runtime} config`, following `{pkg_path}/{utils.REFERENCES_DIRNAME}/{compute.SETUP_GUIDE.name}`. If the configured location is missing, report that as the blocker.",
        "- Retain the active `run_id` assigned at session start, include it in every delegation request, and use only that run's artifact references.",
    ]

    delegation_mech = delegation_mechanism_lines(harness, delegates,
                                                 root=True).strip()
    if delegation_mech:
        sections.append(delegation_mech)

    sections.extend([
        f"- Direct delegation targets: {targets}. Do not invoke any other package agent directly.",
        "- Run one branch at a time. A subagent may wait for its one nested helper; do not run sibling branches in parallel.",
        f"- Store durable outputs below `{utils.RUNS_HOME_SPELLING}/<run_id>/artifacts/`; never overwrite an existing artifact.",
    ])

    delegation_payloads = delegation_contract_section(
        manifest, delegates, package_dir, agent_prefix=agent_prefix)
    if delegation_payloads:
        sections.extend(["", delegation_payloads])

    if declares_shell:
        sections.extend(["", journal_contract_section(runtime, skill["name"])])

    compute_sec = (compute_contract_section(runtime, pkg_tag)
                   if shell_available else "")
    if compute_sec:
        sections.extend(["", compute_sec])

    return "\n\n" + "\n".join(sections).strip() + "\n"


def skill_document(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    skill: dict[str, Any],
    agent_prefix: str = "",
) -> str:
    """Generate the SKILL.md of one skill.

    A skill that delegates carries the installed package contract, which
    names its delegates and how to reach them. A skill that does not is a
    plain document: it drives a runtime command rather than orchestrating.
    """
    frontmatter = skill_frontmatter(skill["name"], skill["description"])
    # Called for its warning when the harness can't enforce `tools`.
    skill_tools_enforced(skill, target["harness"])
    header = generated_header(manifest, target)

    source_path = rooted(package_dir, skill["instructions"],
                         f"instructions for skill {skill['name']}")
    if not source_path.is_file():
        raise TpuAgentSuiteError(
            f"skill {skill['name']!r} declares instructions that do not "
            f"exist: {skill['instructions']}")
    raw_content = source_path.read_text(encoding="utf-8")
    if raw_content.lstrip().startswith("---"):
        _, body = parse_frontmatter_document(raw_content, source_path.name)
    else:
        body = raw_content
    rendered_body = render_harness_tokens(
        render_agent_references(body, manifest, agent_prefix).rstrip(),
        target["harness"])

    if not skill.get("delegates"):
        return f"{frontmatter}{header}{rendered_body}\n"
    contract = skill_package_contract(
        manifest,
        skill,
        target["harness"],
        package_dir=package_dir,
        agent_prefix=agent_prefix,
    )

    return f"{frontmatter}{header}{rendered_body}{contract}"


def iter_skill_documents(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent_prefix: str = "",
) -> list[tuple[str, str]]:
    """Yield (skill name, SKILL.md body) for every released skill."""
    return [(skill["name"],
             skill_document(package_dir, manifest, target, skill,
                            agent_prefix)) for skill in manifest["skills"]]


def write_skill_documents(
    skills_root: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent_prefix: str = "",
) -> None:
    """Write every skill's SKILL.md beneath `skills_root`."""
    for name, document in iter_skill_documents(package_dir, manifest, target,
                                               agent_prefix):
        path = skills_root / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")


def load_produces_schema(
    package_dir: Path | None,
    produces: str | None,
) -> dict[str, Any] | None:
    """Load an agent's produces schema, or None if it cannot be read."""
    if not package_dir or not produces:
        return None
    path = Path(package_dir) / produces
    if not path.is_file():
        return None
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return schema if isinstance(schema, dict) else None


def result_contract_lines(schema: dict[str, Any] | None) -> list[str]:
    """State the status vocabulary and payload rules an agent's schema declares.

    Everything is read off the schema, so the generated prompt cannot
    contradict the schema it tells the agent to conform to. The statuses are
    the `status` enum plus any status `status_requires` keys a rule on; a
    schema naming none gets no lines.

    Returns: the status line first, then one line per payload rule. Empty when
    the schema names no statuses.
    """
    statuses: list[str] = []
    payload_rules: dict[str, Any] = {}
    declares_artifacts = False
    if schema:
        properties = schema.get("properties", {})
        declares_artifacts = isinstance(properties,
                                        dict) and "artifacts" in properties
        status_prop = properties.get("status", {}) if isinstance(
            properties, dict) else {}
        if isinstance(status_prop, dict) and isinstance(
                status_prop.get("enum"), list):
            statuses = [str(value) for value in status_prop["enum"]]
        if isinstance(schema.get("status_requires"), dict):
            payload_rules = schema["status_requires"]
            # A rule keyed on a status the enum omits is still a status this
            # schema expects, so it belongs in the sentence naming them.
            statuses += [
                str(status) for status in payload_rules
                if str(status) not in statuses
            ]

    if not statuses:
        return []

    vocabulary = ", ".join(f"`{status}`" for status in statuses)
    lines = [f"- Return a result whose `status` is one of {vocabulary}."]

    if payload_rules:
        for status, fields in payload_rules.items():
            if not isinstance(fields, list) or not fields:
                continue
            owed = ", ".join(f"`{field}`" for field in fields)
            article = "An" if status[:1].lower() in "aeiou" else "A"
            lines.append(
                f"- {article} `{status}` result must carry {owed}, each"
                " non-empty; the runtime rejects it otherwise. If the work did"
                f" not produce them, return another status, not `{status}`.")
    elif declares_artifacts and "completed" in statuses:
        # The runtime enforces this only for a schema declaring `artifacts`,
        # and only on the literal status `completed`; anywhere else the line
        # would promise a check that never runs.
        lines.append(
            "- A `completed` result must carry at least one artifact; the"
            " runtime rejects it otherwise. Write findings to a file under the"
            " run's artifacts directory, not only into the summary.")
    return lines


def agent_execution_contract(
    manifest: dict[str, Any],
    agent: dict[str, Any],
    harness: str,
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Build the runtime execution contract section for an agent prompt."""
    delegates = [agent_prefix + name for name in agent.get("delegates", [])]
    targets = ", ".join(delegates) or "none"
    pkg_tag, pkg_path, runtime = package_runtime_paths(manifest)

    sections = [
        "## Generated execution contract",
        "",
    ]
    if agent.get("accepts"):
        sections.append("- Require the request's `run_id`.")

    produces_schema = load_produces_schema(package_dir, agent.get("produces"))
    result_lines = result_contract_lines(produces_schema)

    if agent.get("produces"):
        sections.extend(result_lines[:1])

    has_shell = "shell" in (agent.get("tools") or [])

    sections.extend([
        f"- Keep every output under `{utils.RUNS_HOME_SPELLING}/<run_id>/artifacts/`.",
        f"- Read only your own run's directory under `{utils.RUNS_HOME_SPELLING}/`, or a run the user specifies. Other runs hold other sessions' work, possibly other people's, so don't search or read them.",
        "- Never overwrite an existing artifact.",
        "- Reject artifact paths that escape the run's artifacts directory.",
        "- Materialize durable outputs.",
        "- Wait for slow work inside the command that started it, not by asking again in a later turn. Every turn resends the whole conversation, so polling for a result costs a full context each time to learn one bit.",
        "- When a command will outlast your shell call's timeout or wait window, raise it for the call past the command's own `--timeout`, or sleep inside the command. Polling is what is left when neither is done, not a technique in its own right.",
    ])

    if agent.get("produces"):
        sections.extend(result_lines[1:])

    delegation_mech = delegation_mechanism_lines(harness,
                                                 delegates,
                                                 root=is_root_agent(
                                                     manifest, agent)).strip()
    if delegation_mech:
        sections.append(delegation_mech)

    sections.extend([
        f"- Allowed delegation targets: {targets}.",
        "- Run sequentially. Never weaken inherited permissions or approvals.",
    ])

    delegation_payloads = delegation_contract_section(
        manifest, delegates, package_dir, agent_prefix=agent_prefix)
    if delegation_payloads:
        sections.extend(["", delegation_payloads])

    response_deliverable = response_deliverable_section(agent, package_dir)
    if response_deliverable:
        sections.extend(["", response_deliverable])

    if has_shell:
        sections.extend(["", journal_contract_section(runtime, agent["name"])])

    compute_sec = compute_contract_section(runtime, pkg_tag)
    if compute_sec:
        sections.extend(["", compute_sec])

    return "\n\n" + "\n".join(sections).strip() + "\n"


def agent_prompt(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent: dict[str, Any],
    agent_prefix: str = "",
) -> str:
    """Generate agent instruction prompt document with generated execution contract."""
    source_path = rooted(package_dir, agent["instructions"],
                         "agent instructions")
    raw_content = source_path.read_text(encoding="utf-8")
    if raw_content.lstrip().startswith("---"):
        _, body = parse_frontmatter_document(raw_content, source_path.name)
    else:
        body = raw_content
    rendered_body = render_harness_tokens(
        render_agent_references(body, manifest, agent_prefix).rstrip(),
        target["harness"])

    contract = agent_execution_contract(
        manifest,
        agent,
        target["harness"],
        package_dir=package_dir,
        agent_prefix=agent_prefix,
    )

    return f"{generated_header(manifest, target)}{rendered_body}{contract}"


def resolve_agent_tools(
    agent: dict[str, Any],
    declared_tools: dict[str, dict[str, Any]],
    include_internal: bool,
) -> list[dict[str, Any]]:
    """Resolve tool metadata for an agent, filtering internal tools with warnings if unrequested."""
    resolved: list[dict[str, Any]] = []
    agent_name = agent.get("name", "unnamed")
    for tool_name in agent.get("tools", []):
        tool = validate_agent_tool(tool_name, declared_tools, agent_name)
        if tool.get("visibility") == "internal" and not include_internal:
            print(
                f"WARNING: Skipping internal tool '{tool_name}' for agent"
                f" '{agent['name']}' (use --include-internal to include).",
                file=sys.stderr,
            )
            continue
        resolved.append(tool)
    return resolved


def load_harness_tool_map(harness: str) -> dict[str, list[str]]:
    """Derive the tool mapping for a harness from tool-mappings.json.

    `tool-mappings.json` is the single source of truth for tool vocabulary
    across both runtime hook interception and compiler frontmatter generation.
    """
    mappings = utils.get_tool_mappings()
    tool_map: dict[str, list[str]] = {}
    for canonical_name, spec in mappings.items():
        if canonical_name == "delegate" or not isinstance(spec, dict):
            continue
        harnesses = spec.get("harnesses") or {}
        h_info = utils.harness_mapping(harnesses, harness)
        tools = h_info.get("tools")
        if tools:
            tools_list = list(tools)
            if canonical_name == "shell" and utils.is_google_harness(harness):
                if "manage_task" not in tools_list:
                    tools_list.append("manage_task")
            tool_map[canonical_name] = tools_list
    return tool_map


CLAUDE_TOOL_MAP: dict[str, list[str]] = load_harness_tool_map("claude-code")
# Two tools for a delegating agent, per how agy behaves headless (`-p`,
# which is how the eval runs):
# - `invoke_subagent` returns a handle; the result later arrives as a message
#   that wakes an idle parent.
# - `manage_subagents` stays for stopping a subagent, not for waiting.
# - But a headless run drops a *root* agent that stays idle for 30 minutes
#   (changelog 1.2.9; not exposed by any flag, setting or env var, and not
#   lifted by `--print-timeout`), with a clean exit and partial output. Short
#   delegations pass; long ones are silently abandoned. So the root (the
#   skill, and its agent twin that `--agent` starts from) is also granted
#   `schedule` and arms a heartbeat on the subagent id that wakes it before
#   the cap and cancels itself when the result arrives. Nested delegators
#   need none: their idleness ends nothing. Staying in the turn and reading
#   the transcript instead is a context-filling busy-wait.
GOOGLE_DELEGATION_TOOLS = ["invoke_subagent", "manage_subagents"]
GOOGLE_DELEGATION_TOOL = GOOGLE_DELEGATION_TOOLS[0]
ANTIGRAVITY_DELEGATION_TOOL = GOOGLE_DELEGATION_TOOL

# Root heartbeat interval: a 2x margin under the 30-minute cap, so an
# hour-long delegation costs a few near-empty turns. 10 minutes was tested.
ANTIGRAVITY_HEARTBEAT_SECONDS = 900

# Granted to the root alone: the heartbeat, and `manage_task` to cancel it
# when a subagent fails -- a headless run waits for a pending timer, so one
# left armed holds the run open until it fires.
GOOGLE_ROOT_TOOLS = ["schedule", "manage_task"]


def is_root_agent(manifest: dict[str, Any], agent: dict[str, Any]) -> bool:
    """Report whether `agent` is a skill's twin, the agent a run starts from.

    A skill that delegates is also built as an agent of the same name
    (`--agent <skill>`), and that is how the eval launches the root: its
    idleness is what ends a headless run, so it alone carries the heartbeat
    and the tool to arm it.
    """
    return any(skill["name"] == agent["name"]
               for skill in manifest.get("skills", []))


# A harness that runs every agent in one context, by construction. Cursor is
# the case: `translate_cursor` inlines each agent into one rules file as a
# persona, so there is no subagent to call and saying otherwise would be a lie.
DELEGATION_NONE = "@none"

# A harness whose delegation call has not been confirmed. The contract stays
# silent about the mechanism rather than naming one that may not exist, and the
# build warns. This is a placeholder, not a resting state: an unnamed mechanism
# is what lets a model fall back to running an agent's instructions inline.
DELEGATION_UNVERIFIED = "@unverified"

# Harnesses whose delegation call the runtime intercepts but nobody has watched
# a model actually make. The name is known; that it works is not. Naming an
# unconfirmed mechanism in a contract is worse than saying nothing, because a
# model that tries it and fails falls back to running the agent inline.
UNVERIFIED_DELEGATION_HARNESSES: frozenset[str] = frozenset({"codex"})

# Harnesses that run every agent in one context by construction. Cursor is the
# case: `translate_cursor` inlines each agent into one rules file as a persona,
# so there is no subagent to call and saying otherwise would be a lie. The
# delegate mapping still lists a tool for it; that tool is for intercepting a
# call this build never emits.
INLINING_HARNESSES: frozenset[str] = frozenset({"cursor"})


def load_harness_delegation_tools() -> dict[str, str]:
    """Read the delegation tool per harness from the shared tool mapping.

    `tool-mappings.json` already answers this for the runtime, which matches
    hook events against the same tool names. Deriving the compiler's answer
    from it keeps one registry: a second copy here would name the same tools
    today and silently disagree the first time a harness is added to one and
    not the other.
    """
    mappings = utils.load_json(
        Path(__file__).resolve().parent / "schemas" / "tool-mappings.json")
    harnesses = ((mappings.get("tools") or {}).get("delegate")
                 or {}).get("harnesses") or {}
    tools: dict[str, str] = {}
    for harness, spec in harnesses.items():
        if harness in INLINING_HARNESSES:
            tools[harness] = DELEGATION_NONE
        elif harness in UNVERIFIED_DELEGATION_HARNESSES:
            tools[harness] = DELEGATION_UNVERIFIED
        else:
            names = spec.get("tools") or []
            if names:
                tools[harness] = names[0]

    # A Google harness that inherits its mapping block inherits the tool named
    # in it. Leaving it out here would build a package whose runtime
    # intercepts a delegation the contract never names.
    for harness in utils.GOOGLE_HARNESSES:
        inherited = utils.harness_mapping(harnesses,
                                          harness).get("tools") or []
        if harness not in tools and inherited:
            tools[harness] = inherited[0]
    return tools


# Which tool call actually performs a delegation, per harness. A harness absent
# from this map cannot be built once anything routes to an agent -- see
# `delegation_tool_for`. Adding a harness means answering this question, and the
# place to answer it is `tool-mappings.json`.
HARNESS_DELEGATION_TOOLS: dict[str, str] = load_harness_delegation_tools()
CLAUDE_DELEGATION_TOOL: str = HARNESS_DELEGATION_TOOLS["claude-code"]


def delegation_tool_for(harness: str, delegates: list[str]) -> str:
    """Return the tool that performs a delegation on `harness`.

    Empty when nothing is delegated. Raises when work *is* routed to an agent
    but the harness has no entry: shipping that package produces instructions
    the runtime cannot honor, and the model's remaining move is to read the
    agent's definition and run it inline -- which silently collapses the
    isolation the delegation existed to provide. Failing the build is cheaper
    than discovering it in a transcript.
    """
    if not delegates:
        return ""
    tool = HARNESS_DELEGATION_TOOLS.get(harness)
    if not tool:
        raise TpuAgentSuiteError(
            f"harness {harness} has no delegation mechanism declared, but"
            f" delegation to {sorted(delegates)} was requested; add it to"
            " HARNESS_DELEGATION_TOOLS or drop the routing")
    if tool == DELEGATION_UNVERIFIED:
        print(
            f"WARNING: harness {harness} delegates to {sorted(delegates)} but"
            " its delegation tool is unverified; the generated contract cannot"
            " name how to delegate.",
            file=sys.stderr,
        )
    return tool


def delegation_is_foreground(harness: str) -> bool:
    """Report whether a delegation call on `harness` blocks until the result.

    Decided by the install plan, not the harness name: a Claude Code build
    sets `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`, so `Agent` runs the subagent
    to completion and returns its result as the call's output. Drop that
    setting and the call returns early; reading it here keeps the contract
    below true either way.
    """
    if utils.is_google_harness(harness):
        return False
    return CLAUDE_CODE_SETTINGS_ENV.get(
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS") == "1"


def delegation_mechanism_lines(harness: str,
                               delegates: list[str],
                               *,
                               root: bool = False) -> str:
    """Render the contract line naming *how* to delegate, not just to whom.

    `root` marks the agent whose idleness ends a headless run (the skill, on
    Antigravity); only it carries the heartbeat.

    Naming targets alone leaves the mechanism to be guessed, and the guess that
    costs the most is doing the work inline: it looks like progress, produces a
    plausible answer, and quietly spends the orchestrator's context on output
    that was supposed to stay behind an agent boundary.

    How to wait is the second guess, and the right answer differs by harness,
    so this is the one place that states it: a skill body is compiled for
    every harness and cannot know which one it is on.
    """
    tool = delegation_tool_for(harness, delegates)
    if not tool or tool == DELEGATION_UNVERIFIED:
        return ""
    if tool == DELEGATION_NONE:
        return ("\n- This harness runs every agent in one context: there is no"
                " subagent to call. Adopt each persona's contract yourself,"
                " in sequence.")
    how = (f"\n- Delegate by calling the `{tool}` tool. Reading an agent's"
           " definition and following it yourself is not delegation: it"
           " discards the isolation delegating exists to provide, and is"
           " never the fallback when the call seems unavailable.")
    if delegation_is_foreground(harness):
        return how + (
            f"\n- `{tool}` runs the subagent to completion and returns its"
            " result as the call's output. Act on that result; there is"
            " nothing to wait for, poll, or read while it runs."
            "\n- Delegate in whole pieces: one call hands over the job;"
            " supervising it step by step costs a call per step.")
    if not utils.is_google_harness(harness):
        # Background delegation on a non-Google harness: the call returns a
        # handle and nothing is known to wake the caller, so it has to follow
        # the work itself. Unmeasured; the install plan forces foreground.
        return how + (
            f"\n- `{tool}` returns a handle, not a result, while the work"
            " runs. Nothing wakes you when it finishes, so a turn that ends"
            " while it is running abandons it: stay in the turn and read the"
            " subagent's transcript, whose path came back with the handle,"
            " until the result is there."
            "\n- Delegate in whole pieces: handing over the job costs one call"
            " plus what you choose to read.")
    # Antigravity: the call returns a handle while the work runs and the
    # result arrives as a message that wakes the parent. A headless run drops
    # an idle *root* after 30 minutes, so the root alone keeps a heartbeat;
    # see the note above GOOGLE_DELEGATION_TOOLS.
    wait = (
        f"\n- `{tool}` returns a handle (the subagent's conversation id), not"
        " a result. The result arrives later as a message that wakes you."
        " A subagent that dies sends no result, only a system message that"
        " it went idle without reporting, stopped, or failed to start: that"
        " message is its result, a failure, not a result still to come.")
    if root:
        wait += (
            "\n- Before ending the turn, call `schedule` with"
            f" `DurationSeconds: {ANTIGRAVITY_HEARTBEAT_SECONDS}` and"
            " `TimerCondition` set to that conversation id: a headless run"
            " drops a root agent idle for 30 minutes, and this wakes you"
            " first. It cancels itself when the result arrives."
            "\n- Then end your turn and wait idle: never read the subagent's"
            " transcript, and never act on a result that has not arrived."
            " What wakes you decides what you do. The result: act on it. A"
            " failure message: cancel the heartbeat with `manage_task`"
            " (`Action: kill`, the task id `schedule` returned) -- otherwise"
            " the run stays open until it fires -- then report. The"
            " heartbeat, with no result yet: one `manage_subagents` call"
            " (`Action: list`), then end the turn either way -- still running,"
            " re-arm the heartbeat first; not running, report it as stopped"
            " without a result and do not re-arm. Beyond that,"
            " `manage_subagents` is only for stopping a subagent no longer"
            " needed.")
    else:
        wait += (
            "\n- Then end your turn and wait idle: never read the subagent's"
            " transcript, never check on it, and never act on a result that"
            " has not arrived. Act on the result when its message wakes you."
            " Use `manage_subagents` only to stop a subagent no longer"
            " needed.")
    return how + wait + (
        "\n- Delegate in whole pieces -- say what to produce and how it will be"
        " checked -- since you will not see intermediate steps.")


def map_agent_tools(
    resolved_tools: list[dict[str, Any]],
    tool_map: dict[str, list[str]],
) -> list[str]:
    """Map an agent's resolved tools into one harness's native vocabulary.

    Order is the declaration order of the agent's tools, de-duplicated: a
    harness config is easier to review when it does not reshuffle between
    builds. Script and MCP tools pass through under their own names.
    """
    tools: list[str] = []
    for tool in resolved_tools:
        if tool["category"] == "native":
            for native in tool_map.get(tool["name"], [tool["name"]]):
                if native not in tools:
                    tools.append(native)
        elif tool["category"] in ("script", "mcp"):
            if tool["name"] not in tools:
                tools.append(tool["name"])
    return tools


def claude_frontmatter(agent: dict[str, Any],
                       resolved_tools: list[dict[str, Any]]) -> str:
    """Generate Claude agent YAML frontmatter with mapped native and custom tools."""
    tools = map_agent_tools(resolved_tools, CLAUDE_TOOL_MAP)

    if agent.get("delegates"):
        tools.append(CLAUDE_DELEGATION_TOOL)

    # Claude Code reads it from an agent's own definition; the Agent tool has
    # no per-call effort. Other harnesses get nothing until an equivalent is
    # confirmed, and run the agent at their default.
    optional = f"effort: {agent['effort']}\n" if agent.get("effort") else ""
    return ("---\n"
            f"name: {agent['name']}\n"
            f"description: {agent['description']}\n"
            f"tools: {', '.join(tools)}\n"
            f"{optional}"
            "background: false\n"
            "---\n\n")


def google_harness_frontmatter(
    agent: dict[str, Any],
    resolved_tools: list[dict[str, Any]],
    tool_map: dict[str, list[str]],
    *,
    root: bool = False,
) -> str:
    """Generate Google harness (Antigravity) YAML frontmatter."""
    tools = map_agent_tools(resolved_tools, tool_map)

    if agent.get("delegates"):
        tools.extend(t for t in GOOGLE_DELEGATION_TOOLS if t not in tools)
        if root:
            tools.extend(t for t in GOOGLE_ROOT_TOOLS if t not in tools)

    tools_yaml = "\n".join(f"  - {t}" for t in tools)
    return ("---\n"
            f"name: {agent['name']}\n"
            f'description: "{agent["description"]}"\n'
            f"tools:\n{tools_yaml}\n"
            "subagent: true\n"
            "---\n\n")


def toml_basic_string(value: str) -> str:
    """Escape a value for a TOML basic (double-quoted) string."""
    return (value.replace("\\", "\\\\").replace('"', '\\"').replace(
        "\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


def toml_multiline_string(value: str) -> str:
    """Escape a value for a TOML multi-line basic string.

    Prompts are authored as markdown and routinely contain backslashes -- an
    escaped paren such as `1\\)` carried over from the source package is
    enough. TOML reads `\\` as the start of an escape sequence, so an
    unescaped prompt produces an agent config that does not parse at all.
    Backslashes are doubled first, then any `\"\"\"` that would otherwise
    terminate the literal early.
    """
    return value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def codex_sandbox_mode(resolved_tools: list[dict[str, Any]]) -> str:
    """Determine Codex sandbox mode based on action_levels of resolved tools."""
    has_host_or_exec = any(
        t.get("action_level", {}).get("scope") == "host"
        or t.get("action_level", {}).get("effect") == "execute"
        or t.get("name") == "shell" for t in resolved_tools)
    if has_host_or_exec:
        return "danger-unrestricted"

    has_workspace_write = any(
        (t.get("action_level", {}).get("scope") == "workspace"
         and t.get("action_level", {}).get("effect") in ("write", "delete"))
        or t.get("name") in ("write", "edit") for t in resolved_tools)
    if has_workspace_write:
        return "workspace-write"

    return "read-only"


def codex_approval_policy(resolved_tools: list[dict[str, Any]]) -> str:
    """Determine Codex interactive approval policy based on tool action levels."""
    if any(
            t.get("action_level", {}).get("approval") == "always"
            for t in resolved_tools):
        return "always"
    if any(
            t.get("action_level", {}).get("approval") == "on-demand"
            for t in resolved_tools):
        return "on-demand"
    return "never"


# =====================================================================
# Harness Translation
# =====================================================================

DIST_EXCLUDED_NAMES: tuple[str, ...] = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    ".git",
    "instructions",
    # wikify-repo working state and agent-skill installs next to a wiki
    ".cache",
    ".wikify",
    ".agents",
    ".claude",
)

# Paths relative to a copied asset root that never ship, whatever they hold. The knowledge
# base pins source checkouts (some private) under raw/code and cites them by URL at the
# pinned commit, so the dist needs the wiki pages, never the code. Matched by path, not by
# name: a "code" directory anywhere else (wiki/code, ...) still ships.
DIST_EXCLUDED_PATHS: tuple[str, ...] = ("raw/code", )


def dist_excluded(rel_path: Path) -> bool:
    """True when a package-relative path never ships: an excluded name anywhere in it, or an
    excluded path pair (`raw/code`) under any asset root. Used by verify so files inside a
    pinned source checkout (their own requirements files, for one) are not the package's."""
    parts = rel_path.parts
    if any(
            fnmatch.fnmatch(part, pat) for part in parts
            for pat in DIST_EXCLUDED_NAMES):
        return True
    joined = "/" + "/".join(parts) + "/"
    return any(f"/{excluded}/" in joined for excluded in DIST_EXCLUDED_PATHS)


def dist_ignore(source: Path) -> Callable[[str, list[str]], set[str]]:
    """copytree ignore callable: excluded names anywhere, excluded paths under source."""
    by_name = shutil.ignore_patterns(*DIST_EXCLUDED_NAMES)
    excluded = {Path(p) for p in DIST_EXCLUDED_PATHS}

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(by_name(directory, names))
        rel = Path(os.path.relpath(directory, source))
        skipped.update(name for name in names if (rel / name) in excluded)
        return skipped

    return ignore


def copy_asset(source: Path, destination: Path) -> None:
    """Copy a file or directory tree to destination, creating parent directories as needed."""
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=dist_ignore(source),
        )
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise TpuAgentSuiteError(f"copy path does not exist: {source}")


# What a build leaves out of framework/: the build tooling, which only runs in
# the repository, the schemas only it reads, and what is for maintainers.
BUILD_ONLY = ("compiler.py", "lint.py", "frontmatter.py", "definitions",
              "templates", "tests", "package-manifest.schema.json",
              "*-frontmatter.schema.json", "*.md", "__pycache__", "*.pyc")


def bundle_framework(destination: Path) -> None:
    """Ship framework/ as is, minus BUILD_ONLY: the runtime and the installer.

    A file keeps its path: framework/compute/cli.py in the repository is
    framework/compute/cli.py in a build and in the central store. Markdown
    stays behind too; compute's setup guide ships rendered under
    references/ (copy_runtime_assets).
    """
    shutil.copytree(Path(__file__).resolve().parent,
                    destination,
                    ignore=shutil.ignore_patterns(*BUILD_ONLY))
    installer = destination / "installer.py"
    installer.chmod(installer.stat().st_mode | 0o111)


def bundle_runtime_schemas(schemas_dir: Path) -> None:
    """Copy the study schemas into the distribution's schemas/.

    The runtime reads every framework schema from framework/schemas/; these
    are here too because the package's own schemas and prompts name them
    beside its request and result schemas.
    """
    schemas_dir.mkdir(parents=True, exist_ok=True)
    framework_schemas_dir = Path(__file__).resolve().parent / "schemas"
    for schema_name in ("study-protocol.json", "study-round.json",
                        "study-baseline.json"):
        src = framework_schemas_dir / schema_name
        if src.is_file():
            shutil.copy2(src, schemas_dir / schema_name)


def copy_runtime_assets(package_dir: Path, output: Path,
                        manifest: dict[str, Any], harness: str) -> None:
    """Copy configured package assets, agent references/scripts, and bundle framework runtime worker scripts."""
    for item in manifest.get("copy", []):
        rel = safe_relative(item, "copy path")
        source = rooted(package_dir, str(rel), "copy path")
        copy_asset(source, output / rel)

    # Memory-block sources are compiler input -- rendered and spliced into the
    # harness memory file at install time -- not distributable content, so a
    # copy tree that contains one sheds it (a verbatim copy would carry raw
    # resource tokens).
    for entry in manifest.get("workspace_memory", []):
        shipped = output / safe_relative(entry["content"],
                                         "workspace memory content")
        shipped.unlink(missing_ok=True)

    shipped: set[str] = set()
    units = [
        *[s for s in manifest.get("skills", []) if isinstance(s, dict)],
        *manifest.get("agents", []),
    ]
    for unit in units:
        for ref_file in unit.get("references", []):
            if ref_file in shipped:
                continue
            shipped.add(ref_file)
            rel = safe_relative(ref_file, "reference path")
            source = rooted(package_dir, str(rel), "reference path")
            # Markdown references reach the reading agent exactly as a prompt
            # does, so every token a prompt may use resolves here too, the
            # harness's included, and the frontmatter, which is compiler
            # input, is left behind; other assets ship verbatim.
            if source.is_file() and source.suffix == ".md":
                text = source.read_text(encoding="utf-8")
                if text.lstrip().startswith("---"):
                    _, text = parse_frontmatter_document(text, source.name)
                destination = output / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(render_markdown(text, manifest,
                                                       harness),
                                       encoding="utf-8")
            else:
                copy_asset(source, output / rel)

        for script_file in unit.get("scripts", []):
            if script_file in shipped:
                continue
            shipped.add(script_file)
            rel = safe_relative(script_file, "script path")
            source = rooted(package_dir, str(rel), "script path")
            copy_asset(source, output / rel)

    for agent in get_manifest_agent_map(manifest).values():
        for s_key in ("accepts", "produces"):
            s_file = agent.get(s_key)
            if s_file:
                rel = safe_relative(s_file, f"agent {s_key} schema path")
                source = rooted(package_dir, str(rel),
                                f"agent {s_key} schema path")
                copy_asset(source, output / rel)

    for hook in manifest.get("hooks", []):
        action = hook.get("action", "")
        if action.startswith("script:"):
            rel = safe_relative(action[len("script:"):], "hook script path")
            source = rooted(package_dir, str(rel), "hook script path")
            copy_asset(source, output / rel)

    for script in manifest.get("script_tools", []):
        if "path" in script:
            rel = safe_relative(script["path"], "script tool path")
            source = rooted(package_dir, str(rel), "script tool path")
            copy_asset(source, output / rel)

    bundle_framework(output / "framework")
    bundle_runtime_schemas(output / "schemas")
    # Session start and the compute contract point at the setup guide, so it
    # ships in every build, whichever skills the build holds.
    guide = output / utils.REFERENCES_DIRNAME / compute.SETUP_GUIDE.name
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text(render_markdown(
        compute.SETUP_GUIDE.read_text(encoding="utf-8"), manifest, harness),
                     encoding="utf-8")


def canonical_pkg_name(name: str) -> str:
    """Normalize a package name according to PEP 503 (lowercase, dashes)."""
    stripped = name.strip()
    if not stripped or stripped.startswith("-"):
        return ""
    base = re.split(r"[><=~!;\s\[]", stripped)[0]
    return base.lower().replace("_", "-")


def extract_requirement_names(requirements_text: str) -> set[str]:
    """Extract canonical package names declared in a requirements file."""
    names: set[str] = set()
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        pkg = canonical_pkg_name(line)
        if pkg:
            names.add(pkg)
    return names


def filter_requirements_content(source_text: str,
                                allowed_names: set[str]) -> str:
    """Filter requirements lines by package name, preserving comments and header."""
    canonical_allowed = {
        canonical_pkg_name(name)
        for name in allowed_names if name.strip()
    }
    output_lines: list[str] = []
    has_requirements = False
    for line in source_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue
        if stripped.startswith("#") or stripped.startswith("-"):
            output_lines.append(line)
            continue
        pkg_name = canonical_pkg_name(stripped)
        if pkg_name in canonical_allowed:
            output_lines.append(line)
            has_requirements = True

    while output_lines and output_lines[-1] == "":
        output_lines.pop()
    if not output_lines or not has_requirements:
        return "# No dependencies required for this distribution.\n"
    return "\n".join(output_lines) + "\n"


def undeclared_catalog_entries(package_dir: Path,
                               manifest: dict[str, Any]) -> list[str]:
    """Catalog entries no built unit declares, as warnings.

    A catalog pins versions; frontmatter says what is needed. An entry that
    nothing declares never ships, which is either a package nothing uses or a
    declaration somebody forgot. Neither can be told from here, so it is
    reported and fails nothing.
    """
    warnings: list[str] = []
    for filename, key in (
        ("requirements-client.txt", "host_dependencies"),
        ("requirements-tpu.txt", "tpu_dependencies"),
    ):
        catalog = package_dir / filename
        if not catalog.is_file():
            continue
        declared = {canonical_pkg_name(name) for name in manifest.get(key, [])}
        listed = extract_requirement_names(catalog.read_text(encoding="utf-8"))
        for name in sorted(listed - declared):
            warnings.append(
                f"{filename} lists {name}, which no built unit declares"
                f" under {key}, so it does not ship")
    return warnings


def emit_distribution_requirements(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    """Emit full or trimmed requirements files into the distribution root."""
    host_req = package_dir / "requirements-client.txt"
    if host_req.is_file():
        if "host_dependencies" in manifest:
            content = filter_requirements_content(
                host_req.read_text(encoding="utf-8"),
                set(manifest.get("host_dependencies", [])),
            )
        else:
            content = host_req.read_text(encoding="utf-8")
        (output / "requirements-client.txt").write_text(content,
                                                        encoding="utf-8")

    tpu_req = package_dir / "requirements-tpu.txt"
    if tpu_req.is_file():
        if "tpu_dependencies" in manifest:
            content = filter_requirements_content(
                tpu_req.read_text(encoding="utf-8"),
                set(manifest.get("tpu_dependencies", [])),
            )
        else:
            content = tpu_req.read_text(encoding="utf-8")
        (output / "requirements-tpu.txt").write_text(content, encoding="utf-8")


def _skill_copies(skill_names: Sequence[str], source_root: str,
                  destination_root: str) -> list[dict[str, str]]:
    """One copy rule per shipped skill."""
    return [{
        "source": f"{source_root}/{name}",
        "destination": f"{destination_root}/{name}",
    } for name in skill_names]


def _google_install_spec(dest_root: str,
                         skill_names: Sequence[str]) -> dict[str, Any]:
    """Build install spec for Google agent harnesses (Antigravity)."""
    return {
        "version_command":
        None,
        "memory_file":
        "AGENTS.md",
        "copies": [
            {
                "source": "plugin.json",
                "destination": f"{dest_root}/plugin.json",
            },
            *_skill_copies(skill_names, "skills", f"{dest_root}/skills"),
            {
                "source": "agents",
                "destination": f"{dest_root}/agents",
            },
        ],
        "hook_copies": [
            {
                "source": "hooks.json",
                "destination": f"{dest_root}/hooks.json",
            },
        ],
        "json_merges": [],
        "toml_sets": [],
    }


def harness_install_specs(
    skill_names: Sequence[str] = (),
    package_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Return harness-specific copy rules, version command, and configuration settings.

    `skill_names` lists every skill the distribution ships; each gets one
    copy rule per harness.
    """
    specs: dict[str, dict[str, Any]] = {
        "claude-code": {
            "version_command": ["claude", "--version"],
            "memory_file":
            "CLAUDE.md",
            "copies": [
                *_skill_copies(skill_names, "project/skills",
                               ".claude/skills"),
                {
                    "source": "project/agents",
                    "destination": ".claude/agents",
                },
            ],
            "hook_copies": [{
                "source": "project/hooks.json",
                "destination": ".claude/hooks.json",
            }],
            "json_merges": [{
                "path": ".claude/settings.local.json",
                "value": {
                    "env": dict(CLAUDE_CODE_SETTINGS_ENV),
                    "approved_directories":
                    [utils.TPU_AGENT_SUITE_HOME_SPELLING],
                },
            }],
            "toml_sets": [],
        },
        "codex": {
            "version_command": ["codex", "--version"],
            "memory_file":
            "AGENTS.md",
            "copies": [
                *_skill_copies(skill_names, "skills", ".agents/skills"),
                {
                    "source": "agents",
                    "destination": ".codex/agents",
                },
            ],
            "hook_copies": [{
                "source": ".codex/hooks.json",
                "destination": ".codex/hooks.json",
            }],
            "json_merges": [],
            "toml_sets": [
                {
                    "path": ".codex/config.toml",
                    "section": "agents",
                    "key": "enabled",
                    "value": True,
                },
                {
                    "path": ".codex/config.toml",
                    "section": "agents",
                    "key": "max_concurrent_threads_per_session",
                    "value": MAX_CONCURRENT_DELEGATES,
                },
                {
                    "path": ".codex/config.toml",
                    "section": "sandbox",
                    "key": "allowed_paths",
                    "value": [utils.TPU_AGENT_SUITE_HOME_SPELLING],
                },
            ],
        },
        "cursor": {
            "version_command": ["cursor", "--version"],
            "copies": [
                *_skill_copies(skill_names, "skills", ".cursor/skills"),
                {
                    "source": f".cursor/rules/{package_name}.mdc",
                    "destination": f".cursor/rules/{package_name}.mdc",
                },
            ],
            "hook_copies": [{
                "source": ".cursor/hooks.json",
                "destination": ".cursor/hooks.json",
            }],
            "json_merges": [],
            "toml_sets": [],
        },
    }

    # `.agents`, not `.agent`. The Google harnesses accept either as a general
    # customization root, but they discover *subagents* only at
    # `.agents/agents/<name>.md` and `~/.gemini/config/agents/`. Installing
    # to `.agent/agents/` leaves every agent on disk and invisible, and the
    # orchestrator's `invoke_subagent` call fails with "not found or not
    # allowed to be invoked" -- after which running the agent's
    # instructions inline is the only path left.
    #
    # One spec for the family: those discovery paths are the family's, so a
    # per-member entry would be a copy of this one under another name.
    for google_harness in sorted(utils.GOOGLE_HARNESSES):
        specs[google_harness] = _google_install_spec(".agents", skill_names)

    return specs


def workspace_memory_merges(
    package_dir: Path,
    manifest: dict[str, Any],
    memory_file: str,
) -> list[dict[str, str]]:
    """Build managed workspace memory blocks declared by the package manifest.

  The block text is package content rather than framework text, so it is
  reviewed and versioned alongside the material it points at, and is inlined
  here so the installer never needs the package sources. Resource references
  render through the same token registry as every instruction file, at
  compile time; the installer applies blocks verbatim.
  """
    merges: list[dict[str, str]] = []
    for entry in manifest.get("workspace_memory", []):
        content_path = rooted(package_dir, entry["content"],
                              "workspace memory content")
        content = render_agent_references(
            content_path.read_text(encoding="utf-8").strip(), manifest)

        # Memory files load in every session, from arbitrary working
        # directories (a user-scope CLAUDE.md most of all), so only
        # location-independent references survive there: the workspace
        # environment slot must not appear in a block.
        env_path = f".tpu-agent-suite/{utils.ENVIRONMENT_FILE}"
        if env_path in content:
            raise TpuAgentSuiteError(
                f"workspace memory block '{entry['marker']}' references the"
                " workspace environment file; memory blocks may reference"
                " central resources only")

        merges.append({
            "path": memory_file,
            "marker": f"tpu-agent-suite:{manifest['name']}:{entry['marker']}",
            "content": content,
        })
    return merges


def build_install_plan(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    harness: str,
) -> dict[str, Any]:
    """Generate tpu-agent-suite-install.json installation plan for target harness."""
    name = manifest["name"]
    has_hooks = bool(manifest.get("hooks"))

    asset_roots: set[str] = set()
    for item in manifest.get("copy", []):
        if Path(item).parts:
            asset_roots.add(Path(item).parts[0])

    units = [
        *[s for s in manifest.get("skills", []) if isinstance(s, dict)],
        *get_manifest_agent_map(manifest).values(),
    ]
    for agent in units:
        for ref_file in agent.get("references", []):
            if Path(ref_file).parts:
                asset_roots.add(Path(ref_file).parts[0])
        for script_file in agent.get("scripts", []):
            if Path(script_file).parts:
                asset_roots.add(Path(script_file).parts[0])
        for s_key in ("accepts", "produces"):
            s_file = agent.get(s_key, "")
            if Path(s_file).parts:
                asset_roots.add(Path(s_file).parts[0])

    for hook in manifest.get("hooks", []):
        action = hook.get("action", "")
        if action.startswith("script:"):
            rel = action[len("script:"):]
            if Path(rel).parts:
                asset_roots.add(Path(rel).parts[0])

    for script in manifest.get("script_tools", []):
        script_path = script.get("path", "")
        if Path(script_path).parts:
            asset_roots.add(Path(script_path).parts[0])

    skill_names = [skill["name"] for skill in manifest["skills"]]
    spec = harness_install_specs(skill_names, name)[harness]
    # Placement is declared per entry, never inferred from the path: harness
    # files (skills, agents, hooks) are workspace-store; the payload -- the
    # embedded runtime, contract schemas, build identity, and every shipped
    # asset root -- is central-store, one immutable copy per version with
    # store-relative destinations (the version appears once, in the plan's
    # package identity). The installer roots each entry at its store and
    # rejects entries that do not declare one.
    # The installer travels with the payload, so an install can always be
    # removed by the installer of the version that wrote its ledger, without
    # the distribution it came from.
    payload_sources = {"framework", "schemas", "tpu-agent-suite-build.json"
                       } | set(asset_roots)
    if (package_dir / "requirements-tpu.txt").is_file():
        # What `compute setup verify` checks the target's software against.
        payload_sources.add("requirements-tpu.txt")
    if (package_dir / "requirements-client.txt").is_file():
        # What session start checks this machine's software against.
        payload_sources.add("requirements-client.txt")
    copies = [{
        **mapping, "store": "workspace"
    } for mapping in spec["copies"]] + [{
        "source": src,
        "destination": src,
        "store": "central",
    } for src in sorted(payload_sources)]
    if not manifest["agents"]:
        # A build whose skills delegate to nothing emits no agent files, so
        # the harness's agents mapping would point the installer at a
        # directory the distribution does not contain.
        copies = [
            c for c in copies
            if pathlib.PurePosixPath(c["source"]).name != "agents"
        ]
    json_merges = [dict(m) for m in spec["json_merges"]]
    toml_sets = [dict(t) for t in spec["toml_sets"]]

    if has_hooks:
        copies.extend({
            **mapping, "store": "workspace"
        } for mapping in spec.get("hook_copies", []))

        if harness == "claude-code":
            hooks_config = build_harness_hooks_config(name, manifest,
                                                      "claude-code")
            if hooks_config:
                claude_settings_hooks: dict[str, list[Any]] = {}
                for h_events in hooks_config.values():
                    for evt, evt_list in h_events.items():
                        if evt not in claude_settings_hooks:
                            claude_settings_hooks[evt] = []
                        claude_settings_hooks[evt].extend(evt_list)

                # Merge hooks into .claude/settings.local.json alongside env settings
                merged_into_existing = False
                for m in json_merges:
                    if m.get("path"
                             ) == ".claude/settings.local.json" and isinstance(
                                 m.get("value"), dict):
                        m["value"]["hooks"] = claude_settings_hooks
                        merged_into_existing = True
                        break
                if not merged_into_existing:
                    json_merges.append({
                        "path": ".claude/settings.local.json",
                        "value": {
                            "hooks": claude_settings_hooks
                        },
                    })

        elif harness == "codex":
            toml_sets.append({
                "path": ".codex/config.toml",
                "section": "hooks",
                "key": "enabled",
                "value": True,
            })

    return {
        "harness":
        harness,
        "version_range":
        target["versions"],
        "environment_path":
        f".tpu-agent-suite/{utils.ENVIRONMENT_FILE}",
        "ownership_path":
        f".tpu-agent-suite/{utils.OWNERSHIP_FILE}",
        "version_command":
        spec["version_command"],
        "copies":
        copies,
        "json_merges":
        json_merges,
        "markdown_merges":
        (workspace_memory_merges(package_dir, manifest, spec["memory_file"])
         if spec.get("memory_file") else []),
        "toml_sets":
        toml_sets,
    }


def build_harness_hooks_config(
    package_name: str,
    manifest: dict[str, Any],
    harness: str,
    include_internal: bool = False,
) -> dict[str, Any]:
    """Generate harness-specific hook definitions for lifecycle events and tool interception."""
    hooks_dict: dict[str, Any] = {}
    for hook in manifest.get("hooks", []):
        if hook.get("visibility") == "internal" and not include_internal:
            continue
        # Skip the skill-tools hook on harnesses that can't enforce skill tools.
        if (hook.get("action") == "builtin:skill-tools"
                and SKILL_TOOL_RESTRICTION.get(harness) != SKILL_TOOLS_HOOK):
            continue

        hook_name = hook["name"]
        event = hook["event"]
        action = hook.get("action", "")
        timeout = hook.get("timeout", DEFAULT_HOOK_TIMEOUT)

        pkg_version = manifest.get("version", "")
        pkg_tag = utils.format_package_tag(package_name, pkg_version)
        # The payload lives only in the central store, so hook commands are
        # home-anchored: correct from any cwd, expanded by the shell at run
        # time on the target machine.
        runtime_prefix = utils.TPU_AGENT_SUITE_HOME_SPELLING

        # Map event name to harness-specific lifecycle event identifier
        harness_event = to_harness_event(event, harness)

        matchers = hook.get("matchers", [])
        check_hook_matchers(hook_name, event, matchers)
        if not matchers:
            matcher_str = "*"
        elif event in ("PreToolUse", "PostToolUse"):
            tool_mappings = get_tool_mappings()
            mapped = [
                harness_mapping(
                    tool_mappings.get(m, {}).get("harnesses", {}),
                    harness).get("matcher", m) for m in matchers
            ]
            matcher_str = "|".join(mapped)
        else:  # SubagentStop, the one other event check_hook_matchers allows
            matcher_str = "|".join(matchers)

        # Always spelled out, default included: the command line is the one
        # place the policy is read from when dispatch cannot start.
        on_failure = hook.get("on_failure", DEFAULT_HOOK_ON_FAILURE)
        on_failure_args = f" --on-failure {on_failure}"

        base_cmd = (
            f"python3 {runtime_prefix}/packages/{pkg_tag}/framework/runtime.py hook-dispatch "
            f"--harness {harness} --package {pkg_tag} --hook {hook_name} "
            f"--event {event} --timeout {timeout}{on_failure_args}")

        if action.startswith("builtin:"):
            builtin_name = action[len("builtin:"):]
            action_args = f" --builtin {shlex.quote(builtin_name)}"
        elif action.startswith("script:"):
            script_rel = action[len("script:"):]
            action_args = f" --script {script_rel}"
            if hook.get("args"):
                escaped_args = " ".join(shlex.quote(a) for a in hook["args"])
                action_args = f"{action_args} -- {escaped_args}"
        elif hook.get("command"):
            raw_cmd = hook["command"]
            if hook.get("args"):
                escaped_args = " ".join(shlex.quote(a) for a in hook["args"])
                raw_cmd = f"{raw_cmd} {escaped_args}"
            action_args = f" --command {shlex.quote(raw_cmd)}"
        else:
            raise TpuAgentSuiteError(
                f"hook '{hook_name}' must specify either 'action' (script: or builtin:) or 'command'"
            )

        # The flag that makes the runtime filter by matcher itself, for
        # registrations whose native matcher cannot select the subagent.
        emulated_matcher_args = ("" if matcher_str == "*" else
                                 f" --matcher {shlex.quote(matcher_str)}")

        # Emulate matcher filtering when host engine does not support matchers
        # natively (e.g. Stop on the Google harnesses)
        if is_google_harness(harness) and harness_event == "Stop":
            dispatch_cmd = f"{base_cmd}{emulated_matcher_args}{action_args}"
        else:
            dispatch_cmd = f"{base_cmd}{action_args}"

        # The policy is applied by dispatch, which has to be running for
        # that. The shell applies it when dispatch is not.
        hook_entry = {
            "type":
            "command",
            "command":
            fail_safe_hook_command(dispatch_cmd, hook_name, on_failure, event),
            "timeout":
            timeout + HOOK_DISPATCH_HEADROOM_S,
        }

        if hook_name not in hooks_dict:
            hooks_dict[hook_name] = {}
        if harness_event not in hooks_dict[hook_name]:
            hooks_dict[hook_name][harness_event] = []

        # On the Google harnesses, 'PreInvocation', 'PostInvocation', and 'Stop' events
        # use a flat array of handler objects in hooks.json because they do not target specific tools.
        # Tool events ('PreToolUse', 'PostToolUse'), as well as other harnesses, require
        # grouping handlers under a 'matcher' object (e.g. {"matcher": "...", "hooks": [...]}).
        # Wrapping a loop-level handler in a matcher group makes it silently
        # never fire, which is why this is a shape difference and not a detail.
        if is_google_harness(harness) and harness_event in ("Stop",
                                                            "PreInvocation"):
            hooks_dict[hook_name][harness_event].append(hook_entry)
        else:
            hooks_dict[hook_name][harness_event].append({
                "matcher": matcher_str,
                "hooks": [hook_entry],
            })

        # A subagent's result can reach its caller by more than one road. On
        # Claude Code a foreground subagent ends its turn and the result is
        # its last message (native SubagentStop), but a background or
        # auto-mode subagent must call the injected `SubagentHandback` tool
        # instead, and the text it writes after that call is not delivered.
        # Registering the same hook body on that tool call keeps the package's
        # SubagentStop contract enforced whichever road the harness takes.
        # The command still says `--event SubagentStop`: the runtime recognizes
        # the tool payload and answers it in that event's native shape. It
        # also names that event with `--handback-event`, for a dispatch that
        # fails before it can read the payload: a reply in SubagentStop's
        # shape is rejected by the tool event, and the tool runs. The tool
        # matcher selects the call, so the subagent matchers move into the
        # command, where the runtime emulates them.
        handback_mapping = (subagent_handback_mapping(harness)
                            if event == "SubagentStop" else None)
        if handback_mapping:
            handback_args = (" --handback-event "
                             f"{shlex.quote(handback_mapping['event_name'])}")
            hooks_dict[hook_name].setdefault(
                handback_mapping["event_name"], []).append({
                    "matcher":
                    handback_mapping["matcher"],
                    "hooks": [{
                        **hook_entry,
                        "command":
                        fail_safe_hook_command(
                            f"{base_cmd}{handback_args}"
                            f"{emulated_matcher_args}{action_args}", hook_name,
                            on_failure, event),
                    }],
                })

    return hooks_dict


def translate_claude(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Claude Code plugin and project layouts."""
    declared_tools = get_declared_tools(package_dir, manifest)
    plugin = output / ".claude-plugin"
    write_json(
        plugin / "plugin.json",
        {
            "name": manifest["name"],
            "displayName": manifest.get("display_name", manifest["name"]),
            "version": manifest["version"],
            "description": manifest["description"],
            "author": {
                "name": manifest.get("author", "TPU agent suite framework")
            },
        },
    )

    plugin_agent_prefix = f"{manifest['name']}:"

    # Write plugin skill and standalone project skill
    write_skill_documents(output / "skills", package_dir, manifest, target,
                          plugin_agent_prefix)
    write_skill_documents(output / "project" / "skills", package_dir, manifest,
                          target)

    # Write plugin agents and project agents
    for agent in manifest["agents"]:
        resolved_tools = resolve_agent_tools(agent, declared_tools,
                                             include_internal)
        fm = claude_frontmatter(agent, resolved_tools)
        path = output / "agents" / f"{agent['name']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            fm + agent_prompt(package_dir, manifest, target, agent,
                              plugin_agent_prefix),
            encoding="utf-8",
        )

        project_agent = output / "project" / "agents" / f"{agent['name']}.md"
        project_agent.parent.mkdir(parents=True, exist_ok=True)
        project_agent.write_text(
            fm + agent_prompt(package_dir, manifest, target, agent),
            encoding="utf-8",
        )

    # Export MCP server configurations if declared
    mcp_configs: dict[str, Any] = {}
    for mcp in manifest.get("mcp_servers", []):
        if mcp.get("visibility") == "internal" and not include_internal:
            continue
        mcp_configs[mcp["name"]] = {
            "command": mcp["command"],
            "args": mcp.get("args", []),
            "env": mcp.get("env", {}),
        }
    if mcp_configs:
        write_json(output / ".claude-plugin" / "mcp.json",
                   {"mcpServers": mcp_configs})

    hooks_config = build_harness_hooks_config(manifest["name"], manifest,
                                              "claude-code", include_internal)
    if hooks_config:
        write_json(output / ".claude-plugin" / "hooks.json", hooks_config)
        write_json(output / "project" / "hooks.json", hooks_config)

    write_json(
        output / "tpu-agent-suite-install.json",
        build_install_plan(package_dir, manifest, target, "claude-code"),
    )


def translate_codex(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Codex plugin and agents layouts."""
    declared_tools = get_declared_tools(package_dir, manifest)
    hooks_config = build_harness_hooks_config(manifest["name"], manifest,
                                              "codex", include_internal)
    plugin = output / ".codex-plugin"
    plugin_manifest: dict[str, Any] = {
        "name": manifest["name"],
        "version": manifest["version"],
        "description": manifest["description"],
        "author": {
            "name": manifest.get("author", "TPU agent suite framework")
        },
        "skills": "./skills/",
        "interface": {
            "displayName": manifest.get("display_name", manifest["name"]),
            "shortDescription": manifest["description"][:80],
            "longDescription": manifest["description"],
            "developerName": manifest.get("author",
                                          "TPU agent suite framework"),
            "category": "Productivity",
            "capabilities": ["Interactive", "Write"],
            "defaultPrompt": [default_prompt(package_dir)[:128]],
        },
    }
    # Codex defaults to `hooks/hooks.json` at the plugin root, which here holds
    # the package's own hook scripts. Declaring the path keeps the generated
    # file beside the rest of the generated plugin config.
    if hooks_config:
        plugin_manifest["hooks"] = "./.codex-plugin/hooks.json"
    write_json(plugin / "plugin.json", plugin_manifest)

    write_skill_documents(output / "skills", package_dir, manifest, target)

    for agent in manifest["agents"]:
        resolved_tools = resolve_agent_tools(agent, declared_tools,
                                             include_internal)
        config = output / "agents" / f"{agent['name']}.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        prompt = toml_multiline_string(
            agent_prompt(package_dir, manifest, target, agent))
        sandbox = codex_sandbox_mode(resolved_tools)
        approval = codex_approval_policy(resolved_tools)
        config.write_text(
            f'name = "{toml_basic_string(agent["name"])}"\n'
            f'description = "{toml_basic_string(agent["description"])}"\n'
            f'sandbox_mode = "{sandbox}"\n'
            f'approval_policy = "{approval}"\n'
            f'developer_instructions = """\n{prompt}\n"""\n',
            encoding="utf-8",
        )

    if hooks_config:
        events: dict[str, list[dict[str, Any]]] = {}
        for hook_name in sorted(hooks_config):
            for event, groups in sorted(hooks_config[hook_name].items()):
                events.setdefault(event, []).extend(groups)
        codex_hooks = {
            "description": manifest["description"],
            "hooks": events,
        }
        write_json(plugin / "hooks.json", codex_hooks)
        write_json(output / ".codex" / "hooks.json", codex_hooks)

    write_json(
        output / "tpu-agent-suite-install.json",
        build_install_plan(package_dir, manifest, target, "codex"),
    )


def translate_google_harness_layout(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    harness: str,
    tool_map: dict[str, list[str]],
    include_internal: bool = False,
) -> None:
    """Translate package into the layout every Google harness shares."""
    declared_tools = get_declared_tools(package_dir, manifest)
    write_json(
        output / "plugin.json",
        {
            "name": manifest["name"],
            "version": manifest["version"],
            "description": manifest["description"],
        },
    )

    write_skill_documents(output / "skills", package_dir, manifest, target)

    for agent in manifest["agents"]:
        resolved_tools = resolve_agent_tools(agent, declared_tools,
                                             include_internal)
        fm = google_harness_frontmatter(agent,
                                        resolved_tools,
                                        tool_map,
                                        root=is_root_agent(manifest, agent))
        agent_path = output / "agents" / f"{agent['name']}.md"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(
            fm + agent_prompt(package_dir, manifest, target, agent),
            encoding="utf-8",
        )

    mcp_configs: dict[str, Any] = {}
    for mcp in manifest.get("mcp_servers", []):
        if mcp.get("visibility") == "internal" and not include_internal:
            continue
        mcp_configs[mcp["name"]] = {
            "command": mcp["command"],
            "args": mcp.get("args", []),
            "env": mcp.get("env", {}),
        }
    if mcp_configs:
        write_json(output / "mcp_config.json", {"mcpServers": mcp_configs})

    hooks_config = build_harness_hooks_config(manifest["name"], manifest,
                                              harness, include_internal)
    if hooks_config:
        write_json(output / "hooks.json", hooks_config)

    write_json(
        output / "tpu-agent-suite-install.json",
        build_install_plan(package_dir, manifest, target, harness),
    )


def google_harness_translator(
    harness: str,
    tool_map: dict[str, list[str]],
) -> Callable[[Path, Path, dict[str, Any], dict[str, Any], bool], None]:
    """Bind a Google harness and its tool map to the family translator.

    The layout a Google harness is compiled into is the family's; only the
    native tool names differ. Binding the pair here is what lets a new member
    be registered by naming it and its tool map, with no translator of its own
    to keep in step with this one.
    """

    def translate_google_harness(
        package_dir: Path,
        output: Path,
        manifest: dict[str, Any],
        target: dict[str, Any],
        include_internal: bool = False,
    ) -> None:
        translate_google_harness_layout(package_dir, output, manifest, target,
                                        harness, tool_map, include_internal)

    return translate_google_harness


DEFAULT_PROMPT_FILE = "default-prompt.md"


def default_prompt(package_dir: Path) -> str:
    """The prompt a harness offers a new user, from `default-prompt.md`.

    Build input, not shipped content: it names what the package is for in one
    line a user can run as written.
    """
    path = package_dir / DEFAULT_PROMPT_FILE
    if not path.is_file():
        return "Run this workflow."
    text = " ".join(path.read_text(encoding="utf-8").split())
    return text or "Run this workflow."


def translate_cursor(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Cursor rules, skills, and MCP configurations."""
    write_skill_documents(output / "skills", package_dir, manifest, target)

    # Generate consolidated Cursor rule (.mdc)
    rules_dir = output / ".cursor" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    rule_content = [
        "---",
        f'description: "{manifest["description"]}"',
        'globs: "*"',
        "alwaysApply: true",
        "---",
        "",
        f"# {manifest.get('display_name', manifest['name'])} Workflow Rules",
        "",
        # Cursor runs one context, so a skill that delegates needs the
        # agents it names in that context with it. A skill that does not
        # delegate is served from `.cursor/skills` like any other.
        *[
            skill_document(package_dir, manifest, target, skill)
            for skill in manifest["skills"] if skill["delegates"]
        ],
        "",
        "## Agent Personas & Execution Contracts",
        "",
    ]

    for agent in manifest["agents"]:
        rule_content.append(f"### Persona: {agent['name']}")
        rule_content.append(agent_prompt(package_dir, manifest, target, agent))
        rule_content.append("")

    (rules_dir / f"{manifest['name']}.mdc").write_text("\n".join(rule_content),
                                                       encoding="utf-8")

    mcp_configs: dict[str, Any] = {}
    for mcp in manifest.get("mcp_servers", []):
        if mcp.get("visibility") == "internal" and not include_internal:
            continue
        mcp_configs[mcp["name"]] = {
            "command": mcp["command"],
            "args": mcp.get("args", []),
            "env": mcp.get("env", {}),
        }
    if mcp_configs:
        write_json(output / ".cursor" / "mcp.json",
                   {"mcpServers": mcp_configs})

    hooks_config = build_harness_hooks_config(manifest["name"], manifest,
                                              "cursor", include_internal)
    if hooks_config:
        events: dict[str, list[dict[str, Any]]] = {}
        for hook_name in sorted(hooks_config):
            for event, groups in sorted(hooks_config[hook_name].items()):
                for group in groups:
                    for entry in group["hooks"]:
                        events.setdefault(event, []).append({
                            "type":
                            "command",
                            "command":
                            entry["command"],
                            "timeout":
                            entry["timeout"],
                            "matcher":
                            group["matcher"],
                        })
        write_json(output / ".cursor" / "hooks.json", {
            "version": 1,
            "hooks": events
        })

    write_json(
        output / "tpu-agent-suite-install.json",
        build_install_plan(package_dir, manifest, target, "cursor"),
    )


Translator = Callable[[Path, Path, dict[str, Any], dict[str, Any], bool], None]

# The native tool vocabulary of each Google harness. The family shares a
# translator; what it does not share is the names the tools go by, so this is
# the whole of what a member has to state about itself.
GOOGLE_HARNESS_TOOL_MAPS: dict[str, dict[str, list[str]]] = {
    harness: load_harness_tool_map(harness)
    for harness in utils.GOOGLE_HARNESSES
}

# Which function compiles a package for each harness. Registering a Google
# harness means adding it to `GOOGLE_HARNESS_TOOL_MAPS` above and to
# `GOOGLE_HARNESSES`; the translator, hook shape, and mappings all follow from
# the family.
HARNESS_TRANSLATORS: dict[str, Translator] = {
    "claude-code": translate_claude,
    "codex": translate_codex,
    "cursor": translate_cursor,
    **{
        harness: google_harness_translator(harness, tool_map)
        for harness, tool_map in GOOGLE_HARNESS_TOOL_MAPS.items()
    },
}


def translate(
    package_dir: Path,
    harness: str,
    output: Path,
    include_internal: bool = False,
    allow_extra_fields: bool = False,
    skills: Sequence[str] | None = None,
    manifest_name: str = DEFAULT_MANIFEST,
) -> None:
    """Translate a package into a target harness distribution."""
    assert set(GOOGLE_HARNESS_TOOL_MAPS) == utils.GOOGLE_HARNESSES, (
        "GOOGLE_HARNESS_TOOL_MAPS and utils.GOOGLE_HARNESSES must name the"
        " same harnesses")
    manifest = load_package(package_dir,
                            allow_extra_fields=allow_extra_fields,
                            skills=skills,
                            manifest=manifest_name)
    target = target_for(manifest, harness)

    if output.exists():
        marker = output / "tpu-agent-suite-build.json"
        if not marker.is_file() or not load_json(marker).get("generated"):
            raise TpuAgentSuiteError(
                f"refusing to replace non-generated output directory: {output}"
            )
        shutil.rmtree(output)

    output.mkdir(parents=True)

    HARNESS_TRANSLATORS[harness](package_dir, output, manifest, target,
                                 include_internal)

    copy_runtime_assets(package_dir, output, manifest, harness)
    emit_distribution_requirements(package_dir, output, manifest)

    contracts = {
        name: {
            "accepts": a.get("accepts", ""),
            "produces": a.get("produces", ""),
        }
        for name, a in get_manifest_agent_map(manifest).items()
    }

    write_json(
        output / "tpu-agent-suite-build.json",
        {
            "version": manifest.get("version", TPU_AGENT_SUITE_VERSION),
            "source": manifest["name"],
            "compatibility_target": target["name"],
            "harness": harness,
            "version_range": target["versions"],
            "include_internal": include_internal,
            "contracts": contracts,
            "skill_allowed_tools": hook_allowed_skill_tools(manifest, harness),
            "generated": True,
        },
    )

    bundle_standalone_distribution(output, manifest, harness)


def bundle_standalone_distribution(
    output: Path,
    manifest: dict[str, Any],
    harness: str,
) -> None:
    """Bundle the standalone installer and documentation into the distribution."""
    framework_dir = Path(__file__).resolve().parent

    # The installer ships in framework/ (bundle_framework).

    # Generate README.md
    display_name = HARNESS_DISPLAY_NAMES.get(harness, harness)
    harness_dir = HARNESS_CONFIG_DIRS.get(harness, f".{harness}")
    pkg_name = manifest.get("name", "tpu-agent-suite")
    version = manifest.get("version", TPU_AGENT_SUITE_VERSION)
    desc = manifest.get("description", "").strip()

    template_path = framework_dir / "templates" / "README.md"
    readme_content = template_path.read_text(encoding="utf-8").format(
        title_name=manifest.get("display_name", pkg_name),
        display_name=display_name,
        version=version,
        desc=desc,
        harness_dir=harness_dir,
    )
    (output / "README.md").write_text(readme_content, encoding="utf-8")


def create_archives(
    dist_dir: Path,
    output_dir: Path,
    harnesses: Sequence[str] | None = None,
) -> list[Path]:
    """Package compiled harness distributions into standalone .tar.gz archives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    target_harnesses = set(harnesses) if harnesses else set(
        HARNESS_TRANSLATORS)

    if not dist_dir.is_dir():
        raise TpuAgentSuiteError(
            f"Distribution directory does not exist: {dist_dir}")

    for harness in sorted(target_harnesses):
        harness_dir = dist_dir / harness
        if not harness_dir.is_dir():
            continue
        for pkg_dir in sorted(harness_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            build_file = pkg_dir / "tpu-agent-suite-build.json"
            if not build_file.is_file():
                continue
            build_meta = load_json(build_file)
            source_name = build_meta.get("source", pkg_dir.name)
            version = build_meta.get("version", TPU_AGENT_SUITE_VERSION)

            archive_base = f"{source_name}-{harness}-v{version}"

            # Create .tar.gz
            tar_path = output_dir / f"{archive_base}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(pkg_dir, arcname=archive_base)
            created.append(tar_path)

    if not created:
        raise TpuAgentSuiteError(
            f"No distribution packages found in '{dist_dir}' for harnesses: {', '.join(sorted(target_harnesses))}"
        )

    return created


def compose_package(
    package_dir: Path,
    output_file: Path | None = None,
    allow_extra_fields: bool = False,
    skills: Sequence[str] | None = None,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Compose and validate a modular package into a single harness-agnostic manifest."""
    manifest = load_package(package_dir,
                            allow_extra_fields=allow_extra_fields,
                            skills=skills,
                            manifest=manifest_name)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_file, manifest)
    return manifest


def run_lint(
    package_dir: Path,
    checks: Sequence[str] | None = None,
    output_format: str = "text",
    allow_extra_fields: bool = False,
    manifest_name: str = DEFAULT_MANIFEST,
) -> int:
    """Report content problems; returns 1 when any are found.

    These are advisory checks -- the ones strict enough to gate on live in
    `verify_package`, and what remains here is heuristic and can legitimately
    fire on correct content. The non-zero exit is still the honest signal: a
    linter that always succeeds is indistinguishable from one finding nothing.
    Keep this out of the *required* status checks rather than muting it.

    `--output-format github` emits workflow annotations, so findings surface
    inline on the changed lines of a pull request.

    """
    try:
        manifest = load_package(package_dir,
                                allow_extra_fields=allow_extra_fields,
                                manifest=manifest_name)
    except TpuAgentSuiteError as exc:
        print(f"Error loading package for linting: {exc}")
        return 1
    findings = lint.run_checks(package_dir, manifest, only=checks)
    print_delegation_warnings(manifest)
    for warning in undeclared_catalog_entries(package_dir, manifest):
        print(f"WARNING: {warning}", file=sys.stderr)
    for rel in lint.unreachable_files(package_dir, manifest):
        print(
            f"WARNING: {rel} is not reachable from any released skill and"
            " will not ship",
            file=sys.stderr)

    # Workflow annotations are resolved from the repository root, not from the
    # job's working-directory, so anchor paths there when a checkout is present.
    root = package_dir.resolve()
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            root = root.relative_to(parent)
            break
    else:
        root = Path(package_dir.name)

    for finding in sorted(findings, key=lambda f: (f.check, f.file, f.line)):
        if output_format == "github":
            # ::warning file=<path>,line=<n>,title=<check>::<detail>
            detail = finding.detail.replace("\n", " ")
            print(
                f"::warning file={(root / finding.file).as_posix()},"
                f"line={max(finding.line, 1)},title={finding.check}::{detail}")
        else:
            print(finding)

    if not findings:
        print("No findings!")
        return 0
    print(f"\n{len(findings)} finding(s).")
    return 1


def parse_skill_list(value: str) -> list[str]:
    """Parse a comma-separated --skills value into skill names."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--skills must name at least one"
                                         " skill")
    for name in names:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise argparse.ArgumentTypeError(
                f"--skills takes skill names, not paths: {name!r}")
    return names


def build_compiler_parser() -> argparse.ArgumentParser:
    """Construct parser for standalone compiler commands."""
    parser = argparse.ArgumentParser(
        prog="compiler",
        description=
        "Verification and translation compiler for TPU agent suite packages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compose
    p_compose = sub.add_parser(
        "compose",
        help="Print the manifest with everything discovered from its skills",
    )
    p_compose.add_argument("package",
                           type=Path,
                           help="Path to package directory")
    p_compose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (prints formatted JSON to stdout if omitted)",
    )

    # verify
    p_verify = sub.add_parser("verify",
                              help="Verify package structure and schemas")
    p_verify.add_argument("package",
                          type=Path,
                          help="Path to package directory")

    # lint
    p_lint = sub.add_parser(
        "lint",
        help="Report content problems in instructions and bundled scripts",
    )
    p_lint.add_argument("package", type=Path, help="Path to package directory")
    p_lint.add_argument(
        "--output-format",
        choices=["text", "github"],
        default="text",
        help="'github' emits ::warning annotations for pull-request review",
    )
    p_lint.add_argument(
        "--check",
        action="append",
        choices=sorted(lint.CHECKS),
        default=None,
        help="Run only this check (repeatable; all checks run by default)",
    )

    # translate
    p_translate = sub.add_parser("translate",
                                 help="Translate package to target harness")
    p_translate.add_argument("package",
                             type=Path,
                             help="Path to package directory")
    p_translate.add_argument(
        "--harness",
        choices=sorted(HARNESS_TRANSLATORS),
        required=True,
        help="Target harness",
    )
    p_translate.add_argument("--output",
                             type=Path,
                             required=True,
                             help="Output directory")
    p_translate.add_argument(
        "--include-internal",
        action="store_true",
        help=
        "Include internal scripts and MCP servers in compiled distribution",
    )

    for p in (p_compose, p_verify, p_lint, p_translate):
        p.add_argument(
            "--manifest",
            default=DEFAULT_MANIFEST,
            help="Manifest file in the package directory that names the"
            f" package to build (default: {DEFAULT_MANIFEST})",
        )
        p.add_argument(
            "--allow-extra-fields",
            action="store_true",
            help="Allow unrecognized or extra properties in package manifest",
        )

    # Not p_lint: it reports on the package as authored, which is what
    # package.release.json releases.
    for p in (p_compose, p_verify, p_translate):
        p.add_argument(
            "--skills",
            type=parse_skill_list,
            default=None,
            help="Comma-separated skill names to build in place of the"
            " `skills` package.release.json lists. Each is built whole, with"
            " everything it reaches.",
        )

    # archive
    p_archive = sub.add_parser(
        "archive",
        help="Create standalone .tar.gz archives from built distributions",
    )
    p_archive.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Root directory containing harness distributions (default: dist)",
    )
    p_archive.add_argument(
        "--output",
        type=Path,
        default=Path("dist/archives"),
        help="Output directory for archives (default: dist/archives)",
    )
    p_archive.add_argument(
        "--harness",
        choices=sorted(HARNESS_TRANSLATORS),
        action="append",
        default=None,
        help="Harnesses to archive (repeatable; default: all)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for standalone compiler operations."""
    parser = build_compiler_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "compose":
            manifest = compose_package(
                args.package,
                args.output,
                allow_extra_fields=args.allow_extra_fields,
                skills=args.skills,
                manifest_name=args.manifest,
            )
            if args.output is None:
                print(json.dumps(manifest, indent=2))

        elif args.command == "verify":
            verify_package(
                args.package,
                allow_extra_fields=args.allow_extra_fields,
                skills=args.skills,
                manifest_name=args.manifest,
            )
            print("Successfully verified!")

        elif args.command == "lint":
            return run_lint(
                args.package,
                checks=args.check,
                output_format=args.output_format,
                allow_extra_fields=args.allow_extra_fields,
                manifest_name=args.manifest,
            )

        elif args.command == "translate":
            translate(
                args.package,
                args.harness,
                args.output,
                include_internal=args.include_internal,
                allow_extra_fields=args.allow_extra_fields,
                skills=args.skills,
                manifest_name=args.manifest,
            )

        elif args.command == "archive":
            created = create_archives(
                dist_dir=args.dist_dir,
                output_dir=args.output,
                harnesses=args.harness,
            )
            for path in created:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"Created archive: {path} ({size_mb:.2f} MB)")

        return 0
    except TpuAgentSuiteError as exc:
        print(f"compiler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
