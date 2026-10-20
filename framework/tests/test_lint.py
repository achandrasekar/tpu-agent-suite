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
"""Unit tests for the package content checks in framework.lint."""

import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from framework import lint

Path = pathlib.Path


class TestLint(unittest.TestCase):
    """Each check is exercised against both a clean and a violating fixture."""

    def setUp(self):
        super().setUp()
        self.pkg = Path(tempfile.mkdtemp())
        (self.pkg / "agent_a" / "instructions").mkdir(parents=True)
        (self.pkg / "agent_a" / "references").mkdir(parents=True)
        (self.pkg / "agent_a" / "scripts").mkdir(parents=True)
        (self.pkg / "environment").mkdir(parents=True)
        (self.pkg / "schemas").mkdir(parents=True)
        self._write("agent_a/references/guide.md", "# Guide")
        # The skill is checked like any other prompt, so the fixture keeps it
        # clean: it names its one delegate with a token. A test that wants
        # findings on the skill overwrites this file.
        self._write("agent_a/instructions/entry.md",
                    "Entry point. Delegate to {{agent:agent-a}}.")
        self._write("agent_a/instructions/a.md", "Do the thing.")
        self.manifest = {
            "name":
            "fixture",
            "skills": [{
                "name": "entry",
                "description": "Entry description",
                "instructions": "agent_a/instructions/entry.md",
                "delegates": ["agent-a"],
            }],
            "agents": [{
                "name": "agent-a",
                "instructions": "agent_a/instructions/a.md",
                "delegates": [],
                "scripts": [],
                "references": ["agent_a/references/guide.md"],
            }],
        }

    def tearDown(self):
        shutil.rmtree(self.pkg, ignore_errors=True)
        super().tearDown()

    def _write(self, rel, text):
        path = self.pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _run(self, check):
        return lint.CHECKS[check](self.pkg, self.manifest)

    # --- instruction-links ------------------------------------------------

    def test_link_to_declared_reference_is_clean(self):
        self._write("agent_a/instructions/a.md",
                    "See [guide](agent_a/references/guide.md).")
        self.assertEqual(self._run("instruction-links"), [])

    def test_external_and_anchor_links_are_ignored(self):
        self._write(
            "agent_a/instructions/a.md",
            "See [docs](https://example.com) and [top](#heading).",
        )
        self.assertEqual(self._run("instruction-links"), [])

    def test_source_relative_link_is_flagged(self):
        """The classic migration bug: a path relative to the instructions file."""
        self._write("agent_a/instructions/a.md",
                    "See [guide](references/guide.md).")
        findings = self._run("instruction-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("references/guide.md", findings[0].detail)
        self.assertIn("does not resolve", findings[0].detail)

    def test_existing_but_undeclared_file_is_flagged(self):
        """Resolves in the source tree, but translate would not bundle it."""
        self._write("agent_a/references/extra.md", "# Extra")
        self._write("agent_a/instructions/a.md",
                    "See [extra](agent_a/references/extra.md).")
        findings = self._run("instruction-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("not declared", findings[0].detail)

    def test_runtime_and_placeholder_link_targets_are_ignored(self):
        """A link target resolved at run time is not a package path.

        `.tpu-agent-suite/` was already a known runtime root, but only `bare-paths`
        consulted the set, so a link to one was still reported here.
        """
        self._write(
            "agent_a/instructions/a.md",
            "See [report](<artifacts>/0/report_0.md),"
            " [state](.tpu-agent-suite/pkg/runs/<run_id>/artifacts/state.md),"
            " and [out](workspace/pkg/runs/<run_id>/artifacts/out.json).",
        )
        self.assertEqual(self._run("instruction-links"), [])

    def test_home_relative_link_targets_are_ignored(self):
        """`~/.tpu-agent-suite/...` is the install destination, spelled home-relative.

        Only the bare spelling was a known runtime root, so the `~/` form was
        reported as an unresolvable package path -- and this check gates
        `verify`, so the false positive failed the build.
        """
        self._write(
            "agent_a/instructions/a.md",
            "Source run: [report](~/.tpu-agent-suite/runs/<run_id>/artifacts/...).",
        )
        self.assertEqual(self._run("instruction-links"), [])

    def test_skill_links_are_checked_too(self):
        """A skill is a prompt like any other."""
        self._write("agent_a/instructions/entry.md",
                    "See [guide](agent_a/references/guide.md).")
        findings = self._run("instruction-links")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].agent, "entry")
        # The skill declares no scripts/references, so nothing it names is
        # bundled for it -- even a file another agent declares.
        self.assertIn("not declared", findings[0].detail)

    # --- bare-paths -------------------------------------------------------

    def test_declared_bare_path_is_clean(self):
        self._write("agent_a/instructions/a.md",
                    "Read `agent_a/references/guide.md` first.")
        self.assertEqual(self._run("bare-paths"), [])

    def test_a_backticked_token_is_not_read_as_a_bare_path(self):
        """The text after the colon of a token is not a path of its own."""
        self._write(
            "agent_a/instructions/a.md",
            "Read `{{reference:agent_a/guide.md}}`, then run"
            " `python3 {{script:agent_a/tool.py}} --fast`.")
        self.assertEqual(self._run("bare-paths"), [])

    def test_a_bare_path_beside_a_token_is_still_checked(self):
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{script:agent_a/tool.py}} agent_a/scripts/nope.py`.")
        findings = self._run("bare-paths")
        self.assertEqual(len(findings), 1)
        self.assertIn("agent_a/scripts/nope.py", findings[0].detail)

    def _add_agent_b(self, *, delegated_from_a: bool) -> None:
        """A second agent owning `agent_b/references/knobs.md`.

        Whether agent-a can reach it by delegating is the only thing separating
        the two bare-paths cases below, so it is the only knob.
        """
        self._write("agent_b/references/knobs.md", "# Knobs")
        self._write("agent_b/instructions/b.md", "Nothing here.")
        if delegated_from_a:
            self.manifest["agents"][0]["delegates"] = ["agent-b"]
        self.manifest["agents"].append({
            "name":
            "agent-b",
            "instructions":
            "agent_b/instructions/b.md",
            "delegates": [],
            "scripts": [],
            "references": ["agent_b/references/knobs.md"],
        })

    def test_bare_path_declared_by_a_delegate_is_ignored(self):
        """Describing a delegate's work names the delegate's files, not its own.

        tpu-sharding's orchestrator explains that the Optimizer reads the knob wiki;
        it never opens the file itself, so the path belongs to the delegate's
        declarations, not a gap in the orchestrator's.
        """
        self._add_agent_b(delegated_from_a=True)
        self._write(
            "agent_a/instructions/a.md",
            "Delegate to {{agent:agent-b}}. It reads"
            " `agent_b/references/knobs.md` for allowed parameters.",
        )
        self.assertEqual(self._run("bare-paths"), [])

    def test_bare_path_declared_by_an_unrelated_agent_is_flagged(self):
        """The exemption follows the delegation edge -- it is not package-wide."""
        self._add_agent_b(delegated_from_a=False)
        self._write("agent_a/instructions/a.md",
                    "Read `agent_b/references/knobs.md`.")
        findings = self._run("bare-paths")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].agent, "agent-a")

    def test_link_to_a_delegates_file_is_still_flagged(self):
        """`bare-paths` exempts a delegate's file; `instruction-links` must not.

        A link tells the reader to open the target, and translate bundles it for
        the delegate only -- so it dangles here whoever else declares it.
        """
        self._add_agent_b(delegated_from_a=True)
        self._write("agent_a/instructions/a.md",
                    "See [knobs](agent_b/references/knobs.md).")
        findings = self._run("instruction-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("not declared", findings[0].detail)

    def test_undeclared_but_existing_bare_path_is_flagged(self):
        """The README-recommended form, naming a file translate will not bundle."""
        self._write("agent_a/scripts/tool.py", "print(1)\n")
        self._write("agent_a/instructions/a.md",
                    "Run `python3 agent_a/scripts/tool.py --flag`.")
        findings = self._run("bare-paths")
        self.assertEqual(len(findings), 1)
        self.assertIn("not declared", findings[0].detail)

    def test_unresolvable_bare_path_is_flagged(self):
        self._write("agent_a/instructions/a.md",
                    "Run `skills/agent_a/scripts/gone.py`.")
        findings = self._run("bare-paths")
        self.assertEqual(len(findings), 1)
        self.assertIn("does not resolve", findings[0].detail)

    def test_runtime_paths_are_not_package_paths(self):
        """`workspace/` and `.tpu-agent-suite/` are resolved at run time, not build time."""
        self._write(
            "agent_a/instructions/a.md",
            "Write `workspace/state.json` and read"
            " `.tpu-agent-suite/pkg/knowledgebase/index.md`.",
        )
        self.assertEqual(self._run("bare-paths"), [])

    def test_bare_path_without_a_directory_is_ignored(self):
        """A bare filename is prose, not a package reference."""
        self._write("agent_a/instructions/a.md", "Edit `setup.py` as needed.")
        self.assertEqual(self._run("bare-paths"), [])

    def test_placeholder_rooted_paths_are_runtime_paths(self):
        """A path rooted at a placeholder the prompt defines cannot be resolved.

        tpu-sharding writes `<artifacts> = .tpu-agent-suite/<pkg>/runs/<run_id>/artifacts`
        and then refers to `<artifacts>/<N>/report_<N>.md` throughout.
        """
        self._write(
            "agent_a/instructions/a.md",
            "Write `<artifacts>/0/report_0.md` and `<workspace_dir>/state.md`.",
        )
        self.assertEqual(self._run("bare-paths"), [])

    def test_markdown_link_inside_a_code_span_reports_neither_half(self):
        """The label is display text; the target belongs to instruction-links."""
        self._write(
            "agent_a/instructions/a.md",
            "Log the link `[0/report_0.md](<artifacts>/0/report_0.md)` in"
            " `state.md`.",
        )
        self.assertEqual(self._run("bare-paths"), [])

    # --- reference-links --------------------------------------------------

    def test_sibling_link_between_declared_references_is_clean(self):
        self._write("agent_a/references/other.md", "# Other")
        self._write("agent_a/references/guide.md", "See [other](other.md).")
        self.manifest["agents"][0]["references"] = [
            "agent_a/references/guide.md",
            "agent_a/references/other.md",
        ]
        self.assertEqual(self._run("reference-links"), [])

    def test_link_to_an_undeclared_sibling_is_flagged(self):
        """It resolves on disk, but nothing bundles it, so it dangles at run time."""
        self._write("agent_a/references/other.md", "# Other")
        self._write("agent_a/references/guide.md", "See [other](other.md).")
        findings = self._run("reference-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("other.md", findings[0].detail)

    def test_reference_link_to_origin_tree_is_flagged(self):
        self._write("agent_a/references/guide.md",
                    "See [src](learning/agents/x.py).")
        findings = self._run("reference-links")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].agent, "agent-a")

    def test_external_reference_links_are_ignored(self):
        self._write("agent_a/references/guide.md",
                    "See [docs](https://example.com).")
        self.assertEqual(self._run("reference-links"), [])

    def test_runtime_reference_links_are_ignored(self):
        """A reference doc describes run-time artifact paths like a prompt does."""
        self._write(
            "agent_a/references/guide.md",
            "Report at [run](<artifacts>/<N>/report_<N>.md), state at"
            " [state](.tpu-agent-suite/pkg/runs/<run_id>/artifacts/state.md), and the"
            " same destination home-relative at"
            " [home](~/.tpu-agent-suite/runs/<run_id>/artifacts/...).",
        )
        self.assertEqual(self._run("reference-links"), [])

    # --- unused-declarations ----------------------------------------------

    def test_declaration_named_by_the_prompt_is_used(self):
        self._write("agent_a/instructions/a.md", "Read `guide.md`.")
        self.assertEqual(self._run("unused-declarations"), [])

    def test_declaration_named_only_by_a_sibling_is_used(self):
        """Indirect use is real: one bundled script starts another."""
        self._write("agent_a/scripts/client.py", "SERVER = 'server.py'\n")
        self._write("agent_a/scripts/server.py", "pass\n")
        self.manifest["agents"][0]["scripts"] = [
            "agent_a/scripts/client.py",
            "agent_a/scripts/server.py",
        ]
        self.manifest["agents"][0]["references"] = []
        self._write("agent_a/instructions/a.md", "Run `client.py`.")
        self.assertEqual(self._run("unused-declarations"), [])

    def test_declaration_imported_by_a_sibling_is_used(self):
        """A Python dependency is named by its module, not its filename.

        A caller naming a helper as a literal string or filename;
        tpu-sharding's `query_tpu_sharding.py` reaches `entry.py` as `from entry import
        ...`. Both are the same correct pattern -- an indirect dependency
        declared alongside its caller -- and only the second needs this match.
        """
        self._write("agent_a/scripts/query.py",
                    "from entry import Entry\nimport persistence\n")
        self._write("agent_a/scripts/entry.py", "class Entry: pass\n")
        self._write("agent_a/scripts/persistence.py", "pass\n")
        self.manifest["agents"][0]["scripts"] = [
            "agent_a/scripts/query.py",
            "agent_a/scripts/entry.py",
            "agent_a/scripts/persistence.py",
        ]
        self.manifest["agents"][0]["references"] = []
        self._write("agent_a/instructions/a.md", "Run `query.py`.")
        self.assertEqual(self._run("unused-declarations"), [])

    def test_import_match_does_not_excuse_an_unimported_module(self):
        """The import allowance is a match, not a blanket pass for `.py`."""
        self._write("agent_a/scripts/query.py", "from entry import Entry\n")
        self._write("agent_a/scripts/entry.py", "class Entry: pass\n")
        self._write("agent_a/scripts/orphan.py", "pass\n")
        self.manifest["agents"][0]["scripts"] = [
            "agent_a/scripts/query.py",
            "agent_a/scripts/entry.py",
            "agent_a/scripts/orphan.py",
        ]
        self.manifest["agents"][0]["references"] = []
        self._write("agent_a/instructions/a.md", "Run `query.py`.")
        findings = self._run("unused-declarations")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, "agent_a/scripts/orphan.py")

    def test_declaration_nothing_names_is_flagged(self):
        self._write("agent_a/instructions/a.md", "Do the thing.")
        findings = self._run("unused-declarations")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, "agent_a/references/guide.md")

    # --- delegation-consistency -------------------------------------------

    def test_delegation_consistency_clean(self):
        self.manifest["agents"][0]["delegates"] = ["agent-b"]
        self._write("agent_a/instructions/a.md", "Invoke {{agent:agent-b}}.")
        self.assertEqual(self._run("delegation-consistency"), [])

    def test_skill_delegates_are_checked_too(self):
        """A skill that names its delegate in prose declares it silently."""
        self._write("agent_a/instructions/entry.md",
                    "For everything, invoke_subagent agent-a.")
        findings = self._run("delegation-consistency")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].agent, "entry")
        self.assertIn("never references", findings[0].detail)

    def test_token_outside_the_allowed_set_is_flagged(self):
        """An agent that *can* delegate, naming a target it may not invoke."""
        self.manifest["agents"][0]["delegates"] = ["agent-b"]
        self._write("agent_a/instructions/a.md",
                    "Invoke {{agent:agent-b}}, then {{agent:agent-c}}.")
        findings = self._run("delegation-consistency")
        self.assertEqual(len(findings), 1)
        self.assertIn("agent-c", findings[0].detail)
        self.assertIn("does not declare", findings[0].detail)

    def test_token_in_a_non_delegating_agent_is_ignored(self):
        """A leaf agent naming the next phase for its caller, not invoking it.

        `translate` grants the delegation tool only when `delegates` is
        non-empty, so this agent is handed none and its contract reads
        "Allowed delegation targets: none" -- the token cannot be an
        invocation, and it renders the prefixed id the caller dispatches on.
        """
        self.assertEqual(self.manifest["agents"][0]["delegates"], [])
        self._write(
            "agent_a/instructions/a.md",
            "PHASE 1 COMPLETE. Report to the worker and request it to invoke"
            " {{agent:agent-b}}.",
        )
        self.assertEqual(self._run("delegation-consistency"), [])

    def test_declared_delegate_never_referenced_is_flagged(self):
        self.manifest["agents"][0]["delegates"] = ["agent-b"]
        findings = self._run("delegation-consistency")
        self.assertEqual(len(findings), 1)
        self.assertIn("never references", findings[0].detail)

    # --- legacy-paths -----------------------------------------------------

    def test_compute_contract_clean(self):
        self.assertEqual(self._run("compute-contract"), [])

    def test_compute_contract_flags_unknown_flag(self):
        """A flag the runtime does not define is a command the agent cannot run."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute run --code x.py --no-such-flag`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("--no-such-flag", findings[0].detail)

    def test_compute_contract_accepts_a_nested_subcommands_flag(self):
        """`setup plan --fields` is the runtime's own fix for a missing field."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute setup plan --fields staging_bucket`,"
            " not `{{runtime}} compute setup plan --no-such-flag`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("--no-such-flag", findings[0].detail)

    def test_compute_contract_flags_unknown_subcommand(self):
        self._write("agent_a/instructions/a.md",
                    "Run `{{runtime}} compute exec --code x.py`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("unknown compute subcommand", findings[0].detail)

    def test_compute_contract_flags_unknown_backend(self):
        """A backend name that is not registered fails at resolve time."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute --backend gke run --code x.py`.\n"
            "Or `{{runtime}} compute run --backend=gke --code x.py`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 2)
        for finding in findings:
            self.assertIn("unknown backend 'gke'", finding.detail)

    def test_compute_contract_accepts_registered_backend_either_side_of_verb(
            self):
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute --backend local run --code x.py` or"
            " `{{runtime}} compute run --backend gke_xpk --code x.py`.")
        self.assertEqual(self._run("compute-contract"), [])

    def test_compute_contract_allows_script_args_after_delimiter(self):
        """Everything right of `--` belongs to the script, not to compute."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute run --code x.py -- --model m --tp 4`.")
        self.assertEqual(self._run("compute-contract"), [])

    def test_compute_contract_folds_line_continuations(self):
        """A wrapped invocation is one command; its `--` may land on a later line."""
        self._write(
            "agent_a/instructions/a.md",
            "```bash\n"
            "{{runtime}} compute run --code x.py --timeout 2000 \\\n"
            "  -- payload.json --trial-timeout 120 --total-timeout 1800\n"
            "```\n",
        )
        self.assertEqual(self._run("compute-contract"), [])

    def test_compute_contract_validates_chained_commands_separately(self):
        """A second call chained with `&&`/`;`/`|` is checked against its own verb.

        Without splitting, `--code` on the second command would be read as a
        flag of the first command's `job` verb and reported as unknown.
        """
        self._write(
            "agent_a/instructions/a.md",
            "```bash\n"
            "{{runtime}} compute job status --job-id X && "
            "{{runtime}} compute run --code y.py\n"
            "{{runtime}} compute inspect --json | tee out.json; "
            "{{runtime}} compute submit --recipe r.sh\n"
            "```\n",
        )
        self.assertEqual(self._run("compute-contract"), [])

        self._write(
            "agent_a/instructions/a.md",
            "`{{runtime}} compute job status --job-id X && "
            "{{runtime}} compute run --bogus y.py`\n")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("`compute run` has no --bogus", findings[0].detail)

    def test_compute_contract_reports_when_the_runtime_cannot_be_introspected(
            self):
        """A runtime that fails to load must not silently switch the check off."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute run --code x.py --no-such-flag`.")
        saved = (lint._COMPUTE_FLAGS_CACHE, lint._COMPUTE_BACKENDS_CACHE,
                 lint._COMPUTE_INTROSPECTION_ERROR)
        lint._COMPUTE_FLAGS_CACHE = None
        lint._COMPUTE_INTROSPECTION_ERROR = None
        try:
            with mock.patch("framework.runtime.build_runtime_parser",
                            side_effect=SyntaxError("broken runtime")):
                findings = self._run("compute-contract")
        finally:
            (lint._COMPUTE_FLAGS_CACHE, lint._COMPUTE_BACKENDS_CACHE,
             lint._COMPUTE_INTROSPECTION_ERROR) = saved
        self.assertEqual(len(findings), 1)
        self.assertIn("introspection failed", findings[0].detail)
        self.assertIn("broken runtime", findings[0].detail)
        self.assertIn("validation skipped", findings[0].detail)

    def test_compute_contract_flags_raw_transports(self):
        """Hand-rolled transport is what the unified runtime exists to replace."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `scp f.py $USER@$HOST:/tmp/`, `kubectl exec pod -- ls`,"
            " `xpk workload create --cluster c`, `gcloud compute ssh vm`.")
        labels = {
            f.detail.split(":")[0]
            for f in self._run("compute-contract")
        }
        self.assertEqual(
            labels,
            {"raw ssh", "raw kubectl", "raw xpk", "raw gcloud-compute"})

    def test_compute_contract_ignores_prose_mentions_of_host_tools(self):
        """Naming a host tool is not invoking it."""
        self._write(
            "agent_a/instructions/a.md",
            "XPK / kubectl / gcloud are host tools installed separately.")
        self.assertEqual(self._run("compute-contract"), [])

    def test_compute_contract_scans_declared_references(self):
        """A declared reference reaches the agent exactly as a prompt does."""
        self._write(
            "agent_a/references/guide.md",
            "Run `{{runtime}} compute run --code x.py --no-such-flag`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, "agent_a/references/guide.md")

    def test_compute_contract_scans_declared_scripts(self):
        """A bundled tool that opens its own connection defeats the runtime.

        It does so whether or not any prompt mentions it, so scripts are
        scanned as well as prompts. This is the shape a package ported from a
        codebase with its own transport arrives in: the prompts get migrated
        and the tools they call keep shelling out.
        """
        self._write(
            "agent_a/scripts/tool.py", "import subprocess\n"
            'cmd = f"ssh {host} python3 -u -"\n'
            "subprocess.run(cmd, shell=True)\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file, "agent_a/scripts/tool.py")
        self.assertIn("raw ssh", findings[0].detail)

    def test_compute_contract_flags_legacy_tpu_client_in_a_script(self):
        """A docstring naming the dead client teaches the agent a dead command."""
        self._write(
            "agent_a/scripts/tool.py",
            '"""Submits the check through tpu_client.py."""\n',
        )
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("raw tpu-client", findings[0].detail)

    def test_compute_contract_flags_ssh_with_optional_arg_notation(self):
        """`ssh [-i key] host` is the form every reference doc used; it must match."""
        self._write(
            "agent_a/instructions/a.md",
            "```bash\nssh [-i <key_path>] <user>@<host> \"python3 -u -\"\n```\n"
        )
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("raw ssh", findings[0].detail)

    def test_compute_contract_flags_legacy_tpu_client_server(self):
        """Legacy tpu-kernel-development TPU server/client scripts are superseded by unified compute."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `python3 scripts/tpu-kernel-development/tpu_client.py --action test`.",
        )
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("raw tpu-client", findings[0].detail)

    def test_compute_contract_flags_ssh_config_file(self):
        """Relying on ~/.ssh/config bypasses the environment.json compute configuration."""
        self._write(
            "agent_a/instructions/a.md",
            "Check `~/.ssh/config` to identify available remote hosts.",
        )
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("raw ssh-config", findings[0].detail)

    def test_legacy_paths_clean(self):
        self.assertEqual(self._run("legacy-paths"), [])

    def test_legacy_paths_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "Run `bazel run //x:y`, read experimental/foo, then plan_kernel_agent.",
        )
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertEqual(labels,
                         {"bazel", "experimental-path", "agy-agent-name"})

    def test_every_legacy_finding_carries_a_remediation_hint(self):
        """The match says something is wrong; the hint says what replaces it."""
        self._write("agent_a/instructions/a.md", "Read experimental/foo.")
        findings = self._run("legacy-paths")
        self.assertEqual(len(findings), 1)
        self.assertIn(" -- ", findings[0].detail)

    def test_bazel_is_flagged_in_prose_not_just_commands(self):
        """A prose mention is the same debt."""
        self._write("agent_a/instructions/a.md",
                    "Build it locally to avoid Bazel overhead.")
        labels = [f.detail.split(":")[0] for f in self._run("legacy-paths")]
        self.assertEqual(labels, ["bazel"])

    def test_agy_invoke_is_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "Read preflight_check/SKILL.md then invoke_subagent torch-coder.",
        )
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertIn("agy-invoke", labels)

    def test_skill_layout_spellings_are_not_legacy(self):
        """`skills/<name>/SKILL.md` is the package layout, not origin rot."""
        self._write("agent_a/instructions/a.md",
                    "Read `skills/tpu-agent/SKILL.md` and follow it.")
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertNotIn("origin-skills-dir", labels)
        self.assertNotIn("agy-skill-file", labels)

    def test_harness_specific_tool_name_is_flagged(self):
        self._write("agent_a/instructions/a.md",
                    "Use the Bash tool, then call TodoWrite.")
        labels = [f.detail.split(":")[0] for f in self._run("legacy-paths")]
        # Once, not twice: harness-tool already owns TodoWrite, and
        # harness-tool-directive covers only the ask-the-user family.
        self.assertEqual(labels, ["harness-tool"])

    def test_a_file_tool_directive_is_left_to_the_portable_tool_check(self):
        """The directive check is the ask-the-user family and nothing wider.

        `list_dir`, `run_command` and the rest read as ordinary prose too often
        to be worth matching, and the tools that do matter are already named by
        harness-tool above.
        """
        self._write("agent_a/references/guide.md",
                    "Use list_dir to see what is there, then run_command it.")
        self.assertEqual(self._run("legacy-paths"), [])

    def test_portable_tool_names_are_not_flagged(self):
        """Lowercase primitives are the portable grants; they must stay clean."""
        self._write("agent_a/instructions/a.md",
                    "You may read, write, edit, search, and shell.")
        self.assertEqual(self._run("legacy-paths"), [])

    def test_agent_name_pattern_ignores_path_segments(self):
        """`torch_agent` inside a path is reported as a path, not an agent name."""
        self._write(
            "agent_a/instructions/a.md",
            "See learning/agents/tpu_suite/torch_agent/skills/preflight_check.py",
        )
        labels = [f.detail.split(":")[0] for f in self._run("legacy-paths")]
        self.assertEqual(labels, ["origin-agent-path"])

    def test_leak_patterns_are_checked_in_declared_references(self):
        """A reference doc shipped from a public repo leaks like a prompt does.

        This is the tpu-sharding `xpk.md` case: the offending command sat in a
        reference, so the instruction-only scope never saw it.
        """
        self._write("agent_a/references/guide.md",
                    "Run experimental/tools/xprof_cli.py list.")
        findings = self._run("legacy-paths")
        self.assertTrue(
            any(f.file == "agent_a/references/guide.md"
                and "experimental-path" in f.detail
                for f in findings), findings)

    def test_non_leak_patterns_stay_instruction_only(self):
        """A doc may explain what replaced Bazel without being reported for it."""
        self._write("agent_a/references/guide.md",
                    "Previously built with bazel; now use the compiler.")
        self.assertEqual([
            f for f in self._run("legacy-paths")
            if f.file == "agent_a/references/guide.md"
        ], [])

    def test_a_reference_telling_the_agent_to_use_a_harness_tool_is_flagged(
            self):
        """The benchmark_execution.md case: an instruction, in a reference.

        "use `ask_question`" works in the one harness that has the tool, and
        the instruction-only scope never saw it because it sat in a reference.
        """
        for line in ("Before running, use `ask_question` tool to check.",
                     "Call AskUserQuestion to confirm the mode.",
                     "invoke the `request_user_input` tool first."):
            with self.subTest(line=line):
                self._write("agent_a/references/guide.md", line)
                findings = [
                    f for f in self._run("legacy-paths")
                    if f.file == "agent_a/references/guide.md"
                ]
                self.assertEqual([f.detail.split(":")[0] for f in findings],
                                 ["harness-tool-directive"])

    def test_a_reference_may_name_a_harness_tool_without_directing_its_use(
            self):
        """Naming the tool is not telling the agent to use it."""
        self._write(
            "agent_a/references/guide.md",
            "`ask_question` only exists in Antigravity, so ask the user"
            " in plain prose instead.")
        self.assertEqual([
            f for f in self._run("legacy-paths")
            if f.file == "agent_a/references/guide.md"
        ], [])

    def test_a_reference_declared_twice_is_reported_once(self):
        """Several agents declaring one file still means one file in the build."""
        self.manifest["agents"].append({
            "name":
            "agent-b",
            "instructions":
            "agent_a/instructions/a.md",
            "delegates": [],
            "scripts": [],
            "references": ["agent_a/references/guide.md"],
        })
        self._write("agent_a/references/guide.md",
                    "See experimental/storage/data.")
        findings = [
            f for f in self._run("legacy-paths")
            if f.file == "agent_a/references/guide.md"
        ]
        self.assertEqual(len(findings), 1, findings)

    def test_internal_shortlinks_and_domains_are_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "See go/tpu-guide, b/1234567, cl/987654, and https://dashboard.corp.google.com.",
        )
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertIn("internal-shortlink", labels)

    def test_experimental_and_origin_agent_and_workstation_paths_are_flagged(
            self):
        self._write(
            "agent_a/instructions/a.md",
            "Read experimental/tpu_kernel_development/tool.py, learning/agents/foo/bar.py, and /usr/local/google/home/user/venv/bin/python3.",
        )
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertLessEqual(
            {"experimental-path", "origin-agent-path", "workstation-path"},
            labels)

    def test_agy_tool_names_are_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "Use view_file to read code, write_to_file to save, ask_question to prompt user, and avoid code_search MCP.",
        )
        labels = {f.detail.split(":")[0] for f in self._run("legacy-paths")}
        self.assertIn("agy-tool", labels)

    # --- environment-contract ---------------------------------------------

    def _setup_environment(self, questions, properties, closed=True):
        self._write(
            "environment/questions.json",
            json.dumps({
                "questions": [{
                    "name": q,
                    "type": "string"
                } for q in questions]
            }),
        )
        schema = {"type": "object", "properties": {p: {} for p in properties}}
        if closed:
            schema["additionalProperties"] = False
        self._write("schemas/env.json", json.dumps(schema))
        self.manifest["environment"] = {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        }

    def test_environment_contract_clean(self):
        self._setup_environment(["host"], ["host"])
        self._write(
            "agent_a/scripts/tool.py",
            'p = "environment.json"\nh = env.get("host")\n',
        )
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        self.assertEqual(self._run("environment-contract"), [])

    def test_unanswerable_environment_key_is_flagged(self):
        self._setup_environment(["host"], ["host"])
        self._write(
            "agent_a/scripts/tool.py",
            'p = "environment.json"\nz = env.get("zone", "us-central2-b")\n',
        )
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("environment-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("'zone'", findings[0].detail)
        self.assertIn("additionalProperties", findings[0].detail)

    def test_question_missing_from_schema_is_flagged(self):
        self._setup_environment(["host", "zone"], ["host"])
        findings = self._run("environment-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("'zone'", findings[0].detail)

    def test_nested_question_path_is_resolved_against_the_schema(self):
        """A question landing at `compute.params.zone` needs that nested property."""
        self._write(
            "environment/questions.json",
            json.dumps({
                "questions": [
                    {
                        "name": "backend",
                        "path": "compute.backend",
                        "type": "string"
                    },
                    {
                        "name": "zone",
                        "path": "compute.params.zone",
                        "type": "string",
                        "when": {
                            "backend": ["gce_tpu_vm"]
                        }
                    },
                    {
                        "name": "pod",
                        "path": "compute.params.pod",
                        "type": "string",
                        "when": {
                            "flavour": ["x"]
                        }
                    },
                ]
            }))
        self._write(
            "schemas/env.json",
            json.dumps({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "compute": {
                        "type": "object",
                        "properties": {
                            "backend": {
                                "type": "string"
                            },
                            "params": {
                                "type": "object",
                                "properties": {
                                    "zone": {
                                        "type": "string"
                                    }
                                },
                            },
                        },
                    }
                },
            }))
        self.manifest["environment"] = {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        }
        findings = self._run("environment-contract")
        details = "\n".join(f.detail for f in findings)
        self.assertEqual(len(findings), 2, details)
        self.assertIn("'pod' lands at 'compute.params.pod'", details)
        self.assertIn("conditional on 'flavour'", details)

    def test_scripts_that_ignore_the_environment_are_skipped(self):
        self._setup_environment(["host"], ["host"])
        self._write("agent_a/scripts/tool.py", 'z = env.get("zone")\n')
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        self.assertEqual(self._run("environment-contract"), [])

    # --- legacy-imports ---------------------------------------------------

    def test_legacy_imports_clean(self):
        self._write("agent_a/scripts/tool.py",
                    "import json\nimport tensorflow\nfrom . import sibling\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        self.assertEqual(self._run("legacy-imports"), [])

    def test_origin_rooted_import_is_flagged(self):
        self._write("agent_a/scripts/tool.py",
                    "from skills.preflight.scripts import preflight\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("legacy-imports")
        self.assertEqual(len(findings), 1)
        self.assertIn("skills.preflight.scripts", findings[0].detail)

    def test_experimental_and_learning_rooted_imports_are_flagged(self):
        self._write(
            "agent_a/scripts/tool.py",
            "import experimental.tpu_kernel_development.tools\nfrom learning.agents import agent\n",
        )
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("legacy-imports")
        self.assertEqual(len(findings), 2)
        modules = {f.detail.split("'")[1] for f in findings}
        self.assertEqual(
            modules,
            {"experimental.tpu_kernel_development.tools", "learning.agents"})

    def test_companion_test_beside_a_declared_script_is_scanned(self):
        """The bug hid in *_test.py, which no frontmatter declares."""
        self._write("agent_a/scripts/tool.py", "import json\n")
        self._write("agent_a/scripts/tool_test.py",
                    "import experimental.foo\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("legacy-imports")
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].file.endswith("tool_test.py"))

    def test_uninstalled_third_party_import_is_not_flagged(self):
        """`not installed here` must never be the signal."""
        self._write(
            "agent_a/scripts/tool.py",
            "import torch_tpu\nimport matplotlib.pyplot\nimport vertexai\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        self.assertEqual(self._run("legacy-imports"), [])

    def test_unparseable_script_is_reported(self):
        self._write("agent_a/scripts/tool.py", "def broken(:\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("legacy-imports")
        self.assertEqual(len(findings), 1)
        self.assertIn("cannot be parsed", findings[0].detail)

    # --- frontmatter-contracts --------------------------------------------

    def test_frontmatter_contracts_clean(self):
        self._write(
            "agent_a/instructions/entry.md",
            "---\n"
            "name: entry\n"
            "description: Entry description.\n"
            "delegates: ['agent-a']\n"
            "---\n"
            "Prompt.",
        )
        self._write(
            "agent_a/instructions/a.md",
            "---\n"
            "name: agent-a\n"
            "description: Agent description.\n"
            "tools:\n"
            "  - read\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "delegates: []\n"
            "---\n"
            "Prompt.",
        )
        self.assertEqual(self._run("frontmatter-contracts"), [])

    def test_frontmatter_contracts_missing_frontmatter(self):
        self._write("agent_a/instructions/a.md", "# Just Markdown")
        findings = self._run("frontmatter-contracts")
        self.assertTrue(
            any("missing leading '---' YAML frontmatter" in f.detail
                for f in findings))

    def test_frontmatter_contracts_unclosed(self):
        self._write("agent_a/instructions/a.md",
                    "---\nname: agent-a\nNo closing delimiter")
        findings = self._run("frontmatter-contracts")
        self.assertTrue(
            any("unclosed YAML frontmatter" in f.detail for f in findings))

    def test_frontmatter_contracts_missing_fields(self):
        self._write("agent_a/instructions/a.md",
                    "---\nname: agent-a\n---\nPrompt.")
        findings = self._run("frontmatter-contracts")
        missing_fields = {f.detail for f in findings}
        self.assertTrue(
            any("missing required contract field 'description'" in d
                for d in missing_fields))
        self.assertFalse(
            any("missing required contract field 'accepts'" in d
                for d in missing_fields))
        self.assertFalse(
            any("missing required contract field 'produces'" in d
                for d in missing_fields))

    def test_frontmatter_contracts_omitted_accepts_and_produces_clean(self):
        self._write(
            "agent_a/instructions/entry.md",
            "---\n"
            "name: entry\n"
            "description: Entry description.\n"
            "delegates: ['agent-a']\n"
            "---\n"
            "Prompt.",
        )
        self._write(
            "agent_a/instructions/a.md",
            "---\n"
            "name: agent-a\n"
            "description: Agent description.\n"
            "---\n"
            "Prompt.",
        )
        self.assertEqual(self._run("frontmatter-contracts"), [])

    def test_frontmatter_contracts_omitted_tools_and_delegates_clean(self):
        self._write(
            "agent_a/instructions/entry.md",
            "---\n"
            "name: entry\n"
            "description: Entry description.\n"
            "delegates: ['agent-a']\n"
            "---\n"
            "Prompt.",
        )
        self._write(
            "agent_a/instructions/a.md",
            "---\n"
            "name: agent-a\n"
            "description: Agent description.\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Prompt.",
        )
        self.assertEqual(self._run("frontmatter-contracts"), [])

    def test_frontmatter_contracts_skill_missing_description(self):
        self._write(
            "agent_a/instructions/entry.md",
            "---\nname: entry\ndelegates: ['agent-a']\n---\nPrompt.",
        )
        findings = self._run("frontmatter-contracts")
        self.assertTrue(
            any("missing required field 'description'" in f.detail
                for f in findings))

    def test_frontmatter_contracts_invalid_tools(self):
        self._write(
            "agent_a/instructions/a.md",
            "---\n"
            "name: agent-a\n"
            "description: Desc.\n"
            "tools:\n"
            "  - bash_tool\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "delegates: []\n"
            "---\n"
            "Prompt.",
        )
        findings = self._run("frontmatter-contracts")
        self.assertTrue(
            any("invalid tool grants" in f.detail for f in findings))

    def test_frontmatter_contracts_name_mismatch(self):
        self._write(
            "agent_a/instructions/a.md",
            "---\n"
            "name: wrong-name\n"
            "description: Desc.\n"
            "tools:\n"
            "  - read\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "delegates: []\n"
            "---\n"
            "Prompt.",
        )
        findings = self._run("frontmatter-contracts")
        self.assertTrue(
            any("does not match agent name" in f.detail for f in findings))

    # --- runner -----------------------------------------------------------
    def test_run_checks_honours_the_only_filter(self):
        self._write("agent_a/instructions/a.md",
                    "experimental/foo and [x](nope.md) and {{agent:ghost}}")
        only = lint.run_checks(self.pkg, self.manifest, only=["legacy-paths"])
        self.assertTrue(only)
        self.assertEqual({f.check for f in only}, {"legacy-paths"})

    def test_finding_key_is_line_insensitive(self):
        a = lint.Finding("c", "agent", "f.md", 10, "detail")
        b = lint.Finding("c", "agent", "f.md", 99, "detail")
        self.assertEqual(a.key(), b.key())


class XprofOwnershipChecks(unittest.TestCase):
    """Only tpu-diagnostics reads traces or runs the xprof CLI."""

    def setUp(self):
        super().setUp()
        self.pkg = Path(tempfile.mkdtemp())
        for rel in ("agent_a/instructions", "agent_a/scripts",
                    "tpu-diagnostics/instructions"):
            (self.pkg / rel).mkdir(parents=True)
        self._write(
            "agent_a/instructions/entry.md",
            "Delegate to {{agent:agent-a}} or {{agent:tpu-diagnostics}}.")
        self._write("agent_a/instructions/a.md", "Do the thing.")
        self._write("tpu-diagnostics/instructions/x.md", "Analyse.")
        self.manifest = {
            "name":
            "fixture",
            "skills": [{
                "name": "entry",
                "description": "Entry",
                "instructions": "agent_a/instructions/entry.md",
                "delegates": ["agent-a", "tpu-diagnostics"],
            }],
            "agents": [{
                "name": "agent-a",
                "instructions": "agent_a/instructions/a.md",
                "delegates": [],
                "scripts": [],
                "references": [],
            }, {
                "name": "tpu-diagnostics",
                "instructions": "tpu-diagnostics/instructions/x.md",
                "delegates": [],
                "scripts": [],
                "references": [],
            }],
        }

    def tearDown(self):
        shutil.rmtree(self.pkg, ignore_errors=True)
        super().tearDown()

    def _write(self, rel, text):
        path = self.pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _run(self, check):
        return lint.CHECKS[check](self.pkg, self.manifest)

    # --- xplane-access ----------------------------------------------------

    def test_script_reading_xplane_proto_is_flagged(self):
        self._write(
            "agent_a/scripts/tool.py",
            "from tensorflow.tsl.profiler.protobuf import xplane_pb2\n"
            "from xprof.convert import raw_to_tool_data\n"
            "import tensorboard_plugin_profile.convert\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("xplane-access")
        self.assertEqual([f.line for f in findings], [1, 2, 3])
        self.assertTrue(all("tpu-diagnostics" in f.detail for f in findings))

    def test_companion_test_reading_xplane_proto_is_flagged(self):
        self._write("agent_a/scripts/tool.py", "import json\n")
        self._write("agent_a/scripts/tool_test.py",
                    "from tsl.profiler.protobuf import xplane_pb2\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        findings = self._run("xplane-access")
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].file.endswith("tool_test.py"))

    def test_ordinary_profiling_imports_are_clean(self):
        """Capturing a trace is fine; only reading one directly is not."""
        self._write("agent_a/scripts/tool.py",
                    "import jax.profiler\nimport xprof\nimport tensorflow\n")
        self.manifest["agents"][0]["scripts"] = ["agent_a/scripts/tool.py"]
        self.assertEqual(self._run("xplane-access"), [])

    # --- xprof-outside-analyst -------------------------------------------

    def test_cli_invocation_outside_tpu_diagnostics_package_is_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "Run `xprof get_overview \"${LOGDIR}\"` and then\n"
            "xprof get_top_hlo_ops LOGDIR --limit=30\n")
        findings = self._run("xprof-outside-analyst")
        self.assertEqual([f.line for f in findings], [1, 2])
        self.assertIn("delegates", findings[0].detail)

    def test_cli_invocation_inside_tpu_diagnostics_package_is_allowed(self):
        self._write("tpu-diagnostics/instructions/x.md",
                    "Run `xprof get_overview LOGDIR`.")
        self.assertEqual(self._run("xprof-outside-analyst"), [])

    def test_the_analyst_is_known_by_name_wherever_its_prompt_sits(self):
        """The exemption followed a directory, which the layout by kind
        removed: the analyst's prompt is `agents/tpu-diagnostics.md` now."""
        self._write("agents/tpu-diagnostics.md",
                    "Run `xprof get_overview LOGDIR`.")
        analyst = next(a for a in self.manifest["agents"]
                       if a["name"] == "tpu-diagnostics")
        analyst["instructions"] = "agents/tpu-diagnostics.md"
        self.assertEqual(self._run("xprof-outside-analyst"), [])

    def test_a_prompt_beside_the_analyst_is_not_exempt(self):
        """A shared directory is not ownership."""
        self._write("agents/agent-a.md", "Run `xprof get_overview LOGDIR`.")
        self.manifest["agents"][0]["instructions"] = "agents/agent-a.md"
        findings = self._run("xprof-outside-analyst")
        self.assertEqual([f.agent for f in findings], ["agent-a"])

    def test_mentioning_xprof_without_a_subcommand_is_clean(self):
        self._write(
            "agent_a/instructions/a.md",
            "Delegate XProf analysis to {{agent:tpu-diagnostics}}; it pins"
            " xprof and installs the xprof skill.")
        self.assertEqual(self._run("xprof-outside-analyst"), [])

    # --- generic-compute-prose -------------------------------------------

    def test_generic_run_vs_submit_is_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "When execution on accelerator compute is required, execute via "
            "`{{runtime}} compute`: use `{{runtime}} compute run` for synchronous "
            "execution or `{{runtime}} compute submit` for asynchronous batch jobs.\n"
        )
        findings = self._run("generic-compute-prose")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line, 1)
        self.assertIn("generic run vs submit explanation", findings[0].detail)

    def test_generic_headers_and_backend_list_are_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "1. **Unified Compute Access**: Execute accelerator operations via "
            "`{{runtime}} compute`. It automatically utilizes configuration in "
            "`environment.json` (supports `local`, `gce_tpu_vm`, `gke_pod`).\n"
            "2. **Universal Compute Operations**:\n"
            "- **Job Lifecycle Management**:\n"
            "- Status: `{{runtime}} compute job status --job-id <job_id>`\n")
        findings = self._run("generic-compute-prose")
        lines = [f.line for f in findings]
        self.assertEqual(lines, [1, 2, 3, 4])
        self.assertTrue(
            all(f.check == "generic-compute-prose" for f in findings))

    def test_specific_compute_commands_are_clean(self):
        self._write(
            "agent_a/instructions/a.md",
            "Run mock validation on TPU synchronously:\n"
            "```bash\n"
            "{{runtime}} compute run --code scripts/tpu-kernel-development/run_autotune.py --timeout 180\n"
            "```\n"
            "Dispatch long-running recipe:\n"
            "```bash\n"
            "JOB_ID=$({{runtime}} compute submit --recipe <artifacts>/recipe.sh)\n"
            "```\n")
        self.assertEqual(self._run("generic-compute-prose"), [])


if __name__ == "__main__":
    unittest.main()
