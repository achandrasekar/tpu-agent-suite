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
"""The layout by kind: skills/, agents/, references/, scripts/.

A manifest names only its skills. Everything else -- agents, references,
scripts, schemas, dependencies -- is found by following frontmatter from
them. These tests build one small package in that layout and check what the
compiler derives from it.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import compiler, lint, utils  # noqa: E402

Path = pathlib.Path


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_package(root: Path) -> Path:
    """A package with two skills, a delegating reference, and a bundle dir."""
    pkg = root / "layout-pkg"
    _write(
        pkg, "package.release.json",
        json.dumps({
            "name":
            "layout-pkg",
            "version":
            "1.0.0",
            "description":
            "Layout test package",
            "skills": ["main", "aux"],
            "copy": ["schemas", "environment"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [{
                "name": "claude",
                "harness": "claude-code",
                "versions": ">=2.0.0",
            }],
        }))
    _write(pkg, "environment/questions.json", json.dumps({"questions": []}))
    _write(pkg, "schemas/env.json", json.dumps({"type": "object"}))
    _write(pkg, "schemas/in.json",
           json.dumps({
               "type": "object",
               "required": ["run_id"]
           }))
    _write(
        pkg, "schemas/out.json",
        json.dumps({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "failed"]
                }
            },
        }))
    _write(pkg, "requirements-client.txt", "pyyaml>=6.0\n")
    _write(
        pkg, "skills/main/SKILL.md", "---\n"
        "name: main\n"
        "description: The primary skill.\n"
        "tools:\n  - read\n"
        "delegates:\n  - worker\n  - aux\n"
        "references:\n  - references/guide.md\n  - references/bundle\n"
        "scripts:\n  - scripts/tool.py\n"
        "---\n"
        "Delegate to {{agent:worker}} or hand off to {{skill:aux}}.\n"
        "Run `{{script:tool.py}}`; see `{{reference:guide.md}}`.\n")
    _write(
        pkg, "skills/aux/SKILL.md", "---\n"
        "name: aux\n"
        "description: A leaf skill that drives a runtime command.\n"
        "---\n"
        "Run {{runtime}} something.\n")
    _write(
        pkg, "agents/worker.md", "---\n"
        "name: worker\n"
        "description: Does the work.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "accepts: schemas/in.json\n"
        "produces: schemas/out.json\n"
        "references:\n  - references/loop.md\n"
        "---\n"
        "Read `{{reference:loop.md}}` and follow it.\n")
    _write(
        pkg, "agents/helper.md", "---\n"
        "name: helper\n"
        "description: A helper the loop reference dispatches.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "---\n"
        "Help.\n")
    _write(
        pkg, "agents/orphan.md", "---\n"
        "name: orphan\n"
        "description: Nothing reaches this one.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "---\n"
        "Unreleased.\n")
    _write(
        pkg, "references/loop.md", "---\n"
        "delegates:\n  - helper\n"
        "scripts:\n  - scripts/helper.py\n"
        "host_dependencies:\n  - pyyaml\n"
        "---\n"
        "Loop: dispatch {{agent:helper}}, then run"
        " `{{script:helper.py}}`.\n")
    _write(
        pkg, "references/guide.md", "---\n"
        "scripts:\n  - scripts/tool.py\n"
        "references:\n  - references/bundle\n"
        "---\n"
        "Tool lives at {{script:tool.py}}; the bundle at"
        " {{reference:bundle/a.md}}.\n"
        "Running on {{harness}}.\n")
    _write(pkg, "references/bundle/a.md", "# A\n")
    _write(pkg, "references/bundle/data.jsonl", '{"k": 1}\n')
    _write(pkg, "scripts/tool.py", "print('tool')\n")
    _write(pkg, "scripts/helper.py", "print('helper')\n")
    return pkg


class LayoutDiscoveryTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pkg = build_package(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, **kwargs):
        return compiler.load_package(self.pkg, **kwargs)

    def test_agents_are_found_by_following_delegates(self):
        manifest = self._load()
        names = {a["name"] for a in manifest["agents"]}
        self.assertIn("worker", names)
        # Named only by the loop reference's frontmatter.
        self.assertIn("helper", names)
        # Nothing reaches it, so it is not built.
        self.assertNotIn("orphan", names)

    def test_a_reference_adds_what_it_declares_to_its_reader(self):
        manifest = self._load()
        worker = next(a for a in manifest["agents"] if a["name"] == "worker")
        self.assertEqual(worker["own_delegates"], [])
        self.assertEqual(worker["delegates"], ["helper"])
        self.assertIn("scripts/helper.py", worker["scripts"])
        self.assertEqual(worker["declared_scripts"], [])
        self.assertIn("pyyaml", manifest["host_dependencies"])

    def test_a_directory_reference_brings_every_file_under_it(self):
        manifest = self._load()
        main = manifest["skills"][0]
        self.assertEqual(main["name"], "main")
        self.assertIn("references/bundle/a.md", main["references"])
        self.assertIn("references/bundle/data.jsonl", main["references"])
        self.assertEqual(main["declared_references"],
                         ["references/guide.md", "references/bundle"])

    def test_schemas_are_registered_from_the_directory(self):
        manifest = self._load()
        self.assertEqual(
            manifest["schemas"],
            ["schemas/env.json", "schemas/in.json", "schemas/out.json"])

    def test_a_skill_that_delegates_gets_a_twin(self):
        manifest = self._load()
        self.assertNotIn("entrypoint", manifest)
        twin = next(a for a in manifest["agents"] if a["name"] == "main")
        self.assertEqual(twin["instructions"], "skills/main/SKILL.md")
        self.assertEqual(twin["delegates"], ["worker", "aux"])
        # A skill without delegates is only a skill.
        self.assertNotIn("aux", {a["name"] for a in manifest["agents"]})

    def test_a_delegate_nothing_declares_fails_the_load(self):
        _write(self.pkg, "references/loop.md",
               "---\ndelegates:\n  - nonesuch\n---\nLoop.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._load()
        self.assertIn("delegates to unknown agents", str(ctx.exception))

    def test_a_token_naming_a_missing_file_fails_the_load(self):
        _write(self.pkg, "references/guide.md", "See {{script:missing.py}}.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._load()
        self.assertIn("scripts/missing.py", str(ctx.exception))

    def test_a_released_skill_without_a_document_fails_the_load(self):
        manifest = json.loads((self.pkg / "package.release.json").read_text())
        manifest["skills"].append("ghost")
        (self.pkg / "package.release.json").write_text(json.dumps(manifest))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._load()
        self.assertIn("skills/ghost/SKILL.md", str(ctx.exception))

    def test_skills_override_builds_the_named_skill_whole(self):
        """`--skills` replaces the list; it never trims what a skill reaches."""
        whole = self._load()
        only_main = self._load(skills=["main"])
        main = only_main["skills"][0]
        self.assertEqual(main["delegates"], whole["skills"][0]["delegates"])
        # `main` delegates to the skill `aux`, so `aux` is built with it.
        self.assertEqual([s["name"] for s in only_main["skills"]],
                         ["main", "aux"])
        self.assertEqual({a["name"]
                          for a in only_main["agents"]},
                         {a["name"]
                          for a in whole["agents"]})

    def test_a_skill_that_delegates_to_a_skill_is_warned_about(self):
        """Allowed, and the guide expects it to be rare."""
        warnings = compiler.delegation_warnings(self._load())
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("skill 'main' delegates to skill 'aux'", warnings[0])

    def test_a_skill_built_only_through_delegation_is_warned_about(self):
        warnings = compiler.delegation_warnings(self._load(skills=["main"]))
        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("skill 'main' delegates to skill 'aux'", warnings[0])
        self.assertIn("skill 'aux' is not one of the skills this build names",
                      warnings[1])
        self.assertIn("only because main delegates to it", warnings[1])

    def test_a_skill_that_delegates_to_agents_only_is_not_warned_about(self):
        skill = self.pkg / "skills" / "main" / "SKILL.md"
        skill.write_text(skill.read_text().replace(
            "delegates:\n  - worker\n  - aux\n",
            "delegates:\n  - worker\n").replace(
                " or hand off to {{skill:aux}}", ""))
        self.assertEqual(compiler.delegation_warnings(self._load()), [])

    def test_skills_override_may_build_a_skill_that_delegates_to_nothing(self):
        manifest = self._load(skills=["aux"])
        self.assertEqual([s["name"] for s in manifest["skills"]], ["aux"])
        self.assertEqual(manifest["agents"], [])
        self.assertEqual(manifest["host_dependencies"], [])

    def test_skills_override_may_name_a_skill_the_manifest_does_not(self):
        unreleased = self.pkg / "skills" / "extra"
        unreleased.mkdir()
        (unreleased / "SKILL.md").write_text(
            "---\nname: extra\ndescription: Not released.\n---\nAlone.\n")
        manifest = self._load(skills=["extra"])
        self.assertEqual([s["name"] for s in manifest["skills"]], ["extra"])

    def test_skills_override_rejects_a_name_with_no_document(self):
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._load(skills=["nonesuch"])
        self.assertIn("skills/nonesuch/SKILL.md", str(ctx.exception))

    def test_unreachable_files_are_reported_not_built(self):
        manifest = self._load()
        self.assertEqual(lint.unreachable_files(self.pkg, manifest),
                         ["agents/orphan.md"])

    def test_lint_is_clean_on_the_layout(self):
        manifest = self._load()
        findings = lint.run_checks(self.pkg,
                                   manifest,
                                   only=[
                                       "bare-paths", "delegation-consistency",
                                       "unused-declarations",
                                       "instruction-links",
                                       "frontmatter-contracts"
                                   ])
        self.assertEqual([str(f) for f in findings], [])


class LayoutTranslationTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pkg = build_package(self.tmp)
        self.dist = self.tmp / "dist"
        compiler.translate(self.pkg, "claude-code", self.dist)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_references_and_scripts_ship_at_their_package_paths(self):
        for rel in ("references/guide.md", "references/loop.md",
                    "references/bundle/a.md", "references/bundle/data.jsonl",
                    "scripts/tool.py", "scripts/helper.py"):
            self.assertTrue((self.dist / rel).is_file(), rel)

    def test_a_reference_ships_without_its_frontmatter(self):
        shipped = (self.dist / "references/loop.md").read_text()
        self.assertFalse(shipped.startswith("---"))
        self.assertIn("Loop: dispatch", shipped)

    def test_tokens_render_to_installed_paths(self):
        shipped = (self.dist / "references/guide.md").read_text()
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/scripts/tool.py",
            shipped)
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/references/bundle/a.md",
            shipped)

    def test_a_token_must_name_what_its_own_document_ships(self):
        """A file another unit declares does not count: a build that holds
        this document need not hold that one."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() + "\nAlso {{script:tool.py}}.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("agent worker names script 'scripts/tool.py'",
                      str(ctx.exception))
        self.assertIn("does not declare it", str(ctx.exception))

    def test_a_token_in_a_reference_must_name_what_the_reference_ships(self):
        guide = self.pkg / "references" / "guide.md"
        guide.write_text(guide.read_text().replace(
            "scripts:\n  - scripts/tool.py\n", ""))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("reference references/guide.md names script",
                      str(ctx.exception))

    def test_a_token_may_name_a_file_under_a_declared_directory(self):
        guide = self.pkg / "references" / "guide.md"
        self.assertIn("{{reference:bundle/a.md}}", guide.read_text())
        self.assertIn("references:\n  - references/bundle\n",
                      guide.read_text())
        compiler.load_package(self.pkg)

    def test_a_plain_path_in_a_prompt_fails_the_load(self):
        """A path typed out is read from the workspace, where it is not."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nThen run `python3 scripts/helper.py`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("agent worker names package files by path", message)
        self.assertIn("scripts/helper.py -> {{script:helper.py}}", message)

    def test_the_error_says_how_to_write_a_workspace_path(self):
        """A bare `scripts/train.py` may mean the user's file, not the
        package's. The rule cannot tell, so it says how to write that."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRun `python3 scripts/train.py`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("scripts/train.py -> {{script:train.py}}", message)
        self.assertIn("./scripts/train.py or <work>/scripts/train.py", message)

    def test_a_workspace_path_written_as_one_is_left_alone(self):
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRun `python3 ./scripts/train.py` on"
                          " `./data/dataset.csv`.\n")
        compiler.load_package(self.pkg)

    def test_a_plain_path_in_a_reference_fails_the_load(self):
        guide = self.pkg / "references" / "guide.md"
        guide.write_text(guide.read_text() + "See references/bundle/a.md.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("reference references/guide.md names package files",
                      message)
        self.assertIn("-> {{reference:bundle/a.md}}", message)

    def test_a_path_built_on_the_package_token_fails_the_load(self):
        """It renders to the right place and is checked against nothing."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRun {{package}}/scripts/helper.py.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("-> {{script:helper.py}}", str(ctx.exception))

    def test_a_path_under_the_package_token_that_no_token_names_fails(self):
        """`{{package}}/fam/scripts/x.py` is how a stale layout survives."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRun {{package}}/fam/scripts/helper.py.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("fam/scripts/helper.py -> no token names it", message)

    def test_the_package_token_alone_is_allowed(self):
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nNever write into `{{package}}`.\n")
        compiler.load_package(self.pkg)

    def test_a_skill_that_delegates_has_its_tools_checked_as_a_skill(self):
        skill = self.pkg / "skills" / "main" / "SKILL.md"
        skill.write_text(skill.read_text().replace(
            "tools:\n  - read\n", "tools:\n  - read\n  - bogus\n", 1))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("skill 'main' references undeclared tool 'bogus'",
                      str(ctx.exception))

    def test_a_skill_that_does_not_delegate_has_its_tools_checked_too(self):
        skill = self.pkg / "skills" / "aux" / "SKILL.md"
        skill.write_text(skill.read_text().replace(
            "description:", "tools:\n  - bash\ndescription:", 1))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("skill 'aux' references undeclared tool 'bash'", message)
        self.assertIn("did you mean canonical tool 'shell'?", message)

    def test_a_skill_may_hold_ask_and_so_may_the_agent_built_from_it(self):
        """A skill is what the user talks to. The agent built from a skill
        that delegates is that skill's own document."""
        skill = self.pkg / "skills" / "main" / "SKILL.md"
        skill.write_text(skill.read_text().replace(
            "tools:\n  - read\n", "tools:\n  - read\n  - ask\n", 1))
        manifest = compiler.load_package(self.pkg)
        twin = {a["name"]: a for a in manifest["agents"]}["main"]
        self.assertIn("ask", twin["tools"])

    def test_an_agent_may_not_hold_ask(self):
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text().replace(
            "tools:\n  - read\n", "tools:\n  - read\n  - ask\n", 1))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("only a skill may hold", str(ctx.exception))

    def test_a_script_declared_outside_scripts_fails_the_load(self):
        """A file declared elsewhere ships where no token can name it."""
        _write(self.pkg, "elsewhere/x.py", "print('x')\n")
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text().replace(
            "---\n", "---\nscripts:\n  - elsewhere/x.py\n", 1))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("'^scripts/'", str(ctx.exception))

    def test_a_reference_declared_outside_references_fails_the_load(self):
        _write(self.pkg, "schemas/extra.json", "{}\n")
        skill = self.pkg / "skills" / "aux" / "SKILL.md"
        skill.write_text(skill.read_text().replace(
            "---\n", "---\nreferences:\n  - schemas/extra.json\n", 1))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("'^references/'", str(ctx.exception))

    def test_a_reference_may_not_declare_a_script_elsewhere_either(self):
        _write(self.pkg, "elsewhere/y.py", "print('y')\n")
        guide = self.pkg / "references" / "guide.md"
        guide.write_text(guide.read_text().replace(
            "scripts:\n  - scripts/tool.py\n",
            "scripts:\n  - scripts/tool.py\n  - elsewhere/y.py\n"))
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("'^scripts/'", str(ctx.exception))

    def test_the_retired_knowledgebase_token_fails_the_load(self):
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRead `{{knowledgebase}}/wiki/concepts/x.md`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("retired {{knowledgebase}} token", message)
        self.assertIn("{{knowledgebase}}/wiki/concepts/x.md", message)

    def test_a_misspelt_token_kind_fails_the_load(self):
        """`{{scripts:x}}` matched no pattern, so it shipped as written."""
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nRun `{{scripts:helper.py}}`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("{{scripts:helper.py}}", message)
        self.assertIn("did you mean {{script:...}}?", message)

    def test_a_misspelt_token_kind_in_a_reference_fails_the_load(self):
        guide = self.pkg / "references" / "guide.md"
        guide.write_text(guide.read_text() + "\nSee {{refrence:bundle}}.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("did you mean {{reference:...}}?", str(ctx.exception))

    def _add_memory_block(self, text):
        _write(self.pkg, "memory/note.md", text)
        manifest_path = self.pkg / "package.release.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["workspace_memory"] = [{
            "marker": "note",
            "content": "memory/note.md"
        }]
        manifest_path.write_text(json.dumps(manifest))

    def test_a_memory_block_may_name_what_the_build_ships(self):
        self._add_memory_block("Ask {{agent:worker}}; run"
                               " `{{script:helper.py}}`.\n")
        compiler.load_package(self.pkg)

    def test_a_plain_path_in_a_memory_block_fails_the_load(self):
        self._add_memory_block("Run `scripts/helper.py`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        message = str(ctx.exception)
        self.assertIn("workspace memory block 'note'", message)
        self.assertIn("{{script:helper.py}}", message)

    def test_a_memory_block_token_must_name_what_the_build_ships(self):
        _write(self.pkg, "scripts/unshipped.py", "print('no')\n")
        self._add_memory_block("Run `{{script:unshipped.py}}`.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("workspace memory block 'note'", str(ctx.exception))

    def test_a_memory_block_may_not_name_an_agent_outside_the_build(self):
        self._add_memory_block("Ask {{agent:orphan}}.\n")
        with self.assertRaises(utils.TpuAgentSuiteError):
            compiler.load_package(self.pkg)

    def test_a_path_that_is_not_a_package_file_is_left_alone(self):
        worker = self.pkg / "agents" / "worker.md"
        worker.write_text(worker.read_text() +
                          "\nWrite `<work>/scripts/out.py`, read"
                          " `workspace/references/notes.md`, and keep"
                          " `~/.cache/scripts/x.py`.\n")
        compiler.load_package(self.pkg)

    def test_frontmatter_is_not_prose(self):
        """A declaration is a path by design; only the body takes tokens."""
        self.assertIn("scripts:\n  - scripts/tool.py\n",
                      (self.pkg / "skills/main/SKILL.md").read_text())
        compiler.load_package(self.pkg)

    def test_data_and_schema_tokens_name_what_the_package_copies(self):
        _write(self.pkg, "data/store/rows.jsonl", "{}\n")
        worker = self.pkg / "agents" / "worker.md"
        original = worker.read_text()
        worker.write_text(original + "\nRows: {{data:store/rows.jsonl}};"
                          " shape: {{schema:in.json}}.\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(self.pkg)
        self.assertIn("package.release.json does not list it under `copy`",
                      str(ctx.exception))
        manifest = json.loads((self.pkg / "package.release.json").read_text())
        manifest["copy"].append("data/store")
        (self.pkg / "package.release.json").write_text(json.dumps(manifest))
        compiler.load_package(self.pkg)
        out = self.pkg.parent / "dist-data"
        compiler.translate(self.pkg, "claude-code", out)
        built = (out / "project/agents/worker.md").read_text()
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/data/store/rows.jsonl",
            built)
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/schemas/in.json",
            built)

    def test_a_shipped_reference_resolves_harness_tokens(self):
        """A reference is read as a prompt is, so it takes the same tokens."""
        shipped = (self.dist / "references/guide.md").read_text()
        self.assertIn("Running on claude-code.", shipped)
        self.assertNotIn("{{", shipped)

    def test_every_skill_renders_and_only_delegating_ones_get_a_contract(self):
        main = (self.dist / "project/skills/main/SKILL.md").read_text()
        aux = (self.dist / "project/skills/aux/SKILL.md").read_text()
        self.assertIn("Installed package contract", main)
        self.assertIn("Delegate to worker or hand off to aux.", main)
        self.assertNotIn("Installed package contract", aux)
        self.assertIn('description: "A leaf skill', aux)

    def test_the_twin_agent_is_emitted_beside_the_others(self):
        agents = {p.stem for p in (self.dist / "project/agents").glob("*.md")}
        self.assertEqual(agents, {"main", "worker", "helper"})

    def test_the_install_plan_ships_the_new_roots_centrally(self):
        plan = utils.load_json(self.dist / "tpu-agent-suite-install.json")
        central = {
            c["source"]
            for c in plan["copies"] if c["store"] == "central"
        }
        self.assertLessEqual({"references", "scripts", "schemas"}, central)
        skills = {
            c["destination"]
            for c in plan["copies"] if c["store"] == "workspace"
        }
        self.assertIn(".claude/skills/main", skills)
        self.assertIn(".claude/skills/aux", skills)


if __name__ == "__main__":
    unittest.main()
