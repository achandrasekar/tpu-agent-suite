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
"""Unit tests for compiler module (verification and translation)."""

import contextlib
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
import unittest
from unittest import mock

from framework import compiler, utils

Path = pathlib.Path


class TestCompiler(unittest.TestCase):
    """Unit tests for package verification, delegation graph, and translation."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _create_valid_package(self) -> Path:
        pkg_dir = self.temp_dir / "valid-pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts" / "custom.py").write_text("print('hello')",
                                                       encoding="utf-8")

        manifest = {
            "name":
            "valid-pkg",
            "version":
            "1.0.0",
            "description":
            "Valid test package",
            "native_tools": [
                {
                    "name": "read",
                    "visibility": "external",
                    "action_level": {
                        "effect": "read",
                        "scope": "workspace"
                    },
                },
                {
                    "name": "write",
                    "visibility": "external",
                    "action_level": {
                        "effect": "write",
                        "scope": "workspace"
                    },
                },
            ],
            "script_tools": [{
                "name": "custom_runner",
                "path": "scripts/custom.py",
                "visibility": "internal",
                "action_level": {
                    "effect": "execute",
                    "scope": "host"
                },
                "description": "Custom runner",
            }],
            "mcp_servers": [{
                "name": "query_db",
                "command": "python3",
                "args": ["-m", "db_server"],
                "visibility": "external",
                "action_level": {
                    "effect": "read",
                    "scope": "workspace"
                },
                "description": "DB server",
            }],
            "skills": ["entry"],
            "copy": ["schemas", "environment", "scripts"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [
                {
                    "name": "claude",
                    "harness": "claude-code",
                    "versions": ">=2.0.0",
                },
                {
                    "name": "codex",
                    "harness": "codex",
                    "versions": ">=0.1.0",
                },
                {
                    "name": "antigravity",
                    "harness": "antigravity",
                    "versions": ">=1.0.0",
                },
                {
                    "name": "cursor",
                    "harness": "cursor",
                    "versions": ">=0.40.0",
                },
            ],
        }
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "Call {{agent:worker}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "worker.md").write_text(
            "---\n"
            "name: worker\n"
            "description: Worker\n"
            "tools:\n"
            "  - read\n"
            "  - write\n"
            "  - custom_runner\n"
            "  - query_db\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Work.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "schemas" / "in.json").write_text(json.dumps(
            {"type": "object"}),
                                                     encoding="utf-8")
        (pkg_dir / "schemas" / "out.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "schemas" / "env.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "tpu_target": {
                        "type": "string"
                    }
                },
                "required": ["tpu_target"],
            }),
            encoding="utf-8",
        )
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({
                "questions": [{
                    "name": "tpu_target",
                    "prompt": "TPU target",
                    "type": "string",
                    "default": "local",
                }]
            }),
            encoding="utf-8",
        )
        return pkg_dir

    def _delegate_to(self, pkg_dir: Path, *names: str) -> None:
        """Point the entry skill at `names`, which is how an agent is reached."""
        delegates = "".join(f"  - {name}\n" for name in names)
        routes = "\n".join(f"Call {{{{agent:{name}}}}}." for name in names)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n" +
            (f"delegates:\n{delegates}" if names else "delegates: []\n") +
            "---\n" + (routes or "Standalone."),
            encoding="utf-8")

    def _declare_dependencies(self, pkg_dir: Path, host=(), tpu=()) -> None:
        """Declare dependencies where they are read from: a unit's frontmatter."""
        path = pkg_dir / "agents" / "worker.md"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^(host|tpu)_dependencies:\n(?:  - .*\n)*", "",
                      text)
        extra = ""
        if host:
            extra += "host_dependencies:\n" + "".join(f"  - {d}\n"
                                                      for d in host)
        if tpu:
            extra += "tpu_dependencies:\n" + "".join(f"  - {d}\n" for d in tpu)
        path.write_text(text.replace("delegates: []\n",
                                     "delegates: []\n" + extra, 1),
                        encoding="utf-8")

    def test_verify_valid_package(self):
        pkg_dir = self._create_valid_package()
        compiler.verify_package(pkg_dir)

    def test_second_manifest_builds_its_own_package_from_the_same_files(self):
        """--manifest picks another manifest file in the same directory."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["name"] = "valid-pkg-experimental"
        (pkg_dir / "package.experimental.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        compiler.verify_package(pkg_dir,
                                manifest_name="package.experimental.json")
        self.assertEqual(
            compiler.compose_package(pkg_dir)["name"], "valid-pkg")
        self.assertEqual(
            compiler.compose_package(
                pkg_dir, manifest_name="package.experimental.json")["name"],
            "valid-pkg-experimental")

        out = self.temp_dir / "dist_experimental"
        ret = compiler.main([
            "translate",
            str(pkg_dir), "--manifest", "package.experimental.json",
            "--harness", "claude-code", "--output",
            str(out)
        ])
        self.assertEqual(ret, 0)
        build = json.loads(
            (out / "tpu-agent-suite-build.json").read_text(encoding="utf-8"))
        self.assertEqual(build["source"], "valid-pkg-experimental")

    def _write_extending_manifest(self, pkg_dir: Path, **fields) -> None:
        (pkg_dir / "skills" / "extra").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "extra" / "SKILL.md").write_text(
            "---\nname: extra\ndescription: Extra description\n---\nExtra.",
            encoding="utf-8")
        (pkg_dir / "package.experimental.json").write_text(json.dumps({
            "extends":
            "package.release.json",
            "name":
            "valid-pkg-experimental",
            **fields
        }),
                                                           encoding="utf-8")

    def test_extending_manifest_inherits_the_base_and_adds_skills(self):
        """`extends` takes every base field; `add_skills` appends to skills."""
        pkg_dir = self._create_valid_package()
        self._write_extending_manifest(pkg_dir, add_skills=["extra"])
        base = compiler.load_package(pkg_dir)
        variant = compiler.load_package(pkg_dir,
                                        manifest="package.experimental.json")

        self.assertEqual(variant["name"], "valid-pkg-experimental")
        self.assertEqual(variant["version"], base["version"])
        self.assertEqual([skill["name"] for skill in variant["skills"]],
                         ["entry", "extra"])
        compiler.verify_package(pkg_dir,
                                manifest_name="package.experimental.json")

    def test_extending_manifest_errors(self):
        cases = {
            "repeats skills already listed": {
                "add_skills": ["entry"]
            },
            "must be a list of skill names": {
                "add_skills": "extra"
            },
            "must name another manifest file": {
                "extends": "package.experimental.json"
            },
        }
        for message, fields in cases.items():
            with self.subTest(message):
                pkg_dir = self._create_valid_package()
                self._write_extending_manifest(pkg_dir, **fields)
                with self.assertRaisesRegex(compiler.TpuAgentSuiteError,
                                            message):
                    compiler.load_package(pkg_dir,
                                          manifest="package.experimental.json")

    def test_add_skills_without_extends_is_rejected(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "package.release.json").write_text(json.dumps({
            **manifest, "add_skills": ["entry"]
        }),
                                                      encoding="utf-8")
        with self.assertRaisesRegex(compiler.TpuAgentSuiteError,
                                    "`add_skills` needs `extends`"):
            compiler.load_package(pkg_dir)

    def test_extends_is_one_level(self):
        pkg_dir = self._create_valid_package()
        self._write_extending_manifest(pkg_dir)
        (pkg_dir / "package.third.json").write_text(json.dumps({
            "extends": "package.experimental.json",
            "name": "third"
        }),
                                                    encoding="utf-8")
        with self.assertRaisesRegex(compiler.TpuAgentSuiteError,
                                    "only one level is supported"):
            compiler.load_package(pkg_dir, manifest="package.third.json")

    def test_missing_manifest_file_is_rejected(self):
        pkg_dir = self._create_valid_package()
        with self.assertRaises(utils.TpuAgentSuiteError):
            compiler.load_package(pkg_dir, manifest="package.missing.json")

    def test_unknown_agent_reference_is_rejected(self):
        """Line-dropping is scoped to a subset: a dangling ref still fails."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "- Call {{agent:worker}}.\n- Call {{agent:ghost}}.\n",
            encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("unknown agent: ghost", str(ctx.exception))

    def test_strict_manifest_rejects_extra_fields(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))

        # Extra root property is strictly rejected
        manifest["unexpected_root_field"] = "value"
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn(
            "unexpected_root_field: additional property is not allowed",
            str(ctx.exception))

        # Extra agent property is strictly rejected
        del manifest["unexpected_root_field"]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        worker_md = (pkg_dir / "agents" /
                     "worker.md").read_text(encoding="utf-8")
        worker_md = worker_md.replace(
            "description: Worker\n",
            "description: Worker\nunexpected_agent_field: 123\n")
        (pkg_dir / "agents" / "worker.md").write_text(worker_md,
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn(
            "unexpected_agent_field: additional property is not allowed",
            str(ctx.exception))

    def test_tool_name_uniqueness(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        # Introduce duplicate tool name between native and script
        manifest["script_tools"].append({
            "name": "read",
            "path": "scripts/custom.py",
            "visibility": "external",
            "action_level": {
                "effect": "read",
                "scope": "session"
            },
        })
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("duplicate tool name", str(ctx.exception).lower())

    def test_invalid_action_level(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["script_tools"][0]["action_level"] = {
            "effect": "destroy_all",
            "scope": "host",
        }
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("action_level/effect", str(ctx.exception).lower())

    def test_invalid_visibility(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["script_tools"][0]["visibility"] = "secret"
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("visibility", str(ctx.exception).lower())

    def test_undeclared_agent_tool(self):
        pkg_dir = self._create_valid_package()
        worker_md = (pkg_dir / "agents" /
                     "worker.md").read_text(encoding="utf-8")
        worker_md = worker_md.replace(
            "  - query_db\n", "  - query_db\n  - undeclared_custom_tool\n")
        (pkg_dir / "agents" / "worker.md").write_text(worker_md,
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("undeclared tool", str(ctx.exception).lower())

    def test_non_canonical_tool_alias_suggestion(self):
        pkg_dir = self._create_valid_package()
        worker_md = (pkg_dir / "agents" /
                     "worker.md").read_text(encoding="utf-8")
        worker_md = worker_md.replace("  - read\n", "  - bash\n")
        (pkg_dir / "agents" / "worker.md").write_text(worker_md,
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("did you mean canonical tool 'shell'?",
                      str(ctx.exception))

    def test_tool_missing_action_level_rejected(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        # Remove action_level from a native tool
        del manifest["native_tools"][0]["action_level"]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("action_level", str(ctx.exception).lower())

    def test_cycle_detection(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents" / "worker2.md").write_text(
            "---\n"
            "name: worker2\n"
            "description: Worker 2\n"
            "tools:\n"
            "  - read\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "Work 2.",
            encoding="utf-8")
        worker_md = (pkg_dir / "agents" /
                     "worker.md").read_text(encoding="utf-8")
        worker_md = worker_md.replace("delegates: []",
                                      "delegates:\n  - worker2")
        (pkg_dir / "agents" / "worker.md").write_text(worker_md,
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("cycle", str(ctx.exception).lower())

    def _chain_manifest(self, pkg_dir, length):
        """Extend the package into a skill -> a1 -> ... -> a<length> chain."""
        names = [f"chain{i}" for i in range(1, length + 1)]
        for i, name in enumerate(names):
            next_agent = [names[i + 1]] if i + 1 < len(names) else []
            delegates_yaml = ("delegates:\n" + "\n".join(
                f"  - {d}"
                for d in next_agent)) if next_agent else "delegates: []"
            (pkg_dir / "agents" / f"{name}.md").write_text(
                f"---\n"
                f"name: {name}\n"
                f"description: Chain link {name}\n"
                f"tools:\n"
                f"  - read\n"
                f"{delegates_yaml}\n"
                f"accepts: schemas/in.json\n"
                f"produces: schemas/out.json\n"
                f"---\n"
                f"Chain work {name}.",
                encoding="utf-8")

        # The skill fixture references {{agent:worker}}, which this chain
        # replaces; repoint it at the head of the chain.
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            f"---\n"
            f"name: entry\n"
            f"description: Entry description\n"
            f"delegates:\n"
            f"  - {names[0]}\n"
            f"---\n"
            f"Entry. Delegate to {{{{agent:{names[0]}}}}}.",
            encoding="utf-8")
        return pkg_dir

    def test_delegation_depth_at_the_limit_is_allowed(self):
        """Levels of work below the skill, which is level 0 and is not counted."""
        pkg_dir = self._chain_manifest(self._create_valid_package(),
                                       compiler.MAX_DELEGATION_DEPTH)
        compiler.verify_package(pkg_dir)

    def test_delegation_past_the_limit_is_rejected(self):
        pkg_dir = self._chain_manifest(self._create_valid_package(),
                                       compiler.MAX_DELEGATION_DEPTH + 1)
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("levels below skill entry", str(ctx.exception))

    def test_build_depth_matches_the_harness_nesting_limit(self):
        """A package that builds here and exceeds the harness limit there fails
        at the point of delegation, long after anyone would look for it."""
        specs = compiler.harness_install_specs(["entry"], "pkg")
        merges = specs["claude-code"]["json_merges"]
        env = merges[0]["value"]["env"]
        self.assertEqual(env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"],
                         str(compiler.MAX_DELEGATION_DEPTH))

    def test_frontmatter_name_mismatch_rejection(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents" / "worker.md").write_text(
            "---\nname: invalid\ndescription: Invalid worker\n---\nWork.",
            encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn(
            "frontmatter name 'invalid' does not match expected agent name "
            "'worker'", str(ctx.exception))

    def test_translate_claude_with_and_without_internal(self):
        pkg_dir = self._create_valid_package()

        # 1. Translate without internal tools (default)
        out_default = self.temp_dir / "dist_claude_default"
        compiler.translate(pkg_dir,
                           "claude-code",
                           out_default,
                           include_internal=False)
        agent_md = (out_default / "project" / "agents" /
                    "worker.md").read_text(encoding="utf-8")
        # 'custom_runner' is internal so it should not appear in tools header
        self.assertNotIn("custom_runner", agent_md.split("---")[1])
        self.assertIn("query_db", agent_md.split("---")[1])

        # 2. Translate with internal tools
        out_internal = self.temp_dir / "dist_claude_internal"
        compiler.translate(pkg_dir,
                           "claude-code",
                           out_internal,
                           include_internal=True)
        agent_md_int = (out_internal / "project" / "agents" /
                        "worker.md").read_text(encoding="utf-8")
        self.assertIn("custom_runner", agent_md_int.split("---")[1])

    def test_translate_codex_sandbox_mode(self):
        pkg_dir = self._create_valid_package()

        # worker has host:execute tool 'custom_runner' with approval='on-demand'
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["script_tools"][0]["action_level"]["approval"] = "on-demand"
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        out = self.temp_dir / "dist_codex"
        compiler.translate(pkg_dir, "codex", out, include_internal=True)
        agent_toml = (out / "agents" /
                      "worker.toml").read_text(encoding="utf-8")
        self.assertIn('sandbox_mode = "danger-unrestricted"', agent_toml)
        self.assertIn('approval_policy = "on-demand"', agent_toml)

        # When internal tool is omitted, worker only has workspace tools
        out_no_int = self.temp_dir / "dist_codex_no_int"
        compiler.translate(pkg_dir,
                           "codex",
                           out_no_int,
                           include_internal=False)
        agent_toml_no_int = (out_no_int / "agents" /
                             "worker.toml").read_text(encoding="utf-8")
        self.assertIn('sandbox_mode = "workspace-write"', agent_toml_no_int)

    def test_manifest_property_level_reference_rejected(self):
        """A property naming a file instead of holding its value is rejected."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))

        scripts = manifest.pop("script_tools")
        (pkg_dir / "tools").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "tools" / "scripts.json").write_text(json.dumps(scripts),
                                                        encoding="utf-8")

        # Point property directly to JSON file
        manifest["script_tools"] = "tools/scripts.json"
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("expected array, got str", str(ctx.exception))

    def test_verify_package_allows_root_requirements(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text("torch>=2.4.0\n",
                                                      encoding="utf-8")
        compiler.verify_package(pkg_dir)

    def test_verify_package_rejects_unpinned_dependencies(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic\n",
                                                         encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("Unpinned package in 'requirements-client.txt'",
                      str(ctx.exception))

        (pkg_dir / "requirements-client.txt").write_text(
            "pydantic; python_version >= '3.10'\n", encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("Unpinned package in 'requirements-client.txt'",
                      str(ctx.exception))

        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text("torch\n",
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("Unpinned package in 'requirements-tpu.txt'",
                      str(ctx.exception))

    def test_verify_package_rejects_scattered_requirements(self):
        pkg_dir = self._create_valid_package()
        sub_dir = pkg_dir / "subagent" / "references"
        sub_dir.mkdir(parents=True)
        (sub_dir / "requirements.txt").write_text("torch\n", encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("Scattered requirements file found", str(ctx.exception))
        self.assertIn("subagent/references/requirements.txt",
                      str(ctx.exception))

    def test_verify_package_rejects_missing_requirements_file_when_deps_declared(
            self):
        pkg_dir = self._create_valid_package()
        self._declare_dependencies(pkg_dir, host=["pydantic"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn(
            "host_dependencies but 'requirements-client.txt' is missing",
            str(ctx.exception))

        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("tpu_dependencies but 'requirements-tpu.txt' is missing",
                      str(ctx.exception))

    def test_verify_package_rejects_undeclared_catalog_dependency(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["unknown-lib"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("Undeclared host dependency: 'unknown-lib'",
                      str(ctx.exception))

    def test_translate_emits_filtered_requirements(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text(
            "pydantic>=2.0.0\nextra-host>=1.0.0\n", encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text(
            "torch>=2.4.0\nextra-tpu>=1.0.0\n", encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])

        output = self.temp_dir / "dist-reqs"
        compiler.translate(pkg_dir, "claude-code", output)

        host_reqs = (output /
                     "requirements-client.txt").read_text(encoding="utf-8")
        tpu_reqs = (output /
                    "requirements-tpu.txt").read_text(encoding="utf-8")
        self.assertIn("pydantic>=2.0.0", host_reqs)
        self.assertNotIn("extra-host", host_reqs)
        self.assertIn("torch>=2.4.0", tpu_reqs)
        self.assertNotIn("extra-tpu", tpu_reqs)

    def test_a_catalog_entry_nothing_declares_is_warned_about(self):
        """It never ships: a package nothing uses, or a declaration somebody
        forgot."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text(
            "pydantic>=2.0.0\nextra-host>=1.0.0\n", encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text(
            "torch>=2.4.0\nextra-tpu>=1.0.0\n", encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])
        manifest = compiler.load_package(pkg_dir)

        warnings = compiler.undeclared_catalog_entries(pkg_dir, manifest)

        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("requirements-client.txt lists extra-host", warnings[0])
        self.assertIn("requirements-tpu.txt lists extra-tpu", warnings[1])

    def test_translate_preserves_pip_option_flags(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text(
            "# Host deps\n"
            "--extra-index-url https://download.pytorch.org/whl/cpu\n"
            "-f https://storage.googleapis.com/jax-releases/jax_releases.html\n"
            "pydantic>=2.0.0\n"
            "unused-dep>=1.0.0\n",
            encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"])

        compiler.verify_package(pkg_dir)

        output = self.temp_dir / "dist-pip-flags"
        compiler.translate(pkg_dir, "claude-code", output)

        host_reqs = (output /
                     "requirements-client.txt").read_text(encoding="utf-8")
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu",
                      host_reqs)
        self.assertIn(
            "-f https://storage.googleapis.com/jax-releases/jax_releases.html",
            host_reqs)
        self.assertIn("pydantic>=2.0.0", host_reqs)
        self.assertNotIn("unused-dep", host_reqs)

    def test_compose_package_subcommand(self):
        pkg_dir = self._create_valid_package()

        # 1. Test compose_package returning manifest dict
        composed = compiler.compose_package(pkg_dir)
        self.assertEqual(composed["name"], "valid-pkg")
        self.assertEqual(len(composed["script_tools"]), 1)
        self.assertEqual(composed["script_tools"][0]["name"], "custom_runner")

        # 2. Test compose_package writing output file
        out_file = self.temp_dir / "composed" / "package.release.json"
        compiler.compose_package(pkg_dir, output_file=out_file)
        self.assertTrue(out_file.is_file())
        written = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(written["name"], "valid-pkg")
        self.assertEqual(len(written["script_tools"]), 1)

        # 3. Test CLI compose invocation
        ret = compiler.main(
            ["compose", str(pkg_dir), "--output",
             str(out_file)])
        self.assertEqual(ret, 0)

    def test_translate_antigravity(self):
        pkg_dir = self._create_valid_package()
        out = self.temp_dir / "dist_antigravity"
        compiler.translate(pkg_dir, "antigravity", out, include_internal=True)

        self.assertTrue((out / "plugin.json").is_file())
        self.assertTrue((out / "skills" / "entry" / "SKILL.md").is_file())
        self.assertTrue((out / "agents" / "worker.md").is_file())
        self.assertTrue((out / "mcp_config.json").is_file())

        worker_md = (out / "agents" / "worker.md").read_text(encoding="utf-8")
        self.assertIn("subagent: true", worker_md)
        self.assertIn("view_file", worker_md)
        self.assertIn("write_to_file", worker_md)

    def test_translate_cursor(self):
        pkg_dir = self._create_valid_package()
        out = self.temp_dir / "dist_cursor"
        compiler.translate(pkg_dir, "cursor", out, include_internal=True)

        self.assertTrue((out / "skills" / "entry" / "SKILL.md").is_file())
        rule_file = out / ".cursor" / "rules" / "valid-pkg.mdc"
        self.assertTrue(rule_file.is_file())
        self.assertTrue((out / ".cursor" / "mcp.json").is_file())

        rule_content = rule_file.read_text(encoding="utf-8")
        self.assertIn("alwaysApply: true", rule_content)
        self.assertIn('globs: "*"', rule_content)
        self.assertIn("### Persona: worker", rule_content)

    def test_skill_contract_names_the_delegation_tool(self):
        """The contract must say *how* to delegate, not only to whom.

        Naming targets alone leaves the mechanism to be guessed, and the guess
        that costs the most is reading the agent's definition and running it
        inline: the orchestrator's context then absorbs everything the agent
        boundary existed to keep out.
        """
        pkg_dir = self._create_valid_package()
        for harness, tool in (("antigravity", "invoke_subagent"),
                              ("claude-code", "Agent")):
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_mech_{harness}"
                compiler.translate(pkg_dir,
                                   harness,
                                   out,
                                   include_internal=True)
                skill = (out / "skills" / "entry" /
                         "SKILL.md").read_text(encoding="utf-8")
                self.assertIn(f"`{tool}`", skill)
                self.assertIn("is not delegation", skill)

    def test_effort_reaches_claude_code_agents_only(self):
        """Claude Code reads it from an agent's own definition; the other
        harnesses have no confirmed equivalent and get nothing."""
        pkg_dir = self._create_valid_package()
        worker_md = pkg_dir / "agents" / "worker.md"
        worker_md.write_text(worker_md.read_text(encoding="utf-8").replace(
            "description: Worker\n", "description: Worker\neffort: low\n", 1),
                             encoding="utf-8")
        for harness in ("claude-code", "antigravity", "codex"):
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_effort_{harness}"
                compiler.translate(pkg_dir,
                                   harness,
                                   out,
                                   include_internal=True)
                rendered = "".join(
                    f.read_text(encoding="utf-8", errors="ignore")
                    for f in out.rglob("*") if f.is_file())
                if harness == "claude-code":
                    self.assertIn("\neffort: low\nbackground:", rendered)
                else:
                    self.assertNotIn("effort: low", rendered)

    def test_a_model_pin_is_rejected(self):
        """A pinned model outlives its deprecation in every old install, so an
        agent may not name one: it runs on the session's model."""
        pkg_dir = self._create_valid_package()
        worker_md = pkg_dir / "agents" / "worker.md"
        worker_md.write_text(worker_md.read_text(encoding="utf-8").replace(
            "description: Worker\n", "description: Worker\nmodel: m\n", 1),
                             encoding="utf-8")
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("$/model", str(ctx.exception))

    def test_an_effort_the_harness_does_not_have_is_rejected(self):
        pkg_dir = self._create_valid_package()
        worker_md = pkg_dir / "agents" / "worker.md"
        worker_md.write_text(worker_md.read_text(encoding="utf-8").replace(
            "description: Worker\n", "description: Worker\neffort: huge\n", 1),
                             encoding="utf-8")
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("$/effort", str(ctx.exception))

    def test_leaf_agent_contract_omits_the_mechanism(self):
        """An agent that delegates to nothing gets no mechanism line, and no
        blank line standing in for one."""
        pkg_dir = self._create_valid_package()
        out = self.temp_dir / "dist_leaf"
        compiler.translate(pkg_dir, "antigravity", out, include_internal=True)

        worker_md = (out / "agents" / "worker.md").read_text(encoding="utf-8")
        self.assertIn("- Allowed delegation targets: none.", worker_md)
        self.assertNotIn("Delegate by calling", worker_md)
        self.assertNotIn("\n\n- Allowed delegation targets", worker_md)

    def test_ask_user_question_token_renders_per_harness(self):
        """Two harnesses have the tool under different names; two have none.

        The ones without get the plain instruction rather than a build error,
        so one authored sentence ships to all four.
        """
        pkg_dir = self._create_valid_package()
        worker_md = pkg_dir / "agents" / "worker.md"
        worker_md.write_text(
            worker_md.read_text(encoding="utf-8") +
            "\nTo choose a mode, {{built-in:ask_user_question}} first.\n",
            encoding="utf-8")
        for harness, expected in (("claude-code", "AskUserQuestion"),
                                  ("antigravity", "ask_question"),
                                  ("codex", "ask the user"), ("cursor",
                                                              "ask the user")):
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_ask_{harness}"
                compiler.translate(pkg_dir,
                                   harness,
                                   out,
                                   include_internal=True)
                # Searched across the whole distribution: each harness files
                # an agent somewhere different, and the token, not the layout,
                # is what this is about.
                rendered = "".join(
                    f.read_text(encoding="utf-8", errors="ignore")
                    for f in out.rglob("*") if f.is_file())
                self.assertIn(f"To choose a mode, {expected} first.", rendered)
                self.assertNotIn("{{built-in:ask_user_question}}", rendered)

    def test_only_a_skill_may_hold_ask(self):
        """An agent granted `ask` would call a tool nobody answers.

        Its caller is blocked on the result, so the question reaches no one.
        Caught at build time because at run time it fails silently.
        """
        pkg_dir = self._create_valid_package()
        # The fixture declares its own native tools rather than the shipped
        # catalog, so `ask` has to exist before an agent can name it.
        manifest_path = pkg_dir / "package.release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["native_tools"].append({
            "name": "ask",
            "visibility": "external",
            "action_level": {
                "effect": "read",
                "scope": "session"
            },
        })
        manifest_path.write_text(json.dumps(manifest, indent=2),
                                 encoding="utf-8")
        worker_md = pkg_dir / "agents" / "worker.md"
        text = worker_md.read_text(encoding="utf-8")
        worker_md.write_text(text.replace("tools:\n", "tools:\n  - ask\n", 1),
                             encoding="utf-8")
        with self.assertRaises(compiler.TpuAgentSuiteError) as caught:
            compiler.load_package(pkg_dir)
        message = str(caught.exception)
        self.assertIn("only a skill may hold", message)
        self.assertIn("needs_input", message)

    def test_an_unregistered_built_in_token_is_rejected(self):
        """The guard used to match [a-z_] only, so a prefixed typo slipped by."""
        pkg_dir = self._create_valid_package()
        worker_md = pkg_dir / "agents" / "worker.md"
        worker_md.write_text(worker_md.read_text(encoding="utf-8") +
                             "\nNow {{built-in:ask_the_user}} about it.\n",
                             encoding="utf-8")
        with self.assertRaises(compiler.TpuAgentSuiteError) as caught:
            compiler.load_package(pkg_dir)
        self.assertIn("unknown resource tokens", str(caught.exception))

    def test_shell_grants_task_management_on_google_harnesses(self):
        """`run_command` backgrounds slow work and returns a task id, so an
        agent granted shell without `manage_task` can start a profile capture
        or a long XProf query and never collect its result."""
        pkg_dir = self._create_valid_package()
        manifest_path = pkg_dir / "package.release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["native_tools"].append({
            "name": "shell",
            "visibility": "external",
            "action_level": {
                "effect": "execute",
                "scope": "host"
            },
        })
        worker_md = (pkg_dir / "agents" /
                     "worker.md").read_text(encoding="utf-8")
        worker_md = worker_md.replace(
            "tools:\n  - read\n  - write\n  - custom_runner\n  - query_db\n",
            "tools:\n  - shell\n")
        (pkg_dir / "agents" / "worker.md").write_text(worker_md,
                                                      encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        for harness in ("antigravity", ):
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_shell_{harness}"
                compiler.translate(pkg_dir,
                                   harness,
                                   out,
                                   include_internal=True)
                worker_md = (out / "agents" /
                             "worker.md").read_text(encoding="utf-8")
                self.assertIn("run_command", worker_md)
                self.assertIn("manage_task", worker_md)

    def test_loop_level_hooks_emit_without_a_matcher_wrapper(self):
        """Stop names no tool, so the harness reads a flat handler list there.
        Wrapping it in a matcher group makes the handler silently never fire,
        which looks exactly like having no hook at all."""
        pkg_dir = self._create_valid_package()
        manifest_path = pkg_dir / "package.release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["hooks"] = [
            {
                "name": "on-stop",
                "event": "Stop",
                "command": "true",
                "visibility": "external",
            },
            {
                "name": "on-tool",
                "event": "PreToolUse",
                "command": "true",
                "matchers": ["shell"],
                "visibility": "external",
            },
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        config = compiler.build_harness_hooks_config("valid-pkg", manifest,
                                                     "antigravity")
        stop_handler = config["on-stop"]["Stop"][0]
        self.assertNotIn("matcher", stop_handler)
        self.assertEqual(stop_handler["type"], "command")

        tool_handler = config["on-tool"]["PreToolUse"][0]
        self.assertIn("matcher", tool_handler)
        self.assertIn("hooks", tool_handler)

    def test_matchers_on_an_event_with_nothing_to_match_are_rejected(self):
        """Only a tool or a subagent can be matched. Matchers on any other
        event used to leave the compiler reusing the previous hook's matcher,
        and the runtime then skipped every fire of the hook: it never ran,
        and nothing said so."""
        pkg_dir = self._create_valid_package()
        manifest_path = pkg_dir / "package.release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for event in ("Stop", "SessionEnd", "SessionStart"):
            manifest["hooks"] = [
                {
                    "name": "on-subagent",
                    "event": "SubagentStop",
                    "matchers": ["worker"],
                    "command": "true",
                },
                {
                    "name": "on-loop",
                    "event": event,
                    "matchers": ["worker"],
                    "command": "true",
                },
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(event=event, stage="verify"):
                with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                            "'on-loop' declares matchers"):
                    compiler.verify_package(pkg_dir)
            with self.subTest(event=event, stage="build"):
                with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                            "'on-loop' declares matchers"):
                    compiler.build_harness_hooks_config(
                        "valid-pkg", manifest, "antigravity")

    def test_contract_environment_path_matches_what_install_writes(self):
        """The path the contract tells agents to read must be the path install
        creates.

        These drifted: the contract said `.tpu-agent-suite/environment.json` while
        install wrote `.tpu-agent-suite/<package>/environment.json`. The file agents
        were told to consult never existed, so the settings it carries -- which
        machine the work runs on, which interpreter serves which role -- were
        invisible, and an agent on a TPU host concluded it had no accelerator.

        Nothing about that failure points at a path mismatch, which is why it
        is worth pinning here rather than hoping the next reader notices.
        """
        pkg_dir = self._create_valid_package()
        for harness in ("antigravity", "claude-code", "codex"):
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_envpath_{harness}"
                compiler.translate(pkg_dir,
                                   harness,
                                   out,
                                   include_internal=True)
                plan = json.loads(
                    (out / "tpu-agent-suite-install.json").read_text(
                        encoding="utf-8"))
                skill_dir = ("skills"
                             if harness != "claude-code" else "project/skills")
                skill = (out / skill_dir / "entry" /
                         "SKILL.md").read_text(encoding="utf-8")
                self.assertIn(f"`{plan['environment_path']}`", skill)

    def _restrict_entry_skill(self, pkg_dir: Path, tools: list[str]) -> None:
        """Give the entry skill a `tools` list, keeping its delegation."""
        listed = "".join(f"  - {name}\n" for name in tools)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - worker\n"
            f"tools:\n{listed}"
            "---\n"
            "Call {{agent:worker}}.",
            encoding="utf-8")

    @staticmethod
    def _skill_frontmatter(text: str) -> str:
        return text.split("---\n", 2)[1]

    def test_claude_skill_leaves_its_tool_limit_to_the_hook(self):
        """A tool-limited skill carries no disallowed-tools on Claude Code.

        Claude Code removes a skill's disallowed-tools from the pool for the
        rest of the turn, subagents included, so a router that loads its skill
        would start every specialist without Bash. The skill-tools hook limits
        the main thread alone.
        """
        pkg_dir = self._create_valid_package()
        self._restrict_entry_skill(pkg_dir, ["read"])
        out = self.temp_dir / "dist_skill_tools"
        compiler.translate(pkg_dir, "claude-code", out, include_internal=True)
        for skill_dir in ("skills", "project/skills"):
            with self.subTest(skill_dir=skill_dir):
                fm = self._skill_frontmatter(
                    (out / skill_dir / "entry" /
                     "SKILL.md").read_text(encoding="utf-8"))
                self.assertNotIn("disallowed-tools", fm)
                self.assertNotIn("allowed-tools", fm)
        build_info = json.loads(
            (out / "tpu-agent-suite-build.json").read_text(encoding="utf-8"))
        self.assertNotIn("shell", build_info["skill_allowed_tools"]["entry"])

    def test_skill_without_tools_is_unrestricted(self):
        """A skill with no tools field remains unrestricted."""
        pkg_dir = self._create_valid_package()
        out = self.temp_dir / "dist_skill_no_tools"
        compiler.translate(pkg_dir, "claude-code", out, include_internal=True)
        text = (out / "skills" / "entry" /
                "SKILL.md").read_text(encoding="utf-8")
        self.assertNotIn("disallowed-tools", self._skill_frontmatter(text))
        self.assertIn("compute inspect", text)

    def test_skill_without_shell_gets_no_compute_contract(self):
        """Skills that deny shell do not get compute command instructions."""
        pkg_dir = self._create_valid_package()
        self._restrict_entry_skill(pkg_dir, ["read"])
        out = self.temp_dir / "dist_skill_no_shell"
        compiler.translate(pkg_dir, "claude-code", out, include_internal=True)
        text = (out / "skills" / "entry" /
                "SKILL.md").read_text(encoding="utf-8")
        self.assertNotIn("compute inspect", text)
        self.assertIn("## Installed package contract", text)

    def test_unenforced_harness_warns_about_skill_tools(self):
        """Harnesses that can't enforce skill tools warn and keep the compute section."""
        pkg_dir = self._create_valid_package()
        self._restrict_entry_skill(pkg_dir, ["read"])
        for harness, mode in compiler.SKILL_TOOL_RESTRICTION.items():
            if mode != compiler.SKILL_TOOLS_UNENFORCED:
                continue
            with self.subTest(harness=harness):
                out = self.temp_dir / f"dist_{harness}"
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    compiler.translate(pkg_dir, harness, out)
                self.assertIn("cannot enforce tool restrictions",
                              stderr.getvalue())
                text = (out / "skills" / "entry" /
                        "SKILL.md").read_text(encoding="utf-8")
                self.assertNotIn("disallowed-tools",
                                 self._skill_frontmatter(text))
                self.assertIn("compute inspect", text)

    def test_unclassified_harness_cannot_build_a_restricted_skill(self):
        skill = {"name": "entry", "tools": ["read"]}
        with self.assertRaisesRegex(compiler.TpuAgentSuiteError,
                                    "SKILL_TOOL_RESTRICTION"):
            compiler.skill_tools_enforced(skill, "no-such-harness")

    def test_every_harness_says_how_it_enforces_skill_tools(self):
        """All supported harnesses must declare their skill tool restriction mode."""
        self.assertEqual(set(compiler.SKILL_TOOL_RESTRICTION),
                         set(compiler.HARNESS_TRANSLATORS))

    def test_released_tpu_agent_suite_skill_limits_only_the_main_thread_on_claude(
            self):
        """tpu-agent's shell limit ships as the hook's table, not frontmatter.

        Its delegates run preflight_check.py and kb_lookup.py through Bash;
        a disallowed-tools line on the router's skill took it from them.
        """
        pkg_dir = Path(__file__).resolve().parents[2] / "package"
        out = self.temp_dir / "dist_released_claude"
        with contextlib.redirect_stderr(io.StringIO()):
            compiler.translate(pkg_dir, "claude-code", out)
        fm = self._skill_frontmatter((out / "skills" / "tpu-agent" /
                                      "SKILL.md").read_text(encoding="utf-8"))
        self.assertNotIn("disallowed-tools", fm)
        build_info = json.loads(
            (out / "tpu-agent-suite-build.json").read_text(encoding="utf-8"))
        allowed = build_info["skill_allowed_tools"]["tpu-agent"]
        self.assertNotIn("shell", allowed)
        self.assertIn("delegate", allowed)
        onboarding = (out / "agents" /
                      "tpu-model-onboarding.md").read_text(encoding="utf-8")
        self.assertRegex(self._skill_frontmatter(onboarding),
                         r"(?m)^tools: .*\bBash\b")

    def test_build_info_lists_every_skills_allowed_tools(self):
        """Build info lists the skill's tools plus the always-safe set for the hook."""
        pkg_dir = self._create_valid_package()
        self._restrict_entry_skill(pkg_dir, ["read"])
        out = self.temp_dir / "dist_skill_allowed_tools"
        compiler.translate(pkg_dir, "claude-code", out)
        build_info = json.loads(
            (out / "tpu-agent-suite-build.json").read_text(encoding="utf-8"))
        allowed = build_info["skill_allowed_tools"]["entry"]
        self.assertIn("read", allowed)
        self.assertTrue(
            set(compiler.SKILL_ALWAYS_ALLOWED_TOOLS["claude-code"]) <= set(
                allowed))
        self.assertNotIn("shell", allowed)

    def test_skill_tools_hook_ships_only_where_it_is_enforced(self):
        """The skill-tools hook is only built for harnesses that enforce skill tools."""
        pkg_dir = Path(__file__).resolve().parents[2] / "package"
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        names = {
            hook["name"]
            for hook in manifest["hooks"]
            if hook.get("action") == "builtin:skill-tools"
        }
        self.assertTrue(names)
        for harness in compiler.HARNESS_TRANSLATORS:
            with self.subTest(harness=harness):
                config = compiler.build_harness_hooks_config(
                    manifest["name"], manifest, harness)
                enforced = (compiler.SKILL_TOOL_RESTRICTION[harness] ==
                            compiler.SKILL_TOOLS_HOOK)
                self.assertEqual(names <= set(config), enforced)

    def test_skill_tools_hook_matches_every_tool(self):
        """On Claude Code the skill-tools PreToolUse hook runs for every tool."""
        pkg_dir = Path(__file__).resolve().parents[2] / "package"
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        config = compiler.build_harness_hooks_config(manifest["name"],
                                                     manifest, "claude-code")
        matchers = [
            entry["matcher"] for hooks in config.values()
            for entry in hooks.get("PreToolUse", [])
            if "--builtin skill-tools" in entry["hooks"][0]["command"]
        ]
        self.assertEqual(matchers, ["*"])

    def test_antigravity_installs_agents_where_it_discovers_them(self):
        """Antigravity discovers subagents at `.agents/agents/<name>.md` and
        `~/.gemini/config/agents/`, and nowhere else.

        `.agent/agents/` is accepted as a general customization root but is not
        a subagent discovery path, so agents installed there exist on disk and
        stay invisible. The orchestrator's invocation then fails with "not
        found or not allowed to be invoked", and running the agent's
        instructions inline is the only path left -- which is precisely the
        isolation failure the delegation contract exists to prevent.
        """
        pkg_dir = self._create_valid_package()
        out = self.temp_dir / "dist_agy_paths"
        compiler.translate(pkg_dir, "antigravity", out, include_internal=True)

        plan = json.loads(
            (out / "tpu-agent-suite-install.json").read_text(encoding="utf-8"))
        agent_dests = [
            c["destination"] for c in plan["copies"] if c["source"] == "agents"
        ]
        self.assertEqual(agent_dests, [".agents/agents"])

    def test_delegation_tool_matches_the_runtime_tool_mapping(self):
        """The compiler names the tool in a contract and the runtime matches
        hook events against it. Two copies of that name would agree today and
        diverge the first time a harness is added to only one."""
        mappings = utils.load_json(
            pathlib.Path(compiler.__file__).resolve().parent / "schemas" /
            "tool-mappings.json")
        declared = mappings["tools"]["delegate"]["harnesses"]
        # A Google harness inherits the family's block rather than declaring
        # one, so it is sourced from this mapping without appearing in it.
        inherited = {
            h
            for h in utils.GOOGLE_HARNESSES
            if utils.harness_mapping(declared, h)
        }
        self.assertEqual(set(compiler.HARNESS_DELEGATION_TOOLS),
                         set(declared) | inherited)
        for harness, tool in compiler.HARNESS_DELEGATION_TOOLS.items():
            if tool in (compiler.DELEGATION_NONE,
                        compiler.DELEGATION_UNVERIFIED):
                continue
            self.assertEqual(
                tool,
                utils.harness_mapping(declared, harness)["tools"][0])

    def test_skill_may_also_be_an_agent_when_it_is_one_document(self):
        """A skill that delegates must be selectable with `--agent`.

        As a skill alone it is subject to progressive disclosure: a model can
        decline to open it and route nothing, which is observed behaviour, not
        a hypothetical. Emitting the same document as an agent gives it the
        delegation contract and a name a user can select.
        """
        pkg_dir = self._create_valid_package()
        manifest = compiler.load_package(pkg_dir)
        twin = next(agent for agent in manifest["agents"]
                    if agent["name"] == "entry")
        self.assertEqual(twin["instructions"], "skills/entry/SKILL.md")
        self.assertEqual(twin["delegates"], ["worker"])

    def test_unknown_harness_delegation_is_a_build_error(self):
        """A harness with no declared mechanism must fail the build rather than
        ship routing instructions its runtime cannot honor."""
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.delegation_tool_for("no-such-harness", ["worker"])
        self.assertIn("no delegation mechanism", str(ctx.exception))

    def test_no_delegates_needs_no_mechanism(self):
        """Nothing routed means nothing to declare, even on an unknown harness."""
        self.assertEqual(compiler.delegation_tool_for("no-such-harness", []),
                         "")

    def test_single_context_harness_says_so(self):
        """Cursor inlines every agent as a persona, so the contract must tell
        the model to adopt them rather than imply a subagent exists."""
        rendered = compiler.delegation_mechanism_lines("cursor", ["worker"])
        self.assertIn("no subagent to call", rendered)
        self.assertTrue(rendered.startswith("\n- "))

    def test_google_root_ends_turn_with_a_heartbeat(self):
        """Antigravity's call returns a handle and the result wakes the
        caller, but a headless run drops an idle root after 30 minutes, so
        the root's contract ends the turn behind a `schedule` heartbeat armed
        on the subagent id, and never waits by reading its transcript."""
        rendered = compiler.delegation_mechanism_lines("antigravity",
                                                       ["worker"],
                                                       root=True)
        self.assertIn("`invoke_subagent`", rendered)
        self.assertIn("returns a handle", rendered)
        self.assertIn("wakes you", rendered)
        self.assertIn("`schedule`", rendered)
        self.assertIn(
            f"`DurationSeconds: {compiler.ANTIGRAVITY_HEARTBEAT_SECONDS}`",
            rendered)
        self.assertIn("`TimerCondition`", rendered)
        self.assertIn("end your turn", rendered)
        self.assertIn("never read the subagent's transcript", rendered)
        self.assertNotIn("Stay in the turn", rendered)
        self.assertNotIn("Nothing wakes you", rendered)
        # A dead subagent sends a system message, not a result; a silent one
        # is caught by exactly one status check per heartbeat wake. Measured:
        # one `list` per wake over three wakes, no re-arm after a failure.
        self.assertIn("went idle without reporting", rendered)
        self.assertIn("one `manage_subagents` call", rendered)
        self.assertIn("then end the turn either way", rendered)
        self.assertIn("not running, report it as stopped", rendered)
        self.assertIn("cancel the heartbeat with `manage_task`", rendered)

    def test_google_nested_agent_ends_turn_without_a_heartbeat(self):
        """A nested delegator's idleness ends nothing, so it waits by ending
        its turn and gets no heartbeat and no `schedule`."""
        rendered = compiler.delegation_mechanism_lines("antigravity",
                                                       ["worker"])
        self.assertIn("returns a handle", rendered)
        self.assertIn("end your turn", rendered)
        self.assertIn("never read the subagent's transcript", rendered)
        self.assertIn("never check on it", rendered)
        self.assertIn("went idle without reporting", rendered)
        for absent in ("`schedule`", "TimerCondition", "30 minutes",
                       "`Action: list`", "manage_task"):
            self.assertNotIn(absent, rendered)
        self.assertNotIn("schedule", compiler.GOOGLE_DELEGATION_TOOLS)

    def test_google_root_twin_is_granted_schedule(self):
        """The eval starts the root with `--agent <skill>`, which loads the
        skill's agent twin and its explicit tool list, so the heartbeat tool
        has to be granted there; nested delegators do not get it."""
        manifest = {"skills": [{"name": "tpu-agent"}]}
        root = {
            "name": "tpu-agent",
            "description": "r",
            "delegates": ["worker"]
        }
        nested = {
            "name": "worker",
            "description": "n",
            "delegates": ["helper"]
        }
        self.assertTrue(compiler.is_root_agent(manifest, root))
        self.assertFalse(compiler.is_root_agent(manifest, nested))
        root_fm = compiler.google_harness_frontmatter(root, [], {}, root=True)
        nested_fm = compiler.google_harness_frontmatter(nested, [], {})
        self.assertIn("  - schedule\n", root_fm)
        self.assertIn("  - manage_task\n", root_fm)
        self.assertNotIn("schedule", nested_fm)
        self.assertNotIn("manage_task", nested_fm)
        for fm in (root_fm, nested_fm):
            self.assertIn("  - invoke_subagent\n", fm)
            self.assertIn("  - manage_subagents\n", fm)

    def test_google_heartbeat_is_under_the_headless_cap(self):
        """A heartbeat longer than the 30-minute idle cap would never fire."""
        self.assertLess(compiler.ANTIGRAVITY_HEARTBEAT_SECONDS, 30 * 60)

    def test_foreground_harness_gets_no_waiting_instructions(self):
        """Claude Code's build disables background tasks, so `Agent` blocks
        and returns the result: the contract must not describe a handle, a
        transcript, a heartbeat, or any way to wait."""
        rendered = compiler.delegation_mechanism_lines("claude-code",
                                                       ["worker"])
        self.assertIn("`Agent`", rendered)
        self.assertIn("returns its result", rendered)
        for absent in ("handle", "transcript", "manage_subagents",
                       "`schedule`", "TimerCondition", "end your turn",
                       "Stay in the turn", "30 minutes"):
            self.assertNotIn(absent, rendered)

    def test_waiting_contract_follows_the_install_plan(self):
        """The foreground branch is decided by the setting that makes it
        true. Dropping the setting must flip the text, not leave it
        asserting a blocking call that no longer blocks."""
        self.assertTrue(compiler.delegation_is_foreground("claude-code"))
        self.assertFalse(compiler.delegation_is_foreground("antigravity"))
        with mock.patch.dict(compiler.CLAUDE_CODE_SETTINGS_ENV, clear=True):
            self.assertFalse(compiler.delegation_is_foreground("claude-code"))
            rendered = compiler.delegation_mechanism_lines(
                "claude-code", ["worker"])
            self.assertIn("returns a handle", rendered)
            self.assertNotIn("`schedule`", rendered)

    def test_unverified_harness_names_nothing(self):
        """Better silent than wrong: an unconfirmed call must not be asserted."""
        rendered = compiler.delegation_mechanism_lines("codex", ["worker"])
        self.assertEqual(rendered, "")

    def test_translate_hooks_all_harnesses(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "sensitive_check.py").write_text(
            "print('checked')", encoding="utf-8")

        manifest["hooks"] = [{
            "name": "sensitive-guard",
            "event": "PreToolUse",
            "matchers": ["read", "write", "shell"],
            "action": "script:scripts/sensitive_check.py",
            "timeout": 5,
            "visibility": "external",
        }, {
            "name": "auto-artifact-tracker",
            "event": "PostToolUse",
            "matchers": ["write", "shell"],
            "command": "python3 track_cli.py",
            "args": ["--verbose"],
            "visibility": "external",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        # 1. Claude Code
        out_claude = self.temp_dir / "dist_claude_hooks"
        compiler.translate(pkg_dir, "claude-code", out_claude)
        self.assertTrue(
            (out_claude / ".claude-plugin" / "hooks.json").is_file())
        self.assertTrue((out_claude / "project" / "hooks.json").is_file())
        self.assertTrue(
            (out_claude / "scripts" / "sensitive_check.py").is_file())

        claude_hooks = json.loads((out_claude / ".claude-plugin" /
                                   "hooks.json").read_text(encoding="utf-8"))
        self.assertEqual(
            claude_hooks["sensitive-guard"]["PreToolUse"][0]["matcher"],
            "Read|Write|Bash")
        self.assertEqual(
            claude_hooks["sensitive-guard"]["PreToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package valid-pkg@1.0.0 --hook sensitive-guard "
                "--event PreToolUse --timeout 5 --on-failure deny --script scripts/sensitive_check.py",
                "sensitive-guard", "deny", "PreToolUse"),
        )
        # The harness outwaits dispatch's own --timeout, so dispatch's verdict
        # on a hung hook reaches the harness instead of being killed with it.
        self.assertEqual(
            claude_hooks["sensitive-guard"]["PreToolUse"][0]["hooks"][0]
            ["timeout"], 5 + compiler.HOOK_DISPATCH_HEADROOM_S)
        self.assertEqual(
            claude_hooks["auto-artifact-tracker"]["PostToolUse"][0]["matcher"],
            "Write|Bash")
        self.assertEqual(
            claude_hooks["auto-artifact-tracker"]["PostToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package valid-pkg@1.0.0 --hook auto-artifact-tracker "
                "--event PostToolUse --timeout 10 --on-failure deny --command 'python3 track_cli.py --verbose'",
                "auto-artifact-tracker", "deny", "PostToolUse"),
        )
        self.assertTrue((out_claude / "framework" / "schemas" /
                         "hook-output.schema.json").is_file())

        # Check tpu-agent-suite-install.json has hooks.json in copy plan
        install_plan = json.loads(
            (out_claude /
             "tpu-agent-suite-install.json").read_text(encoding="utf-8"))
        destinations = [c["destination"] for c in install_plan["copies"]]
        self.assertIn(".claude/hooks.json", destinations)

        # 2. Antigravity
        out_antigravity = self.temp_dir / "dist_ag_hooks"
        compiler.translate(pkg_dir, "antigravity", out_antigravity)
        self.assertTrue((out_antigravity / "hooks.json").is_file())
        self.assertTrue(
            (out_antigravity / "scripts" / "sensitive_check.py").is_file())
        self.assertTrue((out_antigravity / "framework" / "schemas" /
                         "hook-output.schema.json").is_file())
        ag_hooks = json.loads(
            (out_antigravity / "hooks.json").read_text(encoding="utf-8"))
        self.assertEqual(
            ag_hooks["sensitive-guard"]["PreToolUse"][0]["matcher"],
            "view_file|write_to_file|run_command")
        self.assertEqual(
            ag_hooks["sensitive-guard"]["PreToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness antigravity --package valid-pkg@1.0.0 --hook sensitive-guard "
                "--event PreToolUse --timeout 5 --on-failure deny --script scripts/sensitive_check.py",
                "sensitive-guard", "deny", "PreToolUse"),
        )
        self.assertEqual(
            ag_hooks["auto-artifact-tracker"]["PostToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness antigravity --package valid-pkg@1.0.0 --hook auto-artifact-tracker "
                "--event PostToolUse --timeout 10 --on-failure deny --command 'python3 track_cli.py --verbose'",
                "auto-artifact-tracker", "deny", "PostToolUse"),
        )

        # 3. Cursor
        out_cursor = self.temp_dir / "dist_cursor_hooks"
        compiler.translate(pkg_dir, "cursor", out_cursor)
        self.assertTrue((out_cursor / ".cursor" / "hooks.json").is_file())
        self.assertTrue(
            (out_cursor / "scripts" / "sensitive_check.py").is_file())
        self.assertTrue((out_cursor / "framework" / "schemas" /
                         "hook-output.schema.json").is_file())
        cursor_hooks = json.loads((out_cursor / ".cursor" /
                                   "hooks.json").read_text(encoding="utf-8"))
        # Cursor's native shape: version-tagged, keyed by event, with the
        # matcher on the definition rather than wrapping it.
        self.assertEqual(cursor_hooks["version"], 1)
        self.assertEqual(sorted(cursor_hooks["hooks"]),
                         ["postToolUse", "preToolUse"])
        pre_defs = cursor_hooks["hooks"]["preToolUse"]
        self.assertEqual(len(pre_defs), 1)
        self.assertEqual(pre_defs[0]["matcher"], "Read|Write|Shell")
        self.assertEqual(pre_defs[0]["type"], "command")
        self.assertEqual(pre_defs[0]["timeout"],
                         5 + compiler.HOOK_DISPATCH_HEADROOM_S)
        self.assertEqual(
            pre_defs[0]["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness cursor --package valid-pkg@1.0.0 --hook sensitive-guard "
                "--event PreToolUse --timeout 5 --on-failure deny --script scripts/sensitive_check.py",
                "sensitive-guard", "deny", "PreToolUse"),
        )

        # 4. Codex
        out_codex = self.temp_dir / "dist_codex_hooks"
        compiler.translate(pkg_dir, "codex", out_codex)
        self.assertTrue((out_codex / ".codex" / "hooks.json").is_file())
        self.assertTrue((out_codex / ".codex-plugin" / "hooks.json").is_file())
        self.assertTrue(
            (out_codex / "scripts" / "sensitive_check.py").is_file())
        self.assertTrue((out_codex / "framework" / "schemas" /
                         "hook-output.schema.json").is_file())
        codex_hooks = json.loads(
            (out_codex / ".codex" / "hooks.json").read_text(encoding="utf-8"))
        # Codex's native shape: one document keyed by event, each holding
        # matcher groups. `read` has no Codex tool, so it falls through to the
        # canonical name and matches nothing.
        self.assertEqual(sorted(codex_hooks["hooks"]),
                         ["PostToolUse", "PreToolUse"])
        pre_groups = codex_hooks["hooks"]["PreToolUse"]
        self.assertEqual(len(pre_groups), 1)
        self.assertEqual(pre_groups[0]["matcher"], "read|Write|Bash")
        self.assertEqual(
            pre_groups[0]["hooks"][0]["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness codex --package valid-pkg@1.0.0 --hook sensitive-guard "
                "--event PreToolUse --timeout 5 --on-failure deny --script scripts/sensitive_check.py",
                "sensitive-guard", "deny", "PreToolUse"),
        )
        codex_install_plan = json.loads(
            (out_codex /
             "tpu-agent-suite-install.json").read_text(encoding="utf-8"))
        codex_destinations = [
            c["destination"] for c in codex_install_plan["copies"]
        ]
        self.assertIn(".codex/hooks.json", codex_destinations)
        self.assertTrue(
            any(
                t.get("path") == ".codex/config.toml"
                and t.get("section") == "hooks" and t.get("key") == "enabled"
                and t.get("value") is True
                for t in codex_install_plan.get("toml_sets", [])))

    def test_hook_on_failure_reaches_the_dispatch_command(self):
        """Both policies are passed to dispatch, the default `deny` included."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "guard.py").write_text("print('{}')",
                                                      encoding="utf-8")
        manifest["hooks"] = [{
            "name": "lenient-guard",
            "event": "PreToolUse",
            "action": "script:scripts/guard.py",
            "timeout": 5,
            "on_failure": "allow",
        }, {
            "name": "strict-guard",
            "event": "PreToolUse",
            "action": "script:scripts/guard.py",
            "timeout": 5,
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        out_claude = self.temp_dir / "dist_claude_on_failure"
        compiler.translate(pkg_dir, "claude-code", out_claude)
        claude_hooks = json.loads((out_claude / ".claude-plugin" /
                                   "hooks.json").read_text(encoding="utf-8"))

        lenient = claude_hooks["lenient-guard"]["PreToolUse"][0]["hooks"][0]
        strict = claude_hooks["strict-guard"]["PreToolUse"][0]["hooks"][0]

        prefix = (
            "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py"
            " hook-dispatch --harness claude-code --package valid-pkg@1.0.0")
        self.assertEqual(
            lenient["command"],
            compiler.fail_safe_hook_command(
                f"{prefix} --hook lenient-guard --event PreToolUse --timeout 5 "
                "--on-failure allow --script scripts/guard.py",
                "lenient-guard", "allow", "PreToolUse"))
        self.assertEqual(
            strict["command"],
            compiler.fail_safe_hook_command(
                f"{prefix} --hook strict-guard --event PreToolUse --timeout 5 "
                "--on-failure deny --script scripts/guard.py", "strict-guard",
                "deny", "PreToolUse"))

    def _run_hook_command(self,
                          command: str,
                          home: Path,
                          path: str | None = None):
        """Run a compiled hook command as a harness would: in a shell, with
        an empty payload on stdin, `~` pointing at `home`."""
        env = {"HOME": str(home), "PATH": path or os.environ.get("PATH", "")}
        return subprocess.run(["/bin/sh", "-c", command],
                              input="{}",
                              capture_output=True,
                              text=True,
                              env=env,
                              check=False)

    def test_compiled_hook_command_applies_policy_when_dispatch_cannot_run(
            self):
        """The command exits by `on_failure` when hook-dispatch cannot run.

        Without the fallback a missing runtime.py exits 2, which Claude Code
        reads as a block: every `allow` hook denied every tool and Stop hooks
        kept the turn from ending.
        """
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "guard.py").write_text("print('{}')",
                                                      encoding="utf-8")
        manifest["hooks"] = [{
            "name": "lenient-guard",
            "event": "PreToolUse",
            "action": "script:scripts/guard.py",
            "on_failure": "allow",
        }, {
            "name": "strict-guard",
            "event": "PreToolUse",
            "action": "script:scripts/guard.py",
            "on_failure": "deny",
        }]
        hooks = compiler.build_harness_hooks_config("valid-pkg", manifest,
                                                    "claude-code")
        lenient = hooks["lenient-guard"]["PreToolUse"][0]["hooks"][0]
        strict = hooks["strict-guard"]["PreToolUse"][0]["hooks"][0]

        empty_home = self.temp_dir / "empty_home"
        empty_home.mkdir()

        res = self._run_hook_command(lenient["command"], empty_home)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("hook lenient-guard: hook-dispatch exited 2", res.stderr)
        self.assertIn("on_failure=allow", res.stderr)

        res = self._run_hook_command(strict["command"], empty_home)
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("on_failure=deny", res.stderr)

        # Keyed on the exit, not the file: missing interpreter (127) and a
        # dispatcher that dies after starting (1) are answered the same.
        res = self._run_hook_command(lenient["command"],
                                     empty_home,
                                     path="/nonexistent")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("hook-dispatch exited 127", res.stderr)

        stub_home = self.temp_dir / "stub_home"
        stub_runtime = (stub_home / ".tpu-agent-suite" / "packages" /
                        "valid-pkg@1.0.0" / "framework" / "runtime.py")
        stub_runtime.parent.mkdir(parents=True)
        stub_runtime.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        res = self._run_hook_command(lenient["command"], stub_home)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("hook-dispatch exited 1", res.stderr)

    def test_compiled_stop_hook_command_never_blocks_when_dispatch_cannot_run(
            self):
        """On Stop and SubagentStop a block means "keep going", so a dead
        dispatcher's deny would block every stop and re-run every subagent
        with nothing able to end it. There the fallback exits 0 whatever the
        policy -- the default deny included -- and says so. The hand-back
        road is a SubagentStop registered on a tool event; same rule."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "guard.py").write_text("print('{}')",
                                                      encoding="utf-8")
        manifest["hooks"] = [{
            "name": "strict-stop",
            "event": "Stop",
            "action": "script:scripts/guard.py",
        }, {
            "name": "strict-subagent-stop",
            "event": "SubagentStop",
            "action": "script:scripts/guard.py",
        }]
        hooks = compiler.build_harness_hooks_config("valid-pkg", manifest,
                                                    "claude-code")
        empty_home = self.temp_dir / "empty_home"
        empty_home.mkdir()

        commands = {
            "Stop":
            hooks["strict-stop"]["Stop"][0]["hooks"][0]["command"],
            "SubagentStop":
            hooks["strict-subagent-stop"]["SubagentStop"][0]["hooks"][0]
            ["command"],
            "SubagentStop hand-back":
            hooks["strict-subagent-stop"]["PreToolUse"][0]["hooks"][0]
            ["command"],
        }
        for label, command in commands.items():
            with self.subTest(event=label):
                res = self._run_hook_command(command, empty_home)
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertIn("allowing: on_failure=deny would loop on",
                              res.stderr)

    def _store_with_framework(self, name: str) -> Path:
        """A home whose store holds this framework under valid-pkg@1.0.0."""
        home = self.temp_dir / name
        pkg_root = home / ".tpu-agent-suite" / "packages" / "valid-pkg@1.0.0"
        shutil.copytree(Path(compiler.__file__).parent,
                        pkg_root / "framework",
                        ignore=shutil.ignore_patterns("tests", "__pycache__"))
        (pkg_root / "package.release.json").write_text(json.dumps({
            "name":
            "valid-pkg",
            "version":
            "1.0.0"
        }),
                                                       encoding="utf-8")
        return home

    def test_compiled_hook_command_leaves_a_working_dispatch_alone(self):
        """The fallback is inert when dispatch answers: its reply and its
        exit 0 pass through, and nothing is added to stderr."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["hooks"] = [{
            "name": "strict-guard",
            "event": "PreToolUse",
            "command": "true",
            "on_failure": "deny",
        }]
        hooks = compiler.build_harness_hooks_config("valid-pkg", manifest,
                                                    "claude-code")
        strict = hooks["strict-guard"]["PreToolUse"][0]["hooks"][0]

        home = self._store_with_framework("home_with_store")
        res = self._run_hook_command(strict["command"], home)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertNotIn("before a verdict", res.stderr)
        self.assertEqual(
            json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"],
            "allow")

    def test_dispatcher_answers_by_policy_when_its_own_import_fails(self):
        """runtime.py present but package.release.json gone: `utils` raises at import,
        and the dispatcher exits by `on_failure` with the reason, not a
        traceback."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["hooks"] = [{
            "name": "lenient-guard",
            "event": "PreToolUse",
            "command": "true",
            "on_failure": "allow",
        }, {
            "name": "strict-guard",
            "event": "PreToolUse",
            "command": "true",
            "on_failure": "deny",
        }]
        hooks = compiler.build_harness_hooks_config("valid-pkg", manifest,
                                                    "claude-code")
        home = self._store_with_framework("home_half_gone")
        (home / ".tpu-agent-suite" / "packages" / "valid-pkg@1.0.0" /
         "package.release.json").unlink()

        res = self._run_hook_command(
            hooks["lenient-guard"]["PreToolUse"][0]["hooks"][0]["command"],
            home)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("[lenient-guard] hook failed (startup)", res.stderr)
        self.assertIn("TpuAgentSuiteError", res.stderr)
        self.assertNotIn("Traceback", res.stderr)
        # The dispatcher answered, so the shell fallback had nothing to add.
        self.assertNotIn("before a verdict", res.stderr)

        res = self._run_hook_command(
            hooks["strict-guard"]["PreToolUse"][0]["hooks"][0]["command"],
            home)
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("[strict-guard] hook failed (startup)", res.stderr)
        # A deny settled by exit code is exit 2, which the shell fallback
        # cannot tell from a crash, so it reports too. Two lines, one exit:
        # the fallback must not go quiet on 2, or a missing runtime.py
        # (python3 exits 2) would leave no trace at all.
        self.assertIn("hook strict-guard: hook-dispatch exited 2", res.stderr)

    def test_fail_safe_exit_codes_name_every_policy(self):
        """The schema and `utils.HOOK_FAILURE_POLICIES` own the policy list;
        the exit-code table must name the same set, or a new policy would
        validate and then fail at compile time."""
        schema = utils.get_package_manifest_schema()
        declared = schema["properties"]["hooks"]["items"]["properties"][
            "on_failure"]["enum"]
        self.assertEqual(set(declared), set(utils.HOOK_FAILURE_POLICIES))
        self.assertEqual(set(compiler.HOOK_FAIL_SAFE_EXIT_CODES),
                         set(utils.HOOK_FAILURE_POLICIES))

    def test_hooks_with_arguments(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "guard.py").write_text("print('{}')",
                                                      encoding="utf-8")
        manifest["hooks"] = [
            {
                "name": "script-with-args",
                "event": "PreToolUse",
                "action": "script:scripts/guard.py",
                "args": ["--strict", "param with space"],
            },
            {
                "name": "cmd-with-args",
                "event": "PostToolUse",
                "command": "python3 track.py",
                "args": ["-m", "msg with space"],
            },
        ]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        hooks = compiler.build_harness_hooks_config("test-pkg", manifest,
                                                    "claude-code")
        self.assertEqual(
            hooks["script-with-args"]["PreToolUse"][0]["hooks"][0]["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/test-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package test-pkg@1.0.0 --hook script-with-args "
                "--event PreToolUse --timeout 10 --on-failure deny --script scripts/guard.py -- --strict 'param with space'",
                "script-with-args", "deny", "PreToolUse"),
        )
        expected_cmd = compiler.fail_safe_hook_command(
            "python3 ~/.tpu-agent-suite/packages/test-pkg@1.0.0/framework/runtime.py hook-dispatch "
            "--harness claude-code --package test-pkg@1.0.0 --hook cmd-with-args "
            f"--event PostToolUse --timeout 10 --on-failure deny --command {shlex.quote('python3 track.py -m ' + shlex.quote('msg with space'))}",
            "cmd-with-args", "deny", "PostToolUse")
        self.assertEqual(
            hooks["cmd-with-args"]["PostToolUse"][0]["hooks"][0]["command"],
            expected_cmd,
        )

    def test_subagent_hooks_compilation(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "subagent_guard.py").write_text(
            "print('{}')", encoding="utf-8")
        manifest["hooks"] = [
            {
                "name": "kernel-subagent-guard",
                "event": "PreToolUse",
                "matchers": ["delegate"],
                "action": "script:scripts/subagent_guard.py",
            },
            {
                "name": "global-subagent-verifier",
                "event": "PostToolUse",
                "matchers": ["delegate"],
                "command": "python3 verify_subagent.py",
            },
        ]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

        # Compile for Claude Code
        claude_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "claude-code")
        self.assertIn("kernel-subagent-guard", claude_hooks)
        self.assertIn("PreToolUse", claude_hooks["kernel-subagent-guard"])
        self.assertEqual(
            claude_hooks["kernel-subagent-guard"]["PreToolUse"][0]["matcher"],
            "Agent")
        cmd_claude = claude_hooks["kernel-subagent-guard"]["PreToolUse"][0][
            "hooks"][0]["command"]
        self.assertIn("--event PreToolUse", cmd_claude)

        self.assertIn("global-subagent-verifier", claude_hooks)
        self.assertIn("PostToolUse", claude_hooks["global-subagent-verifier"])
        self.assertEqual(
            claude_hooks["global-subagent-verifier"]["PostToolUse"][0]
            ["matcher"], "Agent")
        self.assertIn(
            "--event PostToolUse", claude_hooks["global-subagent-verifier"]
            ["PostToolUse"][0]["hooks"][0]["command"])

        # Compile for Antigravity
        ag_hooks = compiler.build_harness_hooks_config("test-pkg", manifest,
                                                       "antigravity")
        self.assertEqual(
            ag_hooks["kernel-subagent-guard"]["PreToolUse"][0]["matcher"],
            "invoke_subagent")
        self.assertEqual(
            ag_hooks["global-subagent-verifier"]["PostToolUse"][0]["matcher"],
            "invoke_subagent")

    def test_subagent_stop_hook_compilation(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "verify_output.py").write_text(
            "print('verify')", encoding="utf-8")
        manifest["hooks"] = [
            {
                "name": "verify-subagent-result",
                "event": "SubagentStop",
                "matchers": ["torch_coder", "tpu-kernel-development"],
                "action": "script:scripts/verify_output.py",
            },
            {
                "name": "catch-all-stop",
                "event": "SubagentStop",
                "action": "script:scripts/verify_output.py",
            },
        ]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        manifest = compiler.load_package(pkg_dir)

        # 1. Claude Code
        claude_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "claude-code")
        self.assertIn("verify-subagent-result", claude_hooks)
        self.assertIn("SubagentStop", claude_hooks["verify-subagent-result"])
        self.assertEqual(
            claude_hooks["verify-subagent-result"]["SubagentStop"][0]
            ["matcher"],
            "torch_coder|tpu-kernel-development",
        )
        self.assertEqual(
            claude_hooks["catch-all-stop"]["SubagentStop"][0]["matcher"],
            "*",
        )
        self.assertIn(
            "--event SubagentStop",
            claude_hooks["verify-subagent-result"]["SubagentStop"][0]["hooks"]
            [0]["command"],
        )

        # A background or auto-mode subagent delivers through the injected
        # SubagentHandback tool instead of its final message, so the same
        # hook body is also registered on that tool call. The tool matcher
        # selects the call; the subagent matchers move into the command.
        handback = claude_hooks["verify-subagent-result"]["PreToolUse"]
        self.assertEqual(len(handback), 1)
        self.assertEqual(handback[0]["matcher"], "SubagentHandback")
        handback_cmd = handback[0]["hooks"][0]["command"]
        self.assertIn("--event SubagentStop", handback_cmd)
        self.assertIn(
            "--handback-event PreToolUse --matcher 'torch_coder|tpu-kernel-development' "
            "--script ", handback_cmd)
        # The script path closes the dispatch invocation; the shell fallback
        # follows it on this road too.
        dispatch_part, fallback = handback_cmd.split(" || ", 1)
        self.assertTrue(dispatch_part.endswith("verify_output.py"))
        self.assertIn("hook verify-subagent-result: hook-dispatch exited",
                      fallback)
        catch_all_cmd = (claude_hooks["catch-all-stop"]["PreToolUse"][0]
                         ["hooks"][0]["command"])
        self.assertNotIn("--matcher", catch_all_cmd)
        # Only the hand-back road names a tool event to answer in.
        self.assertNotIn(
            "--handback-event", claude_hooks["verify-subagent-result"]
            ["SubagentStop"][0]["hooks"][0]["command"])

        # Also verify install plan merges SubagentStop into .claude/settings.local.json
        target = compiler.target_for(manifest, "claude-code")
        plan = compiler.build_install_plan(pkg_dir, manifest, target,
                                           "claude-code")
        settings_merge = [
            m for m in plan["json_merges"]
            if m.get("path") == ".claude/settings.local.json"
        ]
        self.assertEqual(len(settings_merge), 1)
        self.assertIn("SubagentStop", settings_merge[0]["value"]["hooks"])
        self.assertIn("PreToolUse", settings_merge[0]["value"]["hooks"])

        # 2. Codex
        codex_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "codex")
        self.assertIn("verify-subagent-result", codex_hooks)
        self.assertIn("SubagentStop", codex_hooks["verify-subagent-result"])
        self.assertEqual(
            codex_hooks["verify-subagent-result"]["SubagentStop"][0]
            ["matcher"],
            "torch_coder|tpu-kernel-development",
        )
        # No hand-back tool on the other harnesses.
        self.assertNotIn("PreToolUse", codex_hooks["verify-subagent-result"])

        # 3. Cursor
        cursor_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "cursor")
        self.assertIn("verify-subagent-result", cursor_hooks)
        self.assertIn("subagentStop", cursor_hooks["verify-subagent-result"])
        self.assertEqual(
            cursor_hooks["verify-subagent-result"]["subagentStop"][0]
            ["matcher"],
            "torch_coder|tpu-kernel-development",
        )

        # 4. Antigravity
        ag_hooks = compiler.build_harness_hooks_config("test-pkg", manifest,
                                                       "antigravity")
        self.assertIn("verify-subagent-result", ag_hooks)
        self.assertEqual(
            ag_hooks["verify-subagent-result"]["Stop"][0]["type"],
            "command",
        )
        self.assertIn(
            "verify_output.py",
            ag_hooks["verify-subagent-result"]["Stop"][0]["command"],
        )
        self.assertIn(
            "--matcher 'torch_coder|tpu-kernel-development'",
            ag_hooks["verify-subagent-result"]["Stop"][0]["command"],
        )
        self.assertNotIn(
            "--matcher",
            ag_hooks["catch-all-stop"]["Stop"][0]["command"],
        )

    def test_session_start_hook_compilation(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        (pkg_dir / "scripts" / "init_session.py").write_text("print('init')",
                                                             encoding="utf-8")
        manifest["hooks"] = [
            {
                "name": "init-session-guard",
                "event": "SessionStart",
                "action": "script:scripts/init_session.py",
            },
        ]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        manifest = compiler.load_package(pkg_dir)

        # 1. Claude Code
        claude_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "claude-code")
        self.assertIn("init-session-guard", claude_hooks)
        self.assertIn("SessionStart", claude_hooks["init-session-guard"])
        self.assertEqual(
            claude_hooks["init-session-guard"]["SessionStart"][0]["matcher"],
            "*",
        )
        self.assertIn(
            "--event SessionStart",
            claude_hooks["init-session-guard"]["SessionStart"][0]["hooks"][0]
            ["command"],
        )

        # Verify install plan merges SessionStart into .claude/settings.local.json
        target = compiler.target_for(manifest, "claude-code")
        plan = compiler.build_install_plan(pkg_dir, manifest, target,
                                           "claude-code")
        settings_merge = [
            m for m in plan["json_merges"]
            if m.get("path") == ".claude/settings.local.json"
        ]
        self.assertEqual(len(settings_merge), 1)
        self.assertIn("SessionStart", settings_merge[0]["value"]["hooks"])

        # 2. Codex
        codex_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "codex")
        self.assertIn("init-session-guard", codex_hooks)
        self.assertIn("SessionStart", codex_hooks["init-session-guard"])
        self.assertEqual(
            codex_hooks["init-session-guard"]["SessionStart"][0]["matcher"],
            "*",
        )

        # 3. Cursor
        cursor_hooks = compiler.build_harness_hooks_config(
            "test-pkg", manifest, "cursor")
        self.assertIn("init-session-guard", cursor_hooks)
        self.assertIn("sessionStart", cursor_hooks["init-session-guard"])
        self.assertEqual(
            cursor_hooks["init-session-guard"]["sessionStart"][0]["matcher"],
            "*",
        )

        # 4. Antigravity (maps to PreInvocation flat list)
        ag_hooks = compiler.build_harness_hooks_config("test-pkg", manifest,
                                                       "antigravity")
        self.assertIn("init-session-guard", ag_hooks)
        self.assertIn("PreInvocation", ag_hooks["init-session-guard"])
        self.assertEqual(
            ag_hooks["init-session-guard"]["PreInvocation"][0]["type"],
            "command",
        )
        self.assertIn(
            "init_session.py",
            ag_hooks["init-session-guard"]["PreInvocation"][0]["command"],
        )
        self.assertIn(
            "--event SessionStart",
            ag_hooks["init-session-guard"]["PreInvocation"][0]["command"],
        )

    def test_hooks_validation_errors(self):
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))

        # Duplicate hook names
        manifest["hooks"] = [
            {
                "name": "dup",
                "event": "PreToolUse",
                "command": "true",
            },
            {
                "name": "dup",
                "event": "PostToolUse",
                "command": "true",
            },
        ]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("duplicate hook name", str(ctx.exception))

        # Missing script file
        manifest["hooks"] = [{
            "name": "bad_script",
            "event": "PreToolUse",
            "action": "script:scripts/non_existent.py"
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("script file does not exist", str(ctx.exception))

        # Unsupported hook event
        manifest["hooks"] = [{
            "name": "bad_event",
            "event": "InvalidEvent",
            "command": "true",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("is not one of", str(ctx.exception))

        # Hook name that would not survive filename sanitisation. Per-session
        # hook state lives at sessions/<id>.contexts/<name>.json, so "a:b" and
        # "a_b" would share one file and serve each other's cached context.
        manifest["hooks"] = [{
            "name": "bad:name",
            "event": "PreToolUse",
            "command": "true",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("does not match", str(ctx.exception))

        # Builtin attached to an event it cannot serve. start-run rebinds the
        # session to a fresh run namespace, so firing it per tool call would
        # orphan the artifacts of the run it replaced.
        manifest["hooks"] = [{
            "name": "start-run-on-tool-use",
            "event": "PreToolUse",
            "action": "builtin:start-run",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("cannot be attached to event 'PreToolUse'",
                      str(ctx.exception))

        # reap-jobs cancels the run's leftover workloads; on Stop it would
        # fire at every turn boundary and kill a job just submitted.
        manifest["hooks"] = [{
            "name": "reap-on-stop",
            "event": "Stop",
            "action": "builtin:reap-jobs",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("cannot be attached to event 'Stop'", str(ctx.exception))

        # validate-schema attaches only to PreToolUse and SubagentStop; on SessionStart
        # it is rejected.
        manifest["hooks"] = [{
            "name": "validate-on-session-start",
            "event": "SessionStart",
            "action": "builtin:validate-schema",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("cannot be attached to event 'SessionStart'",
                      str(ctx.exception))

        # Missing both action and command
        manifest["hooks"] = [{
            "name": "missing_action_and_cmd",
            "event": "PreToolUse",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("must specify either an 'action' or 'command'",
                      str(ctx.exception))

        # Both action and command specified
        manifest["hooks"] = [{
            "name": "both_action_and_cmd",
            "event": "PreToolUse",
            "action": "script:scripts/guard.py",
            "command": "python3 guard.py",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("cannot specify both 'action' and 'command'",
                      str(ctx.exception))

        # Invalid action prefix (missing script:)
        manifest["hooks"] = [{
            "name": "invalid_action_prefix",
            "event": "PreToolUse",
            "action": "scripts/guard.py",
        }]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("action must start with 'script:'", str(ctx.exception))

        # Invalid timeouts (non-positive integer, zero, boolean, string)
        for invalid_timeout in [-5, 0, True, "10"]:
            manifest["hooks"] = [{
                "name": "invalid_timeout",
                "event": "PreToolUse",
                "action": "script:scripts/guard.py",
                "timeout": invalid_timeout,
            }]
            (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                          encoding="utf-8")
            with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                compiler.load_package(pkg_dir)
            err_str = str(ctx.exception)
            self.assertTrue(
                "below minimum" in err_str or "expected integer" in err_str
                or "timeout must be a positive integer" in err_str,
                f"Unexpected error message: {err_str}",
            )

        # Invalid on_failure policies (unknown word, wrong case, non-string),
        # each with the error the schema reports for it
        for invalid_policy, expected_error in [
            ("maybe", "on_failure: value is not one of ['deny', 'allow']"),
            ("Allow", "on_failure: value is not one of ['deny', 'allow']"),
            (True, "on_failure: expected string, got bool"),
        ]:
            manifest["hooks"] = [{
                "name": "invalid_on_failure",
                "event": "PreToolUse",
                "action": "script:scripts/guard.py",
                "on_failure": invalid_policy,
            }]
            (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                          encoding="utf-8")
            with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                compiler.load_package(pkg_dir)
            self.assertIn(expected_error, str(ctx.exception))

    def _package_with_memory_block(self, entry: dict) -> Path:
        """Build a valid package carrying one workspace_memory declaration."""
        pkg_dir = self._create_valid_package()
        manifest = json.loads(
            (pkg_dir / "package.release.json").read_text(encoding="utf-8"))
        manifest["workspace_memory"] = [entry]
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        (pkg_dir / "memory").mkdir(parents=True, exist_ok=True)
        return pkg_dir

    def test_workspace_memory_requires_existing_content_file(self):
        """A block naming a missing content file fails verification."""
        pkg_dir = self._package_with_memory_block({
            "marker":
            "kb",
            "content":
            "memory/absent.md",
        })

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn("workspace memory content", str(ctx.exception))

    def test_workspace_memory_rejects_unknown_tokens(self):
        """A memory block naming an unknown resource is a compile error."""
        pkg_dir = self._package_with_memory_block({
            "marker":
            "kb",
            "content":
            "memory/contract.md",
        })
        (pkg_dir / "memory" / "contract.md").write_text(
            "Base at {{install_root}}/.", encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.translate(pkg_dir, "claude-code",
                               self.temp_dir / "dist_no_asset")
        self.assertIn("unknown resource tokens", str(ctx.exception))

    def test_workspace_memory_rejects_workspace_resources(self):
        """Memory blocks load from arbitrary cwds, so only central resources may appear."""
        pkg_dir = self._package_with_memory_block({
            "marker":
            "kb",
            "content":
            "memory/contract.md",
        })
        (pkg_dir / "memory" / "contract.md").write_text(
            "Config at .tpu-agent-suite/environment.json.", encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.translate(pkg_dir, "claude-code",
                               self.temp_dir / "dist_bad_asset")
        self.assertIn("central resources only", str(ctx.exception))

    def test_document_driven_agent_loading_explicit(self):
        """Document-driven agents defined in .md with frontmatter load and verify."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        agent_md = ("---\n"
                    "name: doc-agent\n"
                    "description: Document-driven agent description.\n"
                    "tools:\n"
                    "  - read\n"
                    "  - write\n"
                    "accepts: schemas/in.json\n"
                    "produces: schemas/out.json\n"
                    "delegates: []\n"
                    "scripts: []\n"
                    "references: []\n"
                    "---\n"
                    "# Doc Agent Instructions\n"
                    "Execute tasks directly.")
        (pkg_dir / "agents" / "doc-agent.md").write_text(agent_md,
                                                         encoding="utf-8")
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - doc-agent\n"
            "---\n"
            "Call {{agent:doc-agent}}.",
            encoding="utf-8")

        compiler.verify_package(pkg_dir)
        dist_claude = self.temp_dir / "dist_doc_claude"
        compiler.translate(pkg_dir, "claude-code", dist_claude)
        self.assertTrue((dist_claude / "agents" / "doc-agent.md").is_file())
        claude_content = (dist_claude / "agents" /
                          "doc-agent.md").read_text(encoding="utf-8")
        self.assertIn("tools: Read, Write", claude_content)
        self.assertIn("Execute tasks directly.", claude_content)

    def test_document_driven_agent_loading_explicit_paths(self):
        """Document-driven agents loaded via explicit file paths."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        for name in ("agent-a", "agent-b"):
            agent_md = (f"---\n"
                        f"name: {name}\n"
                        f"description: {name} description.\n"
                        f"tools:\n"
                        f"  - read\n"
                        f"accepts: schemas/in.json\n"
                        f"produces: schemas/out.json\n"
                        f"delegates: []\n"
                        f"---\n"
                        f"# {name} prompt.")
            (pkg_dir / "agents" / f"{name}.md").write_text(agent_md,
                                                           encoding="utf-8")

        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - agent-a\n"
            "  - agent-b\n"
            "---\n"
            "Call {{agent:agent-a}} and {{agent:agent-b}}.",
            encoding="utf-8")
        composed = compiler.compose_package(pkg_dir)
        agent_names = [a["name"] for a in composed["agents"]]
        # The skill's own twin follows the agents it reaches.
        self.assertEqual(agent_names, ["agent-a", "agent-b", "entry"])
        compiler.verify_package(pkg_dir)

    def test_skill_name_cannot_escape_package(self):
        """A skill is named, not located: a path in `skills` is rejected."""
        pkg_dir = self._create_valid_package()
        manifest = utils.load_json(pkg_dir / "package.release.json")
        for bad_name in ("../outside", "nested/skill",
                         "skills/entry/SKILL.md"):
            with self.subTest(name=bad_name):
                manifest["skills"] = [bad_name]
                utils.write_json(pkg_dir / "package.release.json", manifest)
                with self.assertRaises(compiler.TpuAgentSuiteError):
                    compiler.load_package(pkg_dir)

    def test_removed_manifest_fields_are_rejected_by_name(self):
        """A field package.release.json once carried fails with where it lives now,
        not as an anonymous unknown property."""
        pkg_dir = self._create_valid_package()
        original = utils.load_json(pkg_dir / "package.release.json")
        values = {
            "entrypoint": "skills/entry/SKILL.md",
            "agents": ["agents/worker.md"],
            "include": ["fragments/tools.json"],
            "schemas": ["schemas/in.json"],
            "host_dependencies": ["pydantic"],
            "tpu_dependencies": ["torch"],
            "required_capabilities": ["subagents"],
        }
        self.assertEqual(set(values), set(compiler.REMOVED_MANIFEST_FIELDS))
        for field, value in values.items():
            with self.subTest(field=field):
                utils.write_json(pkg_dir / "package.release.json", {
                    **original, field: value
                })
                with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
                    compiler.load_package(pkg_dir)
                self.assertIn("no longer supported", str(ctx.exception))
                self.assertIn(f"  {field}: ", str(ctx.exception))

    def test_manifest_composition_does_not_mutate_input(self):
        """resolve_manifest_composition must not mutate caller manifest."""
        pkg_dir = self._create_valid_package()
        raw_manifest = {
            "name": "immutability-test",
            "version": "0.2.0",
            "skills": ["entry"],
        }
        before = json.loads(json.dumps(raw_manifest))
        composed = compiler.resolve_manifest_composition(pkg_dir, raw_manifest)
        self.assertEqual(composed["skills"][0]["description"],
                         "Entry description")
        self.assertEqual([a["name"] for a in composed["agents"]],
                         ["worker", "entry"])
        self.assertEqual(raw_manifest, before)

    def test_frontmatter_horizontal_rule_diagnostic(self):
        """Document starting with horizontal rule gives diagnostic."""
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.parse_frontmatter_document("---\nJust text\n---",
                                                "doc.md")
        self.assertIn("must be a mapping", str(ctx.exception))
        self.assertIn("horizontal rule", str(ctx.exception))

    def test_document_driven_agent_missing_frontmatter_rejected(self):
        """Document-driven agent without leading frontmatter is rejected."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents" / "bad.md").write_text("# Just Markdown",
                                                   encoding="utf-8")

        self._delegate_to(pkg_dir, "bad")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.verify_package(pkg_dir)
        self.assertIn("must start with '---' YAML frontmatter",
                      str(ctx.exception))

    def test_frontmatter_schema_rejects_missing_required_fields(self):
        """Agent markdown frontmatter missing 'name' or 'description' is rejected by schema."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)

        # Missing name
        (pkg_dir / "agents" / "missing-name.md").write_text(
            "---\ndescription: Has description but no name\ntools:"
            " [read]\n---\n# Content",
            encoding="utf-8",
        )
        self._delegate_to(pkg_dir, "missing-name")
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn(
            "invalid YAML frontmatter in agent document 'missing-name.md'",
            str(ctx.exception))
        self.assertIn("$/name: required property is missing",
                      str(ctx.exception))

        # Missing description
        (pkg_dir / "agents" / "missing-desc.md").write_text(
            "---\nname: missing-desc\ntools: [read]\n---\n# Content",
            encoding="utf-8",
        )
        self._delegate_to(pkg_dir, "missing-desc")
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn(
            "invalid YAML frontmatter in agent document 'missing-desc.md'",
            str(ctx.exception))
        self.assertIn("$/description: required property is missing",
                      str(ctx.exception))

    def test_frontmatter_schema_rejects_invalid_field_types(self):
        """Agent markdown frontmatter with invalid field types is rejected by schema."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        self._delegate_to(pkg_dir, "invalid-types")

        cases = [
            (
                "tools as string",
                "---\nname: invalid-types\ndescription: Desc\ntools:"
                " read\n---\n# C",
                "$/tools: expected array, got str",
            ),
            (
                "delegates as string",
                "---\nname: invalid-types\ndescription: Desc\ndelegates:"
                " worker\n---\n# C",
                "$/delegates: expected array, got str",
            ),
            (
                "scripts as string",
                "---\nname: invalid-types\ndescription: Desc\nscripts:"
                " run.py\n---\n# C",
                "$/scripts: expected array, got str",
            ),
            (
                "references as string",
                "---\nname: invalid-types\ndescription: Desc\nreferences:"
                " doc.md\n---\n# C",
                "$/references: expected array, got str",
            ),
            (
                "accepts as array",
                "---\nname: invalid-types\ndescription: Desc\naccepts:"
                " [schema.json]\n---\n# C",
                "$/accepts: expected string, got list",
            ),
            (
                "produces as array",
                "---\nname: invalid-types\ndescription: Desc\nproduces:"
                " [schema.json]\n---\n# C",
                "$/produces: expected string, got list",
            ),
        ]
        for desc, content, expected_err in cases:
            with self.subTest(case=desc):
                (pkg_dir / "agents" / "invalid-types.md").write_text(
                    content, encoding="utf-8")
                with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
                    compiler.load_package(pkg_dir)
                self.assertIn(expected_err, str(ctx.exception))

    def test_frontmatter_schema_rejects_unexpected_properties(self):
        """Agent markdown frontmatter rejects unknown keys due to additionalProperties: false."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents" / "extra-key.md").write_text(
            "---\nname: extra-key\ndescription: Has extra key\nunknown_property:"
            " true\n---\n# C",
            encoding="utf-8",
        )
        self._delegate_to(pkg_dir, "extra-key")
        with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
            compiler.load_package(pkg_dir)
        self.assertIn(
            "invalid YAML frontmatter in agent document 'extra-key.md'",
            str(ctx.exception))
        self.assertIn("$/unknown_property: additional property is not allowed",
                      str(ctx.exception))

    def test_extended_frontmatter_scripts_and_references_bundled(self):
        """Scripts and references declared in agent frontmatter are bundled without copy entries."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "references").mkdir(parents=True, exist_ok=True)

        (pkg_dir / "scripts" / "worker_tool.py").write_text(
            "#!/usr/bin/env python3\nprint('worker')", encoding="utf-8")
        (pkg_dir / "references" / "reference_manual.md").write_text(
            "# Reference Manual\nDetails here.", encoding="utf-8")

        agent_content = (
            "---\n"
            "name: bundled-worker\n"
            "description: Worker with scripts and references in frontmatter\n"
            "tools: [read]\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "delegates: []\n"
            "scripts:\n"
            "  - scripts/worker_tool.py\n"
            "references:\n"
            "  - references/reference_manual.md\n"
            "---\n"
            "# Worker\n"
            "Follow manual: {{reference:reference_manual.md}}")
        (pkg_dir / "agents" / "bundled-worker.md").write_text(agent_content,
                                                              encoding="utf-8")
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - bundled-worker\n"
            "---\n"
            "Delegate to {{agent:bundled-worker}}.",
            encoding="utf-8")

        raw = utils.load_json(pkg_dir / "package.release.json")
        raw["copy"] = ["schemas"]
        utils.write_json(pkg_dir / "package.release.json", raw)

        compiler.verify_package(pkg_dir)

        dist_out = self.temp_dir / "dist_bundled"
        compiler.translate(pkg_dir, "claude-code", dist_out)

        self.assertTrue((dist_out / "scripts" / "worker_tool.py").is_file(),
                        "Agent script was not bundled into dist")
        self.assertTrue(
            (dist_out / "references" / "reference_manual.md").is_file(),
            "Agent reference was not bundled into dist")
        self.assertEqual(
            (dist_out / "scripts" /
             "worker_tool.py").read_text(encoding="utf-8"),
            "#!/usr/bin/env python3\nprint('worker')",
        )
        self.assertIn(
            "Details here.",
            (dist_out / "references" /
             "reference_manual.md").read_text(encoding="utf-8"),
        )
        # The compute setup guide ships in every build, rendered for the
        # harness, whatever the skills declare; framework/ gets no raw copy.
        guide = (dist_out / "references" /
                 "compute-setup.md").read_text(encoding="utf-8")
        self.assertIn("AskUserQuestion every returned question", guide)
        self.assertNotIn("{{", guide)
        self.assertFalse(
            (dist_out / "framework" / "compute" / "compute-setup.md").exists())

    def test_canonical_tool_alias_suggestions_matrix(self):
        """Aliases for canonical tools are rejected with helpful suggestions."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - alias-tester\n"
            "---\n"
            "Call {{agent:alias-tester}}.",
            encoding="utf-8")

        alias_matrix = [
            ("view_file", "read"),
            ("write_to_file", "write"),
            ("edit_file", "edit"),
            ("replace_file_content", "edit"),
            ("multi_replace_file_content", "edit"),
            ("apply_patch", "write"),
            ("grep_search", "search"),
            ("grep", "search"),
            ("glob", "search"),
            ("find_by_name", "search"),
            ("run_command", "shell"),
            ("bash", "shell"),
            ("sh", "shell"),
        ]

        for alias, canonical in alias_matrix:
            with self.subTest(alias=alias, expected_canonical=canonical):
                agent_content = ("---\n"
                                 "name: alias-tester\n"
                                 "description: Tests tool alias rejection\n"
                                 f"tools:\n  - {alias}\n"
                                 "delegates: []\n"
                                 "---\n"
                                 "# Prompt")
                (pkg_dir / "agents" / "alias-tester.md").write_text(
                    agent_content, encoding="utf-8")
                with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
                    compiler.verify_package(pkg_dir)
                err_msg = str(ctx.exception)
                self.assertIn(f"references undeclared tool '{alias}'", err_msg)
                self.assertIn(f"did you mean canonical tool '{canonical}'?",
                              err_msg)

    def test_web_tools_support_and_rejection_without_aliases(self):
        """web_search and web_fetch are valid canonical tools, and non-canonical variants are rejected without aliases."""
        pkg_dir = self._create_valid_package()
        manifest_path = pkg_dir / "package.release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("native_tools", None)
        manifest_path.write_text(json.dumps(manifest, indent=2),
                                 encoding="utf-8")
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "tools:\n"
            "  - web_search\n"
            "  - web_fetch\n"
            "delegates:\n"
            "  - web-agent\n"
            "---\n"
            "Call {{agent:web-agent}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "web-agent.md").write_text(
            "---\n"
            "name: web-agent\n"
            "description: Web agent\n"
            "tools:\n"
            "  - web_search\n"
            "  - web_fetch\n"
            "delegates: []\n"
            "---\n"
            "# Prompt",
            encoding="utf-8")
        # Valid package passes verification
        compiler.verify_package(pkg_dir)

        # Non-canonical variants must be rejected without alias suggestions
        for non_canonical in ("search_web", "WebSearch", "websearch",
                              "read_url_content", "WebFetch"):
            with self.subTest(non_canonical=non_canonical):
                (pkg_dir / "agents" / "web-agent.md").write_text(
                    "---\n"
                    "name: web-agent\n"
                    "description: Web agent\n"
                    f"tools:\n  - {non_canonical}\n"
                    "delegates: []\n"
                    "---\n"
                    "# Prompt",
                    encoding="utf-8")
                with self.assertRaises(compiler.TpuAgentSuiteError) as ctx:
                    compiler.verify_package(pkg_dir)
                err_msg = str(ctx.exception)
                self.assertIn(f"references undeclared tool '{non_canonical}'",
                              err_msg)
                self.assertNotIn("did you mean canonical tool", err_msg)

    def test_skill_without_delegates_reaches_no_agents(self):
        """Nothing delegated means nothing discovered, and no twin either."""
        pkg_dir = self._create_valid_package()
        self._delegate_to(pkg_dir)
        manifest = compiler.load_package(pkg_dir)
        self.assertEqual(manifest["agents"], [])


class TestTomlEscaping(unittest.TestCase):
    """Codex agent configs are TOML; prompt text must survive the round trip.

    This covers a specific case that was seen in tpu-kernel-development, where `summarize-profile`
    prompt contains `1\\)`, carried over from the source package's markdown.
    """

    def setUp(self):
        super().setUp()
        if tomllib is None:
            self.skipTest("tomllib is not available on Python < 3.11")

    def _round_trip(self, value: str) -> str:
        document = f'x = """\n{compiler.toml_multiline_string(value)}\n"""\n'
        return tomllib.loads(document)["x"].strip("\n")

    def test_backslash_survives_multiline(self):
        self.assertEqual(self._round_trip("three parts: 1\\) A summary"),
                         "three parts: 1\\) A summary")

    def test_windows_path_survives_multiline(self):
        self.assertEqual(self._round_trip(r"C:\Users\tmp\n"),
                         r"C:\Users\tmp\n")

    def test_triple_quote_does_not_terminate_multiline(self):
        self.assertEqual(self._round_trip('say """hi""" now'),
                         'say """hi""" now')

    def test_basic_string_escapes_quotes_and_backslashes(self):
        document = f'x = "{compiler.toml_basic_string(chr(92) + chr(34))}"\n'
        self.assertEqual(tomllib.loads(document)["x"], "\\\"")

    def test_basic_string_escapes_newlines(self):
        val = "a\nb"
        document = f'x = "{compiler.toml_basic_string(val)}"\n'
        self.assertEqual(tomllib.loads(document)["x"], "a\nb")


class TestContractSchemaTemplates(unittest.TestCase):
    """Test schema examples and contract prompt generation."""

    def test_delegation_and_response_contract_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir)
            schemas_dir = pkg_dir / "schemas"
            schemas_dir.mkdir(parents=True)
            (schemas_dir / "req.json").write_text(
                json.dumps({
                    "type":
                    "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string"
                        },
                        "run_id": {
                            "type": "string"
                        },
                    },
                    "examples": [{
                        "query": "test query",
                        "run_id": "20260914T120000Z-000000000000",
                    }],
                }),
                encoding="utf-8",
            )
            (schemas_dir / "res.json").write_text(
                json.dumps({
                    "type": "object",
                    "required": ["status"],
                    "properties": {
                        "status": {
                            "enum": ["success", "failed"]
                        },
                    },
                    "examples": [{
                        "status": "success",
                    }],
                }),
                encoding="utf-8",
            )
            (schemas_dir / "no_example.json").write_text(
                json.dumps({
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "string"
                        }
                    },
                }),
                encoding="utf-8",
            )

            manifest = {
                "name":
                "pkg",
                "agents": [
                    {
                        "name": "sub-a",
                        "accepts": "schemas/req.json"
                    },
                    {
                        "name": "sub-b",
                        "accepts": "schemas/req.json"
                    },
                    {
                        "name": "sub-c",
                        "accepts": "schemas/no_example.json"
                    },
                ],
            }
            del_section = compiler.delegation_contract_section(
                manifest, ["sub-a", "sub-b", "sub-c"], package_dir=pkg_dir)
            self.assertIn("### Delegation request payloads", del_section)
            self.assertIn("- Targets: `sub-a`, `sub-b` (`schemas/req.json`):",
                          del_section)
            self.assertIn('"query": "test query"', del_section)
            self.assertIn('"run_id": "20260914T120000Z-000000000000"',
                          del_section)
            # Schema without examples lists target and schema path without a json codeblock
            self.assertIn("- Targets: `sub-c` (`schemas/no_example.json`):",
                          del_section)

            resp_section = compiler.response_deliverable_section(
                {
                    "name": "worker",
                    "produces": "schemas/res.json"
                },
                package_dir=pkg_dir,
            )
            self.assertIn("### Response deliverable format", resp_section)
            self.assertIn("`schemas/res.json`", resp_section)
            self.assertIn('"status": "success"', resp_section)


class TestHarnessTokens(unittest.TestCase):
    """Tokens that depend on the harness being built, not on the manifest."""

    def test_harness_renders_to_its_name(self):
        for harness in ("claude-code", "codex", "antigravity", "cursor"):
            with self.subTest(harness=harness):
                self.assertEqual(
                    compiler.render_harness_tokens("on {{harness}}", harness),
                    f"on {harness}")

    def test_text_without_token_is_untouched(self):
        self.assertEqual(compiler.render_harness_tokens("plain", "emacs"),
                         "plain")


class TestCopyAssetExclusions(unittest.TestCase):
    """copy_asset never ships wikify working state, agent-skill installs, or raw/code."""

    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "knowledgebase"
        for rel in (
                "wiki/codebases/torch_tpu/overview.md",
                "wiki/code/page.md",  # a "code" dir outside raw/ still ships
                "raw/docs/guide.md",
                "raw/code/.keep",
                ".cache/state.json",
                ".wikify/plan.md",
                ".agents/skills/wikify-ingest-repo/SKILL.md",
                ".claude/skills/wikify-ingest-repo/SKILL.md",
        ):
            path = self.src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
        # A pinned checkout is a symlink to a tree outside the asset (private code);
        # copytree follows symlinks, so the path exclusion is what keeps it out.
        private = self.tmp / "torch_tpu"
        (private / "torch_tpu").mkdir(parents=True)
        (private / "torch_tpu" / "secret.py").write_text("x", encoding="utf-8")
        (self.src / "raw" / "code" / "torch_tpu").symlink_to(private)

    def test_ships_wiki_and_docs_but_never_raw_code_or_tool_state(self):
        dest = self.tmp / "dist" / "knowledgebase"
        compiler.copy_asset(self.src, dest)
        for rel in ("wiki/codebases/torch_tpu/overview.md",
                    "wiki/code/page.md", "raw/docs/guide.md"):
            self.assertTrue((dest / rel).is_file(), f"{rel} must ship")
        for rel in ("raw/code", ".cache", ".wikify", ".agents", ".claude"):
            self.assertFalse((dest / rel).exists(), f"{rel} must not ship")
        self.assertEqual([p for p in dest.rglob("*") if p.name == "secret.py"],
                         [])

    def test_dist_excluded_covers_raw_code_and_tool_state_by_path(self):
        for rel, expected in (
            ("knowledgebase/raw/code/torch_tpu/requirements/requirements_torch.txt",
             True),
            ("knowledgebase/.cache/scip/requirements.txt", True),
            ("knowledgebase/wiki/code/page.md", False),
            ("knowledgebase/raw/docs/requirements-notes.txt", False),
            ("requirements.txt", False),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(compiler.dist_excluded(Path(rel)), expected)

    def test_dist_ignore_matches_paths_relative_to_the_asset_root(self):
        ignore = compiler.dist_ignore(self.src)
        self.assertEqual(ignore(str(self.src / "raw"), ["code", "docs"]),
                         {"code"})
        self.assertEqual(ignore(str(self.src / "wiki"), ["code", "codebases"]),
                         set())
        self.assertEqual(ignore(str(self.src), [".cache", "wiki", ".git"]),
                         {".cache", ".git"})


class TestStandaloneArchives(unittest.TestCase):
    """Test standalone distribution bundling and archive creation."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_archives_packages_distribution(self):
        dist_dir = self.tmp / "dist"
        claude_dist = dist_dir / "claude-code" / "test-pkg"
        claude_dist.mkdir(parents=True)
        utils.write_json(
            claude_dist / "tpu-agent-suite-build.json",
            {
                "source": "test-pkg",
                "version": "1.2.3",
                "harness": "claude-code",
            },
        )
        (claude_dist / "test.txt").write_text("hello", encoding="utf-8")

        out_dir = self.tmp / "archives"
        archives = compiler.create_archives(dist_dir,
                                            out_dir,
                                            harnesses=["claude-code"])
        self.assertEqual(len(archives), 1)
        archive_path = archives[0]
        self.assertEqual(archive_path.name,
                         "test-pkg-claude-code-v1.2.3.tar.gz")
        self.assertTrue(archive_path.is_file())

        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
            self.assertIn("test-pkg-claude-code-v1.2.3", names)
            self.assertIn("test-pkg-claude-code-v1.2.3/test.txt", names)

    def test_create_archives_empty_raises_error(self):
        dist_dir = self.tmp / "dist"
        dist_dir.mkdir(parents=True)
        out_dir = self.tmp / "archives"
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            compiler.create_archives(dist_dir,
                                     out_dir,
                                     harnesses=["claude-code"])
        self.assertIn("No distribution packages found", str(ctx.exception))

    def test_generated_contract_states_the_agents_own_status_vocabulary(self):
        """A generated prompt must not contradict the schema it points at.

        The status line names exactly the statuses the schema declares.
        """
        schema = {
            "properties": {
                "status": {
                    "enum": ["optimized", "invalid_input", "failed"]
                },
            },
            "status_requires": {
                "optimized": ["best_kernel_artifact", "artifacts"]
            },
        }
        lines = compiler.result_contract_lines(schema)
        joined = "\n".join(lines)
        self.assertIn("`optimized`, `invalid_input`, `failed`", joined)
        self.assertIn("An `optimized` result must carry", joined)
        self.assertIn("`best_kernel_artifact`, `artifacts`", joined)
        self.assertNotIn("completed", joined)
        self.assertNotIn("needs_input", joined)

    def test_status_requires_alone_still_names_the_statuses(self):
        """A schema with rules but no enum still names its statuses.

        The statuses keyed in `status_requires` form the status line, so the
        agent is told to return `optimized` and what `optimized` owes.
        """
        joined = "\n".join(
            compiler.result_contract_lines({
                "properties": {
                    "status": {
                        "type": "string"
                    }
                },
                "status_requires": {
                    "optimized": ["best_kernel_artifact"]
                },
            }))
        self.assertIn("`optimized`", joined)
        self.assertNotIn("completed", joined)

    def test_no_contract_states_a_rule_for_a_status_the_schema_cannot_emit(
            self):
        """The artifact line is stated only where the runtime enforces it.

        `check_completed_has_artifact` runs only for a schema that declares
        `artifacts`, and fires only on the literal status `completed`. Stating
        the rule to any other schema promises enforcement never happens.
        """
        own_vocabulary = compiler.result_contract_lines({
            "properties": {
                "status": {
                    "enum": ["cached", "crawled", "uncited", "failed"]
                }
            },
        })
        joined = "\n".join(own_vocabulary)
        self.assertIn("`cached`", joined)
        self.assertNotIn("completed", joined)

        # A schema that names `completed` and declares `artifacts` still gets
        # the rule, because that is exactly when the runtime enforces it.
        keeps_it = "\n".join(
            compiler.result_contract_lines({
                "properties": {
                    "status": {
                        "enum": ["completed", "failed"]
                    },
                    "artifacts": {
                        "type": "array"
                    },
                },
            }))
        self.assertIn("A `completed` result must carry", keeps_it)

        # Naming `completed` without declaring `artifacts` does not: the
        # runtime runs `check_completed_has_artifact` only for schemas that do.
        no_artifacts = "\n".join(
            compiler.result_contract_lines({
                "properties": {
                    "status": {
                        "enum": ["completed", "failed"]
                    }
                },
            }))
        self.assertIn("`completed`", no_artifacts)
        self.assertNotIn("at least one artifact", no_artifacts)

    def test_schema_naming_no_statuses_gets_no_result_lines(self):
        """No statuses in the schema means no status line to state.

        Inventing `completed`/`needs_input` for a schema that names neither
        would contradict the schema the same prompt says to conform to.
        """
        for schema in (None, {}, {
                "properties": {
                    "status": {
                        "type": "string"
                    }
                }
        }):
            with self.subTest(schema=schema):
                self.assertEqual(compiler.result_contract_lines(schema), [])


if __name__ == "__main__":
    unittest.main()
