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
"""Tests for shipping the export-trajectory skill: emission, install, config.

Covers the packaging half of trajectory export -- that a manifest-declared
skill is rendered, installed on every harness, and kept consistent with its
instruction document.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from framework import compiler, lint, runtime, utils

Path = pathlib.Path
REPO_PACKAGE = Path(__file__).resolve().parents[2] / "package"


class HarnessTokenTest(unittest.TestCase):
    """`{{harness}}` names the harness a skill is being compiled for."""

    def test_renders_per_harness(self) -> None:
        """Each harness gets its own name baked into the runtime call."""
        for harness in ("claude-code", "codex", "antigravity", "cursor"):
            with self.subTest(harness=harness):
                self.assertEqual(
                    compiler.render_harness_tokens(
                        "push --harness {{harness}}", harness),
                    f"push --harness {harness}")

    def test_resolves_without_a_skills_dir(self) -> None:
        """The token works alone, even for a harness with no skills root."""
        self.assertEqual(
            compiler.render_harness_tokens("{{harness}}", "emacs"), "emacs")

    def test_is_a_known_token(self) -> None:
        """Declared as harness-scoped, so verification does not reject it."""
        self.assertIn("{{harness}}", compiler.HARNESS_SCOPED_TOKENS)


class AuxiliarySkillDocumentTest(unittest.TestCase):
    """Rendering a manifest-declared skill into SKILL.md."""

    def setUp(self) -> None:
        """A one-skill package whose instruction file each test writes."""
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = {
            "name":
            "p",
            "version":
            "1.0.0",
            "description":
            "package description",
            "agents": [],
            "skills": [{
                "name": "p-export",
                "description": "save the trajectory",
                "instructions": "instructions/p-trajectory.md",
            }],
        }
        (self.tmp / "instructions").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _render(self, body: str, harness: str = "claude-code") -> str:
        """Write the instruction body and render it for one harness."""
        (self.tmp / "instructions" / "p-trajectory.md").write_text(
            body, encoding="utf-8")
        return compiler.skill_document(self.tmp, self.manifest,
                                       {"harness": harness},
                                       self.manifest["skills"][0])

    def test_frontmatter_comes_from_the_manifest(self) -> None:
        """The harness lists the skill by its manifest name and description."""
        document = self._render("# Body\n")
        self.assertTrue(document.startswith("---\nname: p-export\n"))
        self.assertIn('description: "save the trajectory"', document)

    def test_source_frontmatter_is_stripped(self) -> None:
        """An authored contract block never reaches the rendered skill twice."""
        document = self._render(
            "---\nname: p-export\ndescription: x\n---\n\n# Body\n")
        self.assertEqual(document.count("name: p-export"), 1)
        self.assertIn("# Body", document)

    def test_runtime_and_harness_tokens_resolve(self) -> None:
        """A skill body names the runtime and its harness without hardcoding."""
        document = self._render(
            "{{runtime}} trajectory export --harness {{harness}} --package {{package_tag}}\n",
            harness="antigravity")
        self.assertIn("framework/runtime.py trajectory export", document)
        self.assertIn("--harness antigravity", document)
        self.assertIn("--package p@1.0.0", document)
        self.assertNotIn("{{", document)

    def test_missing_instructions_file_is_an_error(self) -> None:
        """A declared skill with no document fails the build, not the install."""
        with self.assertRaises(compiler.TpuAgentSuiteError):
            compiler.skill_document(
                self.tmp, self.manifest, {"harness": "claude-code"}, {
                    "name": "absent",
                    "description": "d",
                    "instructions": "instructions/absent.md",
                })


class InstallSpecSkillCopiesTest(unittest.TestCase):
    """Every declared skill gets one copy rule per harness."""

    HARNESSES = ("claude-code", "codex", "antigravity", "cursor")

    def test_one_skill_gets_one_copy_rule(self) -> None:
        """A package releasing one skill installs exactly that one."""
        specs = compiler.harness_install_specs(["entry"], "pkg")
        for harness in self.HARNESSES:
            copies = [c["source"] for c in specs[harness]["copies"]]
            skill_copies = [c for c in copies if "skills/" in c]
            self.assertEqual(len(skill_copies), 1, harness)
            self.assertTrue(skill_copies[0].endswith("skills/entry"), harness)

    def test_each_declared_skill_gets_a_copy_rule(self) -> None:
        """A second skill installs beside the first on every harness."""
        specs = compiler.harness_install_specs(["entry", "entry-export"],
                                               "pkg")
        for harness in self.HARNESSES:
            with self.subTest(harness=harness):
                copies = specs[harness]["copies"]
                sources = [
                    c["source"] for c in copies if "skills/" in c["source"]
                ]
                self.assertEqual(len(sources), 2)
                self.assertTrue(
                    any(s.endswith("skills/entry") for s in sources))
                self.assertTrue(
                    any(s.endswith("skills/entry-export") for s in sources))

    def test_destinations_track_the_skill_name(self) -> None:
        """Each rule names its own destination, not just its source."""
        specs = compiler.harness_install_specs(["entry", "entry-export"],
                                               "pkg")
        destinations = [
            c["destination"] for c in specs["cursor"]["copies"]
            if "skills/" in c["destination"]
        ]
        self.assertIn(".cursor/skills/entry", destinations)
        self.assertIn(".cursor/skills/entry-export", destinations)

    def test_non_skill_copies_are_untouched(self) -> None:
        """Adding a skill adds a skill rule, never an agents or hooks rule."""
        specs = compiler.harness_install_specs(["entry", "entry-export"],
                                               "pkg")
        agents = [
            c for c in specs["codex"]["copies"]
            if c["source"].endswith("agents")
        ]
        self.assertEqual(len(agents), 1)


class SkillFrontmatterLintTest(unittest.TestCase):
    """Manifest and skill document may not drift apart."""

    def setUp(self) -> None:
        """A one-skill package for exercising the lint rule."""
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "instructions").mkdir(parents=True)
        self.manifest = {
            "name":
            "p",
            "agents": [],
            "skills": [{
                "name": "p-export",
                "description": "save the trajectory",
                "instructions": "instructions/p-trajectory.md",
            }],
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check(self, body: str) -> list:
        """Run the frontmatter check over a one-skill package."""
        (self.tmp / "instructions" / "p-trajectory.md").write_text(
            body, encoding="utf-8")
        return lint.check_frontmatter_contracts(self.tmp, self.manifest)

    def test_matching_frontmatter_passes(self) -> None:
        """Agreement between manifest and document is clean."""
        findings = self._check(
            "---\nname: p-export\ndescription: save the trajectory\n---\n\nBody\n"
        )
        self.assertEqual(findings, [])

    def test_name_mismatch_is_reported(self) -> None:
        """A renamed skill must not advertise one name and describe another."""
        findings = self._check(
            "---\nname: other\ndescription: save the trajectory\n---\n\nBody\n"
        )
        self.assertTrue(
            any("does not match skill name" in f.detail for f in findings),
            findings)

    def test_missing_frontmatter_is_reported(self) -> None:
        """A skill document follows the same contract as every instruction."""
        findings = self._check("Body with no contract\n")
        self.assertTrue(any("missing leading" in f.detail for f in findings))

    def test_missing_instructions_file_is_reported(self) -> None:
        """A skill pointing at no document fails lint, not only the build."""
        findings = lint.check_frontmatter_contracts(self.tmp, self.manifest)
        self.assertTrue(any("does not exist" in f.detail for f in findings),
                        findings)

    def test_an_invalid_tool_is_reported(self) -> None:
        """A skill's tools are held to the same list as an agent's."""
        findings = self._check(
            "---\nname: p-export\ndescription: save the trajectory\n"
            "tools:\n  - read\n  - bash\n---\n\nBody\n")
        self.assertTrue(
            any("invalid tool grants: ['bash']" in f.detail for f in findings),
            findings)

    def test_missing_description_is_reported(self) -> None:
        """Both required fields are enforced."""
        findings = self._check("---\nname: p-export\n---\n\nBody\n")
        self.assertTrue(any("description" in f.detail for f in findings))


class ShippedPackageTest(unittest.TestCase):
    """The real package declares the export-trajectory skill coherently."""

    def setUp(self) -> None:
        self.manifest = json.loads(
            (REPO_PACKAGE /
             "package.release.json").read_text(encoding="utf-8"))

    @staticmethod
    def _skills(manifest) -> dict[str, str]:
        """Skill name -> instructions path, for both manifest spellings."""
        out = {}
        for entry in manifest.get("skills", []):
            if isinstance(entry, str):
                out[entry] = f"skills/{entry}/SKILL.md"
            else:
                out[entry["name"]] = entry["instructions"]
        return out

    def test_skill_is_declared_and_present(self) -> None:
        """The manifest names it and the document exists."""
        skills = self._skills(self.manifest)
        self.assertIn("tpu-agent-export-trajectory", skills)
        path = REPO_PACKAGE / skills["tpu-agent-export-trajectory"]
        self.assertTrue(path.is_file())

    def test_skill_body_drives_the_runtime_not_the_model(self) -> None:
        """The body is one runtime call, with tokens rather than literals."""
        skills = self._skills(self.manifest)
        body = (REPO_PACKAGE /
                skills["tpu-agent-export-trajectory"]).read_text()
        self.assertIn("{{runtime}} trajectory export", body)
        self.assertIn("--harness {{harness}}", body)
        # The consent flag must never be presented as something to add freely.
        self.assertIn("Never add `--all` on your own initiative", body)

    def test_sharing_is_declined_by_default(self) -> None:
        """One gate gets the whole feature, and it defaults to off."""
        questions = json.loads((REPO_PACKAGE / "environment" /
                                "questions.json").read_text())["questions"]
        by_name = {q["name"]: q for q in questions}
        self.assertIn("trajectory_enabled", by_name)
        self.assertIs(by_name["trajectory_enabled"]["default"], False)
        # Every other trajectory question hangs off that gate, directly or
        # through the sink, so declining it ends the flow.
        for name, question in by_name.items():
            if name.startswith("trajectory") and name != "trajectory_enabled":
                with self.subTest(question=name):
                    self.assertIn("when", question)

    def test_environment_schema_accepts_a_declined_bucket(self) -> None:
        """An empty bucket must validate, or declining breaks the install."""
        schema = json.loads(
            (REPO_PACKAGE / "schemas" / "environment.json").read_text())
        params = schema["properties"]["trajectory"]["properties"]["params"]
        self.assertEqual(params["properties"]["bucket"]["type"], "string")
        self.assertNotIn("pattern", params["properties"]["bucket"])
        self.assertNotIn("minLength", params["properties"]["bucket"])


class UserIdResolutionTest(unittest.TestCase):
    """Which identifier uploaded trajectories are filed under."""

    def test_configured_answer_wins(self) -> None:
        """A configured user id is never second-guessed."""
        self.assertEqual(runtime.resolve_trajectory_user_id("me@example.com"),
                         "me@example.com")

    def test_blank_falls_back(self) -> None:
        """Without an answer some identifier is still produced."""
        resolved = runtime.resolve_trajectory_user_id("")
        self.assertTrue(resolved)
        self.assertNotEqual(resolved, "(unset)")


class ShippedFrameworkTest(unittest.TestCase):
    """framework/ as a build ships it imports and runs with nothing else.

    In the repository the tests also see the repository root and the build
    tooling; a copy made the way the compiler makes it, run on its own,
    shows the shipped runtime needs neither.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.package = self.tmp / "pkg"
        compiler.bundle_framework(self.package / "framework")
        (self.package / "tpu-agent-suite-build.json").write_text(
            json.dumps({
                "version": utils.TPU_AGENT_SUITE_VERSION,
                "source": "tpu-agent-suite"
            }),
            encoding="utf-8",
        )
        self.elsewhere = self.tmp / "elsewhere"
        self.elsewhere.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run([sys.executable, *argv],
                              cwd=cwd,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=60)

    def test_the_build_tooling_is_left_out(self) -> None:
        for name in ("compiler.py", "lint.py", "frontmatter.py", "tests",
                     "schemas/package-manifest.schema.json",
                     "schemas/skill-frontmatter.schema.json"):
            self.assertFalse((self.package / "framework" / name).exists(),
                             name)

    def test_the_package_imports(self) -> None:
        """`from framework import trajectory` works from the package root."""
        result = self._run([
            "-c", "from framework import trajectory; "
            "print(trajectory.ARTIFACT_BYTE_CAP)"
        ], self.package)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10485760", result.stdout)

    def test_gcs_helper_resolves(self) -> None:
        """The lazy gcs_staging import finds its sibling package."""
        result = self._run([
            "-c", "from framework.trajectory import bundle as b; "
            "print(b._clean_bucket('gs://x/'))"
        ], self.package)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "x")

    def test_runtime_runs_as_a_script_from_anywhere(self) -> None:
        """The hooks run framework/runtime.py by path, from any directory."""
        result = self._run([
            str(self.package / "framework" / "runtime.py"), "trajectory",
            "export", "--harness", "codex"
        ], self.elsewhere)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not supported", result.stderr)


if __name__ == "__main__":
    unittest.main()

    def test_deploy_imports(self) -> None:
        """`from framework import deploy` works from the package root."""
        result = self._run([
            "-c", "from framework import deploy; "
            "print(deploy.PRODUCTION_LABEL)"
        ], self.package)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("production", result.stdout)
