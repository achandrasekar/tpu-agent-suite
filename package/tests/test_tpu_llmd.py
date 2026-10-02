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
"""Package-level tests for tpu-llmd: shipped scripts, schemas and wiring.

The script tests live beside the scripts (`tpu_llmd_scripts_test.py`) so
they travel with them; this module loads them into CI discovery the way
`test_tpu_performance.py` does, and adds the checks that only make sense
from the package root: every llm-d schema's examples validate against it,
the arm taxonomy stays one list across framework and package, the
environment questions name keys the environment schema has, and the agents
are reachable from the entrypoint.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_NEXUS_ROOT = _PACKAGE_ROOT.parent
_LLMD_SCRIPTS = _PACKAGE_ROOT / "scripts" / "tpu-llmd"

if str(_NEXUS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEXUS_ROOT))

from framework import compiler, frontmatter, study, utils  # noqa: E402


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_scripts_test = _load_script_module(
    "tpu_llmd_scripts_test", _LLMD_SCRIPTS / "tpu_llmd_scripts_test.py")


class TestTpuLlmdRenderProtocol(_scripts_test.RenderProtocolTest):
    """Wire the render_protocol tests into package discovery."""


class TestTpuLlmdSeedArms(_scripts_test.SeedArmsTest):
    """Wire the seed_arms tests into package discovery."""


class TestTpuLlmdEvaluateArm(_scripts_test.EvaluateArmTest):
    """Wire the evaluate_llmd_arm tests into package discovery."""


LLMD_SCHEMAS = (
    "llmd-request.json",
    "llmd-result.json",
    "llmd-workload-spec.json",
    "llmd-config.json",
    "serving-recipe.json",
    "llmd-platform-request.json",
    "llmd-platform-result.json",
    "llmd-arm-request.json",
    "llmd-bench-report.json",
    "engine-capabilities.json",
)


def _schema(name: str) -> dict:
    return json.loads(
        (_PACKAGE_ROOT / "schemas" / name).read_text(encoding="utf-8"))


class TestTpuLlmdSchemas(unittest.TestCase):

    def test_schemas_are_registered_and_well_formed(self):
        manifest = compiler.load_package(_PACKAGE_ROOT)
        registered = set(manifest["schemas"])
        for name in LLMD_SCHEMAS:
            with self.subTest(schema=name):
                self.assertIn(f"schemas/{name}", registered)
                self.assertEqual(utils.validate_schema_definition(_schema(name)),
                                 [])

    # The two documents the runtime writes (`llmd-bench-report`,
    # `engine-capabilities`) are validated from real `run_bench`/`run_probe`
    # output in framework/tests/test_deploy.py, which is a stronger check
    # than a hand-written example that could drift from the writer.
    RUNTIME_WRITTEN = {"llmd-bench-report.json", "engine-capabilities.json"}

    def test_every_example_validates(self):
        for name in LLMD_SCHEMAS:
            if name in self.RUNTIME_WRITTEN:
                continue
            schema = _schema(name)
            examples = schema.get("examples", [])
            with self.subTest(schema=name):
                self.assertTrue(examples, f"{name} ships no example")
                for example in examples:
                    self.assertEqual(
                        utils.validate_instance(example, schema), [])

    def test_request_schemas_carry_prompt_and_results_carry_status(self):
        for name in ("llmd-request.json", "llmd-platform-request.json",
                     "llmd-arm-request.json"):
            self.assertIn("prompt", _schema(name)["properties"])
        for name in ("llmd-result.json", "llmd-platform-result.json"):
            self.assertIn("status", _schema(name)["required"])

    def test_arm_kinds_are_one_list_across_framework_and_package(self):
        round_kinds = tuple(
            json.loads((_NEXUS_ROOT / "framework" / "schemas" /
                        "study-round.json").read_text(encoding="utf-8"))
            ["properties"]["arms"]["items"]["properties"]["kind"]["enum"])
        llmd_arm_kinds = tuple(
            _schema("llmd-arm-request.json")["properties"]["kind"]["enum"])
        self.assertEqual(study.BOTTLENECK_KINDS, round_kinds)
        self.assertEqual(study.BOTTLENECK_KINDS, llmd_arm_kinds)
        for fleet_kind in ("fleet-topology", "kv-reuse", "routing-policy",
                           "engine-fleet-knob", "platform-capacity"):
            self.assertIn(fleet_kind, study.BOTTLENECK_KINDS)

    def test_arm_result_carries_llmd_extensions(self):
        props = _schema("arm-result.json")["properties"]
        for key in ("config_path", "sweep_ledger", "recommendations",
                    "chips_total"):
            self.assertIn(key, props)

    def test_seeded_arm_kinds_are_in_the_request_enum(self):
        seed_arms = _load_script_module("seed_arms",
                                        _LLMD_SCRIPTS / "seed_arms.py")
        allowed = set(
            _schema("llmd-arm-request.json")["properties"]["arm_kind"]["enum"])
        self.assertTrue(set(seed_arms.COARSE_KIND) <= allowed,
                        set(seed_arms.COARSE_KIND) - allowed)
        self.assertTrue(set(seed_arms.SWEEP_KNOBS) <= allowed)


class TestTpuLlmdEnvironment(unittest.TestCase):

    def test_llmd_questions_name_keys_the_schema_has(self):
        schema = json.loads((_PACKAGE_ROOT / "schemas" /
                             "environment.json").read_text(encoding="utf-8"))
        llmd_props = schema["properties"]["llmd"]["properties"]
        questions = json.loads(
            (_PACKAGE_ROOT / "environment" /
             "questions.json").read_text(encoding="utf-8"))["questions"]
        llmd_questions = [q for q in questions
                          if q["name"].startswith("llmd_")]
        self.assertTrue(llmd_questions)
        for question in llmd_questions:
            with self.subTest(question=question["name"]):
                path = question["path"].split(".")
                self.assertEqual(path[0], "llmd")
                self.assertIn(path[1], llmd_props)
                # Only asked on a GKE backend: nothing else can host llm-d.
                when = question.get("when") or {}
                self.assertIn("gke_job", when.get("backend", []))


class TestTpuLlmdWiring(unittest.TestCase):

    def _frontmatter(self, rel: str) -> dict:
        text = (_PACKAGE_ROOT / rel).read_text(encoding="utf-8")
        return frontmatter.parse_frontmatter_document(text)[0]

    def test_agents_json_lists_the_three_agents(self):
        agents = json.loads((_PACKAGE_ROOT / "tpu-llmd" /
                             "agents.json").read_text(encoding="utf-8"))
        self.assertEqual(agents["agents"], [
            "instructions/tpu-llmd.md",
            "instructions/llmd-platform.md",
            "instructions/llmd-arm-executor.md",
        ])
        manifest = json.loads(
            (_PACKAGE_ROOT / "package.release.json").read_text(encoding="utf-8"))
        self.assertIn("tpu-llmd/agents.json", manifest["include"])

    def test_orchestrator_delegates_only_to_its_specialists(self):
        fm = self._frontmatter("agents/tpu-llmd.md")
        self.assertEqual(fm["accepts"], "schemas/llmd-request.json")
        self.assertEqual(fm["produces"], "schemas/llmd-result.json")
        self.assertEqual(set(fm["delegates"]),
                         {"llmd-platform", "llmd-arm-executor", "tpu-knowledge-base"})
        # Recommend-only toward the engine domains: never a delegation.
        self.assertNotIn("tpu-performance-optimization", fm["delegates"])
        self.assertNotIn("tpu-model-onboarding", fm["delegates"])

    def test_specialists_are_leaves(self):
        for rel, accepts in (
            ("agents/llmd-platform.md",
             "schemas/llmd-platform-request.json"),
            ("agents/llmd-arm-executor.md",
             "schemas/llmd-arm-request.json"),
        ):
            fm = self._frontmatter(rel)
            self.assertEqual(fm["accepts"], accepts)
            self.assertEqual(fm["delegates"], ["tpu-knowledge-base"])
            self.assertNotIn("ask", fm["tools"])

    def test_entrypoint_routes_to_tpu_llmd(self):
        rel = "skills/tpu-agent/SKILL.md"
        fm = self._frontmatter(rel)
        self.assertIn("tpu-llmd", fm["delegates"])
        body = (_PACKAGE_ROOT / rel).read_text(encoding="utf-8")
        self.assertIn("{{agent:tpu-llmd}}", body)
        self.assertIn("promotion_approved", body)

    def test_prompts_never_deploy_from_the_orchestrator(self):
        body = (_PACKAGE_ROOT / "agents" /
                "tpu-llmd.md").read_text(encoding="utf-8")
        for verb in ("deploy apply", "deploy render", "deploy bench",
                     "deploy teardown"):
            self.assertNotIn(f"{{{{runtime}}}} {verb}", body)
        executor = (_PACKAGE_ROOT / "agents" /
                    "llmd-arm-executor.md").read_text(encoding="utf-8")
        self.assertIn("{{runtime}} deploy teardown", executor)


if __name__ == "__main__":
    unittest.main()
