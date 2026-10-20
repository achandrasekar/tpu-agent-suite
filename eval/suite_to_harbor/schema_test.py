# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for schema."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import pydantic
import schema
from absl.testing import absltest, parameterized

Path = pathlib.Path


class SchemaTest(parameterized.TestCase):

    def test_default_evaluation_config(self) -> None:
        eval_cfg = schema.EvaluationConfig(
            cases=[schema.TestCase(name="c1", prompt="p1")])
        self.assertEqual(eval_cfg.judge_model, "gemini/gemini-3.6-flash")
        self.assertEmpty(eval_cfg.script_verifiers)
        self.assertEqual(eval_cfg.cases[0].timeout_sec, 300.0)

    @parameterized.named_parameters(
        (
            "eval_suite_extra_field",
            schema.EvalSuite,
            {
                "evaluation": {
                    "cases": [{
                        "name": "c1",
                        "prompt": "p1"
                    }]
                },
                "unknown_pillar": {},
            },
        ),
        (
            "test_case_extra_field",
            schema.TestCase,
            {
                "name": "c1",
                "prompt": "p1",
                "forbidden_field": 123,
            },
        ),
    )
    def test_extra_forbidden_on_models(self,
                                       model_cls: type[pydantic.BaseModel],
                                       data: dict[str, Any]) -> None:
        with self.assertRaises(pydantic.ValidationError):
            model_cls.model_validate(data)

    def test_expectation_criterion_alias_choices(self) -> None:
        crit1 = schema.ExpectationCriterion.model_validate({
            "criterion": "Criterion 1",
            "weight": 2.0,
            "type": "binary",
        })
        self.assertEqual(crit1.criterion, "Criterion 1")
        self.assertEqual(crit1.weight, 2.0)
        crit2 = schema.ExpectationCriterion.model_validate({
            "description": "Description alias",
            "weight": 1.5,
            "type": "numeric",
        })
        self.assertEqual(crit2.criterion, "Description alias")
        self.assertEqual(crit2.weight, 1.5)
        self.assertEqual(crit2.type, "numeric")
        crit3 = schema.ExpectationCriterion.model_validate({
            "criterion": "Likert criterion",
            "type": "likert",
        })
        self.assertEqual(crit3.type, "likert")

    @parameterized.named_parameters(
        ("empty_suite", schema.EvalSuite, {}),
        ("empty_evaluation", schema.EvaluationConfig, {}),
        ("missing_prompt_case", schema.TestCase, {
            "name": "c1"
        }),
        ("whitespace_prompt_case", schema.TestCase, {
            "name": "c1",
            "prompt": "   ",
        }),
    )
    def test_missing_required_fields_raise_error(self, model_cls: type[
        pydantic.BaseModel], data: dict[str, Any]) -> None:
        with self.assertRaises(pydantic.ValidationError):
            model_cls.model_validate(data)

    @parameterized.named_parameters(
        (
            "negative_weight_script",
            schema.ScriptCheck,
            {
                "script": "s.py",
                "weight": -0.5
            },
        ),
        (
            "negative_weight_expectation",
            schema.ExpectationCriterion,
            {
                "criterion": "c",
                "weight": -1.0
            },
        ),
        (
            "negative_timeout_case",
            schema.TestCase,
            {
                "name": "c",
                "prompt": "p",
                "timeout_sec": -10.0
            },
        ),
        ("zero_cpus", schema.EnvironmentConfig, {
            "cpus": 0
        }),
        ("none_cpus", schema.EnvironmentConfig, {
            "cpus": None
        }),
    )
    def test_negative_boundary_values_raise_error(
            self, model_cls: type[pydantic.BaseModel],
            data: dict[str, Any]) -> None:
        with self.assertRaises(pydantic.ValidationError):
            model_cls.model_validate(data)

    def test_from_yaml_string(self) -> None:
        yaml_content = """
name: test_suite
environment:
  base_image: python:3.12-slim
  pip_packages:
    - pytest
  workspace_files:
    - "shared/setup.sh:setup.sh"
evaluation:
  judge_model: gemini/gemini-2.0-flash
  cases:
    - name: test_case_1
      prompt: |-
        Solve problem line 1.
        Solve problem line 2.
      expectations:
        - "Valid JSON output"
        - criterion: "Passes logic"
          weight: 2.0
      script_verifiers:
        - "tests/check.py"
        - script: "tests/check2.py"
          args: ["--arg1"]
          weight: 1.5
    - name: test_case_2
      prompt_file: "tasks/a1/TASK.md"
      workspace_files:
        - "tasks/a1/TASK.md:TASK.md"
        - "traces/j3:inputs/trace_a"
"""
        expected = schema.EvalSuite(
            name="test_suite",
            environment=schema.EnvironmentConfig(
                base_image="python:3.12-slim",
                pip_packages=["pytest"],
                workspace_files=["shared/setup.sh:setup.sh"],
            ),
            evaluation=schema.EvaluationConfig(
                judge_model="gemini/gemini-2.0-flash",
                cases=[
                    schema.TestCase(
                        name="test_case_1",
                        prompt="Solve problem line 1.\nSolve problem line 2.",
                        expectations=[
                            "Valid JSON output",
                            schema.ExpectationCriterion(
                                criterion="Passes logic", weight=2.0),
                        ],
                        script_verifiers=[
                            "tests/check.py",
                            schema.ScriptCheck(
                                script="tests/check2.py",
                                args=["--arg1"],
                                weight=1.5,
                            ),
                        ],
                    ),
                    schema.TestCase(
                        name="test_case_2",
                        prompt_file="tasks/a1/TASK.md",
                        workspace_files=[
                            "tasks/a1/TASK.md:TASK.md",
                            "traces/j3:inputs/trace_a",
                        ],
                    ),
                ],
            ),
        )
        suite = schema.EvalSuite.from_yaml(yaml_content)
        self.assertEqual(suite, expected)

    def test_from_yaml_file_path(self) -> None:
        yaml_content = """
evaluation:
  cases:
    - name: case
      prompt: "prompt"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = pathlib.Path(tmp_dir)
            yaml_file = temp_path / "suite.yaml"
            yaml_file.write_text(yaml_content, encoding="utf-8")

            suite_from_path = schema.EvalSuite.from_yaml(yaml_file)
            self.assertEqual(suite_from_path.evaluation.cases[0].name, "case")

    def test_file_staging_coercion(self) -> None:
        # Docker style colon shorthand
        fs1 = schema.FileStaging.coerce("artifacts/traces/j3:inputs/trace_a")
        self.assertEqual(fs1.src, "artifacts/traces/j3")
        self.assertEqual(fs1.dest, "inputs/trace_a")

        # Single string shorthand
        fs2 = schema.FileStaging.coerce("TASK.md")
        self.assertEqual(fs2.src, "TASK.md")
        self.assertEqual(fs2.dest, "TASK.md")

        # Explicit dict
        fs3 = schema.FileStaging.coerce({
            "src": "path/to/src",
            "dest": "path/to/dest"
        })
        self.assertEqual(fs3.src, "path/to/src")
        self.assertEqual(fs3.dest, "path/to/dest")

        # FileStaging instance
        fs4 = schema.FileStaging(src="a", dest="b")
        self.assertIs(schema.FileStaging.coerce(fs4), fs4)

        # Default dest
        fs5 = schema.FileStaging(src="only_src.txt")
        self.assertEqual(fs5.src, "only_src.txt")
        self.assertEqual(fs5.dest, ".")

        # GCS URI with colon destination
        fs_gcs1 = schema.FileStaging.coerce(
            "gs://xprof_oss_skills/traces/v1/j1_baseline:inputs/trace")
        self.assertEqual(fs_gcs1.src,
                         "gs://xprof_oss_skills/traces/v1/j1_baseline")
        self.assertEqual(fs_gcs1.dest, "inputs/trace")

        # GCS URI without destination
        fs_gcs2 = schema.FileStaging.coerce(
            "gs://xprof_oss_skills/traces/v1/trace.pb")
        self.assertEqual(fs_gcs2.src,
                         "gs://xprof_oss_skills/traces/v1/trace.pb")
        self.assertEqual(fs_gcs2.dest, "trace.pb")

        # Invalid type
        with self.assertRaises(TypeError):
            schema.FileStaging.coerce(123)

    def test_environment_workspace_files(self) -> None:
        env = schema.EnvironmentConfig(workspace_files=[
            "common/data:data",
            "TASK.md",
            {
                "src": "extra",
                "dest": "extra_dest"
            },
        ])
        norm = env.normalized_workspace_files()
        self.assertLen(norm, 3)
        self.assertEqual(norm[0].src, "common/data")
        self.assertEqual(norm[0].dest, "data")
        self.assertEqual(norm[1].src, "TASK.md")
        self.assertEqual(norm[1].dest, "TASK.md")
        self.assertEqual(norm[2].src, "extra")
        self.assertEqual(norm[2].dest, "extra_dest")

        # Test None coercion
        env_none = schema.EnvironmentConfig.model_validate(
            {"workspace_files": None})
        self.assertEqual(env_none.workspace_files, [])

    def test_case_resolve_prompt(self) -> None:
        # Resolve prompt from prompt string
        tc = schema.TestCase(name="tc", prompt="tpu-hello-world")
        self.assertEqual(tc.resolve_prompt(), "tpu-hello-world")

        with tempfile.TemporaryDirectory() as tmp_dir:
            suite_dir = Path(tmp_dir)
            task_md = suite_dir / "tasks" / "TASK.md"
            task_md.parent.mkdir(parents=True, exist_ok=True)
            task_md.write_text("# Task Instructions\nDo this step.",
                               encoding="utf-8")

            # Resolve relative to source_root
            tc_file = schema.TestCase(name="tc_file",
                                      prompt_file="tasks/TASK.md")
            self.assertEqual(
                tc_file.resolve_prompt(source_root=suite_dir),
                "# Task Instructions\nDo this step.",
            )

            # Resolve absolute path
            tc_abs = schema.TestCase(name="tc_abs",
                                     prompt_file=str(task_md.resolve()))
            self.assertEqual(
                tc_abs.resolve_prompt(),
                "# Task Instructions\nDo this step.",
            )

            # Nonexistent prompt_file raises FileNotFoundError
            tc_nonexistent = schema.TestCase(name="tc_bad",
                                             prompt_file="missing/TASK.md")
            with self.assertRaises(FileNotFoundError):
                tc_nonexistent.resolve_prompt(source_root=suite_dir)

    def test_case_workspace_files(self) -> None:
        tc = schema.TestCase(
            name="tc",
            prompt="do task",
            workspace_files=["tasks/a1/TASK.md:TASK.md"],
        )
        norm = tc.normalized_workspace_files()
        self.assertLen(norm, 1)
        self.assertEqual(norm[0].src, "tasks/a1/TASK.md")
        self.assertEqual(norm[0].dest, "TASK.md")

        # Test None coercion
        tc_none = schema.TestCase.model_validate({
            "name": "tc",
            "prompt": "p",
            "workspace_files": None
        })
        self.assertEqual(tc_none.workspace_files, [])

    def test_case_test_files(self) -> None:
        tc = schema.TestCase.model_validate({
            "name":
            "tc",
            "prompt":
            "solve task",
            "test_files": [
                "configs/schema.json:schema.json",
                "configs/key.json:key.json",
            ],
        })
        self.assertLen(tc.test_files, 2)
        norm = tc.normalized_test_files()
        self.assertEqual(norm[0].src, "configs/schema.json")
        self.assertEqual(norm[0].dest, "schema.json")
        self.assertEqual(norm[1].src, "configs/key.json")
        self.assertEqual(norm[1].dest, "key.json")

        # Test None coercion
        tc_none = schema.TestCase.model_validate({
            "name": "tc",
            "prompt": "p",
            "test_files": None,
        })
        self.assertEqual(tc_none.test_files, [])

    def test_script_check_verifier_params(self) -> None:
        sc_default = schema.ScriptCheck(script="tests/check.py")
        self.assertEqual(sc_default.verifier_params, {})

        sc_custom = schema.ScriptCheck(
            script="tests/check.py",
            verifier_params={
                "target_agent": "test",
                "retries": 3
            },
        )
        self.assertEqual(
            sc_custom.verifier_params,
            {
                "target_agent": "test",
                "retries": 3
            },
        )

        sc_coerced = schema.ScriptCheck.coerce({
            "script": "tests/check.py",
            "verifier_params": None,
        })
        self.assertEqual(sc_coerced.verifier_params, {})


if __name__ == "__main__":
    absltest.main()
