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
"""Strict Pydantic v2 schema definitions for evaluation suites.

Defines data models for evaluating AI agents:
  - environment: Base container, packages, and workspace files.
  - evaluation: Judge model, shared script verifiers, and test cases.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal, TextIO

import pydantic
import yaml

Path = pathlib.Path


class StrictModel(pydantic.BaseModel):
    """Base class enforcing strict validation and forbidding unknown fields."""

    model_config = pydantic.ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class FileStaging(StrictModel):
    """File or directory to stage into the container workspace or test directory."""

    src: str = pydantic.Field(
        ...,
        description="Host source path (relative to suite YAML or absolute).",
    )
    dest: str = pydantic.Field(
        default=".",
        description=("Destination path relative to target dir (default: '.' or"
                     " source filename)."),
    )

    @classmethod
    def coerce(cls, value: str | FileStaging | dict[str, Any]) -> FileStaging:
        """Normalizes string shorthand, dict, or instance into a FileStaging."""
        if isinstance(value, FileStaging):
            return value
        if isinstance(value, str):
            if "://" in value:
                scheme, rest = value.split("://", 1)
                if ":" in rest:
                    src_rest, dest = rest.split(":", 1)
                    return cls(
                        src=f"{scheme}://{src_rest.strip()}",
                        dest=dest.strip(),
                    )
                fallback_dest = Path(rest.strip("/")).name or "."
                return cls(src=value.strip(), dest=fallback_dest)
            if ":" in value:
                src, dest = value.split(":", 1)
                return cls(src=src.strip(), dest=dest.strip())
            return cls(src=value, dest=value)
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to FileStaging")


class ScriptCheck(StrictModel):
    """Detailed configuration for a script-based verifier."""

    script: str = pydantic.Field(
        ...,
        description=
        "Path to python verification script (relative or absolute).",
    )
    args: list[str] = pydantic.Field(
        default_factory=list,
        description="Command-line arguments passed to the verifier script.",
    )
    weight: float = pydantic.Field(
        default=1.0,
        ge=0.0,
        description="Relative scoring weight in reward aggregation.",
    )
    name: str | None = pydantic.Field(
        default=None,
        description="Optional custom folder name under tests/.",
    )
    timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description="Optional execution timeout in seconds.",
    )
    verifier_params: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=
        "Key-value parameters to be written to tests/verifer_params.json",
    )

    @pydantic.field_validator("args", mode="before")
    @classmethod
    def _coerce_none_lists(cls, v: Any) -> Any:
        return [] if v is None else v

    @pydantic.field_validator("verifier_params", mode="before")
    @classmethod
    def _coerce_none_dict(cls, v: Any) -> Any:
        return {} if v is None else v

    @classmethod
    def coerce(cls, value: str | ScriptCheck | dict[str, Any]) -> ScriptCheck:
        """Normalizes string shorthand, dict, or instance into a ScriptCheck instance."""
        if isinstance(value, ScriptCheck):
            return value
        if isinstance(value, str):
            return cls(script=value)
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to ScriptCheck")


class ExpectationCriterion(StrictModel):
    """Detailed configuration for an LLM judge rubric criterion."""

    criterion: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("criterion", "description"),
        description="Natural language evaluation criterion for LLM judge.",
    )
    weight: float = pydantic.Field(
        default=1.0,
        ge=0.0,
        description="Relative scoring weight in reward aggregation.",
    )
    type: Literal["binary", "numeric", "likert"] = pydantic.Field(
        default="binary",
        description="Evaluation scale type ('binary', 'numeric', or 'likert').",
    )

    @classmethod
    def coerce(
        cls, value: str | ExpectationCriterion | dict[str, Any]
    ) -> ExpectationCriterion:
        """Normalizes string shorthand, dict, or instance into an ExpectationCriterion."""
        if isinstance(value, ExpectationCriterion):
            return value
        if isinstance(value, str):
            return cls(criterion=value)
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to ExpectationCriterion")


class TestCase(StrictModel):
    """Individual test case in an evaluation suite."""

    name: str = pydantic.Field(
        ...,
        min_length=1,
        description="Unique identifier for the test case.",
    )
    prompt: str | None = pydantic.Field(
        default=None,
        description=
        "Instruction prompt provided to the agent in instruction.md.",
    )
    prompt_file: str | None = pydantic.Field(
        default=None,
        description=
        "Path to markdown/text file containing prompt (e.g. TASK.md).",
    )
    workspace_files: list[str | FileStaging] = pydantic.Field(
        default_factory=list,
        description=
        "Case-specific files or directories to stage into workspace.",
    )
    test_files: list[str | FileStaging] = pydantic.Field(
        default_factory=list,
        description=(
            "Case-specific auxiliary files or directories to stage into tests/."
        ),
    )
    expectations: list[str | ExpectationCriterion] = pydantic.Field(
        default_factory=list,
        description=(
            "List of LLM judge criteria (strings or ExpectationCriterion"
            " objects)."),
    )
    script_verifiers: list[str | ScriptCheck] = pydantic.Field(
        default_factory=list,
        description="Case-specific script verifiers.",
    )
    timeout_sec: float = pydantic.Field(
        default=300.0,
        gt=0.0,
        description="Execution timeout for the agent in seconds.",
    )
    metadata: dict[str, Any] | None = pydantic.Field(
        default=None,
        description="Optional metadata dictionary.",
    )
    verifier_params: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=(
            "Case-level parameter overrides for tests/verifier_params.json."),
    )
    task_name: str | None = pydantic.Field(
        default=None,
        description="Optional explicit full task name override.",
    )
    env: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "Environment variables set in this case's container, written to "
            "task.toml's [environment] env."),
    )

    @pydantic.field_validator(
        "expectations",
        "script_verifiers",
        "workspace_files",
        "test_files",
        mode="before",
    )
    @classmethod
    def _coerce_none_lists(cls, v: Any) -> Any:
        return [] if v is None else v

    @pydantic.field_validator("verifier_params", mode="before")
    @classmethod
    def _coerce_none_dict(cls, v: Any) -> Any:
        return {} if v is None else v

    @pydantic.model_validator(mode="after")
    def _validate_prompt_present(self) -> TestCase:
        has_prompt = bool(self.prompt and self.prompt.strip())
        has_prompt_file = bool(self.prompt_file and self.prompt_file.strip())
        if not has_prompt and not has_prompt_file:
            raise ValueError(
                f"TestCase '{self.name}' must specify either 'prompt' or 'prompt_file'"
            )
        return self

    def resolve_prompt(self, source_root: Path | None = None) -> str:
        """Resolves instruction prompt text from prompt string or prompt_file."""
        if self.prompt:
            return self.prompt.strip()
        if self.prompt_file:
            p = Path(self.prompt_file)
            if not p.is_absolute():
                if source_root and (source_root / p).is_file():
                    p = (source_root / p).resolve()
                elif (Path.cwd() / p).is_file():
                    p = (Path.cwd() / p).resolve()
                elif source_root:
                    p = (source_root / p).resolve()
            if not p.is_file():
                raise FileNotFoundError(
                    f"prompt_file not found: {self.prompt_file}")
            return p.read_text(encoding="utf-8").rstrip()
        raise ValueError(f"No prompt source found for test case '{self.name}'")

    def normalized_expectations(self) -> list[ExpectationCriterion]:
        """Returns all expectations coerced to ExpectationCriterion objects."""
        return [ExpectationCriterion.coerce(e) for e in self.expectations]

    def normalized_script_verifiers(self) -> list[ScriptCheck]:
        """Returns all script verifiers coerced to ScriptCheck objects."""
        return [ScriptCheck.coerce(sv) for sv in self.script_verifiers]

    def normalized_workspace_files(self) -> list[FileStaging]:
        """Returns all workspace files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.workspace_files]

    def normalized_test_files(self) -> list[FileStaging]:
        """Returns all test files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.test_files]


class EnvironmentConfig(StrictModel):
    """Environment and container configuration."""

    base_image: str = pydantic.Field(
        default="python:3.12-slim",
        description="Base Docker image.",
    )

    apt_packages: list[str] = pydantic.Field(
        default_factory=list,
        description="System apt packages to install during container build.",
    )
    pip_packages: list[str] = pydantic.Field(
        default_factory=list,
        description="Python pip packages to install during container build.",
    )
    workspace_files: list[str | FileStaging] = pydantic.Field(
        default_factory=list,
        description=(
            "Suite-level files or directories to stage into the workspace."),
    )
    cpus: int = pydantic.Field(
        default=2,
        gt=0,
        description="CPU cores limit for task container.",
    )
    memory_mb: int = pydantic.Field(
        default=4096,
        gt=0,
        description="Memory allocation in MB.",
    )
    storage_mb: int = pydantic.Field(
        default=10240,
        gt=0,
        description="Storage allocation in MB.",
    )
    workdir: str = pydantic.Field(
        default="/workspace",
        description="Working directory in container.",
    )

    @pydantic.field_validator(
        "apt_packages",
        "pip_packages",
        "workspace_files",
        mode="before",
    )
    @classmethod
    def _coerce_none_lists(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            v_str = v.strip()
            return [v_str] if v_str else []
        return list(v)

    def normalized_workspace_files(self) -> list[FileStaging]:
        """Returns suite-level workspace files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.workspace_files]


class EvaluationConfig(StrictModel):
    """Evaluation configuration and test cases."""

    judge_model: str = pydantic.Field(
        default="gemini/gemini-3.6-flash",
        description="Default LLM judge model identifier.",
    )
    script_verifiers: list[str | ScriptCheck] = pydantic.Field(
        default_factory=list,
        description=
        "Suite-level script verifiers applied across all test cases.",
    )
    cases: list[TestCase] = pydantic.Field(
        ...,
        description="List of test cases.",
    )

    @pydantic.field_validator("script_verifiers", "cases", mode="before")
    @classmethod
    def _coerce_none_lists(cls, v: Any) -> Any:
        return [] if v is None else v

    def normalized_script_verifiers(self) -> list[ScriptCheck]:
        """Returns suite-level script verifiers coerced to ScriptCheck objects."""
        return [ScriptCheck.coerce(sv) for sv in self.script_verifiers]


class EvalSuite(StrictModel):
    """Root evaluation suite document."""

    version: str = pydantic.Field(
        default="1.0",
        description="Schema version string.",
    )
    name: str | None = pydantic.Field(
        default=None,
        description="Name of the evaluation suite.",
    )
    suite_name: str | None = pydantic.Field(
        default=None,
        description="Alias for suite name.",
    )
    environment: EnvironmentConfig = pydantic.Field(
        default_factory=EnvironmentConfig,
        description="Environment and container configuration.",
    )
    evaluation: EvaluationConfig = pydantic.Field(
        ...,
        description=
        ("Evaluation configuration including judge model, verifiers, and test"
         " cases."),
    )

    @pydantic.model_validator(mode="after")
    def _resolve_name(self) -> EvalSuite:
        if not self.name and self.suite_name:
            self.name = self.suite_name
        elif not self.suite_name and self.name:
            self.suite_name = self.name
        return self

    @classmethod
    def from_file(cls, file_path: str | Path) -> EvalSuite:
        """Loads and validates an EvalSuite from a .yaml or .yml file path."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Suite YAML file not found: {file_path}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected YAML mapping at root of {file_path}, got {type(data).__name__}"
            )
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_content: str | Path | TextIO) -> EvalSuite:
        """Loads and validates an EvalSuite from a YAML string, file path, or file stream."""
        if isinstance(yaml_content, Path):
            with open(yaml_content, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(yaml_content, str):
            path_candidate = Path(yaml_content)
            if "\n" not in yaml_content and path_candidate.is_file():
                with open(path_candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            elif "\n" not in yaml_content and (yaml_content.endswith(".yaml")
                                               or
                                               yaml_content.endswith(".yml")):
                raise FileNotFoundError(
                    f"Suite YAML file not found: {yaml_content}")
            else:
                data = yaml.safe_load(yaml_content)
        elif hasattr(yaml_content, "read"):
            data = yaml.safe_load(yaml_content.read())
        else:
            raise TypeError(
                f"Unsupported YAML content type: {type(yaml_content).__name__}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected YAML mapping at root, got {type(data).__name__}")
        return cls.model_validate(data)


def normalize_script_verifier(
    item: str | ScriptCheck | dict[str, Any], ) -> ScriptCheck:
    """Normalizes a string path, dict, or ScriptCheck instance into a ScriptCheck."""
    return ScriptCheck.coerce(item)


def normalize_expectation(
    item: str | ExpectationCriterion | dict[str, Any],
) -> ExpectationCriterion:
    """Normalizes a string rubric, dict, or ExpectationCriterion into an ExpectationCriterion."""
    return ExpectationCriterion.coerce(item)


def normalize_file_staging(
    item: str | FileStaging | dict[str, Any], ) -> FileStaging:
    """Normalizes a string path, dict, or FileStaging instance into a FileStaging."""
    return FileStaging.coerce(item)
