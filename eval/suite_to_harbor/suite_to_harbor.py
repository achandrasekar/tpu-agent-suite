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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydantic>=2.0",
#     "pyyaml>=6.0",
# ]
# ///
"""YAML to Harbor task generator and CLI."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import pathlib
import re
import shutil
import sys
from collections.abc import Sequence
from typing import Any

from . import gcs_utils, schema, template_generators

Path = pathlib.Path


def sanitize_name(name: str) -> str:
    """Sanitizes strings for safe directory names, identifiers, and container tags."""
    clean = name.strip().lower()
    clean = re.sub(r"[^a-z0-9_.-]+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_.-") or "eval_task"


def resolve_path(
    path_str: str,
    source_root: Path | None = None,
    cache_root: Path | None = None,
) -> Path | None:
    """Resolves file paths relative to suite directory, cwd, absolute paths, or GCS URIs."""
    if not path_str or not path_str.strip():
        return None
    clean_path = path_str.strip()
    if clean_path.startswith("gs://"):
        return gcs_utils.fetch_gcs_asset(clean_path, cache_root=cache_root)
    p = Path(clean_path)
    if p.is_absolute() and p.exists():
        return p.resolve()
    for root in (source_root, Path.cwd()):
        if root:
            cand = (root / p).resolve()
            if cand.exists():
                return cand
    return None


def link_or_copy(src: str | Path, dest: str | Path) -> None:
    """Hardlinks cached assets to dest or copies local repo files."""
    src_p = Path(src).resolve()
    dest_p = Path(dest)
    # Cleanup dest
    if dest_p.exists():
        if dest_p.is_dir():
            shutil.rmtree(dest_p)
        else:
            dest_p.unlink()
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    cache_root = gcs_utils.get_gcs_cache_root().resolve()
    # Hardlink cache data into generated tasks.
    if cache_root in src_p.parents:
        try:
            os.link(src_p, dest_p)
            return
        except OSError as e:
            logging.debug(
                "Could not hardlink %s to %s (%s); falling back to copy.",
                src_p,
                dest_p,
                e,
            )
    shutil.copy2(src_p, dest_p)


def copy_tree_with_patterns(src: Path, dest: Path, *ignore_patterns:
                            str) -> None:
    """Recursively copies/links src to dest using glob ignore patterns."""
    if not src.is_dir():
        return
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(*ignore_patterns),
        copy_function=link_or_copy,
        dirs_exist_ok=True,
    )


def copy_filtered_tree(src: Path, dest: Path) -> None:
    """Recursively copies files, ignoring hidden and test files."""
    copy_tree_with_patterns(src, dest, ".*", "*_test.py", "test_*")


def stage_file(
    staging: schema.FileStaging,
    dest_dir: Path,
    source_root: Path | None = None,
) -> bool:
    """Stages a source file or directory into dest_dir based on FileStaging."""
    resolved = resolve_path(staging.src, source_root=source_root)
    if not resolved:
        logging.warning("File path not found: %s", staging.src)
        return False

    dest_rel = staging.dest.strip()
    if dest_rel.startswith("/workspace/"):
        dest_rel = dest_rel[len("/workspace/"):]
    elif dest_rel.startswith("workspace/"):
        dest_rel = dest_rel[len("workspace/"):]
    elif dest_rel.startswith("/tests/"):
        dest_rel = dest_rel[len("/tests/"):]
    elif dest_rel.startswith("tests/"):
        dest_rel = dest_rel[len("tests/"):]
    is_dir_dest = dest_rel.endswith("/") or dest_rel in ("", ".")
    dest_rel = dest_rel.strip("/")

    dest_dir.mkdir(parents=True, exist_ok=True)

    if resolved.is_dir():
        target_dir = dest_dir / dest_rel if dest_rel else dest_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_filtered_tree(src=resolved, dest=target_dir)
        return True
    elif resolved.is_file():
        dest_path = dest_dir / dest_rel if dest_rel else dest_dir
        if is_dir_dest or dest_path.is_dir():
            target_file = dest_path / resolved.name
        else:
            target_file = dest_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(resolved, target_file)
        return True
    return False


def bundle_workspace_data(
    env: schema.EnvironmentConfig,
    task_dir: Path,
    source_root: Path | None = None,
    case: schema.TestCase | None = None,
    case_workspace_files: Sequence[schema.FileStaging | str] | None = None,
) -> bool:
    """Stages workspace files into environment/data/."""
    data_dir = task_dir / "environment" / "data"

    # 1. Suite level files
    for wf in env.normalized_workspace_files():
        stage_file(wf, dest_dir=data_dir, source_root=source_root)

    # 2. Case level files
    staged_case_files: list[schema.FileStaging] = []
    if case_workspace_files is not None:
        staged_case_files.extend(
            [schema.FileStaging.coerce(f) for f in case_workspace_files])
    elif case is not None:
        staged_case_files.extend(case.normalized_workspace_files())

    for wf in staged_case_files:
        stage_file(wf, dest_dir=data_dir, source_root=source_root)

    if data_dir.is_dir():
        has_files = any(f.is_file() for f in data_dir.rglob("*"))
        if not has_files:
            shutil.rmtree(data_dir, ignore_errors=True)
            return False
        return True
    return False


def bundle_script_verifiers(
    verifiers: Sequence[Any],
    task_dir: Path,
    source_root: Path | None = None,
) -> None:
    """Stages script verifiers into tests/<folder_name>/<script_filename>."""
    if not verifiers:
        return

    used_names: set[str] = set()
    for raw_v in verifiers:
        sc = schema.ScriptCheck.coerce(raw_v)
        resolved = resolve_path(sc.script, source_root=source_root)
        clean_stem = sanitize_name(sc.name or Path(sc.script).stem)

        folder_name = clean_stem
        count = 2
        while folder_name in used_names:
            folder_name = f"{clean_stem}_{count}"
            count += 1
        used_names.add(folder_name)

        check_dir = task_dir / "tests" / folder_name
        check_dir.mkdir(parents=True, exist_ok=True)
        dest_file = check_dir / (resolved.name
                                 if resolved else f"{folder_name}.py")

        if resolved and resolved.is_file():
            if sc.args:
                content = resolved.read_text(encoding="utf-8")
                # Prepending an assignment pushes the file's own imports down,
                # so ruff reads every one of them as E402. The generated file
                # is machine-written and cannot be edited to satisfy that, and
                # a bundle the repo's own hooks reject is a bundle somebody has
                # to patch by hand after every regeneration.
                header = (f"_SPECIFIED_ARGS = {json.dumps(list(sc.args))}\n"
                          "# ruff: noqa: E402\n\n")
                # After the licence header *and* the module docstring, not
                # before them. Prepending at position 0 puts code above the
                # licence, which `addlicense` reports as missing, and above the
                # docstring, which stops being one -- `__doc__` goes None and
                # anything reading it breaks.
                lines = content.splitlines(keepends=True)
                at = 0
                for n, line in enumerate(lines):
                    if line.startswith("#") or not line.strip():
                        at = n + 1
                    else:
                        break
                try:
                    tree = ast.parse(content)
                    if (tree.body and isinstance(tree.body[0], ast.Expr)
                            and isinstance(tree.body[0].value, ast.Constant)
                            and isinstance(tree.body[0].value.value, str)):
                        at = max(at, tree.body[0].end_lineno)
                except SyntaxError:
                    pass
                dest_file.write_text("".join(lines[:at]) + header +
                                     "".join(lines[at:]),
                                     encoding="utf-8")
            else:
                shutil.copy2(resolved, dest_file)
        else:
            logging.warning("Script verifier file not found on disk: %s",
                            sc.script)


def compile_case(
    suite_name: str,
    case: schema.TestCase,
    env: schema.EnvironmentConfig,
    eval_cfg: schema.EvaluationConfig,
    output_dir: Path,
    judge_model_override: str | None = None,
    source_root: Path | None = None,
) -> Path:
    """Compiles a single TestCase into a self-contained Harbor task bundle."""
    task_dir = output_dir / sanitize_name(case.name)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = case.resolve_prompt(source_root=source_root)

    # 1. Write instruction.md & task.toml
    (task_dir / "instruction.md").write_text(prompt_text, encoding="utf-8")
    (task_dir / "task.toml").write_text(
        template_generators.generate_task_toml(
            suite_name=suite_name,
            case=case,
            env=env,
            prompt_text=prompt_text,
        ),
        encoding="utf-8",
    )

    # 2. Bundle workspace data & generate Dockerfile
    has_data = bundle_workspace_data(
        env=env,
        task_dir=task_dir,
        source_root=source_root,
        case=case,
    )
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        template_generators.generate_dockerfile(env=env, has_data=has_data),
        encoding="utf-8",
    )

    # 3. Setup RewardKit tests harness
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "reward.toml").write_text(
        template_generators.generate_reward_toml(), encoding="utf-8")

    # 3b. Stage case-level test files into tests/
    for tf in case.normalized_test_files():
        stage_file(staging=tf, dest_dir=tests_dir, source_root=source_root)

    # 4. LLM judge expectations (tests/expectations/judge.toml)
    judge_model = judge_model_override or eval_cfg.judge_model
    judge_toml = template_generators.generate_judge_toml(
        expectations=case.normalized_expectations(),
        judge_model=judge_model,
    )
    if judge_toml:
        expectations_dir = tests_dir / "expectations"
        expectations_dir.mkdir(parents=True, exist_ok=True)
        (expectations_dir / "judge.toml").write_text(judge_toml,
                                                     encoding="utf-8")

    # 5. Script verifiers (tests/<check>/...)
    all_verifiers = list(eval_cfg.normalized_script_verifiers()) + list(
        case.normalized_script_verifiers())
    if all_verifiers:
        bundle_script_verifiers(all_verifiers,
                                task_dir=task_dir,
                                source_root=source_root)

    # 5b. Write custom params into verifier_params.json.
    effective_params: dict[str, Any] = {}
    for sv in all_verifiers:
        params = {}
        sc = schema.ScriptCheck.coerce(sv)
        if sc.verifier_params:
            params.update(sc.verifier_params)

        if case.verifier_params:
            params.update(case.verifier_params)
        effective_params[sv.name or Path(sv.script).stem] = params

    if effective_params:
        (tests_dir / "verifier_params.json").write_text(
            json.dumps(effective_params, indent=2) + "\n",
            encoding="utf-8",
        )

    # 6. Executable test.sh
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(template_generators.generate_test_sh(),
                       encoding="utf-8")
    os.chmod(test_sh, 0o755)

    return task_dir


def compile_suite(
    suite_path: str | Path,
    output_dir: str | Path,
    judge_model_override: str | None = None,
    task_name_filter: str | None = None,
) -> list[Path]:
    """Compiles a YAML evaluation suite into Harbor task bundles."""
    output_dir_p = Path(output_dir).resolve()
    output_dir_p.mkdir(parents=True, exist_ok=True)

    if isinstance(suite_path, Path) or (isinstance(suite_path, str)
                                        and "\n" not in suite_path):
        cand_p = Path(suite_path)
        if cand_p.exists():
            content = cand_p.read_text(encoding="utf-8")
            suite = schema.EvalSuite.from_yaml(content)
            suite_name = suite.suite_name or suite.name or cand_p.stem
            source_root = cand_p.parent if cand_p.is_file() else Path.cwd()
        else:
            resolved_suite = resolve_path(str(suite_path))
            if resolved_suite and resolved_suite.exists():
                content = resolved_suite.read_text(encoding="utf-8")
                suite = schema.EvalSuite.from_yaml(content)
                suite_name = suite.suite_name or suite.name or resolved_suite.stem
                source_root = resolved_suite.parent
            else:
                suite = schema.EvalSuite.from_yaml(str(suite_path))
                suite_name = suite.suite_name or suite.name or "evaluation"
                source_root = Path.cwd()
    else:
        suite = schema.EvalSuite.from_yaml(suite_path)
        suite_name = suite.suite_name or suite.name or "evaluation"
        source_root = Path.cwd()

    cases = [
        c for c in suite.evaluation.cases
        if not task_name_filter or c.name == task_name_filter
        or sanitize_name(c.name) == sanitize_name(task_name_filter) or (
            c.task_name and c.task_name == task_name_filter)
    ]

    if task_name_filter and not cases:
        available = [c.name for c in suite.evaluation.cases]
        raise ValueError(
            f"No test case found matching task_name filter '{task_name_filter}'. "
            f"Available cases in suite: {available}")

    return [
        compile_case(
            suite_name=suite_name,
            case=case,
            env=suite.environment,
            eval_cfg=suite.evaluation,
            output_dir=output_dir_p,
            judge_model_override=judge_model_override,
            source_root=source_root,
        ) for case in cases
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for converting YAML suites to Harbor bundles."""
    parser = argparse.ArgumentParser(
        description="Convert YAML evaluation suites to Harbor task bundles.")
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        required=False,
        help="Path to YAML evaluation suite definition file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="Output directory to write Harbor task bundle(s).",
    )
    parser.add_argument(
        "-j",
        "--judge_model",
        type=str,
        default=None,
        help="Judge model override (e.g. 'gemini/gemini-3.6-flash').",
    )
    parser.add_argument(
        "-t",
        "--task_name",
        type=str,
        default=None,
        help="Filter to convert only a single test case matching this name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    if args.suite and args.output_dir:
        print(f"Compiling Harbor evaluation suite from: {args.suite}")
        converted = compile_suite(
            suite_path=args.suite,
            output_dir=args.output_dir,
            judge_model_override=args.judge_model,
            task_name_filter=args.task_name,
        )
        for p in converted:
            print(f"  ✓ {p.name}")

        print(
            f"\nSuccessfully compiled {len(converted)} Harbor task bundles to:"
            f" {args.output_dir}")
        return 0

    if args.suite and not args.output_dir:
        parser.error("--output_dir is required when --suite is specified.")

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
