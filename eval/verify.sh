#!/bin/bash
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

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Optional dataset suite to scope the task check to (e.g. tpu-hello-world,
# smoke-check, smart-hf-search). Configs are always validated as a whole,
# since eval/configs/ is no longer partitioned by package.
SUITE="${1:-}"

if [ -n "$SUITE" ]; then
  echo "=== Verifying Harbor Setup (datasets/$SUITE) ==="
else
  echo "=== Verifying Harbor Setup (all configs and datasets) ==="
fi

# Resolve Harbor binary (fallback to PATH)
HARBOR_BIN="${HARBOR_BIN:-harbor}"

if ! command -v "$HARBOR_BIN" >/dev/null 2>&1; then
  echo "Error: '$HARBOR_BIN' CLI not found." >&2
  echo "Please install Harbor or set HARBOR_BIN environment variable." >&2
  exit 1
fi

# Resolve Python binary (fallback to PATH or infer from harbor path)
if [ -z "${PYTHON_BIN:-}" ]; then
  if [[ "$HARBOR_BIN" == */bin/harbor ]]; then
    PYTHON_BIN="${HARBOR_BIN%/harbor}/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' not found." >&2
  exit 1
fi

echo "Using Harbor: $HARBOR_BIN ($($HARBOR_BIN --version))"
echo "Using Python: $PYTHON_BIN"

# 2. Check Docker
if ! docker info >/dev/null 2>&1; then
  echo "WARNING: Docker daemon is not running. Live runs will fail."
fi

# Check API Key
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "WARNING: GEMINI_API_KEY is not set. Live runs will fail."
fi

# 3. Validate schemas
echo "Validating schemas..."
export SUITE
"$PYTHON_BIN" - << 'EOF'
import os, sys, yaml
from pathlib import Path
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

eval_dir = Path("eval")
suite = os.environ.get("SUITE", "")

# Validate JobConfigs under configs/
configs_dir = eval_dir / "configs"
if not configs_dir.is_dir():
    print(f"Error: Configs directory not found: {configs_dir}", file=sys.stderr)
    sys.exit(1)

# Recursive: configs are grouped by harness, configs/<harness>/<name>.yaml.
config_files = sorted(configs_dir.glob("**/*.yaml"))
if not config_files:
    print(f"Error: No config (.yaml) files found under {configs_dir}",
          file=sys.stderr)
    sys.exit(1)

for cfg_file in config_files:
    try:
        with open(cfg_file, "r") as f:
            job_data = yaml.safe_load(f)
        JobConfig.model_validate(job_data)
        print(f"✓ JobConfig schema OK ({cfg_file})")
    except Exception as e:
        print(f"Error validating JobConfig {cfg_file}: {e}", file=sys.stderr)
        sys.exit(1)

# Validate TaskConfigs under datasets/ and .datasets/
try:
    import tomllib
    suites_dir = eval_dir / "suites"
    from eval.suite_to_harbor.suite_to_harbor import compile_suite
    if suites_dir.is_dir():
        dot_datasets = eval_dir / ".datasets"
        for suite_file in suites_dir.glob("**/*.yaml"):
            with open(suite_file, "r", encoding="utf-8") as f:
                s_data = yaml.safe_load(f) or {}
            s_name = s_data.get("name") or s_data.get("suite_name") or suite_file.stem
            sub_rel = suite_file.parent.relative_to(suites_dir)
            target_suite_dir = dot_datasets / sub_rel if str(sub_rel) != "." else dot_datasets / s_name
            task_tomls = list(target_suite_dir.glob("**/task.toml"))
            stale = any(suite_file.stat().st_mtime > t.stat().st_mtime for t in task_tomls)
            if not target_suite_dir.is_dir() or not task_tomls or stale:
                compile_suite(suite_file, target_suite_dir)

    # Every dataset is top-level now, so there is one place to look.
    dataset_dir = eval_dir / "datasets" / suite if suite else eval_dir / "datasets"
    if not dataset_dir.is_dir():
        print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(dataset_dir.glob("**/task.toml"))
    if not suite:
        task_files += sorted(eval_dir.glob("**/.*datasets/**/task.toml"))

    if not task_files:
        print(f"WARNING: No task.toml files found under {dataset_dir}")

    for task_path in task_files:
        try:
            with open(task_path, "rb") as f:
                task_data = tomllib.load(f)
            TaskConfig.model_validate(task_data)
            print(f"✓ TaskConfig schema OK ({task_path})")
        except Exception as e:
            print(f"Error validating TaskConfig {task_path}: {e}", file=sys.stderr)
            sys.exit(1)
except ImportError:
    print("Skipping TaskConfig schema validation (requires python 3.11+ or tomllib)")
EOF

echo "Verification complete."
