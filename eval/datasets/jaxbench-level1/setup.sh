#!/usr/bin/env bash
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

# ==============================================================================
# JaxBench Level 1 Dataset Setup
# ==============================================================================
# Ephemeral just-in-time environment setup for JaxBench Level 1 tasks.
# Stages shared common assets (Dockerfile, instruction.md) into each task directory.
#
# Usage:
#   ./setup.sh          # Stage common assets for all tasks
#   ./setup.sh --clean  # Remove ephemeral common assets
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DOCKERFILE="${SCRIPT_DIR}/common/Dockerfile"
COMMON_INSTRUCTION="${SCRIPT_DIR}/common/instruction.md"

if [ ! -f "$COMMON_DOCKERFILE" ]; then
  echo "Error: Common Dockerfile not found at '$COMMON_DOCKERFILE'" >&2
  exit 1
fi

if [ ! -f "$COMMON_INSTRUCTION" ]; then
  echo "Error: Common instruction not found at '$COMMON_INSTRUCTION'" >&2
  exit 1
fi

ACTION="${1:-stage}"

case "$ACTION" in
  --clean|clean)
    echo "[JaxBench-L1] Cleaning ephemeral task assets (Dockerfiles, instructions)..."
    for task_dir in "${SCRIPT_DIR}"/*/; do
      if [ -f "${task_dir}/task.toml" ]; then
        rm -f "${task_dir}/environment/Dockerfile"
        rm -f "${task_dir}/instruction.md"
      fi
    done
    echo "[JaxBench-L1] Clean complete."
    ;;
  *)
    TASK_COUNT=0
    for task_dir in "${SCRIPT_DIR}"/*/; do
      if [ -f "${task_dir}/task.toml" ]; then
        mkdir -p "${task_dir}/environment"
        cp "$COMMON_DOCKERFILE" "${task_dir}/environment/Dockerfile"
        cp "$COMMON_INSTRUCTION" "${task_dir}/instruction.md"
        TASK_COUNT=$((TASK_COUNT + 1))
      fi
    done
    echo "[JaxBench-L1] Ephemerally staged Dockerfile & instruction.md for ${TASK_COUNT} tasks from common/."
    ;;
esac
