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

# Configures GKE compute profile and runs preflight checks.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/e2e_env.sh"

gcloud container clusters get-credentials "$GKE_CLUSTER" --location="$GKE_LOCATION" --project="$GKE_PROJECT" --dns-endpoint

cat <<EOF > "$WORK_ROOT/answers.json"
{
  "backend": "$COMPUTE_BACKEND",
  "namespace": "$GKE_NAMESPACE",
  "queue": "$GKE_QUEUE",
  "priority": "$GKE_PRIORITY",
  "declared_duration_minutes": $GKE_DECLARED_DURATION_MINUTES,
  "image": "$TASK_DOCKER",
  "tpu_accelerator": "$GKE_TPU_ACCELERATOR",
  "tpu_topology": "$GKE_TPU_TOPOLOGY"
}
EOF

cd "$WORKSPACE_DIR"
python3 "$DOWNLOAD_DIR/$DIST_NAME/framework/runtime.py" compute --package tpu-agent-suite setup apply --profile default --answers "$WORK_ROOT/answers.json"
python3 "$DOWNLOAD_DIR/$DIST_NAME/framework/runtime.py" compute --package tpu-agent-suite preflight
