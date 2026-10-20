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

# Scores the planning turn with RewardKit: trajectory checks under checks/ and
# an LLM judge under expectations/. A failure to run at all scores zero.
mkdir -p /logs/verifier /logs/agent
echo '{"reward": 0.0}' > /logs/verifier/reward.json
cd /workspace || exit 1

# The checks read /logs/agent/trajectory.json; some harnesses write it one
# level up.
if [ -f /logs/trajectory.json ] && [ ! -f /logs/agent/trajectory.json ]; then
  cp /logs/trajectory.json /logs/agent/trajectory.json
fi

if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh |
    env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1 || true
  export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
fi

export LITELLM_NUM_RETRIES=8
export LITELLM_RETRY_AFTER=3
RETRY_DELAY=3
EXIT_CODE=1
for i in 1 2 3 4 5; do
  if uvx --with google-auth --from "harbor-rewardkit>=0.1.7" \
      rewardkit /tests --workspace /workspace; then
    exit 0
  fi
  EXIT_CODE=$?
  echo "[test.sh] Verifier error ($EXIT_CODE); retrying in ${RETRY_DELAY}s ($i/5)"
  sleep "$RETRY_DELAY"
  RETRY_DELAY=$((RETRY_DELAY * 2))
done
exit "$EXIT_CODE"
