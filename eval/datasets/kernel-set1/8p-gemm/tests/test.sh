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

# Placeholder scoring: did the agent profile with xprof. It does not check
# correctness and does not measure speedup, so the instruction's
# 30%-of-roofline ask is currently unscored.
set -uo pipefail

LOGS_DIR="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS_DIR"

if grep -rqi xprof /logs/agent /logs/artifacts "${WORKSPACE_DIR:-/workspace}" 2>/dev/null; then
  echo "1.0000" > "$LOGS_DIR/reward.txt"
  echo "xprof found"
else
  echo "0.0000" > "$LOGS_DIR/reward.txt"
  echo "no xprof evidence"
fi
