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
#
# Run before building. Catches the two ways this task silently becomes invalid:
# shipping a file the instruction does not name, and shipping a file that gives
# the answer away. Both happened: a rename left the Dockerfile copying the old
# multi-stage workload, so the agent was handed the tuned tile sizes and never
# found the file it was told to edit. The trial burned 48 minutes and scored 0.
# Run this before sending the task anywhere:
#
#     bash eval/datasets/xprof-attention/kernel-tuning/tests/preflight.sh
#
# It compares the repo against itself -- does the instruction name the file the
# Dockerfile ships, and does any workload file carry the answer -- which is a
# question the verifier cannot ask, because it runs inside the container and
# never sees the Dockerfile. The image-side half of this check now lives in
# `test.sh`, which does run in the built image.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

named="$(grep -oE '/workspace/[a-z_]+\.py' "$HERE/instruction.md" | sort -u)"
shipped="$(grep -oE 'COPY [^ ]+ (/workspace/[a-z_]+\.py)' "$HERE/environment/Dockerfile" | awk '{print $3}' | sort -u)"
echo "  instruction names: ${named:-<none>}"
echo "  dockerfile ships:  ${shipped:-<none>}"
[ "$named" = "$shipped" ] || { echo "  FAIL: the file the agent is told to edit is not the file shipped"; fail=1; }

# Nothing in the workload directory may carry a worked answer.
for f in "$HERE"/environment/*.py; do
  if grep -qE 'block_q *= *[0-9]|block_kv *= *[0-9]|BlockSizes\(|splash_attention|V2_BLOCKS' "$f"; then
    echo "  FAIL: $(basename "$f") contains tile sizes or a kernel -- that is the answer"; fail=1
  fi
done
[ "$fail" = 0 ] && echo "  preflight OK" || echo "  preflight FAILED"
exit "$fail"
