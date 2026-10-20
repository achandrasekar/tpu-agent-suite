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
#
# Capture `vllm serve --help` from the vLLM installed on the target, so the
# launcher is generated against the flags that build actually accepts rather
# than against remembered ones.
#
# Runs on the accelerator host, dispatched with `compute run`.
#
# No virtualenv is activated here. The compute framework runs this script
# under bash and passes the interpreter configured in the compute profile, or
# the one named with `--interpreter`, as TPU_AGENT_SUITE_PYTHON; `vllm` is resolved
# from the same directory as that interpreter so a venv install is found
# without sourcing anything.
#
# Usage:
#   compute run --code .../discover_vllm_flags.sh \
#     --output vllm_flags.txt:<local> -- vllm_flags.txt

set -uo pipefail

OUT="${1:-vllm_flags.txt}"

# Prefer the vllm sitting beside the Python that the framework selected; fall
# back to PATH. Nothing is installed here -- a missing vllm is a finding.
# Resolved, since the default interpreter is a bare `python3`, and with a
# leading ~ expanded, since Kubernetes passes env values through as is.
PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}" || true)"
VLLM=""
if [ -n "$PY" ] && [ -x "$(dirname "$PY")/vllm" ]; then
  VLLM="$(dirname "$PY")/vllm"
elif command -v vllm >/dev/null 2>&1; then
  VLLM="$(command -v vllm)"
fi

if [ -z "$VLLM" ]; then
  echo "vllm was not found on the target (looked beside ${PY:-python3} and on PATH)." >&2
  echo "vllm not found" > "$OUT"
  exit 1
fi

echo "# vllm: $VLLM" > "$OUT"
# Recent vLLM prints only argument groups under plain --help; the flags
# themselves (--tensor-parallel-size, --max-model-len, ...) appear only under
# --help=all. Older builds reject --help=all, so fall back to --help.
if "$VLLM" serve --help=all >> "$OUT" 2>&1; then
  STATUS=0
else
  echo "# vllm: $VLLM" > "$OUT"
  "$VLLM" serve --help >> "$OUT" 2>&1
  STATUS=$?
fi

cat "$OUT"
if [ "$STATUS" -ne 0 ]; then
  echo "\`vllm serve --help\` exited $STATUS; the output above is what it printed." >&2
fi
exit "$STATUS"
