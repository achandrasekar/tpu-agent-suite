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

# Harbor reads one number from /logs/verifier/reward.txt. Every graded row is
# checked against what the xprof CLI itself reports for the profile baked into
# the image, so a right answer is one the tools would give, not one that
# sounds plausible. The routing check (did the run go through tpu-diagnostics?)
# is reported, not scored: the control arm has no package and must still be
# able to score on the numbers.

set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
AGENT_LOGS=(/logs/agent/*)

table_row() {
  # Unanchored: the agent log is JSONL, so the table lives inside JSON strings
  # with escaped newlines and a row never starts a physical line. The `| name |`
  # structure is what makes the match specific.
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|]+" "${AGENT_LOGS[@]}" 2>/dev/null || true; } |
    tail -n 1 | sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//" | sed -E 's/[[:space:]]+$//'
}

# Numeric compare with an absolute tolerance; strips units, %, commas, and any
# trailing prose so "233.8 ms" and "233.8" both read as 233.8.
num() { printf '%s' "$1" | sed -E 's/^[^0-9.-]*//; s/[^0-9.].*$//'; }
near() {
  local got want tol
  got="$(num "$1")"; want="$2"; tol="$3"
  [ -n "$got" ] || return 1
  awk -v a="$got" -v b="$want" -v t="$tol" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= t) }'
}
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
contains() { case "$(lower "$1")" in *"$(lower "$2")"*) return 0 ;; *) return 1 ;; esac; }

SCORE=0
TOTAL=0
LINES=()
check() {
  # check <label> <got> <ok: 0|1> [why]
  local ok="$3"
  TOTAL=$((TOTAL + 1)); [ "$ok" = 1 ] && SCORE=$((SCORE + 1))
  LINES+=("$(printf '%-22s %-40s %s%s' "$1" "${2:-<no table row>}" "$([ "$ok" = 1 ] && echo ok || echo FAIL)" "${4:+  ($4)}")")
}

DA="$(table_row device_type_a)";              check device_type_a "$DA" "$(contains "$DA" "v6" && echo 1 || echo 0)"
DB="$(table_row device_type_b)";              check device_type_b "$DB" "$(contains "$DB" "v6" && echo 1 || echo 0)"
SA="$(table_row step_time_ms_a)";             check step_time_ms_a "$SA" "$(near "$SA" 233.8 7 && echo 1 || echo 0)" "expected 233.8"
SB="$(table_row step_time_ms_b)";             check step_time_ms_b "$SB" "$(near "$SB" 6676.2 200 && echo 1 || echo 0)" "expected 6676.2"
FAST="$(table_row faster_run)";               check faster_run "$FAST" "$(contains "$FAST" "a" && ! contains "$FAST" "b" && echo 1 || echo 0)" "expected A"
CAT="$(table_row slower_run_top_op_category)"; check slower_run_top_op_category "$CAT" "$(contains "$CAT" "all-reduce" && echo 1 || echo 0)" "expected all-reduce"
MXU="$(table_row slower_run_mxu_utilization_percent)"; check slower_run_mxu "$MXU" "$(near "$MXU" 30.6 1 && echo 1 || echo 0)" "expected 30.6"

REWARD="$(awk -v s="$SCORE" -v t="$TOTAL" 'BEGIN { printf "%.4f", (t > 0 ? s / t : 0) }')"
printf '%s\n' "$REWARD" >"$LOGS/reward.txt"
cat >"$LOGS/reward.json" <<JSON
{"reward": $REWARD, "correct": $SCORE, "graded": $TOTAL}
JSON

# Keyed on the CLI's own subagent accounting, which appears in the terminal
# result event as `"by_type":{"tpu-diagnostics":N}`. A grep for the CLI name
# would misreport: the control arm runs `xprof` itself, by design.
if grep -qh '"tpu-diagnostics":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  ROUTED=yes
else
  ROUTED=no
fi

printf '%s\n' "${LINES[@]}"
echo "Routed via tpu-diagnostics: $ROUTED  (reported, not scored)"
echo "Verifier final score: $(cat "$LOGS/reward.txt")  ($SCORE/$TOTAL graded rows)"
