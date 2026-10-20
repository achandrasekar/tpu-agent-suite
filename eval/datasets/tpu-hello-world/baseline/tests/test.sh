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

# Harbor reads one number from /logs/verifier/reward.txt. For a tpu-hello-world
# that is the whole verifier -- no rewardkit, no criteria files, no judge.
#
# The number is the hardware result and nothing else: did a TPU actually do the
# arithmetic. Whether the agent got there by invoking `tpu-hello-world` or by writing
# its own manifest is reported alongside it, not folded into it -- both arms are
# given the same cluster and the same image, so both can legitimately pass, and
# a score that conflated the two would say the package failed whenever the
# control arm succeeded.
set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

# The pod's output reaches the host through the agent's captured session: the
# agent runs `kubectl logs`, so what the TPU printed is quoted verbatim in
# there. It has to be read this way rather than from the cluster, because the
# Job is deleted once its logs have been read -- a Job left holding a TPU node
# keeps the next run Pending.
AGENT_LOGS=(/logs/agent/*)

# The LAST occurrence, not any occurrence. An agent's session records its
# exploration as well as its result: probes that failed, approaches abandoned,
# and the instructions it read along the way, which quote every device label as
# prose. Scoring "does this string appear anywhere" makes a run that succeeded
# after a false start indistinguishable from one that never succeeded.
# Read the row out of the result table the prompt requires, not a string only
# the packaged script prints. The control arm can produce this table too, which
# is the point: the reward measures the hardware result, not which agent ran.
table_row() {
  # Unanchored: the agent log is JSONL, so the table lives inside JSON strings
  # with escaped newlines and a row never starts a physical line. The `| name |`
  # structure is what makes the match specific -- that is what stops a bare
  # generation string in prose, or the cluster's own name, from satisfying it.
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|[:space:]]+" "${AGENT_LOGS[@]}" 2>/dev/null || true; } |
    tail -n 1 | sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//" | tr -d '[:space:]'
}

DEVICE_TYPE="$(table_row device_type)"
ACCELERATOR="$(table_row accelerator)"

# One question: did a real TPU do the arithmetic. `native-tpu`
# is the only label where it landed on silicon -- "cpu-fallback" and
# "cpu-simulated" (MOCK_TPU=1) did not.
#
# The chip generation is reported but not scored. Gating on it would pin the
# task to one cluster: this one is tpu7x, but the same check is meant to run
# against v6e and whatever comes next.
#
# The value is read out of the reported table, so it cannot be satisfied by
# prose or by the cluster's own name -- which matters, since this cluster is
# called `torchtpu-distributed-v7x-8`.
# One aspect, but named the same as in `cited` and `profiled`, which are this
# task plus more. Harbor reads reward.json in preference to reward.txt and
# takes a {name: number} map, so `hardware` means the same thing across all
# three and the ladder can be compared on it.
HARDWARE_OK=no
case "$DEVICE_TYPE" in
"native-tpu") HARDWARE_OK=yes ;;
esac

h=0; [ "$HARDWARE_OK" = yes ] && h=1
printf '%s\n' "$h" >"$LOGS/reward.txt"
cat >"$LOGS/reward.json" <<JSON
{"reward": $h, "hardware": $h}
JSON

# The second signal, reported and not scored. An installed subagent is an offer
# rather than an instruction, and a capable model handed a cluster and a clear
# question will often just do the work itself -- so this is the measurement the
# A/B is actually for.
#
# Keyed on the CLI's own subagent accounting, which appears in the terminal
# result event as `"by_type":{"tpu-hello-world":N}`. The obvious grep for the script
# filename reported `yes` for a run with no package installed at all: the
# control arm wrote its own check and, taking the name from the prompt, called
# it hello_tpu.py.
if grep -qh '"tpu-hello-world":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  USED_PACKAGE=yes
else
  USED_PACKAGE=no
fi

echo "Device type:         ${DEVICE_TYPE:-<no table row>}"
echo "Accelerator:         ${ACCELERATOR:-<no table row>}"
echo "Hardware ran on TPU: $HARDWARE_OK"
echo "Used packaged agent: $USED_PACKAGE"
echo "Verifier final score: $(cat "$LOGS/reward.txt")"
