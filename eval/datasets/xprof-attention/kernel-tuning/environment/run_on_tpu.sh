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
# Runs attention.py on a TPU node and brings its trace back.
#
#   ./run_on_tpu.sh <label>
#
# Getting a file off a finished Job is the awkward part: `kubectl cp` needs a
# live pod, and a Job's pod is gone the moment its container exits. So the
# container writes the trace, drops a READY marker and then sleeps -- the pod is
# kept alive on purpose, copied from, and deleted here. No shared volume, no
# bucket, no race with the pod's lifetime.
#
# The earlier version piped the tarball to stdout as base64 and read it back
# from `kubectl logs`, which is simpler and worked until the profiling flags
# started producing real LLO data: the trace went from 285 KB to 15.8 MB, the
# container log rotated, and the marker at the *front* of the stream was the
# part that got dropped -- so a job that had succeeded reported "no trace in
# the job output" and the trace was lost.
#
# Traces land in /logs/artifacts/xprof/<label>/, which is collected off the
# machine when the run ends.
set -uo pipefail

LABEL="${1:?usage: run_on_tpu.sh <label>}"

# The budget binds here, not only in the grader. Enforcing it solely at scoring
# time meant an agent could spend eight profiles and report a figure from the
# seventh -- which is what run I did: its FINAL cited 136.28 us from
# `sharded_final`, a profile the verifier never read, while the best it had
# actually shown was 1007.72 us. Wasted chips, and a claim about work nobody
# scored. Refusing the seventh capture makes the limit real to the agent at the
# moment it matters.
MAX_PROFILES="${MAX_PROFILE_RUNS:-6}"
# Counted from a ledger as well as from the directories, and the larger wins.
# Counting directories alone made the budget reversible: `rm -rf` on one
# capture bought another, and the cap is supposed to measure whether the
# instruction was followed. The ledger is append-only in practice -- deleting
# it does not restore the budget either, because a ledger shorter than the
# captures on disk is itself the discrepancy the verifier reports.
LEDGER="/logs/artifacts/xprof/.captures"
mkdir -p /logs/artifacts/xprof 2>/dev/null || true
DIRS="$(find /logs/artifacts/xprof -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d " ")"
LOGGED="$(wc -l < "$LEDGER" 2>/dev/null | tr -d " ")"
TAKEN="$DIRS"
[ "${LOGGED:-0}" -gt "$TAKEN" ] && TAKEN="$LOGGED"
if [ "$TAKEN" -ge "$MAX_PROFILES" ]; then
  echo "refusing: $TAKEN of $MAX_PROFILES profiles already taken." >&2
  echo "The budget is spent. Report the best result you have measured." >&2
  exit 1
fi
printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LABEL" >> "$LEDGER" 2>/dev/null || true

NS="${GKE_NAMESPACE:-default}"
JOB="attn-${LABEL//[^a-z0-9-]/-}-$RANDOM"
# Built here rather than inline in the YAML: `${GKE_QUEUE:+...{...}}` ends its
# expansion at the first unescaped brace, so with no queue the metadata line
# renders one `}` too many and kubectl rejects the whole document with a line
# number that points at `kind: Job`.
QLABEL=""
[ -n "${GKE_QUEUE:-}" ] && QLABEL=", labels: {kueue.x-k8s.io/queue-name: $GKE_QUEUE}"

# Chips per node, from the topology rather than a constant. GKE turns the chip
# request into a third node selector, `gke-accelerator-count`, so asking for 4
# on an 8-chip v6e node (2x4) leaves the pod Pending forever behind a message
# about node affinity that never mentions the count. tpu7x 2x2x1 -> 4,
# v6e 2x4 -> 8. Single-host slices only; a multi-host topology would need the
# per-node share, not the product.
CHIPS="${GKE_TPU_CHIPS:-}"
if [ -z "$CHIPS" ]; then
  CHIPS=1
  for n in ${GKE_TPU_TOPOLOGY//x/ }; do CHIPS=$((CHIPS * n)); done
fi
OUT="/logs/artifacts/xprof/$LABEL"

kubectl create configmap "$JOB-src" -n "$NS" \
  --from-file=attention.py=/workspace/attention.py >/dev/null || exit 1

cat <<YAML | kubectl apply -f - >/dev/null || exit 1
apiVersion: batch/v1
kind: Job
metadata: {name: $JOB, namespace: $NS$QLABEL}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: ${GKE_TPU_ACCELERATOR}
        cloud.google.com/gke-tpu-topology: ${GKE_TPU_TOPOLOGY}
      volumes: [{name: src, configMap: {name: $JOB-src}}]
      containers:
        - name: main
          image: python:3.12-slim
          command: ["bash","-c"]
          args:
            - |
              pip install -q --index-url https://pypi.org/simple 'jax[tpu]==0.11.0' 2>&1 | tail -1
              python3 /src/attention.py --label "$LABEL" --logdir /tmp/tr
              rc=\$?
              tar -czf /tmp/trace.tgz -C /tmp/tr . 2>/dev/null
              echo "\$rc" > /tmp/RC
              touch /tmp/READY
              sleep ${GKE_KEEPALIVE_S:-900}
          volumeMounts: [{name: src, mountPath: /src}]
          resources:
            limits: {google.com/tpu: $CHIPS}
            requests: {google.com/tpu: $CHIPS}
YAML

echo "submitted $JOB; waiting..."
# Polled, not `kubectl wait`. Straight after an apply the Job is often not yet
# visible, so `wait` returns an error instantly and the `|| wait --for=failed
# --timeout=10s` fallback returns ten seconds later -- long before the workload
# has run. And `--for=condition=complete` would never fire now anyway: the
# container sleeps after writing, so what is waited on is the READY marker.
POD=""
READY=0
DEADLINE=$(( SECONDS + ${GKE_JOB_TIMEOUT_S:-1800} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  POD="$(kubectl get pods -n "$NS" -l "job-name=$JOB" \
           -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"
  if [ -n "$POD" ] && kubectl exec -n "$NS" "$POD" -- test -f /tmp/READY 2>/dev/null; then
    READY=1; break
  fi
  case "$(kubectl get job "$JOB" -n "$NS" \
            -o jsonpath='{.status.failed}' 2>/dev/null)" in
    [1-9]*) echo "job failed before it wrote a trace" >&2; break ;;
  esac
  sleep 5
done

kubectl logs "job/$JOB" -n "$NS" --tail=-1 2>/dev/null | grep -v '^$'

mkdir -p "$OUT"
if [ "$READY" = 1 ]; then
  RC="$(kubectl exec -n "$NS" "$POD" -- cat /tmp/RC 2>/dev/null | tr -d '[:space:]')"
  [ "${RC:-1}" = 0 ] || echo "workload exited $RC; copying whatever it wrote" >&2
  TGZ="$(mktemp)"
  if kubectl cp -n "$NS" "$POD:/tmp/trace.tgz" "$TGZ" >/dev/null 2>&1 \
     && tar -xzf "$TGZ" -C "$OUT" 2>/dev/null; then
    echo "trace -> $OUT ($(find "$OUT" -name '*.xplane.pb' | wc -l | tr -d ' ') xplane files)"
  else
    echo "could not copy the trace off $POD" >&2
  fi
  rm -f "$TGZ"
else
  echo "no trace: the pod never signalled READY" >&2
fi
kubectl delete job "$JOB" configmap "$JOB-src" -n "$NS" --ignore-not-found >/dev/null 2>&1
