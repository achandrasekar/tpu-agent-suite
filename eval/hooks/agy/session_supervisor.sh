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
# Antigravity CLI Session Supervisor (/root/.local/bin/agy)
# ==============================================================================
# Pure Shell script supervisor mounted inside evaluation containers.
# Maintains open stdin for streaming multi-agent execution, translates CLI prompts
# into streaming JSON events, and detects subagent quiescence before exiting.
set -euo pipefail

REAL_AGY="/usr/local/bin/agy-real"
[ -x "$REAL_AGY" ] || REAL_AGY="/root/.local/bin/agy-real"
export AGY_ADC_AUTH="${AGY_ADC_AUTH:-1}"
# The eval pins the CLI's version; a background self-update would replace it.
export AGY_CLI_DISABLE_AUTO_UPDATE=true
# Only when an ADC file is mounted: pointing this at a missing file makes
# ADC fail instead of falling back to the pod's GKE metadata server.
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f /root/.config/gcloud/application_default_credentials.json ]; then
  export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
fi
export GOOGLE_CLOUD_QUOTA_PROJECT="${GOOGLE_CLOUD_QUOTA_PROJECT:-${VERTEXAI_PROJECT:-tpu-nexus-767207}}"

SUPERVISOR_LOG="/logs/agent/session_supervisor.log"

log_supervisor() {
  local msg="$1"
  echo "$msg" >&2
  if [ -d "/logs/agent" ] || mkdir -p "/logs/agent" 2>/dev/null; then
    echo "$msg" >> "$SUPERVISOR_LOG" 2>/dev/null || true
  fi
}

if [ ! -x "$REAL_AGY" ]; then
  log_supervisor "[AgySupervisor] Error: Native Antigravity CLI binary not found at $REAL_AGY"
  exit 1
fi

# 1. Pass through inspection commands directly
if [[ "$1" == "--version" || "$1" == "models" || "$1" == "agents" ]]; then
  exec "$REAL_AGY" "$@"
fi

if ! command -v jq &>/dev/null; then
  log_supervisor "[AgySupervisor] jq is not available; bypassing supervisor and executing native CLI directly."
  exec "$REAL_AGY" "$@"
fi

# 2. Extract prompt argument; forward all other flags to agy
PROMPT=""
CLI_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt=*)
      PROMPT="${1#--prompt=}"
      shift
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

CLI_ARGS+=("--input-format" "stream-json")
if [ -n "${AGENT_PERSONA:-}" ] && [[ ! " ${CLI_ARGS[*]} " =~ " --agent " ]]; then
  CLI_ARGS+=("--agent" "$AGENT_PERSONA")
fi

log_supervisor "[AgySupervisor] Launching native Antigravity CLI: $REAL_AGY ${CLI_ARGS[*]}"

# 3. Output stream capture and FIFO to keep stdin open for multi-agent execution
RAW_STREAM="/tmp/agy_raw_stream_$$.jsonl"
FIFO="/tmp/agy_stdin_$$"
mkfifo "$FIFO"
exec 3<>"$FIFO"

# 4. Launch agy-real with streaming I/O bound to the FIFO and capture stream output
"$REAL_AGY" "${CLI_ARGS[@]}" <"$FIFO" 3>&- > >(tee "$RAW_STREAM") &
AGY_PID=$!

STITCHED=0

# ==============================================================================
# Post-Run Trajectory Stitcher (100% Pure Shell via jq)
# ==============================================================================
# Harbor collects a single trajectory from ~/.gemini/antigravity-cli/brain via:
#   src=$(find "$HOME/.gemini/antigravity-cli/brain" -type f -name 'transcript_full.jsonl' ... | sort -nr | head -n1)
# Antigravity CLI isolates child subagents in separate UUID directories.
# To show the complete end-to-end multi-agent execution in Harbor UI, we stitch
# all subagent transcripts chronologically into a unified trajectory file, ensure
# it has the newest timestamp, and sync it across transcript files.
stitch_trajectories() {
  if [ "$STITCHED" -eq 1 ]; then
    return 0
  fi
  STITCHED=1

  local brain_dir="${HOME}/.gemini/antigravity-cli/brain"
  if [ ! -d "$brain_dir" ]; then
    return 0
  fi

  local files=()
  while IFS= read -r f; do
    [ -f "$f" ] && files+=("$f")
  done < <(find "$brain_dir" -type f -name 'transcript_full.jsonl' ! -path '*/unified/*' 2>/dev/null)

  if [ "${#files[@]}" -eq 0 ]; then
    return 0
  fi

  # Discover subagent conversation metadata from the stream logs or transcripts
  declare -A conv_meta=()
  local stream
  for stream in "$RAW_STREAM" "/logs/agent/antigravity-stream.jsonl"; do
    if [ -f "$stream" ]; then
      while IFS=$'\t' read -r cid tname; do
        if [ -n "$cid" ] && [ -n "$tname" ]; then
          conv_meta["$cid"]="${tname}|1"
        fi
      done < <(grep '"subagent_info"' "$stream" 2>/dev/null | jq -r '
        .step_update?.subagent_info?.subagents[]?
        | [.conversation_id, (.type_name // .role // "subagent")]
        | @tsv
      ' 2>/dev/null || true)
    fi
  done

  # Also inspect transcripts for conversationId references
  for f in "${files[@]}"; do
    while IFS= read -r cid; do
      if [ -n "$cid" ] && [ -z "${conv_meta[$cid]:-}" ]; then
        conv_meta["$cid"]="subagent|1"
      fi
    done < <(grep -o '"conversationId": *"[^"]*"' "$f" 2>/dev/null | cut -d'"' -f4 || true)
  done

  local db_file="${HOME}/.gemini/antigravity-cli/conversation_summaries.db"

  local tagged_files=()

  # Tag each conversation with its agent identity and depth
  for f in "${files[@]}"; do
    local conv_id
    conv_id=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    local agent_name=""
    local depth=0

    if [ -f "$db_file" ]; then
      local meta
      meta=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COALESCE(NULLIF(agent_name, ''), NULLIF('${AGENT_PERSONA:-}', '')) || '|' || COALESCE(nesting_depth, 0) FROM conversation_summaries WHERE conversation_id = '$conv_id';" 2>/dev/null || true)
      if [ -n "$meta" ]; then
        agent_name="${meta%|*}"
        depth="${meta##*|}"
      fi
    fi
    if [ -z "$agent_name" ] && [ -n "${conv_meta[$conv_id]:-}" ]; then
      local meta="${conv_meta[$conv_id]}"
      agent_name="${meta%|*}"
      depth="${meta##*|}"
    elif [ -z "$agent_name" ]; then
      agent_name="${AGENT_PERSONA:-}"
      depth=0
    fi
    [ -z "$depth" ] && depth=0

    local tag="[Depth ${depth}]"
    if [ "$depth" -eq 0 ]; then
      if [ -n "$agent_name" ]; then
        tag="[Depth 0 | ${agent_name}]"
      fi
    else
      local display_name="${agent_name:-subagent}"
      tag="[Depth ${depth} | ${display_name}]"
    fi

    local tagged_tmp="${f}.tagged"
    jq -R -c --arg agent "${agent_name:-agent}" --argjson depth "$depth" --arg tag "$tag" '
      fromjson?
      | select(type == "object" and .created_at != null and .type != null)
      | .agent = $agent
      | .depth = $depth
      | if (.content | type == "string" and length > 0) then
          (if (.content | startswith("[Depth ")) then . else .content = ($tag + " " + .content) end)
        elif (.tool_calls != null or .thinking != null) then
          .content = $tag
        else
          .
        end
      | if (.thinking | type == "string" and length > 0) then
          (if (.thinking | startswith("[Depth ")) then . else .thinking = ($tag + " " + .thinking) end)
        else
          .
        end
    ' "$f" > "$tagged_tmp" 2>/dev/null || true
    tagged_files+=("$tagged_tmp")
  done

  # If single conversation (e.g. vanilla AGY baseline), overwrite in-place
  if [ "${#files[@]}" -eq 1 ]; then
    local single_file="${files[0]}"
    local tagged_single="${tagged_files[0]}"
    if [ -s "$tagged_single" ]; then
      mv "$tagged_single" "$single_file"
      local compact="${single_file%/transcript_full.jsonl}/transcript.jsonl"
      if [ -f "$compact" ]; then
        cp "$single_file" "$compact" 2>/dev/null || true
      fi
      log_supervisor "[AgySupervisor] Tagged single session trajectory with [Depth 0]."
    else
      rm -f "$tagged_single" 2>/dev/null || true
    fi
    return 0
  fi

  log_supervisor "[AgySupervisor] Found ${#files[@]} agent transcripts. Stitching into unified trajectory via jq..."

  local unified_dir="${brain_dir}/unified/.system_generated/logs"
  mkdir -p "$unified_dir"
  local unified_file="${unified_dir}/transcript_full.jsonl"

  # Merge all tagged records, sort chronologically by created_at, and renumber step_index 0..N-1
  cat "${tagged_files[@]}" | jq -c -s '
    sort_by(.created_at)
    | to_entries
    | map(.value + {step_index: .key})[]
  ' > "$unified_file.tmp" 2>/dev/null || true

  rm -f "${tagged_files[@]}" 2>/dev/null || true

  if [ -s "$unified_file.tmp" ]; then
    mv "$unified_file.tmp" "$unified_file"
    touch "$unified_file"
    for f in "${files[@]}"; do
      cp "$unified_file" "$f" 2>/dev/null || true
      local compact="${f%/transcript_full.jsonl}/transcript.jsonl"
      if [ -f "$compact" ]; then
        cp "$unified_file" "$compact" 2>/dev/null || true
      fi
    done
    log_supervisor "[AgySupervisor] Stitched unified trajectory generated: $(wc -l < "$unified_file") steps with agent tags."
  else
    rm -f "$unified_file.tmp"
  fi
}

# Cleanup trap to ensure FIFO, background process, and trajectory stitching are handled
# shellcheck disable=SC2317
cleanup() {
  exec 3>&- 2>/dev/null || true
  rm -f "$FIFO" 2>/dev/null || true
  if [ -n "${AGY_PID:-}" ] && kill -0 "$AGY_PID" 2>/dev/null; then
    kill -TERM "$AGY_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$AGY_PID" 2>/dev/null; then
      kill -KILL "$AGY_PID" 2>/dev/null || true
    fi
  fi
  stitch_trajectories || true
  rm -f "$RAW_STREAM" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 5. Dispatch initial prompt event over the open pipe (RFC 8259 JSON via jq)
if [ -n "$PROMPT" ]; then
  jq -nc --arg p "$PROMPT" '{"event":"user","message":{"content":$p}}' >&3
fi

# 6. Active Agent Completion Checker
check_work_completed() {
  # 1. Primary check: Native stream event completion in RAW_STREAM or /logs/agent/antigravity-stream.jsonl
  local stream
  for stream in "$RAW_STREAM" "/logs/agent/antigravity-stream.jsonl"; do
    if [ -f "$stream" ] && grep -q '"event":"result"' "$stream" 2>/dev/null; then
      return 0
    fi
  done

  # 2. Fallback check: conversation_summaries.db (e.g. AGY full desktop/daemon)
  local db_file="${HOME}/.gemini/antigravity-cli/conversation_summaries.db"
  if [ -f "$db_file" ]; then
    local total
    total=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COUNT(*) FROM conversation_summaries;" 2>/dev/null || echo "0")
    if [ "${total:-0}" -gt 0 ]; then
      local active
      active=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COUNT(*) FROM conversation_summaries WHERE status LIKE '%RUNNING%' OR not_fully_idle != 0;" 2>/dev/null || echo "1")
      if [ "${active:-1}" -eq 0 ]; then
        return 0
      fi
    fi
  fi

  return 1
}

# 7. Monitoring Loop: Keep looking for active working agents to complete their work
WORK_DONE=0
while kill -0 "$AGY_PID" 2>/dev/null; do
  sleep 2
  if check_work_completed; then
    log_supervisor "[AgySupervisor] All working agents completed their work. Concluding session."
    WORK_DONE=1
    break
  fi
done

# 8. Close stdin to signal EOF to agy-real and wait for clean exit
exec 3>&- 2>/dev/null || true
rm -f "$FIFO" 2>/dev/null || true

set +e
WAIT_COUNT=0
while kill -0 "$AGY_PID" 2>/dev/null && [ "$WAIT_COUNT" -lt 15 ]; do
  sleep 1
  WAIT_COUNT=$((WAIT_COUNT + 1))
done

if kill -0 "$AGY_PID" 2>/dev/null; then
  kill -TERM "$AGY_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$AGY_PID" 2>/dev/null; then
    kill -KILL "$AGY_PID" 2>/dev/null || true
  fi
fi

AGY_EXIT_CODE=0
wait "$AGY_PID" 2>/dev/null || AGY_EXIT_CODE=$?

stitch_trajectories || true

if [ "$WORK_DONE" -eq 1 ]; then
  exit 0
fi
exit "$AGY_EXIT_CODE"
