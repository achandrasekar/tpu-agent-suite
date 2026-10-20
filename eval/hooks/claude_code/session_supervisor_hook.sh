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
# Claude Code Session Supervisor Environment Hook (/usr/local/bin/claude_session_supervisor_hook.sh)
# ==============================================================================
# Sourced via BASH_ENV in evaluation containers.
# When Harbor installs native claude CLI, this moves it to claude-real in the
# same directory and replaces it with /usr/local/bin/claude_session_supervisor.sh.
# /usr/local/bin is root-only (see eval/adapter/kube_gate_setup.sh), so
# claude-real stays in the user's directory.
# ==============================================================================

SUPERVISOR="/usr/local/bin/claude_session_supervisor.sh"

# `pip install` as `agent` puts commands in ~/.local/bin.
case ":$PATH:" in
  *":${HOME:-/root}/.local/bin:"*) ;;
  *) export PATH="${HOME:-/root}/.local/bin:$PATH" ;;
esac

for bin in /root/.local/bin/claude /usr/local/bin/claude "${HOME}/.local/bin/claude"; do
  if [ -f "$bin" ] && [ ! -f "${bin}-real" ] && [ -w "$(dirname "$bin")" ]; then
    mv "$bin" "${bin}-real"
    cp "$SUPERVISOR" "$bin"
    chmod +x "$bin" "${bin}-real"
    break
  fi
done
