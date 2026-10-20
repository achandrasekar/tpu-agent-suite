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
# Antigravity CLI Session Supervisor Environment Hook (/usr/local/bin/session_supervisor_hook.sh)
# ==============================================================================
# Sourced via BASH_ENV in evaluation containers.
# When Harbor installs native agy to /root/.local/bin/agy (or ~/.local/bin/agy),
# this moves it to agy-real in the same directory and replaces it with
# /usr/local/bin/session_supervisor.sh. /usr/local/bin is root-only (see
# eval/adapter/kube_gate_setup.sh), so agy-real stays in the user's directory.
# ==============================================================================
# `pip install` as `agent` puts commands in ~/.local/bin.
case ":$PATH:" in
  *":${HOME:-/root}/.local/bin:"*) ;;
  *) export PATH="${HOME:-/root}/.local/bin:$PATH" ;;
esac
for bin in /root/.local/bin/agy "${HOME:-/root}/.local/bin/agy" /usr/local/bin/agy; do
  if [ -f "$bin" ] && [ ! -f "${bin}-real" ] && [ -w "$(dirname "$bin")" ]; then
    mv "$bin" "${bin}-real"
    cp /usr/local/bin/session_supervisor.sh "$bin"
    chmod +x "$bin" "${bin}-real"
    export AGY_ADC_AUTH=1
    # Only when an ADC file is mounted: pointing this at a missing file makes
    # ADC fail instead of falling back to the pod's GKE metadata server.
    if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f /root/.config/gcloud/application_default_credentials.json ]; then
      export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
    fi
    export GOOGLE_CLOUD_QUOTA_PROJECT="${GOOGLE_CLOUD_QUOTA_PROJECT:-${VERTEXAI_PROJECT:-tpu-nexus-767207}}"
    unset BASH_ENV
    break
  fi
done
