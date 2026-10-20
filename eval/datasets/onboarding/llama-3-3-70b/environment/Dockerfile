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

# The nightly the instruction names, plus what an agent run needs on top.
FROM us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-dev:nightly-20260804

# node for the agent CLI Harbor installs; git because the deliverable is a
# local commit history; jq for the session supervisor hook;
# google-cloud-cli-gke-gcloud-auth-plugin for kubectl GKE authentication.
RUN apt-get update && apt-get install -y --no-install-recommends \
      nodejs npm git ca-certificates curl gnupg jq \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends \
      google-cloud-cli-gke-gcloud-auth-plugin \
    && rm -rf /var/lib/apt/lists/*

# tokamax needs jax >= 0.11, which the nightly's flax 0.12.0 cannot load.
RUN pip install --no-cache-dir tokamax "flax>=0.12.10"

WORKDIR /workspace
