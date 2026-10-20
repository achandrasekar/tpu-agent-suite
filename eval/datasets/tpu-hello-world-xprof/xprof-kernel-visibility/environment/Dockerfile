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

FROM us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host@sha256:7ff58a98ae2655dbcf25fa0aeb3e8081281bfb3061f97581a98951e98371fff4

ARG XPROF_DEMO_COMMIT=317ed23c45ae43d8953963ec87f8330c661c11e4
RUN mkdir -p /workspace/profiles/v6e-4-training /workspace/profiles/tpu-training \
 && curl -fsSL -o /workspace/profiles/v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb \
      "https://raw.githubusercontent.com/openxla/xprof/${XPROF_DEMO_COMMIT}/demo/plugins/profile/v6e-4-training/t1v-n-9bfa07b4-w-0.xplane.pb" \
 && curl -fsSL -o /workspace/profiles/tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb \
      "https://raw.githubusercontent.com/openxla/xprof/${XPROF_DEMO_COMMIT}/demo/plugins/profile/tpu-training/gke-tpu-b309f56b-rq5s.xplane.pb"

WORKDIR /workspace
