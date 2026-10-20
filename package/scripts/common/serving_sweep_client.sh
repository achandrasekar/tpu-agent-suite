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
# One shape of a serving sweep (serving_sweep.py): serving_benchmark.py with
# the flags it is given. Its exit code is kept in a file rather than returned,
# so the job ends COMPLETED and its files come back however the shape went.
# Submitted without --interpreter: for a .sh recipe gke_job exports that as
# TPU_AGENT_SUITE_PYTHON, and the client would then run under bash.
PY="${TPU_AGENT_SUITE_PYTHON:-$(command -v python3)}"
mkdir -p serving_bench
"$PY" serving_benchmark.py "$@" > serving_bench/client.json
echo $? > serving_bench/exit_code
exit 0
