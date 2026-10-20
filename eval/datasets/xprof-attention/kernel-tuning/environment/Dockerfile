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

# The environment only: the TPU agent suite package is compiled to dist/ and
# mounted at run time. Harbor builds with environment/ as the context, so every
# COPY source must sit beside this file.
FROM us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host@sha256:7ff58a98ae2655dbcf25fa0aeb3e8081281bfb3061f97581a98951e98371fff4

WORKDIR /workspace

# JAX on CPU for both arms: the verifier checks numerics with
# `xprof verify_numerical_parity`, which needs no chip. JAX_VERSION matches the
# TPU build in run_on_tpu.sh so a toolchain skew cannot pass for a numerical
# one. xprof comes from nexus-host, which installs requirements-client.txt and
# pins what it installed in /etc/nexus-host/constraints.txt; installing under
# those constraints keeps it. The verifier records the version it measured with.
ARG JAX_VERSION=0.11.0
RUN pip install --no-cache-dir -c /etc/nexus-host/constraints.txt --break-system-packages --index-url https://pypi.org/simple \
      "jax[cpu]==${JAX_VERSION}" \
  || pip install --no-cache-dir -c /etc/nexus-host/constraints.txt --index-url https://pypi.org/simple \
      "jax[cpu]==${JAX_VERSION}"

# Fail the build, not the run, on an older JAX or a missing xprof CLI.
RUN python3 -c "import jax; \
v=tuple(int(p) for p in jax.__version__.split('.')[:2]); \
assert v >= (0, 11), f'jax {jax.__version__} is older than the verifier reference (0.11)'" \
 && xprof --help 2>&1 | grep -q get_kernel_stats

# The subject of the task, the same for both arms.
COPY attention.py /workspace/attention.py
COPY run_on_tpu.sh /workspace/run_on_tpu.sh
RUN chmod +x /workspace/run_on_tpu.sh
