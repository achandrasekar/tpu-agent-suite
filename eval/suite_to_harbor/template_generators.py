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
"""String template generators for Harbor task bundles."""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from inspect import cleandoc
from typing import Any

from . import schema


def toml_escape(val: Any) -> str:
    """Formats Python primitives and collections into valid TOML values."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple)):
        return f"[{', '.join(toml_escape(x) for x in val)}]"
    if isinstance(val, dict):
        inner = ", ".join(f"{k} = {toml_escape(v)}" for k, v in val.items()
                          if v is not None)
        return f"{{{inner}}}"
    return json.dumps(str(val))


def generate_dockerfile(env: schema.EnvironmentConfig,
                        has_data: bool = False) -> str:
    """Generates a clean, minimal environment/Dockerfile."""
    pip_pkgs = ["harbor-rewardkit"
                ] + [p for p in env.pip_packages if p != "harbor-rewardkit"]
    quoted_pip = " ".join(shlex.quote(p) for p in pip_pkgs)

    combined_apt = list(
        dict.fromkeys([
            "curl", "ca-certificates", "gnupg", "jq", "sudo", *env.apt_packages
        ]))
    apt_lines = "\n".join(f"    {shlex.quote(pkg)} \\" for pkg in combined_apt)

    data_copy_section = ""
    if has_data:
        data_copy_section = cleandoc(f"""
            COPY data/ {env.workdir.rstrip('/')}/
            RUN chmod -R 777 {env.workdir}
        """) + "\n"

    dockerfile = cleandoc(f"""
        FROM --platform=linux/amd64 {env.base_image}

        # System packages
        RUN apt-get update && apt-get install -y --no-install-recommends \\
        {apt_lines}
            && rm -rf /var/lib/apt/lists/*

        # Create non-root agent user (uid 1000) and kube-gate system user (uid 1001)
        RUN groupadd -g 1000 agent && \\
            useradd -u 1000 -g agent -M -d /root -s /bin/bash agent && \\
            groupadd -g 1001 kube-gate && \\
            useradd -u 1001 -g kube-gate --system --no-create-home -s /usr/sbin/nologin kube-gate && \\
            mkdir -p /etc/kube-gate /usr/libexec /root {env.workdir} && \\
            chown -R kube-gate:kube-gate /etc/kube-gate && \\
            chmod 0700 /etc/kube-gate && \\
            chown -R agent:agent /root {env.workdir}

        # Install kubectl for Unified Compute cluster access
        RUN ARCH=$(dpkg --print-architecture) && \\
            curl -fsSL "https://dl.k8s.io/release/v1.31.0/bin/linux/${{ARCH}}/kubectl" -o /usr/local/bin/kubectl && \\
            chmod +x /usr/local/bin/kubectl

        # Install gke-gcloud-auth-plugin for GKE cluster authentication
        RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \\
            > /etc/apt/sources.list.d/google-cloud-sdk.list && \\
            curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \\
            | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg && \\
            apt-get update && apt-get install -y --no-install-recommends google-cloud-cli-gke-gcloud-auth-plugin \\
            && rm -rf /var/lib/apt/lists/*

        # Python dependencies
        RUN pip install --no-cache-dir {quoted_pip}

        WORKDIR {env.workdir}
    """)

    return f"{dockerfile}\n\n{data_copy_section}".rstrip() + "\n"


def generate_task_toml(
    case: schema.TestCase,
    suite_name: str,
    env: schema.EnvironmentConfig,
    prompt_text: str | None = None,
) -> str:
    """Generates task.toml configuration for Harbor."""
    raw_prompt = prompt_text if prompt_text is not None else case.prompt
    desc = (raw_prompt.strip().splitlines()[0]
            if raw_prompt and raw_prompt.strip() else case.name)

    return cleandoc(f"""
        version = "1.0"

        artifacts = [
          {{ source = {toml_escape(env.workdir)}, destination = 'workspace' }},
          {{ source = "/root/.tpu-agent-suite/runs", destination = "tpu_agent_suite_runs" }},
        ]

        [task]
        name = {toml_escape(f"{suite_name}/{case.name}")}
        description = {toml_escape(desc)}

        [metadata]
        category = {toml_escape(suite_name)}

        [agent]
        timeout_sec = {float(case.timeout_sec):.1f}

        [verifier]
        timeout_sec = 300.0

        [environment]
        workdir = {toml_escape(env.workdir)}
        cpus = {env.cpus}
        memory_mb = {env.memory_mb}
        storage_mb = {env.storage_mb}
    """) + "\n" + _env_line(case.env)


def _env_line(variables: dict[str, str]) -> str:
    """`[environment] env`, as an inline table; nothing when there are none."""
    if not variables:
        return ""
    pairs = ", ".join(f"{name} = {toml_escape(value)}"
                      for name, value in sorted(variables.items()))
    return f"env = {{ {pairs} }}\n"


def generate_reward_toml() -> str:
    """Generates tests/reward.toml configuration for RewardKit."""
    return cleandoc("""
        [[reward]]
        name = "reward"
        aggregation = "weighted_mean"
    """) + "\n"


def generate_judge_toml(expectations: Sequence[Any],
                        judge_model: str) -> str | None:
    """Generates tests/expectations/judge.toml for LLM rubric evaluation."""
    norm = [
        schema.ExpectationCriterion.coerce(e) for e in expectations
        if str(e).strip()
    ]
    if not norm:
        return None

    criteria_blocks = "\n\n".join(
        cleandoc(f"""
            [[criterion]]
            description = {toml_escape(exp.criterion)}
            type = {toml_escape(exp.type)}
            weight = {float(exp.weight):.1f}
        """) for exp in norm)

    # Dedent the template before inserting the blocks: their lines after the
    # first carry no indent, so dedenting afterwards would find none to strip.
    judge = cleandoc(f"""
        [judge]
        judge = {toml_escape(judge_model)}
        atif-trajectory = "/logs/agent/trajectory.json"
    """)
    scoring = cleandoc("""
        [scoring]
        aggregation = "weighted_mean"
    """)
    return f"{judge}\n\n{criteria_blocks}\n\n{scoring}\n"


def generate_test_sh() -> str:
    """Generates executable tests/test.sh with transient retry handling."""
    # TODO: kevinaram - Consider utilizing .sh file when setup grows.
    return cleandoc("""
        #!/bin/bash
        export LITELLM_NUM_RETRIES=8
        export LITELLM_RETRY_AFTER=3

        MAX_RETRIES=5
        RETRY_DELAY=3

        for ((i=1; i<=MAX_RETRIES; i++)); do
          rewardkit /tests --workspace /workspace && exit 0
          EXIT_CODE=$?
          if [ "$i" -lt "$MAX_RETRIES" ]; then
            echo "[test.sh] Verifier transient error ($EXIT_CODE). Retrying in ${RETRY_DELAY}s ($i/$MAX_RETRIES)..."
            sleep "$RETRY_DELAY"
            RETRY_DELAY=$((RETRY_DELAY * 2))
          fi
        done

        exit "$EXIT_CODE"
    """) + "\n"
