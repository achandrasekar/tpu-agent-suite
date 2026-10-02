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
"""llm-d releases on GKE: the runtime's `deploy` command.

A release is one incumbent `llmd-config.json` rendered against a serving
recipe and a workload spec into self-contained manifests, applied to a
namespace, benchmarked with inference-perf, reconfigured in place where
that is cheap, and torn down by one label. The agents in `tpu-llmd` drive
these verbs; nothing in them reaches a cluster any other way.
"""

from .cli import handle_deploy_cli, register_deploy_parser
from .config import RELEASE_BACKEND, DeployEnv, load_env
from .kube import CommandError, DeployError
from .release import wrap_reap_resolver

__all__ = [
    "CommandError",
    "DeployEnv",
    "DeployError",
    "RELEASE_BACKEND",
    "handle_deploy_cli",
    "load_env",
    "register_deploy_parser",
    "wrap_reap_resolver",
]
