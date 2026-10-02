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
"""What `deploy` reads from the package environment, and the release registry.

The environment file already carries the GKE facts the compute drivers use
(`compute.params`: namespace, kubeconfig, accelerator, topology, project).
`deploy` adds one optional `llmd` section for what only an llm-d release
needs: the production namespace, the gateway mode, the HF token secret, and
where Prometheus is. Both are folded into one `DeployEnv` here so no verb
reads the raw document.

A release is found by name through the compute job store: `apply` records a
`JobRecord` with backend `gke_release`, and `status`, `bench`, `reconfigure`
and `teardown` look it up there. That is also how the session-end reaper
finds a study release nobody tore down.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

try:
    from framework.compute import config as compute_config
    from framework.compute.drivers import jobs
except ImportError:
    try:
        from ..compute import config as compute_config  # type: ignore
        from ..compute.drivers import jobs  # type: ignore
    except ImportError:  # bundled runtime: `deploy` sits beside `compute`
        from compute import config as compute_config  # type: ignore
        from compute.drivers import jobs  # type: ignore

from . import pins
from .kube import DeployError, Kube

# The job-store backend name a release is filed under. Not a compute driver:
# nothing can `compute run` against it, but `reap_jobs` resolves it through
# `release.wrap_reap_resolver` so a leftover study release is torn down with
# the run's other jobs.
RELEASE_BACKEND = "gke_release"

GATEWAY_MODES = ("standalone", "gke")

DEFAULT_HF_TOKEN_SECRET = "llm-d-hf-token"
DEFAULT_MAX_CHIPS = 32


@dataclasses.dataclass
class DeployEnv:
    """Everything a deploy verb needs to know about where it is deploying."""

    namespace: str = "default"
    production_namespace: str = ""
    gateway_mode: str = "standalone"
    kubeconfig: str | None = None
    kube_context: str | None = None
    project: str = ""
    zone: str = ""
    tpu_accelerator: str = ""
    tpu_topology: str = ""
    hf_token_secret: str = DEFAULT_HF_TOKEN_SECRET
    prometheus_url: str = ""
    prometheus_project: str = ""
    inference_perf_image: str = pins.INFERENCE_PERF_IMAGE
    max_chips_default: int = DEFAULT_MAX_CHIPS
    compute_backend: str = ""
    # Pod sizes for the non-accelerator parts of a release (bench Job, EPP,
    # Envoy proxy). The defaults are the llm-d guides' sizes; a shared
    # cluster whose CPU pools are nearly full sets smaller ones here rather
    # than leaving every bench Job Pending.
    sizing: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_environment(cls, env_data: dict[str, Any]) -> DeployEnv:
        """Fold `compute.params` and the optional `llmd` section together."""
        backend, params = compute_config.parse_compute_section(env_data
                                                              or {})
        llmd = dict((env_data or {}).get("llmd") or {})
        namespace = str(llmd.get("namespace") or params.get("namespace")
                        or "default")
        mode = str(llmd.get("gateway_mode") or "standalone")
        if mode not in GATEWAY_MODES:
            raise DeployError(
                f"llmd.gateway_mode must be one of {GATEWAY_MODES}, got"
                f" {mode!r}",
                error_code="INVALID_FIELD")
        sizing = llmd.get("sizing") or {}
        if not isinstance(sizing, dict):
            raise DeployError("llmd.sizing must be an object",
                              error_code="INVALID_FIELD")
        return cls(
            namespace=namespace,
            production_namespace=str(
                llmd.get("production_namespace") or ""),
            gateway_mode=mode,
            kubeconfig=params.get("kubeconfig") or None,
            kube_context=llmd.get("kube_context") or None,
            project=str(params.get("project") or ""),
            zone=str(params.get("zone") or ""),
            tpu_accelerator=str(llmd.get("tpu_accelerator")
                                or params.get("tpu_accelerator") or ""),
            tpu_topology=str(llmd.get("tpu_topology")
                             or params.get("tpu_topology") or ""),
            hf_token_secret=str(
                llmd.get("hf_token_secret") or DEFAULT_HF_TOKEN_SECRET),
            prometheus_url=str(llmd.get("prometheus_url") or ""),
            prometheus_project=str(
                llmd.get("prometheus_project") or params.get("project")
                or ""),
            inference_perf_image=str(
                llmd.get("inference_perf_image") or pins.INFERENCE_PERF_IMAGE),
            max_chips_default=int(
                llmd.get("max_chips_default") or DEFAULT_MAX_CHIPS),
            compute_backend=str(backend or ""),
            sizing=dict(sizing),
        )

    def kube(self, *, production: bool = False, runner: Any = None) -> Kube:
        """A `Kube` bound to the study namespace, or production's."""
        namespace = self.namespace
        if production:
            namespace = self.production_namespace or self.namespace
        kwargs: dict[str, Any] = {
            "namespace": namespace,
            "kubeconfig": self.kubeconfig,
            "context": self.kube_context,
        }
        if runner is not None:
            kwargs["runner"] = runner
        return Kube(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_env(workspace: pathlib.Path | str | None,
             package: str | None) -> DeployEnv:
    """The deploy environment for `package`, from its environment.json.

    `package` is optional here, unlike for compute: `render` and the golden
    tests need no cluster, and a missing package then simply yields defaults.
    """
    data = compute_config.find_environment_data(workspace, package)
    return DeployEnv.from_environment(data)


# -- release registry --------------------------------------------------------


def find_release_record(name: str) -> jobs.JobRecord | None:
    """The newest non-terminal job record filed for release `name`."""
    matches = [
        record for record in jobs.list_jobs()
        if record.backend == RELEASE_BACKEND and record.job_name == name
    ]
    live = [
        record for record in matches
        if record.status not in jobs.TERMINAL_STATUSES
    ]
    pool = live or matches
    return pool[-1] if pool else None


def release_dir_for(name: str,
                    release_dir: str | None = None) -> pathlib.Path:
    """Locate a release's on-disk directory by explicit path or by name."""
    if release_dir:
        path = pathlib.Path(release_dir)
        if not (path / "release.json").is_file():
            raise DeployError(f"{path} holds no release.json",
                              error_code="RELEASE_NOT_FOUND")
        return path
    record = find_release_record(name)
    if record is None or not record.extra.get("release_dir"):
        raise DeployError(
            f"no applied release named {name!r}; pass --release DIR for a"
            " rendered-but-unapplied one",
            error_code="RELEASE_NOT_FOUND")
    path = pathlib.Path(record.extra["release_dir"])
    if not (path / "release.json").is_file():
        raise DeployError(
            f"release {name!r} points at {path}, which holds no release.json",
            error_code="RELEASE_NOT_FOUND")
    return path
