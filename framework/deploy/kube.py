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
"""The one place `deploy` shells out to kubectl and helm.

Every verb goes through `Kube`, which carries the kubeconfig, context and
namespace once, and through a `runner` that tests replace with a fake. No
other module in `deploy/` imports subprocess, so a verb that reaches the
cluster some other way is a review finding rather than a surprise.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from typing import Any, Callable, Sequence

Runner = Callable[..., subprocess.CompletedProcess]

READ_TIMEOUT_SEC = 60
WRITE_TIMEOUT_SEC = 300
HELM_TIMEOUT_SEC = 900


class DeployError(Exception):
    """A deploy verb could not do what it was asked.

    `error_code` is a stable machine-readable name; `detail` is for a person.
    The CLI prints both, as JSON under `--json`.
    """

    def __init__(self, message: str, error_code: str = "DEPLOY_ERROR",
                 detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out = {"error": self.error_code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out


class CommandError(DeployError):
    """kubectl or helm exited non-zero."""

    def __init__(self, argv: Sequence[str], stderr: str) -> None:
        tool = argv[0] if argv else "?"
        shown = " ".join(argv[:6]) + (" ..." if len(argv) > 6 else "")
        super().__init__(f"{tool} failed: {shown}",
                         error_code=f"{tool.upper()}_FAILED",
                         detail=(stderr or "").strip())
        self.argv = list(argv)
        self.stderr = stderr or ""


def default_runner(cmd: list[str],
                   *,
                   stdin: str | None = None,
                   timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a command, capturing text output, never raising on exit code."""
    return subprocess.run(cmd,
                          input=stdin,
                          capture_output=True,
                          text=True,
                          timeout=timeout,
                          check=False)


@dataclasses.dataclass
class Kube:
    """kubectl and helm bound to one cluster, context and namespace."""

    namespace: str = "default"
    kubeconfig: str | None = None
    context: str | None = None
    runner: Runner = default_runner

    # -- plumbing ----------------------------------------------------------

    def _global_flags(self) -> list[str]:
        flags: list[str] = []
        if self.kubeconfig:
            flags += ["--kubeconfig", self.kubeconfig]
        if self.context:
            flags += ["--context", self.context]
        return flags

    def _run(self,
             cmd: list[str],
             *,
             stdin: str | None = None,
             check: bool = True,
             timeout: int | None = None) -> subprocess.CompletedProcess:
        try:
            proc = self.runner(cmd, stdin=stdin, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise CommandError(cmd, f"no answer within {timeout}s") from exc
        except FileNotFoundError as exc:
            raise DeployError(f"{cmd[0]} is not on PATH",
                              error_code="TOOL_MISSING",
                              detail=str(exc)) from exc
        if check and proc.returncode != 0:
            raise CommandError(cmd, proc.stderr)
        return proc

    # -- kubectl -----------------------------------------------------------

    def kubectl(self,
                argv: Sequence[str],
                *,
                stdin: str | None = None,
                check: bool = True,
                namespaced: bool = True,
                timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run one kubectl command in the bound namespace."""
        cmd = ["kubectl", *self._global_flags()]
        if namespaced:
            cmd += ["-n", self.namespace]
        cmd += list(argv)
        return self._run(cmd,
                         stdin=stdin,
                         check=check,
                         timeout=timeout or READ_TIMEOUT_SEC)

    def kubectl_json(self,
                     argv: Sequence[str],
                     *,
                     namespaced: bool = True,
                     check: bool = True) -> dict[str, Any]:
        """`kubectl ... -o json`, parsed. An empty answer is an empty dict."""
        proc = self.kubectl([*argv, "-o", "json"],
                            namespaced=namespaced,
                            check=check)
        if proc.returncode != 0:
            return {}
        text = (proc.stdout or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError as exc:
            raise DeployError(f"kubectl {' '.join(argv)} returned unparseable"
                              " JSON",
                              error_code="KUBECTL_BAD_OUTPUT",
                              detail=text[:500]) from exc

    def apply(self, manifest_text: str, *, timeout: int | None = None
              ) -> subprocess.CompletedProcess:
        """`kubectl apply -f -` with the manifest on stdin."""
        return self.kubectl(["apply", "-f", "-"],
                            stdin=manifest_text,
                            timeout=timeout or WRITE_TIMEOUT_SEC)

    def delete(self,
               argv: Sequence[str],
               *,
               ignore_not_found: bool = True,
               wait: bool = False,
               timeout: int | None = None) -> subprocess.CompletedProcess:
        """`kubectl delete ...`; absent objects are not an error by default."""
        extra = []
        if ignore_not_found:
            extra.append("--ignore-not-found=true")
        extra.append(f"--wait={'true' if wait else 'false'}")
        return self.kubectl(["delete", *argv, *extra],
                            timeout=timeout or WRITE_TIMEOUT_SEC)

    def exists(self, kind: str, name: str, *, namespaced: bool = True) -> bool:
        proc = self.kubectl(["get", kind, name],
                            check=False,
                            namespaced=namespaced)
        return proc.returncode == 0

    def logs(self, selector: str, *, container: str | None = None,
             tail: int = 200) -> str:
        argv = ["logs", "-l", selector, "--tail", str(tail), "--prefix=true",
                "--all-containers=false"]
        if container:
            argv += ["-c", container]
        proc = self.kubectl(argv, check=False)
        return (proc.stdout or "") + (proc.stderr or "")

    # -- pod proxy -----------------------------------------------------------
    # The API server proxies HTTP to a pod port
    # (/api/v1/namespaces/<ns>/pods/<pod>:<port>/proxy/<path>), which reaches
    # an engine's /metrics or /reset_prefix_cache from anywhere kubectl
    # works: no exec (distroless images have no shell), no port-forward,
    # no Prometheus. `kubectl get --raw` is a GET, `create --raw` a POST.

    def pod_proxy_get(self, pod: str, port: int, path: str,
                      *, timeout: int | None = None) -> str | None:
        """Body of GET <pod>:<port>/<path>, or None when it failed."""
        raw = (f"/api/v1/namespaces/{self.namespace}/pods/{pod}:{port}"
               f"/proxy/{path.lstrip('/')}")
        proc = self.kubectl(["get", "--raw", raw], check=False,
                            namespaced=False, timeout=timeout or 30)
        if proc.returncode != 0:
            return None
        return proc.stdout or ""

    def pod_proxy_post(self, pod: str, port: int, path: str,
                       body: str = "{}", *, timeout: int | None = None
                       ) -> tuple[bool, str]:
        """POST <pod>:<port>/<path>; (ok, body or stderr)."""
        raw = (f"/api/v1/namespaces/{self.namespace}/pods/{pod}:{port}"
               f"/proxy/{path.lstrip('/')}")
        proc = self.kubectl(["create", "--raw", raw, "-f", "-"],
                            stdin=body, check=False, namespaced=False,
                            timeout=timeout or 60)
        return proc.returncode == 0, (proc.stdout or proc.stderr or "")

    # -- helm --------------------------------------------------------------

    def helm(self,
             argv: Sequence[str],
             *,
             check: bool = True,
             timeout: int | None = None) -> subprocess.CompletedProcess:
        cmd = ["helm", *self._helm_global_flags(), *argv]
        return self._run(cmd, check=check, timeout=timeout or HELM_TIMEOUT_SEC)

    def _helm_global_flags(self) -> list[str]:
        flags = ["-n", self.namespace]
        if self.kubeconfig:
            flags += ["--kubeconfig", self.kubeconfig]
        if self.context:
            flags += ["--kube-context", self.context]
        return flags

    def helm_upgrade_install(self,
                             release: str,
                             chart: str,
                             version: str,
                             values_files: Sequence[str],
                             *,
                             wait: bool = False,
                             timeout_sec: int = HELM_TIMEOUT_SEC
                             ) -> subprocess.CompletedProcess:
        argv = ["upgrade", "--install", release, chart, "--version", version]
        for path in values_files:
            argv += ["-f", path]
        if wait:
            argv += ["--wait", "--timeout", f"{timeout_sec}s"]
        return self.helm(argv, timeout=timeout_sec + 60)

    def helm_uninstall(self, release: str) -> subprocess.CompletedProcess:
        proc = self.helm(["uninstall", release, "--wait"], check=False)
        if proc.returncode != 0 and "not found" not in (proc.stderr or ""):
            raise CommandError(["helm", "uninstall", release], proc.stderr)
        return proc

    def helm_release_exists(self, release: str) -> bool:
        proc = self.helm(["status", release], check=False)
        return proc.returncode == 0


def tools_present() -> dict[str, str | None]:
    """Where the tools a deploy needs are, or None for each that is missing."""
    return {tool: shutil.which(tool) for tool in ("kubectl", "helm", "gcloud")}


TOKEN_COMMANDS = (
    ["gcloud", "auth", "application-default", "print-access-token"],
    ["gcloud", "auth", "print-access-token"],
)


def gcloud_access_token(runner: Runner = default_runner) -> str:
    """A bearer token for Google APIs from the caller's gcloud login.

    Managed Prometheus is queried over HTTPS with this token; `collect`
    never shells out to anything else, and never stores the token.

    Application Default Credentials are tried first, the user login second:
    found live, a corp workstation's user token was rejected by the
    Monitoring API (401 UNAUTHENTICATED) while its ADC token was accepted,
    and a plain workstation often has only the user login. The first
    command that prints a token wins; the error names both when neither does.
    """
    errors: list[str] = []
    for cmd in TOKEN_COMMANDS:
        try:
            proc = runner(cmd, stdin=None, timeout=READ_TIMEOUT_SEC)
        except FileNotFoundError as exc:
            raise DeployError("gcloud is not on PATH",
                              error_code="TOOL_MISSING",
                              detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandError(cmd, "no answer within 60s") from exc
        if proc.returncode == 0 and (proc.stdout or "").strip():
            return proc.stdout.strip()
        errors.append(f"{' '.join(cmd)}: "
                      f"{(proc.stderr or 'no token printed').strip()[:200]}")
    raise CommandError(list(TOKEN_COMMANDS[-1]), "; ".join(errors))

