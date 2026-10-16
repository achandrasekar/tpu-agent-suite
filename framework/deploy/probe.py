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
"""What the serving image can do: the engine-capabilities sanity check.

llm-d's TPU guides assume the vLLM TPU image carries P/D and host-offload
KV connectors. A study plans on that assumption and this probe confirms it
before any arm needing a connector is deployed. It is a sanity check rather
than a gate -- a missing connector marks the arms that need it `blocked`, it
does not stop the study.

Two ways to ask the same question:

* `job` mode (managed fleets): one short Job on a TPU node runs the script
  and exits. It costs a slice for about a minute.
* `exec` mode (adopted fleets): the script runs inside a pod that is
  already serving. An adopted fleet usually holds every chip the namespace
  has, so a Job could never schedule; exec answers without touching the
  fleet. The accelerator is disabled for the probe process
  (`JAX_PLATFORMS=cpu`, `PJRT_DEVICE=CPU`) so it cannot contend with the
  server for the chips it owns.

Presence is checked with `importlib.util.find_spec`, not an import: the
connector modules pull in the engine, and importing the engine inside a
serving pod would try to initialise a runtime that pod already owns.

Connector modules, per the guides at pins.LLMD_REF:
  https://github.com/llm-d/llm-d/tree/v0.10.0/guides/pd-disaggregation
  https://github.com/llm-d/llm-d/tree/v0.10.0/guides/tiered-prefix-cache
  https://github.com/vllm-project/tpu-inference
The `vllm_torchtpu.*` paths are the torchtpu serving image's own
connectors (classes TPUConnector / TPURaidenConnector / TPUMultiConnector
and the raiden host-offload connector); they are what the image used in
this repo's first spike actually carried.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import time
import uuid
from typing import Any, Callable

from . import pins, render
from .config import DeployEnv
from .kube import DeployError, Kube

PROBE_TIMEOUT_S = 900
POLL_S = 10
PROBE_MARK = "===CAPABILITIES==="

# Module path -> capability name. Each is tried in order; the first module
# that resolves sets the capability.
CAPABILITY_MODULES: dict[str, list[str]] = {
    "pd_connector": [
        "tpu_inference.distributed.tpu_connector",
        "tpu_inference.distributed.tpu_connector_hma",
        "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    ],
    "offload_connector": [
        "tpu_inference.distributed.offload.tpu_offload_connector",
        "tpu_inference.distributed.tpu_offload_connector",
        "vllm_torchtpu.offload.raiden_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    ],
    "kv_events": ["vllm.distributed.kv_events"],
    "tpu_inference": ["tpu_inference"],
    "vllm_torchtpu": ["vllm_torchtpu"],
    "vllm": ["vllm"],
}

# Distributions whose versions are recorded; read from package metadata so
# nothing is imported.
VERSION_DISTS = ("vllm", "tpu_inference", "vllm_torchtpu", "torch", "jax")

# Environment for the exec-mode probe process: no accelerator for us.
EXEC_ENV = {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "CPU",
            "TPU_SKIP_MDS_QUERY": "1"}

PROBE_SCRIPT = r"""
import importlib.util, json, sys
try:
    import importlib.metadata as md
except Exception:  # noqa: BLE001
    md = None
mods = %s
dists = %s
out = {"capabilities": {}, "versions": {}, "modules": {},
       "method": "find_spec"}
for cap, paths in mods.items():
    found = None
    errors = {}
    for p in paths:
        try:
            spec = importlib.util.find_spec(p)
        except Exception as exc:  # noqa: BLE001
            errors[p] = str(exc)[:200]
            continue
        if spec is None:
            errors[p] = "not found"
            continue
        found = p
        break
    out["capabilities"][cap] = found is not None
    out["modules"][cap] = {"found": found, "errors": errors}
for name in dists:
    ver = None
    if md is not None:
        for cand in (name, name.replace("_", "-")):
            try:
                ver = md.version(cand)
                break
            except Exception:  # noqa: BLE001
                continue
    out["versions"][name] = ver
print(%r)
print(json.dumps(out))
print(%r)
"""


def probe_script() -> str:
    return PROBE_SCRIPT % (json.dumps(CAPABILITY_MODULES),
                           json.dumps(list(VERSION_DISTS)), PROBE_MARK,
                           PROBE_MARK)


def probe_job(name: str, image: str, env: DeployEnv) -> dict[str, Any]:
    chips = render.chips_for_topology(env.tpu_topology)
    labels = {
        pins.MANAGED_BY_LABEL: pins.MANAGED_BY_VALUE,
        "tpu-coworker.llmd/probe": "true"
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        pins.TOPOLOGY_LABEL: env.tpu_topology,
                        pins.ACCELERATOR_LABEL: env.tpu_accelerator,
                    },
                    "containers": [{
                        "name": "probe",
                        "image": image,
                        "command": ["python3", "-c", probe_script()],
                        "resources": {
                            "limits": {pins.TPU_RESOURCE: chips},
                            "requests": {pins.TPU_RESOURCE: chips},
                        },
                    }],
                },
            },
        },
    }


def parse_probe_logs(text: str) -> dict[str, Any] | None:
    inside = False
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.split("] ", 1)[1] if raw.startswith("[") and "] " in raw \
            else raw
        if line.strip() == PROBE_MARK:
            if inside:
                try:
                    return json.loads("\n".join(buf))
                except ValueError:
                    return None
            inside = True
            continue
        if inside:
            buf.append(line)
    return None


def _probe_by_job(kube: Kube, env: DeployEnv, image: str, timeout_s: int,
                  sleep: Callable[[float], None],
                  clock: Callable[[], float]) -> tuple[str, str]:
    if not env.tpu_accelerator or not env.tpu_topology:
        raise DeployError(
            "probe needs compute.params.tpu_accelerator and tpu_topology",
            error_code="INVALID_FIELD")
    name = f"llmd-probe-{uuid.uuid4().hex[:8]}"
    kube.apply(json.dumps(probe_job(name, image, env)))
    started = clock()
    outcome = "timeout"
    while clock() - started < timeout_s:
        doc = kube.kubectl_json(["get", "job", name], check=False)
        st = (doc.get("status") or {}) if doc else {}
        if int(st.get("succeeded") or 0) >= 1:
            outcome = "succeeded"
            break
        if int(st.get("failed") or 0) >= 1:
            outcome = "failed"
            break
        sleep(POLL_S)
    proc = kube.kubectl(["logs", f"job/{name}", "--tail=-1"], check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    kube.delete(["job", name])
    return outcome, text


def pick_pod(kube: Kube, selector: str) -> tuple[str, str, str]:
    """First Ready pod matching `selector`: (name, container, image)."""
    doc = kube.kubectl_json(["get", "pods", "-l", selector], check=False)
    for item in doc.get("items") or []:
        status = item.get("status") or {}
        if status.get("phase") != "Running":
            continue
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in status.get("conditions") or [])
        if not ready:
            continue
        containers = (item.get("spec") or {}).get("containers") or []
        if not containers:
            continue
        return (item["metadata"]["name"], containers[0]["name"],
                containers[0].get("image", ""))
    raise DeployError(f"no Ready pod matches {selector!r}",
                      error_code="NO_ADOPTED_POD")


def _probe_by_exec(kube: Kube, pod: str, container: str | None,
                   timeout_s: int) -> tuple[str, str]:
    argv = ["exec", pod]
    if container:
        argv += ["-c", container]
    argv += ["--", "env", *[f"{k}={v}" for k, v in EXEC_ENV.items()],
             "python3", "-c", probe_script()]
    proc = kube.kubectl(argv, check=False, timeout=timeout_s)
    text = (proc.stdout or "") + (proc.stderr or "")
    return ("succeeded" if proc.returncode == 0 else "failed"), text


def _write_capabilities(out_dir: pathlib.Path,
                        doc: dict[str, Any]) -> dict[str, Any]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engine-capabilities.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def run_probe(env: DeployEnv,
              *,
              image: str | None = None,
              out_dir: pathlib.Path,
              pod: str | None = None,
              selector: str | None = None,
              container: str | None = None,
              timeout_s: int = PROBE_TIMEOUT_S,
              runner: Any = None,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Probe by Job (`image`) or by exec (`pod` or `selector`).

    Exec mode records the image the chosen pod actually runs, so the
    capabilities document always names the image it describes.
    """
    if image and image.split(":")[0].split("@")[0] == pins.SIM_IMAGE:
        # A simulated engine is a Go binary with no Python to import; its
        # capabilities are what the simulator is, not what a probe finds.
        return _write_capabilities(out_dir, {
            "schema": "engine-capabilities",
            "probed_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "image": image, "mode": "sim", "pod": None,
            "accelerator": {"tpu_accelerator": env.tpu_accelerator,
                            "tpu_topology": env.tpu_topology},
            "outcome": "sim",
            "capabilities": {"sim": True, "pd_connector": True,
                             "offload_connector": False, "kv_events": False,
                             "tpu_inference": False, "vllm": False,
                             "vllm_torchtpu": False},
            "versions": {}, "modules": {}, "log_tail": "",
        })
    kube: Kube = env.kube(runner=runner)
    mode = "exec" if (pod or selector) else "job"
    if mode == "exec":
        if pod is None:
            pod, found_container, found_image = pick_pod(kube, selector or "")
            container = container or found_container
            image = image or found_image
        elif image is None:
            doc = kube.kubectl_json(["get", "pod", pod], check=False)
            containers = (doc.get("spec") or {}).get("containers") or []
            if containers:
                image = containers[0].get("image", "")
                container = container or containers[0]["name"]
        outcome, text = _probe_by_exec(kube, pod, container, timeout_s)
    else:
        if not image:
            raise DeployError("probe needs --image, --pod or --selector",
                              error_code="INVALID_FIELD")
        outcome, text = _probe_by_job(kube, env, image, timeout_s, sleep,
                                      clock)
    parsed = parse_probe_logs(text)
    doc = {
        "schema": "engine-capabilities",
        "probed_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "image": image or "",
        "mode": mode,
        "pod": pod,
        "accelerator": {
            "tpu_accelerator": env.tpu_accelerator,
            "tpu_topology": env.tpu_topology
        },
        "outcome": outcome,
        "capabilities": (parsed or {}).get("capabilities", {}),
        "versions": (parsed or {}).get("versions", {}),
        "modules": (parsed or {}).get("modules", {}),
        "log_tail": text[-2000:] if parsed is None else "",
    }
    _write_capabilities(out_dir, doc)
    if parsed is None:
        raise DeployError(f"probe {outcome}: no capabilities printed",
                          error_code="PROBE_FAILED",
                          detail=text[-500:])
    return doc
