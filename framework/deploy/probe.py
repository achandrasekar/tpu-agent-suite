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

llm-d's TPU guides assume the vLLM TPU image carries tpu-inference's P/D
and host-offload KV connectors. A study plans on that assumption and this
probe confirms it before any arm needing a connector is deployed: one short
Job on a TPU node imports the modules and prints what it found. It does not
load a model, so it costs a node for about a minute, and it is a sanity
check rather than a gate -- a missing connector marks the arms that need it
`blocked`, it does not stop the study.

Connector modules, per the guides at pins.LLMD_REF:
  https://github.com/llm-d/llm-d/tree/v0.10.0/guides/pd-disaggregation
  https://github.com/llm-d/llm-d/tree/v0.10.0/guides/tiered-prefix-cache
  https://github.com/vllm-project/tpu-inference
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

# Module path -> capability name. Each is tried in order; the first import
# that succeeds sets the capability.
CAPABILITY_MODULES: dict[str, list[str]] = {
    "pd_connector": [
        "tpu_inference.distributed.tpu_connector",
        "tpu_inference.distributed.tpu_connector_hma",
    ],
    "offload_connector": [
        "tpu_inference.distributed.offload.tpu_offload_connector",
        "tpu_inference.distributed.tpu_offload_connector",
    ],
    "kv_events": ["vllm.distributed.kv_events"],
    "tpu_inference": ["tpu_inference"],
    "vllm": ["vllm"],
}

PROBE_SCRIPT = r"""
import importlib, json, sys
mods = %s
out = {"capabilities": {}, "versions": {}, "modules": {}}
for cap, paths in mods.items():
    found = None
    errors = {}
    for p in paths:
        try:
            importlib.import_module(p)
            found = p
            break
        except Exception as exc:  # noqa: BLE001
            errors[p] = str(exc)[:200]
    out["capabilities"][cap] = found is not None
    out["modules"][cap] = {"found": found, "errors": errors}
for name in ("vllm", "tpu_inference", "torch", "jax"):
    try:
        m = importlib.import_module(name)
        out["versions"][name] = getattr(m, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        out["versions"][name] = None
print(%r)
print(json.dumps(out))
print(%r)
"""


def probe_job(name: str, image: str, env: DeployEnv) -> dict[str, Any]:
    script = PROBE_SCRIPT % (json.dumps(CAPABILITY_MODULES), PROBE_MARK,
                             PROBE_MARK)
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
                        "command": ["python3", "-c", script],
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


def run_probe(env: DeployEnv,
              *,
              image: str,
              out_dir: pathlib.Path,
              timeout_s: int = PROBE_TIMEOUT_S,
              runner: Any = None,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    if not env.tpu_accelerator or not env.tpu_topology:
        raise DeployError(
            "probe needs compute.params.tpu_accelerator and tpu_topology",
            error_code="INVALID_FIELD")
    kube: Kube = env.kube(runner=runner)
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
    parsed = parse_probe_logs(text)
    doc = {
        "schema": "engine-capabilities",
        "probed_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "image": image,
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
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engine-capabilities.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if parsed is None:
        raise DeployError(f"probe {outcome}: no capabilities printed",
                          error_code="PROBE_FAILED",
                          detail=text[-500:])
    return doc
