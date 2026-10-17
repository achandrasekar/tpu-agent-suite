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
"""Stage model weights on a PersistentVolumeClaim before a release.

A script-driven engine (recipe `launch`) reads its weights from a volume
the recipe names (`pod.volumes[].persistentVolumeClaim`, mounted at
`MODEL_HUB_CACHE`'s parent) and loads them with
`snapshot_download(local_files_only=True)`; nothing in `render`/`apply`
fills that volume. This is the InferenceX launcher's `populate-qwen-cache`
step as a `deploy` verb:

1. the claim (default: the first PVC the recipe mounts) is created when
   missing -- `--size`, `--storage-class`, `--access-mode`, or
   `--from-snapshot <VolumeSnapshot>` to clone a populated one;
2. unless cloned, a Job on a CPU node (or `--tolerate-tpu` for a TPU node,
   which Hyperdisk ML classes may require) runs `hf download <model>`
   into the hub-cache layout (`<mount>/hub/models--<org>--<name>`) with
   the HF token Secret the environment names;
3. `--ro-via-snapshot <VolumeSnapshotClass>` reproduces the launcher's
   read-only path: populate a ReadWriteOnce claim `<pvc>-rw`, snapshot
   it as `<pvc>-snap`, and create the ReadOnlyMany claim `<pvc>` from
   that snapshot, so every engine pod on every node can mount it.

`preflight --recipe` reports whether the recipe's claims exist and are
Bound, and names this verb as the remediation.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Callable

from . import pins
from .config import DeployEnv, DeployError
from .kube import Kube

HF_DOWNLOAD_IMAGE = pins.AIPERF_BENCH_IMAGE  # uv + python, same pin as bench
HUB_SUBDIR = "hub"
DEFAULT_SIZE = "500Gi"
DEFAULT_TIMEOUT_S = 4 * 3600
POLL_S = 15
SNAPSHOT_API = "snapshot.storage.k8s.io/v1"
SNAPSHOT_CRD = "volumesnapshots.snapshot.storage.k8s.io"


def recipe_claims(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Every PVC the recipe's pod mounts: claim name, read-only flag, the
    mount path, and the hub sub-directory the role env points at."""
    pod = recipe.get("pod") or {}
    mounts = {m.get("name"): m for m in pod.get("volume_mounts") or []}
    hub = _hub_path(recipe)
    out = []
    for vol in pod.get("volumes") or []:
        pvc = vol.get("persistentVolumeClaim")
        if not pvc:
            continue
        mount = mounts.get(vol.get("name")) or {}
        mount_path = str(mount.get("mountPath") or "")
        subdir = HUB_SUBDIR
        if hub and mount_path and hub.startswith(mount_path.rstrip("/") + "/"):
            subdir = hub[len(mount_path.rstrip("/")) + 1:]
        out.append({"claim": str(pvc.get("claimName")),
                    "read_only": bool(pvc.get("readOnly") or
                                      mount.get("readOnly")),
                    "mount_path": mount_path, "hub_subdir": subdir})
    return out


def _hub_path(recipe: dict[str, Any]) -> str:
    role_env = (recipe.get("launch") or {}).get("role_env") or {}
    for role in ("prefill", "decode"):
        hub = (role_env.get(role) or {}).get("MODEL_HUB_CACHE")
        if hub:
            return str(hub)
    env = recipe.get("env") or {}
    return str(env.get("HF_HUB_CACHE") or env.get("MODEL_HUB_CACHE") or "")


def pvc_manifest(name: str, namespace: str, *, size: str,
                 storage_class: str | None, access_mode: str,
                 from_snapshot: str | None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "accessModes": [access_mode],
        "resources": {"requests": {"storage": size}},
    }
    if storage_class:
        spec["storageClassName"] = storage_class
    if from_snapshot:
        spec["dataSource"] = {"name": from_snapshot, "kind": "VolumeSnapshot",
                              "apiGroup": "snapshot.storage.k8s.io"}
    return {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
            "metadata": {"name": name, "namespace": namespace,
                         "labels": {pins.WEIGHTS_LABEL: "true",
                                    "app.kubernetes.io/component": "weights"}},
            "spec": spec}


def snapshot_manifest(name: str, namespace: str, *, claim: str,
                      snapshot_class: str) -> dict[str, Any]:
    return {"apiVersion": SNAPSHOT_API, "kind": "VolumeSnapshot",
            "metadata": {"name": name, "namespace": namespace,
                         "labels": {pins.WEIGHTS_LABEL: "true"}},
            "spec": {"volumeSnapshotClassName": snapshot_class,
                     "source": {"persistentVolumeClaimName": claim}}}


def download_job(name: str, env: DeployEnv, *, claim: str, models: list[str],
                 hub_subdir: str, node_selector: dict[str, str] | None,
                 tolerate_tpu: bool, token_secret: str | None) -> dict[str, Any]:
    """`hf download` of every model into the hub layout on the claim."""
    mount = "/mnt/weights"
    hub = f"{mount}/{hub_subdir}".rstrip("/")
    cmds = " && ".join(
        f"uvx --from 'huggingface_hub>=1.0' hf download"
        f" {json.dumps(m)} --max-workers 16" for m in models)
    # huggingface_hub >= 1.0 ships the `hf` CLI in the base package and
    # moves bulk transfer to Xet (HF_XET_HIGH_PERFORMANCE).
    script = ("set -euo pipefail; export HF_HUB_CACHE=" + json.dumps(hub) +
              "; export HF_XET_HIGH_PERFORMANCE=1; mkdir -p \"$HF_HUB_CACHE\";"
              f" {cmds}; echo staged; du -sh \"$HF_HUB_CACHE\"/models--* || true;"
              " echo '===STAGED===' && ls \"$HF_HUB_CACHE\"")
    container: dict[str, Any] = {
        "name": "hf-download", "image": HF_DOWNLOAD_IMAGE,
        "command": ["bash", "-c"], "args": [script],
        "env": [{"name": "HOME", "value": "/tmp"},
                {"name": "UV_CACHE_DIR", "value": "/tmp/uv"}],
        "resources": {"requests": {"cpu": "2", "memory": "4Gi"},
                      "limits": {"memory": "8Gi"}},
        "volumeMounts": [{"name": "weights", "mountPath": mount}],
    }
    if token_secret:
        container["env"].append({"name": "HF_TOKEN", "valueFrom": {
            "secretKeyRef": {"name": token_secret, "key": "HF_TOKEN",
                             "optional": True}}})
    pod: dict[str, Any] = {
        "restartPolicy": "OnFailure",   # a resumed download picks up
        "containers": [container],
        "volumes": [{"name": "weights",
                     "persistentVolumeClaim": {"claimName": claim}}],
    }
    if node_selector:
        pod["nodeSelector"] = dict(node_selector)
    if tolerate_tpu:
        pod["tolerations"] = [{"key": pins.TPU_RESOURCE, "operator": "Exists",
                               "effect": "NoSchedule"}]
    return {"apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": name, "namespace": env.namespace,
                         "labels": {pins.WEIGHTS_LABEL: "true",
                                    "app.kubernetes.io/component": "weights"}},
            "spec": {"backoffLimit": 6, "template": {
                "metadata": {"labels": {pins.WEIGHTS_LABEL: "true"}},
                "spec": pod}}}


def _pvc_phase(kube: Kube, claim: str) -> str | None:
    doc = kube.kubectl_json(["get", "pvc", claim], check=False)
    if not doc:
        return None
    return str((doc.get("status") or {}).get("phase") or "Pending")


def _wait(pred: Callable[[], bool], *, timeout_s: int,
          sleep: Callable[[float], None], clock: Callable[[], float]) -> bool:
    started = clock()
    while not pred():
        if clock() - started > timeout_s:
            return False
        sleep(POLL_S)
    return True


def _job_state(kube: Kube, name: str) -> str:
    doc = kube.kubectl_json(["get", "job", name], check=False)
    st = (doc.get("status") or {}) if doc else {}
    if int(st.get("succeeded") or 0) >= 1:
        return "succeeded"
    if int(st.get("failed") or 0) >= 1:
        return "failed"
    return "running" if doc else "absent"


def stage_weights(env: DeployEnv, *, recipe: dict[str, Any],
                  claim: str | None = None, size: str = DEFAULT_SIZE,
                  storage_class: str | None = None,
                  access_mode: str = "ReadWriteOnce",
                  from_snapshot: str | None = None,
                  ro_via_snapshot: str | None = None,
                  node_selector: dict[str, str] | None = None,
                  tolerate_tpu: bool = False,
                  extra_models: list[str] | None = None,
                  wait: bool = True, timeout_s: int = DEFAULT_TIMEOUT_S,
                  runner: Any = None,
                  sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic
                  ) -> dict[str, Any]:
    claims = recipe_claims(recipe)
    target = claim or (claims[0]["claim"] if claims else None)
    if not target:
        raise DeployError("the recipe mounts no PersistentVolumeClaim and"
                          " --pvc was not given", error_code="INVALID_FIELD")
    info = next((c for c in claims if c["claim"] == target),
                {"claim": target, "hub_subdir": HUB_SUBDIR, "read_only": False})
    models = [str(recipe["model"])]
    tok = recipe.get("tokenizer")
    if tok and tok != recipe["model"] and "/" in str(tok):
        models.append(str(tok))
    models += [m for m in (extra_models or []) if m not in models]
    kube = env.kube(runner=runner)
    node_selector = (dict(node_selector) if node_selector is not None
                     else (dict(env.cpu_node_selector) if not tolerate_tpu
                           else None))
    steps: list[dict[str, Any]] = []
    result: dict[str, Any] = {"claim": target, "models": models,
                              "hub_subdir": info["hub_subdir"],
                              "namespace": env.namespace, "steps": steps}

    def ensure_pvc(name: str, mode: str, snapshot: str | None) -> str:
        phase = _pvc_phase(kube, name)
        if phase is None:
            kube.apply(json.dumps(pvc_manifest(
                name, env.namespace, size=size, storage_class=storage_class,
                access_mode=mode, from_snapshot=snapshot)))
            steps.append({"step": "pvc.create", "claim": name, "mode": mode,
                          "from_snapshot": snapshot})
            return "created"
        steps.append({"step": "pvc.exists", "claim": name, "phase": phase})
        return "exists"

    if from_snapshot:
        # A populated snapshot: no download, just the clone.
        ensure_pvc(target, access_mode, from_snapshot)
        result["populated"] = "from_snapshot"
        result["status"] = "completed"
        return result

    if ro_via_snapshot:
        if not kube.exists("crd", SNAPSHOT_CRD, namespaced=False):
            raise DeployError(
                f"{SNAPSHOT_CRD} is not installed; the read-only path needs"
                " the CSI snapshotter (GKE: enabled with the Compute Engine"
                " persistent disk CSI driver)", error_code="PREFLIGHT_FAILED")
        rw_claim, snap = f"{target}-rw", f"{target}-snap"
    else:
        if access_mode == "ReadOnlyMany":
            raise DeployError(
                "a ReadOnlyMany claim cannot be populated by a Job; stage"
                " with --ro-via-snapshot <VolumeSnapshotClass> (populate"
                f" {target}-rw, snapshot, clone read-only) or"
                " --from-snapshot <existing snapshot>",
                error_code="INVALID_FIELD")
        rw_claim, snap = target, None

    ensure_pvc(rw_claim, "ReadWriteOnce" if ro_via_snapshot else access_mode,
               None)
    job_name = f"stage-weights-{rw_claim}"[:63].rstrip("-")
    state = _job_state(kube, job_name)
    if state in ("absent", "failed"):
        if state == "failed":
            kube.delete(["job", job_name], ignore_not_found=True)
        kube.apply(json.dumps(download_job(
            job_name, env, claim=rw_claim, models=models,
            hub_subdir=info["hub_subdir"], node_selector=node_selector,
            tolerate_tpu=tolerate_tpu, token_secret=env.hf_token_secret)))
        steps.append({"step": "job.create", "job": job_name,
                      "image": HF_DOWNLOAD_IMAGE,
                      "node_selector": node_selector,
                      "tolerate_tpu": tolerate_tpu})
    else:
        steps.append({"step": "job.exists", "job": job_name, "state": state})
    result["job"] = job_name
    if not wait:
        result["status"] = "staging"
        result["populated"] = "pending"
        return result
    done = _wait(lambda: _job_state(kube, job_name) in ("succeeded", "failed"),
                 timeout_s=timeout_s, sleep=sleep, clock=clock)
    final = _job_state(kube, job_name)
    proc = kube.kubectl(["logs", f"job/{job_name}", "--tail=40"], check=False)
    result["job_log_tail"] = (proc.stdout or proc.stderr or "")[-2000:]
    if not done or final != "succeeded":
        result["status"] = "failed" if final == "failed" else "timeout"
        result["populated"] = "no"
        steps.append({"step": "job.wait", "state": final, "waited": done})
        return result
    steps.append({"step": "job.wait", "state": "succeeded"})
    if not ro_via_snapshot:
        result["populated"] = "yes"
        result["status"] = "completed"
        return result
    # Read-only path: snapshot the populated claim, clone it ReadOnlyMany.
    if not kube.exists("volumesnapshot", snap):
        kube.apply(json.dumps(snapshot_manifest(
            snap, env.namespace, claim=rw_claim,
            snapshot_class=ro_via_snapshot)))
        steps.append({"step": "snapshot.create", "snapshot": snap})

    def snapshot_ready() -> bool:
        doc = kube.kubectl_json(["get", "volumesnapshot", snap], check=False)
        return bool(((doc or {}).get("status") or {}).get("readyToUse"))
    if not _wait(snapshot_ready, timeout_s=timeout_s, sleep=sleep,
                 clock=clock):
        result.update({"status": "timeout", "populated": "rw_only",
                       "snapshot": snap})
        return result
    steps.append({"step": "snapshot.ready", "snapshot": snap})
    ensure_pvc(target, "ReadOnlyMany", snap)
    result.update({"status": "completed", "populated": "yes",
                   "snapshot": snap, "rw_claim": rw_claim})
    return result


def check_claims(kube: Kube, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """For preflight: each claim the recipe mounts, with its phase."""
    out = []
    for c in recipe_claims(recipe):
        phase = _pvc_phase(kube, c["claim"])
        # Pending is fine for a WaitForFirstConsumer class: it binds when
        # the first engine pod lands.
        out.append({**c, "phase": phase, "ok": phase is not None})
    return out
