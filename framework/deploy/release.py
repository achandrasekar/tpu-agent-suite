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
"""The life of a rendered release on the cluster.

`apply` puts it there and files a job record; `wait` blocks until it serves
or says why it will not; `reconfigure` changes only what is cheap to change
and re-applies only that; `teardown` removes everything the release label
selects. Every release is a `JobRecord` with backend `gke_release`, so a
study that ends without tearing down leaves something the run-end reaper
can find and remove -- except a production release, which is `serve=True`
and outlives the session on purpose.
"""

from __future__ import annotations

import copy
import datetime
import json
import pathlib
import time
from typing import Any, Callable

try:
    from framework.compute.drivers import jobs
except ImportError:
    try:
        from ..compute.drivers import jobs  # type: ignore
    except ImportError:  # bundled runtime: `deploy` sits beside `compute`
        from compute.drivers import jobs  # type: ignore

from . import pins, render
from .config import RELEASE_BACKEND, DeployEnv, find_release_record
from .kube import DeployError, Kube

DEFAULT_WAIT_TIMEOUT_S = 1800
DEFAULT_POLL_S = 15
# Pod waiting reasons that will not clear by waiting longer.
FATAL_WAITING_REASONS = ("InvalidImageName", "ErrImageNeverPull",
                         "CrashLoopBackOff", "CreateContainerConfigError")
PULL_FAILURE_GRACE_S = 180

# Which incumbent paths `reconfigure` may change, by what the change costs.
# Hot paths change the EPP only; warm paths roll the model-server pods on
# the nodes they already hold. Anything else needs new capacity and belongs
# to the outer loop as a fresh release.
HOT_PREFIXES = ("router.",)
WARM_PREFIXES = ("engine.", "kv.block_size")
# Hot-looking paths that are nonetheless cold: changing them changes which
# components exist, not how the existing ones are configured. `router.profile`
# swaps the plugin set and its data producers, so scorer weights swept under
# one profile say nothing about another; the study measures each profile as
# its own arm.
COLD_EXCEPTIONS = ("router.profile", "router.flow_control",
                   "router.predicted_latency")

# Kinds the release label selects, deleted at teardown in this order so the
# Jobs stop sending before the servers go.
RELEASE_KINDS = ("job", "configmap", "deployment", "service",
                 "serviceaccount", "podmonitoring")


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")


def _kube_for(env: DeployEnv, release: dict[str, Any],
              runner: Any = None) -> Kube:
    kwargs: dict[str, Any] = {
        "namespace": release["namespace"],
        "kubeconfig": env.kubeconfig,
        "context": env.kube_context,
    }
    if runner is not None:
        kwargs["runner"] = runner
    return Kube(**kwargs)


def _read(release_dir: pathlib.Path, rel: str) -> str:
    return (pathlib.Path(release_dir) / rel).read_text(encoding="utf-8")


def _history(release: dict[str, Any], action: str, **fields: Any) -> None:
    release.setdefault("history", []).append({
        "at": _now(),
        "action": action,
        **fields
    })


# -- apply ---------------------------------------------------------------------


def apply(env: DeployEnv,
          release_dir: pathlib.Path,
          *,
          run_id: str,
          production: bool = False,
          runner: Any = None) -> dict[str, Any]:
    """Apply a rendered release and file it in the job store.

    Order: model servers first (they take minutes to pull and load), then
    monitoring, then the router, so the EPP's pool watch finds pods already
    coming up. Monitoring is best-effort: a cluster without Managed
    Prometheus still serves, it just cannot be `collect`ed from.
    """
    release_dir = pathlib.Path(release_dir)
    release = render.load_release(release_dir)
    if production and not release.get("production"):
        raise DeployError(
            "this release was rendered for a study; re-render with"
            " --production before applying it as production",
            error_code="NOT_PRODUCTION_RENDER")
    kube = _kube_for(env, release, runner)

    kube.apply(_read(release_dir, release["files"]["modelserver"]))
    warnings: list[str] = []
    try:
        kube.apply(_read(release_dir, release["files"]["monitoring"]))
    except DeployError as exc:
        warnings.append(f"monitoring not applied: {exc.message};"
                        f" {exc.detail[:200]}")
    kube.helm_upgrade_install(
        release["router"]["helm_release"],
        release["router"]["chart"],
        release["router"]["chart_version"],
        [str(release_dir / release["files"]["router_values"])])

    record = jobs.JobRecord.create(
        job_name=release["name"],
        run_id=run_id,
        backend=RELEASE_BACKEND,
        serve=bool(production),
        workload_name=release["model"],
        extra={
            "release_dir": str(release_dir),
            "namespace": release["namespace"],
            "helm_release": release["router"]["helm_release"],
            "production": bool(production),
            "chips_total": release["chips"]["total"],
            "config_digest": release["config_digest"],
        },
    )
    record.mark("RUNNING")
    release.update({
        "state": "applied",
        "applied_at": _now(),
        "job_id": record.job_id,
        "run_id": run_id,
    })
    _history(release, "apply", config_digest=release["config_digest"],
             job_id=record.job_id, warnings=warnings)
    render.save_release(release_dir, release)
    return {
        "name": release["name"],
        "namespace": release["namespace"],
        "job_id": record.job_id,
        "chips_total": release["chips"]["total"],
        "router_service": release["router"]["service"],
        "warnings": warnings,
    }


# -- wait ------------------------------------------------------------------------


def _deployment_ready(doc: dict[str, Any]) -> tuple[bool, int, int]:
    spec = int((doc.get("spec") or {}).get("replicas") or 0)
    status = doc.get("status") or {}
    ready = int(status.get("readyReplicas") or 0)
    generation_ok = int(status.get("observedGeneration") or 0) >= int(
        (doc.get("metadata") or {}).get("generation") or 0)
    updated = int(status.get("updatedReplicas") or 0)
    return (generation_ok and ready >= spec and updated >= spec
            and spec > 0), ready, spec


def _fatal_pod(pods: dict[str, Any], started: float) -> str | None:
    """Why the release will never be ready, if a pod already says so."""
    for pod in pods.get("items", []):
        name = pod.get("metadata", {}).get("name", "?")
        for cs in (pod.get("status") or {}).get("containerStatuses", []) or []:
            waiting = (cs.get("state") or {}).get("waiting") or {}
            reason = waiting.get("reason", "")
            if reason in FATAL_WAITING_REASONS:
                return f"{name}: {reason}: {waiting.get('message', '')[:200]}"
            if reason in ("ErrImagePull", "ImagePullBackOff") and (
                    time.monotonic() - started) > PULL_FAILURE_GRACE_S:
                return (f"{name}: {reason} for more than"
                        f" {PULL_FAILURE_GRACE_S}s:"
                        f" {waiting.get('message', '')[:200]}")
        phase = (pod.get("status") or {}).get("phase")
        if phase == "Failed":
            return f"{name}: pod Failed"
    return None


def wait(env: DeployEnv,
         release_dir: pathlib.Path,
         *,
         timeout_s: int = DEFAULT_WAIT_TIMEOUT_S,
         poll_s: int = DEFAULT_POLL_S,
         runner: Any = None,
         sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Block until every model-server Deployment and the EPP are ready.

    Bounded: a release that cannot come up (bad image, missing secret,
    crash loop) is reported as soon as a pod says so, and the rest time out
    with `WAIT_TIMEOUT`, which the CLI turns into exit code 3 so a caller
    can tell "not yet" from "broken".
    """
    release_dir = pathlib.Path(release_dir)
    release = render.load_release(release_dir)
    kube = _kube_for(env, release, runner)
    name = release["name"]
    started = clock()
    last: dict[str, Any] = {}
    while True:
        last = {"deployments": {}, "epp": {}}
        all_ready = True
        for dep in release["deployments"]:
            doc = kube.kubectl_json(["get", "deployment", dep], check=False)
            ready, have, want = _deployment_ready(doc) if doc else (False, 0,
                                                                     0)
            last["deployments"][dep] = {"ready": have, "want": want}
            all_ready = all_ready and ready
        epp = kube.kubectl_json([
            "get", "deployment", "-l", f"app.kubernetes.io/instance={name}"
        ], check=False)
        epp_items = epp.get("items", []) if epp else []
        if not epp_items:
            all_ready = False
            last["epp"] = {"found": False}
        for item in epp_items:
            ready, have, want = _deployment_ready(item)
            last["epp"][item["metadata"]["name"]] = {
                "ready": have,
                "want": want
            }
            all_ready = all_ready and ready
        if all_ready:
            release["state"] = "ready"
            release["ready_at"] = _now()
            _history(release, "ready", waited_s=round(clock() - started))
            render.save_release(release_dir, release)
            return {
                "name": name,
                "state": "ready",
                "waited_s": round(clock() - started),
                **last
            }
        pods = kube.kubectl_json(
            ["get", "pods", "-l", f"{pins.RELEASE_LABEL}={name}"],
            check=False)
        fatal = _fatal_pod(pods, started) if pods else None
        if fatal:
            release["state"] = "failed"
            _history(release, "failed", reason=fatal)
            render.save_release(release_dir, release)
            raise DeployError(f"release {name} cannot become ready",
                              error_code="RELEASE_FAILED",
                              detail=fatal)
        if clock() - started >= timeout_s:
            raise DeployError(
                f"release {name} not ready after {timeout_s}s",
                error_code="WAIT_TIMEOUT",
                detail=json.dumps(last, sort_keys=True))
        sleep(poll_s)


# -- reconfigure ------------------------------------------------------------------


def classify_path(path: str) -> str:
    """hot, warm or cold: which loop may change this incumbent path."""
    if path in COLD_EXCEPTIONS:
        return "cold"
    if path.startswith(HOT_PREFIXES):
        return "hot"
    if path.startswith(WARM_PREFIXES):
        return "warm"
    return "cold"


def parse_set_value(text: str) -> Any:
    """`--set` values are JSON when they parse, strings otherwise."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = doc
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def reconfigure(env: DeployEnv,
                release_dir: pathlib.Path,
                sets: dict[str, Any],
                *,
                runner: Any = None) -> dict[str, Any]:
    """Change hot and warm paths of a live release in place.

    Cold paths are refused: they need new capacity and a new release, and
    letting them through here would quietly turn an inner-loop step into an
    outer-loop deployment. The incumbent config in the release directory is
    updated so the next `bench` measures what is actually running.
    """
    if not sets:
        raise DeployError("nothing to set", error_code="INVALID_FIELD")
    release_dir = pathlib.Path(release_dir)
    release = render.load_release(release_dir)
    classes = {path: classify_path(path) for path in sets}
    cold = sorted(p for p, c in classes.items() if c == "cold")
    if cold:
        raise DeployError(
            f"{cold} are cold paths; render and apply a new release",
            error_code="COLD_PATH")
    config = json.loads(_read(release_dir, release["files"]["config"]))
    before = render.config_digest(render.normalize_config(config))
    for path, value in sets.items():
        set_path(config, path, value)
    new_release = render.rerender(release_dir, config)
    kube = _kube_for(env, new_release, runner)

    did = {"hot": False, "warm": False}
    if any(c == "warm" for c in classes.values()):
        kube.apply(_read(release_dir, new_release["files"]["modelserver"]))
        did["warm"] = True
    if any(c == "hot" for c in classes.values()):
        kube.helm_upgrade_install(
            new_release["router"]["helm_release"],
            new_release["router"]["chart"],
            new_release["router"]["chart_version"],
            [str(release_dir / new_release["files"]["router_values"])])
        did["hot"] = True
    new_release["state"] = "reconfiguring"
    _history(new_release, "reconfigure", sets=sets, classes=classes,
             from_digest=before, to_digest=new_release["config_digest"])
    render.save_release(release_dir, new_release)
    return {
        "name": new_release["name"],
        "applied": did,
        "classes": classes,
        "config_digest": new_release["config_digest"],
        "needs_wait": did["warm"] or did["hot"],
    }


# -- status, logs, events ------------------------------------------------------


def status(env: DeployEnv, release_dir: pathlib.Path,
           *, runner: Any = None) -> dict[str, Any]:
    release = render.load_release(release_dir)
    kube = _kube_for(env, release, runner)
    name = release["name"]
    out: dict[str, Any] = {
        "name": name,
        "namespace": release["namespace"],
        "state": release.get("state", "rendered"),
        "config_digest": release["config_digest"],
        "chips": release["chips"],
        "deployments": {},
        "pods": [],
        "router": {},
        "bench_jobs": [],
    }
    for dep in release["deployments"]:
        doc = kube.kubectl_json(["get", "deployment", dep], check=False)
        ready, have, want = _deployment_ready(doc) if doc else (False, 0, 0)
        out["deployments"][dep] = {
            "exists": bool(doc),
            "ready": ready,
            "ready_replicas": have,
            "replicas": want
        }
    pods = kube.kubectl_json(
        ["get", "pods", "-l", f"{pins.RELEASE_LABEL}={name}"], check=False)
    for pod in (pods.get("items", []) if pods else []):
        statuses = (pod.get("status") or {}).get("containerStatuses") or []
        waiting = ""
        for cs in statuses:
            w = (cs.get("state") or {}).get("waiting") or {}
            if w.get("reason"):
                waiting = w["reason"]
        out["pods"].append({
            "name": pod["metadata"]["name"],
            "phase": (pod.get("status") or {}).get("phase"),
            "node": (pod.get("spec") or {}).get("nodeName"),
            "waiting": waiting,
            "restarts": sum(int(cs.get("restartCount") or 0)
                            for cs in statuses),
        })
    out["router"] = {
        "helm_release": release["router"]["helm_release"],
        "installed": kube.helm_release_exists(
            release["router"]["helm_release"]),
        "service": release["router"]["service"],
    }
    bench = kube.kubectl_json(
        ["get", "jobs", "-l", f"{pins.RELEASE_LABEL}={name}"], check=False)
    for job in (bench.get("items", []) if bench else []):
        st = job.get("status") or {}
        out["bench_jobs"].append({
            "name": job["metadata"]["name"],
            "succeeded": int(st.get("succeeded") or 0),
            "failed": int(st.get("failed") or 0),
            "active": int(st.get("active") or 0),
        })
    return out


def logs(env: DeployEnv, release_dir: pathlib.Path, *, component: str,
         tail: int = 200, runner: Any = None) -> str:
    release = render.load_release(release_dir)
    kube = _kube_for(env, release, runner)
    name = release["name"]
    if component == "modelserver":
        return kube.logs(f"{pins.RELEASE_LABEL}={name},"
                         f"{pins.LLMD_ROLE_LABEL}=decode",
                         container=render.MODELSERVER_CONTAINER,
                         tail=tail)
    if component == "epp":
        return kube.logs(f"app.kubernetes.io/instance={name}", tail=tail)
    if component == "bench":
        return kube.logs(f"{pins.RELEASE_LABEL}={name},"
                         "tpu-coworker.llmd/bench-stage",
                         container=render.BENCH_CONTAINER,
                         tail=tail)
    raise DeployError("component must be modelserver, epp or bench",
                      error_code="INVALID_FIELD")


def events(env: DeployEnv, release_dir: pathlib.Path,
           *, runner: Any = None) -> list[dict[str, Any]]:
    release = render.load_release(release_dir)
    kube = _kube_for(env, release, runner)
    doc = kube.kubectl_json(["get", "events", "--sort-by=.lastTimestamp"],
                            check=False)
    name = release["name"]
    out = []
    for ev in (doc.get("items", []) if doc else []):
        obj = ev.get("involvedObject") or {}
        if name not in str(obj.get("name", "")):
            continue
        out.append({
            "time": ev.get("lastTimestamp") or ev.get("eventTime"),
            "type": ev.get("type"),
            "reason": ev.get("reason"),
            "object": f"{obj.get('kind')}/{obj.get('name')}",
            "message": (ev.get("message") or "")[:300],
        })
    return out[-100:]


# -- teardown -----------------------------------------------------------------------


def teardown(env: DeployEnv,
             release_dir: pathlib.Path | None = None,
             *,
             record: jobs.JobRecord | None = None,
             include_production: bool = False,
             runner: Any = None) -> dict[str, Any]:
    """Remove everything the release label selects, then the router chart.

    Works from the release directory when it exists and from the job record
    alone when it does not (a run directory may have been cleaned before the
    reaper got to it): the label and the helm release name are all teardown
    needs, and both are in the record.
    """
    release: dict[str, Any] | None = None
    if release_dir is not None and (pathlib.Path(release_dir) /
                                    render.RELEASE_FILE).is_file():
        release = render.load_release(release_dir)
    if record is None and release is not None and release.get("job_id"):
        record = jobs.JobRecord.load(release["job_id"])
    if release is None and record is None:
        raise DeployError("nothing identifies the release to tear down",
                          error_code="RELEASE_NOT_FOUND")

    name = release["name"] if release else record.job_name  # type: ignore
    namespace = (release["namespace"]
                 if release else record.extra.get("namespace", env.namespace))
    helm_release = (release["router"]["helm_release"]
                    if release else record.extra.get("helm_release", name))
    production = bool(release.get("production") if release else record.extra.
                      get("production"))
    if production and not include_production:
        raise DeployError(
            f"release {name} is production; pass --include-production",
            error_code="PRODUCTION_GUARD")

    kwargs: dict[str, Any] = {
        "namespace": namespace,
        "kubeconfig": env.kubeconfig,
        "context": env.kube_context
    }
    if runner is not None:
        kwargs["runner"] = runner
    kube = Kube(**kwargs)
    selector = f"{pins.RELEASE_LABEL}={name}"
    removed: list[str] = []
    problems: list[str] = []
    for kind in RELEASE_KINDS:
        try:
            proc = kube.delete([kind, "-l", selector])
            removed.append(f"{kind}: {(proc.stdout or '').strip() or 'none'}")
        except DeployError as exc:
            # PodMonitoring is absent on clusters without Managed
            # Prometheus; that is not a failed teardown.
            if kind == "podmonitoring":
                continue
            problems.append(f"{kind}: {exc.message} {exc.detail[:200]}")
    try:
        kube.helm_uninstall(helm_release)
        removed.append(f"helm: {helm_release}")
    except DeployError as exc:
        problems.append(f"helm: {exc.message} {exc.detail[:200]}")

    if record is not None:
        record.mark("CANCELLED" if not problems else "FAILED",
                    error="; ".join(problems) if problems else None,
                    torn_down_at=_now())
    if release is not None:
        release["state"] = "torn_down" if not problems else "teardown_failed"
        _history(release, "teardown", removed=removed, problems=problems)
        render.save_release(release_dir, release)  # type: ignore[arg-type]
    if problems:
        raise DeployError(f"teardown of {name} left resources behind",
                          error_code="TEARDOWN_INCOMPLETE",
                          detail="; ".join(problems))
    return {"name": name, "namespace": namespace, "removed": removed}


# -- reaper integration ---------------------------------------------------------------


class ReleaseReaper:
    """The one method `reap_jobs` needs: cancel a release by its job id."""

    def __init__(self, env: DeployEnv, runner: Any = None) -> None:
        self.env = env
        self.runner = runner

    def __getattr__(self, item: str) -> Any:
        # Only reached for attributes not set above: `gke_release` is a job
        # record backend, not a compute driver, and must say so if someone
        # tries to run on it.
        raise DeployError(
            f"{RELEASE_BACKEND} is not a compute backend; it only supports"
            " teardown through `compute job reap`",
            error_code="INVALID_BACKEND")

    def cancel(self, job_id: str) -> None:
        record = jobs.JobRecord.load(job_id)
        if record is None:
            raise DeployError(f"no job record {job_id}",
                              error_code="RELEASE_NOT_FOUND")
        release_dir = record.extra.get("release_dir")
        teardown(self.env,
                 pathlib.Path(release_dir) if release_dir else None,
                 record=record,
                 include_production=True,
                 runner=self.runner)


def wrap_reap_resolver(resolve: Callable[..., Any],
                       env_loader: Callable[..., DeployEnv] | None = None
                       ) -> Callable[..., Any]:
    """A driver resolver that answers `gke_release` with a `ReleaseReaper`.

    Everything else passes through to the compute resolver untouched, so
    `gke_release` never appears as a compute backend anywhere but here.
    """
    from .config import load_env  # pylint: disable=import-outside-toplevel
    loader = env_loader or load_env

    def _resolve(backend_override: str | None = None, **kwargs: Any) -> Any:
        if backend_override == RELEASE_BACKEND:
            return ReleaseReaper(
                loader(kwargs.get("workspace"), kwargs.get("package")))
        return resolve(backend_override=backend_override, **kwargs)

    return _resolve


def release_by_name(name: str) -> jobs.JobRecord | None:
    return find_release_record(name)


def snapshot_config(release_dir: pathlib.Path) -> dict[str, Any]:
    """The incumbent config as the release directory currently holds it."""
    return copy.deepcopy(
        json.loads(_read(pathlib.Path(release_dir), render.CONFIG_FILE)))
