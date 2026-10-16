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
"""The `deploy` command: how agents reach every verb in this package.

Each verb prints one JSON document on success. Failures print a JSON error
on stderr and exit 2, except a bounded wait that ran out, which exits 3 so
a caller can tell "not yet" from "broken" without parsing text.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

try:
    from framework import utils
except ImportError:
    try:
        from .. import utils  # type: ignore
    except ImportError:  # bundled runtime: utils.py beside runtime.py
        import utils  # type: ignore

from . import bench as bench_mod
from . import collect as collect_mod
from . import preflight as preflight_mod
from . import probe as probe_mod
from . import release as release_mod
from . import render, workload
from .config import load_env, release_dir_for
from .kube import DeployError

EXIT_ERROR = 2
EXIT_TIMEOUT = 3


def _load(path: str) -> dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise DeployError(f"{path} does not exist", error_code="INVALID_FIELD")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DeployError(f"{path} is not valid JSON",
                          error_code="INVALID_FIELD",
                          detail=str(exc)) from exc
    if not isinstance(data, dict):
        raise DeployError(f"{path} must hold a JSON object",
                          error_code="INVALID_FIELD")
    return data


def register_deploy_parser(sub: Any) -> None:
    """Attach `deploy` and its verbs to the runtime's subparsers."""
    # `common` is attached at both the `deploy` level and every verb, and the
    # compiler injects `--package` right after `deploy`. argparse lets the
    # verb's parser re-apply its defaults over values the parent already
    # parsed, so a `None` default here would silently erase the injected
    # package. SUPPRESS leaves the attribute absent instead, and `dispatch`
    # reads all three through `getattr(..., None)`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--package",
                        default=argparse.SUPPRESS,
                        help="Package whose environment.json configures the"
                        " cluster")
    common.add_argument("--workspace",
                        type=pathlib.Path,
                        default=argparse.SUPPRESS,
                        help="Workspace root override")
    common.add_argument("--run-id",
                        default=argparse.SUPPRESS,
                        help="Run the release belongs to (default: active run)")

    p = sub.add_parser("deploy",
                       parents=[common],
                       help="Render, apply, benchmark and tear down llm-d"
                       " releases on GKE")
    verbs = p.add_subparsers(dest="deploy_action", required=True)

    v = verbs.add_parser("preflight", parents=[common],
                         help="Check the cluster is ready for a release")
    v.add_argument("--fix", action="store_true",
                   help="Create the namespace and install GAIE CRDs if missing")
    v.add_argument("--sim", action="store_true",
                   help="A simulated fleet: no TPU node is required")

    v = verbs.add_parser("probe", parents=[common],
                         help="Report what the serving image can do")
    v.add_argument("--image", help="Run a probe Job with this image "
                   "(managed fleets)")
    v.add_argument("--pod", help="Run the probe inside this serving pod "
                   "(adopted fleets)")
    v.add_argument("--selector", help="Run the probe inside the first Ready"
                   " pod matching this label selector")
    v.add_argument("--container")
    v.add_argument("--out", required=True, type=pathlib.Path)
    v.add_argument("--timeout", type=int, default=probe_mod.PROBE_TIMEOUT_S)

    v = verbs.add_parser("workload", parents=[common],
                         help="Inspect the workload catalog or resolve a spec")
    v.add_argument("workload_action", choices=["catalog", "resolve"])
    v.add_argument("--spec", type=pathlib.Path)
    v.add_argument("--out", type=pathlib.Path)

    v = verbs.add_parser("render", parents=[common],
                         help="Render a release directory from config,"
                         " recipe and workload spec")
    v.add_argument("--name", required=True)
    v.add_argument("--recipe", required=True)
    v.add_argument("--workload", required=True,
                   help="Workload spec JSON (not the resolved profile)")
    v.add_argument("--config", default=None,
                   help="Incumbent llmd-config.json; default is the baseline")
    v.add_argument("--replicas", type=int, default=1,
                   help="Replica count for the default baseline config")
    v.add_argument("--out", required=True, type=pathlib.Path)
    v.add_argument("--production", action="store_true")
    v.add_argument("--namespace", default=None,
                   help="Override the environment's namespace")
    v.add_argument("--slo-duration", type=int, default=None,
                   help="Seconds per SLO-stage benchmark run; default"
                   f" {workload.DEFAULT_SLO_STAGE_DURATION_S}")
    v.add_argument("--sweep-duration", type=int, default=None,
                   help="Seconds per rung of an explicit rate.sweep ladder;"
                   f" default {workload.DEFAULT_SWEEP_STAGE_DURATION_S}")
    v.add_argument("--exposure", choices=list(render.EXPOSURE_MODES),
                   default=None,
                   help="Front door of a --production release; default"
                   f" {render.EXPOSURE_CLUSTER_IP}")

    v = verbs.add_parser("apply", parents=[common],
                         help="Apply a rendered release")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None, help="Release directory")
    v.add_argument("--production", action="store_true")

    v = verbs.add_parser("wait", parents=[common],
                         help="Block until the release is ready")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--timeout", type=int,
                   default=release_mod.DEFAULT_WAIT_TIMEOUT_S)
    v.add_argument("--poll", type=int, default=release_mod.DEFAULT_POLL_S)

    v = verbs.add_parser("reconfigure", parents=[common],
                         help="Change hot/warm paths of a live release")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")

    v = verbs.add_parser("bench", parents=[common],
                         help="Run the release's inference-perf Jobs")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--stage", choices=list(render.BENCH_STAGES), required=True)
    v.add_argument("--replicates", type=int, default=1)
    v.add_argument("--out", required=True, type=pathlib.Path)
    v.add_argument("--execution-id", default=None)
    v.add_argument("--timeout", type=int,
                   default=bench_mod.DEFAULT_JOB_TIMEOUT_S)

    v = verbs.add_parser("collect", parents=[common],
                         help="Fleet metrics over a window from Prometheus")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--window", default="10m")
    v.add_argument("--out", required=True, type=pathlib.Path)

    v = verbs.add_parser("drill", parents=[common],
                         help="Run one fault drill and measure recovery")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--drill", choices=list(release_mod.DRILLS),
                   default=release_mod.DRILL_EPP_POD_KILL)
    v.add_argument("--timeout", type=int,
                   default=release_mod.DEFAULT_DRILL_TIMEOUT_S)
    v.add_argument("--out", type=pathlib.Path, default=None,
                   help="Also write the drill row to this JSON file")

    for verb, help_text in (("status", "Release state on the cluster"),
                            ("events", "Recent events for the release")):
        v = verbs.add_parser(verb, parents=[common], help=help_text)
        v.add_argument("--name", required=True)
        v.add_argument("--release", default=None)

    v = verbs.add_parser("logs", parents=[common], help="Component logs")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--component", choices=["modelserver", "epp", "bench"],
                   default="modelserver")
    v.add_argument("--tail", type=int, default=200)

    v = verbs.add_parser("teardown", parents=[common],
                         help="Remove the release")
    v.add_argument("--name", required=True)
    v.add_argument("--release", default=None)
    v.add_argument("--include-production", action="store_true")


def _run_id(args: argparse.Namespace) -> str:
    run_id = getattr(args, "run_id", None) or utils.get_active_run_id(
        getattr(args, "workspace", None))
    if not run_id:
        raise DeployError(
            "apply needs a run to file the release under: pass --run-id or"
            " run inside an initialized session",
            error_code="MISSING_FIELD")
    return run_id


def dispatch(args: argparse.Namespace) -> dict[str, Any] | str:
    env = load_env(getattr(args, "workspace", None),
                   getattr(args, "package", None))
    action = args.deploy_action

    if action == "preflight":
        return preflight_mod.run_preflight(env, fix=args.fix,
                                           sim=bool(getattr(args, "sim",
                                                            False)))

    if action == "probe":
        return probe_mod.run_probe(env,
                                   image=args.image,
                                   out_dir=args.out,
                                   pod=args.pod,
                                   selector=args.selector,
                                   container=args.container,
                                   timeout_s=args.timeout)

    if action == "workload":
        if args.workload_action == "catalog":
            return {
                "inference_perf_ref": workload.catalog_index()[
                    "inference_perf_ref"],
                "entries": {
                    name: workload.catalog_summary(name)
                    for name in workload.catalog_entries()
                },
            }
        if not args.spec:
            raise DeployError("workload resolve needs --spec",
                              error_code="MISSING_FIELD")
        profile = workload.resolve_workload(_load(str(args.spec)),
                                            spec_dir=args.spec.parent)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / render.PROFILE_FILE).write_text(
                json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        shown = dict(profile)
        shown.pop("base_config", None)
        return shown

    if action == "render":
        recipe = _load(args.recipe)
        spec_path = pathlib.Path(args.workload)
        profile = workload.resolve_workload(_load(str(spec_path)),
                                            spec_dir=spec_path.parent)
        config = (_load(args.config) if args.config else
                  render.default_config(recipe, replicas=args.replicas))
        constraints = profile.get("constraints") or {}
        namespace = (args.namespace or constraints.get("namespace")
                     or (env.production_namespace if args.production and
                         env.production_namespace else env.namespace))
        env_subset = {
            "namespace": namespace,
            "hf_token_secret": env.hf_token_secret,
            "inference_perf_image": env.inference_perf_image,
            "gateway_mode": env.gateway_mode,
        }
        if env.sizing:
            env_subset["sizing"] = env.sizing
        if env.cpu_node_selector:
            env_subset["cpu_node_selector"] = env.cpu_node_selector
        if getattr(args, "slo_duration", None):
            env_subset["slo_stage_duration_s"] = int(args.slo_duration)
        if getattr(args, "sweep_duration", None):
            env_subset["sweep_stage_duration_s"] = int(args.sweep_duration)
        if getattr(args, "exposure", None):
            if not args.production:
                raise DeployError("--exposure only applies to a --production"
                                  " render", error_code="INVALID_FIELD")
            env_subset["exposure"] = args.exposure
        return render.render_release(name=args.name,
                                     config=config,
                                     recipe=recipe,
                                     profile=profile,
                                     env=env_subset,
                                     out_dir=args.out,
                                     production=args.production)

    if action == "apply":
        rd = release_dir_for(args.name, args.release)
        return release_mod.apply(env,
                                 rd,
                                 run_id=_run_id(args),
                                 production=args.production)

    rd = release_dir_for(args.name, getattr(args, "release", None))
    if action == "wait":
        return release_mod.wait(env, rd, timeout_s=args.timeout,
                                poll_s=args.poll)
    if action == "drill":
        row = release_mod.drill(env, rd, name=args.drill,
                                timeout_s=args.timeout)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(row, indent=2) + "\n",
                                encoding="utf-8")
        return row
    if action == "reconfigure":
        sets: dict[str, Any] = {}
        for item in args.set:
            if "=" not in item:
                raise DeployError(f"--set {item!r} is not PATH=VALUE",
                                  error_code="INVALID_FIELD")
            path, value = item.split("=", 1)
            sets[path.strip()] = release_mod.parse_set_value(value.strip())
        return release_mod.reconfigure(env, rd, sets)
    if action == "bench":
        return bench_mod.run_bench(env,
                                   rd,
                                   stage=args.stage,
                                   replicates=args.replicates,
                                   out_dir=args.out,
                                   execution_id=args.execution_id,
                                   timeout_s=args.timeout)
    if action == "collect":
        return collect_mod.collect(env, rd, window=args.window,
                                   out_dir=args.out)
    if action == "status":
        return release_mod.status(env, rd)
    if action == "events":
        return {"events": release_mod.events(env, rd)}
    if action == "logs":
        return release_mod.logs(env, rd, component=args.component,
                                tail=args.tail)
    if action == "teardown":
        return release_mod.teardown(
            env, rd, include_production=args.include_production)
    raise DeployError(f"unknown deploy action {action!r}",
                      error_code="INVALID_FIELD")


def handle_deploy_cli(args: argparse.Namespace) -> int:
    try:
        result = dispatch(args)
    except DeployError as exc:
        print(json.dumps(exc.to_dict()), file=sys.stderr)
        return EXIT_TIMEOUT if exc.error_code == "WAIT_TIMEOUT" else EXIT_ERROR
    if isinstance(result, str):
        print(result, end="" if result.endswith("\n") else "\n")
    else:
        print(utils.format_json(result))
    return 0
