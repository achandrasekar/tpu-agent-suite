#!/usr/bin/env python3
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
"""Promote a study's incumbent llm-d configuration to a production bundle.

Three subcommands, in the order the productionizer runs them around the
runtime's `deploy render --production / apply --production / wait / bench /
drill` verbs:

  prepare --request R --out DIR
      The guard and the production config. Refuses (exit 2, a `refused`
      result on disk) unless `promotion_approved` is literally true, then
      writes `production-config.json`: the incumbent byte-for-byte plus the
      only production settings this step may add -- EPP replica count and
      HA mode. Prints the release name, namespace and exposure the caller
      must pass to `deploy render`.

  bundle --request R --out DIR --release-dir REL [--study-state S]
         [--acceptance BENCH] [--baseline-report BENCH] [--drill ROW ...]
      Writes the handoff bundle the design doc (§9) lists: the incumbent,
      the search ledger from the study state, the rendered router values,
      EPP config, InferenceObjectives, exposure, workload, observability,
      an autoscaling note, RUNBOOK.md, acceptance-report.md and release.json.
      Also judges acceptance -- every gate in the acceptance report passed,
      the primary metric is inside the noise floor of the incumbent's own
      measurement or better, every drill passed -- and writes the
      `llmd-promote-result.json` envelope.

  refuse --request R --out DIR --reason TEXT
      Writes a `refused` result for a reason the caller established
      (an existing production release, a user who did not confirm).

The configuration is never edited here beyond `router.epp_replicas` and
`router.epp_ha_mode`: what goes to production is what was measured.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any

RESULT_FILE = "llmd-promote-result.json"
PRODUCTION_CONFIG_FILE = "production-config.json"
EXIT_REFUSED = 2
EXIT_ACCEPTANCE_FAILED = 3

# Design §6.5 asks for more than this cluster, this runtime and an adopted
# fleet can establish. Each is written into the bundle as UNVERIFIED rather
# than left out, so the reader knows what was not done and why.
UNVERIFIED_ALWAYS = [
    "autoscaling: no ScaledObject/HPA is rendered; an adopted fleet's pods"
    " are not the release's to scale and no soak was run against a scaler",
    "zonal failover: not exercised (single cluster, single zone)",
    "canary rollout: the release is applied whole; weighted splits need a"
    " second router release or Gateway-mode HTTPRoute weights",
]
UNVERIFIED_ADOPTED = [
    "model-server drills (decode/prefill pod kill): refused on an adopted"
    " fleet, whose pods the release does not own",
]
UNVERIFIED_PASSTHROUGH = [
    "EPP HA, exposure and the EPP drill: the incumbent is the passthrough"
    " profile, so no router was installed",
]


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")


def _load(path: str | pathlib.Path) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n",
                    encoding="utf-8")


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _to_yaml(doc: Any, indent: int = 0) -> str:
    """A small JSON-to-YAML printer: the bundle is read by people and by
    cluster tooling that expects YAML, and PyYAML is not a runtime dependency."""
    pad = "  " * indent
    if isinstance(doc, dict):
        if not doc:
            return pad + "{}"
        lines = []
        for key, value in doc.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.append(_to_yaml(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
        return "\n".join(lines)
    if isinstance(doc, list):
        if not doc:
            return pad + "[]"
        lines = []
        for item in doc:
            if isinstance(item, (dict, list)) and item:
                body = _to_yaml(item, indent + 1)
                first, _, rest = body.partition("\n")
                lines.append(f"{pad}- {first.strip()}")
                if rest:
                    lines.append(rest)
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return pad + _scalar(doc)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    text = str(value)
    if text == "" or any(c in text for c in ":#{}[],&*?|<>=!%@`'\"\n") \
            or text.strip() != text or text.lower() in ("yes", "no", "on",
                                                        "off", "null",
                                                        "true", "false"):
        return json.dumps(text)
    return text


# -- guard ------------------------------------------------------------------------


def release_name_for(request: dict[str, Any]) -> str:
    name = request.get("release_name") or f"{request['study_id']}-prod"
    name = name.lower()
    return name[:40].rstrip("-")


def refusal(request: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "refused",
        "release_name": release_name_for(request),
        "namespace": request.get("production_namespace"),
        "summary": reason,
        "unverified": [],
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    request = _load(args.request)
    out = pathlib.Path(args.out)
    if request.get("promotion_approved") is not True:
        result = refusal(
            request, "promotion_approved is not true: the user has not"
            " approved moving the incumbent to production in this session."
            " Nothing was rendered or applied.")
        _write_json(out / RESULT_FILE, result)
        print(json.dumps(result, indent=2))
        return EXIT_REFUSED
    if request.get("existing_production_release"):
        result = refusal(
            request, f"a production release"
            f" ({request['existing_production_release']}) already serves this"
            " fleet; say what happens to it before a second one is applied.")
        _write_json(out / RESULT_FILE, result)
        print(json.dumps(result, indent=2))
        return EXIT_REFUSED

    incumbent = _load(request["incumbent_config_path"])
    config = copy.deepcopy(incumbent)
    router = config.setdefault("router", {})
    passthrough = router.get("profile", "passthrough") == "passthrough"
    if not passthrough:
        router["epp_replicas"] = int(request.get("epp_replicas") or 2)
        if router["epp_replicas"] > 1:
            router["epp_ha_mode"] = request.get("epp_ha_mode") or \
                router.get("epp_ha_mode") or "active-passive"
    provenance = dict(config.get("provenance") or {})
    provenance["promoted_from"] = {
        "study_id": request["study_id"],
        "incumbent_config_path": request["incumbent_config_path"],
        "incumbent_digest": _digest(
            pathlib.Path(request["incumbent_config_path"])),
        "promoted_at": _now(),
        "production_changes": ([] if passthrough else
                               ["router.epp_replicas", "router.epp_ha_mode"]),
    }
    config["provenance"] = provenance
    _write_json(out / PRODUCTION_CONFIG_FILE, config)
    (out / RESULT_FILE).unlink(missing_ok=True)
    print(json.dumps({
        "status": "prepared",
        "release_name": release_name_for(request),
        "namespace": request["production_namespace"],
        "exposure": request.get("exposure") or "cluster-ip",
        "config_path": str(out / PRODUCTION_CONFIG_FILE),
        "router_installed": not passthrough,
        "epp_replicas": router.get("epp_replicas"),
        "epp_ha_mode": router.get("epp_ha_mode"),
    }, indent=2))
    return 0


def cmd_refuse(args: argparse.Namespace) -> int:
    request = _load(args.request)
    result = refusal(request, args.reason)
    _write_json(pathlib.Path(args.out) / RESULT_FILE, result)
    print(json.dumps(result, indent=2))
    return EXIT_REFUSED


# -- acceptance ----------------------------------------------------------------------


def judge_acceptance(acceptance: dict[str, Any] | None,
                     baseline: dict[str, Any] | None,
                     drills: list[dict[str, Any]],
                     noise_floor_pct: float,
                     router_installed: bool) -> dict[str, Any]:
    """Gates, metric-vs-incumbent and drills, each with its own verdict.

    The production release is the incumbent re-applied in its own
    namespace, so the right expectation is "the same" -- inside the noise
    floor of the incumbent's measurement -- not "better". Better is
    allowed; worse by more than the floor is a failed acceptance, because it
    means the production render is not what was measured.
    """
    rows: list[dict[str, Any]] = []
    passed = True
    if acceptance is None:
        rows.append({"check": "acceptance benchmark", "passed": False,
                     "detail": "no acceptance report"})
        passed = False
    else:
        for gate in acceptance.get("gates") or []:
            rows.append({
                "check": f"gate {gate['name']}",
                "passed": bool(gate.get("passed")),
                "detail": f"{gate.get('value')} {gate.get('comparator')}"
                          f" {gate.get('bound')} {gate.get('unit', '')}",
            })
            passed = passed and bool(gate.get("passed"))
        if not acceptance.get("admissible", False):
            rows.append({"check": "admissible", "passed": False,
                         "detail": "the report is not admissible"})
            passed = False
        primary = acceptance.get("primary_metric")
        value = ((acceptance.get("metrics") or {}).get(primary) or {}).get(
            "value")
        if baseline is not None and value is not None:
            ref = ((baseline.get("metrics") or {}).get(primary) or {}).get(
                "value")
            if ref:
                delta = (value - ref) / ref * 100.0
                ok = delta >= -float(noise_floor_pct)
                rows.append({
                    "check": f"{primary} vs incumbent measurement",
                    "passed": ok,
                    "detail": f"{value:.4g} vs {ref:.4g} ({delta:+.1f}%,"
                              f" floor -{noise_floor_pct}%)",
                })
                passed = passed and ok
        else:
            rows.append({"check": "metric vs incumbent measurement",
                         "passed": True,
                         "detail": "no incumbent report given; gates only"})
    if router_installed:
        if not drills:
            rows.append({"check": "drill epp-pod-kill", "passed": False,
                         "detail": "not run"})
            passed = False
        for row in drills:
            rows.append({
                "check": f"drill {row.get('drill')}",
                "passed": bool(row.get("passed")),
                "detail": f"killed {row.get('killed')}; recovered in"
                          f" {row.get('recovery_s')} s; front door"
                          f" {'ok' if row.get('front_door_ok') else 'down'}",
            })
            passed = passed and bool(row.get("passed"))
    return {"passed": passed, "rows": rows}


# -- bundle ------------------------------------------------------------------------


def search_ledger(study: dict[str, Any] | None,
                  incumbent_arm: str | None) -> str:
    lines = ["# Search ledger", ""]
    if not study:
        lines.append("No study state was supplied; see the study's own"
                     " `study notes` output.")
        return "\n".join(lines)
    protocol = study.get("protocol") or {}
    metric = protocol.get("primary_metric") or {}
    lines += [
        f"Study `{study.get('study_id')}`, status {study.get('status')},"
        f" optimizing `{metric.get('name')}` ({metric.get('direction')},"
        f" {metric.get('unit')}); noise floor {protocol.get('noise_floor_pct')}%"
        f" at {protocol.get('replicate_count')} replicate(s).",
        "",
    ]
    baseline = study.get("baseline")
    if baseline:
        lines += [
            f"Baseline: {baseline['metric'].get('value')}"
            f" {baseline['metric'].get('unit', '')} from execution"
            f" `{baseline.get('execution_id')}`.",
            "",
        ]
    lines += [
        "| arm | kind | status | verdict | metric | delta | execution | claim |",
        "| :-- | :-- | :-- | :-- | --: | --: | :-- | :-- |",
    ]
    for arm in study.get("arms") or []:
        result = arm.get("result") or {}
        met = result.get("metric") or {}
        value = met.get("value")
        delta = met.get("delta_pct")
        mark = " **(incumbent)**" if arm.get("arm") == incumbent_arm else ""
        lines.append(
            f"| {arm.get('arm')}{mark} | {arm.get('kind')} |"
            f" {arm.get('status')} | {result.get('verdict', '-')} |"
            f" {'' if value is None else f'{value:.4g}'} |"
            f" {'' if delta is None else f'{delta:+.1f}%'} |"
            f" {result.get('execution_id', '')} |"
            f" {str(arm.get('claim', '')).replace('|', '/')} |")
    blocked = [a for a in study.get("arms") or [] if a.get("status") ==
               "blocked"]
    if blocked:
        lines += ["", "## Blocked arms (findings, not failures)", ""]
        for arm in blocked:
            lines.append(f"- `{arm.get('arm')}` ({arm.get('kind')}):"
                         f" {arm.get('blocked_on', 'blocked')}")
    return "\n".join(lines)


def acceptance_report(request: dict[str, Any], release: dict[str, Any],
                      judged: dict[str, Any],
                      acceptance: dict[str, Any] | None,
                      unverified: list[str]) -> str:
    lines = [
        f"# Acceptance report: {release['name']}",
        "",
        f"Namespace `{release['namespace']}`; config digest"
        f" `{release['config_digest']}`; model `{release['model']}`;"
        f" fleet mode `{(release.get('fleet') or {}).get('mode')}`;"
        f" router `{release['router'].get('profile')}`"
        f" ({'installed' if release['router'].get('installed') else 'not installed'}).",
        "",
        f"**Result: {'ACCEPTED' if judged['passed'] else 'NOT ACCEPTED'}**",
        "",
        "| check | result | detail |",
        "| :-- | :-- | :-- |",
    ]
    for row in judged["rows"]:
        lines.append(f"| {row['check']} | {'pass' if row['passed'] else 'FAIL'}"
                     f" | {row['detail']} |")
    if acceptance:
        lines += [
            "",
            "## Acceptance measurement",
            "",
            f"- execution: `{acceptance.get('execution_id')}`"
            f" ({acceptance.get('stage')} stage,"
            f" {acceptance.get('replicates')} replicate(s))",
            f"- primary metric: `{acceptance.get('primary_metric')}` ="
            f" {((acceptance.get('metrics') or {}).get(acceptance.get('primary_metric')) or {}).get('value')}",
            f"- raw reports: `{acceptance.get('raw_dir')}`",
        ]
        for tenant, block in (acceptance.get("tenants") or {}).items():
            parts = []
            for key, slo in sorted((block.get("slo") or {}).items()):
                measured = (slo or {}).get("measured_ms")
                if isinstance(measured, (int, float)):
                    parts.append(f"{key}={measured:.1f}")
            rate = block.get("error_rate")
            if isinstance(rate, (int, float)):
                # One decimal hid a 0.03 % rate behind "0.0"; the gate
                # judged the exact value, so show it.
                parts.append(f"error_rate={rate:.6g}")
            if "requests_per_sec" in block:
                parts.append(f"rps={block['requests_per_sec']:.2f}")
            lines.append(f"- tenant `{tenant}`: " + ", ".join(parts))
    lines += ["", "## Not verified", ""]
    for item in unverified:
        lines.append(f"- UNVERIFIED: {item}")
    lines += [
        "",
        "## Provenance",
        "",
        f"- promoted from study `{request['study_id']}` at {_now()}",
        f"- incumbent: `{request['incumbent_config_path']}`",
        f"- llm-d ref `{(release.get('pins') or {}).get('llmd_ref')}`,"
        f" router chart `{release['router'].get('chart_version')}`,"
        f" inference-perf `{(release.get('pins') or {}).get('inference_perf_ref')}`",
    ]
    return "\n".join(lines)


def runbook(request: dict[str, Any], release: dict[str, Any],
            bundle_dir: pathlib.Path) -> str:
    name = release["name"]
    ns = release["namespace"]
    adopted = (release.get("fleet") or {}).get("mode") == "adopted"
    front = (release.get("exposure") or {}).get("host") or release["router"][
        "service"]
    rt = "{{runtime}} deploy"
    tenants = [str(t) for t in (release.get("workload") or {}).get(
        "tenants") or []]
    # Release-scoped, matching framework.deploy.render.objective_name; the
    # runtime is not importable from a package script so the rule is
    # repeated here and pinned by the bundle test.
    objectives = ", ".join(
        f"`{t}` -> `x-llm-d-inference-objective: {name.lower()[:40]}-{t.lower()}`"
        for t in tenants)
    lines = [
        f"# Runbook: {name}",
        "",
        f"Production llm-d release `{name}` in namespace `{ns}`."
        f" Front door: `http://{front}:{release['router'].get('port', 80)}`"
        " (OpenAI-compatible; `/v1/models`, `/v1/completions`,"
        " `/v1/chat/completions`).",
        "",
        "Clients send their tenant's objective header so the router places"
        " them in the right priority band (without it a request is"
        " served at the default priority): " + (objectives or "none"),
        "",
        "Every operation below goes through the runtime so the release's"
        " history records it. The bundle directory is"
        f" `{bundle_dir}`; the rendered release it was applied from is"
        f" `{bundle_dir / 'release'}`.",
        "",
        "## Check health",
        "",
        "```bash",
        f"{rt} status --name {name} --release {bundle_dir / 'release'}",
        f"{rt} events --name {name} --release {bundle_dir / 'release'}",
        f"{rt} logs --name {name} --release {bundle_dir / 'release'} --component epp",
        "```",
        "",
        "## Re-tune a hot path (router weights, flow-control thresholds)",
        "",
        "Hot paths re-render the EPP config and roll the EPP only; the model"
        " servers are untouched. Measure before you keep it.",
        "",
        "```bash",
        f"{rt} reconfigure --name {name} --release {bundle_dir / 'release'}"
        " --set router.plugins.queue-scorer.weight=2",
        f"{rt} bench --name {name} --release {bundle_dir / 'release'}"
        " --stage slo --replicates 1 --out <dir>",
        "```",
        "",
        "## Roll back",
        "",
        "The incumbent that was promoted is `llmd-config.json` in this"
        " bundle. Re-render it with `--production` and apply; the chart"
        " upgrade is in place and the InferencePool keeps its endpoints.",
        "",
        "```bash",
        f"{rt} render --name {name} --recipe {bundle_dir / 'serving-recipe.json'}"
        f" --workload {bundle_dir / 'workload' / 'workload-spec.json'}"
        f" --config {bundle_dir / 'llmd-config.json'} --production"
        f" --exposure {(release.get('exposure') or {}).get('mode', 'cluster-ip')}"
        f" --out {bundle_dir / 'release'}",
        f"{rt} apply --name {name} --release {bundle_dir / 'release'} --production",
        f"{rt} wait --name {name} --release {bundle_dir / 'release'}",
        "```",
        "",
        "## Scale",
        "",
    ]
    if adopted:
        lines += [
            "This release adopted a fleet it does not own"
            f" (selector `{(release.get('fleet') or {}).get('selector')}`)."
            " Scaling the model servers is the fleet owner's operation; the"
            " router's InferencePool follows the selector, so new pods join"
            " routing on their own. Re-run the study (below) after a scale"
            " change: the tuned router profile was measured at"
            f" {release.get('replicas')} replicas.",
        ]
    else:
        lines += [
            "Replica count is a cold path: re-render with"
            " `topology.decode.replicas` changed and apply. Autoscaling is"
            " UNVERIFIED for this release (see acceptance-report.md).",
        ]
    lines += [
        "",
        "## Rotate the serving image",
        "",
        "A new image is a new recipe: run `tpu-onboarding` for the"
        " single-replica tuning, then re-run the study with the new recipe;"
        " the router profile that won on the old engine is a hypothesis on"
        " the new one, not a result.",
        "",
        "## Re-run the study",
        "",
        "Open a new study with the same workload spec and this recipe; the"
        " incumbent here is its natural first arm. The search ledger"
        " (`search-ledger.md`) says which arms were blocked on this cluster"
        " and why.",
        "",
        "## Drill",
        "",
        "```bash",
        f"{rt} drill --name {name} --release {bundle_dir / 'release'}"
        " --drill epp-pod-kill",
        "```",
        "",
        "## Remove the release",
        "",
        "Production is guarded: teardown refuses without"
        " `--include-production`, and the hook `llmd-production-guard` keeps"
        " agents from passing that flag. Removal is a user action: place the"
        f" approval marker `{bundle_dir / '.teardown-approved'}` (an empty"
        " file next to this runbook), then run the teardown yourself."
        + (" The adopted model servers are not part of the release and stay"
           " up." if adopted else
           " The managed model servers are part of the release and go with"
           " it."),
        "",
        "```bash",
        f"touch {bundle_dir / '.teardown-approved'}",
        f"{rt} teardown --name {name} --release {bundle_dir / 'release'}"
        " --include-production",
        "```",
        "",
        "## Migrate to GKE Inference Gateway",
        "",
        "Set `llmd.gateway_mode: gke` in the environment and re-render; the"
        " EPP config and InferenceObjectives carry over, the exposure becomes"
        " a Gateway + HTTPRoute. Not exercised by this promotion.",
    ]
    return "\n".join(lines)


def autoscaling_note(release: dict[str, Any]) -> str:
    return "\n".join([
        "# Autoscaling: UNVERIFIED",
        "",
        "No ScaledObject or HPA is part of this bundle.",
        "",
        "- The fleet mode is"
        f" `{(release.get('fleet') or {}).get('mode')}`."
        + (" An adopted fleet's Deployment is not the release's to scale."
           if (release.get('fleet') or {}).get('mode') == 'adopted' else
           " A managed fleet can be scaled on `topology.decode.replicas`,"
           " a cold path."),
        "- No KEDA installation was found or installed, and no soak was run"
        " against a scaler, so the thresholds the design derives from the"
        " saturation model (`llm_d_epp_inflight_tokens` against TTFT,"
        " KV occupancy for decode) were not calibrated here.",
        "",
        "When autoscaling is wanted: install KEDA, scale on the EPP's"
        " inflight-token metric against the measured knee in"
        " `acceptance-report.md`, min replicas = the capacity floor the"
        " study measured, max = the chip budget, and re-run acceptance under"
        " the sweep ladder so the scaler's reaction is part of the"
        " measurement.",
    ])


def cmd_bundle(args: argparse.Namespace) -> int:
    request = _load(args.request)
    out = pathlib.Path(args.out)
    release_dir = pathlib.Path(args.release_dir)
    release = _load(release_dir / "release.json")
    if not release.get("production"):
        raise SystemExit("the release directory was not rendered with"
                         " --production; refusing to bundle a study render")
    study = _load(args.study_state) if args.study_state else None
    acceptance = _load(args.acceptance) if args.acceptance else None
    baseline = _load(args.baseline_report) if args.baseline_report else None
    drills = [_load(p) for p in (args.drill or [])]
    router_installed = bool(release["router"].get("installed"))
    adopted = (release.get("fleet") or {}).get("mode") == "adopted"
    noise = float(((request.get("acceptance") or {}).get("noise_floor_pct"))
                  or 5.0)

    judged = judge_acceptance(acceptance, baseline, drills, noise,
                              router_installed)
    unverified = list(UNVERIFIED_ALWAYS)
    if adopted:
        unverified += UNVERIFIED_ADOPTED
    if not router_installed:
        unverified += UNVERIFIED_PASSTHROUGH

    # The bundle. The rendered release directory travels with it so the
    # runbook's commands point at files that exist.
    out.mkdir(parents=True, exist_ok=True)
    bundle_release = out / "release"
    if bundle_release.resolve() != release_dir.resolve():
        if bundle_release.exists():
            shutil.rmtree(bundle_release)
        shutil.copytree(release_dir, bundle_release)
    shutil.copy2(release_dir / "llmd-config.json", out / "llmd-config.json")
    shutil.copy2(release_dir / "serving-recipe.json",
                 out / "serving-recipe.json")
    workload_dir = out / "workload"
    workload_dir.mkdir(exist_ok=True)
    shutil.copy2(request["workload_spec_path"],
                 workload_dir / "workload-spec.json")
    shutil.copy2(release_dir / "workload_profile.json",
                 workload_dir / "workload_profile.json")
    for key, files in ((release.get("files") or {}).get("bench") or {}).items():
        cm = _load(release_dir / files["configmap"])
        cfg = (cm.get("data") or {}).get("config.json")
        if cfg:
            _write_text(workload_dir / f"inference-perf.{key}.json", cfg)

    if router_installed:
        values = _load(release_dir / release["files"]["router_values"])
        _write_text(out / "router.values.yaml", _to_yaml(values))
        epp = _load(release_dir / release["files"]["epp_config"])
        _write_text(out / "epp-config.yaml", _to_yaml(epp))
        objectives = (values.get("router") or {}).get("inferenceObjectives")
        _write_text(out / "inference-objectives.yaml", _to_yaml({
            "note": "Rendered by the llm-d-router-standalone chart from"
                    " router.inferenceObjectives; listed here for review.",
            "inferenceObjectives": objectives or [],
        }))
        if release.get("files", {}).get("exposure"):
            exposure = _load(release_dir / release["files"]["exposure"])
            _write_text(out / "exposure.yaml", _to_yaml(exposure))
    obs = out / "observability"
    obs.mkdir(exist_ok=True)
    monitoring = _load(release_dir / release["files"]["monitoring"])
    _write_text(obs / "podmonitoring.yaml", _to_yaml(monitoring))
    _write_text(obs / "README.md", "\n".join([
        "# Observability",
        "",
        "`podmonitoring.yaml` is what the release applied: Managed"
        " Prometheus scrapes of the model servers' `/metrics` and the EPP's."
        " Fleet metrics over a window come from"
        " `{{runtime}} deploy collect --window 10m`.",
        "",
        "Dashboards and alert policies are not part of this bundle"
        " (UNVERIFIED); the SLO the alerts should burn against is in"
        " `workload/workload-spec.json`.",
    ]))
    (out / "autoscaling").mkdir(exist_ok=True)
    _write_text(out / "autoscaling" / "README.md", autoscaling_note(release))

    incumbent_arm = None
    if study:
        for arm in study.get("arms") or []:
            if (arm.get("result") or {}).get("verdict") == "win":
                incumbent_arm = arm.get("arm")
    _write_text(out / "search-ledger.md", search_ledger(study, incumbent_arm))
    _write_text(out / "acceptance-report.md",
                acceptance_report(request, release, judged, acceptance,
                                  unverified))
    _write_text(out / "RUNBOOK.md", runbook(request, release, out))
    _write_json(out / "release.json", {
        "release": release["name"],
        "namespace": release["namespace"],
        "production": True,
        "state": release.get("state"),
        "config_digest": release["config_digest"],
        "incumbent_digest": _digest(out / "llmd-config.json"),
        "model": release["model"],
        "image": release["image"],
        "accelerator": release["accelerator"],
        "fleet": release.get("fleet"),
        "router": {k: release["router"].get(k) for k in (
            "installed", "profile", "chart", "chart_version", "epp_image",
            "epp_ha", "service", "port")},
        "exposure": release.get("exposure"),
        "pins": release.get("pins"),
        "acceptance_execution_id": (acceptance or {}).get("execution_id"),
        "accepted": judged["passed"],
        "bundled_at": _now(),
        "files": sorted(str(p.relative_to(out)) for p in out.rglob("*")
                        if p.is_file() and "release/" not in
                        str(p.relative_to(out)) + "/"),
    })

    endpoint = None
    if router_installed:
        host = (release.get("exposure") or {}).get("host") or release[
            "router"]["service"]
        endpoint = f"http://{host}:{release['router'].get('port', 80)}"
    else:
        fleet = release.get("fleet") or {}
        if fleet.get("service_host"):
            endpoint = f"http://{fleet['service_host']}:{fleet['service_port']}"
    result = {
        "status": "promoted" if judged["passed"] else "acceptance_failed",
        "release_name": release["name"],
        "namespace": release["namespace"],
        "endpoint": endpoint,
        "bundle_path": str(out),
        "config_digest": release["config_digest"],
        "acceptance": {
            "execution_id": (acceptance or {}).get("execution_id"),
            "metric": ((acceptance or {}).get("metrics") or {}).get(
                (acceptance or {}).get("primary_metric")),
            "gates": judged["rows"],
            "drills": drills,
            "passed": judged["passed"],
        },
        "unverified": unverified,
        "summary": (
            f"{release['name']} applied to {release['namespace']};"
            f" acceptance {'passed' if judged['passed'] else 'FAILED'}"
            f" ({sum(1 for r in judged['rows'] if r['passed'])}/"
            f"{len(judged['rows'])} checks); {len(unverified)} item(s)"
            " UNVERIFIED; bundle at " + str(out)),
        "artifacts": {
            "bundle": str(out),
            "acceptance_report": str(out / "acceptance-report.md"),
            "runbook": str(out / "RUNBOOK.md"),
            "config": str(out / "llmd-config.json"),
        },
    }
    _write_json(out / RESULT_FILE, result)
    print(json.dumps(result, indent=2))
    return 0 if judged["passed"] else EXIT_ACCEPTANCE_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("refuse")
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_refuse)
    p = sub.add_parser("bundle")
    p.add_argument("--request", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--release-dir", required=True)
    p.add_argument("--study-state", default=None)
    p.add_argument("--acceptance", default=None)
    p.add_argument("--baseline-report", default=None)
    p.add_argument("--drill", action="append", default=[])
    p.set_defaults(func=cmd_bundle)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
