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
"""Run a release's benchmark Jobs and score the result against the SLOs.

One `bench` invocation is one execution: K replicates of every tenant's
inference-perf Job for one stage, all tenants of a replicate running at the
same time so the fleet sees the mixed load the SLOs are about. The Jobs
print their reports to their logs between markers (see `render.bench_job`),
so this module reads logs and nothing else.

The output, `llmd-bench-report.json`, carries the raw inference-perf
numbers, each tenant's SLO attainment, the two primary-metric candidates
(`goodput_at_slo_per_chip`, `cost_per_1M_tokens_at_slo`) and the gate
results, so an arm's verdict can be computed from this one file.

inference-perf report fields:
  https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/reports.md
  https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/goodput.md
"""

from __future__ import annotations

import datetime
import json
import pathlib
import statistics
import time
import uuid
from typing import Any, Callable

from . import render, workload
from .config import DeployEnv
from .kube import DeployError, Kube

DEFAULT_JOB_TIMEOUT_S = 3600
POLL_S = 20
# Where inference-perf puts TPOT; the first key found is used, and the
# report says which so a change upstream is visible rather than silent.
TPOT_FIELDS = ("time_per_output_token", "normalized_time_per_output_token")
GOODPUT_RPS_KEYS = ("request_goodput", "good_requests_per_sec",
                    "requests_per_sec")
GOODPUT_TPS_KEYS = ("token_goodput", "good_tokens_per_sec",
                    "total_tokens_per_sec")


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")


def new_execution_id() -> str:
    return (datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])


# -- log parsing -------------------------------------------------------------------


def parse_report_logs(text: str) -> dict[str, Any]:
    """Reports echoed by the bench Job, keyed by file basename."""
    reports: dict[str, Any] = {}
    current: str | None = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw
        # `kubectl logs --prefix` prepends "[pod/.../container] ".
        if line.startswith("[") and "] " in line:
            line = line.split("] ", 1)[1]
        if line.startswith(render.BENCH_REPORT_MARK):
            current = line[len(render.BENCH_REPORT_MARK):].strip()
            buf = []
            continue
        if line.strip() == render.BENCH_REPORT_END and current is not None:
            try:
                reports[pathlib.Path(current).name] = json.loads(
                    "\n".join(buf))
            except ValueError:
                reports[pathlib.Path(current).name] = {
                    "_unparseable": "\n".join(buf)[:2000]
                }
            current = None
            buf = []
            continue
        if current is not None:
            buf.append(line)
    return reports


def _find(reports: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    for name, doc in reports.items():
        if name.endswith(suffix) and isinstance(doc, dict):
            return doc
    return None


def _stage_reports(reports: dict[str, Any]) -> list[tuple[int, dict]]:
    out = []
    for name, doc in reports.items():
        if "stage_" in name and name.endswith("_lifecycle_metrics.json"):
            try:
                idx = int(name.split("stage_", 1)[1].split("_", 1)[0])
            except ValueError:
                continue
            if isinstance(doc, dict):
                out.append((idx, doc))
    return sorted(out, key=lambda t: t[0])


def _latency_ms(doc: dict[str, Any], field: str, percentile: str
                ) -> tuple[float | None, str]:
    """A latency percentile in ms, and the field it was read from."""
    latency = ((doc.get("successes") or {}).get("latency") or {})
    candidates = TPOT_FIELDS if field == "time_per_output_token" else (field,)
    for cand in candidates:
        block = latency.get(cand)
        if isinstance(block, dict) and block.get(percentile) is not None:
            return float(block[percentile]) * 1000.0, cand
    return None, field


def summarize_report(doc: dict[str, Any], tenant: dict[str, Any],
                     error_rate_max: float) -> dict[str, Any]:
    """One tenant's numbers from one lifecycle report, judged against SLOs."""
    successes = doc.get("successes") or {}
    failures = doc.get("failures") or {}
    ok = int(successes.get("count") or 0)
    bad = int(failures.get("count") or 0)
    total = ok + bad
    throughput = successes.get("throughput") or {}
    goodput = doc.get("goodput_metrics") or {}

    slo: dict[str, Any] = {}
    attains = total > 0
    for target in workload.slo_targets(tenant):
        value, source = _latency_ms(doc, target["field"], target["percentile"])
        passed = value is not None and value <= target["bound_ms"]
        slo[target["key"]] = {
            "bound_ms": target["bound_ms"],
            "measured_ms": value,
            "source_field": source,
            "pass": passed,
        }
        attains = attains and passed
    error_rate = (bad / total) if total else 1.0
    attains = attains and error_rate <= error_rate_max

    rps = float(throughput.get("requests_per_sec") or 0.0)
    tps = float(throughput.get("total_tokens_per_sec") or 0.0)
    good_rps = None
    for key in GOODPUT_RPS_KEYS:
        if goodput.get(key) is not None:
            good_rps = float(goodput[key])
            break
    good_tps = None
    for key in GOODPUT_TPS_KEYS:
        if goodput.get(key) is not None:
            good_tps = float(goodput[key])
            break
    # Without inference-perf's per-request goodput, the tenant's whole
    # throughput counts when its percentiles meet the SLO and none of it
    # when they do not.
    if good_rps is None:
        good_rps = rps if attains else 0.0
    if good_tps is None:
        good_tps = tps if attains else 0.0

    return {
        "requests": {"ok": ok, "failed": bad},
        "error_rate": round(error_rate, 6),
        "requests_per_sec": rps,
        "total_tokens_per_sec": tps,
        "output_tokens_per_sec": float(
            throughput.get("output_tokens_per_sec") or 0.0),
        "goodput_requests_per_sec": good_rps,
        "goodput_tokens_per_sec": good_tps,
        "goodput_source": "inference-perf" if goodput else "percentile-gate",
        "slo": slo,
        "attains": attains,
        "failures_by_label": {
            k: int((v or {}).get("count") or 0)
            for k, v in (failures.get("by_label") or {}).items()
        },
        "load_summary": doc.get("load_summary"),
    }


# -- scoring ----------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.median(clean) if clean else None


def score(release: dict[str, Any], profile: dict[str, Any],
          per_replicate: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Primary-metric candidates and gates from per-replicate tenant summaries.

    `per_replicate[r][tenant]` is `summarize_report`'s output. Replicates
    are reduced by median: fleet benchmarks are noisy and a mean lets one
    autoscaler hiccup decide an arm.
    """
    chips = int(release["chips"]["total"]) or 1
    budget = profile.get("budget") or {}
    error_rate_max = float(
        (profile.get("availability") or {}).get("error_rate_max", 0.001))

    goodput_reps = []
    tps_reps = []
    for rep in per_replicate:
        goodput_reps.append(
            sum(t["goodput_requests_per_sec"] for t in rep.values()))
        tps_reps.append(sum(t["goodput_tokens_per_sec"] for t in rep.values()))
    goodput = _median(goodput_reps) or 0.0
    good_tps = _median(tps_reps) or 0.0

    metrics: dict[str, Any] = {
        "goodput_at_slo": {
            "value": goodput,
            "unit": "req/s",
            "replicates": goodput_reps
        },
        "goodput_at_slo_per_chip": {
            "value": goodput / chips,
            "unit": "req/s/chip",
            "replicates": [g / chips for g in goodput_reps],
        },
        "good_tokens_per_sec": {"value": good_tps, "unit": "tok/s"},
    }
    cost = budget.get("chip_hour_cost")
    if cost and good_tps > 0:
        usd_per_hour = float(cost) * chips
        tokens_per_hour = good_tps * 3600.0
        metrics["cost_per_1M_tokens_at_slo"] = {
            "value": usd_per_hour / (tokens_per_hour / 1e6),
            "unit": "usd/1M-tokens",
            "replicates": [
                (usd_per_hour / (t * 3600.0 / 1e6)) if t > 0 else None
                for t in tps_reps
            ],
        }
    else:
        metrics["cost_per_1M_tokens_at_slo"] = None

    tenants: dict[str, Any] = {}
    gates: list[dict[str, Any]] = []
    for tenant in profile["tenants"]:
        name = tenant["name"]
        reps = [rep[name] for rep in per_replicate if name in rep]
        slo_summary: dict[str, Any] = {}
        for target in workload.slo_targets(tenant):
            measured = _median([
                r["slo"][target["key"]]["measured_ms"] for r in reps
                if target["key"] in r["slo"]
            ])
            passed = measured is not None and measured <= target["bound_ms"]
            slo_summary[target["key"]] = {
                "bound_ms": target["bound_ms"],
                "measured_ms": measured,
                "pass": passed,
            }
            gates.append({
                "name": f"{name}.{target['key']}",
                "metric": target["key"],
                "comparator": "lte",
                "bound": target["bound_ms"],
                "unit": "ms",
                "value": measured,
                "passed": passed,
                "same_execution": True,
            })
        err = _median([r["error_rate"] for r in reps])
        err_ok = err is not None and err <= error_rate_max
        gates.append({
            "name": f"{name}.error_rate",
            "metric": "error_rate",
            "comparator": "lte",
            "bound": error_rate_max,
            "unit": "ratio",
            "value": err,
            "passed": err_ok,
            "same_execution": True,
        })
        tenants[name] = {
            "priority": tenant["priority"],
            "share": tenant["share"],
            "slo": slo_summary,
            "error_rate": err,
            "requests_per_sec": _median([r["requests_per_sec"] for r in reps]),
            "attains": all(r["attains"] for r in reps) and bool(reps),
            "replicates": reps,
        }
    max_chips = budget.get("max_chips")
    if max_chips:
        gates.append({
            "name": "budget.max_chips",
            "metric": "chips",
            "comparator": "lte",
            "bound": int(max_chips),
            "unit": "chips",
            "value": chips,
            "passed": chips <= int(max_chips),
            "same_execution": True,
        })
    return {
        "metrics": metrics,
        "tenants": tenants,
        "gates": gates,
        "admissible": all(g["passed"] for g in gates) and bool(per_replicate),
    }


def sweep_summary(per_tenant_stages: dict[str, list[dict[str, Any]]],
                  profile: dict[str, Any]) -> dict[str, Any]:
    """Per-stage attainment and the knee: the last stage every tenant met."""
    stages: list[dict[str, Any]] = []
    count = max((len(v) for v in per_tenant_stages.values()), default=0)
    knee = None
    for idx in range(count):
        entry: dict[str, Any] = {"stage": idx, "tenants": {}}
        all_ok = True
        total_rps = 0.0
        for tenant in profile["tenants"]:
            rows = per_tenant_stages.get(tenant["name"], [])
            if idx >= len(rows):
                all_ok = False
                continue
            row = rows[idx]
            entry["tenants"][tenant["name"]] = {
                "requests_per_sec": row["requests_per_sec"],
                "attains": row["attains"],
                "slo": {k: v["measured_ms"] for k, v in row["slo"].items()},
                "load_summary": row.get("load_summary"),
            }
            total_rps += row["requests_per_sec"]
            all_ok = all_ok and row["attains"]
        entry["requests_per_sec"] = total_rps
        entry["attains"] = all_ok
        stages.append(entry)
        if all_ok:
            knee = {"stage": idx, "requests_per_sec": total_rps}
    return {"stages": stages, "knee": knee}


# -- running ------------------------------------------------------------------------


def _job_doc_with_name(job: dict[str, Any], name: str) -> dict[str, Any]:
    doc = json.loads(json.dumps(job))
    doc["metadata"]["name"] = name
    return doc


def _wait_jobs(kube: Kube, names: list[str], *, timeout_s: int,
               sleep: Callable[[float], None],
               clock: Callable[[], float]) -> dict[str, str]:
    started = clock()
    outcome: dict[str, str] = {}
    while len(outcome) < len(names):
        for name in names:
            if name in outcome:
                continue
            doc = kube.kubectl_json(["get", "job", name], check=False)
            st = (doc.get("status") or {}) if doc else {}
            if int(st.get("succeeded") or 0) >= 1:
                outcome[name] = "succeeded"
            elif int(st.get("failed") or 0) >= 1:
                outcome[name] = "failed"
        if len(outcome) < len(names):
            if clock() - started > timeout_s:
                for name in names:
                    outcome.setdefault(name, "timeout")
                break
            sleep(POLL_S)
    return outcome


def run_bench(env: DeployEnv,
              release_dir: pathlib.Path,
              *,
              stage: str,
              replicates: int,
              out_dir: pathlib.Path,
              execution_id: str | None = None,
              timeout_s: int = DEFAULT_JOB_TIMEOUT_S,
              runner: Any = None,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run `replicates` rounds of the stage's tenant Jobs; write the report."""
    if stage not in render.BENCH_STAGES:
        raise DeployError("stage must be sweep or slo",
                          error_code="INVALID_FIELD")
    if replicates < 1:
        raise DeployError("replicates must be >= 1", error_code="INVALID_FIELD")
    release_dir = pathlib.Path(release_dir)
    release = render.load_release(release_dir)
    profile = json.loads(
        (release_dir / release["files"]["profile"]).read_text("utf-8"))
    kwargs: dict[str, Any] = {
        "namespace": release["namespace"],
        "kubeconfig": env.kubeconfig,
        "context": env.kube_context
    }
    if runner is not None:
        kwargs["runner"] = runner
    kube = Kube(**kwargs)
    execution_id = execution_id or new_execution_id()
    out_dir = pathlib.Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    error_rate_max = float(
        (profile.get("availability") or {}).get("error_rate_max", 0.001))

    per_replicate: list[dict[str, dict[str, Any]]] = []
    per_tenant_stages: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []
    for rep in range(replicates):
        names: dict[str, str] = {}
        for tenant in profile["tenants"]:
            key = f"{tenant['name']}.{stage}"
            files = release["files"]["bench"][key]
            kube.apply((release_dir / files["configmap"]).read_text("utf-8"))
            job = json.loads((release_dir / files["job"]).read_text("utf-8"))
            job_name = (f"{job['metadata']['name']}-{execution_id[-8:]}"
                        f"-r{rep}")
            kube.apply(json.dumps(_job_doc_with_name(job, job_name)))
            names[tenant["name"]] = job_name
        outcome = _wait_jobs(kube,
                             list(names.values()),
                             timeout_s=timeout_s,
                             sleep=sleep,
                             clock=clock)
        summaries: dict[str, dict[str, Any]] = {}
        for tenant in profile["tenants"]:
            job_name = names[tenant["name"]]
            proc = kube.kubectl(["logs", f"job/{job_name}", "--tail=-1"],
                                check=False)
            text = (proc.stdout or "") + (proc.stderr or "")
            (raw_dir / f"{job_name}.log").write_text(text, encoding="utf-8")
            reports = parse_report_logs(text)
            (raw_dir / f"{job_name}.reports.json").write_text(
                json.dumps(reports, indent=2), encoding="utf-8")
            summary_doc = _find(reports, "summary_lifecycle_metrics.json")
            if outcome.get(job_name) != "succeeded" or summary_doc is None:
                failures.append(
                    f"{job_name}: {outcome.get(job_name)}; reports:"
                    f" {sorted(reports)}")
                summaries[tenant["name"]] = summarize_report(
                    {}, tenant, error_rate_max)
            else:
                summaries[tenant["name"]] = summarize_report(
                    summary_doc, tenant, error_rate_max)
                if stage == "sweep":
                    per_tenant_stages[tenant["name"]] = [
                        summarize_report(doc, tenant, error_rate_max)
                        for _, doc in _stage_reports(reports)
                    ]
            kube.delete(["job", job_name])
        per_replicate.append(summaries)

    scored = score(release, profile, per_replicate)
    report: dict[str, Any] = {
        "schema": "llmd-bench-report",
        "execution_id": execution_id,
        "stage": stage,
        "replicates": replicates,
        "started_at": started_at,
        "finished_at": _now(),
        "release": {
            "name": release["name"],
            "namespace": release["namespace"],
            "config_digest": release["config_digest"],
            "replicas": release["replicas"],
            "chips_total": release["chips"]["total"],
            "router_profile": release["router"]["profile"],
        },
        "workload": release["workload"],
        "primary_metric": profile["objective"]["primary_metric"],
        "telemetry_valid": not failures,
        "job_failures": failures,
        "raw_dir": str(raw_dir),
        **scored,
    }
    if stage == "sweep":
        report["sweep"] = sweep_summary(per_tenant_stages, profile)
    (out_dir / "llmd-bench-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # The release remembers its executions so a sweep ledger can be rebuilt.
    release.setdefault("history", []).append({
        "at": _now(),
        "action": "bench",
        "execution_id": execution_id,
        "stage": stage,
        "config_digest": release["config_digest"],
        "primary": (scored["metrics"].get(profile["objective"]["primary_metric"])
                    or {}).get("value"),
        "admissible": scored["admissible"],
        "report": str(out_dir / "llmd-bench-report.json"),
    })
    render.save_release(release_dir, release)
    return report
