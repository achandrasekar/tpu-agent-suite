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
"""Fleet metrics for a release over a window, from Managed Prometheus.

The benchmark says what clients saw; this says what the fleet was doing
meanwhile: KV occupancy and its spread across pods, queue depth, prefix hit
rate, server-side TTFT. The planner's ceiling formulas read these, and the
study baseline records this file as its `profile` in place of an XProf
trace. Each query is recorded with its PromQL and either a value or the
error it hit, so a missing metric is a visible gap and not a zero.

Metric names: vLLM's Prometheus metrics
  https://docs.vllm.ai/en/latest/design/metrics.html
Managed Prometheus HTTP API:
  https://cloud.google.com/stackdriver/docs/managed-prometheus/query-api-ui
"""

from __future__ import annotations

import datetime
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import render
from .config import DeployEnv
from .kube import DeployError, default_runner, gcloud_access_token

GMP_URL = ("https://monitoring.googleapis.com/v1/projects/{project}"
           "/location/global/prometheus/api/v1/query")

# Window-parameterised PromQL, keyed by the name the planner reads. `{w}` is
# the window, `{ns}` the namespace, `{sel}` the release's pod selector.
QUERIES: dict[str, str] = {
    "kv_util_mean":
        'avg(avg_over_time(vllm:kv_cache_usage_perc{{{sel}}}[{w}]))',
    "kv_util_max":
        'max(max_over_time(vllm:kv_cache_usage_perc{{{sel}}}[{w}]))',
    "kv_util_pod_stddev":
        'stddev(avg_over_time(vllm:kv_cache_usage_perc{{{sel}}}[{w}]))',
    "requests_running_mean":
        'avg(avg_over_time(vllm:num_requests_running{{{sel}}}[{w}]))',
    "requests_waiting_mean":
        'avg(avg_over_time(vllm:num_requests_waiting{{{sel}}}[{w}]))',
    "requests_waiting_max":
        'max(max_over_time(vllm:num_requests_waiting{{{sel}}}[{w}]))',
    "prefix_hit_rate":
        ('sum(increase(vllm:prefix_cache_hits_total{{{sel}}}[{w}])) /'
         ' sum(increase(vllm:prefix_cache_queries_total{{{sel}}}[{w}]))'),
    "ttft_p90_server_s":
        ('histogram_quantile(0.9, sum by (le)'
         ' (rate(vllm:time_to_first_token_seconds_bucket{{{sel}}}[{w}])))'),
    # vLLM renamed this histogram to `inter_token_latency` (the fleet found
    # live exports only the new name); `or` takes whichever the server has.
    "tpot_p90_server_s":
        ('histogram_quantile(0.9, sum by (le)'
         ' (rate(vllm:inter_token_latency_seconds_bucket{{{sel}}}[{w}])'
         ' or rate(vllm:time_per_output_token_seconds_bucket{{{sel}}}[{w}])))'),
    "generation_tokens_per_s":
        'sum(rate(vllm:generation_tokens_total{{{sel}}}[{w}]))',
    "prompt_tokens_per_s":
        'sum(rate(vllm:prompt_tokens_total{{{sel}}}[{w}]))',
    "preempted_per_s":
        'sum(rate(vllm:num_preemptions_total{{{sel}}}[{w}]))',
}


def _query(url: str, token: str, promql: str, timeout: int = 30) -> Any:
    data = urllib.parse.urlencode({"query": promql}).encode("utf-8")
    req = urllib.request.Request(url,
                                 data=data,
                                 headers={
                                     "Authorization": f"Bearer {token}",
                                     "Content-Type":
                                     "application/x-www-form-urlencoded",
                                 })
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _scalar(answer: Any) -> float | None:
    result = ((answer or {}).get("data") or {}).get("result") or []
    if not result:
        return None
    value = result[0].get("value") or [None, None]
    try:
        number = float(value[1])
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN from an empty histogram
        return None
    return number


def _read_release_file(release_dir: pathlib.Path, release: dict[str, Any],
                       key: str) -> str:
    return (pathlib.Path(release_dir) / release["files"][key]).read_text(
        encoding="utf-8")


def collect(env: DeployEnv,
            release_dir: pathlib.Path,
            *,
            window: str,
            out_dir: pathlib.Path,
            runner: Any = None,
            query_fn: Any = None) -> dict[str, Any]:
    """Query every metric in `QUERIES` over `window`; write metrics.json."""
    release = render.load_release(release_dir)
    url = env.prometheus_url
    token = ""
    if not url:
        project = env.prometheus_project or env.project
        if not project:
            raise DeployError(
                "no Prometheus: set llmd.prometheus_url or a GCP project in"
                " the environment", error_code="NO_PROMETHEUS")
        url = GMP_URL.format(project=project)
        token = gcloud_access_token(runner or default_runner)
    query = query_fn or (lambda promql: _query(url, token, promql))
    namespace = release["namespace"]
    # Managed Prometheus labels every sample a PodMonitoring scrapes with
    # `job=<PodMonitoring name>`; the release's model-server PodMonitoring
    # is `<name>-modelserver` for a managed fleet (every pool, prefill
    # included) and for an adopted one (whose pods carry the customer's
    # names, which a `<name>-decode-.*` regex would never match).
    sel = (f'namespace="{namespace}",'
           f'job="{render.modelserver_monitoring_name(release["name"])}"')
    def run_queries(selector: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, template in QUERIES.items():
            promql = template.format(w=window, ns=namespace, sel=selector)
            entry: dict[str, Any] = {"promql": promql, "value": None}
            try:
                entry["value"] = _scalar(query(promql))
            except (urllib.error.URLError, ValueError, OSError) as exc:
                entry["error"] = str(exc)[:300]
            out[name] = entry
        return out

    metrics = run_queries(sel)
    # Per-role breakdown under P/D: the PodMonitoring copies the pod's role
    # label into `llmd_role`, so each pool is one more selector. The xPyD
    # planner reads these to decide which axis of the shape to move; an
    # aggregated fleet has one pool and the breakdown is skipped.
    roles: dict[str, Any] = {}
    config = json.loads(_read_release_file(release_dir, release, "config"))
    if (config.get("topology") or {}).get("mode") == "disaggregated":
        for role in (render.ROLE_PREFILL, render.ROLE_DECODE):
            roles[role] = run_queries(
                f'{sel},{render.ROLE_TARGET_LABEL}="{role}"')
    doc = {
        "schema": "llmd-fleet-metrics",
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "window": window,
        "source": url,
        "release": {
            "name": release["name"],
            "namespace": namespace,
            "config_digest": release["config_digest"],
            "replicas": release["replicas"],
            "chips_total": release["chips"]["total"],
        },
        "metrics": metrics,
        "roles": roles,
        "missing": sorted(k for k, v in metrics.items() if v["value"] is None),
    }
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(doc, indent=2) + "\n",
                                          encoding="utf-8")
    return doc
