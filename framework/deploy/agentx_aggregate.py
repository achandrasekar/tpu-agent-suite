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
"""Reduce one aiperf AgentX run to the numbers the leaderboard ranks on.

Runs inside the bench pod (standard library only) after `aiperf profile
--scenario inferencex-agentx-mvp`, and writes one small JSON the bench
runner reads back out of the Job log. The formulas are InferenceX's
(SemiAnalysisAI/InferenceX,
utils/agentic/aggregation/request_metrics.py and
process_agentic_result.py), re-stated here so a point measured by this
suite is the same number the submission pipeline would compute:

* records: `profile_export.jsonl` rows whose `metadata.benchmark_phase`
  is `profiling` (warm-up rows are dropped) and that carry no `error`;
* latencies (ms in the records): `time_to_first_token`,
  `request_latency`, `inter_token_latency` (TPOT is the ITL series),
  reported as seconds; interactivity = 1 / ITL per percentile;
* throughput: duration = (max request_end_ns - min request_start_ns),
  `total = (sum input + sum output) / duration`, output and input alike;
* per chip: the caller divides by physical chips (the runner knows the
  fleet; InferenceX divides by `ceil(logical devices / 2)` on tpuv7).

Prefix-cache hit rates come from aiperf's `--server-metrics` export when
present (`vllm:prefix_cache_hits|queries` for the HBM tier,
`vllm:external_prefix_cache_hits|queries` for the CPU/external tier).
Nothing here needs the InferenceX checkout.

Usage: agentx_aggregate.py <artifact_dir> <out.json> --concurrency N
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
from typing import Any

PERCENTILES = (50, 75, 90, 95, 99)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank-free linear interpolation, as InferenceX's
    `aggregation_common.percentile` (numpy default)."""
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return data[int(k)]
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    out = {"mean": statistics.mean(values), "min": min(values),
           "max": max(values)}
    for pct in PERCENTILES:
        out[f"p{pct}"] = percentile(values, pct)
    out["std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return out


def _metric(record: dict[str, Any], key: str) -> Any:
    metric = (record.get("metrics") or {}).get(key)
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


def _floats(records: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in records:
        v = _metric(r, key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _ms_to_s(values: list[float]) -> list[float]:
    return [v / 1000.0 for v in values if v > 0]


def _find_artifact_dir(base: pathlib.Path) -> pathlib.Path:
    if (base / "profile_export.jsonl").is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "profile_export.jsonl").is_file():
                return child
    return base


def load_records(path: pathlib.Path) -> tuple[list[dict[str, Any]],
                                              dict[str, Any]]:
    """Profiling records without errors, plus the accounting of what was
    dropped (the ok/total column of the hill-climb tables)."""
    records: list[dict[str, Any]] = []
    acct: dict[str, Any] = {"records_total": 0, "records_profiled": 0,
                            "records_warmup_dropped": 0,
                            "records_error_dropped": 0,
                            "error_categories": {}}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            acct["records_total"] += 1
            phase = (obj.get("metadata") or {}).get("benchmark_phase")
            warm = phase is not None and phase != "profiling"
            error = obj.get("error")
            if warm:
                acct["records_warmup_dropped"] += 1
            if error:
                acct["records_error_dropped"] += 1
                cat = _error_category(error)
                acct["error_categories"][cat] = \
                    acct["error_categories"].get(cat, 0) + 1
            if warm or error:
                continue
            records.append(obj)
    acct["records_profiled"] = len(records)
    acct["records_dropped_total"] = acct["records_total"] - len(records)
    return records, acct


def _error_category(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("code") or error.get("status")
        if code is not None:
            return f"http_{code}"
        kind = error.get("type") or error.get("message")
        return str(kind)[:40] if kind else "unknown"
    return str(error)[:40]


def reduce_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = _ms_to_s(_floats(records, "time_to_first_token"))
    e2el = _ms_to_s(_floats(records, "request_latency"))
    itl = _ms_to_s(_floats(records, "inter_token_latency"))
    # The dashboard's x axis: when aiperf reports the full-response ITL
    # ((request_end - first_content) / (OSL - 1)) the ingest replaces ITL
    # with it before taking 1 / pXX (InferenceX-app benchmark-mapper.ts,
    # full-response-interactivity.ts); mirrored here as `dashboard_itl`.
    full_itl = _ms_to_s(_floats(records, "full_response_inter_token_latency"))
    dashboard_itl = full_itl if full_itl else itl
    isl = _floats(records, "input_sequence_length")
    osl = _floats(records, "output_sequence_length")
    starts = [int(r["metadata"]["request_start_ns"]) for r in records
              if (r.get("metadata") or {}).get("request_start_ns")]
    ends = [int(r["metadata"]["request_end_ns"]) for r in records
            if (r.get("metadata") or {}).get("request_end_ns")]
    duration = ((max(ends) - min(starts)) / 1e9) if starts and ends else 0.0
    itl_stats = stats(itl)
    intvty = {k: (1.0 / v if v else None) for k, v in itl_stats.items()
              if k.startswith("p") or k == "mean"}
    # The dashboard's E2E-normalised interactivity (MODELS.md): per request
    # E2EL / OSL, a percentile of that, inverted -- the whole response time
    # per generated token, not just the inter-token gaps.
    per_token = []
    for r in records:
        e = _metric(r, "request_latency")
        o = _metric(r, "output_sequence_length")
        try:
            e, o = float(e), float(o)
        except (TypeError, ValueError):
            continue
        if e > 0 and o > 0:
            per_token.append((e / 1000.0) / o)
    e2e_norm_stats = stats(per_token)
    e2e_norm_intvty = {k: (1.0 / v if v else None)
                       for k, v in e2e_norm_stats.items()
                       if k.startswith("p") or k == "mean"}
    throughput: dict[str, Any] = {"duration_s": duration}
    if duration > 0:
        throughput.update({
            "input_tokens_per_s": sum(isl) / duration,
            "output_tokens_per_s": sum(osl) / duration,
            "total_tokens_per_s": (sum(isl) + sum(osl)) / duration,
            "requests_per_s": len(records) / duration,
        })
    dash_stats = stats(dashboard_itl)
    dashboard_intvty = {k: (1.0 / v if v else None)
                        for k, v in dash_stats.items()
                        if k.startswith("p") or k == "mean"}
    return {
        "latency_s": {"ttft": stats(ttft), "e2el": stats(e2el),
                      "itl": itl_stats, "tpot": itl_stats,
                      "full_response_itl": stats(full_itl),
                      "dashboard_itl": dash_stats},
        "interactivity_tok_s_per_user": intvty,
        "dashboard_interactivity_tok_s_per_user": dashboard_intvty,
        "dashboard_interactivity_source": "full_response_itl" if full_itl
        else "itl",
        "e2e_norm_interactivity_tok_s_per_user": e2e_norm_intvty,
        "tokens": {"input": stats(isl), "output": stats(osl),
                   "input_total": sum(isl), "output_total": sum(osl)},
        "throughput": throughput,
    }


def _metric_rows(doc: Any) -> list[dict[str, Any]]:
    """Flatten aiperf's server_metrics_export.json, whose shape has varied
    (a list of samples, or {endpoint: {metric: value}})."""
    rows: list[dict[str, Any]] = []
    if isinstance(doc, list):
        for item in doc:
            if isinstance(item, dict):
                rows.append(item)
    elif isinstance(doc, dict):
        if "metrics" in doc and isinstance(doc["metrics"], (list, dict)):
            return _metric_rows(doc["metrics"])
        for key, value in doc.items():
            if isinstance(value, dict):
                rows.append({"endpoint": key, **value})
            elif isinstance(value, list):
                rows += _metric_rows(value)
    return rows


CACHE_COUNTERS = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
                  "vllm:external_prefix_cache_hits",
                  "vllm:external_prefix_cache_queries")


def _series_totals(doc: Any, endpoints: set[str] | None = None
                   ) -> tuple[dict[str, float], bool]:
    """aiperf >= 0.12 `server_metrics_export.json` (schema_version 1.1):
    `metrics.<name>.{type, series: [{endpoint_url, labels, stats: {total,
    rate, ...}}]}`, totals over the profiling phase (`metrics_phase`).
    Counter names carry no `_total` suffix. When `endpoints` is given only
    those series count: aiperf also scrapes the inference URL's own
    `/metrics`, which behind the router's proxy Service is one engine's
    page again and would be double counted. Returns (totals, seen)."""
    totals = {k: 0.0 for k in CACHE_COUNTERS}
    seen = False
    metrics = doc.get("metrics") if isinstance(doc, dict) else None
    if not isinstance(metrics, dict):
        return totals, False
    for name in CACHE_COUNTERS:
        entry = metrics.get(name) or metrics.get(name + "_total")
        if not isinstance(entry, dict):
            continue
        for series in entry.get("series") or []:
            if endpoints and (series or {}).get("endpoint_url") not in endpoints:
                continue
            stats = (series or {}).get("stats") or {}
            value = stats.get("total")
            if value is None:
                value = stats.get("delta", stats.get("last"))
            try:
                totals[name] += float(value)
                seen = True
            except (TypeError, ValueError):
                continue
    return totals, seen


def cache_rates(server_metrics: Any, endpoints: set[str] | None = None
                ) -> dict[str, Any]:
    """HBM and CPU/external prefix-cache hit rates from aiperf's server
    metrics export, by the same counters InferenceX's vLLM backend reads
    (`vllm:prefix_cache_{hits,queries}` for HBM,
    `vllm:external_prefix_cache_{hits,queries}` for the CPU tier);
    `present: false` when the export carries none of them. Reads the
    schema-1.1 series shape first, then the older flat shapes."""
    totals, seen = _series_totals(server_metrics, endpoints)
    if not seen:
        totals = {k: 0.0 for k in CACHE_COUNTERS}
        for row in _metric_rows(server_metrics):
            name = row.get("name") or row.get("metric")
            if name in totals:
                try:
                    totals[name] += float(row.get("value") or
                                          row.get("last") or 0.0)
                    seen = True
                except (TypeError, ValueError):
                    continue
            for key in list(totals):
                if key in row and name is None:
                    try:
                        totals[key] += float(row[key])
                        seen = True
                    except (TypeError, ValueError):
                        continue
    if not seen:
        return {"present": False}
    q = totals["vllm:prefix_cache_queries"]
    eq = totals["vllm:external_prefix_cache_queries"]
    return {
        "present": True,
        "gpu_hit_rate": (totals["vllm:prefix_cache_hits"] / q) if q else None,
        "cpu_hit_rate": (totals["vllm:external_prefix_cache_hits"] / eq)
        if eq else None,
        "counters": totals,
    }


SERVER_METRICS_SUMMARY_FILE = "server_metrics_summary.json"


def server_metrics_summary(doc: Any) -> Any:
    """aiperf's server_metrics_export.json without the per-second
    `timeslices`: the endpoint summary and each series' `stats`, which is
    all the hit-rate and queue readings need. The full export stays in
    the Job's artifact dir; the container log cannot hold it."""
    if not isinstance(doc, dict):
        return doc

    def trim(metrics: Any) -> Any:
        if not isinstance(metrics, dict):
            return metrics
        return {
            name: {**{k: v for k, v in entry.items() if k != "series"},
                   "series": [{k: v for k, v in (s or {}).items()
                               if k != "timeslices"}
                              for s in entry.get("series") or []]}
            if isinstance(entry, dict) else entry
            for name, entry in metrics.items()}
    # `metrics` is the profiling phase; aiperf also keeps a full copy per
    # other phase (`warmup_metrics`), each with its own timeslices.
    out = {k: (trim(v) if k == "metrics" or k.endswith("_metrics") else v)
           for k, v in doc.items()}
    out["trimmed"] = "timeslices dropped"
    return out


def _write_server_metrics_summary(src: pathlib.Path, dst: pathlib.Path
                                  ) -> None:
    doc = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(json.dumps(server_metrics_summary(doc)) + "\n",
                   encoding="utf-8")


def aggregate(artifact_dir: pathlib.Path, concurrency: int | None,
              out_dir: pathlib.Path | None = None) -> dict[str, Any]:
    art = _find_artifact_dir(artifact_dir)
    out_dir = out_dir or art.parent
    export = art / "profile_export.jsonl"
    if not export.is_file():
        return {"schema": "agentx-point", "status": "no_export",
                "artifact_dir": str(art), "concurrency": concurrency}
    records, acct = load_records(export)
    doc: dict[str, Any] = {
        "schema": "agentx-point",
        "status": "ok" if records else "no_profiled_records",
        "concurrency": concurrency,
        "artifact_dir": str(art),
        "requests": acct,
        **(reduce_records(records) if records else {}),
    }
    agg_path = art / "profile_export_aiperf.json"
    if agg_path.is_file():
        try:
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            agg = {}
        meta = agg.get("metadata") if isinstance(agg, dict) else None
        if isinstance(meta, dict):
            doc["aiperf"] = {k: meta.get(k) for k in (
                "dataset", "benchmark_duration", "concurrency",
                "aiperf_version", "version") if meta.get(k) is not None}
        for key in ("request_count", "error_request_count",
                    "completed_request_count"):
            block = agg.get(key) if isinstance(agg, dict) else None
            if isinstance(block, dict) and block.get("avg") is not None:
                doc.setdefault("aiperf_counts", {})[key] = block["avg"]
    sm_path = art / "server_metrics_export.json"
    if sm_path.is_file():
        try:
            _write_server_metrics_summary(
                sm_path, out_dir / SERVER_METRICS_SUMMARY_FILE)
        except (OSError, ValueError):
            pass
        # The engine endpoints the Job was told to scrape; the runner sets
        # the env from the live pods of the metrics role.
        endpoints = {u.strip() for u in os.environ.get(
            "AGENTX_SERVER_METRICS_URLS", "").split(",") if u.strip()}
        try:
            doc["cache"] = cache_rates(json.loads(
                sm_path.read_text(encoding="utf-8")), endpoints or None)
        except json.JSONDecodeError:
            doc["cache"] = {"present": False}
    else:
        doc["cache"] = {"present": False}
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact_dir", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = aggregate(args.artifact_dir, args.concurrency, args.out.parent)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": doc["status"],
                      "records": doc.get("requests", {}).get(
                          "records_profiled")}))
    return 0 if doc["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
