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
    throughput: dict[str, Any] = {"duration_s": duration}
    if duration > 0:
        throughput.update({
            "input_tokens_per_s": sum(isl) / duration,
            "output_tokens_per_s": sum(osl) / duration,
            "total_tokens_per_s": (sum(isl) + sum(osl)) / duration,
            "requests_per_s": len(records) / duration,
        })
    return {
        "latency_s": {"ttft": stats(ttft), "e2el": stats(e2el),
                      "itl": itl_stats, "tpot": itl_stats},
        "interactivity_tok_s_per_user": intvty,
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


def cache_rates(server_metrics: Any) -> dict[str, Any]:
    """HBM and CPU/external prefix-cache hit rates from the scrape, by the
    same counters InferenceX's vLLM backend reads; absent when the export
    has none of them."""
    totals = {"vllm:prefix_cache_hits": 0.0, "vllm:prefix_cache_queries": 0.0,
              "vllm:external_prefix_cache_hits": 0.0,
              "vllm:external_prefix_cache_queries": 0.0}
    seen = False
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


def aggregate(artifact_dir: pathlib.Path, concurrency: int | None
              ) -> dict[str, Any]:
    art = _find_artifact_dir(artifact_dir)
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
            doc["cache"] = cache_rates(json.loads(
                sm_path.read_text(encoding="utf-8")))
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
    doc = aggregate(args.artifact_dir, args.concurrency)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": doc["status"],
                      "records": doc.get("requests", {}).get(
                          "records_profiled")}))
    return 0 if doc["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
