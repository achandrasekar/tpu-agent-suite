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
"""The AgentX leaderboard view of a study: tables and charts.

Reads every AgentX bench report the study produced (one point per
concurrency per configuration), the leaderboard's own entries for the
same model and scenario, and the price the leaderboard applies per
accelerator-hour, and writes:

* `leaderboard.md` -- the hill-climb table (config, concurrency, chips,
  total and output tok/s/chip, TTFT p50/p90, TPOT, interactivity, E2E,
  prefix-hit rates, ok/total, qualifying) and the best qualifying point
  per configuration, next to the leaderboard entries;
* `throughput_vs_interactivity.svg` -- total tok/s per chip against
  interactivity (tok/s/user), one series per configuration and per
  leaderboard entry, points labelled by concurrency, hollow when the
  point failed the qualifying gate;
* `tokens_per_dollar_vs_interactivity.svg` -- total tokens per dollar
  against interactivity at the leaderboard's price per chip-hour
  (`tokens/$ = tok/s/chip x 3600 / $/chip-hour`);
* `points.csv` and `points.json` -- every plotted point with its source.

Axes are the dashboard's (InferenceX-app, read 2026-10-16): x is
**P90 interactivity** = 1 / p90 of the (full-response) inter-token
latency, `--interactivity intvty_p90` by default; `intvty_p50` (median
interactivity) and `e2e_norm_intvty_p90` (the "North Star"
E2E-normalised form) are the alternatives. Each series is drawn as its
Pareto frontier (maximise x and y), the dashboard's "optimal" view, with
the dominated points as faded markers. Prices are the dashboard's
(`pins.AGENTX_PRICES`) under `--tco-basis internal|external|rental`
(default `internal`, the dashboard default: tpu7x $1.03 per chip-hour).

Leaderboard entries come from `--leaderboard <json>` -- the file this
package ships (`tpu-llmd/references/agentx/leaderboard.json`) -- refreshed
with `fetch-live` from the dashboard's public views API, or extended
with `import-inferencex` from InferenceX agg JSONs.

Usage:
    agentx_leaderboard.py build --study-dir <llmd dir> [--reports <glob>...]
        --leaderboard <json> [--tco-basis internal] [--price hw=<usd>]
        --out <dir> [--title ...] [--interactivity intvty_p90]
    agentx_leaderboard.py fetch-live --out <entries.json>
        [--model Qwen-3.5-397B-A17B] [--merge <existing entries.json>]
    agentx_leaderboard.py import-inferencex --name <series> --hardware <hw>
        --framework <fw> --precision <p> <agg_conc*.json>... --out <entries.json>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE.parents[1] / "runtime", _HERE.parents[2]):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
try:
    from framework.deploy import pins, svgchart  # noqa: E402
except ImportError:  # installed layout
    from deploy import pins, svgchart  # type: ignore  # noqa: E402

INTERACTIVITY_KEYS = ("intvty_p90", "intvty_p75", "intvty_p50",
                      "e2e_norm_intvty_p90")
X_LABELS = {
    "intvty_p90": "P90 interactivity (tok/s/user = 1 / p90 inter-token latency)",
    "intvty_p75": "P75 interactivity (tok/s/user = 1 / p75 inter-token latency)",
    "intvty_p50": "P50 interactivity (tok/s/user = 1 / p50 inter-token latency)",
    "e2e_norm_intvty_p90": "P90 E2E-normalised interactivity (1 / p90 of E2E latency per output token)",
}


def price_for(hardware: str, basis: str, overrides: dict[str, float]
              ) -> float | None:
    key = str(hardware or "").lower().split("_")[0].split("-")[0]
    if key in overrides:
        return overrides[key]
    table = pins.AGENTX_PRICES.get(key)
    return table.get(basis) if table else None


def pareto(points: list[dict[str, Any]], x_key: str, y_key: str
           ) -> list[dict[str, Any]]:
    """The dashboard's `upper_left` frontier: sort by x descending, keep
    the points whose y beats every point with higher x (maximise both)."""
    cands = [p for p in points
             if p.get(x_key) is not None and p.get(y_key) is not None
             and float(p[x_key]) > 0]
    best_y = float("-inf")
    frontier = []
    for p in sorted(cands, key=lambda p: -float(p[x_key])):
        if float(p[y_key]) > best_y:
            frontier.append(p)
            best_y = float(p[y_key])
    return frontier


def _read(path: str | pathlib.Path) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


# -- our reports -> points ------------------------------------------------------


def _config_tag(report: dict[str, Any], config: dict[str, Any] | None) -> str:
    """A short name for the configuration a report measured."""
    if config:
        topo = config.get("topology") or {}
        if topo.get("mode") == "disaggregated":
            p = (topo.get("prefill") or {}).get("replicas", "?")
            d = (topo.get("decode") or {}).get("replicas", "?")
            shape = f"{p}P{d}D"
        else:
            shape = f"{(topo.get('decode') or {}).get('replicas', '?')}x agg"
        router = (config.get("router") or {}).get("profile", "")
        router = {"agentx-turn-priority": "turn-priority",
                  "agentx-session-affinity": "nofc"}.get(router, router)
        knobs = []
        for role in ("prefill", "decode"):
            for k, v in ((topo.get(role) or {}).get("engine") or {}).items():
                knobs.append(f"{role[0]}.{k}={v}")
        for k, v in (config.get("engine") or {}).items():
            knobs.append(f"{k}={v}")
        return " ".join([shape, router] + knobs).strip()
    rel = report.get("release") or {}
    return f"{rel.get('name', '?')} {rel.get('router_profile', '')}".strip()


def points_from_report(report: dict[str, Any], *, config: dict[str, Any] | None,
                       tag: str | None = None) -> list[dict[str, Any]]:
    """One point per concurrency in an AgentX bench report."""
    ax = report.get("agentx") or {}
    chips = int((report.get("release") or {}).get("chips_total") or 0)
    tag = tag or _config_tag(report, config)
    out = []
    for p in ax.get("points") or []:
        tenant = p.get("tenant") or {}
        reps = tenant.get("replicates") or []
        rep = reps[-1] if reps else {}
        a = rep.get("agentx") or {}
        metrics = p.get("metrics") or {}
        ttft = a.get("ttft_s") or {}
        tpot = a.get("tpot_ms") or {}
        e2e = a.get("e2e_s") or {}
        gates = {g["name"]: g for g in p.get("gates") or []}
        gate = next((g for n, g in gates.items() if n.endswith("ttft_p50_ms")),
                    None)
        total = (metrics.get("total_tokens_per_s_per_chip") or {}).get("value")
        out.append({
            "series": tag,
            "source": "study",
            "execution_id": report.get("execution_id"),
            "conc": p.get("concurrency"),
            "chips": chips,
            "total_tokens_per_s_per_chip": total,
            "output_tokens_per_s_per_chip": (
                metrics.get("output_tokens_per_s_per_chip") or {}).get("value"),
            "total_tokens_per_s": (metrics.get("total_tokens_per_s") or {}
                                   ).get("value"),
            "intvty_p90": a.get("interactivity_p90"),
            "intvty_p75": a.get("interactivity_p75"),
            "intvty_p50": a.get("interactivity_p50"),
            "intvty_source": a.get("interactivity_source"),
            "e2e_norm_intvty_p90": a.get("e2e_norm_interactivity_p90"),
            "ttft_p50_s": ttft.get("p50"),
            "ttft_p90_s": ttft.get("p90"),
            "tpot_p50_ms": tpot.get("p50"),
            "e2e_p50_s": e2e.get("p50"),
            "gpu_prefix_hit_rate": a.get("gpu_prefix_hit_rate"),
            "cpu_prefix_hit_rate": a.get("cpu_prefix_hit_rate"),
            "ok": (rep.get("requests") or {}).get("ok"),
            # profiled requests: ok + failed (warm-ups are not measured)
            "total": ((rep.get("requests") or {}).get("ok") or 0)
            + ((rep.get("requests") or {}).get("failed") or 0)
            if rep.get("requests") else None,
            "admissible": bool(p.get("admissible")),
            "gate_ttft_p50_ms": (gate or {}).get("bound"),
            "submission_valid": bool(ax.get("submission_valid")),
            "duration_s": ax.get("duration_s"),
            "harness": "aiperf",
        })
    return out


def _config_for(path: pathlib.Path, report: dict[str, Any],
                study_dir: pathlib.Path | None) -> dict[str, Any] | None:
    """The llmd-config.json that produced a report: the arm ledger's
    snapshot for its execution id first (exact, survives a later
    reconfigure), then the release directory it ran against, then a
    config beside the report."""
    exec_id = str(report.get("execution_id") or "")
    name = str((report.get("release") or {}).get("name") or "")
    cands: list[pathlib.Path] = []
    if study_dir and exec_id:
        cands += sorted(pathlib.Path(study_dir).rglob(
            f"configs/{exec_id}.json"))
    if study_dir and name:
        cands.append(pathlib.Path(study_dir) / "releases" / name /
                     "llmd-config.json")
    cands += [path.parent / "llmd-config.json",
              path.parent.parent / "llmd-config.json"]
    for rel in path.parents:
        if (rel / "release.json").is_file():
            cands.append(rel / "llmd-config.json")
            break
    for cand in cands:
        if cand.is_file():
            try:
                return _read(cand)
            except (OSError, ValueError):
                continue
    return None


def collect_study_points(study_dir: pathlib.Path | None,
                         report_globs: list[str]) -> list[dict[str, Any]]:
    """Points from every AgentX report under the study dir (and globs),
    each tagged with the config its release carried."""
    paths: list[pathlib.Path] = []
    if study_dir:
        paths += sorted(pathlib.Path(study_dir).rglob("llmd-bench-report.json"))
    for pattern in report_globs:
        paths += [pathlib.Path(p) for p in sorted(glob.glob(pattern,
                                                            recursive=True))]
    seen: set[str] = set()
    points: list[dict[str, Any]] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            report = _read(path)
        except (OSError, ValueError):
            continue
        if not report.get("agentx"):
            continue
        config = _config_for(path, report, study_dir)
        for pt in points_from_report(report, config=config):
            pt["report"] = str(path)
            points.append(pt)
    return points


# -- leaderboard entries ---------------------------------------------------------


def leaderboard_points(entries: list[dict[str, Any]], basis: str,
                       overrides: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for entry in entries:
        hw = str(entry.get("hardware") or "")
        price = price_for(hw, basis, overrides)
        if price is None:
            price = entry.get("price_per_chip_hour")
        for p in entry.get("points") or []:
            pt = {"series": entry["name"], "source": "leaderboard",
                  "hardware": hw, "framework": entry.get("framework"),
                  "precision": entry.get("precision"),
                  "price_per_chip_hour": price,
                  "citation": entry.get("source"), "admissible": True}
            pt.update(p)
            if "intvty_p90" not in pt and "interactivity" in pt:
                pt["intvty_p90"] = pt["interactivity"]
            out.append(pt)
    return out


# -- the dashboard's views API ----------------------------------------------------


def fetch_live(model: str, *, timeout: int = 60) -> dict[str, Any]:
    """Every point of every series the dashboard has for `model`'s agentic
    sequence, on the P90-interactivity x axis, as entries."""
    query = urllib.parse.urlencode({
        "model": model, "sequence": pins.AGENTX_LEADERBOARD_SEQUENCE,
        "metric": "tpPerGpu", "percentile": "p90", "xmode": "interactivity",
        "best": "false", "optimal": "false", "allPoints": "true",
        "format": "json"})
    url = f"{pins.AGENTX_LEADERBOARD_API}?{query}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "Accept-Encoding": "identity",
        "User-Agent": "tpu-llmd/agentx_leaderboard"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        doc = json.loads(resp.read().decode("utf-8"))
    return views_to_entries(doc, url)


def views_to_entries(doc: dict[str, Any], url: str = "") -> dict[str, Any]:
    """`/api/v1/views/inference` -> {snapshot, entries}; x is the response's
    x (P90 interactivity), y its tpPerGpu, chips its `tp` (physical for
    TPU), one entry per hardware+framework series."""
    entries = []
    for s in doc.get("series") or []:
        pts = []
        for p in s.get("points") or []:
            m = p.get("metrics") or {}
            pts.append({
                "conc": p.get("concurrency"), "chips": p.get("tp"),
                "total_tokens_per_s_per_chip": m.get("tpPerGpu", p.get("y")),
                "output_tokens_per_s_per_chip": m.get("outputTputPerGpu"),
                "input_tokens_per_s_per_chip": m.get("inputTputPerGpu"),
                "intvty_p90": p.get("x"),
                "precision": p.get("precision"),
                "topology": p.get("topologyKey"),
                "date": p.get("date"), "run_id": p.get("runId"),
                "dashboard_frontier": bool(p.get("frontier")),
                "admissible": True,
            })
        entries.append({
            "name": f"{s.get('label') or s.get('hwKey')}",
            "hw_key": s.get("hwKey"), "hardware": s.get("gpu"),
            "framework": s.get("framework"),
            "precision": (pts[0].get("precision") if pts else None),
            "spec_method": s.get("specMethod"),
            "deployment": s.get("deployment"),
            "best_per_sku": bool(s.get("bestPerSku")),
            "source": f"InferenceX dashboard views API ({url or 'live'})",
            "points": pts,
        })
    return {"snapshot": {"fetched_at": datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds"),
        "x_axis": doc.get("xAxis"), "metric": doc.get("metric"),
        "count": doc.get("count"), "url": url},
        "entries": entries}


def tokens_per_dollar(point: dict[str, Any], price: float | None
                      ) -> float | None:
    tps = point.get("total_tokens_per_s_per_chip")
    if tps is None or not price or price <= 0:
        return None
    return float(tps) * 3600.0 / float(price)


# -- import InferenceX agg JSONs ------------------------------------------------


def import_inferencex(paths: list[str]) -> list[dict[str, Any]]:
    """Points from InferenceX `${RESULT_FILENAME}_conc<C>.json` aggregates
    (utils/agentic/aggregation/process_agentic_result.py)."""
    pts = []
    for path in paths:
        agg = _read(path)
        rm = agg.get("request_metrics") or {}
        lat = rm.get("latency") or {}
        tp = rm.get("throughput") or {}
        per = tp.get("per_gpu") or {}
        cache = (agg.get("server_metrics") or {}).get("cache") or {}
        pts.append({
            "conc": agg.get("conc"),
            "chips": agg.get("num_gpus"),
            "total_tokens_per_s_per_chip": per.get("total_tput_tps"),
            "output_tokens_per_s_per_chip": per.get("output_tput_tps"),
            "total_tokens_per_s": (tp.get("total") or {}).get(
                "tokens_per_second"),
            # The dashboard's x: 1 / p90 of full-response ITL, else ITL.
            "intvty_p90": _recip(((lat.get("full_response_itl") or {}).get("p90"))
                                 or (lat.get("itl") or {}).get("p90")),
            "intvty_p75": _recip(((lat.get("full_response_itl") or {}).get("p75"))
                                 or (lat.get("itl") or {}).get("p75")),
            "intvty_p50": (lat.get("intvty") or {}).get("p50"),
            "e2e_norm_intvty_p90": (lat.get("e2e_norm_intvty") or {}).get("p90"),
            "ttft_p50_s": (lat.get("ttft") or {}).get("p50"),
            "ttft_p90_s": (lat.get("ttft") or {}).get("p90"),
            "tpot_p50_ms": ((lat.get("tpot") or {}).get("p50") or 0) * 1000.0
            if (lat.get("tpot") or {}).get("p50") is not None else None,
            "e2e_p50_s": (lat.get("e2el") or {}).get("p50"),
            "gpu_prefix_hit_rate": cache.get("gpu_cache_hit_rate"),
            "cpu_prefix_hit_rate": cache.get("cpu_cache_hit_rate"),
            "ok": agg.get("num_requests_successful"),
            "total": agg.get("num_requests_total"),
            "file": str(path),
        })
    return pts


def _recip(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return (1.0 / f) if f > 0 else None


# -- outputs ---------------------------------------------------------------------


def _f(v: Any, nd: int = 1, scale: float = 1.0, pct: bool = False) -> str:
    if v is None:
        return "—"
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return str(v)
    if pct:
        return f"{x * 100:.1f}%"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def _frontier_flags(points: list[dict[str, Any]], x_key: str,
                    y_key: str = "total_tokens_per_s_per_chip") -> None:
    """Mark every point `frontier` = on its series' Pareto frontier."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault(p["series"], []).append(p)
    for group in groups.values():
        front = {id(p) for p in pareto(group, x_key, y_key)}
        for p in group:
            p["frontier"] = id(p) in front


def markdown(points: list[dict[str, Any]], lb_points: list[dict[str, Any]],
             *, title: str, interactivity: str, our_price: float | None,
             price_note: str, basis: str) -> str:
    xk = interactivity
    xl = X_LABELS.get(xk, xk)
    lines = [f"# {title}", "",
             f"Interactivity axis: **{xl}** (`{xk}`). Tokens per dollar ="
             f" total tok/s/chip x 3600 / price per chip-hour, prices on the"
             f" dashboard's `{basis}` TCO basis ({price_note}). Each series'"
             f" frontier is the dashboard's `upper_left` Pareto set"
             f" (maximise interactivity and throughput); points off it are"
             f" marked dominated.", ""]
    lines += ["## This study's points", "",
              "| configuration | conc | chips | total tok/s/chip | out tok/s/chip"
              " | tok/$ | TTFT p50 (s) | TTFT p90 (s) | TPOT p50 (ms) |"
              f" {xk} | intvty p50 | E2E p50 (s) | GPU hit | CPU hit |"
              " ok/total | qualifies | frontier |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for p in sorted(points, key=lambda p: (p["series"], p.get("conc") or 0)):
        lines.append(
            f"| {p['series']} | {p.get('conc')} | {p.get('chips')} |"
            f" {_f(p.get('total_tokens_per_s_per_chip'), 0)} |"
            f" {_f(p.get('output_tokens_per_s_per_chip'))} |"
            f" {_f(tokens_per_dollar(p, our_price), 0)} |"
            f" {_f(p.get('ttft_p50_s'), 2)} | {_f(p.get('ttft_p90_s'), 2)} |"
            f" {_f(p.get('tpot_p50_ms'), 2)} | {_f(p.get(xk), 1)} |"
            f" {_f(p.get('intvty_p50'), 0)} |"
            f" {_f(p.get('e2e_p50_s'), 2)} |"
            f" {_f(p.get('gpu_prefix_hit_rate'), pct=True)} |"
            f" {_f(p.get('cpu_prefix_hit_rate'), pct=True)} |"
            f" {p.get('ok')}/{p.get('total')} |"
            f" {'✅' if p.get('admissible') else '❌'}"
            f"{'' if p.get('submission_valid', True) else ' (short run)'} |"
            f" {'●' if p.get('frontier') else '○'} |")
    if not points:
        lines.append("| (no AgentX report found) | | | | | | | | | | | | | | | | |")
    best: dict[str, dict[str, Any]] = {}
    for p in points:
        if not p.get("admissible") or p.get(
                "total_tokens_per_s_per_chip") is None:
            continue
        cur = best.get(p["series"])
        if cur is None or p["total_tokens_per_s_per_chip"] > cur[
                "total_tokens_per_s_per_chip"]:
            best[p["series"]] = p
    lines += ["", "## Best qualifying point per configuration", "",
              "| configuration | conc | total tok/s/chip | out tok/s/chip |"
              f" tok/$ | {xk} | TTFT p50 (s) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, p in sorted(best.items(), key=lambda kv: -kv[1][
            "total_tokens_per_s_per_chip"]):
        lines.append(f"| {name} | {p.get('conc')} |"
                     f" {_f(p['total_tokens_per_s_per_chip'], 0)} |"
                     f" {_f(p.get('output_tokens_per_s_per_chip'))} |"
                     f" {_f(tokens_per_dollar(p, our_price), 0)} |"
                     f" {_f(p.get(xk), 1)} |"
                     f" {_f(p.get('ttft_p50_s'), 2)} |")
    if not best:
        lines.append("| (no qualifying point yet) | | | | | | |")
    # Leaderboard: the frontier of each entry, then its best point.
    lines += ["", "## Leaderboard entries drawn for comparison", "",
              "Frontier points only (the dashboard's optimal view); every"
              " point is in `points.csv`.", "",
              "| entry | hardware | stack | conc | chips | topology |"
              " total tok/s/chip | out tok/s/chip | $/chip-h | tok/$ |"
              f" {xk} | source |",
              "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for p in sorted(lb_points, key=lambda p: (p["series"], p.get("conc") or 0)):
        if not p.get("frontier"):
            continue
        price = p.get("price_per_chip_hour")
        lines.append(
            f"| {p['series']} | {p.get('hardware', '')} |"
            f" {p.get('framework', '')} {p.get('precision', '')} |"
            f" {p.get('conc')} | {p.get('chips', '')} |"
            f" {_topology_short(p.get('topology'))} |"
            f" {_f(p.get('total_tokens_per_s_per_chip'), 0)} |"
            f" {_f(p.get('output_tokens_per_s_per_chip'))} |"
            f" {_f(price, 2)} | {_f(tokens_per_dollar(p, price), 0)} |"
            f" {_f(p.get(xk), 1)} |"
            f" {p.get('citation', '')} |")
    if not lb_points:
        lines.append("| (none given) | | | | | | | | | | | |")
    omitted = sorted({p["series"] for p in lb_points if p.get(xk) is None})
    if omitted:
        lines += ["", f"Entries without `{xk}` (not drawn on the charts; their"
                  f" tables record what they do have): "
                  + ", ".join(f"`{s}`" for s in omitted) + "."]
    lines += ["", "## Peak per series (any interactivity)", "",
              f"| series | source | best total tok/s/chip | at conc | {xk} |"
              " tok/$ |", "|---|---|---:|---:|---:|---:|"]
    peaks: dict[str, dict[str, Any]] = {}
    for p in points + lb_points:
        if p.get("total_tokens_per_s_per_chip") is None or (
                p.get("source") == "study" and not p.get("admissible")):
            continue
        cur = peaks.get(p["series"])
        if cur is None or p["total_tokens_per_s_per_chip"] > cur[
                "total_tokens_per_s_per_chip"]:
            peaks[p["series"]] = p
    for name, p in sorted(peaks.items(), key=lambda kv: -kv[1][
            "total_tokens_per_s_per_chip"]):
        price = our_price if p.get("source") == "study" else p.get(
            "price_per_chip_hour")
        lines.append(f"| {name} | {p.get('source')} |"
                     f" {_f(p['total_tokens_per_s_per_chip'], 0)} |"
                     f" {p.get('conc')} | {_f(p.get(xk), 1)} |"
                     f" {_f(tokens_per_dollar(p, price), 0)} |")
    lines += ["", "Charts: `throughput_vs_interactivity.svg`,"
              " `tokens_per_dollar_vs_interactivity.svg`; data: `points.csv`,"
              " `points.json`.", ""]
    return "\n".join(lines)


def _topology_short(key: Any) -> str:
    """`PD|GPU=32|...|P-GPU=20|D-GPU=12|...` -> `PD 32 (P20/D12)`."""
    if not key:
        return ""
    parts = dict(kv.split("=", 1) for kv in str(key).split("|") if "=" in kv)
    mode = str(key).split("|")[0]
    gpus = parts.get("GPU", "?")
    if mode == "PD":
        return f"PD {gpus} (P{parts.get('P-GPU', '?')}/D{parts.get('D-GPU', '?')})"
    return f"{mode} {gpus} TP{parts.get('TP', '?')} EP{parts.get('EP', '?')}"


def render_png(svg_path: pathlib.Path, png_path: pathlib.Path,
               size: tuple[int, int] = (1000, 640)) -> str | None:
    """Rasterise with a headless Chrome/Chromium if one is installed
    (ImageMagick's built-in SVG renderer fills stroked paths; browsers
    render these charts correctly). Returns the browser used, or None."""
    for exe in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(exe)
        if not path:
            continue
        cmd = [path, "--headless=new", "--no-sandbox", "--disable-gpu",
               "--hide-scrollbars", f"--window-size={size[0]},{size[1]}",
               f"--screenshot={png_path}", svg_path.resolve().as_uri()]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if png_path.is_file():
            return exe
    return None


def build_charts(points: list[dict[str, Any]], lb_points: list[dict[str, Any]],
                 *, interactivity: str, our_price: float | None,
                 title: str, price_note: str, basis: str) -> tuple[str, str]:
    def series_of(pts: list[dict[str, Any]], y_key: str, price_fn,
                  *, study: bool) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for p in pts:
            groups.setdefault(p["series"], []).append(p)
        out = []
        for name, group in groups.items():
            rows = []
            for p in group:
                y = price_fn(p) if y_key == "tokens_per_dollar" else p.get(y_key)
                x = p.get(interactivity)
                if x is None or y is None or float(x) <= 0:
                    continue
                rows.append({"x": float(x), "y": float(y), "_p": p})
            if not rows:
                continue
            front = {id(r) for r in pareto(rows, "x", "y")}
            plotted = []
            for r in rows:
                p = r["_p"]
                on = id(r) in front
                plotted.append({
                    "x": r["x"], "y": r["y"], "on_line": on,
                    "faded": not on,
                    "label": f"c{p.get('conc')}" if (on or study) else "",
                    "hollow": not p.get("admissible", True)})
            out.append({"name": name, "points": plotted,
                        "dashed": not study, "width": 3 if study else 2,
                        **({"marker": "diamond"} if study else {})})
        return out

    x_label = X_LABELS.get(interactivity, interactivity)
    tput = svgchart.scatter_svg(
        series_of(points, "total_tokens_per_s_per_chip", None, study=True) +
        series_of(lb_points, "total_tokens_per_s_per_chip", None, study=False),
        title=f"{title}: total tokens/s per chip vs interactivity",
        x_label=x_label, y_label="total tokens/s per chip (physical chips)",
        subtitle="solid: this study (labels = concurrency); dashed:"
                 " leaderboard series; lines follow each series' Pareto"
                 " frontier",
        footnote="AgentX agentic-coding replay; qualifying gate TTFT p50"
                 " <= 5 s unless the workload spec says otherwise")
    tpd = svgchart.scatter_svg(
        series_of(points, "tokens_per_dollar",
                  lambda p: tokens_per_dollar(p, our_price), study=True) +
        series_of(lb_points, "tokens_per_dollar",
                  lambda p: tokens_per_dollar(p, p.get("price_per_chip_hour")),
                  study=False),
        title=f"{title}: total tokens per dollar vs interactivity",
        x_label=x_label,
        y_label="total tokens per $ (tok/s/chip x 3600 / $ per chip-hour)",
        subtitle=f"prices: dashboard `{basis}` TCO basis; {price_note}",
        footnote=f"price table: {pins.AGENTX_PRICE_SOURCE}; a series"
                 " without a price is omitted here")
    return tput, tpd


def cmd_build(args: argparse.Namespace) -> int:
    overrides: dict[str, float] = {}
    for item in args.price or []:
        hw, _, val = item.partition("=")
        overrides[hw.strip().lower()] = float(val)
    basis = args.tco_basis
    points = collect_study_points(args.study_dir, args.reports or [])
    doc = _read(args.leaderboard) if args.leaderboard else {"entries": []}
    if isinstance(doc, list):
        doc = {"entries": doc}
    # A file-level price table (legacy `prices: {hw: usd}`) sits between
    # the pinned table and the command line.
    for k, v in (doc.get("prices") or {}).items():
        overrides.setdefault(str(k).lower(), float(v))
    entries = doc.get("entries") or []
    lb_points = leaderboard_points(entries, basis, overrides)
    our_hw = (args.hardware or "tpu7x").lower()
    our_price = price_for(our_hw, basis, overrides)
    price_note = (f"{our_hw} at ${our_price:.2f} per chip-hour"
                  if our_price else
                  f"no price for {our_hw}; pass --price {our_hw}=<usd>")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in points:
        p["price_per_chip_hour"] = our_price
        p["tokens_per_dollar"] = tokens_per_dollar(p, our_price)
    for p in lb_points:
        p["tokens_per_dollar"] = tokens_per_dollar(
            p, p.get("price_per_chip_hour"))
    _frontier_flags(points, args.interactivity)
    _frontier_flags(lb_points, args.interactivity)
    tput, tpd = build_charts(points, lb_points, interactivity=args.interactivity,
                             our_price=our_price, title=args.title,
                             price_note=price_note, basis=basis)
    (out / "throughput_vs_interactivity.svg").write_text(tput, encoding="utf-8")
    (out / "tokens_per_dollar_vs_interactivity.svg").write_text(
        tpd, encoding="utf-8")
    pngs: dict[str, str | None] = {}
    if args.png:
        for stem in ("throughput_vs_interactivity",
                     "tokens_per_dollar_vs_interactivity"):
            pngs[stem] = render_png(out / f"{stem}.svg", out / f"{stem}.png")
    (out / "leaderboard.md").write_text(markdown(
        points, lb_points, title=args.title,
        interactivity=args.interactivity, our_price=our_price,
        price_note=price_note, basis=basis), encoding="utf-8")
    everything = points + lb_points
    snapshot = doc.get("snapshot") or doc.get("_note")
    (out / "points.json").write_text(json.dumps(
        {"interactivity": args.interactivity, "tco_basis": basis,
         "price_overrides": overrides, "price_source": pins.AGENTX_PRICE_SOURCE,
         "leaderboard_snapshot": snapshot, "points": everything},
        indent=2) + "\n", encoding="utf-8")
    keys = ["series", "source", "hardware", "framework", "precision", "conc",
            "chips", "topology", "total_tokens_per_s_per_chip",
            "output_tokens_per_s_per_chip", "tokens_per_dollar",
            "price_per_chip_hour", "intvty_p90", "intvty_p75", "intvty_p50",
            "e2e_norm_intvty_p90", "ttft_p50_s", "ttft_p90_s", "tpot_p50_ms",
            "e2e_p50_s", "gpu_prefix_hit_rate", "cpu_prefix_hit_rate", "ok",
            "total", "admissible", "frontier", "dashboard_frontier", "date",
            "citation", "report"]
    with (out / "points.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for p in everything:
            writer.writerow(p)
    best = max((p for p in points if p.get("admissible")),
               key=lambda p: p.get("total_tokens_per_s_per_chip") or 0,
               default=None)
    summary = {"status": "completed", "out": str(out),
               "interactivity": args.interactivity, "tco_basis": basis,
               "study_points": len(points), "leaderboard_points": len(lb_points),
               "price_per_chip_hour": our_price, "png": pngs or None,
               "best": {k: best.get(k) for k in (
                   "series", "conc", "total_tokens_per_s_per_chip",
                   "tokens_per_dollar", args.interactivity)} if best else None}
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_fetch_live(args: argparse.Namespace) -> int:
    if args.from_file:
        doc = views_to_entries(_read(args.from_file), f"file:{args.from_file}")
    else:
        doc = fetch_live(args.model)
    out = pathlib.Path(args.out)
    if args.merge:
        base = _read(args.merge)
        if isinstance(base, list):
            base = {"entries": base}
        live_names = {e["name"] for e in doc["entries"]}
        kept = [e for e in base.get("entries") or []
                if e.get("name") not in live_names]
        doc = {**base, "snapshot": doc["snapshot"],
               "entries": doc["entries"] + kept}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "out": str(out),
                      "entries": len(doc["entries"]),
                      "points": sum(len(e["points"]) for e in doc["entries"]),
                      "snapshot": doc.get("snapshot")}, indent=2))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    entry = {"name": args.name, "hardware": args.hardware,
             "framework": args.framework, "precision": args.precision,
             "price_per_chip_hour": args.price_per_chip_hour,
             "source": args.source or "InferenceX agg JSON",
             "points": import_inferencex(args.files)}
    out = pathlib.Path(args.out)
    doc = _read(out) if out.is_file() else {"entries": []}
    if isinstance(doc, list):
        doc = {"entries": doc}
    doc["entries"] = [e for e in doc.get("entries") or []
                      if e.get("name") != args.name] + [entry]
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "entry": args.name,
                      "points": len(entry["points"]), "out": str(out)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--study-dir", type=pathlib.Path, default=None)
    b.add_argument("--reports", action="append", default=[],
                   help="glob of llmd-bench-report.json files (repeatable)")
    b.add_argument("--leaderboard", default=None,
                   help="entries JSON ({snapshot?, prices?, entries} or a list)")
    b.add_argument("--tco-basis", choices=pins.AGENTX_TCO_BASES,
                   default="internal",
                   help="dashboard price basis (default: internal)")
    b.add_argument("--price", action="append", default=[],
                   metavar="HW=USD_PER_HOUR",
                   help="override the pinned price of one SKU")
    b.add_argument("--hardware", default="tpu7x")
    b.add_argument("--interactivity", choices=INTERACTIVITY_KEYS,
                   default="intvty_p90")
    b.add_argument("--title", default="Qwen3.5-397B AgentX on tpu7x + llm-d")
    b.add_argument("--png", action="store_true",
                   help="also rasterise the charts with a headless Chrome")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    f = sub.add_parser("fetch-live")
    f.add_argument("--model", default=pins.AGENTX_LEADERBOARD_MODEL)
    f.add_argument("--from-file", default=None,
                   help="convert a saved views-API response instead of fetching")
    f.add_argument("--merge", default=None,
                   help="existing entries JSON whose non-live entries to keep")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fetch_live)
    i = sub.add_parser("import-inferencex")
    i.add_argument("files", nargs="+")
    i.add_argument("--name", required=True)
    i.add_argument("--hardware", required=True)
    i.add_argument("--framework", default="")
    i.add_argument("--precision", default="")
    i.add_argument("--price-per-chip-hour", type=float, default=None)
    i.add_argument("--source", default=None)
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_import)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
