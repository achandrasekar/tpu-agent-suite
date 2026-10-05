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
"""Score a fleet arm from its bench reports; keep the inner-sweep ledger.

Two subcommands, one ledger file:

  ledger  --ledger L --report R [--sets JSON] [--config CFG]
      Append one `deploy bench` report to the arm's sweep ledger and print
      the best admissible entry so far. The executor calls this after every
      measurement; the ledger is what makes the chosen configuration
      traceable to the execution that measured it.

  score   --ledger L --protocol P --arm A [--baseline-value X]
          --out arm-result.json
      Pick the best admissible ledger entry, compute the verdict against the
      baseline with `framework.study.compute_arm_verdict` (the same function
      every study uses), and write the arm-result envelope.

The verdict rules are the study's, not this script's: a win must clear the
noise floor at the pre-registered replicate count with every gate passing in
the same execution; anything the Jobs could not measure is invalid.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve()
_SHIPPED_RUNTIME = _HERE.parents[2] / "runtime"
_REPO_NEXUS = _HERE.parents[3]
for _candidate in (_SHIPPED_RUNTIME, _REPO_NEXUS):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    from framework import study  # noqa: E402
except ImportError:
    import study  # type: ignore[no-redef]  # noqa: E402


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _direction(protocol: dict[str, Any]) -> str:
    return protocol["primary_metric"]["direction"]


def better(a: float | None, b: float | None, direction: str) -> bool:
    """Is `a` a better primary value than `b`?"""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if direction == "maximize" else a < b


def ledger_entry(report: dict[str, Any], sets: dict[str, Any] | None,
                 primary: str) -> dict[str, Any]:
    metric = (report.get("metrics") or {}).get(primary) or {}
    return {
        "execution_id": report["execution_id"],
        "sets": sets or {},
        "config_digest": report["release"]["config_digest"],
        "stage": report.get("stage"),
        "value": metric.get("value"),
        "admissible": bool(report.get("admissible"))
        and bool(report.get("telemetry_valid", True)),
        "telemetry_valid": bool(report.get("telemetry_valid", True)),
        "replicates": int(report.get("replicates") or 0),
        "chips_total": int(report["release"]["chips_total"]),
        "report": report.get("_path", ""),
        "config_path": "",
    }


def best_entry(entries: list[dict[str, Any]],
               direction: str) -> dict[str, Any] | None:
    """The best admissible SLO-stage entry; sweep stages never win an arm."""
    best: dict[str, Any] | None = None
    for entry in entries:
        if entry.get("stage") != "slo" or not entry.get("admissible"):
            continue
        if best is None or better(entry["value"], best["value"], direction):
            best = entry
    return best


def cmd_ledger(args: argparse.Namespace) -> int:
    report = _load(args.report)
    report["_path"] = str(args.report)
    ledger = _load(args.ledger) if args.ledger.is_file() else {
        "schema": "llmd-sweep-ledger",
        "primary_metric": args.primary_metric or report.get("primary_metric"),
        "direction": args.direction,
        "entries": [],
    }
    primary = ledger["primary_metric"]
    sets = json.loads(args.sets) if args.sets else {}
    entry = ledger_entry(report, sets, primary)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    if args.config:
        # `reconfigure` rewrites the release's config in place, so the config
        # that produced this measurement is snapshotted now, by execution id.
        snap_dir = args.ledger.parent / "configs"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = snap_dir / f"{report['execution_id']}.json"
        shutil.copy2(args.config, snap)
        entry["config_path"] = str(snap)
    ledger["entries"].append(entry)
    args.ledger.write_text(json.dumps(ledger, indent=2) + "\n",
                           encoding="utf-8")
    best = best_entry(ledger["entries"], ledger["direction"])
    print(json.dumps({
        "entries": len(ledger["entries"]),
        "last": ledger["entries"][-1],
        "best": best,
        "remaining_budget": (None if args.budget is None else max(
            0, args.budget - (len(ledger["entries"]) - 1))),
    }, indent=2))
    return 0


def score_arm(ledger: dict[str, Any], protocol: dict[str, Any], *,
              arm: str, baseline_value: float | None,
              config_path: str) -> dict[str, Any]:
    direction = _direction(protocol)
    primary = protocol["primary_metric"]["name"]
    entries = ledger.get("entries") or []
    best = best_entry(entries, direction)
    slo_entries = [e for e in entries if e.get("stage") == "slo"]
    telemetry_valid = any(e.get("telemetry_valid") for e in slo_entries)
    chosen = best or (slo_entries[-1] if slo_entries else None)

    if chosen is None:
        return {
            "status": "failed",
            "arm": arm,
            "verdict": "invalid",
            "summary": f"arm {arm}: no SLO-stage measurement in the ledger",
            "sweep_ledger": entries,
            "artifacts": [],
        }
    report = _load(pathlib.Path(chosen["report"])) if chosen.get(
        "report") and pathlib.Path(chosen["report"]).is_file() else {}
    gates_passed = bool(chosen["admissible"])
    required = int(protocol["replicate_count"])
    if baseline_value is None:
        # The baseline arm itself: no comparison, the measurement stands.
        verdict, delta = ("tie" if gates_passed else "invalid"), None
    else:
        verdict, delta = study.compute_arm_verdict(
            metric_value=chosen["value"],
            baseline_value=float(baseline_value),
            direction=direction,
            noise_floor_pct=float(protocol["noise_floor_pct"]),
            gates_passed=gates_passed,
            replicates=int(chosen["replicates"]),
            required_replicates=required,
            telemetry_valid=bool(chosen["telemetry_valid"]),
        )
    gates = [{
        "name": g["name"],
        "passed": bool(g["passed"]),
        **({"value": float(g["value"])} if g.get("value") is not None else {}),
        "execution_id": chosen["execution_id"],
    } for g in (report.get("gates") or [])]
    # Every protocol gate must be reported, even when the Job never measured
    # it; an unreported gate is a gate that failed.
    reported = {g["name"] for g in gates}
    for g in protocol["admissibility_gates"]:
        if g["name"] not in reported:
            gates.append({"name": g["name"], "passed": False,
                          "execution_id": chosen["execution_id"]})
            if verdict == "win":
                verdict = "invalid"
    metric = {
        "name": primary,
        "value": float(chosen["value"]) if chosen["value"] is not None else 0.0,
        "unit": protocol["primary_metric"].get("unit", ""),
        "replicates": int(chosen["replicates"]) or 1,
        "provenance": "measured",
    }
    if delta is not None:
        metric["delta_pct"] = round(float(delta), 4)
    summary = (f"arm {arm}: {verdict}; {primary}={metric['value']:.4g}"
               + (f" ({delta:+.2f}% vs baseline)" if delta is not None else "")
               + f" on {chosen['chips_total']} chips after"
               f" {len(entries)} measurement(s); gates"
               f" {'pass' if gates_passed else 'FAIL'}")
    return {
        "status": "completed" if telemetry_valid else "failed",
        "arm": arm,
        "verdict": verdict,
        "execution_id": chosen["execution_id"],
        "metric": metric,
        "gates": gates,
        "artifacts": sorted({e["report"] for e in entries if e.get("report")}),
        "config_path": config_path,
        "sweep_ledger": [{
            k: v for k, v in e.items()
            if k in ("execution_id", "sets", "config_digest", "value",
                     "admissible", "report")
        } for e in entries],
        "chips_total": int(chosen["chips_total"]),
        "summary": summary,
    }


def cmd_score(args: argparse.Namespace) -> int:
    ledger = _load(args.ledger)
    protocol = _load(args.protocol)
    direction = _direction(protocol)
    best = best_entry(ledger.get("entries") or [], direction)
    config_path = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if best and best.get("config_path") and pathlib.Path(
            best["config_path"]).is_file():
        # The snapshot taken when the best entry was measured becomes the
        # arm's config, one file beside the result.
        dst = args.out.parent / "llmd-config.json"
        shutil.copy2(best["config_path"], dst)
        config_path = str(dst)
    result = score_arm(ledger, protocol, arm=args.arm,
                       baseline_value=args.baseline_value,
                       config_path=config_path)
    if config_path and result.get("execution_id"):
        cfg_doc = _load(pathlib.Path(config_path))
        prov = dict(cfg_doc.get("provenance") or {})
        prov["arm"] = args.arm
        prov["execution_id"] = result["execution_id"]
        if result.get("metric"):
            prov["metric"] = {
                k: result["metric"][k]
                for k in ("name", "value", "provenance")
                if k in result["metric"]
            }
        cfg_doc["provenance"] = prov
        pathlib.Path(config_path).write_text(
            json.dumps(cfg_doc, indent=2) + "\n", encoding="utf-8")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ledger")
    p.add_argument("--ledger", required=True, type=pathlib.Path)
    p.add_argument("--report", required=True, type=pathlib.Path)
    p.add_argument("--sets", default=None, help="JSON of the hot/warm sets"
                   " this measurement was taken under")
    p.add_argument("--config", default=None, type=pathlib.Path,
                   help="The release's llmd-config.json as measured;"
                   " snapshotted beside the ledger")
    p.add_argument("--primary-metric", default=None)
    p.add_argument("--direction", default="maximize",
                   choices=["maximize", "minimize"])
    p.add_argument("--budget", type=int, default=None,
                   help="inner_sweep_budget, to report what is left")
    p.set_defaults(fn=cmd_ledger)
    p = sub.add_parser("score")
    p.add_argument("--ledger", required=True, type=pathlib.Path)
    p.add_argument("--protocol", required=True, type=pathlib.Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--baseline-value", type=float, default=None)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.set_defaults(fn=cmd_score)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
