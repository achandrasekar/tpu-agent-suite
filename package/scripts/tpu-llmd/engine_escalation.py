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
"""Decide whether the fleet search should hand off to the engine, and
write everything that hand-off needs.

The llm-d study moves the fleet: shape, router, KV reuse, the fleet
whitelist of engine knobs. When it plateaus with the objective unmet, the
remaining headroom is inside one engine -- usually one role's engine --
and that is `tpu-performance`'s domain. This script is the seam between
the two studies. Given the llm-d study's state and the incumbent's
measurements it:

1. decides (`decision.json`): escalate only when the loop is enabled, the
   objective is unmet (a failed gate at target rate, or a stated target
   value not reached), the search is at a plateau, and a bound role is
   left that an earlier escalation did not already exhaust;
2. attributes the bound to a role and a bottleneck kind (size_pd);
3. writes the **engine escalation** (`engine-escalation.json`): the
   operating point the fleet measured (ISL/OSL, prefix hit rate,
   concurrency per replica, offered rate), the engine objective for that
   role (prefill: prompt tokens/s under the TTFT gate; decode: generation
   tokens/s under the TPOT gate), a `study-protocol` for the engine study,
   an inference-perf microbenchmark config for the role, and hypothesis
   seeds in `hypothesis_bank.py`'s shape whose `target_parameters.serving`
   name the args/env/image change each arm makes, which is what
   `recipe_bump.py` reads back.

It writes no study state: the orchestrator opens the engine study from
the protocol file and admits the seeds. Usage:

    engine_escalation.py prepare --study-state <status.json> \\
        --llmd-study <llmd-study.json> --workload <spec> --recipe <recipe> \\
        --incumbent <llmd-config.json> --report <llmd-bench-report.json> \\
        [--profile <metrics.json>] [--plateau yes|no] [--escalations N] \\
        --run-id <run_id> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve()
_SHIPPED_RUNTIME = _HERE.parents[2] / "runtime"
_REPO_NEXUS = _HERE.parents[3]
for _candidate in (_SHIPPED_RUNTIME, _REPO_NEXUS):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    from framework.deploy import render, workload  # noqa: E402
except ImportError:  # shipped runtime: `deploy` is top-level
    from deploy import render, workload  # type: ignore  # noqa: E402

if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import size_pd  # noqa: E402  (same directory)

ESCALATION_FILE = "engine-escalation.json"
PROTOCOL_FILE = "engine-protocol.json"
HYPOTHESES_FILE = "hypotheses.json"
DECISION_FILE = "decision.json"
DEFAULT_MAX_ESCALATIONS = 3
DEFAULT_ENGINE_MAX_ROUNDS = 8
DEFAULT_ENGINE_PLATEAU = 3
MICROBENCH_STAGE_S = 120
# Engine-level metric and gate per role. Prefill is paid for in TTFT,
# decode in TPOT; an aggregated engine answers for both.
ROLE_OBJECTIVE: dict[str, dict[str, Any]] = {
    "prefill": {"metric": "prompt_tokens_per_s", "unit": "tok/s",
                "gate_metrics": ("ttft",)},
    "decode": {"metric": "generation_tokens_per_s", "unit": "tok/s",
               "gate_metrics": ("tpot",)},
    "aggregated": {"metric": "output_tokens_per_s", "unit": "tok/s",
                   "gate_metrics": ("ttft", "tpot")},
}


def _read(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _digest(doc: Any) -> str:
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _metric(doc: dict[str, Any] | None, name: str,
            role: str | None = None) -> float | None:
    return size_pd._metric(doc, name, role)


# -- decision -------------------------------------------------------------------


def loop_settings(llmd_study: dict[str, Any] | None) -> dict[str, Any]:
    """The loop's knobs from llmd-study.json, with defaults: enabled, up
    to three escalations, no role exhausted yet."""
    loop = dict((llmd_study or {}).get("engine_loop") or {})
    return {
        "enabled": bool(loop.get("enabled", True)),
        "max_escalations": int(loop.get("max_escalations")
                               or DEFAULT_MAX_ESCALATIONS),
        "exhausted_roles": list(loop.get("exhausted_roles") or []),
        "escalations": list(loop.get("escalations") or []),
    }


def objective_unmet(report: dict[str, Any] | None,
                    llmd_study: dict[str, Any] | None,
                    study_state: dict[str, Any] | None) -> dict[str, Any]:
    """Is the user's objective still open on the incumbent?

    Two readings, either suffices: a failed SLO gate at target rate in the
    incumbent's bench report, or an `objective_target` in llmd-study.json
    the incumbent's primary metric has not reached.
    """
    failed = size_pd._failed_gates(report)
    reasons: list[str] = []
    if failed:
        reasons.append(f"gates failed at target rate: {', '.join(failed)}")
    target = (llmd_study or {}).get("objective_target") or {}
    value = target.get("value")
    if value is not None:
        direction = str(target.get("direction") or "maximize")
        current = _incumbent_value(study_state)
        if current is None:
            reasons.append("objective_target set but the incumbent has no"
                           " measured primary metric yet")
        elif (direction == "maximize" and current < float(value)) or (
                direction == "minimize" and current > float(value)):
            reasons.append(f"primary metric {current:.4g} has not reached"
                           f" the target {float(value):.4g} ({direction})")
    return {"unmet": bool(reasons), "reasons": reasons,
            "failed_gates": failed}


def _incumbent_value(state: dict[str, Any] | None) -> float | None:
    if not state:
        return None
    best = None
    for arm in state.get("arms") or []:
        result = arm.get("result") or {}
        metric = result.get("metric") or {}
        if arm.get("status") == "decided" and result.get("verdict") in (
                "win", "tie", "baseline") and metric.get("value") is not None:
            if arm["arm"] == "baseline" and best is None:
                best = float(metric["value"])
            elif result.get("verdict") == "win":
                best = float(metric["value"])
    if best is None:
        base = ((state.get("baseline") or {}).get("metric") or {}).get("value")
        best = float(base) if base is not None else None
    return best


def at_plateau(state: dict[str, Any] | None, plateau_flag: str | None
               ) -> dict[str, Any]:
    """The search has nothing left to try at the fleet level.

    The orchestrator already evaluates the protocol's plateau rule and
    passes it in; independently, a state with no open arm is a plateau.
    """
    open_arms = [a["arm"] for a in (state or {}).get("arms") or []
                 if a.get("status") == "open"]
    flagged = str(plateau_flag or "").lower() in ("yes", "true", "1")
    return {"plateau": flagged or not open_arms,
            "open_arms": open_arms, "flagged_by_orchestrator": flagged}


def decide(*, loop: dict[str, Any], unmet: dict[str, Any],
           plateau: dict[str, Any], attribution: dict[str, Any]
           ) -> dict[str, Any]:
    role = attribution["role"]
    if role == "none":
        # Nothing in the fleet metrics names a pool; the objective is
        # still unmet, so the whole engine is the subject.
        role = "aggregated"
    reasons: list[str] = []
    if not loop["enabled"]:
        reasons.append("engine_loop disabled in llmd-study.json")
    if len(loop["escalations"]) >= loop["max_escalations"]:
        reasons.append(f"max_escalations {loop['max_escalations']} reached")
    if not unmet["unmet"]:
        reasons.append("objective met on the incumbent; nothing to escalate")
    if not plateau["plateau"]:
        reasons.append(f"fleet search still has open arms"
                       f" {plateau['open_arms']}")
    if role in loop["exhausted_roles"]:
        reasons.append(f"role {role} already escalated and exhausted")
    if role == "loadgen":
        reasons.append("incumbent measurement is client-limited: scale the"
                       " load generator out and re-measure before deciding"
                       " anything about the engine")
    return {"escalate": not reasons, "role": role,
            "kind": attribution["kind"] if attribution["kind"] != "none"
            else "scheduler-batching",
            "blockers": reasons, "objective": unmet, "plateau": plateau,
            "attribution": attribution}


# -- operating point and objective ------------------------------------------------


def operating_point(profile: dict[str, Any], incumbent: dict[str, Any],
                    metrics: dict[str, Any] | None, role: str
                    ) -> dict[str, Any]:
    """What the engine must be measured at: the fleet's shape and load."""
    shape = profile.get("shape") or {}
    hit = _metric(metrics, "prefix_hit_rate")
    pool_role = role if role in ("prefill", "decode") else None
    running = _metric(metrics, "requests_running_mean", pool_role) or \
        _metric(metrics, "requests_running_mean")
    pool = incumbent["topology"].get(pool_role or "decode") or \
        incumbent["topology"]["decode"]
    replicas = int(pool.get("replicas") or 1)
    return {
        "target_qps": float(profile["rate"]["target_qps"]),
        "isl_mean": float((shape.get("input_sequence_length") or {}).get(
            "mean") or 0),
        "osl_mean": float((shape.get("output_sequence_length") or {}).get(
            "mean") or 0),
        "prefix_hit_rate": hit,
        "concurrency_per_replica": max(1, int(round(running or 1))),
        "replicas": replicas,
        "engine": render.role_engine(incumbent, pool_role or "decode"),
        "fleet_shape": {
            "mode": incumbent["topology"].get("mode"),
            "prefill_replicas": int((incumbent["topology"].get("prefill")
                                     or {}).get("replicas") or 0),
            "decode_replicas": int(incumbent["topology"]["decode"].get(
                "replicas") or 1),
        },
    }


def _tightest_bound(profile: dict[str, Any], key_prefix: str
                    ) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for tenant in profile.get("tenants") or []:
        for name, bound in (tenant.get("slo") or {}).items():
            if name.startswith(key_prefix) and bound is not None:
                if best is None or float(bound) < best[1]:
                    best = (name, float(bound))
    return best


def engine_objective(profile: dict[str, Any], role: str
                     ) -> dict[str, Any]:
    spec = ROLE_OBJECTIVE[role]
    gates: list[dict[str, Any]] = []
    for prefix in spec["gate_metrics"]:
        found = _tightest_bound(profile, prefix)
        if found:
            name, bound = found
            gates.append({"name": f"slo.{name}", "metric": name,
                          "comparator": "lte", "bound": bound, "unit": "ms",
                          "same_execution": True})
    gates.append({"name": "engine.error_rate", "metric": "error_rate",
                  "comparator": "lte", "bound": 0.001, "unit": "ratio",
                  "same_execution": True})
    return {"metric": spec["metric"], "unit": spec["unit"],
            "direction": "maximize", "gates": gates}


def engine_protocol(profile: dict[str, Any], recipe: dict[str, Any],
                    llmd_protocol: dict[str, Any] | None, role: str,
                    point: dict[str, Any]) -> dict[str, Any]:
    """A study-protocol for the engine study: one replica of the role's
    slice at the fleet's per-replica concurrency; replicates, noise floor
    and mode inherited from the llm-d protocol."""
    objective = engine_objective(profile, role)
    accel = recipe.get("accelerator") or {}
    llmd_protocol = llmd_protocol or {}
    shape = (f"{role}:{int(point['isl_mean'])}-{int(point['osl_mean'])}"
             if role != "prefill" else f"prefill:{int(point['isl_mean'])}-1")
    return {
        "primary_metric": {"name": objective["metric"],
                           "unit": objective["unit"],
                           "direction": objective["direction"]},
        "measurement_cell": {
            "topology": f"{accel.get('tpu_accelerator')}-"
                        f"{accel.get('tpu_topology')}",
            "shape": shape,
            "concurrency": int(point["concurrency_per_replica"]),
        },
        "replicate_count": int(llmd_protocol.get("replicate_count") or 3),
        "noise_floor_pct": float(llmd_protocol.get("noise_floor_pct") or 5.0),
        "stopping_rule": {"max_rounds": DEFAULT_ENGINE_MAX_ROUNDS,
                          "plateau_rounds": DEFAULT_ENGINE_PLATEAU},
        "admissibility_gates": objective["gates"],
        "mode": str(llmd_protocol.get("mode") or "approve"),
    }


def microbench_config(profile: dict[str, Any], recipe: dict[str, Any],
                      role: str, point: dict[str, Any]) -> dict[str, Any]:
    """An inference-perf config that loads one replica the way the fleet
    loads this role: prefill = the workload's prompt length with a
    one-token answer, decode = a short prompt with the workload's output
    length, both at the measured per-replica concurrency, closed loop."""
    isl = max(1, int(point["isl_mean"]))
    osl = max(1, int(point["osl_mean"]))
    if role == "prefill":
        question, output = isl, 1
    elif role == "decode":
        question, output = min(128, isl), osl
    else:
        question, output = isl, osl
    hit = point.get("prefix_hit_rate") or 0.0
    system = int(isl * hit) if role != "decode" else 0
    return {
        "api": {"type": "completion", "streaming": True},
        "server": {"type": "vllm", "model_name": recipe["model"],
                   "base_url": "http://localhost:8000"},
        "data": {"type": "shared_prefix",
                 "shared_prefix": {"num_groups": 32, "num_prompts_per_group": 16,
                                   "system_prompt_len": system,
                                   "question_len": max(1, question - system),
                                   "output_len": output}},
        "load": {"type": "concurrent",
                 "stages": [{"concurrency_level": int(
                     point["concurrency_per_replica"]),
                     "duration": MICROBENCH_STAGE_S}]},
        "report": {"request_lifecycle": {"summary": True,
                                         "per_stage": True}},
        "tokenizer": {"pretrained_model_name_or_path": recipe["model"]},
        "metrics": {"type": "prometheus",
                    "prometheus": {"url": "http://localhost:9090"}},
    }


# -- hypothesis seeds -------------------------------------------------------------


def _h(id_: str, kind: str, claim: str, serving: dict[str, Any],
       evidence: list[dict[str, str]], *, share: float = 0.5,
       headroom: float = 0.1, details: str = "") -> dict[str, Any]:
    return {"id": id_, "kind": kind, "claim": claim, "details": details,
            "evidence": evidence, "share": share, "headroom": headroom,
            "target_parameters": {"serving": serving}}


def hypothesis_seeds(role: str, kind: str, point: dict[str, Any],
                     attribution: dict[str, Any], recipe: dict[str, Any],
                     exhausted_knobs: list[str], evidence_refs: dict[str, str]
                     ) -> dict[str, Any]:
    """Opening hypotheses for the engine study, as serving-config deltas.

    Each `target_parameters.serving` is `{args: {flag: value}, env: {},
    image: ""}` so a win can be bumped into the recipe mechanically. Knobs
    the fleet sweep already walked are listed as `excluded` with their
    best values, so the researcher does not re-propose them; anything
    deeper (kernels, compilation buckets, sharding) is the researcher's,
    seeded here only as a named direction with no serving delta.
    """
    ev = [{"type": "fleet-report", "ref": evidence_refs.get("report", "-"),
           "note": "; ".join(attribution.get("evidence") or [
               "objective unmet at target rate"])}]
    if evidence_refs.get("metrics"):
        ev.append({"type": "fleet-metrics", "ref": evidence_refs["metrics"],
                   "note": f"per-role metrics; role={role}"})
    engine = point.get("engine") or {}
    conc = int(point["concurrency_per_replica"])
    seeds: list[dict[str, Any]] = []
    if role == "prefill":
        bt = int(engine.get("max_num_batched_tokens") or 0)
        seeds.append(_h(
            f"e-{role}-batched-tokens", "scheduler-batching",
            "a larger prefill batched-token budget raises prompt tokens/s"
            " at the same TTFT bound on the prefill engine",
            {"args": {"--max-num-batched-tokens": (bt * 2) if bt else 16384}},
            ev, share=0.6, headroom=0.2,
            details="prefill-only engine; TTFT gate from the workload SLO"))
        seeds.append(_h(
            f"e-{role}-async-sched", "host-critical-path",
            "async scheduling hides host scheduling time between prefill"
            " batches", {"args": {"--async-scheduling": True}}, ev,
            share=0.3, headroom=0.1))
    elif role == "decode":
        ms = int(engine.get("max_num_seqs") or 0)
        seeds.append(_h(
            f"e-{role}-max-seqs", "scheduler-batching",
            "a larger decode batch raises generation tokens/s while TPOT"
            f" stays under the bound at concurrency {conc}",
            {"args": {"--max-num-seqs": (ms * 2) if ms else 256}}, ev,
            share=0.6, headroom=0.2))
        seeds.append(_h(
            f"e-{role}-kv-dtype", "memory-capacity",
            "an 8-bit KV cache doubles resident sequences and removes"
            " preemption at the decode batch the fleet runs",
            {"args": {"--kv-cache-dtype": "fp8"}}, ev,
            share=0.4, headroom=0.15,
            details="needs the parity gate; accuracy is an admissibility"
                    " check, not a hope"))
    else:
        seeds.append(_h(
            "e-engine-batched-tokens", "scheduler-batching",
            "a larger batched-token budget raises throughput at the same"
            " SLO on a single replica",
            {"args": {"--max-num-batched-tokens": 16384}}, ev,
            share=0.5, headroom=0.15))
    if kind == "memory-capacity":
        seeds.append(_h(
            f"e-{role}-mem-util", "memory-capacity",
            "a higher memory-utilisation target gives the KV cache the"
            " headroom the fleet metrics show it lacks",
            {"args": {"--gpu-memory-utilization": 0.95}}, ev,
            share=0.4, headroom=0.1))
    # Directions without a mechanical delta: the researcher's.
    seeds.append({
        "id": f"e-{role}-recompile-padding", "kind": "recompilation-padding",
        "claim": "padding buckets sized to the fleet's measured ISL/OSL"
                 " distribution cut recompiles and padded compute",
        "details": "no serving delta; the profile decides",
        "evidence": ev, "share": 0.3, "headroom": 0.1,
        "target_parameters": {"serving": {}}})
    seeds.append({
        "id": f"e-{role}-kernel-block", "kind": "kernel-block",
        "claim": "the dominant kernel at this operating point has headroom"
                 " a kernel arm can take",
        "details": "no serving delta; attributed from the XProf profile",
        "evidence": ev, "share": 0.3, "headroom": 0.1,
        "target_parameters": {"serving": {}}})
    seeds = [s for s in seeds if not any(
        k in json.dumps(s["target_parameters"]) for k in exhausted_knobs)]
    return {"hypotheses": seeds, "excluded": exhausted_knobs}


def exhausted_fleet_knobs(arm_plans: dict[str, Any] | None,
                          state: dict[str, Any] | None) -> list[str]:
    """Engine knobs the fleet's inner sweep already walked (vllm flags),
    so the engine study starts past them."""
    flags: list[str] = []
    plans = (arm_plans or {}).get("plans") or {}
    decided = {a["arm"] for a in (state or {}).get("arms") or []
               if a.get("status") == "decided"}
    for arm_id, plan in plans.items():
        if arm_id not in decided:
            continue
        for knob in plan.get("sweep_knobs") or []:
            leaf = knob.split(".")[-1]
            flag = render.ENGINE_FLAGS.get(leaf) or render.ENGINE_SWITCHES.get(
                leaf)
            if flag and flag not in flags:
                flags.append(flag)
    return flags


# -- cli ------------------------------------------------------------------------


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = pathlib.Path(args.workload)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    profile = workload.resolve_workload(spec, spec_dir=spec_path.parent)
    recipe = json.loads(pathlib.Path(args.recipe).read_text(encoding="utf-8"))
    incumbent = render.normalize_config(json.loads(
        pathlib.Path(args.incumbent).read_text(encoding="utf-8")))
    state = _read(args.study_state)
    llmd_study = _read(args.llmd_study)
    llmd_protocol = _read(args.protocol) or (state or {}).get("protocol")
    report = _read(args.report)
    metrics = _read(args.profile)
    arm_plans = _read(args.arm_plans)
    loop = loop_settings(llmd_study)
    unmet = objective_unmet(report, llmd_study, state)
    plateau = at_plateau(state, args.plateau)
    attribution = size_pd.attribute(metrics, report)
    decision = decide(loop=loop, unmet=unmet, plateau=plateau,
                      attribution=attribution)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / DECISION_FILE).write_text(json.dumps(decision, indent=2) + "\n",
                                     encoding="utf-8")
    if not decision["escalate"]:
        return {"status": "no_escalation", "decision": decision,
                "artifacts": [str(out / DECISION_FILE)]}
    role, kind = decision["role"], decision["kind"]
    point = operating_point(profile, incumbent, metrics, role)
    protocol = engine_protocol(profile, recipe, llmd_protocol, role, point)
    micro = microbench_config(profile, recipe, role, point)
    exhausted = exhausted_fleet_knobs(arm_plans, state)
    seeds = hypothesis_seeds(role, kind, point, attribution, recipe,
                             exhausted, {"report": args.report or "",
                                         "metrics": args.profile or ""})
    study_id = (state or {}).get("study_id") or args.study_id or "llmd"
    n = len(loop["escalations"]) + 1
    engine_study_id = f"{study_id[:40]}-engine-{role}-{n}"
    micro_path = out / f"microbench.{role}.inference-perf.json"
    micro_path.write_text(json.dumps(micro, indent=2) + "\n", encoding="utf-8")
    (out / PROTOCOL_FILE).write_text(json.dumps(protocol, indent=2) + "\n",
                                     encoding="utf-8")
    (out / HYPOTHESES_FILE).write_text(json.dumps(seeds, indent=2) + "\n",
                                       encoding="utf-8")
    doc = {
        "schema": "engine-escalation",
        "study_id": study_id,
        "run_id": args.run_id,
        "escalation": n,
        "engine_study_id": engine_study_id,
        "role": role,
        "kind": kind,
        "recipe_path": str(pathlib.Path(args.recipe).resolve()),
        "recipe_digest": _digest(recipe),
        "operating_point": point,
        "engine_objective": engine_objective(profile, role),
        "protocol_path": str(out / PROTOCOL_FILE),
        "microbench_path": str(micro_path),
        "hypotheses_path": str(out / HYPOTHESES_FILE),
        "excluded_knobs": exhausted,
        "evidence": {
            "attribution": attribution,
            "objective": unmet,
            "incumbent_config_path": str(pathlib.Path(args.incumbent).resolve()),
            "report_path": args.report or "",
            "metrics_path": args.profile or "",
        },
        "return_contract": {
            "arm_result_target_parameters": "serving.{args,env,image}",
            "recipe_bump": "recipe_bump.py reads the engine study's winning"
                           " arm from its hypothesis bank and applies the"
                           " serving delta to the recipe",
        },
        "summary": (f"escalate {role} ({kind}) to tpu-performance as"
                    f" {engine_study_id}: {'; '.join(unmet['reasons'])}"),
    }
    (out / ESCALATION_FILE).write_text(json.dumps(doc, indent=2) + "\n",
                                       encoding="utf-8")
    return {"status": "escalate", "decision": decision,
            "engine_study_id": engine_study_id,
            "escalation_path": str(out / ESCALATION_FILE),
            "artifacts": [str(out / f) for f in (
                DECISION_FILE, ESCALATION_FILE, PROTOCOL_FILE,
                HYPOTHESES_FILE)] + [str(micro_path)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--study-state", required=True)
    p.add_argument("--llmd-study")
    p.add_argument("--protocol")
    p.add_argument("--workload", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--report")
    p.add_argument("--profile")
    p.add_argument("--arm-plans")
    p.add_argument("--plateau", default="no")
    p.add_argument("--study-id")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = prepare(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
