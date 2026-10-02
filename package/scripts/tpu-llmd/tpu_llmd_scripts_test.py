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
"""Tests for the tpu-llmd scripts: protocol, arms, evaluation."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE.parents[1] / "runtime", _HERE.parents[2]):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


render_protocol = _load("render_protocol")
seed_arms = _load("seed_arms")
evaluate_llmd_arm = _load("evaluate_llmd_arm")
plan_arms = _load("plan_arms")


class _Quiet(unittest.TestCase):
    """The scripts print their result JSON for the agent that called them;
    under test that is noise, so stdout is captured per test."""

    def setUp(self) -> None:
        self._stdout = contextlib.redirect_stdout(io.StringIO())
        self._stdout.__enter__()
        self.addCleanup(self._stdout.__exit__, None, None, None)

RECIPE = {
    "model": "Qwen/Qwen3-32B",
    "image": "docker.io/vllm/vllm-tpu:v0.29.0",
    "args": ["--max-model-len=4096"],
    "accelerator": {"tpu_accelerator": "tpu7x", "tpu_topology": "2x2x1"},
    "tensor_parallel": 8,
    "measured": {"knee_rps": 45},
}
SPEC = {
    "name": "acme-chat",
    "workload": {
        "catalog_entry": "interactive-chat",
        "overrides": {"rate": {"target_qps": 120}},
    },
    "tenancy": [
        {"name": "interactive", "share": 0.7, "priority": 10,
         "slo": {"ttft_p90_ms": 500, "tpot_p90_ms": 30}},
        {"name": "batch", "share": 0.3, "priority": -5,
         "slo": {"ttft_p90_ms": 5000, "tpot_p90_ms": 80}},
    ],
    "budget": {"max_chips": 64},
}


def _report(execution_id, value, admissible=True, stage="slo",
            digest="abc", chips=12, telemetry_valid=True):
    return {
        "schema": "llmd-bench-report",
        "execution_id": execution_id,
        "stage": stage,
        "replicates": 3,
        "release": {"name": "r", "config_digest": digest, "chips_total": chips,
                    "replicas": 3},
        "primary_metric": "goodput_at_slo_per_chip",
        "telemetry_valid": telemetry_valid,
        "metrics": {"goodput_at_slo": {"value": value * chips},
                    "goodput_at_slo_per_chip": {"value": value}},
        "tenants": {},
        "gates": [
            {"name": "interactive.ttft_p90_ms", "metric": "ttft_p90_ms",
             "comparator": "lte", "bound": 500, "value": 300,
             "passed": admissible},
        ],
        "admissible": admissible,
    }


class RenderProtocolTest(_Quiet):

    def test_protocol_and_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "spec.json").write_text(json.dumps(SPEC))
            (tmp / "recipe.json").write_text(json.dumps(RECIPE))
            rc = render_protocol.main([
                "--workload-spec", str(tmp / "spec.json"), "--recipe",
                str(tmp / "recipe.json"), "--out", str(tmp / "out")
            ])
            self.assertEqual(rc, 0)
            protocol = json.loads((tmp / "out/protocol.json").read_text())
            extras = json.loads((tmp / "out/llmd-study.json").read_text())
        self.assertEqual(protocol["primary_metric"]["name"],
                         "goodput_at_slo_per_chip")
        self.assertEqual(protocol["primary_metric"]["direction"], "maximize")
        names = {g["name"] for g in protocol["admissibility_gates"]}
        self.assertIn("interactive.ttft_p90_ms", names)
        self.assertIn("batch.error_rate", names)
        self.assertIn("budget.max_chips", names)
        self.assertTrue(all(g["same_execution"]
                            for g in protocol["admissibility_gates"]))
        self.assertEqual(protocol["mode"], "approve")
        self.assertEqual(extras["chips_per_replica"], 4)
        self.assertEqual(extras["inner_sweep_budget"], 8)

    def test_cost_metric_direction(self):
        spec = dict(SPEC)
        spec["objective"] = {"primary_metric": "cost_per_1M_tokens_at_slo"}
        spec["budget"] = {"max_chips": 64, "chip_hour_cost": 4.2}
        from framework.deploy import workload  # noqa: PLC0415
        profile = workload.resolve_workload(spec)
        protocol = render_protocol.build_protocol(profile, RECIPE,
                                                  mode="auto", replicates=2,
                                                  noise_floor_pct=3,
                                                  max_rounds=5,
                                                  plateau_rounds=2)
        self.assertEqual(protocol["primary_metric"]["direction"], "minimize")
        self.assertEqual(protocol["replicate_count"], 2)


class SeedArmsTest(_Quiet):

    def test_seed_staged_queue(self):
        from framework.deploy import workload  # noqa: PLC0415
        profile = workload.resolve_workload(SPEC)
        arms, plans = seed_arms.seed(profile, RECIPE, max_chips=None)
        ids = [a["arm"] for a in arms["arms"]]
        self.assertEqual(ids[0], "baseline")
        self.assertEqual(plans["capacity_floor"]["replicas"], 3)  # 120/45
        self.assertEqual(plans["plans"]["baseline"]["sets"],
                         {"topology.decode.replicas": 3})
        self.assertEqual(plans["plans"]["a1"]["sets"],
                         {"topology.decode.replicas": 2})
        by_id = {a["arm"]: a for a in arms["arms"]}
        self.assertEqual(by_id["a1"]["kind"], "fleet-topology")
        self.assertEqual(by_id["a5"]["kind"], "routing-policy")
        # interactive-chat is multi-turn with a shared system prompt, so the
        # reuse arms exist; on a managed fleet with no probe they are open.
        self.assertEqual(by_id["a3"]["status"], "open")
        self.assertEqual(plans["plans"]["a3"]["sets"]["kv.precise_routing"],
                         True)
        self.assertEqual(by_id["a4"]["status"], "open")
        self.assertIn("cpu_chunks", plans["plans"]["a4"]["sets"]["kv.tiered_host"])
        self.assertEqual(by_id["a7"]["status"], "open")
        # Two tenant priorities seed a flow-control arm.
        self.assertEqual(by_id["a9"]["status"], "open")
        bands = plans["plans"]["a9"]["sets"]["router.flow_control"][
            "priority_bands"]
        self.assertEqual([b["priority"] for b in bands], [10, -5])
        self.assertNotIn("a0", by_id)  # managed: baseline already routes
        for arm in arms["arms"]:
            self.assertIn(arm["kind"], (
                "fleet-topology", "kv-reuse", "routing-policy",
                "engine-fleet-knob", "platform-capacity"))

    def test_probe_gates_connector_arms(self):
        from framework.deploy import workload  # noqa: PLC0415
        profile = workload.resolve_workload(SPEC)
        caps = {"kv_events": False, "offload_connector": True,
                "pd_connector": False}
        arms, plans = seed_arms.seed(profile, RECIPE, max_chips=None,
                                     capabilities=caps)
        by_id = {a["arm"]: a for a in arms["arms"]}
        self.assertEqual(by_id["a3"]["status"], "blocked")
        self.assertIn("blocked_on_engine_capability", by_id["a3"]["blocked_on"])
        self.assertIn("kv_events", by_id["a3"]["blocked_on"])
        self.assertEqual(by_id["a4"]["status"], "open")
        self.assertEqual(plans["capabilities"], caps)

    def test_adopted_fleet_seeds_passthrough_baseline(self):
        from framework.deploy import workload  # noqa: PLC0415
        recipe = dict(RECIPE)
        recipe["fleet"] = {"mode": "adopted", "selector": {"app": "ds"},
                           "service": "ds", "port": 8000, "replicas": 2}
        profile = workload.resolve_workload(SPEC)
        arms, plans = seed_arms.seed(profile, recipe, max_chips=8)
        by_id = {a["arm"]: a for a in arms["arms"]}
        self.assertEqual(plans["capacity_floor"]["replicas"], 2)
        self.assertEqual(plans["capacity_floor"]["provenance"], "adopted")
        self.assertEqual(plans["plans"]["baseline"]["sets"], {})
        # Scaling, KV and engine arms would restart servers the release
        # does not own; routing arms stay open.
        for arm_id in ("a1", "a3", "a4", "a8"):
            self.assertEqual(by_id[arm_id]["status"], "blocked", arm_id)
            self.assertIn("blocked_on_adopted_fleet",
                          by_id[arm_id]["blocked_on"])
        self.assertEqual(by_id["a0"]["status"], "open")
        self.assertEqual(plans["plans"]["a0"]["sets"],
                         {"router.profile": "sticky-until-saturated"})
        self.assertEqual(by_id["a5"]["status"], "open")
        self.assertEqual(by_id["a9"]["status"], "open")

    def test_capacity_floor_capped_by_budget(self):
        from framework.deploy import workload  # noqa: PLC0415
        profile = workload.resolve_workload(SPEC)
        _, plans = seed_arms.seed(profile, RECIPE, max_chips=8)
        self.assertEqual(plans["capacity_floor"]["replicas"], 2)
        self.assertNotIn("a2", plans["plans"])

    def test_apply_sets(self):
        from framework.deploy import render  # noqa: PLC0415
        config = render.default_config(RECIPE, replicas=3)
        out = seed_arms.apply_sets(config, {"router.profile": "load-only",
                                            "engine.max_num_seqs": 256})
        self.assertEqual(out["router"]["profile"], "load-only")
        self.assertEqual(out["engine"]["max_num_seqs"], 256)
        self.assertEqual(out["topology"]["decode"]["replicas"], 3)


class EvaluateArmTest(_Quiet):

    def _protocol(self):
        return {
            "primary_metric": {"name": "goodput_at_slo_per_chip",
                               "unit": "req/s/chip", "direction": "maximize"},
            "replicate_count": 3,
            "noise_floor_pct": 5.0,
            "admissibility_gates": [
                {"name": "interactive.ttft_p90_ms", "metric": "ttft_p90_ms",
                 "comparator": "lte", "bound": 500, "same_execution": True}],
        }

    def test_ledger_then_score_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            ledger = tmp / "ledger.json"
            cfg = tmp / "cfg.json"
            for i, (value, ok) in enumerate([(0.70, True), (0.90, False),
                                             (0.83, True)]):
                rep = tmp / f"r{i}.json"
                rep.write_text(json.dumps(_report(f"e{i}", value, ok,
                                                  digest=f"d{i}")))
                cfg.write_text(json.dumps({"engine": {"max_num_seqs": 128 * (i + 1)}}))
                rc = evaluate_llmd_arm.main([
                    "ledger", "--ledger", str(ledger), "--report", str(rep),
                    "--config", str(cfg), "--budget", "8",
                    "--sets", json.dumps({"engine.max_num_seqs": 128 * (i + 1)})
                ])
                self.assertEqual(rc, 0)
            proto = tmp / "protocol.json"
            proto.write_text(json.dumps(self._protocol()))
            out = tmp / "arm/arm-result.json"
            rc = evaluate_llmd_arm.main([
                "score", "--ledger", str(ledger), "--protocol", str(proto),
                "--arm", "a1", "--baseline-value", "0.71", "--out", str(out)
            ])
            self.assertEqual(rc, 0)
            result = json.loads(out.read_text())
            chosen_cfg = json.loads((tmp / "arm/llmd-config.json").read_text())
        # The inadmissible 0.90 is skipped; 0.83 beats 0.71 by >5%.
        self.assertEqual(result["verdict"], "win")
        self.assertEqual(result["execution_id"], "e2")
        self.assertAlmostEqual(result["metric"]["value"], 0.83)
        self.assertEqual(chosen_cfg["engine"]["max_num_seqs"], 384)
        self.assertEqual(len(result["sweep_ledger"]), 3)
        self.assertTrue(all(g["execution_id"] == "e2" for g in result["gates"]))

    def test_tie_inside_noise_floor_and_invalid_without_telemetry(self):
        ledger = {"primary_metric": "goodput_at_slo_per_chip",
                  "direction": "maximize", "entries": [
                      evaluate_llmd_arm.ledger_entry(
                          _report("e1", 0.72), {}, "goodput_at_slo_per_chip")]}
        res = evaluate_llmd_arm.score_arm(ledger, self._protocol(), arm="a",
                                          baseline_value=0.71, config_path="")
        self.assertEqual(res["verdict"], "tie")
        bad = {"primary_metric": "goodput_at_slo_per_chip",
               "direction": "maximize", "entries": [
                   evaluate_llmd_arm.ledger_entry(
                       _report("e1", 0.95, telemetry_valid=False), {},
                       "goodput_at_slo_per_chip")]}
        res = evaluate_llmd_arm.score_arm(bad, self._protocol(), arm="a",
                                          baseline_value=0.71, config_path="")
        self.assertEqual(res["verdict"], "invalid")
        self.assertEqual(res["status"], "failed")

    def test_unreported_protocol_gate_blocks_win(self):
        proto = self._protocol()
        proto["admissibility_gates"].append({
            "name": "batch.error_rate", "metric": "error_rate",
            "comparator": "lte", "bound": 0.001, "same_execution": True})
        with tempfile.TemporaryDirectory() as tmp:
            rep = pathlib.Path(tmp) / "r.json"
            rep.write_text(json.dumps(_report("e9", 0.95)))
            entry = evaluate_llmd_arm.ledger_entry(
                json.loads(rep.read_text()), {}, "goodput_at_slo_per_chip")
            entry["report"] = str(rep)
            ledger = {"primary_metric": "goodput_at_slo_per_chip",
                      "direction": "maximize", "entries": [entry]}
            res = evaluate_llmd_arm.score_arm(ledger, proto, arm="a",
                                              baseline_value=0.71,
                                              config_path="")
        self.assertEqual(res["verdict"], "invalid")
        names = {g["name"]: g["passed"] for g in res["gates"]}
        self.assertFalse(names["batch.error_rate"])


if __name__ == "__main__":
    unittest.main()


class PlanArmsTest(_Quiet):
    """plan_arms proposes incumbent-relative arms, gated not pruned."""

    def _request(self, tmp: pathlib.Path, recipe, incumbent, *, decided=(),
                 open_arms=(), caps=None, profile_metrics=None):
        (tmp / "spec.json").write_text(json.dumps(SPEC))
        (tmp / "recipe.json").write_text(json.dumps(recipe))
        (tmp / "incumbent.json").write_text(json.dumps(incumbent))
        req = {
            "run_id": "r", "study_id": "s",
            "workload_spec_path": str(tmp / "spec.json"),
            "recipe_path": str(tmp / "recipe.json"),
            "incumbent_config_path": str(tmp / "incumbent.json"),
            "decided_arms": list(decided), "open_arms": list(open_arms),
            "max_new_arms": 4,
        }
        if caps is not None:
            (tmp / "caps.json").write_text(json.dumps(
                {"capabilities": caps}))
            req["capabilities_path"] = str(tmp / "caps.json")
        if profile_metrics is not None:
            (tmp / "metrics.json").write_text(json.dumps(
                {"metrics": {k: {"value": v}
                             for k, v in profile_metrics.items()}}))
            req["baseline_profile_path"] = str(tmp / "metrics.json")
        return req

    def test_managed_incumbent_proposes_reuse_first(self):
        from framework.deploy import render  # noqa: PLC0415
        incumbent = render.default_config(RECIPE, replicas=3)
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, RECIPE, incumbent,
                                profile_metrics={"prefix_hit_rate": 0.2})
            result = plan_arms.propose(req)
        self.assertEqual(result["status"], "planned")
        open_arms = [a for a in result["arms"] if a["status"] == "open"]
        self.assertTrue(open_arms)
        self.assertLessEqual(len(open_arms), 4)
        kinds = [result["plans"][a["arm"]]["arm_kind"] for a in open_arms]
        # Low measured hit rate on a shared-prefix workload raises precise
        # routing's headroom to the cap; the N-1 capacity arm (share 1.0)
        # still ranks above it by headroom x share, as the staged order says.
        self.assertEqual(kinds[0], "topology.replicas")
        self.assertIn("kv.precise_routing", kinds)
        precise = next(a for a in open_arms if result["plans"][a["arm"]][
            "arm_kind"] == "kv.precise_routing")
        self.assertAlmostEqual(precise["headroom"], 0.5)
        self.assertAlmostEqual(result["model"]["reuse_headroom"], 0.5)
        for arm in result["arms"]:
            plan = result["plans"][arm["arm"]]
            self.assertIn("sets", plan)
            self.assertIn("evidence", plan)
            self.assertTrue(arm["arm"].startswith("p"))

    def test_dedup_against_decided_and_open(self):
        from framework.deploy import render  # noqa: PLC0415
        incumbent = render.default_config(RECIPE, replicas=3)
        decided = [{"arm": "a5", "arm_kind": "router.profile",
                    "verdict": "loss", "sets": {"router.profile":
                                                "prefix-weighted"}}]
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, RECIPE, incumbent, decided=decided,
                                open_arms=["p1"])
            result = plan_arms.propose(req)
        all_sets = [json.dumps(p["sets"], sort_keys=True)
                    for p in result["plans"].values()]
        self.assertNotIn(json.dumps({"router.profile": "prefix-weighted"},
                                    sort_keys=True), all_sets)
        self.assertNotIn("p1", result["plans"])
        self.assertTrue(any("already proposed" in c
                            for c in result["model"]["closed"]))

    def test_load_only_win_lowers_reuse_headroom(self):
        from framework.deploy import render  # noqa: PLC0415
        incumbent = render.default_config(RECIPE, replicas=3)
        incumbent["router"]["profile"] = "load-only"
        decided = [{"arm": "a6", "arm_kind": "router.profile",
                    "verdict": "win", "sets": {"router.profile": "load-only"}}]
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, RECIPE, incumbent, decided=decided,
                                profile_metrics={"prefix_hit_rate": 0.2})
            result = plan_arms.propose(req)
        self.assertTrue(result["model"]["load_only_won"])
        self.assertAlmostEqual(result["model"]["reuse_headroom"], 0.15)

    def test_adopted_passthrough_gates_and_orders(self):
        from framework.deploy import render  # noqa: PLC0415
        recipe = dict(RECIPE)
        recipe["fleet"] = {"mode": "adopted", "selector": {"app": "ds"},
                           "service": "ds", "port": 8000, "replicas": 2}
        incumbent = render.default_config(recipe)
        caps = {"kv_events": True, "offload_connector": True,
                "pd_connector": False}
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, recipe, incumbent, caps=caps)
            result = plan_arms.propose(req)
        by_kind = {}
        for arm in result["arms"]:
            by_kind.setdefault(result["plans"][arm["arm"]]["arm_kind"],
                               []).append(arm)
        first = result["arms"][0]
        self.assertEqual(result["plans"][first["arm"]]["sets"],
                         {"router.profile": "sticky-until-saturated"})
        for arm in by_kind["router.profile"]:
            self.assertEqual(arm["status"], "open")
        # Capability beats fleet in the reason: the image lacks pd_connector.
        pd = by_kind["topology.pd"][0]
        self.assertEqual(pd["status"], "blocked")
        self.assertIn("blocked_on_engine_capability", pd["blocked_on"])
        tiered = by_kind["kv.tiered_host"][0]
        self.assertIn("blocked_on_adopted_fleet", tiered["blocked_on"])
        self.assertEqual(result["model"]["fleet_mode"], "adopted")
        # Blocked arms survive the max_new_arms cut.
        self.assertGreater(len(result["arms"]), 4)

    def test_main_writes_result_and_rejects_missing_keys(self):
        from framework.deploy import render  # noqa: PLC0415
        incumbent = render.default_config(RECIPE, replicas=3)
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, RECIPE, incumbent)
            (tmp / "req.json").write_text(json.dumps(req))
            rc = plan_arms.main(["--request", str(tmp / "req.json"),
                                 "--out", str(tmp / "out")])
            self.assertEqual(rc, 0)
            written = json.loads((tmp / "out/llmd-plan-result.json").read_text())
            self.assertEqual(written["status"], "planned")
            del req["incumbent_config_path"]
            (tmp / "bad.json").write_text(json.dumps(req))
            self.assertEqual(plan_arms.main(["--request", str(tmp / "bad.json"),
                                             "--out", str(tmp / "out2")]), 2)

