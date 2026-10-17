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
size_pd = _load("size_pd")
agentx_search = _load("agentx_search")
agentx_leaderboard = _load("agentx_leaderboard")
engine_escalation = _load("engine_escalation")
recipe_bump = _load("recipe_bump")
promote_release = _load("promote_release")
production_guard = _load("llmd_production_guard")
chip_budget_guard = _load("llmd_chip_budget_guard")


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

    def test_agentx_protocol_ranks_total_tokens_per_chip(self):
        from framework.deploy import workload  # noqa: PLC0415
        profile = workload.resolve_workload(json.loads(json.dumps(AGENTX_SPEC)))
        protocol = render_protocol.build_protocol(profile, AGENTX_RECIPE,
                                                  mode="auto", replicates=1,
                                                  noise_floor_pct=5,
                                                  max_rounds=6,
                                                  plateau_rounds=2)
        self.assertEqual(protocol["primary_metric"],
                         {"name": "total_tokens_per_s_per_chip",
                          "unit": "tok/s/chip", "direction": "maximize"})
        self.assertTrue(protocol["measurement_cell"]["shape"].startswith(
            "inferencex-agentx-mvp@c"))
        names = [g["name"] for g in protocol["admissibility_gates"]]
        self.assertIn("agentx.ttft_p50_ms", names)
        self.assertIn("agentx.error_rate", names)
        extras = render_protocol.build_study_extras(profile, AGENTX_RECIPE,
                                                    inner_sweep_budget=4)
        self.assertEqual(extras["secondary_metric"],
                         "output_tokens_per_s_per_chip")
        self.assertEqual(extras["workload"]["agentx"]["concurrency"],
                         profile["agentx"]["concurrency"])
        self.assertEqual(extras["benchmark_duration_s"],
                         profile["agentx"]["duration_s"])


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
        self.assertEqual(chosen_cfg["provenance"]["arm"], "a1")
        self.assertEqual(chosen_cfg["provenance"]["execution_id"], "e2")
        self.assertAlmostEqual(
            chosen_cfg["provenance"]["metric"]["value"], 0.83)
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

    def test_harness_change_makes_the_arm_invalid(self):
        rep = _report("e1", 0.95)
        rep["release"]["harness_digest"] = "h-arm"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "r.json"
            path.write_text(json.dumps(rep))
            entry = evaluate_llmd_arm.ledger_entry(
                rep, {}, "goodput_at_slo_per_chip")
            entry["report"] = str(path)
            self.assertEqual(entry["harness_digest"], "h-arm")
            ledger = {"primary_metric": "goodput_at_slo_per_chip",
                      "direction": "maximize", "entries": [entry]}
            same = evaluate_llmd_arm.score_arm(
                ledger, self._protocol(), arm="a", baseline_value=0.71,
                config_path="", baseline_harness="h-arm")
            other = evaluate_llmd_arm.score_arm(
                ledger, self._protocol(), arm="a", baseline_value=0.71,
                config_path="", baseline_harness="h-base")
            # No baseline digest known (an older report): no check.
            unknown = evaluate_llmd_arm.score_arm(
                ledger, self._protocol(), arm="a", baseline_value=0.71,
                config_path="")
        self.assertEqual(same["verdict"], "win")
        self.assertEqual(same["harness_digest"], "h-arm")
        self.assertEqual(other["verdict"], "invalid")
        self.assertIn("harness differs", other["summary"])
        self.assertEqual(unknown["verdict"], "win")

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

    def test_first_split_is_sized_and_a_split_walks_its_neighbours(self):
        from framework.deploy import render  # noqa: PLC0415
        recipe = dict(RECIPE, measured={"knee_rps": 45,
                                        "peak_prefill_throughput": 60000,
                                        "peak_decode_throughput": 6000})
        incumbent = render.default_config(recipe, replicas=3)
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            req = self._request(tmp, recipe, incumbent, caps={
                "pd_connector": True})
            result = plan_arms.propose(req)
            pd = [p for p in result["plans"].values()
                  if p["arm_kind"] == "topology.pd"]
            self.assertEqual(len(pd), 1)
            sets = pd[0]["sets"]
            self.assertEqual(sets["topology.mode"], "disaggregated")
            self.assertGreaterEqual(sets["topology.prefill"]["replicas"], 1)
            self.assertIn("topology.decode.replicas", sets)
            self.assertEqual(pd[0]["prerequisites"]["sizing_provenance"],
                             "modeled")
            # Now the incumbent is a 2P:2D split whose decode pool queues:
            # the planner proposes shape neighbours that move the decode
            # axis, and no aggregated capacity arms.
            split = render.default_config(recipe, replicas=2)
            split["topology"]["mode"] = "disaggregated"
            split["topology"]["prefill"] = {"replicas": 2, "tp": 8}
            (tmp / "inc2.json").write_text(json.dumps(split))
            (tmp / "prof.json").write_text(json.dumps({"metrics": {}, "roles": {
                "decode": {"requests_waiting_mean": {"value": 3.0}},
                "prefill": {"requests_waiting_mean": {"value": 0.0}}}}))
            req2 = self._request(tmp, recipe, split, caps={
                "pd_connector": True})
            req2["incumbent_config_path"] = str(tmp / "inc2.json")
            req2["incumbent_profile_path"] = str(tmp / "prof.json")
            req2["max_new_arms"] = 8
            result2 = plan_arms.propose(req2)
        shapes = [p for p in result2["plans"].values()
                  if p["arm_kind"] == "topology.pd_shape"]
        self.assertTrue(shapes)
        self.assertEqual(shapes[0]["sets"], {"topology.prefill.replicas": 1,
                                             "topology.decode.replicas": 3})
        self.assertEqual(result2["model"]["attribution"]["role"], "decode")
        self.assertFalse([p for p in result2["plans"].values()
                          if p["arm_kind"] == "topology.replicas"])
        for p in shapes:
            self.assertIn("topology.decode.engine.max_num_seqs",
                          p["sweep_knobs"])

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


# -- M4: promotion and the two guards ------------------------------------------------


def _production_release(tmp: pathlib.Path, *, profile: str = "sticky-until-saturated",
                        production: bool = True) -> pathlib.Path:
    """A release directory the way `deploy render --production` writes it.

    Rendered through the framework rather than hand-written so the bundle
    test reads the same `release.json`/`files` layout the runtime produces.
    """
    import os  # noqa: PLC0415
    from framework.deploy import pins, render, workload  # noqa: PLC0415
    os.environ.setdefault("TPU_AGENT_SUITE_HOME", str(tmp / "home"))
    cfg = render.default_config(RECIPE, replicas=2)
    cfg["router"]["profile"] = profile
    if profile != render.PASSTHROUGH_PROFILE:
        cfg["router"]["epp_replicas"] = 2
        cfg["router"]["epp_ha_mode"] = "active-passive"
    env = {"namespace": "prod", "hf_token_secret": "hf-token",
           "inference_perf_image": pins.INFERENCE_PERF_IMAGE,
           "gateway_mode": "standalone", "exposure": "cluster-ip"}
    out = tmp / "release"
    render.render_release(name="acme-prod", config=cfg, recipe=RECIPE,
                          profile=workload.resolve_workload(json.loads(
                              json.dumps(SPEC))),
                          env=env, out_dir=out, production=production,
                          now="2026-10-01T00:00:00+00:00")
    return out


def _promote_request(tmp: pathlib.Path, incumbent: dict, **overrides) -> dict:
    (tmp / "incumbent.json").write_text(json.dumps(incumbent))
    (tmp / "spec.json").write_text(json.dumps(SPEC))
    (tmp / "recipe.json").write_text(json.dumps(RECIPE))
    req = {
        "run_id": "run-1",
        "study_id": "acme-chat-study",
        "incumbent_config_path": str(tmp / "incumbent.json"),
        "recipe_path": str(tmp / "recipe.json"),
        "workload_spec_path": str(tmp / "spec.json"),
        "production_namespace": "prod",
        "promotion_approved": True,
    }
    req.update(overrides)
    (tmp / "promote-request.json").write_text(json.dumps(req))
    return req


class PromoteReleaseTest(_Quiet):

    def _incumbent(self, profile="sticky-until-saturated"):
        from framework.deploy import render  # noqa: PLC0415
        cfg = render.default_config(RECIPE, replicas=2)
        cfg["router"]["profile"] = profile
        return cfg

    def test_prepare_refuses_without_explicit_approval(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            for value in (False, "yes", 1, None):
                _promote_request(tmp, self._incumbent(), promotion_approved=value)
                rc = promote_release.main([
                    "prepare", "--request", str(tmp / "promote-request.json"),
                    "--out", str(tmp / "out")])
                self.assertEqual(rc, promote_release.EXIT_REFUSED, value)
                result = json.loads(
                    (tmp / "out" / promote_release.RESULT_FILE).read_text())
                self.assertEqual(result["status"], "refused")
                self.assertFalse(
                    (tmp / "out" / promote_release.PRODUCTION_CONFIG_FILE)
                    .exists())

    def test_prepare_refuses_when_production_already_exists(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            _promote_request(tmp, self._incumbent(),
                             existing_production_release="old-prod")
            rc = promote_release.main([
                "prepare", "--request", str(tmp / "promote-request.json"),
                "--out", str(tmp / "out")])
            self.assertEqual(rc, promote_release.EXIT_REFUSED)
            result = json.loads(
                (tmp / "out" / promote_release.RESULT_FILE).read_text())
        self.assertIn("old-prod", result["summary"])

    def test_prepare_adds_only_epp_ha_to_a_router_incumbent(self):
        incumbent = self._incumbent()
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            _promote_request(tmp, incumbent, epp_replicas=3)
            rc = promote_release.main([
                "prepare", "--request", str(tmp / "promote-request.json"),
                "--out", str(tmp / "out")])
            self.assertEqual(rc, 0)
            config = json.loads(
                (tmp / "out" / promote_release.PRODUCTION_CONFIG_FILE)
                .read_text())
            self.assertFalse((tmp / "out" / promote_release.RESULT_FILE)
                             .exists())
        self.assertEqual(config["router"]["epp_replicas"], 3)
        self.assertEqual(config["router"]["epp_ha_mode"], "active-passive")
        promoted = config["provenance"]["promoted_from"]
        self.assertEqual(promoted["study_id"], "acme-chat-study")
        self.assertEqual(promoted["production_changes"],
                         ["router.epp_replicas", "router.epp_ha_mode"])
        # Everything else is the incumbent byte-for-byte.
        for key in ("topology", "kv", "engine"):
            self.assertEqual(config[key], incumbent[key])
        self.assertEqual(config["router"]["profile"],
                         incumbent["router"]["profile"])

    def test_prepare_leaves_a_passthrough_incumbent_alone(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            _promote_request(tmp, self._incumbent("passthrough"))
            rc = promote_release.main([
                "prepare", "--request", str(tmp / "promote-request.json"),
                "--out", str(tmp / "out")])
            self.assertEqual(rc, 0)
            config = json.loads(
                (tmp / "out" / promote_release.PRODUCTION_CONFIG_FILE)
                .read_text())
        self.assertEqual(config["router"]["epp_replicas"], 1)
        self.assertNotIn("epp_ha_mode", config["router"])
        self.assertEqual(
            config["provenance"]["promoted_from"]["production_changes"], [])

    def _drill_row(self, passed=True):
        return {"drill": "epp-pod-kill", "release": "acme-prod",
                "killed": "acme-prod-epp-abc", "recovered": passed,
                "recovery_s": 12.5, "front_door_ok": passed, "passed": passed}

    def _bundle(self, tmp, *, acceptance, baseline, drill=None,
                profile="sticky-until-saturated", study=None):
        release_dir = _production_release(tmp, profile=profile)
        incumbent = json.loads((release_dir / "llmd-config.json").read_text())
        _promote_request(tmp, incumbent)
        (tmp / "acceptance.json").write_text(json.dumps(acceptance))
        (tmp / "baseline.json").write_text(json.dumps(baseline))
        argv = ["bundle", "--request", str(tmp / "promote-request.json"),
                "--out", str(tmp / "bundle"), "--release-dir", str(release_dir),
                "--acceptance", str(tmp / "acceptance.json"),
                "--baseline-report", str(tmp / "baseline.json")]
        if drill is not None:
            (tmp / "drill.json").write_text(json.dumps(drill))
            argv += ["--drill", str(tmp / "drill.json")]
        if study is not None:
            (tmp / "study.json").write_text(json.dumps(study))
            argv += ["--study-state", str(tmp / "study.json")]
        rc = promote_release.main(argv)
        result = json.loads(
            (tmp / "bundle" / promote_release.RESULT_FILE).read_text())
        return rc, result

    def test_bundle_writes_the_handoff_and_accepts_within_noise(self):
        from framework import utils  # noqa: PLC0415
        study = {
            "study_id": "acme-chat-study", "status": "converged",
            "protocol": {"primary_metric": {
                "name": "goodput_at_slo_per_chip", "direction": "maximize",
                "unit": "rps/chip"}, "noise_floor_pct": 5,
                "replicate_count": 3},
            "baseline": {"metric": {"value": 2.5, "unit": "rps/chip"},
                         "execution_id": "e0"},
            "arms": [
                {"arm": "a1", "kind": "routing-policy", "status": "decided",
                 "claim": "sticky routing | keeps prefix", "result": {
                     "verdict": "win", "execution_id": "e1",
                     "metric": {"value": 2.6, "delta_pct": 4.0}}},
                {"arm": "a2", "kind": "fleet-topology", "status": "blocked",
                 "claim": "P/D", "blocked_on": "blocked_on_adopted_fleet"},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, result = self._bundle(
                tmp, acceptance=_report("e2", 2.55), baseline=_report("e1", 2.6),
                drill=self._drill_row(), study=study)
            self.assertEqual(rc, 0, result)
            bundle = tmp / "bundle"
            for rel in ("llmd-config.json", "serving-recipe.json",
                        "search-ledger.md", "router.values.yaml",
                        "epp-config.yaml", "inference-objectives.yaml",
                        "exposure.yaml", "workload/workload-spec.json",
                        "workload/workload_profile.json",
                        "observability/podmonitoring.yaml",
                        "observability/README.md", "autoscaling/README.md",
                        "acceptance-report.md", "RUNBOOK.md", "release.json",
                        "release/release.json"):
                self.assertTrue((bundle / rel).is_file(), rel)
            ledger = (bundle / "search-ledger.md").read_text()
            report = (bundle / "acceptance-report.md").read_text()
            runbook = (bundle / "RUNBOOK.md").read_text()
            manifest = json.loads((bundle / "release.json").read_text())
            schema = json.loads(
                (_HERE.parents[1] / "schemas" / "llmd-promote-result.json")
                .read_text())
        self.assertEqual(result["status"], "promoted")
        self.assertEqual(utils.validate_instance(result, schema), [])
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["endpoint"],
                         "http://acme-prod-ingress.prod.svc.cluster.local:80")
        self.assertIn("a1 **(incumbent)**", ledger)
        self.assertIn("Blocked arms", ledger)
        self.assertIn("ACCEPTED", report)
        self.assertIn("-1.9%", report)
        self.assertIn("UNVERIFIED: autoscaling", report)
        self.assertIn("deploy teardown", runbook)
        # The runbook's objective names match the chart's InferenceObjectives.
        from framework.deploy import render  # noqa: PLC0415
        for tenant in ("interactive", "batch"):
            self.assertIn("x-llm-d-inference-objective: "
                          + render.objective_name("acme-prod", tenant), runbook)
        self.assertTrue(manifest["accepted"])
        self.assertEqual(manifest["router"]["epp_ha"]["replicas"], 2)
        self.assertNotIn("release/release.json", manifest["files"])

    def test_bundle_fails_acceptance_on_regression_or_failed_drill(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, result = self._bundle(
                tmp, acceptance=_report("e2", 2.2), baseline=_report("e1", 2.6),
                drill=self._drill_row())
        self.assertEqual(rc, promote_release.EXIT_ACCEPTANCE_FAILED)
        self.assertEqual(result["status"], "acceptance_failed")
        failed = [r["check"] for r in result["acceptance"]["gates"]
                  if not r["passed"]]
        self.assertEqual(failed,
                         ["goodput_at_slo_per_chip vs incumbent measurement"])
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, result = self._bundle(
                tmp, acceptance=_report("e2", 2.6), baseline=_report("e1", 2.6),
                drill=self._drill_row(passed=False))
        self.assertEqual(result["status"], "acceptance_failed")
        self.assertIn("drill epp-pod-kill",
                      [r["check"] for r in result["acceptance"]["gates"]
                       if not r["passed"]])

    def test_bundle_requires_a_drill_when_a_router_is_installed(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, result = self._bundle(
                tmp, acceptance=_report("e2", 2.6), baseline=_report("e1", 2.6))
        self.assertEqual(result["status"], "acceptance_failed")
        self.assertIn({"check": "drill epp-pod-kill", "passed": False,
                       "detail": "not run"}, result["acceptance"]["gates"])

    def test_bundle_of_passthrough_skips_router_files_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, result = self._bundle(
                tmp, acceptance=_report("e2", 2.6), baseline=_report("e1", 2.6),
                profile="passthrough")
            self.assertEqual(rc, 0, result)
            self.assertFalse((tmp / "bundle" / "router.values.yaml").exists())
            self.assertFalse((tmp / "bundle" / "exposure.yaml").exists())
        self.assertEqual(result["status"], "promoted")
        self.assertTrue(any("passthrough" in u for u in result["unverified"]))

    def test_bundle_refuses_a_study_render(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            release_dir = _production_release(tmp, production=False)
            incumbent = json.loads(
                (release_dir / "llmd-config.json").read_text())
            _promote_request(tmp, incumbent)
            with self.assertRaises(SystemExit):
                promote_release.main([
                    "bundle", "--request", str(tmp / "promote-request.json"),
                    "--out", str(tmp / "bundle"),
                    "--release-dir", str(release_dir)])

    def test_bundle_minimize_cost_metric_acceptance(self):
        def _cost_report(exec_id, cost_val):
            rep = _report(exec_id, 2.5)
            rep["primary_metric"] = "cost_per_1M_tokens_at_slo"
            rep["metrics"]["cost_per_1M_tokens_at_slo"] = {"value": cost_val}
            return rep

        # Lower cost (-10%) is an improvement under minimize and must pass;
        # higher cost (+20%) exceeds the +5% ceiling and must fail.
        passed = promote_release.judge_acceptance(
            _cost_report("e2", 0.90), _cost_report("e1", 1.00),
            [self._drill_row()], noise_floor_pct=5.0, router_installed=True)
        self.assertTrue(passed["passed"], passed)
        self.assertTrue(any("ceiling +5.0%" in r["detail"]
                            for r in passed["rows"]))

        failed = promote_release.judge_acceptance(
            _cost_report("e3", 1.20), _cost_report("e1", 1.00),
            [self._drill_row()], noise_floor_pct=5.0, router_installed=True)
        self.assertFalse(failed["passed"], failed)
        self.assertEqual(
            [r["check"] for r in failed["rows"] if not r["passed"]],
            ["cost_per_1M_tokens_at_slo vs incumbent measurement"])

    def test_bundle_prefers_provenance_arm_and_falls_back_to_baseline(self):
        study = {
            "study_id": "acme-chat-study", "status": "converged",
            "protocol": {"primary_metric": {
                "name": "goodput_at_slo_per_chip", "direction": "maximize",
                "unit": "rps/chip"}, "noise_floor_pct": 5,
                "replicate_count": 3},
            "baseline": {"metric": {"value": 2.5, "unit": "rps/chip"},
                         "execution_id": "e0"},
            "arms": [
                {"arm": "baseline", "kind": "baseline", "status": "decided",
                 "claim": "control", "result": {
                     "verdict": "tie", "execution_id": "e0",
                     "metric": {"value": 2.5}}},
                {"arm": "a0", "kind": "routing-policy", "status": "decided",
                 "claim": "sticky", "result": {
                     "verdict": "tie", "execution_id": "e1",
                     "metric": {"value": 2.52, "delta_pct": 0.8}}},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rc, _ = self._bundle(
                tmp, acceptance=_report("e2", 2.5), baseline=_report("e0", 2.5),
                drill=self._drill_row(), study=study)
            self.assertEqual(rc, 0)
            ledger = (tmp / "bundle" / "search-ledger.md").read_text()
            self.assertIn("baseline **(incumbent)**", ledger)

        # When provenance.arm is explicitly stamped on the incumbent config,
        # it takes precedence over scanning study arms.
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            release_dir = _production_release(tmp)
            cfg_path = release_dir / "llmd-config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["provenance"] = {"arm": "a0", "execution_id": "e1"}
            cfg_path.write_text(json.dumps(cfg))
            _promote_request(tmp, cfg)
            (tmp / "acceptance.json").write_text(
                json.dumps(_report("e2", 2.5)))
            (tmp / "baseline.json").write_text(
                json.dumps(_report("e1", 2.52)))
            (tmp / "drill.json").write_text(
                json.dumps(self._drill_row()))
            (tmp / "study.json").write_text(json.dumps(study))
            rc = promote_release.main([
                "bundle", "--request", str(tmp / "promote-request.json"),
                "--out", str(tmp / "bundle"),
                "--release-dir", str(release_dir),
                "--acceptance", str(tmp / "acceptance.json"),
                "--baseline-report", str(tmp / "baseline.json"),
                "--drill", str(tmp / "drill.json"),
                "--study-state", str(tmp / "study.json")])
            self.assertEqual(rc, 0)
            ledger = (tmp / "bundle" / "search-ledger.md").read_text()
            self.assertIn("a0 **(incumbent)**", ledger)


class SizePdTest(_Quiet):
    """The ratio model opens the xP:yD lattice and names the bound pool."""

    def _profile(self):
        from framework.deploy import workload  # noqa: PLC0415
        return workload.resolve_workload(json.loads(json.dumps(SPEC)))

    def test_opening_from_measured_capacities(self):
        from framework.deploy import render  # noqa: PLC0415
        recipe = dict(RECIPE, measured={"knee_rps": 45,
                                        "peak_prefill_throughput": 60000,
                                        "peak_decode_throughput": 6000})
        incumbent = render.default_config(recipe, replicas=3)
        profile = self._profile()
        opening = size_pd.opening_shape(profile, recipe, incumbent,
                                        hit_rate=0.5, max_chips=64)
        dem = opening["demand"]
        isl = profile["shape"]["input_sequence_length"]["mean"]
        osl = profile["shape"]["output_sequence_length"]["mean"]
        self.assertAlmostEqual(dem["prefill_tokens_per_s"], 120 * isl * 0.5)
        self.assertAlmostEqual(dem["decode_tokens_per_s"], 120 * osl)
        import math  # noqa: PLC0415
        self.assertEqual(opening["prefill_replicas"], max(1, math.ceil(
            dem["prefill_tokens_per_s"] / (60000 * size_pd.HEADROOM))))
        self.assertEqual(opening["decode_replicas"], max(1, math.ceil(
            dem["decode_tokens_per_s"] / (6000 * size_pd.HEADROOM))))
        self.assertEqual(opening["provenance"], "modeled")
        self.assertLessEqual(opening["chips"], 64)

    def test_opening_without_capacities_is_a_labelled_heuristic(self):
        from framework.deploy import render  # noqa: PLC0415
        incumbent = render.default_config(RECIPE, replicas=4)
        opening = size_pd.opening_shape(self._profile(), RECIPE, incumbent,
                                        hit_rate=None, max_chips=64)
        self.assertEqual(opening["provenance"], "heuristic")
        self.assertEqual(opening["prefill_replicas"]
                         + opening["decode_replicas"], 4)
        self.assertGreaterEqual(opening["prefill_replicas"], 1)
        self.assertGreaterEqual(opening["decode_replicas"], 1)
        self.assertEqual(opening["demand"]["prefix_hit_source"], "assumed")

    def test_budget_shrinks_the_opening_shape(self):
        from framework.deploy import render  # noqa: PLC0415
        recipe = dict(RECIPE, measured={"peak_prefill_throughput": 1000,
                                        "peak_decode_throughput": 100})
        incumbent = render.default_config(recipe, replicas=1)
        opening = size_pd.opening_shape(self._profile(), recipe, incumbent,
                                        hit_rate=0.0, max_chips=16)
        self.assertEqual(opening["chips"], 16)
        self.assertEqual(opening["bound_by"], "max_chips")

    def test_attribution_reads_gates_then_queues(self):
        prof = {"metrics": {}, "roles": {
            "prefill": {"requests_waiting_mean": {"value": 0.1},
                        "kv_util_max": {"value": 0.3}},
            "decode": {"requests_waiting_mean": {"value": 2.0},
                       "kv_util_max": {"value": 0.95}}}}
        report = {"tenants": {"chat": {"slo": {
            "ttft_p99_ms": {"pass": True}, "tpot_p99_ms": {"pass": False}}}}}
        att = size_pd.attribute(prof, report)
        self.assertEqual(att["role"], "decode")
        self.assertEqual(att["kind"], "memory-capacity")
        self.assertIn("chat.tpot_p99_ms", att["failed_gates"])
        # No failed gate: the queueing pool is named; no pressure: none.
        att2 = size_pd.attribute(prof, None)
        self.assertEqual(att2["role"], "decode")
        self.assertEqual(size_pd.attribute({"roles": {}}, None)["role"],
                         "none")
        # Client-limited: the load generator is named, no pool, no shape.
        limited = dict(report, client_limited=["chat/r0", "chat/r1"])
        att3 = size_pd.attribute(prof, limited)
        self.assertEqual((att3["role"], att3["kind"]), ("loadgen", "none"))
        self.assertEqual(size_pd.neighbours(1, 2, "loadgen", chips_per=4,
                                            max_chips=24), [])

    def test_neighbours_move_the_bound_axis_first(self):
        nb = size_pd.neighbours(2, 2, "prefill", chips_per=4, max_chips=16)
        self.assertEqual([(n["prefill_replicas"], n["decode_replicas"])
                          for n in nb], [(3, 1), (2, 1)])  # (3,2) over budget
        nb = size_pd.neighbours(2, 2, "decode", chips_per=4, max_chips=24)
        self.assertEqual([(n["prefill_replicas"], n["decode_replicas"])
                          for n in nb], [(1, 3), (2, 3), (1, 2)])
        nb = size_pd.neighbours(2, 3, "none", chips_per=4, max_chips=None)
        self.assertEqual([(n["prefill_replicas"], n["decode_replicas"])
                          for n in nb], [(1, 3), (2, 2), (1, 2)])
        self.assertEqual(size_pd.neighbours(1, 1, "none", chips_per=4,
                                            max_chips=None), [])

    def test_main_writes_the_sizing_file(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            from framework.deploy import render  # noqa: PLC0415
            (tmp / "spec.json").write_text(json.dumps(SPEC))
            (tmp / "recipe.json").write_text(json.dumps(RECIPE))
            cfg = render.default_config(RECIPE, replicas=2)
            cfg["topology"]["mode"] = "disaggregated"
            cfg["topology"]["prefill"] = {"replicas": 1}
            (tmp / "inc.json").write_text(json.dumps(cfg))
            rc = size_pd.main(["--workload", str(tmp / "spec.json"),
                               "--recipe", str(tmp / "recipe.json"),
                               "--incumbent", str(tmp / "inc.json"),
                               "--max-chips", "64",
                               "--out", str(tmp / "out")])
            doc = json.loads((tmp / "out" / size_pd.SIZING_FILE).read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(doc["incumbent_shape"],
                         {"prefill_replicas": 1, "decode_replicas": 2})
        self.assertTrue(doc["neighbours"])


def _llmd_state(arms, baseline_value=2.5, study_id="acme-chat"):
    return {"study_id": study_id, "status": "open", "round": 4,
            "protocol": {"primary_metric": {"name": "goodput_at_slo_per_chip",
                                            "direction": "maximize",
                                            "unit": "req/s/chip"},
                         "replicate_count": 3, "noise_floor_pct": 5.0,
                         "mode": "approve",
                         "stopping_rule": {"max_rounds": 12,
                                           "plateau_rounds": 3}},
            "baseline": {"metric": {"name": "goodput_at_slo_per_chip",
                                    "value": baseline_value},
                         "execution_id": "e0"},
            "arms": arms}


def _failing_report(gate="tpot_p99_ms"):
    rep = _report("e9", 2.4, admissible=False)
    rep["tenants"] = {"interactive": {"slo": {
        "ttft_p90_ms": {"pass": gate != "ttft_p90_ms", "measured_ms": 300},
        gate: {"pass": False, "measured_ms": 80}}, "error_rate": 0.0,
        "requests_per_sec": 84.0}}
    return rep


class EngineEscalationTest(_Quiet):
    """The seam between the fleet search and the engine study."""

    def _inputs(self, tmp, *, state, report, metrics=None, llmd_study=None,
                incumbent=None, arm_plans=None):
        from framework.deploy import render  # noqa: PLC0415
        (tmp / "spec.json").write_text(json.dumps(SPEC))
        (tmp / "recipe.json").write_text(json.dumps(RECIPE))
        inc = incumbent or render.default_config(RECIPE, replicas=3)
        (tmp / "inc.json").write_text(json.dumps(inc))
        (tmp / "state.json").write_text(json.dumps(state))
        (tmp / "report.json").write_text(json.dumps(report))
        argv = ["prepare", "--study-state", str(tmp / "state.json"),
                "--workload", str(tmp / "spec.json"),
                "--recipe", str(tmp / "recipe.json"),
                "--incumbent", str(tmp / "inc.json"),
                "--report", str(tmp / "report.json"),
                "--run-id", "run-1", "--out", str(tmp / "esc")]
        if metrics is not None:
            (tmp / "metrics.json").write_text(json.dumps(metrics))
            argv += ["--profile", str(tmp / "metrics.json")]
        if llmd_study is not None:
            (tmp / "study.json").write_text(json.dumps(llmd_study))
            argv += ["--llmd-study", str(tmp / "study.json")]
        if arm_plans is not None:
            (tmp / "plans.json").write_text(json.dumps(arm_plans))
            argv += ["--arm-plans", str(tmp / "plans.json")]
        return argv

    def _run(self, argv):
        import argparse  # noqa: PLC0415
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        p = sub.add_parser("prepare")
        for flag in ("--study-state", "--llmd-study", "--protocol",
                     "--workload", "--recipe", "--incumbent", "--report",
                     "--profile", "--arm-plans", "--plateau", "--study-id",
                     "--run-id", "--out"):
            p.add_argument(flag, default="no" if flag == "--plateau" else None)
        return engine_escalation.prepare(parser.parse_args(argv))

    def test_no_escalation_while_arms_are_open_or_objective_met(self):
        arms = [{"arm": "a5", "kind": "routing-policy", "status": "open",
                 "claim": "x", "since_round": 1}]
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            res = self._run(self._inputs(tmp, state=_llmd_state(arms),
                                         report=_failing_report()))
            self.assertEqual(res["status"], "no_escalation")
            self.assertIn("open arms", " ".join(res["decision"]["blockers"]))
            # Objective met: nothing to escalate even at a plateau.
            res = self._run(self._inputs(tmp, state=_llmd_state([]),
                                         report=_report("e1", 2.6)))
            self.assertEqual(res["status"], "no_escalation")
            self.assertTrue(any("objective met" in b
                                for b in res["decision"]["blockers"]))
            # Loop disabled.
            res = self._run(self._inputs(
                tmp, state=_llmd_state([]), report=_failing_report(),
                llmd_study={"engine_loop": {"enabled": False}}))
            self.assertTrue(any("disabled" in b
                                for b in res["decision"]["blockers"]))
            # Client-limited incumbent: the load generator is the finding,
            # not the engine.
            res = self._run(self._inputs(
                tmp, state=_llmd_state([]),
                report=dict(_failing_report(), client_limited=["chat/r0"])))
            self.assertEqual(res["status"], "no_escalation")
            self.assertEqual(res["decision"]["role"], "loadgen")
            self.assertTrue(any("client-limited" in b
                                for b in res["decision"]["blockers"]))

    def test_escalates_the_bound_role_with_protocol_and_seeds(self):
        from framework import utils  # noqa: PLC0415
        metrics = {"metrics": {"prefix_hit_rate": {"value": 0.7},
                               "requests_running_mean": {"value": 60}},
                   "roles": {"decode": {"requests_waiting_mean": {"value": 4},
                                        "requests_running_mean": {"value": 96},
                                        "kv_util_max": {"value": 0.5}},
                             "prefill": {"requests_waiting_mean": {"value": 0}}}}
        from framework.deploy import render  # noqa: PLC0415
        inc = render.default_config(RECIPE, replicas=2)
        inc["topology"]["mode"] = "disaggregated"
        inc["topology"]["prefill"] = {"replicas": 1, "tp": 8}
        inc["topology"]["decode"]["engine"] = {"max_num_seqs": 128}
        plans = {"plans": {"a0": {"arm_kind": "topology.pd_shape",
                                  "sweep_knobs": [
                                      "topology.decode.engine.max_num_seqs"]}}}
        arms = [{"arm": "a0", "kind": "fleet-topology", "status": "decided",
                 "claim": "x", "since_round": 1,
                 "result": {"verdict": "tie", "execution_id": "e9",
                            "metric": {"name": "goodput_at_slo_per_chip",
                                       "value": 2.4, "replicates": 3,
                                       "delta_pct": -4.0,
                                       "provenance": "measured"}}}]
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            res = self._run(self._inputs(tmp, state=_llmd_state(arms),
                                         report=_failing_report(),
                                         metrics=metrics, incumbent=inc,
                                         arm_plans=plans))
            self.assertEqual(res["status"], "escalate", res)
            esc = json.loads((tmp / "esc" / engine_escalation.ESCALATION_FILE)
                             .read_text())
            proto = json.loads((tmp / "esc" / engine_escalation.PROTOCOL_FILE)
                               .read_text())
            seeds = json.loads((tmp / "esc" / engine_escalation.HYPOTHESES_FILE)
                               .read_text())
            micro = json.loads(pathlib.Path(esc["microbench_path"]).read_text())
            esc_schema = json.loads(
                (_HERE.parents[1] / "schemas" / "engine-escalation.json")
                .read_text())
            proto_schema = json.loads(
                (_HERE.parents[2] / "framework" / "schemas" /
                 "study-protocol.json").read_text())
        self.assertEqual(utils.validate_instance(esc, esc_schema), [])
        self.assertEqual(utils.validate_instance(proto, proto_schema), [])
        self.assertEqual(esc["role"], "decode")
        self.assertEqual(esc["kind"], "scheduler-batching")
        self.assertEqual(esc["engine_study_id"], "acme-chat-engine-decode-1")
        self.assertEqual(proto["primary_metric"]["name"],
                         "generation_tokens_per_s")
        self.assertEqual(proto["measurement_cell"]["concurrency"], 96)
        gate_metrics = {g["metric"] for g in proto["admissibility_gates"]}
        self.assertIn("tpot_p90_ms", gate_metrics)
        self.assertNotIn("ttft_p90_ms", gate_metrics)
        self.assertEqual(proto["mode"], "approve")
        self.assertEqual(proto["replicate_count"], 3)
        # Seeds carry serving deltas; the knob the fleet swept is excluded.
        self.assertEqual(esc["excluded_knobs"], ["--max-num-seqs"])
        ids = [h["id"] for h in seeds["hypotheses"]]
        self.assertNotIn("e-decode-max-seqs", ids)
        self.assertIn("e-decode-kv-dtype", ids)
        for h in seeds["hypotheses"]:
            self.assertIn("serving", h["target_parameters"])
            self.assertTrue(h["evidence"])
        self.assertEqual(micro["load"]["stages"][0]["concurrency_level"], 96)
        self.assertEqual(micro["data"]["shared_prefix"]["output_len"],
                         int(esc["operating_point"]["osl_mean"]))

    def test_target_value_counts_as_unmet_and_role_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            res = self._run(self._inputs(
                tmp, state=_llmd_state([], baseline_value=2.0),
                report=_report("e1", 2.0),
                llmd_study={"objective_target": {"value": 3.0,
                                                 "direction": "maximize"}}))
            self.assertEqual(res["status"], "escalate", res)
            self.assertEqual(res["decision"]["role"], "aggregated")
            # A second escalation on an exhausted role is refused.
            res2 = self._run(self._inputs(
                tmp, state=_llmd_state([], baseline_value=2.0),
                report=_report("e1", 2.0),
                llmd_study={"objective_target": {"value": 3.0},
                            "engine_loop": {"exhausted_roles": ["aggregated"],
                                            "escalations": [{"n": 1}]}}))
            self.assertEqual(res2["status"], "no_escalation")


class RecipeBumpTest(_Quiet):
    """An engine study's win comes back as a versioned recipe."""

    def _engine_state(self, verdict="win", delta=12.0, arm="e-decode-max-seqs",
                      metric_name="generation_tokens_per_s", value=9000.0):
        return {"study_id": "acme-chat-engine-decode-1", "status": "closed",
                "arms": [{"arm": arm, "kind": "scheduler-batching",
                          "status": "decided", "claim": "x", "since_round": 1,
                          "result": {"verdict": verdict, "execution_id": "x1",
                                     "metric": {"name": metric_name,
                                                "value": value,
                                                "delta_pct": delta,
                                                "replicates": 3,
                                                "provenance": "measured"}}},
                         {"arm": "e-other", "kind": "kernel-block",
                          "status": "decided", "claim": "y", "since_round": 2,
                          "result": {"verdict": "win", "execution_id": "x2",
                                     "metric": {"name": metric_name,
                                                "value": 8000.0,
                                                "delta_pct": 3.0,
                                                "replicates": 3,
                                                "provenance": "measured"}}}]}

    def _bank(self, arm="e-decode-max-seqs"):
        return {"arms": {arm: {"hypothesis_id": arm, "target_parameters": {
            "serving": {"args": {"--max-num-seqs": 256,
                                 "--max-model-len": None},
                        "env": {"VLLM_FOO": "1"}}}}}}

    def _run(self, tmp, state, bank, escalation=None):
        import argparse  # noqa: PLC0415
        (tmp / "serving-recipe.json").write_text(json.dumps(RECIPE))
        (tmp / "estate.json").write_text(json.dumps(state))
        (tmp / "bank.json").write_text(json.dumps(bank))
        (tmp / "esc.json").write_text(json.dumps(
            escalation or {"study_id": "acme-chat", "role": "decode",
                           "escalation": 1}))
        ns = argparse.Namespace(recipe=str(tmp / "serving-recipe.json"),
                                engine_study_state=str(tmp / "estate.json"),
                                hypothesis_bank=str(tmp / "bank.json"),
                                escalation=str(tmp / "esc.json"),
                                out=str(tmp / "bump"))
        return recipe_bump.bump(ns)

    def test_bumps_the_recipe_from_the_best_win(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            doc = self._run(tmp, self._engine_state(), self._bank())
            new = json.loads(pathlib.Path(doc["recipe_after"]).read_text())
            amend = json.loads((tmp / "bump" / recipe_bump.AMENDMENT_FILE)
                               .read_text())
        self.assertEqual(doc["status"], "bumped")
        self.assertEqual(doc["winning_arm"], "e-decode-max-seqs")
        self.assertTrue(doc["recipe_after"].endswith("serving-recipe.v2.json"))
        # --max-model-len removed (null), --max-num-seqs set, env merged.
        self.assertIn("--max-num-seqs=256", new["args"])
        self.assertFalse(any(a.startswith("--max-model-len")
                             for a in new["args"]))
        self.assertEqual(new["env"]["VLLM_FOO"], "1")
        self.assertEqual(new["measured"]["peak_decode_throughput"], 9000.0)
        self.assertEqual(new["measured"]["knee_rps"], 45)
        self.assertEqual(new["source"]["engine_study_id"],
                         "acme-chat-engine-decode-1")
        self.assertEqual(amend["fingerprint"]["commit_sha"],
                         doc["recipe_digest_after"])
        # The runtime refuses a new baseline under measured arms, so the
        # amendment names the successor study the search continues in.
        self.assertEqual(amend["apply"], "successor_study")
        self.assertEqual(amend["predecessor_study_id"], "acme-chat")
        self.assertEqual(amend["successor_study_id"], "acme-chat-v2")
        self.assertEqual(recipe_bump.successor_study_id(
            "acme-chat-v2", pathlib.Path("/x/serving-recipe.v3.json")),
            "acme-chat-v3")
        self.assertNotEqual(doc["recipe_digest_before"],
                            doc["recipe_digest_after"])
        self.assertEqual(recipe_bump.next_version_path(
            pathlib.Path("/x/serving-recipe.v2.json")).name,
            "serving-recipe.v3.json")

    def test_no_win_and_no_delta_leave_the_recipe_alone(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            state = self._engine_state(verdict="tie")
            state["arms"][1]["result"]["verdict"] = "loss"
            doc = self._run(tmp, state, self._bank())
            self.assertEqual(doc["status"], "no_win")
            self.assertFalse((tmp / "serving-recipe.v2.json").exists())
            doc = self._run(tmp, self._engine_state(), {"arms": {}})
            self.assertEqual(doc["status"], "no_delta")
            self.assertFalse((tmp / "serving-recipe.v2.json").exists())

    def test_launch_recipe_takes_env_per_role_and_flags_through_knobs(self):
        """A role-script engine has no vLLM argv in the recipe: env lands on
        `launch.role_env.<escalated role>`, a flag a `launch.knobs` entry
        maps goes through that knob's env on its roles, any other flag is
        reported `unapplied` for a person to carry into the script."""
        recipe = json.loads(json.dumps(AGENTX_RECIPE))
        recipe["launch"]["role_env"] = {"prefill": {"PREFILL_TP": 8},
                                        "decode": {"DECODE_TP": 8,
                                                   "MAX_NUM_SEQS_DECODE": 32}}
        recipe["launch"]["knobs"]["max_num_seqs"]["arg"] = "--max-num-seqs"
        delta = {"args": {"--max-num-seqs": 48, "--max-num-batched-tokens": 8192},
                 "env": {"TPU_HMA_MAX_INFLIGHT_PULLS": "8"}, "image": ""}
        out, changes = recipe_bump.apply_delta(recipe, delta, "decode")
        dec, pf = out["launch"]["role_env"]["decode"], out["launch"]["role_env"]["prefill"]
        self.assertEqual(dec["MAX_NUM_SEQS_DECODE"], 48)
        self.assertEqual(dec["TPU_HMA_MAX_INFLIGHT_PULLS"], "8")
        self.assertNotIn("TPU_HMA_MAX_INFLIGHT_PULLS", pf)   # decode only
        self.assertNotIn("args", out)                        # never argv
        self.assertEqual(out["env"] if "env" in out else {}, {})
        unapplied = [c for c in changes if c.startswith("unapplied:")]
        self.assertEqual(len(unapplied), 1)
        self.assertIn("--max-num-batched-tokens", unapplied[0])
        # An aggregated escalation writes env on every role.
        out2, _ = recipe_bump.apply_delta(recipe, {"env": {"X": "1"}},
                                          "aggregated")
        self.assertEqual(out2["launch"]["role_env"]["prefill"]["X"], "1")
        self.assertEqual(out2["launch"]["role_env"]["decode"]["X"], "1")
        # Through bump(): a delta of only unmapped flags leaves the recipe
        # alone and says why.
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            (tmp / "serving-recipe.json").write_text(json.dumps(recipe))
            (tmp / "estate.json").write_text(json.dumps(self._engine_state()))
            (tmp / "bank.json").write_text(json.dumps({"arms": {
                "e-decode-max-seqs": {"target_parameters": {"serving": {
                    "args": {"--max-num-batched-tokens": 8192}}}}}}))
            (tmp / "esc.json").write_text(json.dumps(
                {"study_id": "ax", "role": "decode", "escalation": 1}))
            import argparse  # noqa: PLC0415
            doc = recipe_bump.bump(argparse.Namespace(
                recipe=str(tmp / "serving-recipe.json"),
                engine_study_state=str(tmp / "estate.json"),
                hypothesis_bank=str(tmp / "bank.json"),
                escalation=str(tmp / "esc.json"), out=str(tmp / "bump")))
            self.assertEqual(doc["status"], "unapplied")
            self.assertEqual(len(doc["unapplied"]), 1)
            self.assertFalse((tmp / "serving-recipe.v2.json").exists())


AGENTX_RECIPE = {
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "image": "example/torchtpu-vllm:infx",
    "accelerator": {"tpu_accelerator": "tpu7x", "tpu_topology": "2x2x1",
                    "chips_per_host": 4},
    "tensor_parallel": 8,
    "router_images": {"epp": "ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:abc",
                      "sidecar": "ghcr.io/llm-d/llm-d-router-disagg-sidecar@sha256:def"},
    "launch": {"command": ["bash", "server.sh"],
               "ports": {"prefill": 8000, "decode": 7200},
               "knobs": {"max_num_seqs": {"env": "MAX_NUM_SEQS_DECODE",
                                          "roles": ["decode"]},
                         "kv_cpu_offload_bytes": {
                             "env": "PREFILL_KV_CPU_OFFLOAD_BYTES",
                             "roles": ["prefill"], "unit": "GiB"}}},
    "measured": {"sessions_per_prefill_engine": 96,
                 "sessions_per_decode_engine": 112},
}
AGENTX_SPEC = {
    "name": "qwen35-agentx",
    "workload": {"agentx": {"concurrency": [96, 192, 240, 288],
                            "fast": True}},
    "budget": {"max_chips": 24, "accelerator": "tpu7x"},
}


def _agentx_report(*, admissible: bool, ttft_p50_ms: float,
                   prefill_hit: float | None, dec_running: float,
                   dec_kv: float, pf_waiting: float = 0.0,
                   total_per_chip: float = 30000.0) -> dict:
    return {
        "admissible": admissible,
        "metrics": {"total_tokens_per_s_per_chip": {"value": total_per_chip}},
        "tenants": {"agentx": {
            "slo": {"ttft_p50_ms": {"bound_ms": 5000,
                                    "measured_ms": ttft_p50_ms,
                                    "pass": ttft_p50_ms <= 5000}},
            "replicates": [{
                "agentx": {"gpu_prefix_hit_rate": prefill_hit},
                "roles": {
                    "prefill": {"pods": 1, "requests_waiting_max": pf_waiting,
                                "prefix_hit_rate": prefill_hit},
                    "decode": {"pods": 1, "requests_running_max": dec_running,
                               "kv_util_max": dec_kv}}}]}}}


class AgentXSearchTest(_Quiet):
    """The closed-loop hill-climb: seed, climb, attribute, move the shape."""

    def _profile(self):
        from framework.deploy import workload  # noqa: PLC0415
        return workload.resolve_workload(json.loads(json.dumps(AGENTX_SPEC)))

    def test_seed_opens_on_sessions_per_engine_and_the_router_ab(self):
        arms, plans = seed_arms.seed(self._profile(), AGENTX_RECIPE,
                                     max_chips=24)
        by_id = {a["arm"]: a for a in arms["arms"]}
        base = plans["plans"]["baseline"]["sets"]
        # 96 sessions / 96 per prefill = 1P; / 112 per decode = 1D.
        self.assertEqual(base["topology.prefill"]["replicas"], 1)
        self.assertEqual(base["topology.decode.replicas"], 1)
        self.assertEqual(base["workload.concurrency"], 96)
        self.assertEqual(base["router.profile"], "agentx-turn-priority")
        self.assertEqual(plans["plans"]["c1"]["arm_kind"],
                         "operating_point.concurrency")
        self.assertEqual(plans["plans"]["c1"]["sets"],
                         {"workload.concurrency": 192})
        self.assertEqual(plans["plans"]["r1"]["sets"],
                         {"router.profile": "agentx-session-affinity"})
        # Only the knobs the recipe maps become arms.
        kinds = {k: v["arm_kind"] for k, v in plans["plans"].items()}
        role_knobs = [plans["plans"][k]["sets"] for k, v in kinds.items()
                      if v == "engine.role_knob"]
        self.assertEqual(role_knobs, [
            {"topology.decode.engine.max_num_seqs": 48},
            {"topology.prefill.engine.kv_cpu_offload_bytes": 256}])
        self.assertEqual(by_id["baseline"]["kind"], "fleet-topology")
        self.assertEqual(by_id["c1"]["status"], "open")
        self.assertEqual(plans["agentx"]["opening"]["provenance"], "measured")
        # Without router images the v0.11.0 P/D profile is the baseline and
        # there is no A/B; 240 sessions open 3P:3D.
        plain = {k: v for k, v in AGENTX_RECIPE.items() if k != "router_images"}
        spec = json.loads(json.dumps(AGENTX_SPEC))
        spec["workload"]["agentx"]["concurrency"] = [240]
        from framework.deploy import workload  # noqa: PLC0415
        _, plans2 = seed_arms.seed(workload.resolve_workload(spec), plain,
                                   max_chips=24)
        b2 = plans2["plans"]["baseline"]["sets"]
        self.assertEqual((b2["topology.prefill"]["replicas"],
                          b2["topology.decode.replicas"]), (3, 3))
        self.assertEqual(b2["router.profile"], "sticky-until-saturated")
        self.assertNotIn("r1", plans2["plans"])

    def test_attribution_reads_cliff_signatures(self):
        from framework.deploy import render  # noqa: PLC0415
        inc = render.default_config(AGENTX_RECIPE)
        inc["topology"]["mode"] = "disaggregated"
        inc["topology"]["prefill"] = {"replicas": 1, "tp": 8}
        inc["topology"]["decode"]["engine"] = {"max_num_seqs": 32}
        decode_cliff = _agentx_report(admissible=False, ttft_p50_ms=52000,
                                      prefill_hit=0.35, dec_running=32,
                                      dec_kv=0.91)
        att = agentx_search.attribute(decode_cliff, inc)
        self.assertEqual((att["role"], att["kind"]),
                         ("decode", "memory-capacity"))
        prefill_cliff = _agentx_report(admissible=False, ttft_p50_ms=127000,
                                       prefill_hit=0.40, dec_running=17,
                                       dec_kv=0.34, pf_waiting=200)
        att = agentx_search.attribute(prefill_cliff, inc)
        self.assertEqual(att["role"], "prefill")
        healthy = _agentx_report(admissible=True, ttft_p50_ms=2650,
                                 prefill_hit=0.83, dec_running=20, dec_kv=0.5)
        self.assertEqual(agentx_search.attribute(healthy, inc)["role"],
                         "none")
        queued = _agentx_report(admissible=False, ttft_p50_ms=6000,
                                prefill_hit=0.82, dec_running=10, dec_kv=0.3)
        att = agentx_search.attribute(queued, inc)
        self.assertEqual((att["role"], att["kind"]),
                         ("prefill", "scheduler-batching"))

    def test_propose_climbs_then_moves_the_bound_axis(self):
        from framework.deploy import render  # noqa: PLC0415
        inc = render.default_config(AGENTX_RECIPE)
        inc["topology"]["mode"] = "disaggregated"
        inc["topology"]["prefill"] = {"replicas": 2, "tp": 8}
        inc["topology"]["decode"]["replicas"] = 2
        inc["router"]["profile"] = "agentx-session-affinity"
        inc["workload"]["concurrency"] = 192
        collected: list[tuple[str, dict]] = []

        def add(arm_kind, claim, *, share, headroom, sets, rationale,
                prerequisites=None):
            collected.append((arm_kind, sets))

        # Qualifying at 192 on 2P:2D with nothing saturated: climb to 240
        # first; no router A/B again once both profiles were measured.
        healthy = _agentx_report(admissible=True, ttft_p50_ms=2800,
                                 prefill_hit=0.83, dec_running=15, dec_kv=0.36)
        decided = [{"arm": "r1", "arm_kind": "router.profile",
                    "sets": {"router.profile": "agentx-turn-priority"},
                    "verdict": "loss"}]
        agentx_search.propose(self._profile(), AGENTX_RECIPE, inc,
                              decided=decided, incumbent_report=healthy,
                              max_chips=24, chips_per=4, add=add)
        self.assertEqual(collected[0],
                         ("operating_point.concurrency",
                          {"workload.concurrency": 240}))
        self.assertFalse(any(k == "router.profile" for k, _ in collected))
        shapes = [s for k, s in collected if k == "topology.pd_shape"]
        # Both pools idle and qualifying: leaner shapes are proposed.
        self.assertIn({"topology.prefill.replicas": 1,
                       "topology.decode.replicas": 2}, shapes)
        # Decode-saturated and failing: move a slice to decode, no climb.
        collected.clear()
        cliff = _agentx_report(admissible=False, ttft_p50_ms=52000,
                               prefill_hit=0.35, dec_running=32, dec_kv=0.91)
        inc["topology"]["decode"]["engine"] = {"max_num_seqs": 32}
        agentx_search.propose(self._profile(), AGENTX_RECIPE, inc,
                              decided=decided, incumbent_report=cliff,
                              max_chips=24, chips_per=4, add=add)
        kinds = [k for k, _ in collected]
        self.assertNotIn("operating_point.concurrency", kinds)
        shapes = [s for k, s in collected if k == "topology.pd_shape"]
        self.assertEqual(shapes[0], {"topology.prefill.replicas": 1,
                                     "topology.decode.replicas": 3})
        # Failing with no saturation: back off the ladder.
        collected.clear()
        knee = _agentx_report(admissible=False, ttft_p50_ms=6000,
                              prefill_hit=0.82, dec_running=10, dec_kv=0.3)
        knee["tenants"]["agentx"]["replicates"][0]["roles"]["prefill"][
            "requests_waiting_max"] = 0
        agentx_search.propose(self._profile(), AGENTX_RECIPE, inc,
                              decided=decided, incumbent_report=knee,
                              max_chips=24, chips_per=4, add=add)
        att = agentx_search.attribute(knee, inc)
        self.assertEqual(att["role"], "prefill")  # TTFT failed -> prefill
        # (a TTFT failure without saturation names prefill queueing, so the
        # shape moves toward prefill rather than backing off)
        shapes = [s for k, s in collected if k == "topology.pd_shape"]
        self.assertEqual(shapes[0]["topology.prefill.replicas"], 3)

    def test_shape_arm_claims_the_rung_the_incumbent_could_not_hold(self):
        """What the kind run showed: 1P:1D qualified at 4 and broke the
        TTFT gate at 8 (prefill-bound). The shape move exists to sustain 8,
        so the 2P:1D arm carries workload.concurrency 8, and so does the
        bound role's knob; the other role's knob stays at the incumbent."""
        from framework.deploy import render  # noqa: PLC0415
        inc = render.default_config(AGENTX_RECIPE)
        inc["topology"]["mode"] = "disaggregated"
        inc["topology"]["prefill"] = {"replicas": 1, "tp": 8}
        inc["topology"]["decode"]["replicas"] = 1
        inc["router"]["profile"] = "agentx-session-affinity"
        inc["workload"]["concurrency"] = 96
        collected: list[tuple[str, dict]] = []

        def add(arm_kind, claim, *, share, headroom, sets, rationale,
                prerequisites=None):
            collected.append((arm_kind, sets, claim))
        failed = _agentx_report(admissible=False, ttft_p50_ms=11200,
                                prefill_hit=0.67, dec_running=7, dec_kv=0.62)
        failed["tenants"]["agentx"]["replicates"][0]["roles"]["prefill"][
            "requests_waiting_max"] = 0
        decided = [{"arm": "baseline", "arm_kind": "baseline",
                    "verdict": "tie", "sets": {}},
                   {"arm": "c1", "arm_kind": "operating_point.concurrency",
                    "verdict": "invalid",
                    "sets": {"workload.concurrency": 192}}]
        model = agentx_search.propose(self._profile(), AGENTX_RECIPE, inc,
                                      decided=decided,
                                      incumbent_report=failed,
                                      max_chips=24, chips_per=4, add=add)
        self.assertEqual(model["cliff"], 192)
        self.assertEqual(model["attribution"]["role"], "prefill")
        shapes = [(s, c) for k, s, c in collected if k == "topology.pd_shape"]
        self.assertEqual(shapes[0][0], {"topology.prefill.replicas": 2,
                                        "topology.decode.replicas": 1,
                                        "workload.concurrency": 192})
        self.assertIn("broke the gate there", shapes[0][1])
        knobs = {next(k for k in s if k.startswith("topology.")): s
                 for kind, s, _ in collected if kind == "engine.role_knob"}
        self.assertIn("workload.concurrency",
                      knobs["topology.prefill.engine.kv_cpu_offload_bytes"])
        self.assertNotIn("workload.concurrency",
                         knobs["topology.decode.engine.max_num_seqs"])
        # Not climbing (the gate failed) and not backing off (a pool is bound).
        self.assertFalse(any(k == "operating_point.concurrency"
                             for k, _, _ in collected))

    def test_plan_arms_routes_agentx_specs(self):
        from framework.deploy import render  # noqa: PLC0415
        inc = render.default_config(AGENTX_RECIPE)
        inc["topology"]["mode"] = "disaggregated"
        inc["topology"]["prefill"] = {"replicas": 1, "tp": 8}
        inc["router"]["profile"] = "agentx-turn-priority"
        inc["workload"]["concurrency"] = 96
        rep = _agentx_report(admissible=True, ttft_p50_ms=2540,
                             prefill_hit=0.82, dec_running=18, dec_kv=0.45)
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            (tmp / "spec.json").write_text(json.dumps(AGENTX_SPEC))
            (tmp / "recipe.json").write_text(json.dumps(AGENTX_RECIPE))
            (tmp / "inc.json").write_text(json.dumps(inc))
            (tmp / "report.json").write_text(json.dumps(rep))
            result = plan_arms.propose({
                "run_id": "r", "study_id": "s",
                "workload_spec_path": str(tmp / "spec.json"),
                "recipe_path": str(tmp / "recipe.json"),
                "incumbent_config_path": str(tmp / "inc.json"),
                "incumbent_report_path": str(tmp / "report.json"),
                "decided_arms": [], "open_arms": [], "max_new_arms": 4})
        self.assertEqual(result["status"], "planned")
        kinds = [result["plans"][a["arm"]]["arm_kind"] for a in result["arms"]]
        self.assertEqual(kinds[0], "operating_point.concurrency")
        self.assertIn("router.profile", kinds)
        self.assertEqual(result["model"]["attribution"]["role"], "none")


class AgentXLeaderboardTest(_Quiet):
    """The leaderboard view: the dashboard's frontier, prices and axes."""

    def test_pareto_is_the_dashboards_upper_left_frontier(self):
        pts = [{"x": 500, "y": 10}, {"x": 300, "y": 30}, {"x": 400, "y": 5},
               {"x": 100, "y": 60}, {"x": 200, "y": 20}, {"x": 0, "y": 99}]
        front = agentx_leaderboard.pareto(pts, "x", "y")
        self.assertEqual([(p["x"], p["y"]) for p in front],
                         [(500, 10), (300, 30), (100, 60)])

    def test_price_for_uses_the_pinned_table_per_basis_and_overrides(self):
        self.assertEqual(agentx_leaderboard.price_for("tpuv7", "internal", {}),
                         1.03)
        self.assertEqual(agentx_leaderboard.price_for("tpu7x", "external", {}),
                         1.21)
        self.assertEqual(agentx_leaderboard.price_for("gb300_dynamo-sglang",
                                                      "internal", {}), 2.31)
        self.assertEqual(agentx_leaderboard.price_for("tpu7x", "internal",
                                                      {"tpu7x": 2.0}), 2.0)
        self.assertIsNone(agentx_leaderboard.price_for("abacus", "internal", {}))
        # tokens/$ = tok/s/chip x 3600 / $/chip-hour.
        self.assertAlmostEqual(agentx_leaderboard.tokens_per_dollar(
            {"total_tokens_per_s_per_chip": 20000}, 1.03), 69_902_913, delta=1)

    def test_views_to_entries_keeps_the_dashboards_x_y_and_chips(self):
        doc = {"xAxis": {"field": "p90_intvty"}, "metric": {"key": "tpPerGpu"},
               "count": 2, "series": [{
                   "hwKey": "tpuv7_vllm", "label": "TPU7x (vLLM)", "gpu": "tpuv7",
                   "framework": "vllm", "specMethod": "mtp",
                   "deployment": "single-node", "bestPerSku": True,
                   "points": [
                       {"x": 100.0, "y": 20000.0, "concurrency": 32, "tp": 4,
                        "precision": "fp8", "frontier": True,
                        "topologyKey": "Single|GPU=4|TP=4|EP=1",
                        "metrics": {"tpPerGpu": 20000.0,
                                    "outputTputPerGpu": 400.0}},
                       {"x": 120.0, "y": 15000.0, "concurrency": 16, "tp": 4,
                        "precision": "fp8", "frontier": True,
                        "metrics": {"tpPerGpu": 15000.0}}]}]}
        out = agentx_leaderboard.views_to_entries(doc, "u")
        e = out["entries"][0]
        self.assertEqual(e["name"], "TPU7x (vLLM)")
        self.assertEqual(e["hardware"], "tpuv7")
        self.assertEqual(e["points"][0]["chips"], 4)
        self.assertEqual(e["points"][0]["intvty_p90"], 100.0)
        self.assertEqual(e["points"][0]["total_tokens_per_s_per_chip"], 20000.0)
        self.assertTrue(e["points"][0]["dashboard_frontier"])
        lb = agentx_leaderboard.leaderboard_points(out["entries"], "internal", {})
        self.assertEqual(lb[0]["price_per_chip_hour"], 1.03)
        self.assertEqual(agentx_leaderboard._topology_short(
            "PD|GPU=32|P-GPU=20|D-GPU=12"), "PD 32 (P20/D12)")

    def test_shipped_leaderboard_frontier_matches_the_dashboard_flag(self):
        ref = _HERE.parents[1] / "references" / "tpu-llmd" / "agentx" / "leaderboard.json"
        doc = json.loads(ref.read_text())
        self.assertEqual(doc["entries"], [])
        entries = [{
            "name": "Synthetic (vLLM)", "hardware": "tpuv7", "framework": "vllm",
            "points": [
                {"conc": 8, "chips": 4, "intvty_p90": 150.0,
                 "total_tokens_per_s_per_chip": 10000.0,
                 "dashboard_frontier": True},
                {"conc": 16, "chips": 4, "intvty_p90": 120.0,
                 "total_tokens_per_s_per_chip": 18000.0,
                 "dashboard_frontier": True},
                {"conc": 24, "chips": 4, "intvty_p90": 110.0,
                 "total_tokens_per_s_per_chip": 16000.0,
                 "dashboard_frontier": False},
            ],
        }]
        pts = agentx_leaderboard.leaderboard_points(entries, "internal", {})
        agentx_leaderboard._frontier_flags(pts, "intvty_p90")
        self.assertEqual([p["frontier"] for p in pts],
                         [p["dashboard_frontier"] for p in pts])
        for hw in ("tpuv7", "gb300", "b300", "b200", "mi355x", "h200", "h100"):
            for basis in ("internal", "external", "rental"):
                self.assertIsNotNone(agentx_leaderboard.price_for(
                    hw, basis, {}), (hw, basis))

    def test_build_draws_our_points_against_the_leaderboard(self):
        report = {
            "schema": "llmd-bench-report", "execution_id": "x",
            "release": {"name": "ax", "chips_total": 8,
                        "router_profile": "agentx-turn-priority"},
            "agentx": {"submission_valid": True, "duration_s": 900, "points": [
                {"concurrency": c, "admissible": c < 24,
                 "gates": [{"name": "agentx.ttft_p50_ms", "bound": 5000}],
                 "metrics": {"total_tokens_per_s_per_chip": {"value": y},
                             "output_tokens_per_s_per_chip": {"value": y / 90},
                             "total_tokens_per_s": {"value": y * 8}},
                 "tenant": {"replicates": [{
                     "requests": {"ok": 10, "total": 10},
                     "agentx": {"interactivity_p90": x, "interactivity_p50": x * 1.3,
                                "ttft_s": {"p50": 1.0, "p90": 2.0},
                                "tpot_ms": {"p50": 1000 / x},
                                "e2e_s": {"p50": 30}}}]}}
                for c, x, y in ((8, 300.0, 9000.0), (16, 200.0, 18000.0),
                                (24, 100.0, 26000.0))]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rel = root / "releases" / "ax"
            rel.mkdir(parents=True)
            (rel / "release.json").write_text("{}")
            (rel / "llmd-config.json").write_text(json.dumps({
                "topology": {"mode": "disaggregated",
                             "prefill": {"replicas": 1}, "decode": {"replicas": 1}},
                "router": {"profile": "agentx-turn-priority"}}))
            bench = rel / "bench" / "c"
            bench.mkdir(parents=True)
            (bench / "llmd-bench-report.json").write_text(json.dumps(report))
            lb = {"entries": [{"name": "TPU7x (vLLM)", "hardware": "tpuv7",
                               "framework": "vllm", "points": [
                                   {"conc": 32, "chips": 4, "intvty_p90": 100.0,
                                    "total_tokens_per_s_per_chip": 20000.0}]}]}
            (root / "lb.json").write_text(json.dumps(lb))
            rc = agentx_leaderboard.main([
                "build", "--study-dir", str(root), "--leaderboard",
                str(root / "lb.json"), "--out", str(root / "out")])
            self.assertEqual(rc, 0)
            pts = json.loads((root / "out" / "points.json").read_text())
            ours = [p for p in pts["points"] if p["source"] == "study"]
            self.assertEqual([p["conc"] for p in ours], [8, 16, 24])
            self.assertEqual(ours[0]["series"], "1P1D turn-priority")
            self.assertEqual(ours[0]["price_per_chip_hour"], 1.03)
            self.assertAlmostEqual(ours[0]["tokens_per_dollar"],
                                   9000 * 3600 / 1.03)
            self.assertTrue(all(p["frontier"] for p in ours))
            md = (root / "out" / "leaderboard.md").read_text()
            self.assertIn("| 1P1D turn-priority | 16 |", md)
            self.assertIn("internal", md)
            svg = (root / "out" / "throughput_vs_interactivity.svg").read_text()
            self.assertIn("c16", svg)
            self.assertIn("TPU7x (vLLM)", svg)
            # The gate-failing c24 point is hollow.
            self.assertIn('fill="white"', svg)
            tpd = (root / "out" / "tokens_per_dollar_vs_interactivity.svg"
                   ).read_text()
            self.assertIn("$1.03", tpd)
            csv_text = (root / "out" / "points.csv").read_text()
            self.assertIn("intvty_p90", csv_text.splitlines()[0])

    def test_fetch_live_from_file_merges_non_live_entries(self):
        doc = {"series": [{"hwKey": "h100_sglang", "label": "H100 (SGLang)",
                           "gpu": "h100", "framework": "sglang",
                           "points": [{"x": 50.0, "y": 5000.0,
                                       "concurrency": 8, "tp": 8}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "views.json").write_text(json.dumps(doc))
            (root / "base.json").write_text(json.dumps({"entries": [
                {"name": "H100 (SGLang)", "points": []},
                {"name": "custom series", "hardware": "tpu7x", "points": []}]}))
            rc = agentx_leaderboard.main([
                "fetch-live", "--from-file", str(root / "views.json"),
                "--merge", str(root / "base.json"),
                "--out", str(root / "lb.json")])
            self.assertEqual(rc, 0)
            out = json.loads((root / "lb.json").read_text())
            self.assertEqual([e["name"] for e in out["entries"]],
                             ["H100 (SGLang)", "custom series"])
            self.assertEqual(out["entries"][0]["points"][0]["chips"], 8)
            self.assertIn("snapshot", out)


class ProductionGuardTest(unittest.TestCase):

    def _shell(self, command, tool="run_shell_command"):
        return {"tool": tool, "args": {"command": command}}

    def test_denies_include_production_teardown_without_marker(self):
        verdict = production_guard.decide(self._shell(
            "python3 runtime.py deploy teardown --name p --release /tmp/x"
            " --include-production"))
        self.assertEqual(verdict["decision"], "deny")
        self.assertIn("user action", verdict["reason"])
        # The flag is what is denied, not teardown itself.
        self.assertEqual(production_guard.decide(self._shell(
            "python3 runtime.py deploy teardown --name p --release /tmp/x"))
            ["decision"], "allow")

    def test_marker_in_the_bundle_lifts_the_teardown_rule(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "release").mkdir()
            cmd = (f"runtime deploy teardown --name p --release"
                   f" {bundle / 'release'} --include-production")
            self.assertEqual(production_guard.decide(self._shell(cmd))
                             ["decision"], "deny")
            (bundle / production_guard.APPROVAL_MARKER).write_text("")
            self.assertEqual(production_guard.decide(self._shell(cmd))
                             ["decision"], "allow")

    def test_denies_raw_mutations_that_name_production(self):
        label = production_guard.PRODUCTION_LABEL
        self.assertEqual(production_guard.decide(self._shell(
            f"kubectl delete deploy -l {label}=true"))["decision"], "deny")
        self.assertEqual(production_guard.decide(self._shell(
            f"helm uninstall acme-prod -l {label}=true"))["decision"], "deny")
        # A read naming the label is fine; so is a mutation elsewhere.
        self.assertEqual(production_guard.decide(self._shell(
            f"kubectl get pods -l {label}=true"))["decision"], "allow")
        self.assertEqual(production_guard.decide(self._shell(
            "kubectl delete job bench-r0 -n llmd-study"))["decision"], "allow")

    def test_production_namespace_comes_from_the_environment(self):
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            env_path = pathlib.Path(d) / "environment.json"
            env_path.write_text(json.dumps(
                {"llmd": {"production_namespace": "serving-prod"}}))
            with mock.patch.dict(os.environ,
                                 {"TPU_AGENT_SUITE_ENVIRONMENT": str(env_path)}):
                deny = production_guard.decide(self._shell(
                    "kubectl scale deploy x --replicas=0 -n serving-prod"))
                allow = production_guard.decide(self._shell(
                    "kubectl scale deploy x --replicas=0 -n serving-prod-2"))
        self.assertEqual(deny["decision"], "deny")
        self.assertEqual(allow["decision"], "allow")

    def test_non_shell_tools_and_empty_payloads_pass(self):
        self.assertEqual(production_guard.decide(self._shell(
            "deploy teardown --include-production", tool="read_file"))
            ["decision"], "allow")
        self.assertEqual(production_guard.decide({})["decision"], "allow")


class ChipBudgetGuardTest(unittest.TestCase):

    def _payload(self, tmp, sets, *, recipe=RECIPE, budget=64,
                 subagent="llmd-arm-executor"):
        from framework.deploy import render  # noqa: PLC0415
        spec = json.loads(json.dumps(SPEC))
        spec["budget"]["max_chips"] = budget
        (tmp / "spec.json").write_text(json.dumps(spec))
        (tmp / "recipe.json").write_text(json.dumps(recipe))
        (tmp / "incumbent.json").write_text(
            json.dumps(render.default_config(recipe, replicas=3)))
        request = {"arm": "a9", "sets": sets,
                   "incumbent_config_path": str(tmp / "incumbent.json"),
                   "recipe_path": str(tmp / "recipe.json"),
                   "workload_spec_path": str(tmp / "spec.json")}
        prompt = ("Evaluate this arm. Request:\n```json\n"
                  + json.dumps(request, indent=2) + "\n```\nReport back.")
        return {"tool": "delegate", "subagent": subagent,
                "args": {"prompt": prompt}}

    def test_only_executor_delegations_are_examined(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            huge = {"topology.decode.replicas": 100}
            self.assertEqual(chip_budget_guard.decide(self._payload(
                tmp, huge, subagent="llmd-planner"))["decision"], "allow")
            self.assertEqual(chip_budget_guard.decide(
                {"tool": "shell", "args": {"command": "ls"}})["decision"],
                "allow")
            self.assertEqual(chip_budget_guard.decide(self._payload(
                tmp, huge))["decision"], "deny")

    def test_arm_arithmetic_counts_prefill_under_pd(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            # 2x2x1 = 4 chips per replica; 16 decode x 4 = 64 fits exactly.
            self.assertEqual(chip_budget_guard.decide(self._payload(
                tmp, {"topology.decode.replicas": 16}))["decision"], "allow")
            verdict = chip_budget_guard.decide(self._payload(
                tmp, {"topology.mode": "disaggregated",
                      "topology.decode.replicas": 14,
                      "topology.prefill": {"replicas": 3, "tp": 4}}))
        self.assertEqual(verdict["decision"], "deny")
        self.assertIn("68 chips", verdict["reason"])
        self.assertIn("14 decode + 3 prefill", verdict["reason"])
        self.assertIn("not pruned", verdict["reason"])

    def test_adopted_fleets_cost_nothing(self):
        recipe = dict(RECIPE)
        recipe["fleet"] = {"mode": "adopted", "selector": {"app": "ds"},
                           "service": "ds", "port": 8000, "replicas": 2}
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            self.assertEqual(chip_budget_guard.decide(self._payload(
                tmp, {"topology.decode.replicas": 100}, recipe=recipe))
                ["decision"], "allow")

    def test_production_holdings_shrink_the_budget(self):
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            home = tmp / "home"
            for job, extra, status in (
                ("j1", {"production": True, "chips_held": 48}, "RUNNING"),
                ("j2", {"production": True, "chips_held": 48}, "CANCELLED"),
                ("j3", {"production": False, "chips_held": 48}, "RUNNING"),
            ):
                path = home / "runs" / "jobs" / job / "job.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"backend": "gke_release",
                                            "status": status, "extra": extra}))
            with mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(home)}):
                self.assertEqual(chip_budget_guard.production_chips(), 48)
                ok = chip_budget_guard.decide(self._payload(
                    tmp, {"topology.decode.replicas": 4}))
                over = chip_budget_guard.decide(self._payload(
                    tmp, {"topology.decode.replicas": 5}))
        self.assertEqual(ok["decision"], "allow")
        self.assertEqual(over["decision"], "deny")
        self.assertIn("production holds 48", over["reason"])

    def test_unreadable_inputs_fail_open(self):
        payload = {"tool": "delegate", "subagent": "llmd-arm-executor",
                   "args": {"prompt": json.dumps({
                       "incumbent_config_path": "/nonexistent/i.json",
                       "recipe_path": "/nonexistent/r.json",
                       "workload_spec_path": "/nonexistent/s.json",
                       "sets": {"topology.decode.replicas": 999}})}}
        self.assertEqual(chip_budget_guard.decide(payload)["decision"], "allow")
