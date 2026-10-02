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
