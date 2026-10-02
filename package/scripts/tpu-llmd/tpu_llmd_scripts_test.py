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
        # reuse arms exist but are blocked until the runtime renders them.
        self.assertEqual(by_id["a3"]["status"], "blocked")
        self.assertIn("precise", by_id["a3"]["blocked_on"])
        self.assertEqual(by_id["a7"]["status"], "open")
        for arm in arms["arms"]:
            self.assertIn(arm["kind"], (
                "fleet-topology", "kv-reuse", "routing-policy",
                "engine-fleet-knob", "platform-capacity"))

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
