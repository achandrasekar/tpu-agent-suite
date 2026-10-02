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
"""Tests for `framework.deploy`: the llm-d release verbs behind `deploy`.

Nothing here touches a cluster. `Kube` takes a runner, so every kubectl and
helm call is recorded by a fake that answers with canned output, and the
job store is redirected with `TPU_AGENT_SUITE_HOME`. The render tests are the
golden surface: a change to what a release contains should fail here first.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_NEXUS = pathlib.Path(__file__).resolve().parents[2]
if str(_NEXUS) not in sys.path:
    sys.path.insert(0, str(_NEXUS))

from framework import runtime, utils  # noqa: E402
from framework.deploy import (bench, cli, config, pins, release, render,  # noqa: E402
                              workload)
from framework.deploy.kube import DeployError  # noqa: E402

RECIPE = {
    "model": "Qwen/Qwen3-32B",
    "image": "docker.io/vllm/vllm-tpu:v0.29.0",
    "args": [
        "--max-model-len", "4096", "--max-num-seqs=128",
        "--enable-prefix-caching"
    ],
    "env": {"TORCH_TPU_FOO": "1"},
    "accelerator": {"tpu_accelerator": "tpu7x", "tpu_topology": "2x2x1"},
    "tensor_parallel": 4,
    "max_model_len": 4096,
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
    "budget": {"max_chips": 64, "accelerator": "tpu7x"},
    "objective": {"primary_metric": "goodput_at_slo_per_chip"},
}

ENV_SUBSET = {
    "namespace": "llmd-study",
    "hf_token_secret": "hf-token",
    "inference_perf_image": pins.INFERENCE_PERF_IMAGE,
    "gateway_mode": "standalone",
}


class FakeRunner:
    """Records every kubectl/helm argv and answers with canned output.

    `-o json` reads get an empty list document so the callers that parse
    JSON keep working; everything else succeeds with empty stdout unless a
    prefix in `fail` matches, in which case it fails with that stderr.
    """

    def __init__(self, fail: dict[str, str] | None = None,
                 json_docs: dict[str, dict] | None = None,
                 logs: str = "") -> None:
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []
        self.fail = fail or {}
        self.json_docs = json_docs or {}
        self.logs = logs

    def __call__(self, cmd, *, stdin=None, timeout=None):
        self.calls.append(list(cmd))
        self.stdins.append(stdin)
        joined = " ".join(cmd)
        for prefix, err in self.fail.items():
            if prefix in joined:
                return subprocess.CompletedProcess(cmd, 1, "", err)
        if "-o" in cmd and "json" in cmd:
            for key, doc in self.json_docs.items():
                if key in joined:
                    return subprocess.CompletedProcess(cmd, 0,
                                                       json.dumps(doc), "")
            return subprocess.CompletedProcess(cmd, 0, '{"items": []}', "")
        if "logs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.logs, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def matching(self, *words: str) -> list[list[str]]:
        return [c for c in self.calls if all(w in c for w in words)]


def _render(out: pathlib.Path, *, replicas: int = 3,
            config_edit=None) -> tuple[dict, dict]:
    profile = workload.resolve_workload(copy.deepcopy(SPEC))
    cfg = render.default_config(RECIPE, replicas=replicas)
    if config_edit:
        config_edit(cfg)
    rel = render.render_release(name="acme-r1", config=cfg, recipe=RECIPE,
                                profile=profile, env=dict(ENV_SUBSET),
                                out_dir=out, now="2026-10-01T00:00:00+00:00")
    return rel, cfg


class WorkloadTest(unittest.TestCase):

    def test_catalog_is_pinned_to_the_inference_perf_ref(self):
        self.assertEqual(workload.catalog_index()["inference_perf_ref"],
                         pins.INFERENCE_PERF_REF)
        self.assertIn("interactive-chat", workload.catalog_entries())

    def test_resolve_catalog_spec(self):
        profile = workload.resolve_workload(copy.deepcopy(SPEC))
        self.assertEqual(profile["source"]["kind"], "catalog")
        self.assertEqual(profile["rate"]["target_qps"], 120)
        self.assertEqual([t["name"] for t in profile["tenants"]],
                         ["interactive", "batch"])
        self.assertEqual(profile["objective"]["primary_metric"],
                         "goodput_at_slo_per_chip")
        targets = workload.slo_targets(profile["tenants"][0])
        self.assertEqual({t["key"] for t in targets},
                         {"ttft_p90_ms", "tpot_p90_ms"})

    def test_bad_slo_key_and_trace_are_refused(self):
        bad = copy.deepcopy(SPEC)
        bad["tenancy"][0]["slo"] = {"ttft_p95_ms": 500}
        with self.assertRaises(DeployError):
            workload.resolve_workload(bad)
        trace = copy.deepcopy(SPEC)
        trace["workload"] = {"trace": "gs://bucket/trace.jsonl"}
        with self.assertRaises(DeployError) as ctx:
            workload.resolve_workload(trace)
        self.assertEqual(ctx.exception.error_code, "NOT_IMPLEMENTED")

    def test_cost_metric_needs_chip_hour_cost(self):
        spec = copy.deepcopy(SPEC)
        spec["objective"] = {"primary_metric": "cost_per_1M_tokens_at_slo"}
        with self.assertRaises(DeployError):
            workload.resolve_workload(spec)
        spec["budget"]["chip_hour_cost"] = 4.2
        profile = workload.resolve_workload(spec)
        self.assertEqual(profile["budget"]["chip_hour_cost"], 4.2)


class RenderTest(unittest.TestCase):

    def test_release_files_and_chip_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rel, _ = _render(out)
            names = sorted(str(p.relative_to(out)) for p in out.rglob("*")
                           if p.is_file())
            for expected in ("release.json", "manifests/10-modelserver.json",
                             "manifests/40-monitoring.json",
                             "router.values.json", "epp-config.json",
                             "llmd-config.json", "serving-recipe.json",
                             "workload_profile.json",
                             "bench/interactive.slo.job.json",
                             "bench/batch.sweep.configmap.json"):
                self.assertIn(expected, names)
            self.assertEqual(rel["replicas"], 3)
            self.assertEqual(rel["chips"]["total"], 12)
            self.assertEqual(rel["router"]["profile"], "sticky-until-saturated")
            self.assertEqual(rel["router"]["service"],
                             "acme-r1-epp.llmd-study.svc.cluster.local")
            self.assertEqual(rel["pins"]["llmd_ref"], pins.LLMD_REF)

    def test_engine_overrides_replace_recipe_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)

            def edit(cfg):
                cfg["engine"]["max_num_seqs"] = 256

            _render(out, config_edit=edit)
            doc = json.loads((out / "manifests/10-modelserver.json").read_text())
            deployment = next(i for i in doc["items"]
                              if i["kind"] == "Deployment")
            args = deployment["spec"]["template"]["spec"]["containers"][0][
                "args"]
            joined = " ".join(args)
            self.assertIn("--max-num-seqs=256", joined)
            self.assertNotIn("--max-num-seqs=128", joined)
            self.assertIn("--enable-prefix-caching", joined)
            self.assertEqual(deployment["spec"]["replicas"], 3)
            selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
            self.assertEqual(selector[pins.ACCELERATOR_LABEL], "tpu7x")
            self.assertEqual(selector[pins.TOPOLOGY_LABEL], "2x2x1")

    def test_epp_config_overrides_weight_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)

            def edit(cfg):
                cfg["router"]["plugins"] = {
                    "prefix-cache-affinity-filter": {
                        "peakPrefillThroughput": 27336
                    },
                    "token-load-scorer": {"weight": 1.5},
                }

            _render(out, config_edit=edit)
            epp = json.loads((out / "epp-config.json").read_text())
            self.assertEqual(epp["apiVersion"], pins.EPP_CONFIG_API_VERSION)
            self.assertEqual(epp["kind"], pins.EPP_CONFIG_KIND)
            by_type = {p["type"]: p for p in epp["plugins"]}
            self.assertEqual(
                by_type["prefix-cache-affinity-filter"]["parameters"]
                ["peakPrefillThroughput"], 27336)
            refs = {p["pluginRef"]: p
                    for p in epp["schedulingProfiles"][0]["plugins"]}
            self.assertEqual(refs["token-load-scorer"]["weight"], 1.5)

    def test_unknown_plugin_and_not_implemented_features_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)

            def unknown(cfg):
                cfg["router"]["plugins"] = {"made-up-scorer": {"weight": 1}}

            with self.assertRaises(DeployError):
                _render(out / "a", config_edit=unknown)

            def pd(cfg):
                cfg["topology"]["mode"] = "pd"

            with self.assertRaises(DeployError) as ctx:
                _render(out / "b", config_edit=pd)
            self.assertEqual(ctx.exception.error_code, "NOT_IMPLEMENTED")

            def bad_profile(cfg):
                cfg["router"]["profile"] = "round-robin"

            with self.assertRaises(DeployError):
                _render(out / "c", config_edit=bad_profile)

            def foreign_engine_knob(cfg):
                cfg["engine"]["speculative_config"] = {"n": 3}

            with self.assertRaises(DeployError):
                _render(out / "d", config_edit=foreign_engine_knob)

    def test_bench_job_carries_goodput_constraints_in_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            _render(out)
            cm = json.loads(
                (out / "bench/interactive.slo.configmap.json").read_text())
            ip_cfg = json.loads(cm["data"]["config.json"])
            goodput = ip_cfg["report"]["goodput"]["constraints"]
            self.assertAlmostEqual(goodput["ttft"], 0.5)
            self.assertAlmostEqual(goodput["tpot"], 0.03)
            job = json.loads((out / "bench/interactive.slo.job.json").read_text())
            self.assertEqual(job["kind"], "Job")
            container = job["spec"]["template"]["spec"]["containers"][0]
            self.assertEqual(container["image"], pins.INFERENCE_PERF_IMAGE)

    def test_config_digest_ignores_provenance(self):
        cfg = render.default_config(RECIPE, replicas=2)
        other = copy.deepcopy(cfg)
        other["provenance"] = {"note": "different"}
        self.assertEqual(render.config_digest(cfg), render.config_digest(other))
        other["engine"]["max_num_seqs"] = 999
        self.assertNotEqual(render.config_digest(cfg),
                            render.config_digest(other))

    def test_rerender_keeps_name_and_changes_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rel, cfg = _render(out)
            cfg2 = copy.deepcopy(cfg)
            cfg2["engine"]["max_num_seqs"] = 512
            rel2 = render.rerender(out, cfg2)
            self.assertEqual(rel2["name"], rel["name"])
            self.assertNotEqual(rel2["config_digest"], rel["config_digest"])


class ReleaseLifecycleTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name) / "home"
        self.home.mkdir()
        self._env = mock.patch.dict(os.environ,
                                    {"TPU_AGENT_SUITE_HOME": str(self.home)})
        self._env.start()
        self.out = pathlib.Path(self._tmp.name) / "release"
        self.env = config.DeployEnv(namespace="llmd-study",
                                    tpu_accelerator="tpu7x",
                                    tpu_topology="2x2x1")
        _render(self.out)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_classify_path(self):
        self.assertEqual(release.classify_path("router.profile"), "cold")
        self.assertEqual(
            release.classify_path("router.plugins.queue-scorer.weight"),
            "hot")
        self.assertEqual(release.classify_path("engine.max_num_seqs"), "warm")
        self.assertEqual(release.classify_path("kv.block_size"), "warm")
        self.assertEqual(release.classify_path("topology.decode.replicas"),
                         "cold")
        self.assertEqual(release.classify_path("router.flow_control"), "cold")

    def test_apply_files_a_release_record_and_installs_router_last(self):
        runner = FakeRunner()
        result = release.apply(self.env, self.out, run_id="run-1",
                               runner=runner)
        self.assertEqual(result["chips_total"], 12)
        self.assertTrue(result["job_id"])
        applies = runner.matching("apply")
        self.assertEqual(len(applies), 2)  # model servers, monitoring
        helm = runner.matching("helm", "upgrade", "--install")
        self.assertEqual(len(helm), 1)
        self.assertIn(pins.ROUTER_STANDALONE_CHART, helm[0])
        # The router is installed after the servers were applied.
        self.assertGreater(runner.calls.index(helm[0]),
                           runner.calls.index(applies[-1]))
        rel = render.load_release(self.out)
        self.assertEqual(rel["state"], "applied")
        record = config.find_release_record("acme-r1")
        self.assertIsNotNone(record)
        self.assertEqual(record.backend, config.RELEASE_BACKEND)
        self.assertEqual(record.extra["release_dir"], str(self.out))

    def test_apply_refuses_study_render_as_production(self):
        with self.assertRaises(DeployError) as ctx:
            release.apply(self.env, self.out, run_id="run-1",
                          production=True, runner=FakeRunner())
        self.assertEqual(ctx.exception.error_code, "NOT_PRODUCTION_RENDER")

    def test_reconfigure_hot_warm_and_cold(self):
        runner = FakeRunner()
        release.apply(self.env, self.out, run_id="run-1", runner=runner)
        before = render.load_release(self.out)["config_digest"]

        with self.assertRaises(DeployError) as ctx:
            release.reconfigure(self.env, self.out,
                                {"topology.decode.replicas": 4},
                                runner=runner)
        self.assertEqual(ctx.exception.error_code, "COLD_PATH")
        with self.assertRaises(DeployError) as ctx:
            release.reconfigure(self.env, self.out,
                                {"router.profile": "load-only"},
                                runner=runner)
        self.assertEqual(ctx.exception.error_code, "COLD_PATH")

        runner.calls.clear()
        hot = release.reconfigure(
            self.env, self.out,
            {"router.plugins.token-load-scorer.weight": 2.0}, runner=runner)
        self.assertEqual(hot["classes"],
                         {"router.plugins.token-load-scorer.weight": "hot"})
        self.assertTrue(runner.matching("helm", "upgrade"))
        self.assertFalse(runner.matching("kubectl", "apply"))
        self.assertNotEqual(hot["config_digest"], before)

        runner.calls.clear()
        warm = release.reconfigure(self.env, self.out,
                                   {"engine.max_num_seqs": 512},
                                   runner=runner)
        self.assertTrue(warm["needs_wait"])
        self.assertTrue(runner.matching("kubectl", "apply"))
        cfg = json.loads((self.out / "llmd-config.json").read_text())
        self.assertEqual(cfg["engine"]["max_num_seqs"], 512)
        self.assertEqual(
            cfg["router"]["plugins"]["token-load-scorer"]["weight"], 2.0)

    def test_teardown_removes_by_label_and_marks_record(self):
        runner = FakeRunner()
        applied = release.apply(self.env, self.out, run_id="run-1",
                                runner=runner)
        runner.calls.clear()
        result = release.teardown(self.env, self.out, runner=runner)
        deletes = runner.matching("kubectl", "delete")
        self.assertTrue(deletes)
        for call in deletes:
            self.assertIn(f"{pins.RELEASE_LABEL}=acme-r1", call)
        self.assertTrue(runner.matching("helm", "uninstall"))
        self.assertIn("name", result)
        self.assertEqual(render.load_release(self.out)["state"], "torn_down")
        from framework.compute.drivers import jobs
        self.assertEqual(
            jobs.JobRecord.load(applied["job_id"]).status, "CANCELLED")

    def test_teardown_guards_production(self):
        prod = pathlib.Path(self._tmp.name) / "prod"
        profile = workload.resolve_workload(copy.deepcopy(SPEC))
        env = dict(ENV_SUBSET, namespace="serving")
        render.render_release(name="acme-prod",
                              config=render.default_config(RECIPE, replicas=2),
                              recipe=RECIPE, profile=profile, env=env,
                              out_dir=prod, production=True,
                              now="2026-10-01T00:00:00+00:00")
        runner = FakeRunner()
        release.apply(self.env, prod, run_id="run-1", production=True,
                      runner=runner)
        with self.assertRaises(DeployError) as ctx:
            release.teardown(self.env, prod, runner=runner)
        self.assertEqual(ctx.exception.error_code, "PRODUCTION_GUARD")
        release.teardown(self.env, prod, runner=runner,
                         include_production=True)

    def test_wrap_reap_resolver_only_intercepts_release_backend(self):
        sentinel = object()

        def resolve(backend_override=None, **kwargs):
            return sentinel

        wrapped = release.wrap_reap_resolver(resolve,
                                             env_loader=lambda *a: self.env)
        self.assertIs(wrapped(backend_override="gke_job"), sentinel)
        reaper = wrapped(backend_override=config.RELEASE_BACKEND)
        self.assertIsInstance(reaper, release.ReleaseReaper)
        with self.assertRaises(DeployError) as ctx:
            reaper.submit()
        self.assertEqual(ctx.exception.error_code, "INVALID_BACKEND")


def _lifecycle_doc(*, ok: int, bad: int, ttft_s: float, tpot_s: float,
                   rps: float) -> dict:
    return {
        "successes": {
            "count": ok,
            "latency": {
                "time_to_first_token": {"p90": ttft_s},
                "time_per_output_token": {"p90": tpot_s},
            },
            "throughput": {
                "requests_per_sec": rps,
                "total_tokens_per_sec": rps * 4300,
                "output_tokens_per_sec": rps * 300,
            },
        },
        "failures": {"count": bad},
    }


class BenchTest(unittest.TestCase):

    def test_parse_report_logs_handles_prefixed_lines(self):
        text = "\n".join([
            "[pod/x/c] starting",
            f"[pod/x/c] {render.BENCH_REPORT_MARK} summary_lifecycle_metrics.json",
            '[pod/x/c] {"successes": {"count": 3},',
            '[pod/x/c]  "failures": {"count": 0}}',
            f"[pod/x/c] {render.BENCH_REPORT_END}",
            "done",
        ])
        reports = bench.parse_report_logs(text)
        self.assertEqual(list(reports), ["summary_lifecycle_metrics.json"])
        self.assertEqual(reports["summary_lifecycle_metrics.json"]
                         ["successes"]["count"], 3)

    def test_score_gates_match_protocol_names_and_per_chip_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rel, _ = _render(out)
            profile = json.loads((out / "workload_profile.json").read_text())
            tenants = {t["name"]: t for t in profile["tenants"]}
            good = {
                "interactive": bench.summarize_report(
                    _lifecycle_doc(ok=1000, bad=0, ttft_s=0.3, tpot_s=0.02,
                                   rps=84.0), tenants["interactive"], 0.001),
                "batch": bench.summarize_report(
                    _lifecycle_doc(ok=400, bad=0, ttft_s=2.0, tpot_s=0.05,
                                   rps=36.0), tenants["batch"], 0.001),
            }
            scored = bench.score(rel, profile, [good, good, good])
            self.assertTrue(scored["admissible"])
            names = {g["name"] for g in scored["gates"]}
            self.assertEqual(names, {
                "interactive.ttft_p90_ms", "interactive.tpot_p90_ms",
                "interactive.error_rate", "batch.ttft_p90_ms",
                "batch.tpot_p90_ms", "batch.error_rate", "budget.max_chips"
            })
            self.assertTrue(all(g["same_execution"] for g in scored["gates"]))
            per_chip = scored["metrics"]["goodput_at_slo_per_chip"]
            self.assertAlmostEqual(per_chip["value"], 120.0 / 12)
            self.assertEqual(len(per_chip["replicates"]), 3)
            self.assertIsNone(scored["metrics"]["cost_per_1M_tokens_at_slo"])

            # One tenant over its TPOT bound: inadmissible, goodput drops.
            slow_batch = dict(good)
            slow_batch["batch"] = bench.summarize_report(
                _lifecycle_doc(ok=400, bad=0, ttft_s=2.0, tpot_s=0.09,
                               rps=36.0), tenants["batch"], 0.001)
            scored2 = bench.score(rel, profile, [slow_batch] * 3)
            self.assertFalse(scored2["admissible"])
            failed = {g["name"] for g in scored2["gates"] if not g["passed"]}
            self.assertEqual(failed, {"batch.tpot_p90_ms"})
            self.assertLess(scored2["metrics"]["goodput_at_slo_per_chip"]["value"],
                            per_chip["value"])

    def test_run_bench_writes_a_report_the_schema_accepts(self):
        doc = _lifecycle_doc(ok=1000, bad=0, ttft_s=0.3, tpot_s=0.02,
                             rps=60.0)
        logs = "\n".join([
            f"{render.BENCH_REPORT_MARK} summary_lifecycle_metrics.json",
            json.dumps(doc),
            render.BENCH_REPORT_END,
        ])
        runner = FakeRunner(json_docs={"get job": {"status": {"succeeded": 1}}},
                            logs=logs)
        env = config.DeployEnv(namespace="llmd-study")
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "release"
            _render(out)
            report = bench.run_bench(env, out, stage="slo", replicates=2,
                                     out_dir=pathlib.Path(tmp) / "bench",
                                     execution_id="exec-1", timeout_s=30,
                                     runner=runner, sleep=lambda s: None,
                                     clock=lambda: 0.0)
            self.assertEqual(report["execution_id"], "exec-1")
            self.assertEqual(report["replicates"], 2)
            self.assertTrue(report["telemetry_valid"])
            self.assertTrue(report["admissible"])
            # Two tenants x two replicates: a ConfigMap and a Job each time,
            # and each Job deleted afterwards so a rerun starts clean.
            self.assertEqual(len(runner.matching("kubectl", "apply")), 8)
            self.assertEqual(len(runner.matching("kubectl", "delete", "job")),
                             4)
            schema = json.loads(
                (_NEXUS / "package" / "schemas" /
                 "llmd-bench-report.json").read_text(encoding="utf-8"))
            self.assertEqual(utils.validate_instance(report, schema), [])
            written = json.loads((pathlib.Path(tmp) / "bench" /
                                  "llmd-bench-report.json").read_text())
            self.assertEqual(written["execution_id"], "exec-1")
            history = render.load_release(out)["history"]
            self.assertEqual(history[-1]["action"], "bench")
            self.assertEqual(history[-1]["execution_id"], "exec-1")

    def test_run_bench_with_a_failed_job_is_not_telemetry_valid(self):
        runner = FakeRunner(json_docs={"get job": {"status": {"failed": 1}}})
        env = config.DeployEnv(namespace="llmd-study")
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "release"
            _render(out)
            report = bench.run_bench(env, out, stage="slo", replicates=1,
                                     out_dir=pathlib.Path(tmp) / "bench",
                                     timeout_s=30, runner=runner,
                                     sleep=lambda s: None, clock=lambda: 0.0)
            self.assertFalse(report["telemetry_valid"])
            self.assertFalse(report["admissible"])
            self.assertEqual(len(report["job_failures"]), 2)


class ProbeTest(unittest.TestCase):

    def test_run_probe_writes_capabilities_the_schema_accepts(self):
        from framework.deploy import probe
        payload = {
            "capabilities": {"pd_connector": True, "offload_connector": True,
                             "kv_events": True, "tpu_inference": True,
                             "vllm": True},
            "versions": {"vllm": "0.29.0", "tpu_inference": "0.12.0"},
            "modules": {"pd_connector": "tpu_inference.distributed.tpu_connector"},
        }
        logs = "\n".join([probe.PROBE_MARK, json.dumps(payload), probe.PROBE_MARK])
        runner = FakeRunner(json_docs={"get job": {"status": {"succeeded": 1}}},
                            logs=logs)
        env = config.DeployEnv(namespace="llmd-study", tpu_accelerator="tpu7x",
                               tpu_topology="2x2x1")
        with tempfile.TemporaryDirectory() as tmp:
            doc = probe.run_probe(env, image=RECIPE["image"],
                                  out_dir=pathlib.Path(tmp), runner=runner,
                                  sleep=lambda s: None)
            self.assertEqual(doc["outcome"], "succeeded")
            self.assertTrue(doc["capabilities"]["pd_connector"])
            schema = json.loads(
                (_NEXUS / "package" / "schemas" /
                 "engine-capabilities.json").read_text(encoding="utf-8"))
            self.assertEqual(utils.validate_instance(doc, schema), [])
            self.assertTrue((pathlib.Path(tmp) /
                             "engine-capabilities.json").is_file())
            # The probe Job asked for one slice of the configured topology.
            applied = json.loads(runner.stdins[0])
            container = applied["spec"]["template"]["spec"]["containers"][0]
            self.assertEqual(container["image"], RECIPE["image"])
            self.assertEqual(container["resources"]["limits"][pins.TPU_RESOURCE],
                             4)
            self.assertTrue(runner.matching("kubectl", "delete", "job"))


class CliTest(unittest.TestCase):

    def test_package_given_at_deploy_level_survives_verb_parsing(self):
        parser = argparse.ArgumentParser(prog="runtime")
        sub = parser.add_subparsers(dest="cmd")
        cli.register_deploy_parser(sub)
        # This is where the compiler injects `--package`.
        args = parser.parse_args([
            "deploy", "--package", "pkg@1.0", "render", "--name", "r",
            "--recipe", "recipe.json", "--workload", "spec.json", "--out",
            "/tmp/x"
        ])
        self.assertEqual(args.package, "pkg@1.0")
        self.assertEqual(args.deploy_action, "render")
        args = parser.parse_args(["deploy", "workload", "catalog"])
        self.assertIsNone(getattr(args, "package", None))

    def test_runtime_registers_deploy(self):
        parser = runtime.build_runtime_parser()
        args = parser.parse_args(["deploy", "status", "--name", "r"])
        self.assertEqual(args.subcommand, "deploy")
        self.assertEqual(args.deploy_action, "status")

    def test_workload_catalog_needs_no_cluster(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            with mock.patch.object(cli, "load_env",
                                   return_value=config.DeployEnv()):
                out = cli.dispatch(argparse.Namespace(
                    deploy_action="workload", workload_action="catalog",
                    spec=None, out=None))
        self.assertEqual(out["inference_perf_ref"], pins.INFERENCE_PERF_REF)
        self.assertIn("interactive-chat", out["entries"])


if __name__ == "__main__":
    unittest.main()
