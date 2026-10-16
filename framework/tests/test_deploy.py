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
from framework.deploy import (bench, cli, config, pins, preflight,  # noqa: E402
                              release, render, workload)
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

    def test_unknown_plugin_and_malformed_features_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)

            def unknown(cfg):
                cfg["router"]["plugins"] = {"made-up-scorer": {"weight": 1}}

            with self.assertRaises(DeployError):
                _render(out / "a", config_edit=unknown)

            def pd_without_prefill(cfg):
                cfg["topology"]["mode"] = "disaggregated"

            with self.assertRaises(DeployError) as ctx:
                _render(out / "b", config_edit=pd_without_prefill)
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

            def pd_and_offload(cfg):
                cfg["topology"]["mode"] = "disaggregated"
                cfg["topology"]["prefill"] = {"replicas": 1, "tp": 4}
                cfg["kv"]["tiered_host"] = {"cpu_chunks": 1000}

            with self.assertRaises(DeployError) as ctx:
                _render(out / "b2", config_edit=pd_and_offload)
            self.assertEqual(ctx.exception.error_code, "NOT_IMPLEMENTED")

            def hot_path_without_epp(cfg):
                cfg["router"]["profile"] = "passthrough"
                cfg["router"]["flow_control"] = {"max_requests": 10}

            with self.assertRaises(DeployError) as ctx:
                _render(out / "b3", config_edit=hot_path_without_epp)
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

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


class AdvancedArmRenderTest(unittest.TestCase):
    """Each advanced arm renders exactly the shape its llm-d guide documents."""

    def _rendered(self, config_edit, recipe=None, env_extra=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = pathlib.Path(tmp.name)
        profile = workload.resolve_workload(copy.deepcopy(SPEC))
        recipe = recipe or RECIPE
        cfg = render.default_config(recipe, replicas=2)
        config_edit(cfg)
        env = dict(ENV_SUBSET)
        env.update(env_extra or {})
        rel = render.render_release(name="adv", config=cfg, recipe=recipe,
                                    profile=profile, env=env, out_dir=out,
                                    now="2026-10-01T00:00:00+00:00")
        return out, rel, cfg

    def _epp(self, out):
        values = json.loads((out / "router.values.json").read_text())
        return (values,
                json.loads(values["router"]["epp"]["pluginsCustomConfig"]
                           [render.EPP_CONFIG_KEY]))

    def _modelserver(self, out):
        doc = json.loads((out / render.MODELSERVER_FILE).read_text())
        return {i["metadata"]["name"]: i for i in doc["items"]
                if i["kind"] == "Deployment"}

    def test_per_role_engine_overlays_merge_over_the_fleet_default(self):
        def edit(cfg):
            cfg["topology"]["mode"] = "disaggregated"
            cfg["topology"]["prefill"] = {
                "replicas": 1, "tp": 4, "slice": "2x2x1",
                "engine": {"max_num_batched_tokens": 16384,
                           "max_num_seqs": None}}
            cfg["topology"]["decode"]["engine"] = {"max_num_seqs": 512}
            cfg["engine"] = {"max_num_seqs": 128,
                             "max_num_batched_tokens": 4096}
        out, _, cfg = self._rendered(edit)
        pools = self._modelserver(out)
        d_args = pools["adv-decode"]["spec"]["template"]["spec"][
            "containers"][0]["args"]
        p_args = pools["adv-prefill"]["spec"]["template"]["spec"][
            "containers"][0]["args"]
        # Decode: overlay wins for max_num_seqs, fleet default for the rest.
        self.assertIn("--max-num-seqs=512", d_args)
        self.assertIn("--max-num-batched-tokens=4096", d_args)
        # Prefill: overlay wins for batched tokens; null removes the fleet
        # default so the recipe's own `--max-num-seqs=128` applies again.
        self.assertIn("--max-num-batched-tokens=16384", p_args)
        self.assertIn("--max-num-seqs=128", p_args)
        self.assertNotIn("--max-num-seqs=512", p_args)
        self.assertEqual(render.role_engine(cfg, "decode"),
                         {"max_num_seqs": 512, "max_num_batched_tokens": 4096})
        self.assertEqual(render.role_engine(cfg, "prefill"),
                         {"max_num_batched_tokens": 16384})

    def test_per_role_overlay_is_whitelisted_and_needs_its_pool(self):
        with self.assertRaises(DeployError) as ctx:
            self._rendered(lambda cfg: cfg["topology"]["decode"].update(
                {"engine": {"speculative_config": {}}}))
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertIn("topology.decode.engine", str(ctx.exception))
        with self.assertRaises(DeployError) as ctx:
            self._rendered(lambda cfg: cfg["topology"].update(
                {"prefill": {"replicas": 1, "engine": {"max_num_seqs": 8}}}))
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_pd_renders_two_pools_sidecar_and_disagg_profiles(self):
        def edit(cfg):
            cfg["topology"]["mode"] = "disaggregated"
            cfg["topology"]["prefill"] = {"replicas": 1, "tp": 4,
                                          "slice": "2x2x1"}
        out, rel, _ = self._rendered(edit)
        pools = self._modelserver(out)
        self.assertEqual(set(pools), {"adv-decode", "adv-prefill"})
        self.assertEqual(rel["deployments"], ["adv-decode", "adv-prefill"])
        # 2 decode + 1 prefill replicas of 4 chips each.
        self.assertEqual(rel["chips"]["total"], 12)
        decode = pools["adv-decode"]["spec"]["template"]["spec"]
        prefill = pools["adv-prefill"]["spec"]["template"]["spec"]
        # Decode sits behind the routing sidecar on the model-server port.
        (sidecar,) = decode["initContainers"]
        self.assertEqual(sidecar["restartPolicy"], "Always")
        self.assertIn(f"--port={pins.MODEL_SERVER_PORT}", sidecar["args"])
        self.assertIn("--kv-connector=nixlv2", sidecar["args"])
        self.assertTrue(sidecar["image"].startswith(pins.PD_SIDECAR_IMAGE))
        self.assertNotIn("initContainers", prefill)
        d_args = decode["containers"][0]["args"]
        p_args = prefill["containers"][0]["args"]
        self.assertIn(f"--port={pins.PD_DECODE_PORT}", d_args)
        self.assertIn(f"--port={pins.MODEL_SERVER_PORT}", p_args)
        d_kv = json.loads([a for a in d_args
                           if a.startswith("--kv-transfer-config=")][0]
                          .split("=", 1)[1])
        p_kv = json.loads([a for a in p_args
                           if a.startswith("--kv-transfer-config=")][0]
                          .split("=", 1)[1])
        self.assertEqual(d_kv["kv_role"], "kv_consumer")
        self.assertEqual(p_kv["kv_role"], "kv_producer")
        self.assertEqual(p_kv["kv_connector_module_path"],
                         pins.TPU_PD_MODULE_TPU_INFERENCE)
        self.assertEqual(p_kv["kv_ip"], "$(POD_IP)")
        env_names = {e["name"] for e in prefill["containers"][0]["env"]}
        self.assertTrue({"POD_IP", "TPU_SIDE_CHANNEL_PORT",
                         "TPU_KV_TRANSFER_PORT"} <= env_names)
        labels = pools["adv-prefill"]["spec"]["template"]["metadata"]["labels"]
        self.assertEqual(labels[pins.LLMD_ROLE_LABEL], "prefill")
        values, epp = self._epp(out)
        types = [p["type"] for p in epp["plugins"]]
        self.assertIn("disagg-profile-handler", types)
        self.assertIn("always-disagg-pd-decider", types)
        self.assertIn("prefill-filter", types)
        self.assertIn("decode-filter", types)
        self.assertNotIn("pd-profile-handler", types)
        names = [p["name"] for p in epp["schedulingProfiles"]]
        self.assertEqual(names, ["prefill", "decode"])
        # The router selects both pools by the serving label.
        self.assertEqual(values["router"]["modelServers"]["matchLabels"],
                         {pins.RELEASE_LABEL: "adv",
                          pins.LLMD_SERVING_LABEL: "true"})
        # The Service still fronts decode, so passthrough works under P/D.
        doc = json.loads((out / render.MODELSERVER_FILE).read_text())
        svc = [i for i in doc["items"] if i["kind"] == "Service"][0]
        self.assertEqual(svc["spec"]["selector"][pins.LLMD_ROLE_LABEL],
                         "decode")
        self.assertEqual(svc["spec"]["ports"][0]["targetPort"],
                         pins.MODEL_SERVER_PORT)

    def test_pd_uses_the_torchtpu_connector_when_the_recipe_says_so(self):
        recipe = dict(RECIPE, engine_flavor="torchtpu")

        def edit(cfg):
            cfg["topology"]["mode"] = "disaggregated"
            cfg["topology"]["prefill"] = {"replicas": 1}
        out, _, _ = self._rendered(edit, recipe=recipe)
        pools = self._modelserver(out)
        args = pools["adv-prefill"]["spec"]["template"]["spec"][
            "containers"][0]["args"]
        kv = json.loads([a for a in args
                         if a.startswith("--kv-transfer-config=")][0]
                        .split("=", 1)[1])
        self.assertEqual(kv["kv_connector_module_path"],
                         pins.TPU_PD_MODULE_TORCHTPU)
        self.assertEqual(kv["kv_connector"], pins.TPU_PD_CONNECTOR_TORCHTPU)

    def test_precise_routing_renders_kv_events_and_precise_producer(self):
        def edit(cfg):
            cfg["kv"]["precise_routing"] = True
            cfg["kv"]["block_size"] = 64
        out, _, _ = self._rendered(edit)
        pools = self._modelserver(out)
        c = pools["adv-decode"]["spec"]["template"]["spec"]["containers"][0]
        ev = json.loads([a for a in c["args"]
                         if a.startswith("--kv-events-config=")][0]
                        .split("=", 1)[1])
        self.assertTrue(ev["enable_kv_cache_events"])
        self.assertEqual(ev["publisher"], "zmq")
        self.assertEqual(ev["endpoint"], f"tcp://*:{pins.KV_EVENTS_PORT}")
        self.assertEqual(ev["replay_endpoint"],
                         f"tcp://*:{pins.KV_EVENTS_REPLAY_PORT}")
        self.assertTrue(ev["topic"].startswith("kv@$(POD_IP):8000@"))
        self.assertIn("--enable-scale-out", c["args"])
        self.assertIn("--block-size=64", c["args"])
        port_names = {p["name"] for p in c["ports"]}
        self.assertTrue({"kv-events", "kv-replay"} <= port_names)
        values, epp = self._epp(out)
        by_type = {p["type"]: p for p in epp["plugins"]}
        self.assertNotIn("approx-prefix-cache-producer", by_type)
        precise = by_type["precise-prefix-cache-producer"]["parameters"]
        self.assertEqual(precise["tokenProcessorConfig"]["blockSizeTokens"], 64)
        disc = precise["kvEventsConfig"]["podDiscoveryConfig"]
        self.assertEqual(disc["socketPort"], pins.KV_EVENTS_PORT)
        self.assertEqual(disc["replaySocketPort"], pins.KV_EVENTS_REPLAY_PORT)
        tok = by_type["token-producer"]["parameters"]
        self.assertEqual(tok["backend"], "vllm")
        self.assertEqual(tok["modelName"], RECIPE["model"])
        self.assertEqual(tok["vllm"]["url"],
                         "http://adv-modelserver.llmd-study.svc.cluster.local"
                         ":8000")
        self.assertEqual(
            by_type["prefix-cache-affinity-filter"]["parameters"]
            ["prefixMatchInfoProducerName"], "precise-prefix-cache-producer")

    def test_tiered_host_offload_renders_connector_and_env(self):
        def edit(cfg):
            cfg["kv"]["tiered_host"] = {"cpu_chunks": 25000,
                                        "staging_blocks": 1000}
        out, _, _ = self._rendered(edit)
        pools = self._modelserver(out)
        c = pools["adv-decode"]["spec"]["template"]["spec"]["containers"][0]
        kv = json.loads([a for a in c["args"]
                         if a.startswith("--kv-transfer-config=")][0]
                        .split("=", 1)[1])
        self.assertEqual(kv["kv_connector"],
                         pins.TPU_OFFLOAD_CONNECTOR_TPU_INFERENCE)
        self.assertEqual(kv["kv_connector_module_path"],
                         pins.TPU_OFFLOAD_MODULE_TPU_INFERENCE)
        self.assertEqual(kv["kv_role"], "kv_both")
        env = {e["name"]: e.get("value") for e in c["env"]}
        self.assertEqual(env[pins.TPU_OFFLOAD_CPU_CHUNKS_ENV], "25000")
        self.assertEqual(env[pins.TPU_OFFLOAD_STAGING_BLOCKS_ENV], "1000")

    def test_torchtpu_offload_passes_extra_config_verbatim(self):
        recipe = dict(RECIPE, engine_flavor="torchtpu")

        def edit(cfg):
            cfg["kv"]["tiered_host"] = {
                "extra_config": {"cpu_bytes_to_use": 2 ** 36}}
        out, _, _ = self._rendered(edit, recipe=recipe)
        pools = self._modelserver(out)
        c = pools["adv-decode"]["spec"]["template"]["spec"]["containers"][0]
        kv = json.loads([a for a in c["args"]
                         if a.startswith("--kv-transfer-config=")][0]
                        .split("=", 1)[1])
        self.assertEqual(kv["kv_connector_module_path"],
                         pins.TPU_OFFLOAD_MODULE_TORCHTPU)
        self.assertEqual(kv["kv_connector_extra_config"],
                         {"cpu_bytes_to_use": 2 ** 36})

    def test_flow_control_sets_feature_gate_bands_and_detector(self):
        def edit(cfg):
            cfg["router"]["flow_control"] = {
                "saturation_detector": {"type": "utilization-detector",
                                        "queueDepthThreshold": 8,
                                        "kvCacheUtilThreshold": 0.9},
                "max_requests": 2000,
                "default_request_ttl": "30s",
                "priority_bands": [{"priority": 10, "max_requests": 500},
                                   {"priority": -1}],
            }
        out, _, _ = self._rendered(edit)
        values, epp = self._epp(out)
        self.assertEqual(epp["featureGates"], ["flowControl"])
        fc = epp["flowControl"]
        self.assertEqual(fc["saturationDetector"]["pluginRef"],
                         "utilization-detector")
        self.assertEqual(fc["maxRequests"], 2000)
        self.assertEqual(fc["defaultRequestTTL"], "30s")
        self.assertEqual([b["priority"] for b in fc["priorityBands"]],
                         [10, -1])
        self.assertEqual(fc["priorityBands"][0]["maxRequests"], 500)
        det = [p for p in epp["plugins"]
               if p["type"] == "utilization-detector"][0]
        self.assertEqual(det["parameters"]["queueDepthThreshold"], 8)
        # Tenant priorities become InferenceObjectives, named per release
        # so two routers can share a namespace.
        self.assertEqual(values["router"]["inferenceObjectives"],
                         [{"name": "adv-interactive", "priority": 10},
                          {"name": "adv-batch", "priority": -5}])
        slo_cm = json.loads(
            (out / "bench/interactive.slo.configmap.json").read_text())
        ip_cfg = json.loads(slo_cm["data"]["config.json"])
        self.assertEqual(ip_cfg["api"]["headers"][workload.OBJECTIVE_HEADER],
                         "adv-interactive")

    def test_predicted_latency_adds_plugins_and_predictor(self):
        def edit(cfg):
            cfg["router"]["predicted_latency"] = True
        out, _, _ = self._rendered(edit)
        values, epp = self._epp(out)
        types = [p["type"] for p in epp["plugins"]]
        for t in ("predicted-latency-producer", "latency-scorer",
                  "slo-headroom-tier-filter", "latency-slo-admitter"):
            self.assertIn(t, types)
        refs = [p["pluginRef"] for p in epp["schedulingProfiles"][0]["plugins"]]
        self.assertEqual(refs[0], "slo-headroom-tier-filter")
        self.assertTrue(values["router"]["latencyPredictor"]["enabled"])
        self.assertTrue(values["router"]["epp"]["flags"]
                        ["allow-experimental-plugins"])

    def test_epp_ha_and_proxy_mode_and_sizing(self):
        def edit(cfg):
            cfg["router"]["epp_replicas"] = 2
            cfg["router"]["epp_ha_mode"] = "active-active"
            cfg["router"]["proxy_mode"] = "service"
            cfg["router"]["proxy_replicas"] = 2
        sizing = {"epp": {"requests": {"cpu": "1", "memory": "1Gi"}},
                  "proxy": {"requests": {"cpu": 1, "memory": "1Gi"}},
                  "bench": {"requests": {"cpu": "2", "memory": "3Gi"},
                            "limits": {"memory": "6Gi"}}}
        out, _, _ = self._rendered(edit, env_extra={"sizing": sizing})
        values, _ = self._epp(out)
        epp = values["router"]["epp"]
        self.assertEqual(epp["replicas"], 2)
        self.assertIs(epp["flags"]["ha-enable-leader-election"], False)
        self.assertEqual(epp["resources"]["requests"]["cpu"], "1")
        self.assertEqual(epp["resources"]["limits"]["memory"], "16Gi")
        proxy = values["router"]["proxy"]
        self.assertEqual(proxy["mode"], "service")
        self.assertEqual(proxy["replicas"], 2)
        self.assertEqual(proxy["resources"]["requests"]["cpu"], "1")
        job = json.loads((out / "bench/interactive.slo.job.json").read_text())
        res = job["spec"]["template"]["spec"]["containers"][0]["resources"]
        self.assertEqual(res["requests"]["cpu"], "2")
        self.assertEqual(res["limits"]["memory"], "6Gi")

    def test_active_passive_ha_leaves_leader_election_to_the_chart(self):
        def edit(cfg):
            cfg["router"]["epp_replicas"] = 2
        out, _, _ = self._rendered(edit)
        values, _ = self._epp(out)
        self.assertNotIn("ha-enable-leader-election",
                         values["router"]["epp"]["flags"])

    def test_envoy_configmap_is_named_per_release(self):
        # The chart's preset names it a bare `envoy`; installing the
        # production router next to a study arm failed Helm's ownership
        # check on it live. The name is the only field overridden so the
        # preset's envoy.yaml still applies.
        out, _, _ = self._rendered(lambda cfg: None)
        values, _ = self._epp(out)
        proxy = values["router"]["proxy"]
        # The preset path is what the chart's volume mount reads; a
        # top-level `configMap` would rename the object but not the mount.
        self.assertEqual(proxy["presets"]["envoy"]["configMap"],
                         {"name": render.proxy_configmap_name("adv")})
        self.assertNotIn("configMap", proxy)
        self.assertNotEqual(proxy["presets"]["envoy"]["configMap"]["name"],
                            "envoy")
        self.assertNotEqual(render.proxy_configmap_name("a"),
                            render.proxy_configmap_name("b"))

    def test_every_advanced_arm_is_refused_on_an_adopted_fleet(self):
        edits = {
            "topology.mode": lambda c: c["topology"].update(
                {"mode": "disaggregated", "prefill": {"replicas": 1}}),
            "kv.precise_routing": lambda c: c["kv"].update(
                {"precise_routing": True}),
            "kv.tiered_host": lambda c: c["kv"].update(
                {"tiered_host": {"cpu_chunks": 100}}),
            "engine": lambda c: c["engine"].update({"max_num_seqs": 64}),
        }
        for path, edit in edits.items():
            with self.subTest(path=path):
                def full_edit(cfg, edit=edit):
                    cfg["router"]["profile"] = "sticky-until-saturated"
                    edit(cfg)
                with self.assertRaises(DeployError) as ctx:
                    self._rendered(full_edit, recipe=ADOPTED_RECIPE)
                self.assertEqual(ctx.exception.error_code,
                                 "ADOPTED_FLEET_IMMUTABLE")

    def test_flow_control_is_a_hot_path_on_an_adopted_fleet(self):
        def edit(cfg):
            cfg["router"]["profile"] = "sticky-until-saturated"
            cfg["router"]["flow_control"] = {"max_requests": 100}
        out, rel, _ = self._rendered(edit, recipe=ADOPTED_RECIPE)
        self.assertTrue(rel["router"]["installed"])
        _, epp = self._epp(out)
        self.assertEqual(epp["featureGates"], ["flowControl"])


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
        # The disaggregation decider is EPP config; a pool's engine overlay
        # rolls that pool only.
        self.assertEqual(release.classify_path(
            "topology.pd_decider.parameters.nonCachedTokens"), "hot")
        self.assertEqual(release.classify_path(
            "topology.decode.engine.max_num_seqs"), "warm")
        self.assertEqual(release.classify_path(
            "topology.prefill.engine.max_num_batched_tokens"), "warm")
        self.assertEqual(release.classify_path("topology.prefill.replicas"),
                         "cold")

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


class PreflightTest(unittest.TestCase):
    """The router chart needs two CRD families; preflight checks both.

    Found live: a GKE cluster with the Inference Gateway's InferencePool CRD
    installed still fails `helm upgrade --install` of the v0.11.0 standalone
    chart with "no matches for kind InferenceObjective in version
    llm-d.ai/v1alpha2", because that CRD ships with llm-d-router, not GAIE.
    """

    def setUp(self):
        self.env = config.DeployEnv(namespace="llmd-study",
                                    tpu_accelerator="tpu7x",
                                    tpu_topology="2x2x1")

    def _run(self, runner, fix=False):
        with mock.patch.object(preflight, "tools_present", return_value={
                "kubectl": "/usr/bin/kubectl", "helm": "/usr/bin/helm",
                "gcloud": None}):
            return preflight.run_preflight(self.env, fix=fix, runner=runner)

    def test_missing_router_crd_fails_and_fix_applies_the_pinned_manifest(self):
        missing = {f"get crd {pins.INFERENCE_OBJECTIVE_CRD}": "NotFound"}
        report = self._run(FakeRunner(fail=missing))
        checks = {c["name"]: c for c in report["checks"]}
        self.assertEqual(checks["crd.inferencepool"]["status"], "ok")
        self.assertEqual(checks["crd.inferenceobjective"]["status"], "fail")
        self.assertIn(pins.ROUTER_CRD_URL,
                      checks["crd.inferenceobjective"]["fix"])
        self.assertIn("crd.inferenceobjective", report["failed"])

        runner = FakeRunner(fail=missing)
        report = self._run(runner, fix=True)
        self.assertEqual(report["fixed"],
                         [f"applied llm-d-router CRDs {pins.ROUTER_RELEASE_VERSION}"])
        self.assertTrue(runner.matching("apply", "-f", pins.ROUTER_CRD_URL))
        self.assertFalse(runner.matching("apply", "-f", pins.GAIE_CRD_URL))
        self.assertNotIn("crd.inferenceobjective", report["failed"])

    def test_router_crd_pin_follows_the_router_release(self):
        self.assertIn(pins.ROUTER_RELEASE_VERSION, pins.ROUTER_CRD_URL)
        self.assertTrue(pins.ROUTER_CRD_URL.endswith("/manifests.yaml"))


SIM_RECIPE = {
    "model": "sim/qwen3.5-397b",
    "image": f"{pins.SIM_IMAGE}:{pins.SIM_VERSION}",
    "engine_flavor": "sim",
    "args": ["--mode=random", "--latency-calculator=per-token",
             "--prefill-overhead=20ms", "--prefill-time-per-token=50us",
             "--inter-token-latency=20ms", "--max-num-seqs=64",
             "--enable-kvcache", "--kv-cache-size=4096",
             "--kv-transfer-config={}"],
    "accelerator": {"tpu_accelerator": "tpu7x", "tpu_topology": "2x2x1"},
    "tensor_parallel": 4,
    "multihost": {"worker_command": ["sleep", "infinity"]},
}


class MultiHostAndSimRenderTest(unittest.TestCase):
    """M5a: a replica may span hosts; a fleet may be simulated."""

    def _render(self, recipe, edit, name="mh"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = pathlib.Path(tmp.name)
        profile = workload.resolve_workload(copy.deepcopy(SPEC))
        cfg = render.default_config(recipe, replicas=2)
        edit(cfg)
        rel = render.render_release(name=name, config=cfg, recipe=recipe,
                                    profile=profile, env=dict(ENV_SUBSET),
                                    out_dir=out,
                                    now="2026-10-01T00:00:00+00:00")
        doc = json.loads((out / render.MODELSERVER_FILE).read_text())
        return out, rel, cfg, {(i["kind"], i["metadata"]["name"]): i
                               for i in doc["items"]}

    def test_multi_host_pool_renders_a_leader_worker_set(self):
        recipe = dict(RECIPE, multihost={
            "worker_command": ["ray", "start", "--block",
                               "--address=$(LWS_LEADER_ADDRESS):6379"],
            "env": {"TPU_MULTIHOST_BACKEND": "ray"}})
        recipe["accelerator"] = {"tpu_accelerator": "tpu7x",
                                 "tpu_topology": "2x2x2",
                                 "placement_policy": "pp-tpu7x-8"}

        def edit(cfg):
            cfg["topology"]["decode"]["hosts"] = 2
            cfg["topology"]["decode"]["slice"] = "2x2x2"
        out, rel, cfg, items = self._render(recipe, edit)
        lws = items[("LeaderWorkerSet", "mh-decode")]
        self.assertEqual(lws["apiVersion"], pins.LWS_API_VERSION)
        spec = lws["spec"]
        self.assertEqual(spec["replicas"], 2)
        self.assertEqual(spec["leaderWorkerTemplate"]["size"], 2)
        self.assertEqual(spec["leaderWorkerTemplate"]["restartPolicy"],
                         "RecreateGroupOnPodRestart")
        leader = spec["leaderWorkerTemplate"]["leaderTemplate"]
        worker = spec["leaderWorkerTemplate"]["workerTemplate"]
        # The leader serves and is the only member the selectors see.
        self.assertEqual(leader["metadata"]["labels"][pins.LLMD_ROLE_LABEL],
                         "decode")
        self.assertEqual(leader["metadata"]["labels"][pins.MEMBER_LABEL],
                         "leader")
        self.assertNotIn(pins.LLMD_ROLE_LABEL, worker["metadata"]["labels"])
        self.assertNotIn(pins.LLMD_SERVING_LABEL,
                         worker["metadata"]["labels"])
        self.assertEqual(worker["metadata"]["labels"][pins.MEMBER_LABEL],
                         "worker")
        # Every member requests all of its node's chips; the replica is
        # hosts x chips per host (tpu7x: 4).
        for tpl in (leader, worker):
            limits = tpl["spec"]["containers"][0]["resources"]["limits"]
            self.assertEqual(limits[pins.TPU_RESOURCE], 4)
            sel = tpl["spec"]["nodeSelector"]
            self.assertEqual(sel[pins.TOPOLOGY_LABEL], "2x2x2")
            self.assertEqual(sel[pins.PLACEMENT_POLICY_LABEL], "pp-tpu7x-8")
        self.assertEqual(rel["chips"]["total"], 16)
        self.assertEqual(worker["spec"]["containers"][0]["command"][0], "ray")
        env = {e["name"]: e.get("value") for e in
               worker["spec"]["containers"][0]["env"]}
        self.assertEqual(env["TPU_MULTIHOST_BACKEND"], "ray")
        self.assertEqual(rel["pools"], [{
            "name": "mh-decode", "role": "decode", "kind": "LeaderWorkerSet",
            "replicas": 2, "hosts": 2}])
        # The Service still selects by release + role, i.e. leaders only.
        svc = items[("Service", "mh-modelserver")]
        self.assertEqual(svc["spec"]["selector"],
                         {pins.RELEASE_LABEL: "mh",
                          pins.LLMD_ROLE_LABEL: "decode"})

    def test_multi_host_refuses_without_worker_command_or_host_size(self):
        with self.assertRaises(DeployError) as ctx:
            self._render(RECIPE, lambda c: c["topology"]["decode"].update(
                {"hosts": 2}))
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertIn("worker_command", str(ctx.exception))
        unknown = dict(RECIPE, multihost={"worker_command": ["sleep"]})
        unknown["accelerator"] = {"tpu_accelerator": "tpu9z",
                                  "tpu_topology": "2x2x2"}
        with self.assertRaises(DeployError) as ctx:
            self._render(unknown, lambda c: c["topology"]["decode"].update(
                {"hosts": 2}))
        self.assertIn("chips_per_host", str(ctx.exception))

    def test_sim_fleet_renders_the_simulator_without_accelerators(self):
        def edit(cfg):
            cfg["topology"]["mode"] = "disaggregated"
            cfg["topology"]["prefill"] = {"replicas": 1, "hosts": 2}
            cfg["topology"]["decode"]["engine"] = {"max_num_seqs": 128,
                                                   "max_model_len": 8192}
            cfg["engine"] = {"max_num_batched_tokens": 8192}
        out, rel, cfg, items = self._render(SIM_RECIPE, edit, name="simf")
        decode = items[("Deployment", "simf-decode")]
        prefill = items[("LeaderWorkerSet", "simf-prefill")]
        dc = decode["spec"]["template"]["spec"]["containers"][0]
        self.assertNotIn("command", dc)
        self.assertNotIn(pins.TPU_RESOURCE, dc["resources"]["limits"])
        self.assertNotIn("nodeSelector", decode["spec"]["template"]["spec"])
        args = dc["args"]
        self.assertIn(f"--model={SIM_RECIPE['model']}", args)
        self.assertIn(f"--port={pins.PD_DECODE_PORT}", args)
        self.assertIn("--render-url=", args)
        self.assertIn("--max-num-seqs=128", args)
        self.assertIn("--max-model-len=8192", args)
        self.assertIn("--enable-kvcache", args)
        # vLLM-only flags never reach the simulator (it aborts on them).
        for forbidden in ("--kv-transfer-config", "--tensor-parallel-size",
                          "--max-num-batched-tokens", "--kv-events-config"):
            self.assertFalse(any(a.startswith(forbidden) for a in args),
                             forbidden)
        # P/D still goes through the routing sidecar on decode.
        (sidecar,) = decode["spec"]["template"]["spec"]["initContainers"]
        self.assertIn("--kv-connector=nixlv2", sidecar["args"])
        # No TPU side-channel env or ports on a simulated pod; POD_IP is
        # always there (the simulator's KV cache needs it).
        names = {e["name"] for e in dc["env"]}
        self.assertNotIn("TPU_SIDE_CHANNEL_PORT", names)
        self.assertIn("POD_IP", names)
        self.assertEqual([p["name"] for p in dc["ports"]],
                         [render.MODELSERVER_CONTAINER])
        # The prefill group is an LWS of sim leader + sleeping worker.
        worker = prefill["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        self.assertEqual(worker["spec"]["containers"][0]["command"],
                         ["sleep", "infinity"])
        # Chip accounting stays virtual: 2 decode x 4 + 1 prefill x (2 x 4).
        self.assertEqual(rel["chips"]["total"], 16)

    def test_sim_refuses_precise_routing_and_host_tiering(self):
        for edit in (lambda c: c["kv"].update({"precise_routing": True}),
                     lambda c: c["kv"].update(
                         {"tiered_host": {"cpu_chunks": 10}})):
            with self.assertRaises(DeployError) as ctx:
                self._render(SIM_RECIPE, edit)
            self.assertEqual(ctx.exception.error_code, "NOT_IMPLEMENTED")

    def test_probe_short_circuits_for_the_simulator(self):
        from framework.deploy import probe
        env = config.DeployEnv(namespace="llmd-study")
        with tempfile.TemporaryDirectory() as tmp:
            doc = probe.run_probe(env, image=SIM_RECIPE["image"],
                                  out_dir=pathlib.Path(tmp),
                                  runner=FakeRunner(fail={"": "no cluster"}))
            schema = json.loads(
                (_NEXUS / "package" / "schemas" /
                 "engine-capabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(utils.validate_instance(doc, schema), [])
        self.assertEqual(doc["mode"], "sim")
        self.assertTrue(doc["capabilities"]["sim"])
        self.assertTrue(doc["capabilities"]["pd_connector"])
        self.assertFalse(doc["capabilities"]["kv_events"])


class ProductionTest(unittest.TestCase):
    """What a `--production` render adds and what the drill measures."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self._tmp.name) / "home"
        self.home.mkdir()
        self._env = mock.patch.dict(os.environ,
                                    {"TPU_AGENT_SUITE_HOME": str(self.home)})
        self._env.start()
        self.env = config.DeployEnv(namespace="serving",
                                    tpu_accelerator="tpu7x",
                                    tpu_topology="2x2x1")
        self.profile = workload.resolve_workload(copy.deepcopy(SPEC))

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _prod(self, name="acme-prod", exposure=None, epp_replicas=2,
              ha_mode=None, production=True):
        out = pathlib.Path(self._tmp.name) / name
        cfg = render.default_config(RECIPE, replicas=2)
        cfg["router"]["epp_replicas"] = epp_replicas
        if ha_mode:
            cfg["router"]["epp_ha_mode"] = ha_mode
        env = dict(ENV_SUBSET, namespace="serving")
        if exposure:
            env["exposure"] = exposure
        rel = render.render_release(name=name, config=cfg, recipe=RECIPE,
                                    profile=self.profile, env=env,
                                    out_dir=out, production=production,
                                    now="2026-10-01T00:00:00+00:00")
        return out, rel

    def test_exposure_is_rendered_for_production_only(self):
        out, rel = self._prod()
        self.assertEqual(rel["files"]["exposure"], render.EXPOSURE_FILE)
        self.assertEqual(rel["exposure"]["mode"], render.EXPOSURE_CLUSTER_IP)
        doc = json.loads((out / render.EXPOSURE_FILE).read_text())
        svc = doc["items"][0]
        self.assertEqual(svc["spec"]["type"], "ClusterIP")
        self.assertEqual(svc["spec"]["selector"],
                         {"llm-d-router-gateway": "acme-prod-epp"})
        self.assertEqual(svc["metadata"]["labels"][pins.PRODUCTION_LABEL],
                         "true")
        _, study = self._prod(name="acme-study", production=False)
        self.assertIsNone(study["files"]["exposure"])
        self.assertIsNone(study["exposure"])

    def test_internal_lb_exposure_carries_the_gke_annotation(self):
        out, rel = self._prod(name="acme-lb",
                              exposure=render.EXPOSURE_INTERNAL_LB)
        svc = json.loads((out / render.EXPOSURE_FILE).read_text())["items"][0]
        self.assertEqual(svc["spec"]["type"], "LoadBalancer")
        self.assertEqual(
            svc["metadata"]["annotations"][render.INTERNAL_LB_ANNOTATION],
            "Internal")
        with self.assertRaises(DeployError):
            render.exposure_manifests("x", "ns", "public")

    def test_epp_ha_min_ready_follows_leader_election(self):
        _, rel = self._prod()
        self.assertEqual(rel["router"]["epp_ha"],
                         {"replicas": 2, "mode": "active-passive",
                          "min_ready": 1})
        _, aa = self._prod(name="acme-aa", ha_mode="active-active")
        self.assertEqual(aa["router"]["epp_ha"]["min_ready"], 2)
        _, single = self._prod(name="acme-one", epp_replicas=1)
        self.assertEqual(single["router"]["epp_ha"],
                         {"replicas": 1, "mode": None, "min_ready": 1})

    def test_apply_creates_the_front_door_after_the_chart(self):
        out, _ = self._prod()
        runner = FakeRunner()
        release.apply(self.env, out, run_id="run-1", production=True,
                      runner=runner)
        helm = runner.matching("helm", "upgrade", "--install")[0]
        exposure_apply = [i for i, s in enumerate(runner.stdins)
                          if s and "acme-prod-ingress" in s]
        self.assertTrue(exposure_apply)
        self.assertGreater(exposure_apply[0], runner.calls.index(helm))

    def test_wait_accepts_one_ready_epp_under_leader_election(self):
        out, _ = self._prod()
        epp = {"metadata": {"generation": 1},
               "spec": {"replicas": 2},
               "status": {"observedGeneration": 1, "readyReplicas": 1,
                          "updatedReplicas": 2}}
        runner = FakeRunner(json_docs={
            "get deployment acme-prod-epp": epp,
            "get deployment acme-prod-decode": {
                "metadata": {"generation": 1}, "spec": {"replicas": 2},
                "status": {"observedGeneration": 1, "readyReplicas": 2,
                           "updatedReplicas": 2}},
        })
        result = release.wait(self.env, out, runner=runner, timeout_s=5,
                              poll_s=0, sleep=lambda s: None)
        self.assertEqual(result["state"], "ready")

    def test_drill_kills_the_ready_epp_and_reports_recovery(self):
        out, _ = self._prod()
        epp = {"metadata": {"generation": 1},
               "spec": {"replicas": 2},
               "status": {"observedGeneration": 1, "readyReplicas": 1,
                          "updatedReplicas": 2}}

        class DrillRunner(FakeRunner):
            def __init__(self):
                super().__init__(json_docs={"get deployment acme-prod-epp": epp})
                self.killed = False

            def __call__(self, cmd, *, stdin=None, timeout=None):
                joined = " ".join(cmd)
                if "delete pod" in joined:
                    self.killed = True
                if "get pods" in joined and "-o" in cmd:
                    self.calls.append(list(cmd))
                    pods = ([_ready_pod("epp-b", False), _ready_pod("epp-c")]
                            if self.killed else
                            [_ready_pod("epp-a"), _ready_pod("epp-b", False)])
                    return subprocess.CompletedProcess(
                        cmd, 0, json.dumps({"items": pods}), "")
                if cmd[-3:-1] == ["--max-time", "10"] or "curl" in joined:
                    self.calls.append(list(cmd))
                    return subprocess.CompletedProcess(
                        cmd, 0, '{"object":"list","data":[]}', "")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)

        runner = DrillRunner()
        clock = iter([0.0, 0.0, 4.0, 4.5, 5.0, 6.0, 7.0])
        row = release.drill(self.env, out, name=release.DRILL_EPP_POD_KILL,
                            runner=runner, timeout_s=60, poll_s=0,
                            sleep=lambda s: None,
                            clock=lambda: next(clock))
        self.assertEqual(row["killed"], "epp-a")
        self.assertTrue(row["recovered"])
        self.assertTrue(row["front_door_ok"])
        self.assertTrue(row["passed"])
        self.assertTrue(runner.matching("kubectl", "delete", "pod", "epp-a"))
        history = render.load_release(out)["history"]
        self.assertEqual(history[-1]["action"], "drill")
        with self.assertRaises(DeployError):
            release.drill(self.env, out, name="decode-pod-kill",
                          runner=runner)

    def test_drill_refuses_a_passthrough_release(self):
        out = pathlib.Path(self._tmp.name) / "pt"
        cfg = render.default_config(RECIPE, replicas=2)
        cfg["router"]["profile"] = render.PASSTHROUGH_PROFILE
        render.render_release(name="acme-pt", config=cfg, recipe=RECIPE,
                              profile=self.profile,
                              env=dict(ENV_SUBSET, namespace="serving"),
                              out_dir=out, production=True,
                              now="2026-10-01T00:00:00+00:00")
        with self.assertRaises(DeployError) as ctx:
            release.drill(self.env, out, name=release.DRILL_EPP_POD_KILL,
                          runner=FakeRunner())
        self.assertEqual(ctx.exception.error_code, "NOTHING_TO_DRILL")


ADOPTED_RECIPE = {
    **RECIPE,
    "fleet": {
        "mode": "adopted",
        "selector": {"app": "deepseek-v41-tpu7x"},
        "service": "deepseek-v41-tpu7x",
        "port": 8000,
        "replicas": 2,
    },
}


def _ready_pod(name: str, ready: bool = True) -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready",
                            "status": "True" if ready else "False"}],
        },
    }


class AdoptedFleetTest(unittest.TestCase):
    """A release around servers it does not own: router and bench only."""

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
        self.profile = workload.resolve_workload(copy.deepcopy(SPEC))

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _render(self, config_edit=None, name="ds-r1"):
        cfg = render.default_config(ADOPTED_RECIPE)
        if config_edit:
            config_edit(cfg)
        rel = render.render_release(name=name, config=cfg,
                                    recipe=ADOPTED_RECIPE,
                                    profile=self.profile,
                                    env=dict(ENV_SUBSET,
                                             namespace="llmd-study"),
                                    out_dir=self.out,
                                    now="2026-10-02T00:00:00+00:00")
        return rel, cfg

    def test_baseline_is_passthrough_at_the_fleets_replica_count(self):
        rel, cfg = self._render()
        self.assertEqual(cfg["router"]["profile"], render.PASSTHROUGH_PROFILE)
        self.assertEqual(cfg["topology"]["decode"]["replicas"], 2)
        self.assertEqual(rel["fleet"]["mode"], "adopted")
        self.assertEqual(rel["deployments"], [])
        self.assertIsNone(rel["files"]["modelserver"])
        self.assertIsNone(rel["files"]["router_values"])
        self.assertFalse(rel["router"]["installed"])
        self.assertFalse((self.out / render.MODELSERVER_FILE).exists())
        self.assertFalse((self.out / render.ROUTER_VALUES_FILE).exists())
        # The benchmark goes straight to the fleet's own Service.
        self.assertEqual(
            rel["bench"]["base_url"],
            "http://deepseek-v41-tpu7x.llmd-study.svc.cluster.local:8000")
        cm = json.loads((self.out / rel["files"]["bench"]
                         ["interactive.slo"]["configmap"]).read_text())
        ip_cfg = json.loads(cm["data"]["config.json"])
        self.assertEqual(ip_cfg["server"]["base_url"],
                         rel["bench"]["base_url"])
        # Chips are the fleet's, reported but not held by the release.
        self.assertEqual(rel["chips"]["total"], 8)

    def test_router_profile_selects_adopted_pods_by_their_labels(self):
        rel, _ = self._render(lambda c: c["router"].update(
            profile="sticky-until-saturated"))
        self.assertTrue(rel["router"]["installed"])
        self.assertEqual(rel["router"]["epp_deployment"], "ds-r1-epp")
        self.assertEqual(rel["router"]["epp_selector"],
                         "llm-d-router-gateway=ds-r1-epp")
        values = json.loads((self.out / render.ROUTER_VALUES_FILE).read_text())
        servers = values["router"]["modelServers"]
        self.assertEqual(servers["matchLabels"], {"app": "deepseek-v41-tpu7x"})
        self.assertEqual(servers["targetPorts"], [{"number": 8000}])
        mon = json.loads((self.out / render.MONITORING_FILE).read_text())
        by_name = {i["metadata"]["name"]: i for i in mon["items"]}
        self.assertEqual(
            by_name["ds-r1-modelserver"]["spec"]["selector"]["matchLabels"],
            {"app": "deepseek-v41-tpu7x"})
        self.assertEqual(
            by_name["ds-r1-epp"]["spec"]["selector"]["matchLabels"],
            {"llm-d-router-gateway": "ds-r1-epp"})
        self.assertEqual(rel["bench"]["base_url"],
                         "http://ds-r1-epp.llmd-study.svc.cluster.local:80")

    def test_paths_that_touch_the_fleet_are_refused(self):
        for edit in (
                lambda c: c["engine"].update(max_num_seqs=256),
                lambda c: c["topology"]["decode"].update(replicas=3),
                lambda c: c["kv"].update(block_size=64),
        ):
            with self.assertRaises(DeployError) as ctx:
                self._render(edit)
            self.assertEqual(ctx.exception.error_code,
                             "ADOPTED_FLEET_IMMUTABLE")
        # Router plugins under passthrough are a contradiction, not a knob.
        with self.assertRaises(DeployError) as ctx:
            self._render(lambda c: c["router"].update(
                plugins={"queue-scorer": {"weight": 2}}))
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_apply_wait_and_teardown_never_name_the_adopted_pods(self):
        self._render()
        runner = FakeRunner(json_docs={
            "get pods -l app=deepseek-v41-tpu7x": {
                "items": [_ready_pod("deepseek-v41-tpu7x"),
                          _ready_pod("deepseek-v41-tpu7x-2")]
            },
        })
        applied = release.apply(self.env, self.out, run_id="run-1",
                                runner=runner)
        self.assertEqual(applied["chips_held"], 0)
        self.assertFalse(applied["router_installed"])
        # Only monitoring was applied; no servers, no chart.
        self.assertEqual(len(runner.matching("kubectl", "apply")), 1)
        self.assertFalse(runner.matching("helm", "upgrade"))
        waited = release.wait(self.env, self.out, timeout_s=10, poll_s=0,
                              runner=runner, sleep=lambda s: None)
        self.assertEqual(waited["state"], "ready")
        self.assertEqual(waited["adopted"], {"ready": 2, "want": 2})
        self.assertEqual(waited["epp"], {"installed": False})
        runner.calls.clear()
        result = release.teardown(self.env, self.out, runner=runner)
        self.assertFalse(result["adopted_fleet_touched"])
        for call in runner.matching("kubectl", "delete"):
            self.assertIn(f"{pins.RELEASE_LABEL}=ds-r1", call)
            self.assertNotIn("deepseek-v41-tpu7x", " ".join(call))
        self.assertFalse(runner.matching("helm", "uninstall"))

    def test_wait_waits_for_the_chart_epp_by_its_real_name(self):
        self._render(lambda c: c["router"].update(profile="load-only"))
        runner = FakeRunner(json_docs={
            "get pods -l app=deepseek-v41-tpu7x": {
                "items": [_ready_pod("a"), _ready_pod("b")]
            },
            "get deployment ds-r1-epp": {
                "metadata": {"generation": 1},
                "spec": {"replicas": 1},
                "status": {"observedGeneration": 1, "readyReplicas": 1,
                           "updatedReplicas": 1},
            },
        })
        release.apply(self.env, self.out, run_id="run-1", runner=runner)
        self.assertTrue(runner.matching("helm", "upgrade", "--install"))
        waited = release.wait(self.env, self.out, timeout_s=10, poll_s=0,
                              runner=runner, sleep=lambda s: None)
        self.assertEqual(waited["epp"], {"ds-r1-epp": {"ready": 1, "want": 1}})
        with self.assertRaises(DeployError) as ctx:
            release.reconfigure(self.env, self.out,
                                {"engine.max_num_seqs": 64}, runner=runner)
        self.assertEqual(ctx.exception.error_code, "ADOPTED_FLEET_IMMUTABLE")

    def test_hot_path_under_passthrough_is_refused(self):
        self._render()
        runner = FakeRunner()
        release.apply(self.env, self.out, run_id="run-1", runner=runner)
        with self.assertRaises(DeployError) as ctx:
            release.reconfigure(self.env, self.out,
                                {"router.plugins.queue-scorer.weight": 2.0},
                                runner=runner)
        self.assertEqual(ctx.exception.error_code, "NO_ROUTER")

    def test_data_overrides_clamp_the_catalog_entry(self):
        spec = copy.deepcopy(SPEC)
        spec["workload"]["overrides"]["data"] = {
            "conversation_replay": {
                "dynamic_system_prompt_len": {"max": 1500, "mean": 800},
            }
        }
        profile = workload.resolve_workload(spec)
        cfg = workload.render_inference_perf(
            profile, profile["tenants"][0], stage="slo",
            base_url="http://x:80", model_name="m")
        dyn = cfg["data"]["conversation_replay"]["dynamic_system_prompt_len"]
        self.assertEqual(dyn["max"], 1500)
        self.assertEqual(dyn["mean"], 800)
        self.assertEqual(dyn["type"], "lognormal")  # untouched keys survive
        self.assertEqual(cfg["data"]["conversation_replay"]["seed"], 42)


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

    def test_client_limited_replicate_is_inadmissible(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rel, _ = _render(out)
            profile = json.loads((out / "workload_profile.json").read_text())
            tenants = {t["name"]: t for t in profile["tenants"]}
            doc = _lifecycle_doc(ok=1000, bad=0, ttft_s=0.3, tpot_s=0.02,
                                 rps=84.0)
            doc["load_summary"] = {"requested_rate": 84.0,
                                   "achieved_rate": 70.0,
                                   "schedule_delay": {"mean": 12.0}}
            slow = bench.summarize_report(doc, tenants["interactive"], 0.001)
            self.assertTrue(slow["client_limited"])
            fine = bench.summarize_report(
                _lifecycle_doc(ok=400, bad=0, ttft_s=2.0, tpot_s=0.05,
                               rps=36.0), tenants["batch"], 0.001)
            self.assertFalse(fine["client_limited"])
            scored = bench.score(rel, profile,
                                 [{"interactive": slow, "batch": fine}])
            self.assertFalse(scored["admissible"])
            self.assertEqual(scored["client_limited"], ["interactive/r0"])
            gate = next(g for g in scored["gates"]
                        if g["name"] == "load.client_limited")
            self.assertFalse(gate["passed"])
        self.assertFalse(bench.client_limited(None))
        self.assertFalse(bench.client_limited(
            {"requested_rate": 20, "achieved_rate": 19.5,
             "schedule_delay": {"mean": 0.02}}))

    def test_loadgen_sizing_scales_with_rate(self):
        self.assertEqual(workload.loadgen_sizing(1.0), (1, 100))
        workers, concurrency = workload.loadgen_sizing(20.0)
        self.assertEqual(workers, 2)
        self.assertGreaterEqual(workers * concurrency, 600)
        workers, _ = workload.loadgen_sizing(500.0)
        self.assertEqual(workers, workload.LOADGEN_MAX_WORKERS)
        spec = copy.deepcopy(SPEC)
        spec["workload"]["overrides"]["rate"]["target_qps"] = 20
        profile = workload.resolve_workload(spec)
        tenant = profile["tenants"][0]
        cfg = workload.render_inference_perf(
            profile, tenant, stage="slo", base_url="http://x:8000",
            model_name="m", objective="acme-r1-chat")
        self.assertGreaterEqual(cfg["load"]["num_workers"], 1)
        self.assertIn("worker_max_concurrency", cfg["load"])
        self.assertEqual(cfg["load"]["request_retries"],
                         workload.LOADGEN_REQUEST_RETRIES)
        self.assertEqual(cfg["load"]["request_timeout"],
                         workload.LOADGEN_REQUEST_TIMEOUT_S)
        self.assertEqual(cfg["load"]["stage_teardown_grace_seconds"],
                         workload.LOADGEN_TEARDOWN_GRACE_S)
        hardened = workload.harden_load({"request_retries": 0, "stages": []})
        self.assertEqual(hardened["request_retries"], 0)
        self.assertIn("request_timeout", hardened)
        self.assertEqual(cfg["api"]["headers"][workload.OBJECTIVE_HEADER],
                         "acme-r1-chat")
        cfg = workload.render_inference_perf(
            profile, tenant, stage="slo", base_url="http://x:8000",
            model_name="m")
        self.assertNotIn("headers", cfg["api"])

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
                "batch.tpot_p90_ms", "batch.error_rate", "budget.max_chips",
                "load.client_limited"
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
            # Two replicates of (interactive at 84 rps = 2 shards, batch at
            # 36 rps = 1 shard): a ConfigMap and a Job per shard each time,
            # and each Job deleted afterwards so a rerun starts clean.
            self.assertEqual(len(runner.matching("kubectl", "apply")), 12)
            self.assertEqual(len(runner.matching("kubectl", "delete", "job")),
                             6)
            self.assertEqual(report["shards"], {"interactive": 2, "batch": 1})
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
            # Three Jobs (two interactive shards, one batch), all failed.
            self.assertEqual(len(report["job_failures"]), 3)


class CollectTest(unittest.TestCase):
    """Fleet-metrics collection: the selector and the token path.

    Both pins come from the live spike: a `pod=~"<name>-decode-.*"` selector
    matched nothing for an adopted fleet (its pods carry the customer's
    names), and the workstation's user token was rejected by the
    Monitoring API while its ADC token was accepted.
    """

    def _release_dir(self, tmp: str) -> pathlib.Path:
        out = pathlib.Path(tmp) / "rel"
        _render(out)
        return out

    def test_queries_select_by_the_podmonitoring_job_label(self):
        from framework.deploy import collect
        seen: list[str] = []

        def query(promql):
            seen.append(promql)
            return {"status": "success", "data": {"resultType": "vector",
                    "result": [{"value": [0, "0.42"]}]}}

        with tempfile.TemporaryDirectory() as tmp:
            env = config.DeployEnv(namespace="llmd-study",
                                   prometheus_url="http://prom.local")
            doc = collect.collect(env, self._release_dir(tmp), window="10m",
                                  out_dir=pathlib.Path(tmp) / "profile",
                                  query_fn=query)
        self.assertEqual(len(seen), len(collect.QUERIES))
        expected = (f'namespace="llmd-study",'
                    f'job="{render.modelserver_monitoring_name("acme-r1")}"')
        for promql in seen:
            self.assertIn(expected, promql)
            self.assertNotIn("pod=~", promql)
        self.assertEqual(doc["metrics"]["kv_util_mean"]["value"], 0.42)

    def test_pd_release_collects_each_role_through_the_pod_label(self):
        from framework.deploy import collect
        seen: list[str] = []

        def query(promql):
            seen.append(promql)
            value = "1" if 'llmd_role="prefill"' in promql else "2"
            return {"status": "success", "data": {"resultType": "vector",
                    "result": [{"value": [0, value]}]}}

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "rel"

            def pd(cfg):
                cfg["topology"]["mode"] = "disaggregated"
                cfg["topology"]["prefill"] = {"replicas": 1}
            _render(out, config_edit=pd)
            mon = json.loads((out / render.MONITORING_FILE).read_text())
            env = config.DeployEnv(namespace="llmd-study",
                                   prometheus_url="http://prom.local")
            doc = collect.collect(env, out, window="10m",
                                  out_dir=pathlib.Path(tmp) / "profile",
                                  query_fn=query)
        server = [i for i in mon["items"] if i["metadata"]["name"]
                  == render.modelserver_monitoring_name("acme-r1")][0]
        self.assertEqual(server["spec"]["targetLabels"]["fromPod"],
                         [{"from": pins.LLMD_ROLE_LABEL,
                           "to": render.ROLE_TARGET_LABEL}])
        self.assertEqual(len(seen), 3 * len(collect.QUERIES))
        self.assertEqual(doc["roles"]["prefill"]["kv_util_mean"]["value"], 1)
        self.assertEqual(doc["roles"]["decode"]["kv_util_mean"]["value"], 2)
        self.assertEqual(doc["metrics"]["kv_util_mean"]["value"], 2)

    def test_rendered_podmonitoring_name_matches_the_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._release_dir(tmp)
            mon = json.loads((out / render.MONITORING_FILE).read_text())
        names = [item["metadata"]["name"] for item in mon["items"]]
        self.assertIn(render.modelserver_monitoring_name("acme-r1"), names)

    def test_token_prefers_adc_and_falls_back_to_the_user_login(self):
        from framework.deploy import kube

        class TokenRunner(FakeRunner):
            def __call__(self, cmd, *, stdin=None, timeout=None):
                self.calls.append(list(cmd))
                if "application-default" in cmd:
                    return subprocess.CompletedProcess(cmd, 1, "", "no ADC")
                return subprocess.CompletedProcess(cmd, 0, "user-tok\n", "")

        runner = TokenRunner()
        self.assertEqual(kube.gcloud_access_token(runner), "user-tok")
        self.assertEqual(runner.calls, [list(c) for c in kube.TOKEN_COMMANDS])

        adc_calls: list[list[str]] = []

        def adc_runner(cmd, *, stdin=None, timeout=None):
            adc_calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "adc-tok", "")
        self.assertEqual(kube.gcloud_access_token(adc_runner), "adc-tok")
        self.assertEqual(adc_calls, [list(kube.TOKEN_COMMANDS[0])])

    def test_token_error_names_both_commands(self):
        from framework.deploy import kube
        runner = FakeRunner(fail={"print-access-token": "not logged in"})
        with self.assertRaises(kube.CommandError) as ctx:
            kube.gcloud_access_token(runner)
        self.assertIn("application-default", ctx.exception.stderr)
        self.assertIn("gcloud auth print-access-token", ctx.exception.stderr)
        self.assertIn("not logged in", ctx.exception.stderr)


class LoadgenShardTest(unittest.TestCase):
    """M5e: a rate one load-generator pod cannot carry is split and pooled."""

    def test_shards_and_split_rates(self):
        self.assertEqual(workload.loadgen_shards(20), 1)
        self.assertEqual(workload.loadgen_shards(workload.LOADGEN_RPS_PER_POD),
                         1)
        self.assertEqual(workload.loadgen_shards(
            workload.LOADGEN_RPS_PER_POD + 1), 2)
        cfg = {"load": {"type": "poisson",
                        "stages": [{"rate": 200.0, "duration": 120},
                                   {"rate": 260.0, "duration": 120}],
                        "num_workers": 8, "worker_max_concurrency": 1000},
               "api": {"type": "completion"}}
        pieces = workload.shard_load(cfg, 3)
        self.assertEqual(len(pieces), 3)
        for piece in pieces:
            rates = [s["rate"] for s in piece["load"]["stages"]]
            self.assertEqual(rates, [round(200 / 3, 4), round(260 / 3, 4)])
            self.assertEqual(piece["api"], cfg["api"])
            w, c = workload.loadgen_sizing(round(260 / 3, 4))
            self.assertEqual(piece["load"]["num_workers"], w)
            self.assertEqual(piece["load"]["worker_max_concurrency"], c)
        closed = {"load": {"type": "concurrent",
                           "stages": [{"concurrency_level": 5}]}}
        self.assertEqual(workload.shard_load(closed, 2)[0]["load"]["stages"],
                         [{"concurrency_level": 3}])
        self.assertEqual(workload.shard_load(cfg, 1), [cfg])

    def test_pool_reports_sums_rates_and_takes_the_worst_percentile(self):
        def doc(ok, rps, p99, delay, goodput):
            return {"successes": {"count": ok,
                                  "throughput": {"requests_per_sec": rps,
                                                 "output_tokens_per_sec": rps * 10},
                                  "latency": {"time_to_first_token": {"p99": p99,
                                                                      "mean": p99 / 2}},
                                  "goodput_metrics": {"goodput_requests_per_sec": goodput}},
                    "failures": {"count": 1, "by_label": {"timeout": {"count": 1}}},
                    "load_summary": {"requested_rate": 40, "achieved_rate": rps,
                                     "schedule_delay": {"mean": delay}}}
        pooled = bench.pool_reports([doc(100, 39.0, 0.20, 0.01, 38.0),
                                     doc(120, 40.0, 0.35, 0.02, 40.0)])
        self.assertEqual(pooled["successes"]["count"], 220)
        self.assertEqual(pooled["failures"]["count"], 2)
        self.assertEqual(pooled["failures"]["by_label"]["timeout"]["count"], 2)
        self.assertAlmostEqual(
            pooled["successes"]["throughput"]["requests_per_sec"], 79.0)
        self.assertAlmostEqual(
            pooled["successes"]["latency"]["time_to_first_token"]["p99"], 0.35)
        self.assertAlmostEqual(
            pooled["successes"]["goodput_metrics"]["goodput_requests_per_sec"],
            78.0)
        self.assertAlmostEqual(pooled["load_summary"]["requested_rate"], 80.0)
        self.assertAlmostEqual(pooled["load_summary"]["schedule_delay"]["mean"],
                               0.02)
        self.assertEqual(pooled["pooled"], "max-of-shards")
        single = doc(1, 1.0, 0.1, 0.0, 1.0)
        self.assertIs(bench.pool_reports([single]), single)


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

    def test_exec_probe_runs_inside_a_ready_adopted_pod_without_a_job(self):
        from framework.deploy import probe
        payload = {
            "capabilities": {"pd_connector": True, "offload_connector": True,
                             "kv_events": True, "tpu_inference": False,
                             "vllm_torchtpu": True, "vllm": True},
            "versions": {"vllm": "0.29.0", "tpu_inference": None},
            "modules": {},
        }
        logs = "\n".join([probe.PROBE_MARK, json.dumps(payload), probe.PROBE_MARK])

        class ExecRunner(FakeRunner):

            def __call__(self, cmd, *, stdin=None, timeout=None):
                if "exec" in cmd:
                    self.calls.append(list(cmd))
                    self.stdins.append(stdin)
                    return subprocess.CompletedProcess(cmd, 0, self.logs, "")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)

        pods = {"items": [
            _ready_pod("ds-1", ready=False),
            _ready_pod("ds-2", ready=True),
        ]}
        for item in pods["items"]:
            item["spec"] = {"containers": [{"name": "modelserver",
                                            "image": "img@sha256:abc"}]}
        runner = ExecRunner(json_docs={"get pods": pods}, logs=logs)
        env = config.DeployEnv(namespace="llmd-study")
        with tempfile.TemporaryDirectory() as tmp:
            doc = probe.run_probe(env, out_dir=pathlib.Path(tmp),
                                  selector="app=ds", runner=runner,
                                  sleep=lambda s: None)
        self.assertEqual(doc["mode"], "exec")
        self.assertEqual(doc["pod"], "ds-2")
        self.assertEqual(doc["image"], "img@sha256:abc")
        self.assertTrue(doc["capabilities"]["vllm_torchtpu"])
        schema = json.loads(
            (_NEXUS / "package" / "schemas" /
             "engine-capabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(utils.validate_instance(doc, schema), [])
        # No Job was applied and the fleet pod kept its chips: the probe
        # process ran with the accelerator disabled.
        self.assertFalse(runner.matching("kubectl", "apply"))
        (exec_call,) = runner.matching("kubectl", "exec")
        self.assertIn("ds-2", exec_call)
        self.assertIn("JAX_PLATFORMS=cpu", exec_call)
        self.assertIn("-c", exec_call)
        self.assertIn("modelserver", exec_call)


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
