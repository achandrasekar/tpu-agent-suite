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
            "replicas": 2, "hosts": 2, "engine_port": 8000}])
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
                if "curl" in joined or "-probe-" in joined:
                    # The attach-free probe: run, wait, logs, phase, delete.
                    self.calls.append(list(cmd))
                    if "jsonpath={.status.phase}" in joined and "get" in cmd:
                        return subprocess.CompletedProcess(cmd, 0,
                                                           "Succeeded", "")
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
            release.drill(self.env, out, name="no-such-drill",
                          runner=runner)

    def test_decode_pod_kill_drill_waits_for_the_pool(self):
        out, _ = self._prod()
        pool_doc = {"metadata": {"generation": 1}, "spec": {"replicas": 2},
                    "status": {"observedGeneration": 1, "readyReplicas": 2,
                               "updatedReplicas": 2}}

        class DrillRunner(FakeRunner):
            def __init__(self):
                super().__init__(json_docs={
                    "get deployment acme-prod-decode": pool_doc})
                self.killed = False

            def __call__(self, cmd, *, stdin=None, timeout=None):
                joined = " ".join(cmd)
                if "delete pod" in joined:
                    self.killed = True
                if "get pods" in joined and "-o" in cmd:
                    # A LeaderWorkerSet recreates the leader under the
                    # same name with a new UID; the drill must see that
                    # as recovery.
                    def pod(name, uid, ready=True):
                        doc = _ready_pod(name, ready)
                        doc["metadata"]["uid"] = uid
                        return doc
                    pods = ([pod("dec-a", "uid-a2"), pod("dec-b", "uid-b")]
                            if self.killed else
                            [pod("dec-a", "uid-a1"), pod("dec-b", "uid-b")])
                    return subprocess.CompletedProcess(
                        cmd, 0, json.dumps({"items": pods}), "")
                if "curl" in joined or "-probe-" in joined:
                    if "jsonpath={.status.phase}" in joined and "get" in cmd:
                        return subprocess.CompletedProcess(cmd, 0,
                                                           "Succeeded", "")
                    return subprocess.CompletedProcess(
                        cmd, 0, '{"object":"list","data":[]}', "")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)

        runner = DrillRunner()
        clock = iter([0.0, 12.0, 12.5, 13.0, 14.0])
        row = release.drill(self.env, out, name=release.DRILL_DECODE_POD_KILL,
                            runner=runner, timeout_s=60, poll_s=0,
                            sleep=lambda s: None, clock=lambda: next(clock))
        self.assertEqual(row["killed"], "dec-a")
        self.assertEqual(row["pool"]["name"], "acme-prod-decode")
        self.assertTrue(row["recovered"])
        self.assertTrue(row["passed"])
        self.assertTrue(runner.matching("kubectl", "delete", "pod", "dec-a"))

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

    def test_harness_digest_follows_the_apparatus_not_the_fleet(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel, _ = _render(pathlib.Path(tmp))
        base = bench.harness_digest(rel)
        self.assertEqual(len(base), 16)
        # The fleet's own config does not move it...
        other = json.loads(json.dumps(rel))
        other["config_digest"] = "ffff"
        other["replicas"] = 7
        self.assertEqual(bench.harness_digest(other), base)
        # ...the load generator's sizing does.
        other["env"]["sizing"] = {"bench": {"requests": {"cpu": "1"}}}
        self.assertNotEqual(bench.harness_digest(other), base)

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


# A recipe shaped like the InferenceX AgentX Qwen3.5 stack: the engine is a
# role script with env, one whole tpu7x host per engine, weights on a
# ReadOnlyMany PVC, llm-d-router main images, the turn-priority EPP config.
AGENTX_RECIPE = {
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "image": "us-docker.pkg.dev/x/vllm-tpu/torchtpu-vllm-prod:infx-qwen35",
    "accelerator": {"tpu_accelerator": "tpu7x", "tpu_topology": "2x2x1",
                    "chips_per_host": 4},
    "tensor_parallel": 8,
    "max_model_len": 262144,
    "engine_flavor": "torchtpu",
    "source": {"repo": "SemiAnalysisAI/InferenceX",
               "ref": "main", "token_secret": "infx-token"},
    "router_images": {"epp": pins.AGENTX_EPP_IMAGE,
                      "sidecar": pins.AGENTX_SIDECAR_IMAGE},
    "launch": {
        "command": ["bash", "benchmarks/multi_node/agentic/"
                            "qwen3.5_fp8_tpuv7_vllm-disagg_server.sh"],
        "ports": {"prefill": 8000, "decode": 7200},
        "role_env": {
            "prefill": {"MODEL": "Qwen/Qwen3.5-397B-A17B-FP8",
                        "PREFILL_TP": 8, "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": 900,
                        "PREFILL_RAIDEN_ENGINE_ID": "$(POD_NAME)",
                        "TPU_RAIDEN_ADVERTISE_HOST": "$(POD_IP)"},
            "decode": {"MODEL": "Qwen/Qwen3.5-397B-A17B-FP8", "DECODE_TP": 8,
                       "MAX_NUM_SEQS_DECODE": 32,
                       "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": 900},
        },
        "knobs": {
            "max_num_seqs": {"env": "MAX_NUM_SEQS_DECODE", "roles": ["decode"]},
            "kv_cpu_offload_bytes": {"env": "PREFILL_KV_CPU_OFFLOAD_BYTES",
                                     "roles": ["prefill"], "unit": "GiB"},
            "hma_max_inflight_pulls": {"env": "TPU_HMA_MAX_INFLIGHT_PULLS"},
        },
        "sidecar_args": pins.AGENTX_SIDECAR_ARGS,
    },
    "pod": {
        "privileged": True,
        "working_dir": "/workdir",
        "resources": {"limits": {"google.com/tpu": 4}},
        "tolerations": [{"key": "google.com/tpu", "operator": "Exists",
                         "effect": "NoSchedule"}],
        "volumes": [
            {"name": "tpu-cache", "hostPath": {
                "path": "/var/lib/tpu-cache", "type": "DirectoryOrCreate"}},
            {"name": "model-cache", "persistentVolumeClaim": {
                "claimName": "qwen-cache-ro", "readOnly": True}},
        ],
        "volume_mounts": [
            {"name": "tpu-cache", "mountPath": "/root/.cache"},
            {"name": "model-cache", "mountPath": "/mnt/models-ro",
             "readOnly": True},
        ],
        "init_containers": [{"name": "tpu-node-setup", "image": "busybox",
                             "securityContext": {"privileged": True},
                             "command": ["sh", "-c", "sysctl -w x=1"]}],
    },
    "measured": {"sessions_per_prefill_engine": 96,
                 "sessions_per_decode_engine": 112},
}


class AgentXLaunchRenderTest(unittest.TestCase):

    def _config(self, **engine):
        cfg = render.default_config(AGENTX_RECIPE, replicas=1)
        cfg["topology"]["mode"] = "disaggregated"
        cfg["topology"]["prefill"] = {"replicas": 3, "tp": 8}
        cfg["topology"]["decode"]["tp"] = 8
        cfg["router"]["profile"] = "agentx-turn-priority"
        for role, knobs in engine.items():
            cfg["topology"][role]["engine"] = knobs
        return cfg

    def test_image_values_fold_a_digest_into_the_chart_fields(self):
        v = render.image_values(pins.AGENTX_EPP_IMAGE)
        self.assertEqual(v["registry"], "ghcr.io")
        self.assertEqual(v["repository"],
                         "llm-d/llm-d-router-endpoint-picker@sha256")
        self.assertEqual(f"{v['registry']}/{v['repository']}:{v['tag']}",
                         pins.AGENTX_EPP_IMAGE)
        v = render.image_values("ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0")
        self.assertEqual((v["repository"], v["tag"]),
                         ("llm-d/llm-d-router-endpoint-picker", "v0.11.0"))

    def test_document_profiles_render_as_written(self):
        cfg = self._config()
        doc = render.epp_config(cfg)
        self.assertEqual(doc["apiVersion"], "llm-d.ai/v1")
        self.assertEqual(doc["featureGates"], ["flowControl"])
        self.assertEqual(doc["flowControl"]["defaultPriorityBand"]
                         ["fairnessPolicyRef"], "program-aware-fairness")
        types = [p["type"] for p in doc["plugins"]]
        self.assertEqual(types[:2], ["agent-identity", "program-aware-fairness"])
        self.assertIn("session-affinity-filter", types)
        self.assertEqual([sp["name"] for sp in doc["schedulingProfiles"]],
                         ["prefill", "decode"])
        # nofc: the same document minus its flow-control half.
        cfg["router"]["profile"] = "agentx-session-affinity"
        nofc = render.epp_config(cfg)
        self.assertNotIn("featureGates", nofc)
        self.assertNotIn("flowControl", nofc)
        self.assertNotIn("program-aware-fairness",
                         [p["type"] for p in nofc["plugins"]])
        self.assertEqual(nofc["schedulingProfiles"],
                         doc["schedulingProfiles"])
        # Saturation thresholds merge into the flow-control block; a plugin
        # override reaches a plugin by its name.
        cfg["router"]["profile"] = "agentx-turn-priority"
        cfg["router"]["flow_control"] = {"saturation_detector": {
            "queueDepthThreshold": 20, "kvCacheUtilThreshold": 0.95}}
        cfg["router"]["plugins"] = {"active-requests": {"weight": 2},
                                    "program-aware-fairness": {
                                        "turnPriorityTimeWeight": 0.1}}
        tuned = render.epp_config(cfg)
        self.assertEqual(tuned["flowControl"]["saturationDetector"]
                         ["queueDepthThreshold"], 20)
        fair = next(p for p in tuned["plugins"]
                    if p["type"] == "program-aware-fairness")
        self.assertEqual(fair["parameters"]["turnPriorityTimeWeight"], 0.1)
        # A document profile that routes P/D refuses an aggregated fleet,
        # and flow control cannot be switched on where the document has none.
        agg = render.default_config(AGENTX_RECIPE)
        agg["router"]["profile"] = "agentx-session-affinity"
        with self.assertRaises(DeployError):
            render.validate_config(agg)
        cfg["router"]["profile"] = "agentx-session-affinity"
        with self.assertRaises(DeployError):
            render.epp_config(cfg)

    def test_router_values_take_the_recipe_images_and_experimental_flag(self):
        cfg = self._config()
        values = render.router_values("agx", cfg, env=dict(ENV_SUBSET),
                                      recipe=AGENTX_RECIPE)
        epp = values["router"]["epp"]
        self.assertTrue(epp["flags"]["allow-experimental-plugins"])
        self.assertEqual(epp["image"]["tag"],
                         pins.AGENTX_EPP_IMAGE.split("sha256:")[1])
        plain = render.router_values("agx", render.default_config(RECIPE),
                                     env=dict(ENV_SUBSET), recipe=RECIPE)
        self.assertNotIn("image", plain["router"]["epp"])

    def test_launch_recipe_renders_the_inferencex_pod(self):
        cfg = self._config(decode={"max_num_seqs": 48},
                           prefill={"kv_cpu_offload_bytes": 256})
        cfg["engine"] = {"hma_max_inflight_pulls": 16}
        docs = render.modelserver_manifests("agx", AGENTX_RECIPE, cfg,
                                            dict(ENV_SUBSET), False)["items"]
        deps = {d["metadata"]["name"]: d for d in docs
                if d["kind"] == "Deployment"}
        dec = deps["agx-decode"]["spec"]["template"]["spec"]
        pf = deps["agx-prefill"]["spec"]["template"]["spec"]
        c = dec["containers"][0]
        env = {e["name"]: e.get("value") for e in c["env"]}
        self.assertEqual(c["command"], AGENTX_RECIPE["launch"]["command"])
        self.assertNotIn("args", c)
        self.assertEqual(c["workingDir"], "/workdir")
        self.assertEqual(env["ROLE"], "decode")
        self.assertEqual(env["DECODE_TP"], "8")
        # The knob overrides the role default by name, once.
        self.assertEqual(env["MAX_NUM_SEQS_DECODE"], "48")
        self.assertEqual([e["name"] for e in c["env"]].count(
            "MAX_NUM_SEQS_DECODE"), 1)
        self.assertEqual(env["TPU_HMA_MAX_INFLIGHT_PULLS"], "16")
        self.assertNotIn("PREFILL_KV_CPU_OFFLOAD_BYTES", env)
        # Neither the renderer's P/D env nor its offload env leaks in.
        self.assertNotIn("TPU_SIDE_CHANNEL_PORT", env)
        self.assertEqual(c["ports"][0]["containerPort"], 7200)
        self.assertEqual(c["resources"], {"limits": {"google.com/tpu": 4}})
        self.assertTrue(c["securityContext"]["privileged"])
        self.assertEqual(c["startupProbe"]["httpGet"]["path"], "/health")
        self.assertEqual(c["startupProbe"]["failureThreshold"], 240)
        mounts = {m["name"]: m["mountPath"] for m in c["volumeMounts"]}
        self.assertEqual(mounts["model-cache"], "/mnt/models-ro")
        self.assertEqual(mounts["workdir"], "/workdir")
        self.assertNotIn("torch-compile-cache", mounts)
        vols = {v["name"] for v in dec["volumes"]}
        self.assertEqual(vols, {"dshm", "workdir", "tpu-cache", "model-cache"})
        inits = [i["name"] for i in dec["initContainers"]]
        self.assertEqual(inits, ["tpu-node-setup", "clone", "routing-proxy"])
        clone = dec["initContainers"][1]
        self.assertIn("main", clone["args"][0])
        self.assertIn("x-access-token:${TOKEN}@github.com/"
                      "SemiAnalysisAI/InferenceX.git",
                      clone["args"][0])
        sidecar = dec["initContainers"][2]
        self.assertEqual(sidecar["image"], pins.AGENTX_SIDECAR_IMAGE)
        self.assertEqual(sidecar["args"], [
            "--port=8000", "--model-server-port=7200",
            "--kv-connector=nixlv2", "--secure-serving=false"])
        self.assertEqual(dec["tolerations"][0]["key"], "google.com/tpu")
        # Prefill: its own port, its own knobs, bytes from GiB.
        pc = pf["containers"][0]
        penv = {e["name"]: e.get("value") for e in pc["env"]}
        self.assertEqual(pc["ports"][0]["containerPort"], 8000)
        self.assertEqual(penv["PREFILL_KV_CPU_OFFLOAD_BYTES"],
                         str(256 * (1 << 30)))
        self.assertNotIn("MAX_NUM_SEQS_DECODE", penv)
        self.assertEqual(penv["TPU_HMA_MAX_INFLIGHT_PULLS"], "16")
        names = [e["name"] for e in pc["env"]]
        self.assertLess(names.index("POD_NAME"),
                        names.index("PREFILL_RAIDEN_ENGINE_ID"))
        self.assertEqual(penv["PREFILL_RAIDEN_ENGINE_ID"], "$(POD_NAME)")
        self.assertEqual(deps["agx-prefill"]["spec"]["replicas"], 3)
        self.assertEqual(render.total_chips(AGENTX_RECIPE, cfg), 16)

    def test_unmapped_knob_is_refused_by_name(self):
        cfg = self._config(decode={"max_num_batched_tokens": 4096})
        with self.assertRaises(DeployError) as ctx:
            render.modelserver_manifests("agx", AGENTX_RECIPE, cfg,
                                         dict(ENV_SUBSET), False)
        self.assertEqual(ctx.exception.error_code, "ENGINE_KNOB_UNMAPPED")
        # A plain image still takes the whitelisted flags without a map,
        # and refuses a mapped-only knob it cannot express.
        plain = render.default_config(RECIPE)
        plain["engine"] = {"max_num_seqs": 64}
        self.assertIn("--max-num-seqs=64", render.vllm_args(RECIPE, plain))
        plain["engine"] = {"hma_max_inflight_pulls": 4}
        with self.assertRaises(DeployError) as ctx:
            render.modelserver_manifests("p", RECIPE, plain,
                                         dict(ENV_SUBSET), False)
        self.assertEqual(ctx.exception.error_code, "ENGINE_KNOB_UNMAPPED")


AGENTX_SPEC = {
    "name": "qwen35-agentx",
    "workload": {"agentx": {"concurrency": [8, 16, 24], "fast": True}},
    "budget": {"max_chips": 24, "accelerator": "tpu7x"},
}


def _agentx_records(n: int = 20, ttft_ms: float = 1200.0,
                    itl_ms: float = 4.0, warm: int = 3,
                    errors: int = 1) -> str:
    """A profile_export.jsonl: `warm` warm-up rows, `errors` failed rows,
    `n` profiled rows 10 s apart, 1000 in / 200 out tokens each."""
    rows = []
    t0 = 1_700_000_000_000_000_000
    for i in range(warm):
        rows.append({"metadata": {"benchmark_phase": "warmup",
                                  "request_start_ns": t0,
                                  "request_end_ns": t0 + 10**9},
                     "metrics": {"time_to_first_token": {"value": 50.0}}})
    for i in range(errors):
        rows.append({"metadata": {"benchmark_phase": "profiling",
                                  "request_start_ns": t0,
                                  "request_end_ns": t0 + 10**9},
                     "error": {"code": 502, "message": "bad gateway"}})
    for i in range(n):
        start = t0 + i * 10 * 10**9
        rows.append({"metadata": {"benchmark_phase": "profiling",
                                  "request_start_ns": start,
                                  "request_end_ns": start + 5 * 10**9},
                     "metrics": {"time_to_first_token": {"value": ttft_ms},
                                 "request_latency": {"value": 5000.0},
                                 "inter_token_latency": {"value": itl_ms},
                                 "input_sequence_length": {"value": 1000},
                                 "output_sequence_length": {"value": 200}}})
    return "\n".join(json.dumps(r) for r in rows) + "\n"


class AgentXWorkloadTest(unittest.TestCase):

    def test_agentx_spec_resolves_to_a_closed_loop_profile(self):
        profile = workload.resolve_workload(copy.deepcopy(AGENTX_SPEC))
        self.assertTrue(workload.is_agentx(profile))
        ax = profile["agentx"]
        self.assertEqual(ax["concurrency"], [8, 16, 24])
        self.assertEqual(ax["duration_s"], pins.AGENTX_FAST_DURATION_S)
        self.assertEqual(ax["warmup_requests_per_lane"], 1)
        self.assertEqual(ax["dataset"],
                         "semianalysisai/cc-traces-weka-062126-256k")
        self.assertTrue(ax["submission_valid"])
        self.assertFalse(ax["unsafe_override"])
        self.assertEqual(profile["objective"]["primary_metric"],
                         "total_tokens_per_s_per_chip")
        self.assertEqual(profile["tenants"][0]["name"], "agentx")
        self.assertEqual(profile["tenants"][0]["slo"], {"ttft_p50_ms": 5000})
        self.assertEqual(profile["availability"]["error_rate_max"], 0.10)
        self.assertIsNone(profile["rate"]["target_qps"])
        # A range ladder, a short duration, and the metric guard.
        spec = copy.deepcopy(AGENTX_SPEC)
        spec["workload"]["agentx"] = {"concurrency": {"start": 96, "step": 48,
                                                      "max": 240},
                                      "duration_s": 300}
        prof = workload.resolve_workload(spec)
        self.assertEqual(prof["agentx"]["concurrency"], [96, 144, 192, 240])
        self.assertTrue(prof["agentx"]["unsafe_override"])
        self.assertFalse(prof["agentx"]["submission_valid"])
        spec["objective"] = {"primary_metric": "goodput_at_slo_per_chip"}
        with self.assertRaises(DeployError):
            workload.resolve_workload(spec)

    def test_render_writes_one_aiperf_job_for_both_stages(self):
        profile = workload.resolve_workload(copy.deepcopy(AGENTX_SPEC))
        cfg = render.default_config(AGENTX_RECIPE, replicas=1)
        cfg["topology"]["mode"] = "disaggregated"
        cfg["topology"]["prefill"] = {"replicas": 2, "tp": 8}
        cfg["topology"]["decode"]["tp"] = 8
        cfg["router"]["profile"] = "agentx-session-affinity"
        cfg["workload"]["concurrency"] = 96
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            rel = render.render_release(
                name="agx", config=cfg, recipe=AGENTX_RECIPE, profile=profile,
                env=dict(ENV_SUBSET), out_dir=out,
                now="2026-10-16T00:00:00+00:00")
            files = rel["files"]["bench"]
            self.assertEqual(files["agentx.slo"], files["agentx.sweep"])
            job = json.loads((out / files["agentx.slo"]["job"]).read_text())
            cm = json.loads((out / files["agentx.slo"]["configmap"]).read_text())
            run = cm["data"]["run.sh"]
            self.assertIn(render.AGENTX_AGGREGATE_FILE, cm["data"])
        self.assertEqual(rel["workload"]["harness"], "aiperf")
        self.assertEqual(rel["workload"]["agentx"]["operating_point"], 96)
        self.assertEqual(rel["pools"][1]["engine_port"], 8000)
        self.assertEqual(rel["pools"][0]["engine_port"], 7200)
        c = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(c["image"], pins.AIPERF_BENCH_IMAGE)
        names = {e["name"] for e in c["env"]}
        self.assertIn(render.AGENTX_CONC_ENV, names)
        self.assertIn(render.AGENTX_METRICS_ENV, names)
        # The command InferenceX builds, flag for flag.
        for flag in ("--scenario \"inferencex-agentx-mvp\"",
                     "--endpoint \"/v1/chat/completions\"",
                     "--endpoint-type \"chat\"", "--streaming",
                     "aiperf profile --scenario",
                     "--concurrency $AGENTX_CONCURRENCY",
                     "--benchmark-duration \"1200\"",
                     "--random-seed \"42\"",
                     "--failed-request-threshold \"0.25\"",
                     "--trajectory-start-min-ratio \"0.25\"",
                     "--warmup-requests-per-lane \"1\"",
                     "--trace-idle-gap-cap-seconds \"300\"",
                     "--warmup-grace-period \"1800\"",
                     "--use-server-token-count", "--no-gpu-telemetry",
                     "--tokenizer-trust-remote-code",
                     "--num-dataset-entries \"393\"",
                     "--slice-duration \"1.0\"",
                     "--public-dataset"
                     " \"semianalysis_cc_traces_weka_062126_256k\"",
                     "--tokenizer \"Qwen/Qwen3.5-397B-A17B-FP8\"",
                     f"git+{pins.AIPERF_REPO}@{pins.AIPERF_REF}",
                     "hf download --repo-type dataset"
                     " \"semianalysisai/cc-traces-weka-062126-256k\"",
                     "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
                     "--server-metrics"):
            self.assertIn(flag, run, flag)
        self.assertNotIn("--unsafe-override", run)
        self.assertNotIn("--max-context-length", run)

    def test_reducer_applies_the_inferencex_formulas(self):
        from framework.deploy import agentx_aggregate as agg
        with tempfile.TemporaryDirectory() as tmp:
            art = pathlib.Path(tmp) / "aiperf_artifacts"
            art.mkdir()
            (art / "profile_export.jsonl").write_text(_agentx_records())
            (art / "profile_export_aiperf.json").write_text(json.dumps({
                "metadata": {"dataset": {"name": "x"}, "concurrency": 8},
                "request_count": {"avg": 24}}))
            (art / "server_metrics_export.json").write_text(json.dumps([
                {"name": "vllm:prefix_cache_hits", "value": 800},
                {"name": "vllm:prefix_cache_queries", "value": 1000},
                {"name": "vllm:external_prefix_cache_hits", "value": 50},
                {"name": "vllm:external_prefix_cache_queries", "value": 1000},
            ]))
            doc = agg.aggregate(art.parent, 8)
        self.assertEqual(doc["status"], "ok")
        reqs = doc["requests"]
        self.assertEqual((reqs["records_total"], reqs["records_profiled"],
                          reqs["records_warmup_dropped"],
                          reqs["records_error_dropped"]), (24, 20, 3, 1))
        self.assertEqual(reqs["error_categories"], {"http_502": 1})
        # 20 requests, first start to last end = 19*10 + 5 = 195 s.
        th = doc["throughput"]
        self.assertAlmostEqual(th["duration_s"], 195.0)
        self.assertAlmostEqual(th["total_tokens_per_s"], 24000 / 195.0)
        self.assertAlmostEqual(th["output_tokens_per_s"], 4000 / 195.0)
        self.assertAlmostEqual(doc["latency_s"]["ttft"]["p50"], 1.2)
        self.assertAlmostEqual(doc["latency_s"]["tpot"]["p50"], 0.004)
        self.assertAlmostEqual(doc["interactivity_tok_s_per_user"]["p50"],
                               250.0)
        # E2E-normalised: 5 s / 200 tokens = 25 ms per token -> 40 tok/s.
        self.assertAlmostEqual(
            doc["e2e_norm_interactivity_tok_s_per_user"]["p90"], 40.0)
        self.assertAlmostEqual(doc["cache"]["gpu_hit_rate"], 0.8)
        self.assertAlmostEqual(doc["cache"]["cpu_hit_rate"], 0.05)
        self.assertEqual(doc["aiperf"]["concurrency"], 8)

    def test_run_agentx_bench_drains_resets_samples_and_scores(self):
        class ProxyRunner(FakeRunner):
            """Answers the pod proxy: /metrics with running=0 once drained,
            counters that advance per call; records POSTs."""
            def __init__(self, logs):
                super().__init__(json_docs={
                    "get job": {"status": {"succeeded": 1}},
                    "get pods": {"items": [
                        {"metadata": {"name": "agx-prefill-0", "labels": {
                            pins.LLMD_ROLE_LABEL: "prefill"}},
                         "status": {"podIP": "10.0.0.1"}},
                        {"metadata": {"name": "agx-decode-0", "labels": {
                            pins.LLMD_ROLE_LABEL: "decode"}},
                         "status": {"podIP": "10.0.0.2"}}]}}, logs=logs)
                self.scrapes = 0

            def __call__(self, cmd, *, stdin=None, timeout=None):
                joined = " ".join(cmd)
                if "get job" in joined and self.scrapes < 6:
                    # Still running for the first polls, so the sampler
                    # sees the point more than once.
                    self.calls.append(list(cmd))
                    return subprocess.CompletedProcess(
                        cmd, 0, '{"status": {"active": 1}}', "")
                if "get --raw" in joined and "/metrics" in joined:
                    self.calls.append(list(cmd))
                    self.scrapes += 1
                    text = ("vllm:num_requests_running{m=\"x\"} 0\n"
                            "vllm:num_requests_waiting{m=\"x\"} 0\n"
                            "vllm:kv_cache_usage_perc{m=\"x\"} 0.4\n"
                            f"vllm:prefix_cache_hits{{m=\"x\"}} {80 * self.scrapes}\n"
                            f"vllm:prefix_cache_queries{{m=\"x\"}} {100 * self.scrapes}\n")
                    return subprocess.CompletedProcess(cmd, 0, text, "")
                if "create --raw" in joined:
                    self.calls.append(list(cmd))
                    return subprocess.CompletedProcess(
                        cmd, 0, '{"success": true}', "")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)

        from framework.deploy import agentx_aggregate as agg
        with tempfile.TemporaryDirectory() as tmp:
            art = pathlib.Path(tmp) / "aiperf_artifacts"
            art.mkdir()
            (art / "profile_export.jsonl").write_text(_agentx_records())
            point = agg.aggregate(art.parent, 16)
        logs = "\n".join([f"{render.BENCH_REPORT_MARK} /reports/agentx_point.json",
                          json.dumps(point), render.BENCH_REPORT_END])
        runner = ProxyRunner(logs)
        env = config.DeployEnv(namespace="llmd-study")
        profile = workload.resolve_workload(copy.deepcopy(AGENTX_SPEC))
        cfg = render.default_config(AGENTX_RECIPE, replicas=1)
        cfg["topology"]["mode"] = "disaggregated"
        cfg["topology"]["prefill"] = {"replicas": 1, "tp": 8}
        cfg["topology"]["decode"]["tp"] = 8
        cfg["router"]["profile"] = "agentx-session-affinity"
        cfg["workload"]["concurrency"] = 16
        ticks = iter(range(0, 100000, 40))
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "release"
            render.render_release(name="agx", config=cfg, recipe=AGENTX_RECIPE,
                                  profile=profile, env=dict(ENV_SUBSET),
                                  out_dir=out, now="2026-10-16T00:00:00+00:00")
            report = bench.run_bench(env, out, stage="slo", replicates=1,
                                     out_dir=pathlib.Path(tmp) / "bench",
                                     execution_id="exec-ax", timeout_s=600,
                                     runner=runner, sleep=lambda s: None,
                                     clock=lambda: float(next(ticks)))
            schema = json.loads(
                (_NEXUS / "package" / "schemas" /
                 "llmd-bench-report.json").read_text(encoding="utf-8"))
            self.assertEqual(utils.validate_instance(report, schema), [])
        self.assertEqual(report["primary_metric"], "total_tokens_per_s_per_chip")
        self.assertEqual(report["agentx"]["best_concurrency"], 16)
        self.assertEqual(len(report["agentx"]["points"]), 1)
        chips = report["release"]["chips_total"]
        self.assertEqual(chips, 8)
        self.assertAlmostEqual(report["metrics"]["total_tokens_per_s_per_chip"]
                               ["value"], 24000 / 195.0 / 8)
        self.assertTrue(report["admissible"])
        t = report["tenants"]["agentx"]
        self.assertAlmostEqual(t["slo"]["ttft_p50_ms"]["measured_ms"], 1200.0)
        self.assertAlmostEqual(t["error_rate"], 1 / 21)
        # The point's Job got its concurrency and the prefill metrics URL.
        applied = [json.loads(s) for s in runner.stdins
                   if s and s.lstrip().startswith("{") and '"Job"' in s]
        env_map = {e["name"]: e.get("value") for e in applied[-1]["spec"]
                   ["template"]["spec"]["containers"][0]["env"]}
        self.assertEqual(env_map[render.AGENTX_CONC_ENV], "16")
        self.assertEqual(env_map[render.AGENTX_METRICS_ENV],
                         "http://10.0.0.1:8000/metrics")
        # Drained through the pod proxy, reset both engines, sampled roles.
        self.assertEqual(len(runner.matching("create", "--raw")), 2)
        rep = report["agentx"]["points"][0]["tenant"]["replicates"][0]
        self.assertTrue(rep["drain"]["drained"])
        self.assertEqual([r["ok"] for r in rep["prefix_reset"]], [True, True])
        self.assertIn("prefill", rep["roles"])
        self.assertAlmostEqual(rep["roles"]["prefill"]["prefix_hit_rate"], 0.8)
        self.assertEqual(rep["roles"]["decode"]["kv_util_max"], 0.4)

    def test_reducer_reads_aiperf_012_server_metrics_per_endpoint(self):
        """aiperf 0.12's schema-1.1 export (observed on kind 2026-10-16):
        per-series totals; the inference URL's own /metrics (the router's
        proxy Service, one engine's page again) is not counted when the Job
        knows its engine endpoints."""
        from framework.deploy import agentx_aggregate as agg

        def counter(*series):
            return {"type": "counter", "series": [
                {"endpoint_url": url, "labels": {"model_name": "m"},
                 "stats": {"total": total, "rate": 1.0}}
                for url, total in series]}
        router = "http://agx-epp.ns.svc.cluster.local:80/metrics"
        e1, e2 = "http://10.0.0.1:8000/metrics", "http://10.0.0.2:8000/metrics"
        doc = {"schema_version": "1.1", "aiperf_version": "0.12.0",
               "metrics_phase": "profiling", "metrics": {
                   "vllm:prefix_cache_hits": counter((router, 800.0),
                                                     (e1, 800.0), (e2, 100.0)),
                   "vllm:prefix_cache_queries": counter((router, 1000.0),
                                                        (e1, 1000.0),
                                                        (e2, 200.0)),
                   "vllm:prompt_tokens": counter((e1, 5.0))}}
        both = agg.cache_rates(doc, {e1, e2})
        self.assertTrue(both["present"])
        self.assertAlmostEqual(both["gpu_hit_rate"], 900 / 1200)
        self.assertIsNone(both["cpu_hit_rate"])
        everything = agg.cache_rates(doc)
        self.assertAlmostEqual(everything["gpu_hit_rate"], 1700 / 2200)
        self.assertEqual(agg.cache_rates({"metrics": {}}), {"present": False})
        # The exposition format spells the same counter with `_total`.
        sample = bench.parse_prometheus(
            'vllm:prefix_cache_hits_total{model_name="m"} 1.046016e+06\n'
            'vllm:prefix_cache_queries_total{model_name="m"} 1.404219e+06\n'
            'vllm:kv_cache_usage_perc{model_name="m"} 0.25\n')
        self.assertEqual(sample["vllm:prefix_cache_hits"], 1046016.0)
        self.assertEqual(sample["vllm:prefix_cache_queries_total"], 1404219.0)
        self.assertEqual(sample["vllm:kv_cache_usage_perc"], 0.25)

    def test_reducer_writes_a_trimmed_server_metrics_summary(self):
        """The Job echoes the summary, not the export: a 25-minute point's
        export (per-second timeslices) outgrew the container log on kind
        and rotated the point itself away."""
        from framework.deploy import agentx_aggregate as agg
        counter = {"vllm:prefix_cache_hits": {"type": "counter", "series": [
            {"endpoint_url": "u", "stats": {"total": 5.0},
             "timeslices": [{"start_ns": 1, "total": 1.0}] * 500}]}}
        export = {"schema_version": "1.1", "summary": {"x": 1},
                  "metrics": counter, "warmup_metrics": counter}
        trimmed = agg.server_metrics_summary(export)
        self.assertNotIn("timeslices", trimmed["warmup_metrics"][
            "vllm:prefix_cache_hits"]["series"][0])
        series = trimmed["metrics"]["vllm:prefix_cache_hits"]["series"][0]
        self.assertNotIn("timeslices", series)
        self.assertEqual(series["stats"]["total"], 5.0)
        self.assertEqual(trimmed["summary"], {"x": 1})
        self.assertLess(len(json.dumps(trimmed)), len(json.dumps(export)) / 10)
        with tempfile.TemporaryDirectory() as tmp:
            art = pathlib.Path(tmp) / "aiperf_artifacts"
            art.mkdir()
            (art / "profile_export.jsonl").write_text(_agentx_records())
            (art / "server_metrics_export.json").write_text(json.dumps(export))
            doc = agg.aggregate(art.parent, 8, art.parent)
            summary = art.parent / agg.SERVER_METRICS_SUMMARY_FILE
            self.assertTrue(summary.is_file())
            self.assertEqual(json.loads(summary.read_text())["trimmed"],
                             "timeslices dropped")
        self.assertTrue(doc["cache"]["present"])
        script = render.agentx_bench_script(
            workload.resolve_workload(copy.deepcopy(AGENTX_SPEC)),
            base_url="http://router:80", model_name="m", tokenizer=None)
        self.assertIn(render.AGENTX_SERVER_METRICS_SUMMARY_FILE, script)
        self.assertNotIn("'server_metrics_export.json'", script)
        # The point is echoed last.
        self.assertGreater(script.rfind(render.AGENTX_POINT_FILE),
                           script.rfind(render.AGENTX_SERVER_METRICS_SUMMARY_FILE))

    def test_source_patches_ride_in_a_configmap_and_git_apply_after_clone(self):
        """A change the pinned ref lacks (such as an offload env hook)
        travels as a unified diff in the recipe: the CLI inlines the
        file, the release holds it with its sha256, the clone init
        container applies it, and the Deployment mounts it."""
        recipe = copy.deepcopy(AGENTX_RECIPE)
        diff = ("--- a/server.sh\n+++ b/server.sh\n@@ -1 +1,2 @@\n echo hi\n"
                "+echo offload\n")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "patches").mkdir()
            (root / "patches" / "offload.patch").write_text(diff)
            recipe["source"]["patches"] = [{
                "name": "kv offload", "path": "patches/offload.patch",
                "why": "not in pinned ref"}]
            (root / "recipe.json").write_text(json.dumps(recipe))
            loaded = cli._load_recipe(str(root / "recipe.json"))
            patch = loaded["source"]["patches"][0]
            self.assertEqual(patch["content"], diff)
            self.assertEqual(len(patch["sha256"]), 64)
            profile = workload.resolve_workload(copy.deepcopy(AGENTX_SPEC))
            cfg = render.default_config(loaded, replicas=1)
            out = root / "release"
            render.render_release(name="agx", config=cfg, recipe=loaded,
                                  profile=profile, env=dict(ENV_SUBSET),
                                  out_dir=out, now="2026-10-17T00:00:00+00:00")
            manifests = json.loads((out / "manifests" /
                                    "10-modelserver.json").read_text())
        items = manifests["items"]
        cm = next(i for i in items if i["kind"] == "ConfigMap"
                  and i["metadata"]["name"] == "agx-source-patches")
        self.assertEqual(cm["data"], {"00-kv-offload.patch": diff})
        dep = next(i for i in items if i["kind"] == "Deployment")
        pod = dep["spec"]["template"]["spec"]
        clone = next(c for c in pod["initContainers"] if c["name"] == "clone")
        self.assertIn("git apply --verbose", clone["args"][0])
        self.assertIn(render.SOURCE_PATCH_MOUNT, clone["args"][0])
        self.assertIn({"name": "source-patches",
                       "mountPath": render.SOURCE_PATCH_MOUNT,
                       "readOnly": True}, clone["volumeMounts"])
        self.assertIn({"name": "source-patches",
                       "configMap": {"name": "agx-source-patches"}},
                      pod["volumes"])
        # A patch given by path the CLI cannot find is refused up front.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            recipe["source"]["patches"] = [{"name": "x", "path": "missing.patch"}]
            (root / "recipe.json").write_text(json.dumps(recipe))
            with self.assertRaises(DeployError):
                cli._load_recipe(str(root / "recipe.json"))

    def test_stage_weights_creates_the_claim_and_the_download_job(self):
        """The recipe mounts `qwen-cache-ro` read-only at /mnt/models-ro
        and reads `MODEL_HUB_CACHE=/mnt/models-ro/hub`; staging a plain
        RWO claim creates it and runs `hf download` into `hub/` on a CPU
        node with the HF token Secret; the read-only path snapshots a
        populated `<pvc>-rw` and clones it ReadOnlyMany."""
        from framework.deploy import weights  # noqa: PLC0415
        recipe = copy.deepcopy(AGENTX_RECIPE)
        claims = weights.recipe_claims(recipe)
        self.assertEqual(claims[0]["claim"], "qwen-cache-ro")
        self.assertTrue(claims[0]["read_only"])
        self.assertEqual(claims[0]["hub_subdir"], "hub")
        env = config.DeployEnv(namespace="llmd-study",
                               cpu_node_selector={"pool": "cpu"})
        # The Job succeeds on the second poll.
        polls = {"n": 0}

        class Runner(FakeRunner):
            def __call__(self, cmd, *, stdin=None, timeout=None):
                joined = " ".join(cmd)
                if "get pvc" in joined and "-o json" in joined:
                    return subprocess.CompletedProcess(cmd, 1, "", "NotFound")
                if "get job" in joined and "-o json" in joined:
                    polls["n"] += 1
                    if polls["n"] == 1:
                        return subprocess.CompletedProcess(cmd, 1, "", "NotFound")
                    return subprocess.CompletedProcess(cmd, 0, json.dumps(
                        {"status": {"succeeded": 1}}), "")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)
        runner = Runner(logs="===STAGED===\nmodels--Qwen--Qwen3.5-397B-A17B-FP8")
        result = weights.stage_weights(
            env, recipe=recipe, runner=runner, sleep=lambda s: None,
            clock=iter(range(0, 10 ** 6, 20)).__next__)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["claim"], "qwen-cache-ro")
        applied = [json.loads(s) for s in runner.stdins if s]
        kinds = [(d["kind"], d["metadata"]["name"]) for d in applied]
        self.assertIn(("PersistentVolumeClaim", "qwen-cache-ro"), kinds)
        self.assertIn(("Job", "stage-weights-qwen-cache-ro"), kinds)
        pvc = next(d for d in applied if d["kind"] == "PersistentVolumeClaim")
        self.assertEqual(pvc["spec"]["accessModes"], ["ReadWriteOnce"])
        self.assertEqual(pvc["spec"]["resources"]["requests"]["storage"],
                         weights.DEFAULT_SIZE)
        job = next(d for d in applied if d["kind"] == "Job")
        pod = job["spec"]["template"]["spec"]
        self.assertEqual(pod["nodeSelector"], {"pool": "cpu"})
        self.assertNotIn("tolerations", pod)
        script = pod["containers"][0]["args"][0]
        self.assertIn('hf download "Qwen/Qwen3.5-397B-A17B-FP8"', script)
        self.assertIn("HF_HUB_CACHE=\"/mnt/weights/hub\"", script)
        self.assertEqual(pod["containers"][0]["env"][-1]["valueFrom"]
                         ["secretKeyRef"]["name"], env.hf_token_secret)
        self.assertIn("models--Qwen", result["job_log_tail"])
        # A ReadOnlyMany claim cannot be filled by a Job.
        with self.assertRaises(DeployError):
            weights.stage_weights(env, recipe=recipe, runner=Runner(),
                                  access_mode="ReadOnlyMany", wait=False)
        # The read-only path: <pvc>-rw + Job, VolumeSnapshot, ROX clone.
        polls["n"] = 0

        class SnapRunner(Runner):
            seen_snapshot = False

            def __call__(self, cmd, *, stdin=None, timeout=None):
                joined = " ".join(cmd)
                if "get volumesnapshot" in joined and "-o json" in joined:
                    return subprocess.CompletedProcess(cmd, 0, json.dumps(
                        {"status": {"readyToUse": True}}), "")
                if "get volumesnapshot" in joined and not self.seen_snapshot:
                    self.seen_snapshot = True   # absent until created
                    return subprocess.CompletedProcess(cmd, 1, "", "NotFound")
                return super().__call__(cmd, stdin=stdin, timeout=timeout)
        runner = SnapRunner()
        result = weights.stage_weights(
            env, recipe=recipe, runner=runner, ro_via_snapshot="pd-snapshot-class",
            storage_class="hyperdisk-ml-sc", sleep=lambda s: None,
            clock=iter(range(0, 10 ** 6, 20)).__next__)
        self.assertEqual(result["status"], "completed")
        applied = [json.loads(s) for s in runner.stdins if s]
        kinds = [(d["kind"], d["metadata"]["name"]) for d in applied]
        self.assertEqual(kinds, [
            ("PersistentVolumeClaim", "qwen-cache-ro-rw"),
            ("Job", "stage-weights-qwen-cache-ro-rw"),
            ("VolumeSnapshot", "qwen-cache-ro-snap"),
            ("PersistentVolumeClaim", "qwen-cache-ro")])
        ro = applied[-1]["spec"]
        self.assertEqual(ro["accessModes"], ["ReadOnlyMany"])
        self.assertEqual(ro["dataSource"]["name"], "qwen-cache-ro-snap")
        self.assertEqual(ro["storageClassName"], "hyperdisk-ml-sc")
        # --from-snapshot: no Job at all.
        runner = Runner()
        result = weights.stage_weights(env, recipe=recipe, runner=runner,
                                       access_mode="ReadOnlyMany",
                                       from_snapshot="qwen-cache-ro-snap")
        self.assertEqual(result["populated"], "from_snapshot")
        self.assertFalse(runner.matching("apply") and any(
            '"Job"' in (s or "") for s in runner.stdins))

    def test_preflight_reports_the_recipes_claims_and_source_secret(self):
        from framework.deploy import preflight  # noqa: PLC0415
        env = config.DeployEnv(namespace="llmd-study", tpu_accelerator="tpu7x",
                               tpu_topology="2x2x1")
        recipe = copy.deepcopy(AGENTX_RECIPE)
        runner = FakeRunner(fail={"get pvc qwen-cache-ro": "NotFound",
                                  "get secret infx-token": "NotFound"})
        with mock.patch.object(preflight, "tools_present", return_value={
                "kubectl": "/usr/bin/kubectl", "helm": "/usr/bin/helm",
                "gcloud": None}):
            report = preflight.run_preflight(env, runner=runner, recipe=recipe)
        checks = {c["name"]: c for c in report["checks"]}
        self.assertEqual(checks["weights.pvc.qwen-cache-ro"]["status"], "fail")
        self.assertIn("stage-weights", checks["weights.pvc.qwen-cache-ro"]["fix"])
        self.assertIn("--ro-via-snapshot",
                      checks["weights.pvc.qwen-cache-ro"]["fix"])
        self.assertEqual(checks["secret.source_token"]["status"], "fail")
        self.assertIn("weights.pvc.qwen-cache-ro", report["failed"])

    def test_concurrency_is_a_bench_path(self):
        self.assertEqual(release.classify_path("workload.concurrency"),
                         "bench")
        self.assertEqual(release.classify_path("router.plugins.x"), "hot")


if __name__ == "__main__":
    unittest.main()
