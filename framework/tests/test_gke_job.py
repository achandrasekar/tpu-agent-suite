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
"""Unit tests for the gke_job compute driver.

kubectl is replaced by a scripted fake rather than mocked per call, so the
assertions are about the commands the driver issues and the manifest it sends
-- the two things a cluster actually sees. Nothing here talks to a cluster.

Coverage:
- TestGkeJobParams: parameter validation, defaults, CLI flag declaration.
- TestGkeJobSubmit: ConfigMap + Job dispatch, the load-bearing manifest
  fields, Kueue labelling, and rollback when the apply fails.
- TestGkeJobLifecycle: status mapping (including Kueue suspension), logs,
  artifacts, cancellation.
- TestGkeJobRun: synchronous dispatch, cleanup, and the wait timeout.
- TestGkeJobInspect: node-pool reporting and chip contention.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime
import hashlib
import http.server
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

_framework_dir = pathlib.Path(__file__).resolve().parents[1]
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

try:
    from framework.compute import ComputeConfigError
    from framework.compute.drivers import base, gcs_staging, gke_job, jobs
except ImportError:
    from compute import ComputeConfigError
    from compute.drivers import base, gcs_staging, gke_job, jobs

Path = pathlib.Path

try:
    from google.auth import exceptions as _google_auth_errors
except ImportError:
    _google_auth_errors = None


def setUpModule():  # pylint: disable=invalid-name
    """Stop tests from minting real tokens with the developer's ADC.

    Any submit with a staging bucket tries to mint one. By default, act like
    CI (no credentials). Token tests patch this themselves.
    """
    patcher = patch.object(gcs_staging,
                           "mint_bucket_token",
                           side_effect=ComputeConfigError(
                               "no credentials in tests",
                               error_code="NO_CREDENTIALS"))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


@contextlib.contextmanager
def _fake_google_auth(creds: Any):
    """Stand in for `google.auth` so the ambient-credential path is testable
    with or without the SDK installed. The code under test imports it lazily;
    `import google.auth` consults `sys.modules` for the submodule but then
    reads it as an attribute of the `google` package, so both are patched."""
    auth = types.ModuleType("google.auth")
    auth.default = MagicMock(return_value=(creds, "proj"))
    transport = types.ModuleType("google.auth.transport")
    requests = types.ModuleType("google.auth.transport.requests")
    requests.Request = MagicMock()
    auth.transport = transport
    transport.requests = requests
    google_pkg = sys.modules.get("google") or types.ModuleType("google")
    with patch.dict(
            sys.modules, {
                "google": google_pkg,
                "google.auth": auth,
                "google.auth.transport": transport,
                "google.auth.transport.requests": requests,
            }), patch.object(google_pkg, "auth", auth, create=True):
        yield


def workload_command(container: dict[str, Any]) -> list[str]:
    """The container's command behind the cache prologue, if it has one."""
    command = container["command"]
    if command[:2] == ["sh", "-c"] and command[3] == "sh":
        return command[4:]
    return command


class FakeKubectl:
    """Scripted stand-in for kubectl that records what it was asked to do.

    Responses are keyed by a substring of the joined argv rather than the
    whole command, so a test states the one verb it cares about and stays
    readable when an unrelated flag is added to the driver.
    """

    def __init__(self,
                 responses: dict[str, tuple[int, str, str]] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []

    def __call__(self, cmd: list[str], **kwargs: Any):
        self.calls.append(cmd)
        self.stdins.append(kwargs.get("input"))
        joined = " ".join(cmd)
        for key, (code, out, err) in self.responses.items():
            if key in joined:
                return subprocess.CompletedProcess(cmd, code, out, err)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def find(self, needle: str) -> list[str] | None:
        """The first recorded call whose argv contains `needle`."""
        for cmd in self.calls:
            if needle in " ".join(cmd):
                return cmd
        return None

    def applied_manifest(self) -> dict[str, Any]:
        """The Job sent to `kubectl apply -f -`."""
        for cmd, stdin in zip(self.calls, self.stdins):
            if "apply" in cmd and stdin and json.loads(stdin)["kind"] == "Job":
                return json.loads(stdin)
        raise AssertionError("no manifest was applied")


def make_driver(**overrides: Any) -> gke_job.GkeJobDriver:
    params: dict[str, Any] = {
        "namespace": "default",
        "image": "registry/torch-tpu@sha256:abc",
        "tpu_accelerator": "tpu7x",
        "tpu_topology": "2x2x1",
    }
    params.update(overrides)
    return gke_job.GkeJobDriver.from_params(params)


class TestGkeJobParams(unittest.TestCase):

    def test_defaults_applied_in_from_params(self):
        driver = gke_job.GkeJobDriver.from_params({})
        self.assertEqual(driver.namespace, gke_job.DEFAULT_NAMESPACE)
        self.assertEqual(driver.tpu_chips, gke_job.DEFAULT_CHIPS)
        self.assertEqual(driver.python_interpreter, "python3")
        self.assertIsNone(driver.queue)

    def test_empty_strings_are_treated_as_unset(self):
        # environment.json answers default to "" rather than being absent, so
        # a blank must not become a namespace named "" or an image named "".
        driver = gke_job.GkeJobDriver.from_params({
            "namespace": "",
            "image": "",
            "queue": "",
        })
        self.assertEqual(driver.namespace, gke_job.DEFAULT_NAMESPACE)
        self.assertIsNone(driver.image)
        self.assertIsNone(driver.queue)

    def test_inspect_does_not_require_dispatch_fields(self):
        # Rejecting these up front would make `inspect` unusable on a cluster
        # whose node pool has not been chosen yet.
        driver = gke_job.GkeJobDriver.from_params({"namespace": "ns"})
        self.assertIsNone(driver.image)
        self.assertIsNone(driver.tpu_accelerator)

    def test_non_integer_chips_rejected(self):
        with self.assertRaises(ComputeConfigError) as ctx:
            gke_job.GkeJobDriver.from_params({"tpu_chips": "four"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_zero_chips_rejected(self):
        # A pod that requests no chips is scheduled without a device and
        # silently computes on the CPU, which is the one outcome this whole
        # path exists to detect.
        with self.assertRaises(ComputeConfigError) as ctx:
            gke_job.GkeJobDriver.from_params({"tpu_chips": 0})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_string_chips_from_json_are_coerced(self):
        driver = gke_job.GkeJobDriver.from_params({"tpu_chips": "8"})
        self.assertEqual(driver.tpu_chips, 8)

    def test_priority_validation(self):
        for valid in ("medium", "low", "very-low"):
            driver = gke_job.GkeJobDriver.from_params({"priority": valid})
            self.assertEqual(driver.priority, valid)

        with self.assertRaises(ComputeConfigError) as ctx:
            gke_job.GkeJobDriver.from_params({"priority": "high"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_declared_duration_minutes_validation(self):
        driver = gke_job.GkeJobDriver.from_params(
            {"declared_duration_minutes": "15"})
        self.assertEqual(driver.declared_duration_minutes, 15)

        with self.assertRaises(ComputeConfigError) as ctx:
            gke_job.GkeJobDriver.from_params(
                {"declared_duration_minutes": "invalid"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

        driver = gke_job.GkeJobDriver.from_params(
            {"declared_duration_minutes": 0})
        self.assertIsNone(driver.declared_duration_minutes)

        with self.assertRaises(ComputeConfigError) as ctx:
            gke_job.GkeJobDriver.from_params({"declared_duration_minutes": -1})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_shm_size_gb_validation(self):
        # Absent and blank (an unanswered environment.json value) both take
        # the default rather than leaving the pod on the runtime's 64 MB.
        for unset in ({}, {"shm_size_gb": ""}):
            driver = gke_job.GkeJobDriver.from_params(unset)
            self.assertEqual(driver.shm_size_gb, gke_job.DEFAULT_SHM_SIZE_GB)

        driver = gke_job.GkeJobDriver.from_params({"shm_size_gb": "128"})
        self.assertEqual(driver.shm_size_gb, 128)

        for invalid in ("lots", 0, -1):
            with self.assertRaises(ComputeConfigError) as ctx:
                gke_job.GkeJobDriver.from_params({"shm_size_gb": invalid})
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_hf_cache_pvc_validation(self):
        # Absent and blank (an unanswered install question) mean no cache.
        for unset in ({}, {"hf_cache_pvc": ""}, {"hf_cache_pvc": "  "}):
            driver = gke_job.GkeJobDriver.from_params(unset)
            self.assertIsNone(driver.hf_cache_pvc)

        driver = gke_job.GkeJobDriver.from_params(
            {"hf_cache_pvc": " hf-cache.shared "})
        self.assertEqual(driver.hf_cache_pvc, "hf-cache.shared")

        for invalid in ("HF_Cache", "-cache", "cache/", "a" * 254):
            with self.assertRaises(ComputeConfigError) as ctx:
                gke_job.GkeJobDriver.from_params({"hf_cache_pvc": invalid})
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_compile_cache_max_age_days_validation(self):
        self.assertEqual(
            gke_job.GkeJobDriver.from_params({}).compile_cache_max_age_days,
            gke_job.COMPILE_CACHE_MAX_AGE_DAYS)
        self.assertEqual(
            gke_job.GkeJobDriver.from_params({
                "compile_cache_max_age_days": 0
            }).compile_cache_max_age_days, 0)
        for invalid in ("soon", -1):
            with self.assertRaises(ComputeConfigError) as ctx:
                gke_job.GkeJobDriver.from_params(
                    {"compile_cache_max_age_days": invalid})
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_namespace_is_never_derived_from_the_queue(self):
        """Kueue co-locates a LocalQueue with its Jobs; it does not co-name.

        XPK's `multislice-queue` and Kueue's quickstart `user-queue` both live
        in `default`, so a namespace named after the queue would not exist.
        """
        driver = gke_job.GkeJobDriver.from_params({"queue": "priority-dev"})
        self.assertEqual(driver.namespace, gke_job.DEFAULT_NAMESPACE)

        # Explicit non-default namespace is preserved
        driver = gke_job.GkeJobDriver.from_params({
            "queue": "priority-dev",
            "namespace": "custom-ns"
        })
        self.assertEqual(driver.namespace, "custom-ns")

        # An explicit "default" is kept, queue or no queue.
        driver = gke_job.GkeJobDriver.from_params({
            "queue": "priority-dev",
            "namespace": "default"
        })
        self.assertEqual(driver.namespace, "default")
        driver = gke_job.GkeJobDriver(queue="priority-dev",
                                      namespace="default")
        self.assertEqual(driver.namespace, "default")

        # Blank is unset, as the installer's empty answers are.
        driver = gke_job.GkeJobDriver.from_params({
            "queue": "priority-dev",
            "namespace": " "
        })
        self.assertEqual(driver.namespace, gke_job.DEFAULT_NAMESPACE)

    def test_cli_flags_registered_and_not_reserved(self):
        parser = argparse.ArgumentParser(add_help=False)
        gke_job.GkeJobDriver.register_cli_args(parser)
        args = parser.parse_args([
            "--tpu-accelerator", "tpu7x", "--queue", "q",
            "--declared-duration-minutes", "30", "--job-name-prefix", "agent"
        ])
        self.assertEqual(args.tpu_accelerator, "tpu7x")
        self.assertEqual(args.queue, "q")
        self.assertEqual(args.declared_duration_minutes, 30)
        self.assertEqual(args.job_name_prefix, "agent")
        self.assertFalse(hasattr(args, "cpu_requests"))
        self.assertEqual(
            parser.parse_args(["--cpu-requests",
                               "cpu=2,memory=8Gi"]).cpu_requests,
            "cpu=2,memory=8Gi")
        # Unspecified flags must be absent, not None: the CLI forwards every
        # registered flag it finds into param_overrides, which outrank
        # environment.json.
        self.assertFalse(hasattr(args, "image"))

    def test_kube_context_from_params(self):
        for raw, want in ((None, None), ("  ", None), (" ctx-a ", "ctx-a")):
            self.assertEqual(
                gke_job.GkeJobDriver.from_params({
                    "kube_context": raw
                }).kube_context, want)
        self.assertEqual(
            make_driver(kube_context="ctx-a",
                        kubeconfig="/k")._kubectl_cmd(["get", "pods"]), [
                            "kubectl", "--kubeconfig", "/k", "--context",
                            "ctx-a", "get", "pods"
                        ])

    def test_preflight_with_bad_context_still_checks_the_profile(self):
        driver = make_driver(kube_context="nope")
        check = gke_job.base.PreflightCheck
        with patch.object(gke_job.base, "kube_context_check",
                          return_value=check("kube_context", "fail", "x")), \
             patch.object(gke_job.base, "gke_credential_checks",
                          return_value=[check("gcloud:account", "ok", "me")]), \
             patch.object(gke_job, "_image_check",
                          return_value=check("image", "ok", "i")), \
             patch.object(gke_job.auth_utils, "_cluster") as cluster, \
             patch.object(driver, "_preflight_cluster") as nodes:
            checks: list = []
            self.assertFalse(driver._preflight_target(checks))
        cluster.assert_not_called()
        nodes.assert_not_called()
        self.assertEqual([c.name for c in checks],
                         ["kube_context", "gcloud:account", "image"])

    def test_job_name_prefix_validation(self):
        for raw, want in ((None, None), (" ", None), ("Agent_Evals",
                                                      "agent-evals")):
            driver = gke_job.GkeJobDriver.from_params({"job_name_prefix": raw})
            self.assertEqual(driver.job_name_prefix, want)
        for invalid in ("___", "a" * (gke_job.MAX_JOB_NAME_PREFIX_LEN + 1)):
            with self.assertRaises(ComputeConfigError) as ctx:
                gke_job.GkeJobDriver.from_params({"job_name_prefix": invalid})
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")


class TestGkeJobName(unittest.TestCase):
    """How `_job_name` turns a requested name into a Job name."""

    @patch.object(gke_job.jobs, "new_job_id", return_value="job-1")
    def test_names_are_cleaned_and_prefixed_once(self, _):
        plain, agent = make_driver(), make_driver(job_name_prefix="agent")
        for driver, requested, unique, want in (
            (plain, None, True, "job-1"),  # A fresh id needs no suffix.
            (plain, " My_Job.v2 ", False, "my-job-v2"),
            (agent, None, False, "agent-job-1"),
            (agent, "bench", False, "agent-bench"),
            (agent, "agent-bench", False, "agent-bench"),
            (agent, "agent", False, "agent"),
        ):
            self.assertEqual(driver._job_name(requested, unique), want)
        with self.assertRaises(ComputeConfigError) as ctx:
            plain._job_name("___", unique=False)
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")

    def test_unique_suffix_survives_trimming_and_renaming(self):
        driver = make_driver(job_name_prefix="agent")
        name = driver._job_name("Bench_" + "x" * 100, unique=True)
        self.assertEqual(len(name), gke_job.MAX_JOB_NAME_LEN)
        self.assertRegex(name, r"^agent-bench-x+-[0-9a-f]{6}$")
        self.assertNotEqual(driver._job_name("b", unique=True),
                            driver._job_name("b", unique=True))
        # `run` passes the name it picked to `submit`, which names it again.
        self.assertEqual(driver._job_name(name, unique=False), name)


class TestGkeJobSubmit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "hello_tpu.py"
        self.script.write_text("print('hi')\n", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(_job_store())

    def submit(self, fake: FakeKubectl, **overrides: Any) -> str:
        driver = make_driver(**overrides)
        with patch.object(gke_job.subprocess, "run", fake):
            return driver.submit(str(self.script), job_name="run-1")

    def test_configmap_created_before_job(self):
        fake = FakeKubectl()
        self.submit(fake)
        # After the lookup that checks the name is free.
        verbs = [" ".join(c) for c in fake.calls]
        self.assertIn("get configmap,job run-1", verbs[0])
        self.assertIn("create configmap run-1", verbs[1])
        self.assertIn(f"--from-file=hello_tpu.py={self.script}", verbs[1])
        self.assertIn("apply", verbs[2])

    def test_job_id_is_the_job_name(self):
        self.assertEqual(self.submit(FakeKubectl()), "run-1")

    def test_submit_adds_the_prefix_but_no_suffix(self):
        fake = FakeKubectl()
        self.assertEqual(self.submit(fake, job_name_prefix="agent"),
                         "agent-run-1")
        self.assertEqual(fake.applied_manifest()["metadata"]["name"],
                         "agent-run-1")

    def test_a_name_held_outside_the_run_gets_a_suffix(self):
        # Refusing with "cancel it if it is yours" led an agent to cancel
        # another run's job.
        taken = "get configmap,job agent-run-1 "
        fake = FakeKubectl({taken: (0, "job.batch/agent-run-1\n", "")})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            job_id = self.submit(fake, job_name_prefix="agent")
        self.assertRegex(job_id, r"^agent-run-1-[0-9a-f]{6}$")
        self.assertEqual(fake.applied_manifest()["metadata"]["name"], job_id)
        self.assertIn(f"submitting as {job_id!r}", err.getvalue())
        self.assertIsNone(fake.find("delete"))

    def test_a_name_held_by_this_runs_live_job_is_refused(self):
        jobs.JobRecord.create(job_id="agent-run-1",
                              backend="gke_job",
                              run_id="run-a")
        fake = FakeKubectl({"get configmap,job": (0, "job.batch/x\n", "")})
        driver = make_driver(job_name_prefix="agent")
        with patch.object(gke_job.subprocess, "run", fake), \
                self.assertRaises(ComputeConfigError) as ctx:
            driver.submit(str(self.script), job_name="run-1", run_id="run-a")
        self.assertEqual(ctx.exception.error_code, "NAME_TAKEN")
        self.assertIn("already this run's job", ctx.exception.message)
        self.assertEqual(len(fake.calls), 1)

    def test_a_name_whose_suffixes_are_all_taken_is_refused(self):
        fake = FakeKubectl({"get configmap,job": (0, "job.batch/x\n", "")})
        with self.assertRaises(ComputeConfigError) as ctx:
            self.submit(fake, job_name_prefix="agent")
        self.assertEqual(ctx.exception.error_code, "NAME_TAKEN")
        self.assertIn("'agent-run-1' (from --name 'run-1')",
                      ctx.exception.message)
        self.assertNotIn("job cancel", ctx.exception.message)
        # The name, then each suffixed candidate; nothing created.
        self.assertEqual(len(fake.calls), 1 + gke_job.NAME_SUFFIX_ATTEMPTS)
        self.assertIsNone(fake.find("create"))

    def test_a_failed_lookup_leaves_the_check_to_the_create(self):
        # e.g. an account allowed to create ConfigMaps but not to read them.
        fake = FakeKubectl(
            {"get configmap,job": (1, "", "Forbidden: cannot get")})
        self.assertEqual(self.submit(fake), "run-1")
        self.assertIsNotNone(fake.find("apply"))

    def test_manifest_carries_the_load_bearing_fields(self):
        fake = FakeKubectl()
        self.submit(fake)
        manifest = fake.applied_manifest()
        pod_spec = manifest["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        self.assertEqual(manifest["spec"]["backoffLimit"], 0)
        self.assertEqual(
            pod_spec["nodeSelector"],
            {
                gke_job.ACCELERATOR_LABEL: "tpu7x",
                gke_job.TOPOLOGY_LABEL: "2x2x1",
            },
        )
        self.assertNotIn("serviceAccountName", pod_spec)
        self.assertEqual(pod_spec["tolerations"][0]["key"],
                         gke_job.TPU_RESOURCE)
        self.assertEqual(
            container["resources"]["limits"][gke_job.TPU_RESOURCE],
            gke_job.DEFAULT_CHIPS,
        )
        # /workspace is taken inside the torch_tpu images; mounting over it
        # fails container creation.
        self.assertEqual(container["volumeMounts"][0]["mountPath"], "/app")
        self.assertEqual(workload_command(container),
                         ["python3", "/app/hello_tpu.py"])
        self.assertEqual(pod_spec["volumes"][0]["configMap"]["name"], "run-1")

    def test_manifest_mounts_a_memory_backed_dev_shm(self):
        # The runtime's 64 MB default breaks vLLM past TP=4 and torch_tpu's
        # compilation cache, and the container cannot remount it.
        fake = FakeKubectl()
        self.submit(fake, shm_size_gb=16)
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        mounts = {
            m["name"]: m["mountPath"]
            for m in pod_spec["containers"][0]["volumeMounts"]
        }
        volumes = {v["name"]: v for v in pod_spec["volumes"]}
        self.assertEqual(mounts["dshm"], "/dev/shm")
        self.assertEqual(volumes["dshm"]["emptyDir"], {
            "medium": "Memory",
            "sizeLimit": "16Gi"
        })
        # Its cap, for fetch_weights to size a download against: statvfs in
        # the pod can report the node's whole memory instead.
        env = {e["name"]: e["value"] for e in pod_spec["containers"][0]["env"]}
        self.assertEqual(env["TPU_AGENT_SUITE_SHM_LIMIT_BYTES"],
                         str(16 * 2**30))

    def test_manifest_mounts_the_hf_cache_pvc(self):
        # Without it each pod starts with an empty filesystem and downloads
        # the model again: a 65 GB Qwen took ~50 minutes per pod.
        fake = FakeKubectl()
        self.submit(fake, hf_cache_pvc="hf-cache")
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
        volumes = {v["name"]: v for v in pod_spec["volumes"]}
        self.assertEqual(mounts["hf-cache"], "/hf-cache")
        self.assertEqual(volumes["hf-cache"]["persistentVolumeClaim"],
                         {"claimName": "hf-cache"})
        # The hub cache only: HF_HOME would also put the token on it.
        env = {e["name"]: e["value"] for e in container["env"]}
        env.pop("TPU_AGENT_SUITE_SHM_LIMIT_BYTES")
        self.assertEqual(env, {"HF_HUB_CACHE": "/hf-cache"})
        # The compile caches are set in the container, not the pod spec,
        # which would override a cache the image's ENV points at.
        prologue = container["command"][:4]
        cache = "/hf-cache/compile-cache/tpu7x-2x2x1/sha256-abc"
        names = ("VLLM_CACHE_ROOT",
                 "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
                 "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
        cases = [
            # (the image's ENV, what the workload sees)
            ({}, {
                "VLLM_CACHE_ROOT": f"{cache}/vllm",
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
                f"{cache}/torch_tpu",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": f"{cache}/torch_tpu",
            }),
            # An image with its own caches keeps them; torch_tpu's two
            # spellings move together.
            ({
                "VLLM_CACHE_ROOT": "/opt/vllm",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/opt/tpu",
            }, {
                "VLLM_CACHE_ROOT": "/opt/vllm",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/opt/tpu",
            }),
        ]
        for image_env, seen in cases:
            with self.subTest(image_env=image_env):
                proc = subprocess.run(prologue + [
                    sys.executable, "-c",
                    "import json, os, sys; print(json.dumps({n: "
                    "os.environ[n] for n in sys.argv[1:] "
                    "if n in os.environ}))", *names
                ],
                                      env={
                                          "PATH": os.environ["PATH"],
                                          **image_env
                                      },
                                      capture_output=True,
                                      text=True,
                                      check=True)
                self.assertEqual(json.loads(proc.stdout), seen)
        self.assertRegex(gke_job.compile_cache_key("gcr.io/p/img:v1"),
                         r"^img-v1-[0-9a-f]{12}$")
        self.assertEqual(mounts["dshm"], "/dev/shm")

    def test_compile_cache_prune_removes_only_stale_folders(self):
        mount = Path(self.tmp.name) / "vol"
        shape = mount / "compile-cache" / "tpu7x-2x2x1"
        # (folder, age in days of its file, of its stamp or None, kept)
        folders = [("old", 30, 30, False), ("old-unstamped", 30, None, False),
                   ("restamped", 30, 1, True),
                   ("new-unstamped", 1, None, True), ("current", 30, 30, True)]
        for name, file_age, stamp_age, _ in folders:
            (shape / name / "vllm").mkdir(parents=True)
            for path, age in ((shape / name / "vllm" / "f", file_age),
                              (shape / name / ".last_used", stamp_age)):
                if age is not None:
                    subprocess.run(["touch", "-d", f"{age} days ago", path],
                                   check=True)
            for path in (shape / name / "vllm", shape / name):
                subprocess.run(["touch", "-d", f"{file_age} days ago", path],
                               check=True)
        # Over 90% full: the caches stay in the pod, so `current` is not
        # restamped, and it is still never removed.
        bin_dir = Path(self.tmp.name) / "bin"
        bin_dir.mkdir()
        (bin_dir /
         "df").write_text("#!/bin/sh\necho head\necho fs 100 95 5 95% /vol\n")
        (bin_dir / "df").chmod(0o755)
        command = gke_job._compile_cache_command([
            sys.executable, "-c", "import os; print(os.environ['"
            "VLLM_CACHE_ROOT'])"
        ], str(shape / "current"), 14, str(mount))
        # Returns once the background prune closes the pipes it inherited.
        proc = subprocess.run(command,
                              env={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
                              capture_output=True,
                              text=True,
                              check=True)
        self.assertEqual(proc.stdout.strip(),
                         gke_job.POD_CACHE_ENV["VLLM_CACHE_ROOT"][0])
        self.assertIn("95% full", proc.stderr)
        for name, _, _, kept in folders:
            self.assertEqual((shape / name).is_dir(), kept, name)

    def test_manifest_without_hf_cache_pvc_caches_on_dev_shm(self):
        # The node's overlay can hold less than one model's weights; a pod
        # that downloads onto it is evicted with exit 137 part-way.
        fake = FakeKubectl()
        self.submit(fake)
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        # Set in the container, not the pod spec, which would override the
        # image's own ENV.
        self.assertEqual([e["name"] for e in container["env"]],
                         ["TPU_AGENT_SUITE_SHM_LIMIT_BYTES"])
        prologue = container["command"][:4]
        defaults = {
            "HF_HUB_CACHE": "/dev/shm/cache/huggingface/hub",
            "TPU_AGENT_SUITE_POD_HF_CACHE": "/dev/shm/cache/huggingface/hub",
            "VLLM_CACHE_ROOT": "/dev/shm/cache/vllm",
            "XDG_CACHE_HOME": "/dev/shm/cache",
            "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/cache/torchinductor",
            "TMPDIR": "/dev/shm",
        }
        cases = [
            # (the image's ENV, what the workload sees)
            ({}, defaults),
            # An image that moves its caches keeps them: HF_HOME moves the
            # hub cache and XDG_CACHE_HOME vLLM's, so neither is set, and
            # the hub cache is not named as the pod's to prune.
            ({
                "HF_HOME": "/opt/hf",
                "XDG_CACHE_HOME": "/opt/cache",
                "TMPDIR": "/scratch",
            }, {
                "HF_HOME": "/opt/hf",
                "XDG_CACHE_HOME": "/opt/cache",
                "TMPDIR": "/scratch",
                "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/cache/torchinductor",
            }),
        ]
        names = sorted({*defaults, "HF_HOME"})
        for image_env, seen in cases:
            with self.subTest(image_env=image_env):
                proc = subprocess.run(prologue + [
                    sys.executable, "-c",
                    "import json, os, sys; print(json.dumps({n: "
                    "os.environ[n] for n in sys.argv[1:] "
                    "if n in os.environ}))", *names
                ],
                                      env={
                                          "PATH": os.environ["PATH"],
                                          **image_env
                                      },
                                      capture_output=True,
                                      text=True,
                                      check=True)
                self.assertEqual(json.loads(proc.stdout), seen)
        # No volume of its own: it shares /dev/shm's shm_size_gb cap.
        self.assertEqual({v["name"]
                          for v in pod_spec["volumes"]}, {"script", "dshm"})

    def test_manifest_disables_service_links(self):
        # Service links put every Service in a shared namespace into the
        # pod's environment, e.g. VLLM_..._PORT=tcp://..., which vLLM
        # misreads as its own settings.
        fake = FakeKubectl()
        self.submit(fake)
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        self.assertIs(pod_spec["enableServiceLinks"], False)

    def test_script_args_reach_the_container_command(self):
        fake = FakeKubectl()
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.submit(str(self.script),
                          job_name="run-1",
                          args=["--dim", "8"])
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        command = workload_command(container)
        self.assertEqual(command[-2:], ["--dim", "8"])

    def test_no_kueue_labels_on_a_direct_cluster(self):
        fake = FakeKubectl()
        self.submit(fake)
        self.assertNotIn("labels", fake.applied_manifest()["metadata"])
        self.assertNotIn("metadata",
                         fake.applied_manifest()["spec"]["template"])

    def test_kueue_labels_applied_when_queue_configured(self):
        fake = FakeKubectl()
        self.submit(fake, queue="default", priority="low")
        manifest = fake.applied_manifest()
        labels = manifest["metadata"]["labels"]
        self.assertEqual(labels[gke_job.QUEUE_LABEL], "default")
        self.assertEqual(labels[gke_job.PRIORITY_LABEL], "low")

        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        self.assertEqual(pod_labels["namespace"], "default")
        self.assertEqual(pod_labels["priority"], "low")

    def test_queue_without_priority_omits_priority_label(self):
        """A cluster with no WorkloadPriorityClass admits unlabelled Jobs only.

        Kueue refuses a Job naming a priority class that does not exist, so
        a default nobody chose would make every Job inadmissible there.
        """
        for priority in (None, ""):
            with self.subTest(priority=priority):
                fake = FakeKubectl()
                self.submit(fake, queue="default", priority=priority)
                manifest = fake.applied_manifest()
                labels = manifest["metadata"]["labels"]
                self.assertEqual(labels[gke_job.QUEUE_LABEL], "default")
                self.assertNotIn(gke_job.PRIORITY_LABEL, labels)
                pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
                self.assertNotIn("priority", pod_labels)

    def test_kueue_pod_labels_with_declared_duration(self):
        fake = FakeKubectl()
        self.submit(fake, queue="default", declared_duration_minutes=15)
        pod_labels = fake.applied_manifest(
        )["spec"]["template"]["metadata"]["labels"]
        self.assertEqual(pod_labels["declared-duration-minutes"], "15")

    def test_chip_count_is_configurable(self):
        fake = FakeKubectl()
        self.submit(fake, tpu_chips=8)
        limits = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        self.assertEqual(limits[gke_job.TPU_RESOURCE], 8)

    def test_missing_dispatch_fields_named_together(self):
        driver = gke_job.GkeJobDriver.from_params({})
        with self.assertRaises(ComputeConfigError) as ctx:
            driver.submit(str(self.script))
        message = ctx.exception.message
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")
        # All three at once: reporting them one per attempt costs three round
        # trips to find out what a dispatch needs.
        for field in ("image", "tpu_accelerator", "tpu_topology"):
            self.assertIn(field, message)
        self.assertIn("--tpu-accelerator", message)

    def test_configmap_removed_when_the_apply_fails(self):
        # Otherwise the ConfigMap is orphaned and invisible, and a retry
        # collides with the name instead of dispatching.
        fake = FakeKubectl({"apply": (1, "", "admission webhook denied")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            with self.assertRaises(ComputeConfigError):
                driver.submit(str(self.script), job_name="run-1")
        self.assertIsNotNone(fake.find("delete configmap run-1"))

    def test_kubectl_stderr_reaches_the_error_message(self):
        fake = FakeKubectl({"create configmap": (1, "", "Forbidden: no RBAC")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            with self.assertRaises(ComputeConfigError) as ctx:
                driver.submit(str(self.script), job_name="run-1")
        self.assertEqual(ctx.exception.error_code, "KUBECTL_FAILED")
        self.assertIn("Forbidden: no RBAC", ctx.exception.message)

    def test_kubeconfig_passed_to_every_call(self):
        fake = FakeKubectl()
        self.submit(fake, kubeconfig="/tmp/kubeconfig")
        for cmd in fake.calls:
            self.assertIn("--kubeconfig", cmd)

    def test_kube_context_passed_to_every_call_only_when_set(self):
        fake = FakeKubectl()
        self.submit(fake, kube_context="ctx-a")
        for cmd in fake.calls:
            self.assertEqual(cmd[cmd.index("--context") + 1], "ctx-a")
        fake = FakeKubectl()
        self.submit(fake)
        for cmd in fake.calls:
            self.assertNotIn("--context", cmd)

    def test_colocate_takes_the_servers_placement_without_chips(self):
        server = {
            "nodeName": "n1",
            "nodeSelector": {
                "flavor": "tpu"
            },
            "tolerations": [{
                "key": "reservation"
            }]
        }
        pods = json.dumps({"items": [{"spec": server}]})
        fake = FakeKubectl({"get pods": (0, pods, "")})
        self.submit(fake, colocate_with="srv", colocate_requests="cpu=1")
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        self.assertEqual(pod_spec["nodeSelector"], {
            "flavor": "tpu",
            gke_job.HOSTNAME_LABEL: "n1"
        })
        self.assertEqual(pod_spec["tolerations"], server["tolerations"])
        self.assertEqual(pod_spec["containers"][0]["resources"],
                         {"requests": {
                             "cpu": "1"
                         }})
        with self.assertRaises(ComputeConfigError):
            make_driver(colocate_requests="cpu")
        with self.assertRaises(ComputeConfigError):
            make_driver(colocate_with="srv")

    def test_cpu_requests_book_no_tpu_node(self):
        """No chips, and nothing that sends the pod to a TPU node: only the
        Workload Identity selector, no TPU toleration. Not with colocation."""
        fake = FakeKubectl()
        self.submit(fake,
                    cpu_requests="cpu=2,memory=8Gi",
                    pod_service_account="agent-sa")
        pod_spec = fake.applied_manifest()["spec"]["template"]["spec"]
        self.assertEqual(pod_spec["nodeSelector"],
                         {gke_job.WORKLOAD_IDENTITY_LABEL: "true"})
        self.assertNotIn("tolerations", pod_spec)
        self.assertEqual(pod_spec["containers"][0]["resources"],
                         {"requests": {
                             "cpu": "2",
                             "memory": "8Gi"
                         }})
        with self.assertRaises(ComputeConfigError):
            make_driver(cpu_requests="cpu=2",
                        colocate_with="srv",
                        colocate_requests="cpu=1")

    def test_a_cpu_job_is_told_the_profiles_tpu_generation(self):
        """As probe names it, for a check that plans for that TPU without a
        chip; a TPU Job has the chip and is told nothing."""
        for cpu_requests, expected in (("cpu=2", "tpu7x"), (None, None)):
            with self.subTest(cpu_requests=cpu_requests):
                fake = FakeKubectl()
                self.submit(fake, cpu_requests=cpu_requests)
                container = fake.applied_manifest(
                )["spec"]["template"]["spec"]["containers"][0]
                env = {e["name"]: e.get("value") for e in container["env"]}
                self.assertEqual(env.get(gke_job.CHIPLESS_TPU_ENV), expected)

    def test_a_cpu_run_that_needs_a_chip_runs_again_on_a_tpu_node(self):
        """Once, and only when no CPU node took it or it exited
        NEEDS_TPU_EXIT_CODE; any other result is the run's own."""

        def ran(code, **details):
            return base.SynchronousExecutionResult(exit_code=code,
                                                   stdout="",
                                                   stderr="",
                                                   details=details)

        cases = [
            (ran(1, never_started=True), "no CPU node took it"),
            (ran(gke_job.NEEDS_TPU_EXIT_CODE), "it exited 75"),
            (ran(1), None),
        ]
        for first, why in cases:
            with self.subTest(why=why):
                driver = make_driver(cpu_requests="cpu=2")
                results, held = [first, ran(0)], []

                def once(**_kwargs):
                    held.append(dict(driver.cpu_requests))
                    return results.pop(0)

                with patch.object(driver, "_run_once", side_effect=once), \
                        contextlib.redirect_stderr(io.StringIO()):
                    result = driver.run(str(self.script))
                if why is None:
                    self.assertEqual(held, [{"cpu": "2"}])
                    self.assertIs(result, first)
                    continue
                self.assertEqual(held, [{"cpu": "2"}, {}])
                self.assertEqual(result.exit_code, 0)
                self.assertIn(why, result.details["cpu_fallback"])
                self.assertEqual(driver.cpu_requests, {"cpu": "2"})

    def test_serve_adds_a_headless_service_owned_by_the_job(self):
        fake = FakeKubectl({"jsonpath": (0, "uid-1", "")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.submit(str(self.script), job_name="run-1", serve=True)
        service = json.loads(fake.stdins[-1])
        self.assertEqual(service["spec"]["clusterIP"], "None")
        self.assertEqual(service["metadata"]["ownerReferences"][0]["uid"],
                         "uid-1")

    def test_a_colocated_client_gives_its_batch_server_the_service(self):
        """The server needs no flag to be reached, and stays a batch Job."""
        pods = json.dumps({"items": [{"spec": {"nodeName": "n1"}}]})
        fake = FakeKubectl({
            "get pods": (0, pods, ""),
            "get job srv": (0, "uid-srv", "")
        })
        self.submit(fake, colocate_with="srv", colocate_requests="cpu=1")
        services = [
            json.loads(stdin) for stdin in fake.stdins
            if stdin and '"kind": "Service"' in stdin
        ]
        self.assertEqual(len(services), 1)
        self.assertEqual(services[0]["metadata"]["name"], "srv")
        self.assertEqual(services[0]["spec"]["selector"], {"job-name": "srv"})
        self.assertEqual(services[0]["metadata"]["ownerReferences"][0]["uid"],
                         "uid-srv")
        # Before the client Job, so its address exists when the client starts.
        kinds = [json.loads(stdin)["kind"] for stdin in fake.stdins if stdin]
        self.assertEqual(kinds, ["Service", "Job"])

    def test_a_batch_job_gets_no_service(self):
        fake = FakeKubectl({"jsonpath": (0, "uid-1", "")})
        self.submit(fake)
        self.assertFalse(
            any('"kind": "Service"' in (stdin or "") for stdin in fake.stdins))

    def test_submit_keeps_the_job_uid(self):
        """The UID comes back from the apply itself, with no extra call."""
        fake = FakeKubectl({"jsonpath={.metadata.uid}": (0, "uid-7\n", "")})
        with patch.object(gke_job.subprocess, "run", fake):
            make_driver().submit(str(self.script), job_name="run-1")
        self.assertEqual(jobs.JobRecord.load("run-1").extra["uid"], "uid-7")
        self.assertIn("apply", fake.find("jsonpath={.metadata.uid}"))
        self.assertIsNone(fake.find("get job"))

    def test_a_refused_service_keeps_the_job_and_its_record(self):
        """RBAC refusing the Service warns; the running Job stays tracked.

        A 403 goes through the ADC retry and comes back as
        ClusterAccessDenied, not KubectlError. It used to escape submit()
        after the Job was applied, delete the Job's ConfigMap in the cleanup
        and leave no record of the job id.
        """
        forbidden = ('Error from server (Forbidden): services is forbidden: '
                     'User "x" cannot create resource "services"')

        class RefuseService(FakeKubectl):

            def __call__(self, cmd, **kwargs):
                if '"kind": "Service"' in (kwargs.get("input") or ""):
                    self.calls.append(cmd)
                    self.stdins.append(kwargs.get("input"))
                    return subprocess.CompletedProcess(cmd, 1, "", forbidden)
                return super().__call__(cmd, **kwargs)

        fake = RefuseService({"jsonpath": (0, "uid-1", "")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gke_job.auth_utils, "adc_token",
                             return_value=("", "no ADC")), \
                patch("sys.stderr", io.StringIO()) as err:
            job_id = driver.submit(str(self.script),
                                   job_name="run-1",
                                   serve=True)
        self.assertEqual(job_id, "run-1")
        self.assertIsNotNone(jobs.JobRecord.load("run-1"))
        self.assertIsNone(fake.find("delete configmap"))
        self.assertIn("no Service for run-1", err.getvalue())


def job_json(**status: Any) -> str:
    spec = {"suspend": status.pop("suspend", False)}
    return json.dumps({"spec": spec, "status": status})


def failed_pod(*,
               exit_code: int | None = None,
               container_reason: str | None = None,
               reason: str | None = None,
               message: str | None = None,
               conditions: list[dict[str, Any]] | None = None,
               phase: str = "Failed",
               node: str | None = "gke-tpu-node-1",
               name: str = "run-1-abcde") -> dict[str, Any]:
    """A Job's pod as `kubectl get pods -o json` lists it.

    The container is terminated only when `exit_code` or
    `container_reason` is given; `reason`, `message` and `conditions` are
    the pod's own status. Defaults to a failed pod on a TPU node; pass
    `phase="Running"` for a live one.
    """
    status: dict[str, Any] = {"phase": phase}
    if exit_code is not None or container_reason is not None:
        terminated = {"exitCode": exit_code, "reason": container_reason}
        status["containerStatuses"] = [{
            "state": {
                "terminated": {
                    k: v
                    for k, v in terminated.items() if v is not None
                }
            }
        }]
    for key, value in (("reason", reason), ("message", message), ("conditions",
                                                                  conditions)):
        if value is not None:
            status[key] = value
    pod: dict[str, Any] = {"metadata": {"name": name}, "status": status}
    if node:
        pod["spec"] = {"nodeName": node}
    return pod


_RUNNING_POD = json.dumps({"items": [{"status": {"phase": "Running"}}]})
_DONE_POD = json.dumps({"items": [{"status": {"phase": "Succeeded"}}]})
_FAILED_POD = json.dumps({
    "items": [{
        "status": {
            "containerStatuses": [{
                "state": {
                    "terminated": {
                        "exitCode": 1,
                        "reason": "Error"
                    }
                }
            }]
        }
    }]
})


@contextlib.contextmanager
def _job_store():
    """A throwaway job-record store, so tests never touch ~/.tpu-agent-suite."""
    with tempfile.TemporaryDirectory() as home, patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": home}):
        yield


class TestGkeJobLifecycle(unittest.TestCase):

    def setUp(self):
        self.enterContext(_job_store())

    def status(self, fake: FakeKubectl):
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            return driver.get_status("run-1")

    def test_succeeded_job_is_completed_with_exit_zero(self):
        fake = FakeKubectl({
            "get job":
            (0, job_json(succeeded=1, startTime="t0", completionTime="t1"), "")
        })
        result = self.status(fake)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.end_time, "t1")

    def test_active_job_with_a_running_pod_is_running(self):
        pods = json.dumps({"items": [{"status": {"phase": "Running"}}]})
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        self.assertEqual(self.status(fake).status, "RUNNING")

    def test_a_hung_status_read_times_out_as_a_kubectl_error(self):
        # Unbounded, one unreachable API server hung `job list` for good.
        def hang(cmd: list[str], **kwargs: Any):
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with self.assertRaises(gke_job.KubectlError) as ctx:
            self.status(hang)
        self.assertIn(f"no answer within {gke_job.KUBECTL_READ_TIMEOUT_SEC}s",
                      str(ctx.exception))

    def test_active_job_with_an_unschedulable_pod_is_pending(self):
        """`active` counts pending pods too, so it cannot mean "running".

        A pod that will never schedule -- no free chips, or a selector
        matching nothing -- keeps the Job active indefinitely. Reported as
        RUNNING it looks like healthy progress until the caller's timeout
        expires, which is the one thing a caller must not be told here.
        """
        pods = json.dumps({
            "items": [{
                "status": {
                    "phase":
                    "Pending",
                    "conditions": [{
                        "type":
                        "PodScheduled",
                        "status":
                        "False",
                        "reason":
                        "Unschedulable",
                        "message": ("0/2 nodes are available: 2 Insufficient "
                                    "google.com/tpu."),
                    }],
                }
            }]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        result = self.status(fake)
        self.assertEqual(result.status, "PENDING")
        self.assertIn("Insufficient google.com/tpu", result.error)
        self.assertEqual(result.details["pod_phase"], "Pending")
        # The reason is what `job wait` reads: unbound, no queue.
        self.assertEqual(result.details["pending_reason"],
                         gke_job.base.PENDING_UNSCHEDULED)

        # The same pod under a queue is admitted but unschedulable.
        labelled = json.loads(job_json(active=1))
        labelled["metadata"] = {"labels": {gke_job.QUEUE_LABEL: "lq"}}
        fake.responses["get job"] = (0, json.dumps(labelled), "")
        self.assertEqual(
            self.status(fake).details["pending_reason"],
            gke_job.base.PENDING_ADMITTED_NO_NODE)

        # Bound to a node, it is starting, queue or not.
        bound = json.loads(pods)
        bound["items"][0]["spec"] = {"nodeName": "gke-tpu-node-1"}
        fake.responses["get pods"] = (0, json.dumps(bound), "")
        self.assertEqual(
            self.status(fake).details["pending_reason"],
            gke_job.base.PENDING_STARTING)

    FIRST_FAILURE = "2026-01-01T00:04:00Z"

    @staticmethod
    def _pulling_pods(reason: str) -> str:
        return json.dumps({
            "items": [{
                "metadata": {
                    "name": "run-1-abcde"
                },
                "status": {
                    "phase":
                    "Pending",
                    "containerStatuses": [{
                        "image": "img:1",
                        "state": {
                            "waiting": {
                                "reason": reason,
                                "message": "manifest for img:1 not found"
                            }
                        },
                    }],
                },
            }]
        })

    @classmethod
    def _pull_events(cls, span: float) -> str:
        """The kubelet's events for pulls failing from FIRST_FAILURE for
        `span` seconds, and for back-offs that went on for an hour."""
        first = gke_job._parse_time(cls.FIRST_FAILURE)

        def stamp(offset: float) -> str:
            return datetime.datetime.fromtimestamp(
                first + offset,
                datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Neither of these is a pull attempt: they repeat while it waits.
        waiting = [("BackOff", 'Back-off pulling image "img:1"'),
                   ("Failed", "Error: ImagePullBackOff")]
        return json.dumps({
            "items": [{
                "reason": "Pulling",
                "message": 'Pulling image "img:1"',
                "firstTimestamp": stamp(-240)
            }, {
                "reason": "Failed",
                "message": 'Failed to pull image "img:1": not found',
                "firstTimestamp": stamp(0),
                "lastTimestamp": stamp(span),
            }] + [{
                "reason": reason,
                "message": message,
                "firstTimestamp": stamp(0),
                "lastTimestamp": stamp(3600),
            } for reason, message in waiting]
        })

    def _pulling(self,
                 reason: str = "ImagePullBackOff",
                 span: float = gke_job.PULL_FAILURE_GRACE_SEC,
                 **responses: Any) -> FakeKubectl:
        return FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, self._pulling_pods(reason), ""),
            "get events": (0, self._pull_events(span), ""),
            **responses,
        })

    def test_pulls_failing_for_the_grace_period_fail_the_job(self):
        """The kubelet retries a bad pull forever; the Job must not wait."""
        fake = self._pulling()
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertIn("ImagePullBackOff: manifest for img:1 not found",
                      result.error)
        self.assertIn("involvedObject.name=run-1-abcde",
                      " ".join(fake.find("get events")))
        # The live pod would hold its TPU node forever: it is deleted.
        self.assertIsNotNone(fake.find("delete job run-1"))

    def test_one_failed_pull_is_not_final_however_long_ago(self):
        """The waiting reason outlasts the failure while a retry pulls; a
        multi-gigabyte retry after one blip must be left to finish."""
        for reason in ("ErrImagePull", "ImagePullBackOff"):
            fake = self._pulling(reason, span=0)
            result = self.status(fake)
            self.assertEqual(result.status, "PENDING", reason)
            self.assertIn("manifest for img:1 not found", result.error)
            self.assertIsNone(fake.find("delete job"))

    def test_a_pull_failure_without_readable_events_is_never_final(self):
        # Not allowed to read events, or they aged out: wait, as before.
        for events in ((1, "", "Forbidden"), (0, '{"items": []}', "")):
            fake = self._pulling(**{"get events": events})
            self.assertEqual(self.status(fake).status, "PENDING")

    def test_an_image_reference_that_can_never_pull_fails_at_once(self):
        fake = self._pulling("InvalidImageName")
        self.assertEqual(self.status(fake).status, "FAILED")
        self.assertIsNone(fake.find("get events"))

    GONE = (1, "", 'Error from server (NotFound): jobs.batch "run-1" not '
            "found")

    def test_a_pull_failure_on_a_submitted_job_is_deleted_and_recorded(self):
        """`job status` settles the record, and the reaper skips a settled
        record: without the delete, the Job would pull forever on a TPU
        node with nothing left to clean it up.

        The status read is the one recorded: reading the Job again to
        delete it could find it vanishing and record a bare CANCELLED.
        Later reads answer from that record, not with a kubectl NotFound,
        and logs say there are none.
        """
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._pulling()
        self.assertEqual(self.status(fake).status, "FAILED")
        self.assertIsNotNone(fake.find("delete job run-1"))
        self.assertIsNotNone(fake.find("delete configmap run-1"))
        self.assertEqual(
            sum("get job run-1" in " ".join(cmd) for cmd in fake.calls), 1)
        record = jobs.JobRecord.load("run-1")
        self.assertEqual(record.status, "FAILED")
        self.assertTrue(record.extra["never_started"])
        gone = FakeKubectl({"get job": self.GONE})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", gone):
            result = driver.get_status("run-1")
            self.assertEqual(result.status, "FAILED")
            self.assertEqual(
                result.error, "The image could not be pulled: "
                "ImagePullBackOff: manifest for img:1 not found")
            self.assertTrue(result.details["never_started"])
            self.assertTrue(result.details["from_record"])
            with self.assertRaises(ComputeConfigError) as ctx:
                driver.fetch_logs("run-1")
            self.assertEqual(ctx.exception.error_code, "JOB_NEVER_STARTED")
            self.assertIn("manifest for img:1 not found",
                          ctx.exception.message)
        # Nothing is asked of a container that never ran.
        self.assertIsNone(gone.find("logs"))
        self.assertIsNone(gone.find("delete"))

    def test_a_pull_failure_whose_delete_fails_stays_unsettled(self):
        """Left unsettled, the run-end reaper still tries to delete it."""
        record = MagicMock(status="RUNNING")
        fake = self._pulling(**{"delete job": (1, "", "Forbidden")})
        with patch.object(gke_job.jobs.JobRecord, "load",
                          return_value=record), \
                patch.object(gke_job.GkeJobDriver, "_save_logs"), \
                contextlib.redirect_stderr(io.StringIO()):
            result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        record.mark.assert_not_called()

    def test_a_missing_job_is_only_answered_by_a_settled_record(self):
        driver = make_driver()
        # No record, or one still live: the Job's absence is unexplained.
        for create in (False, True):
            with self.subTest(record=create), _job_store(), \
                    patch.object(gke_job.time, "sleep") as sleep:
                if create:
                    record = jobs.JobRecord.create(job_id="run-1",
                                                   backend="gke_job",
                                                   status="RUNNING")
                    record.start_time = "2026-01-01T00:00:00+00:00"
                    record.save()
                with patch.object(gke_job.subprocess, "run",
                                  FakeKubectl({"get job": self.GONE})):
                    with self.assertRaises(ComputeConfigError) as ctx:
                        driver.get_status("run-1")
                sleep.assert_not_called()  # Not just submitted: no retry.
                if create:
                    # Removed outside compute: said so, and the record
                    # is kept, since the Job may run on another cluster.
                    self.assertEqual(ctx.exception.error_code, "JOB_GONE")
                    self.assertEqual(
                        jobs.JobRecord.load("run-1").status, "RUNNING")
        # A settled record does not hide a cluster that cannot be asked.
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job").mark("COMPLETED", exit_code=0)
        unauthorized = FakeKubectl(
            {"get job": (1, "", "error: You must be logged in to the server")})
        with patch.object(gke_job.subprocess, "run", unauthorized):
            with self.assertRaises(ComputeConfigError):
                driver.get_status("run-1")

    def test_a_just_submitted_job_is_read_again_on_not_found(self):
        """The read right after submit can race the create."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        found = FakeKubectl({"get job": (0, job_json(), "")})
        answers = iter([FakeKubectl({"get job": self.GONE})] * 2)

        def kubectl(cmd: list[str], **kwargs: Any):
            if "get job run-1" in " ".join(cmd):
                return next(answers, found)(cmd, **kwargs)
            return found(cmd, **kwargs)

        with patch.object(gke_job.time, "sleep") as sleep:
            self.assertEqual(self.status(kubectl).status, "PENDING")
        self.assertEqual(sleep.call_count, 2)

    def test_cancel_keeps_a_running_jobs_logs(self):
        """`job logs` after `job cancel` used to fail on NotFound."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        alive = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, _RUNNING_POD, ""),
            "logs": (0, "loading weights\n", ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", alive):
            driver.cancel("run-1")
        self.assertEqual(jobs.JobRecord.load("run-1").status, "CANCELLED")
        gone = FakeKubectl({"get job": self.GONE, "logs": self.GONE})
        with patch.object(gke_job.subprocess, "run", gone):
            self.assertEqual(driver.fetch_logs("run-1"), "loading weights\n")

    def test_run_reports_a_pull_failure_not_missing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "hello_tpu.py"
            script.write_text("print('hi')\n", encoding="utf-8")
            fake = self._pulling()
            with patch.object(gke_job.subprocess, "run", fake), \
                    patch.object(gke_job.time, "sleep"):
                result = make_driver().run(str(script))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("manifest for img:1 not found", result.stderr)
        self.assertTrue(result.details["never_started"])
        self.assertIsNone(fake.find("logs"))
        self.assertIsNotNone(fake.find("delete job"))

    def test_active_job_with_unreadable_pods_stays_running(self):
        # No phase to read is not evidence of a stuck pod; the Job is active.
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (1, "", "Forbidden"),
        })
        self.assertEqual(self.status(fake).status, "RUNNING")

    def test_unstarted_job_is_pending(self):
        fake = FakeKubectl({"get job": (0, job_json(), "")})
        result = self.status(fake)
        self.assertEqual(result.status, "PENDING")
        self.assertNotIn("suspended", result.details)

    def test_suspended_job_is_distinguishable_from_plain_pending(self):
        # A suspended Job exists, reports no error and starts no pod, so a
        # caller that cannot tell it from pending reads a full queue as
        # unreachable hardware.
        fake = FakeKubectl({"get job": (0, job_json(suspend=True), "")})
        result = self.status(fake)
        self.assertEqual(result.status, "PENDING")
        self.assertTrue(result.details["suspended"])
        self.assertEqual(result.details["pending_reason"],
                         gke_job.base.PENDING_QUEUED)
        self.assertIn("queue", result.error)

    # -- Kueue refusing a suspended Job ------------------------------------

    CREATED = "2026-01-01T00:00:00Z"
    WORKLOAD = "job-run-1-3f2a1"
    LABELLED = f"-l {gke_job.JOB_UID_LABEL}=uid-1"
    UNLABELLED = f"{gke_job.WORKLOAD_RESOURCE} -n default -o json"
    NO_LQ = "LocalQueue lq doesn't exist"
    QUOTA_WAIT = ("couldn't assign flavors to pod set main: insufficient "
                  "unused quota for google.com/tpu in flavor tpu7x, 4 more "
                  "needed")
    PREEMPTED = ("Preempted to accommodate a workload (UID: w-2, JobUID: "
                 "uid-2) due to prioritization in the ClusterQueue")

    @classmethod
    def _suspended_job(cls, **metadata: Any) -> str:
        return json.dumps({
            "metadata": {
                "name": "run-1",
                "uid": "uid-1",
                "creationTimestamp": cls.CREATED,
                "labels": {
                    gke_job.QUEUE_LABEL: "lq",
                    gke_job.PRIORITY_LABEL: "medium"
                },
                **metadata,
            },
            "spec": {
                "suspend": True
            },
        })

    @staticmethod
    def _items(*items: dict[str, Any]) -> tuple[int, str, str]:
        return 0, json.dumps({"items": list(items)}), ""

    @staticmethod
    def _cond(kind: str,
              reason: str,
              message: str,
              status: str = "True") -> dict[str, Any]:
        return {
            "type": kind,
            "status": status,
            "reason": reason,
            "message": message
        }

    @classmethod
    def _workload(cls,
                  *conditions: dict[str, Any],
                  owner: str = "uid-1",
                  **spec: Any) -> dict[str, Any]:
        return {
            "metadata": {
                "name": cls.WORKLOAD,
                "ownerReferences": [{
                    "kind": "Job",
                    "uid": owner
                }],
            },
            "spec": spec,
            "status": {
                "conditions": list(conditions)
            },
        }

    @classmethod
    def _unreserved(cls, reason: str, message: str) -> dict[str, Any]:
        """A Workload Kueue has not reserved quota for, and why."""
        return cls._workload(
            cls._cond("QuotaReserved", reason, message, status="False"))

    @staticmethod
    def _queue(reason: str) -> tuple[int, str, str]:
        """A LocalQueue or ClusterQueue Kueue has made inactive for `reason`."""
        active = {"type": "Active", "status": "False", "reason": reason}
        return 0, json.dumps({"status": {"conditions": [active]}}), ""

    @staticmethod
    def _warnings(*lines: tuple[str, str]) -> tuple[int, str, str]:
        return 0, json.dumps({
            "items": [{
                "type": "Normal",
                "reason": "CreatedWorkload",
                "message": "Created Workload: default/job-run-1-3f2a1",
                "lastTimestamp": "2026-01-01T00:00:01Z",
            }] + [{
                "type": "Warning",
                "reason": reason,
                "message": message,
                "lastTimestamp": f"2026-01-01T00:00:0{i + 2}Z",
            } for i, (reason, message) in enumerate(lines)]
        }), ""

    def _kueue(self,
               *workloads: dict[str, Any],
               job: str | None = None,
               **responses: Any) -> FakeKubectl:
        """A suspended Job in a Kueue queue, its Workloads, and the rest."""
        if workloads:
            responses = {self.LABELLED: self._items(*workloads), **responses}
        return FakeKubectl({
            "get job": (0, job or self._suspended_job(), ""),
            **responses,
        })

    def kueue_status(self, fake: Any, age: float = 3600):
        created = gke_job._parse_time(self.CREATED)
        with patch.object(gke_job.time, "time", return_value=created + age):
            return self.status(fake)

    def test_a_workload_kueue_refuses_fails_the_job_with_kueues_reason(self):
        """An inadmissible Workload is never admitted, however long the wait.

        Polling it burned the whole fifteen minutes to report a "full queue"
        whose real cause was only in the Workload's conditions. It fails at
        once, with Kueue's own words, and is deleted: left queued, it would
        run with nobody watching the day the queue is fixed.
        """
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._kueue(
            self._unreserved("Inadmissible", self.NO_LQ), **{
                "involvedObject.kind=Workload":
                self._warnings(("Inadmissible", self.NO_LQ))
            })
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(
            result.error, f"Kueue will not admit the Job (Inadmissible): "
            f"{self.NO_LQ}")
        self.assertEqual(result.details["kueue_reason"], "Inadmissible")
        self.assertEqual(result.details["kueue_workload"], self.WORKLOAD)
        self.assertEqual(result.details["kueue_events"],
                         [f"Inadmissible: {self.NO_LQ}"])
        self.assertNotIn("suspended", result.details)
        self.assertIn(f"involvedObject.name={self.WORKLOAD}",
                      " ".join(fake.find("involvedObject.kind=Workload")))
        self.assertIsNotNone(fake.find("delete job run-1"))
        record = jobs.JobRecord.load("run-1")
        self.assertEqual(record.status, "FAILED")
        self.assertTrue(record.extra["never_started"])
        # Nothing ran, so there are no logs to save or look for.
        self.assertIsNone(fake.find("logs"))

    def test_every_kind_of_kueue_refusal_fails_the_job(self):
        refusals = {
            "granular reason":
            self._unreserved("Misconfigured", "ClusterQueue cq doesn't exist"),
            # Inactive for want of its flavor, not held: nobody will lift it.
            "inactive queue":
            self._unreserved("Inadmissible", "ClusterQueue cq is inactive"),
            "request no quota can hold":
            self._unreserved("ExceedsMaxQuota", "request > maximum capacity"),
            # The scheduler can report an inactive queue as plain Pending.
            "queue problem under Pending":
            self._unreserved("Pending", "LocalQueue lq is inactive"),
            "deactivated":
            self._workload(self._cond("Evicted", "Deactivated",
                                      "deactivated by an AdmissionCheck"),
                           active=False),
            "finished unadmitted":
            self._workload(
                self._cond("Finished", "AdmissionChecksRejected",
                           "Admission checks rejected")),
        }
        for case, workload in refusals.items():
            with self.subTest(case), _job_store():
                result = self.status(
                    self._kueue(
                        workload,
                        **{"queues.kueue": self._queue("FlavorNotFound")}))
                self.assertEqual(result.status, "FAILED")
                self.assertTrue(result.details["never_started"])
                self.assertIn(result.details["kueue_message"], result.error)

    def test_a_job_the_clusters_time_limit_stopped_fails_and_is_kept(self):

        def stopped(reason: str, message: str) -> dict[str, Any]:
            w = self._workload(self._cond("Evicted", "Deactivated",
                                          "Deactivated"),
                               active=False)
            w["metadata"]["annotations"] = {
                gke_job.EVICTION_REASON_ANNOTATION: reason,
                gke_job.EVICTION_MESSAGE_ANNOTATION: message,
                gke_job.EVICTED_AT_ANNOTATION: "2026-01-01T02:00:24Z",
            }
            return w

        msg = "Suspended by time-limit-controller: elapsed 120.4m"
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._kueue(stopped("timeout", msg))
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertIn(f"stopped the Job after it ran: {msg}", result.error)
        self.assertEqual(result.details["eviction_reason"], "timeout")
        self.assertNotIn("never_started", result.details)
        self.assertIsNone(fake.find("delete job run-1"))
        self.assertIsNone(
            gke_job._time_limit_stop(stopped("finished-gc", "GC")))

    def test_kueue_waits_that_end_on_their_own_stay_pending(self):
        """No free quota, a queue someone has put on hold, and a preemption
        Kueue requeued are all waits, not refusals. Kueue's message says
        which, and is passed on."""
        held = self._unreserved("Inadmissible", "ClusterQueue cq is inactive")
        stopped = self._queue("Stopped")
        # (case, the Workload, its queue's answer, kueue_reason, in the error)
        waits = [
            ("no quota", self._unreserved("Pending", self.QUOTA_WAIT), None,
             "Pending", self.QUOTA_WAIT),
            ("no quota, newer Kueue",
             self._unreserved("WaitingForQuota", self.QUOTA_WAIT), None,
             "WaitingForQuota", self.QUOTA_WAIT),
            ("held ClusterQueue", held, stopped, "Stopped", "cq is inactive"),
            ("held LocalQueue",
             self._unreserved("Inadmissible", "LocalQueue lq is inactive"),
             stopped, "Stopped", "lq is inactive"),
            # Unreadable, the queue may be held: only an answer fails a Job.
            ("unreadable queue", held, (1, "", "Forbidden"), "Stopped",
             "cq is inactive"),
            ("preempted",
             self._workload(
                 self._cond("Preempted", "InClusterQueue", self.PREEMPTED),
                 self._cond("QuotaReserved", "Pending", self.QUOTA_WAIT,
                            "False")), None, "Preempted",
             f"Kueue preempted the Job to make room for another workload and "
             f"requeued it; it runs again when quota frees up. Kueue says: "
             f"{self.PREEMPTED}"),
            ("preempted, older Kueue",
             self._workload(self._cond("Evicted", "Preempted",
                                       self.PREEMPTED)), None, "Preempted",
             self.PREEMPTED),
            ("not looked at yet", self._workload(), None, None, "Free chips"),
        ]
        for case, workload, queue, reason, said in waits:
            with self.subTest(case):
                fake = self._kueue(
                    workload, **({
                        "queues.kueue": queue
                    } if queue else {}))
                result = self.kueue_status(fake)
                self.assertEqual(result.status, "PENDING")
                self.assertTrue(result.details["suspended"])
                self.assertEqual(result.details.get("kueue_reason"), reason)
                self.assertIn(said, result.error)
                self.assertIsNone(fake.find("delete job"))

    def test_no_workload_after_the_grace_period_fails_the_job(self):
        """Kueue makes no Workload for a Job naming a priority class the
        cluster lacks; that Job stays suspended forever."""
        late = gke_job.KUEUE_WORKLOAD_GRACE_SEC + 1
        none = {self.LABELLED: self._items(), self.UNLABELLED: self._items()}
        fake = self._kueue(
            **{
                **none,
                # A Workload another Job owns is not this Job's.
                self.UNLABELLED:
                self._items(self._workload(owner="uid-2")),
                "workloadpriorityclasses":
                (1, "", "Error from server (NotFound): "
                 'workloadpriorityclasses.kueue.x-k8s.io "medium" not found'),
            })
        result = self.kueue_status(fake, age=late)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.details["kueue_reason"], "NoWorkload")
        self.assertIn("no Kueue Workload was created", result.error)
        self.assertIn("WorkloadPriorityClass 'medium'", result.error)
        # The label is checked by owner too: an older Kueue does not set it.
        self.assertIsNotNone(fake.find(self.UNLABELLED))
        self.assertIsNotNone(fake.find("delete job run-1"))

        # A class the cluster does define is not blamed.
        result = self.kueue_status(self._kueue(**none), age=late)
        self.assertEqual(result.status, "FAILED")
        self.assertNotIn("'medium'", result.error)
        self.assertIn("kueue-controller-manager", result.error)

        # Kueue's own event on the Job, when it records one, says it better.
        found = 'WorkloadPriorityClass "medium" not found'
        fake = self._kueue(
            **{
                **none, "involvedObject.kind=Job":
                self._warnings(("WorkloadPriorityClassNotFound", found))
            })
        result = self.kueue_status(fake, age=late)
        self.assertEqual(result.details["kueue_message"],
                         f"WorkloadPriorityClassNotFound: {found}")

    def test_no_workload_is_pending_within_the_grace_period_or_for_a_child(
            self):
        # Kueue has not reconciled the Job yet: that is not a refusal.
        fake = self._kueue(**{self.LABELLED: self._items()})
        result = self.kueue_status(fake,
                                   age=gke_job.KUEUE_WORKLOAD_GRACE_SEC - 1)
        self.assertEqual(result.status, "PENDING")
        self.assertIsNone(fake.find(self.UNLABELLED))
        # A JobSet's Job never has a Workload: Kueue admits the JobSet.
        child = self._suspended_job(ownerReferences=[{
            "kind": "JobSet",
            "uid": "js-1",
            "controller": True
        }])
        fake = self._kueue(job=child, **{self.LABELLED: self._items()})
        self.assertEqual(self.kueue_status(fake).status, "PENDING")
        self.assertIsNone(fake.find(gke_job.WORKLOAD_RESOURCE))

    def test_an_unreadable_workload_list_stays_pending(self):
        """Only an answer the cluster gave may fail a Job: no Kueue CRD, no
        right to list Workloads, or no answer at all says nothing."""
        unreadable = {
            "no CRD": (1, "", 'error: the server doesn\'t have a resource '
                       'type "workloads"'),
            "forbidden": (1, "", "Error from server (Forbidden)"),
            "garbage": (0, "not json", ""),
        }
        for case, answer in unreadable.items():
            with self.subTest(case):
                fake = self._kueue(**{self.LABELLED: answer})
                result = self.kueue_status(fake)
                self.assertEqual(result.status, "PENDING")
                self.assertTrue(result.details["suspended"])
        # Found by neither label nor owner, but the owner scan failed.
        fake = self._kueue(**{
            self.LABELLED: self._items(),
            self.UNLABELLED: (1, "", "Forbidden")
        })
        self.assertEqual(self.kueue_status(fake).status, "PENDING")
        # A read that times out raises in `_kubectl`; it must not escape.
        answering = self._kueue()

        def hang_on_workloads(cmd: list[str], **kwargs: Any):
            if gke_job.WORKLOAD_RESOURCE in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return answering(cmd, **kwargs)

        self.assertEqual(
            self.kueue_status(hang_on_workloads).status, "PENDING")

    def test_run_reports_a_kueue_refusal_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "hello_tpu.py"
            script.write_text("print('hi')\n", encoding="utf-8")
            fake = self._kueue(self._unreserved("Inadmissible", self.NO_LQ))
            with patch.object(gke_job.subprocess, "run", fake), \
                    patch.object(gke_job.time, "sleep") as sleep:
                result = make_driver(queue="lq").run(str(script))
        sleep.assert_not_called()
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(self.NO_LQ, result.stderr)
        self.assertEqual(result.details["job_status"], "FAILED")
        self.assertIsNone(fake.find("logs"))

    def test_failed_job_reports_the_pod_exit_code(self):
        pods = json.dumps({
            "items": [{
                "status": {
                    "containerStatuses": [{
                        "state": {
                            "terminated": {
                                "exitCode": 1,
                                "reason": "Error"
                            }
                        }
                    }]
                }
            }]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(failed=1), ""),
            "get pods": (0, pods, ""),
        })
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.exit_code, 1)
        # Not Kubernetes' bare "Error", which reads as the whole diagnosis.
        self.assertEqual(result.error, "container exited with code 1")
        self.assertEqual(gke_job._termination_error(137, "OOMKilled"),
                         "container exited with code 137 (OOMKilled)")

    def test_failed_job_survives_an_unreadable_pod(self):
        # The pod is garbage-collected out from under a late status read; the
        # Job still failed and must still say so.
        fake = FakeKubectl({
            "get job": (0, job_json(failed=1), ""),
            "get pods": (1, "", "not found"),
        })
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.exit_code)

    LOW_DISK = ("The node was low on resource: ephemeral-storage. Threshold "
                "quantity: 10Gi, available: 1Gi.")
    SHUTDOWN = "Pod was terminated in response to imminent node shutdown."

    def test_failed_job_says_why_the_pod_ended(self):
        """Exit 137 and no traceback reads as a crash; the pod's own reason
        (an eviction for a full disk, a node shutdown) is the diagnosis."""
        killed = {"exit_code": 137, "container_reason": "Error"}
        evicted = {"reason": "Evicted", "message": self.LOW_DISK}
        cases = [
            # (failed_pod arguments, error)
            ({
                **killed,
                **evicted
            }, (f"pod was evicted: {self.LOW_DISK} "
                "(container exited with code 137)")),
            # Evicted before the kubelet recorded the container's end.
            (evicted, f"pod was evicted: {self.LOW_DISK}"),
            # Only the DisruptionTarget condition kept the kubelet's words.
            ({
                **killed, "conditions": [{
                    "type": "DisruptionTarget",
                    "status": "True",
                    "reason": "TerminationByKubelet",
                    "message": self.LOW_DISK,
                }]
            }, (f"pod was evicted: {self.LOW_DISK} "
                "(container exited with code 137)")),
            ({
                **killed, "reason": "Terminated",
                "message": self.SHUTDOWN
            }, ("container exited with code 137; pod Terminated: "
                f"{self.SHUTDOWN}")),
            # No reason anywhere: say what sends a SIGKILL.
            (killed, ("container exited with code 137 (SIGKILL: out of "
                      "memory, an eviction, or the node going away)")),
            # A specific reason needs no hint.
            ({
                "exit_code": 137,
                "container_reason": "OOMKilled"
            }, "container exited with code 137 (OOMKilled)"),
        ]
        for pod, error in cases:
            with self.subTest(error=error), _job_store():
                jobs.JobRecord.create(job_id="run-1", backend="gke_job")
                pods = json.dumps({"items": [failed_pod(**pod)]})
                fake = FakeKubectl({
                    "get job": (0, job_json(failed=1), ""),
                    "get pods": (0, pods, ""),
                    "logs": (0, "", ""),
                })
                result = self.status(fake)
                self.assertEqual(result.status, "FAILED")
                self.assertEqual(result.error, error)
                # Kept on the record, for once the pod is gone.
                self.assertEqual(
                    jobs.JobRecord.load("run-1").extra["error"], error)

    def test_a_vanished_job_says_where_it_was_last_seen(self):
        """A `--serve` Job deleted out from under the run used to give only
        kubectl's NotFound."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job", serve=True)
        record = jobs.JobRecord.load("run-1")
        record.start_time = "2026-01-01T00:00:00+00:00"
        record.save()
        pods = json.dumps({"items": [failed_pod(phase="Running")]})
        self.status(
            FakeKubectl({
                "get job": (0, job_json(active=1), ""),
                "get pods": (0, pods, ""),
            }))
        with self.assertRaises(ComputeConfigError) as ctx:
            self.status(FakeKubectl({"get job": self.GONE}))
        self.assertEqual(ctx.exception.error_code, "JOB_GONE")
        self.assertRegex(
            ctx.exception.message,
            r"^Job run-1 not found in namespace 'default'; last seen RUNNING "
            r"on node gke-tpu-node-1 at \S+ \(pod run-1-abcde\)\. ")

    @staticmethod
    def _entry(text: str, **payload: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "resource": {
                "labels": {
                    "pod_name": "run-1-abcde"
                }
            }
        }
        entry.update({"jsonPayload": payload}
                     if payload else {"textPayload": text + "\n"})
        return entry

    def _fetch_gone_pod_logs(self,
                             answer: tuple[int, str, str],
                             on_path: bool = True) -> tuple[Any, Any]:
        """fetch_logs for a Job whose pod and copies are gone, with gcloud
        answering `answer`. Returns (logs or the raised error, the fake)."""
        record = jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        record.start_time = "2026-01-01T00:20:00+00:00"
        record.extra["uid"] = "uid-1"
        record.save()
        fake = FakeKubectl({
            "logging read": answer,
            "get job": self.GONE,
            "logs": (1, "", 'pods "run-1-abcde" not found'),
        })
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gke_job.shutil, "which",
                             return_value="/bin/gcloud" if on_path else None):
            try:
                return make_driver().fetch_logs("run-1"), fake
            except ComputeConfigError as exc:
                return exc, fake

    def test_logs_of_a_gone_pod_come_from_cloud_logging(self):
        """No pod, no saved copy, no GCS copy: Cloud Logging's lines, oldest
        first, under a line saying where they came from."""
        newest_first = json.dumps([
            self._entry("", message="Killed"),
            self._entry("Downloading shards: 40%"),
        ])
        logs, fake = self._fetch_gone_pod_logs((0, newest_first, ""))
        first, *rest = logs.splitlines()
        self.assertTrue(first.startswith("[compute] Job run-1's pod is gone"))
        self.assertIn("Cloud Logging kept", first)
        self.assertEqual(rest, ["Downloading shards: 40%", "Killed"])
        query = fake.find("logging read")
        self.assertIn("--order=desc", query)
        self.assertIn(f"--limit={gke_job.CLOUD_LOGGING_MAX_LINES}", query)
        # The record's UID and start reach the query (the window itself is
        # covered by test_cloud_logging_query_keeps_to_one_job).
        self.assertIn('labels."k8s-pod/controller-uid"="uid-1"', query[3])
        self.assertIn('timestamp>="2026-01-01T00:10:00Z"', query[3])

    def test_no_cloud_logging_lines_says_why(self):
        cases = [
            # (gcloud on PATH, its answer, words in the error)
            (False, (0, "[]", ""), "gcloud is not on PATH"),
            (True, (1, "", "ERROR: (gcloud.logging.read) PERMISSION_DENIED"),
             "PERMISSION_DENIED"),
            (True, (0, "[]", ""), "holds no output for the Job"),
        ]
        for on_path, answer, expected in cases:
            with self.subTest(expected=expected), _job_store():
                exc, _ = self._fetch_gone_pod_logs(answer, on_path)
                self.assertIsInstance(exc, ComputeConfigError)
                self.assertEqual(exc.error_code, "LOGS_UNAVAILABLE")
                self.assertIn(expected, exc.message)
                # The query to run by hand is always given.
                self.assertIn('labels."k8s-pod/job-name"="run-1"', exc.message)

    def test_cloud_logging_query_keeps_to_one_job(self):
        """Review on #680: cancel and resubmit with the same --name, and a
        window widened around one Job's times takes in the other's output.

        With the UID, only this Job's pods match, so the window can be wide.
        Without it (an older record), the window is exactly the record's.
        """
        start = gke_job._parse_time("2026-01-01T00:20:00Z")
        end = gke_job._parse_time("2026-01-01T01:00:00Z")
        base = ('resource.type="k8s_container" AND '
                'resource.labels.namespace_name="ns" AND '
                'labels."k8s-pod/job-name"="run-1"')
        self.assertEqual(
            gke_job._cloud_logging_query("ns", "run-1", "c1", start, end,
                                         "uid-2"),
            base + ' AND (labels."k8s-pod/controller-uid"="uid-2" OR '
            'labels."k8s-pod/batch_kubernetes_io/controller-uid"="uid-2") '
            'AND resource.labels.cluster_name="c1" AND '
            'timestamp>="2026-01-01T00:10:00Z" AND '
            'timestamp<="2026-01-01T01:10:00Z"')
        self.assertEqual(
            gke_job._cloud_logging_query("ns", "run-1", None, start, end),
            base + ' AND timestamp>="2026-01-01T00:20:00Z" AND '
            'timestamp<="2026-01-01T01:00:00Z"')

    def test_unreadable_logs_raise_rather_than_becoming_the_log(self):
        reaped = "error: timed out waiting for the condition\n"
        fake = FakeKubectl({"logs": (1, "", reaped)})
        with tempfile.TemporaryDirectory() as out, \
                patch.object(gke_job.subprocess, "run", fake):
            with self.assertRaises(ComputeConfigError) as ctx:
                make_driver().fetch_artifacts("run-1", Path(out))
            self.assertEqual(ctx.exception.error_code, "LOGS_UNAVAILABLE")
            self.assertIn("timed out", ctx.exception.message)
            self.assertEqual(list(Path(out).iterdir()), [])

    def test_finished_job_logs_outlive_its_pod(self):
        """Seeing the Job finished, by status read or cancel, keeps its logs."""
        alive = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "step_time_ms: 41.7\n", ""),
        })
        reaped = FakeKubectl({"logs": (1, "", "timed out")})
        for observe in ("get_status", "cancel"):
            with self.subTest(observe), _job_store():
                jobs.JobRecord.create(job_id="run-1", backend="gke_job")
                driver = make_driver()
                with patch.object(gke_job.subprocess, "run", alive):
                    getattr(driver, observe)("run-1")
                with patch.object(gke_job.subprocess, "run", reaped):
                    self.assertEqual(driver.fetch_logs("run-1"),
                                     "step_time_ms: 41.7\n")

    def _lost_node_fake(self,
                        cloud: list[str],
                        node_log: str = "c1 385\nc2 729\n") -> FakeKubectl:
        """A Job whose node was reset: the pod is left `Unknown`/255, its log
        on the node is `node_log`, and Cloud Logging has `cloud`."""
        pods = json.dumps(
            {"items": [failed_pod(exit_code=255, container_reason="Unknown")]})
        return FakeKubectl({
            "logging read":
            (0, json.dumps([self._entry(line)
                            for line in reversed(cloud)]), ""),
            "get job": (0, job_json(failed=1), ""),
            "get pods": (0, pods, ""),
            "logs": (0, node_log, ""),
        })

    def test_a_lost_nodes_log_gets_cloud_loggings_tail(self):
        """Hard-resetting the node mid-benchmark left `job logs` with c1 and
        c2, while Cloud Logging also had c4 and c8. The node's copy can end
        mid-line; that line is read from Cloud Logging whole. Status reads
        never ask Cloud Logging; the first log read waits until it has had
        time to ingest the tail, asks once, and keeps the answer."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        driver = make_driver()
        cloud = ["c1 385", "c2 729", "c4 1393", "c8 2613"]
        status = self._lost_node_fake(cloud, "c1 385\nc2 72")
        with patch.object(gke_job.subprocess, "run", status), \
                patch.object(gke_job.shutil, "which",
                             return_value="/bin/gcloud"):
            driver.get_status("run-1")
        self.assertIsNone(status.find("logging read"))
        read = self._lost_node_fake(cloud)
        with patch.object(gke_job.subprocess, "run", read), \
                patch.object(gke_job.shutil, "which",
                             return_value="/bin/gcloud"), \
                patch.object(gke_job.time, "sleep") as sleep:
            logs = driver.fetch_logs("run-1").splitlines()
        self.assertAlmostEqual(sleep.call_args[0][0],
                               gke_job.CLOUD_LOGGING_SETTLE_SEC,
                               delta=5)
        self.assertEqual(logs[0], "c1 385")
        self.assertTrue(logs[1].startswith("[compute] The node was lost"))
        self.assertEqual(logs[2:], ["c2 729", "c4 1393", "c8 2613"])
        settled = FakeKubectl({"logs": (1, "", "timed out")})
        with patch.object(gke_job.subprocess, "run", settled):
            self.assertEqual(driver.fetch_logs("run-1").splitlines(), logs)
        self.assertIsNone(settled.find("logging read"))

    def test_a_lost_nodes_log_read_before_the_job_settles(self):
        """`job logs` on a pod left Unknown, before any status read, keeps
        and settles the copy the same way; a copy with no complete line
        takes all of Cloud Logging's."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run",
                          self._lost_node_fake(["c1 385", "c2 729"], "c1 3")), \
                patch.object(gke_job.shutil, "which",
                             return_value="/bin/gcloud"), \
                patch.object(gke_job.time, "sleep"):
            logs = driver.fetch_logs("run-1").splitlines()
        self.assertTrue(logs[0].startswith("[compute] The node was lost"))
        self.assertEqual(logs[1:], ["c1 385", "c2 729"])
        self.assertFalse(driver._lost_node_marker("run-1").exists())

    def test_logs_of_a_job_yet_to_start_are_empty(self):
        fake = FakeKubectl({
            "logs": (1, "", "container is waiting to start"),
            "get job": (0, job_json(), ""),
        })
        with patch.object(gke_job.subprocess, "run", fake):
            self.assertEqual(make_driver().fetch_logs("run-1"), "")

    def test_logs_fall_back_to_the_copy_the_pod_uploaded(self):
        fake = FakeKubectl({"logs": (1, "", "timed out")})

        def download(bucket, blob, dest):
            self.assertEqual((bucket, blob),
                             ("b", "run-1/out/.compute-output.log"))
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("step_time_ms: 41.7\n", encoding="utf-8")
            return True

        with \
                patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "download_file_from_gcs", download):
            self.assertEqual(
                make_driver(staging_bucket="gs://b").fetch_logs("run-1"),
                "step_time_ms: 41.7\n")

    def test_status_without_a_pod_skips_kubectl_logs(self):
        """`kubectl logs` on a podless Job blocks for 20s on every read."""
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        with patch.object(gke_job.subprocess, "run", fake):
            make_driver().get_status("run-1")
        self.assertIsNone(fake.find("logs"))

    def test_a_gcs_failure_breaks_neither_status_nor_cancel(self):
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})

        def download(bucket, blob, dest):
            Path(dest).write_text("half a lo", encoding="utf-8")
            raise OSError("mTLS handshake failed")

        for observe in ("get_status", "cancel"):
            with self.subTest(observe), _job_store(), \
                    patch.object(gke_job.subprocess, "run", fake), \
                    patch.object(gcs_staging, "download_file_from_gcs",
                                 download), \
                    contextlib.redirect_stderr(io.StringIO()):
                jobs.JobRecord.create(job_id="run-1", backend="gke_job")
                getattr(make_driver(staging_bucket="gs://b"), observe)("run-1")
                self.assertEqual(
                    jobs.JobRecord.load("run-1").status, "COMPLETED")
                # A partial download must not pass for the saved log.
                self.assertEqual(
                    list((jobs.jobs_dir() / "run-1").glob("output.log*")), [])
        self.assertIsNotNone(fake.find("delete job run-1"))

    def test_fetch_artifacts_writes_the_logs(self):
        fake = FakeKubectl({"logs": (0, "SUCCESS: native-tpu\n", "")})
        driver = make_driver()
        with tempfile.TemporaryDirectory() as out:
            with patch.object(gke_job.subprocess, "run", fake):
                written = driver.fetch_artifacts("run-1", Path(out))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].name, "run-1.log")
            self.assertIn("native-tpu", written[0].read_text(encoding="utf-8"))

    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.download_file_from_gcs")
    def test_declared_outputs_keep_their_directories(self, mock_download,
                                                     mock_delete):
        """Two outputs that share a basename are two files, not one."""
        del mock_delete
        fake = FakeKubectl({"logs": (0, "done\n", "")})
        driver = make_driver(staging_bucket="gs://test-staging-bucket")

        def fake_download(bucket, blob, dest):
            del bucket
            Path(dest).write_text(blob)
            return True

        mock_download.side_effect = fake_download
        jobs.JobRecord.create(job_id="job-nested",
                              backend="gke_job",
                              extra={
                                  "outputs": [
                                      {
                                          "local": "eval/metrics.json",
                                          "remote": "eval/metrics.json",
                                      },
                                      {
                                          "local": "train/metrics.json",
                                          "remote": "train/metrics.json",
                                      },
                                  ]
                              })
        with tempfile.TemporaryDirectory() as out:
            with patch.object(gke_job.subprocess, "run", fake):
                written = driver.fetch_artifacts("job-nested", Path(out))
            landed = sorted(
                str(f.relative_to(out)) for f in written
                if f.suffix == ".json")
            self.assertEqual(landed,
                             ["eval/metrics.json", "train/metrics.json"])
            self.assertIn("job-nested/out/eval/metrics.json",
                          (Path(out) / "eval" / "metrics.json").read_text())

    def test_cancel_removes_job_and_configmap(self):
        # Both: a Job left holding a TPU node keeps the next run pending, and
        # a stranded ConfigMap blocks the name.
        fake = FakeKubectl()
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.cancel("run-1")
        self.assertIsNotNone(fake.find("delete job run-1"))
        self.assertIsNotNone(fake.find("delete configmap run-1"))

    def test_cancel_tolerates_an_absent_job(self):
        fake = FakeKubectl({"delete": (1, "", "NotFound")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.cancel("run-1")  # must not raise

    def test_cancel_that_cannot_delete_raises_and_keeps_the_record_live(self):
        """A CANCELLED record over a running Job is a leak; refuse to write it."""
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, _RUNNING_POD, ""),
            "delete": (1, "", "error: You must be logged in to the server "
                       "(Unauthorized)"),
        })
        driver = make_driver()
        jobs.JobRecord.create(job_id="run-1",
                              job_name="run-1",
                              backend="gke_job")
        with patch.object(gke_job.subprocess, "run", fake):
            with self.assertRaises(gke_job.KubectlError):
                driver.cancel("run-1")
        self.assertEqual(jobs.JobRecord.load("run-1").status, "PENDING")

    def test_cancel_of_a_finished_job_records_how_it_finished(self):
        """Deleting a Job that already ended reclaims its objects, no more.

        The record says COMPLETED or FAILED, as the cluster did; a Job that
        succeeded and was then tidied up must not read as cancelled.
        """
        for scripted, expected, exit_code in (
            (job_json(succeeded=1), "COMPLETED", 0),
            (job_json(failed=1), "FAILED", 1),
        ):
            with self.subTest(expected=expected), _job_store():
                fake = FakeKubectl({
                    "get job": (0, scripted, ""),
                    "get pods": (0, _FAILED_POD, ""),
                })
                driver = make_driver()
                jobs.JobRecord.create(job_id="run-1",
                                      job_name="run-1",
                                      backend="gke_job")
                with patch.object(gke_job.subprocess, "run", fake):
                    driver.cancel("run-1")
                record = jobs.JobRecord.load("run-1")
                self.assertEqual(record.status, expected)
                self.assertEqual(record.exit_code, exit_code)
                self.assertIsNotNone(fake.find("delete job run-1"))

    def test_cancel_of_a_running_job_records_cancelled(self):
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, _RUNNING_POD, ""),
        })
        driver = make_driver()
        jobs.JobRecord.create(job_id="run-1",
                              job_name="run-1",
                              backend="gke_job")
        with patch.object(gke_job.subprocess, "run", fake):
            driver.cancel("run-1")
        self.assertEqual(jobs.JobRecord.load("run-1").status, "CANCELLED")

    def test_status_read_settles_the_record(self):
        """`job list` reads the record; a finished Job must show up as such."""
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})
        driver = make_driver()
        jobs.JobRecord.create(job_id="run-1",
                              job_name="run-1",
                              backend="gke_job")
        with patch.object(gke_job.subprocess, "run", fake):
            self.assertEqual(driver.get_status("run-1").status, "COMPLETED")
        self.assertEqual(jobs.JobRecord.load("run-1").status, "COMPLETED")

    def test_status_reports_pod_name_when_pod_exists(self):
        pods = json.dumps({
            "items": [{
                "metadata": {
                    "name": "job-worker-xyz"
                },
                "status": {
                    "phase": "Running"
                }
            }]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            result = driver.get_status("run-1")
            self.assertEqual(result.status, "RUNNING")
            self.assertEqual(result.details.get("pod_name"), "job-worker-xyz")

    def test_port_forward_targets_pod_when_available(self):
        pods = json.dumps({
            "items": [{
                "metadata": {
                    "name": "job-pod-123"
                },
                "status": {
                    "phase": "Running"
                }
            }]
        })
        fake = FakeKubectl({
            "get pods": (0, pods, ""),
            "port-forward":
            (0, "Forwarding from 127.0.0.1:8080 -> 8000\n", ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            rc = driver.port_forward(local_port=8080,
                                     remote_port=8000,
                                     job_id="my-job")
            self.assertEqual(rc, 0)
            self.assertIsNotNone(
                fake.find("port-forward pod/job-pod-123 8080:8000"))

        # `_listening` stands in for the local socket: the Popen here is a
        # mock, so nothing binds 8080 and the readiness poll would otherwise
        # wait out its timeout before calling this forward dead.
        with patch.object(gke_job.subprocess, "run",
                          fake), patch.object(base,
                                              "_listening",
                                              return_value=True), patch.object(
                                                  gke_job.subprocess,
                                                  "Popen") as mock_popen:
            mock_popen.return_value.pid = 7777
            mock_popen.return_value.poll.return_value = None
            pid = driver.port_forward(local_port=8080,
                                      remote_port=8000,
                                      job_id="my-job",
                                      background=True)
            self.assertEqual(pid, 7777)

        def run_with_keyboard_interrupt(cmd, *args, **kwargs):
            if "port-forward" in cmd:
                raise KeyboardInterrupt()
            return fake(cmd, *args, **kwargs)

        with patch.object(gke_job.subprocess, "run",
                          run_with_keyboard_interrupt):
            rc = driver.port_forward(local_port=8080,
                                     remote_port=8000,
                                     job_id="my-job")
            self.assertEqual(rc, 0)

    def test_port_forward_skips_a_dead_pod(self):
        """`-l job-name=X` returns finished pods too, in no useful order."""
        pods = json.dumps({
            "items": [
                {
                    "metadata": {
                        "name": "job-pod-old"
                    },
                    "status": {
                        "phase": "Failed"
                    }
                },
                {
                    "metadata": {
                        "name": "job-pod-live"
                    },
                    "status": {
                        "phase": "Running"
                    }
                },
            ]
        })
        fake = FakeKubectl({
            "get pods": (0, pods, ""),
            "port-forward": (0, "Forwarding\n", ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.port_forward(local_port=8080,
                                remote_port=8000,
                                job_id="my-job")
        self.assertIsNotNone(fake.find("port-forward pod/job-pod-live"))
        self.assertIsNone(fake.find("job-pod-old"))

    def test_port_forward_falls_back_to_the_job_when_no_pod_runs(self):
        pods = json.dumps({
            "items": [{
                "metadata": {
                    "name": "job-pod-gone"
                },
                "status": {
                    "phase": "Succeeded"
                }
            }]
        })
        fake = FakeKubectl({
            "get pods": (0, pods, ""),
            "port-forward": (0, "Forwarding\n", ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.port_forward(local_port=8080,
                                remote_port=8000,
                                job_id="my-job")
        self.assertIsNotNone(fake.find("port-forward job/my-job"))

    def test_pending_diagnostic_survives_a_finished_pod(self):
        """A retried Job has several pods; only one of them is the news.

        The Pending pod is why the caller is still waiting, and a Succeeded
        or Failed attempt sorting ahead of it would report the Job as healthy
        progress until the timeout expired.
        """
        pods = json.dumps({
            "items": [
                {
                    "metadata": {
                        "name": "job-pod-attempt-1"
                    },
                    "status": {
                        "phase": "Failed"
                    }
                },
                {
                    "metadata": {
                        "name": "job-pod-attempt-2"
                    },
                    "status": {
                        "phase":
                        "Pending",
                        "conditions": [{
                            "type":
                            "PodScheduled",
                            "status":
                            "False",
                            "message":
                            "0/2 nodes are available: 2 "
                            "Insufficient google.com/tpu",
                        }],
                    },
                },
            ]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            status = driver.get_status("my-job")
        self.assertEqual(status.status, "PENDING")
        self.assertEqual(status.details["pod_phase"], "Pending")
        self.assertIn("Insufficient google.com/tpu",
                      status.details["pod_reason"])

    def test_running_pod_outranks_a_pending_one(self):
        """A Job with a live pod is progressing, whatever else is queued."""
        pods = json.dumps({
            "items": [
                {
                    "metadata": {
                        "name": "job-pod-queued"
                    },
                    "status": {
                        "phase": "Pending"
                    }
                },
                {
                    "metadata": {
                        "name": "job-pod-live"
                    },
                    "status": {
                        "phase": "Running"
                    }
                },
            ]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            status = driver.get_status("my-job")
        self.assertEqual(status.status, "RUNNING")
        self.assertEqual(status.details["pod_name"], "job-pod-live")

    def test_status_reads_the_pod_list_once(self):
        """Every poll of the synchronous wait goes through here."""
        pods = json.dumps({
            "items": [{
                "metadata": {
                    "name": "job-pod-1"
                },
                "status": {
                    "phase": "Pending"
                }
            }]
        })
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, pods, ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.get_status("my-job")
        pod_reads = [c for c in fake.calls if "pods" in " ".join(c)]
        self.assertEqual(len(pod_reads), 1)

    def test_port_forward_requires_job_id(self):
        driver = make_driver()
        with self.assertRaises(ComputeConfigError) as ctx:
            driver.port_forward(local_port=8000, remote_port=8000, job_id=None)
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")


class TestGkeJobRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "hello_tpu.py"
        self.script.write_text("print('hi')\n", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(_job_store())

    def test_dispatch_waits_reads_logs_then_cleans_up(self):
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "logs": (0, "device_type: native-tpu\n", ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            result = driver.run(str(self.script))

        self.assertEqual(result.exit_code, 0)
        self.assertIn("native-tpu", result.stdout)
        # Order matters: deleting the Job deletes its pod and the logs go with
        # it, so the read has to come first.
        verbs = [" ".join(c) for c in fake.calls]
        self.assertLess(
            next(i for i, v in enumerate(verbs) if "logs" in v),
            next(i for i, v in enumerate(verbs) if "delete job" in v))

    def test_run_tags_its_job_with_the_run_id(self):
        # Without the tag, a run killed mid-wait leaves a Job the run-end
        # reaper cannot find: it looks jobs up by run id.
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "logs": (0, "ok\n", ""),
        })
        driver = make_driver()
        driver.run_id = "20260101T000000Z-123456789abc"
        with patch.object(gke_job.subprocess, "run", fake):
            result = driver.run(str(self.script))
        record = jobs.JobRecord.load(result.details["job_id"])
        self.assertEqual(record.run_id, driver.run_id)

    def test_dispatch_cleans_up_after_a_failed_job(self):
        fake = FakeKubectl({
            "get job": (0, job_json(failed=1), ""),
            "get pods": (0, json.dumps({"items": []}), ""),
        })
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            result = driver.run(str(self.script))
        self.assertEqual(result.exit_code, 1)
        self.assertIsNotNone(fake.find("delete job"))

    def test_dispatch_gives_up_at_the_timeout_and_says_so(self):
        fake = FakeKubectl({"get job": (0, job_json(active=1), "")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake), \
             patch.object(gke_job.time, "sleep"):
            result = driver.run(str(self.script), timeout=0)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("RUNNING", result.stderr)
        self.assertIsNotNone(fake.find("delete job"))

    UNSCHEDULABLE = ("0/15 nodes are available: 8 Insufficient "
                     "google.com/tpu. No preemption victims found.")

    def _pending_fake(self) -> FakeKubectl:
        """A cluster with no free chips: the Job stays active, its pod
        Pending with the scheduler's reason, for as long as anyone asks."""
        pod = {
            "metadata": {
                "name": "run-1-abcde",
                "namespace": "default"
            },
            "spec": {},
            "status": {
                "phase":
                "Pending",
                "conditions": [{
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": self.UNSCHEDULABLE,
                }],
            },
        }
        return FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, json.dumps({"items": [pod]}), ""),
        })

    def test_a_run_that_times_out_in_the_queue_is_deleted_not_leaked(self):
        # Timed out before any node took it: no log read, Job deleted.
        fake = self._pending_fake()
        driver = make_driver()
        out = Path(self.tmp.name) / "probe.json"
        with patch.object(gke_job.subprocess, "run", fake), \
             patch.object(gke_job.time, "sleep"):
            result = driver.run(
                str(self.script),
                timeout=0,
                outputs=[base.StagedFile(local=str(out), remote="probe.json")])
        self.assertTrue(result.details["never_started"])
        self.assertIn(self.UNSCHEDULABLE, result.stderr)
        self.assertIn("The Job has been deleted.", result.stderr)
        self.assertIn("probe.json", result.stderr)
        self.assertIsNone(fake.find("logs"))
        self.assertIsNotNone(fake.find("delete job"))
        record = jobs.JobRecord.load(result.details["job_id"])
        self.assertEqual(record.status, "CANCELLED")
        self.assertTrue(record.extra["never_started"])
        self.assertTrue(record.extra["error"].startswith("Cancelled while: "))
        self.assertIn(self.UNSCHEDULABLE, record.extra["error"])

    def test_sigterm_during_the_wait_still_deletes_the_job(self):
        # The caller's shell timeout kills with SIGTERM.
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("the guard is main-thread only; the kill would "
                          "take the runner down")
        before = signal.getsignal(signal.SIGTERM)
        fake = self._pending_fake()
        driver = make_driver()

        def sleep(_sec: float) -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        with patch.object(gke_job.subprocess, "run", fake), \
             patch.object(gke_job.time, "sleep", sleep), \
             contextlib.redirect_stderr(io.StringIO()) as err, \
             self.assertRaises(SystemExit) as ctx:
            driver.run(str(self.script), timeout=3600)
        self.assertEqual(ctx.exception.code, 128 + signal.SIGTERM)
        self.assertIsNotNone(fake.find("delete job"))
        # The caller sees why, not just its own timeout.
        self.assertIn("was PENDING", err.getvalue())
        self.assertIn(self.UNSCHEDULABLE, err.getvalue())
        self.assertIn("A resubmit would wait the same way", err.getvalue())
        # Whatever handler the runner had is put back.
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def _wait_through(self, timeout: int, *phases: tuple[float, str]):
        """Run `_wait` on a fake clock against a Job that moves through
        `phases`, each (seconds, status); the last one never ends."""
        clock = [0.0]

        def status(job_id: str) -> base.AsynchronousExecutionResult:
            at, state = clock[0], phases[-1][1]
            for seconds, phase in phases:
                if at < seconds:
                    state = phase
                    break
                at -= seconds
            return base.AsynchronousExecutionResult(job_id=job_id,
                                                    status=state)

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        driver = make_driver(queue="lq")
        with patch.object(driver, "get_status", side_effect=status), \
                patch.object(gke_job.time, "sleep", side_effect=sleep), \
                patch.object(gke_job.time, "monotonic",
                             side_effect=lambda: clock[0]):
            return driver._wait("job-1", timeout)

    def test_queue_time_does_not_count_against_the_timeout(self):
        # 40 minutes queued, then 25 of work, on a 1h timeout.
        result = self._wait_through(3600, (2400, "PENDING"), (1500, "RUNNING"),
                                    (1, "COMPLETED"))
        self.assertEqual(result.status, "COMPLETED")

    def test_run_timeout_reports_the_time_queued(self):
        result = self._wait_through(3600, (600, "PENDING"), (1, "RUNNING"))
        self.assertEqual(result.details["queued_sec"], 600)
        self.assertIn("running for 3600s", result.error)

    def test_a_job_that_never_starts_still_times_out(self):
        result = self._wait_through(300, (1, "PENDING"))
        self.assertEqual(result.status, "PENDING")
        self.assertIn("never ran", result.error)

    def test_interpreter_override_reaches_the_container(self):
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.run(str(self.script),
                       python_interpreter="/opt/py/bin/python")
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        command = workload_command(container)
        self.assertEqual(command[0], "/opt/py/bin/python")

    def test_shell_script_runs_under_bash(self):
        # A `.sh` run without `--interpreter bash` went to python3 and died
        # with a SyntaxError on `set -uo pipefail`.
        script = Path(self.tmp.name) / "discover.sh"
        script.write_text("set -uo pipefail\n", encoding="utf-8")
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})
        driver = make_driver()
        with patch.object(gke_job.subprocess, "run", fake):
            driver.run(str(script), python_interpreter="/opt/py/bin/python")
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(workload_command(container),
                         ["bash", "/app/discover.sh"])
        # The interpreter still reaches the script, to find `vllm` beside it.
        self.assertIn(
            {
                "name": "TPU_AGENT_SUITE_PYTHON",
                "value": "/opt/py/bin/python"
            }, container["env"])

    def test_named_run_creates_a_job_under_that_name(self):
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})
        driver = make_driver(job_name_prefix="agent")
        with patch.object(gke_job.subprocess, "run", fake):
            result = driver.run(str(self.script), job_name="qwen-bench")
        name = result.details["job_id"]
        self.assertRegex(name, r"^agent-qwen-bench-[0-9a-f]{6}$")
        self.assertEqual(fake.applied_manifest()["metadata"]["name"], name)
        self.assertIsNotNone(fake.find(f"create configmap {name}"))
        self.assertIsNotNone(fake.find(f"delete job {name}"))


def node_json(count: int = 1,
              chips: int = 4,
              accelerator: str = "tpu7x") -> str:
    return json.dumps({
        "items": [{
            "metadata": {
                "name": f"node-{i}",
                "labels": {
                    gke_job.ACCELERATOR_LABEL: accelerator,
                    gke_job.TOPOLOGY_LABEL: "2x2x1",
                },
            },
            "status": {
                "allocatable": {
                    gke_job.TPU_RESOURCE: str(chips)
                }
            },
        } for i in range(count)]
    })


def pods_json(*requests: tuple[str, int, str]) -> str:
    """Pods as `kubectl get pods` lists them; a Pending one is unscheduled."""
    return json.dumps({
        "items": [{
            "metadata": {
                "name": name,
                "namespace": "default"
            },
            "status": {
                "phase": phase
            },
            "spec": {
                **({} if phase == "Pending" else {
                       "nodeName": "gke-tpu-node-1"
                   }), "containers": [{
                    "resources": {
                        "limits": {
                            gke_job.TPU_RESOURCE: str(chips)
                        }
                    }
                }]
            },
        } for name, chips, phase in requests]
    })


class TestGkeJobInspect(unittest.TestCase):

    def inspect(self, fake: FakeKubectl, **overrides: Any):
        driver = make_driver(**overrides)
        with patch.object(gke_job.subprocess, "run", fake):
            return driver.inspect()

    def _preflight(self, responses, **overrides):
        """Preflight against a cluster that answers, unless told otherwise.

        `responses` are matched first, so a narrower key can answer one of
        several calls a default would.
        """
        defaults = {
            "config current-context": (0, "ctx", ""),
            "version": (0, '"serverVersion"', ""),
            "auth can-i": (0, "yes\n", ""),
            "get nodes": (0, "node/tpu-1\n", ""),
            "auth list": (0, "me@example.com\n", ""),
        }
        fake = FakeKubectl({
            **responses,
            **{
                k: v
                for k, v in defaults.items() if k not in responses
            }
        })
        env = {
            k: v
            for k, v in os.environ.items()
            if k != gke_job.base.GAC_ENV and not k.startswith("GKE_")
        }
        env.update(overrides.pop("env", {}))
        with patch.object(gke_job.shutil, "which",
                          return_value="/bin/tool"), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(gke_job.subprocess, "run", fake):
            report = make_driver(**overrides).preflight()
        return report, {c.name: c for c in report.checks}, fake

    def test_preflight(self):
        """The cluster answers, and the settings a Job needs are there."""
        report, checks, _ = self._preflight(
            {"version": (1, "", "Unable to connect")})
        self.assertFalse(report.reachable)
        self.assertEqual(checks["cluster"].status, "fail")
        self.assertNotIn("rbac:create jobs", checks)
        report, checks, _ = self._preflight({}, image="")
        self.assertTrue(report.reachable)
        self.assertEqual(
            {name
             for name, c in checks.items() if c.status != "ok"},
            {"dispatch_settings"})

    def test_preflight_checks_the_image_in_artifact_registry(self):
        image = "us-docker.pkg.dev/p/repo/torch-tpu:v1"
        _, checks, fake = self._preflight(
            {"images describe": (0, "sha256:abc\n", "")}, image=image)
        self.assertEqual(checks["image"].status, "ok")
        self.assertIn(image, " ".join(fake.find("images describe")))
        # What gcloud really prints for a missing tag or digest: the error,
        # then a usage block whose last line says nothing is wrong.
        real_not_found = (
            "ERROR: (gcloud.artifacts.docker.images.describe) Image not "
            "found.\n\nA valid container image can be referenced by tag or "
            "digest, has the format of\n"
            "  LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE:tag\n"
            "  LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:"
            "digest\n")
        _, checks, _ = self._preflight(
            {"images describe": (1, "", real_not_found)}, image=image)
        self.assertEqual(checks["image"].status, "fail")
        self.assertIn("Image not found", checks["image"].detail)
        self.assertIn(
            "gcloud artifacts docker images list "
            "us-docker.pkg.dev/p/repo/torch-tpu", checks["image"].fix)
        self.assertNotIn("sha256:digest", checks["image"].detail)
        # Not allowed to read it: the nodes may still pull it.
        _, checks, _ = self._preflight(
            {"images describe": (1, "", "PERMISSION_DENIED: denied")},
            image=image)
        self.assertEqual(checks["image"].status, "warn")
        # Elsewhere, it cannot be checked without pulling it.
        _, checks, fake = self._preflight({}, image="docker.io/me/img:1")
        self.assertEqual(checks["image"].status, "warn")
        self.assertIsNone(fake.find("images describe"))

    CONTEXT = "gke_tpu-nexus-767207_us-central2-b_tpu-nexus-cluster"
    NODE_SA = "141197224043-compute@developer.gserviceaccount.com"

    @staticmethod
    def _readers(member):
        """A repository policy granting `member` roles/artifactregistry.reader."""
        return {
            "bindings": [{
                "role": "roles/artifactregistry.reader",
                "members": [member]
            }]
        }

    def _node_pull(self, image, policy=None, driver=None, **responses):
        """Preflight of `image` on a gke_* context, with the TPU nodes in
        pool 'tpu-pool' running as the default Compute Engine account."""
        pools = [{
            "name": "tpu-pool",
            "config": {
                "serviceAccount": "default"
            }
        }, {
            "name": "other-pool",
            "config": {
                "serviceAccount": "other@x.iam.gserviceaccount.com"
            }
        }]
        _, checks, fake = self._preflight(
            {
                "config current-context": (0, self.CONTEXT, ""),
                "images describe": (0, "sha256:abc\n", ""),
                "get nodes": (0, "tpu-pool\ntpu-pool\n", ""),
                "node-pools list": (0, json.dumps(pools), ""),
                "projects describe":
                (0, json.dumps({"projectNumber": "141197224043"}), ""),
                "get-iam-policy":
                ((0, json.dumps(policy), "") if policy is not None else
                 (1, "", "ERROR: PERMISSION_DENIED")),
                **responses
            },
            image=image,
            **(driver or {}))
        return checks, fake

    def test_preflight_warns_the_nodes_may_not_pull_a_cross_project_image(
            self):
        """The host can read the image; the nodes pull as themselves."""
        image = ("us-docker.pkg.dev/ml-oss-artifacts-transient/"
                 "torch-tpu-docker-container/torch-tpu:nightly-latest")
        checks, fake = self._node_pull(image)
        pull = checks["image:node_pull"]
        self.assertEqual(pull.status, "warn")
        # Only the pool the TPU nodes are in, with "default" resolved.
        self.assertIn(self.NODE_SA, pull.detail)
        self.assertNotIn("other@x", pull.detail)
        self.assertIn("ml-oss-artifacts-transient", pull.detail)
        self.assertIn("cannot read its policy", pull.detail)
        self.assertIn(
            "add-iam-policy-binding torch-tpu-docker-container --location us "
            "--project ml-oss-artifacts-transient --role "
            f"roles/artifactregistry.reader --member serviceAccount:"
            f"{self.NODE_SA}", pull.fix)
        self.assertIn("--cluster tpu-nexus-cluster --location us-central2-b",
                      " ".join(fake.find("node-pools list")))
        # A policy that leaves them out says so.
        checks, _ = self._node_pull(image,
                                    policy=self._readers("group:someone@x"))
        self.assertIn("does not name them", checks["image:node_pull"].detail)
        # One that grants them, or everyone, read: ok.
        for member in (f"serviceAccount:{self.NODE_SA}", "allUsers"):
            checks, _ = self._node_pull(image, policy=self._readers(member))
            self.assertEqual(checks["image:node_pull"].status, "ok", member)

    def test_preflight_node_pull_without_the_node_accounts(self):
        """Pools gcloud cannot list: a generic warning, unless the
        repository is public."""
        public = self._readers("allAuthenticatedUsers")
        for policy, status, says in ((None, "warn",
                                      "the node pools' service account"),
                                     (public, "ok", "grants the nodes read")):
            checks, _ = self._node_pull(
                "us-docker.pkg.dev/p/repo/img:1",
                policy=policy,
                **{"node-pools list": (1, "", "ERROR: PERMISSION_DENIED")})
            pull = checks["image:node_pull"]
            self.assertEqual(pull.status, status)
            self.assertIn(says, pull.detail)

    def test_preflight_node_pull_picks_a_scaled_down_pool_by_topology(self):
        """No TPU nodes to name the pool: the pool of that topology."""
        pools = [{
            "name": "tpu-pool",
            "placementPolicy": {
                "tpuTopology": "2x2x1"
            },
            "config": {
                "serviceAccount": "tpu@x.iam.gserviceaccount.com"
            }
        }, {
            "name": "cpu-pool",
            "config": {
                "serviceAccount": "cpu@x.iam.gserviceaccount.com"
            }
        }]
        checks, _ = self._node_pull(
            "us-docker.pkg.dev/p/repo/img:1", **{
                "get nodes": (0, "", ""),
                "node-pools list": (0, json.dumps(pools), "")
            })
        pull = checks["image:node_pull"]
        self.assertIn("tpu@x", pull.detail)
        self.assertNotIn("cpu@x", pull.detail)
        # No pool of that topology: not every pool's account, a generic one.
        checks, _ = self._node_pull("us-docker.pkg.dev/p/repo/img:1",
                                    driver={"tpu_topology": "4x4"},
                                    **{
                                        "get nodes": (0, "", ""),
                                        "node-pools list":
                                        (0, json.dumps(pools), "")
                                    })
        self.assertIn("the node pools' service account",
                      checks["image:node_pull"].detail)

    def test_preflight_says_nothing_of_node_pulls_it_cannot_judge(self):
        # The cluster's own project.
        checks, fake = self._node_pull(
            "us-central2-docker.pkg.dev/tpu-nexus-767207/nexus-dev/img:1")
        self.assertNotIn("image:node_pull", checks)
        self.assertIsNone(fake.find("node-pools list"))
        # Not Artifact Registry.
        checks, _ = self._node_pull("docker.io/me/img:1")
        self.assertNotIn("image:node_pull", checks)
        # A context that does not name a GKE cluster, without GKE_*.
        renamed = {"config current-context": (0, "ctx", "")}
        checks, _ = self._node_pull("us-docker.pkg.dev/p/repo/img:1",
                                    **renamed)
        self.assertNotIn("image:node_pull", checks)
        # GKE_* names it instead.
        gke = {
            f"GKE_{part}": value
            for part, value in (("CLUSTER", "c"), ("LOCATION", "l"),
                                ("PROJECT", "cp"))
        }
        checks, fake = self._node_pull("us-docker.pkg.dev/p/repo/img:1",
                                       driver={"env": gke},
                                       **renamed)
        self.assertEqual(checks["image:node_pull"].status, "warn")
        self.assertIn("--cluster c --location l --project cp",
                      " ".join(fake.find("node-pools list")))
        # An image that does not exist: its failure says enough.
        checks, _ = self._node_pull(
            "us-docker.pkg.dev/p/repo/img:1",
            **{"images describe": (1, "", "ERROR: Image not found.")})
        self.assertNotIn("image:node_pull", checks)

    def test_preflight_times_out_as_a_network_problem(self):
        """A cluster that never answers is not a credentials problem."""
        _, checks, _ = self._preflight({
            "version": (1, "", "Unable to connect to the server: context "
                        "deadline exceeded")
        })
        self.assertEqual(checks["cluster"].status, "fail")
        self.assertIn("authorized networks", checks["cluster"].fix)
        self.assertIn("--dns-endpoint", checks["cluster"].fix)
        _, checks, _ = self._preflight(
            {"version": (1, "", "error: You must be logged in")})
        self.assertNotIn("authorized networks", checks["cluster"].fix)

    def test_preflight_warns_on_an_unknown_accelerator(self):
        report, checks, fake = self._preflight({"get nodes": (0, "", "")},
                                               tpu_accelerator='pu-v6e-slice"')
        self.assertEqual(checks["tpu_accelerator"].status, "warn")
        self.assertIn("tpu-v6e-slice", checks["tpu_accelerator"].fix)
        # No node yet may be a pool that scales from zero: never a failure.
        self.assertEqual(checks["tpu_nodes"].status, "warn")
        self.assertIn("scales from zero", checks["tpu_nodes"].fix)
        self.assertIn('pu-v6e-slice"', " ".join(fake.find("get nodes")))
        self.assertNotEqual(report.status, "fail")
        _, checks, _ = self._preflight({})
        self.assertNotIn("tpu_accelerator", checks)
        self.assertEqual(checks["tpu_nodes"].status, "ok")

    def test_preflight_checks_the_namespace_rights_and_kueue(self):
        _, checks, fake = self._preflight(
            {
                "get namespace": (1, "", 'Error from server (Forbidden): '
                                  'namespaces "ns" is forbidden'),
                "get localqueue": (1, "", "error: the server doesn't have a "
                                   'resource type "localqueue"'),
                "get workloadpriorityclass": (0, "low", ""),
            },
            namespace="ns",
            queue="q",
            priority="low")
        self.assertEqual(checks["namespace"].status, "warn")
        self.assertEqual(checks["rbac:create jobs"].status, "ok")
        self.assertEqual(checks["rbac:create configmaps"].status, "ok")
        self.assertEqual(checks["kueue:queue"].status, "fail")
        self.assertIn("Install Kueue", checks["kueue:queue"].fix)
        self.assertEqual(checks["kueue:priority"].status, "ok")
        self.assertIn("-n ns", " ".join(fake.find("get localqueue")))
        self.assertIn("-n ns", " ".join(fake.find("auth can-i")))

    def test_preflight_fails_an_account_that_may_not_create_jobs(self):
        _, checks, _ = self._preflight({"auth can-i": (1, "no\n", "")})
        self.assertEqual(checks["rbac:create jobs"].status, "fail")
        self.assertIn("Ask a cluster admin", checks["rbac:create jobs"].fix)

    def test_preflight_without_a_queue_skips_kueue(self):
        _, checks, fake = self._preflight({}, priority="low")
        self.assertNotIn("namespace", checks)  # `default` always exists
        self.assertNotIn("kueue:queue", checks)
        self.assertNotIn("kueue:priority", checks)
        self.assertIsNone(fake.find("localqueue"))

    def test_preflight_checks_staging_only_with_a_bucket(self):
        staged = [base.PreflightCheck("staging:signing", "fail", "no")]
        with patch.object(gcs_staging, "preflight_checks",
                          return_value=staged) as staging:
            _, checks, _ = self._preflight({}, staging_bucket="gs://b/")
            staging.assert_called_once_with("b",
                                            credentials=None,
                                            pod_service_account=None)
            self.assertEqual(checks["staging:signing"].status, "fail")
            # Setup asks again for what lets the pod reach the bucket.
            self.assertEqual(checks["staging:signing"].fields,
                             list(gcs_staging.POD_UPLOAD_FIELDS))
            self._preflight({})
            staging.assert_called_once()

    def test_preflight_checks_the_pod_service_account_and_its_nodes(self):
        """A missing account leaves the Job with no pod, and a
        node without the metadata server gives the pod no token."""
        _, checks, fake = self._preflight(
            {
                "get serviceaccount":
                (1, "", 'Error from server (NotFound): serviceaccounts "ksa" '
                 "not found"),
                "get nodes": (0, "", ""),
            },
            pod_service_account="ksa")
        self.assertEqual(checks["pod_service_account"].status, "fail")
        fix = checks["pod_service_account"].fix
        self.assertIn("kubectl create serviceaccount ksa -n default", fix)
        self.assertIn("iam.gke.io/gcp-service-account", fix)
        self.assertIn(f"{gke_job.WORKLOAD_IDENTITY_LABEL}=true",
                      " ".join(fake.find("get nodes")))
        # No TPU node at all: the plain warning, not the Workload Identity one.
        self.assertNotIn("GKE_METADATA", checks["tpu_nodes"].fix)
        _, checks, _ = self._preflight({}, pod_service_account="ksa")
        self.assertEqual(checks["pod_service_account"].status, "ok")
        _, checks, _ = self._preflight({})
        self.assertNotIn("pod_service_account", checks)

    def test_preflight_names_tpu_pools_without_workload_identity(self):
        _, checks, _ = self._preflight(
            {
                f"{gke_job.WORKLOAD_IDENTITY_LABEL}=true": (0, "", ""),
                "get nodes": (0, "tpu-pool\n", ""),
            },
            pod_service_account="ksa")
        self.assertEqual(checks["tpu_nodes"].status, "warn")
        self.assertIn("tpu-pool", checks["tpu_nodes"].detail)
        self.assertIn("GKE_METADATA", checks["tpu_nodes"].fix)

    @staticmethod
    def _pvc(phase: str = "Bound",
             modes: tuple[str, ...] = ("ReadWriteMany", ),
             storage_class: str = "filestore-rwx") -> tuple[int, str, str]:
        status: dict[str, Any] = {"phase": phase}
        if phase == "Bound":
            status.update(accessModes=list(modes), capacity={"storage": "1Ti"})
        return (0,
                json.dumps({
                    "spec": {
                        "accessModes": list(modes),
                        "storageClassName": storage_class
                    },
                    "status": status
                }), "")

    def test_preflight_checks_the_hf_cache_pvc_only_when_set(self):
        _, checks, fake = self._preflight({})
        self.assertNotIn("hf_cache_pvc", checks)
        self.assertIsNone(fake.find("get pvc"))

        report, checks, fake = self._preflight({"get pvc": self._pvc()},
                                               namespace="ns",
                                               hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "ok")
        self.assertIn("Bound, ReadWriteMany, 1Ti",
                      checks["hf_cache_pvc"].detail)
        self.assertIn("-n ns", " ".join(fake.find("get pvc hf-cache")))

    def test_preflight_fails_a_missing_hf_cache_pvc(self):
        # Every Job naming it would sit Pending, so the installer must flag
        # it before anything is written.
        report, checks, _ = self._preflight(
            {
                "get pvc": (1, "", 'Error from server (NotFound): '
                            'persistentvolumeclaims "hf-cache" not found')
            },
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "fail")
        self.assertIn("not found", checks["hf_cache_pvc"].detail)
        self.assertIn("Ask a cluster admin to create a ReadWriteMany",
                      checks["hf_cache_pvc"].fix)
        self.assertEqual(report.status, "fail")

    def test_preflight_fails_an_hf_cache_pvc_that_is_not_read_write_many(self):
        _, checks, _ = self._preflight(
            {"get pvc": self._pvc(modes=("ReadWriteOnce", ))},
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "fail")
        self.assertIn("ReadWriteOnce, not ReadWriteMany",
                      checks["hf_cache_pvc"].detail)

    def test_preflight_hf_cache_pvc_pending(self):
        # Pending is healthy under a class that binds on first use...
        _, checks, fake = self._preflight(
            {
                "get pvc": self._pvc(phase="Pending"),
                "get storageclass": (0, "WaitForFirstConsumer", ""),
            },
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "warn")
        self.assertIn("until the first Job", checks["hf_cache_pvc"].detail)
        self.assertIn("filestore-rwx", " ".join(fake.find("get storageclass")))
        # ...and a failure under one that should already have bound it.
        _, checks, _ = self._preflight(
            {
                "get pvc": self._pvc(phase="Pending"),
                "get storageclass": (0, "Immediate", ""),
            },
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "fail")
        self.assertIn("not Bound", checks["hf_cache_pvc"].detail)
        # A class this account cannot read leaves it undecided.
        _, checks, _ = self._preflight(
            {
                "get pvc": self._pvc(phase="Pending"),
                "get storageclass": (1, "", "Error from server (Forbidden)"),
            },
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "warn")

    def test_preflight_warns_when_the_hf_cache_pvc_cannot_be_read(self):
        _, checks, _ = self._preflight(
            {
                "get pvc": (1, "", 'Error from server (Forbidden): '
                            'persistentvolumeclaims is forbidden')
            },
            hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "warn")

    def test_no_matching_nodes_reports_no_accelerator(self):
        fake = FakeKubectl({"get nodes": (0, json.dumps({"items": []}), "")})
        profile = self.inspect(fake)
        self.assertEqual(profile.accelerator, "none")
        self.assertEqual(profile.chips, 0)
        self.assertEqual(profile.status, "no_accelerator")

    def test_a_pool_with_no_node_still_reports_the_profiles_tpu(self):
        """Review on #840: an autoscaled pool at zero nodes read as
        `accelerator: none`, leaving a CPU preflight Job nothing to plan for
        and TP nothing to be sized to. What the profile names is reported in
        `details`, and the top-level fields still say no node is there."""
        fake = FakeKubectl({"get nodes": (0, json.dumps({"items": []}), "")})
        with patch.object(gke_job,
                          "_chip_spec",
                          return_value=({
                              "hbm_gb": 192.0,
                              "devices_per_chip": 2
                          }, "")):
            profile = self.inspect(fake, tpu_accelerator="tpu7x")
        self.assertEqual(profile.status, "no_accelerator")
        details = profile.details
        self.assertEqual((details["tpu_accelerator"], details["job_devices"],
                          details["hbm_gb_per_device"]), ("tpu7x", 8, 96.0))

    def test_chips_summed_across_the_pool(self):
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=4, chips=4), ""),
            "get pods": (0, pods_json(), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual(profile.accelerator, "tpu7x")
        self.assertEqual(profile.topology, "2x2x1")
        self.assertEqual(profile.chips, 16)
        self.assertEqual(profile.available_chips, 16)
        self.assertEqual(profile.status, "healthy")
        self.assertEqual(profile.hbm_gb, 192.0)

    def test_job_devices_and_hbm_come_from_tpu_info(self):
        # A 2x2x1 tpu7x Job is 4 chips but 8 devices of 96 GB each; reporting
        # only chips and 192 GB per chip is what led a run to TP=4 on half a
        # slice. What tpu_info cannot tell is left out rather than guessed.
        cases = [
            # (label, tpu_info importable, hbm_gb, devices_per_chip, note)
            ("tpu7x", True, 192.0, 2, None),
            ("tpu-v6e-slice", True, 32.0, 1, None),
            ("tpu-v99-slice", True, 0.0, None, "tpu-v99-slice"),
            ("tpu7x", False, 0.0, None,
             "Before anything else, ask the user to install this package"),
        ]
        for label, has_tpu_info, hbm_gb, per_chip, note in cases:
            with self.subTest(label=label, tpu_info=has_tpu_info):
                fake = FakeKubectl({
                    "get nodes": (0, node_json(accelerator=label), ""),
                    "get pods": (0, pods_json(), ""),
                })
                missing = {} if has_tpu_info else {
                    "tpu_info": None,
                    "tpu_info.device": None
                }
                with patch.dict(sys.modules, missing):
                    profile = self.inspect(fake, tpu_accelerator=label)
                details = profile.details
                self.assertEqual(profile.hbm_gb, hbm_gb)
                self.assertEqual(details["job_chips"], 4)
                self.assertEqual(details.get("devices_per_chip"), per_chip)
                if per_chip:
                    self.assertEqual(details["job_devices"], 4 * per_chip)
                    self.assertEqual(details["hbm_gb_per_device"],
                                     hbm_gb / per_chip)
                else:
                    self.assertNotIn("job_devices", details)
                    self.assertIn(note, details["hbm_gb_note"])

    def test_running_pods_subtract_from_available_chips(self):
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=2, chips=4), ""),
            "get pods": (0, pods_json(("busy", 4, "Running")), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual(profile.available_chips, 4)
        self.assertEqual(profile.held_by, ["default/busy"])

    def test_finished_and_unscheduled_pods_do_not_hold_chips(self):
        # A Pending pod with no node is waiting for chips, not holding them;
        # counted, it read an empty cluster as busy.
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=1, chips=4), ""),
            "get pods":
            (0, pods_json(("old", 4, "Succeeded"),
                          ("waiting", 4, "Pending")), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual(profile.available_chips, 4)
        self.assertEqual(profile.held_by, [])

    def test_full_pool_reports_busy(self):
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=1, chips=4), ""),
            "get pods": (0, pods_json(("busy", 4, "Running")), ""),
        })
        self.assertEqual(self.inspect(fake).status, "busy")

    def test_unlistable_pods_leave_contention_unknown(self):
        # Not 0 and not `chips`: one would report a healthy pool as saturated
        # and the other would send a scheduler at chips somebody else holds.
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=1, chips=4), ""),
            "get pods": (1, "", "Forbidden"),
        })
        profile = self.inspect(fake)
        self.assertIsNone(profile.available_chips)
        self.assertEqual(profile.status, "unknown_contention")

    def test_unknown_generation_reports_no_hbm_and_says_why(self):
        nodes = json.dumps({
            "items": [{
                "metadata": {
                    "name": "n",
                    "labels": {
                        gke_job.ACCELERATOR_LABEL: "tpu9z",
                        gke_job.TOPOLOGY_LABEL: "1x1",
                    },
                },
                "status": {
                    "allocatable": {
                        gke_job.TPU_RESOURCE: "1"
                    }
                },
            }]
        })
        fake = FakeKubectl({
            "get nodes": (0, nodes, ""),
            "get pods": (0, pods_json(), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual(profile.hbm_gb, 0.0)
        self.assertIn("tpu9z", profile.details["hbm_gb_note"])

    def test_selector_narrows_to_the_configured_pool(self):
        fake = FakeKubectl({
            "get nodes": (0, node_json(), ""),
            "get pods": (0, pods_json(), ""),
        })
        self.inspect(fake, tpu_accelerator="tpu7x", tpu_topology="2x2x1")
        selector = fake.find("get nodes")
        self.assertIn(f"{gke_job.ACCELERATOR_LABEL}=tpu7x", " ".join(selector))
        self.assertIn(f"{gke_job.TOPOLOGY_LABEL}=2x2x1", " ".join(selector))


class TestGkeJobStaging(unittest.TestCase):
    """File staging via GCS and signed URLs on gke_job."""

    def setUp(self):
        # Dispatch writes job records; none of them belong in the real store.
        self.enterContext(_job_store())

    def test_clean_bucket_name(self):
        self.assertEqual(gcs_staging.clean_bucket_name("gs://my-bucket"),
                         "my-bucket")
        self.assertEqual(gcs_staging.clean_bucket_name("gs://my-bucket/"),
                         "my-bucket")
        self.assertEqual(gcs_staging.clean_bucket_name("my-bucket"),
                         "my-bucket")
        self.assertEqual(gcs_staging.clean_bucket_name(""), "")

    def test_build_pod_bootstrap_command(self):
        cmd = gcs_staging.build_pod_bootstrap_command(
            interpreter="python3",
            script_mount="/opt/workload",
            script_name="workload.py",
            args=["--flag", "val"],
            input_urls=[("sub/in.txt", "https://gcs/in?sig=1")],
            output_urls=[("out.txt", "https://gcs/out?sig=2")],
        )
        self.assertEqual(cmd[0], "bash")
        self.assertEqual(cmd[1], "-c")
        script = cmd[2]
        self.assertIn("mkdir -p sub", script)
        self.assertIn(
            f"mkdir -p {gcs_staging.POD_WORK_DIR} && cd "
            f"{gcs_staging.POD_WORK_DIR}", script)
        self.assertIn("curl -fsSL --retry 3 -o sub/in.txt", script)
        self.assertIn("python3 /opt/workload/workload.py --flag val", script)
        self.assertIn(
            "curl -fsSL --retry 3 -X PUT -H 'Content-Type: application/octet-stream' "
            "--upload-file out.txt",
            script,
        )
        self.assertIn("exit $WORKLOAD_EXIT_CODE", script)

    def test_bootstrap_uploads_the_log_without_failing_on_it(self):
        script = gcs_staging.build_pod_bootstrap_command(
            interpreter="python3",
            script_mount="/opt/workload",
            script_name="workload.py",
            log_url="https://gcs/log?sig=3",
        )[2]
        self.assertIn(
            f"tee {gcs_staging.POD_LOG_FILE} < {gcs_staging.LOG_FIFO} &",
            script)
        self.assertIn(f"> {gcs_staging.LOG_FIFO} 2>&1 &", script)
        self.assertIn("https://gcs/log?sig=3' || echo", script)

    @staticmethod
    def _client_with(creds: Any) -> tuple[MagicMock, MagicMock]:
        client = MagicMock()
        client._credentials = creds
        blob = MagicMock()
        client.bucket.return_value.blob.return_value = blob
        blob.generate_signed_url.return_value = (
            "https://storage.googleapis.com/test-signed-url")
        return client, blob

    def test_key_or_impersonated_credentials_sign_locally(self):
        """Credentials with their own signer need no IAM detour."""

        class KeyCreds:
            signer_email = "sa@proj.iam.gserviceaccount.com"

            def sign_bytes(self, data):
                return b"sig"

        client, blob = self._client_with(KeyCreds())
        url = gcs_staging.generate_v4_signed_url("my-bucket",
                                                 "in/data.txt",
                                                 client=client)
        self.assertEqual(url, "https://storage.googleapis.com/test-signed-url")
        kwargs = blob.generate_signed_url.call_args.kwargs
        self.assertNotIn("service_account_email", kwargs)
        self.assertNotIn("access_token", kwargs)

    def test_ambient_gce_credentials_sign_through_iam_after_refresh(self):
        """A GCE/GKE identity has no key; it is refreshed for its real
        email and token, and those go to the IAM signBlob path."""

        class GceCreds:

            def __init__(self):
                self.service_account_email = "default"
                self.token = None

            def refresh(self, req):
                self.service_account_email = "my-sa@proj.iam.gserviceaccount.com"
                self.token = "fake-token"

        creds = GceCreds()
        client, blob = self._client_with(creds)
        with _fake_google_auth(creds):
            url = gcs_staging.generate_v4_signed_url("my-bucket",
                                                     "in/data.txt",
                                                     client=client)
        self.assertEqual(url, "https://storage.googleapis.com/test-signed-url")
        kwargs = blob.generate_signed_url.call_args.kwargs
        self.assertEqual(kwargs["service_account_email"],
                         "my-sa@proj.iam.gserviceaccount.com")
        self.assertEqual(kwargs["access_token"], "fake-token")

    def test_iam_signing_failure_names_the_missing_role(self):

        class GceCreds:
            service_account_email = "my-sa@proj.iam.gserviceaccount.com"
            token = "fake-token"

        client, blob = self._client_with(GceCreds())
        blob.generate_signed_url.side_effect = Exception(
            "403 Forbidden: Permission denied")
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("my-bucket",
                                               "in/data.txt",
                                               client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))
        self.assertIn("my-sa@proj.iam.gserviceaccount.com", str(ctx.exception))

    @contextlib.contextmanager
    def _fake_storage(self, creds: Any):
        client, blob = self._client_with(creds)
        storage = types.ModuleType("google.cloud.storage")
        storage.Client = MagicMock(return_value=client)
        cloud = types.ModuleType("google.cloud")
        cloud.storage = storage
        with _fake_google_auth(creds), patch.dict(
                sys.modules, {
                    "google.cloud": cloud,
                    "google.cloud.storage": storage
                }):
            yield client, blob

    def test_preflight_signs_and_lists_the_bucket(self):

        class KeyCreds:
            signer_email = "sa@proj.iam.gserviceaccount.com"

            def sign_bytes(self, data):
                return b"sig"

        with self._fake_storage(KeyCreds()) as (client, blob):
            checks = gcs_staging.preflight_checks("b")
        self.assertEqual([(c.name, c.status) for c in checks],
                         [("staging:credentials", "ok"),
                          ("staging:signing", "ok"), ("staging:bucket", "ok")])
        blob.upload_from_filename.assert_not_called()
        client.list_blobs.assert_called_once_with("b",
                                                  prefix=None,
                                                  max_results=1)
        # A folder in the bucket is listed as one, not as a bucket name.
        with self._fake_storage(KeyCreds()) as (client, _):
            gcs_staging.preflight_checks("gs://b/runs/")
        client.list_blobs.assert_called_once_with("b",
                                                  prefix="runs/",
                                                  max_results=1)

    def test_preflight_does_not_ask_a_pod_token_setup_to_sign(self):
        """With pod_service_account nothing is signed, so a
        person's login, which cannot sign, is not a warning."""

        class UserCreds:
            token = "mock-user-token"

        with self._fake_storage(UserCreds()) as (_, blob):
            checks = gcs_staging.preflight_checks("b",
                                                  pod_service_account="ksa")
        self.assertEqual({c.name: c.status
                          for c in checks}, {
                              "staging:credentials": "ok",
                              "staging:signing": "ok",
                              "staging:bucket": "ok"
                          })
        blob.generate_signed_url.assert_not_called()

    def test_preflight_reports_credentials_that_cannot_sign(self):

        class UserCreds:
            token = "mock-user-token"

        with self._fake_storage(UserCreds()) as (client, _):
            error = Exception("403 no list")
            error.code = 403
            client.list_blobs.side_effect = error
            checks = {c.name: c for c in gcs_staging.preflight_checks("b")}
        self.assertEqual(checks["staging:signing"].status, "fail")
        self.assertIn("--impersonate-service-account",
                      checks["staging:signing"].fix)
        # Forbidden to list is not proof the bucket is missing.
        self.assertEqual(checks["staging:bucket"].status, "warn")

    def test_user_credentials_are_refused_with_a_way_forward(self):
        """`gcloud auth application-default login` on a laptop: no key, no
        service account. Say so before touching the library, and say what
        to do instead."""

        class UserCreds:
            token = "mock-user-token"

        client, blob = self._client_with(UserCreds())
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("my-bucket",
                                               "in/data.txt",
                                               client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_UNSUPPORTED")
        message = str(ctx.exception)
        self.assertIn("--impersonate-service-account", message)
        # The staging-only credential comes first: it signs without changing
        # who kubectl is.
        self.assertLess(message.index(gcs_staging.SERVICE_ACCOUNT_PARAM),
                        message.index("--impersonate-service-account"))
        self.assertIn(gcs_staging.KEY_FILE_PARAM, message)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", message)
        # Never mentioned without what it does to kubectl.
        self.assertIn(gke_job.base.KUBECTL_IDENTITY_WARNING, message)
        blob.generate_signed_url.assert_not_called()

    def test_staging_credential_validation(self):
        self.assertIsNone(gcs_staging.staging_credentials(None, None))
        self.assertIsNone(gcs_staging.staging_credentials(" ", ""))
        for fn in (
                lambda: gcs_staging.staging_credentials(
                    "sa@p.iam.gserviceaccount.com", "/k.json"),
                lambda: gke_job.GkeJobDriver.from_params({
                    gcs_staging.SERVICE_ACCOUNT_PARAM:
                    "sa@p.iam.gserviceaccount.com",
                    gcs_staging.KEY_FILE_PARAM:
                    "/k.json",
                }),
                lambda: gcs_staging.staging_credentials(
                    key_file="/nonexistent/key.json"),
        ):
            with self.assertRaises(ComputeConfigError) as ctx:
                fn()
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertIn(gcs_staging.KEY_FILE_PARAM, str(ctx.exception))
        driver = make_driver(
            staging_bucket="gs://b",
            staging_service_account_key="/nonexistent/key.json")
        self.assertEqual(driver._staging(best_effort=True), {})
        with self.assertRaises(ComputeConfigError):
            driver._staging()

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_impersonated_staging_credential_signs_locally(self):
        """Impersonation from ADC, built without touching the environment."""
        from google.auth import impersonated_credentials  # noqa: PLC0415

        source = MagicMock()
        with patch("google.auth.default",
                   return_value=(source,
                                 "proj")) as default, patch.dict(os.environ,
                                                                 {},
                                                                 clear=False):
            os.environ.pop(gke_job.base.GAC_ENV, None)
            creds = gcs_staging.staging_credentials(
                "stager@p.iam.gserviceaccount.com")
            self.assertNotIn(gke_job.base.GAC_ENV, os.environ)
        default.assert_called_once()
        self.assertIsInstance(creds, impersonated_credentials.Credentials)
        self.assertEqual(creds.signer_email,
                         "stager@p.iam.gserviceaccount.com")
        self.assertEqual(gcs_staging._signing_kwargs(creds), {})
        client, blob = self._client_with(creds)
        blob.generate_signed_url.side_effect = Exception("403 signBlob denied")
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))
        self.assertIn("stager@p.iam.gserviceaccount.com", str(ctx.exception))
        with self.assertRaises(ComputeConfigError) as ctx:
            with gcs_staging._gcs_errors("upload to", "gs://b/x", client):
                raise _google_auth_errors.RefreshError("getAccessToken denied")
        self.assertEqual(ctx.exception.error_code, "GCS_FORBIDDEN")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))

    def test_preflight_uses_the_staging_credential_not_adc(self):

        class KeyCreds:
            signer_email = "stager@p.iam.gserviceaccount.com"

            def sign_bytes(self, data):
                return b"sig"

        with self._fake_storage(MagicMock()) as (client, _):
            storage = sys.modules["google.cloud.storage"]
            with patch("google.auth.default") as default:
                checks = gcs_staging.preflight_checks("b",
                                                      credentials=KeyCreds())
        default.assert_not_called()
        self.assertIsInstance(storage.Client.call_args.kwargs["credentials"],
                              KeyCreds)
        self.assertEqual(checks[0].name, "staging:credentials")
        self.assertEqual(checks[0].status, "ok")

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_credentials_that_will_not_refresh_cannot_sign(self):
        """An ambient identity whose token cannot be had, off GCE or with
        no network: a ComputeConfigError, which callers fall back on."""

        class AmbientCreds:
            service_account_email = "default"
            token = None

            def refresh(self, request):
                raise _google_auth_errors.RefreshError("metadata unreachable")

        client, _ = self._client_with(AmbientCreds())
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("metadata unreachable", str(ctx.exception))

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    def test_run_requires_staging_bucket_when_staging(self):
        driver = make_driver()
        with tempfile.NamedTemporaryFile("w", suffix=".py") as tf:
            tf.write("print('ok')\n")
            tf.flush()
            staged = base.StagedFile(tf.name, "in.txt")
            with self.assertRaises(ComputeConfigError) as ctx:
                driver.run(tf.name, inputs=[staged])
            self.assertEqual(ctx.exception.error_code,
                             "STAGING_NOT_CONFIGURED")
            self.assertEqual(ctx.exception.missing, ["staging_bucket"])

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.download_file_from_gcs")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_run_with_gcs_staging(self, mock_upload, mock_sign, mock_download,
                                  mock_delete):
        mock_sign.return_value = "https://storage.googleapis.com/test-url"
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "wait": (0, "job.batch/run-1 condition met", ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "hello output\n", ""),
        })
        driver = make_driver(staging_bucket="gs://test-staging-bucket")

        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            in_file = Path(td) / "input.dat"
            in_file.write_text("in-data")
            out_file = Path(td) / "output.dat"

            def fake_download(bucket, blob, dest):
                Path(dest).write_text("downloaded-result")
                return True

            mock_download.side_effect = fake_download

            staged_in = base.StagedFile(str(in_file), "input.dat")
            staged_out = base.StagedFile(str(out_file), "output.dat")

            with patch.object(gke_job.subprocess, "run", fake):
                res = driver.run(
                    str(script_path),
                    inputs=[staged_in],
                    outputs=[staged_out],
                )
            self.assertEqual(res.exit_code, 0)
            self.assertEqual(res.stdout, "hello output\n")
            self.assertTrue(out_file.is_file())
            self.assertEqual(out_file.read_text(), "downloaded-result")
            mock_upload.assert_called_once()
            mock_delete.assert_called_with("test-staging-bucket",
                                           f"{res.details['job_id']}/")

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.get_storage_client")
    @patch("framework.compute.drivers.gcs_staging.staging_credentials")
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.download_file_from_gcs")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_run_stages_with_its_own_credential(self, mock_upload, mock_sign,
                                                mock_download, mock_delete,
                                                mock_creds, mock_client):
        """With a staging credential configured, every staging
        call acts as it, and nothing sets $GOOGLE_APPLICATION_CREDENTIALS
        for kubectl to inherit."""
        mock_sign.return_value = "https://storage.googleapis.com/test-url"
        client = mock_client.return_value
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "wait": (0, "job.batch/run-1 condition met", ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "hello output\n", ""),
        })
        driver = make_driver(
            staging_bucket="gs://test-staging-bucket",
            staging_service_account="stager@p.iam.gserviceaccount.com",
        )
        mock_download.side_effect = lambda _b, _k, dest, **_kw: bool(
            Path(dest).write_text("downloaded-result"))
        environ = dict(os.environ)
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            in_file = Path(td) / "input.dat"
            in_file.write_text("in-data")
            with patch.object(gke_job.subprocess, "run", fake):
                res = driver.run(
                    str(script_path),
                    inputs=[base.StagedFile(str(in_file), "input.dat")],
                    outputs=[
                        base.StagedFile(str(Path(td) / "o.dat"), "o.dat")
                    ],
                )
        self.assertEqual(res.exit_code, 0)
        mock_creds.assert_called_once_with("stager@p.iam.gserviceaccount.com",
                                           None)
        mock_client.assert_called_once_with(
            credentials=mock_creds.return_value)
        for mock in (mock_upload, mock_sign, mock_download, mock_delete):
            self.assertTrue(mock.call_args_list, mock)
            for call in mock.call_args_list:
                self.assertIs(call.kwargs.get("client"), client)
        self.assertEqual(dict(os.environ), environ)

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_submit_signs_for_the_long_haul(self, mock_upload, mock_sign,
                                            mock_delete):
        """A submitted job may outlive any wait, so its URLs must too."""
        del mock_upload, mock_delete
        mock_sign.return_value = "https://storage.googleapis.com/test-url"
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://test-staging-bucket")
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            in_file = Path(td) / "input.dat"
            in_file.write_text("in-data")
            with patch.object(gke_job.subprocess, "run", fake):
                driver.submit(
                    str(script_path),
                    inputs=[base.StagedFile(str(in_file), "input.dat")],
                    outputs=[
                        base.StagedFile(str(Path(td) / "o.dat"), "o.dat")
                    ],
                )
        for call in mock_sign.call_args_list:
            self.assertEqual(call.kwargs["expiration_seconds"],
                             gke_job.SUBMIT_URL_EXP_SEC)

    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    def test_submit_without_files_survives_credentials_that_cannot_sign(
            self, mock_sign):
        """The log copy is a backup; a person's ADC must not fail the Job."""
        mock_sign.side_effect = ComputeConfigError(
            "cannot sign", error_code="SIGNING_UNSUPPORTED")
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://test-staging-bucket")
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            with patch.object(gke_job.subprocess, "run", fake), \
                    patch("sys.stderr", new_callable=io.StringIO) as err:
                driver.submit(str(script_path))
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        command = workload_command(container)
        self.assertEqual(command[0], "python3")
        self.assertIn("SIGNING_UNSUPPORTED", err.getvalue())

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_submit_cleans_up_staged_blobs_when_it_fails(self, mock_upload):
        """Blobs uploaded for a Job that never started have nothing to own them.

        The upload comes first, so every later step that can fail must clean
        it up: the signing, the ConfigMap create and the Job apply.
        """
        del mock_upload
        refused = ComputeConfigError("cannot sign",
                                     error_code="SIGNING_UNSUPPORTED")
        for step, sign, responses in (
            ("signing", refused, {}),
            ("configmap", None, {
                "create configmap": (1, "", "exists")
            }),
            ("apply", None, {
                "apply": (1, "", "webhook denied")
            }),
        ):
            with self.subTest(step), \
                    patch.object(gcs_staging, "generate_v4_signed_url",
                                 side_effect=sign, return_value="https://u"), \
                    patch.object(gcs_staging,
                                 "delete_staged_prefix") as mock_delete, \
                    tempfile.TemporaryDirectory() as td:
                in_file = Path(td) / "input.dat"
                in_file.write_text("in-data")
                with patch.object(gke_job.subprocess, "run",
                                  FakeKubectl(responses)):
                    with self.assertRaises(Exception):
                        make_driver(staging_bucket="gs://b").submit(
                            str(in_file),
                            job_name="job-1",
                            inputs=[base.StagedFile(str(in_file), "in.dat")],
                        )
                mock_delete.assert_called_once_with("b", "job-1/")

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_run_cleans_up_an_upload_whose_signing_fails(
            self, mock_upload, mock_sign, mock_delete):
        """`run` stages before `submit`, which cleans up only its own."""
        mock_sign.side_effect = ComputeConfigError(
            "cannot sign", error_code="SIGNING_UNSUPPORTED")
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://test-staging-bucket")
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            in_file = Path(td) / "input.dat"
            in_file.write_text("in-data")
            with patch.object(gke_job.subprocess, "run", fake):
                with self.assertRaises(ComputeConfigError):
                    driver.run(
                        str(script_path),
                        inputs=[base.StagedFile(str(in_file), "input.dat")],
                    )
        mock_upload.assert_called_once()
        mock_delete.assert_called_once()
        self.assertEqual(mock_delete.call_args[0][0], "test-staging-bucket")
        self.assertTrue(mock_delete.call_args[0][1].endswith("/"))
        # Only the lookup that checks the name is free; nothing was created.
        self.assertEqual([c for c in fake.calls if "get" not in c], [])

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url",
           return_value="https://u")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_a_name_taken_after_the_lookup_keeps_the_other_jobs_files(
            self, mock_upload, mock_sign, mock_delete):
        """The ConfigMap and staged prefix under the name are another job's."""
        del mock_sign
        fake = FakeKubectl({
            "create configmap": (1, "", 'error: failed to create configmap: '
                                 'configmaps "run-1" already exists')
        })
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "script.py"
            script_path.write_text("print('test')")
            in_file = Path(td) / "input.dat"
            in_file.write_text("in-data")
            with patch.object(gke_job.subprocess, "run", fake):
                with self.assertRaises(ComputeConfigError):
                    make_driver(staging_bucket="gs://b").submit(
                        str(script_path),
                        job_name="run-1",
                        inputs=[base.StagedFile(str(in_file), "input.dat")])
        mock_upload.assert_called_once()
        mock_delete.assert_not_called()
        self.assertIsNone(fake.find("delete configmap"))

    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    def test_cancel_deletes_staged_inputs_only(self, mock_delete):
        """Outputs nobody has fetched must survive a reap; inputs need not."""
        fake = FakeKubectl({"get job": (0, job_json(active=1), "")})
        driver = make_driver(staging_bucket="gs://test-staging-bucket")
        record = jobs.JobRecord.create(
            job_id="job-staged-1",
            backend="gke_job",
            status="RUNNING",
            extra={"staged_prefix": "job-staged-1/"},
        )
        with patch.object(gke_job.subprocess, "run", fake):
            driver.cancel(record.job_id)
        mock_delete.assert_called_once_with("test-staging-bucket",
                                            "job-staged-1/in/")

    def test_serving_job_is_dispatched_with_a_restart_policy(self):
        """A service that a single crash ends is not a service."""
        fake = FakeKubectl()
        driver = make_driver()
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "server.py"
            script_path.write_text("serve()")
            with patch.object(gke_job.subprocess, "run", fake):
                driver.submit(str(script_path), serve=True)
        manifest = fake.applied_manifest()
        self.assertEqual(manifest["spec"]["template"]["spec"]["restartPolicy"],
                         "OnFailure")
        self.assertEqual(manifest["spec"]["backoffLimit"],
                         gke_job.SERVE_BACKOFF_LIMIT)

    def test_batch_job_still_gets_one_attempt(self):
        fake = FakeKubectl()
        driver = make_driver()
        with tempfile.TemporaryDirectory() as td:
            script_path = Path(td) / "batch.py"
            script_path.write_text("train()")
            with patch.object(gke_job.subprocess, "run", fake):
                driver.submit(str(script_path))
        manifest = fake.applied_manifest()
        self.assertEqual(manifest["spec"]["template"]["spec"]["restartPolicy"],
                         "Never")
        self.assertEqual(manifest["spec"]["backoffLimit"], 0)


def _framed(index: int, data: bytes, job: str = "run-1") -> str:
    """An output as the pod's bootstrap prints it into `job`'s log."""
    import base64
    import hashlib
    body = base64.b64encode(data).decode()
    lines = [body[i:i + 76] for i in range(0, len(body), 76)]
    return "\n".join([
        f"{gcs_staging.output_frame(job)} begin {index} "
        f"{hashlib.sha256(data).hexdigest()}", *lines,
        f"{gcs_staging.output_frame(job)} end {index}"
    ]) + "\n"


# The pod's note that output 0 of `run-1` did not fit in its log.
_TOO_LARGE = f"{gcs_staging.output_frame('run-1')} too-large 0 21600000\n"


class TestGkeJobSmallFiles(unittest.TestCase):
    """Inputs in the ConfigMap and outputs in the log: no bucket, no signing."""

    def setUp(self):
        self.enterContext(_job_store())
        # `run` names its Job, and the frames in a scripted log name it too.
        self.enterContext(
            patch.object(gke_job.jobs, "new_job_id", return_value="run-1"))
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.script = self.tmp / "job.py"
        self.script.write_text("print('hi')\n")
        self.config = self.tmp / "config.json"
        self.config.write_text('{"lr": 0.1}')

    def test_small_inputs_ride_in_the_configmap(self):
        fake = FakeKubectl()
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "upload_file_to_gcs") as upload:
            make_driver().submit(
                str(self.script),
                inputs=[base.StagedFile(str(self.config), "cfg/c.json")])
        upload.assert_not_called()
        self.assertIn(f"--from-file=tpu-agent-suite-input-0={self.config}",
                      fake.find("create configmap"))
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        script = workload_command(container)[2]
        self.assertIn("mkdir -p cfg", script)
        self.assertIn("cp /app/tpu-agent-suite-input-0 cfg/c.json", script)

    def test_run_returns_outputs_through_the_log_without_a_bucket(self):
        result_file = self.tmp / "local" / "result.json"
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "step 1\n" + _framed(0, b'{"ok": 1}'), ""),
        })
        with patch.object(gke_job.subprocess, "run", fake):
            res = make_driver().run(
                str(self.script),
                outputs=[
                    base.StagedFile(str(result_file), "result.json"),
                    base.StagedFile(str(self.tmp / "never"), "never.txt"),
                ])
        self.assertEqual(result_file.read_bytes(), b'{"ok": 1}')
        self.assertEqual(res.stdout, "step 1\n")
        self.assertEqual(res.exit_code, 1)
        self.assertIn(
            "'never.txt' was not produced on the target: was not "
            "created", res.stderr)

    def test_outputs_fall_back_to_the_log_when_signing_fails(self):
        fake = FakeKubectl()
        refused = ComputeConfigError("cannot sign",
                                     error_code="SIGNING_UNSUPPORTED")
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "generate_v4_signed_url",
                             side_effect=refused), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            job_id = make_driver(staging_bucket="gs://b").submit(
                str(self.script), outputs=[base.StagedFile("r", "r.json")])
        self.assertIn("through the pod's log", err.getvalue())
        self.assertEqual(
            jobs.JobRecord.load(job_id).extra["outputs_in_log"],
            "SIGNING_UNSUPPORTED")
        container = fake.applied_manifest(
        )["spec"]["template"]["spec"]["containers"][0]
        script = workload_command(container)[2]
        self.assertIn("log_output 0 r.json", script)

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_outputs_fall_back_to_the_log_without_any_credentials(self):
        """A laptop that never ran `application-default login`."""
        from google.cloud import storage  # noqa: PLC0415
        fake = FakeKubectl()
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(storage, "Client", side_effect=(
                    _google_auth_errors.DefaultCredentialsError("no ADC"))), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            job_id = make_driver(staging_bucket="gs://b").submit(
                str(self.script), outputs=[base.StagedFile("r", "r.json")])
        self.assertIn("NO_CREDENTIALS: could not sign upload URLs",
                      err.getvalue())
        self.assertIn("not keeping a GCS copy", err.getvalue())
        self.assertTrue(jobs.JobRecord.load(job_id).extra["outputs_in_log"])

    def _run_losing_an_output(self, log: str, **driver_kw: Any):
        """`run` declaring one output, whose Job's log is `log`."""
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, log, ""),
        })
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "delete_staged_prefix"), \
                contextlib.redirect_stderr(io.StringIO()):
            return make_driver(**driver_kw).run(
                str(self.script),
                outputs=[base.StagedFile(str(self.tmp / "p"), "p.tgz")])

    def test_an_output_too_large_for_the_log_asks_for_a_pod_upload(self):
        """A bucket set, but ADC that cannot sign, lost a 20 MiB
        profile, and the error asked for the bucket that was already set."""
        refused = ComputeConfigError("cannot sign",
                                     error_code="SIGNING_UNSUPPORTED")
        with patch.object(gcs_staging,
                          "generate_v4_signed_url",
                          side_effect=refused):
            res = self._run_losing_an_output(_TOO_LARGE,
                                             staging_bucket="gs://b")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("signing its URL failed (SIGNING_UNSUPPORTED)",
                      res.stderr)
        self.assertEqual(res.details["setup_fields"],
                         list(gcs_staging.POD_UPLOAD_FIELDS))
        # With no bucket, only the bucket: `setup verify` then asks for
        # signing only if this host cannot.
        res = self._run_losing_an_output(_TOO_LARGE)
        self.assertEqual(res.details["setup_fields"], ["staging_bucket"])

    def test_an_output_the_upload_refused_and_the_log_lost_says_so(self):
        with patch.object(gcs_staging,
                          "generate_v4_signed_url",
                          return_value="https://u"):
            res = self._run_losing_an_output(_TOO_LARGE,
                                             staging_bucket="gs://b")
        self.assertIn("after the pod failed to upload it to gs://b",
                      res.stderr)
        self.assertNotIn("setup_fields", res.details)

    def test_artifacts_lost_in_the_log_fail_with_why_at_submit(self):
        """The reason is the one recorded at submit: a bucket set since then
        would not have helped that Job."""
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job",
                              extra={
                                  "outputs": [{
                                      "remote": "p.tgz"
                                  }],
                                  "outputs_in_log": gke_job.NO_BUCKET
                              })
        fake = FakeKubectl({"logs": (0, _TOO_LARGE, "")})
        with patch.object(gke_job.subprocess, "run", fake), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(base.NotConfigured) as ctx:
            make_driver(staging_bucket="gs://b").fetch_artifacts(
                "run-1", self.tmp / "out")
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))
        self.assertIn("no 'staging_bucket' was set", err.getvalue())
        self.assertTrue((self.tmp / "out" / "run-1.log").is_file())

    def test_a_missing_script_is_reported_not_raised_raw(self):
        with self.assertRaises(ComputeConfigError) as ctx:
            make_driver().submit(str(self.tmp / "gone.py"))
        self.assertEqual(ctx.exception.error_code, "INPUT_NOT_FOUND")

    def test_run_takes_an_output_the_bucket_refused_from_the_log(self):
        result_file = self.tmp / "result.json"
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, _framed(0, b"42"), ""),
        })
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "generate_v4_signed_url",
                             return_value="https://u"), \
                patch.object(gcs_staging, "delete_staged_prefix"), \
                patch.object(gcs_staging, "download_file_from_gcs") as get:
            res = make_driver(staging_bucket="gs://b").run(
                str(self.script),
                outputs=[base.StagedFile(str(result_file), "r.json")])
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(result_file.read_bytes(), b"42")
        get.assert_not_called()

    def test_a_failed_submit_touches_no_bucket_it_did_not_upload_to(self):
        fake = FakeKubectl({"create configmap": (1, "", "exists")})
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "delete_staged_prefix") as delete:
            with self.assertRaises(ComputeConfigError):
                make_driver(staging_bucket="gs://b").submit(
                    str(self.script),
                    inputs=[base.StagedFile(str(self.config), "c.json")])
        delete.assert_not_called()

    def test_artifacts_of_a_submitted_job_come_from_its_log(self):
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job",
                              extra={
                                  "outputs": [{
                                      "local": "r.json",
                                      "remote": "eval/r.json"
                                  }],
                                  "outputs_in_log":
                                  True
                              })
        fake = FakeKubectl({"logs": (0, "done\n" + _framed(0, b"42"), "")})
        with patch.object(gke_job.subprocess, "run", fake):
            written = make_driver().fetch_artifacts("run-1", self.tmp / "out")
        self.assertEqual((self.tmp / "out" / "run-1.log").read_text(),
                         "done\n")
        self.assertEqual((self.tmp / "out" / "eval" / "r.json").read_bytes(),
                         b"42")
        self.assertEqual(len(written), 2)
        # Outputs staged to a bucket that is no longer configured.
        jobs.JobRecord.create(job_id="run-2",
                              backend="gke_job",
                              extra={"outputs": [{
                                  "remote": "r.json"
                              }]})
        with patch.object(gke_job.subprocess, "run", fake), \
                self.assertRaises(ComputeConfigError) as ctx:
            make_driver().fetch_artifacts("run-2", self.tmp / "out")
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))

    def test_a_frame_is_accepted_only_whole(self):
        good = _framed(0, b"payload")
        damaged = good.replace(good.splitlines()[1], "AAAA")
        for log, expected in (
            (good, b"payload"),
            (damaged, "arrived damaged in the pod's log"),
            (good.rsplit("\n", 2)[0], "was cut off in the pod's log"),
        ):
            with self.subTest(expected=expected):
                kept, found = gcs_staging.split_logged_outputs(
                    "a\n" + log, "run-1")
                self.assertEqual(kept, "a\n")
                self.assertEqual(found, {0: expected})
        # A frame whose start rotated out of the log says so.
        kept, found = gcs_staging.split_logged_outputs(
            good.split("\n", 1)[1], "run-1")
        self.assertIn("log rotated", found[0])

    def test_a_copy_sent_while_running_survives_what_follows_it(self):
        """Review on #817 (Sandeep): a later copy cut off by a lost node, or
        too large for the log, leaves the last whole one standing; and the
        workload's own short lines (`ok`, `/workspace/out`), printed in the
        middle of a copy, stay in the log and leave the copy whole."""
        marker = gcs_staging.output_frame("run-1")
        whole = b'{"cells": 1}'
        body = base64.b64encode(whole).decode()
        digest = hashlib.sha256(whole).hexdigest()
        sent = (f"\n{marker} begin 0 {digest}\n\n{marker} {body[:8]}\n"
                f"ok\n/workspace/out\n\n{marker} {body[8:]}\n"
                f"\n{marker} end 0\n")
        self.assertEqual(gcs_staging.split_logged_outputs(sent, "run-1"),
                         ("ok\n/workspace/out\n", {
                             0: whole
                         }))
        cut = sent + f"\n{marker} begin 0 {'0' * 64}\n\n{marker} QUJD\n"
        later_too_large = sent + f"{marker} too-large 0 9999999\n"
        for log in (cut, later_too_large):
            self.assertEqual(
                gcs_staging.split_logged_outputs(log, "run-1")[1], {0: whole})

    def test_nothing_else_in_the_log_can_break_a_frame(self):
        good = _framed(0, b"payload")
        # Another run's log, printed by this workload: its frames, and one
        # left open, are the workload's output, not this Job's frames.
        other = (_framed(0, b"old", "run-0") +
                 _framed(1, b"old", "run-0").rsplit("\n", 2)[0])
        log = other + "\n" + good
        self.assertEqual(gcs_staging.split_logged_outputs(log, "run-1"),
                         (other + "\n", {
                             0: b"payload"
                         }))
        # A line written meanwhile, off the frame's stream: kept, not decoded.
        body = good.splitlines(keepends=True)
        stray = "".join(body[:2] + ["curl: (7) refused\n"] + body[2:])
        self.assertEqual(gcs_staging.split_logged_outputs(stray, "run-1"),
                         ("curl: (7) refused\n", {
                             0: b"payload"
                         }))
        # Output that ends without a newline keeps its last line as it was.
        self.assertEqual(
            gcs_staging.split_logged_outputs("done" + good, "run-1"),
            ("done", {
                0: b"payload"
            }))

    @unittest.skipUnless(
        all(shutil.which(t) for t in ("bash", "base64", "sha256sum")),
        "needs bash, base64 and sha256sum")
    def test_bootstrap_round_trips_files_in_bash(self):
        """What the pod runs, run for real: copy in, frame out, parse back."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "tpu-agent-suite-input-0").write_bytes(b"\x00\xffcfg")
        # Output that ends with a newline or without one comes back as the
        # workload printed it, with no line added before the frames.
        for printed in ("line\n", "no newline"):
            with self.subTest(printed=printed):
                (mount / "job.py").write_text(
                    "import os\n"
                    "os.makedirs('eval')\n"
                    "open('eval/r.bin', 'wb').write("
                    "open('c/in', 'rb').read() * 2)\n"
                    "open('big', 'wb').write(b'x' * (2 << 20))\n"
                    f"print({printed!r}, end='')\n"
                    "raise SystemExit(3)\n")
                work = self.tmp / "w"
                shutil.rmtree(work, ignore_errors=True)
                # No copies while it runs: one would end the unfinished line.
                with patch.object(gcs_staging, "POD_WORK_DIR", str(work)), \
                        patch.object(gcs_staging, "LIVE_OUTPUT_BUDGET_BYTES", 0):
                    command = gcs_staging.build_pod_bootstrap_command(
                        script_mount=str(mount),
                        script_name="job.py",
                        interpreter=sys.executable,
                        mounted_inputs=[("c/in", "tpu-agent-suite-input-0")],
                        logged_outputs=["eval/r.bin", "big"],
                        job="run-1",
                    )
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 3)
                log, found = gcs_staging.split_logged_outputs(
                    proc.stdout, "run-1")
                self.assertEqual(log, printed)
                self.assertEqual(found[0], b"\x00\xffcfg" * 2)
                self.assertIn("more than the pod's log carries", found[1])

    @unittest.skipUnless(
        all(shutil.which(t) for t in ("bash", "base64", "sha256sum", "stat")),
        "needs bash, base64, sha256sum and stat")
    def test_bootstrap_sends_outputs_while_the_workload_runs(self):
        """A pod lost mid-run (its node reset: everything killed at once)
        still leaves the output as it last was in its log, even with the
        workload printing half lines all the while."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text(
            "import os, sys, time\n"
            "def save(text):\n"
            "    open('s.tmp', 'w').write(text)\n"
            "    os.replace('s.tmp', 'summary.json')\n"
            "save('{\"cells\": 1}')\n"
            "end = time.time() + 2.5\n"
            "while time.time() < end:\n"
            "    sys.stderr.write('\\r 10%|##')\n"
            "    print('ok', flush=True)\n"
            "    sys.stderr.flush()\n"
            "    time.sleep(0.01)\n"
            "save('{\"cells\": 2}')\n"
            "time.sleep(60)\n")
        with patch.object(gcs_staging, "POD_WORK_DIR", str(self.tmp / "w")), \
                patch.object(gcs_staging, "LIVE_OUTPUT_INTERVAL_SEC", 1):
            command = gcs_staging.build_pod_bootstrap_command(
                script_mount=str(mount),
                script_name="job.py",
                interpreter=sys.executable,
                logged_outputs=["summary.json"],
                log_url="http://127.0.0.1:9/log",
                job="run-1",
            )
        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                text=True,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
        log, found = gcs_staging.split_logged_outputs(out, "run-1")
        self.assertEqual(found, {0: b'{"cells": 2}'})
        self.assertIn(" 10%|##", log)
        self.assertIn("ok\n", log)

    @unittest.skipUnless(
        all(shutil.which(t) for t in ("bash", "base64", "sha256sum")),
        "needs bash, base64 and sha256sum")
    def test_bootstrap_sends_outputs_when_the_pod_is_stopped(self):
        """SIGTERM (a timeout, a preemption) reaches the workload, and its
        outputs come back before the grace period's SIGKILL."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text("import time\n"
                                      "open('r.txt', 'w').write('kept')\n"
                                      "print('started', flush=True)\n"
                                      "time.sleep(60)\n")
        with patch.object(gcs_staging, "POD_WORK_DIR", str(self.tmp / "w")):
            command = gcs_staging.build_pod_bootstrap_command(
                script_mount=str(mount),
                script_name="job.py",
                interpreter=sys.executable,
                logged_outputs=["r.txt"],
                job="run-1",
            )
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        self.assertEqual(proc.stdout.readline(), "started\n")
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=10)
        # The workload's own exit: killed by the signal.
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM)
        self.assertEqual(
            gcs_staging.split_logged_outputs(out, "run-1")[1], {0: b"kept"})

    @unittest.skipUnless(
        all(
            shutil.which(t)
            for t in ("bash", "base64", "sha256sum", "curl", "sed")),
        "needs bash, base64, sha256sum, curl and sed")
    def test_bootstrap_frames_an_output_the_bucket_refuses(self):
        """An upload that fails comes back in the log and its GCS copy; so
        does one with no Workload Identity token to send."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text("open('r.bin', 'wb').write(b'res')\n")
        refused = "http://127.0.0.1:9/refused"  # Nothing listens on port 9.
        for pod_token in (False, True):
            with self.subTest(pod_token=pod_token):
                work = self.tmp / f"w-{pod_token}"
                with patch.object(gcs_staging, "POD_WORK_DIR", str(work)), \
                        patch.object(gcs_staging, "POD_TOKEN_URL", refused):
                    command = gcs_staging.build_pod_bootstrap_command(
                        script_mount=str(mount),
                        script_name="job.py",
                        interpreter=sys.executable,
                        output_urls=[("r.bin", refused)],
                        log_url=refused,
                        job="run-1",
                        pod_token=pod_token,
                    )
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout)
                # curl's error and the notice share the frame's stream.
                log, found = gcs_staging.split_logged_outputs(
                    proc.stdout, "run-1")
                self.assertEqual(found, {0: b"res"})
                self.assertIn("returning it in the log", log)
                self.assertEqual(
                    "The GKE metadata server returned no Google token (curl: "
                    in log, pod_token)
                self.assertNotIn("--workload-metadata", log)
                self.assertEqual(proc.stderr, "")
                log_copy = (work / gcs_staging.POD_LOG_FILE).read_text()
                self.assertEqual(
                    gcs_staging.split_logged_outputs(log_copy, "run-1")[1],
                    {0: b"res"})

    @unittest.skipUnless(shutil.which("curl"), "needs curl")
    def test_bootstrap_explains_a_transfer_gcs_refuses(self):
        """A token GCS refuses: name the object and the missing grant."""

        class Gcs(http.server.BaseHTTPRequestHandler):

            def log_message(self, *args):
                del args

            def do_GET(self):
                token = self.path == "/token"
                self.send_response(200 if token else 403)
                self.end_headers()
                if token:
                    self.wfile.write(b'{"access_token": "t"}')

            do_PUT = do_GET

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gcs)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text("open('r.bin', 'w').write('res')\n")
        for what, urls, code in (
            ("download", {
                "input_urls": [("in", f"{endpoint}/b/in")]
            }, 22),
            ("upload", {
                "output_urls": [("r.bin", f"{endpoint}/b/r.bin")]
            }, 0),
        ):
            with self.subTest(what=what), \
                    patch.object(gcs_staging, "POD_WORK_DIR",
                                 str(self.tmp / what)), \
                    patch.object(gcs_staging, "POD_TOKEN_URL",
                                 f"{endpoint}/token"), \
                    patch.object(gcs_staging, "GCS_OBJECT_ENDPOINT", endpoint):
                command = gcs_staging.build_pod_bootstrap_command(
                    script_mount=str(mount),
                    script_name="job.py",
                    interpreter=sys.executable,
                    job="j",
                    pod_token=True,
                    service_account="ksa",
                    namespace="ns",
                    bucket="b",
                    **urls)
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, code, proc.stdout)
                self.assertIn(f"GCS refused to {what} gs://b/", proc.stdout)
                self.assertIn(
                    "Grant roles/storage.objectAdmin on gs://b to Kubernetes "
                    "service account 'ksa' in namespace 'ns'", proc.stdout)


class TestGcsStagingBucket(unittest.TestCase):

    def test_a_bucket_may_name_a_folder_in_it(self):
        client = MagicMock()
        with tempfile.NamedTemporaryFile() as f:
            gcs_staging.upload_file_to_gcs(f.name,
                                           "gs://b/runs/",
                                           "j/in/x",
                                           client=client)
        client.bucket.assert_called_once_with("b")
        client.bucket.return_value.blob.assert_called_once_with("runs/j/in/x")

    def test_a_refused_request_is_a_config_error(self):

        class GoogleAPICallError(Exception):
            code, message = 403, "storage.objects.create denied"

        exceptions = types.SimpleNamespace(
            GoogleAPICallError=GoogleAPICallError)
        api_core = types.SimpleNamespace(exceptions=exceptions)
        client = MagicMock()
        client.bucket.return_value.blob.return_value.upload_from_filename \
            .side_effect = GoogleAPICallError()
        with tempfile.NamedTemporaryFile() as f, patch.dict(
                sys.modules, {
                    "google": types.SimpleNamespace(api_core=api_core),
                    "google.api_core": api_core,
                }):
            with self.assertRaises(ComputeConfigError) as ctx:
                gcs_staging.upload_file_to_gcs(f.name, "b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "GCS_FORBIDDEN")
        self.assertIn("storage.objects.create denied", ctx.exception.message)


class TestGkeJobWorkloadIdentity(unittest.TestCase):
    """With `pod_service_account`, the pod moves staged files
    with its own Workload Identity token, so nothing has to be signed."""

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    def test_submit_runs_the_pod_as_the_account_and_signs_nothing(self):
        self.enterContext(_job_store())
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = tmp / "job.py"
        script.write_text("print('x')\n")
        data = tmp / "in.bin"
        data.write_bytes(b"in")
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://b/runs",
                             pod_service_account="ksa")
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gcs_staging, "upload_file_to_gcs") as upload, \
                patch.object(gcs_staging, "generate_v4_signed_url",
                             side_effect=AssertionError("signed")):
            job = driver.submit(
                str(script),
                inputs=[base.StagedFile(str(data), "a b.bin")],
                outputs=[base.StagedFile(str(tmp / "o.bin"), "o.bin")])
        upload.assert_called_once()
        spec = fake.applied_manifest()["spec"]["template"]["spec"]
        self.assertEqual(spec["serviceAccountName"], "ksa")
        self.assertEqual(spec["nodeSelector"][gke_job.WORKLOAD_IDENTITY_LABEL],
                         "true")
        pod_script = workload_command(spec["containers"][0])[2]
        # Plain object URLs under the bucket's folder, names URL-encoded.
        prefix = f"https://storage.googleapis.com/b/runs/{job}"
        for url in (f"{prefix}/in/a%20b.bin", f"{prefix}/out/o.bin",
                    f"{prefix}/out/{gcs_staging.POD_LOG_FILE}"):
            self.assertIn(url, pod_script)
        # The input, the output (as it changes, and at the end) and the log
        # copy each send a fresh token.
        self.assertEqual(
            pod_script.count('-H "Authorization: Bearer $(gcs_token)"'), 4)


class TestGkeJobPodToken(unittest.TestCase):
    """With a staging bucket and no pod_service_account, the pod gets a GCS
    token limited to the staging bucket, in a Secret the Job owns."""

    TOKEN = "fake-pod-token-value"

    def setUp(self):
        self.enterContext(_job_store())

    def submit(self, **params):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = tmp / "job.py"
        script.write_text("print('x')\n")
        fake = FakeKubectl()
        with patch.object(gke_job.subprocess, "run", fake), \
                patch.object(gke_job.gcs_staging, "mint_bucket_token",
                             return_value=(self.TOKEN, time.time() + 3600)
                             ) as mint, \
                contextlib.redirect_stderr(io.StringIO()):
            job = make_driver(**params).submit(str(script))
        applied = {
            o["kind"]: o
            for o in (json.loads(s) for c, s in zip(fake.calls, fake.stdins)
                      if "apply" in c and s)
        }
        return job, applied, mint

    def test_the_pod_gets_a_token_scoped_to_the_whole_bucket(self):
        # The whole bucket, not just the staging folder: fetch_weights.py
        # reads checkpoints staged elsewhere in it.
        _, applied, mint = self.submit(staging_bucket="gs://b/runs")
        mint.assert_called_once_with("b")
        self.assertEqual(applied["Secret"]["stringData"],
                         {"token": self.TOKEN})
        self.assertNotIn(self.TOKEN, json.dumps(applied["Job"]))
        c = applied["Job"]["spec"]["template"]["spec"]["containers"][0]
        self.assertIn(
            {
                "name": "GCS_TOKEN_FILE",
                "value": f"{gke_job.TOKEN_MOUNT}/token"
            }, c["env"])
        # subPath mounts don't pick up renewed tokens.
        mount = next(m for m in c["volumeMounts"] if m["name"] == "gcs-token")
        self.assertNotIn("subPath", mount)


if __name__ == "__main__":
    unittest.main()
