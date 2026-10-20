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
"""Unit tests for the framework compute runtime layer.

Test Coverage Overview:
- TestDriverContracts / TestDriverRegistry: the data contracts, the abstract
  driver interface, plugin registration, and driver auto-discovery.
- TestLocalDriver: host probing and the local job lifecycle.
- TestRemoteShellDrivers: the lifecycle GceTpuVmDriver and GkePodDriver share,
  exercised once against both transports, plus each transport's own commands.
- TestGkeXpkDriver: cluster status vocabulary, occupancy, and GCS retrieval.
- TestComputeConfig / TestEnvironmentResolution: 3-tier configuration
  resolution, environment.json schemas, and CLI override precedence.
- TestComputeCli / TestContractGeneration: `runtime.py compute` behaviour,
  output envelopes, and the generated agent contract.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Allow imports both from within the repository root and standalone
_framework_dir = pathlib.Path(__file__).resolve().parents[1]
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
_repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from framework import compiler, compute, runtime, utils
    from framework.compute import (DRIVER_REGISTRY,
                                   AsynchronousExecutionResult,
                                   ComputeConfigError, ComputeDriver,
                                   GceTpuVmDriver, GkeJobDriver, GkePodDriver,
                                   GkeXpkDriver, HardwareProfile, LocalDriver,
                                   StagedFile, SynchronousExecutionResult)
    from framework.compute import config as compute_config
    from framework.compute import (get_driver, get_registered_driver_cli_flags,
                                   get_registered_drivers, list_drivers,
                                   register_driver, resolve_driver)
    from framework.compute.drivers import (_auto_discover_drivers, auth_utils,
                                           base, gke_pod, jobs, probe)
    from framework.compute.drivers import registry as driver_registry
    from framework.compute.drivers import shell
except ImportError:
    import compiler
    import compute
    import runtime
    import utils
    from compute import (DRIVER_REGISTRY, AsynchronousExecutionResult,
                         ComputeConfigError, ComputeDriver, GceTpuVmDriver,
                         GkeJobDriver, GkePodDriver, GkeXpkDriver,
                         HardwareProfile, LocalDriver, StagedFile,
                         SynchronousExecutionResult)
    from compute import config as compute_config
    from compute import (get_driver, get_registered_driver_cli_flags,
                         get_registered_drivers, list_drivers, register_driver,
                         resolve_driver)
    from compute.drivers import (_auto_discover_drivers, auth_utils, base,
                                 gke_pod, jobs, probe)
    from compute.drivers import registry as driver_registry
    from compute.drivers import shell

Path = pathlib.Path

# The compute CLI falls back to the session leased to the harness process
# above it. Run from inside an agent session, that would be the developer's
# own run; tests that exercise the fallback patch it back in themselves.
_NO_SESSION = patch.object(compute.cli.lease, "session_run", return_value=None)

# Nor may a test reach the developer's real state some other way. Jobs and runs
# are recorded in the central store, and a command given no workspace finds one
# by climbing from the current directory, which in a checkout reaches its
# active run. Both point into a temp directory for the whole module; a test that
# needs its own store still patches `TPU_AGENT_SUITE_HOME` over this one.
_ISOLATION = contextlib.ExitStack()


def setUpModule():  # noqa: N802 - unittest's spelling
    _NO_SESSION.start()
    scratch = Path(tempfile.mkdtemp())
    _ISOLATION.callback(shutil.rmtree, scratch, ignore_errors=True)
    _ISOLATION.enter_context(
        patch.dict(os.environ,
                   {"TPU_AGENT_SUITE_HOME": str(scratch / "store")}))
    # chdir rather than patching Path.cwd, so that subprocesses the tests start
    # resolve their workspace inside the temp directory too.
    _ISOLATION.callback(os.chdir, os.getcwd())
    os.chdir(scratch)


def tearDownModule():  # noqa: N802 - unittest's spelling
    _ISOLATION.close()
    _NO_SESSION.stop()


def profile_env(backend: str, **params: Any) -> dict[str, Any]:
    """An environment.json holding one profile, `default`, the default."""
    profile = {"backend": backend, **({"params": params} if params else {})}
    return {"default": "default", "profiles": {"default": profile}}


def _decode_inner_script(ssh_command: str) -> str:
    """Pull the workload back out of the base64 payload `run` stages.

    `run` no longer passes the command inline: it writes a script to the
    target and runs that, so the command text a test wants to assert on is
    inside the base64 blob rather than in the SSH invocation.
    """
    blob = re.search(r"printf %s '([A-Za-z0-9+/=]+)'", ssh_command)
    assert blob, f"no staged script found in: {ssh_command}"
    return base64.b64decode(blob.group(1)).decode("utf-8")


def _completed(stdout: str = "",
               returncode: int = 0,
               stderr: str = "") -> subprocess.CompletedProcess:
    """Build a CompletedProcess stand-in for a mocked subprocess call."""
    return subprocess.CompletedProcess(args=[],
                                       returncode=returncode,
                                       stdout=stdout,
                                       stderr=stderr)


class _ScriptMixin:
    """Writes throwaway scripts that are removed when the test finishes."""

    def _script(self, body: str, suffix: str = ".py") -> str:
        with tempfile.NamedTemporaryFile("w", suffix=suffix,
                                         delete=False) as handle:
            handle.write(body)
            path = handle.name
        self.addCleanup(os.remove, path)
        return path


@register_driver("mock_test")
class MockTestDriver(ComputeDriver):
    """Mock compute driver used to verify base framework contracts and CLI routing."""

    def __init__(
        self,
        endpoint: str = "default-endpoint",
        token: str = "default-token",
        python_interpreter: str = "python3",
    ):
        self.endpoint = endpoint
        self.token = token
        self.python_interpreter = python_interpreter

    @classmethod
    def from_params(cls, params: dict) -> MockTestDriver:
        endpoint = params.get("endpoint", "default-endpoint")
        return cls(
            endpoint=endpoint,
            token=params.get("token", "default-token"),
            python_interpreter=params.get("python_interpreter", "python3"),
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        cls.add_cli_arg(parser, "--endpoint", type=str, help="Mock endpoint")
        cls.add_cli_arg(parser, "--token", type=str, help="Mock token")

    def inspect(self) -> HardwareProfile:
        return HardwareProfile(
            accelerator="mock_tpu",
            topology="2x2x1",
            chips=4,
            hbm_gb=64.0,
            status="healthy",
            available_chips=4,
            held_by=[],
        )

    def run(self, code_path: str, *args,
            **kwargs) -> SynchronousExecutionResult:
        return SynchronousExecutionResult(
            exit_code=0,
            stdout=f"mock run executed {code_path}",
            stderr="",
        )

    def submit(self, recipe_path: str, *args, **kwargs) -> str:
        return "mock-job-999"

    def get_status(self, job_id: str) -> AsynchronousExecutionResult:
        return AsynchronousExecutionResult(job_id=job_id, status="RUNNING")

    def fetch_logs(self, job_id: str) -> str:
        return f"mock logs for {job_id}"

    def fetch_artifacts(self, job_id: str, dest_dir: str) -> list[str]:
        return [str(Path(dest_dir) / "output.txt")]

    def cancel(self, job_id: str) -> None:
        pass


class TestDriverContracts(unittest.TestCase):
    """The dataclasses drivers return, and the abstractness of the interface."""

    def test_execution_result_defaults(self):
        """Verify execution results keep unknown fields unset rather than zeroed."""
        sync = SynchronousExecutionResult(exit_code=0,
                                          stdout="out",
                                          stderr="err")
        self.assertEqual((sync.exit_code, sync.stdout, sync.stderr),
                         (0, "out", "err"))

        job = AsynchronousExecutionResult(job_id="job-1", status="PENDING")
        self.assertEqual(job.job_id, "job-1")
        self.assertIsNone(job.exit_code)
        self.assertIsNone(job.start_time)
        self.assertIsNone(job.end_time)

    def test_availability_preserves_none_and_zero(self):
        """Verify unmeasured (None) vs measured saturated (0) availability distinctions are preserved."""
        unmeasured = HardwareProfile(accelerator="v4",
                                     topology="2x2x1",
                                     chips=4,
                                     hbm_gb=32.0,
                                     status="healthy")
        self.assertIsNone(unmeasured.available_chips)
        self.assertIsNone(unmeasured.to_dict()["available_chips"])
        self.assertEqual(unmeasured.held_by, [])

        saturated = HardwareProfile(accelerator="v6e",
                                    topology="2x2x1",
                                    chips=4,
                                    hbm_gb=32.0,
                                    status="busy",
                                    available_chips=0,
                                    held_by=["PID 1 on /dev/accel0"])
        self.assertEqual(saturated.to_dict()["available_chips"], 0)

    def test_compute_driver_is_abstract(self):
        """Verify ComputeDriver cannot be instantiated without its methods."""
        with self.assertRaises(TypeError):
            ComputeDriver()  # type: ignore[abstract]


class TestDriverRegistry(unittest.TestCase):
    """Driver registration, discovery, and CLI flag declaration."""

    def test_driver_registration_and_lookup(self):
        """Verify the register_driver decorator, list_drivers, and valid backends."""
        self.assertIn("mock_test", list_drivers())
        self.assertIs(get_driver("mock_test"), MockTestDriver)
        self.assertIn("mock_test", get_registered_drivers())
        self.assertEqual(set(compute_config.get_valid_backends()),
                         set(list_drivers()))

    def test_registered_backends_are_declared_in_schema(self):
        """Verify every shipped driver is a documented environment.json backend."""
        schema_path = (pathlib.Path(__file__).resolve().parent.parent.parent /
                       "package" / "schemas" / "environment.json")
        if not schema_path.is_file():
            self.skipTest("package schema not available in this layout")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        compute = schema["properties"]["profiles"]["additionalProperties"][
            "properties"]
        declared = set(compute["backend"]["enum"])
        shipped = {n for n in list_drivers() if not n.startswith("mock")}
        self.assertTrue(
            shipped <= declared,
            f"backends missing from environment.json schema: {shipped - declared}"
        )

    def test_declared_flags_are_collected(self):
        """Verify flags a driver declares are recorded for parameter extraction."""
        MockTestDriver.register_cli_args(argparse.ArgumentParser())
        flags = get_registered_driver_cli_flags()
        self.assertIn("endpoint", flags)
        self.assertIn("token", flags)

    def test_add_cli_arg_enforces_rules(self):
        """Verify driver defaults are stripped and reserved names are rejected."""
        parser = argparse.ArgumentParser()

        class _DefaultingDriver(MockTestDriver):

            @classmethod
            def register_cli_args(cls, target: Any) -> None:
                cls.add_cli_arg(target,
                                "--zone-probe",
                                default="us-central2-b",
                                help="zone")

        self.addCleanup(driver_registry.REGISTERED_DRIVER_CLI_FLAGS.discard,
                        "zone_probe")
        _DefaultingDriver.register_cli_args(parser)
        self.assertFalse(hasattr(parser.parse_args([]), "zone_probe"))

        with self.assertRaises(ValueError) as ctx:
            MockTestDriver.add_cli_arg(argparse.ArgumentParser(),
                                       "--workspace",
                                       help="clash")
        self.assertIn("reserved", str(ctx.exception))
        self.assertNotIn("workspace", get_registered_driver_cli_flags())

    def test_auto_discovery_handles_errors_and_subpackages(self):
        """Verify auto-discovery records import errors and discovers subpackages."""
        self.addCleanup(driver_registry.DISCOVERY_ERRORS.pop, "zz_broken",
                        None)
        with patch("pkgutil.iter_modules",
                   return_value=[(None, "zz_broken", False)]), \
             patch("importlib.import_module",
                   side_effect=RuntimeError("probe boom")):
            _auto_discover_drivers()
        errors = driver_registry.get_discovery_errors()
        self.assertIn("probe boom", errors["zz_broken"])
        self.assertIn("zz_broken", driver_registry.describe_discovery_errors())

        with patch("pkgutil.iter_modules",
                   return_value=[(None, "zz_probe_pkg", True)]), \
             patch("importlib.import_module") as mock_import:
            _auto_discover_drivers()
        parent_pkg = driver_registry.__name__.rsplit(".", 1)[0]
        mock_import.assert_called_with(f"{parent_pkg}.zz_probe_pkg")

    def test_invalid_backend_error_reports_discovery_failures(self):
        """Verify a failed driver import is named in the INVALID_BACKEND message.

        Otherwise a driver with a missing dependency vanishes silently and the
        user is told only that their backend is 'invalid'.
        """
        driver_registry.DISCOVERY_ERRORS["ghost_backend"] = (
            "ModuleNotFoundError: No module named 'internal_dep'")
        self.addCleanup(driver_registry.DISCOVERY_ERRORS.pop, "ghost_backend",
                        None)
        with self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver(backend_override="ghost_backend")
        self.assertEqual(ctx.exception.error_code, "INVALID_BACKEND")
        self.assertIn("ghost_backend", ctx.exception.message)
        self.assertIn("internal_dep", ctx.exception.message)

    def test_dynamic_plugin_registration(self):
        """Verify an out-of-tree driver resolves and enforces its own parameters."""

        @register_driver("mock_plugin")
        class MockPluginDriver(MockTestDriver):

            def __init__(self, custom_val: str = ""):
                super().__init__()
                self.custom_val = custom_val

            @classmethod
            def from_params(cls, params):
                val = params.get("custom_val")
                if not val:
                    raise ComputeConfigError("mock_plugin requires custom_val",
                                             error_code="MISSING_FIELD")
                return cls(custom_val=val)

        self.addCleanup(DRIVER_REGISTRY.pop, "mock_plugin", None)
        self.assertIn("mock_plugin", compute_config.get_valid_backends())

        driver = resolve_driver(backend_override="mock_plugin",
                                param_overrides={"custom_val": "test_123"})
        self.assertIsInstance(driver, MockPluginDriver)
        self.assertEqual(driver.custom_val, "test_123")

        with self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver(backend_override="mock_plugin", param_overrides={})
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")


class TestHostProbe(unittest.TestCase):
    """The host probe LocalDriver runs in-process and remote drivers ship out."""

    def setUp(self):
        self.driver = LocalDriver(python_interpreter=sys.executable)

    @contextlib.contextmanager
    def _probed(self,
                accel=(),
                vfio=(),
                pci=(0, None),
                meta=None,
                holders=([], set(), set()),
                locked=False):
        """Patch every input the host probe reads.

        The probe consults metadata, device nodes, PCI, and /proc. A test that
        stubs only some of those inherits the real machine for the rest, which
        is how these tests used to pass in CI and fail on a TPU host.
        """
        patches = (
            patch.object(probe, "read_metadata", return_value=meta or {}),
            patch.object(probe,
                         "list_devices",
                         return_value=(list(accel), list(vfio))),
            patch.object(probe, "scan_pci", return_value=pci),
            patch.object(probe, "scan_holders", return_value=holders),
            patch.object(probe, "_lockfile_is_locked", return_value=locked),
        )
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            yield

    def _holders(self, devices):
        """Build a scan_holders return value for the given held devices."""
        held = set(devices)
        return ([f"PID 42 (trainer) on {d}"
                 for d in sorted(held)], held, set())

    def test_no_accelerator(self):
        """Verify a host with no devices reports 'none' and no capacity."""
        with self._probed():
            profile = self.driver.inspect()
        self.assertEqual(profile.accelerator, "none")
        self.assertEqual(profile.chips, 0)
        self.assertEqual(profile.status, "no_accelerator")

    def test_accelerator_identification(self):
        """Verify generation, chip count, topology, and HBM across device styles.

        tpu7x exposes two devices per chip, so its chip count is half the device
        count; GKE describes the same hardware through kube-label metadata
        rather than an ACCELERATOR_TYPE, and both must land on the same answer.
        """
        cases = [
            (
                "v6e accel nodes",
                dict(accel=["/dev/accel0", "/dev/accel1"],
                     meta={"ACCELERATOR_TYPE": "v6e-2"}),
                dict(accelerator="v6e", chips=2, status="healthy"),
            ),
            (
                "tpu7x halves devices per chip",
                dict(vfio=[f"/dev/vfio/{i}" for i in range(8)],
                     meta={
                         "ACCELERATOR_TYPE": "tpu7x-8",
                         "TOPOLOGY": "2x2x1"
                     }),
                dict(accelerator="tpu7x",
                     chips=4,
                     topology="2x2x1",
                     hbm_gb=192.0,
                     available_chips=4),
            ),
            (
                "gke kube-label metadata",
                dict(vfio=[f"/dev/vfio/{i}" for i in range(8)],
                     meta={
                         "cloud.google.com/gke-tpu-accelerator":
                         "tpu-v6e-slice",
                         "cloud.google.com/gke-tpu-topology": "2x4",
                         "cloud.google.com/machine-family": "ct6e",
                     }),
                dict(accelerator="v6e",
                     chips=8,
                     topology="2x4",
                     hbm_gb=32.0,
                     available_chips=8),
            ),
            (
                "unrecognised chip count is not guessed at",
                dict(accel=[f"/dev/accel{i}" for i in range(6)],
                     meta={"ACCELERATOR_TYPE": "v6e-6"}),
                dict(chips=6, topology="unknown"),
            ),
        ]
        for name, probe_args, expected in cases:
            with self.subTest(name):
                with self._probed(**probe_args):
                    profile = self.driver.inspect()
                for field, value in expected.items():
                    self.assertEqual(getattr(profile, field), value, field)

    def test_contention_counts_chips_not_devices(self):
        """Verify availability subtracts held chips, converting devices first.

        Four held tpu7x devices are two physical chips, not four; a lockfile
        held by libtpu means nothing is usable however free the devices look.
        """
        cases = [
            (
                "some accel nodes held",
                dict(accel=["/dev/accel0", "/dev/accel1"],
                     meta={"ACCELERATOR_TYPE": "v6e-2"},
                     holders=self._holders(["/dev/accel0"])),
                dict(chips=2, available_chips=1, status="partially_busy"),
            ),
            (
                "all accel nodes held",
                dict(accel=["/dev/accel0", "/dev/accel1"],
                     meta={"ACCELERATOR_TYPE": "v6e-2"},
                     holders=self._holders(["/dev/accel0", "/dev/accel1"])),
                dict(chips=2, available_chips=0, status="busy"),
            ),
            (
                "tpu7x devices convert to chips",
                dict(vfio=[f"/dev/vfio/{i}" for i in range(8)],
                     meta={"ACCELERATOR_TYPE": "tpu7x-8"},
                     holders=self._holders(
                         [f"/dev/vfio/{i}" for i in range(4)])),
                dict(chips=4, available_chips=2, status="partially_busy"),
            ),
            (
                "held lockfile overrides free devices",
                dict(accel=["/dev/accel0", "/dev/accel1"],
                     meta={"ACCELERATOR_TYPE": "v6e-2"},
                     locked=True),
                dict(available_chips=0, status="busy"),
            ),
        ]
        for name, probe_args, expected in cases:
            with self.subTest(name):
                with self._probed(**probe_args):
                    profile = self.driver.inspect()
                for field, value in expected.items():
                    self.assertEqual(getattr(profile, field), value, field)

    def test_held_devices_are_named_in_the_profile(self):
        """Verify the holder descriptions reach the caller, not just a count."""
        with self._probed(accel=["/dev/accel0", "/dev/accel1"],
                          meta={"ACCELERATOR_TYPE": "v6e-2"},
                          holders=self._holders(["/dev/accel0"])):
            profile = self.driver.inspect()
        self.assertEqual(profile.held_by, ["PID 42 (trainer) on /dev/accel0"])

    def test_devices_and_hbm_per_device_are_reported(self):
        """Verify TP inputs match gke_job's keys, and unknown chips say why."""
        cases = [
            ("v6e is one device per chip",
             dict(accel=["/dev/accel0"], meta={"ACCELERATOR_TYPE": "v6e-1"}),
             dict(job_devices=1, hbm_gb_per_device=32.0)),
            ("tpu7x chips are two devices each",
             dict(vfio=[f"/dev/vfio/{i}" for i in range(8)],
                  meta={"ACCELERATOR_TYPE":
                        "tpu7x-4"}), dict(job_devices=8,
                                          hbm_gb_per_device=96.0)),
        ]
        for name, probe_args, expected in cases:
            with self.subTest(name):
                with self._probed(**probe_args):
                    details = self.driver.inspect().details
                for field, value in expected.items():
                    self.assertEqual(details[field], value, field)
                self.assertNotIn("hbm_gb_note", details)

        with self._probed(accel=["/dev/accel0"],
                          meta={"ACCELERATOR_TYPE": "tpu9z-1"}):
            details = self.driver.inspect().details
        self.assertNotIn("job_devices", details)
        self.assertNotIn("hbm_gb_per_device", details)
        self.assertIn("tpu9z", details["hbm_gb_note"])

    def test_generation_lookup_normalizes_vendor_names(self):
        """Verify PCI ids and hyphenated product names resolve to a generation."""
        self.assertEqual(probe.generation_for_pci_id("0x006f"), "v6e")
        spec = probe.spec_for("v6e")
        self.assertEqual(spec["devices_per_chip"], 1)
        self.assertEqual(spec["hbm_gb"], 32.0)
        self.assertEqual(probe.generation_for_name("tpu-v5-lite-podslice"),
                         "v5e")
        self.assertEqual(probe.generation_for_name("tpu-v6e-slice"), "v6e")
        self.assertEqual(probe.generation_for_name("ct6e-standard-8t"), "v6e")

    def test_missing_tpu_info_raises_runtime_error(self):
        """Import must survive a missing tpu_info; only probing itself raises.

        The compute CLI import chain pulls this module in on controller
        machines that never probe local hardware; an import-time raise there
        takes down every runtime command, including hook dispatch.
        """
        with patch.dict(sys.modules, {
                "tpu_info": None,
                "tpu_info.device": None
        }):
            importlib.reload(probe)  # must not raise
            with self.assertRaises(probe.MissingPackageError) as ctx:
                probe.spec_for("v6e")
            self.assertEqual(ctx.exception.package, "tpu_info")
            self.assertEqual(
                str(ctx.exception),
                f"tpu_info is not installed for {sys.executable}")
        importlib.reload(probe)


def _terminate_quietly(pid: int) -> None:
    """Stop a test-spawned process group without failing if it already exited."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


class TestLocalDriver(_ScriptMixin, unittest.TestCase):
    """LocalDriver's synchronous and detached execution on this host."""

    def setUp(self):
        self.driver = LocalDriver(python_interpreter=sys.executable)

    def test_shell_script_gets_the_interpreter_as_tpu_agent_suite_python(self):
        """Verify a .sh runs under bash with TPU_AGENT_SUITE_PYTHON set."""
        result = self.driver.run(
            self._script('echo "py=$TPU_AGENT_SUITE_PYTHON"\n', suffix=".sh"))
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertIn(f"py={sys.executable}", result.stdout)

    def _await_terminal(self, job_id: str, timeout: float = 5.0):
        """Poll a submitted job until it reaches a terminal state."""
        deadline = time.monotonic() + timeout
        status = self.driver.get_status(job_id)
        while status.status not in ("COMPLETED",
                                    "FAILED") and time.monotonic() < deadline:
            time.sleep(0.05)
            status = self.driver.get_status(job_id)
        record = jobs.JobRecord.load(job_id)
        if record and record.pid:
            _terminate_quietly(record.pid)
        return status

    def test_interpreter_path_expands_home(self):
        """Verify a `~/...` interpreter from environment.json is usable locally.

        The command is exec'd as an argv list with no shell in between, so
        without expansion the tilde would be looked up as a literal directory.
        """
        with tempfile.TemporaryDirectory() as fake_home:
            os.symlink(sys.executable, os.path.join(fake_home, "py"))
            with patch.dict(os.environ, {"HOME": fake_home}):
                driver = LocalDriver(python_interpreter="~/py")
                res = driver.run(
                    self._script("import sys; print(sys.executable)\n"))
            # Inside the block: once the temp dir is gone, realpath can no
            # longer follow the `py` symlink.
            self.assertEqual(res.exit_code, 0, res.stderr)
            self.assertEqual(os.path.realpath(res.stdout.strip()),
                             os.path.realpath(os.path.join(fake_home, "py")))

    def test_declared_outputs_survive_a_missing_artifacts_dir(self):
        """Outputs sit beside `artifacts/`, so its absence must not hide them."""
        with tempfile.TemporaryDirectory() as home, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": home}):
            record = jobs.JobRecord.create(job_id="job-out-1",
                                           backend="local",
                                           extra={
                                               "outputs": [{
                                                   "local":
                                                   "results.json",
                                                   "remote":
                                                   "results.json",
                                               }]
                                           })
            (record.directory / "results.json").write_text('{"n": 42}')
            shutil.rmtree(record.directory / "artifacts")
            with tempfile.TemporaryDirectory() as dest:
                files = self.driver.fetch_artifacts("job-out-1", Path(dest))
                self.assertEqual([f.name for f in files], ["results.json"])
                self.assertEqual(files[0].read_text(), '{"n": 42}')

    def test_a_name_claimed_by_both_halves_is_reported_once(self):
        """`artifacts/m.json` and a declared `m.json` want the same target.

        Both halves of the fetch write it, and the declared output is the one
        the caller asked for by name, so it lands last and wins. What must not
        happen is the path being counted twice, which reads as two files
        retrieved where there is one.
        """
        with tempfile.TemporaryDirectory() as home, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": home}):
            record = jobs.JobRecord.create(
                job_id="job-out-3",
                backend="local",
                extra={"outputs": [{
                    "local": "m.json",
                    "remote": "m.json",
                }]})
            (record.directory / "artifacts" / "m.json").write_text('"stray"')
            (record.directory / "m.json").write_text('"declared"')
            with tempfile.TemporaryDirectory() as dest:
                files = self.driver.fetch_artifacts("job-out-3", Path(dest))
                self.assertEqual(files, list(dict.fromkeys(files)))
                self.assertEqual((Path(dest) / "m.json").read_text(),
                                 '"declared"')

    def test_run_stages_inputs_and_outputs_like_a_remote(self):
        """Verify the local backend honours the same --input/--output spelling.

        A prompt written for a TPU VM names the payload by bare name and
        expects the result at a workspace path; running it on `local` must
        not require a second spelling.
        """
        work = pathlib.Path(tempfile.mkdtemp())
        payload = work / "in" / "payload.json"
        payload.parent.mkdir()
        payload.write_text(json.dumps({"n": 41}))
        script = self._script(
            "import json, sys, os\n"
            "n = json.load(open(sys.argv[1]))['n']\n"
            "json.dump({'n': n + 1, 'cwd': os.getcwd()}, open(sys.argv[2], 'w'))\n"
        )
        result_local = work / "out" / "results.json"

        res = self.driver.run(
            script,
            args=["payload.json", "results.json"],
            inputs=[StagedFile(local=str(payload), remote="payload.json")],
            outputs=[
                StagedFile(local=str(result_local), remote="results.json")
            ],
        )
        self.assertEqual(res.exit_code, 0, res.stderr)
        produced = json.loads(result_local.read_text())
        self.assertEqual(produced["n"], 42)
        # Ran in a scratch directory, not the caller's, and that directory
        # did not outlive the run.
        self.assertNotEqual(produced["cwd"], os.getcwd())
        self.assertFalse(pathlib.Path(produced["cwd"]).exists())

    def test_run_without_staging_keeps_the_callers_directory(self):
        """Verify plain `run` still executes in cwd, as scripts relying on it expect."""
        res = self.driver.run(self._script("import os; print(os.getcwd())\n"))
        self.assertEqual(res.stdout.strip(), os.getcwd())

    def test_run_returns_output_and_exit_code(self):
        """Verify synchronous execution reports stdout, exit status, and interpreter."""
        path = self._script("import sys; print('ran_ok'); sys.exit(3)\n")
        res = self.driver.run(path)
        self.assertIsInstance(res, SynchronousExecutionResult)
        self.assertEqual(res.exit_code, 3)
        self.assertIn("ran_ok", res.stdout)

        interp_path = self._script("import sys; print(sys.executable)\n")
        res_interp = self.driver.run(interp_path,
                                     python_interpreter=sys.executable)
        self.assertEqual(res_interp.exit_code, 0)
        self.assertIn(sys.executable, res_interp.stdout)

    def test_run_kills_the_whole_process_tree_on_timeout(self):
        """A timed-out workload must not leave children holding the device.

        `subprocess.run` kills the direct child and nothing below it. The
        workload this package times out most often -- an autotune sweep --
        spawns a subprocess per trial, so the children are the part that
        matters: orphaned, they keep the accelerator and the next run collides
        with them.

        The assertion is on elapsed time, because killing only the parent has
        two symptoms and this is the sharper one: the grandchild inherits the
        stdout pipe, so `communicate()` blocks until it exits on its own. The
        orphan does not merely survive, it hangs the caller for as long as it
        runs. A test that only checked "is the pid gone" would pass against the
        broken version too -- eventually.
        """
        marker = pathlib.Path(tempfile.mkdtemp()) / "child_alive.txt"
        child_lifetime = 30
        path = self._script(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c',\n"
            f"    \"import time,pathlib;\"\n"
            f"    \"pathlib.Path(r'{marker}').write_text('1');\"\n"
            f"    \"time.sleep({child_lifetime})\"])\n"
            "print(child.pid, flush=True)\n"
            f"time.sleep({child_lifetime})\n")

        started = time.monotonic()
        res = self.driver.run(path, timeout=3)
        elapsed = time.monotonic() - started

        self.assertEqual(res.exit_code, 124)
        self.assertIn("process group was terminated", res.stderr)
        self.assertLess(
            elapsed, child_lifetime / 2,
            f"run() took {elapsed:.1f}s for a 3s timeout: it waited on a "
            "grandchild that should have been killed with the group")

        # The grandchild must have started, or the test proves nothing.
        self.assertTrue(marker.is_file(), "the grandchild never started")

        # Poll rather than check once: the signal lands immediately but the pid
        # survives as a zombie until init reaps it, which is not instant.
        child_pid = int(res.stdout.split()[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:
            self.fail("the grandchild survived and still holds the device")

    def test_missing_input_raises_the_same_error_for_run_and_submit(self):
        """Verify both entry points answer a missing file identically."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nonexistent", "script.py")
            for call in (self.driver.run, self.driver.submit):
                with self.subTest(call.__name__):
                    with self.assertRaises(ComputeConfigError) as ctx:
                        call(missing)
                    self.assertEqual(ctx.exception.error_code,
                                     "INPUT_NOT_FOUND")
                    self.assertIn("does not exist", ctx.exception.message)

    def test_unreadable_input_reports_permission_cause(self):
        """Verify an inaccessible parent directory reports why it could not be stat'd."""
        with patch.object(Path,
                          "is_file",
                          side_effect=PermissionError(13,
                                                      "Permission denied")):
            with self.assertRaises(ComputeConfigError) as ctx:
                self.driver.run("/restricted/dir/script.py")
            self.assertEqual(ctx.exception.error_code, "INPUT_NOT_FOUND")
            self.assertIn("is not accessible", ctx.exception.message)
            self.assertIn("Permission denied", ctx.exception.message)

    def test_submitted_job_tracks_status_logs_and_exit_code(self):
        """Verify a detached job reaches a terminal state with its output kept."""
        for name, exit_code, expected in (("success", 0, "COMPLETED"),
                                          ("failure", 42, "FAILED")):
            with self.subTest(name):
                path = self._script(
                    f"import sys; print('logged_output'); sys.exit({exit_code})\n"
                )
                # Through ComputeDriver._new_record, which every driver uses.
                self.driver.compute_profile = "vm"
                job_id = self.driver.submit(path)
                self.assertTrue(job_id.startswith("job-"))
                self.assertEqual(
                    jobs.JobRecord.load(job_id).compute_profile, "vm")

                status = self._await_terminal(job_id)
                self.assertIsInstance(status, AsynchronousExecutionResult)
                self.assertEqual(status.status, expected)
                self.assertEqual(status.exit_code, exit_code)
                self.assertIn("logged_output", self.driver.fetch_logs(job_id))

    def test_cancel_marks_the_job_cancelled(self):
        """Verify terminating a running job reports CANCELLED, not FAILED."""
        job_id = self.driver.submit(
            self._script("import time\ntime.sleep(10)\n"))
        self.driver.cancel(job_id)
        self.assertEqual(self.driver.get_status(job_id).status, "CANCELLED")
        record = jobs.JobRecord.load(job_id)
        if record and record.pid:
            _terminate_quietly(record.pid)

    def test_a_reused_pid_is_neither_running_nor_signalled(self):
        """A pid now held by another process is not the job, and is left be.

        The reapers run long after submit, even after a reboot, when a
        recorded pid can belong to anything; SIGTERM must not reach it.
        """
        job_id = self.driver.submit(
            self._script("import time\ntime.sleep(10)\n"))
        record = jobs.JobRecord.load(job_id)
        self.addCleanup(_terminate_quietly, record.pid)
        self.assertEqual(record.pid_started, jobs.process_started(record.pid))
        record.pid_started = "another process"
        record.save()

        with patch("os.killpg") as killpg, patch("os.kill") as kill:
            self.driver.cancel(job_id)
        killpg.assert_not_called()
        self.assertNotIn(signal.SIGTERM, [c.args[1] for c in kill.mock_calls])
        self.assertNotEqual(jobs.JobRecord.load(job_id).status, "RUNNING")


_PROBE_PAYLOAD = {
    "accelerator": "v4",
    "topology": "2x2x1",
    "chips": 4,
    "hbm_gb": 32.0,
    "status": "healthy",
    "available_chips": 4,
    "held_by": [],
    "details": {
        "accelerator_type": "v4-8"
    },
}


def _shell_drivers() -> dict[str, Any]:
    """The RemoteShellDriver subclasses, keyed by backend name."""
    return {
        "gce_tpu_vm":
        lambda: GceTpuVmDriver(
            tpu_name="test-tpu", zone="us-central2-b", project="test-proj"),
        "gke_pod":
        lambda: GkePodDriver(namespace="default", pod="tpu-pod"),
    }


class TestRemoteShellDrivers(_ScriptMixin, unittest.TestCase):
    """The job lifecycle GceTpuVmDriver and GkePodDriver inherit and share.

    Both reach their target as "run this shell command there", so the launch
    wrapper, the pid/exit-code status protocol, log retrieval, and cancellation
    are one implementation; these run it against both transports.
    """

    @patch("subprocess.run")
    def test_probe_output_becomes_a_profile(self, mock_run):
        """Verify the shipped probe's JSON is parsed into a HardwareProfile."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed(json.dumps(_PROBE_PAYLOAD))
                profile = make_driver().inspect()
                self.assertEqual(profile.chips, 4)
                self.assertEqual(profile.accelerator, "v4")
                self.assertEqual(profile.topology, "2x2x1")
                self.assertEqual(profile.status, "healthy")

    @patch("subprocess.run")
    def test_profile_names_the_host_it_measured(self, mock_run):
        """Verify each transport labels the profile with its own target.

        The probe reports what it found but not where; without these details a
        cached profile cannot be told apart from another backend's.
        """
        mock_run.return_value = _completed(json.dumps(_PROBE_PAYLOAD))
        gce = GceTpuVmDriver(tpu_name="test-tpu",
                             zone="us-central2-b",
                             project="test-proj").inspect()
        self.assertEqual(gce.details["tpu_name"], "test-tpu")
        self.assertEqual(gce.details["zone"], "us-central2-b")
        self.assertEqual(gce.details["project"], "test-proj")

        pod = GkePodDriver(namespace="default", pod="tpu-pod").inspect()
        self.assertEqual(pod.details["pod"], "tpu-pod")
        self.assertEqual(pod.details["namespace"], "default")

    @patch("subprocess.run")
    def test_remote_contention_is_reported(self, mock_run):
        """Verify a busy remote host reads as busy rather than healthy.

        `gcloud describe` would call such a VM READY; only an on-host probe can
        see that another process already holds the chips.
        """
        payload = dict(_PROBE_PAYLOAD,
                       status="busy",
                       available_chips=0,
                       held_by=[f"PID 42 on /dev/accel{i}" for i in range(4)])
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed(json.dumps(payload))
                profile = make_driver().inspect()
                self.assertEqual(profile.status, "busy")
                self.assertEqual(profile.available_chips, 0)

    @patch("subprocess.run")
    def test_unreachable_host_is_not_free_capacity(self, mock_run):
        """Verify a host that could not be probed reports unknown availability."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed(
                    returncode=255, stderr="ssh: connect refused")
                profile = make_driver().inspect()
                self.assertEqual(profile.status, "unreachable")
                self.assertIsNone(profile.available_chips)
                self.assertEqual(profile.chips, 0)

    @patch("subprocess.run")
    def test_probe_runs_under_the_configured_interpreter(self, mock_run):
        """Verify inspect probes with `python_interpreter`, like run does.

        `tpu_info` is installed into the user's venv per `requirements-tpu.txt`,
        so a probe that shells out to the target's bare `python3` reports a
        healthy VM as unreachable while `run` on the same VM succeeds.
        """
        venv_python = "~/tpu_venv/bin/python3"
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.reset_mock()
                mock_run.return_value = _completed(json.dumps(_PROBE_PAYLOAD))
                driver = make_driver()
                driver.python_interpreter = venv_python
                driver.inspect()
                remote = " ".join(mock_run.call_args.args[0])
                self.assertIn(f"{venv_python} -c ", remote)
                self.assertNotIn(" python3 -c ", remote)

    @patch("subprocess.run")
    def test_failed_probe_names_its_interpreter(self, mock_run):
        """Verify an unreachable profile says which interpreter the probe used.

        A missing `tpu_info` looks identical to a dead ssh link in the status
        field; the interpreter in `details` is what points at the venv.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed(
                    returncode=1, stderr="RuntimeError: tpu_info is required")
                driver = make_driver()
                driver.python_interpreter = "/opt/venv/bin/python3"
                profile = driver.inspect()
                self.assertEqual(profile.status, "unreachable")
                self.assertEqual(profile.details["python_interpreter"],
                                 "/opt/venv/bin/python3")
                self.assertIn("tpu_info", profile.details["error"])

    @patch("subprocess.run")
    def test_run_copies_then_executes(self, mock_run):
        """Verify synchronous execution stages the file and returns its output."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [
                    _completed(),  # copy in
                    _completed("remote_ok\n"),  # execute
                ]
                res = make_driver().run(self._script("print('hi')\n"))
                self.assertEqual(res.exit_code, 0)
                self.assertEqual(res.stdout, "remote_ok\n")

    @patch("subprocess.run")
    def test_run_stages_inputs_and_copies_outputs_back(self, mock_run):
        """Verify --input files land beside the script and --output files return.

        The autotune and profiling prompts hand a script a payload by bare
        name and read a result by bare name; on a remote backend that only
        works if both travel with the script and it runs where they landed.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                payload = self._script("{}", suffix=".json")
                out_dir = pathlib.Path(tempfile.mkdtemp())
                out_local = out_dir / "nested" / "results.json"

                responses = iter([
                    _completed(),  # mkdir run dir
                    _completed(),  # copy script in
                    _completed(),  # copy payload in
                    _completed("ran\n"),  # execute
                    _completed(),  # copy results out
                ])

                def fake_run(cmd, *a, **k):
                    # The copy-out names the local path; "deliver" the file.
                    if any(str(out_local) == str(x) for x in cmd):
                        out_local.parent.mkdir(parents=True, exist_ok=True)
                        out_local.write_text("{}")
                    return next(responses)

                mock_run.side_effect = fake_run
                res = make_driver().run(
                    self._script("print('hi')\n"),
                    args=["payload.json", "results.json"],
                    inputs=[StagedFile(local=payload, remote="payload.json")],
                    outputs=[
                        StagedFile(local=str(out_local), remote="results.json")
                    ],
                )
                self.assertEqual(res.exit_code, 0, res.stderr)
                self.assertTrue(out_local.is_file())

                # The script ran inside the run directory, by bare name, with
                # the payload copied to that same directory. The workload
                # travels base64-encoded, so assert on what the target will
                # actually execute rather than on the SSH command around it.
                calls = [
                    " ".join(str(x) for x in c.args[0])
                    for c in mock_run.call_args_list
                ]
                execute = calls[3]
                inner = _decode_inner_script(execute)
                self.assertIn("cd /tmp/tpu_agent_suite_run_", inner)
                self.assertIn("script.py payload.json results.json", inner)
                self.assertIn("/payload.json", calls[2])
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_run_keeps_a_shell_recipe_apart_from_its_launch_wrapper(
            self, mock_run):
        """Verify an unstaged .sh recipe isn't overwritten by the wrapper."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [_completed(), _completed()]
                make_driver().run(self._script("echo hi\n", suffix=".sh"))
                copy_in, execute = (" ".join(map(str, c.args[0]))
                                    for c in mock_run.call_args_list)
                recipe = re.search(r"/tmp/tpu_agent_suite_run_\S+?\.sh",
                                   copy_in).group(0)
                self.assertNotIn(f"base64 -d > {recipe}", execute)
                self.assertIn(f"exec bash {recipe}",
                              _decode_inner_script(execute))
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_run_reaps_the_remote_process_when_it_times_out(self, mock_run):
        """A dead SSH client does not stop the workload; it must be killed.

        This is the failure that silently costs an accelerator: the client
        gives up, the caller sees exit 124 and moves on, and the workload is
        still running on the target holding the chips the next run wants.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                calls = []

                def fake_run(cmd, **kwargs):
                    joined = " ".join(str(x) for x in cmd)
                    calls.append(joined)
                    if "base64 -d" in joined:
                        raise subprocess.TimeoutExpired(cmd, 5)
                    return _completed("REAPED\n")

                mock_run.side_effect = fake_run
                res = make_driver().run(self._script("print('hi')\n"),
                                        timeout=5)

                self.assertEqual(res.exit_code, 124)
                self.assertIn("timed out", res.stderr)
                self.assertIn("process group was terminated", res.stderr)
                reap = calls[-1]
                self.assertIn("kill -TERM", reap)
                self.assertIn("kill -KILL", reap)
                self.assertIn("ps -o pgid=", reap)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_run_serialises_on_the_target_run_lock(self, mock_run):
        """Concurrent runs against one accelerator must not interleave."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [_completed(), _completed("ok\n")]
                make_driver().run(self._script("print('hi')\n"), timeout=600)
                execute = " ".join(
                    str(x) for x in mock_run.call_args_list[-1].args[0])
                self.assertIn("flock -w", execute)
                self.assertIn(shell.RUN_LOCK_PATH, execute)
                # Half the call's timeout, so a contended target still leaves
                # the workload most of its budget.
                self.assertIn("-w 300", execute)
                # Absent flock, the run proceeds rather than failing.
                self.assertIn("command -v flock", execute)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_run_reports_a_busy_target_as_never_having_started(self, mock_run):
        """Losing the lock race is not a failed run, and must not read as one."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [
                    _completed(),
                    _completed(returncode=shell.LOCK_BUSY_EXIT),
                ]
                res = make_driver().run(self._script("print('hi')\n"),
                                        timeout=60)
                self.assertEqual(res.exit_code, shell.LOCK_BUSY_EXIT)
                self.assertIn("busy", res.stderr)
                self.assertIn("no measurement was taken", res.stderr)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_run_reports_a_declared_output_that_never_appeared(self, mock_run):
        """Verify a clean exit without a promised output is not reported as success."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                missing = pathlib.Path(tempfile.mkdtemp()) / "never.json"
                mock_run.side_effect = [
                    _completed(),  # mkdir
                    _completed(),  # copy script
                    _completed("ok\n"),  # execute, exit 0
                    _completed(returncode=1,
                               stderr="No such file"),  # copy out
                ]
                res = make_driver().run(self._script("print('hi')\n"),
                                        outputs=[
                                            StagedFile(local=str(missing),
                                                       remote="never.json")
                                        ])
                self.assertEqual(res.exit_code, 1)
                self.assertIn("never.json", res.stderr)
                self.assertIn("not produced", res.stderr)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_submit_then_status_reads_the_remote_exit_code(self, mock_run):
        """Verify a detached job is tracked through the remote exit-code file."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [
                    _completed(),  # mkdir
                    _completed(),  # copy recipe in
                    _completed(),  # detached launch
                    _completed("COMPLETED 0\n"),  # status probe
                ]
                driver = make_driver()
                job_id = driver.submit(self._script("print('submitted')\n"))
                self.assertTrue(job_id.startswith("job-"))
                status = driver.get_status(job_id)
                self.assertEqual(status.status, "COMPLETED")
                self.assertEqual(status.exit_code, 0)

    @patch("subprocess.run")
    def test_fetch_logs_returns_the_remote_log(self, mock_run):
        """Verify logs are read back from the job's remote directory."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed("remote_log_contents\n")
                self.assertEqual(make_driver().fetch_logs("job-123"),
                                 "remote_log_contents\n")

    @patch("subprocess.run")
    def test_cancel_of_a_finished_job_keeps_its_exit_status(self, mock_run):
        """The status probe says the job already ended; cancel records that."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.return_value = _completed("COMPLETED 0\n")
                driver.cancel(job_id)
                self.assertEqual(
                    jobs.JobRecord.load(job_id).status, "COMPLETED")
                for call in mock_run.call_args_list:
                    self.assertNotIn("kill -TERM", " ".join(call[0][0]))

    @patch("subprocess.run")
    def test_cancel_that_cannot_reach_the_target_raises(self, mock_run):
        """Unreachable target: the record stays live rather than lying."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = _completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.side_effect = [
                    _completed("RUNNING\n"),
                    _completed(returncode=255, stderr="ssh: connect refused"),
                ]
                with self.assertRaises(ComputeConfigError) as ctx:
                    driver.cancel(job_id)
                mock_run.side_effect = None
                self.assertEqual(ctx.exception.error_code, "CANCEL_FAILED")
                self.assertIn("connect refused", str(ctx.exception))
                self.assertEqual(jobs.JobRecord.load(job_id).status, "RUNNING")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "reads /proc")
    def test_only_the_launched_process_counts_as_the_job(self):
        """The target-side check behind status and cancel, run for real."""
        proc = subprocess.Popen(["sleep", "30"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        job_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, job_dir)
        (job_dir / "pid.txt").write_text(f"{proc.pid}\n")
        # The target reads field 22 of /proc/<pid>/stat, as process_started.
        (job_dir /
         "pid_started.txt").write_text(f"{jobs.process_started(proc.pid)}\n")

        def live() -> bool:
            check = f"{shell._live_pid(str(job_dir))} echo \"$LIVE\""
            return subprocess.run(["sh", "-c", check],
                                  capture_output=True,
                                  text=True).stdout.strip() == "1"

        self.assertTrue(live())
        (job_dir / "pid_started.txt").write_text("1\n")  # pid reused
        self.assertFalse(live())
        (job_dir / "pid_started.txt").unlink()  # launched before the file
        self.assertTrue(live())

    @patch("subprocess.run")
    def test_cancelled_job_stays_cancelled(self, mock_run):
        """Verify cancelling then polling does not relabel the job as FAILED.

        Killing the remote process makes the status probe report `FAILED 1`;
        cancellation is the truer account of why it stopped.
        """
        mock_run.return_value = _completed()
        driver = GkePodDriver(namespace="default", pod="tpu-pod")
        job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
        driver.cancel(job_id)

        mock_run.return_value = _completed("FAILED 1\n")
        self.assertEqual(driver.get_status(job_id).status, "CANCELLED")
        record = jobs.JobRecord.load(job_id)
        if record and record.pid:
            _terminate_quietly(record.pid)

    def test_gce_command_carries_vm_coordinates_and_ssh_flags(self):
        """Verify gcloud calls name the VM, zone, project, and separate ssh flags."""
        cmd = GceTpuVmDriver(tpu_name="t", zone="z",
                             project="p")._gcloud_base("ssh")
        self.assertEqual(cmd[:6],
                         ["gcloud", "compute", "tpus", "tpu-vm", "ssh", "t"])
        self.assertIn("--zone=z", cmd)
        self.assertIn("--project=p", cmd)
        ssh_flags = [part for part in cmd if part.startswith("--ssh-flag=")]
        self.assertEqual(len(ssh_flags), 3)
        for flag in ssh_flags:
            self.assertEqual(flag.count("-o "), 1)

    def test_gce_internal_ip_reaches_ssh_and_scp_only(self):
        """Verify internal_ip adds --internal-ip to ssh and scp, not describe."""
        driver = GceTpuVmDriver.from_params({
            "tpu_name": "t",
            "zone": "z",
            "project": "p",
            "internal_ip": "true"
        })
        self.assertIn("--internal-ip", driver._gcloud_base("ssh"))
        self.assertNotIn("--internal-ip", driver._gcloud_base("describe"))
        with patch("subprocess.run") as run:
            driver._scp("a", "t:b")
        self.assertIn("--internal-ip", run.call_args.args[0])

    def test_shell_recipe_runs_under_bash_with_tpu_agent_suite_python(self):
        """Verify a .sh runs under bash, handed the interpreter, not run by it."""
        driver = GkePodDriver(namespace="default",
                              pod="tpu-pod",
                              python_interpreter="~/venv/bin/python3")
        self.assertEqual(driver._runner_for(pathlib.Path("r.sh"), None),
                         "TPU_AGENT_SUITE_PYTHON=~/venv/bin/python3 bash")
        self.assertEqual(driver._runner_for(pathlib.Path("r.py"), None),
                         "~/venv/bin/python3")

    def test_run_inner_script_executes_a_shell_recipe(self):
        """Verify the wrapper really runs a .sh recipe with its interpreter."""
        driver = GkePodDriver(namespace="default",
                              pod="tpu-pod",
                              python_interpreter="~/venv/bin/python3")
        with tempfile.TemporaryDirectory() as tmp:
            recipe = pathlib.Path(tmp) / "recipe.sh"
            recipe.write_text('echo "py=$TPU_AGENT_SUITE_PYTHON arg=$1"\n')
            command = (f"{driver._runner_for(recipe, None)} "
                       f"{shlex.quote(str(recipe))} a1")
            proc = subprocess.run([
                "sh", "-c",
                shell.run_inner_script(command, f"{tmp}/pid", None)
            ],
                                  capture_output=True,
                                  text=True,
                                  timeout=30,
                                  check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"py={os.path.expanduser('~')}/venv/bin/python3 arg=a1",
                      proc.stdout)

    def test_gke_pod_command_carries_namespace_and_container(self):
        """Verify kubectl calls are scoped to the pod's namespace and container."""
        driver = GkePodDriver(namespace="ns-1", pod="p-1", container="trainer")
        cmd = driver._exec_prefix()
        self.assertEqual(cmd[:5], ["kubectl", "-n", "ns-1", "exec", "p-1"])
        self.assertEqual(cmd[5:], ["-c", "trainer"])
        # A pinned context goes on every call; unset, none does.
        pinned = GkePodDriver.from_params({"pod": "p-1", "kube_context": "c"})
        self.assertEqual(pinned._kubectl_base(),
                         ["kubectl", "--context", "c", "-n", "default"])


class TestGkeXpkDriver(_ScriptMixin, unittest.TestCase):
    """Cluster workloads via xpk: status vocabulary, occupancy, and GCS reads."""

    def setUp(self):
        self.driver = GkeXpkDriver(cluster_name="c1",
                                   project="p1",
                                   zone="z1",
                                   gcs_bucket="gs://test-bucket")
        self.empty_listing = _completed("Jobset Name  Status\n")

    def _listing(self, name: str, status: str) -> subprocess.CompletedProcess:
        """An `xpk workload list` table with one workload in the given state.

        The name sits in the second column, so this also covers the driver
        matching an identifier wherever xpk chooses to print it.
        """
        return _completed(f"Jobset Name  Workload Name  Status\n"
                          f"jobset-1     {name}         {status}\n")

    def test_run_is_refused_before_anything_executes(self):
        """Verify sync `run` raises rather than running on the calling host.

        The script must not run at all: a real exit code from a chipless host
        is the failure mode being closed off, so the marker file the script
        would write is the assertion.
        """
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "ran"
            path = self._script(f"open({str(marker)!r}, 'w').close()\n")
            with self.assertRaises(ComputeConfigError) as ctx:
                self.driver.run(path, inputs=[StagedFile(path, "in.py")])
            self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_OPERATION")
            self.assertIn("compute submit", ctx.exception.message)
            self.assertFalse(marker.exists())

    def test_submit_refuses_staging_it_cannot_do(self):
        """The recipe builds the workload, so nothing here can stage for it.

        Accepting the flags and staging nothing leaves the recipe to die on a
        missing path at runtime, long after the command reported success.
        """
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "ran"
            path = self._script(f"open({str(marker)!r}, 'w').close()\n")
            with self.assertRaises(ComputeConfigError) as ctx:
                self.driver.submit(path, outputs=[StagedFile(path, "o.pt")])
            self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_OPERATION")
            self.assertFalse(marker.exists())

    def test_command_coordinates_and_bucket_formatting(self):
        """Verify xpk commands carry cluster coordinates and bucket paths normalize."""
        cmd = self.driver._xpk("workload", "list")
        self.assertEqual(cmd[:3], ["xpk", "workload", "list"])
        self.assertIn("--cluster=c1", cmd)
        self.assertIn("--project=p1", cmd)
        self.assertIn("--zone=z1", cmd)

        driver = GkeXpkDriver(cluster_name="c",
                              project="p",
                              zone="z",
                              gcs_bucket="gs://my-bucket/")
        self.assertEqual(driver.gcs_bucket, "gs://my-bucket")
        # Spelled either way in environment.json, read one way: never
        # `gs://gs://...`, never a bare name handed to gcloud.
        for spelled in ("my-bucket", "my-bucket/", "gs://my-bucket"):
            with self.subTest(spelled=spelled):
                driver = GkeXpkDriver(cluster_name="c",
                                      project="p",
                                      zone="z",
                                      gcs_bucket=spelled)
                self.assertEqual(driver.gcs_bucket, "gs://my-bucket")
        self.assertEqual(
            GkeXpkDriver(cluster_name="c", project="p", zone="z").gcs_bucket,
            "")

    @patch("subprocess.run")
    def test_workload_state_vocabulary(self, mock_run):
        """Verify each xpk/Kueue state maps to the framework's own vocabulary.

        Kueue's ADMITTED, SUSPENDED, PREEMPTED, and EVICTED are not terminal --
        the workload still holds or is reclaiming quota and may be requeued, so
        calling a preempted job FAILED would end a poll loop on a live job.
        """
        cases = [
            ("COMPLETE", "COMPLETED", 0),
            ("SUCCESSFUL", "COMPLETED", 0),
            ("FAILED", "FAILED", 1),
            ("FINISHED", "COMPLETED", 0),
            ("FINISHED JobSet failed: Pod backoff limit reached", "FAILED", 1),
            ("RUNNING", "RUNNING", None),
            ("QUEUED", "PENDING", None),
            ("ADMITTED", "PENDING", None),
            ("PREEMPTED", "PENDING", None),
        ]
        for raw, expected, exit_code in cases:
            with self.subTest(raw):
                mock_run.return_value = self._listing("wl-1", raw)
                result = self.driver.get_status("wl-1")
                self.assertEqual(result.status, expected)
                self.assertEqual(result.exit_code, exit_code)

    @patch("subprocess.run")
    def test_unrecognised_state_is_reported_not_hidden(self, mock_run):
        """Verify an unknown state names itself instead of looking like absence.

        A vocabulary this build does not know is a different problem from a
        workload that was never created, and the message must say which.
        """
        mock_run.return_value = self._listing("wl-x", "FLARGLE")
        result = self.driver.get_status("wl-x")
        self.assertEqual(result.status, "UNKNOWN")
        self.assertIn("unrecognised workload state", result.error)
        self.assertIn("FLARGLE", result.error)

    @patch("subprocess.run")
    def test_missing_workload_is_unknown_not_running(self, mock_run):
        """Verify an absent workload reports UNKNOWN, not a perpetual RUNNING.

        An orchestrator polls until a terminal state; reporting RUNNING for a
        workload that does not exist hangs that loop forever.
        """
        mock_run.return_value = self.empty_listing
        result = self.driver.get_status("job-nonexistent")
        self.assertEqual(result.status, "UNKNOWN")
        self.assertIn("not found", result.error)

    @patch("subprocess.run")
    def test_recipe_exit_code_does_not_stand_in_for_the_workload(
            self, mock_run):
        """Verify the local recipe's exit code cannot report job completion.

        The recipe only *submits* the workload, so exit 0 proves nothing about
        the job -- but a non-zero exit does explain an absent workload.
        """
        mock_run.return_value = self.empty_listing
        for exit_code, expected in (("0\n", "UNKNOWN"), ("3\n", "FAILED")):
            with self.subTest(exit_code.strip()):
                record = jobs.JobRecord.create(job_name="j",
                                               workload_name="wl-sub")
                (record.directory / "exit_code.txt").write_text(exit_code)
                result = self.driver.get_status(record.job_id)
                self.assertEqual(result.status, expected)
                if expected == "FAILED":
                    self.assertEqual(result.exit_code, 3)
                    self.assertIn("before the workload was created",
                                  result.error)

    @patch("subprocess.run")
    def test_in_flight_submission_is_pending_not_missing(self, mock_run):
        """Verify a still-submitting job is PENDING, not reported as absent.

        `xpk workload create` takes tens of seconds; during that window the
        workload is legitimately not on the cluster yet, and calling it missing
        tells an orchestrator the job died before it started.
        """
        mock_run.return_value = _completed("Workload created\n")
        job_id = self.driver.submit(self._script("sleep 30\n", suffix=".sh"),
                                    job_name="wl-inflight")
        # Reap the background recipe directly: `cancel` would shell out to the
        # real xpk binary once this test's mock is torn down.
        record = jobs.JobRecord.load(job_id)
        if record and record.pid:
            self.addCleanup(_terminate_quietly, record.pid)

        mock_run.return_value = self.empty_listing
        result = self.driver.get_status(job_id)
        self.assertEqual(result.status, "PENDING")
        self.assertIsNone(result.error)

    @patch("subprocess.run")
    def test_occupancy_matches_status_columns_not_row_text(self, mock_run):
        """Verify a workload named `pending-sweep` does not count as pending.

        Occupancy drives capacity reporting, so statuses are matched as whole
        columns rather than as substrings anywhere in the row.
        """
        mock_run.return_value = _completed(
            "Jobset Name    Status\npending-sweep  FINISHED\n")
        self.assertEqual(self.driver._occupying_workloads(), [])

        mock_run.return_value = _completed(
            "Jobset Name  Status\nwl-admitted  ADMITTED\n")
        self.assertEqual(self.driver._occupying_workloads(), ["wl-admitted"])

    @patch("subprocess.run")
    def test_unlistable_workloads_report_unknown_availability(self, mock_run):
        """Verify a cluster whose workloads cannot be listed is not called free."""
        mock_run.return_value = _completed(returncode=1, stderr="boom")
        self.assertIsNone(self.driver._occupying_workloads())

    @patch("subprocess.run")
    def test_inspect_scrapes_cluster_description(self, mock_run):
        """Verify cluster TPU type and health come from `xpk cluster describe`."""
        mock_run.return_value = _completed(
            "device_type: v5e\nchips_per_node: 4\n")
        profile = self.driver.inspect()
        self.assertEqual(profile.accelerator, "v5e")
        self.assertEqual(profile.status, "healthy")

    def test_inspect_handles_failures_gracefully(self):
        """Verify inspect degrades to an error profile on non-zero exit or missing binary."""
        with patch("subprocess.run",
                   return_value=_completed(returncode=1,
                                           stderr="cluster not found")):
            profile = self.driver.inspect()
            self.assertEqual(profile.status, "degraded")
            self.assertIsNone(profile.available_chips)
            self.assertIn("cluster not found", profile.details["error"])

        with patch("subprocess.run",
                   side_effect=OSError("xpk binary missing")):
            profile = self.driver.inspect()
            self.assertIn("error", profile.status)
            self.assertIsNone(profile.available_chips)
            self.assertIn("xpk binary missing", profile.details["error"])

    @patch("subprocess.run")
    def test_gcs_log_name_is_discovered_not_assumed(self, mock_run):
        """Verify log retrieval lists the prefix rather than assuming a filename.

        The name is recipe-specific: MaxText writes `train-0.log`, others use
        `main.log` or `rank-0.log`.
        """

        def fake(cmd, **_kwargs):
            if cmd[:3] == ["gcloud", "storage", "ls"]:
                return _completed("gs://test-bucket/wl/logs/rank-0.log\n")
            return _completed("log body")

        mock_run.side_effect = fake
        self.assertEqual(self.driver._gcs_logs("wl"), "log body")

    def test_fetch_logs_returns_empty_for_unknown_job(self):
        """Verify an unknown job yields no logs rather than an error."""
        with patch("subprocess.run", return_value=_completed(returncode=1)):
            self.assertEqual(self.driver.fetch_logs("nonexistent-job"), "")

    @patch("subprocess.run")
    def test_cancel_deletes_the_workload(self, mock_run):
        """Verify cancelling asks xpk to delete the workload and records it."""
        mock_run.return_value = _completed("Deleted\n")
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-xyz")
        self.driver.cancel(record.job_id)

        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:3], ["xpk", "workload", "delete"])
        self.assertIn("--workload=wl-xyz", cmd)
        self.assertEqual(
            jobs.JobRecord.load(record.job_id).status, "CANCELLED")

    @patch("subprocess.run")
    def test_cancel_of_a_finished_workload_records_how_it_finished(
            self, mock_run):
        """A FINISHED workload is COMPLETED, not CANCELLED, and not deleted."""
        mock_run.return_value = self._listing("wl-done", "FINISHED")
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-done")
        self.driver.cancel(record.job_id)
        self.assertEqual(
            jobs.JobRecord.load(record.job_id).status, "COMPLETED")
        for call in mock_run.call_args_list:
            self.assertNotIn("delete", call[0][0])

    @patch("subprocess.run")
    def test_cancel_that_cannot_delete_raises_and_keeps_the_record_live(
            self, mock_run):
        """A CANCELLED record over a live workload is a leak; refuse it."""
        mock_run.side_effect = [
            self._listing("wl-live", "RUNNING"),
            _completed(returncode=1,
                       stderr="ERROR: permission denied on cluster c1"),
        ]
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-live")
        with self.assertRaises(ComputeConfigError) as ctx:
            self.driver.cancel(record.job_id)
        self.assertEqual(ctx.exception.error_code, "CANCEL_FAILED")
        self.assertIn("permission denied", str(ctx.exception))
        self.assertEqual(jobs.JobRecord.load(record.job_id).status, "RUNNING")

    @patch("subprocess.run")
    def test_cancel_tolerates_a_workload_already_gone(self, mock_run):
        mock_run.side_effect = [
            self.empty_listing,
            _completed(returncode=1, stderr="workload wl-gone not found"),
        ]
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-gone")
        self.driver.cancel(record.job_id)
        self.assertEqual(
            jobs.JobRecord.load(record.job_id).status, "CANCELLED")

    def test_fetch_artifacts_requires_bucket(self):
        driver = GkeXpkDriver(cluster_name="c1", project="p1", zone="z1")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ComputeConfigError) as ctx:
                driver.fetch_artifacts("job-1", Path(td))
            self.assertEqual(ctx.exception.error_code, "BUCKET_NOT_CONFIGURED")
            self.assertEqual(ctx.exception.missing, ["gcs_bucket"])

    @patch("subprocess.run")
    def test_fetch_artifacts_tolerates_not_found(self, mock_run):
        mock_run.return_value = _completed(
            returncode=1,
            stderr=
            ("ERROR: (gcloud.storage.cp) One or more URLs matched no objects."
             ),
        )
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-1")
        with tempfile.TemporaryDirectory() as td:
            files = self.driver.fetch_artifacts(record.job_id, Path(td))
            self.assertEqual(files, [])

    @patch("subprocess.run")
    def test_fetch_artifacts_raises_on_failure(self, mock_run):
        mock_run.return_value = _completed(
            returncode=1,
            stderr=
            "ERROR: (gcloud.storage.cp) 403 Forbidden: Permission denied.",
        )
        record = jobs.JobRecord.create(job_name="j", workload_name="wl-2")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ComputeConfigError) as ctx:
                self.driver.fetch_artifacts(record.job_id, Path(td))
            self.assertEqual(ctx.exception.error_code, "FETCH_FAILED")
            self.assertIn("403 Forbidden", str(ctx.exception))


class TestComputeConfig(unittest.TestCase):
    """Driver resolution, environment.json schemas, and override precedence."""

    @contextlib.contextmanager
    def _environment(self, data: dict[str, Any]):
        """Resolve against a stubbed environment.json payload."""
        with patch.object(compute_config,
                          "find_environment_data",
                          return_value=data):
            yield

    def test_missing_or_unknown_backend_raises(self):
        """Verify missing or unrecognized backend names fail with proper error codes."""
        with self._environment(
            {}), self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver()
        self.assertEqual(ctx.exception.error_code, "NO_COMPUTE_BACKEND")
        self.assertIn("--package <name@version>", str(ctx.exception))

        for backend in ("auto", "nonexistent_backend_xyz"):
            with self.subTest(backend), self.assertRaises(
                    ComputeConfigError) as ctx:
                resolve_driver(backend_override=backend)
            self.assertEqual(ctx.exception.error_code, "INVALID_BACKEND")

    def test_cli_overrides_win_but_unset_flags_do_not_erase_config(self):
        """Verify overrides outrank environment.json while `None` leaves it alone.

        Unset argparse flags arrive as None; treating those as values would
        blank out configured fields on every invocation.
        """
        with self._environment(
                profile_env("mock_test",
                            endpoint="base-ep",
                            token="base-token")):
            driver = resolve_driver(backend_override="mock_test",
                                    param_overrides={
                                        "endpoint": "override-ep",
                                        "token": None
                                    })
        self.assertEqual(driver.endpoint, "override-ep")
        self.assertEqual(driver.token, "base-token")

    def test_backends_resolve_from_their_own_parameters(self):
        """Verify each shipped backend builds from its configured fields."""
        cases = [
            (profile_env("local"), LocalDriver, {}),
            (profile_env("gce_tpu_vm",
                         tpu_name="node-1",
                         zone="us-central2-b",
                         project="proj-1"), GceTpuVmDriver, {
                             "tpu_name": "node-1",
                             "zone": "us-central2-b",
                             "project": "proj-1"
                         }),
            (profile_env("gke_xpk",
                         cluster_name="my-cluster",
                         project="proj-1",
                         zone="us-central2-b"), GkeXpkDriver, {
                             "cluster_name": "my-cluster"
                         }),
            (profile_env("gke_pod", pod="my-pod",
                         namespace="my-ns"), GkePodDriver, {
                             "pod": "my-pod",
                             "namespace": "my-ns"
                         }),
            (profile_env("gke_job",
                         namespace="my-ns",
                         tpu_chips=8,
                         queue="my-queue",
                         tpu_accelerator="tpu7x",
                         tpu_topology="2x2x1"), GkeJobDriver, {
                             "namespace": "my-ns",
                             "tpu_chips": 8,
                             "queue": "my-queue",
                         }),
        ]
        for cfg, expected_cls, attrs in cases:
            with self.subTest(cfg["profiles"]["default"]["backend"]):
                run_id = "20260101T000000Z-123456789abc"
                with self._environment(cfg):
                    driver = resolve_driver(run_id=run_id)
                self.assertIsInstance(driver, expected_cls)
                self.assertEqual(driver.compute_profile, "default")
                self.assertEqual(driver.run_id, run_id)
                for field, value in attrs.items():
                    self.assertEqual(getattr(driver, field), value, field)

    def test_incomplete_profile_is_compute_not_configured(self):
        """A profile `setup apply` would not save cannot be used.

        The error names the fields to ask for and the plan command that asks.
        """
        cases = [
            ("unknown backend", profile_env("mock_test"), ["backend"]),
            ("gke_pod without pod", profile_env("gke_pod"), ["pod"]),
        ]
        for name, cfg, missing in cases:
            with self.subTest(name):
                with self._environment(cfg):
                    with self.assertRaises(ComputeConfigError) as ctx:
                        resolve_driver(package="pkg@1")
                error = ctx.exception.to_dict()
                self.assertEqual(error["error_code"], "COMPUTE_NOT_CONFIGURED")
                self.assertEqual(error["missing"], missing)
                self.assertIn(
                    "compute --package pkg@1 setup plan --profile "
                    "default", error["fix"])

    def test_python_interpreter_passed_to_driver(self):
        """Verify the shared interpreter parameter reaches the driver instance."""
        with self._environment(
                profile_env("local", python_interpreter="/opt/custom/python")):
            driver = resolve_driver()
        self.assertEqual(driver.python_interpreter, "/opt/custom/python")

    def test_cli_passes_the_job_and_fills_in_a_fields_fix(self):
        """The job's id reaches the resolver; a driver's `<GROUP>_NOT_CONFIGURED`
        gets the `setup plan --fields` command for this package and the
        driver's profile as its fix."""
        driver, seen = MockTestDriver(), {}
        driver.compute_profile = "c"
        args = argparse.Namespace(compute_action="inspect",
                                  backend=None,
                                  package="pkg@1",
                                  workspace=None,
                                  json=True,
                                  profile=None,
                                  run_id=None,
                                  job_id="job-1")
        with patch.object(driver,
                          "inspect",
                          side_effect=base.NotConfigured(
                              "unset", "STAGING_NOT_CONFIGURED",
                              ["staging_bucket"])):
            with self.assertRaises(ComputeConfigError) as ctx:
                compute.handle_compute_cli(
                    args, resolve=lambda **kw: seen.update(kw) or driver)
        self.assertEqual(seen["job_id"], "job-1")
        error = ctx.exception.to_dict()
        self.assertEqual((error["error_code"], error["missing"]),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))
        self.assertTrue(error["fix"].endswith(
            "compute --package pkg@1 setup plan --profile c --fields "
            "staging_bucket"))

    def test_run_turns_setup_fields_into_a_fix(self):
        """An output lost in a gke_job pod's log comes back as `setup_fields`;
        `run` adds the command that asks for them, as for NotConfigured."""
        driver = MockTestDriver()
        driver.compute_profile = "c"
        lost = SynchronousExecutionResult(
            exit_code=1,
            stdout="",
            stderr="",
            details={"setup_fields": ["pod_service_account"]})
        with tempfile.TemporaryDirectory() as td:
            code = Path(td) / "job.py"
            code.write_text("")
            args = argparse.Namespace(compute_action="run",
                                      backend=None,
                                      package="pkg@1",
                                      workspace=None,
                                      json=True,
                                      profile=None,
                                      run_id=None,
                                      job_id=None,
                                      code=str(code),
                                      timeout=None,
                                      run_args=[])
            with patch.object(driver, "run", return_value=lost), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                compute.handle_compute_cli(args, resolve=lambda **_: driver)
        fix = json.loads(out.getvalue())["details"]["fix"]
        self.assertTrue(
            fix.endswith("compute --package pkg@1 setup plan --profile c "
                         "--fields pod_service_account"))


class TestEnvironmentResolution(unittest.TestCase):
    """environment.json lookup must stay package-agnostic."""

    def _write(self, directory: Path, backend: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "environment.json").write_text(
            json.dumps(profile_env(backend)),
            encoding="utf-8",
        )

    def _ledger(self, directory: Path, package: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ownership.json").write_text(json.dumps(
            {"package": package}),
                                                  encoding="utf-8")

    def test_environment_resolution_precedence_and_fallback(self):
        """Ownership-keyed resolution: workspace is authoritative, central
        serves the not-installed-here case, and nothing else is consulted."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            home = Path(tmp) / "home"
            old = os.environ.get("TPU_AGENT_SUITE_HOME")
            os.environ["TPU_AGENT_SUITE_HOME"] = str(home)
            try:
                self._write(ws / ".tpu-agent-suite", "scoped")
                self._ledger(ws / ".tpu-agent-suite", "demo-pkg")
                self._write(home, "central")
                self._ledger(home, "demo-pkg")

                # Workspace config is authoritative, bare name or tag alike.
                for package in ("demo-pkg", "demo-pkg@2.0.0"):
                    self.assertEqual(
                        compute_config.find_environment_data(
                            workspace=ws,
                            package=package)["profiles"]["default"]["backend"],
                        "scoped")

                # No package: nothing is consulted.
                self.assertEqual(
                    compute_config.find_environment_data(workspace=ws), {})

                # Not installed in the workspace: the user-scope install in
                # the central store, which is not version-namespaced, so a
                # bare name reaches it too.
                ws2 = Path(tmp) / "elsewhere"
                ws2.mkdir()
                for package in ("demo-pkg", "demo-pkg@2.0.0"):
                    self.assertEqual(
                        compute_config.find_environment_data(
                            workspace=ws2,
                            package=package)["profiles"]["default"]["backend"],
                        "central")

                # Another package's install is not this one's: skipped.
                other = Path(tmp) / "other"
                self._write(other / ".tpu-agent-suite", "not-mine")
                self._ledger(other / ".tpu-agent-suite", "other-pkg")
                self.assertEqual(
                    compute_config.find_environment_data(
                        workspace=other,
                        package="demo-pkg")["profiles"]["default"]["backend"],
                    "central")

                # An unreadable ledger names no package: skipped as well.
                unreadable = Path(tmp) / "unreadable"
                self._write(unreadable / ".tpu-agent-suite", "not-mine")
                (unreadable / ".tpu-agent-suite" /
                 "ownership.json").write_text("{", encoding="utf-8")
                self.assertEqual(
                    compute_config.find_environment_data(
                        workspace=unreadable,
                        package="demo-pkg")["profiles"]["default"]["backend"],
                    "central")

                # Broken install: ledger present, config gone -> loud error.
                broken = Path(tmp) / "broken"
                self._ledger(broken / ".tpu-agent-suite", "demo-pkg")
                with self.assertRaises(runtime.TpuAgentSuiteError):
                    compute_config.find_environment_data(
                        workspace=broken, package="demo-pkg@2.0.0")

                # The resolver itself refuses an empty package outright.
                with self.assertRaises(runtime.TpuAgentSuiteError):
                    runtime.utils.resolve_environment("", ws)

                # A package installed nowhere: clean miss, though the central
                # store holds another package's config.
                self.assertEqual(
                    compute_config.find_environment_data(
                        workspace=ws2, package="ghost-pkg@1.0.0"), {})
            finally:
                if old is not None:
                    os.environ["TPU_AGENT_SUITE_HOME"] = old
                else:
                    os.environ.pop("TPU_AGENT_SUITE_HOME", None)


class TestRuntimeCliCompute(_ScriptMixin, unittest.TestCase):
    """`runtime.py compute` argument parsing and dispatch."""

    def _run_cli(self, argv: list[str]) -> tuple[int, str]:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            rc = runtime.main(argv)
            return rc, buf.getvalue()

    def test_inspect_accepts_the_backend_flag_in_either_position(self):
        """Verify --backend parses before or after the subcommand, JSON or not."""
        variants = [
            ["compute", "--backend", "mock_test", "inspect", "--json"],
            ["compute", "inspect", "--backend", "mock_test", "--json"],
            # Without --json the output is pretty-printed but still JSON.
            ["compute", "--backend", "mock_test", "inspect"],
        ]
        for argv in variants:
            with self.subTest(" ".join(argv)):
                rc, out = self._run_cli(argv)
                self.assertEqual(rc, 0)
                payload = json.loads(out.strip())
                self.assertEqual(payload["accelerator"], "mock_tpu")
                self.assertEqual(payload["chips"], 4)

    def test_run_dispatches_to_the_driver(self):
        """Verify `compute run` reaches the resolved driver with the script path."""
        path = self._script("print('from_runtime_cli')\n")
        rc, out = self._run_cli([
            "compute", "--backend", "mock_test", "run", "--code", path,
            "--json"
        ])
        self.assertEqual(rc, 0)
        payload = json.loads(out.strip())
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("mock run executed", payload["stdout"])

    def test_run_forwards_the_name_flag(self):
        """Verify `compute run --name` reaches the driver as `job_name`."""
        path = self._script("print('named')\n")
        with patch.object(MockTestDriver,
                          "run",
                          autospec=True,
                          side_effect=MockTestDriver.run) as run:
            self._run_cli([
                "compute", "--backend", "mock_test", "run", "--code", path,
                "--name", "qwen-bench"
            ])
        self.assertEqual(run.call_args.kwargs["job_name"], "qwen-bench")

    def test_run_executes_python_and_bash_locally(self):
        """Verify the local backend runs both script kinds end to end."""
        cases = [
            (".py", "print('from_python')\n", "from_python"),
            (".sh", "#!/bin/bash\necho 'from_bash'\n", "from_bash"),
        ]
        for suffix, body, expected in cases:
            with self.subTest(suffix):
                path = self._script(body, suffix=suffix)
                rc, out = self._run_cli([
                    "compute", "--backend", "local", "run", "--code", path,
                    "--json"
                ])
                self.assertEqual(rc, 0)
                self.assertIn(expected, json.loads(out.strip())["stdout"])

    def test_run_forwards_the_interpreter_flag(self):
        """Verify --interpreter selects the binary that executes the script."""
        path = self._script("import sys; print(sys.executable)\n")
        rc, out = self._run_cli([
            "compute", "--backend", "local", "run", "--interpreter",
            sys.executable, "--code", path, "--json"
        ])
        self.assertEqual(rc, 0)
        self.assertIn(sys.executable, json.loads(out.strip())["stdout"])

    def test_run_input_and_output_specs_are_parsed_and_validated(self):
        """Verify `--input LOCAL[:REMOTE]` and `--output REMOTE[:LOCAL]` reach the driver."""
        self.assertEqual(StagedFile.parse_input("a/b.json"),
                         StagedFile(local="a/b.json", remote="b.json"))
        self.assertEqual(StagedFile.parse_input("a/b.json:sub/c.json"),
                         StagedFile(local="a/b.json", remote="sub/c.json"))
        self.assertEqual(StagedFile.parse_output("r.json"),
                         StagedFile(local="r.json", remote="r.json"))
        self.assertEqual(StagedFile.parse_output("r.json:out/r.json"),
                         StagedFile(local="out/r.json", remote="r.json"))
        for bad in ("/abs/path.json", "../up.json", "a/../../b"):
            with self.assertRaises(ComputeConfigError):
                StagedFile.parse_output(bad)

        path = self._script("print('x')\n")
        with io.StringIO() as err, contextlib.redirect_stderr(err):
            rc, _ = self._run_cli([
                "compute", "--backend", "mock_test", "run", "--code", path,
                "--input", "/definitely/missing.json", "--json"
            ])
            self.assertNotEqual(rc, 0)
            self.assertEqual(
                json.loads(err.getvalue())["error_code"], "INPUT_NOT_FOUND")

    def test_staged_local_paths_expand_a_leading_tilde(self):
        """`~` after the colon reaches us unexpanded; the prompts rely on it.

        `--output r.json:~/.tpu-agent-suite/runs/...` is how a prompt lands an
        output in its run directory, and the shell leaves that `~` alone.
        """
        home = os.path.expanduser("~")
        self.assertEqual(
            StagedFile.parse_output("x.xplane.pb:~/runs/r/x.xplane.pb"),
            StagedFile(local=f"{home}/runs/r/x.xplane.pb",
                       remote="x.xplane.pb"))
        self.assertEqual(StagedFile.parse_input("~/in.json"),
                         StagedFile(local=f"{home}/in.json", remote="in.json"))
        # The target side is never a home directory, so `~` there stays an
        # ordinary (relative) name.
        self.assertEqual(StagedFile.parse_output("~x.json"),
                         StagedFile(local="~x.json", remote="~x.json"))

    def test_submit_without_json_prints_only_the_job_id(self):
        """Verify plain `submit` output is capturable as `JOB_ID=$(...)`.

        The executor prompts do exactly that and then pass the value to
        `--job-id`, so anything but the bare id on stdout breaks polling.
        """
        path = self._script("print('submitted')\n")
        rc, out = self._run_cli([
            "compute", "--backend", "mock_test", "submit", "--recipe", path,
            "--run-id", "20260101T000000Z-123456789abc"
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "mock-job-999")

    def test_submit_and_job_lifecycle_subcommands(self):
        """Verify submit, job status, job logs, and job cancel all route through."""
        path = self._script("print('submitted')\n")
        rc, out = self._run_cli([
            "compute", "--backend", "mock_test", "submit", "--recipe", path,
            "--run-id", "20260101T000000Z-123456789abc", "--json"
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.strip())["job_id"], "mock-job-999")

        rc, out = self._run_cli([
            "compute", "--backend", "mock_test", "job", "status", "--job-id",
            "mock-job-999", "--json"
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.strip())["status"], "RUNNING")

        rc, out = self._run_cli([
            "compute", "--backend", "mock_test", "job", "logs", "--job-id",
            "mock-job-999"
        ])
        self.assertEqual(rc, 0)
        self.assertIn("mock logs for mock-job-999", out)

        # A real driver's submit leaves a record, which cancel checks.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            jobs.JobRecord.create(job_id="mock-job-999", backend="mock_test")
            rc, _ = self._run_cli([
                "compute", "--backend", "mock_test", "job", "cancel",
                "--job-id", "mock-job-999"
            ])
        self.assertEqual(rc, 0)

    def test_driver_declared_flags_reach_param_overrides(self):
        """Verify dynamically mounted driver flags are forwarded to the resolver."""
        mock_driver = MagicMock()
        mock_driver.inspect.return_value = MockTestDriver().inspect()
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            # Patch the resolver where the CLI reads it, so the test exercises
            # the real lookup path rather than a package-level alias.
            with patch.object(runtime.compute.config,
                              "resolve_driver",
                              return_value=mock_driver) as mock_resolve:
                rc = runtime.main([
                    "compute", "--backend", "mock_test", "--endpoint",
                    "https://custom.mock:8080", "--token", "custom-token-xyz",
                    "inspect", "--json"
                ])
        self.assertEqual(rc, 0)
        _, kwargs = mock_resolve.call_args
        self.assertEqual(kwargs["backend_override"], "mock_test")
        self.assertEqual(kwargs["param_overrides"]["endpoint"],
                         "https://custom.mock:8080")
        self.assertEqual(kwargs["param_overrides"]["token"],
                         "custom-token-xyz")


def _free_port() -> int:
    """A loopback port nothing is bound to, for a forward to bind or miss."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestSpawnPortForward(unittest.TestCase):
    """The liveness check behind every backgrounded port forward."""

    def setUp(self):
        home = tempfile.mkdtemp()
        patcher = patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, home, True)

    def test_a_forward_that_dies_is_reported_as_dead(self):
        """The pid alone proves nothing: Popen returns one either way."""
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward(
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                "dead-1",
                local_port=_free_port())
        self.assertEqual(ctx.exception.error_code, "PORT_FORWARD_FAILED")
        self.assertIn("code 7", str(ctx.exception))

    def test_the_failure_is_quoted_back_and_kept(self):
        """`/dev/null` is why this failure used to be undiagnosable."""
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward([
                sys.executable, "-c",
                "import sys; sys.stderr.write('unable to forward port\n');"
                " sys.exit(1)"
            ],
                                    "dead-2",
                                    local_port=_free_port())
        message = str(ctx.exception)
        self.assertIn("unable to forward port", message)
        log_path = Path(message.rsplit("Log: ", 1)[1].strip())
        self.assertIn("unable to forward port", log_path.read_text())

    def test_a_forward_that_binds_its_port_is_ready(self):
        """The socket answering is the proof; the poll ends as soon as it does."""
        script = ("import socket, time;"
                  " s = socket.socket();"
                  " s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
                  " s.bind(('127.0.0.1', PORT)); s.listen(5); time.sleep(30)")
        port = _free_port()
        pid = base.spawn_port_forward(
            [sys.executable, "-c",
             script.replace("PORT", str(port))],
            "ready-1",
            local_port=port,
        )
        self.addCleanup(_terminate_quietly, pid)
        self.assertGreater(pid, 0)

    def test_a_forward_that_never_binds_is_not_called_established(self):
        """Alive is not the same as serving -- this is the silent failure."""
        port = _free_port()
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                "never-1",
                local_port=port,
                ready_timeout_sec=0.5,
            )
        self.assertEqual(ctx.exception.error_code, "PORT_FORWARD_FAILED")
        self.assertIn("nothing is listening", str(ctx.exception))

    def test_a_forward_that_dies_late_is_still_caught(self):
        """The old fixed sleep returned a pid for anything slower than it."""
        port = _free_port()
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward([
                sys.executable, "-c", "import sys, time; time.sleep(1.5);"
                " sys.stderr.write('ssh: connect to host: timed out\\n');"
                " sys.exit(255)"
            ],
                                    "late-1",
                                    local_port=port,
                                    ready_timeout_sec=10.0)
        self.assertIn("code 255", str(ctx.exception))
        self.assertIn("timed out", str(ctx.exception))


class TestComputeCliContracts(_ScriptMixin, unittest.TestCase):
    """handle_compute_cli behaviour reachable without the runtime parser."""

    def _args(self, **overrides: Any) -> argparse.Namespace:
        base_args = {
            "compute_action": "inspect",
            "backend": "mock_test",
            "workspace": None,
            "json": True,
            "job_id": "job-1",
            "out_dir": None,
        }
        base_args.update(overrides)
        return argparse.Namespace(**base_args)

    def _invoke(self, args: argparse.Namespace, driver: Any):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            rc = compute.handle_compute_cli(args, resolve=lambda **_: driver)
            return rc, buf.getvalue()

    def test_resolver_is_injectable(self):
        """Verify an injected resolver replaces the default without patching modules."""
        mock_driver = MagicMock()
        mock_driver.inspect.return_value = MockTestDriver().inspect()
        calls: list[dict[str, Any]] = []

        def fake_resolve(**kwargs: Any):
            calls.append(kwargs)
            return mock_driver

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            rc = compute.handle_compute_cli(self._args(), resolve=fake_resolve)
            payload = json.loads(buf.getvalue())

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["backend_override"], "mock_test")
        self.assertEqual(payload["chips"], 4)

    def test_missing_tool_is_a_structured_error_not_a_traceback(self):
        """A missing kubectl crashed inspect with a traceback."""
        driver = MagicMock(required_tools=base.GKE_TOOLS)
        with patch.object(base.shutil, "which", return_value=None):
            with self.assertRaises(ComputeConfigError) as ctx:
                self._invoke(self._args(), driver)
            self.assertEqual(ctx.exception.error_code, "MISSING_TOOL")
            self.assertIn("gcloud components install kubectl",
                          ctx.exception.message)
            self.assertIn("gke-gcloud-auth-plugin", ctx.exception.message)
            driver.inspect.assert_not_called()
            # Preflight reports the tools itself, so it still runs.
            driver.preflight.return_value.to_dict.return_value = {
                "status": "fail"
            }
            self._invoke(self._args(compute_action="preflight"), driver)
            driver.preflight.assert_called_once()

    def test_job_subcommands_emit_parseable_envelopes(self):
        """Verify each `job` subcommand prints a JSON envelope or plain text."""
        driver = MagicMock()
        driver.fetch_logs.return_value = "line one\nline two"
        driver.fetch_artifacts.return_value = [Path("/tmp/a.pb")]

        _, out = self._invoke(
            self._args(compute_action="job", job_action="logs"), driver)
        payload = json.loads(out)
        self.assertEqual(payload["job_id"], "job-1")
        self.assertIn("line one", payload["logs"])

        _, plain = self._invoke(
            self._args(compute_action="job", job_action="logs", json=False),
            driver)
        self.assertIn("line one", plain)
        self.assertNotIn('"job_id"', plain)

        _, out = self._invoke(
            self._args(compute_action="job", job_action="artifacts"), driver)
        payload = json.loads(out)
        self.assertEqual(payload["artifacts"], ["/tmp/a.pb"])
        self.assertIn("out_dir", payload)
        # cancel's envelope: test_job_cancel_only_deletes_this_runs_jobs.

    def test_job_cancel_only_deletes_this_runs_jobs(self):
        """`job cancel` only cancels jobs this run submitted."""
        mine = "20261007T151000Z-0123456789ab"
        other = "20261007T150000Z-ba9876543210"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            jobs.JobRecord.create(job_id="theirs", run_id=other)
            jobs.JobRecord.create(job_id="ours", run_id=mine)
            for job_id, why in (("someone-elses", "tell them compute can't"),
                                ("theirs", f"belongs to run {other}")):
                with self.subTest(job_id):
                    driver = MagicMock()
                    with self.assertRaises(ComputeConfigError) as ctx:
                        self._invoke(
                            self._args(compute_action="job",
                                       job_action="cancel",
                                       job_id=job_id,
                                       run_id=mine), driver)
                    self.assertEqual(ctx.exception.error_code,
                                     compute.cli.NOT_OWN_JOB)
                    self.assertIn(why, ctx.exception.message)
                    driver.cancel.assert_not_called()
            # This run's job, and the other run's when it is named outright.
            for job_id, run_id in (("ours", mine), ("theirs", other)):
                with self.subTest(job_id=job_id, run_id=run_id):
                    driver = MagicMock()
                    rc, out = self._invoke(
                        self._args(compute_action="job",
                                   job_action="cancel",
                                   job_id=job_id,
                                   run_id=run_id), driver)
                    self.assertEqual(rc, 0)
                    driver.cancel.assert_called_once_with(job_id)
                    self.assertEqual(json.loads(out), {
                        "job_id": job_id,
                        "status": "CANCELLED"
                    })

    def test_job_action_requires_job_id(self):
        """status, logs, artifacts, cancel require --job-id.

        Raised as a ComputeConfigError so the runtime's one error handler
        renders it -- JSON under --json, prose otherwise -- like every
        other configuration failure.
        """
        driver = MagicMock()
        for action in ("status", "logs", "artifacts", "cancel"):
            with self.subTest(action):
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._invoke(
                        self._args(compute_action="job",
                                   job_action=action,
                                   job_id=None), driver)
                self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")
                self.assertIn("--job-id", str(ctx.exception))

    def test_job_artifacts_fails_when_a_declared_output_never_arrived(self):
        """Exit 0 with nothing to read is what makes an agent invent numbers."""
        driver = MagicMock()
        driver.fetch_artifacts.side_effect = lambda jid, dest: [
            dest / f"{jid}.log"
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            jobs.JobRecord.create(job_id="job-empty",
                                  extra={
                                      "outputs": [{
                                          "local":
                                          "autotune_summary.json",
                                          "remote":
                                          "autotune_summary.json",
                                      }]
                                  })
            with tempfile.TemporaryDirectory() as dest:
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="artifacts",
                               job_id="job-empty",
                               out_dir=dest),
                    driver,
                )
        self.assertEqual(rc, 1)
        self.assertEqual(
            json.loads(out)["missing_outputs"], ["autotune_summary.json"])

    def test_job_artifacts_is_a_success_once_the_outputs_are_there(self):
        driver = MagicMock()

        def _fetch(jid, dest):
            (dest / "eval").mkdir(parents=True, exist_ok=True)
            (dest / "eval" / "metrics.json").write_text("{}")
            return [dest / "eval" / "metrics.json"]

        driver.fetch_artifacts.side_effect = _fetch
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            jobs.JobRecord.create(job_id="job-full",
                                  extra={
                                      "outputs": [{
                                          "local":
                                          "eval/metrics.json",
                                          "remote":
                                          "eval/metrics.json",
                                      }]
                                  })
            with tempfile.TemporaryDirectory() as dest:
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="artifacts",
                               job_id="job-full",
                               out_dir=dest),
                    driver,
                )
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["missing_outputs"], [])

    def test_job_artifacts_reports_where_each_output_landed(self):
        """Agents looked for outputs at a submit-time ':LOCAL'."""
        driver = MagicMock()

        def _fetch(jid, dest):
            (dest / "r.json").write_text("{}")
            return [dest / "r.json"]

        driver.fetch_artifacts.side_effect = _fetch
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            dest = Path(tmp) / "out"
            jobs.JobRecord.create(
                job_id="j",
                extra={"outputs": [{
                    "local": "x/r.json",
                    "remote": "r.json"
                }]})
            rc, out = self._invoke(
                self._args(compute_action="job",
                           job_action="artifacts",
                           job_id="j",
                           out_dir=str(dest)), driver)
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(out)["outputs"], {"r.json": str(dest / "r.json")})

    def test_submit_maps_outputs_and_warns_that_local_is_ignored(self):
        """Only a spec with a ':LOCAL' warns; every output is mapped."""
        driver = MagicMock()
        driver.submit.return_value = "job-s"
        driver.get_status.return_value = AsynchronousExecutionResult(
            "job-s", "RUNNING")
        run_id = "20260101T000000Z-cccccccccccc"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}), \
                patch.object(compute.cli.lease, "spawn_watcher"), \
                io.StringIO() as err, contextlib.redirect_stderr(err):
            recipe = Path(tmp) / "recipe.py"
            recipe.write_text("print('hi')\n")
            rc, out = self._invoke(
                self._args(compute_action="submit",
                           recipe=str(recipe),
                           name="s",
                           submit_args=[],
                           interpreter=None,
                           run_id=run_id,
                           output=["results.json:out/r.json", "a.json"]),
                driver)
            stderr = err.getvalue()
            landing = utils.get_run_artifacts_dir(run_id) / "job-s"
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(out)["outputs"], {
                "results.json": str(landing / "results.json"),
                "a.json": str(landing / "a.json")
            })
        self.assertIn(
            f"ignores ':out/r.json'; results.json will land at "
            f"{landing / 'results.json'}", stderr)
        self.assertEqual(stderr.count("warning:"), 1)

    def test_job_artifacts_declaring_nothing_has_nothing_to_miss(self):
        """The batch jobs that predate `--output` must still exit 0."""
        driver = MagicMock()
        driver.fetch_artifacts.side_effect = lambda jid, dest: []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            jobs.JobRecord.create(job_id="job-plain")
            with tempfile.TemporaryDirectory() as dest:
                rc, _ = self._invoke(
                    self._args(compute_action="job",
                               job_action="artifacts",
                               job_id="job-plain",
                               out_dir=dest),
                    driver,
                )
        self.assertEqual(rc, 0)

    def test_job_artifacts_defaults_to_run_artifact_dir(self):
        """When --out-dir is omitted, artifacts default to {{runs}}/<run_id>/artifacts/<job_id>/."""
        driver = MagicMock()
        driver.fetch_artifacts.side_effect = lambda jid, dest: [
            dest / "output.bin"
        ]
        valid_run_id = "20260101T000000Z-123456789abc"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)}):
                # 1. JobRecord has run_id
                record = jobs.JobRecord.create(
                    job_id="job-custom-42",
                    job_name="test-job",
                    run_id=valid_run_id,
                )
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="artifacts",
                               job_id="job-custom-42",
                               out_dir=None),
                    driver,
                )
                self.assertEqual(rc, 0)
                payload = json.loads(out)
                expected_dir = tmp_path / "runs" / valid_run_id / "artifacts" / "job-custom-42"
                self.assertEqual(payload["out_dir"], str(expected_dir))
                self.assertTrue(expected_dir.is_dir())

                # 2. No run anywhere: cwd / job_id, so two jobs still cannot
                # collide. Run from the temp dir so the directory it makes
                # does not land in the repository.
                jobs.JobRecord.create(job_id="job-no-run",
                                      job_name="test-job-2",
                                      run_id="")
                scratch = tmp_path / "cwd"
                scratch.mkdir()
                previous = os.getcwd()
                os.chdir(scratch)
                try:
                    rc, out = self._invoke(
                        self._args(compute_action="job",
                                   job_action="artifacts",
                                   job_id="job-no-run",
                                   out_dir=None,
                                   run_id=None,
                                   workspace=scratch),
                        driver,
                    )
                finally:
                    os.chdir(previous)
                self.assertEqual(rc, 0)
                payload = json.loads(out)
                self.assertEqual(payload["out_dir"],
                                 str(scratch / "job-no-run"))

    def test_job_list_and_reap(self):
        """Verify compute job list filters records and compute job reap cancels active jobs."""
        run_a = "20260101T000000Z-aaaaaaaaaaaa"
        run_b = "20260101T000000Z-bbbbbbbbbbbb"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)}):
                # Create some test jobs
                j1 = jobs.JobRecord.create(job_id="job-101",
                                           job_name="train",
                                           run_id=run_a,
                                           backend="mock_test")
                j1.mark("RUNNING")
                j2 = jobs.JobRecord.create(job_id="job-102",
                                           job_name="eval",
                                           run_id=run_a,
                                           backend="mock_test")
                j2.mark("COMPLETED")
                j3 = jobs.JobRecord.create(job_id="job-103",
                                           job_name="tune",
                                           run_id=run_b,
                                           backend="mock_test")
                j3.mark("RUNNING")

                # List all jobs as JSON
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status=None),
                    MagicMock(),
                )
                self.assertEqual(rc, 0)
                records = json.loads(out)
                self.assertEqual(len(records), 3)

                # Filter by run_id
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=run_a,
                               status=None),
                    MagicMock(),
                )
                self.assertEqual(rc, 0)
                records = json.loads(out)
                self.assertEqual(len(records), 2)
                self.assertEqual({r["job_id"]
                                  for r in records}, {"job-101", "job-102"})

                # Filter by status
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status="running"),
                    MagicMock(),
                )
                self.assertEqual(rc, 0)
                records = json.loads(out)
                self.assertEqual(len(records), 2)
                self.assertEqual({r["job_id"]
                                  for r in records}, {"job-101", "job-103"})

                # List in plain text
                rc, plain = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status=None,
                               json=False),
                    MagicMock(),
                )
                self.assertEqual(rc, 0)
                self.assertIn("job-101", plain)
                self.assertIn("train", plain)

                # Reap jobs for run-A. A driver's cancel records the
                # outcome itself; the reaper reads the record back.
                cancelled_jobs: list[str] = []

                def fake_cancel(jid: str) -> None:
                    cancelled_jobs.append(jid)
                    jobs.JobRecord.load(jid).mark("CANCELLED")

                mock_driver = MagicMock()
                mock_driver.cancel.side_effect = fake_cancel

                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="reap",
                               run_id=run_a),
                    mock_driver,
                )
                self.assertEqual(rc, 0)
                payload = json.loads(out)
                self.assertEqual(payload["run_id"], run_a)
                self.assertEqual(payload["reaped"], ["job-101"])
                self.assertEqual(cancelled_jobs, ["job-101"])

                # Check job-101 is now CANCELLED on disk
                reloaded = jobs.JobRecord.load("job-101")
                self.assertEqual(reloaded.status, "CANCELLED")

    def test_job_list_refreshes_the_active_runs_unfinished_jobs(self):
        """A record says PENDING from submit until the job ends unless a
        status read updates it, so `job list` first asks the backend of each
        unfinished job in the run -- once per backend if it cannot answer."""

        def answering(**result: Any) -> MagicMock:
            driver = MagicMock()
            driver.get_status.side_effect = (
                lambda jid: AsynchronousExecutionResult(job_id=jid, **result))
            return driver

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": tmp}):
            active, older = ("20260101T000000Z-000000000001",
                             "20260101T000000Z-000000000000")
            utils.record_active_run(active, workspace=Path(tmp))
            for jid in ("a", "b"):
                jobs.JobRecord.create(job_id=jid, run_id=active, backend="x")
            jobs.JobRecord.create(job_id="done", run_id=active,
                                  backend="x").mark("COMPLETED")
            jobs.JobRecord.create(job_id="old-run", run_id=older, backend="x")

            def listed(driver: MagicMock, **flags: Any) -> dict[str, Any]:
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status=None,
                               workspace=Path(tmp),
                               **flags), driver)
                self.assertEqual(rc, 0)
                return {r["job_id"]: r for r in json.loads(out)}

            running = answering(status="RUNNING")
            self.assertEqual(
                listed(running, no_refresh=True)["a"]["status"], "PENDING")
            running.get_status.assert_not_called()

            # Only the active run's unfinished jobs are asked about; an older
            # run's record may name a cluster today's config does not.
            rows = listed(running)
            self.assertEqual([rows[j]["status"] for j in ("a", "b", "done")],
                             ["RUNNING", "RUNNING", "COMPLETED"])
            self.assertEqual(rows["old-run"]["status"], "PENDING")
            self.assertEqual(running.get_status.call_count, 2)

            # A target that cannot answer -- by raising, or as UNKNOWN with
            # an error -- is asked once, and every job on it says so. A
            # target is a backend and profile: "c" is a second one.
            jobs.JobRecord.create(job_id="c",
                                  run_id=active,
                                  backend="x",
                                  compute_profile="other")
            raising = MagicMock()
            raising.get_status.side_effect = RuntimeError("no cluster")
            for driver, why in ((raising, "no cluster"),
                                (answering(status="UNKNOWN",
                                           error="ssh: timed out"),
                                 "ssh: timed out")):
                with self.subTest(why=why):
                    for jid in ("a", "b"):
                        jobs.JobRecord.load(jid).mark("PENDING")
                    rows = listed(driver)
                    self.assertEqual(driver.get_status.call_count, 2)
                    for jid in ("a", "b"):
                        self.assertEqual(rows[jid]["status"], "PENDING")
                        self.assertIn(why, rows[jid]["refresh_error"])
                    self.assertNotIn("refresh_error", rows["old-run"])

            # A Job deleted outside compute is its own error: the target
            # answered for the rest.
            def a_is_gone(jid: str) -> AsynchronousExecutionResult:
                if jid == "a":
                    raise ComputeConfigError("deleted",
                                             error_code=compute.cli.JOB_GONE)
                return AsynchronousExecutionResult(job_id=jid,
                                                   status="RUNNING")

            gone = MagicMock()
            gone.get_status.side_effect = a_is_gone
            rows = listed(gone)
            self.assertEqual(rows["a"]["refresh_error"], "deleted")
            self.assertEqual(rows["b"]["status"], "RUNNING")
            self.assertNotIn("refresh_error", rows["b"])

    def test_job_reap_by_lease(self):
        """`--orphaned` needs no run, `--watch` one; both are tested in
        test_lease."""
        resolve = MagicMock()

        def reap(**flags):
            flags.setdefault("run_id", None)
            args = self._args(compute_action="job", job_action="reap", **flags)
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                self.assertEqual(compute.handle_compute_cli(args, resolve), 0)
                return json.loads(buf.getvalue())

        sweep = ({"run-a": ["job-1"]}, {})
        with patch.object(compute.lease,
                          "reap_orphaned_jobs",
                          return_value=sweep):
            self.assertEqual(reap(orphaned=True)["reaped"], sweep[0])
        watched = ([], {"job-2": "x"})
        with patch.object(compute.lease, "watch",
                          return_value=watched) as watch:
            self.assertEqual(
                reap(watch=True, run_id="run-b")["failed"], watched[1])
        watch.assert_called_once_with("run-b", resolve)

    def test_submit_after_the_session_ended(self):
        """A batch job starts its run's watcher; an ended session's active
        run is no fallback, as a job tagged with it would be reaped."""
        driver = MagicMock()
        driver.submit.return_value = "job-1"
        driver.get_status.return_value = AsynchronousExecutionResult(
            job_id="job-1", status="RUNNING")
        run_id = "20260101T000000Z-123456789abc"
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        utils.record_active_run(run_id, workspace=tmp)

        def submit(**overrides):
            return self._invoke(
                self._args(compute_action="submit",
                           recipe=self._script("echo hi\n", suffix=".sh"),
                           name=None,
                           submit_args=[],
                           interpreter=None,
                           workspace=str(tmp),
                           **overrides), driver)[0]

        with patch.object(compute.lease, "spawn_watcher") as spawn, \
                patch.object(compute.lease, "has_ended", return_value=True):
            with self.assertRaisesRegex(base.ComputeConfigError, "has ended"):
                submit(run_id=None)
            self.assertEqual(submit(run_id=run_id), 0)
            self.assertEqual(submit(run_id=run_id, serve=True), 0)
        spawn.assert_called_once_with(run_id)

    def test_submit_tags_job_record_with_run_id(self):
        """compute submit passes run_id to driver and tags the JobRecord."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)}):
                driver = LocalDriver()
                recipe_path = tmp_path / "test_recipe.py"
                recipe_path.write_text("print('hello')\n")

                run_id = "20260101T000000Z-123456789abc"
                rc, out = self._invoke(
                    self._args(
                        compute_action="submit",
                        recipe=str(recipe_path),
                        name="test-run-tag",
                        submit_args=[],
                        interpreter=None,
                        run_id=run_id,
                    ),
                    driver,
                )
                self.assertEqual(rc, 0)
                payload = json.loads(out)
                job_id = payload["job_id"]
                record = jobs.JobRecord.load(job_id)
                self.assertIsNotNone(record)
                self.assertEqual(record.run_id, run_id)
                self.assertEqual(record.backend, "local")
                driver.cancel(job_id)

    def test_submit_reports_initial_status_failure(self):
        """Verify a failing initial status check is reported, not hidden as PENDING.

        The job was submitted, so its id must still reach the caller -- but the
        optimistic "PENDING" must not be passed off as an observed state.
        """
        driver = MagicMock()
        driver.submit.return_value = "job-123"
        driver.get_status.side_effect = RuntimeError("cluster unreachable")

        rc, out = self._invoke(
            self._args(compute_action="submit",
                       recipe=self._script("echo hi\n", suffix=".sh"),
                       name=None,
                       submit_args=[],
                       interpreter=None,
                       run_id="20260101T000000Z-123456789abc"), driver)
        payload = json.loads(out)
        self.assertEqual(rc, 0)
        self.assertEqual(payload["job_id"], "job-123")
        self.assertEqual(payload["status"], "PENDING")
        self.assertIn("cluster unreachable", payload["status_error"])

    def test_submit_with_serve_and_staged_files(self):
        """Verify compute submit accepts --serve, --input, and --output."""
        driver = MagicMock()
        driver.submit.return_value = "job-serv-1"
        driver.get_status.return_value = AsynchronousExecutionResult(
            job_id="job-serv-1", status="RUNNING")
        recipe = self._script("print('server')\n")
        in_file = self._script("weights\n")
        rc, out = self._invoke(
            self._args(
                compute_action="submit",
                recipe=recipe,
                name="server-test",
                submit_args=[],
                interpreter=None,
                run_id="20260101T000000Z-123456789abc",
                serve=True,
                input=[f"{in_file}:weights.bin"],
                output=["metrics.json"],
            ),
            driver,
        )
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["job_id"], "job-serv-1")
        self.assertTrue(payload["serve"])
        driver.submit.assert_called_once()
        _, kwargs = driver.submit.call_args
        self.assertTrue(kwargs["serve"])
        self.assertEqual(len(kwargs["inputs"]), 1)
        self.assertEqual(kwargs["inputs"][0].remote, "weights.bin")
        self.assertEqual(len(kwargs["outputs"]), 1)
        self.assertEqual(kwargs["outputs"][0].remote, "metrics.json")

    def test_port_forward_cli_dispatches_to_driver(self):
        """Verify compute port-forward parses [LOCAL:]REMOTE and calls driver."""
        driver = MagicMock()
        driver.port_forward.return_value = 0
        rc, _ = self._invoke(
            self._args(
                compute_action="port-forward",
                port="8000",
                job_id=None,
            ),
            driver,
        )
        self.assertEqual(rc, 0)
        driver.port_forward.assert_called_with(local_port=8000,
                                               remote_port=8000,
                                               job_id=None,
                                               background=False)

        driver.reset_mock()
        rc, _ = self._invoke(
            self._args(
                compute_action="port-forward",
                port="8080:8000",
                job_id="job-456",
            ),
            driver,
        )
        self.assertEqual(rc, 0)
        driver.port_forward.assert_called_with(local_port=8080,
                                               remote_port=8000,
                                               job_id="job-456",
                                               background=False)

        driver.reset_mock()
        driver.port_forward.return_value = 12345
        rc, out = self._invoke(
            self._args(
                compute_action="port-forward",
                port="8000",
                job_id=None,
                background=True,
            ),
            driver,
        )
        self.assertEqual(rc, 0)
        driver.port_forward.assert_called_with(local_port=8000,
                                               remote_port=8000,
                                               job_id=None,
                                               background=True)
        self.assertIn("12345", out)

    def test_port_forward_rejects_a_port_that_is_not_one(self):
        """Rejected here, not deep inside kubectl, where the message names
        neither the flag nor the number."""
        driver = MagicMock()
        for spec in ("0", "-1", "99999", "8080:70000", "not_a_port", "8080:x"):
            with self.assertRaises(compute.ComputeConfigError) as ctx:
                self._invoke(
                    self._args(
                        compute_action="port-forward",
                        port=spec,
                        job_id=None,
                    ),
                    driver,
                )
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        driver.port_forward.assert_not_called()

    def test_port_forward_gce_tpu_vm(self):
        driver = compute.GceTpuVmDriver(tpu_name="v4-node",
                                        zone="us-central2-b",
                                        project="test-proj")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[],
                                                                returncode=0)
            rc = driver.port_forward(local_port=8080, remote_port=8000)
            self.assertEqual(rc, 0)
            cmd = mock_run.call_args[0][0]
            self.assertIn("-L8080:localhost:8000", cmd)
            self.assertIn("-N", cmd)

    def test_port_forward_gke_pod(self):
        driver = compute.GkePodDriver(namespace="default", pod="my-pod")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[],
                                                                returncode=0)
            rc = driver.port_forward(local_port=8080, remote_port=8000)
            self.assertEqual(rc, 0)
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd, [
                "kubectl", "-n", "default", "port-forward", "pod/my-pod",
                "8080:8000"
            ])

    def test_port_forward_local(self):
        driver = compute.LocalDriver()
        rc = driver.port_forward(local_port=8000, remote_port=8000)
        self.assertEqual(rc, 0)

    def test_port_forward_local_refuses_a_remapping(self):
        """Nothing would be listening on the local port it was asked for."""
        driver = compute.LocalDriver()
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            driver.port_forward(local_port=9000,
                                remote_port=8000,
                                background=True)
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_OPERATION")

    def test_port_forward_unsupported_on_xpk(self):
        driver = compute.GkeXpkDriver(cluster_name="test-cl",
                                      zone="us-central1-a",
                                      project="test-p")
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            driver.port_forward(local_port=8000, remote_port=8000)
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_OPERATION")

    def test_missing_input_fails_before_the_driver_is_asked(self):
        """Verify a missing --code/--recipe fails against the flag the user typed.

        Drivers validate again for programmatic callers, but checking here keeps
        the failure from surfacing midway through SSH or cloud staging.
        """
        driver = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            cases = [
                ("run",
                 dict(compute_action="run",
                      code=os.path.join(tmp, "nonexistent", "x.py"),
                      timeout=None,
                      run_args=[],
                      interpreter=None)),
                ("submit",
                 dict(compute_action="submit",
                      recipe=os.path.join(tmp, "nonexistent", "r.sh"),
                      name=None,
                      submit_args=[],
                      interpreter=None)),
            ]
            for action, overrides in cases:
                with self.subTest(action):
                    with self.assertRaises(ComputeConfigError) as ctx:
                        self._invoke(self._args(**overrides), driver)
                    self.assertEqual(ctx.exception.error_code,
                                     "INPUT_NOT_FOUND")
                    self.assertIn("does not exist", ctx.exception.message)
        driver.run.assert_not_called()
        driver.submit.assert_not_called()

    def test_invalid_run_id_fails_before_the_driver_is_asked(self):
        """A malformed --run-id is INVALID_FIELD, and nothing is dispatched.

        Checked after the dispatch, it left a job running on the target
        that no run could find, and the job id was never printed.
        """
        driver = MagicMock()
        cases = [
            ("run",
             dict(compute_action="run",
                  code=self._script("print('hi')\n"),
                  timeout=None,
                  run_args=[],
                  interpreter=None)),
            ("submit",
             dict(compute_action="submit",
                  recipe=self._script("echo hi\n", suffix=".sh"),
                  name=None,
                  submit_args=[],
                  interpreter=None)),
        ]
        for action, overrides in cases:
            with self.subTest(action), \
                    patch.object(compute.lease, "spawn_watcher") as spawn:
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._invoke(self._args(run_id="test-run1", **overrides),
                                 driver)
                self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
                self.assertIn("invalid run ID", ctx.exception.message)
                spawn.assert_not_called()
        driver.run.assert_not_called()
        driver.submit.assert_not_called()

    def test_job_fallback_to_active_run_id(self):
        """Artifacts, reap, and submit fall back to utils.get_active_run_id."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            active_run = "20260101T000000Z-999999999999"
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)},
                            clear=False):
                utils.record_active_run(active_run, workspace=tmp_path)

                driver = LocalDriver()
                # The recipe must still be running when reap looks at it:
                # submit settles the status right after launch, and a script
                # that has already exited is COMPLETED, which reap skips.
                recipe_path = tmp_path / "test_active_recipe.py"
                recipe_path.write_text("import time\ntime.sleep(60)\n")

                # 1. Submit fallback to active run id
                rc, out = self._invoke(
                    self._args(
                        compute_action="submit",
                        recipe=str(recipe_path),
                        name="test-active",
                        submit_args=[],
                        interpreter=None,
                        run_id=None,
                        workspace=tmp_path,
                    ),
                    driver,
                )
                self.assertEqual(rc, 0)
                job_id = json.loads(out)["job_id"]
                rec = jobs.JobRecord.load(job_id)
                self.assertEqual(rec.run_id, active_run)

                # 2. Reap fallback to active run id
                rc, out = self._invoke(
                    self._args(
                        compute_action="job",
                        job_action="reap",
                        run_id=None,
                        workspace=tmp_path,
                    ),
                    driver,
                )
                self.assertEqual(rc, 0)
                reap_payload = json.loads(out)
                self.assertEqual(reap_payload["run_id"], active_run)
                self.assertIn(job_id, reap_payload["reaped"])

    def test_submit_without_any_run_is_refused(self):
        """No --run-id and no active run: refuse rather than tag with ''.

        A job tagged with no run is one the run-end reaper can never find.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)},
                            clear=False):
                recipe_path = tmp_path / "recipe.py"
                recipe_path.write_text("print('hi')\n")
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._invoke(
                        self._args(
                            compute_action="submit",
                            recipe=str(recipe_path),
                            name="untagged",
                            submit_args=[],
                            interpreter=None,
                            run_id=None,
                            workspace=tmp_path,
                        ),
                        LocalDriver(),
                    )
                self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")
                self.assertIn("--run-id", str(ctx.exception))
                # The directory the search started from, not the cwd.
                self.assertIn(str(tmp_path), str(ctx.exception))
                self.assertEqual(jobs.list_jobs(), [])

    def test_submit_outside_the_workspace_uses_the_sessions_run(self):
        """A `cd` out of the workspace keeps the session's run and workspace.

        The lease the session-start hook wrote names both, so `submit` from
        /tmp tags the job with the session's run and runs it under the
        interpreter the workspace's environment.json configures, as `run`
        from inside the workspace would, rather than failing with
        MISSING_FIELD or falling back to the default.
        """
        with tempfile.TemporaryDirectory() as tmp:
            home, ws, elsewhere = (Path(tmp) / d
                                   for d in ("home", "ws", "elsewhere"))
            for d in (home, ws / ".tpu-agent-suite", elsewhere):
                d.mkdir(parents=True)
            (ws / ".tpu-agent-suite" / "environment.json").write_text(
                json.dumps(
                    profile_env("local",
                                python_interpreter="/opt/venv/bin/python")))
            session_run = "20260101T000000Z-aaaaaaaaaaaa"
            recipe = ws / "recipe.py"
            recipe.write_text("print('hi')\n")
            with patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(home)}), \
                    patch.object(compute.cli.lease, "session_run",
                                 return_value=(session_run, ws)), \
                    patch.object(compute.cli.lease, "spawn_watcher"), \
                    patch.object(LocalDriver, "submit", autospec=True,
                                 return_value="job-x") as submit, \
                    patch.object(LocalDriver, "get_status",
                                 return_value=AsynchronousExecutionResult(
                                     "job-x", "RUNNING")), \
                    contextlib.chdir(elsewhere), \
                    io.StringIO() as buf, contextlib.redirect_stdout(buf):
                rc = compute.handle_compute_cli(
                    self._args(compute_action="submit",
                               backend=None,
                               package="tpu-agent-suite@1.0.0",
                               recipe=str(recipe),
                               name="n",
                               submit_args=[],
                               interpreter=None,
                               run_id=None),
                    resolve=compute_config.resolve_driver)
            self.assertEqual(rc, 0)
            self.assertEqual(submit.call_args.kwargs["run_id"], session_run)
            self.assertEqual(submit.call_args.args[0].python_interpreter,
                             "/opt/venv/bin/python")

    def test_the_lease_wins_over_the_workspace_the_cwd_is_in(self):
        """A job launched from this session is this session's to reap, so
        the lease names the run and workspace wherever the cwd is, even
        inside another workspace with its own active run; `--workspace` or
        `--run-id` is how to act on that one instead."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            active = "20260101T000000Z-bbbbbbbbbbbb"
            leased = "20260101T000000Z-cccccccccccc"
            other_ws = tmp_path / "other-ws"
            other_ws.mkdir()
            with patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(tmp_path)}):
                utils.record_active_run(active, workspace=tmp_path)
                (tmp_path / "sub").mkdir()
                args = self._args(compute_action="job",
                                  job_action="reap",
                                  run_id=None)
                with patch.object(compute.cli.lease, "session_run",
                                  return_value=(leased, other_ws)), \
                        contextlib.chdir(tmp_path / "sub"):
                    rc, out = self._invoke(args, LocalDriver())
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["run_id"], leased)
            self.assertEqual(args.workspace, other_ws)

    def test_two_sessions_in_one_workspace_each_keep_their_own_run(self):
        """A second session's start overwrites the workspace's active-run
        marker; the lease is per harness process, so this session's `job
        list` still refreshes its own jobs, not the other session's."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            mine = "20260101T000000Z-aaaaaaaaaaaa"
            theirs = "20260101T000000Z-bbbbbbbbbbbb"
            driver = MagicMock()
            driver.get_status.side_effect = (
                lambda jid: AsynchronousExecutionResult(jid, "RUNNING"))
            with patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": tmp}), \
                    patch.object(compute.cli.lease, "session_run",
                                 return_value=(mine, ws)), \
                    contextlib.chdir(ws):
                utils.record_active_run(theirs, workspace=ws)
                jobs.JobRecord.create(job_id="my-job",
                                      run_id=mine,
                                      backend="x")
                jobs.JobRecord.create(job_id="their-job",
                                      run_id=theirs,
                                      backend="x")
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status=None), driver)
            self.assertEqual(rc, 0)
            self.assertEqual(
                {r["job_id"]: r["status"]
                 for r in json.loads(out)}, {
                     "my-job": "RUNNING",
                     "their-job": "PENDING"
                 })
            driver.get_status.assert_called_once_with("my-job")


class _ScriptedJob:
    """A driver whose job moves through `states`, one per status read.

    A state is a status, a (status, error) pair, a (status, error, details)
    triple, or an exception the read raises. `logs` is the text `fetch_logs`
    returns, or a function of how many reads there have been, for logs that
    grow as the job runs. Each read advances `clock` by `poll_sec`.
    """

    def __init__(self, states: list[Any], logs: Any = "") -> None:
        self.states = states
        self.logs = logs
        self.reads = 0
        self.clock = [0.0]
        self.poll_sec = 0.0

    def get_status(self, job_id: str) -> AsynchronousExecutionResult:
        state = self.states[min(self.reads, len(self.states) - 1)]
        self.reads += 1
        self.clock[0] += self.poll_sec
        if isinstance(state, Exception):
            raise state
        details: dict[str, Any] = {}
        if isinstance(state, tuple):
            status, error, *rest = state
            if rest:
                details = rest[0]
        else:
            status, error = state, None
        return AsynchronousExecutionResult(job_id=job_id,
                                           status=status,
                                           error=error,
                                           details=details)

    def fetch_logs(self, job_id: str) -> str:
        return self.logs(self.reads) if callable(self.logs) else self.logs


class TestJobWait(unittest.TestCase):
    """`compute job wait`: one exit code per outcome, so agents stop writing
    sleep loops that outlive the shell or miss a crash."""

    def wait(self, job: _ScriptedJob, **flags: Any) -> tuple[int, dict]:
        args = argparse.Namespace(compute_action="job",
                                  job_action="wait",
                                  backend="mock_test",
                                  workspace=None,
                                  json=True,
                                  job_id="serve-1",
                                  until=None,
                                  fail_on_log=None,
                                  no_default_fail_patterns=False,
                                  timeout=None)
        vars(args).update(flags)
        cli = sys.modules[compute.handle_compute_cli.__module__]
        clock = job.clock

        def sleep(sec: float) -> None:
            clock[0] += sec

        with io.StringIO() as buf, contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch.object(cli.time, "sleep", sleep), \
                patch.object(cli.time, "monotonic", lambda: clock[0]):
            rc = compute.handle_compute_cli(args, resolve=lambda **_: job)
            return rc, json.loads(buf.getvalue())

    def test_each_outcome_has_its_exit_code(self):
        on_ready = dict(until="log:READY")

        def ready_on_read_2(first: str):
            return lambda reads: first + ("READY\n" if reads >= 2 else "")

        cases = [
            # (states, logs, flags, exit code, outcome, line)
            (["PENDING",
              "RUNNING"], "", dict(until="running"), 0, "reached", None),
            (["RUNNING"], ready_on_read_2("loading\n"), on_ready, 0, "reached",
             "READY"),
            (["RUNNING", "COMPLETED"], "", {}, 0, "reached", None),
            # A warning-like line does not fail a job that succeeded.
            (["COMPLETED"], "No space left on device", {}, 0, "reached", None),
            # The run that went on for 8 minutes: the compile cache filled
            # /dev/shm while the Job still read RUNNING.
            (["RUNNING"], "RESOURCE_EXHAUSTED: write", on_ready, 2,
             "failure_logged", "RESOURCE_EXHAUSTED: write"),
            (["RUNNING"], "HBM OOM", dict(on_ready, fail_on_log=["OOM"]), 2,
             "failure_logged", "HBM OOM"),
            # A TP=8 attempt failed, then TP=4 came up: the server is up.
            (["RUNNING"], "RuntimeError: cancelled\nREADY", on_ready, 0,
             "reached", "READY"),
            (["RUNNING"], ready_on_read_2("RuntimeError: cancelled\n"),
             on_ready, 2, "failure_logged", "RuntimeError: cancelled"),
            (["RUNNING"], ready_on_read_2("RuntimeError: cancelled\n"),
             dict(on_ready,
                  no_default_fail_patterns=True), 0, "reached", "READY"),
            # Up, then dead before the wait looked: not up.
            (["FAILED"], "READY\nRaising signal 7", on_ready, 4, "job_ended",
             "Raising signal 7"),
            (["COMPLETED"], "bye", on_ready, 4, "job_ended", None),
        ]
        for states, logs, flags, code, outcome, line in cases:
            with self.subTest(states=states, logs=logs, **flags):
                rc, result = self.wait(_ScriptedJob(states, logs), **flags)
                self.assertEqual((rc, result["outcome"], result.get("line")),
                                 (code, outcome, line))

    def test_failed_reads_are_retried_then_raised(self):
        # An API blip does not end the wait; a job no backend knows does,
        # with the runtime's error exit rather than a nine-minute timeout.
        rc, _ = self.wait(_ScriptedJob([RuntimeError("blip"), "RUNNING"]),
                          until="running")
        self.assertEqual(rc, 0)
        job = _ScriptedJob([("UNKNOWN", "Job not found")])
        with self.assertRaises(ComputeConfigError) as ctx:
            self.wait(job)
        self.assertIn("Job not found", ctx.exception.message)
        self.assertEqual(job.reads, 3)

    def test_timeout_leaves_room_for_the_next_poll(self):
        rc, result = self.wait(_ScriptedJob(["PENDING"]), timeout=0)
        self.assertEqual((rc, result["outcome"]), (3, "timed_out"))
        self.assertIn("again", result["next"])

        # Polls take 30s: after the one ending at 70s, another would end
        # past 100s, so the wait stops at 70s rather than overrunning.
        job = _ScriptedJob(["PENDING"])
        job.poll_sec = 30
        _, result = self.wait(job, timeout=100)
        self.assertEqual(result["waited_sec"], 70)

    def test_a_long_pending_job_is_told_why_before_waiting_again(self):
        # An unqueued pod on a full cluster, past the ceiling.
        cli = sys.modules[compute.handle_compute_cli.__module__]
        long = cli.WAIT_PENDING_GIVE_UP_SEC + 180  # the wait stops a poll early
        reason = "0/15 nodes are available: 8 Insufficient google.com/tpu."
        job = _ScriptedJob([("PENDING", reason, {
            "pending_reason": base.PENDING_UNSCHEDULED
        })])
        job.poll_sec = 60
        _, result = self.wait(job, timeout=long)
        self.assertGreaterEqual(result["since_submit_sec"],
                                cli.WAIT_PENDING_GIVE_UP_SEC)
        self.assertIn("job cancel --job-id serve-1", result["next"])
        self.assertIn(reason, result["next"])

        # Same ceiling, but queued, admitted, starting, running, or a backend
        # that gives no reason (xpk, whose PENDING is always a queue): wait.
        kept = [("PENDING", None, {
            "pending_reason": r
        }) for r in (base.PENDING_QUEUED, base.PENDING_ADMITTED_NO_NODE,
                     base.PENDING_STARTING)] + ["RUNNING", "PENDING"]
        for state in kept:
            with self.subTest(state=state):
                job = _ScriptedJob([state])
                job.poll_sec = 60
                _, result = self.wait(job, until="log:READY", timeout=long)
                self.assertIn("run the same `job wait` again", result["next"])
                self.assertNotIn("job cancel", result["next"])

    def test_bad_until_is_rejected(self):
        for until in ("ready", "log:", "log:(unclosed"):
            with self.assertRaises(ComputeConfigError):
                self.wait(_ScriptedJob(["RUNNING"]), until=until)


class TestReapJobs(unittest.TestCase):
    """`reap_jobs` hands every live job to its own backend's driver."""

    RUN = "20260101T000000Z-000000000001"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.patcher = patch.dict(os.environ,
                                  {"TPU_AGENT_SUITE_HOME": str(self.tmp)})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _live(self,
              job_id: str,
              backend: str,
              serve: bool = False) -> jobs.JobRecord:
        rec = jobs.JobRecord.create(job_id=job_id,
                                    job_name=job_id,
                                    run_id=self.RUN,
                                    backend=backend,
                                    serve=serve)
        return rec.mark("RUNNING")

    def test_serving_jobs_are_exempt_from_reap(self):
        """A job with serve=True is skipped by reap_jobs on session end."""
        self._live("job-serve-1", "local", serve=True)
        driver = MagicMock()
        reaped, failed = jobs.reap_jobs(self.RUN, lambda _: driver)
        self.assertEqual(reaped, [])
        self.assertEqual(failed, {})
        driver.cancel.assert_not_called()
        self.assertEqual(jobs.JobRecord.load("job-serve-1").status, "RUNNING")

    def test_local_job_is_cancelled_through_the_local_driver(self):
        """Local jobs go through LocalDriver.cancel too: it is what knows
        to signal the whole process group, not just the wrapper shell."""
        self._live("job-local", "local")
        resolved: list[str] = []

        def resolver(job_id):
            resolved.append(job_id)
            return LocalDriver()

        reaped, failed = jobs.reap_jobs(self.RUN, resolver)
        self.assertEqual((reaped, failed), (["job-local"], {}))
        self.assertEqual(resolved, ["job-local"])
        self.assertEqual(jobs.JobRecord.load("job-local").status, "CANCELLED")

    def test_untagged_backend_is_treated_as_local(self):
        """A record from before backends were tagged was a local job, so it
        gets the local driver whatever the profile's backend is."""
        self._live("job-old", "")
        with patch.object(compute_config,
                          "find_environment_data",
                          return_value=profile_env("gke_pod")):
            driver = resolve_driver(job_id="job-old")
        self.assertIsInstance(driver, LocalDriver)

    def test_driver_failure_keeps_the_record_live_and_is_reported(self):
        self._live("job-xpk", "gke_xpk")
        driver = MagicMock()
        driver.cancel.side_effect = RuntimeError("GCP cluster unavailable")
        reaped, failed = jobs.reap_jobs(self.RUN, lambda _: driver)
        self.assertEqual(reaped, [])
        self.assertEqual(failed, {"job-xpk": "GCP cluster unavailable"})
        # RUNNING, so a later reap or an operator can retry.
        self.assertEqual(jobs.JobRecord.load("job-xpk").status, "RUNNING")

    def test_reaped_means_the_driver_recorded_cancelled(self):
        """The driver owns the record's final word. One that finds the job
        already finished records that instead, and the job is not "reaped"."""
        self._live("job-live", "gke_job")
        self._live("job-done", "gke_job")

        def cancel(job_id):
            rec = jobs.JobRecord.load(job_id)
            rec.mark("CANCELLED" if job_id == "job-live" else "COMPLETED")

        driver = MagicMock()
        driver.cancel.side_effect = cancel
        reaped, failed = jobs.reap_jobs(self.RUN, lambda _: driver)
        self.assertEqual((reaped, failed), (["job-live"], {}))
        self.assertEqual(jobs.JobRecord.load("job-done").status, "COMPLETED")

    def test_explicit_reap_can_take_the_serving_jobs_too(self):
        """`job reap --include-serving` is a decision to clear the run."""
        self._live("job-serve-2", "local", serve=True)
        reaped, failed = jobs.reap_jobs(self.RUN,
                                        lambda _: LocalDriver(),
                                        include_serving=True)
        self.assertEqual((reaped, failed), (["job-serve-2"], {}))
        self.assertEqual(
            jobs.JobRecord.load("job-serve-2").status, "CANCELLED")

    def test_serving_jobs_can_be_named_without_reaping_them(self):
        """What the reaper left behind has to be reportable."""
        self._live("job-serve-3", "local", serve=True)
        self._live("job-batch-3", "local")
        self._live("job-old-3", "local", serve=True).mark("COMPLETED")
        self.assertEqual(jobs.serving_jobs(self.RUN), ["job-serve-3"])

    def test_terminal_jobs_are_left_alone(self):
        for i, status in enumerate(("COMPLETED", "FAILED", "CANCELLED")):
            self._live(f"job-term-{i}", "gke_job").mark(status)
        driver = MagicMock()
        self.assertEqual(jobs.reap_jobs(self.RUN, lambda _: driver), ([], {}))
        driver.cancel.assert_not_called()


class TestContractGeneration(unittest.TestCase):
    """The generated agent contract must reflect the real registry."""

    def test_required_flags_are_reported_per_backend_and_ordered_first(self):
        """Verify `from_params` requirements match the registry and lead before optional flags."""
        required = compute.describe_required_flags()
        self.assertIn("--pod", required["gke_pod"])
        self.assertNotIn("--container", required["gke_pod"])
        self.assertIn("--cluster-name", required["gke_xpk"])
        self.assertNotIn("--gcs-bucket", required["gke_xpk"])

        flags = compute.describe_backends()["gke_pod"]
        self.assertEqual(flags[0], "--pod")
        self.assertEqual(flags[1:],
                         ["--container", "--kube-context", "--namespace"])

    def test_contract_lists_registered_backends_and_marks_optional_flags(self):
        """Verify backends, their flags, and optionality come from the registry."""
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertIn("`mock_test`", section)
        self.assertIn("--endpoint", section)
        self.assertIn("RUNTIME compute --package pkg@1.0.0 inspect", section)
        self.assertIn("`--container <value>` (optional)", section)
        self.assertNotIn("`--pod <value>` (optional)", section)

    def test_contract_states_when_no_backends_registered(self):
        """Verify an empty registry is stated plainly, not papered over."""
        with patch.object(compute, "describe_backends", return_value={}):
            section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertIn("no compute backends", section)
        self.assertNotIn("gce_tpu_vm", section)

    def test_every_contract_command_names_the_package(self):
        """Without --package the resolver never looks in the package's own
        environment.json, and a correctly installed package reports
        NO_COMPUTE_BACKEND."""
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        commands = [
            line for line in section.splitlines() if "RUNTIME compute" in line
        ]
        self.assertTrue(commands)
        for line in commands:
            self.assertIn("--package pkg@1.0.0", line)

    def test_contract_tells_agent_to_outlast_the_run(self):
        """A shell call that times out kills the run inside it, so the section
        must say to raise any per-call shell timeout past `--timeout`."""
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertIn(f"+ {compiler.SHELL_TIMEOUT_HEADROOM_S} seconds",
                      section)
        self.assertIn(f"under {compiler.MAX_SYNC_COMPUTE_TIMEOUT_S}", section)

    def test_contract_says_what_a_recipe_is(self):
        """Verify the instructions explain what a recipe is and how it runs.

        Before this, they only listed `submit --recipe <path>`, so agents had
        to read the driver source to find out.
        """
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertIn("A recipe (`--recipe`", section)
        self.assertIn("a `.sh` file runs under `bash`", section)
        self.assertIn("[-- <recipe args>]", section)
        self.assertIn(
            "RUNTIME compute --package pkg@1.0.0 submit --recipe train.sh "
            "--input train.py", section)
        # Every agent gets this section, and torch-inference-coder says to
        # start vLLM only through serve_vllm.sh, so don't show `vllm serve`.
        self.assertNotIn("vllm serve", section)
        # Shell recipes get the configured Python as TPU_AGENT_SUITE_PYTHON (#656).
        self.assertIn("as `$TPU_AGENT_SUITE_PYTHON`", section)
        self.assertIn('exec "$TPU_AGENT_SUITE_PYTHON" train.py', section)
        self.assertNotIn("binaries by path", section)
        self.assertIn("`$WORKLOAD_NAME`", section)
        without_xpk = {
            k: v
            for k, v in compute.describe_backends().items() if k != "gke_xpk"
        }
        with patch.object(compute,
                          "describe_backends",
                          return_value=without_xpk):
            section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertNotIn("gke_xpk", section)

    def test_contract_keeps_agents_off_raw_cluster_commands(self):
        """Agents left compute for raw kubectl and touched other runs' jobs.
        Every agent gets this section, so it is the one place the rule needs
        to be."""
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        self.assertIn("Don't run `kubectl`", section)
        # Recipes may (gke_xpk ones create their own XPK workload).
        self.assertIn("a recipe that compute runs may", section)
        self.assertIn("belongs to someone else", section)
        self.assertIn("rather than working around it", section)

    def test_contract_matches_the_job_cli(self):
        """`job wait` shipped in the CLI but not in this section, so agents
        wrote their own sleep loops. Every action the `job`
        parser accepts must be listed, and every flag the section names for
        `job` must exist on it."""
        parser = runtime.build_runtime_parser()
        for name in ("compute", "job"):
            parser = next(
                a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices[name]
        section = compiler.compute_contract_section("RUNTIME", "pkg@1.0.0")
        job_lines = " ".join(line for line in section.splitlines()
                             if "pkg@1.0.0 job " in line)
        choices = next(a.choices for a in parser._actions
                       if a.dest == "job_action")
        for action in choices:
            self.assertRegex(job_lines,
                             rf"job (\[[^\]]*\b{action}\b|{action}\b)")
        options = {o for a in parser._actions for o in a.option_strings}
        for flag in set(re.findall(r"--[a-z][a-z-]*",
                                   job_lines)) - {"--package"}:
            self.assertIn(flag, options)


class TestPreflight(unittest.TestCase):
    """`compute preflight`: the same report shape for every backend."""

    def _run(self, driver, responses=None, tools="/bin/tool"):
        """Preflight with subprocess answering by a substring of the argv."""
        calls: list[str] = []

        def fake(cmd, **_kwargs):
            calls.append(" ".join(cmd))
            for key, (code, out) in (responses or {}).items():
                if key in calls[-1]:
                    return _completed(out, code, "boom")
            return _completed()

        with patch("shutil.which", return_value=tools), \
                patch("subprocess.run", fake):
            report = driver.preflight()
        return report, {c.name: c for c in report.checks}, calls

    ACCOUNT = {"auth list": (0, "me@example.com\n")}

    def test_probe_error_prefers_the_tools_error_line(self):
        # gcloud: the error, then a usage block that says nothing is wrong.
        gcloud = ("ERROR: (gcloud.x) Image not found.\n\nA valid image has "
                  "the format of\n  HOST/PROJECT/REPO/IMAGE:tag\n")
        self.assertEqual(base.probe_error(_completed("", 1, gcloud)),
                         "ERROR: (gcloud.x) Image not found.")
        self.assertEqual(
            base.probe_error(
                _completed(
                    "", 1, "E0925 memcache.go:265] noise\n"
                    'Error from server (Forbidden): nodes is "x"\n')),
            'Error from server (Forbidden): nodes is "x"')
        # No error line: the last line, as before.
        self.assertEqual(
            base.probe_error(
                _completed(
                    "", 1, "E0925 noise\nUnable to connect to the "
                    "server: dial tcp: i/o timeout\n")),
            "Unable to connect to the server: dial tcp: i/o timeout")
        self.assertEqual(base.probe_error(_completed("", 3, "")), "exit 3")

    def _gce(self):
        return GceTpuVmDriver(tpu_name="vm", zone="us-east5-b", project="p")

    def test_a_missing_tool_is_a_failure_and_the_target_is_not_checked(self):
        report, checks, calls = self._run(GkeXpkDriver(cluster_name="c",
                                                       project="p",
                                                       zone="z"),
                                          tools=None)
        self.assertEqual((report.reachable, report.status), (False, "fail"))
        self.assertIn("docs/install.md", checks["tool:xpk"].fix)
        self.assertEqual(calls, [])

    def test_local(self):
        with patch.object(probe, "list_devices", return_value=([], [])):
            report, checks, _ = self._run(LocalDriver())
        self.assertFalse(report.reachable)
        self.assertEqual(checks["tpu_devices"].status, "fail")
        with patch.object(probe, "list_devices",
                          return_value=(["/dev/accel0"], [])), \
                patch.object(probe, "scan_holders",
                             return_value=(["PID 7"], set(), set())):
            report, checks, _ = self._run(LocalDriver())
        self.assertEqual((report.reachable, report.status), (True, "warn"))

    def test_gce(self):
        report, checks, calls = self._run(self._gce(), {
            **self.ACCOUNT, "describe": (0, "READY\tv6e-8\n")
        })
        self.assertEqual((report.reachable, report.status), (True, "ok"))
        self.assertFalse(any(" ssh " in call for call in calls))
        _, checks, _ = self._run(self._gce(), {
            **self.ACCOUNT, "describe": (0, "STOPPED v6e-8\n")
        })
        self.assertIn("tpu-vm start vm --zone=us-east5-b",
                      checks["tpu_vm"].fix)
        _, checks, _ = self._run(self._gce(), {"auth list": (0, "")})
        self.assertIn("gcloud auth login", checks["gcloud:account"].fix)
        self.assertNotIn("tpu_vm", checks)

    def test_gke_pod(self):
        cluster = {
            "current-context": (0, "ctx"),
            "version": (0, '"serverVersion"')
        }
        _, checks, calls = self._run(GkePodDriver(pod="p1", namespace="ns"), {
            **cluster, "get pod": (0, "Pending")
        })
        self.assertIn("is Pending", checks["pod"].detail)
        # The cluster is probed with the same kubectl the driver runs with.
        kubectl = [call for call in calls if call.startswith("kubectl")]
        self.assertTrue(kubectl)
        self.assertTrue(
            all(call.startswith("kubectl -n ns ") for call in kubectl))
        self.assertIn("gcloud:account", checks)

    @staticmethod
    def _cluster(kubectl=("kubectl", ), kubeconfig=None):
        """The `cluster` check `auth_utils.cluster_check` appends."""
        checks: list = []
        auth_utils.cluster_check(types.SimpleNamespace(), list(kubectl),
                                 checks, kubeconfig)
        return next(c for c in checks if c.name == "cluster")

    def _cluster_check(self, env, kubeconfig=None, context=None):
        """The cluster check with kubectl knowing no context, under `env`."""
        with patch.dict(os.environ, env, clear=True), \
                patch("subprocess.run", return_value=_completed("", 1)):
            return self._cluster(base.kubectl_prefix(kubeconfig, context),
                                 kubeconfig)

    def test_a_missing_kubeconfig_is_named(self):
        """kubectl calls a missing file "no context"; the path is the fault."""
        with tempfile.TemporaryDirectory() as home:
            gone = os.path.join(home, "missing-kubeconfig")
            there = os.path.join(home, "config")
            Path(there).write_text("", encoding="utf-8")
            # Given with --kubeconfig: named, and credentials written there.
            check = self._cluster_check({"HOME": home}, kubeconfig=gone)
            self.assertEqual(check.status, "fail")
            self.assertIn(f"kubeconfig {gone}, which does not exist",
                          check.detail)
            self.assertTrue(check.fix.startswith(f"KUBECONFIG={gone} gcloud"))
            # Named by $KUBECONFIG, alongside one that is there.
            check = self._cluster_check({
                "HOME":
                home,
                "KUBECONFIG":
                f"{gone}{os.pathsep}{there}"
            })
            self.assertIn(f"$KUBECONFIG names {gone}, which does not exist",
                          check.detail)
            self.assertNotIn(there, check.detail)
            self.assertIn(f"KUBECONFIG={gone} ", check.fix)
            # Neither: the default file, written without naming it.
            check = self._cluster_check({"HOME": home})
            self.assertIn("~/.kube/config, which does not exist", check.detail)
            self.assertTrue(check.fix.startswith("gcloud container clusters"))
            # The file is there but has no context: nothing to name.
            check = self._cluster_check({"HOME": home, "KUBECONFIG": there})
            self.assertIn("no cluster is configured", check.detail)
            # A pinned context is named, and its cluster is in the fix.
            check = self._cluster_check({"HOME": home},
                                        kubeconfig=gone,
                                        context="gke_p_us-central1_c9")
            self.assertIn(
                "context 'gke_p_us-central1_c9' cannot be used: "
                f"kubeconfig {gone}", check.detail)
            self.assertIn("get-credentials c9", check.fix)

    def test_kube_context_check(self):
        """A pinned context: ok if current, a warning if another is, and a
        failure if the kubeconfig does not have it or cannot be read."""
        with tempfile.TemporaryDirectory() as home:
            config = os.path.join(home, "config")
            Path(config).write_text("{}", encoding="utf-8")

            def check(context, contexts="a\nb\n", current="b", code=0):
                out = {"name": contexts, "current-context": current}

                def run(cmd, **_):
                    return _completed(out.get(cmd[-1], ""), code, "boom")

                with patch.object(base, "probe_run", side_effect=run):
                    return base.kube_context_check(config, context)

            self.assertIsNone(check(None))
            self.assertEqual(check("b").status, "ok")
            moved = check("a")
            self.assertEqual(moved.status, "warn")
            self.assertIn("'a', not kubectl's current context 'b'",
                          moved.detail)
            for contexts, code, detail, others in (
                ("a\nb\n", 0, "not in the kubeconfig",
                 "one of: a, b"), ("", 0, "not in the kubeconfig", None),
                ("", 1, "could not read the kubeconfig's contexts: boom",
                 None)):
                with self.subTest(contexts=contexts, code=code):
                    failed = check("gke_p_us-central1_c9", contexts, code=code)
                    self.assertEqual((failed.name, failed.status),
                                     ("kube_context", "fail"))
                    self.assertIn(detail, failed.detail)
                    self.assertIn("get-credentials c9", failed.fix)
                    self.assertEqual("one of" in failed.fix, bool(others))
                    if others:
                        self.assertIn(others, failed.fix)

    def test_the_credentials_fix_is_filled_from_the_environment(self):
        """CI and evals say which cluster they mean: no placeholders."""
        with tempfile.TemporaryDirectory() as home:
            gke = {
                "HOME": home,
                "GKE_CLUSTER": "tpu-nexus-cluster",
                "GKE_LOCATION": "us-central2-b",
                "GKE_PROJECT": "tpu-nexus-767207",
            }
            check = self._cluster_check(gke)
            self.assertIn(
                "get-credentials tpu-nexus-cluster --location us-central2-b "
                "--project tpu-nexus-767207", check.fix)
            self.assertNotIn("<", check.fix)
            # Only what is unknown stays a placeholder.
            del gke["GKE_PROJECT"]
            check = self._cluster_check(gke)
            self.assertIn("--project <project>", check.fix)
            self.assertIn("get-credentials tpu-nexus-cluster ", check.fix)

    def test_the_credentials_fix_names_the_contexts_cluster(self):
        """Without GKE_*, a gke_* context names the cluster; GKE_* wins over
        it, whole, never mixed with it; what GKE_* lacks is a placeholder."""
        context = "gke_tpu-nexus-767207_us-central2-b_tpu-nexus-cluster"

        def kubectl(name, error):
            return lambda cmd, **_: (_completed(name, 0) if "current-context"
                                     in cmd else _completed("", 1, error))

        for error in ("error: You must be logged in to the server",
                      "Unable to connect to the server: i/o timeout"):
            with self.subTest(error=error), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch("subprocess.run", kubectl(context, error)):
                check = self._cluster()
            self.assertIn(
                "get-credentials tpu-nexus-cluster --location us-central2-b "
                "--project tpu-nexus-767207", check.fix)
        # The DNS-endpoint hint names the same cluster.
        self.assertIn("update tpu-nexus-cluster --enable-dns-access",
                      check.fix)
        gke = {"GKE_CLUSTER": "other", "GKE_PROJECT": "other-project"}
        for name in (context, "my-cluster"):
            with patch.dict(os.environ, gke, clear=True), \
                    patch("subprocess.run",
                          kubectl(name, "error: You must be logged in")):
                check = self._cluster()
            self.assertIn(
                "get-credentials other --location <location> "
                "--project other-project", check.fix)
        # A cluster that answers is the context's, whatever GKE_* says.
        with patch.dict(os.environ, gke, clear=True):
            self.assertEqual(
                base.gke_target(context, prefer_context=True), {
                    "cluster": "tpu-nexus-cluster",
                    "location": "us-central2-b",
                    "project": "tpu-nexus-767207"
                })

    SA = "123-compute@developer.gserviceaccount.com"

    def _metadata(self, email, scopes):
        """base._metadata_get answering as a VM whose account is `email`."""
        values = {"email": email, "scopes": "\n".join(scopes)}
        return patch.object(base,
                            "_metadata_get",
                            side_effect=lambda p: values[p.rsplit("/", 1)[-1]])

    def _account_check(self, account, other=None):
        with patch("subprocess.run", return_value=_completed(account, 0)):
            return base.gcloud_account_check(other)

    def test_a_vm_account_without_cloud_platform_scope(self):
        """A VM's service account is limited to the VM's scopes, whatever
        IAM grants it: say so, rather than let every call be refused."""
        storage = "https://www.googleapis.com/auth/devstorage.read_only"
        with self._metadata(self.SA, [storage]):
            check = self._account_check(self.SA)
            self.assertEqual(check.status, "fail")
            self.assertIn("does not grant it the cloud-platform scope",
                          check.detail)
            self.assertIn("devstorage.read_only", check.detail)
            self.assertIn("gcloud auth login", check.fix)
            self.assertIn("--scopes=cloud-platform", check.fix)
            # Another credential that works: a warning, and it is named.
            check = self._account_check(self.SA, "kubectl reaches the cluster")
            self.assertEqual(check.status, "warn")
            self.assertIn("not blocking: kubectl reaches the cluster",
                          check.detail)
        with self._metadata(self.SA, [storage, base.CLOUD_PLATFORM_SCOPE]):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # Some other service account (a key gcloud activated): not the VM's.
        with self._metadata("other@p.iam.gserviceaccount.com", [storage]):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # Off GCE: nothing to check.
        with patch.object(base, "_metadata_get", side_effect=OSError("no")):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # A user account is never limited by scopes, so it is not looked up.
        with patch.object(base,
                          "_metadata_get",
                          side_effect=AssertionError("looked up")):
            self.assertEqual(
                self._account_check("me@example.com").status, "ok")
        # No account at all: the same rule.
        self.assertEqual(self._account_check("").status, "fail")
        self.assertEqual(self._account_check("", "$X works").status, "warn")

    def test_application_credentials_are_reported_with_a_warning(self):
        """$GOOGLE_APPLICATION_CREDENTIALS is a credential kubectl will use:
        list it, but warn rather than recommend."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(base.application_credentials_check(),
                             (None, None))
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "user.json": {
                    "type": "authorized_user",
                    "client_id": "x"
                },
                "sa.json": {
                    "type": "service_account",
                    "client_email": "bot@p.iam.gserviceaccount.com"
                },
                "odd.json": {
                    "type": "gdch_service_account"
                },
            }
            for name, body in files.items():
                Path(tmp, name).write_text(json.dumps(body), encoding="utf-8")
            Path(tmp, "bad.json").write_text("{", encoding="utf-8")

            def check(name):
                path = os.path.join(tmp, name)
                with patch.dict(os.environ, {base.GAC_ENV: path}, clear=True):
                    return base.application_credentials_check()

            gac, usable = check("user.json")
            self.assertEqual(gac.status, "warn")
            self.assertIn("authorized_user credential", gac.detail)
            self.assertIn(base.KUBECTL_IDENTITY_WARNING, gac.detail)
            self.assertIn("env -u GOOGLE_APPLICATION_CREDENTIALS", gac.fix)
            self.assertIn("authorized_user", usable)
            gac, usable = check("sa.json")
            self.assertIn(
                "service_account credential bot@p.iam.gserviceaccount.com",
                gac.detail)
            self.assertIsNotNone(usable)
            # Reported, but not one the auth plugin is known to use.
            gac, usable = check("odd.json")
            self.assertIn("gdch_service_account", gac.detail)
            self.assertIsNone(usable)
            for name, says in (("gone.json", "No such file"),
                               ("bad.json", "not a readable credential")):
                gac, usable = check(name)
                self.assertIn(says, gac.detail)
                self.assertIn(base.KUBECTL_IDENTITY_WARNING, gac.detail)
                self.assertIsNone(usable)

    def test_the_credentials_repro(self):
        """KUBECONFIG names a missing file, the gcloud
        account is a VM's without cloud-platform scope, and an
        authorized_user $GOOGLE_APPLICATION_CREDENTIALS is set. The missing
        file is the blocker; the scope is not."""
        with tempfile.TemporaryDirectory() as tmp:
            gone = os.path.join(tmp, "config")
            gac = os.path.join(tmp, "adc.json")
            Path(gac).write_text('{"type": "authorized_user"}',
                                 encoding="utf-8")
            env = {
                "HOME": tmp,
                "KUBECONFIG": gone,
                base.GAC_ENV: gac,
                "GKE_CLUSTER": "c",
                "GKE_LOCATION": "l",
                "GKE_PROJECT": "p",
            }
            for driver in (GkePodDriver(pod="p1", namespace="ns"),
                           GkeJobDriver(kubeconfig=None)):
                with self.subTest(driver=driver.backend_name), \
                        patch.dict(os.environ, env, clear=True), \
                        self._metadata(self.SA, []):
                    report, checks, _ = self._run(driver, {
                        "current-context": (1, ""),
                        "auth list": (0, self.SA)
                    })
                self.assertFalse(report.reachable)
                self.assertEqual(checks["cluster"].status, "fail")
                self.assertIn(f"{gone}, which does not exist",
                              checks["cluster"].detail)
                self.assertIn("get-credentials c --location l --project p",
                              checks["cluster"].fix)
                self.assertEqual(checks["gcloud:account"].status, "warn")
                self.assertIn("cloud-platform scope",
                              checks["gcloud:account"].detail)
                self.assertIn(
                    "not blocking: $GOOGLE_APPLICATION_CREDENTIALS holds "
                    "another credential (authorized_user credential)",
                    checks["gcloud:account"].detail)
                self.assertEqual(checks["credentials:gac"].status, "warn")
                self.assertIn("authorized_user credential",
                              checks["credentials:gac"].detail)

    def test_a_cluster_that_answers_makes_account_problems_warnings(self):
        cluster = base.PreflightCheck("cluster", "ok", "context 'c' answers")
        with patch.dict(os.environ, {}, clear=True), \
                patch("shutil.which", return_value="/bin/gcloud"), \
                patch("subprocess.run", return_value=_completed("", 0)):
            checks = base.gke_credential_checks(cluster)
        self.assertEqual([(c.name, c.status) for c in checks],
                         [("gcloud:account", "warn")])
        self.assertIn("kubectl reaches the cluster", checks[0].detail)
        # No gcloud: no account for the plugin to use, so nothing to check.
        with patch.dict(os.environ, {}, clear=True), \
                patch("shutil.which", return_value=None):
            self.assertEqual(base.gke_credential_checks(cluster), [])

    def test_gke_xpk(self):
        driver = GkeXpkDriver(cluster_name="c",
                              project="p",
                              zone="us-central1-a")
        report, checks, _ = self._run(driver, {
            **self.ACCOUNT, "clusters list": (0, "europe-west4 RUNNING\n")
        })
        self.assertTrue(report.reachable)
        self.assertEqual(checks["zone"].status, "warn")
        report, checks, _ = self._run(driver, {
            **self.ACCOUNT, "clusters list": (0, "")
        })
        self.assertFalse(report.reachable)
        self.assertIn("no cluster 'c'", checks["cluster"].detail)
        self.assertNotIn("bucket", checks)
        driver = GkeXpkDriver(cluster_name="c",
                              project="p",
                              zone="us-central1-a",
                              gcs_bucket="b/runs")
        _, checks, calls = self._run(
            driver, {
                **self.ACCOUNT, "clusters list": (0, "us-central1 RUNNING\n"),
                "buckets describe": (1, "")
            })
        self.assertEqual(checks["bucket"].status, "fail")
        self.assertTrue(any("describe gs://b " in call for call in calls))

    def test_gke_backends_need_the_auth_plugin(self):
        for driver in (GkePodDriver, GkeJobDriver, GkeXpkDriver):
            self.assertIn("gke-gcloud-auth-plugin", driver.required_tools)

    def test_auth_plugin_needed_only_when_the_kubeconfig_calls_it(self):
        for stdout, tools in (("gke-gcloud-auth-plugin", base.GKE_TOOLS),
                              ("", ("kubectl", ))):
            proc = subprocess.CompletedProcess([], 0, stdout, "")
            with patch("shutil.which", return_value="/bin/kubectl"), \
                    patch.object(base, "probe_run", return_value=proc):
                self.assertEqual(GkeJobDriver.needed_tools(MagicMock()), tools)

    def test_a_probe_that_hangs_is_reported_not_raised(self):

        def hang(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1)

        with patch("shutil.which", return_value="/bin/gcloud"), \
                patch("subprocess.run", hang):
            report = self._gce().preflight()
        self.assertIn("did not answer", report.checks[-1].detail)

    def test_cli_prints_the_report_and_exits_1_on_failure(self):
        from framework.compute import cli as compute_cli
        driver = MagicMock()
        driver.preflight.return_value = base.PreflightReport(
            "local", "this host", False,
            [base.PreflightCheck("tpu_devices", "fail", "none", "use a VM")])
        for as_json in (False, True):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = compute_cli._action_preflight(
                    driver, argparse.Namespace(json=as_json))
            self.assertEqual(code, 1)
            self.assertIn(
                '"status": "fail"' if as_json else "[FAIL] tpu_devices  none",
                out.getvalue())


class _CountingDriver(MockTestDriver):
    """Mock driver with settable preflight and inspect results and a call
    counter for preflight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._checks = []
        self._calls = 0
        self._reachable = True
        self._hardware = base.HardwareProfile("v6e",
                                              "2x2",
                                              4,
                                              32.0,
                                              "healthy",
                                              details={
                                                  "accelerator_type": "v6e-4",
                                                  "job_devices": 4,
                                                  "hbm_gb_per_device": 32.0,
                                              })

    def preflight(self):
        self._calls += 1
        return base.PreflightReport("mock_test",
                                    f"mock target {self.endpoint}",
                                    self._reachable, list(self._checks))

    def inspect(self):
        return self._hardware


class TestComputeCheckPerSession(unittest.TestCase):
    """Session start reports the setup state; a changed profile is rechecked."""

    RUN_ID = "20260101T000000Z-123456789abc"

    def setUp(self):
        root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        patcher = patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        utils.get_run_root(self.RUN_ID).mkdir(parents=True)
        self.workspace = root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.env_file.parent.mkdir(parents=True)
        self._write_env({"python_interpreter": "a"})
        self.driver = _CountingDriver()
        patcher = patch.object(compute.cli.config,
                               "resolve_driver",
                               side_effect=lambda **_: self.driver)
        self.resolve = patcher.start()
        self.addCleanup(patcher.stop)
        # What this machine has installed changes the note; tests that care
        # set it.
        self.host_patcher = patch.object(compute.cli.setup,
                                         "host_requirements_note",
                                         return_value="")
        self.host_note = self.host_patcher.start()
        self.addCleanup(self.host_patcher.stop)

    def _write_env(self, params, backend="local"):
        self.env_file.write_text(
            json.dumps({
                "revision": 1,
                **profile_env(backend, **params)
            }))

    def _verified(self, passed=True, failed_checks=()):
        """Record a verify result for the current default profile."""
        profile = json.loads(self.env_file.read_text())["profiles"]["default"]
        (self.env_file.parent / utils.VERIFICATION_FILE).write_text(
            json.dumps({
                compute.config.profile_hash(profile): {
                    "profile": "default",
                    "passed": passed,
                    "version": utils.TPU_AGENT_SUITE_VERSION,
                    "failed_checks": list(failed_checks),
                }
            }))

    def _session_start(self, hook_timeout=30):
        with contextlib.redirect_stderr(io.StringIO()):
            return compute.cli.check_target_at_session_start(
                "pkg", self.workspace, self.RUN_ID, hook_timeout)

    def _command(self, profile=None):
        args = argparse.Namespace(compute_action="inspect",
                                  backend=None,
                                  package="pkg",
                                  workspace=self.workspace,
                                  json=True,
                                  profile=profile,
                                  run_id=self.RUN_ID)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            rc = compute.handle_compute_cli(args,
                                            resolve=lambda **_: self.driver)
        return rc, err.getvalue()

    READY = ("Compute ready: profile 'default'. Compute target checked: "
             "mock target default-endpoint (ok)\nHardware: accelerator_type "
             "v6e-4, 4 devices, 32 GB HBM per device. Plan from these "
             "numbers, not from the profile, VM or cluster name.")

    def test_session_start_reports_a_pass_and_records_the_profile_it_checked(
            self):
        self._verified()
        note = self._session_start()
        self.assertEqual(note, self.READY)
        # One driver serves both the preflight and the hardware inspect.
        self.assertEqual(self.resolve.call_count, 1)
        self.assertEqual(
            compute.cli._last_checked_hash(self.RUN_ID),
            compute.config.profile_hash({
                "backend": "local",
                "params": {
                    "python_interpreter": "a"
                }
            }))

    def test_hardware_missing_from_the_probe_is_never_guessed(self):
        """No devices or HBM, or an inspect that raises: run inspect."""
        self._verified()
        unread = "Hardware not read at session start{}. Don't guess "
        self.driver._hardware = base.HardwareProfile(
            "tpu",
            "unknown",
            4,
            0.0,
            "healthy",
            details={"hbm_gb_note": "unknown accelerator 'x'"})
        note = self._session_start()
        self.assertIn(unread.format(" (unknown accelerator 'x')"), note)
        self.assertIn("compute --package pkg inspect --json` before sizing",
                      note)
        self.assertIn("leave the sizing to a specialist", note)
        self.driver.inspect = MagicMock(side_effect=RuntimeError("ssh"))
        self.assertIn(unread.format(""), self._session_start())

    def test_session_start_leads_with_missing_host_software(self):
        """The installer's host check warns once; session start repeats it
        to the agent, whatever the compute state, until it passes."""
        self.host_note.return_value = "ASK"
        self._verified()
        self.assertEqual(self._session_start(),
                         f"## Host software\nASK\n\n{self.READY}")
        self._verified(passed=False)
        self.assertTrue(self._session_start().startswith(
            "## Host software\nASK\n\n## Compute setup\n"))
        self.env_file.unlink()
        self.assertEqual(self._session_start(), "## Host software\nASK")

    def test_host_requirements_come_from_the_file_shipped_with_the_runtime(
            self):
        real = self.host_patcher.temp_original  # setUp patched it out
        reqs = self.workspace / utils.HOST_REQUIREMENTS
        reqs.write_text("# host\npyyaml>=6.0\nno-such-package-xyz>=1.0\n")
        with patch.object(compute.setup, "_package_file", return_value=reqs):
            note = real()
        self.assertIn(
            f"`{sys.executable} -m pip install 'no-such-package-xyz>=1.0'`",
            note)
        self.assertNotIn("pyyaml", note)
        missing = compute.setup.ComputeConfigError("no file")
        with patch.object(compute.setup, "_package_file", side_effect=missing):
            self.assertEqual(real(), "")

    def test_session_start_gives_one_message_per_setup_state(self):
        """Only a READY profile is preflighted; the rest say what to run."""
        plan = "compute --package pkg setup plan"
        guide = str(compute.cli.SETUP_GUIDE)
        cases = [
            ("no profile", {
                "default": None,
                "profiles": {}
            }, [
                "Compute is not set up yet.", f"{plan} --json", guide,
                "On gke_job, it can also run a smoke-test Job"
            ]),
            ("none selected", {
                "default": None,
                "profiles": {
                    "vm": {
                        "backend": "local"
                    }
                }
            }, ["environment.json has vm", "config --package pkg use <name>"]),
            ("unknown selected", {
                "default": "gone",
                "profiles": {}
            }, [
                "'gone'", "config --package pkg use <name>",
                f"{plan} --profile <name> --backend <backend> --json",
                "setup verify --profile <name>`"
            ]),
            ("incomplete", profile_env("gke_pod"), [
                "profile 'default' is not fully set up",
                f"{plan} --profile default --json",
                "setup apply --profile default --answers '<json>'",
                "setup verify --profile default`"
            ]),
            ("unverified", profile_env("local"),
             ["setup verify`.", "!--smoke-test"]),
            ("unverified, schedules Jobs",
             profile_env("gke_job", tpu_accelerator="t", tpu_topology="1x1"),
             ["setup verify`.", "add `--smoke-test` only if they agree"]),
        ]
        for name, env, expected in cases:
            with self.subTest(name):
                self.env_file.write_text(json.dumps(env))
                note = self._session_start()
                self.assertTrue(note.startswith("## Compute setup\n"), note)
                for text in expected:
                    if text.startswith("!"):
                        self.assertNotIn(text[1:], note)
                    else:
                        self.assertIn(text, note)
        self.assertEqual(self.driver._calls, 0)

        # FAILED, from the last verify: its checks, fixes and fields.
        self._write_env({})
        self._verified(False, [{
            "name": "software",
            "detail": "no torch_tpu\nmore",
            "fix": "install it",
            "fields": ["python_interpreter"]
        }])
        note = self._session_start()
        self.assertIn("- software: no torch_tpu Fix: install it", note)
        self.assertIn(
            f"{plan} --profile default --fields python_interpreter --json",
            note)
        self.assertEqual(self.driver._calls, 0)

        # A READY profile whose quick preflight fails reads as FAILED; one
        # that did not answer, with no failing check, names that.
        self._verified()
        self.driver._checks = [
            base.PreflightCheck("image", "fail", "gone", "push it", ["image"])
        ]
        note = self._session_start()
        self.assertIn("failed its check.\n- image: gone Fix: push it", note)
        self.driver._checks, self.driver._reachable = [], False
        self.assertIn(
            "- reachable: mock target default-endpoint did not answer",
            self._session_start())
        self.assertEqual(self.driver._calls, 2)

    def test_only_a_changed_environment_is_checked_again(self):
        """The recheck runs once, reports a failure, and the command still runs."""
        self._verified()
        self._session_start()
        self._command()
        self.assertEqual(self.driver._calls, 1)
        self.driver._checks = [
            base.PreflightCheck("image", "fail", "gone", "push it")
        ]
        self._write_env({"python_interpreter": "b"})
        rc, err = self._command()
        self.assertEqual((rc, self.driver._calls), (0, 2))
        self.assertIn("the compute profile changed", err)
        self.assertIn("[FAIL] image  gone", err)
        self.assertIn("fix: push it", err)
        self._command()
        self.assertEqual(self.driver._calls, 2)
        # A change outside the profile is not a change to the target.
        doc = json.loads(self.env_file.read_text())
        self.env_file.write_text(json.dumps({**doc, "xprof_interpreter": "x"}))
        self._command()
        self.assertEqual(self.driver._calls, 2)

    def test_the_recheck_checks_the_profile_the_command_uses(self):
        """--profile is checked, not the run's profile; switching checks again."""
        self._verified()
        self._session_start()
        self._write_env({"python_interpreter": "b"})
        for profile, calls in (("vm", 2), ("vm", 2), (None, 3)):
            self._command(profile)
            self.assertEqual(self.driver._calls, calls)
            self.assertEqual(self.resolve.call_args.kwargs["profile_override"],
                             profile)

    def test_a_session_check_that_did_not_finish_is_left_to_the_next_command(
            self):
        self._verified()
        release = threading.Event()
        self.addCleanup(release.set)
        slow = self.driver.preflight
        self.driver.preflight = lambda: (release.wait(5), slow())[1]
        for timeout in (compute.cli.SESSION_CHECK_HEADROOM_S,
                        compute.cli.SESSION_CHECK_HEADROOM_S + 0.05):
            self.assertTrue(
                self._session_start(timeout).startswith(
                    "Compute ready: profile 'default'.\n"
                    "Hardware not read at session start."))
        self.assertIsNone(compute.cli._last_checked_hash(self.RUN_ID))
        release.set()
        _, err = self._command()
        self.assertIn("Compute target checked", err)
        self.assertIsNotNone(compute.cli._last_checked_hash(self.RUN_ID))


class TestAdcRetry(unittest.TestCase):
    """A cluster that refuses the gcloud account is retried with ADC."""

    ACCOUNT = "sa@proj.iam.gserviceaccount.com"
    ADC_USER = "me@example.com"

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(auth_utils.ACCESS_TOKEN_ENV, None)
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        for target, value in (("shutil.which", "/usr/bin/tool"), (
                "framework.compute.drivers.base.vm_account_scopes", None)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

    def _cluster(self, *allowed, adc=True):
        """A fake kubectl/gcloud whose cluster lets in only `allowed` credentials."""

        def run(cmd, **kwargs):
            token = os.environ.get(auth_utils.ACCESS_TOKEN_ENV)
            cred = "adc" if token == "adc-token" else "default"
            text = " ".join(cmd)
            self.calls.append((cred, text))
            if "print-access-token" in text:
                return subprocess.CompletedProcess(
                    cmd, 0 if adc else 1, "adc-token\n" if adc else "",
                    "" if adc else "ERROR: Your default credentials were "
                    "not found.")
            if cmd[0] == "gcloud":
                return subprocess.CompletedProcess(cmd, 0, self.ACCOUNT + "\n",
                                                   "")
            if "current-context" in text:
                return subprocess.CompletedProcess(
                    cmd, 0, "gke_proj_us-central1_cluster-a\n", "")
            if " version " in f" {text} ":
                return subprocess.CompletedProcess(cmd, 0,
                                                   '{"serverVersion": {}}', "")
            if cred in allowed:
                out = "Running" if "status.phase" in text else "hi\n"
                if "exec" in cmd:
                    out = f"{gke_pod.EXEC_STARTED}\n{out}"
                return subprocess.CompletedProcess(cmd, 0, out, "")
            user = self.ADC_USER if cred == "adc" else self.ACCOUNT
            return subprocess.CompletedProcess(
                cmd, 1, "",
                "Error from server (Forbidden): pods is forbidden: "
                f'User "{user}" cannot list resource "pods"')

        return patch("subprocess.run", run)

    def test_preflight_falls_back_to_adc_and_says_so(self):
        driver = GkePodDriver(pod="p")
        with self._cluster("adc"):
            report = driver.preflight().to_dict()
        checks = {c["name"]: c for c in report["checks"]}
        self.assertTrue(driver.using_adc)
        self.assertEqual(os.environ[auth_utils.ACCESS_TOKEN_ENV], "adc-token")
        self.assertEqual(checks["cluster"]["status"], "ok")
        self.assertIn("answers with application default credentials",
                      checks["cluster"]["detail"])
        self.assertEqual(checks["credentials:adc"]["status"], "warn")
        self.assertIn(f"refused the gcloud account {self.ACCOUNT}",
                      checks["credentials:adc"]["detail"])
        self.assertEqual(checks["pod"]["status"], "ok")  # read with ADC

    def test_preflight_names_both_identities_and_the_fix_when_adc_fails(self):
        with self._cluster():
            report = GkePodDriver(pod="p").preflight().to_dict()
        cluster = next(c for c in report["checks"] if c["name"] == "cluster")
        self.assertEqual(cluster["status"], "fail")
        self.assertIn(f"the gcloud account {self.ACCOUNT}", cluster["detail"])
        self.assertIn(f"application default credentials ({self.ADC_USER})",
                      cluster["detail"])
        self.assertIn("gcloud auth application-default login", cluster["fix"])
        self.assertIn(
            "gcloud projects add-iam-policy-binding proj "
            f"--member=serviceAccount:{self.ACCOUNT} "
            "--role=roles/container.developer", cluster["fix"])
        self.assertNotIn(auth_utils.ACCESS_TOKEN_ENV, os.environ)

    def test_a_cluster_that_lets_the_account_in_is_not_retried(self):
        driver = GkePodDriver(pod="p")
        with self._cluster("default"):
            report = driver.preflight().to_dict()
        self.assertFalse(driver.using_adc)
        self.assertNotIn("credentials:adc",
                         [c["name"] for c in report["checks"]])
        self.assertFalse(any("print-access-token" in c for _, c in self.calls))

    def test_compute_commands_retry_with_adc(self):
        """An exec kubectl refused before it started is retried with ADC."""
        driver = GkePodDriver(pod="p")
        with self._cluster("adc"):
            proc = driver._sh("echo hi")
        self.assertEqual((proc.returncode, proc.stdout), (0, "hi\n"))
        self.assertEqual([cred for cred, _ in self.calls if "exec" in _],
                         ["default", "adc"])

    def test_compute_commands_raise_a_structured_error_when_adc_fails(self):
        with self._cluster(adc=False), self.assertRaises(
                auth_utils.ClusterAccessDenied) as ctx:
            GkeJobDriver(image="img")._kubectl(["get", "pods"])
        payload = ctx.exception.to_dict()
        self.assertEqual(payload["error_code"], "CLUSTER_ACCESS_DENIED")
        self.assertEqual(payload["identities"], [
            f"the gcloud account {self.ACCOUNT}",
            "application default credentials (none usable: ERROR: Your "
            "default credentials were not found.)"
        ])
        self.assertIn("gcloud auth application-default login", payload["fix"])

    def test_only_kubectls_own_refusal_counts(self):
        refused = [
            'Error from server (Forbidden): pods "p" is forbidden: User "x"',
            "error: You must be logged in to the server (Unauthorized)",
            "error: You must be logged in to the server (the server has "
            "asked for the client to provide credentials)",
            "W0930 12:00:00 warning\nError from server (Forbidden): nope",
        ]
        from_the_pod = [
            "curl: (22) The requested URL returned error: 403 Forbidden",
            "PermissionError: (forbidden) /data",
            "requests.exceptions.HTTPError: 401 (Unauthorized)",
            "log: Error from server (Forbidden): quoted mid-line",
        ]
        for stderr in refused:
            self.assertTrue(
                auth_utils.credential_refused(
                    subprocess.CompletedProcess([], 1, "", stderr)), stderr)
        for stderr in from_the_pod:
            self.assertFalse(
                auth_utils.credential_refused(
                    subprocess.CompletedProcess([], 1, "", stderr)), stderr)

    def test_a_workload_refused_inside_the_pod_is_not_retried_with_adc(self):
        """Once the exec started, even kubectl's exact prefix is not a retry."""
        in_pod_refusal = ('Error from server (Forbidden): pods is forbidden: '
                          'User "system:serviceaccount:default:default"')
        real_run = subprocess.run
        calls = []

        def run(cmd, **kwargs):
            if "--" not in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            calls.append(cmd)
            # The pod's side of `kubectl exec ... -- sh -c <script>`, run here.
            return real_run(cmd[cmd.index("--") + 1:], **kwargs)

        driver = GkePodDriver(pod="p")
        with patch("subprocess.run", run):
            proc = driver._sh(
                f"echo 'ran'; echo '{in_pod_refusal}' >&2; exit 1")
        self.assertEqual((proc.returncode, proc.stdout), (1, "ran\n"))
        self.assertIn(in_pod_refusal, proc.stderr)
        self.assertEqual(len(calls), 1)
        self.assertFalse(driver.using_adc)

    def test_port_forward_uses_adc_after_the_gcloud_account_is_refused(self):
        drivers = (GkePodDriver(pod="p"), GkeJobDriver(image="img"))
        for driver in drivers:
            os.environ.pop(auth_utils.ACCESS_TOKEN_ENV, None)
            with self.subTest(driver=type(driver).__name__), \
                    self._cluster("adc"):
                cmd, _ = driver._port_forward_command(8080, 80, "job-1")
                self.assertEqual(os.environ[auth_utils.ACCESS_TOKEN_ENV],
                                 "adc-token")
                self.assertIn("port-forward", cmd)
                self.assertNotIn("adc-token", " ".join(cmd))


class TestSetupFieldsAndValidation(unittest.TestCase):
    """SETUP_FIELDS metadata, format validation, and schema/questions consistency."""

    def test_setup_fields_categories_are_disjoint_and_consistent(self):
        pkg_root = pathlib.Path(__file__).resolve().parents[2] / "package"
        schema_path = pkg_root / "schemas" / "environment.json"
        questions_path = pkg_root / "environment" / "questions.json"
        if not schema_path.is_file() or not questions_path.is_file():
            self.skipTest("package schema/questions not available")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        questions_data = json.loads(questions_path.read_text(encoding="utf-8"))

        compute_schema = schema["properties"]["profiles"][
            "additionalProperties"]
        param_props = compute_schema["properties"]["params"]["properties"]
        questions_by_param = {
            q["path"].removeprefix("compute.params."): q
            for q in questions_data["questions"]
            if q.get("path", "").startswith("compute.params.")
        }
        # A missing `when` means "every backend" in questions.json, which is
        # never what a backend-specific param wants, so each one must say.
        for param_name, q in questions_by_param.items():
            self.assertTrue(
                q.get("when", {}).get("backend"),
                f"{param_name} question has no `when.backend`")
        required_flags = compute.describe_required_flags()
        samples = {
            "project": "my-proj",
            "zone": "us-central1-a",
            "tpu_name": "my-tpu",
            "pod": "my-pod",
            "cluster_name": "my-cluster",
            "image": "us-docker.pkg.dev/p/r/i@sha256:abc",
            "tpu_accelerator": "tpu7x",
            "tpu_topology": "2x2x1",
        }
        shipped = {
            k: v
            for k, v in get_registered_drivers().items()
            if not k.startswith("mock")
        }
        # Nothing may exist only in the schema or questions.json: a param the
        # driver reads but no SETUP_FIELDS declares is invisible to setup.
        declared = set().union(*(d.SETUP_FIELDS.all_fields()
                                 for d in shipped.values()))
        self.assertEqual(
            set(param_props) - declared, set(),
            "schema params missing from every SETUP_FIELDS")
        self.assertEqual(
            set(questions_by_param) - declared, set(),
            "compute.params questions missing from every SETUP_FIELDS")

        for backend, driver_cls in shipped.items():
            sf = driver_cls.SETUP_FIELDS
            all_fields = sf.all_fields()
            with self.subTest(backend=backend):
                self.assertEqual(
                    len(sf.required) + len(sf.optional) + len(sf.conditional) +
                    len(sf.on_demand), len(all_fields))
                for child, trigger in sf.conditional.items():
                    self.assertIn(trigger, all_fields, f"{backend}:{child}")
                for field_name in all_fields:
                    self.assertIn(field_name, param_props)
                    # on_demand fields are set when a command first needs
                    # them, not at install, so they need no prompt here.
                    if field_name in sf.on_demand:
                        continue
                    q = questions_by_param[field_name]
                    self.assertIn(backend, q["when"]["backend"])
                for req_field in sf.required:
                    q = questions_by_param[req_field]
                    self.assertNotEqual(
                        q.get("default"), "",
                        f"{backend}:{req_field} defaults to ''")
                self.assertEqual(
                    required_flags.get(backend, set()),
                    {f"--{f.replace('_', '-')}"
                     for f in sf.required},
                )
                full_params = {f: samples[f] for f in sf.required}
                for missing_field in sf.required:
                    incomplete = {
                        k: v
                        for k, v in full_params.items() if k != missing_field
                    }
                    with self.assertRaises(ComputeConfigError) as ctx:
                        drv = driver_cls.from_params(incomplete)
                        if hasattr(drv, "_require_dispatch_config"):
                            drv._require_dispatch_config()
                    self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

        for param_name, q in questions_by_param.items():
            for backend in q["when"]["backend"]:
                if backend in shipped:
                    self.assertIn(param_name,
                                  shipped[backend].SETUP_FIELDS.all_fields())

    def test_validate_field_format(self):
        v = compute.validate_field_format
        for field, good, bad in [
            ("zone", "us-central1-a", "not_a_zone"),
            ("tpu_topology", "2x2x1", "v6e-8"),
            ("tpu_topology", "2x4", "0x4"),
            ("priority", "low", "urgent"),
            ("staging_bucket", "gs://my-bucket/path", "my-bucket"),
            ("gcs_bucket", "gs://my-bucket", "gs://"),
            ("trajectory_bucket", "gs://my-bucket", "http://my-bucket"),
        ]:
            self.assertIsNone(v(field, good), field)
            self.assertIsNotNone(v(field, bad), field)

        region_err = v("zone", "us-central1")
        self.assertIn("is a region, not a zone", region_err["reason"])
        self.assertIn("us-central1-a", region_err["choices"])
        # Priority choices come out in a fixed order, highest first.
        self.assertEqual(
            v("priority", "urgent")["choices"],
            ["", "medium", "low", "very-low"],
        )
        # An unfamiliar accelerator label is left to preflight, which only
        # warns, so a newer TPU generation is not rejected at setup.
        self.assertIsNone(v("tpu_accelerator", "tpu-v99-slice"))

        # Images are accepted whether pinned by digest or by tag.
        self.assertIsNone(v("image", "gcr.io/p/img@sha256:deadbeef"))
        self.assertIsNone(v("image", "gcr.io/p/img:latest"))


class TestMultiProfileResolution(unittest.TestCase):
    """Multi-profile environment.json: shapes, precedence, and named errors."""

    PKG = "pkg@1.0.0"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        patcher = patch.dict(os.environ,
                             {"TPU_AGENT_SUITE_HOME": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.workspace = self.root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.env_file.parent.mkdir(parents=True)

    @staticmethod
    def _profiles(default: str | None, *names: str) -> dict[str, Any]:
        """A multi-profile document whose profiles run `/<name>/python`."""
        return {
            "revision": 1,
            "default": default,
            "profiles": {
                n: {
                    "backend": "local",
                    "params": {
                        "python_interpreter": f"/{n}/python"
                    }
                }
                for n in names
            },
        }

    def _interpreter(self, env: dict[str, Any] | str, **kwargs: Any) -> str:
        """Write `env` (a dict or JSON text); the resolved driver's Python."""
        self.env_file.write_text(
            env if isinstance(env, str) else json.dumps(env), encoding="utf-8")
        return resolve_driver(workspace=self.workspace,
                              package=self.PKG,
                              **kwargs).python_interpreter

    def test_precedence_is_flag_then_job_then_run_then_default(self):
        abc = self._profiles("a", "a", "b", "c")
        self.assertEqual(self._interpreter(abc), "/a/python")
        run_id = runtime.start_run(package=self.PKG,
                                   workspace=self.workspace,
                                   compute_profile="b")["run_id"]
        self.assertEqual(self._interpreter(abc, run_id=run_id), "/b/python")
        self.assertEqual(
            self._interpreter(abc, run_id=run_id, profile_override="c"),
            "/c/python")
        # A job is reached through the profile it was submitted under; a
        # record from before jobs kept one falls back to the run's.
        job = jobs.JobRecord.create(compute_profile="c").job_id
        old = jobs.JobRecord.create().job_id
        self.assertEqual(self._interpreter(abc, run_id=run_id, job_id=job),
                         "/c/python")
        self.assertEqual(self._interpreter(abc, run_id=run_id, job_id=old),
                         "/b/python")
        self.assertEqual(
            self._interpreter(abc, job_id=job, profile_override="a"),
            "/a/python")
        # A job whose profile was deleted falls back, so it can be cancelled.
        gone = jobs.JobRecord.create(compute_profile="gone").job_id
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(
                self._interpreter(abc, run_id=run_id, job_id=gone),
                "/b/python")
        self.assertIn("'gone' no longer exists", err.getvalue())
        # A driver built for another backend is not the profile's target.
        self.assertIsNone(
            resolve_driver(workspace=self.workspace,
                           package=self.PKG,
                           backend_override="mock_test").compute_profile)
        # The CLI forwards --profile and the active run to the resolver.
        with patch.object(runtime.compute.config, "resolve_driver") as m, \
                contextlib.redirect_stdout(io.StringIO()):
            m.return_value.inspect.return_value = MockTestDriver().inspect()
            runtime.main([
                "compute", "--package", self.PKG, "--workspace",
                str(self.workspace), "--profile", "c", "inspect", "--json"
            ])
        self.assertEqual((m.call_args.kwargs["profile_override"],
                          m.call_args.kwargs["run_id"]), ("c", run_id))

    def test_unresolvable_profiles_are_named_errors(self):
        for env, kwargs, code in [
            ('{"default": "typo", "profiles": {"a": {}}}', {},
             "UNKNOWN_PROFILE"),
            ("{}", {
                "profile_override": "x"
            }, "UNKNOWN_PROFILE"),
            ('{"default": null, "profiles": {}}', {},
             "COMPUTE_NOT_CONFIGURED"),
            ('{"profiles": {}}', {
                "profile_override": " "
            }, "INVALID_FIELD"),
            ('{"backend": "mock_test", "endpoint": "flat"}', {},
             "COMPUTE_NOT_CONFIGURED"),
            ('{"profiles": []}', {}, "INVALID_FIELD"),
            ('{"profiles": {"vm": "gce"}}', {}, "INVALID_FIELD"),
            ('{"default": "", "profiles": {}}', {}, "INVALID_FIELD"),
        ]:
            with self.subTest(env):
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._interpreter(env, **kwargs)
                self.assertEqual(ctx.exception.error_code, code)
                if env == '{"default": null, "profiles": {}}':
                    self.assertEqual(ctx.exception.missing, ["backend"])

        # start-run refuses an unknown name, or a name without a package,
        # before recording a run.
        self.env_file.write_text(json.dumps(self._profiles("a", "a")))
        for kwargs, code, text in [
            (dict(package=self.PKG,
                  workspace=self.workspace,
                  compute_profile="nope"), "UNKNOWN_PROFILE",
             "available profiles: a"),
            (dict(compute_profile="a"), "MISSING_FIELD", "--package"),
            (dict(compute_profile=" "), "INVALID_FIELD", "non-empty"),
        ]:
            with self.assertRaises(ComputeConfigError) as ctx:
                runtime.start_run(**kwargs)
            self.assertEqual(ctx.exception.error_code, code)
            self.assertIn(text, ctx.exception.message)
        self.assertFalse(utils.get_runs_root().is_dir()
                         and any(utils.get_runs_root().iterdir()))

    def test_schema_accepts_profiles_and_checks_their_structure(self):
        # Structural only: per-backend completeness belongs to SETUP_FIELDS.
        schema = json.loads(
            (pathlib.Path(__file__).resolve().parents[2] / "package" /
             "schemas" / "environment.json").read_text(encoding="utf-8"))
        ok = '{"default": "vm", "profiles": {"vm": {"backend": "gce_tpu_vm", "params": {"zone": "z"}}}}'
        for doc in (ok, '{"revision": 1, "default": null, "profiles": {}}'):
            self.assertEqual(utils.validate_instance(json.loads(doc), schema),
                             [])
        bad = json.loads(ok.replace('"zone"', '"no_such_param"'))
        self.assertIn("no_such_param",
                      str(utils.validate_instance(bad, schema)))
        self.assertTrue(utils.validate_instance({"default": ""}, schema))


class TestComputeSetupApplyAndConfig(unittest.TestCase):
    """`compute setup apply` and `config [show|set|use]`."""

    PKG = "pkg@1.0.0"
    GKE = {
        "backend": "gke_job",
        "image": "us-docker.pkg.dev/p/r/img:v1",
        "tpu_accelerator": "tpu7x",
        "tpu_topology": "2x2x1",
    }

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = pathlib.Path(self.temp_dir.name)
        self.workspace = root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.write({"revision": 1, "default": None, "profiles": {}})
        env = patch.dict(os.environ,
                         {"TPU_AGENT_SUITE_HOME": str(root / "home")})
        env.start()
        self.addCleanup(env.stop)

    def write(self, doc):
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.write_text(json.dumps(doc), encoding="utf-8")

    def saved(self):
        return json.loads(self.env_file.read_text(encoding="utf-8"))

    def apply(self, profile, answers, **kwargs):
        return compute.apply_setup(profile=profile,
                                   answers=answers,
                                   package=self.PKG,
                                   workspace=self.workspace,
                                   **kwargs)

    def config_set(self, *assignments, profile=None):
        return compute.config_set(assignments=list(assignments),
                                  profile=profile,
                                  package=self.PKG,
                                  workspace=self.workspace)

    def test_apply_incomplete_writes_nothing(self):
        vm = {"backend": "gce_tpu_vm", "project": "p"}
        cases = [
            ({}, "missing", ["backend"]),
            ({
                "backend": "nope"
            }, "invalid", ["backend"]),
            (vm, "missing", ["zone", "tpu_name"]),
            ({
                **vm, "zone": "us-central1",
                "tpu_name": "v"
            }, "invalid", ["zone"]),
            ({
                **self.GKE, "queue": "q"
            }, "newly_applicable", ["priority"]),
            ({
                **self.GKE, "tpu_chips": "abc"
            }, "invalid", ["tpu_chips"]),
            ({
                **self.GKE, "staging_service_account":
                "sa@p.iam.gserviceaccount.com",
                "staging_service_account_key": "/k.json"
            }, "invalid", ["params"]),
            ({
                "backend": "local",
                "xprof_interpreter": "py"
            }, "invalid", ["xprof_interpreter"]),
        ]
        before = self.env_file.read_text(encoding="utf-8")
        for answers, key, names in cases:
            with self.subTest(answers=answers):
                res = self.apply("p", answers)
                self.assertEqual(res["status"], "INCOMPLETE")
                self.assertEqual(sorted(e["name"] for e in res[key]),
                                 sorted(names))
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("my profile", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), before)

    def test_apply_merges_into_stored_profile(self):
        stored = {k: v for k, v in self.GKE.items() if k != "backend"}
        self.write({
            "revision": 3,
            "xprof_interpreter": "/py",
            **profile_env("gke_job", **stored)
        })
        # An on-demand field alone keeps everything already stored.
        self.assertEqual(
            self.apply("default", {"staging_bucket": "gs://b"})["status"],
            "ok")
        saved = self.saved()
        self.assertEqual(
            (saved["revision"], saved["default"], saved["xprof_interpreter"]),
            (4, "default", "/py"))
        self.assertEqual(saved["profiles"]["default"]["params"], {
            **stored, "staging_bucket": "gs://b"
        })
        # A conditional is asked for only when its trigger is newly set, so
        # a queue saved without a priority doesn't block later edits.
        self.assertEqual(
            self.apply("default", {
                "queue": "q",
                "priority": ""
            })["status"], "ok")
        self.assertNotIn("priority",
                         self.saved()["profiles"]["default"]["params"])
        self.assertEqual(
            self.apply("default", {"namespace": "ns"})["status"], "ok")
        # Unsetting the trigger drops the conditional with it.
        self.apply("default", {"priority": "low"})
        self.apply("default", {"queue": ""})
        self.assertFalse(
            {"queue", "priority"}
            & self.saved()["profiles"]["default"]["params"].keys())
        # A backend change keeps only fields the new backend knows.
        self.apply(
            "default", {
                "backend": "gce_tpu_vm",
                "project": "p",
                "zone": "us-east5-a",
                "tpu_name": "v"
            })
        self.assertEqual(self.saved()["profiles"]["default"]["params"], {
            "project": "p",
            "zone": "us-east5-a",
            "tpu_name": "v"
        })
        # Dry runs write nothing; later profiles leave the default alone.
        self.assertTrue(
            self.apply("dev", {"backend": "local"}, dry_run=True)["dry_run"])
        self.assertNotIn("dev", self.saved()["profiles"])
        self.assertEqual(
            self.apply("dev", {"backend": "local"})["default"], "default")
        # A default naming a missing profile is replaced.
        self.write({"revision": 1, "default": "gone", "profiles": {}})
        self.assertEqual(
            self.apply("dev", {"backend": "local"})["default"], "dev")

    def test_setting_compute_up_again_replaces_the_pre_profiles_format(self):
        self.write({"revision": 4, "compute": {"backend": "local"}})
        self.assertEqual(
            self.apply("default", {"backend": "local"})["status"], "ok")
        saved = self.saved()
        self.assertNotIn("compute", saved)
        self.assertEqual((saved["revision"], saved["default"]), (5, "default"))

    def test_config_set_validates_like_apply(self):
        # A blank field that has a default, like gke_job's image, takes it.
        self.assertEqual(
            self.apply("blank", {
                **self.GKE, "image": ""
            }, dry_run=True)["params"]["image"],
            compute.setup._questions()["image"]["default"])
        self.apply("gke", self.GKE)
        before = self.saved()
        self.assertEqual(
            self.config_set("tpu_topology=")["missing"][0]["name"],
            "tpu_topology")
        self.assertEqual(self.saved(), before)
        res = self.config_set("declared_duration_minutes=30",
                              "xprof_interpreter=/py")
        self.assertEqual((res["status"], res["revision"]), ("ok", 3))
        self.assertEqual(
            self.saved()["profiles"]["gke"]["params"]
            ["declared_duration_minutes"], 30)
        self.assertEqual(self.saved()["xprof_interpreter"], "/py")
        for args, kwargs, code in [(("trajectory_enabled=true", ), {},
                                    "INVALID_FIELD"),
                                   (("queue=x", ), {
                                       "profile": "nope"
                                   }, "UNKNOWN_PROFILE"),
                                   (("no-equals", ), {}, "INVALID_FIELD")]:
            with self.subTest(args=args), self.assertRaises(
                    compute.ComputeConfigError) as ctx:
                self.config_set(*args, **kwargs)
            self.assertEqual(ctx.exception.error_code, code)

    def test_environment_location_is_never_guessed(self):
        self.env_file.write_text("{not json", encoding="utf-8")
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("dev", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        # Other runtime commands, such as trajectory export, say so too.
        with self.assertRaisesRegex(runtime.TpuAgentSuiteError,
                                    "not valid JSON"):
            runtime.get_environment(self.workspace, self.PKG)
        # An ownership ledger without environment.json is repaired ...
        self.env_file.unlink()
        (self.env_file.parent / "ownership.json").write_text(json.dumps(
            {"package": "pkg"}),
                                                             encoding="utf-8")
        self.assertEqual(
            self.apply("dev", {"backend": "local"})["status"], "ok")
        # ... but a package installed nowhere is an error.
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            compute.config_show(package=self.PKG,
                                workspace=pathlib.Path(self.temp_dir.name))
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

    def test_cli(self):
        where = ["--package", self.PKG, "--workspace", str(self.workspace)]

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = runtime.main(list(argv))
            return rc, (json.loads(out.getvalue()) if out.getvalue() else None)

        answers = pathlib.Path(self.temp_dir.name) / "answers.json"
        answers.write_text(json.dumps(self.GKE), encoding="utf-8")
        self.assertEqual(
            run("compute", *where, "setup", "apply", "--profile", "gke",
                "--answers", str(answers), "--json")[0], 0)
        self.assertEqual(
            run("compute", "setup", "apply", *where, "--profile", "dev",
                "--answers", str(answers), "--json")[0], 0)
        self.assertEqual(
            run("config", *where, "set", "queue=q", "--json")[0], 1)
        # Flags before the subcommand are kept.
        rc, shown = run("config", *where, "show", "--json")
        self.assertEqual((rc, shown["active_profile"], shown["active_source"]),
                         (0, "gke", "default"))
        run_id = runtime.start_run(package=self.PKG,
                                   workspace=self.workspace)["run_id"]
        self.assertEqual(run("config", "use", "dev", *where, "--json")[0], 0)
        with self.assertRaisesRegex(compute.ComputeConfigError,
                                    r"^Compute profile 'nope' is not"):
            compute.config_use(profile="nope",
                               package=self.PKG,
                               workspace=self.workspace,
                               run_id=run_id)
        # With no --profile, `set` changes the run's profile, not the default.
        rc, res = run("config", "set", "declared_duration_minutes=45", *where,
                      "--json")
        self.assertEqual((rc, res["profile"]), (0, "dev"))
        rc, shown = run("config", "show", *where, "--run-id", run_id, "--json")
        self.assertEqual((shown["active_profile"], shown["active_source"]),
                         ("dev", "run"))
        self.assertEqual([
            shown["profiles"][p]["params"].get("declared_duration_minutes")
            == 45 for p in ("gke", "dev")
        ], [False, True])
        self.assertEqual(
            run("config", "use", "dev", "--default", *where, "--json")[0], 0)
        self.assertEqual(self.saved()["default"], "dev")
        # Trajectory sharing changes only once the user has agreed, and lands
        # where the export reads it.
        sharing = ("trajectory_enabled=true", "trajectory_bucket=gs://b")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc, _ = run("config", "set", *sharing, *where, "--json")
        self.assertNotEqual(rc, 0)
        self.assertIn("--user-confirmed", err.getvalue())
        self.assertNotIn("trajectory", self.saved())
        self.assertEqual(
            run("config", "set", *sharing, "--user-confirmed", *where,
                "--json")[0], 0)
        self.assertEqual(self.saved()["trajectory"],
                         {"params": {
                             "enabled": True,
                             "bucket": "gs://b"
                         }})
        # A Harbor org is checked with the hub.
        with patch.object(SETUP, "_harbor_org_exists",
                          return_value=False), \
                contextlib.redirect_stderr(err):
            rc, _ = run("config", "set", "trajectory_sink=harbor",
                        "trajectory_share_org=nope", "--user-confirmed",
                        *where, "--json")
        self.assertEqual(rc, 1)
        self.assertIn("no organization named 'nope'", err.getvalue())
        self.assertIn('"INVALID_FIELD"', err.getvalue())
        self.assertNotIn("share_org", self.saved()["trajectory"]["params"])
        # Turning sharing off takes the destination with it, and the export
        # then refuses rather than uploading to the old bucket.
        rc, res = run("config", "set", "trajectory_enabled=false",
                      "--user-confirmed", *where, "--json")
        self.assertEqual(self.saved()["trajectory"],
                         {"params": {
                             "enabled": False
                         }})
        self.assertIn("trajectory_bucket", res["removed"][0])
        with contextlib.redirect_stderr(err):
            rc = runtime.main([
                "trajectory", "export", "--harness", "claude-code",
                "--package", self.PKG, "--workspace",
                str(self.workspace)
            ])
        self.assertEqual(rc, 1)
        self.assertIn("trajectory sharing is off", err.getvalue())

    def test_verify_records_a_result_per_profile_hash(self):
        """A result counts until the profile's settings change."""
        self.apply(
            "vm", {
                "backend": "gce_tpu_vm",
                "project": "p",
                "zone": "us-east5-a",
                "tpu_name": "v"
            })
        driver = MagicMock(quick_run=True, scheduled_run=False)
        driver.preflight.side_effect = lambda: base.PreflightReport(
            "gce_tpu_vm", "vm", True, [])

        def verify(stdout, *flags):
            if stdout is not None:
                driver.run.return_value = base.SynchronousExecutionResult(
                    0 if "ok" in stdout else 1, stdout, "")
            out = io.StringIO()
            with patch.object(compute.GceTpuVmDriver,
                              "from_params",
                              return_value=driver), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", "setup", "verify", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            return rc, json.loads(out.getvalue())

        def state():
            return compute.config_show(
                package=self.PKG,
                workspace=self.workspace)["setup_state"]["vm"]

        self.assertEqual(state(), "UNVERIFIED")
        rc, res = verify("[FAIL] torch_tpu")
        self.assertEqual((rc, res["status"], res["fields"]),
                         (1, "FAILED", ["python_interpreter"]))
        self.assertEqual(state(), "FAILED")
        # The failing checks are kept, for session start to show.
        results = self.env_file.parent / "verification.json"
        (saved, ) = json.loads(results.read_text()).values()
        self.assertEqual([c["fields"] for c in saved["failed_checks"]],
                         [["python_interpreter"]])
        rc, res = verify("Status: ok", "--profile", "vm")
        self.assertEqual((rc, res["status"], res["smoke_test"]),
                         (0, "ok", False))
        self.assertEqual(state(), "READY")
        # A result from another package version doesn't count.
        setup_utils = sys.modules[compute.config_show.__module__].utils
        with patch.object(setup_utils, "TPU_AGENT_SUITE_VERSION", "0.0.1"):
            self.assertEqual(state(), "UNVERIFIED")
        # Flags before the subcommand aren't overwritten by its defaults.
        args = runtime.build_runtime_parser().parse_args(
            ["compute", "--run-id", "r", "--profile", "vm", "setup", "verify"])
        self.assertEqual((args.run_id, args.profile), ("r", "vm"))
        self.assertEqual(len(json.loads(results.read_text())), 1)
        # A smoke test still pending at its timeout passes, but didn't run.
        driver.configure_mock(quick_run=False, scheduled_run=True)
        driver.run.return_value = base.SynchronousExecutionResult(
            1, "", "", {
                "timed_out_after_sec": 900,
                "job_status": "PENDING"
            })
        rc, res = verify(None, "--smoke-test")
        self.assertEqual((rc, res["status"], res["smoke_test"]),
                         (0, "ok", False))
        self.config_set("zone=us-east5-b")
        self.assertEqual(state(), "UNVERIFIED")
        # An incomplete profile is reported, not checked.
        self.write(profile_env("gce_tpu_vm"))
        self.assertEqual(verify("")[1]["status"], "INCOMPLETE")
        self.assertEqual(
            compute.config_show(package=self.PKG,
                                workspace=self.workspace)["setup_state"],
            {"default": "NOT_CONFIGURED"})

    def test_failed_checks_name_the_fields_to_ask_again(self):

        class Driver(compute.LocalDriver):
            CHECK_FIELDS = {"tpu_devices": ("python_interpreter", )}

            def _preflight_target(self, checks):
                checks += [
                    base.PreflightCheck("tpu_devices", "fail"),
                    base.PreflightCheck("other", "fail")
                ]
                return True

        self.assertEqual([c.fields for c in Driver().preflight().checks],
                         [["python_interpreter"], []])
        for name, cls in compute.drivers.registry.DRIVER_REGISTRY.items():
            named = {f for fs in cls.CHECK_FIELDS.values() for f in fs}
            self.assertLessEqual(named, cls.SETUP_FIELDS.all_fields(), name)

    def test_plan_lists_questions_with_suggestions(self):
        outputs = {
            "kubectl get nodes": "tpu7x 2x2x1\ntpu7x 2x2x1\ntpu-v6e-slice 2x4",
            "kubectl get localqueues": "lq other",
            "gcloud config get-value": "other-proj",
            "gcloud compute tpus": "vm1 us-west4-a",
            "kubectl config current-context": "gke_my-proj_us-central1_c1",
        }
        ran = []

        def run(cmd):
            ran.append(" ".join(cmd))
            return outputs.get(
                " ".join(cmd[:3]),
                ""), ("kubectl is not installed" if cmd[2] == "view" else None)

        def plan(*flags, before=()):
            out = io.StringIO()
            with patch.object(compute.probes, "_run", side_effect=run), \
                    patch.object(compute.probes.probe, "list_devices",
                                 return_value=([], [])), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", *before, "setup", "plan", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            res = json.loads(out.getvalue() or "{}")
            return rc, res, {q["name"]: q for q in res.get("questions", [])}

        # No backend: only the backend question, and the profiles to pick.
        self.write({"default": None, "profiles": {"vm": {"backend": "local"}}})
        _, res, qs = plan()
        self.assertEqual((res["profiles"], list(qs)), ({
            "vm": "local"
        }, ["backend"]))
        self.assertEqual(
            (qs["backend"]["suggested"], qs["backend"]["required"]),
            ("gke_job", True))
        # A backend's setup fields, suggested by probes or defaults. Flags
        # before the subcommand count too.
        _, res, qs = plan(before=("--profile", "gke", "--backend", "gke_job"))
        self.assertEqual((res["profile"], res["backend"]), ("gke", "gke_job"))
        self.assertEqual(list(qs), [
            "image", "tpu_accelerator", "tpu_topology", "kube_context",
            "namespace", "queue", "hf_cache_pvc", "priority"
        ])
        # The current context is suggested, so the profile pins it.
        self.assertEqual(
            (qs["kube_context"]["suggested"],
             qs["kube_context"]["suggested_from"]),
            ("gke_my-proj_us-central1_c1", "kubectl config current-context"))
        self.assertEqual([(qs[f]["suggested"], qs[f]["suggested_from"])
                          for f in ("tpu_accelerator", "tpu_topology")],
                         [("tpu7x", "kubectl get nodes"),
                          ("2x2x1", "kubectl get nodes")])
        self.assertEqual(qs["queue"]["suggested"], "lq")
        self.assertEqual(qs["image"]["suggested_from"], "default")
        self.assertEqual(qs["priority"]["asked_if"], "queue")
        self.assertNotIn("suggested", qs["namespace"])
        # The shared HF cache claim is asked at setup, optional and blank by
        # default: nothing else would ever ask for it.
        self.assertEqual(
            (qs["hf_cache_pvc"]["required"], qs["hf_cache_pvc"]["default"]),
            (False, ""))
        self.assertNotIn("suggested", qs["hf_cache_pvc"])
        # A missing tool is reported, deduplicated.
        self.assertEqual(res["probe_warnings"], ["kubectl is not installed"])
        self.assertIn("optional ones too", res["note"])
        # With a backend chosen, a cluster with no TPU nodes is reported;
        # without one, no nodes just means another backend.
        nodes = outputs.pop("kubectl get nodes")
        for flags, reported in (((), False), (("--backend", "gke_job"), True)):
            _, res, _ = plan("--profile", "g2", *flags)
            self.assertEqual(
                any("no TPU nodes" in w
                    for w in res.get("probe_warnings", [])), reported)
        outputs["kubectl get nodes"] = nodes
        # Stored values are suggested as they are, except with --fields,
        # which asks again for values verify rejected.
        self.apply("gke", self.GKE)
        _, res, qs = plan("--profile", "gke")
        self.assertEqual(qs["tpu_accelerator"]["suggested_from"], "profile")
        _, res, qs = plan("--profile", "gke", "--fields",
                          "tpu_accelerator,staging_bucket")
        self.assertEqual(list(qs), ["tpu_accelerator", "staging_bucket"])
        self.assertEqual((qs["tpu_accelerator"]["current"],
                          qs["tpu_accelerator"]["suggested_from"]),
                         ("tpu7x", "kubectl get nodes"))
        # A saved kubeconfig is used by the kubectl probes.
        self.apply("kc", {**self.GKE, "kubeconfig": "/k"})
        plan("--profile", "kc", "--fields", "queue")
        self.assertIn(
            "kubectl get localqueues -o jsonpath="
            "{.items[*].metadata.name} --kubeconfig /k", ran)
        self.assertEqual(plan("--profile", "gke", "--fields", "zone")[0], 1)
        # Trajectory settings need consent, so plan won't ask for them.
        self.assertEqual(
            plan("--profile", "gke", "--fields", "trajectory_enabled")[0], 1)
        # gke_xpk's project, cluster and zone all come from the kubectl
        # context, even when gcloud's project differs.
        _, _, qs = plan("--profile", "xpk", "--backend", "gke_xpk")
        self.assertEqual(
            [qs[f]["suggested"] for f in ("project", "cluster_name", "zone")],
            ["my-proj", "c1", "us-central1"])
        self.assertEqual(
            plan("--profile", "my vm", "--backend", "gke_xpk")[0], 1)
        # gcloud prints "(unset)" when it has no project.
        outputs["gcloud config get-value"] = "(unset)"
        # A TPU VM is only suggested when it's clear which one is meant.
        _, _, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual(qs["tpu_name"]["suggested"], "vm1")
        self.assertNotIn("suggested", qs["project"])
        outputs["gcloud config get-value"] = ""  # some versions print nothing
        _, _, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertNotIn("suggested", qs["project"])
        outputs["gcloud compute tpus"] = "vm1 us-west4-a\nme-vm us-east5-a"
        with patch.dict(os.environ, {"USER": "me"}):
            _, _, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual([qs[f]["suggested"] for f in ("tpu_name", "zone")],
                         ["me-vm", "us-east5-a"])
        with patch.dict(os.environ, {"USER": "nobody"}):
            _, res, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertNotIn("suggested", qs["tpu_name"])
        # With no clear pick the VMs are offered as choices instead, each
        # with its zone, and no zone is suggested when they span several.
        self.assertEqual(qs["tpu_name"]["choices"], [{
            "tpu_name": "vm1",
            "zone": "us-west4-a"
        }, {
            "tpu_name": "me-vm",
            "zone": "us-east5-a"
        }])
        self.assertNotIn("suggested", qs["zone"])
        self.assertNotIn("choices", qs["zone"])
        self.assertNotIn("choices", qs["project"])
        self.assertIn("free text", res["note"])
        # $USER's VMs come first, and the list is capped.
        many = "\n".join(f"vm{i} z{i % 2}" for i in range(20))
        with patch.dict(os.environ, {"USER": "ann"}):
            found = compute.probes._tpu_vm(f"{many}\nann-a z9\nann-b z9")
        self.assertEqual([c["tpu_name"] for c in found["tpu_name"][:3]],
                         ["ann-a", "ann-b", "vm0"])
        self.assertEqual(len(found["tpu_name"]), compute.probes.MAX_CHOICES)
        self.assertNotIn("zone", found)
        self.assertEqual(compute.probes._tpu_vm(""), {})
        # VMs that share a zone still get it suggested.
        self.assertEqual(compute.probes._tpu_vm("a z\nb z")["zone"], "z")
        # ... but only if the VMs left off the list share it too.
        cut = "\n".join(f"vm{i} z" for i in range(20)) + "\nlast elsewhere"
        found = compute.probes._tpu_vm(cut)
        self.assertEqual(len(found["tpu_name"]), compute.probes.MAX_CHOICES)
        self.assertNotIn("zone", found)
        # The same name in two zones stays two distinct choices.
        self.assertEqual(
            compute.probes._tpu_vm("dup za\ndup zb")["tpu_name"], [{
                "tpu_name": "dup",
                "zone": "za"
            }, {
                "tpu_name": "dup",
                "zone": "zb"
            }])
        # Several VMs still point to gce_tpu_vm as the backend.
        self.assertEqual(
            compute.probes._backend({}, outputs["gcloud compute tpus"],
                                    ["gce_tpu_vm"])[0], "gce_tpu_vm")
        # After a backend change, the values apply would keep are current.
        self.apply(
            "vm", {
                "backend": "gce_tpu_vm",
                "project": "p",
                "zone": "us-east5-a",
                "tpu_name": "v"
            })
        _, _, qs = plan("--profile", "vm", "--backend", "gke_xpk")
        self.assertEqual([
            qs[f].get("current") for f in ("project", "zone", "cluster_name")
        ], ["p", "us-east5-a", None])
        # The context's cluster is in another project, so isn't suggested.
        self.assertNotIn("suggested", qs["cluster_name"])
        # --fields needs a backend.
        self.assertEqual(plan("--profile", "new", "--fields", "zone")[0], 1)

    def test_a_probe_reports_a_missing_tool_or_a_timeout(self):
        self.assertEqual(compute.probes._run(["no-such-tool-xyz"]),
                         ("", "no-such-tool-xyz is not installed"))
        with patch.object(compute.probes, "PROBE_TIMEOUT_SEC", 0.2):
            self.assertEqual(compute.probes._run(["sleep", "5"]),
                             ("", "`sleep 5` did not answer within 0.2s"))
        # A command that runs and fails just has nothing to suggest.
        self.assertEqual(compute.probes._run(["false"]), ("", None))


class TestTargetSoftwareCheck(unittest.TestCase):
    """The software check `compute setup verify` runs on the target."""

    def setUp(self):
        super().setUp()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.requirements = tmp / utils.TARGET_REQUIREMENTS
        self.requirements.write_text(
            "# comment\njax>=0.11.0\n--index-url https://x\n",
            encoding="utf-8")

    def _check(self,
               target_run=None,
               *,
               backend="local",
               params=None,
               scheduled=False,
               smoke_test=False):
        """Run `compute.setup.check_target` on a mock driver for `backend`."""
        real = get_driver(backend)
        driver = MagicMock(quick_run=target_run is not None and not scheduled,
                           scheduled_run=scheduled,
                           backend_name=backend,
                           runs_in_container=real.runs_in_container,
                           SETUP_FIELDS=real.SETUP_FIELDS)
        driver.preflight.return_value = base.PreflightReport(
            backend, "this host", True,
            [base.PreflightCheck("tpu_devices", "ok")])
        driver.run.return_value = target_run
        out = io.StringIO()
        report = compute.setup.check_target(
            driver, {"python_interpreter": "/venv/bin/python"}
            if params is None else params,
            self.requirements,
            smoke_test=smoke_test,
            out=out)
        return report, out.getvalue(), driver

    def test_the_software_check_runs_on_the_target_and_counts(self):
        failed = base.SynchronousExecutionResult(1, "[FAIL] torch_tpu", "")
        report, out, driver = self._check(failed)
        self.assertEqual(report["status"], "fail")
        self.assertIn("[FAIL] torch_tpu", out)
        call = driver.run.call_args
        self.assertEqual(call.args[0], str(compute.setup.ENV_CHECK))
        # Arguments, not a staged file: a gke_job without a staging bucket
        # can carry a script but no files.
        self.assertEqual(
            call.kwargs["args"],
            ["--require=jax>=0.11.0", "--require=--index-url https://x"])
        self.assertNotIn("inputs", call.kwargs)
        self.assertEqual(call.kwargs["timeout"], 300)

    def test_the_targets_package_releases_are_returned_not_printed(self):
        report = ("[ok  ] jax  0.11.2 >=0.11.0\n\nStatus: ok\n"
                  'Versions: {"jax": "0.11.2", "torch_tpu": null}\n')
        found, out, _ = self._check(
            base.SynchronousExecutionResult(0, report, ""))
        self.assertEqual({
            "jax": "0.11.2",
            "torch_tpu": None
        }, found["target_versions"])
        # The line is for `compute versions`; the person reads the checks.
        self.assertNotIn("Versions:", out)
        found, _, _ = self._check(
            base.SynchronousExecutionResult(0, "Status: ok\n", ""))
        self.assertIsNone(found["target_versions"])

    def test_a_scheduled_backend_smoke_tests_only_when_asked(self):
        failed = base.SynchronousExecutionResult(
            1, "", "The image could not be pulled: ImagePullBackOff")
        gke = {"backend": "gke_job", "params": {"image": "img:1"}}
        report, out, driver = self._check(failed,
                                          backend=gke["backend"],
                                          params=gke["params"],
                                          scheduled=True)
        driver.run.assert_not_called()
        self.assertIn("pass --smoke-test", out)
        # Skipping the smoke test leaves a target that is otherwise ready.
        self.assertEqual(report["status"], "ok")

        # Asked for: the run's error is shown, and a failure counts.
        report, out, driver = self._check(failed,
                                          backend=gke["backend"],
                                          params=gke["params"],
                                          scheduled=True,
                                          smoke_test=True)
        self.assertEqual(driver.run.call_args.kwargs["timeout"],
                         compute.setup.SMOKE_TEST_TIMEOUT_SEC)
        # A pip install would not outlast the pod; fixes name the image.
        self.assertEqual(driver.run.call_args.kwargs["args"][-1], "--image")
        self.assertIn("ImagePullBackOff", out)
        self.assertEqual(report["status"], "fail")

    def test_a_smoke_test_that_never_started_is_unverified_not_failed(self):
        """A busy queue is not a broken target: nothing was learned."""
        queued = base.SynchronousExecutionResult(
            1,
            "",
            "Job is suspended: the cluster admits work through a queue",
            details={
                "job_status": "PENDING",
                "timed_out_after_sec": 900
            })
        report, out, _ = self._check(queued,
                                     backend="gke_job",
                                     params={"image": "i"},
                                     scheduled=True,
                                     smoke_test=True)
        self.assertIn("[WARN] the check was still pending after 900s", out)
        self.assertIn("Job is suspended", out)
        self.assertNotEqual(report["status"], "fail")

    def test_the_software_check_is_skipped_where_it_cannot_run_quickly(self):
        failed = base.SynchronousExecutionResult(1, "", "")
        # A backend that can neither run a script quickly nor schedule one
        # to (gke_xpk).
        _, _, driver = self._check()
        driver.run.assert_not_called()
        # A build without the requirements to check against.
        self.requirements.unlink()
        _, _, driver = self._check(failed)
        driver.run.assert_not_called()

    def test_a_software_check_that_cannot_run_is_a_failure(self):
        """It asks again for the profile's image, or else its interpreter."""
        for backend, field in (("gke_job", "image"), ("gke_pod",
                                                      "python_interpreter")):
            broken = MagicMock(backend_name=backend,
                               SETUP_FIELDS=get_driver(backend).SETUP_FIELDS)
            broken.run.side_effect = OSError("kubectl cp failed")
            check, versions = compute.setup._run_target_check(
                broken, self.requirements)
            self.assertEqual((check.status, check.fields), ("fail", [field]))
            self.assertIsNone(versions)

    def test_a_failed_report_is_not_followed_by_the_exit_summary(self):
        """The report explains a failed check; stderr is for one that never
        got to report, like an image that did not pull."""
        cases = (
            (base.SynchronousExecutionResult(
                1, "[FAIL] torch  2.10.0 installed\n\nStatus: fail\n",
                "container exited with code 1"), False),
            (base.SynchronousExecutionResult(
                1, "", "The image could not be pulled: ErrImagePull"), True),
        )
        for result, stderr_shown in cases:
            driver = MagicMock()
            driver.run.return_value = result
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(
                    compute.setup._run_target_check(
                        driver, self.requirements)[0].status, "fail")
            self.assertEqual(result.stderr in out.getvalue(), stderr_shown,
                             out.getvalue())

    def test_local_without_an_interpreter_skips_the_software_check(self):
        # It would check this Python, not the workload's.
        _, out, driver = self._check(base.SynchronousExecutionResult(
            1, "", ""),
                                     params={})
        driver.run.assert_not_called()
        self.assertIn("set python_interpreter", out)


# The module `compute setup` and `config` run in.
SETUP = sys.modules[compute.config_show.__module__]


class ShareOrgCheckTest(unittest.TestCase):
    """A Harbor org the hub does not have is refused when saved, not at
    upload."""

    @staticmethod
    def _answers(org: str, sink: str = "harbor") -> dict:
        return {"trajectory": {"params": {"sink": sink, "share_org": org}}}

    def _check(self,
               answers: dict,
               exists: bool | Exception = True,
               previous: dict | None = None):
        """Run the check with the hub lookup stubbed; return it and stderr."""
        stub = ({
            "side_effect": exists
        } if isinstance(exists, Exception) else {
            "return_value": exists
        })
        stderr = io.StringIO()
        with patch.object(SETUP, "_harbor_org_exists", **stub) as lookup, \
                contextlib.redirect_stderr(stderr):
            SETUP.check_share_org(answers, previous)
        return lookup, stderr.getvalue()

    def test_an_unknown_org_is_refused(self) -> None:
        """Every export would otherwise fail at upload, as `tpu-pytorch` did."""
        with self.assertRaises(SETUP.utils.TpuAgentSuiteError) as caught:
            self._check(self._answers("tpu-pytorch"), exists=False)
        self.assertIn("no organization named 'tpu-pytorch'",
                      str(caught.exception))

    def test_an_org_that_cannot_be_checked_is_kept_with_a_warning(
            self) -> None:
        """No harbor, no login or no network is not evidence the org is wrong."""
        _, stderr = self._check(self._answers("my-team"),
                                exists=ImportError("No module named 'harbor'"))
        self.assertIn("harbor is not installed", stderr)
        _, stderr = self._check(self._answers("my-team"),
                                exists=TimeoutError())
        self.assertIn("could not check the Harbor organization", stderr)

    def test_no_lookup_unless_sharing_to_harbor(self) -> None:
        """Export off, a bucket sink, a private upload or no trajectory: no call."""
        disabled = self._answers("my-team")
        disabled["trajectory"]["params"]["enabled"] = False
        for answers in (disabled, self._answers("my-team",
                                                "gcs"), self._answers(""), {}):
            with self.subTest(answers=answers):
                lookup, _ = self._check(answers)
                lookup.assert_not_called()

    def test_only_a_changed_org_is_looked_up(self) -> None:
        lookup, _ = self._check(self._answers("my-team"),
                                previous=self._answers("my-team"))
        lookup.assert_not_called()
        lookup, _ = self._check(self._answers("my-team"),
                                previous=self._answers("old-team"))
        lookup.assert_called_once_with("my-team")


class TestTargetVersions(unittest.TestCase):
    """`compute versions`: the releases on the target, never this machine's."""

    PKG = "pkg@1.0.0"
    GKE = {
        "backend": "gke_job",
        "image": "us-docker.pkg.dev/p/r/img:v1",
        "tpu_accelerator": "tpu7x",
        "tpu_topology": "2x2x1",
    }

    def setUp(self):
        super().setUp()
        self.cli = importlib.import_module(compute.__name__ + ".cli")
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.workspace = self.tmp / "ws"
        self.env_file = (self.workspace / ".tpu-agent-suite" /
                         "environment.json")
        self.env_file.parent.mkdir(parents=True)
        self.env_file.write_text(json.dumps({
            "revision": 1,
            "default": None,
            "profiles": {}
        }),
                                 encoding="utf-8")
        env = patch.dict(os.environ,
                         {"TPU_AGENT_SUITE_HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)

    def _python_with(self, **packages: str) -> str:
        """An interpreter whose installed packages are exactly `packages`."""
        site = self.tmp / "site"
        for name, version in packages.items():
            meta = site / f"{name}-{version}.dist-info"
            meta.mkdir(parents=True)
            (meta / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                encoding="utf-8")
        python = self.tmp / "python"
        # -S: no site-packages, so only `site` is installed.
        python.write_text(
            f'#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} -S "$@"\n',
            encoding="utf-8")
        python.chmod(0o755)
        return str(python)

    def _apply(self, answers: dict[str, Any]) -> None:
        result = compute.apply_setup(profile="default",
                                     answers=answers,
                                     package=self.PKG,
                                     workspace=self.workspace)
        self.assertEqual("ok", result["status"], result)

    def _versions(self, *flags: str) -> dict[str, Any]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = runtime.main([
                "compute", "versions", "--package", self.PKG, "--workspace",
                str(self.workspace), "--json", *flags
            ])
        self.assertEqual(0, rc)
        return json.loads(out.getvalue())

    def _smoke_test(self,
                    report: str,
                    smoke_test: bool = True) -> dict[str, Any]:
        """`compute setup verify --smoke-test` on the gke_job profile, with
        the Job's output `report`; without the flag if not `smoke_test`."""
        driver = MagicMock(quick_run=False,
                           scheduled_run=True,
                           backend_name="gke_job",
                           runs_in_container=True,
                           SETUP_FIELDS=GkeJobDriver.SETUP_FIELDS)
        driver.preflight.side_effect = lambda: base.PreflightReport(
            "gke_job", "cluster", True, [])
        driver.run.return_value = base.SynchronousExecutionResult(
            0, report, "")
        out = io.StringIO()
        with patch.object(GkeJobDriver, "from_params", return_value=driver), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            runtime.main([
                "compute", "setup", "verify", "--package", self.PKG,
                "--workspace",
                str(self.workspace), "--json",
                *(["--smoke-test"] if smoke_test else [])
            ])
        return json.loads(out.getvalue())

    def test_a_local_target_is_asked_with_its_interpreter_not_this_one(self):
        python = self._python_with(jax="0.11.2")
        self._apply({"backend": "local", "python_interpreter": python})
        # This process has another jax; the target's is the one reported.
        with patch("importlib.metadata.version", return_value="0.10.2"):
            found = self._versions()
        self.assertEqual(("live", "0.11.2", None),
                         (found["source"], found["packages"]["jax"],
                          found["packages"]["torch_tpu"]))

    def test_a_local_interpreter_that_prints_first_still_answers(self):
        """A sitecustomize may print before env_check.py's JSON line."""
        python = pathlib.Path(self._python_with(jax="0.11.2"))
        python.write_text(python.read_text(encoding="utf-8").replace(
            "PYTHONPATH", "echo 'site hello'\nPYTHONPATH"),
                          encoding="utf-8")
        found = self._versions("--backend", "local", "--interpreter",
                               str(python))
        self.assertEqual(("live", "0.11.2"),
                         (found["source"], found["packages"]["jax"]))

    def test_a_local_interpreter_that_cannot_answer_is_unknown(self):
        found = self._versions("--backend", "local", "--interpreter",
                               str(self.tmp / "missing"))
        self.assertEqual((None, {}), (found["source"], found["packages"]))

    def test_a_remote_target_reads_what_verify_recorded(self):
        self._apply(self.GKE)
        self.assertIsNone(self._versions()["source"])
        res = self._smoke_test("[ok  ] jax 0.11.0\n\nStatus: ok\n"
                               'Versions: {"jax": "0.11.0"}\n')
        self.assertEqual({"jax": "0.11.0"}, res["target_versions"]["packages"])
        with patch("subprocess.run") as run:
            found = self._versions()
        run.assert_not_called()
        self.assertEqual(("recorded", {
            "jax": "0.11.0"
        }, self.GKE["image"]),
                         (found["source"], found["packages"], found["image"]))
        self.assertEqual(res["checked_at"], found["checked"])

    def test_stderr_in_the_job_log_does_not_lose_the_releases(self):
        """A Job's log is the container's stdout and stderr together. The
        node logs each stream line by line, so stderr written without a
        newline stays off the `Versions:` line (seen on a real gke_job)."""
        self._apply(self.GKE)
        res = self._smoke_test(
            "W1012 a stderr line\n[ok  ] jax 0.11.0\n\nStatus: ok\n"
            'Versions: {"jax": "0.11.0"}\n'
            "W1012 stderr before,  W1012 stderr after\n")
        self.assertEqual({"jax": "0.11.0"}, res["target_versions"]["packages"])
        self.assertEqual("recorded", self._versions()["source"])

    def test_a_record_for_another_image_is_not_the_answer(self):
        """The job image can change after the check: then it's unknown."""
        self._apply(self.GKE)
        self._smoke_test('Status: ok\nVersions: {"jax": "0.11.0"}\n')
        compute.config_set(assignments=["image=us-docker.pkg.dev/p/r/img:v2"],
                           profile=None,
                           package=self.PKG,
                           workspace=self.workspace)
        found = self._versions()
        self.assertEqual((None, {}), (found["source"], found["packages"]))
        self.assertEqual((self.GKE["image"], "us-docker.pkg.dev/p/r/img:v2", {
            "jax": "0.11.0"
        }), (found["stale"]["image"], found["stale"]["configured_image"],
             found["stale"]["packages"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.main([
                "compute", "versions", "--package", self.PKG, "--workspace",
                str(self.workspace)
            ])
        self.assertIn("unknown: the profile changed", out.getvalue())
        self.assertIn("img:v2", out.getvalue())

    def test_a_verify_without_the_smoke_test_keeps_the_releases(self):
        """The profile is unchanged, so what was recorded still holds."""
        self._apply(self.GKE)
        first = self._smoke_test('Status: ok\nVersions: {"jax": "0.11.0"}\n')
        res = self._smoke_test("", smoke_test=False)
        self.assertFalse(res["smoke_test"])
        found = self._versions()
        self.assertEqual(
            ("recorded", {
                "jax": "0.11.0"
            }, first["checked_at"]),
            (found["source"], found["packages"], found["checked"]))

    def test_a_check_that_reports_no_releases_records_none(self):
        self._apply(self.GKE)
        res = self._smoke_test("Status: ok\n")
        self.assertNotIn("target_versions", res)
        self.assertEqual(
            (None, {}),
            (self._versions()["source"], self._versions()["packages"]))

    def test_the_cli_needs_no_driver(self):
        self._apply(self.GKE)
        args = argparse.Namespace(compute_action="versions",
                                  package=self.PKG,
                                  workspace=self.workspace,
                                  json=True)
        resolve = MagicMock()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, compute.handle_compute_cli(args, resolve))
        resolve.assert_not_called()
        self.assertEqual("gke_job", json.loads(out.getvalue())["backend"])


if __name__ == "__main__":
    unittest.main()
