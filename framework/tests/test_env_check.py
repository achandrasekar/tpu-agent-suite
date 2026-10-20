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
"""The software check the installer runs on the target and on this machine."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "compute" / "env_check.py"
_spec = importlib.util.spec_from_file_location("env_check", _SCRIPT)
env_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env_check)


class TestEnvCheck(unittest.TestCase):

    def test_each_requirement_is_checked_with_its_fix(self):
        versions = {"jax": "0.10.0", "libtpu": "0.0.44"}
        with mock.patch.object(env_check,
                               "version_of",
                               side_effect=versions.get):
            results = {
                req: env_check.check_requirement(req)
                for req in ("diffusers>=0.28.0", "jax>=0.11.0",
                            "libtpu>=0.0.41", "torch>=2.13.0,<2.14")
            }
        self.assertEqual(results["diffusers>=0.28.0"][1:],
                         ("fail", "not installed; needs >=0.28.0",
                          "pip install 'diffusers>=0.28.0'"))
        self.assertEqual(results["jax>=0.11.0"][2],
                         "0.10.0 installed; needs >=0.11.0")
        self.assertEqual(results["libtpu>=0.0.41"][1], "ok")
        # torch comes with torch_tpu, never on its own.
        self.assertIn("reinstall torch_tpu", results["torch>=2.13.0,<2.14"][3])

    def test_an_image_target_is_fixed_by_its_image_not_pip(self):
        """A package installed into a pod is gone with it."""
        with mock.patch.object(env_check, "version_of", return_value=None):
            checks = {
                name: fix
                for name, _, _, fix in env_check.run_checks(
                    ["jax>=0.11.0", "torch>=2.13.0,<2.14"], image=True)
            }
        for name in ("torch_tpu", "jax", "torch"):
            self.assertIn(env_check.IMAGE_DOC, checks[name])
            self.assertNotIn("pip install", checks[name])
        self.assertIn("'jax>=0.11.0'", checks["jax"])

    def test_lines_that_name_no_package_are_skipped(self):
        for line in ("-r other.txt", "-e .", "--index-url https://x",
                     "https://x/y.whl", "./local.whl", "/abs/y.whl"):
            self.assertIsNone(env_check.check_requirement(line), line)
        with mock.patch.object(env_check, "version_of", return_value="1.2"):
            for line in ("pkg[extra]>=1", "pkg>=1; python_version < '3.13'"):
                self.assertEqual(env_check.check_requirement(line),
                                 ("pkg", "ok", "1.2 >=1", ""), line)

    def test_host_mode_skips_the_tpu_software_and_devices(self):
        with mock.patch.object(env_check, "version_of", return_value=None):
            names = [c[0] for c in env_check.run_checks(["pyyaml>=6"], True)]
        self.assertEqual(names, ["python", "pyyaml"])

    def test_missing_host_packages_become_one_install_command(self):
        """Session start asks the user once, for everything missing, with
        the interpreter the runtime runs under."""
        versions = {"pyyaml": "6.0", "pydantic": "1.10"}
        with mock.patch.object(env_check,
                               "version_of",
                               side_effect=versions.get):
            missing = env_check.failing_requirements([
                "pyyaml>=6.0", "pydantic>=2.0.0", "tpu-info>=0.14.0",
                "--index-url https://x"
            ])
        self.assertEqual(missing, ["pydantic>=2.0.0", "tpu-info>=0.14.0"])
        self.assertEqual(
            env_check.ask_to_install(missing, python="/v/bin/python"),
            "Before anything else, ask the user to install these packages, "
            "which /v/bin/python is missing or has too old: pydantic>=2.0.0, "
            "tpu-info>=0.14.0. The command is `/v/bin/python -m pip install "
            "'pydantic>=2.0.0' 'tpu-info>=0.14.0'`. Don't start other work, or "
            "look up or measure what these packages would have told you, "
            "until they are installed.")
        with mock.patch.object(env_check,
                               "externally_managed",
                               return_value=False):
            one = env_check.ask_to_install(["x"])
        self.assertTrue(
            one.startswith("Before anything else, ask the user to install "
                           "this package"))
        self.assertIn(f"`{sys.executable} -m pip install 'x'`", one)
        self.assertTrue(one.endswith("until it is installed."))

    def test_a_system_python_is_told_to_use_a_venv_not_pip(self):
        """Review on #679: with the deps in a venv the user didn't activate,
        the hooks run on the system Python, which pip refuses (PEP 668)."""
        with tempfile.TemporaryDirectory() as stdlib:
            marker = Path(stdlib) / "EXTERNALLY-MANAGED"
            for in_venv, marked, managed in ((False, True, True), (True, True,
                                                                   False),
                                             (False, False, False)):
                with self.subTest(in_venv=in_venv, marked=marked), \
                        mock.patch.object(env_check.sysconfig, "get_path",
                                          return_value=stdlib), \
                        mock.patch.object(env_check.sys, "base_prefix",
                                          "/usr" if in_venv else sys.prefix):
                    if marked:
                        marker.touch()
                    else:
                        marker.unlink(missing_ok=True)
                    self.assertEqual(env_check.externally_managed(), managed)
                    note = env_check.ask_to_install(["x"])
                    if managed:
                        self.assertIn("activate it and restart this session",
                                      note)
                        self.assertIn("pip install 'x'`, then restart", note)
                        self.assertNotIn("-m pip install", note)
                    else:
                        self.assertIn(f"`{sys.executable} -m pip install 'x'`",
                                      note)
        # Debian's system Python has no ensurepip until python3-venv is
        # installed, and `venv` fails without it.
        for has_ensurepip, apt in ((True, False), (False, True)):
            with self.subTest(has_ensurepip=has_ensurepip), \
                    mock.patch.object(env_check, "externally_managed",
                                      return_value=True), \
                    mock.patch.object(env_check.importlib.util, "find_spec",
                                      return_value=object() if has_ensurepip
                                      else None):
                note = env_check.ask_to_install(["x"])
                self.assertEqual(
                    "`sudo apt install python3-venv && python3 -m venv"
                    in note, apt)
        # An interpreter named explicitly is never second-guessed.
        with mock.patch.object(env_check,
                               "externally_managed",
                               return_value=True):
            self.assertIn(
                "`/v/bin/python -m pip install 'x'`",
                env_check.ask_to_install(["x"], python="/v/bin/python"))

    def test_the_script_runs_standalone_and_exits_1_on_failure(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as handle:
            handle.write("# comment\nno-such-package-xyz>=1.0\n")
        self.addCleanup(Path(handle.name).unlink)
        proc = subprocess.run(
            [sys.executable,
             str(_SCRIPT), "--requirements", handle.name],
            capture_output=True,
            text=True,
            check=False)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("[FAIL] no-such-package-xyz", proc.stdout)
        self.assertIn("Status: fail", proc.stdout)

    def test_requirements_can_be_given_as_arguments(self):
        """A Job carries the script but, without a bucket, no files."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = env_check.main([
                "--require=no-such-package-xyz>=1.0",
                "--require=--index-url https://x"
            ])
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] no-such-package-xyz", out.getvalue())
        self.assertNotIn("index-url", out.getvalue())

    def test_versions_prints_each_tracked_release_and_checks_nothing(self):
        versions = {"jax": "0.11.2", "torch_tpu": "0.1.1"}
        out = io.StringIO()
        with mock.patch.object(env_check,
                               "version_of",
                               side_effect=versions.get), \
                contextlib.redirect_stdout(out):
            code = env_check.main(["--versions"])
        self.assertEqual(code, 0)
        self.assertEqual(
            {name: versions.get(name)
             for name in env_check.TRACKED}, json.loads(out.getvalue()))

    def test_a_target_check_ends_with_the_releases_it_found(self):
        """The installer records them from this line."""
        out = io.StringIO()
        with mock.patch.object(env_check, "version_of",
                               return_value=None), \
                contextlib.redirect_stdout(out):
            env_check.main(["--require=jax"])
        last = out.getvalue().splitlines()[-1]
        self.assertTrue(last.startswith(env_check.VERSIONS_PREFIX), last)
        self.assertIsNone(
            json.loads(last[len(env_check.VERSIONS_PREFIX):])["jax"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            env_check.main(["--host", "--require=jax"])
        self.assertNotIn(env_check.VERSIONS_PREFIX, out.getvalue())

    def test_some_requirements_must_be_given(self):
        """Else "ok" would say nothing about the package's requirements."""
        with self.assertRaises(SystemExit) as ctx, \
                contextlib.redirect_stderr(io.StringIO()):
            env_check.main([])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
