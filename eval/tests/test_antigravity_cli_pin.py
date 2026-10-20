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
"""Tests for the Antigravity CLI version pin in eval/adapter/sitecustomize.py."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.adapter import sitecustomize as _ADAPTER

try:
    from harbor.agents.installed.antigravity_cli import AntigravityCli
except ImportError:
    AntigravityCli = None


def _fake_release(version: str) -> bytes:
    """A release tarball whose `antigravity` reports `version`."""
    script = (
        f"#!/bin/sh\n"
        f"if [ \"$1\" = --version ]; then echo {version}; fi\n").encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("antigravity")
        info.size = len(script)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(script))
    return buf.getvalue()


class TestPinRequested(unittest.TestCase):

    def test_pinned_unless_latest_is_asked_for(self) -> None:
        env = _ADAPTER.ANTIGRAVITY_CLI_VERSION_ENV
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env, None)
            self.assertTrue(_ADAPTER.antigravity_cli_pin_requested())
        for value in ("latest", " Latest "):
            with mock.patch.dict(os.environ, {env: value}):
                self.assertFalse(_ADAPTER.antigravity_cli_pin_requested())

    def test_pin_is_a_full_release(self) -> None:
        pin = _ADAPTER.ANTIGRAVITY_CLI_PIN
        self.assertRegex(pin["version"], r"^\d+\.\d+\.\d+$")
        self.assertIn(f"/{pin['version']}-", pin["url"])
        self.assertTrue(pin["url"].startswith("https://"))
        self.assertRegex(pin["sha512"], r"^[0-9a-f]{128}$")
        self.assertRegex(pin["patch_before"], r"^[0-9a-f]+$")
        self.assertRegex(pin["patch_after"], r"^[0-9a-f]+$")
        self.assertEqual(
            len(bytes.fromhex(pin["patch_before"])),
            len(bytes.fromhex(pin["patch_after"])),
        )
        self.assertNotEqual(pin["patch_before"], pin["patch_after"])


class TestInstallCommand(unittest.TestCase):
    """Runs the generated script against a local fake release."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.home = root / "home"
        self.home.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        self.release = root / "release.tar.gz"
        # `curl -fsSL -o <dest> <url>` copies the local release instead.
        curl = self.bin / "curl"
        curl.write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\n'
                        f'cp "{self.release}" "$2"\n')
        curl.chmod(0o755)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, pin: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c",
             _ADAPTER.antigravity_cli_install_command(pin)],
            env={
                "HOME": str(self.home),
                "PATH": f"{self.bin}:{os.environ['PATH']}"
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def _pin(self, served: str, expected: str) -> dict[str, str]:
        data = _fake_release(served)
        self.release.write_bytes(data)
        return {
            "version": expected,
            "url": f"https://example.invalid/{expected}-1/cli.tar.gz",
            "sha512": hashlib.sha512(data).hexdigest(),
        }

    def test_installs_the_pinned_release(self) -> None:
        result = self._run(self._pin("9.9.9", "9.9.9"))
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.home / ".local" / "bin" / "agy"
        self.assertTrue(os.access(installed, os.X_OK))
        self.assertIn("9.9.9", result.stdout)

    def test_patches_binary_in_place(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["patch_before"] = b"9.9.9".hex()
        pin["patch_after"] = b"9.9.0".hex()
        pin["version"] = "9.9.0"
        result = self._run(pin)
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.home / ".local" / "bin" / "agy"
        self.assertIn(b"9.9.0", installed.read_bytes())
        self.assertNotIn(b"9.9.9", installed.read_bytes())

    def test_missing_patch_target_installs_nothing(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["patch_before"] = "deadbeef"
        pin["patch_after"] = "cafebabe"
        result = self._run(pin)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / ".local" / "bin" / "agy").exists())

    def test_a_checksum_mismatch_installs_nothing(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["sha512"] = "0" * 128
        result = self._run(pin)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / ".local" / "bin" / "agy").exists())

    def test_a_binary_reporting_another_version_fails(self) -> None:
        result = self._run(self._pin("9.9.8", "9.9.9"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 9.9.9", result.stderr)


@unittest.skipIf(AntigravityCli is None, "harbor is not installed")
class TestPatchedInstall(unittest.TestCase):

    def _install(self, env: dict[str, str]) -> mock.MagicMock:
        agent = mock.MagicMock()
        agent.ensure_system_dependencies = mock.AsyncMock()
        agent.exec_as_agent = mock.AsyncMock()
        with mock.patch.dict(os.environ, env):
            asyncio.run(AntigravityCli.install(agent, mock.MagicMock()))
        return agent

    def test_install_is_patched(self) -> None:
        self.assertTrue(getattr(AntigravityCli, "_eval_cli_pin_patched",
                                False))

    def test_installs_the_pin_by_default(self) -> None:
        env = {_ADAPTER.ANTIGRAVITY_CLI_VERSION_ENV: ""}
        agent = self._install(env)
        command = agent.exec_as_agent.call_args.kwargs["command"]
        self.assertTrue(command.startswith("bash -c "))
        self.assertIn(_ADAPTER.ANTIGRAVITY_CLI_PIN["sha512"], command)
        self.assertIn(_ADAPTER.ANTIGRAVITY_CLI_PIN["patch_before"], command)
        self.assertIn(_ADAPTER.ANTIGRAVITY_CLI_PIN["patch_after"], command)
        self.assertNotIn("install.sh", command)

    def test_validate_reasoning_effort_allows_medium_for_gemini_4(
            self) -> None:
        AntigravityCli._validate_reasoning_effort("medium",
                                                  "gemini/gemini-4-argon")
        with self.assertRaises(ValueError):
            AntigravityCli._validate_reasoning_effort(
                "medium", "gemini/gemini-3.1-pro-preview")


if __name__ == "__main__":
    unittest.main()
