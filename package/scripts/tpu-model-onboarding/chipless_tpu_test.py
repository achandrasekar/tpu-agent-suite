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
"""Tests for chipless_tpu, against stand-ins for torch and torch_tpu.

Each case runs a fresh interpreter, as the shim loads at startup from a
sitecustomize.py. The stand-ins keep the shape the shim relies on: torch
autoloads torch_tpu, which registers the `tpu` device only when its PCI scan
counts a chip.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

HARDWARE = '''
_PCI_DEVICE_ID_TO_NAME = {"0x006f": "TPU v6e", "0x0076": "TPU v7"}
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x006f": {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"},
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},
}

def _scan_pci_tpus():
    return 0, None

def get_tpu_device_count():
    return _scan_pci_tpus()[0]

def get_tpu_device_name():
    return "TPU"
'''

TORCH = '''
_backends = set()

def device(name):
    if name not in _backends and name != "cpu":
        raise RuntimeError(f"Expected one of cpu ... device type: {name}")
    return name

import torch_tpu  # noqa: E402,F401 - torch autoloads it
'''

TORCH_TPU = '''
import torch
from torch_tpu._internal.utils import hardware

if hardware.get_tpu_device_count() > 0:
    torch._backends.add("tpu")
'''


class ChiplessTpuTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        fakes = os.path.join(self.root, "fakes")
        self._write(fakes, "torch/__init__.py", TORCH)
        self._write(fakes, "torch_tpu/__init__.py", TORCH_TPU)
        self._write(fakes, "torch_tpu/_internal/__init__.py", "")
        self._write(fakes, "torch_tpu/_internal/utils/__init__.py", "")
        self._write(fakes, "torch_tpu/_internal/utils/hardware.py", HARDWARE)
        # The image's own sitecustomize, which the shim's must not hide.
        hidden = os.path.join(self.root, "image")
        self._write(hidden, "sitecustomize.py",
                    "import os\nos.environ['IMAGE_SITE_RAN'] = '1'\n")
        # As onboarding_preflight.py lays it out.
        self.shim = os.path.join(self.root, ".chipless")
        os.makedirs(self.shim)
        shutil.copy(os.path.join(HERE, "chipless_tpu.py"), self.shim)
        self._write(self.shim, "sitecustomize.py", "import chipless_tpu\n")
        self.fakes, self.hidden = fakes, hidden

    @staticmethod
    def _write(root, relative, text):
        path = os.path.join(root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))

    def _run(self, argv, generation="v6e", devices="8", shim=True):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        path = [self.hidden, self.fakes]
        if shim:
            path.insert(0, self.shim)
        env["PYTHONPATH"] = os.pathsep.join(path)
        for key, value in (("TPU_AGENT_SUITE_CHIPLESS_TPU", generation),
                           ("CHIPLESS_TPU_DEVICES", devices)):
            env.pop(key, None)
            if value is not None:
                env[key] = value
        return subprocess.run([sys.executable] + argv,
                              env=env,
                              cwd=self.root,
                              capture_output=True,
                              text=True,
                              check=False)

    def _check(self, **kwargs):
        proc = self._run(
            [os.path.join(self.shim, "chipless_tpu.py"), "--check"], **kwargs)
        return proc.returncode, json.loads(proc.stdout), proc.stderr

    def test_torch_tpu_answers_as_the_profiles_tpu(self):
        """Count, generation and layout table as asked, by the name compute
        gives the generation; the image's own sitecustomize still runs, and
        stays the module `sitecustomize`; and with no generation set, nothing
        changes."""
        code = ("import json, os, sys; "
                "from torch_tpu._internal.utils import hardware as h; "
                "n, table = h._scan_pci_tpus(); "
                "print(json.dumps([h.get_tpu_device_name(), n, "
                "(table or {}).get(8), os.environ.get('IMAGE_SITE_RAN'), "
                "sys.modules['sitecustomize'].__file__.split(os.sep)[-2]]))")
        cases = [
            ("v6e", ["TPU v6e", 8, "2,4,1", "1", "image"]),
            ("tpu7x", ["TPU v7", 8, "2,2,1,2", "1", "image"]),
            (None, ["TPU", 0, None, "1", "image"]),
        ]
        for generation, expected in cases:
            with self.subTest(generation=generation):
                proc = self._run(["-c", code], generation=generation)
                self.assertEqual(json.loads(proc.stdout), expected,
                                 proc.stderr)
        code, report, _ = self._check()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["device"], "tpu")

    def test_fails_loudly_rather_than_misreporting(self):
        """Each way the shim can miss exits 3, says CHIPLESS_TPU_UNSUPPORTED
        and names the cause, so the caller runs on a TPU node."""
        cases = [
            ({
                "generation": "v9"
            }, "is not one of"),
            ({
                "devices": "eight"
            }, "invalid literal"),
            ({
                "shim": False
            }, "was not loaded at startup"),
            # torch_tpu without the private function the shim replaces.
            ({
                "changed": True
            }, "no _scan_pci_tpus"),
        ]
        for kwargs, cause in cases:
            with self.subTest(**kwargs):
                if kwargs.pop("changed", False):
                    self._write(
                        self.fakes, "torch_tpu/_internal/utils/hardware.py",
                        HARDWARE.replace("_scan_pci_tpus",
                                         "_scan_pci_devices"))
                code, report, stderr = self._check(**kwargs)
                self.assertEqual(code, 3)
                self.assertEqual(report["status"], "fail")
                self.assertIn(cause, report["error"])
                self.assertIn("CHIPLESS_TPU_UNSUPPORTED", stderr)


if __name__ == "__main__":
    unittest.main()
