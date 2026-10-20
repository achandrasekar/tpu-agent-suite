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
"""Tests for the served-kernel introspection script."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import introspect_kernel

# Executing any of these modules raises: introspection must only read them.
_RAISES = "raise RuntimeError('module code ran')\n"

_DISPATCHER = """\
import json
from fakepkg.kernels.fused import kernel as fused_kernel
from fakepkg.kernels.other.kernel import build


def dispatch(name):
    from ..kernels.lazy import kernel as lazy_kernel
    from .missing import helper
    return fused_kernel, build, lazy_kernel, helper
"""

# A kernel importing a neighbour (copied) and a package helper (not copied).
_FUSED_KERNEL = ("from fakepkg.kernels.fused import configs\n"
                 "from fakepkg.utils import helpers\n" + _RAISES)


class IntrospectKernelTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.site = self.root / "site"
        pkg = self.site / "fakepkg"
        files = {
            "__init__.py": "",
            "layers/__init__.py": "",
            "layers/attention.py": _DISPATCHER,
            "kernels/__init__.py": "",
            "kernels/fused/__init__.py": "",
            "kernels/fused/kernel.py": _FUSED_KERNEL,
            "kernels/fused/configs.py": "BLOCK = 128\n",
            "kernels/fused/tables/v7x.json": "{}\n",
            "kernels/other/__init__.py": "",
            "kernels/other/kernel.py": _RAISES,
            "kernels/lazy/__init__.py": "",
            "kernels/lazy/kernel.py": _RAISES,
            "utils/__init__.py": "",
            "utils/helpers.py": "",
        }
        for relative, text in files.items():
            path = pkg / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cache = pkg / "kernels/fused/__pycache__"
        cache.mkdir()
        (cache / "kernel.cpython-312.pyc").write_bytes(b"\0")
        (pkg / "kernels/fused/ext.so").write_bytes(b"\0")
        self.pkg = pkg
        self.out = self.root / "out"
        sys.path.insert(0, str(self.site))

    def tearDown(self):
        sys.path.remove(str(self.site))
        for name in list(sys.modules):
            if name == "fakepkg" or name.startswith("fakepkg."):
                del sys.modules[name]
        self._tmp.cleanup()

    def _run(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = introspect_kernel.main([*argv, "--out-dir", str(self.out)])
        report = json.loads(
            (self.out / introspect_kernel.JSON_NAME).read_text("utf-8"))
        return code, report

    def _tar_names(self):
        with tarfile.open(self.out / introspect_kernel.TAR_NAME) as tar:
            return sorted(tar.getnames())

    def _assert_copies_match(self, report):
        with tarfile.open(self.out / introspect_kernel.TAR_NAME) as tar:
            for entry in report["files"]:
                data = tar.extractfile(entry["path"]).read()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(data,
                                 (self.site / entry["path"]).read_bytes())

    def test_dispatcher_copies_its_candidates_without_running_them(self):
        dispatcher = self.pkg / "layers/attention.py"
        code, report = self._run("--dispatcher", str(dispatcher), "--image",
                                 "example/image:tag")
        self.assertEqual(code, 0, report.get("error"))
        self.assertEqual(report["image"], "example/image:tag")
        self.assertEqual(report["dispatcher"]["module"],
                         "fakepkg.layers.attention")
        self.assertEqual(report["dispatcher"]["copy"],
                         "fakepkg/layers/attention.py")
        modules = {m["module"]: m for m in report["modules"]}
        self.assertEqual(
            set(modules), {
                "fakepkg.kernels.fused.kernel",
                "fakepkg.kernels.other.kernel",
                "fakepkg.kernels.lazy.kernel",
            })
        self.assertEqual(modules["fakepkg.kernels.fused.kernel"]["copy"],
                         "fakepkg/kernels/fused/kernel.py")
        self.assertEqual(
            modules["fakepkg.kernels.other.kernel"]["imported_by"],
            "fakepkg.layers.attention")
        self.assertEqual(report["unresolved"], ["fakepkg.layers.missing"])
        # Each module's own-package imports, marked present or absent in the
        # copy; an absent one is reported, never supplied.
        self.assertEqual(modules["fakepkg.kernels.fused.kernel"]["imports"], [
            {
                "module": "fakepkg.kernels.fused.configs",
                "present": True
            },
            {
                "module": "fakepkg.utils.helpers",
                "present": False
            },
        ])
        names = self._tar_names()
        self.assertNotIn("fakepkg/utils/helpers.py", names)
        self.assertIn("fakepkg/layers/attention.py", names)
        self.assertIn("fakepkg/kernels/fused/configs.py", names)
        self.assertIn("fakepkg/kernels/lazy/kernel.py", names)
        # Imported directories are copied flat; bytecode is never copied.
        self.assertNotIn("fakepkg/kernels/fused/tables/v7x.json", names)
        self.assertFalse(any(n.endswith((".pyc", ".so")) for n in names))
        self._assert_copies_match(report)

    def test_named_module_copies_its_whole_directory(self):
        code, report = self._run("fakepkg.kernels.fused.kernel")
        self.assertEqual(code, 0, report.get("error"))
        self.assertIsNone(report["dispatcher"])
        self.assertEqual(self._tar_names(), [
            "fakepkg/kernels/fused/__init__.py",
            "fakepkg/kernels/fused/configs.py",
            "fakepkg/kernels/fused/kernel.py",
            "fakepkg/kernels/fused/tables/v7x.json",
        ])
        self._assert_copies_match(report)

    def test_unknown_module_reports_error_and_writes_no_tarball(self):
        code, report = self._run("fakepkg.kernels.absent.kernel")
        self.assertEqual(code, 1)
        self.assertIn("error", report)
        self.assertFalse((self.out / introspect_kernel.TAR_NAME).exists())

    def test_missing_dispatcher_reports_error(self):
        code, report = self._run("--dispatcher",
                                 str(self.pkg / "layers/absent.py"))
        self.assertEqual(code, 1)
        self.assertIn("not a file", report["error"])
        self.assertFalse((self.out / introspect_kernel.TAR_NAME).exists())

    def test_oversized_copy_is_refused(self):
        code, report = self._run("fakepkg.kernels.fused.kernel", "--max-bytes",
                                 "4")
        self.assertEqual(code, 1)
        self.assertIn("--max-bytes", report["error"])
        self.assertFalse((self.out / introspect_kernel.TAR_NAME).exists())


if __name__ == "__main__":
    unittest.main()
