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

import contextlib
import importlib.metadata
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest

import build_plugin
import testing_fakes as fakes

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "examples", "gemma4_decode_mlp")


class BuildPluginTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = os.path.join(self.tmp, "plugin")
        self.kernel = fakes.write(self.tmp, "optimized.py",
                                  "def computation(x):\n    return x\n")

    def build(self, *extra, op="fake_op", op_module=None):
        op_module = op_module or fakes.write(
            self.tmp, "op.py", fakes.OP_MODULE.format(kernel="same"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = build_plugin.main([
                "--out-dir", self.out, "--op", op, "--op-module", op_module,
                "--kernel", self.kernel, *extra
            ])
        return code, json.loads(out.getvalue())

    def test_builds_the_package_entry_point_and_manifest(self):
        code, out = self.build("--tar", os.path.join(self.tmp, "p.tgz"))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["env"], {"TPU_AGENT_SUITE_KERNELS": "fake_op"})
        pkg = os.path.join(self.out, "tpu_agent_suite_kernels")
        for rel in ("__init__.py", "bridge.py", "op_fake_op.py",
                    "kernel_fake_op.py"):
            self.assertTrue(os.path.isfile(os.path.join(pkg, rel)), rel)
        self.assertFalse(any(n.endswith("_test.py") for n in os.listdir(pkg)))
        # vLLM finds the plugin the way it finds any general plugin.
        eps = [
            e for d in importlib.metadata.distributions(path=[self.out])
            for e in d.entry_points if e.group == "vllm.general_plugins"
        ]
        self.assertEqual(
            [(e.name, e.value) for e in eps],
            [("tpu_agent_suite_kernels", "tpu_agent_suite_kernels:register")])
        with open(os.path.join(self.out, "manifest.json")) as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["ops"]["fake_op"]["kernel"]["sha256"]),
                         64)
        with tarfile.open(os.path.join(self.tmp, "p.tgz")) as tar:
            names = tar.getnames()
        self.assertIn("tpu_agent_suite_kernels/op_fake_op.py", names)
        self.assertIn(
            "tpu_agent_suite_kernels-0.1.0.dist-info/entry_points.txt", names)

    def test_a_second_op_joins_the_same_plugin(self):
        self.build()
        code, out = self.build("--base", self.kernel, op="other_op")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["ops"], ["fake_op", "other_op"])
        self.assertEqual(out["env"]["TPU_AGENT_SUITE_KERNELS"],
                         "fake_op,other_op")
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.out, "tpu_agent_suite_kernels",
                             "kernel_other_op_base.py")))

    def test_an_op_module_missing_a_contract_name_is_refused(self):
        bad = fakes.write(self.tmp, "bad.py",
                          "TARGET = 'a:B'\nOWNED_BUCKETS = (1,)\n")
        code, out = self.build(op_module=bad)
        self.assertEqual(code, 2)
        self.assertIn("does not define", out["error"])

    def test_a_bad_slug_or_target_is_refused(self):
        code, out = self.build(op="Bad-Op")
        self.assertEqual(code, 2)
        bad = fakes.write(
            self.tmp, "bad.py",
            fakes.OP_MODULE.format(kernel="same").replace(
                'TARGET = "fake_target:Mlp"', 'TARGET = "fake_target"'))
        code, out = self.build(op_module=bad)
        self.assertEqual(code, 2)
        self.assertIn("TARGET", out["error"])

    def test_the_worked_example_satisfies_the_contract(self):
        build_plugin.check_op_module(
            build_plugin.Path(os.path.join(EXAMPLE, "op.py")))


if __name__ == "__main__":
    unittest.main()
