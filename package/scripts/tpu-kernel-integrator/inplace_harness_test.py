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
"""inplace_harness on CPU: torch.compile's eager backend, a fake op and case."""

import contextlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import build_plugin
import inplace_harness
import testing_fakes as fakes

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "needs torch")
class InplaceHarnessTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        fakes.write(self.tmp, "fake_target.py", fakes.TARGET_MODULE)
        self.case = fakes.write(self.tmp, "case.py", fakes.CASE_MODULE)
        sys.path.insert(0, self.tmp)
        self.addCleanup(sys.path.remove, self.tmp)
        self.addCleanup(fakes.forget_plugin_modules)
        self.addCleanup(os.environ.pop, "TPU_AGENT_SUITE_KERNELS", None)

    def harness(self, kernel, *extra):
        fakes.forget_plugin_modules()
        plugin = os.path.join(self.tmp, f"plugin_{kernel}")
        op = fakes.write(self.tmp, f"op_{kernel}.py",
                         fakes.OP_MODULE.format(kernel=kernel))
        k = fakes.write(self.tmp, "k.py", "\n")
        tgz = os.path.join(self.tmp, f"plugin_{kernel}.tgz")
        with contextlib.redirect_stdout(io.StringIO()):
            build_plugin.main([
                "--out-dir", plugin, "--op", "fake_op", "--op-module", op,
                "--kernel", k, "--tar", tgz
            ])
        # The harness unpacks the .tgz and imports the package from it; the
        # fake bridge replaces jax_op once the package is importable.
        root = inplace_harness.add_plugin(tgz)
        self.addCleanup(lambda: sys.path.remove(root)
                        if root in sys.path else None)
        ck = importlib.import_module("tpu_agent_suite_kernels")
        ck.bridge.make_op = fakes.python_make_op
        ck.bridge.MARKERS.clear()
        report_path = os.path.join(self.tmp, f"report_{kernel}.json")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()):
            code = inplace_harness.main([
                "--plugin", root, "--case", self.case, "--no-vllm", "--device",
                "cpu", "--backend", "eager", "--warmup", "2", "--iters", "10",
                "--repeats", "3", "--out", report_path, *extra
            ])
        with open(report_path) as f:
            return code, json.load(f)

    def test_an_op_written_for_the_unloaded_weight_layout_fails_op_parity(
            self):
        # The server re-lays out weights after loading; the harness must too,
        # or an op reading W the wrong way round would pass the op parity check here and
        # compute garbage in the server (silently, for a square W).
        code, report = self.harness("unloaded")
        self.assertEqual(code, inplace_harness.EXIT_GATE)
        self.assertEqual(report["status"], "mismatch")
        self.assertFalse(report["op_parity"]["buckets"][0]["match"])
        self.assertTrue(report["op_parity"]["buckets"][-1]["match"])

    def test_a_correct_kernel_that_is_not_faster_is_no_gain(self):
        code, report = self.harness("same")
        self.assertEqual(code, 1)
        # Not a win; CPU timing jitter may leave it undecided rather than lost.
        self.assertIn(report["status"], ("no_gain", "inconclusive"))
        self.assertTrue(report["op_parity"]["passed"])
        # JSON keys are strings.
        self.assertEqual(report["op_parity"]["paths"], {
            "2": "kernel",
            "4": "kernel",
            "8": "original"
        })
        self.assertEqual(
            [r["bucket"] for r in report["block_timing"]["buckets"]], [2, 4])

    def test_a_wrong_kernel_fails_op_parity_and_is_not_timed(self):
        code, report = self.harness("wrong")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "mismatch")
        rows = {
            r["bucket"]: r["match"]
            for r in report["op_parity"]["buckets"]
        }
        # Owned buckets ran the wrong kernel; the fallback bucket is fine.
        self.assertEqual(rows, {2: False, 4: False, 8: True})
        self.assertEqual(report["block_timing"], {})

    def test_a_faster_kernel_is_ready(self):
        code, report = self.harness("fast", "--control-buckets", "8")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "ready")
        owned = [r for r in report["block_timing"]["buckets"] if r["owned"]]
        self.assertTrue(all(r["delta_pct"] > 2.0 for r in owned))
        control = [
            r for r in report["block_timing"]["buckets"] if not r["owned"]
        ]
        # The control bucket is timed and reported, but never gates.
        self.assertEqual([r["bucket"] for r in control], [8])

    def test_an_op_that_breaks_the_graph_fails_block_timing(self):
        code, report = self.harness("breaks")
        self.assertEqual(code, 1)
        self.assertTrue(report["op_parity"]["passed"])
        self.assertEqual(report["status"], "graph_break")
        self.assertTrue(report["block_timing"]["graph_break"]["op_on"])

    def test_noise_decides_inconclusive_not_a_win(self):

        def row(delta, spread_off, spread_on=1.0):
            return {
                "delta_pct": delta,
                "off": {
                    "spread_pct": spread_off
                },
                "on": {
                    "spread_pct": spread_on
                }
            }

        verdict = inplace_harness.bucket_verdict
        self.assertEqual(verdict(row(10.0, 3.0), 2.0), "win")
        self.assertEqual(verdict(row(4.0, 3.0), 2.0), "inconclusive")
        self.assertEqual(verdict(row(-19.0, 20.0), 2.0), "inconclusive")
        self.assertEqual(verdict(row(0.5, 1.0), 2.0), "no_gain")
        self.assertEqual(verdict(row(-30.0, 5.0), 2.0), "no_gain")

    def test_a_case_missing_a_name_is_a_setup_error(self):
        bad = fakes.write(self.tmp, "bad_case.py", "OP = 'fake_op'\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = inplace_harness.main([
                "--plugin", self.tmp, "--case", bad, "--no-vllm", "--device",
                "cpu", "--out",
                os.path.join(self.tmp, "r.json")
            ])
        self.assertEqual(code, 2)
        self.assertIn("does not define", json.loads(out.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
