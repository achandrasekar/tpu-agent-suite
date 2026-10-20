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
"""The plugin runtime, with a plain torch module standing in for vLLM's."""

import contextlib
import importlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest

import build_plugin
import testing_fakes as fakes

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "needs torch")
class TpuAgentSuiteKernelsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        plugin = os.path.join(self.tmp, "plugin")
        op = fakes.write(self.tmp, "op.py",
                         fakes.OP_MODULE.format(kernel="same"))
        kernel = fakes.write(self.tmp, "k.py", "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            build_plugin.main([
                "--out-dir", plugin, "--op", "fake_op", "--op-module", op,
                "--kernel", kernel
            ])
        fakes.write(self.tmp, "fake_target.py", fakes.TARGET_MODULE)
        fakes.forget_plugin_modules()
        sys.path[:0] = [plugin, self.tmp]
        self.addCleanup(
            lambda: [sys.path.remove(p) for p in (plugin, self.tmp)])
        self.addCleanup(fakes.forget_plugin_modules)
        self.addCleanup(os.environ.pop, "TPU_AGENT_SUITE_KERNELS", None)
        self.ck = importlib.import_module("tpu_agent_suite_kernels")
        self.ck.bridge.make_op = fakes.python_make_op
        self.ck.bridge.MARKERS.clear()

    def register(self, value):
        os.environ["TPU_AGENT_SUITE_KERNELS"] = value
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.ck.register()
        return err.getvalue()

    def test_nothing_is_patched_without_the_variable(self):
        import fake_target  # pylint: disable=import-outside-toplevel
        original = fake_target.Mlp.forward
        self.register("")
        self.assertIs(fake_target.Mlp.forward, original)

    def test_eligible_instances_take_the_op_and_others_do_not(self):
        import torch  # pylint: disable=import-outside-toplevel
        logs = self.register("fake_op")
        self.assertIn("fake_op patched fake_target:Mlp", logs)
        import fake_target  # pylint: disable=import-outside-toplevel
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            yes = fake_target.Mlp(prefix="layers.0.mlp")
            no = fake_target.Mlp(eligible=False, prefix="layers.1.mlp")
        self.assertIn("fake_op layers.0.mlp KERNEL", err.getvalue())
        self.assertIn("fake_op layers.1.mlp ORIGINAL (not eligible)",
                      err.getvalue())
        self.assertTrue(yes._tpu_agent_suite_fake_op)
        self.assertFalse(no._tpu_agent_suite_fake_op)
        with torch.no_grad():
            yes.weight.copy_(torch.eye(8))
        with contextlib.redirect_stderr(io.StringIO()):
            out = yes(torch.ones(2, 8))
            no(torch.ones(4, 8))
        self.assertTrue(torch.equal(out, torch.ones(2, 8)))
        # Only the eligible instance reached the bridge.
        self.assertEqual(self.ck.bridge.MARKERS, [{
            "op": "fake_op",
            "bucket": 2,
            "path": "kernel"
        }])

    def test_the_bucket_picks_the_path(self):
        import torch  # pylint: disable=import-outside-toplevel
        self.register("fake_op")
        import fake_target  # pylint: disable=import-outside-toplevel
        with contextlib.redirect_stderr(io.StringIO()) as err:
            m = fake_target.Mlp()
            for bucket in (2, 4, 8, 2):
                m(torch.ones(bucket, 8))
        self.assertEqual([(x["bucket"], x["path"])
                          for x in self.ck.bridge.MARKERS], [(2, "kernel"),
                                                             (4, "kernel"),
                                                             (8, "original")])
        # One line per bucket, however often it runs.
        self.assertEqual(err.getvalue().count("bucket=2 path=kernel"), 1)

    def test_an_op_that_cannot_patch_is_logged_not_raised(self):
        logs = self.register("missing_op")
        self.assertIn("missing_op NOT PATCHED", logs)

    def test_the_dispatcher_keeps_the_fallback_signature(self):
        import inspect  # pylint: disable=import-outside-toplevel
        op = self.ck.load_op("fake_op")
        fn = self.ck.bridge.dispatch("fake_op", op.OWNED_BUCKETS, op.kernel_fn,
                                     op.fallback_fn)
        # jax_op derives the torch schema from these.
        self.assertEqual(list(inspect.signature(fn).parameters), ["x", "w"])
        self.assertEqual(fn.__name__, "tpu_agent_suite_fake_op")


if __name__ == "__main__":
    unittest.main()
