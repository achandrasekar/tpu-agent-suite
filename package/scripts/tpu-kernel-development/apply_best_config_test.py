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
"""Tests for apply_best_config.py."""

import importlib
import json
import os
import tempfile
import unittest

try:
    # Hyphenated package dir: not reachable via the `import` statement.
    apply_best_config = importlib.import_module(
        "package.scripts.tpu-kernel-development.apply_best_config")
except ImportError:
    import apply_best_config


class TestApplyBestConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec_path = os.path.join(self.tmp.name, "spec.json")
        self.results_path = os.path.join(self.tmp.name, "results.json")
        self.output_path = os.path.join(self.tmp.name, "optimized.py")
        with open(self.spec_path, "w") as f:
            json.dump({"code_template": "BLOCK_M = {BLOCK_M}\n"}, f)
        with open(self.output_path, "w") as f:
            f.write("BLOCK_M = 8  # phase 3 kernel\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_results(self, results):
        with open(self.results_path, "w") as f:
            json.dump(results, f)

    def test_success_applies_best_config(self):
        self._write_results({
            "status": "success",
            "best_config": {
                "BLOCK_M": 32
            },
            "best_time_ms": 1.0,
        })
        apply_best_config.apply_best_config(self.spec_path, self.results_path,
                                            self.output_path)
        with open(self.output_path) as f:
            self.assertEqual(f.read(), "BLOCK_M = 32\n")

    def test_failed_sweep_is_refused_and_kernel_untouched(self):
        # A sweep with no successful trial has no best_config; the kernel
        # Phase 3 tested must survive intact rather than be overwritten with
        # a configuration that never ran.
        self._write_results({
            "status": "failed",
            "best_config": None,
            "best_time_ms": None,
            "failure_reason": "no trial succeeded (2 tried)",
        })
        with self.assertRaises(ValueError) as ctx:
            apply_best_config.apply_best_config(self.spec_path,
                                                self.results_path,
                                                self.output_path)
        self.assertIn("'failed'", str(ctx.exception))
        self.assertIn("no trial succeeded", str(ctx.exception))
        with open(self.output_path) as f:
            self.assertEqual(f.read(), "BLOCK_M = 8  # phase 3 kernel\n")

    def test_results_without_status_are_refused(self):
        # A results file that predates the status field, or was produced by
        # something other than run_autotune.py, is not trusted.
        self._write_results({"best_config": {"BLOCK_M": 32}})
        with self.assertRaises(ValueError):
            apply_best_config.apply_best_config(self.spec_path,
                                                self.results_path,
                                                self.output_path)


if __name__ == "__main__":
    unittest.main()
