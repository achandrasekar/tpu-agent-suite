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
"""Tests for run_autotune.py."""

import importlib
import json
import os
import tempfile
import unittest

try:
    # Hyphenated package dir: not reachable via the `import` statement.
    run_autotune = importlib.import_module(
        "package.scripts.tpu-kernel-development.run_autotune")
except ImportError:
    import run_autotune


class TestRunAutotune(unittest.TestCase):

    def test_run_autotune_sweep(self):
        payload = {
            "code_template":
            ("import time\n"
             "block_m = {BLOCK_M}\n"
             "block_n = {BLOCK_N}\n"
             "print('compile time: 5000 ms')\n"
             "print(f'RESULT_TIME: {block_m + block_n} ms')\n"),
            "search_space": {
                "BLOCK_M": [16, 32],
                "BLOCK_N": [64, 128],
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as pf:
            json.dump(payload, pf)
            payload_path = pf.name

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as rf:
            results_path = rf.name

        try:
            summary = run_autotune.run_autotune(
                payload_path=payload_path,
                output_path=results_path,
                per_trial_timeout=10,
                total_timeout=60,
            )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["total_trials"], 4)
            self.assertIsNotNone(summary["best_config"])
            # Best is 16 + 64 = 80ms
            self.assertEqual(summary["best_config"]["BLOCK_M"], 16)
            self.assertEqual(summary["best_config"]["BLOCK_N"], 64)
            self.assertAlmostEqual(summary["best_time_ms"], 80.0)

            # Check that results_path was written properly
            with open(results_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["best_config"]["BLOCK_M"], 16)
            self.assertAlmostEqual(saved["all_results"][0]["time_ms"], 80.0)

        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)
            if os.path.exists(results_path):
                os.remove(results_path)

    def test_missing_result_time_marker_is_a_failed_trial(self):
        payload = {
            "code_template": "print('elapsed time: 1 ms')\n",
            "search_space": {
                "BLOCK_M": [16],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as pf:
            json.dump(payload, pf)
            payload_path = pf.name

        try:
            summary = run_autotune.run_autotune(
                payload_path=payload_path,
                per_trial_timeout=5,
                total_timeout=10,
            )
            self.assertEqual(summary["status"], "failed")
            # A failed sweep names no config: falling back to an arbitrary
            # combination would let apply_best_config.py write an unmeasured
            # configuration into the kernel.
            self.assertIsNone(summary["best_config"])
            self.assertIsNone(summary["best_time_ms"])
            self.assertIn("no trial succeeded", summary["failure_reason"])
            trial = summary["all_results"][0]
            self.assertEqual(trial["exit_code"], 1)
            self.assertIsNone(trial["time_ms"])
            self.assertIn("RESULT_TIME", trial["stderr"])
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)

    def test_identical_failures_are_reported_as_environment_not_config(self):
        # Every trial dies the same way regardless of its configuration,
        # which is what hardware contention on a shared target looks like.
        payload = {
            "code_template":
            ("import sys\n"
             "print('TPU already in use by another process', file=sys.stderr)\n"
             "sys.exit(3)\n"),
            "search_space": {
                "BLOCK_M": [16, 32, 64],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as pf:
            json.dump(payload, pf)
            payload_path = pf.name

        try:
            summary = run_autotune.run_autotune(
                payload_path=payload_path,
                per_trial_timeout=5,
                total_timeout=10,
            )
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["best_config"])
            self.assertEqual(summary["total_trials"], 3)
            self.assertIn("all 3 trials failed identically",
                          summary["failure_reason"])
            self.assertIn("exit code 3", summary["failure_reason"])
            self.assertIn("TPU already in use", summary["failure_reason"])
            self.assertIn("not a configuration problem",
                          summary["failure_reason"])
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)

    def test_spawn_failure_records_exception_as_stderr(self):
        payload = {
            "code_template": "print('RESULT_TIME: 1 ms')\n",
            "search_space": {
                "BLOCK_M": [16, 32],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as pf:
            json.dump(payload, pf)
            payload_path = pf.name

        real_run = run_autotune.subprocess.run
        calls = {"n": 0}

        def flaky_run(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("spawn failed")
            return real_run(*args, **kwargs)

        run_autotune.subprocess.run = flaky_run
        try:
            summary = run_autotune.run_autotune(
                payload_path=payload_path,
                per_trial_timeout=5,
                total_timeout=10,
            )
            first, second = summary["all_results"]
            self.assertEqual(first["exit_code"], 1)
            self.assertEqual(first["stderr"], "OSError: spawn failed")
            self.assertEqual(second["exit_code"], 0)
            self.assertEqual(summary["best_config"]["BLOCK_M"], 32)
        finally:
            run_autotune.subprocess.run = real_run
            if os.path.exists(payload_path):
                os.remove(payload_path)

    def test_cli_flags_and_defaults(self):
        payload = {
            "code_template": "print('RESULT_TIME: 10 ms')\n",
            "search_space": {
                "M": [1]
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as pf:
            json.dump(payload, pf)
            payload_path = pf.name

        try:
            # Test main with --trial-timeout and --total-timeout
            ret = run_autotune.main([
                payload_path, "--trial-timeout", "10", "--total-timeout", "20"
            ])
            self.assertEqual(ret, 0)
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)


if __name__ == "__main__":
    unittest.main()
