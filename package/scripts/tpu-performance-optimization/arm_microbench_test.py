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
"""Unit tests for arm_microbench.py fast-fail checks."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import arm_microbench  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "arm_microbench", _SCRIPTS_DIR / "arm_microbench.py")
    arm_microbench = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(arm_microbench)


class TestArmMicrobench(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_syntax_valid_file(self):
        f = self.test_dir / "valid.py"
        f.write_text("import torch\nimport torch_tpu\nx = 1 + 2\n",
                     encoding="utf-8")
        errors = arm_microbench.check_syntax_and_imports(f)
        self.assertEqual(errors, [])

    def test_syntax_error_detected(self):
        f = self.test_dir / "bad_syntax.py"
        f.write_text("def broken(:\n  pass\n", encoding="utf-8")
        errors = arm_microbench.check_syntax_and_imports(f)
        self.assertTrue(any("syntax error" in e.lower() for e in errors))

    def test_forbidden_torch_xla_import(self):
        f = self.test_dir / "deprecated.py"
        f.write_text(
            "import torch_xla\nimport torch_xla.core.xla_model as xm\n",
            encoding="utf-8")
        errors = arm_microbench.check_syntax_and_imports(f)
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("torch_xla" in e for e in errors))

    def test_forbidden_pytorch_xla_from_import(self):
        f = self.test_dir / "deprecated_from.py"
        f.write_text("from pytorch_xla.core import xla_model\n",
                     encoding="utf-8")
        errors = arm_microbench.check_syntax_and_imports(f)
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("pytorch_xla" in e for e in errors))

    def test_microbench_command_execution(self):
        res = arm_microbench.run_microbench(
            ["python3", "-c", "print('bench_ok')"], timeout_s=5.0)
        self.assertTrue(res["success"])
        self.assertIn("bench_ok", res["stdout"])

    def test_microbench_timeout(self):
        res = arm_microbench.run_microbench(
            ["python3", "-c", "import time; time.sleep(2)"],
            timeout_s=0.2,
        )
        self.assertFalse(res["success"])
        self.assertIn("timed out", res["stderr"])

    def test_gotcha_detects_blocking_item_and_cpu_in_loop(self):
        f = self.test_dir / "blocking_sync.py"
        f.write_text(
            "import torch\n"
            "for i in range(10):\n"
            "    val = x.item()\n",
            encoding="utf-8",
        )
        errors = arm_microbench.check_hardware_gotchas(f)
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any(".item()" in e for e in errors))

    def test_gotcha_detects_non_systolic_chunk_size_and_allows_sublane_batch(
            self):
        f = self.test_dir / "dummy.py"
        f.write_text("x = 1\n", encoding="utf-8")
        errors = arm_microbench.check_hardware_gotchas(
            f, config={"chunk_size": 250})
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("MXU lane dimension (128)" in e for e in errors))

        # Batch sizes of 8 and 32 map to the MXU sublane dimension (8) and must pass cleanly
        for valid_bs in (8, 32):
            self.assertEqual(
                arm_microbench.check_hardware_gotchas(f,
                                                      config={
                                                          "batch_size":
                                                          valid_bs,
                                                          "chunk_size": 2048
                                                      }),
                [],
            )

        # Unaligned batch size (e.g. 7) is flagged against sublane dimension (8)
        sublane_errs = arm_microbench.check_hardware_gotchas(
            f, config={"batch_size": 7})
        self.assertTrue(
            any("MXU sublane dimension (8)" in e for e in sublane_errs))

    def test_gotcha_detects_sc_sram_exceeding_128kib(self):
        f = self.test_dir / "dummy.py"
        f.write_text("x = 1\n", encoding="utf-8")
        errors = arm_microbench.check_hardware_gotchas(
            f, config={"sc_sram_bytes": 200 * 1024})
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(
            any("SparseCore SRAM budget exceeded" in e for e in errors))

    def test_gotcha_detects_core_vmem_exceeding_6mb(self):
        f = self.test_dir / "dummy.py"
        f.write_text("x = 1\n", encoding="utf-8")
        errors = arm_microbench.check_hardware_gotchas(
            f, config={"vmem_bytes_per_core": 8 * 1024 * 1024})
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("Core VMEM budget exceeded" in e for e in errors))

    def test_gotcha_passes_valid_code_and_config(self):
        f = self.test_dir / "valid_gotchas.py"
        f.write_text("for i in range(10):\n    x = y + 1\n", encoding="utf-8")
        errors = arm_microbench.check_hardware_gotchas(
            f,
            config={
                "chunk_size": 512,
                "sc_sram_bytes": 64 * 1024,
                "vmem_bytes_per_core": 4 * 1024 * 1024,
            })
        self.assertEqual(errors, [])

    def test_gotcha_ignores_print_item_and_detects_one_hop_helper_sync(self):
        f_print = self.test_dir / "print_item.py"
        f_print.write_text(
            "for step in range(10):\n"
            "    print(f'loss={loss.item()}')\n",
            encoding="utf-8",
        )
        self.assertEqual(arm_microbench.check_hardware_gotchas(f_print), [])

        f_hop = self.test_dir / "helper_hop.py"
        f_hop.write_text(
            "def extract_scalar(t):\n"
            "    return t.item()\n\n"
            "while running:\n"
            "    val = extract_scalar(logits)\n",
            encoding="utf-8",
        )
        hop_errors = arm_microbench.check_hardware_gotchas(f_hop)
        self.assertTrue(len(hop_errors) >= 1)
        self.assertTrue(
            any("calls helper 'extract_scalar()'" in e for e in hop_errors))

    def test_cli_run_with_trailing_gotchas_flag_and_tpu_version_specs(self):
        f = self.test_dir / "bench_target.py"
        f.write_text("x = 1 + 1\n", encoding="utf-8")

        # --run "<cmd>" --gotchas must not swallow --gotchas into the command
        rc = arm_microbench.main([
            "--file",
            str(f),
            "--run",
            "python3 -c \"print('cli_run_ok')\"",
            "--gotchas",
        ])
        self.assertEqual(rc, 0)

        # v5e has 16 MiB core VMEM (so 8 MiB passes on v5e, whereas it fails on v7x's 6.06 MiB)
        self.assertEqual(
            arm_microbench.check_hardware_gotchas(
                f,
                config={"vmem_bytes_per_core": 8 * 1024 * 1024},
                tpu_version="v5e",
            ),
            [],
        )
        v7x_errs = arm_microbench.check_hardware_gotchas(
            f,
            config={"vmem_bytes_per_core": 8 * 1024 * 1024},
            tpu_version="v7x",
        )
        self.assertTrue(any("Core VMEM budget exceeded" in e
                            for e in v7x_errs))


if __name__ == "__main__":
    unittest.main()
