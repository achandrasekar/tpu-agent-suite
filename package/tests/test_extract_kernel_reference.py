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
"""Tests for extract_kernel_reference.py."""

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = (_PACKAGE_ROOT / "scripts" / "tpu-model-onboarding" /
           "extract_kernel_reference.py")

_spec = importlib.util.spec_from_file_location("extract_kernel_reference",
                                               _SCRIPT)
extractor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extractor)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestExtractKernelReference(unittest.TestCase):

    def test_parse_shape_spec(self):
        dtype, dims = extractor.parse_shape_spec("bf16[8,16,128,64]{3,2,1,0}")
        self.assertEqual(dtype, "bf16")
        self.assertEqual(dims, [8, 16, 128, 64])

        dtype2, dims2 = extractor.parse_shape_spec("f32[2, 4]")
        self.assertEqual(dtype2, "f32")
        self.assertEqual(dims2, [2, 4])

    def test_map_dtype_to_torch(self):
        self.assertEqual(extractor.map_dtype_to_torch("bf16"),
                         "torch.bfloat16")
        self.assertEqual(extractor.map_dtype_to_torch("f32"), "torch.float32")
        self.assertEqual(extractor.map_dtype_to_torch("s32"), "torch.int32")

    def test_extraction_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            source_file = tmppath / "sample_layer.py"
            source_file.write_text(
                "import torch\n\n"
                "def custom_add_norm(x, residual):\n"
                "    return (x + residual).to(torch.float32)\n",
                encoding="utf-8",
            )

            shapes_data = {
                "operands": ["bf16[4, 8]", "bf16[4, 8]"],
                "outputs": ["f32[4, 8]"],
            }
            shapes_file = tmppath / "call_shapes.json"
            shapes_file.write_text(json.dumps(shapes_data), encoding="utf-8")

            output_file = tmppath / "base_kernel.py"

            cmd = [
                sys.executable,
                str(_SCRIPT),
                "--source",
                f"{source_file}:3",
                "--call-shapes-json",
                str(shapes_file),
                "--output-path",
                str(output_file),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertTrue(output_file.exists())

            content = output_file.read_text(encoding="utf-8")
            self.assertIn("def computation(*args, **kwargs):", content)
            self.assertIn("custom_add_norm", content)
            self.assertIn("get_dummy_inputs", content)

            # Verify syntax parses cleanly
            ast.parse(content)

            # If torch is available in this environment, execute it
            if _HAS_TORCH:
                exec_res = subprocess.run(
                    [sys.executable, str(output_file)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("Execution verified", exec_res.stdout)


if __name__ == "__main__":
    unittest.main()
