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
"""Tests for Hugging Face models compatibility checklist preflight validation."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

try:
    import httpx
except ImportError:  # huggingface_hub >= 2.0 ships httpx2 instead.
    import httpx2 as httpx
import huggingface_hub
# Import the target library
import preflight_check
import transformers
from absl import flags
from absl.testing import absltest, parameterized

FLAGS = flags.FLAGS


class MockSibling:

    def __init__(self, rfilename: str):
        self.rfilename = rfilename


class MockModelInfo:

    def __init__(
        self,
        gated: bool | str = False,
        siblings: list[MockSibling] | None = None,
        library_name: str | None = None,
    ):
        self.gated = gated
        self.siblings = siblings or []
        self.library_name = library_name


class PreflightCheckTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        # Keep a reference to restore force flag
        self._orig_force = FLAGS.force
        FLAGS.force = False

    def tearDown(self):
        super().tearDown()
        FLAGS.force = self._orig_force

    @mock.patch.object(huggingface_hub, "HfApi")
    def test_check_access_public(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_info = MockModelInfo(gated=False)
        mock_api.model_info.return_value = mock_info

        ok, msg, info = preflight_check.check_access("public-model")
        self.assertTrue(ok)
        self.assertEqual(msg, "Public model.")
        self.assertEqual(info, mock_info)

    @mock.patch.object(huggingface_hub, "HfApi")
    def test_check_access_gated_authorized(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_info = MockModelInfo(gated="manual")
        mock_api.model_info.return_value = mock_info

        ok, msg, info = preflight_check.check_access("gated-model")
        self.assertTrue(ok)
        self.assertIn("Gated model", msg)
        self.assertEqual(info, mock_info)

    @mock.patch.object(huggingface_hub, "HfApi")
    def test_check_access_gated_unauthorized(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.side_effect = huggingface_hub.utils.GatedRepoError(
            "Gated access required",
            response=mock.Mock(),  # pytype: disable=wrong-arg-types
        )

        ok, msg, info = preflight_check.check_access("gated-model")
        self.assertFalse(ok)
        self.assertIn("Access denied (Gated repository)", msg)
        self.assertIsNone(info)

    @mock.patch.object(huggingface_hub, "HfApi")
    def test_check_access_not_found(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.side_effect = (
            huggingface_hub.utils.RepositoryNotFoundError(
                "Repo not found",
                response=mock.Mock(),  # pytype: disable=wrong-arg-types
            ))

        ok, msg, info = preflight_check.check_access("fake-model")
        self.assertFalse(ok)
        self.assertIn("Repository not found", msg)
        self.assertIsNone(info)

    @mock.patch.object(huggingface_hub, "HfApi")
    def test_check_access_generic_exception(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.side_effect = RuntimeError("Connection timed out")

        ok, msg, info = preflight_check.check_access("some-model")
        self.assertFalse(ok)
        self.assertIn("Error checking model info", msg)
        self.assertIsNone(info)

    @parameterized.named_parameters(
        (
            "no_siblings",
            MockModelInfo(siblings=[]),
            True,
            "No file metadata found",
        ),
        (
            "pytorch_safetensors",
            MockModelInfo(siblings=[MockSibling("model.safetensors")]),
            True,
            "Format is likely compatible",
        ),
        (
            "pytorch_bin",
            MockModelInfo(siblings=[MockSibling("pytorch_model.bin")]),
            True,
            "Format is likely compatible",
        ),
        (
            "pytorch_pth",
            MockModelInfo(siblings=[MockSibling("weights.pth")]),
            True,
            "Format is likely compatible",
        ),
        (
            "gguf_only",
            MockModelInfo(siblings=[MockSibling("model.gguf")]),
            False,
            "Incompatible format: GGUF",
        ),
        (
            "tflite_only",
            MockModelInfo(siblings=[MockSibling("model.tflite")]),
            False,
            "Incompatible format: TFLite",
        ),
        (
            "gguf_and_tflite",
            MockModelInfo(siblings=[
                MockSibling("model.gguf"),
                MockSibling("model.tflite")
            ]),
            False,
            "Incompatible format: GGUF, TFLite",
        ),
        (
            "gguf_and_pytorch",
            MockModelInfo(siblings=[
                MockSibling("model.gguf"),
                MockSibling("model.safetensors"),
            ]),
            True,
            "Format is likely compatible",
        ),
    )
    def test_check_formats(
        self,
        model_info: Any,
        expected_ok: bool,
        expected_submsg: str,
    ):
        ok, msg = preflight_check.check_formats(model_info)  # pytype: disable=wrong-arg-types
        self.assertIs(ok, expected_ok)
        self.assertIn(expected_submsg, msg)

    def test_check_for_incompatible_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create standard Python file
            normal_py = os.path.join(tmpdir, "normal.py")
            with open(normal_py, "w", encoding="utf-8") as f:
                f.write("import os\nfrom typing import Any\n")

            # Create a file importing triton
            bad_py1 = os.path.join(tmpdir, "bad1.py")
            with open(bad_py1, "w", encoding="utf-8") as f:
                f.write("import os\nimport triton\n")

            # Create a file using from flash_attn import ...
            bad_py2 = os.path.join(tmpdir, "bad2.py")
            with open(bad_py2, "w", encoding="utf-8") as f:
                f.write("from flash_attn import flash_attn_func\n")

            # Create a file with invalid syntax that should be skipped elegantly
            syntax_error_py = os.path.join(tmpdir, "syntax_error.py")
            with open(syntax_error_py, "w", encoding="utf-8") as f:
                f.write("class Foo\n  pass\n")

            issues = preflight_check.check_for_incompatible_imports(tmpdir)
            issues_dict = {item[0]: item[2] for item in issues}

            self.assertIn("bad1.py", issues_dict)
            self.assertEqual(issues_dict["bad1.py"], "triton")
            self.assertIn("bad2.py", issues_dict)
            self.assertEqual(issues_dict["bad2.py"], "flash_attn")
            self.assertNotIn("normal.py", issues_dict)
            self.assertNotIn("syntax_error.py", issues_dict)

    def test_check_for_custom_cuda_kernels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Standalone .cu file
            cu_file = os.path.join(tmpdir, "kernel.cu")
            with open(cu_file, "w", encoding="utf-8") as f:
                f.write("// CUDA kernel code\n")

            # File importing proprietary CUDA extension
            ext_py = os.path.join(tmpdir, "custom_layer.py")
            with open(ext_py, "w", encoding="utf-8") as f:
                f.write("from inference_extensions_cuda import process_cuda\n")

            # File referencing torch.ops custom op
            ops_py = os.path.join(tmpdir, "custom_ops.py")
            with open(ops_py, "w", encoding="utf-8") as f:
                f.write("import torch\nx = torch.ops.my_custom_kernel(a, b)\n")

            issues = preflight_check.check_for_incompatible_imports(tmpdir)
            issues_dict = {item[0]: item[2] for item in issues}

            self.assertIn("kernel.cu", issues_dict)
            self.assertEqual(issues_dict["kernel.cu"], "custom_cuda")
            self.assertIn("custom_layer.py", issues_dict)
            self.assertEqual(issues_dict["custom_layer.py"],
                             "inference_extensions_cuda")
            self.assertIn("custom_ops.py", issues_dict)
            self.assertEqual(issues_dict["custom_ops.py"], "custom_cuda")

    def test_check_for_incompatible_imports_line_and_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "layer.py"), "w",
                      encoding="utf-8") as f:
                f.write("import os\n"
                        "from foo_cuda import op_a, op_b\n"
                        "import torch\n"
                        "y = torch.ops.my_ext.fused(x)\n")
            issues = preflight_check.check_for_incompatible_imports(tmpdir)
            self.assertCountEqual(issues, [
                ("layer.py", 2, "foo_cuda", "op_a"),
                ("layer.py", 2, "foo_cuda", "op_b"),
                ("layer.py", 4, "custom_cuda", "torch.ops.my_ext"),
            ])

    def test_check_for_incompatible_imports_ignores_builtin_torch_ops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "builtin_ops.py"),
                      "w",
                      encoding="utf-8") as f:
                f.write("import torch\n"
                        "import torch.cuda\n"
                        "a = torch.ops.aten.add(x, y)\n"
                        "b = torch.ops.prims.convert_element_type(x, t)\n"
                        "c = torch.ops.torchvision.nms(bx, sc, 0.5)\n"
                        "d = torch.ops.quantized.linear(x, w)\n")
            issues = preflight_check.check_for_incompatible_imports(tmpdir)
            self.assertEqual(issues, [])

    def test_validate_config_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertFalse(ok)
            self.assertIn("Missing config", msg)
            self.assertIsNone(config)

    def test_validate_config_model_index_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid diffusers model index setup
            model_index = os.path.join(tmpdir, "model_index.json")
            with open(model_index, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "_class_name": "StableDiffusionPipeline",
                        "scheduler": [None, "Scheduler"],
                    },
                    f,
                )

            # scheduler component is None, so no directory needed
            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertTrue(ok)
            self.assertIn(
                "Diffusers pipeline (StableDiffusionPipeline) is valid.", msg)
            assert config is not None
            self.assertEqual(config["_class_name"], "StableDiffusionPipeline")

    def test_validate_config_model_index_missing_component(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_index = os.path.join(tmpdir, "model_index.json")
            with open(model_index, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "_class_name": "StableDiffusionPipeline",
                        "unet": ["unet_lib", "UNet2DConditionModel"],
                    },
                    f,
                )

            # unet is not None but has no component subdirectory
            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertFalse(ok)
            self.assertIn(
                "directory 'unet' (class: UNet2DConditionModel) is missing",
                msg,
            )
            self.assertIsNotNone(config)

    @parameterized.named_parameters(
        ("config_yaml", "config.yaml"),
        ("hyperparams_yaml", "hyperparams.yaml"),
    )
    def test_validate_config_yaml(self, filename: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("model: custom\n")

            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertTrue(ok)
            self.assertIn("YAML Config (pyannote/espnet/speechbrain)", msg)
            assert config is not None
            self.assertEqual(config["model_type"], "yaml_custom")

    def test_validate_config_timm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = os.path.join(tmpdir, "config.json")
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump({
                    "architecture": "resnet50",
                    "pretrained_cfg": {}
                }, f)

            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertTrue(ok)
            self.assertIn("Timm model config", msg)
            assert config is not None
            self.assertEqual(config["architecture"], "resnet50")

    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_validate_config_huggingface_success(self, mock_from_pretrained):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = os.path.join(tmpdir, "config.json")
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump({"model_type": "llama"}, f)

            mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                               instance=True)
            mock_config.quantization_config = None
            mock_config.model_type = "llama"
            mock_config.architectures = ["LlamaForCausalLM"]
            mock_from_pretrained.return_value = mock_config

            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertTrue(ok)
            self.assertIn("config.json is valid and loadable", msg)
            self.assertEqual(config, mock_config)

    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_validate_config_huggingface_quantization_awq(
            self, mock_from_pretrained):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = os.path.join(tmpdir, "config.json")
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump({"model_type": "llama"}, f)

            mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                               instance=True)
            mock_config.quantization_config = {"quant_method": "awq"}
            mock_config.model_type = "llama"
            mock_config.architectures = ["LlamaForCausalLM"]
            mock_from_pretrained.return_value = mock_config

            ok, msg, config = preflight_check.validate_config(tmpdir)
            self.assertFalse(ok)
            self.assertIn("Incompatible quantization method: awq", msg)
            self.assertEqual(config, mock_config)

    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_validate_config_huggingface_specialized(self,
                                                     mock_from_pretrained):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = os.path.join(tmpdir, "config.json")
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump({"model_type": "layoutlm"}, f)

            mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                               instance=True)
            mock_config.quantization_config = None
            mock_config.model_type = "layoutlm"
            mock_config.architectures = ["LayoutLMForTokenClassification"]
            mock_from_pretrained.return_value = mock_config

            # Should pass but print specialized warning
            with mock.patch.object(sys, "stderr") as mock_stderr:
                ok, msg, config = preflight_check.validate_config(tmpdir)
                self.assertTrue(ok)
                self.assertIn("config.json is valid and loadable", msg)
                self.assertIsNotNone(config)
                mock_stderr.write.assert_called()

    @parameterized.named_parameters(
        (
            "diffusers",
            {
                "_class_name": "StableDiffusionPipeline"
            },
            True,
            "Diffusers pipeline",
        ),
        (
            "custom_yaml_espnet",
            {
                "model_type": "yaml_custom"
            },
            True,
            "Custom library model",
        ),
        (
            "speechbrain",
            {
                "speechbrain_interface": "encoder"
            },
            True,
            "Custom library model",
        ),
        (
            "timm",
            {
                "architecture": "resnet50",
                "pretrained_cfg": {}
            },
            True,
            "Timm model",
        ),
    )
    def test_validate_processor_bypasses(
        self,
        config: Any,
        expected_ok: bool,
        expected_submsg: str,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, msg = preflight_check.validate_processor(tmpdir, config)
            self.assertEqual(ok, expected_ok)
            self.assertIn(expected_submsg, msg)

    @mock.patch.object(transformers.AutoTokenizer, "from_pretrained")
    def test_validate_processor_standard_tokenizer_success(self, mock_tok):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_config = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tokenizer_config, "w", encoding="utf-8") as f:
                f.write("{}")

            mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                               instance=True)
            mock_config.model_type = "llama"
            mock_config.architectures = ["LlamaForCausalLM"]

            ok, msg = preflight_check.validate_processor(tmpdir, mock_config)
            self.assertTrue(ok)
            self.assertIn("Tokenizer/Processor check passed.", msg)
            mock_tok.assert_called_once_with(tmpdir, trust_remote_code=True)

    @mock.patch.object(transformers.AutoTokenizer, "from_pretrained")
    def test_validate_processor_standard_tokenizer_failed(self, mock_tok):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_tok.side_effect = Exception("Tokenizer payload corrupt")
            mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                               instance=True)
            mock_config.model_type = "llama"
            mock_config.architectures = ["LlamaForCausalLM"]

            ok, msg = preflight_check.validate_processor(tmpdir, mock_config)
            self.assertFalse(ok)
            self.assertIn("Failed to instantiate tokenizer", msg)

    @mock.patch.object(transformers.AutoTokenizer, "from_pretrained")
    def test_check_vocab_bounds_compatible(self, mock_tokenizer_class):
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.__len__.return_value = 32000
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                           instance=True)
        mock_config.vocab_size = 32000

        ok, msg = preflight_check.check_vocab_bounds("some-dir", mock_config)
        self.assertTrue(ok)
        self.assertIn("passed", msg)

    @mock.patch.object(transformers.AutoTokenizer, "from_pretrained")
    def test_check_vocab_bounds_incompatible(self, mock_tokenizer_class):
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.__len__.return_value = 32005
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_config = mock.create_autospec(transformers.PretrainedConfig,
                                           instance=True)
        mock_config.vocab_size = 32000

        ok, msg = preflight_check.check_vocab_bounds("some-dir", mock_config)
        self.assertFalse(ok)
        self.assertIn("exceeds configuration embedding vocab size", msg)

    def test_handle_failure_no_force(self):
        FLAGS.force = False
        with self.assertRaises(SystemExit) as cm:
            preflight_check.handle_failure("Failed validation")
        self.assertEqual(cm.exception.code, 1)

    def test_handle_failure_with_force(self):
        FLAGS.force = True
        # Should print bypass message and not exit
        try:
            preflight_check.handle_failure("Failed validation but bypassed")
        except SystemExit:
            self.fail(
                "handle_failure raised SystemExit unexpectedly with --force")

    @mock.patch.object(preflight_check, "check_vocab_bounds")
    @mock.patch.object(preflight_check, "validate_processor")
    @mock.patch.object(preflight_check, "validate_config")
    @mock.patch.object(preflight_check, "check_for_incompatible_imports")
    @mock.patch.object(huggingface_hub, "snapshot_download")
    @mock.patch.object(preflight_check, "check_formats")
    @mock.patch.object(preflight_check, "check_access")
    def test_main_execution_sequence(
        self,
        mock_check_access,
        mock_check_formats,
        mock_snapshot_download,
        mock_check_imports,
        mock_validate_config,
        mock_validate_processor,
        mock_check_vocab,
    ):
        orig_model = FLAGS.model
        FLAGS.model = "test-model-id"
        try:
            mock_check_access.return_value = (
                True,
                "Access wrapper output",
                MockModelInfo(),
            )
            mock_check_formats.return_value = (True, "Formats wrapper output")
            mock_snapshot_download.return_value = "/mock/local/dir"
            mock_check_imports.return_value = []
            mock_validate_config.return_value = (
                True,
                "Config validated",
                mock.create_autospec(transformers.PretrainedConfig,
                                     instance=True),
            )
            mock_validate_processor.return_value = (True, "Processor verified")
            mock_check_vocab.return_value = (True, "Vocab bounds passed")

            # Use a parent mock / manager mock to trace call order
            manager = mock.Mock()
            manager.attach_mock(mock_check_imports, "check_imports")
            manager.attach_mock(mock_validate_config, "validate_config")

            preflight_check.main(["preflight_check"])

            # Verify order: check_imports should be called before validate_config
            calls = manager.mock_calls
            call_names = [call[0] for call in calls]
            self.assertIn("check_imports", call_names)
            self.assertIn("validate_config", call_names)
            self.assertLess(
                call_names.index("check_imports"),
                call_names.index("validate_config"),
            )
        finally:
            FLAGS.model = orig_model

    @mock.patch.object(preflight_check, "check_vocab_bounds")
    @mock.patch.object(preflight_check, "validate_processor")
    @mock.patch.object(preflight_check, "validate_config")
    @mock.patch.object(preflight_check, "check_for_incompatible_imports")
    @mock.patch.object(huggingface_hub, "snapshot_download")
    @mock.patch.object(preflight_check, "check_formats")
    @mock.patch.object(preflight_check, "check_access")
    def test_main_incompatible_imports_no_allow(
            self,
            mock_check_access,
            mock_check_formats,
            mock_snapshot_download,
            mock_check_imports,
            mock_validate_config,  # pylint: disable=unused-argument
            mock_validate_processor,  # pylint: disable=unused-argument
            mock_check_vocab,  # pylint: disable=unused-argument
    ):
        orig_model = FLAGS.model
        orig_allow = FLAGS.allow_gpu_imports
        orig_force = FLAGS.force
        FLAGS.model = "test-model-id"
        FLAGS.allow_gpu_imports = False
        FLAGS.force = False
        try:
            mock_check_access.return_value = (True, "Access OK",
                                              MockModelInfo())
            mock_check_formats.return_value = (True, "Formats OK")
            mock_snapshot_download.return_value = "/mock/local/dir"
            # Simulate finding incompatible unmapped imports.
            mock_check_imports.return_value = [
                ("bad.py", 1, "unknown_cuda_kernel", "unknown_cuda_kernel")
            ]

            # We expect handle_failure to raise SystemExit.
            with self.assertRaises(SystemExit) as cm:
                preflight_check.main(["preflight_check"])
            self.assertEqual(cm.exception.code, 1)
        finally:
            FLAGS.model = orig_model
            FLAGS.allow_gpu_imports = orig_allow
            FLAGS.force = orig_force

    @mock.patch.object(preflight_check, "handle_failure")
    @mock.patch.object(preflight_check, "check_vocab_bounds")
    @mock.patch.object(preflight_check, "validate_processor")
    @mock.patch.object(preflight_check, "validate_config")
    @mock.patch.object(preflight_check, "check_for_incompatible_imports")
    @mock.patch.object(huggingface_hub, "snapshot_download")
    @mock.patch.object(preflight_check, "check_formats")
    @mock.patch.object(preflight_check, "check_access")
    def test_main_incompatible_imports_with_allow(
        self,
        mock_check_access,
        mock_check_formats,
        mock_snapshot_download,
        mock_check_imports,
        mock_validate_config,
        mock_validate_processor,
        mock_check_vocab,
        mock_handle_failure,
    ):
        orig_model = FLAGS.model
        orig_allow = FLAGS.allow_gpu_imports
        FLAGS.model = "test-model-id"
        FLAGS.allow_gpu_imports = True
        try:
            mock_check_access.return_value = (True, "Access OK",
                                              MockModelInfo())
            mock_check_formats.return_value = (True, "Formats OK")
            mock_snapshot_download.return_value = "/mock/local/dir"
            # Simulate finding incompatible imports.
            mock_check_imports.return_value = [("bad.py", 1, "triton",
                                                "triton")]

            # Mock subsequent calls to avoid failures.
            mock_validate_config.return_value = (
                True,
                "Config OK",
                mock.create_autospec(transformers.PretrainedConfig,
                                     instance=True),
            )
            mock_validate_processor.return_value = (True, "Processor OK")
            mock_check_vocab.return_value = (True, "Vocab OK")

            # We expect handle_failure NOT to be called
            preflight_check.main(["preflight_check"])
            mock_handle_failure.assert_not_called()
        finally:
            FLAGS.model = orig_model
            FLAGS.allow_gpu_imports = orig_allow

    def test_write_cuda_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "cuda_report.json")
            issues = [("attention.py", 3, "flash_attn", "flash_attn_func"),
                      ("custom_ops.py", 7, "unknown_cuda_kernel", "my_op")]
            playbook_map = {
                "flash_attn":
                "package/data/tpu-knowledge-base/pages/flash-attention-to-sdpa.md"
            }
            preflight_check.write_cuda_report(
                report_path=report_path,
                model_id="test-model",
                mode="benchmark",
                issues=issues,
                playbook_mappings=playbook_map,
            )
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["model_id"], "test-model")
            self.assertEqual(data["mode"], "benchmark")
            self.assertEqual(data["status"], "DISCOVERED")
            self.assertTrue(data["tpu_kernel_development_handoff_ready"])
            self.assertEqual(len(data["kernels"]), 2)
            self.assertEqual(data["kernels"][0]["playbook_status"], "HIT")
            self.assertEqual(data["kernels"][0]["line"], 3)
            self.assertEqual(data["kernels"][0]["symbol"], "flash_attn_func")
            self.assertEqual(data["kernels"][1]["playbook_status"], "MISS")
            self.assertIsNone(data["kernels"][1]["playbook"])
            self.assertEqual(data["kernels"][1]["line"], 7)
            self.assertEqual(data["kernels"][1]["symbol"], "my_op")

    @staticmethod
    def _write_playbook(directory: str, name: str, target: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(f"---\ntarget_imports:\n  - {target}\n---\n# Playbook\n")

    def test_discover_playbook_mappings_env_override_wins(self):
        with tempfile.TemporaryDirectory() as env_kb, \
                tempfile.TemporaryDirectory() as cwd:
            self._write_playbook(os.path.join(env_kb, "pages"), "env.md",
                                 "from_env")
            self._write_playbook(
                os.path.join(cwd, "package", "data", "tpu-knowledge-base",
                             "pages"), "cwd.md", "from_cwd")
            with mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_KNOWLEDGE_BASE_HOME": env_kb}), \
                    mock.patch.object(preflight_check.Path, "cwd",
                                      return_value=preflight_check.Path(cwd)):
                mappings = preflight_check.discover_playbook_mappings()
            self.assertIn("from_env", mappings)
            self.assertNotIn("from_cwd", mappings)

    def test_discover_playbook_mappings_skips_empty_env_dir(self):
        with tempfile.TemporaryDirectory() as env_kb, \
                tempfile.TemporaryDirectory() as cwd:
            os.makedirs(os.path.join(env_kb, "pages"))  # no *.md
            self._write_playbook(
                os.path.join(cwd, "package", "data", "tpu-knowledge-base",
                             "pages"), "cwd.md", "from_cwd")
            with mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_KNOWLEDGE_BASE_HOME": env_kb}), \
                    mock.patch.object(preflight_check.Path, "cwd",
                                      return_value=preflight_check.Path(cwd)):
                mappings = preflight_check.discover_playbook_mappings()
            self.assertIn("from_cwd", mappings)

    def test_check_local_formats_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.safetensors"), "wb") as f:
                f.write(b"")
            ok, msg = preflight_check.check_local_formats(tmpdir)
            self.assertTrue(ok)
            self.assertIn("Format is likely compatible", msg)

    def test_check_local_formats_incompatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.gguf"), "wb") as f:
                f.write(b"")
            ok, msg = preflight_check.check_local_formats(tmpdir)
            self.assertFalse(ok)
            self.assertIn("Incompatible format", msg)

    @mock.patch.object(preflight_check, "check_vocab_bounds")
    @mock.patch.object(preflight_check, "validate_processor")
    @mock.patch.object(preflight_check, "validate_config")
    @mock.patch.object(preflight_check, "check_for_incompatible_imports")
    def test_main_local_checkpoint(
        self,
        mock_check_imports,
        mock_validate_config,
        mock_validate_proc,
        mock_check_vocab,
    ):
        mock_check_imports.return_value = []
        mock_validate_config.return_value = (
            True,
            "Config OK",
            {
                "model_type": "bert"
            },
        )
        mock_validate_proc.return_value = (True, "Processor OK")
        mock_check_vocab.return_value = (True, "Vocab OK")

        orig_model = FLAGS.model
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "model.safetensors"),
                          "wb") as f:
                    f.write(b"")
                FLAGS.model = tmpdir
                preflight_check.main(["preflight_check"])
                mock_check_imports.assert_called_once()
                mock_validate_config.assert_called_once()
                mock_validate_proc.assert_called_once()
                mock_check_vocab.assert_called_once()
        finally:
            FLAGS.model = orig_model

    @parameterized.named_parameters(
        ("transformers", "transformers", "A"),
        ("diffusers_padded", " Diffusers ", "A"),
        ("gguf", "gguf", "None"),
        ("tensorflow", "tensorflow", "None"),
        ("spacy", "spacy", "B"),
    )
    def test_classify_library_tier(self, library_name: str, expected: str):
        self.assertEqual(preflight_check.classify_library_tier(library_name),
                         expected)

    @parameterized.named_parameters(
        ("none", None, None),
        ("empty", {}, None),
        ("awq", {
            "quant_method": "awq"
        }, False),
        ("gptq", {
            "quant_method": "gptq"
        }, False),
        ("mxfp4", {
            "quant_method": "mxfp4"
        }, True),
        ("object", types.SimpleNamespace(quant_method="gptq"), False),
    )
    def test_check_quantization(self, quant_config: Any,
                                expected_ok: bool | None):
        result = preflight_check.check_quantization(quant_config)
        if expected_ok is None:
            self.assertIsNone(result)
        else:
            assert result is not None
            self.assertIs(result[0], expected_ok)

    @parameterized.named_parameters(
        # Benchmark: transformers' quantizer for each needs GPU kernels.
        ("benchmark_bitsandbytes", "bitsandbytes", "benchmark", False),
        ("benchmark_eetq", "eetq", "benchmark", False),
        ("benchmark_fbgemm_fp8", "fbgemm_fp8", "benchmark", False),
        ("benchmark_fp8", "fp8", "benchmark", False),
        ("benchmark_vptq", "vptq", "benchmark", False),
        ("benchmark_aqlm", "aqlm", "benchmark", False),
        ("benchmark_hqq_unproven", "hqq", "benchmark", True),
        ("benchmark_torchao_unproven", "torchao", "benchmark", True),
        ("benchmark_mxfp4", "mxfp4", "benchmark", True),
        # Inference: what the vLLM TorchTPU plugin's quantization map serves.
        ("inference_fp8", "fp8", "inference", True),
        ("inference_mxfp4", "mxfp4", "inference", True),
        ("inference_compressed_tensors", "compressed-tensors", "inference",
         True),
        ("inference_deepseek_v4_fp8", "deepseek_v4_fp8", "inference", True),
        ("inference_hqq", "hqq", "inference", False),
        ("inference_awq", "awq", "inference", False),
        ("inference_gptq", "gptq", "inference", False),
        ("inference_bitsandbytes", "bitsandbytes", "inference", False),
    )
    def test_check_quantization_by_mode(self, method: str, mode: str,
                                        expected_ok: bool):
        result = preflight_check.check_quantization({"quant_method": method},
                                                    mode)
        assert result is not None
        ok, message = result
        self.assertIs(ok, expected_ok)
        if not ok:
            self.assertIn(f"Incompatible quantization method: {method}",
                          message)
        if not ok and mode == "inference":
            self.assertIn("the vLLM TorchTPU plugin does not serve it",
                          message)

    @parameterized.named_parameters(
        ("nvfp4_top_level", {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4"
        }, True),
        ("nvfp4_nested", {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "NVFP4"
            }
        }, True),
        ("modelopt_fp8", {
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        }, False),
    )
    def test_modelopt_is_judged_by_its_algorithm(self, quant_config: Any,
                                                 expected_ok: bool):
        # vLLM names an FP4 ModelOpt checkpoint modelopt_fp4, which the
        # plugin serves; its FP8 one keeps the name modelopt, which it does not.
        ok, _ = preflight_check.check_quantization(quant_config, "inference")
        self.assertIs(ok, expected_ok)

    def test_read_quantization_config_nested_under_text_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(
                    {
                        "text_config": {
                            "quantization_config": {
                                "quant_method": "awq"
                            }
                        }
                    }, f)
            self.assertEqual(preflight_check.read_quantization_config(tmpdir),
                             {"quant_method": "awq"})

    def test_read_quantization_config_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(preflight_check.read_quantization_config(tmpdir))

    def test_module_imports_without_transformers(self):
        # A fresh interpreter, so absl flags register once and nothing this
        # test process already imported can satisfy `import transformers`.
        code = textwrap.dedent("""
            import sys
            import types
            from unittest import mock

            sys.modules["transformers"] = None  # `import transformers` fails
            import huggingface_hub
            import preflight_check

            assert not preflight_check.transformers_available()
            info = types.SimpleNamespace(
                gated=False, library_name="gguf",
                siblings=[types.SimpleNamespace(rfilename="model.gguf")])
            with mock.patch.object(huggingface_hub, "HfApi") as api:
                api.return_value.model_info.return_value = info
                ok, _, got = preflight_check.check_access("org/model")
            assert ok and got is info
            assert preflight_check.check_formats(info)[0] is False
            assert preflight_check.classify_library_tier("gguf") == "None"
            assert preflight_check.check_quantization(
                {"quant_method": "awq"})[0] is False
            print("ok")
            """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.abspath(preflight_check.__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def _run_main_on_snapshot(self, config: dict[str, Any],
                              **flag_values: Any) -> str:
        """Runs main() on a Hub model whose snapshot holds `config`."""
        saved = {name: getattr(FLAGS, name) for name in flag_values}
        saved["model"] = FLAGS.model
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(config, f)
            FLAGS.model = "org/model"
            for name, value in flag_values.items():
                setattr(FLAGS, name, value)
            out = io.StringIO()
            try:
                with mock.patch.object(
                        preflight_check,
                        "check_access",
                        return_value=(True, "Public model.", MockModelInfo(
                            library_name="transformers"))), \
                        mock.patch.object(huggingface_hub,
                                          "snapshot_download",
                                          return_value=tmpdir), \
                        redirect_stdout(out):
                    preflight_check.main(["preflight_check"])
            finally:
                for name, value in saved.items():
                    setattr(FLAGS, name, value)
        return out.getvalue()

    def test_main_hub_only_runs_without_transformers(self):
        with mock.patch.dict(sys.modules, {"transformers": None}):
            out = self._run_main_on_snapshot({"model_type": "llama"},
                                             hub_only=True)
        self.assertIn("Quantization Check: No quantization config.", out)
        self.assertIn("PRE-FLIGHT HUB CHECKS PASSED", out)

    def test_main_writes_empty_cuda_report_when_scan_is_clean(self):
        # compute run fails a run whose declared --output was never created.
        with tempfile.TemporaryDirectory() as out_dir:
            report_path = os.path.join(out_dir, "cuda_kernels_report.json")
            self._run_main_on_snapshot({"model_type": "llama"},
                                       hub_only=True,
                                       cuda_report_json=report_path)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["model_id"], "org/model")
        self.assertEqual(report["kernels"], [])
        self.assertFalse(report["tpu_kernel_development_handoff_ready"])

    def test_main_hub_only_applies_quantization_rule(self):
        with mock.patch.dict(sys.modules, {"transformers": None}), \
                self.assertRaises(SystemExit) as cm:
            self._run_main_on_snapshot(
                {"quantization_config": {
                    "quant_method": "gptq"
                }},
                hub_only=True)
        self.assertEqual(cm.exception.code, 1)

    def test_main_without_transformers_fails_clearly(self):
        with mock.patch.dict(sys.modules, {"transformers": None}), \
                mock.patch.object(sys, "stderr", new=io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            self._run_main_on_snapshot({"model_type": "llama"})
        self.assertIn("transformers is not installed here", err.getvalue())
        self.assertIn("--hub_only", err.getvalue())


def _hub_model(repo_id: str,
               downloads: int = 0,
               params: float | None = None) -> types.SimpleNamespace:
    """A canned `ModelInfo`, carrying the fields list_models is asked to expand."""
    return types.SimpleNamespace(
        id=repo_id,
        downloads=downloads,
        pipeline_tag="text-generation",
        gated=False,
        safetensors=(types.SimpleNamespace(total=params) if params else None),
    )


def _not_found() -> huggingface_hub.utils.RepositoryNotFoundError:
    return huggingface_hub.utils.RepositoryNotFoundError(
        "Repo not found",
        response=mock.Mock(status_code=404),  # pytype: disable=wrong-arg-types
    )


def _http_error(status: int) -> huggingface_hub.utils.HfHubHTTPError:
    return huggingface_hub.utils.HfHubHTTPError(
        f"{status} Server Error",
        response=mock.Mock(status_code=status),  # pytype: disable=wrong-arg-types
    )


# Search results by search term, shaped like the Hub's answers.
GPT_OSS_SEARCH = {
    "gpt-oss-20b": [
        _hub_model("openai/gpt-oss-20b", 9_000_000, 21e9),
        _hub_model("unsloth/gpt-oss-20b-GGUF", 700_000, 21e9),
        _hub_model("openai/gpt-oss-safeguard-20b", 90_000, 21e9),
    ],
}

LLAMA_SEARCH = {
    # Meta-Llama-3-8B does not contain "llama-3-8b", so the first term misses.
    "llama-3-8b": [],
    "llama-3": [
        _hub_model("meta-llama/Meta-Llama-3-8B-Instruct", 2_000_000, 8.03e9),
        _hub_model("meta-llama/Meta-Llama-3-70B", 500_000, 70.6e9),
        _hub_model("meta-llama/Meta-Llama-3-8B", 900_000, 8.03e9),
        _hub_model("TheBloke/Llama-3-8B-GPTQ", 5_000_000, 8.03e9),
        _hub_model("someone/llama-3-8b-finetune", 50_000, 8.03e9),
    ],
}


class ResolveModelNameTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.api = None  # The mocked HfApi of the last _resolve call.

    def _resolve(self,
                 query: str,
                 search: dict[str, list[Any]],
                 model_info: Any = None,
                 limit: int = 3) -> dict[str, Any]:
        """Resolves `query` against a Hub that answers with canned payloads.

        `model_info` is what an exact-id lookup returns; by default the id is
        not found, as for a loose name.
        """
        if model_info is None:
            model_info = _not_found()
        with mock.patch.object(huggingface_hub, "HfApi") as api_class:
            api = api_class.return_value
            api.list_models.side_effect = (
                lambda **kwargs: iter(search.get(kwargs["search"], [])))
            if isinstance(model_info, BaseException):
                api.model_info.side_effect = model_info
            else:
                api.model_info.return_value = model_info
            self.api = api
            return preflight_check.resolve_report(query, limit)

    def test_exact_id(self):
        report = self._resolve("openai/gpt-oss-20b", {},
                               model_info=_hub_model("openai/gpt-oss-20b",
                                                     9_000_000, 21e9))
        self.assertEqual(report["verdict"], "exact")
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")
        self.assertEqual(report["url"],
                         "https://huggingface.co/openai/gpt-oss-20b")
        self.api.list_models.assert_not_called()

    def test_exact_gated_id(self):
        report = self._resolve("meta-llama/Meta-Llama-3-8B", {},
                               model_info=huggingface_hub.utils.GatedRepoError(
                                   "Gated",
                                   response=mock.Mock(status_code=403)))
        self.assertEqual(report["verdict"], "exact")
        self.assertTrue(report["candidates"][0]["gated"])

    @parameterized.named_parameters(
        ("gpt2", "gpt2", "openai-community/gpt2"),
        ("bert", "bert-base-uncased", "google-bert/bert-base-uncased"),
    )
    def test_id_without_an_org_is_exact(self, query: str, redirected: str):
        report = self._resolve(query, {},
                               model_info=_hub_model(redirected, 1_000_000))
        self.assertEqual(report["verdict"], "exact")
        self.assertEqual(report["repo_id"], redirected)
        self.assertEqual(report["reason"],
                         f"the id given is now {redirected} on the Hub")
        self.api.model_info.assert_called_once_with(query)
        self.api.list_models.assert_not_called()

    def test_loose_name_skips_the_exact_lookup(self):
        self._resolve("gpt oss 20b", GPT_OSS_SEARCH)
        self.api.model_info.assert_not_called()

    def test_unknown_id_without_an_org_falls_back_to_search(self):
        report = self._resolve("pythia-1b", {
            "pythia-1b": [_hub_model("EleutherAI/pythia-1b", 400_000, 1.01e9)]
        })
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "EleutherAI/pythia-1b")
        self.api.model_info.assert_called_once_with("pythia-1b")

    def test_unknown_id_falls_back_to_search(self):
        report = self._resolve("gpt-oss/20b", {"gpt-oss/20b": []},
                               model_info=_not_found())
        self.assertEqual(report["verdict"], "not_found")
        self.api.list_models.assert_called()

    def test_named_exactly_is_single_match(self):
        # gpt-oss-safeguard-20b also has every word of the name, but only
        # openai/gpt-oss-20b is named as asked.
        report = self._resolve("gpt oss 20b", GPT_OSS_SEARCH)
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")
        self.assertEqual(
            [c["repo_id"] for c in report["candidates"]],
            [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-safeguard-20b",
                "unsloth/gpt-oss-20b-GGUF",
            ],
        )

    def test_competing_official_repos_are_ambiguous(self):
        report = self._resolve("llama 3 8b", LLAMA_SEARCH, limit=5)
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertEqual(report["reason"],
                         "several official repositories match")
        self.assertIsNone(report["repo_id"])
        # Official before community, unquantized before quantized, the size
        # asked for before other sizes, then downloads.
        self.assertEqual(
            [c["repo_id"] for c in report["candidates"]],
            [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-70B",
                "someone/llama-3-8b-finetune",
                "TheBloke/Llama-3-8B-GPTQ",
            ],
        )
        top = report["candidates"][0]
        self.assertEqual(
            top["url"],
            "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertEqual(top["variant"], "instruct")
        self.assertTrue(report["candidates"][-1]["quantized"])
        searched = [
            c.kwargs["search"] for c in self.api.list_models.call_args_list
        ]
        self.assertEqual(searched, ["llama-3-8b", "llama-3"])

    def test_limit_caps_candidates(self):
        report = self._resolve("llama 3 8b", LLAMA_SEARCH, limit=2)
        self.assertLen(report["candidates"], 2)

    def test_other_size_is_not_a_rival(self):
        # The 70B checkpoint is official and unquantized, but not 8B.
        search = {
            "smollm-8b": [],
            "smollm": [
                _hub_model("HuggingFaceTB/SmolLM-8B-Instruct-v2", 100, 8e9),
                _hub_model("HuggingFaceTB/SmolLM-70B", 50, 70e9),
                _hub_model("someone/SmolLM-8B-GGUF", 1_000),
            ],
        }
        report = self._resolve("smollm 8b", search)
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["reason"],
                         "one official, unquantized repository matches")
        self.assertEqual(report["repo_id"],
                         "HuggingFaceTB/SmolLM-8B-Instruct-v2")

    def test_only_community_repos_are_ambiguous(self):
        search = {"tinyllama": [_hub_model("someone/TinyLlama-chat", 10)]}
        report = self._resolve("tinyllama", search)
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertEqual(report["reason"],
                         "no official unquantized repository matches")
        self.assertEqual(report["candidates"][0]["repo_id"],
                         "someone/TinyLlama-chat")

    @parameterized.named_parameters(
        ("eleutherai", "pythia 1b", "pythia-1b", [
            _hub_model("EleutherAI/pythia-1b", 400_000, 1.01e9),
            _hub_model("EleutherAI/pythia-1b-deduped", 90_000, 1.01e9),
            _hub_model("someone/ppt-pythia-1b-stage2", 10, 1.01e9),
        ], "EleutherAI/pythia-1b"),
        ("baai", "bge m3", "bge-m3", [
            _hub_model("BAAI/bge-m3", 9_000_000, 568e6),
            _hub_model("BAAI/bge-reranker-v2-m3", 3_000_000, 568e6),
            _hub_model("deepvk/USER-bge-m3", 20_000, 359e6),
        ], "BAAI/bge-m3"),
    )
    def test_reference_org_beyond_the_big_labs_is_chosen(
            self, query: str, term: str, hits: list[Any], expected: str):
        report = self._resolve(query, {term: hits})
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], expected)

    def test_not_found(self):
        report = self._resolve("no such model 7b", {})
        self.assertEqual(report["verdict"], "not_found")
        self.assertEqual(report["candidates"], [])

    @parameterized.named_parameters(
        ("qwen", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5 7b"),
        ("llama", "meta-llama/Llama-3.1-8B", "llama 3.1 8b"),
        ("spaced_version", "Qwen/Qwen2.5-7B", "qwen 2.5 7b"),
        ("phi", "microsoft/Phi-3.5-mini-instruct", "phi 3.5 mini"),
        ("family_finds_point_release", "meta-llama/Llama-3.1-8B",
         "llama 3 8b"),
        ("experts", "mistralai/Mixtral-8x7B-v0.1", "mixtral 8x7b"),
        ("version_marker", "deepseek-ai/DeepSeek-V2-Lite", "deepseek v2"),
    )
    def test_dotted_version_matches(self, repo_id: str, query: str):
        self.assertTrue(preflight_check._name_matches(repo_id, query))

    @parameterized.named_parameters(
        ("size_inside_a_size", "openai/gpt-oss-21b", "1b"),
        ("size_inside_a_size_7b", "someone/model-17b", "7b"),
        ("major_inside_a_point_release", "meta-llama/Llama-3.2-1B", "llama 2"),
        ("number_inside_a_version_marker",
         "nvidia/llama-nemotron-rerank-1b-v2", "llama 2 1b"),
        ("other_point_release", "meta-llama/Llama-3.1-8B", "llama 3.2 8b"),
        ("size_prefix", "meta-llama/Llama-3.1-80B", "llama 3.1 8b"),
        ("underscored_point_release", "someone/ultravox-llama-3_2-1b",
         "llama 2 1b"),
    )
    def test_part_of_a_token_does_not_match(self, repo_id: str, query: str):
        self.assertFalse(preflight_check._name_matches(repo_id, query))

    def test_no_false_match_becomes_a_candidate(self):
        # Substring matching found these for a Llama 2 1B, which does not
        # exist: the 2 of "-v2" and of "3.2".
        search = {
            "llama-2-1b": [
                _hub_model("nvidia/llama-nemotron-rerank-1b-v2", 90_000,
                           1.2e9),
                _hub_model("meta-llama/Llama-3.2-1B", 3_000_000, 1.24e9),
            ],
        }
        report = self._resolve("llama 2 1b", search)
        self.assertEqual(report["verdict"], "not_found")

    def test_spaced_version_is_named_exactly(self):
        # "qwen 2.5 7b" names Qwen2.5-7B as surely as "qwen2.5 7b" does.
        search = {
            "qwen-2.5-7b": [
                _hub_model("Qwen/Qwen2.5-7B-Instruct", 9_000_000, 7.6e9),
                _hub_model("Qwen/Qwen2.5-7B", 2_000_000, 7.6e9),
            ],
        }
        report = self._resolve("qwen 2.5 7b", search)
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "Qwen/Qwen2.5-7B")

    def test_dotted_version_resolves(self):
        search = {
            "qwen2.5-7b": [
                _hub_model("Qwen/Qwen2.5-7B-Instruct", 9_000_000, 7.6e9),
                _hub_model("Qwen/Qwen2.5-7B", 2_000_000, 7.6e9),
                _hub_model("Qwen/Qwen2.5-VL-7B-Instruct", 1_000_000, 8.3e9),
            ],
        }
        report = self._resolve("qwen2.5 7b", search)
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "Qwen/Qwen2.5-7B")

    @parameterized.named_parameters(
        ("plain", "https://huggingface.co/openai/gpt-oss-20b"),
        ("tree", "https://huggingface.co/openai/gpt-oss-20b/tree/main"),
        ("short", "hf.co/openai/gpt-oss-20b"),
        ("query", "huggingface.co/openai/gpt-oss-20b?library=transformers"),
    )
    def test_hub_url_is_an_exact_id(self, url: str):
        report = self._resolve(url, {},
                               model_info=_hub_model("openai/gpt-oss-20b",
                                                     9_000_000, 21e9))
        self.assertEqual(report["verdict"], "exact")
        self.assertEqual(report["query"], url)
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")
        self.api.model_info.assert_called_once_with("openai/gpt-oss-20b")

    def test_org_with_loose_name_searches_within_org(self):
        # "openai/gpt oss 20b" is not a valid repository id; model_info raises
        # HFValidationError rather than RepositoryNotFoundError.
        report = self._resolve(
            "openai/gpt oss 20b",
            GPT_OSS_SEARCH,
            model_info=huggingface_hub.utils.HFValidationError("bad id"))
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")
        self.assertEqual(
            [c["repo_id"] for c in report["candidates"]],
            ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"])

    def test_org_without_matches_searches_everywhere(self):
        report = self._resolve(
            "meta/gpt oss 20b",
            GPT_OSS_SEARCH,
            model_info=huggingface_hub.utils.HFValidationError("bad id"))
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")

    def test_top_of_unknown_size_is_not_chosen(self):
        # The only official repository of no stated size is not the 2B asked
        # for: it is a suggestion, not the answer.
        search = {
            "gemma-3-2b": [
                _hub_model("google/gemma-3n-E2B-it-litert-lm", 5_000),
                _hub_model("google/gemma-3n-E2B-it", 2_000, 5.4e9),
            ],
        }
        report = self._resolve("gemma 3 2b", search)
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertEqual(
            report["reason"],
            "no official repository is known to be the size asked for")
        self.assertIsNone(report["repo_id"])

    def test_top_of_another_size_is_not_chosen(self):
        search = {
            "llama-3-13b": [],
            "llama-3": [
                _hub_model("meta-llama/Meta-Llama-3-8B-Instruct", 2_000_000,
                           8.03e9),
                _hub_model("meta-llama/Meta-Llama-3-70B", 500_000, 70.6e9),
            ],
        }
        report = self._resolve("llama 3 13b", search)
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertIsNone(report["repo_id"])

    @parameterized.named_parameters(
        ("dense", "llama 3 8b", 8e9),
        ("millions", "smollm 360m", 360e6),
        ("mixture_of_experts", "mixtral 8x7b", 56e9),
        ("mixture_of_experts_hyphen", "Mixtral-8x22B", 176e9),
        ("none", "phi 3.5 mini", None),
    )
    def test_size_in_params(self, text: str, expected: float | None):
        self.assertEqual(preflight_check._size_in_params(text), expected)

    def test_mixture_of_experts_base_and_instruct_are_ambiguous(self):
        # Read as 7B, neither ~47B Mixtral was the size asked for, so the
        # instruct one used to be chosen with no rival.
        search = {
            "mixtral-8x7b": [
                _hub_model("mistralai/Mixtral-8x7B-Instruct-v0.1", 900_000,
                           46.7e9),
                _hub_model("mistralai/Mixtral-8x7B-v0.1", 300_000, 46.7e9),
            ],
        }
        report = self._resolve("mixtral 8x7b", search)
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertEqual(report["reason"],
                         "several official repositories match")

    def test_rank_candidates(self):

        def candidate(repo_id, official, quantized, params, downloads):
            return {
                "repo_id": repo_id,
                "official": official,
                "quantized": quantized,
                "params": params,
                "downloads": downloads,
            }

        ranked = preflight_check.rank_candidates([
            candidate("community", False, False, 8e9, 10**9),
            candidate("quantized", True, True, 8e9, 10**8),
            candidate("other_size", True, False, 70e9, 10**7),
            candidate("within_tolerance", True, False, 9e9, 10),
            candidate("unknown_size", True, False, None, 100),
        ], 8e9)
        self.assertEqual([c["repo_id"] for c in ranked], [
            "unknown_size", "within_tolerance", "other_size", "quantized",
            "community"
        ])

    @parameterized.named_parameters(
        ("too_many_requests", 429),
        ("server_error", 500),
        ("unavailable", 503),
    )
    def test_http_error_is_hub_error_not_offline(self, status: int):
        # HfHubHTTPError is an OSError, the trap this guards against: caught as
        # OSError, a rate limit would read as "offline".
        self.assertTrue(
            issubclass(huggingface_hub.utils.HfHubHTTPError, OSError))
        with mock.patch.object(huggingface_hub, "HfApi") as api_class:
            api_class.return_value.list_models.side_effect = _http_error(
                status)
            report = preflight_check.resolve_report("gpt oss 20b")
        self.assertEqual(report["verdict"], "hub_error")
        self.assertEqual(report["http_status"], status)
        self.assertIn(f"HTTP {status}", report["reason"])

    def test_http_error_on_exact_id_is_hub_error(self):
        report = self._resolve("openai/gpt-oss-20b", {},
                               model_info=_http_error(429))
        self.assertEqual(report["verdict"], "hub_error")
        self.assertEqual(report["http_status"], 429)

    @parameterized.named_parameters(
        ("httpx_connect", httpx.ConnectError("Name or service not known")),
        ("httpx_timeout", httpx.ReadTimeout("timed out")),
        ("socket", ConnectionRefusedError("refused")),
        ("offline_mode",
         huggingface_hub.utils.OfflineModeIsEnabled("HF_HUB_OFFLINE=1")),
    )
    def test_unreachable_hub_is_offline(self, error: BaseException):
        with mock.patch.object(huggingface_hub, "HfApi") as api_class:
            api_class.return_value.list_models.side_effect = error
            report = preflight_check.resolve_report("gpt oss 20b")
        self.assertEqual(report["verdict"], "offline")
        self.assertIn("unreachable", report["reason"])

    @parameterized.named_parameters(("httpx", "httpx"), ("httpx2", "httpx2"))
    def test_each_http_client_transport_error_is_offline(self, client: str):
        # huggingface_hub 1.x raises httpx's transport errors and 2.x httpx2's;
        # neither is an OSError, and only the installed one can be imported.

        class TransportError(Exception):
            pass

        with mock.patch.dict(
                sys.modules,
            {
                "httpx": None,
                "httpx2": None,
                client: types.SimpleNamespace(TransportError=TransportError),
            }), mock.patch.object(huggingface_hub, "HfApi") as api_class:
            api_class.return_value.list_models.side_effect = TransportError(
                "connection refused")
            report = preflight_check.resolve_report("gpt oss 20b")
        self.assertEqual(report["verdict"], "offline")
        self.assertNotIn("http_status", report)

    def test_main_resolve_prints_json_and_runs_no_check(self):
        orig = FLAGS.resolve
        FLAGS.resolve = "gpt oss 20b"
        out = io.StringIO()
        try:
            with mock.patch.object(huggingface_hub, "HfApi") as api_class, \
                    mock.patch.object(preflight_check,
                                      "check_access") as check_access, \
                    redirect_stdout(out):
                api_class.return_value.list_models.side_effect = (
                    lambda **kwargs: iter(
                        GPT_OSS_SEARCH.get(kwargs["search"], [])))
                preflight_check.main(["preflight_check"])
        finally:
            FLAGS.resolve = orig
        check_access.assert_not_called()
        report = json.loads(out.getvalue())
        self.assertEqual(report["verdict"], "single_match")
        self.assertEqual(report["repo_id"], "openai/gpt-oss-20b")

    def test_discover_playbook_mappings_finds_the_shipped_pages(self):
        """Wherever it is run from, the script finds the pages of the package
        it sits in."""
        with tempfile.TemporaryDirectory() as cwd, \
                mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_KNOWLEDGE_BASE_HOME": ""}), \
                mock.patch.object(preflight_check.Path, "cwd",
                                  return_value=preflight_check.Path(cwd)):
            mappings = preflight_check.discover_playbook_mappings()
        self.assertTrue(mappings["flash_attn"].endswith(
            "tpu-knowledge-base/pages/flash-attention-to-sdpa.md"))


GCS_MODEL = "gs://bucket/models/tiny"


class GcsCheckpointTest(parameterized.TestCase):
    """A gs:// checkpoint folder is checked without the Hub."""

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.saved = {
            name: getattr(FLAGS, name)
            for name in ("model", "hub_only", "force", "resolve",
                         "allow_gpu_imports")
        }
        FLAGS.force = False
        self.listing = {
            "config.json": 10,
            "model.safetensors": 1000,
            "modeling_tiny.py": 10,
        }
        self.copies = []

        def fetch_gcs(url, small_only=False):
            self.copies.append((url, small_only))
            with open(os.path.join(self.tmp, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump({"model_type": "llama"}, f)
            with open(os.path.join(self.tmp, "modeling_tiny.py"),
                      "w",
                      encoding="utf-8") as f:
                f.write("import flash_attn\n")
            return self.tmp

        self.fetch_weights = types.ModuleType("fetch_weights")
        self.fetch_weights.gcs_listing = lambda url: dict(self.listing)
        self.fetch_weights.fetch_gcs = fetch_gcs

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(FLAGS, name, value)
        super().tearDown()

    def run_main(self, **flag_values):
        for name, value in flag_values.items():
            setattr(FLAGS, name, value)
        out = io.StringIO()
        hub_asked = mock.Mock(side_effect=AssertionError("the Hub was asked"))
        with mock.patch.dict(sys.modules,
                             {"fetch_weights": self.fetch_weights}), \
                mock.patch.object(preflight_check, "check_access",
                                  hub_asked), \
                mock.patch.object(huggingface_hub, "snapshot_download",
                                  hub_asked), \
                redirect_stdout(out):
            preflight_check.main(["preflight_check"])
        return out.getvalue()

    def test_hub_only_checks_the_folder_without_the_hub(self):
        out = self.run_main(model=GCS_MODEL,
                            hub_only=True,
                            allow_gpu_imports=True)
        self.assertIn("the Hub is not asked", out)
        self.assertIn("Format Check: Format is likely compatible", out)
        # Only the small files were copied, and the custom code scanned.
        self.assertEqual(self.copies, [(GCS_MODEL, True)])
        self.assertIn("modeling_tiny.py", out)
        self.assertIn("flash_attn", out)
        self.assertIn("PRE-FLIGHT HUB CHECKS PASSED", out)

    def test_a_folder_with_only_gguf_fails_before_copying(self):
        self.listing = {"model.gguf": 1000, "config.json": 10}
        with mock.patch.object(sys, "stderr", new=io.StringIO()) as err, \
                self.assertRaises(SystemExit) as cm:
            self.run_main(model=GCS_MODEL, hub_only=True)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Incompatible format: GGUF", err.getvalue())
        self.assertEqual(self.copies, [])

    def test_no_permission_fails_with_the_reason(self):

        def denied(url):
            raise PermissionError(13, f"no permission to read {url}")

        self.fetch_weights.gcs_listing = denied
        with mock.patch.object(sys, "stderr", new=io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            self.run_main(model=GCS_MODEL, hub_only=True)
        self.assertIn(f"Could not read {GCS_MODEL}", err.getvalue())
        self.assertIn("no permission", err.getvalue())

    def test_without_fetch_weights_the_job_says_what_to_pass(self):
        self.fetch_weights = None  # import fails, as in a job without it
        with mock.patch.object(sys, "stderr", new=io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            self.run_main(model=GCS_MODEL, hub_only=True)
        self.assertIn("fetch_weights.py", err.getvalue())

    @mock.patch.object(preflight_check, "check_vocab_bounds")
    @mock.patch.object(preflight_check, "validate_processor")
    @mock.patch.object(preflight_check, "validate_config")
    def test_the_full_check_runs_on_the_copy(self, validate_config,
                                             validate_processor,
                                             check_vocab_bounds):
        validate_config.return_value = (True, "Config OK", {})
        validate_processor.return_value = (True, "Processor OK")
        check_vocab_bounds.return_value = (True, "Vocab OK")
        out = self.run_main(model=GCS_MODEL,
                            hub_only=False,
                            allow_gpu_imports=True)
        self.assertEqual(validate_config.call_args.args[0], self.tmp)
        self.assertEqual(validate_processor.call_args.args[0], self.tmp)
        self.assertIn("PRE-FLIGHT CHECK PASSED", out)

    def test_resolve_takes_the_folder_as_given(self):
        with mock.patch.object(preflight_check,
                               "resolve_report",
                               side_effect=AssertionError("searched")):
            out = self.run_main(resolve=GCS_MODEL)
        report = json.loads(out)
        self.assertEqual(report["verdict"], "exact")
        self.assertEqual(report["repo_id"], GCS_MODEL)

    def test_listed_formats(self):
        self.assertTrue(
            preflight_check.check_listed_formats(
                ["sub/model.safetensors", "model.gguf"])[0])
        self.assertFalse(
            preflight_check.check_listed_formats(["a/model.tflite"])[0])


if __name__ == "__main__":
    absltest.main()
