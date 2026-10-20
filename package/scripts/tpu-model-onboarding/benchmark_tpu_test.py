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

import builtins
import json
import os
import sys
import tempfile
from unittest import mock

import benchmark_tpu
import huggingface_hub
import torch
import torch.nn.functional as F
import transformers
from absl.testing import absltest, parameterized
from torch import nn  # pylint: disable=unused-import
from torch.nn.utils import weight_norm


class BenchmarkTpuTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        tempfile.tempdir = self.create_tempdir().full_path

    # ============================================================================
    # Transformers Patch (GETTER/SETTER for all_tied_weights_keys)
    # ============================================================================

    def test_transformers_patch_empty(self):
        mock_instance = mock.MagicMock(spec=transformers.PreTrainedModel)
        if hasattr(mock_instance, "_all_tied_weights_keys_storage"):
            delattr(mock_instance, "_all_tied_weights_keys_storage")
        if hasattr(mock_instance, "_tied_weights_keys"):
            delattr(mock_instance, "_tied_weights_keys")

        benchmark_tpu.apply_global_patches()

        val = transformers.PreTrainedModel.all_tied_weights_keys.fget(
            mock_instance)  # pytype: disable=attribute-error
        self.assertEqual(val, {})

    def test_transformers_patch_set_value(self):
        mock_instance = mock.MagicMock(spec=transformers.PreTrainedModel)
        benchmark_tpu.apply_global_patches()

        transformers.PreTrainedModel.all_tied_weights_keys.fset(  # pytype: disable=attribute-error
            mock_instance, {"key1"})
        self.assertEqual(mock_instance._all_tied_weights_keys_storage,
                         {"key1"})

        val = transformers.PreTrainedModel.all_tied_weights_keys.fget(
            mock_instance)  # pytype: disable=attribute-error
        self.assertEqual(val, {"key1"})

    def test_transformers_patch_fallback_to_tied_weights_keys(self):
        mock_instance = mock.MagicMock(spec=transformers.PreTrainedModel)
        benchmark_tpu.apply_global_patches()
        if hasattr(mock_instance, "_all_tied_weights_keys_storage"):
            delattr(mock_instance, "_all_tied_weights_keys_storage")

        mock_instance._tied_weights_keys = {"fallback_key"}
        val = transformers.PreTrainedModel.all_tied_weights_keys.fget(
            mock_instance)  # pytype: disable=attribute-error
        self.assertEqual(val, {"fallback_key"})

    # ============================================================================
    # parse_args()
    # ============================================================================

    def test_parse_args_valid_inputs(self):
        with mock.patch.object(
                sys,
                "argv",
            [
                "benchmark_tpu.py",
                "--model",
                "llama",
                "--device",
                "tpu",
                "--library",
                "diffusers",
            ],
        ):
            args = benchmark_tpu.parse_args()
            self.assertEqual(args.model, "llama")
            self.assertEqual(args.device, "tpu")
            self.assertEqual(args.library, "diffusers")

    def test_parse_args_default_library(self):
        with mock.patch.object(
                sys, "argv",
            ["benchmark_tpu.py", "--model", "llama", "--device", "cpu"]):
            args = benchmark_tpu.parse_args()
            self.assertEqual(args.model, "llama")
            self.assertEqual(args.device, "cpu")
            self.assertEqual(args.library, "transformers")
            self.assertFalse(args.no_verify)
            self.assertEqual(args.ref_dir, ".benchmark_refs")

    def test_parse_args_no_verify(self):
        with mock.patch.object(
                sys,
                "argv",
            [
                "benchmark_tpu.py",
                "--model",
                "llama",
                "--device",
                "cpu",
                "--no-verify",
            ],
        ):
            args = benchmark_tpu.parse_args()
            self.assertTrue(args.no_verify)

    def test_parse_args_missing_required_argument(self):
        with mock.patch.object(sys, "argv",
                               ["benchmark_tpu.py", "--model", "llama"]):
            with self.assertRaises(SystemExit):
                benchmark_tpu.parse_args()

    # ============================================================================
    # patch_interpolate()
    # ============================================================================

    def test_patch_interpolate_bicubic(self):
        benchmark_tpu.patch_interpolate()

        x = torch.zeros(1, 1, 4, 4)

        with mock.patch.object(builtins, "print") as mock_print:
            out = F.interpolate(x,
                                size=(8, 8),
                                mode="bicubic",
                                align_corners=False)
            self.assertEqual(out.shape, (1, 1, 8, 8))
            mock_print.assert_any_call(
                "INFO: Replacing upsample_bicubic2d with bilinear for TPU compat."
            )

    def test_patch_interpolate_bilinear(self):
        benchmark_tpu.patch_interpolate()

        x = torch.zeros(1, 1, 4, 4)

        with mock.patch.object(builtins, "print") as mock_print:
            out = F.interpolate(x,
                                size=(8, 8),
                                mode="bilinear",
                                align_corners=False)
            self.assertEqual(out.shape, (1, 1, 8, 8))
            for (args, *_), _ in mock_print.call_args_list:
                self.assertNotIn("Replacing upsample_bicubic2d with bilinear",
                                 args[0])

    def test_patch_interpolate_nearest(self):
        benchmark_tpu.patch_interpolate()

        x = torch.zeros(1, 1, 4, 4)

        with mock.patch.object(builtins, "print") as mock_print:
            out = F.interpolate(x, size=(8, 8), mode="nearest")
            self.assertEqual(out.shape, (1, 1, 8, 8))
            for (args, *_), _ in mock_print.call_args_list:
                self.assertNotIn("Replacing upsample_bicubic2d with bilinear",
                                 args[0])

    # ============================================================================
    # remove_weight_norm_recursive(module)
    # ============================================================================

    def test_remove_weight_norm_recursive_applied(self):
        linear = nn.Linear(10, 10)
        wn_linear = weight_norm(linear)
        self.assertTrue(
            any(
                name.startswith("weight_g")
                for name, _ in wn_linear.named_parameters()))

        benchmark_tpu.remove_weight_norm_recursive(wn_linear)
        self.assertFalse(
            any(
                name.startswith("weight_g")
                for name, _ in wn_linear.named_parameters()))

    def test_remove_weight_norm_recursive_no_hook(self):
        normal_layer = nn.Linear(5, 5)
        benchmark_tpu.remove_weight_norm_recursive(normal_layer)
        self.assertFalse(
            any(
                name.startswith("weight_g")
                for name, _ in normal_layer.named_parameters()))

    def test_remove_weight_norm_recursive_nested(self):
        parent = nn.Sequential(nn.Linear(5, 5),
                               nn.utils.weight_norm(nn.Linear(5, 5)))
        self.assertTrue(
            any(
                name.startswith("1.weight_g")
                for name, _ in parent.named_parameters()))

        benchmark_tpu.remove_weight_norm_recursive(parent)
        self.assertFalse(
            any(
                name.startswith("1.weight_g")
                for name, _ in parent.named_parameters()))

    # ============================================================================
    # monkey_patch_group_norm()
    # ============================================================================

    def test_monkey_patch_group_norm_tpu_non_contiguous(self):
        old_forward = nn.GroupNorm.forward
        mock_original = mock.create_autospec(nn.GroupNorm.forward)
        nn.GroupNorm.forward = mock_original

        try:
            benchmark_tpu.monkey_patch_group_norm()
            mock_input = mock.create_autospec(torch.Tensor, instance=True)
            mock_input.device.type = "tpu"
            mock_input.is_contiguous.return_value = False
            mock_input.contiguous.return_value = "contiguous_input"

            nn.GroupNorm.forward(None, mock_input)
            mock_input.contiguous.assert_called_once()
            mock_original.assert_called_once_with(None, "contiguous_input")
        finally:
            nn.GroupNorm.forward = old_forward

    def test_monkey_patch_group_norm_tpu_contiguous(self):
        old_forward = nn.GroupNorm.forward
        mock_original = mock.create_autospec(nn.GroupNorm.forward)
        nn.GroupNorm.forward = mock_original

        try:
            benchmark_tpu.monkey_patch_group_norm()
            mock_input = mock.create_autospec(torch.Tensor, instance=True)
            mock_input.device.type = "tpu"
            mock_input.is_contiguous.return_value = True

            nn.GroupNorm.forward(None, mock_input)
            mock_input.contiguous.assert_not_called()
            mock_original.assert_called_once_with(None, mock_input)
        finally:
            nn.GroupNorm.forward = old_forward

    def test_monkey_patch_group_norm_cpu_non_contiguous(self):
        old_forward = nn.GroupNorm.forward
        mock_original = mock.create_autospec(nn.GroupNorm.forward)
        nn.GroupNorm.forward = mock_original

        try:
            benchmark_tpu.monkey_patch_group_norm()
            mock_input = mock.create_autospec(torch.Tensor, instance=True)
            mock_input.device.type = "cpu"
            mock_input.is_contiguous.return_value = False

            nn.GroupNorm.forward(None, mock_input)
            mock_input.contiguous.assert_not_called()
            mock_original.assert_called_once_with(None, mock_input)
        finally:
            nn.GroupNorm.forward = old_forward

    # ============================================================================
    # prepare_patched_metadata(model_id)
    # ============================================================================

    @mock.patch.object(huggingface_hub, "hf_hub_download")
    def test_prepare_patched_metadata_multi_modality_patched(
            self, mock_download):
        temp_dir = self.create_tempdir()
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": "multi_modality"}, f)

        mock_download.return_value = config_path
        patched_dir = benchmark_tpu.prepare_patched_metadata("test-model")
        self.assertIsNotNone(patched_dir)

        with open(os.path.join(patched_dir, "config.json"), "r") as f:
            data = json.load(f)
            self.assertEqual(data["model_type"], "janus")

    @mock.patch.object(huggingface_hub, "hf_hub_download")
    def test_prepare_patched_metadata_llama_no_patch(self, mock_download):
        temp_dir = self.create_tempdir()
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": "llama"}, f)

        mock_download.return_value = config_path
        patched_dir = benchmark_tpu.prepare_patched_metadata("test-model")
        self.assertIsNone(patched_dir)

    @mock.patch.object(huggingface_hub, "hf_hub_download")
    def test_prepare_patched_metadata_download_exception_returns_none(
            self, mock_download):
        mock_download.side_effect = Exception("Network failure")
        patched_dir = benchmark_tpu.prepare_patched_metadata("test-model")
        self.assertIsNone(patched_dir)

    # ============================================================================
    # load_model_and_processor()
    # ============================================================================

    @mock.patch.object(benchmark_tpu, "prepare_patched_metadata")
    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_load_model_and_processor_with_both(self, unused_mock_config,
                                                mock_metadata):
        mock_metadata.return_value = "/dummy/temp/dir"
        mock_model_factory = mock.create_autospec(transformers.AutoModel,
                                                  instance=False)
        mock_processor_factory = mock.create_autospec(
            transformers.AutoProcessor, instance=False)

        benchmark_tpu.load_model_and_processor(
            "test-model",
            "cpu",
            mock_model_factory,
            mock_processor_factory,
            use_bfloat16=True,
        )
        mock_model_factory.from_pretrained.assert_called_once()
        mock_processor_factory.from_pretrained.assert_called_once()

    @mock.patch.object(benchmark_tpu, "prepare_patched_metadata")
    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_load_model_and_processor_model_only(self, unused_mock_config,
                                                 mock_metadata):
        mock_metadata.return_value = None
        mock_model_factory = mock.create_autospec(transformers.AutoModel,
                                                  instance=False)

        benchmark_tpu.load_model_and_processor("test-model",
                                               "cpu",
                                               mock_model_factory,
                                               None,
                                               use_bfloat16=False)
        mock_model_factory.from_pretrained.assert_called_once()

    @mock.patch.object(benchmark_tpu, "prepare_patched_metadata")
    @mock.patch.object(transformers.AutoConfig, "from_pretrained")
    def test_load_model_and_processor_gemma_disables_low_cpu_mem_usage(
            self, unused_mock_config, mock_metadata):
        mock_metadata.return_value = "/dummy/temp/dir"
        mock_model_factory = mock.create_autospec(transformers.AutoModel,
                                                  instance=False)

        benchmark_tpu.load_model_and_processor("gemma-model",
                                               "cpu",
                                               mock_model_factory,
                                               None,
                                               use_bfloat16=False)
        _, loaded_kwargs = mock_model_factory.from_pretrained.call_args
        self.assertFalse(loaded_kwargs.get("low_cpu_mem_usage"))

    # ============================================================================
    # cast_inputs_to_model_dtype(inputs, model, device)
    # ============================================================================

    def test_cast_inputs_to_model_dtype_dict_mixed(self):
        model = nn.Linear(10, 10)
        device = torch.device("cpu")
        inputs = {
            "input_ids": torch.ones(5).long(),
            "pixel_values": torch.ones(5).float(),
        }
        casted = benchmark_tpu.cast_inputs_to_model_dtype(
            inputs, model, device)
        self.assertEqual(casted["input_ids"].dtype, torch.int64)
        self.assertEqual(casted["pixel_values"].dtype, torch.float32)

    def test_cast_inputs_to_model_dtype_single_float_tensor(self):
        model = nn.Linear(5, 5)
        device = torch.device("cpu")
        float_tensor = torch.randn(5)
        casted = benchmark_tpu.cast_inputs_to_model_dtype(
            float_tensor, model, device)
        self.assertEqual(casted.dtype, torch.float32)

    def test_cast_inputs_to_model_dtype_single_int_tensor(self):
        model = nn.Linear(5, 5)
        device = torch.device("cpu")
        int_tensor = torch.ones(5).long()
        casted = benchmark_tpu.cast_inputs_to_model_dtype(
            int_tensor, model, device)
        self.assertEqual(casted.dtype, torch.int64)

    # ============================================================================
    # monkey_patch_compile()
    # ============================================================================

    def test_monkey_patch_compile_replaces(self):
        original = torch.compile
        try:
            benchmark_tpu.monkey_patch_compile()
            self.assertNotEqual(torch.compile, original)
        finally:
            torch.compile = original

    def test_monkey_patch_compile_forces_static(self):
        original = torch.compile
        mock_original = mock.create_autospec(torch.compile)
        torch.compile = mock_original
        try:
            benchmark_tpu.monkey_patch_compile()
            torch.compile("sub_func", dynamic=True)
            mock_original.assert_called_once_with("sub_func", dynamic=False)
        finally:
            torch.compile = original

    def test_monkey_patch_compile_preserves_other_kwargs(self):
        original = torch.compile
        mock_original = mock.create_autospec(torch.compile)
        torch.compile = mock_original
        try:
            benchmark_tpu.monkey_patch_compile()
            torch.compile("sub_func", mode="reduce-overhead", dynamic=True)
            mock_original.assert_called_once_with("sub_func",
                                                  mode="reduce-overhead",
                                                  dynamic=False)
        finally:
            torch.compile = original

    # ============================================================================
    # get_device(device_str)
    # ============================================================================

    def test_get_device_cpu(self):
        dev = benchmark_tpu.get_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_get_device_cuda(self):
        dev = benchmark_tpu.get_device("cuda")
        self.assertEqual(dev.type, "cuda")

    def test_get_device_tpu(self):
        mock_api = mock.MagicMock()
        mock_api.tpu_device.return_value = "mock_tpu_device"
        mock_torch_tpu = mock.MagicMock()
        mock_torch_tpu.api = mock_api
        with mock.patch.dict("sys.modules", {"torch_tpu": mock_torch_tpu}):
            dev = benchmark_tpu.get_device("tpu")
            self.assertEqual(dev, "mock_tpu_device")

    # ============================================================================
    # verify_inputs_bounds(model, inputs)
    # ============================================================================

    def test_verify_inputs_bounds_in_bounds(self):
        mock_model = mock.create_autospec(transformers.PreTrainedModel,
                                          instance=True)
        mock_embeddings = mock.Mock()
        mock_embeddings.weight = mock.Mock()
        mock_embeddings.weight.shape = [1000]
        mock_model.get_input_embeddings.return_value = mock_embeddings

        inputs = {"input_ids": torch.tensor([500])}

        with mock.patch.object(builtins, "print") as mock_print:
            benchmark_tpu.verify_inputs_bounds(mock_model, inputs)
            mock_print.assert_not_called()

    def test_verify_inputs_bounds_out_of_bounds(self):
        mock_model = mock.create_autospec(transformers.PreTrainedModel,
                                          instance=True)
        mock_embeddings = mock.Mock()
        mock_embeddings.weight = mock.Mock()
        mock_embeddings.weight.shape = [1000]
        mock_model.get_input_embeddings.return_value = mock_embeddings

        inputs = {"input_ids": torch.tensor([1200])}

        with mock.patch.object(builtins, "print") as mock_print:
            benchmark_tpu.verify_inputs_bounds(mock_model, inputs)
            mock_print.assert_any_call(
                "WARNING: Out-of-bounds token ID detected "
                "(1200 >= embedding vocab 1000). Execution will crash on CPU.")

    def test_verify_inputs_bounds_no_vocab_graceful(self):
        mock_model = mock.create_autospec(transformers.PreTrainedModel,
                                          instance=True)
        mock_model.get_input_embeddings.side_effect = Exception(
            "No Embeddings")

        inputs = {"input_ids": torch.tensor([1200])}
        with mock.patch.object(builtins, "print") as mock_print:
            benchmark_tpu.verify_inputs_bounds(mock_model, inputs)
            mock_print.assert_any_call(
                "Warning: Failed to verify vocab bounds: Exception('No Embeddings')"
            )

    # ============================================================================
    # import_dependency(module_name, pip_name)
    # ============================================================================

    def test_import_dependency_success(self):
        mod = benchmark_tpu.import_dependency("json")
        self.assertIs(mod, json)

    def test_import_dependency_missing_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            benchmark_tpu.import_dependency("non_existent_module_xyz",
                                            "non-existent-pkg")
        self.assertIn("non-existent-pkg", str(ctx.exception))
        self.assertIn("requirements-tpu.txt", str(ctx.exception))

    # ============================================================================
    # repair_json_string(json_str)
    # ============================================================================

    def test_repair_json_string_missing_comma_quotes(self):
        bad_json = '{"a": "b" "c": "d"}'
        repaired = benchmark_tpu.repair_json_string(bad_json)
        self.assertEqual(json.loads(repaired), {"a": "b", "c": "d"})

    def test_repair_json_string_missing_comma_before_colon(self):
        bad_json = '{"a": "b" "c" : "d"}'
        repaired = benchmark_tpu.repair_json_string(bad_json)
        self.assertEqual(json.loads(repaired), {"a": "b", "c": "d"})

    def test_repair_json_string_valid_remains_valid(self):
        good_json = '{"a": "b", "c": "d"}'
        repaired = benchmark_tpu.repair_json_string(good_json)
        self.assertEqual(json.loads(repaired), {"a": "b", "c": "d"})

    # ============================================================================
    # extract_raw_tensor() and get_ref_path()
    # ============================================================================

    def test_get_ref_path(self):
        ref_dir = self.create_tempdir().full_path
        path = benchmark_tpu.get_ref_path("google/gemma-2b", ref_dir)
        self.assertEqual(os.path.basename(path), "ref_google_gemma-2b.pt")

    def test_extract_raw_tensor_raw_tensor(self):
        t = torch.randn(5)
        extracted = benchmark_tpu.extract_raw_tensor(t)
        self.assertIs(extracted, t)

    def test_extract_raw_tensor_huggingface_output(self):
        mock_out = mock.Mock()
        mock_out.logits = torch.randn(2, 3)
        extracted = benchmark_tpu.extract_raw_tensor(mock_out)
        self.assertIs(extracted, mock_out.logits)

        mock_out_hidden = mock.Mock()
        mock_out_hidden.logits = None
        mock_out_hidden.logits_per_image = None
        mock_out_hidden.logits_per_audio = None
        mock_out_hidden.last_hidden_state = torch.randn(4, 5)
        extracted = benchmark_tpu.extract_raw_tensor(mock_out_hidden)
        self.assertIs(extracted, mock_out_hidden.last_hidden_state)

    def test_extract_raw_tensor_dict(self):
        d = {"logits": torch.randn(2, 2)}
        extracted = benchmark_tpu.extract_raw_tensor(d)
        self.assertIs(extracted, d["logits"])

    def test_extract_raw_tensor_tuple(self):
        tup = (torch.randn(3, 3), "dummy_info")
        extracted = benchmark_tpu.extract_raw_tensor(tup)
        self.assertIs(extracted, tup[0])

    # ============================================================================
    # run_benchmark_pass(model, inputs, forward_fn, postprocess_fn)
    # ============================================================================

    def test_run_benchmark_pass(self):
        mock_model = mock.MagicMock()
        mock_inputs = "dummy_inputs"
        mock_forward = mock.MagicMock(return_value="raw_outputs")
        mock_postprocess = mock.MagicMock(return_value="processed_info")

        duration, info, outputs, warmup_time = benchmark_tpu.run_benchmark_pass(
            mock_model, mock_inputs, mock_forward, mock_postprocess)

        self.assertGreaterEqual(duration, 0.0)
        self.assertGreaterEqual(warmup_time, 0.0)
        self.assertEqual(info, "processed_info")
        self.assertEqual(outputs, "raw_outputs")
        self.assertEqual(mock_forward.call_count, 2)
        mock_postprocess.assert_called_once_with("raw_outputs")

    def test_verify_output_cpu_saves_ref(self):
        mock_model = mock.MagicMock()
        mock_inputs = "dummy_inputs"
        tensor_out = torch.randn(5)
        mock_forward = mock.MagicMock(return_value=tensor_out)

        ref_dir = self.create_tempdir().full_path
        device = torch.device("cpu")

        with mock.patch.object(builtins, "print") as mock_print:
            mse = benchmark_tpu.verify_output(
                device_outputs=tensor_out,
                model=mock_model,
                inputs=mock_inputs,
                forward_fn=mock_forward,
                model_id="test-model",
                device=device,
                ref_dir=ref_dir,
            )
            self.assertIsNone(mse)
            mock_print.assert_any_call(
                "INFO: Saving CPU reference tensor to"
                f" {os.path.join(ref_dir, 'ref_test-model.pt')}...")
            expected_ref = os.path.join(ref_dir, "ref_test-model.pt")
            self.assertTrue(os.path.exists(expected_ref))

    def test_verify_output_tpu_loads_ref(self):
        # Setup: save a fake ref file
        ref_dir = self.create_tempdir().full_path
        ref_path = os.path.join(ref_dir, "ref_test-model.pt")
        ref_tensor = torch.tensor([1.0, 2.0, 3.0])
        torch.save(ref_tensor, ref_path)

        mock_model = mock.MagicMock()
        mock_inputs = "dummy_inputs"
        mock_forward = mock.MagicMock()

        device = torch.device("cuda")  # Simulate accelerator
        device_outputs = torch.tensor([1.0, 2.0, 3.0])

        with mock.patch.object(builtins, "print") as mock_print:
            mse = benchmark_tpu.verify_output(
                device_outputs=device_outputs,
                model=mock_model,
                inputs=mock_inputs,
                forward_fn=mock_forward,
                model_id="test-model",
                device=device,
                ref_dir=ref_dir,
            )
            self.assertEqual(mse, 0.0)
            mock_print.assert_any_call(
                f"INFO: Loaded cached CPU reference from {ref_path}")

    # ============================================================================
    # main() routing and error handling
    # ============================================================================

    @mock.patch.object(benchmark_tpu, "benchmark_model")
    @mock.patch.object(benchmark_tpu, "get_device")
    @mock.patch.object(benchmark_tpu, "parse_args")
    def test_main_routing_success(self, mock_args, mock_get_device,
                                  mock_benchmark_model):
        output_file = os.path.join(tempfile.gettempdir(), "test_output.json")
        mock_args.return_value = mock.Mock(
            model="some-model",
            device="cpu",
            library="transformers",
            pipeline_tag="text-generation",
            no_verify=False,
            ref_dir=".benchmark_refs",
            output_file=output_file,
            compile=False,
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_benchmark_model.return_value = (0.5, "execution log", 0.01, None)

        benchmark_tpu.main()

        with open(output_file, "r") as f:
            results = json.load(f)
        self.assertEqual(results["duration"], 0.5)
        self.assertEqual(results["mse"], 0.01)
        self.assertEqual(results["output_info"], "execution log")
        self.assertIsNone(results["compile_time"])

        mock_benchmark_model.assert_called_once_with(
            "some-model",
            torch.device("cpu"),
            verify=True,
            ref_dir=".benchmark_refs",
            library="transformers",
            pipeline_tag="text-generation",
            compile_flag=False,
        )

    @mock.patch.object(benchmark_tpu, "benchmark_model")
    @mock.patch.object(benchmark_tpu, "get_device")
    @mock.patch.object(benchmark_tpu, "parse_args")
    def test_main_benchmark_failure_exits(self, mock_args, mock_get_device,
                                          mock_benchmark_model):
        mock_args.return_value = mock.Mock(
            model="some-model",
            device="cpu",
            library="transformers",
            no_verify=True,
            ref_dir=".benchmark_refs",
            output_file=None,
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_benchmark_model.side_effect = RuntimeError("Execution crashed")

        with self.assertRaisesRegex(SystemExit, "1"):
            with mock.patch("sys.stderr", new_callable=mock.MagicMock):
                with mock.patch.object(builtins, "print"):
                    benchmark_tpu.main()

    def test_get_cache_dir_uses_hf_hub_cache_when_set(self):
        # An explicit cache_dir overrides HF_HUB_CACHE, so a hard-coded
        # private directory would make every pod download the model again.
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "/hf-cache"}):
            self.assertEqual(benchmark_tpu.get_cache_dir(), "/hf-cache")
            self.assertTrue(benchmark_tpu.cache_is_shared())
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(benchmark_tpu.get_cache_dir().endswith(
                ".cache/torch_tpu_cache"))
            self.assertFalse(benchmark_tpu.cache_is_shared())

    def test_the_pod_cache_gke_job_names_is_not_shared(self):
        # Without hf_cache_pvc, gke_job points HF_HUB_CACHE at the pod's own
        # /dev/shm; left unpruned, a sweep fills it model by model.
        pod = "/dev/shm/cache/huggingface/hub"
        with mock.patch.dict(os.environ, {
                "HF_HUB_CACHE": pod,
                "TPU_AGENT_SUITE_POD_HF_CACHE": pod
        }):
            self.assertEqual(benchmark_tpu.get_cache_dir(), pod)
            self.assertFalse(benchmark_tpu.cache_is_shared())
        # A recipe that points it elsewhere has made it its own.
        with mock.patch.dict(os.environ, {
                "HF_HUB_CACHE": "/data/hub",
                "TPU_AGENT_SUITE_POD_HF_CACHE": pod
        }):
            self.assertTrue(benchmark_tpu.cache_is_shared())

    def test_cleanup_never_prunes_a_shared_cache(self):
        # Other Jobs load from a shared cache: deleting a model there would
        # break them mid-load.
        with tempfile.TemporaryDirectory() as cache:
            model_dir = os.path.join(cache, "models--org--m")
            os.makedirs(model_dir)
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": cache}):
                benchmark_tpu.cleanup_model_cache("org/m")
            self.assertTrue(os.path.isdir(model_dir))

    def test_cleanup_prunes_the_private_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            model_dir = os.path.join(cache, "models--org--m")
            os.makedirs(model_dir)
            env = {k: v for k, v in os.environ.items() if k != "HF_HUB_CACHE"}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(benchmark_tpu, "get_cache_dir",
                                      return_value=cache):
                benchmark_tpu.cleanup_model_cache("org/m")
            self.assertFalse(os.path.exists(model_dir))


if __name__ == "__main__":
    absltest.main()
