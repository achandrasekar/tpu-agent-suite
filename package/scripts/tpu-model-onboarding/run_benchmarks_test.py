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

import json
import os
import tempfile
from unittest import mock

import run_benchmarks
from absl.testing import absltest


class RunBenchmarksTest(absltest.TestCase):

    def test_get_models_by_device_benchmark(self):
        models = [
            {
                "model_id": "model_1",
                "device": "cpu"
            },
            {
                "model_id": "model_2",
                "device": "tpu",
                "library": "diffusers",
                "clean_cache": True,
                "compile": True,
                "pipeline_tag": "text-to-image"
            },
            {
                "device": "cpu"
            },  # Missing model_id
            {
                "model_id": "model_3",
                "device": "cuda"
            },
        ]
        results_dict = {
            ("model_1", "cpu", "transformers", False, False): {
                "status": "success"
            },
            ("model_3", "cuda", "transformers", False, False): {
                "status": "error"
            },
        }

        # Case 1: retry_failed = False
        models_to_run = run_benchmarks.get_models_by_device_benchmark(
            models, results_dict, retry_failed=False)

        # model_1 is skipped because it has status "success"
        # model_3 is skipped because it has status "error" and retry_failed is False
        # Missing model_id is skipped
        # Only model_2 should be in "tpu"
        self.assertIn("tpu", models_to_run)
        self.assertNotIn("cpu", models_to_run)
        self.assertNotIn("cuda", models_to_run)
        self.assertLen(models_to_run["tpu"], 1)

        item = models_to_run["tpu"][0]
        self.assertEqual(item["idx"], 1)
        self.assertEqual(item["model_id"], "model_2")
        self.assertEqual(item["library"], "diffusers")
        self.assertEqual(item["device"], "tpu")
        self.assertTrue(item["clean_cache"])
        self.assertTrue(item["compile"])
        self.assertEqual(item["pipeline_tag"], "text-to-image")

        # Case 2: retry_failed = True
        models_to_run_retry = run_benchmarks.get_models_by_device_benchmark(
            models, results_dict, retry_failed=True)
        # Now model_3 (cuda) is not skipped
        self.assertIn("tpu", models_to_run_retry)
        self.assertIn("cuda", models_to_run_retry)
        self.assertLen(models_to_run_retry["tpu"], 1)
        self.assertLen(models_to_run_retry["cuda"], 1)

        item_cuda = models_to_run_retry["cuda"][0]
        self.assertEqual(item_cuda["model_id"], "model_3")
        self.assertEqual(item_cuda["device"], "cuda")

    @mock.patch("huggingface_hub.snapshot_download")
    def test_prefetch_model_success(self, mock_snapshot_download):
        run_benchmarks.prefetch_model("test_model")
        mock_snapshot_download.assert_called_once_with(
            repo_id="test_model",
            cache_dir=mock.ANY,
            ignore_patterns=["*.msgpack", "*.h5", "*.ckpt", "*.ot"],
        )

    @mock.patch("subprocess.run")
    def test_run_benchmark_subprocess_success(self, mock_run):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # We intercept the subprocess command to write the expected JSON output file
        def side_effect(*args, **_):
            cmd = args[0]
            output_file_index = cmd.index("--output_file") + 1
            output_file = cmd[output_file_index]
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "duration": 4.5,
                        "output_info": "Test output",
                        "mse": 0.001,
                        "compile_time": 1.2,
                        "ppl_stats": {
                            "perplexity": 12.3
                        },
                    },
                    f,
                )
            return mock_result

        mock_run.side_effect = side_effect

        duration, output_info, mse, compile_time, ppl_stats = (
            run_benchmarks.run_benchmark_subprocess(
                model_id="test_model",
                library="transformers",
                pipeline_tag="text-generation",
                device_str="cpu",
                clean_cache=False,
                verify=True,
                ref_dir="/tmp/refs",
                compile_flag=True,
            ))

        self.assertEqual(duration, 4.5)
        self.assertEqual(output_info, "Test output")
        self.assertEqual(mse, 0.001)
        self.assertEqual(compile_time, 1.2)
        self.assertEqual(ppl_stats, {"perplexity": 12.3})

    @mock.patch("subprocess.run")
    def test_run_benchmark_subprocess_failure(self, mock_run):
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Some error"
        mock_result.stdout = "Some output"
        mock_run.return_value = mock_result

        with self.assertRaisesRegex(RuntimeError, "Subprocess failed"):
            run_benchmarks.run_benchmark_subprocess(
                model_id="test_model",
                library="transformers",
                pipeline_tag="text-generation",
                device_str="cpu",
                clean_cache=False,
                verify=True,
                ref_dir="/tmp/refs",
                compile_flag=False,
            )

    @mock.patch("pathlib.Path")
    def test_detect_tpu_chips_success(self, mock_path):
        mock_pci_devices = mock.MagicMock()
        mock_path.return_value = mock_pci_devices
        mock_pci_devices.exists.return_value = True

        def create_mock_dev(name, vendor, device):
            m = mock.MagicMock()
            m.name = name

            def get_file(part):
                f = mock.MagicMock()
                if part == "vendor":
                    f.read_text.return_value = vendor
                elif part == "device":
                    f.read_text.return_value = device
                return f

            m.__truediv__.side_effect = get_file
            return m

        dev1 = create_mock_dev("0000:01:00.0", "0x1ae0", "0x0076")
        dev2 = create_mock_dev("0000:01:00.1", "0x1ae0", "0x0076")
        dev3 = create_mock_dev("0000:02:00.0", "0x1ae0", "0x0076")
        dev4 = create_mock_dev("0000:03:00.0", "0x8086", "0x0076")
        dev5 = create_mock_dev("0000:04:00.0", "0x1ae0", "0x0077")

        mock_pci_devices.iterdir.return_value = [dev1, dev2, dev3, dev4, dev5]

        chips = run_benchmarks.detect_tpu_chips()
        self.assertEqual(chips, [0, 1])

    @mock.patch("pathlib.Path")
    def test_detect_tpu_chips_fallback(self, mock_path):
        mock_pci_devices = mock.MagicMock()
        mock_path.return_value = mock_pci_devices
        mock_pci_devices.exists.return_value = False

        chips = run_benchmarks.detect_tpu_chips()
        self.assertEqual(chips, [0])

    @mock.patch("pathlib.Path")
    def test_get_tpu_chip_vfio_path(self, mock_path):
        mock_pci_devices = mock.MagicMock()
        mock_path.return_value = mock_pci_devices
        mock_pci_devices.exists.return_value = True

        def create_mock_dev(name, vendor, device, iommu_group_name):
            m = mock.MagicMock()
            m.name = name

            def get_file(part):
                f = mock.MagicMock()
                if part == "vendor":
                    f.read_text.return_value = vendor
                elif part == "device":
                    f.read_text.return_value = device
                elif part == "iommu_group":
                    f.readlink.return_value.name = iommu_group_name
                return f

            m.__truediv__.side_effect = get_file
            return m

        dev1 = create_mock_dev("0000:01:00.0", "0x1ae0", "0x0076", "vfio1")
        dev2 = create_mock_dev("0000:01:00.1", "0x1ae0", "0x0076", "vfio2")
        dev3 = create_mock_dev("0000:02:00.0", "0x1ae0", "0x0076", "vfio3")

        mock_pci_devices.iterdir.return_value = [dev3, dev1, dev2]

        path0 = run_benchmarks.get_tpu_chip_vfio_path(0)
        self.assertEqual(path0, "/dev/vfio/vfio1")

        path1 = run_benchmarks.get_tpu_chip_vfio_path(1)
        self.assertEqual(path1, "/dev/vfio/vfio3")

        path2 = run_benchmarks.get_tpu_chip_vfio_path(2)
        self.assertIsNone(path2)

    def test_tpu_device_allocator(self):
        allocator = run_benchmarks.TPUDeviceAllocator([0, 1, 2])
        self.assertEqual(allocator.allocate(), 0)
        self.assertEqual(allocator.allocate(), 1)
        allocator.free(0)
        self.assertEqual(allocator.allocate(), 2)
        self.assertEqual(allocator.allocate(), 0)
        self.assertIsNone(allocator.allocate())

    @mock.patch.object(run_benchmarks, "get_tpu_chip_vfio_path")
    @mock.patch("subprocess.run")
    def test_run_benchmark_subprocess_tpu(self, mock_run, mock_get_vfio):
        mock_get_vfio.return_value = "/dev/vfio/42"
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        def side_effect(*args, **_):
            cmd = args[0]
            output_file_index = cmd.index("--output_file") + 1
            output_file = cmd[output_file_index]
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "duration": 1.0,
                        "output_info": "TPU OK",
                        "mse": 0.0,
                        "compile_time": 0.5,
                    },
                    f,
                )
            return mock_result

        mock_run.side_effect = side_effect

        run_benchmarks.run_benchmark_subprocess(model_id="test_model",
                                                library="transformers",
                                                pipeline_tag="text-generation",
                                                device_str="tpu",
                                                clean_cache=False,
                                                verify=True,
                                                ref_dir="/tmp/refs",
                                                compile_flag=False,
                                                chip_idx=1)

        mock_get_vfio.assert_called_once_with(1)
        _, kwargs = mock_run.call_args
        env = kwargs["env"]
        self.assertEqual(env["TPU_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["TPU_VISIBLE_DEVICE_PATHS"], "/dev/vfio/42")
        self.assertEqual(env["ALLOW_MULTIPLE_LIBTPU_LOAD"], "1")
        self.assertEqual(env["LIBTPU_INIT_ARGS"],
                         "--xla_tpu_enable_diagnostic_coordinator=false")
        self.assertEqual(env["TORCH_TPU_TIER2_COMPILATION_CACHE"], "disabled")

    def test_get_cache_dir_uses_hf_hub_cache_when_set(self):
        # An explicit cache_dir overrides HF_HUB_CACHE, so a hard-coded
        # private directory would make every pod download the model again.
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "/hf-cache"}):
            self.assertEqual(run_benchmarks.get_cache_dir(), "/hf-cache")
            self.assertTrue(run_benchmarks.cache_is_shared())
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(run_benchmarks.get_cache_dir().endswith(
                ".cache/torch_tpu_cache"))
            self.assertFalse(run_benchmarks.cache_is_shared())

    def test_the_pod_cache_gke_job_names_is_not_shared(self):
        # Without hf_cache_pvc, gke_job points HF_HUB_CACHE at the pod's own
        # /dev/shm; left unpruned, a sweep fills it model by model.
        pod = "/dev/shm/cache/huggingface/hub"
        with mock.patch.dict(os.environ, {
                "HF_HUB_CACHE": pod,
                "TPU_AGENT_SUITE_POD_HF_CACHE": pod
        }):
            self.assertEqual(run_benchmarks.get_cache_dir(), pod)
            self.assertFalse(run_benchmarks.cache_is_shared())
        # A recipe that points it elsewhere has made it its own.
        with mock.patch.dict(os.environ, {
                "HF_HUB_CACHE": "/data/hub",
                "TPU_AGENT_SUITE_POD_HF_CACHE": pod
        }):
            self.assertTrue(run_benchmarks.cache_is_shared())

    def test_cleanup_never_prunes_a_shared_cache(self):
        # Other Jobs load from a shared cache: deleting a model there would
        # break them mid-load.
        with tempfile.TemporaryDirectory() as cache:
            model_dir = os.path.join(cache, "models--org--m")
            os.makedirs(model_dir)
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": cache}):
                run_benchmarks.cleanup_model_cache("org/m")
            self.assertTrue(os.path.isdir(model_dir))

    def test_cleanup_prunes_the_private_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            model_dir = os.path.join(cache, "models--org--m")
            os.makedirs(model_dir)
            env = {k: v for k, v in os.environ.items() if k != "HF_HUB_CACHE"}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(run_benchmarks, "get_cache_dir",
                                      return_value=cache):
                run_benchmarks.cleanup_model_cache("org/m")
            self.assertFalse(os.path.exists(model_dir))


if __name__ == "__main__":
    absltest.main()
