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
"""Tests for Hugging Face search functionality, including parameter parsing, compatibility filtering, and library stats fetching."""

from __future__ import annotations

from typing import Any
from unittest import mock

import smart_hf_search
from absl.testing import absltest, parameterized


class MockSafeTensors:

    def __init__(self, total: float | None):
        self.total = total


class MockModel:

    def __init__(
        self,
        model_id: str,
        tags: list[str] | None = None,
        downloads: int = 0,
        likes: int = 0,
        library_name: str | None = None,
        safetensors: MockSafeTensors | None = None,
        gated: bool = False,
    ):
        self.id = model_id
        self.tags = tags or []
        self.downloads = downloads
        self.likes = likes
        self.library_name = library_name
        self.safetensors = safetensors
        self.gated = gated


class MockSibling:

    def __init__(self, rfilename: str, size: int):
        self.rfilename = rfilename
        self.size = size


class MockModelInfo:

    def __init__(self, siblings: list[MockSibling]):
        self.siblings = siblings


class SmartHfSearchTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ("8B", "8B", 8.0e9),
        ("350M", "350M", 350.0e6),
        ("1K", "1K", 1000.0),
        ("float_B", "1.5B", 1.5e9),
        ("raw_float", "12345.6", 12345.6),
        ("raw_int", "500", 500.0),
        ("none", None, None),
        ("invalid", "invalid_limit", None),
    )
    def test_parse_params_limit(self, limit_str: str | None,
                                expected: float | None):
        self.assertEqual(smart_hf_search.parse_params_limit(limit_str),
                         expected)

    @parameterized.named_parameters(
        ("compatible_model", MockModel("safe-model"), False),
        ("missing_model_id", MockModel(""), True),
        ("gguf_extension", MockModel("llama3-8b.gguf"), True),
        ("gguf_in_path", MockModel("meta/gguf/llama3"), True),
        ("gguf_tag", MockModel("safe-model", tags=["gguf"]), True),
        ("tflite_compatible", MockModel("model-tflite", tags=["pytorch"
                                                              ]), False),
        (
            "safetensors_compatible",
            MockModel("model-litert", tags=["safetensors"]),
            False,
        ),
        ("tflite_only_tag", MockModel("model", tags=["tflite"]), True),
        ("litert_only_tag", MockModel("model", tags=["litert"]), True),
        ("tflite_in_id", MockModel("llama-tflite-model"), True),
        ("bitsandbytes_tag", MockModel("model", tags=["bitsandbytes"]), True),
        (
            "bitsandbytes_in_tag",
            MockModel("model", tags=["any-bitsandbytes-tag"]),
            True,
        ),
    )
    def test_is_incompatible(self, model: MockModel, expected: bool):
        self.assertEqual(smart_hf_search.is_incompatible(model), expected)

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_estimate_params_from_file_size(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.return_value = MockModelInfo([
            MockSibling("model.safetensors", 2000),
            MockSibling("config.json", 500),
        ])

        params = smart_hf_search.estimate_params_from_file_size(
            "test-model", "token")
        self.assertEqual(params, 1000.0)

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_estimate_params_from_file_size_prioritization(
            self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.return_value = MockModelInfo([
            MockSibling("model.safetensors", 2000),
            MockSibling("pytorch_model.bin", 3000),
            MockSibling("config.json", 500),
        ])

        params = smart_hf_search.estimate_params_from_file_size(
            "test-model", "token")
        self.assertEqual(params, 1000.0)

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_estimate_params_from_file_size_exception(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.model_info.side_effect = Exception("Network error")

        params = smart_hf_search.estimate_params_from_file_size(
            "test-model", "token")
        self.assertIsNone(params)

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_fetch_library_stats(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.list_models.return_value = [
            MockModel("model1",
                      downloads=100,
                      likes=10,
                      safetensors=MockSafeTensors(1e9)),
            MockModel("model2",
                      downloads=200,
                      likes=20,
                      safetensors=MockSafeTensors(5e9)),
        ]

        stats = smart_hf_search.fetch_library_stats(
            lib={
                "id": "transformers",
                "label": "Transformers"
            },
            api=mock_api,
            n=2,
            avg_limit=2,
            max_params_val=6e9,
        )

        self.assertEqual(stats["id"], "transformers")
        self.assertEqual(stats["avg_downloads"], 150.0)
        self.assertLen(stats["models"], 2)
        self.assertEqual(stats["models"][0]["id"], "model1")
        self.assertEqual(stats["models"][0]["params_str"], "1.00B")

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_fetch_library_stats_exception(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.list_models.side_effect = Exception("error detail")

        stats = smart_hf_search.fetch_library_stats(
            lib={
                "id": "transformers",
                "label": "Transformers"
            },
            api=mock_api,
            n=2,
            avg_limit=2,
            max_params_val=6e9,
        )

        self.assertEqual(stats["id"], "transformers")
        self.assertIn("Exception: error detail", stats["error"])

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_get_top_libraries(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_api.get_model_tags.return_value = {
            "library": [
                {
                    "id": "transformers",
                    "label": "Transformers"
                },
                {
                    "id": "timm",
                    "label": "Timm"
                },
            ]
        }

        def mock_fetch_stats(lib, *unused_args, **unused_kwargs):
            downloads = 100.0 if lib["id"] == "transformers" else 50.0
            return {
                "id": lib["id"],
                "label": lib["label"],
                "avg_downloads": downloads,
                "models": [],
            }

        with mock.patch(
                "smart_hf_search.fetch_library_stats",
                side_effect=mock_fetch_stats,
        ):
            libs = smart_hf_search.get_top_libraries(k=2,
                                                     n=1,
                                                     max_params_val=None,
                                                     exclude_libs=["timm"])

        self.assertLen(libs, 1)
        self.assertEqual(libs[0]["id"], "transformers")

    @mock.patch("huggingface_hub.HfApi", autospec=True)
    def test_get_top_libraries_sorting_and_limiting(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value
        mock_libraries: list[dict[str, Any]] = [{
            "id": f"lib_{i}",
            "label": f"Lib {i}"
        } for i in range(60)]
        mock_libraries[50] = {"id": "transformers", "label": "Transformers"}
        mock_libraries[51] = {
            "id": "custom_lib",
            "label": "Custom",
            "count": 200000,
        }
        mock_libraries[0]["count"] = 5

        mock_api.get_model_tags.return_value = {"library": mock_libraries}

        def mock_fetch_stats(lib, *unused_args, **unused_kwargs):
            return {
                "id": lib["id"],
                "label": lib.get("label", ""),
                "avg_downloads": 100.0,
                "models": [],
            }

        class MockExecutor:

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args, **kwargs):
                res = fn(*args, **kwargs)
                mock_future = mock.Mock()
                mock_future.result.return_value = res
                return mock_future

        def mock_as_completed(fs):
            return fs

        with mock.patch(
                "smart_hf_search.fetch_library_stats",
                side_effect=mock_fetch_stats,
        ) as mock_fetch, mock.patch(
                "smart_hf_search.concurrent.futures.ThreadPoolExecutor",
                MockExecutor,
        ), mock.patch(
                "smart_hf_search.concurrent.futures.as_completed",
                mock_as_completed,
        ):
            _ = smart_hf_search.get_top_libraries(k=5,
                                                  n=1,
                                                  max_params_val=None)

        self.assertEqual(mock_fetch.call_count, 50)
        called_libs = [
            call.args[0]["id"] for call in mock_fetch.call_args_list
        ]
        self.assertEqual(called_libs[0], "custom_lib")
        self.assertEqual(called_libs[1], "transformers")

    @mock.patch("huggingface_hub.HfApi")
    def test_search_models(self, mock_hf_api_class):
        mock_api = mock_hf_api_class.return_value

        # Return models in mock HfApi call
        mock_api.list_models.return_value = [
            MockModel(
                "model1",
                downloads=100,
                likes=10,
                library_name="transformers",
                safetensors=MockSafeTensors(2e9),
            ),
            MockModel(
                "model_incompatible",
                downloads=500,
                likes=50,
                library_name="transformers",
                tags=["gguf"],
            ),
            MockModel(
                "model2",
                downloads=50,
                likes=5,
                library_name="timm",
                safetensors=MockSafeTensors(8e9),
            ),
        ]

        models = smart_hf_search.search_models(
            limit=5,
            sort="trending",
            max_params_val=5e9,
            min_params_val=1e9,
            library=None,
            pipeline_tag=None,
            exclude_libs=["timm"],
            token="some-token",
        )

        self.assertLen(models, 1)
        self.assertEqual(models[0]["id"], "model1")
        self.assertEqual(models[0]["params_str"], "2.00B")
        self.assertEqual(models[0]["library"], "transformers")


if __name__ == "__main__":
    absltest.main()
