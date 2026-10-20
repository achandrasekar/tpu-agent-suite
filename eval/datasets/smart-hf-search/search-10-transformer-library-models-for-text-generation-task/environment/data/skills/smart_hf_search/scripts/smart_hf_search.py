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
r"""Script to search Hugging Face models and libraries based on criteria.

This script identifies trending, highly downloaded, or liked PyTorch models
and libraries matching parameter limits, pipeline tags, and architectures.
It filters out incompatible formats (e.g., GGUF or TFLite/LiteRT) that cannot
be loaded directly via PyTorch XLA compilation backends on Google Cloud TPUs.

Output Example (JSON format when passing `--json` flag in models mode):
  [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "downloads": 10578535,
      "likes": 5891,
      "params_str": "8.03B",
      "library": "transformers",
      "gated": false
    }
  ]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Literal, cast

import huggingface_hub

# A fallback priority mapping of library IDs to their popularity heuristics
# to prioritize fetching stats for more widely-used libraries, especially since
# api.get_model_tags() response tags under the "library" key might lack a
# "count" field.
_POPULAR_LIBRARIES_PRIORITY = {
    "transformers": 100000,
    "pytorch": 90000,
    "diffusers": 80000,
    "timm": 70000,
    "sentence-transformers": 60000,
    "jax": 50000,
    "tf": 40000,
    "onnx": 30000,
    "auto-gptq": 20000,
    "gguf": 10000,
}

_MAX_LIBRARIES_TO_FETCH = 50


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search Hugging Face models and libraries.")
    parser.add_argument(
        "--mode",
        choices=["models", "libraries"],
        default="models",
        help="Search mode: models or libraries.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of models or libraries to return.",
    )
    parser.add_argument(
        "--sort",
        default="trending",
        choices=["trending", "downloads", "likes"],
        help="Sort key: trending, downloads, likes.",
    )
    parser.add_argument(
        "--max_params",
        default=None,
        help="Maximum parameters (e.g., '20B', '350M').",
    )
    parser.add_argument(
        "--min_params",
        default=None,
        help="Minimum parameters (e.g., '1B').",
    )
    parser.add_argument(
        "--library",
        default=None,
        help="Filter by library name (e.g., 'transformers').",
    )
    parser.add_argument(
        "--pipeline_tag",
        default=None,
        help="Filter by pipeline tag (e.g., 'text-generation').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (optional).",
    )
    parser.add_argument(
        "--exclude_library",
        default="nemo",
        help="Comma-separated list of libraries to exclude from search.",
    )
    parser.add_argument(
        "--base_model_only",
        action="store_true",
        help="Only include base models, excluding fine-tuned variants.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Search query to filter models by name/keyword (e.g., 'bert').",
    )
    return parser.parse_args()


def parse_params_limit(limit_str: str | None) -> float | None:
    """Parses a parameter limit string like '8B', '350M' into a float value."""
    if not limit_str:
        return None
    s = limit_str.strip().upper()
    try:
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        elif s.endswith("M"):
            return float(s[:-1]) * 1e6
        elif s.endswith("K"):
            return float(s[:-1]) * 1e3
        else:
            return float(s)
    except ValueError:
        print(
            f"Warning: Could not parse params limit '{limit_str}', ignoring"
            " filter.",
            file=sys.stderr,
        )
        return None


def is_incompatible(model: Any) -> bool:
    """Checks if a model is incompatible with Cloud TPU.

  Filters out:
  - GGUF: Single-file binary format designed for consumer CPU/GPU quantization
    engines (llama.cpp) which cannot be loaded via PyTorch AutoClasses.
  - TFLite/LiteRT: Mobile edge runtime graphs lacking execution delegates
    for Google Cloud TPU XLA hardware.
  - BitsAndBytes: Quantized models using custom CUDA kernels that are not
    supported on Cloud TPU.

  Multi-format Safe-guard:
  If a model has mobile tags but also hosts standard PyTorch/Safetensors weights
  (e.g., as part of a multi-format repository), it is kept as compatible.

  Args:
    model: The model object from Hugging Face Hub API.

  Returns:
    True if the model is considered incompatible with Cloud TPU, False
    otherwise.
  """
    model_id = getattr(model, "id", "")
    if not model_id:
        return True

    # Check GGUF.
    if model_id.lower().endswith("gguf") or "/gguf" in model_id.lower():
        return True
    tags = getattr(model, "tags", None) or []
    tags_lower = [str(t).lower() for t in tags]
    if any("gguf" == t for t in tags_lower):
        return True

    # Check BitsAndBytes.
    if any("bitsandbytes" in t for t in tags_lower):
        return True

    # Check TFLite/LiteRT.
    has_tflite_id = any(x in model_id.lower() for x in ["tflite", "litert"])
    if has_tflite_id:
        # If it has pytorch or safetensors, it might still be compatible.
        has_pytorch = any(x in tags_lower for x in ["pytorch", "safetensors"])
        if not has_pytorch:
            return True
    else:
        has_tflite_tag = any(
            any(x in t for x in ["tflite", "litert"]) for t in tags_lower)
        if has_tflite_tag:
            has_pytorch = any(x in tags_lower
                              for x in ["pytorch", "safetensors"])
            if not has_pytorch:
                return True

    return False


def estimate_params_from_file_size(model_id: str,
                                   token: str | None) -> float | None:
    """Estimates parameter count from weight file sizes in the repository.

  Args:
    model_id: The ID of the model on the Hugging Face Hub.
    token: Optional Hugging Face authentication token.

  Returns:
    The estimated number of parameters as a float (assuming 2 bytes/param for
    f16/bf16 formats if standard formats exist), or None if it cannot estimate
    the parameters.
  """
    api = huggingface_hub.HfApi()
    weight_extensions = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    try:
        info = api.model_info(model_id, files_metadata=True, token=token)
        siblings = getattr(info, "siblings", [])

        files_by_ext = {ext: [] for ext in weight_extensions}
        for sibling in siblings:
            filename = sibling.rfilename.lower()
            for ext in weight_extensions:
                if filename.endswith(ext):
                    files_by_ext[ext].append(sibling.size or 0)
                    break

        selected_sizes = []
        for ext in weight_extensions:
            if files_by_ext[ext]:
                selected_sizes = files_by_ext[ext]
                break

        if selected_sizes:
            total_size = sum(selected_sizes)
            if total_size > 0:
                # Assume 2 bytes per parameter (bfloat16/float16)
                return total_size / 2
    except Exception:  # pylint: disable=broad-except
        pass
    return None


def fetch_library_stats(
    lib: Dict[str, Any],
    api: huggingface_hub.HfApi,
    *,
    n: int,
    avg_limit: int,
    max_params_val: float | None,
    token: str | None = None,
) -> Dict[str, Any]:
    """Fetches stats for a single library.

  Args:
    lib: A dictionary representing the library, which contains keys like 'id'
      and 'label'.
    api: The Hugging Face Hub API client.
    n: The number of top PyTorch-compatible models to retrieve and list under
      the library.
    avg_limit: The number of top models (sorted by downloads) to average
      downloads over in order to calculate the library's popularity metric
      ('avg_downloads').
    max_params_val: The maximum parameter count limit (as a float, e.g., 2e10
      for 20B) used to filter out models that are too large. If None, no
      parameter limit filtering is applied.
    token: Optional Hugging Face authentication token.

  Returns:
    A dictionary containing library stats, including 'id', 'label',
    'avg_downloads', and a list of 'models' details.
  """
    lib_id = lib.get("id")
    lib_label = lib.get("label")
    lib_info = {
        "id": lib_id,
        "label": lib_label,
        "models": [],
        "avg_downloads": 0.0,
    }
    try:
        # If filtering by parameter limit, many models might be discarded.
        # We fetch a larger candidate pool (oversampling by a factor of 5)
        # to ensure that we are left with at least max(n, avg_limit) valid models
        # after filtering out those that are too large.
        fetch_limit = (max(n, avg_limit) *
                       5 if max_params_val is not None else max(n, avg_limit))
        models = api.list_models(
            filter=lib_id,
            sort="downloads",
            limit=fetch_limit,
            expand=["safetensors", "likes", "downloads", "gated"],
            token=token,
        )

        valid_models = []
        for m in models:
            if is_incompatible(m):
                continue
            if max_params_val is not None:
                st = getattr(m, "safetensors", None)
                total_params = getattr(st, "total", None) if st else None
                if total_params is None:
                    total_params = estimate_params_from_file_size(m.id, token)
                if total_params is not None and total_params > max_params_val:
                    continue
                elif total_params is None:
                    continue
            valid_models.append(m)

        if valid_models:
            avg_candidates = valid_models[:avg_limit]
            total_downloads = sum(
                getattr(m, "downloads", 0) or 0 for m in avg_candidates)
            lib_info["avg_downloads"] = (total_downloads / len(avg_candidates)
                                         if avg_candidates else 0.0)

            for model in valid_models[:n]:
                downloads = getattr(model, "downloads", 0) or 0
                likes = getattr(model, "likes", 0) or 0
                st = getattr(model, "safetensors", None)
                params = getattr(st, "total", None) if st else None
                if params is None:
                    params = estimate_params_from_file_size(model.id, token)

                params_str = "Unknown"
                if params is not None:
                    if params >= 1e9:
                        params_str = f"{params / 1e9:.2f}B"
                    elif params >= 1e6:
                        params_str = f"{params / 1e6:.2f}M"
                    else:
                        params_str = f"{params:,}"

                lib_info["models"].append({
                    "id":
                    model.id,
                    "downloads":
                    downloads,
                    "likes":
                    likes,
                    "params_str":
                    params_str,
                    "gated":
                    getattr(model, "gated", False),
                })
    except Exception as e:  # pylint: disable=broad-except
        lib_info["error"] = "".join(traceback.format_exception_only(e)).strip()
    return lib_info


def get_top_libraries(
    k: int,
    *,
    n: int,
    max_params_val: float | None,
    exclude_libs: list[str] | None = None,
    token: str | None = None,
) -> List[Dict[str, Any]]:
    """Extracts the top k libraries ranked by their popular models' average downloads.

  Args:
    k: The number of top libraries to return.
    n: The number of top compatible models to return in the list for each
      library.
    max_params_val: The maximum parameter limit to filter models within
      libraries.
    exclude_libs: A list of library names to exclude.
    token: Optional Hugging Face authentication token.

  Returns:
    A list of dictionaries, where each library contains metadata, its calculated
    average downloads, and details of its top n compatible models.
  """
    api = huggingface_hub.HfApi(token=token)
    results = []
    try:
        tags = api.get_model_tags()
        integrated_libraries = tags.get("library", [])
        exclude_libs_list = exclude_libs or []
        integrated_libraries = [
            lib for lib in integrated_libraries
            if lib.get("id") not in exclude_libs_list
        ]

        # Sort libraries by model count if available, or fall back to popularity
        # priority list.
        integrated_libraries.sort(
            key=lambda x: x.get("count") or _POPULAR_LIBRARIES_PRIORITY.get(
                x.get("id"), 0),
            reverse=True,
        )

        # Limit to top N libraries to avoid fetching stats for too many libraries,
        # which is slow and can trigger API rate limits.
        integrated_libraries = integrated_libraries[:_MAX_LIBRARIES_TO_FETCH]

        # Use ThreadPoolExecutor to fetch stats in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    fetch_library_stats,
                    lib,
                    api,
                    n=n,
                    avg_limit=10,
                    max_params_val=max_params_val,
                    token=token,
                ) for lib in integrated_libraries
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Sort by average downloads descending.
        results.sort(key=lambda x: x.get("avg_downloads", 0.0), reverse=True)
        return results[:k]
    except Exception:  # pylint: disable=broad-except
        logging.exception("Error fetching libraries")
        return []


def is_base_model(model: huggingface_hub.v1.hf_api.ModelInfo) -> bool:
    """Checks if a model is a base model.

  Model tags will include a tag in the form of `base_model:<name>` only if the
  current model is a finetune of the model named <name>. Therefore, any model
  that doesn't include a base_model tag should be considered a base_model.

  Args:
    model: The Hugging Face model info object.

  Returns:
    True if the model is a base model, False otherwise.
  """
    tags = getattr(model, "tags", None) or []
    tags_lower = [str(t).lower() for t in tags]
    for tag in tags_lower:
        if tag.startswith("base-model") or tag.startswith("base_model"):
            return False
    return True


def search_models(
    limit: int,
    *,
    sort: str,
    search: str | None = None,
    max_params_val: float | None,
    min_params_val: float | None,
    library: str | None,
    pipeline_tag: str | None,
    exclude_libs: list[str] | None = None,
    base_model_only: bool = False,
    token: str | None = None,
) -> List[Dict[str, Any]]:
    """Searches for models matching criteria.

  Args:
    limit: The maximum number of model results to return.
    sort: The sort key ('trending', 'downloads', or 'likes').
    search: A search string to filter models.
    max_params_val: The maximum parameter limit (e.g. 2e10 for 20B) to filter
      models.
    min_params_val: The minimum parameter limit to filter models.
    library: Library name filter (e.g., 'transformers').
    pipeline_tag: Pipeline tag filter (e.g., 'text-generation').
    exclude_libs: A list of library names to exclude.
    base_model_only: Whether to only return base models (no fine-tuned).
    token: Optional Hugging Face token.

  Returns:
    A list of dictionaries representing the found compatible models.
  """
    api = huggingface_hub.HfApi(token=token)
    sort_mapping = {
        "trending": "trending_score",
        "downloads": "downloads",
        "likes": "likes",
    }

    # Define fallback sequence if "trending" is requested
    sorts_to_try = [sort]
    if sort == "trending":
        sorts_to_try.extend(["downloads", "likes"])

    filters = ["pytorch"]  # Default to PyTorch for TPU compatibility
    if library:
        filters.append(library)
    if pipeline_tag:
        filters.append(pipeline_tag)

    # Fetch more to allow for param filtering
    fetch_limit = limit * 10 if (max_params_val or min_params_val) else limit

    exclude_libs_list = exclude_libs or []
    valid_models = []
    seen_ids = set()

    for current_sort in sorts_to_try:
        if len(valid_models) >= limit:
            break

        hf_sort = cast(
            Literal[
                "created_at",
                "downloads",
                "last_modified",
                "likes",
                "trending_score",
            ],
            sort_mapping.get(current_sort, "trending_score"),
        )

        try:
            models = api.list_models(
                filter=filters,
                sort=hf_sort,
                limit=fetch_limit,
                search=search,
                expand=[
                    "safetensors",
                    "library_name",
                    "likes",
                    "downloads",
                    "gated",
                    "pipeline_tag",
                    "tags",
                ],
                token=token,
            )

            for m in models:
                if m.id in seen_ids:
                    continue

                lib = getattr(m, "library_name", None)

                if base_model_only and not is_base_model(m):
                    continue

                if lib in exclude_libs_list:
                    continue
                if is_incompatible(m):
                    continue

                st = getattr(m, "safetensors", None)
                total_params = getattr(st, "total", None) if st else None

                if max_params_val or min_params_val:
                    if total_params is None:
                        # Estimate from file size
                        total_params = estimate_params_from_file_size(
                            m.id, token)

                    if total_params is not None:
                        if max_params_val and total_params > max_params_val:
                            continue
                        if min_params_val and total_params < min_params_val:
                            continue
                    else:
                        # Still None after estimation, skip
                        continue

                params_str = "Unknown"
                if total_params is not None:
                    if total_params >= 1e9:
                        params_str = f"{total_params / 1e9:.2f}B"
                    elif total_params >= 1e6:
                        params_str = f"{total_params / 1e6:.2f}M"
                    else:
                        params_str = f"{total_params:,}"

                valid_models.append({
                    "id":
                    m.id,
                    "downloads":
                    getattr(m, "downloads", 0) or 0,
                    "likes":
                    getattr(m, "likes", 0) or 0,
                    "params_str":
                    params_str,
                    "library":
                    getattr(m, "library_name", "Unknown"),
                    "gated":
                    getattr(m, "gated", False),
                    "pipeline_tag":
                    getattr(m, "pipeline_tag", "Unknown"),
                })
                seen_ids.add(m.id)

                if len(valid_models) >= limit:
                    break

        except Exception:  # pylint: disable=broad-except
            logging.exception("Error searching models with sort %s",
                              current_sort)

    return valid_models


def main() -> None:
    args = parse_args()

    max_params_val = parse_params_limit(args.max_params)
    min_params_val = parse_params_limit(args.min_params)
    exclude_libs = (
        [x.strip() for x in args.exclude_library.split(",")
         if x.strip()] if args.exclude_library else [])
    token = args.token or os.environ.get("HF_TOKEN")

    if args.mode == "libraries":
        results = get_top_libraries(
            args.k,
            n=5,
            max_params_val=max_params_val,
            exclude_libs=exclude_libs,
            token=token,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, lib_info in enumerate(results, 1):
                print(
                    f"{i}. Library: {lib_info['label']} ({lib_info['id']}) - Avg"
                    f" Downloads: {lib_info['avg_downloads']:.2f}")
                if "error" in lib_info:
                    print(f"  Error: {lib_info['error']}")
                elif not lib_info["models"]:
                    print("  No models found.")
                else:
                    for j, model in enumerate(lib_info["models"], 1):
                        print(
                            f"  {j}. {model['id']} (Downloads: {model['downloads']:.1f},"
                            f" Likes: {model['likes']}, Params: {model['params_str']})"
                        )
                print("-" * 50)
    else:
        results = search_models(
            args.k,
            sort=args.sort,
            search=args.search,
            max_params_val=max_params_val,
            min_params_val=min_params_val,
            library=args.library,
            pipeline_tag=args.pipeline_tag,
            exclude_libs=exclude_libs,
            base_model_only=args.base_model_only,
            token=token,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Top {args.k} {args.sort} models:")
            for i, model in enumerate(results, 1):
                print(
                    f"{i}. {model['id']} (Downloads: {model['downloads']}, Likes:"
                    f" {model['likes']}, Params: {model['params_str']}, Library:"
                    f" {model['library']}, Pipeline Tag: {model['pipeline_tag']})"
                )


if __name__ == "__main__":
    main()
