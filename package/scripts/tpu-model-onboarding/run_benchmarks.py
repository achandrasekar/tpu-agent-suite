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
"""Orchestrator script for benchmarking multiple Hugging Face models."""

import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

# Disable XET backend to avoid 403 sharing errors.
os.environ["HF_HUB_DISABLE_XET"] = "1"


def get_cache_dir() -> str:
    """The hub cache to download into.

    `HF_HUB_CACHE` when it is set: gke_job sets it to a shared volume
    (`hf_cache_pvc`), or without one to the pod's /dev/shm, and an explicit
    `cache_dir` would override it. Otherwise a private directory.
    """
    return os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
        "~/.cache/torch_tpu_cache")


def cache_is_shared() -> bool:
    """Whether the cache belongs to someone else: never prune it.

    A shared volume holds models other Jobs are loading, and a cache the
    user pointed `HF_HUB_CACHE` at is theirs; deleting a model there breaks
    the next run rather than reclaiming this pod's disk. gke_job names the
    one it gave the pod in `TPU_AGENT_SUITE_POD_HF_CACHE`; that one is ours.
    """
    cache = os.environ.get("HF_HUB_CACHE")
    return bool(cache) and cache != os.environ.get(
        "TPU_AGENT_SUITE_POD_HF_CACHE")


def get_free_space_gb(path: str) -> float:
    """Returns the free space of the filesystem containing path, in GB."""
    try:
        os.makedirs(path, exist_ok=True)
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / (1024 * 1024 * 1024)
    except Exception:  # pylint: disable=broad-exception-caught
        return 0.0


def _scan_tpu_devices() -> list[pathlib.Path]:
    """Scans PCI bus for Google TPU v7e devices."""
    pci_devices = pathlib.Path("/sys/bus/pci/devices")
    if not pci_devices.exists():
        return []

    tpu_devices = []
    for device_path in pci_devices.iterdir():
        try:
            vendor = (device_path / "vendor").read_text().strip()
            if vendor == "0x1ae0":  # Google PCI Vendor ID
                device_id = (device_path / "device").read_text().strip()
                if device_id == "0x0076":  # TPU v7e Device ID
                    tpu_devices.append(device_path)
        except OSError:
            pass
    return tpu_devices


def detect_tpu_chips() -> list[int]:
    """Detects available physical TPU chip indices on the host by scanning PCI bus."""
    tpu_devices = _scan_tpu_devices()
    if not tpu_devices:
        return [0]  # Fallback to default single-chip index.

    chips = set()
    for device_path in tpu_devices:
        # Sibling cores share the same slot name and only differ in Function
        # index (e.g. .0 or .1)
        slot = device_path.name.split(".")[0]
        chips.add(slot)

    # Return logical chip indices 0..N-1.
    return list(range(len(chips)))


def get_tpu_chip_vfio_path(chip_idx: int) -> str | None:
    """Dynamically resolves the Core 0 VFIO path for the given physical TPU chip index."""
    tpu_devices = _scan_tpu_devices()
    if not tpu_devices:
        return None

    # Group detected TPU functions by physical chip slot.
    chips_map = {}
    for dev_path in tpu_devices:
        slot = dev_path.name.split(".")[0]
        func = dev_path.name.split(".")[1]
        if slot not in chips_map:
            chips_map[slot] = {}
        chips_map[slot][func] = dev_path

    # Sort slots to ensure deterministic ordering of chips (0 to N-1).
    sorted_slots = sorted(chips_map.keys())
    if chip_idx >= len(sorted_slots):
        return None

    target_slot = sorted_slots[chip_idx]
    # Target Core 0 (function "0") of that physical chip.
    core0_path = chips_map[target_slot].get("0")
    if core0_path is None:
        core0_path = list(chips_map[target_slot].values())[0]

    # Resolve the VFIO path for that core.
    try:
        iommu_group = (core0_path / "iommu_group").readlink().name
        return f"/dev/vfio/{iommu_group}"
    except OSError as e:
        print(
            f"WARNING: Failed to read IOMMU group for {core0_path.name}: {e!r}"
        )
        return None


class TPUDeviceAllocator:
    """Thread-safe allocator for physical TPU chips."""

    def __init__(self, chip_indices: list[int]):
        self.lock = threading.Lock()
        self.available_chips = sorted(list(chip_indices))

    def allocate(self) -> int | None:
        with self.lock:
            if not self.available_chips:
                return None
            return self.available_chips.pop(0)

    def free(self, chip_idx: int):
        with self.lock:
            self.available_chips.append(chip_idx)


def throttle_prefetch(model_id: str, max_models: int = 50):
    """Throttle pre-fetching of models to avoid overloading the disk."""
    cache_dir = get_cache_dir()
    safe_folder_name = "models--" + model_id.replace("/", "--")
    model_cache_path = os.path.join(cache_dir, safe_folder_name)

    # Only throttle if the folder does NOT exist
    # Downloading it will increase the count
    if not os.path.exists(model_cache_path):
        free_space = get_free_space_gb(cache_dir)
        if free_space < 50:
            print(f"[PRE-FETCHER] Low disk space ({free_space:.1f} GB free). "
                  "Checking folder limit...")
            start_wait = time.time()
            while True:
                try:
                    if os.path.exists(cache_dir):
                        folders = [
                            d for d in os.listdir(cache_dir)
                            if d.startswith("models--")
                        ]
                        print(
                            f"[PRE-FETCHER] {model_id} folders count: {len(folders)}"
                        )
                        if len(folders) < max_models:
                            break
                        if time.time() - start_wait > 300:
                            print(
                                "[PRE-FETCHER] 5 minutes elapsed, forcing pre-fetch for"
                                f" {model_id}")
                            break
                    else:
                        break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(
                        f"[PRE-FETCHER] {model_id} exception in throttling: {e}"
                    )
                    break
                time.sleep(5)
        else:
            print(
                "[PRE-FETCHER] Skipping throttling for"
                f" {model_id} (plenty of disk space: {free_space:.1f} GB free)"
            )


def prefetch_model(
    model_id: str,
    skip_throttle: bool = False,
    max_models: int = 50,
    sleep_seconds: int = 2,
):
    """Background task to pre-download model weights."""
    print("[PRE-FETCHER] Starting of prefetch_model for"
          f" {model_id}, skip_throttle={skip_throttle}")

    if not skip_throttle:
        throttle_prefetch(model_id, max_models)

    try:
        # pylint: disable=g-import-not-at-top
        import huggingface_hub

        print(
            f"\n[PRE-FETCHER] Starting background download for {model_id}...")
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            cache_dir=get_cache_dir(),
            ignore_patterns=["*.msgpack", "*.h5", "*.ckpt", "*.ot"],
        )
        print(f"\n[PRE-FETCHER] Finished download for {model_id}.")
    except ImportError:
        print(
            "\n[PRE-FETCHER] huggingface_hub not installed, skipping pre-fetch."
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"\n[PRE-FETCHER] Failed to pre-fetch {model_id}: {e}")
    finally:
        # Rate limit guard: sleep between API calls to avoid 429
        time.sleep(sleep_seconds)


def run_benchmark_subprocess(
    model_id: str,
    library: str,
    pipeline_tag: str,
    device_str: str,
    clean_cache: bool,
    verify: bool,
    ref_dir: str,
    compile_flag: bool,
    chip_idx: int | None = None,
) -> tuple[float, str, float | None, float | None, dict[str, Any] | None]:
    """Runs the benchmark script in a subprocess to ensure memory isolation."""
    library = library or "transformers"
    pipeline_tag = pipeline_tag or "Unknown"
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "benchmark_tpu.py")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_file = tmp.name

    try:
        cmd = [
            sys.executable or "python3",
            script_path,
            "--model",
            model_id,
            "--device",
            device_str,
            "--library",
            library,
            "--pipeline_tag",
            pipeline_tag,
            "--ref_dir",
            ref_dir,
            "--output_file",
            output_file,
        ]
        if clean_cache:
            cmd.append("--cleanup_cache")
        if not verify:
            cmd.append("--no-verify")
        if compile_flag:
            cmd.append("--compile")

        env = os.environ.copy()
        if device_str == "tpu" and chip_idx is not None:
            vfio_path = get_tpu_chip_vfio_path(chip_idx)
            if vfio_path:
                env["TPU_VISIBLE_DEVICES"] = str(chip_idx)
                env["TPU_VISIBLE_DEVICE_PATHS"] = vfio_path
                env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
                env["LIBTPU_INIT_ARGS"] = (
                    "--xla_tpu_enable_diagnostic_coordinator=false")
                env["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
            else:
                print(
                    f"WARNING: Could not resolve VFIO path for TPU chip {chip_idx}."
                    " Defaulting to unsanitized run.")

        print(f"Executing Subprocess: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "Subprocess timed out:\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}") from e

        if result.returncode != 0:
            raise RuntimeError(
                "Subprocess failed (exit code"
                f" {result.returncode}):\n{result.stderr}\n--- STDOUT"
                f" ---\n{result.stdout}")

        with open(output_file, "r") as f:
            res_json = json.load(f)

        duration = res_json.get("duration")
        output_info = res_json.get("output_info", "")
        mse = res_json.get("mse")
        compile_time = res_json.get("compile_time")
        ppl_stats = res_json.get("ppl_stats")

        if duration is None:
            raise RuntimeError(
                f"Duration was null in JSON output.\n--- STDOUT ---\n{result.stdout}"
            )

        return duration, output_info, mse, compile_time, ppl_stats
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)


def get_models_by_device_benchmark(
    models: list[Any],
    results_dict: dict[tuple[str, str, str, bool, bool], Any],
    global_compile: bool = False,
    retry_failed: bool = False,
):
    """Groups models by device, skipping completed or invalid ones."""
    models_by_device = collections.defaultdict(list)
    for idx, item in enumerate(models):
        model_id = item.get("model_id")
        library = item.get("library") or "transformers"
        pipeline_tag = item.get("pipeline_tag") or "Unknown"
        device_str = item.get("device", "cpu")
        clean_cache = item.get("clean_cache", False)
        compile_flag = item.get("compile", False) or global_compile

        if not model_id:
            print(f"WARNING: Skipping entry {idx} due to missing 'model_id'.")
            continue

        config_key = (model_id, device_str, library, clean_cache, compile_flag)
        if (config_key in results_dict
                and results_dict[config_key].get("status") == "success"):
            print(
                f"\nINFO: Skipping already successfully completed model: {model_id}"
                f" on {device_str}")
            continue
        elif (not retry_failed and config_key in results_dict
              and results_dict[config_key].get("status") == "error"):
            print(f"\nINFO: Skipping previously failed model: {model_id} on"
                  f" {device_str}")
            continue

        item["idx"] = idx
        item["device"] = device_str
        item["library"] = library
        item["pipeline_tag"] = pipeline_tag
        item["clean_cache"] = clean_cache
        item["compile"] = compile_flag
        item["config_key"] = config_key
        models_by_device[str(device_str)].append(item)
    return models_by_device


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Orchestrate benchmarking of multiple HF models.")
    parser.add_argument(
        "--models_json",
        type=str,
        default="models.json",
        help="Path to models JSON file.",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default=".benchmark_refs",
        help="Global reference directory for storing CPU validation tensors.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Globally disable model output correctness verification.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Globally retry failed benchmarks.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results.json",
        help="Path to save the benchmark results JSON.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Globally enable model compilation.",
    )
    parser.add_argument(
        "--tpu_concurrency",
        type=int,
        default=None,
        help=(
            "Maximum parallel concurrency for TPU models (defaults to number of"
            " detected chips). Set to 1 for sequential execution."),
    )
    parser.add_argument(
        "--cpu_concurrency",
        type=int,
        default=1,
        help=("Maximum parallel concurrency for CPU models (defaults to 1)."),
    )
    parser.add_argument(
        "--gpu_concurrency",
        type=int,
        default=1,
        help=("Maximum parallel concurrency for GPU models (defaults to 1)."),
    )
    parser.add_argument(
        "--prefetch_concurrency",
        type=int,
        default=1,
        help=
        ("Maximum parallel concurrency for downloading models (defaults to 1)."
         ),
    )
    parser.add_argument(
        "--prefetch_sleep_seconds",
        type=int,
        default=2,
        help=
        ("Sleep seconds between prefetch API calls to avoid rate limiting."
         " Ideally sleep duration should be prportional to number of workers."
         " (default: 2)"),
    )
    return parser.parse_args()


def get_models_to_benchmark(args: argparse.Namespace) -> list[Any]:
    """Returns the list of models to benchmark."""
    models_json_path = os.path.expanduser(args.models_json)
    if not os.path.exists(models_json_path):
        print(f"ERROR: {models_json_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(models_json_path, "r") as f:
            content = f.read()
            # Replace boolean literals to make the file valid JSON
            content = re.sub(r":\s*True\b", ": true", content)
            content = re.sub(r":\s*False\b", ": false", content)
            data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {models_json_path}: {e}",
              file=sys.stderr)
        sys.exit(1)

    return data.get("models", [])


def get_results_dict(
    args: argparse.Namespace, ) -> dict[tuple[str, str, str, bool, bool], Any]:
    """Returns the results dictionary from the output JSON file."""
    results_dict = {}
    output_path = os.path.expanduser(args.output_json)
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                for r in existing_data.get("results", []):
                    key = (
                        r.get("model_id"),
                        r.get("device"),
                        r.get("library"),
                        r.get("clean_cache"),
                        r.get("compile", False),
                    )
                    results_dict[key] = r
            print(
                f"Loaded {len(results_dict)} existing results from {output_path}."
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"WARNING: Failed to load existing results from {output_path}: {e}"
            )
    return results_dict


def write_results(
    args: argparse.Namespace,
    results_dict: dict[tuple[str, str, str, bool, bool], Any],
):
    """Writes the results dictionary to the output JSON file atomically."""
    output_path = os.path.expanduser(args.output_json)
    try:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump({"results": list(results_dict.values())}, f, indent=2)
        os.replace(temp_path, output_path)
        print(f"\nUpdated benchmark results in {output_path}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"WARNING: Failed to save results to {output_path}: {e}")


def get_worker_pool_executors(
        models_by_device: dict[str, Any], num_chips: int,
        args: argparse.Namespace) -> dict[str, concurrent.futures.Executor]:
    """Returns a dict of executors for each device type."""
    executors = {}
    max_workers = {
        "tpu": args.tpu_concurrency or num_chips,
        "cpu": max(1, args.cpu_concurrency),
        "cuda": max(1, args.gpu_concurrency),
    }
    for device in models_by_device:
        executors[device] = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers.get(device, 1))
    return executors


def cleanup_model_cache(model_id: str) -> None:
    """Removes downloaded Hugging Face model cache to reclaim disk space."""
    cache_dir = get_cache_dir()
    safe_folder_name = "models--" + model_id.replace("/", "--")
    model_cache_path = os.path.join(cache_dir, safe_folder_name)

    if cache_is_shared():
        print(f"INFO: [ORCHESTRATOR] Keeping {model_cache_path}: the cache is "
              "shared (HF_HUB_CACHE), not this pod's to clean up")
        return
    if os.path.exists(model_cache_path):
        print("INFO: [ORCHESTRATOR] Cleaning up model cache directory:"
              f" {model_cache_path}")
        shutil.rmtree(model_cache_path, ignore_errors=True)


def main():
    args = parse_args()
    models = get_models_to_benchmark(args)
    if not models:
        print(f"WARNING: No models found in {args.models_json}.")
        sys.exit(0)

    verify = not args.no_verify

    print(f"Found {len(models)} models to benchmark.")

    results_dict = get_results_dict(args)
    models_to_run = get_models_by_device_benchmark(models, results_dict,
                                                   args.compile,
                                                   args.retry_failed)
    total_runs = len(models)

    detected_chips = detect_tpu_chips()
    num_chips = len(detected_chips)
    print(f"Detected {num_chips} physical TPU chip(s) on host.")

    allocator = TPUDeviceAllocator(detected_chips)
    executors = get_worker_pool_executors(models_to_run, num_chips, args)

    runs_scheduled_count = collections.defaultdict(int)
    for device in models_to_run:
        for item in models_to_run[device]:
            runs_scheduled_count[item["model_id"]] += 1

    prefetch_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.prefetch_concurrency)

    # Pre-populate prefetch queue.
    prefetch_futures = {}
    for device in models_to_run:
        for item in models_to_run[device]:
            model_id = item["model_id"]
            if model_id not in prefetch_futures:
                prefetch_futures[model_id] = prefetch_executor.submit(
                    prefetch_model, model_id)

    def benchmark_worker(item):
        idx = item["idx"]
        config_key = item["config_key"]
        model_id = item["model_id"]
        library = item["library"]
        pipeline_tag = item["pipeline_tag"]
        device_str = item["device"]
        clean_cache = item["clean_cache"]
        compile_flag = item["compile"]

        # Wait for prefetch to complete.
        prefetch_future = prefetch_futures.get(model_id)
        wait_time = 0.0
        if prefetch_future:
            t_start = time.time()
            print(f"INFO: Waiting for prefetch of {model_id}...")
            try:
                prefetch_future.result()
                wait_time = time.time() - t_start
                print(
                    f"INFO: Prefetch wait time for {model_id}: {wait_time:.2f} seconds"
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                wait_time = time.time() - t_start
                print(f"WARNING: Prefetching failed for {model_id}"
                      f" after {wait_time:.2f} seconds: {e}")

        chip_idx = None
        if device_str == "tpu":
            chip_idx = allocator.allocate()
            if chip_idx is None:
                raise RuntimeError("No TPU chip available for execution.")

        print(f"\nINFO: Spawning {model_id} on {device_str.upper()} Chip"
              f" {chip_idx}...")
        try:
            duration, output_info, mse_error, compile_time, ppl_stats = (
                run_benchmark_subprocess(
                    model_id=model_id,
                    library=library,
                    pipeline_tag=pipeline_tag,
                    device_str=device_str,
                    clean_cache=clean_cache,
                    verify=verify,
                    ref_dir=args.ref_dir,
                    compile_flag=compile_flag,
                    chip_idx=chip_idx,
                ))
            return config_key, {
                "idx": idx,
                "model_id": model_id,
                "library": library,
                "device": device_str,
                "clean_cache": clean_cache,
                "compile": compile_flag,
                "status": "success",
                "duration_seconds": duration,
                "compile_time_seconds": compile_time,
                "output_info": output_info,
                "verification_mse": mse_error,
                "ppl_stats": ppl_stats,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            return config_key, {
                "idx": idx,
                "model_id": model_id,
                "library": library,
                "device": device_str,
                "clean_cache": clean_cache,
                "compile": compile_flag,
                "status": "error",
                "error_message": str(e),
            }
        finally:
            if chip_idx is not None:
                allocator.free(chip_idx)

    benchmark_futures = {}
    for device, executor in executors.items():
        for item in models_to_run[device]:
            benchmark_futures[executor.submit(benchmark_worker,
                                              item)] = item["config_key"]

    for future in concurrent.futures.as_completed(benchmark_futures):
        config_key = benchmark_futures[future]
        model_id = config_key[0]
        try:
            key, result_entry = future.result()
            results_dict[key] = result_entry

            if (result_entry["status"] == "success"
                    and result_entry.get("ppl_stats") is not None):
                target_model = result_entry["model_id"]

                # Extract CPU PPL and TPU PPL carefully to bind to the right object
                if result_entry["device"] == "cpu":
                    cpu_ppl = result_entry["ppl_stats"].get("device_ppl")
                    if cpu_ppl:
                        for _, v in results_dict.items():
                            if (v.get("model_id") == target_model
                                    and v.get("device") != "cpu"
                                    and v.get("status") == "success"):
                                dev_ppl = v.get("ppl_stats",
                                                {}).get("device_ppl")
                                if dev_ppl is not None:
                                    v["ppl_stats"]["relative_degradation"] = (
                                        abs(dev_ppl - cpu_ppl) / cpu_ppl)
                else:  # An accelerator (TPU/CUDA) just finished
                    dev_ppl = result_entry["ppl_stats"].get("device_ppl")
                    if dev_ppl is not None:
                        # We only need to find the CPU, and update ONLY ourselves
                        for _, v in results_dict.items():
                            if (v.get("model_id") == target_model
                                    and v.get("device") == "cpu"
                                    and v.get("status") == "success"):
                                cpu_ppl = v.get("ppl_stats",
                                                {}).get("device_ppl")
                                if cpu_ppl:
                                    result_entry["ppl_stats"][
                                        "relative_degradation"] = (
                                            abs(dev_ppl - cpu_ppl) / cpu_ppl)
                                break  # Found the CPU, stop looking

            if result_entry["status"] == "success":
                print(
                    "\n" + "-" * 80 + "\n"
                    f"Finished benchmark for {result_entry['model_id']} "
                    f"[{result_entry['idx']} / {total_runs}]. \n"
                    f"Status: {result_entry['status'].upper()}\n"
                    f"Device: {result_entry['device'].upper()}\n"
                    f"Duration: {result_entry['duration_seconds']:.4f} seconds\n"
                    f"Output Info: {result_entry['output_info']}"
                    "\n" + "-" * 80 + "\n")
            else:
                print(
                    "\n" + "-" * 80 + "\n"
                    f"Finished benchmark for {result_entry['model_id']} "
                    f"[{result_entry['idx']} / {total_runs}]. \n"
                    f"Status: {result_entry['status'].upper()}"
                    f"Error: {result_entry['error_message']}"
                    "\n" + "-" * 80 + "\n", )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Worker crashed for config {config_key}: {e}")
        finally:
            runs_scheduled_count[model_id] -= 1
            if runs_scheduled_count[model_id] == 0:
                cleanup_model_cache(model_id)
            write_results(args, results_dict)

    prefetch_executor.shutdown(wait=True)
    for executor in executors.values():
        executor.shutdown(wait=True)

    print("\nFinished benchmarking. Final results are available at"
          f" {args.output_json}")


if __name__ == "__main__":
    main()
