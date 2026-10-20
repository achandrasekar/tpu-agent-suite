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
"""Template schema for benchmark_tpu.py.

This file contains common PyTorch/XLA TPU compatibility patches, helpers, and
utilities. The actual model execution logic must be synthesized by implementing
the `benchmark_model` placeholder.
"""

import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from torch import nn

# ============================================================================
# Global Patches & Workarounds
# ============================================================================


def set_socket_default_timeout(timeout: int = 60) -> None:
    """Sets the default socket timeout."""
    socket.setdefaulttimeout(timeout)
    print(f"INFO: Set default socket timeout to {timeout} seconds.")


def patch_interpolate() -> None:
    """Upsample_bicubic2d bilinear swap workaround."""
    try:
        original_interpolate = F.interpolate

        def custom_interpolate(*args, **kwargs):
            if kwargs.get("mode") == "bicubic":
                print(
                    "INFO: Replacing upsample_bicubic2d with bilinear for TPU compat."
                )
                kwargs["mode"] = "bilinear"
            return original_interpolate(*args, **kwargs)

        F.interpolate = custom_interpolate
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not patch F.interpolate: {e!r}")


def monkey_patch_group_norm() -> None:
    """GroupNorm contiguous view TPU fix."""
    try:
        original_forward = nn.GroupNorm.forward

        def custom_forward(self, input):  # pylint: disable=redefined-builtin
            if input.device.type in ["tpu", "xla"]:
                if not input.is_contiguous():
                    input = input.contiguous()
            return original_forward(self, input)

        nn.GroupNorm.forward = custom_forward
        print("INFO: Applied GroupNorm contiguous input monkey-patch.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not patch GroupNorm: {e!r}")


def monkey_patch_compile() -> None:
    """Compile with dynamic=False static shape patches."""
    try:
        original_compile = torch.compile

        def custom_compile(*args, **kwargs):
            kwargs["dynamic"] = False
            print("INFO: Applying static shape monkey-patch for torch.compile")
            return original_compile(*args, **kwargs)

        torch.compile = custom_compile
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not monkey-patch torch.compile: {e!r}")


def monkey_patch_hf_hub_download() -> None:
    """Patched huggingface_hub.hf_hub_download for legacy use_auth_token and HTTPError 404 support."""
    try:
        # pylint: disable=g-import-not-at-top
        import huggingface_hub
        import huggingface_hub.errors
        import requests.exceptions
        original_hf_hub_download = huggingface_hub.hf_hub_download

        def custom_hf_hub_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            try:
                return original_hf_hub_download(*args, **kwargs)
            except huggingface_hub.errors.EntryNotFoundError as e:
                raise requests.exceptions.HTTPError(
                    f"404 Client Error: {e}") from e

        huggingface_hub.hf_hub_download = custom_hf_hub_download
        print("INFO: Patched huggingface_hub.hf_hub_download for legacy"
              " use_auth_token and HTTPError 404 support.")
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def patch_torchaudio_list_audio_backends() -> None:
    """Patches torchaudio.list_audio_backends for compatibility."""
    try:
        # pylint: disable=g-import-not-at-top
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            print(
                "INFO: Patching torchaudio.list_audio_backends for compatibility."
            )
            torchaudio.list_audio_backends = lambda: []
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def apply_global_patches() -> None:
    """Applies global compatibility patches for PyTorch/XLA TPU execution."""
    # 1. Backward Compatibility Patch (transformers.PreTrainedModel)
    try:
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):

            def get_tied_keys(self):
                val = getattr(self, "_all_tied_weights_keys_storage", None)
                if val is None:
                    val = getattr(self, "_tied_weights_keys", None)
                return val if val is not None else {}

            def set_tied_keys(self, val):
                # pylint: disable=protected-access
                self._all_tied_weights_keys_storage = val if val is not None else {}

            transformers.PreTrainedModel.all_tied_weights_keys = property(
                get_tied_keys, set_tied_keys)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not patch transformers.PreTrainedModel: {e!r}")

    patch_interpolate()
    monkey_patch_group_norm()
    monkey_patch_compile()
    patch_torchaudio_list_audio_backends()
    monkey_patch_hf_hub_download()
    set_socket_default_timeout()


# ============================================================================
# Utilities & Helpers
# ============================================================================


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


def get_device(device_str: str) -> torch.device:
    """Returns CPU, CUDA, or TPU device identifiers."""
    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "tpu":
        try:
            # pylint: disable=g-import-not-at-top
            from torch_tpu import api
            return api.tpu_device()
        except ImportError:
            return torch.device("tpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    else:
        raise ValueError(f"Unknown device: {device_str}")


def remove_weight_norm_recursive(module: nn.Module) -> None:
    """Recursively strip weight normalizations."""
    # pylint: disable=g-import-not-at-top
    from torch.nn.utils import parametrize, remove_weight_norm

    try:
        remove_weight_norm(module)
        print(f"INFO: Removed weight_norm from {module.__class__.__name__}")
    except ValueError:
        pass

    for name in ["weight", "bias"]:
        try:
            if parametrize.is_parametrized(module, name):
                parametrizations = getattr(module, "parametrizations",
                                           {}).get(name, [])
                has_weight_norm = False
                has_other = False
                for p in parametrizations:
                    class_name = p.__class__.__name__
                    if "WeightNorm" in class_name:
                        has_weight_norm = True
                    else:
                        has_other = True

                if has_weight_norm and not has_other:
                    parametrize.remove_parametrizations(
                        module, name, leave_parametrized=True)
                    print("INFO: Removed weight_norm parametrization from"
                          f" {module.__class__.__name__}.{name}")
                elif has_other:
                    print(
                        "INFO: Preserving non-weight_norm parametrizations on"
                        f" {module.__class__.__name__}.{name} (classes:"
                        f" {[p.__class__.__name__ for p in parametrizations]})"
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"DEBUG: Failed to inspect/remove parametrization {name} from"
                f" {module.__class__.__name__}: {e!r}")

    for child in module.children():
        remove_weight_norm_recursive(child)


def repair_json_string(json_str: str) -> str:
    """Regex JSON comma and syntax repairs."""
    json_str = re.sub(r"(\"[\w\-./]+\"\s*)(\s*\"[\w\-./]+\")([,\s\}]|$)",
                      r"\1,\2\3", json_str)
    json_str = re.sub(r"(\"[\w\-./]+\"\s*)(\s*\"[\w\-./]+\"\s*:)", r"\1,\2",
                      json_str)
    return json_str


def import_dependency(module_name: str, pip_name: str | None = None) -> Any:
    """Imports a module, raising an error if missing."""
    # pylint: disable=g-import-not-at-top
    import importlib

    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f'Required dependency "{pip_name}" is missing. '
            "Please provision the TPU environment using requirements-tpu.txt: "
            f"pip install -r requirements-tpu.txt (or pip install {pip_name})"
        ) from exc


def verify_inputs_bounds(model: Any, inputs: Any) -> None:
    """Checks token bounds safely, avoiding warnings on non-text models."""
    if not hasattr(model, "get_input_embeddings") or "input_ids" not in inputs:
        return
    try:
        embeddings = model.get_input_embeddings()
        if embeddings:
            embed_vocab = embeddings.weight.shape[0]
            max_token_id = inputs["input_ids"].max().item()
            if max_token_id >= embed_vocab:
                print(
                    f"WARNING: Out-of-bounds token ID detected ({max_token_id} >="
                    f" embedding vocab {embed_vocab}). Execution will crash on CPU."
                )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Failed to verify vocab bounds: {e!r}")


def prepare_patched_metadata(model_id: str) -> str | None:
    """Download and patch model metadata configs."""
    # pylint: disable=g-import-not-at-top
    from huggingface_hub import hf_hub_download

    temp_dir = tempfile.mkdtemp(prefix="hf_patched_metadata_")
    cache_dir = get_cache_dir()

    try:
        config_file = hf_hub_download(repo_id=model_id,
                                      filename="config.json",
                                      cache_dir=cache_dir)
        shutil.copy(config_file, os.path.join(temp_dir, "config.json"))

        vocab_files = [
            "tokenizer.model",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ]
        for filename in [
                "processor_config.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "generation_config.json",
        ] + vocab_files:
            try:
                f_path = hf_hub_download(repo_id=model_id,
                                         filename=filename,
                                         cache_dir=cache_dir)
                shutil.copy(f_path, os.path.join(temp_dir, filename))
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        patched = False
        if config_dict.get("model_type") == "multi_modality":
            print("INFO: Patching model_type from multi_modality to janus")
            config_dict["model_type"] = "janus"
            patched = True

        if not patched:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return temp_dir
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f"Warning: Failed to prepare patched metadata: {e!r}. Return None."
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def load_config_with_fallback(model_id: str, cache_dir: str) -> Any:
    """Loads configuration for model_id, with fallback for missing model_type."""
    # pylint: disable=g-import-not-at-top
    from transformers import AutoConfig
    try:
        return AutoConfig.from_pretrained(model_id,
                                          cache_dir=cache_dir,
                                          trust_remote_code=True)
    except ValueError as e:
        if "model_type" in str(e):
            print(
                f"INFO: Detected missing model_type in config for {model_id}. "
                "Attempting fallback...")
            try:
                from transformers.configuration_utils import PretrainedConfig

                config_dict, _ = PretrainedConfig.get_config_dict(
                    model_id, cache_dir=cache_dir)
                model_type = None
                model_id_lower = model_id.lower()
                if "bert" in model_id_lower:
                    model_type = "bert"
                elif "gpt" in model_id_lower:
                    model_type = "gpt2"
                elif "t5" in model_id_lower:
                    model_type = "t5"

                if model_type:
                    print(
                        f"INFO: Guessed model_type: '{model_type}' for {model_id}."
                    )
                    return AutoConfig.for_model(model_type, **config_dict)
            except Exception as inner_e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Fallback config load failed: {inner_e!r}")
        raise e


def load_model_and_processor(
    model_id: str,
    device: torch.device | str,
    model_class_factory: Any,
    processor_class_factory: Any = None,
    use_bfloat16: bool = True,
) -> Any:
    """Load configuration, tokenizer, and weights.

  Args:
    model_id: The Hugging Face model ID.
    device: The torch.device to load the model on.
    model_class_factory: The transformers AutoModel class (e.g.,
        AutoModelForCausalLM).
    processor_class_factory: Optional. The transformers AutoProcessor or
      AutoTokenizer class.
    use_bfloat16: Whether to load the model with torch.bfloat16 dtype.

  Returns:
    If processor_class_factory is provided, returns a tuple of (model,
    processor). Otherwise, returns just the model.
  """
    # pylint: disable=g-import-not-at-top
    from transformers import AutoConfig

    temp_dir = prepare_patched_metadata(model_id)
    cache_dir = get_cache_dir()

    config = None
    processor = None

    if temp_dir:
        try:
            config = AutoConfig.from_pretrained(temp_dir)
            if processor_class_factory:
                try:
                    processor = processor_class_factory.from_pretrained(
                        temp_dir, trust_remote_code=True)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"Warning: Failed to load config from patched temp dir: {e!r}")

    if processor is None and processor_class_factory:
        processor = processor_class_factory.from_pretrained(
            model_id, trust_remote_code=True)

    if config is None:
        config = load_config_with_fallback(model_id, cache_dir)

    kwargs = {
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "ignore_mismatched_sizes": True,
    }
    if "gemma" in model_id.lower() or (config and getattr(
            config, "model_type", None) in ["gemma", "gemma2"]):
        print("INFO: Disabling low_cpu_mem_usage for Gemma model to bypass"
              " meta-device weight-tying limitations.")
        kwargs["low_cpu_mem_usage"] = False
    if config:
        kwargs["config"] = config
    if use_bfloat16:
        kwargs["torch_dtype"] = torch.bfloat16

    print(f"Loading model weights for {model_id}...")
    model = model_class_factory.from_pretrained(model_id, **kwargs)
    if hasattr(device, "type") and device.type in ["tpu", "xla"]:
        remove_weight_norm_recursive(model)
    model = model.to(device)

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if processor_class_factory:
        return model, processor
    return model


def cast_inputs_to_model_dtype(inputs: Any, model: nn.Module,
                               device: torch.device | str) -> Any:
    """Push inputs to device and cast to model precision."""
    model_dtype = torch.float32
    try:
        has_bf16 = False
        has_fp16 = False
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                has_bf16 = True
                break
            elif p.dtype == torch.float16:
                has_fp16 = True
        if has_bf16:
            model_dtype = torch.bfloat16
        elif has_fp16:
            model_dtype = torch.float16
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    if hasattr(inputs, "items"):
        # NOTE: build a new dict rather than mutating `inputs` in place. This
        # function is called on the same inputs object for both the primary
        # accelerator run and (inside verify_output) a separate CPU-reference
        # run; in-place mutation here silently moved the caller's tensors to
        # "cpu", causing device-mismatch crashes in code that reused `inputs`
        # afterward (e.g. calculate_perplexity running on the accelerator).
        new_inputs = dict(inputs)
        for k, v in new_inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    new_inputs[k] = v.to(device).to(model_dtype)
                else:
                    new_inputs[k] = v.to(device)
        return new_inputs
    elif isinstance(inputs, torch.Tensor):
        if inputs.is_floating_point():
            inputs = inputs.to(device).to(model_dtype)
        else:
            inputs = inputs.to(device)
    return inputs


def get_ref_path(model_id: str, ref_dir: str) -> str:
    """Generates a safe filename for the reference tensor."""
    os.makedirs(ref_dir, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)
    return os.path.join(ref_dir, f"ref_{safe_name}.pt")


def extract_raw_tensor(outputs: Any) -> torch.Tensor:
    """Extracts the primary numerical tensor from various model output formats."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if (hasattr(outputs, "logits_per_image")
            and outputs.logits_per_image is not None):
        return outputs.logits_per_image
    if (hasattr(outputs, "logits_per_audio")
            and outputs.logits_per_audio is not None):
        return outputs.logits_per_audio
    if (hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None):
        return outputs.last_hidden_state
    if hasattr(outputs, "sample") and outputs.sample is not None:
        return outputs.sample
    if hasattr(outputs, "images") and outputs.images is not None:
        return outputs.images
    if isinstance(outputs, dict):
        for key in ["logits", "last_hidden_state", "sample", "embeddings"]:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(outputs, (list, tuple)):
        if len(outputs) > 0:
            return extract_raw_tensor(outputs[0])
    raise TypeError(
        "Could not extract a comparable tensor from model output of type"
        f" {type(outputs)}")


def run_benchmark_pass(
    model: Any,
    inputs: Any,
    forward_fn: Any,
    postprocess_fn: Any,
    device: torch.device | str | None = None,
) -> tuple[float, str, Any, float]:
    """Handles timing, execution, and warmup for all model classes.

  Args:
    model: The model to benchmark.
    inputs: The inputs to feed to the model.
    forward_fn: A function that takes the model and inputs and returns the
      model's outputs.
    postprocess_fn: A function that takes the model's outputs and returns a
      string with relevant information.
    device: The device on which the benchmark is running.

  Returns:
    A tuple containing:
      - duration (float): The time taken for the benchmark execution in seconds.
      - output_info (str): A string containing post-processed information
        derived from the model's outputs.
      - outputs (Any): The raw outputs from the model forward pass.
      - warmup_time (float): The time taken for the warmup execution in seconds.
  """

    def _synchronize(dev: torch.device | str | None = None):
        if dev is None:
            return
        dev_str = str(dev).lower()
        if "cuda" in dev_str and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif "tpu" in dev_str:
            try:
                # pylint: disable=g-import-not-at-top
                from torch_tpu._internal import \
                    sync as tpu_sync  # pyrefly: ignore[missing-import]

                tpu_sync.synchronize(wait=True)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Could not synchronize TPU: {e!r}")

    print("Running warmup...")
    _synchronize(device)
    t_warmup_0 = time.time()
    with torch.no_grad():
        _ = forward_fn(model, inputs)
    _synchronize(device)
    t_warmup_1 = time.time()
    warmup_time = t_warmup_1 - t_warmup_0

    print("Running benchmark...")
    _synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = forward_fn(model, inputs)
    _synchronize(device)
    t1 = time.time()

    output_info = postprocess_fn(outputs)
    return t1 - t0, output_info, outputs, warmup_time


def prepare_inputs_for_cpu(inputs: Any) -> Any:
    """Safely clones and converts input tensors to CPU float32 precision."""
    if isinstance(inputs, dict):
        res = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.cpu().float() if v.is_floating_point() else v.cpu()
            else:
                res[k] = v
        return res
    elif isinstance(inputs, torch.Tensor):
        return inputs.cpu().float() if inputs.is_floating_point(
        ) else inputs.cpu()
    return inputs


def calculate_perplexity(
    model: Any,
    inputs: Any,
) -> tuple[float, float | None]:
    """Computes model Perplexity."""
    # pylint: disable=g-import-not-at-top
    import math

    loss_inputs = dict(inputs)
    if "labels" not in loss_inputs and "input_ids" in loss_inputs:
        loss_inputs["labels"] = loss_inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**loss_inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return 0.0, None
        current_ppl = math.exp(loss.item())

    return current_ppl, None


def verify_output(
    device_outputs: Any,
    model: Any,
    inputs: Any,
    forward_fn: Any,
    model_id: str,
    device: torch.device,
    ref_dir: str = ".benchmark_refs",
) -> float | None:
    """Saves CPU reference or loads it and computes MSE error against device output."""
    device_tensor = extract_raw_tensor(device_outputs)
    ref_path = get_ref_path(model_id, ref_dir)

    if device.type == "cpu":
        # We are running on CPU, so save the output as reference.
        print(f"INFO: Saving CPU reference tensor to {ref_path}...")
        try:
            torch.save(device_tensor.cpu(), ref_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Warning: Failed to save CPU reference: {e!r}")
        return None

    # We are running on accelerator, try to compare against reference.
    cpu_tensor = None
    if os.path.exists(ref_path):
        print(f"INFO: Loaded cached CPU reference from {ref_path}")
        try:
            cpu_tensor = torch.load(ref_path, map_location="cpu")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"Warning: Failed to load cached reference: {e!r}. Re-generating."
            )
            cpu_tensor = None

    if cpu_tensor is None:
        print(
            "WARNING: Cached reference not found. Generating on CPU (this may be"
            " slow)...")
        try:
            # Save original device to restore later.
            original_device = device
            model_cpu = model.to("cpu").to(torch.float32)
            if hasattr(model_cpu, "tie_weights"):
                try:
                    model_cpu.tie_weights()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            inputs_cpu = cast_inputs_to_model_dtype(inputs, model_cpu, "cpu")
            with torch.no_grad():
                cpu_outputs = forward_fn(model_cpu, inputs_cpu)
            cpu_tensor = extract_raw_tensor(cpu_outputs).cpu()

            print(f"INFO: Saving newly generated CPU reference to {ref_path}")
            torch.save(cpu_tensor, ref_path)

            # Move model back to original device.
            model.to(original_device)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"ERROR: Failed to generate reference on CPU: {e!r}")
            return None

    # Discrete token ID output accuracy
    if not device_tensor.is_floating_point():
        exact_match = ((
            cpu_tensor.cpu() == device_tensor.cpu()).float().mean().item())
        return 0.0 if exact_match == 1.0 else (1.0 - exact_match)

    # Compute MSE.
    try:
        dev_t = device_tensor.cpu().float()
        cpu_t = cpu_tensor.float()
        if cpu_t.shape != dev_t.shape:
            if cpu_t.ndim == dev_t.ndim:
                min_shape = [
                    min(s1, s2) for s1, s2 in zip(cpu_t.shape, dev_t.shape)
                ]
                slices = tuple(slice(0, s) for s in min_shape)
                cpu_t = cpu_t[slices]
                dev_t = dev_t[slices]
            else:
                return None
        return F.mse_loss(cpu_t, dev_t).item()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"ERROR: Failed to compute MSE: {e!r}")
        return None


def get_dummy_inputs_for_config(
        config: Any,
        device: torch.device | str) -> dict[str, Any] | torch.Tensor:
    """Generates dummy inputs for a given model config."""
    if hasattr(config, "vocab_size"):
        return {
            "input_ids": torch.ones((1, 16), dtype=torch.long, device=device)
        }
    if hasattr(config, "hidden_size") and not hasattr(config, "vocab_size"):
        return {
            "pixel_values":
            torch.ones((1, 3, 224, 224), dtype=torch.float32, device=device)
        }
    if ("audio" in str(type(config)).lower()
            or "speech" in str(type(config)).lower()):
        return {
            "input_values":
            torch.ones((1, 16000), dtype=torch.float32, device=device)
        }
    return torch.ones((1, 10), dtype=torch.long, device=device)


# ============================================================================
# Model Benchmark Placeholder (To be implemented dynamically)
# ============================================================================


def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    library: str = "transformers",
    pipeline_tag: str = "text-generation",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict[str, Any] | None]:
    """Placeholder: Implement bespoke loading and execution logic here."""
    # Coder agent will replace this function using SKILL.md recipes.
    raise NotImplementedError("Bespoke loader not implemented.")


# ============================================================================
# Main Dispatcher
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark HF models.")
    parser.add_argument("--model", type=str, required=True, help="HF model ID")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "tpu", "cuda"],
        required=True,
        help="Device to run on",
    )
    parser.add_argument("--library",
                        type=str,
                        default="transformers",
                        help="HF Library Tag")
    parser.add_argument(
        "--pipeline_tag",
        type=str,
        default="text-generation",
        help="HF Pipeline Tag",
    )
    parser.add_argument(
        "--cleanup_cache",
        action="store_true",
        help="Whether to cleanup model cache after benchmark",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable model output correctness verification.",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default=".benchmark_refs",
        help="Directory to save/load verification reference tensors.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional file path to save benchmark results as JSON.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model",
    )
    return parser.parse_args()


def cleanup_model_cache(model_id: str):
    """Removes downloaded Hugging Face model cache to reclaim disk space."""
    cache_dir = get_cache_dir()
    # Hugging Face names cache folders as 'models--{author}--{model_name}'
    safe_folder_name = "models--" + model_id.replace("/", "--")
    model_cache_path = os.path.join(cache_dir, safe_folder_name)

    if cache_is_shared():
        print(f"INFO: Keeping {model_cache_path}: the cache is shared "
              "(HF_HUB_CACHE), not this pod's to clean up")
        return
    if os.path.exists(model_cache_path):
        print(f"INFO: Cleaning up model cache directory: {model_cache_path}")
        shutil.rmtree(model_cache_path, ignore_errors=True)


def main() -> None:
    """Main device resolution and task routing dispatcher."""
    args = parse_args()
    apply_global_patches()

    try:
        device = get_device(args.device)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"ERROR: Failed to resolve device {args.device}: {e!r}",
              file=sys.stderr)
        sys.exit(1)

    try:
        duration, output_info, mse_error, compile_time, ppl_stats = benchmark_model(
            args.model,
            device,
            verify=not args.no_verify,
            ref_dir=args.ref_dir,
            library=args.library,
            pipeline_tag=args.pipeline_tag,
            compile_flag=args.compile,
        )
        result_dict = {
            "duration": duration,
            "output_info": output_info,
            "mse": mse_error,
            "compile_time": compile_time,
            "ppl_stats": ppl_stats,
        }
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result_dict, f, indent=2)
        else:
            print(f"BENCHMARK_RESULT: {duration:.4f} seconds")
            if compile_time is not None:
                print(f"COMPILATION_TIME: {compile_time:.4f} seconds")
            print(f"OUTPUT_INFO: {output_info}")
            if mse_error is not None:
                print(f"VERIFICATION_MSE: {mse_error:.8f}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pylint: disable=g-import-not-at-top
        import traceback
        traceback.print_exc()
        print(f"ERROR: Benchmark failed: {e!r}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.cleanup_cache:
            cleanup_model_cache(args.model)


if __name__ == "__main__":
    main()
