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
"""Captures the numbers one run of a model produces, for parity comparison.

Run twice, once in each role:

  reference   The model's own plain-PyTorch path, on CPU in fp32, with no
              patches applied. If the model loads and runs here, whatever
              executed was the model's built-in code -- a CUDA kernel cannot
              run on a CPU-only host -- so the reference is independent of
              the rewrite under test.
  candidate   The unwired code on TPU, loading the same patch module the
              inference server loads.

Both roles read the same probe: a JSON file describing it and a safetensors
file holding its inputs byte-for-byte. Neither
role generates text: the fixed token sequence is fed to both (teacher forcing)
so every position is compared against identical inputs. Free-running
generation would let one flipped token change every token after it, which
says nothing about whether the rewrite is correct.

`compare_parity.py` reads the two captures and does the arithmetic.

## Model sources

Everything model-specific lives behind `ModelAdapter`: how to build the
model, how to turn a probe entry into forward arguments, and how to find the
comparable output. The probe file names which adapter to use in its `source`
block, so the rest of this script never assumes Hugging Face.

`HuggingFaceAdapter` ("hf_pretrained") ships here and already covers a hub id
*or* a local directory, any auto class, and `trust_remote_code` for custom
model code. A `gs://` checkpoint folder is copied to the job's disk first, by
`fetch_weights.py` passed beside this script: the probe copies only the
tokenizer, each capture the whole model, and the probe keeps naming the
`gs://` folder so both captures read the same one. Other sources -- a bare
checkpoint paired with a code path, a
Python entrypoint returning a built model, a downloaded repo -- are not
implemented: add a class with the four `ModelAdapter` methods, call
`register_adapter`, and give the probe a matching `source.kind`. Nothing else
in this file, and nothing in `compare_parity.py`, needs to change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# Packages that only run on CUDA. Their presence in a completed CPU reference
# run is worth recording: the run proves the code path taken was CPU code,
# but a loaded module is a hint about what the model wanted to use.
CUDA_ONLY_PACKAGES = (
    "flash_attn",
    "mamba_ssm",
    "triton",
    "apex",
    "deepspeed",
    "xformers",
)

REFERENCE = "reference"
CANDIDATE = "candidate"

# The one model source implemented here. Others register their own kind.
HF_PRETRAINED = "hf_pretrained"

# Both runs load the model in this dtype unless the probe says otherwise.
# Matching the candidate's dtype on the reference side removes precision
# from the comparison, leaving the rewrite (and the device) as the only
# things that can differ. Measured on mamba-130m, running the reference in
# fp32 instead contributed up to 2.45% of apparent difference that had
# nothing to do with any rewrite.
DEFAULT_DTYPE = "bfloat16"

# Inputs live in a safetensors file beside the probe JSON, keyed
# "<probe_id>/<input_name>". Every input travels the same way whatever its
# dtype: the file stores dtype and shape natively and byte-for-byte, so
# "both runs got identical inputs" holds by construction rather than by an
# argument about decimal round-tripping.
TENSOR_SUFFIX = ".safetensors"

# A model in a GCS folder rather than on the hub. The job copies it to its
# own disk with fetch_weights.py, which travels beside this script.
GCS_SCHEME = "gs://"


class CaptureError(Exception):
    """Raised when a capture cannot be produced or would not be meaningful."""


def local_checkpoint(path: str, small_only: bool = False) -> str:
    """A local copy of the gs:// checkpoint at `path`.

  Args:
    path: The gs:// folder holding the checkpoint.
    small_only: Copy only the config and tokenizer files, for a job that
      only tokenizes.

  Returns:
    The local folder the checkpoint was copied to.

  Raises:
    CaptureError: If fetch_weights.py is not beside this script, or the copy
      failed.
  """
    try:
        # pylint: disable=g-import-not-at-top
        import fetch_weights
    except ImportError as e:
        raise CaptureError(
            f"{path} is a gs:// checkpoint: run this job with fetch_weights.py "
            "beside it (--input <fetch_weights.py>:fetch_weights.py) so it can "
            "copy the checkpoint") from e
    try:
        return fetch_weights.fetch_gcs(path, small_only=small_only)
    except (OSError, RuntimeError, ValueError) as e:
        raise CaptureError(f"could not copy {path}: {e}") from e


# ============================================================================
# Probe file
# ============================================================================


def probe_tensors_path(probe_path: str) -> Path:
    """Returns the tensor file that belongs to a probe JSON."""
    path = Path(probe_path)
    return path.with_suffix(TENSOR_SUFFIX)


def probe_digest(probe: dict[str, Any],
                 tensors_path: str | Path | None = None) -> str:
    """Hashes a probe so two captures can prove they used the same inputs.

  The tensor file is hashed with the JSON. Hashing only the text would leave
  the guarantee hollow: two runs could name the same inputs and be handed
  different bytes.

  Args:
    probe: The parsed probe file.
    tensors_path: The companion tensor file. Skipped when None, which is the
      case for a probe held in memory.

  Returns:
    A short hex digest.

  Raises:
    CaptureError: If the tensor file is named but missing.
  """
    digest = hashlib.sha256()
    digest.update(
        json.dumps(probe, sort_keys=True, separators=(",", ":")).encode())
    if tensors_path is not None:
        path = Path(tensors_path)
        if not path.is_file():
            raise CaptureError(f"probe tensor file not found: {path}")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def load_probe(path: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Loads a probe JSON and its companion tensor file.

  Args:
    path: Path to the probe JSON.

  Returns:
    A tuple of the parsed probe and its input tensors.

  Raises:
    CaptureError: If either file is missing, malformed, or incomplete.
  """
    probe_path = Path(path)
    if not probe_path.is_file():
        raise CaptureError(f"probe file not found: {path}")
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CaptureError(f"{path} is not valid JSON: {e}") from e

    if "model_id" not in probe and "source" not in probe:
        raise CaptureError("probe file needs a 'source' block (or a legacy "
                           "'model_id')")
    for field in ("modules", "probes"):
        if field not in probe:
            raise CaptureError(f"probe file is missing {field!r}")
    if not probe["probes"]:
        raise CaptureError("probe file lists no probes")

    tensors_path = probe_tensors_path(path)
    if not tensors_path.is_file():
        raise CaptureError(
            f"probe tensor file not found: {tensors_path}. A probe is the "
            "JSON plus its tensors; stage and copy them together.")
    tensors = load_file(str(tensors_path))

    for entry in probe["probes"]:
        if "id" not in entry:
            raise CaptureError("every probe needs an 'id'")
        names = entry.get("inputs")
        if not names:
            raise CaptureError(
                f"probe {entry['id']} declares no 'inputs'; build the file "
                "with --make_probe so both runs feed the model identical "
                "values")
        missing = [
            name for name in names if f"{entry['id']}/{name}" not in tensors
        ]
        if missing:
            raise CaptureError(
                f"probe {entry['id']} declares input(s) {missing} that "
                f"{tensors_path} does not hold")
    return probe, tensors


def make_probe(
    source: dict[str, Any],
    modules: list[str],
    prompts: list[str] | None = None,
    input_ids: list[list[int]] | None = None,
    max_length: int = 64,
    dtype: str = DEFAULT_DTYPE,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Builds a probe, resolving prompts to token ids once, up front.

  The ids are frozen into the tensor file rather than re-derived per run, so
  the two captures cannot drift apart through tokenizer differences.
  Explicit `input_ids` skip tokenization entirely, which is how a model with
  no Hugging Face tokenizer gets a probe.

  Args:
    source: The model source block.
    modules: Qualified names of the modules both runs will hook.
    prompts: Prompts to tokenize, if any.
    input_ids: Ready-made token id sequences, if any.
    max_length: Truncation length for prompts.
    dtype: The dtype both runs should load the model in. Keeping the two
      sides on one dtype is what makes the rewrite the only thing the
      comparison is measuring.

  Returns:
    A tuple of the probe dict and its input tensors.

  Raises:
    CaptureError: If neither prompts nor input ids were given.
  """
    entries: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}

    def add(ids: list[int], prompt: str | None) -> None:
        entry_id = f"p{len(entries)}"
        entries.append({
            "id": entry_id,
            "prompt": prompt,
            "inputs": ["input_ids"],
            "n_tokens": len(ids),
        })
        tensors[f"{entry_id}/input_ids"] = torch.tensor([ids],
                                                        dtype=torch.long)

    for ids in input_ids or []:
        add(list(ids), None)
    if prompts:
        adapter = get_adapter(source["kind"])
        for prompt in prompts:
            add(adapter.tokenize(source, prompt, max_length), prompt)
    if not entries:
        raise CaptureError(
            "a probe needs at least one --prompt or --input_ids")

    return {
        "model_id": source.get("path"),
        "source": source,
        "dtype": dtype,
        "modules": list(modules),
        "probes": entries,
    }, tensors


def write_probe(path: str, probe: dict[str, Any],
                tensors: dict[str, torch.Tensor]) -> Path:
    """Writes the probe JSON and its tensor file. Returns the tensor path."""
    Path(path).write_text(json.dumps(probe, indent=2), encoding="utf-8")
    tensors_path = probe_tensors_path(path)
    save_file(tensors, str(tensors_path))
    return tensors_path


# ============================================================================
# Model loading
# ============================================================================


def get_device(role: str, override: str | None) -> torch.device:
    """Resolves the torch device for a role."""
    name = override or ("cpu" if role == REFERENCE else "tpu")
    if name == "tpu":
        try:
            # pylint: disable=g-import-not-at-top
            from torch_tpu import api

            return api.tpu_device()
        except ImportError:
            return torch.device("tpu")
    return torch.device(name)


def load_patch_module(path: str) -> types.ModuleType:
    """Imports the unwiring patch module the inference server also imports.

  Args:
    path: Path to the generated patch module.

  Returns:
    The imported module, after calling its `apply()`.

  Raises:
    CaptureError: If the module cannot be imported or has no `apply()`.
  """
    module_path = Path(path)
    if not module_path.is_file():
        raise CaptureError(f"patch module not found: {path}")
    spec = importlib.util.spec_from_file_location("server_patches",
                                                  module_path)
    if spec is None or spec.loader is None:
        raise CaptureError(f"cannot import patch module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "apply"):
        raise CaptureError(f"{path} does not define apply()")
    module.apply()
    return module


def check_reference_host() -> None:
    """Refuses to build a reference on a host that could hide a CUDA path.

  Raises:
    CaptureError: If CUDA is visible to this process.
  """
    if torch.cuda.is_available():
        raise CaptureError(
            "CUDA is visible to this process, so the reference could load a "
            "CUDA kernel instead of the model's plain-PyTorch path. Re-run "
            "with CUDA_VISIBLE_DEVICES= set to empty.")


def resolve_source(probe: dict[str, Any]) -> dict[str, Any]:
    """Returns the probe's `source` block, or one inferred from `model_id`.

  Probes that predate the `source` block name only a Hugging Face id, so that
  spelling keeps working.

  Args:
    probe: The parsed probe file.

  Returns:
    A source dict carrying at least `kind` and `path`.
  """
    source = dict(probe.get("source") or {})
    source.setdefault("kind", HF_PRETRAINED)
    source.setdefault("path", probe.get("model_id"))
    return source


class ModelAdapter:
    """Everything that depends on where a model comes from.

  Subclass this for a source the shipped adapter cannot load -- a bare
  checkpoint, a Python entrypoint, a downloaded repo -- and register it. The
  capture flow calls these four methods and nothing else, so an adapter is
  the complete contract for supporting a new kind of model.
  """

    kind = ""

    def load(self, source: dict[str, Any], dtype: torch.dtype,
             device: torch.device) -> Any:
        """Builds the model, in eval mode, on `device`."""
        raise NotImplementedError

    def make_inputs(self, entry: dict[str, Any], tensors: dict[str,
                                                               torch.Tensor],
                    device: torch.device) -> dict[str, Any]:
        """Turns one probe entry into keyword arguments for the forward.

    Args:
      entry: The probe entry, naming its inputs.
      tensors: The probe's tensors, keyed "<probe_id>/<input_name>".
      device: Where the model lives.
    """
        raise NotImplementedError

    def extract_logits(self, outputs: Any) -> torch.Tensor | None:
        """Picks the comparable output tensor out of a forward result."""
        raise NotImplementedError

    def tokenize(self, source: dict[str, Any], prompt: str,
                 max_length: int) -> list[int]:
        """Turns a prompt into token ids, for --make_probe.

    An adapter whose models take something other than tokens can leave this
    unimplemented and have probes carry explicit `inputs` instead.
    """
        raise NotImplementedError(
            f"{self.kind} cannot tokenize prompts; give the probe explicit "
            "input ids or inputs instead")


class HuggingFaceAdapter(ModelAdapter):
    """Loads through `transformers`, from a hub id or a local directory.

  `source` keys:
    path              hub id or local directory (required)
    auto_class        e.g. AutoModelForCausalLM (default), AutoModel,
                      AutoModelForSeq2SeqLM
    trust_remote_code run the repo's own modeling code (default False)
    revision          pin a specific commit
  """

    kind = HF_PRETRAINED

    def _kwargs(self, source: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(source.get("trust_remote_code", False)),
        }
        if source.get("revision"):
            kwargs["revision"] = source["revision"]
        return kwargs

    def load(self, source: dict[str, Any], dtype: torch.dtype,
             device: torch.device) -> Any:
        # pylint: disable=g-import-not-at-top
        import transformers

        path = source.get("path")
        if not path:
            raise CaptureError("source is missing 'path'")
        class_name = source.get("auto_class", "AutoModelForCausalLM")
        auto_class = getattr(transformers, class_name, None)
        if auto_class is None:
            raise CaptureError(
                f"transformers has no {class_name!r}; set source.auto_class "
                "to one that exists")

        # `torch_dtype` rather than `dtype`, matching the rest of this
        # package (torch_benchmark_coder/scripts/benchmark_tpu.py and the
        # fastapi_server references); switch them together on upgrade.
        kwargs = self._kwargs(source)
        try:
            model = auto_class.from_pretrained(path,
                                               torch_dtype=dtype,
                                               attn_implementation="eager",
                                               **kwargs)
        except (ValueError, TypeError) as e:
            # Not every architecture exposes the eager switch. Fall back and
            # let the manifest record which path was actually taken.
            print(f"INFO: eager attention unavailable ({e!r}); using the "
                  "model's default attention path")
            model = auto_class.from_pretrained(path,
                                               torch_dtype=dtype,
                                               **kwargs)
        return model.to(device).eval()

    def make_inputs(self, entry: dict[str, Any], tensors: dict[str,
                                                               torch.Tensor],
                    device: torch.device) -> dict[str, Any]:
        # Each declared input is passed to the forward under its own name,
        # so a text model gets input_ids and a vision model gets
        # pixel_values without this code knowing the difference.
        return {
            name: tensors[f"{entry['id']}/{name}"].to(device)
            for name in entry["inputs"]
        }

    def extract_logits(self, outputs: Any) -> torch.Tensor | None:
        logits = getattr(outputs, "logits", None)
        return logits if logits is not None else first_tensor(outputs)

    def tokenize(self, source: dict[str, Any], prompt: str,
                 max_length: int) -> list[int]:
        # pylint: disable=g-import-not-at-top
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source["path"],
                                                  **self._kwargs(source))
        encoded = tokenizer(prompt,
                            return_tensors=None,
                            truncation=True,
                            max_length=max_length)
        return list(encoded["input_ids"])


_ADAPTERS: dict[str, ModelAdapter] = {}


def register_adapter(adapter: ModelAdapter) -> None:
    """Makes an adapter available to probes naming its `kind`."""
    _ADAPTERS[adapter.kind] = adapter


def get_adapter(kind: str) -> ModelAdapter:
    """Looks up a registered adapter.

  Args:
    kind: The `source.kind` named by a probe.

  Returns:
    The adapter.

  Raises:
    CaptureError: If no adapter is registered under that kind.
  """
    if kind not in _ADAPTERS:
        raise CaptureError(
            f"no adapter registered for source kind {kind!r}; known kinds "
            f"are {sorted(_ADAPTERS)}")
    return _ADAPTERS[kind]


register_adapter(HuggingFaceAdapter())


def attention_path(model: Any) -> str | None:
    """Reports which attention implementation the loaded model settled on."""
    config = getattr(model, "config", None)
    return getattr(config, "_attn_implementation", None)


def loaded_cuda_packages() -> list[str]:
    """Lists CUDA-only packages that ended up imported in this process."""
    return [name for name in CUDA_ONLY_PACKAGES if name in sys.modules]


# ============================================================================
# Capture
# ============================================================================


def first_tensor(value: Any) -> torch.Tensor | None:
    """Digs the first tensor out of a module output of unknown shape."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def resolve_modules(model: Any, names: list[str]) -> dict[str, Any]:
    """Looks up each hooked module by qualified name.

    A name missing on one side and present on the other would silently drop a
    patched module from the comparison, so this is a hard failure.

    Args:
        model: The loaded model.
        names: Qualified module names.

    Returns:
        A mapping of name to module.

    Raises:
        CaptureError: If any name does not resolve.
    """
    resolved = {}
    for name in names:
        try:
            resolved[name] = model.get_submodule(name)
        except AttributeError as e:
            available = [n for n, _ in model.named_modules() if n][:10]
            raise CaptureError(
                f"module {name!r} does not exist on this model; the first "
                f"few available names are {available}") from e
    return resolved


def _detach(tensor: torch.Tensor) -> torch.Tensor:
    """Brings a tensor to CPU in a form safetensors will accept."""
    return tensor.detach().to("cpu").contiguous().clone()


def synchronize(device: torch.device) -> None:
    """Waits for lazily-executed accelerator work before reading tensors."""
    if device.type != "tpu":
        return
    try:
        # pylint: disable=g-import-not-at-top
        from torch_tpu._internal import sync  # pyrefly: ignore[missing-import]

        sync.synchronize(wait=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: could not synchronize TPU: {e!r}")


def run_capture(
    model: Any,
    probe: dict[str, Any],
    probe_tensors: dict[str, torch.Tensor],
    device: torch.device,
    adapter: ModelAdapter | None = None,
) -> dict[str, torch.Tensor]:
    """Runs one forward pass per probe and collects the tensors to compare.

    Args:
        model: The loaded model.
        probe: The parsed probe file.
        device: Device the model sits on.
        adapter: Source adapter. Defaults to the Hugging Face one.

    Returns:
        A mapping of tensor key to CPU tensor, keyed as `compare_parity` expects.

    Raises:
        CaptureError: If a hooked module produced nothing comparable.
    """
    adapter = adapter or get_adapter(HF_PRETRAINED)
    module_names = list(probe["modules"])
    modules = resolve_modules(model, module_names)
    captured: dict[str, torch.Tensor] = {}
    seen: dict[str, torch.Tensor] = {}

    def hook_for(name):

        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is not None:
                seen[name] = tensor

        return hook

    handles = [
        module.register_forward_hook(hook_for(name))
        for name, module in modules.items()
    ]
    try:
        for entry in probe["probes"]:
            seen.clear()
            inputs = adapter.make_inputs(entry, probe_tensors, device)
            with torch.no_grad():
                outputs = model(**inputs)
            synchronize(device)

            logits = adapter.extract_logits(outputs)
            if logits is None:
                raise CaptureError(
                    f"probe {entry['id']} produced no logits to compare")
            captured[f"{entry['id']}/logits"] = _detach(logits)

            missing = [name for name in module_names if name not in seen]
            if missing:
                raise CaptureError(
                    f"probe {entry['id']}: hooked module(s) {missing} did not "
                    "run during the forward pass, so there is nothing to "
                    "compare for them")
            for name in module_names:
                captured[f"{entry['id']}/module/{name}"] = _detach(seen[name])
    finally:
        for handle in handles:
            handle.remove()
    return captured


def build_manifest(
    role: str,
    probe: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    independent: bool,
    model: Any = None,
    patch_module: types.ModuleType | None = None,
    tensors_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assembles the manifest written beside the tensors."""
    manifest: dict[str, Any] = {
        "role":
        role,
        "model_id":
        probe["model_id"],
        "device":
        device.type,
        "dtype":
        str(dtype).removeprefix("torch."),
        "source":
        resolve_source(probe),
        "probe_sha256":
        probe_digest(probe, tensors_path),
        "probes": [{
            "id": entry["id"],
            "prompt": entry.get("prompt"),
            "inputs": list(entry.get("inputs", [])),
            "n_tokens": entry.get("n_tokens"),
        } for entry in probe["probes"]],
        "independent_reference":
        independent,
        "modules":
        list(probe["modules"]),
        "evidence": {
            "patches_applied": patch_module is not None,
            "cuda_available": torch.cuda.is_available(),
            "torch_tpu_imported": "torch_tpu" in sys.modules,
            "cuda_packages_loaded": loaded_cuda_packages(),
            "attn_implementation": attention_path(model),
        },
        "versions": {
            "torch": torch.__version__,
        },
    }
    if patch_module is not None:
        manifest["applied_playbooks"] = list(
            getattr(patch_module, "APPLIED_PLAYBOOKS", []))
    try:
        # pylint: disable=g-import-not-at-top
        import transformers

        manifest["versions"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    return manifest


def write_capture(prefix: str, manifest: dict[str, Any],
                  tensors: dict[str, torch.Tensor]) -> None:
    """Writes `<prefix>.json` and `<prefix>.safetensors`."""
    Path(f"{prefix}.json").write_text(json.dumps(manifest, indent=2),
                                      encoding="utf-8")
    save_file(tensors, f"{prefix}.safetensors")


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture one run of a model for parity comparison.")
    parser.add_argument("--role", choices=[REFERENCE, CANDIDATE])
    parser.add_argument("--probe", help="Path to the shared probe file.")
    parser.add_argument("--out", help="Output prefix for the capture files.")
    parser.add_argument(
        "--patches",
        default=None,
        help="Generated unwiring patch module. Candidate runs load this. A "
        "reference that needs it is not independent, and is marked so.",
    )
    parser.add_argument("--device", default=None, help="Device override.")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Override the dtype the probe names. Use this when a model has "
        "no working bf16 path on CPU; the comparison then carries a "
        "precision difference as well as the rewrite.",
    )
    parser.add_argument("--make_probe",
                        action="store_true",
                        help="Write a probe file instead of capturing.")
    parser.add_argument(
        "--model",
        default=None,
        help="For --make_probe: hub id or local directory of the model.")
    parser.add_argument(
        "--source_kind",
        default=HF_PRETRAINED,
        help="For --make_probe: which registered adapter loads the model.")
    parser.add_argument(
        "--auto_class",
        default=None,
        help="For --make_probe: transformers auto class, when the model is "
        "not a causal LM.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="For --make_probe: let the repo's own modeling code run.")
    parser.add_argument(
        "--input_ids",
        action="append",
        default=None,
        help="For --make_probe: a comma-separated token id sequence, in "
        "place of a prompt. Repeatable.")
    parser.add_argument("--prompt",
                        action="append",
                        default=None,
                        help="Prompt for --make_probe. Repeatable.")
    parser.add_argument("--modules",
                        default="",
                        help="Comma-separated module names for --make_probe.")
    parser.add_argument("--max_length",
                        type=int,
                        default=64,
                        help="Truncation length for --make_probe.")
    parser.add_argument(
        "--probe_dtype",
        default=DEFAULT_DTYPE,
        help="For --make_probe: the dtype both runs will load the model in.")
    return parser.parse_args(argv)


def run_make_probe(args: argparse.Namespace) -> int:
    """Handles the --make_probe mode."""
    if not args.model or not args.out:
        raise CaptureError("--make_probe needs --model and --out")
    source: dict[str, Any] = {"kind": args.source_kind, "path": args.model}
    if args.auto_class:
        source["auto_class"] = args.auto_class
    if args.trust_remote_code:
        source["trust_remote_code"] = True

    ids = [[int(token) for token in group.split(",") if token]
           for group in (args.input_ids or [])]
    prompts = args.prompt
    if not prompts and not ids:
        prompts = ["The capital of France is Paris."]

    modules = [name for name in args.modules.split(",") if name]
    if not modules:
        print("WARNING: no --modules given; only final logits will be "
              "compared, so a broken layer cannot be localized")

    probe_source = source
    if args.model.startswith(GCS_SCHEME):
        # Tokenizing needs the tokenizer files, not the weights.
        probe_source = {
            **source, "path": local_checkpoint(args.model, small_only=True)
        }
    probe, tensors = make_probe(probe_source, modules, prompts, ids,
                                args.max_length)
    if probe_source is not source:
        # Each capture copies the checkpoint itself, so the probe names the
        # gs:// folder rather than this job's copy of it.
        probe["model_id"], probe["source"] = args.model, source
    tensors_path = write_probe(args.out, probe, tensors)
    print(f"Wrote {args.out} and {tensors_path} "
          f"({len(probe['probes'])} probe(s), {len(modules)} module(s), "
          f"source {source['kind']}, "
          f"digest {probe_digest(probe, tensors_path)})")
    return 0


def run_capture_mode(args: argparse.Namespace) -> int:
    """Handles the reference/candidate capture modes."""
    if not args.role or not args.probe or not args.out:
        raise CaptureError("capture needs --role, --probe and --out")

    probe, probe_tensors = load_probe(args.probe)
    tensors_path = probe_tensors_path(args.probe)
    device = get_device(args.role, args.device)
    # The probe names the dtype so both runs agree without the caller having
    # to remember to pass it twice. --dtype still overrides, for the case
    # where a model has no working bf16 path on CPU.
    dtype = getattr(torch, args.dtype or probe.get("dtype", DEFAULT_DTYPE))

    patch_module = None
    if args.role == REFERENCE:
        check_reference_host()
        if args.patches:
            # The model could not run without our rewrite, so the reference is
            # not independent of it. Recorded, not refused: the numbers still
            # show whether the port to TPU held.
            print("WARNING: reference is loading the patch module, so it is "
                  "NOT independent of the code under test; the comparison "
                  "will be reported as UNVERIFIED")
            patch_module = load_patch_module(args.patches)
    else:
        if not args.patches:
            print("WARNING: candidate has no --patches; it will run whatever "
                  "the stock library does on this device")
        else:
            patch_module = load_patch_module(args.patches)

    independent = args.role == REFERENCE and patch_module is None

    source = resolve_source(probe)
    if str(source.get("path") or "").startswith(GCS_SCHEME):
        source["path"] = local_checkpoint(source["path"])
    adapter = get_adapter(source["kind"])
    model = adapter.load(source, dtype, device)
    tensors = run_capture(model, probe, probe_tensors, device, adapter)
    manifest = build_manifest(args.role, probe, device, dtype, independent,
                              model, patch_module, tensors_path)
    write_capture(args.out, manifest, tensors)

    print(f"Wrote {args.out}.json and {args.out}.safetensors")
    print(f"  role        {args.role}")
    print(f"  device      {device.type} {manifest['dtype']}")
    print(f"  probe       {manifest['probe_sha256']}")
    print(f"  modules     {len(manifest['modules'])}")
    if args.role == REFERENCE:
        print(f"  independent {independent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 2 on a bad or unusable request."""
    args = parse_args(argv)
    try:
        if args.make_probe:
            return run_make_probe(args)
        return run_capture_mode(args)
    except CaptureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
