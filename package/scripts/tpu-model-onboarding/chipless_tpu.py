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
"""Let torch_tpu plan for a TPU generation on a host with no chip.

The vLLM engine compatibility check and the vLLM flag discovery only read
configs and static tables; nothing in them runs on a chip. torch_tpu still
asks the hardware two things: how many chips there are (with none it never
registers the `tpu` device, and vLLM rejects the device string), and which
generation they are (its device-layout table is per generation). Both are
known from the compute profile before any node is booked, so this answers
them, and the checks run on a CPU node.

Loaded at interpreter start by a sitecustomize.py beside it (written by
onboarding_preflight.py), with TPU_AGENT_SUITE_CHIPLESS_TPU (the profile's
TPU generation, which compute's gke_job sets in a --cpu-requests Job) and
CHIPLESS_TPU_DEVICES set, so every Python process in the job is covered,
`vllm serve --help` included. It then runs
the sitecustomize it hides, as Python would have. It does nothing until
torch_tpu's hardware module is imported, then replaces that module's PCI scan
with the answer.

It patches a private torch_tpu function. If torch_tpu changes under it, the
patch records why and `--check` fails with CHIPLESS_TPU_UNSUPPORTED, so the
caller runs the checks on a TPU node as before rather than reading the
failure as the model's.

  python3 chipless_tpu.py --check   # exit 0 if the device registers as asked
"""

import importlib.abc
import importlib.util
import json
import os
import sys

GENERATION_ENV = "TPU_AGENT_SUITE_CHIPLESS_TPU"
DEVICES_ENV = "CHIPLESS_TPU_DEVICES"
HARDWARE = "torch_tpu._internal.utils.hardware"
UNSUPPORTED = "CHIPLESS_TPU_UNSUPPORTED"

# torch_tpu's name (hardware.TpuVersion) for each generation, by the name
# compute's probe.py resolves the profile's accelerator to.
GENERATIONS = {
    "v4": "TPU v4",
    "v5e": "TPU v5e",
    "v5p": "TPU v5p",
    "v6e": "TPU v6e",
    "tpu7x": "TPU v7",
}

error = None
patched = False


def patch(hardware, generation: str, devices: int) -> None:
    """Make `hardware` report `devices` chips of `generation`.

    Raises KeyError, StopIteration or AttributeError when the generation is
    unknown or torch_tpu's tables have changed.
    """
    generation = GENERATIONS[generation]
    pci_id = next(pci for pci, name in hardware._PCI_DEVICE_ID_TO_NAME.items()
                  if name == generation)
    table = hardware._TOPOLOGY_BY_TPU_PCI_DEVICE_ID[pci_id]
    if not callable(getattr(hardware, "_scan_pci_tpus", None)):
        raise AttributeError("torch_tpu has no _scan_pci_tpus to replace")
    hardware._scan_pci_tpus = lambda: (devices, table)
    hardware.get_tpu_device_name = lambda: generation


class _Finder(importlib.abc.MetaPathFinder):
    """Patches torch_tpu's hardware module as it is first imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname != HARDWARE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            global error, patched
            exec_module(module)
            try:
                patch(module, os.environ[GENERATION_ENV],
                      int(os.environ[DEVICES_ENV]))
                patched = True
            except Exception as exc:  # noqa: BLE001 - reported by --check
                # Not raised: torch would fail to import in every process,
                # far from the cause. --check reports it instead.
                error = f"{type(exc).__name__}: {exc}"

        spec.loader.exec_module = exec_and_patch
        return spec


def _run_hidden_sitecustomize() -> None:
    """Run the first sitecustomize.py on sys.path after this one's own.

    Ours is found first, so Python never imports the image's (vllm-torchtpu
    sets torch_tpu's import-time defaults in it).
    """
    here = os.path.dirname(os.path.realpath(__file__))
    for entry in sys.path:
        if not entry or os.path.realpath(entry) == here:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            # As the module `sitecustomize`, which it then stays, as if Python
            # had imported it.
            spec = importlib.util.spec_from_file_location(
                "sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["sitecustomize"] = module
            spec.loader.exec_module(module)
            return


def install() -> None:
    if os.environ.get(GENERATION_ENV):
        sys.meta_path.insert(0, _Finder())
    _run_hidden_sitecustomize()


def check() -> dict:
    """Whether torch_tpu now registers the device as asked, and why not."""
    generation = os.environ.get(GENERATION_ENV, "")
    devices = os.environ.get(DEVICES_ENV, "")
    report = {"generation_asked": generation, "devices": devices}
    if generation not in GENERATIONS:
        report["error"] = (f"{GENERATION_ENV}={generation!r} is not one of "
                           f"{', '.join(GENERATIONS)}")
        return report
    shim = sys.modules.get("chipless_tpu")
    if shim is None:
        report["error"] = ("chipless_tpu was not loaded at startup: its "
                           "sitecustomize.py is not first on PYTHONPATH")
        return report
    try:
        import torch  # noqa: F401 - autoloads torch_tpu
        from torch_tpu._internal.utils import hardware
        if shim.error:
            raise RuntimeError(shim.error)
        if not shim.patched:
            raise RuntimeError("torch_tpu's hardware module loaded without "
                               "the patch")
        report["generation"] = hardware.get_tpu_device_name()
        report["device_count"] = hardware.get_tpu_device_count()
        report["device"] = str(torch.device("tpu"))
        if report["device_count"] != int(devices):
            raise RuntimeError(f"torch_tpu counts {report['device_count']} "
                               f"devices, not {devices}")
    except Exception as exc:  # noqa: BLE001 - this is the report
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    if sys.argv[1:] != ["--check"]:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    report = check()
    report["status"] = "fail" if "error" in report else "ok"
    print(json.dumps(report, sort_keys=True))
    if "error" in report:
        print(
            f"{UNSUPPORTED}: {report['error']}. Run these checks on a TPU "
            "node instead.",
            file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
elif __name__ == "chipless_tpu":
    install()
