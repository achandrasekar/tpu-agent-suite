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
"""Smallest real PyTorch-on-TPU check: one op, cross-checked against the CPU.

Acquires a TPU device, multiplies two matrices on it, copies the result back,
and compares it to the same multiplication done on the CPU. Prints which kind
of device it actually got, so a reader can tell a real TPU run from a silent
fall back to CPU.

Provisions nothing -- no venv, no pip. If no TPU runtime is present that is
reported and the script exits non-zero, rather than being papered over.

Set MOCK_TPU=1 to force the CPU path, which is how this runs in CI on a host
with no accelerator.

Adapted from tests/test_tpu_smoke.py on the test/tpu-smoke-ci branch. If you
change the numerics here, change them there too.
"""

import os
import sys

import torch

# TPU systolic hardware accumulates float32 matmuls in reduced precision, so
# the result differs from a CPU float32 reference by far more than ordinary
# floating-point noise. This tolerance is calibrated to real hardware -- if a
# run fails against it, the hardware is suspect, not this number.
TPU_ATOL = 0.5
CPU_ATOL = 1e-3
DIM = 128

# Labels whose work landed on the accelerator, and so earn TPU_ATOL.
ACCELERATOR_LABELS = frozenset({"native-tpu"})


def acquire_device() -> tuple[torch.device, str]:
    """Returns a device to compute on, and a label for how it was obtained.

  The label is the point of this function: "native-tpu" is the only value that
  means torch_tpu really drove real hardware. Every other value records a
  degradation the caller needs to see.
  """
    if os.environ.get("MOCK_TPU") == "1":
        print("[HELLO-TPU] MOCK_TPU=1 set: using CPU device simulation.")
        return torch.device("cpu"), "cpu-simulated"

    # torch_tpu registers a native PrivateUse1 backend, so the device is
    # reached as "tpu" through plain torch. Constructing the device object
    # alone proves nothing -- allocate on it to force the runtime to init.
    try:
        dev = torch.device("tpu")
        _ = torch.zeros(1, device=dev)
        print(f"[HELLO-TPU] Acquired native PyTorch TPU device: {dev}")
        return dev, "native-tpu"
    except Exception:  # pylint: disable=broad-except
        pass

    print("[HELLO-TPU] No TPU runtime detected: falling back to CPU.")
    return torch.device("cpu"), "cpu-fallback"


def accelerator_name() -> str:
    """The chip generation the runtime is actually sitting on, or "unknown".

    Read from inside the container rather than inferred from a node label. The
    cluster in this eval is *named* `torchtpu-distributed-v7x-8`, so anything
    matching a generation against cluster or node names would report a chip
    for a run that never reached one.
    """
    for var in ("TPU_ACCELERATOR_TYPE", "TPU_CHIPS_PER_HOST_BOUNDS_TYPE"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    for attr in ("get_device_name", "device_name"):
        getter = getattr(getattr(torch, "tpu", None), attr, None)
        if callable(getter):
            try:
                name = str(getter(0)).strip()
                if name:
                    return name
            except Exception:  # pylint: disable=broad-except
                pass
    return "unknown"


def report_table(rows: list[tuple[str, str]]) -> None:
    """Print the findings as a markdown table.

    Markdown, and unprefixed, because it is copied as well as parsed: the
    caller is asked to reproduce these rows verbatim in its own report, and a
    standard table survives that trip where an ad-hoc format does not. The
    verifier reads the rows rather than scraping prose.
    """
    kw = max(len(k) for k, _ in rows)
    vw = max(len(v) for _, v in rows)
    print("[HELLO-TPU] Result table follows -- reproduce it verbatim when "
          "reporting.")
    print()
    print(f"| {'field'.ljust(kw)} | {'value'.ljust(vw)} |")
    print(f"| {'-' * kw} | {'-' * vw} |")
    for key, value in rows:
        print(f"| {key.ljust(kw)} | {value.ljust(vw)} |")
    print()


def hello_tpu() -> str:
    """Runs the check. Returns the device label; raises on any mismatch."""
    device, device_type = acquire_device()
    print(f"[HELLO-TPU] PyTorch Version: {torch.__version__}")
    print(f"[HELLO-TPU] Device: {device} (type: {device_type})")

    print(f"[HELLO-TPU] Allocating ({DIM}x{DIM}) tensors on {device}...")
    torch.manual_seed(42)
    a = torch.randn(DIM, DIM, device=device, dtype=torch.float32)
    b = torch.randn(DIM, DIM, device=device, dtype=torch.float32)

    c = torch.matmul(a, b)

    # The cross-check. Without it a printed number is indistinguishable from
    # a fabricated one.
    c_cpu = c.cpu()
    expected = torch.matmul(a.cpu(), b.cpu())
    max_diff = torch.max(torch.abs(c_cpu - expected)).item()
    atol = TPU_ATOL if device_type in ACCELERATOR_LABELS else CPU_ATOL
    if device_type == "cpu-fallback":
        # No number here: both sides ran on the CPU, so a 0.000000 next to a
        # tolerance would read as a pass in the output of a run that failed.
        print("[HELLO-TPU] CPU reference comparison is not meaningful: both "
              "sides ran on the CPU.")
    else:
        print(f"[HELLO-TPU] Max absolute difference vs CPU reference: "
              f"{max_diff:.6f} (tolerance: {atol})")

    if not torch.allclose(c_cpu, expected, atol=atol, rtol=1e-2):
        raise AssertionError(
            f"Numerical mismatch on {device_type}: max difference {max_diff} "
            f"exceeds tolerance {atol}")

    if device_type == "cpu-fallback":
        # Deliberately not a SUCCESS line: the caller asked about the TPU.
        # main() turns this into the failure it is.
        return device_type

    report_table([
        ("device_type", device_type),
        ("accelerator", accelerator_name()),
        ("max_abs_diff", f"{max_diff:.6f}"),
        ("tolerance", str(atol)),
    ])
    print(f"[HELLO-TPU] SUCCESS: computed on {device_type} and verified "
          "against the CPU reference.")
    return device_type


def main() -> int:
    try:
        device_type = hello_tpu()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[HELLO-TPU] FAILED: {exc}", file=sys.stderr)
        return 1

    # A CPU fall back is not a pass. The caller asked whether the TPU works.
    if device_type == "cpu-fallback":
        print(
            "[HELLO-TPU] FAILED: no TPU runtime was available, so nothing "
            "about the TPU was verified.",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
