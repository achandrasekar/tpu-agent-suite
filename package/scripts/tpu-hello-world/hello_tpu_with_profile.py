#!/usr/bin/env python3
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
"""hello_tpu, plus an xprof trace of the matmul.

Does everything hello_tpu.py does -- acquire a TPU, multiply two matrices on
it, check the answer against the CPU -- and additionally records where the
device time went, which is the question a correctness check cannot answer.

The profiling is `torch_tpu._internal.profiler`: a context manager shaped like
`torch.profiler`, with `xprof_trace_handler` to route the trace to a directory.
It is NOT `torch_xla.debug.profiler`; this runtime ships no torch_xla, so that
import raises ModuleNotFoundError here.

Sources, both inside the project rather than on the open web -- which is why
they are pinned here in comments rather than indexed by the tpu-knowledge-base agent,
whose index only holds pages anyone can fetch:

  * The profiling guide:
    https://github.com/google-pytorch/torch_tpu/blob/main/docs/html/performance_profiling.html
    (source notebook: docs/notebooks/performance_profiling.py)
  * The API and its docstrings:
    src/torch_tpu/_internal/profiler/_api.py  -- `profile`, `ProfilerActivity`,
    `xprof_trace_handler`. Read at torch_tpu main @ cfab6580, 2026-09-13. The
    file was `torch_tpu/_internal/profiler/profiler_api.py` before the tree
    moved under src/; the exported names did not change.

**The guide and this script take different routes, deliberately.** The guide
wraps `torch.profiler.profile()` with `experimental_config=TpuProfilerConfig`
and `on_trace_ready=torch.profiler.tensorboard_trace_handler(...)`, which
targets TensorBoard. This script wants xprof `.xplane.pb` files, so it uses
`xprof_trace_handler` instead. The guide does offer
an xprof route -- `TpuProfilerConfig(run_dir=...)` is documented as where
`.xplane.pb` are written -- but its worked example does not set it, so the two
paths are not reconciled there. If that changes, prefer the documented public
surface over `_internal`.

Two things are deliberate and easy to get wrong by simplifying:

  * The profiled call is warm. The first execution of a shape is a compile, so
    a trace of a cold call is a trace of the compiler and its device time is
    not the device time a loop would see.
  * A capture that returns cleanly proves nothing. The trace is verified on
    disk afterwards, because an empty trace is the failure mode that reads as
    success.

Set MOCK_TPU=1 to force the CPU path, which is how this runs in CI on a host
with no accelerator; profiling is skipped there and reported as skipped.
"""

import os
import pathlib
import shutil
import sys

import torch

# See hello_tpu.py: TPU systolic hardware accumulates float32 matmuls in
# reduced precision, so the tolerance is calibrated to real hardware. Keep
# these in step with that file; a divergence here is a silently different
# check.
TPU_ATOL = 0.5
CPU_ATOL = 1e-3
DIM = 128
ACCELERATOR_LABELS = frozenset({"native-tpu"})

# Relative, so it lands in the job's working directory on the target. An
# absolute path (this used to be /logs/artifacts/xprof) names a directory on
# the TPU node, which on GKE is the pod's own filesystem and goes away with it;
# nothing ever carried it back.
TRACE_DIR = pathlib.Path(os.environ.get("XPROF_DIR", "xprof"))

# The capture, copied to one fixed name beside the script. The profiler picks
# the session directory and the file name, so they cannot be named in advance,
# and `compute run --output` carries back files by name. One `--output` of this
# path brings the trace home on every backend.
TRACE_FILE = pathlib.Path(os.environ.get("XPROF_FILE", "xprof.xplane.pb"))

# Enough iterations that compilation is behind us before the traced one.
WARMUP_ITERS = 5


def acquire_device() -> tuple[torch.device, str]:
    """The device, and a label saying how it was reached."""
    if os.environ.get("MOCK_TPU") == "1":
        print("[HELLO-TPU] MOCK_TPU=1 set: using CPU device simulation.")
        return torch.device("cpu"), "cpu-simulated"

    try:
        dev = torch.device("tpu")
        torch.zeros(1, device=dev)  # constructing the object proves nothing
        print(f"[HELLO-TPU] Acquired native PyTorch TPU device: {dev}")
        return dev, "native-tpu"
    except Exception:  # pylint: disable=broad-except
        pass

    print("[HELLO-TPU] No TPU runtime detected: falling back to CPU.")
    return torch.device("cpu"), "cpu-fallback"


def accelerator_name() -> str:
    """The chip generation, read from inside this container.

    Never inferred from a cluster or node name: a cluster named after a chip
    generation proves nothing about what actually ran.
    """
    return os.environ.get("TPU_ACCELERATOR_TYPE", "unknown")


def trace_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """The xplane files a capture should have written.

    A successful capture lays out `plugins/profile/<session>/<host>.xplane.pb`,
    so finding them is how a capture is confirmed. Globbed rather than
    constructed because the session id is chosen by the profiler.
    """
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*.xplane.pb")
                  if p.stat().st_size > 0)


def profiled_matmul(a: torch.Tensor, b: torch.Tensor,
                    device_type: str) -> tuple[torch.Tensor, str]:
    """Run the matmul warm, with the traced iteration last.

    Returns the result and a status string for the report: `captured`, or why
    not. A failure to profile is not a failure to compute -- the caller still
    gets a result to verify, and the report says the trace is missing rather
    than the run being lost.
    """
    for _ in range(WARMUP_ITERS):
        warm = torch.matmul(a, b)
    del warm

    if device_type != "native-tpu":
        print(f"[HELLO-TPU] Profiling skipped: device is {device_type}, and "
              "the xprof profiler belongs to the native TPU runtime.")
        return torch.matmul(a, b), f"skipped ({device_type})"

    try:
        # pylint: disable-next=g-import-not-at-top
        from torch_tpu._internal import profiler
    except ImportError as exc:
        print(f"[HELLO-TPU] Profiling unavailable: {exc}", file=sys.stderr)
        return torch.matmul(a, b), "unavailable (no profiler module)"

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[HELLO-TPU] Capturing xprof trace to {TRACE_DIR} ...")
    try:
        with profiler.profile(
                activities=[profiler.ProfilerActivity.TPU],
                on_trace_ready=profiler.xprof_trace_handler(
                    dir_name=str(TRACE_DIR)),
        ):
            result = torch.matmul(a, b)
            # Force the work inside the traced region. Anything still queued
            # when the trace stops is not in the trace.
            result.cpu()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[HELLO-TPU] Profiling failed: {exc}", file=sys.stderr)
        return torch.matmul(a, b), f"failed ({type(exc).__name__})"

    # The capture returned. That is not the same as the capture working.
    written = trace_files(TRACE_DIR)
    if not written:
        print(
            "[HELLO-TPU] Profiling produced no non-empty .xplane.pb -- "
            "treating as a failed capture.",
            file=sys.stderr)
        return result, "empty (no xplane written)"

    total = sum(p.stat().st_size for p in written)
    print(f"[HELLO-TPU] Trace written: {len(written)} xplane file(s), "
          f"{total} bytes total.")
    for path in written:
        print(f"[HELLO-TPU]   {path}")
    newest = max(written, key=lambda p: p.stat().st_mtime)
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(newest, TRACE_FILE)
    print(f"[HELLO-TPU] Trace copied to {TRACE_FILE} "
          f"({TRACE_FILE.stat().st_size} bytes) for `--output`.")
    return result, "captured"


def report_table(rows: list[tuple[str, str]]) -> None:
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


def hello_tpu_with_profile() -> tuple[str, str]:
    """Runs the check and the capture. Returns (device label, trace status)."""
    device, device_type = acquire_device()
    print(f"[HELLO-TPU] PyTorch Version: {torch.__version__}")
    print(f"[HELLO-TPU] Device: {device} (type: {device_type})")

    print(f"[HELLO-TPU] Allocating ({DIM}x{DIM}) tensors on {device}...")
    torch.manual_seed(42)
    a = torch.randn(DIM, DIM, device=device, dtype=torch.float32)
    b = torch.randn(DIM, DIM, device=device, dtype=torch.float32)

    c, trace_status = profiled_matmul(a, b, device_type)

    c_cpu = c.cpu()
    expected = torch.matmul(a.cpu(), b.cpu())
    max_diff = torch.max(torch.abs(c_cpu - expected)).item()
    atol = TPU_ATOL if device_type in ACCELERATOR_LABELS else CPU_ATOL
    if device_type == "cpu-fallback":
        # See hello_tpu.py: a 0.000000 here would read as a pass.
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
        return device_type, trace_status

    report_table([
        ("device_type", device_type),
        ("accelerator", accelerator_name()),
        ("max_abs_diff", f"{max_diff:.6f}"),
        ("tolerance", str(atol)),
        ("warmup_iters", str(WARMUP_ITERS)),
        ("xprof_captured", "yes" if trace_status == "captured" else "no"),
        ("trace_status", trace_status),
    ])
    print(f"[HELLO-TPU] SUCCESS: computed on {device_type} and verified "
          "against the CPU reference.")
    return device_type, trace_status


def main() -> int:
    try:
        device_type, trace_status = hello_tpu_with_profile()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[HELLO-TPU] FAILED: {exc}", file=sys.stderr)
        return 1

    if device_type == "cpu-fallback":
        print(
            "[HELLO-TPU] FAILED: no TPU runtime was available, so nothing "
            "about the TPU was verified.",
            file=sys.stderr)
        return 1

    # A missing trace fails the run when a trace was the point. Capture is
    # skipped on purpose off real hardware, and that is not a failure.
    if device_type == "native-tpu" and trace_status != "captured":
        print(
            f"[HELLO-TPU] FAILED: the matmul was verified but no trace was "
            f"captured ({trace_status}).",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
