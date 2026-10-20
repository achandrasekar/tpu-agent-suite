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
"""Check a tpu_agent_suite_kernels op in place, on the target, without a server.

Runs on the accelerator host (through `compute run`). It builds the real vLLM
module the op replaces, at one tensor-parallel rank's shapes, inside a
world-size-1 vLLM context, patches it with the plugin, and runs two checks:

  op parity     the module with the op on matches it with the op off at every
                bucket in the case, and the bridge took the kernel path on
                owned buckets and the fallback everywhere else;
  block timing  the case's block -- the module plus the neighbours XLA fuses
                with it -- compiled, is faster with the op on than off at the
                owned buckets.

Block timing is the measurement a kernel harness cannot make: the op as the server
compiles it, in its neighbourhood, against the same op compiled by XLA. The
case module (see `CASE_CONTRACT`) says how to build the module and block and
what the inputs are; this script owns everything else.

Writes `--out` (JSON) and exits 0 when both checks pass, 1 when one failed
(the report says which), 2 when the case or the plugin could not be loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import tarfile
import time
import traceback

Path = pathlib.Path

EXIT_OK = 0
EXIT_GATE = 1
EXIT_SETUP = 2

CASE_CONTRACT = (
    "OP",  # the op's slug in the plugin
    "BUCKETS",  # token counts to check: owned ones and fallback ones
    "build",  # build(quant_config) -> the vLLM module the op replaces
    "inputs",  # inputs(bucket, generator) -> tuple of CPU tensors
)
# Optional in a case: build_block(module) -> nn.Module (default: the module),
# ATOL / RTOL (default 2e-2), DTYPE (default torch.bfloat16), init(module,
# generator) to fill weights (default: seeded normal, scaled by fan-in).


def load_case(path: str):
    spec = importlib.util.spec_from_file_location("inplace_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [n for n in CASE_CONTRACT if not hasattr(module, n)]
    if missing:
        raise AttributeError(f"case {path} does not define {missing}")
    return module


def add_plugin(plugin: str) -> str:
    """Puts the plugin on sys.path; unpacks it first when it is a .tgz."""
    path = Path(plugin)
    if path.is_file():
        root = path.with_suffix("").with_suffix("")
        root = root.parent / (root.name + "_unpacked")
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tar:
            tar.extractall(root)
        path = root
    sys.path.insert(0, str(path))
    return str(path)


def synchronize(device) -> None:
    if getattr(device, "type", str(device)) != "tpu":
        return
    # pylint: disable=g-import-not-at-top
    from torch_tpu._internal import sync  # pyrefly: ignore[missing-import]
    sync.synchronize(wait=True)


class VllmContext:
    """A world-size-1 vLLM context, enough to construct model layers.

    Layers read the current VllmConfig and the tensor-parallel group when
    they are built; the group needs a process group even at size 1.
    """

    def __init__(self, port: int):
        self.port = port
        self._cm = None

    def __enter__(self):
        # pylint: disable=g-import-not-at-top
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        self._cm = set_current_vllm_config(VllmConfig())
        self._cm.__enter__()
        ps.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{self.port}",
            backend="gloo")
        ps.initialize_model_parallel(1)
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def default_init(module, generator) -> None:
    import torch  # pylint: disable=g-import-not-at-top
    with torch.no_grad():
        for p in module.parameters():
            fan_in = p.shape[-1] if p.dim() > 1 else 1
            scale = fan_in**-0.5 if p.dim() > 1 else 1.0
            p.copy_((torch.randn(p.shape, generator=generator) * scale +
                     (0 if p.dim() > 1 else 1)).to(p.dtype))


def server_quant_config():
    """The quant config the server builds an unquantized model with.

    vllm_torchtpu's model runner replaces vLLM's quant config before it builds
    the model; for an unquantized model it is VllmUnquantizedConfig, whose
    linear method flips weights to [in, out] after loading. A layer built
    without it gets vLLM's stock method and keeps [out, in], a layout the
    server never has. None off vllm_torchtpu.
    """
    try:
        # pylint: disable=g-import-not-at-top
        from vllm_torchtpu.layers.adapter.quantization.unquantized import \
            VllmUnquantizedConfig
    except ImportError:
        return None
    return VllmUnquantizedConfig()


def finish_loading(module) -> None:
    """What vLLM runs once a model's weights are loaded.

    Each layer's quant method may re-lay out its weights here, and the op
    reads them as the server holds them: vllm_torchtpu flips unquantized
    linear weights from vLLM's [out, in] to [in, out]. A module that skipped
    this would show the op a layout the server never has.
    """
    for m in module.modules():
        method = getattr(m, "quant_method", None)
        if method is not None and hasattr(method,
                                          "process_weights_after_loading"):
            method.process_weights_after_loading(m)


def compare(ref, got, atol: float, rtol: float) -> dict:
    import torch  # pylint: disable=g-import-not-at-top
    ref = ref.detach().to("cpu").float()
    got = got.detach().to("cpu").float()
    if ref.shape != got.shape:
        return {
            "match": False,
            "reason": f"shape {tuple(got.shape)} vs "
            f"{tuple(ref.shape)}"
        }
    if not torch.isfinite(got).all():
        return {"match": False, "reason": "NaN or Inf with the op on"}
    diff = (ref - got).abs()
    return {
        "match": bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
        "max_abs_err": float(diff.max()),
        "rel_fro_err": float(diff.norm() / ref.norm().clamp_min(1e-12)),
    }


def time_us(fn, args, device, warmup: int, iters: int, repeats: int) -> dict:
    """Per-call time with calls issued back to back, as a server issues them.

    A server dispatches the next step while the device runs this one, so the
    host-side cost of a call overlaps with device work. Synchronising after
    every call would expose that host cost in full -- on a small op it is most
    of the time -- and charge it to whichever arm has the costlier call. So
    each repeat issues `iters` calls, keeps their outputs alive (an output
    nobody holds may never be computed), synchronises once and divides.
    """
    for _ in range(warmup):
        fn(*args)
    synchronize(device)
    per_call = []
    for _ in range(repeats):
        outputs = []
        start = time.perf_counter()
        for _ in range(iters):
            outputs.append(fn(*args))
        synchronize(device)
        per_call.append((time.perf_counter() - start) * 1e6 / iters)
        del outputs
    median = statistics.median(per_call)
    return {
        "median_us": median,
        "spread_pct": 100.0 * (max(per_call) - min(per_call)) / median,
        "repeats": per_call,
    }


def run(case, args) -> dict:
    # pylint: disable=g-import-not-at-top
    import torch
    import tpu_agent_suite_kernels
    from tpu_agent_suite_kernels import bridge

    op = case.OP
    owned = None
    report: dict = {
        "op": op,
        "device": args.device,
        "op_parity": {},
        "block_timing": {}
    }
    dtype = getattr(case, "DTYPE", torch.bfloat16)
    atol = float(getattr(case, "ATOL", 2e-2))
    rtol = float(getattr(case, "RTOL", 2e-2))
    init = getattr(case, "init", default_init)
    device = torch.device(args.device)
    flag = f"_tpu_agent_suite_{op}"

    tpu_agent_suite_kernels.patch(op)
    owned = sorted(tpu_agent_suite_kernels.load_op(op).OWNED_BUCKETS)
    report["owned_buckets"] = owned
    torch.set_default_dtype(dtype)

    quant_config = server_quant_config()

    def build(on: bool, seed: int = 0):
        module = case.build(quant_config)
        if not getattr(module, flag, False) and on:
            raise RuntimeError(f"the plugin did not take {type(module)}: "
                               "it is not eligible (see the ORIGINAL line)")
        init(module, torch.Generator().manual_seed(seed))
        setattr(module, flag, on)
        module = module.to(device)
        finish_loading(module)
        return module

    # Op parity: one module, op toggled on the instance; same weights, same
    # input.
    span = build(True)
    parity_rows = []
    for bucket in case.BUCKETS:
        x = tuple(
            t.to(device)
            for t in case.inputs(bucket,
                                 torch.Generator().manual_seed(bucket)))
        setattr(span, flag, False)
        off = span(*x)
        setattr(span, flag, True)
        on = span(*x)
        synchronize(device)
        row = {"bucket": bucket, **compare(off, on, atol, rtol)}
        parity_rows.append(row)
    paths = {m["bucket"]: m["path"] for m in bridge.MARKERS if m["op"] == op}
    wrong_path = [
        b for b in case.BUCKETS
        if paths.get(b) != ("kernel" if b in owned else "original")
    ]
    report["op_parity"] = {
        "passed": all(r["match"] for r in parity_rows) and not wrong_path,
        "buckets": parity_rows,
        "paths": paths,
        "wrong_path": wrong_path,
    }
    if not report["op_parity"]["passed"] or args.checks == "op_parity":
        return report

    # Block timing: the block compiled twice, op off and op on, timed per
    # bucket.
    build_block = getattr(case, "build_block", lambda m: m)
    blocks = {}
    for on in (False, True):
        block = build_block(build(on)).to(device)
        # fullgraph: vLLM compiles the model as one graph, so an op that
        # breaks the graph would fail there. Here it fails the gate, instead
        # of running eagerly between compiled pieces and skewing the timing.
        blocks[on] = torch.compile(block,
                                   backend=args.backend,
                                   dynamic=False,
                                   fullgraph=True)
    timed = sorted(set(owned) | set(args.control_buckets or ()))
    for on in (False, True):
        x = tuple(
            t.to(device)
            for t in case.inputs(timed[0],
                                 torch.Generator().manual_seed(timed[0])))
        try:
            blocks[on](*x)
        except Exception as e:  # pylint: disable=broad-exception-caught
            report["block_timing"] = {
                "passed": False,
                "graph_break": {
                    "op_on": on,
                    "error": f"{type(e).__name__}: {e}"[:2000]
                },
            }
            return report
    timing_rows = []
    for bucket in timed:
        x = tuple(
            t.to(device)
            for t in case.inputs(bucket,
                                 torch.Generator().manual_seed(bucket)))
        row = {"bucket": bucket, "owned": bucket in owned}
        for on in (False, True):
            row["on" if on else "off"] = time_us(blocks[on], x, device,
                                                 args.warmup, args.iters,
                                                 args.repeats)
        row.update(compare(blocks[False](*x), blocks[True](*x), atol, rtol))
        off_us, on_us = row["off"]["median_us"], row["on"]["median_us"]
        row["speedup"] = off_us / on_us if on_us else None
        row["delta_pct"] = 100.0 * (off_us - on_us) / off_us
        row["verdict"] = bucket_verdict(row, args.noise_floor_pct)
        timing_rows.append(row)
    owned_rows = [r for r in timing_rows if r["owned"]]
    verdicts = {r["verdict"] for r in owned_rows}
    speedups = [math.log(r["speedup"]) for r in owned_rows]
    report["block_timing"] = {
        "noise_floor_pct":
        args.noise_floor_pct,
        "buckets":
        timing_rows,
        # Speed only: correctness is the op parity check's. The block's on/off error is
        # reported but not gated, because a norm after the op amplifies bf16
        # rounding.
        "passed":
        bool(owned_rows) and verdicts == {"win"},
        # Any clear non-win decides; noise alone leaves it undecided.
        "inconclusive":
        bool(owned_rows) and "no_gain" not in verdicts
        and "inconclusive" in verdicts,
        "geomean_speedup_owned":
        math.exp(statistics.fmean(speedups)) if speedups else None,
    }
    return report


def bucket_verdict(row: dict, floor: float) -> str:
    """`win`, `no_gain` or `inconclusive` for one bucket's off/on timings.

    A win must clear the noise floor plus the spread of the noisier arm. When
    the spread itself is above the floor and the difference sits inside it,
    the measurement cannot tell: a block too small to keep the device busy is
    timed by host dispatch, whose jitter is not the kernel's.
    """
    spread = max(row["off"]["spread_pct"], row["on"]["spread_pct"])
    if row["delta_pct"] > floor + spread:
        return "win"
    if spread > floor and abs(row["delta_pct"]) <= floor + spread:
        return "inconclusive"
    return "no_gain"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--plugin",
        required=True,
        help="plugin root directory, or the .tgz build_plugin packed")
    p.add_argument("--case", required=True, help="the case module")
    p.add_argument("--checks",
                   choices=("op_parity", "all"),
                   default="all",
                   help="op_parity skips the block timing")
    p.add_argument("--device", default="tpu")
    p.add_argument("--backend",
                   default="tpu",
                   help="torch.compile backend for the block timing")
    p.add_argument(
        "--no-vllm",
        action="store_true",
        help="skip the vLLM context (cases that build plain modules)")
    p.add_argument("--port", type=int, default=29517)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--noise-floor-pct", type=float, default=2.0)
    p.add_argument("--control-buckets",
                   type=int,
                   nargs="*",
                   default=None,
                   help="fallback buckets to time as controls in the block "
                   "timing")
    p.add_argument("--out", default="inplace_report.json")
    args = p.parse_args(argv)

    try:
        add_plugin(args.plugin)
        case = load_case(args.case)
        os.environ["TPU_AGENT_SUITE_KERNELS"] = case.OP
        if args.no_vllm:
            report = run(case, args)
        else:
            with VllmContext(args.port):
                report = run(case, args)
    except Exception as e:  # pylint: disable=broad-exception-caught
        report = {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:]
        }
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "error")}))
        return EXIT_SETUP

    parity_ok = report["op_parity"].get("passed", False)
    timing_ok = report["block_timing"].get(
        "passed", False) if args.checks == "all" else True
    report["status"] = (
        "ready" if parity_ok and timing_ok else
        "mismatch" if not parity_ok else "graph_break"
        if report["block_timing"].get("graph_break") else "inconclusive"
        if report["block_timing"].get("inconclusive") else "no_gain")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({
            "status":
            report["status"],
            "op_parity_passed":
            parity_ok,
            "block_geomean_speedup_owned":
            report["block_timing"].get("geomean_speedup_owned"),
        }))
    return EXIT_OK if report["status"] == "ready" else EXIT_GATE


if __name__ == "__main__":
    sys.exit(main())
