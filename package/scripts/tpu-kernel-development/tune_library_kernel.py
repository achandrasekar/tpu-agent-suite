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
"""Autotunes the Tokamax ops a swapped-in library kernel calls.

Tokamax's untuned default configs can be far slower than tuned ones, so a
library kernel measured untuned understates it. This runs Tokamax's own
autotuner on exactly the shapes the harness tests -- every config
`get_inputs()` returns. It writes every measurement as JSON, and prints the
fastest config for each call shape as a `BEST` line, which the kernel file
pins on an explicit op instance.

A full search can outlast the `compute run` timeout, and a killed run returns
nothing. So configs are tuned in batches, the result is rewritten after each
batch, and tuning stops on its own once `--budget` seconds are spent: whatever
was measured by then is kept. The default (heuristic) config is in the first
batch, so a partial result never picks something slower than the default.

Takes the runnable test script `assemble_test_run.py` produced, because it
already binds `get_inputs`, `input_configs` and `opt_computation`; it is
loaded without running its `main()`.
"""

import argparse
import runpy
import time
from pathlib import Path

import jax
import tokamax

BATCH_SIZE = 8


def shapes(ba) -> str:
    """The array arguments of an op call, e.g. `lhs=bfloat16[8192, 2048]`."""
    return ", ".join(f"{name}={arg.dtype}{list(arg.shape)}"
                     for name, arg in ba.arguments.items()
                     if hasattr(arg, "shape") and hasattr(arg, "dtype"))


def op_calls(test_run_path: str) -> list:
    """The Tokamax op calls `opt_computation` makes, for every harness input."""
    ns = runpy.run_path(test_run_path, run_name="tune_library_kernel")
    computation = ns["opt_computation"]

    calls = []
    for dynamic_args, static_args in ns["input_configs"](ns["get_inputs"]()):

        def call(*args, static_args=tuple(static_args)):
            return computation(*args, *static_args)

        calls += tokamax.autotuning.get_bound_args(call, *dynamic_args)
    if not calls:
        raise ValueError(
            "opt_computation calls no Tokamax op, so there is nothing to tune."
        )
    return calls


def print_best(ba, data, source: str = "") -> None:
    """One result line for an op call: its fastest config, or why there is none.

    `fastest_config` raises when every config measured for the call failed;
    that call gets a `NO_CONFIG` line and the other calls still get theirs.
    """
    op = type(ba.op)
    try:
        config = data.fastest_config
    except (ExceptionGroup, ValueError) as e:
        print(f"NO_CONFIG {shapes(ba)}: every config measured failed ({e}); "
              "leave this call untuned")
        return
    print(f"BEST {shapes(ba)}: "
          f"{op.__module__}.{op.__name__}(config={config!r}){source}")


def tune(test_run_path: str, output_path: str, budget_s: float) -> None:
    deadline = time.monotonic() + budget_s
    device_kind = jax.devices()[0].device_kind
    # (bound args, autotuning data) per op call, as Tokamax keeps it.
    tuned = []
    measured = total = 0
    # Written up front, so the file exists even if every call is cached.
    Path(output_path).write_text(
        tokamax.AutotuningResult(device_kind, ()).dumps())

    calls = op_calls(test_run_path)
    # Calls Tokamax already has tuned results for need no search, but still
    # get a BEST line: the kernel file pins one config per call shape.
    cached = [ba for ba in calls if ba.cached_autotuning_data is not None]
    for ba in calls:
        if ba.cached_autotuning_data is not None:
            continue
        default = ba.heuristics_config
        others = [c for c in ba.autotuning_configs if c != default]
        configs = [default] + others
        total += len(configs)
        data = None
        for start in range(0, len(configs), BATCH_SIZE):
            if time.monotonic() >= deadline:
                break
            batch = set(configs[start:start + BATCH_SIZE])
            batch_data = ba.autotune(configs=batch, cache_results=False)
            data = batch_data if data is None else data | batch_data
            measured += len(batch)
            result = tokamax.AutotuningResult(device_kind,
                                              tuple(tuned) + ((ba, data), ))
            Path(output_path).write_text(result.dumps(prune_errors=True))
        if data is not None:
            tuned.append((ba, data))

    for ba in cached:
        print_best(ba, ba.cached_autotuning_data, " (from Tokamax's cache)")
    for ba, data in tuned:
        print_best(ba, data)

    if total == 0:
        print(f"Tuning not needed: all {len(cached)} op call(s) already have "
              f"tuned configs in Tokamax's cache; wrote {output_path}")
        return
    status = "complete" if measured == total else "partial (budget spent)"
    print(f"Tuning {status}: measured {measured} of {total} configs across "
          f"{len(tuned)} op call(s), plus {len(cached)} already tuned in "
          f"Tokamax's cache; wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Autotune the Tokamax ops in a library-kernel test run.")
    parser.add_argument("test_run_path",
                        help="The runnable script from assemble_test_run.py.")
    parser.add_argument("output_path",
                        help="Where to write the AutotuningResult JSON.")
    parser.add_argument(
        "--budget",
        type=float,
        default=1500,
        help="Seconds to spend tuning before stopping with what was measured."
        " Keep it below the compute run timeout.")
    args = parser.parse_args()
    tune(args.test_run_path, args.output_path, args.budget)


if __name__ == "__main__":
    main()
