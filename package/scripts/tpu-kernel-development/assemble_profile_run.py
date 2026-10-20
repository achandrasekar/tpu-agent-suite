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
"""Deterministically assembles this iteration's runnable profile script.

`optimized.py` is a kernel definition plus a `main()` that builds its own
sample inputs -- enough to check that it compiles, not enough to profile. A
profile run has to use the run's real `get_inputs()` (so the trace describes
the shapes the kernel is being optimized for), warm up before it measures, and
initialize libtpu with the LLO debug-info flag. Getting any of those wrong
produces a trace that looks fine and answers the wrong question.

So this script builds that file rather than asking an agent to write it, for
the same reason `assemble_test_harness.py` exists: the parts that must be
exact are copied by a tool, and the only LLM-authored input is the same
`get_inputs()` the correctness harness already uses.

Layout of the emitted file:

  1. `profile_harness_template.py`, byte-for-byte, with TRACE_DIR/
     TRACE_ARCHIVE/NUM_WARMUPS/NUM_RUNS substituted.
  2. the run's `get_inputs()` source, at module level, where it can see the
     template's `jax` / `jnp` imports.
  3. the optimized kernel, `exec()`d into its own namespace and bound as
     `opt_computation` -- so a module-level helper named `kernel` cannot
     collide with anything, the same guarantee `assemble_test_run.py` gives.

Usage:
    python3 assemble_profile_run.py \\
        <optimized_kernel_path> <get_inputs_path> <output_path> \\
        [--trace-dir DIR] [--trace-archive FILE] [--warmups N] [--runs N]

Both target-side paths are relative: `compute run` addresses staged files by
a path relative to the script's working directory on the target, and the
trace comes back as the archive because `--output` carries one file.
"""

import argparse
import pathlib
import re
import sys

Path = pathlib.Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = _SCRIPTS_DIR / "profile_harness_template.py"
COMMON_PATH = _SCRIPTS_DIR / "kernel_exec_common.py"


def assemble(
    optimized_kernel_path: str,
    get_inputs_path: str,
    output_path: str,
    trace_dir: str = "tensorboard",
    trace_archive: str = "trace.tar.gz",
    warmups: int = 3,
    runs: int = 3,
) -> None:
    opt_kernel_src = Path(optimized_kernel_path).read_text()
    if "def computation" not in opt_kernel_src:
        raise ValueError(
            f"{optimized_kernel_path} has no module-level `computation`"
            " function -- cannot bind it as opt_computation.")

    get_inputs_src = Path(get_inputs_path).read_text()
    if "def get_inputs" not in get_inputs_src:
        raise ValueError(f"{get_inputs_path} has no `get_inputs()` function.")

    # A profile with no warm-up traces the compilation along with the kernel,
    # which is the one thing the trace must not contain. Refuse at assembly
    # time rather than emit a script that captures a misleading trace.
    if warmups < 1:
        raise ValueError(
            f"--warmups must be at least 1, got {warmups}: the first call"
            " compiles, so a trace without a warm-up attributes compilation"
            " time to the kernel.")

    template_src = TEMPLATE_PATH.read_text()
    for name, value in (("TRACE_DIR", repr(trace_dir)), ("TRACE_ARCHIVE",
                                                         repr(trace_archive)),
                        ("NUM_WARMUPS", repr(warmups)), ("NUM_RUNS",
                                                         repr(runs))):
        template_src, count = re.subn(rf"^{name} = .*$",
                                      f"{name} = {value}",
                                      template_src,
                                      count=1,
                                      flags=re.M)
        if count != 1:
            raise ValueError(
                f"{TEMPLATE_PATH.name} no longer defines {name} on its own"
                " line -- the template and this assembler are out of step.")

    parts = [
        "# =====================================================================",
        "# Shared compile path, from kernel_exec_common.py -- embedded so the",
        "# benchmark and the profile run reach the accelerator through the",
        "# same compilation. See that file's docstring.",
        "# =====================================================================",
        COMMON_PATH.read_text(),
        template_src,
        "",
        "# =====================================================================",
        f"# get_inputs(), from {get_inputs_path}.",
        "# The only LLM-authored part of this file, and the same one the",
        "# correctness harness uses.",
        "# =====================================================================",
        get_inputs_src,
        "",
        "# =====================================================================",
        f"# Optimized kernel, embedded verbatim from {optimized_kernel_path}.",
        "# Executed into its own namespace -- see this script's module"
        " docstring.",
        "# =====================================================================",
        f"_OPT_KERNEL_SRC = {opt_kernel_src!r}",
        "_opt_ns = {}",
        "exec(compile(_OPT_KERNEL_SRC, " + repr(str(optimized_kernel_path)) +
        ", 'exec'), _opt_ns)",
        "opt_computation = _opt_ns['computation']",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(main())",
        "",
    ]

    Path(output_path).write_text("\n".join(parts))
    print(f"Assembled profile run at {output_path} (trace dir: {trace_dir})")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble this iteration's runnable profile script.")
    parser.add_argument("optimized_kernel_path")
    parser.add_argument("get_inputs_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--trace-dir",
        default="tensorboard",
        help=("Where the script writes the trace, relative to its working"
              " directory on the compute target."),
    )
    parser.add_argument(
        "--trace-archive",
        default="trace.tar.gz",
        help=("The single file the trace directory is packed into, for"
              " `compute run --output` to carry back. Relative, like"
              " --trace-dir."),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    try:
        assemble(
            args.optimized_kernel_path,
            args.get_inputs_path,
            args.output_path,
            trace_dir=args.trace_dir,
            trace_archive=args.trace_archive,
            warmups=args.warmups,
            runs=args.runs,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"Failed to assemble profile run: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
