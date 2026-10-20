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
"""Fast-fail microbench and static syntax/hardware verification for candidate arms.

Executes a two-stage fast-fail check before paying for full cluster E2E runs:
1. Static AST syntax check, forbidden import verification (e.g. deprecated torch_xla),
   and generation-aware TPU hardware gotcha checks (hot-loop host sync with 1-level
   helper call resolution, MXU lane/sublane alignment, SparseCore SRAM & VMEM ceilings).
2. Optional timed execution of a single-step / single-op microbenchmark command.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

Path = pathlib.Path

FORBIDDEN_IMPORTS = {
    "torch_xla",
    "pytorch_xla",
}

# Generation-indexed TPU hardware limits, from the per-generation hardware pages
# tpu-knowledge-base serves (`index.py get tpu-v5e`, `tpu-v6e`, ...).
TPU_HARDWARE_SPECS: dict[str, dict[str, int]] = {
    "v5e": {
        "systolic_lane_dim": 128,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": 16 * 1024 * 1024,
    },
    "v5p": {
        "systolic_lane_dim": 128,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": 32 * 1024 * 1024,
    },
    "v6e": {
        "systolic_lane_dim": 256,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": 32 * 1024 * 1024,
    },
    "trillium": {
        "systolic_lane_dim": 256,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": 32 * 1024 * 1024,
    },
    "v7x": {
        "systolic_lane_dim": 128,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": int(6.06 * 1024 * 1024),
    },
    "ironwood": {
        "systolic_lane_dim": 128,
        "systolic_sublane_dim": 8,
        "max_sc_sram_bytes": 128 * 1024,
        "max_core_vmem_bytes": int(6.06 * 1024 * 1024),
    },
}
DEFAULT_TPU_VERSION = "v7x"


def check_syntax_and_imports(file_path: Path) -> list[str]:
    """Check Python syntax and flag forbidden imports."""
    errors: list[str] = []
    if not file_path.exists():
        return [f"file does not exist: {file_path}"]

    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"failed to read {file_path}: {exc}"]

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_IMPORTS:
                    errors.append(
                        f"line {node.lineno}: forbidden import '{alias.name}' "
                        f"(deprecated; use torch_tpu instead)")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg in FORBIDDEN_IMPORTS:
                errors.append(
                    f"line {node.lineno}: forbidden import from {node.module!r} "
                    f"(deprecated; use torch_tpu instead)")

    return errors


def _is_print_or_logging_call(call_node: ast.Call) -> bool:
    """Return True if call_node is print(...) or logging.<fn>(...) / logger.<fn>(...)."""
    func = call_node.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in ("logging", "logger", "log") and func.attr in (
                "debug",
                "info",
                "warning",
                "warn",
                "error",
                "critical",
                " print",
        ):
            return True
    return False


def _find_unguarded_sync_calls(subtree: ast.AST) -> list[tuple[int, str]]:
    """Find (.item(), .cpu()) calls inside subtree, ignoring arguments to print/logging."""
    ignored_call_ids: set[int] = set()
    for node in ast.walk(subtree):
        if isinstance(node, ast.Call) and _is_print_or_logging_call(node):
            for descendant in ast.walk(node):
                ignored_call_ids.add(id(descendant))

    hits: list[tuple[int, str]] = []
    for node in ast.walk(subtree):
        if isinstance(node, ast.Call) and id(node) not in ignored_call_ids:
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "item",
                    "cpu",
            ):
                hits.append((node.lineno, node.func.attr))
    return hits


def check_hardware_gotchas(
    file_path: Path,
    config: dict[str, Any] | None = None,
    tpu_version: str = DEFAULT_TPU_VERSION,
) -> list[str]:
    """Check code and configuration against TPU generation hardware and compiler gotchas.

    Enforces:
    1. No blocking device-to-host synchronization (.item(), .cpu()) on the hot path inside
       iterative execution loops (including 1-level helper function calls; excludes print/logging).
    2. Systolic dimension alignment: lane/token dims and sublane/batch dims must align to
       the target TPU generation's MXU dimensions to prevent dynamic shape recompilation storms.
    3. SparseCore SRAM ceiling: clamp per-core scratchpad allocation to generation limit.
    4. Core VMEM ceiling: per-core tile/buffer allocation <= generation VMEM limit.
    """
    errors: list[str] = []
    if file_path.exists():
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))

            # Map module-level and class-level functions that perform unguarded host syncs
            # so 1-level function hopping from inside loops is caught.
            syncing_helpers: dict[str, tuple[int, str]] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    direct_syncs = _find_unguarded_sync_calls(node)
                    if direct_syncs:
                        syncing_helpers[node.name] = direct_syncs[0]

            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    for lineno, attr_name in _find_unguarded_sync_calls(node):
                        errors.append(
                            f"line {lineno}: blocking device-to-host synchronization "
                            f"('.{attr_name}()') inside iterative loop stalls TPU systolic pipeline "
                            "(gotcha: #host-cpu-sync-in-loop).")
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            callee_name = None
                            if isinstance(child.func, ast.Name):
                                callee_name = child.func.id
                            elif isinstance(child.func, ast.Attribute):
                                callee_name = child.func.attr
                            if callee_name and callee_name in syncing_helpers:
                                helper_line, helper_attr = syncing_helpers[
                                    callee_name]
                                errors.append(
                                    f"line {child.lineno}: loop calls helper '{callee_name}()' "
                                    f"(which performs blocking '.{helper_attr}()' sync at line {helper_line}) "
                                    "inside iterative loop (gotcha: #host-cpu-sync-in-loop)."
                                )
        except (OSError, SyntaxError):
            pass

    cfg = config or {}
    version_key = str(
        cfg.get("tpu_version") or tpu_version
        or DEFAULT_TPU_VERSION).strip().lower()
    hw_spec = TPU_HARDWARE_SPECS.get(version_key,
                                     TPU_HARDWARE_SPECS[DEFAULT_TPU_VERSION])

    systolic_lane_dim = int(
        cfg.get("systolic_lane_dim", hw_spec["systolic_lane_dim"]))
    systolic_sublane_dim = int(
        cfg.get("systolic_sublane_dim", hw_spec["systolic_sublane_dim"]))
    max_sc_sram_bytes = int(
        cfg.get("max_sc_sram_bytes", hw_spec["max_sc_sram_bytes"]))
    max_core_vmem_bytes = int(
        cfg.get("max_core_vmem_bytes", hw_spec["max_core_vmem_bytes"]))

    lane_alignment_keys = (
        "chunk_size",
        "chunk_prefill_size",
        "token_chunk",
        "block_size",
        "tile_size",
        "max_num_batched_tokens",
    )
    for k in lane_alignment_keys:
        if k in cfg:
            try:
                v = int(cfg[k])
                if v > 0 and (v % systolic_lane_dim != 0):
                    errors.append(
                        f"lane/token dimension {k}={v} is not a multiple of {version_key} MXU lane dimension ({systolic_lane_dim}); "
                        "risks XLA dynamic shape recompilation storm (gotcha: #xla-chunked-prefill-tune)."
                    )
            except (ValueError, TypeError):
                pass

    sublane_alignment_keys = (
        "batch_size",
        "max_num_seqs",
    )
    for k in sublane_alignment_keys:
        if k in cfg:
            try:
                v = int(cfg[k])
                if v > 0 and (v % systolic_sublane_dim != 0):
                    errors.append(
                        f"sublane/batch dimension {k}={v} is not a multiple of {version_key} MXU sublane dimension ({systolic_sublane_dim}); "
                        "risks sublane padding waste (gotcha: #xla-sublane-batch-tile)."
                    )
            except (ValueError, TypeError):
                pass

    if "sc_sram_bytes" in cfg:
        try:
            v = int(cfg["sc_sram_bytes"])
            if v > max_sc_sram_bytes:
                errors.append(
                    f"SparseCore SRAM budget exceeded on {version_key}: {v} bytes > {max_sc_sram_bytes} bytes limit "
                    "(gotcha: #sc-sram-spec-arbiter-v404).")
        except (ValueError, TypeError):
            pass

    if "vmem_bytes_per_core" in cfg:
        try:
            v = int(cfg["vmem_bytes_per_core"])
            if v > max_core_vmem_bytes:
                errors.append(
                    f"Core VMEM budget exceeded on {version_key}: {v} bytes > {max_core_vmem_bytes} bytes limit "
                    "(gotcha: #vmem-spill-ceiling).")
        except (ValueError, TypeError):
            pass

    return errors


def run_microbench(
    cmd: list[str],
    timeout_s: float = 30.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run an isolated microbench command with strict timeout."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
        )
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": exc.stdout or "",
            "stderr": f"microbench timed out after {timeout_s}s",
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def _resolve_bench_command(
    run_arg: str | bool | None,
    remainder: list[str],
) -> tuple[list[str], bool]:
    """Normalize `--run "<cmd>"` and `--run -- <cmd...>` without swallowing `--gotchas`.

    Returns (command_argv, extracted_gotchas_flag).
    """
    extracted_gotchas = False
    cleaned_remainder: list[str] = []
    for tok in remainder:
        if tok == "--gotchas":
            extracted_gotchas = True
            continue
        cleaned_remainder.append(tok)

    if cleaned_remainder and cleaned_remainder[0] == "--":
        cleaned_remainder = cleaned_remainder[1:]

    if isinstance(run_arg, str) and run_arg != "__FLAG_ONLY__":
        cmd_tokens = shlex.split(run_arg)
        if cleaned_remainder:
            cmd_tokens.extend(cleaned_remainder)
        return cmd_tokens, extracted_gotchas

    if run_arg == "__FLAG_ONLY__" and cleaned_remainder:
        if len(cleaned_remainder) == 1 and " " in cleaned_remainder[0]:
            return shlex.split(cleaned_remainder[0]), extracted_gotchas
        return cleaned_remainder, extracted_gotchas

    return [], extracted_gotchas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=
        "Fast-fail microbench and syntax/hardware verification for candidate arms."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Python source file to check.",
    )
    parser.add_argument(
        "--gotchas",
        action="store_true",
        help=
        "Enforce hardware gotcha checks (host sync, systolic alignment, memory caps).",
    )
    parser.add_argument(
        "--config",
        help=
        "Optional JSON string or path to JSON file for hardware gotcha audit.",
    )
    parser.add_argument(
        "--tpu-version",
        default=DEFAULT_TPU_VERSION,
        choices=sorted(TPU_HARDWARE_SPECS.keys()),
        help=
        "Target TPU generation for hardware threshold lookup (default: v7x).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Microbench timeout in seconds.",
    )
    parser.add_argument(
        "--run",
        nargs="?",
        const="__FLAG_ONLY__",
        default=None,
        help=
        'Execute isolated microbench command (accepts `--run "<cmd>"` or `--run -- <cmd>`).',
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Optional microbench command tokens when using `--run -- <cmd>`.",
    )

    args = parser.parse_args(argv)
    bench_cmd, trailing_gotchas = _resolve_bench_command(
        args.run, list(args.cmd))
    enforce_gotchas = bool(args.gotchas or trailing_gotchas or args.config)

    errors = check_syntax_and_imports(args.file)
    if errors:
        result = {
            "valid": False,
            "phase": "syntax_check",
            "errors": errors,
        }
        print(json.dumps(result, indent=2))
        return 1

    cfg: dict[str, Any] = {}
    if args.config:
        cfg_str = args.config.strip()
        if Path(cfg_str).is_file():
            cfg = json.loads(Path(cfg_str).read_text(encoding="utf-8"))
        else:
            cfg = json.loads(cfg_str)

    if enforce_gotchas:
        gotcha_errs = check_hardware_gotchas(args.file,
                                             config=cfg,
                                             tpu_version=args.tpu_version)
        if gotcha_errs:
            result = {
                "valid": False,
                "phase": "gotcha_check",
                "errors": gotcha_errs,
            }
            print(json.dumps(result, indent=2))
            return 1

    if bench_cmd:
        bench_res = run_microbench(bench_cmd, timeout_s=args.timeout)
        result = {
            "valid": bench_res["success"],
            "phase": "microbench_run",
            "run": bench_res,
        }
        print(json.dumps(result, indent=2))
        return 0 if bench_res["success"] else 1

    result = {
        "valid": True,
        "phase": "syntax_check",
        "errors": [],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
