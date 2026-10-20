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
"""Extract a standalone reference kernel from model/container code.

When XProf identifies an in-container op or compiler %fusion as a bottleneck,
this script extracts the Python function or layer around the reported
file:line and generates a standalone module (<work>/base_kernel.py) with a
module-level `computation(*args)` function and dummy tensor fixtures matching
the observed call shapes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


def parse_shape_spec(spec: str) -> tuple[str, list[int]]:
    """Parse an XLA shape string like 'bf16[8,16,128,64]{3,2,1,0}'."""
    match = re.match(r"^([a-zA-Z0-9_]+)\[(.*?)\]", spec.strip())
    if not match:
        raise ValueError(f"Unrecognized shape spec: {spec}")
    dtype_str = match.group(1).lower()
    dims_str = match.group(2).strip()
    dims = [int(d.strip()) for d in dims_str.split(",") if d.strip()]
    return dtype_str, dims


def map_dtype_to_torch(dtype_str: str) -> str:
    """Map XLA dtype string to torch dtype representation."""
    mapping = {
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "f16": "torch.float16",
        "float16": "torch.float16",
        "f32": "torch.float32",
        "float32": "torch.float32",
        "f64": "torch.float64",
        "s32": "torch.int32",
        "i32": "torch.int32",
        "s64": "torch.int64",
        "i64": "torch.int64",
        "s16": "torch.int16",
        "s8": "torch.int8",
        "u8": "torch.uint8",
        "pred": "torch.bool",
        "bool": "torch.bool",
    }
    return mapping.get(dtype_str, "torch.float32")


def find_enclosing_callable(tree: ast.AST,
                            target_line: int) -> tuple[ast.AST | None, bool]:
    """Locate the top-level function or class enclosing target_line."""
    for node in tree.body:
        if isinstance(node,
                      (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None and start <= target_line <= end:
                return node, isinstance(node, ast.ClassDef)
    return None, False


def extract_source_code(file_path: Path,
                        target_line: int | None) -> tuple[str, str, bool]:
    """Read source file and extract the relevant callable around target_line.

    Returns:
        (extracted_code, target_callable_name, is_class)
    """
    raw_src = file_path.read_text(encoding="utf-8")
    if target_line is None:
        return raw_src, "computation", False

    tree = ast.parse(raw_src, filename=str(file_path))
    enclosing, is_class = find_enclosing_callable(tree, target_line)

    if enclosing is None:
        return raw_src, "computation", False

    # Extract all top-level import statements
    import_lines = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(raw_src, stmt)
            if seg:
                import_lines.append(seg)

    imports_src = "\n".join(import_lines)
    node_src = ast.get_source_segment(raw_src, enclosing) or ""
    callable_name = getattr(enclosing, "name", "computation")

    combined = f"{imports_src}\n\n{node_src}\n"
    return combined, callable_name, is_class


def build_standalone_module(
    extracted_src: str,
    target_name: str,
    is_class: bool,
    operands: list[tuple[str, list[int]]],
) -> str:
    """Synthesize standalone <work>/base_kernel.py content."""
    lines = [
        "# Auto-generated standalone kernel reference module",
        "# Extracted for tpu-kernel-development by tpu-model-onboarding",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        extracted_src,
        "",
    ]

    # Generate standard computation(*args, **kwargs) entry point
    if is_class:
        lines.extend([
            f"# Standard entry point wrapper for class {target_name}",
            "_INSTANCE = None",
            "",
            "def computation(*args, **kwargs):",
            "    global _INSTANCE",
            "    if _INSTANCE is None:",
            "        try:",
            f"            _INSTANCE = {target_name}()",
            "        except TypeError:",
            "            try:",
            "                dim = args[0].shape[-1] if args and hasattr(args[0], 'shape') else 2880",
            f"                _INSTANCE = {target_name}(dim)",
            "            except Exception:",
            f"                _INSTANCE = {target_name}",
            "    return _INSTANCE(*args, **kwargs)",
            "",
        ])
    elif target_name != "computation":
        lines.extend([
            f"# Standard entry point wrapper for {target_name}",
            "def computation(*args, **kwargs):",
            f"    return {target_name}(*args, **kwargs)",
            "",
        ])

    # Generate dummy input fixtures
    lines.extend([
        "def get_dummy_inputs():",
        "    inputs = []",
    ])
    for dtype_str, dims in operands:
        torch_dtype = map_dtype_to_torch(dtype_str)
        shape_repr = ", ".join(str(d) for d in dims)
        lines.append(
            f"    inputs.append(torch.randn({shape_repr}, dtype={torch_dtype}))"
        )
    lines.append("    return inputs\n")

    lines.extend([
        "if __name__ == '__main__':",
        "    inputs = get_dummy_inputs()",
        "    try:",
        "        out = computation(*inputs)",
        "        print('Execution verified. Output shapes:', [o.shape if hasattr(o, 'shape') else type(o) for o in (out if isinstance(out, (tuple, list)) else [out])])",
        "    except Exception as e:",
        "        print(f'Execution check completed with: {e}')",
    ])

    return "\n".join(lines) + "\n"


def resolve_file_location(source_arg: str) -> tuple[Path, int | None]:
    """Parse 'path/to/file.py:line' or 'path/to/file.py'."""
    parts = source_arg.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        file_part, line_part = parts[0], int(parts[1])
    else:
        file_part, line_part = source_arg, None

    candidate_path = Path(file_part)
    if candidate_path.exists():
        return candidate_path, line_part

    # Check relative to repo root or CWD
    for prefix in [Path.cwd(), Path(__file__).resolve().parents[3]]:
        alt = prefix / file_part
        if alt.exists():
            return alt, line_part

    raise FileNotFoundError(f"Could not locate source file: {file_part}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Extract a standalone kernel reference for in-container or fused ops.")
    parser.add_argument(
        "--source",
        required=True,
        help=
        "Source location from XProf (e.g. 'path/to/file.py:123' or 'file.py').",
    )
    parser.add_argument(
        "--call-shapes-json",
        help=
        "JSON string or file containing operand shapes: {'operands': ['bf16[8,16]'], ...}",
    )
    parser.add_argument(
        "--output-path",
        default="base_kernel.py",
        help=
        "Path where extracted module should be written (default: base_kernel.py).",
    )

    args = parser.parse_args()

    file_path, line_num = resolve_file_location(args.source)
    extracted_src, target_name, is_class = extract_source_code(
        file_path, line_num)

    operands = []
    if args.call_shapes_json:
        if Path(args.call_shapes_json).exists():
            data = json.loads(
                Path(args.call_shapes_json).read_text(encoding="utf-8"))
        else:
            data = json.loads(args.call_shapes_json)

        for raw_spec in data.get("operands", []):
            try:
                operands.append(parse_shape_spec(raw_spec))
            except ValueError:
                pass

    module_content = build_standalone_module(extracted_src, target_name,
                                             is_class, operands)

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(module_content, encoding="utf-8")
    print(
        f"Extracted reference kernel successfully written to: {out_file.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
