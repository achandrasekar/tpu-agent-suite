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
"""Assemble the `tpu_agent_suite_kernels` vLLM plugin and add ops to it.

Each call adds one op to a plugin directory, creating it from the shipped
template the first time:

  <out-dir>/tpu_agent_suite_kernels/                 the runtime (copied from beside this script)
  <out-dir>/tpu_agent_suite_kernels/op_<op>.py          the op module
  <out-dir>/tpu_agent_suite_kernels/kernel_<op>.py      the kernel (tpu-kernel-development's best artifact)
  <out-dir>/tpu_agent_suite_kernels/kernel_<op>_base.py optional: the scoped base.py
  <out-dir>/tpu_agent_suite_kernels-0.1.0.dist-info/    METADATA + the vllm.general_plugins entry point
  <out-dir>/manifest.json                        sha256 of every op's files

The op module is checked without importing it (it imports torch, JAX and
vLLM, which this host may not have): it must define every name in the
runtime's `OP_CONTRACT` at module level. With `--tar` the directory is also
packed, so it ships to a job as one input and is unpacked onto PYTHONPATH.

Prints one JSON object; exits 0 on success and 2 on an input error.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import sys
import tarfile

Path = pathlib.Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "tpu_agent_suite_kernels"
PACKAGE = "tpu_agent_suite_kernels"
VERSION = "0.1.0"
OP_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
# Mirrors tpu_agent_suite_kernels.OP_CONTRACT, which cannot be imported here
# without importing the package's own imports.
OP_CONTRACT = ("TARGET", "OWNED_BUCKETS", "eligible", "kernel_fn",
               "fallback_fn", "fake", "call")

EXIT_OK = 0
EXIT_USAGE = 2


class BuildError(Exception):
    """An op or plugin directory that cannot be built."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_level_names(source: str) -> set[str]:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node,
                      (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (a.asname or a.name).split(".")[0] for a in node.names)
    return names


def check_op_module(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
        names = module_level_names(source)
    except (OSError, SyntaxError) as e:
        raise BuildError(f"cannot read op module {path}: {e}") from e
    missing = [n for n in OP_CONTRACT if n not in names]
    if missing:
        raise BuildError(f"{path.name} does not define {missing}")
    target = re.search(r"^TARGET\s*=\s*[\"']([^\"']+)[\"']", source, re.M)
    if not target or ":" not in target.group(1):
        raise BuildError(f"{path.name}: TARGET must be a literal "
                         "'module.path:ClassName'")


def ensure_template(out_dir: Path) -> None:
    package = out_dir / PACKAGE
    if not package.exists():
        shutil.copytree(TEMPLATE,
                        package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      "*_test.py"))
    info = out_dir / f"{PACKAGE}-{VERSION}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {PACKAGE}\nVersion: {VERSION}\n",
        encoding="utf-8")
    # vLLM loads every vllm.general_plugins entry point in every process.
    (info / "entry_points.txt").write_text(
        f"[vllm.general_plugins]\n{PACKAGE} = {PACKAGE}:register\n",
        encoding="utf-8")


def add_op(out_dir: Path, op: str, op_module: Path, kernel: Path,
           base: Path | None) -> dict:
    if not OP_RE.match(op):
        raise BuildError(f"op {op!r} must match {OP_RE.pattern}")
    for path in (op_module, kernel) + ((base, ) if base else ()):
        if not path.is_file():
            raise BuildError(f"no file at {path}")
    check_op_module(op_module)
    ensure_template(out_dir)
    package = out_dir / PACKAGE
    files = {
        "op_module": (op_module, package / f"op_{op}.py"),
        "kernel": (kernel, package / f"kernel_{op}.py"),
    }
    if base:
        files["base"] = (base, package / f"kernel_{op}_base.py")
    for src, dst in files.values():
        shutil.copyfile(src, dst)

    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(
        encoding="utf-8")) if manifest_path.exists() else {
            "package": PACKAGE,
            "version": VERSION,
            "ops": {}
        })
    manifest["ops"][op] = {
        role: {
            "source": str(src.resolve()),
            "path": str(dst.relative_to(out_dir)),
            "sha256": sha256(dst),
        }
        for role, (src, dst) in files.items()
    }
    manifest["ops"][op]["added_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def pack(out_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for child in sorted(out_dir.iterdir()):
            tar.add(
                child,
                arcname=child.name,
                filter=lambda t: None
                if "__pycache__" in t.name or t.name.endswith(".pyc") else t)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir",
                   required=True,
                   type=Path,
                   help="plugin root; created from the template if missing")
    p.add_argument("--op", required=True, help="the op's slug")
    p.add_argument("--op-module", required=True, type=Path)
    p.add_argument("--kernel",
                   required=True,
                   type=Path,
                   help="tpu-kernel-development's best_kernel_artifact")
    p.add_argument("--base",
                   type=Path,
                   default=None,
                   help="the scoped base.py, copied in for the op's fallback")
    p.add_argument("--tar",
                   type=Path,
                   default=None,
                   help="also pack the plugin root into this .tgz")
    args = p.parse_args(argv)
    try:
        manifest = add_op(args.out_dir, args.op, args.op_module, args.kernel,
                          args.base)
        if args.tar:
            pack(args.out_dir, args.tar)
    except BuildError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return EXIT_USAGE
    print(
        json.dumps({
            "status": "ok",
            "plugin_dir": str(args.out_dir),
            "tar": str(args.tar) if args.tar else None,
            "ops": sorted(manifest["ops"]),
            "env": {
                "TPU_AGENT_SUITE_KERNELS": ",".join(sorted(manifest["ops"]))
            },
        }))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
