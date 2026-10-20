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
"""Settle, collect, and validate XProf traces from a running vLLM server.

`tpu-diagnostics` is the single profiling entry point in this package. When
given a running `vllm` server (`serve_vllm.sh`), the server's profiler is
already armed at startup and serves `VLLM_PROFILE_DIR` over HTTP on
`files_port` (`8790` by default). Because the XProf trace writer flushes in the
background after `POST /stop_profile` returns, reading or downloading
immediately can copy a partial trace or pick up an earlier session's
`<timestamp>/` directory.

This script handles the three deterministic steps around the profile window:

  1. Snapshot existing trace files before `/start_profile` (`--snapshot`), or
     run a full `/start_profile` -> load -> `/stop_profile` window when given
     `--load-cmd`.
  2. Wait until newly written files under `VLLM_PROFILE_DIR` (over HTTP at
     `--files-url` or on disk at `--profile-dir`) appear and stop growing for
     `--settle-s` seconds, then download/move only the new files into
     `--out-dir`, verifying `Content-Length` on every HTTP file.
  3. Discover captured phased-profiler phase directories (`<phase>/plugins/profile/`,
     or top-level `plugins/profile/` for the plain profiler) and verify that at
     least one `*.xplane.pb` file carries a `/device:TPU:` plane rather than a
     host-only trace (`PYTHON_TRACER_LEVEL=1` overflow).

Standard library only. Prints a JSON summary on stdout and exits 0 when valid
TPU device traces were collected, or 1 on failure (`error` in the JSON output).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

Path = pathlib.Path

_HREF_RE = re.compile(r'<a\s+href="([^"]+)"', re.IGNORECASE)


def default_files_url(base_url: str, files_port: int = 8790) -> str:
    """The HTTP directory server `serve_vllm.sh` runs on `VLLM_PROFILE_FILES_PORT`."""
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return urllib.parse.urlunsplit(
        (parsed.scheme or "http", f"{host}:{files_port}", "", "", ""))


def has_tpu_plane(root: Path) -> tuple[int, int]:
    """Returns `(total *.xplane.pb count, count with a /device:TPU: XPlane)`.

    XProf stores each trace as an `XSpace` protobuf (`*.xplane.pb`) made of
    named `XPlane` timelines: `/host:CPU` for host threads and `/device:TPU:0`,
    `/device:TPU:1`, ... for TPU device execution. A capture without any
    `/device:TPU:` plane is host-only (e.g. when Python tracing overflowed the
    buffer) and has no device kernel timings.
    """
    files = sorted(root.rglob("*.xplane.pb"))
    with_tpu = 0
    for f in files:
        with open(f, "rb") as fh:
            if b"/device:TPU:" in fh.read():
                with_tpu += 1
    return len(files), with_tpu


def captured_phases(root: Path) -> list[str]:
    """Phase names (`prefill_only`, `decode_only`, ...) under `root`."""
    phases = []
    for p in sorted(root.glob("*/plugins/profile")):
        if p.is_dir():
            phases.append(p.parent.parent.name)
    return phases


def _local_file_sizes(root: Path,
                      exclude: set[str] | None = None) -> dict[str, int]:
    if not root.exists():
        return {}
    out: dict[str, int] = {}
    for f in root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(root).as_posix()
            if not exclude or rel not in exclude:
                out[rel] = f.stat().st_size
    return out


def _settle(
    get_sizes: Any,
    quiet_s: float = 15.0,
    limit_s: float = 300.0,
) -> tuple[bool, dict[str, int]]:
    """Polls `get_sizes()` until new trace files appear and stop growing."""
    start = time.monotonic()
    deadline = start + limit_s
    empty_wait = min(limit_s, max(quiet_s, 15.0) if quiet_s > 0 else 0.0)
    last: dict[str, int] | None = None
    since = start
    while True:
        sizes = get_sizes()
        now = time.monotonic()
        if sizes != last:
            last, since = sizes, now
        elif sizes and sum(sizes.values()) > 0 and now - since >= quiet_s:
            return True, sizes
        if not sizes and now - start >= empty_wait:
            return False, sizes
        if now >= deadline:
            return bool(sizes and sum(sizes.values()) > 0), sizes
        time.sleep(min(2.0, max(0.05, quiet_s / 4.0 if quiet_s else 0.05)))


def settle_local(root: Path,
                 before: set[str],
                 quiet_s: float = 15.0,
                 limit_s: float = 300.0) -> bool:
    """Waits until new trace files under `root` appear and stop growing."""
    settled, _ = _settle(lambda: _local_file_sizes(root, exclude=before),
                         quiet_s=quiet_s,
                         limit_s=limit_s)
    return settled


def collect_local_traces(src_root: Path, before: set[str], dest: Path,
                         quiet_s: float, limit_s: float) -> bool:
    """Moves newly written trace files from `src_root` into `dest`."""
    settled = settle_local(src_root, before, quiet_s=quiet_s, limit_s=limit_s)
    new_files = _local_file_sizes(src_root, exclude=before)
    for rel in sorted(new_files):
        src = src_root / rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
    for d in sorted(
        (p for p in src_root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
    ):
        with contextlib.suppress(OSError):
            d.rmdir()
    return settled


def list_http_files(files_url: str,
                    prefix: str = "",
                    timeout: float = 10.0,
                    exclude: set[str] | None = None) -> dict[str, int] | None:
    """Recursively lists files and Content-Lengths from `python3 -m http.server`.

    Returns None when the root is unreachable, else `{rel_path: size_bytes}`.
    """
    base = files_url.rstrip("/")
    url = f"{base}/{prefix}" if prefix else f"{base}/"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None if not prefix else {}
    out: dict[str, int] = {}
    for raw_href in _HREF_RE.findall(html):
        href = urllib.parse.unquote(raw_href)
        if (not href or href.startswith(("/", "?", "#", ".."))
                or ".." in href.split("/")):
            continue
        if href.endswith("/"):
            sub = list_http_files(base,
                                  f"{prefix}{href}",
                                  timeout=timeout,
                                  exclude=exclude)
            if sub:
                out.update(sub)
        else:
            rel = f"{prefix}{href}"
            if exclude and rel in exclude:
                continue
            req = urllib.request.Request(
                f"{base}/{urllib.parse.quote(rel, safe='/')}", method="HEAD")
            size = 0
            with contextlib.suppress(urllib.error.URLError, OSError,
                                     ValueError):
                with urllib.request.urlopen(req, timeout=timeout) as head:
                    size = int(head.headers.get("Content-Length") or 0)
            out[rel] = size
    return out


def settle_http(files_url: str,
                before: set[str],
                quiet_s: float = 15.0,
                limit_s: float = 300.0) -> tuple[bool, dict[str, int]]:
    """Waits until new files on `files_url` appear and stop growing."""
    return _settle(lambda: list_http_files(files_url, exclude=before) or {},
                   quiet_s=quiet_s,
                   limit_s=limit_s)


def collect_http_traces(files_url: str, before: set[str], dest: Path,
                        quiet_s: float,
                        limit_s: float) -> tuple[bool, str | None]:
    """Downloads newly written trace files from `files_url` into `dest`."""
    settled, new_files = settle_http(files_url,
                                     before,
                                     quiet_s=quiet_s,
                                     limit_s=limit_s)
    base = files_url.rstrip("/")
    for rel in sorted(new_files):
        url = f"{base}/{urllib.parse.quote(rel, safe='/')}"
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                expected = resp.headers.get("Content-Length")
                data = resp.read()
        except (urllib.error.URLError, OSError) as e:
            return settled, f"failed to download {url}: {e}"
        if expected is not None and len(data) < int(expected):
            return (
                settled,
                f"incomplete download for {rel}: got {len(data)} of {expected} bytes"
            )
        target.write_bytes(data)
    return settled, None


def post_endpoint(base_url: str,
                  path: str,
                  timeout: float = 60.0) -> str | None:
    """POSTs to `<base_url><path>`, returning an error string or None."""
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return None
    except urllib.error.HTTPError as e:
        e.close()
        return f"POST {url} returned HTTP {e.code} (is the server started with --profiler-config?)"
    except (urllib.error.URLError, OSError) as e:
        return f"POST {url} failed: {e}"


def validate_directory(dest: Path) -> dict[str, Any]:
    """Inspects `dest` for phased/plain profile directories and `/device:TPU:` planes."""
    phases = captured_phases(dest)
    total_xplane, tpu_xplane = has_tpu_plane(dest)
    result: dict[str, Any] = {
        "status": "ok",
        "dir": str(dest),
        "phases": phases,
        "phase_dirs": {
            p: str(dest / p)
            for p in phases
        },
        "xplane_files": total_xplane,
        "xplane_files_with_tpu": tpu_xplane,
    }
    if total_xplane == 0:
        result["status"] = "failed"
        result["error"] = "no *.xplane.pb trace files were written"
    elif tpu_xplane == 0:
        result["status"] = "failed"
        result["error"] = (
            "capture has no /device:TPU: plane (host-only trace; check "
            "PYTHON_TRACER_LEVEL=0)")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url",
                   default="http://localhost:8000",
                   help="vLLM server root URL (default: %(default)s)")
    p.add_argument("--files-url",
                   default=None,
                   help="HTTP directory server URL for VLLM_PROFILE_DIR "
                   "(default: <base-url host>:8790)")
    p.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Local VLLM_PROFILE_DIR on disk (instead of --files-url)")
    p.add_argument("--out-dir",
                   type=Path,
                   default=None,
                   help="Destination directory for collected traces")
    p.add_argument(
        "--snapshot-file",
        type=Path,
        default=None,
        help="File to write (--snapshot) or read (during collection) "
        "the pre-capture file listing so earlier sessions are excluded")
    p.add_argument(
        "--snapshot",
        action="store_true",
        help="Only record the current file listing to --snapshot-file "
        "and exit")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an already-collected --out-dir and exit")
    p.add_argument("--load-cmd",
                   default=None,
                   help="Shell command to run the traffic load. Unless "
                   "--no-toggle-profile is set (e.g. when --load-cmd already "
                   "passes --profile to vllm bench serve), this script POSTs "
                   "/start_profile before --load-cmd and /stop_profile after")
    p.add_argument("--no-toggle-profile",
                   action="store_true",
                   help="Do not POST /start_profile and /stop_profile around "
                   "--load-cmd (use when --load-cmd already triggers them, "
                   "such as `vllm bench serve --profile`)")
    p.add_argument(
        "--settle-s",
        type=float,
        default=15.0,
        help="Seconds trace files must stop growing before download "
        "(default: %(default)s)")
    p.add_argument("--settle-timeout-s",
                   type=float,
                   default=300.0,
                   help="Maximum seconds to wait for trace files to settle "
                   "(default: %(default)s)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/").removesuffix("/v1")
    files_url = args.files_url or default_files_url(base_url)

    if args.validate_only:
        if args.out_dir is None:
            print(
                json.dumps({
                    "status": "failed",
                    "error": "--validate-only requires --out-dir"
                }))
            return 1
        res = validate_directory(args.out_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "ok" else 1

    if args.snapshot:
        if args.profile_dir is not None:
            files = sorted(_local_file_sizes(args.profile_dir))
        else:
            listing = list_http_files(files_url)
            if listing is None:
                print(
                    json.dumps({
                        "status":
                        "failed",
                        "error":
                        f"profile files server at {files_url} did not answer",
                    }))
                return 1
            files = sorted(listing)
        payload = {"files": files}
        if args.snapshot_file is not None:
            args.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot_file.write_text(json.dumps(payload, indent=2) + "\n",
                                          encoding="utf-8")
        print(json.dumps({"status": "ok", **payload}, indent=2))
        return 0

    if args.out_dir is None:
        print(
            json.dumps({
                "status":
                "failed",
                "error":
                "--out-dir is required unless --snapshot is set"
            }))
        return 1

    before: set[str] = set()
    if args.snapshot_file is not None and args.snapshot_file.is_file():
        data = json.loads(args.snapshot_file.read_text(encoding="utf-8"))
        before = set(data.get("files") or [])
    elif args.load_cmd:
        if args.profile_dir is not None:
            before = set(_local_file_sizes(args.profile_dir))
        else:
            listing = list_http_files(files_url)
            if listing is None:
                print(
                    json.dumps({
                        "status":
                        "failed",
                        "error":
                        f"profile files server at {files_url} did not answer",
                    }))
                return 1
            before = set(listing)

    if args.load_cmd:
        if not args.no_toggle_profile:
            err = post_endpoint(base_url, "/start_profile")
            if err:
                print(json.dumps({"status": "failed", "error": err}))
                return 1
        proc = subprocess.run(args.load_cmd, shell=True, check=False)
        stop_err = None
        if not args.no_toggle_profile:
            stop_err = post_endpoint(base_url, "/stop_profile")
        if proc.returncode != 0:
            print(
                json.dumps({
                    "status": "failed",
                    "error": f"--load-cmd exited with {proc.returncode}"
                }))
            return 1
        if stop_err:
            print(json.dumps({"status": "failed", "error": stop_err}))
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dl_error = None
    if args.profile_dir is not None:
        settled = collect_local_traces(args.profile_dir,
                                       before,
                                       args.out_dir,
                                       quiet_s=args.settle_s,
                                       limit_s=args.settle_timeout_s)
    else:
        if list_http_files(files_url) is None:
            print(
                json.dumps({
                    "status":
                    "failed",
                    "error":
                    f"profile files server at {files_url} did not answer",
                }))
            return 1
        settled, dl_error = collect_http_traces(files_url,
                                                before,
                                                args.out_dir,
                                                quiet_s=args.settle_s,
                                                limit_s=args.settle_timeout_s)
    if dl_error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "settled": settled,
                    "error": dl_error
                },
                indent=2))
        return 1

    res = validate_directory(args.out_dir)
    res["settled"] = settled
    if args.load_cmd:
        res["load_cmd"] = args.load_cmd
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
