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
"""Report, and optionally clear, processes holding the TPU device nodes.

Runs on the accelerator host, dispatched with `compute run`.

`compute job cancel` signals the serving job's whole process group, which is
what `setsid` at submit time exists for, so it already reaches workers a
framework spawned -- including the ones vLLM disguises. This script is the
narrower second step: a process that ignored SIGTERM, or one orphaned by an
earlier crash, can keep `/dev/vfio/*` open and every later run then fails with
`Device or resource busy`.

It defaults to reporting only. Killing is destructive and indiscriminate --
anything holding a chip goes, including a workload somebody else is running on
a shared host -- so `--force` must be passed deliberately, after reading what
a plain run listed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys

DEVICE_GLOBS = ("/dev/vfio/*", "/dev/accel*")


def holders(pattern: str) -> dict[str, list[int]]:
    """Map each device matching `pattern` to the pids holding it open.

    Read from /proc rather than by shelling out to `fuser`, which is absent
    from the slim container images this may run in.
    """
    found: dict[str, list[int]] = {}
    devices = [d for d in glob.glob(pattern) if not os.path.isdir(d)]
    if not devices:
        return found
    try:
        targets = {os.path.realpath(d) for d in devices}
    except OSError:
        return found

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target in targets:
                    found.setdefault(target, []).append(int(entry))
                    break
        except OSError:
            # The process exited between listdir and readlink, or is not
            # ours to inspect. Neither is worth failing the scan over.
            continue
    return {dev: sorted(pids) for dev, pids in found.items()}


def describe(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip() or f"pid {pid}"
    except OSError:
        return f"pid {pid}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",
                        action="store_true",
                        help="SIGKILL every process found holding a device")
    parser.add_argument("--output", default="tpu_locks.json")
    args = parser.parse_args()

    held: dict[str, list[int]] = {}
    for pattern in DEVICE_GLOBS:
        held.update(holders(pattern))

    pids = sorted({pid for pids in held.values() for pid in pids})
    processes = {str(pid): describe(pid) for pid in pids}
    killed: list[int] = []
    failed: dict[str, str] = {}

    if args.force:
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError as exc:
                failed[str(pid)] = str(exc)
        # Re-scan so the report states the resulting condition rather than
        # the intent: a pid that survived SIGKILL is the thing worth knowing.
        held = {}
        for pattern in DEVICE_GLOBS:
            held.update(holders(pattern))

    report = {
        "clean": not held,
        "devices": {
            dev: pids
            for dev, pids in sorted(held.items())
        },
        "processes": processes,
        "forced": args.force,
        "killed": killed,
        "kill_failures": failed,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))

    if not held:
        print(
            "Accelerators are detached; the host is ready for new workloads.")
        return 0
    if not args.force:
        print(
            "Devices are still held. Re-run with --force to clear them, "
            "after confirming none of the processes above is a workload "
            "you want to keep.",
            file=sys.stderr)
    else:
        print(
            "Devices are still held after SIGKILL; the host likely needs a "
            "reboot.",
            file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
