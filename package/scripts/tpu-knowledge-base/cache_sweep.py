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
"""SessionStart hook: sweep the tpu-knowledge-base cache, at most once a day.

A cached page is only ever replaced by a lookup of its row, so one nobody
asks for again would stay for good. Something has to reap, and a session's
start is the moment: it always runs, where SessionEnd does not when a
session is killed, and nothing in the session has been handed a cached
path yet. A lookup used to do this on the side, which made a read command
delete files and gave its CLI a `--purge`; neither belongs to a lookup.

As a hook, the event arrives on stdin and is not needed; nothing is
printed, which the dispatcher reads as allow, and a sweep that cannot run
is no reason to hold up the session, so none is raised. By hand, `--now`
sweeps whether or not one ran today and reports what went; with
`--dry-run` it reports and removes nothing.

A hook is in every build; the summariser and catalog beside this file
ship with the tpu-knowledge-base agent. A build without that agent has no
index to sweep against, so the hook exits clean and silent, and `--now`
says why.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str):
    path = _HERE / f"{name}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("kb_summarize")  # loads catalog beside it


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete cached pages no lookup will serve again.")
    parser.add_argument("--now",
                        action="store_true",
                        help="Sweep whether or not one ran today, and say "
                        "what went. Without it: the session-start sweep, "
                        "once a day and silent")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="With --now: say what would go, delete nothing")
    args = parser.parse_args(argv)
    if args.dry_run and not args.now:
        parser.error("--dry-run goes with --now")
    if summarize is None:
        if args.now:
            parser.error("this build has no tpu-knowledge-base index to sweep "
                         "against; the catalog is not beside this script")
        return 0
    if args.now:
        rows = summarize.catalog.rows()
        print(json.dumps(summarize.purge(rows, args.dry_run), indent=2))
        return 0
    summarize.sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
