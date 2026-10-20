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
"""Stop-hook: resume a run that a transient backend refusal ended.

Model access is capped on input tokens per minute. A long TPU run resends its
whole conversation every turn, so it can burn millions of input tokens and be
refused partway through. The refusal kills the request, which means no model
turn exists in which the agent could notice and wait -- prompt guidance cannot
reach this failure at all.

A Stop hook can, because it runs after the loop ends and can send it back in.
That matters more than it sounds: resuming keeps the conversation, so the run
picks up where it stopped. Relaunching the process instead repeats every step
it had already completed, which on a benchmark run is most of the cost and
re-rolls the same dice.

Two things keep this from becoming a machine for hiding failures:

  * Only quota and availability refusals are retried. Anything else -- a bad
    command, a failing test, a real bug -- ends the run as it should.
  * Retries are capped and counted per conversation. At the cap the run ends
    and says why, rather than looping until something else stops it.

The wait is deliberate rather than decorative: the limit is per minute, so a
full minute restores the whole budget, while retrying sooner spends the little
that has accumulated and fails again.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tempfile
import time

MAX_RETRIES = 5
BACKOFF_SECONDS = 60

# Refusals that mean "not now" rather than "not ever". Matched against the
# harness's error text, which carries the status name, the HTTP code, or the
# human-facing sentence depending on which layer produced it.
TRANSIENT = re.compile(
    r"\b429\b|\b503\b|RESOURCE_EXHAUSTED|UNAVAILABLE"
    r"|exceeded your current quota|high demand|rate.?limit",
    re.IGNORECASE,
)

# `RESOURCE_EXHAUSTED` is a status name, not a cause, and the TPU stack borrows
# it: `E1001: CompileTimeScopedVmemOom` at compile time, and an HBM allocation
# with no `E` code at runtime. Neither is the backend being busy, so retrying
# five times spends five minutes proving it. Twelve matches in one run were the
# compile-time form, from an agent sweeping tile sizes -- the search it was
# asked to do.
#
# Textually identical to WORKLOAD in the xprof eval's `test.sh`, which applies
# the same rule across the container boundary and cannot import this. Change
# both together.
WORKLOAD_FAULT = re.compile(
    r"RESOURCE_EXHAUSTED:\s*E\d+:"
    r"|CompileTimeScopedVmemOom|VmemOom"
    r"|Attempting to allocate|Out of memory while trying to allocate"
    r"|RAN OUT OF MEMORY",
    re.IGNORECASE,
)


def state_path(conversation_id: str) -> pathlib.Path:
    """Where this conversation's retry count lives.

    Keyed by conversation so two runs cannot spend each other's budget, and
    hashed so an id that is empty or path-shaped cannot escape the directory.
    """
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
    return pathlib.Path(
        tempfile.gettempdir()) / f"tpu-agent-suite-resume-{digest}"


def read_count(path: pathlib.Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        # No state, or state we cannot read, means no retries spent yet.
        # Under-counting risks one extra wait; failing here would end a run
        # that had every right to continue.
        return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    termination = payload.get("termination") or {}
    reason = str(termination.get("reason", ""))
    error = str(termination.get("error", ""))
    conversation = str(
        (payload.get("context") or {}).get("conversation_id", "")
        or payload.get("conversationId", ""))

    ended_badly = reason == "error" or bool(error)
    blob = f"{reason} {error}"
    if not ended_badly or WORKLOAD_FAULT.search(
            blob) or not TRANSIENT.search(blob):
        json.dump({"decision": "allow"}, sys.stdout)
        return 0

    path = state_path(conversation)
    attempts = read_count(path)
    if attempts >= MAX_RETRIES:
        try:
            path.unlink()
        except OSError:
            pass
        json.dump(
            {
                "decision":
                "allow",
                "reason": (f"Backend refused this run {attempts} times"
                           f" ({error.strip()[:160]}). Giving up rather than"
                           " retrying further; the quota is per minute, so"
                           " this is worth rerunning later."),
            }, sys.stdout)
        return 0

    attempts += 1
    try:
        path.write_text(str(attempts), encoding="utf-8")
    except OSError:
        pass

    time.sleep(BACKOFF_SECONDS)
    json.dump(
        {
            "decision":
            "retry",
            "reason": (f"Backend refused the request (attempt {attempts} of"
                       f" {MAX_RETRIES}). Waited {BACKOFF_SECONDS}s for the"
                       " per-minute token budget to refill. Continue from"
                       " where you stopped; do not restart the task."),
        }, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
