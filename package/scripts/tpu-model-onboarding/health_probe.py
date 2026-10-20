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
"""Poll a serving engine's health endpoint until it answers 200.

Runs on the controller, against a port `compute port-forward` has brought
here -- so `localhost` means the forwarded engine, whichever backend it is
actually running on. On the `local` backend the forward is a no-op and this
still works, because the engine is already on this host.

Exits 0 once the endpoint answers 200, non-zero on timeout. It writes the JSON
report either way, so a caller that timed out can still read how far the
engine got.

A connection refused here has two causes that look identical: the engine is
still starting, or the port forward is not up. The report keeps the raw error
so the caller can tell them apart against `compute job status`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# vLLM compiles XLA graphs before it serves, which on a cold cache runs well
# past ten minutes. A default shorter than that reports a working deployment
# as a failed one.
DEFAULT_TIMEOUT_SEC = 1800
DEFAULT_INTERVAL_SEC = 10


def probe_once(url: str, timeout: float) -> tuple[int | None, str]:
    """Return (status_code, detail); status_code is None when unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(2048).decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        # The server answered, just not with a 2xx. That is a different state
        # from "nothing is listening" and the report must keep them apart.
        return exc.code, exc.reason or ""
    except Exception as exc:  # noqa: BLE001 - reported in the payload
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",
                        type=int,
                        required=True,
                        help="Local port (the forwarded one)")
    parser.add_argument("--path", default="/health")
    parser.add_argument("--timeout",
                        type=int,
                        default=DEFAULT_TIMEOUT_SEC,
                        help="Total seconds to keep polling")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--output",
                        default="health.json",
                        help="Where to write the report")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}{args.path}"
    deadline = time.monotonic() + args.timeout
    attempts = 0
    status: int | None = None
    detail = ""

    while True:
        attempts += 1
        status, detail = probe_once(url, timeout=min(args.interval, 10))
        if status == 200 or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)

    healthy = status == 200
    report = {
        "healthy": healthy,
        "url": url,
        "status_code": status,
        "detail": detail[:1000],
        "attempts": attempts,
        "waited_seconds":
        args.timeout - max(0, int(deadline - time.monotonic())),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if not healthy:
        print(
            f"Endpoint {url} did not return 200 within {args.timeout}s "
            f"(last: {status}). Check `compute job status` before concluding "
            f"the engine failed: a vLLM engine still compiling and a dead "
            f"port forward both look like this.",
            file=sys.stderr,
        )
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
