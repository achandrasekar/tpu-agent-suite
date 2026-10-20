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
"""Model-agnostic online serving benchmark for an already-running server.

Sweeps input/output length x concurrency cells with replicates, and writes one
`summary.json` whose flat metric names (`ttft_p50_ms`, `output_tokens_per_s`,
...) are the names a study protocol's `primary_metric` and
`admissibility_gates` refer to, and the names an onboarding report quotes.

Each run is driven by the engine's client:

-   `--engine vllm` (default): `vllm bench serve` -- the client
    vllm-torchtpu's own perf CI uses -- against the OpenAI-compatible server.
    The served name, the path the server loaded (used as the tokenizer) and
    `max_model_len` are read from `/v1/models`, so the same invocation
    measures any model, and the baseline and every arm of a study are
    measured by an identical client.
-   `--engine custom_fastapi`: `benchmark_fastapi_server.py`, beside this
    script, against a server built from torch-inference-coder's FastAPI
    recipe. It writes vLLM's result keys, so the summary has the same names;
    the recipe does not stream, so TTFT, TPOT and ITL are absent. That server
    has no `/v1/models`, so `--model` is required.

The client holds no chips, and on a pod of the torchtpu image without them
the plain `vllm` CLI does not start. So by default this script runs vllm as
itself, `serving_benchmark.py --as-vllm-cli <vllm CLI args>`, which keeps
torch_tpu and libtpu out of the process; see `run_as_vllm_cli`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import json
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

Path = pathlib.Path

PERCENTILES = (50, 90, 99)
LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")

# vllm bench serve result key -> summary metric name.
THROUGHPUT_KEYS = {
    "output_throughput": "output_tokens_per_s",
    "total_token_throughput": "total_tokens_per_s",
    "request_throughput": "requests_per_s",
    "duration": "duration_s",
    "completed": "completed",
    "failed": "failed",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_output_tokens",
}

ENGINES = ("vllm", "custom_fastapi")
FASTAPI_TASKS = ("generate", "embed", "vqa")
FASTAPI_CLIENT = Path(__file__).resolve().with_name(
    "benchmark_fastapi_server.py")
# First argument that makes this script run the vllm CLI; see run_as_vllm_cli.
AS_VLLM_CLI = "--as-vllm-cli"

ENDPOINTS = {
    "completions": ("vllm", "/v1/completions"),
    "chat": ("openai-chat", "/v1/chat/completions"),
}

EXIT_OK = 0
EXIT_CELL_FAILED = 1
EXIT_SETUP = 2

GCS_SCHEME = "gs://"


def local_tokenizer(name: str) -> str:
    """A local copy of a gs:// checkpoint folder's tokenizer and config files
    (no weights), made by fetch_weights.py.

    In a job, fetch_weights.py is shipped beside this script; in the package
    it is in the tpu-model-onboarding scripts.
    """
    here = Path(__file__).resolve().parent
    for folder in (here, here.parent / "tpu-model-onboarding"):
        if ((folder / "fetch_weights.py").is_file()
                and str(folder) not in sys.path):
            sys.path.insert(0, str(folder))
    try:
        import fetch_weights  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            f"{name} is a gs:// folder: ship fetch_weights.py beside this "
            "script (--input <fetch_weights.py>:fetch_weights.py)") from e
    # Its progress lines go to stderr: stdout is this script's JSON.
    with contextlib.redirect_stdout(sys.stderr):
        return fetch_weights.fetch_gcs(name, small_only=True)


def parse_cells(spec: str) -> list[tuple[int, int]]:
    """Parses "1024:128,2048:256" into [(1024, 128), (2048, 256)]."""
    cells = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        isl, sep, osl = item.partition(":")
        if not sep:
            raise ValueError(f"expected ISL:OSL, got {item!r}")
        cells.append((int(isl), int(osl)))
    if not cells:
        raise ValueError("no ISL:OSL cells given")
    return cells


def parse_ints(spec: str) -> list[int]:
    values = [int(v) for v in spec.split(",") if v.strip()]
    if not values or min(values) < 1:
        raise ValueError(f"expected positive integers, got {spec!r}")
    return values


def parse_avoid_rules(
        spec: str) -> list[tuple[int | None, int | None, int | None, str]]:
    """Parses comma-separated 'isl:osl:c' patterns ('*' matches any value)."""
    rules: list[tuple[int | None, int | None, int | None, str]] = []
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--avoid rule must be isl:osl:c (use * for any), got {raw!r}")
        parsed: list[int | None] = []
        for p in parts:
            p = p.strip()
            if p == "*":
                parsed.append(None)
            else:
                val = int(p)
                if val < 1:
                    raise ValueError(
                        f"expected positive integer or *, got {p!r} in {raw!r}"
                    )
                parsed.append(val)
        rules.append((parsed[0], parsed[1], parsed[2], raw))
    return rules


def avoid_rule_for(
    isl: int,
    osl: int,
    concurrency: int,
    rules: list[tuple[int | None, int | None, int | None, str]],
) -> str | None:
    """Returns the first matching --avoid rule string, or None."""
    for r_isl, r_osl, r_c, raw in rules:
        if ((r_isl is None or r_isl == isl) and (r_osl is None or r_osl == osl)
                and (r_c is None or r_c == concurrency)):
            return raw
    return None


def skipped_cell(isl: int, osl: int, concurrency: int,
                 error: str) -> dict[str, Any]:
    return {
        "cell": cell_name(isl, osl, concurrency),
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "status": "skipped",
        "error": error,
        "replicates": [],
    }


def filter_c_range(
    isl: int,
    osl: int,
    c_range: list[int],
    *,
    max_model_len: int | None = None,
    range_ratio: float = 0.0,
    max_concurrency_limit: int | None = None,
    avoid_rules: list[tuple[int | None, int | None, int | None, str]] = (),
) -> tuple[list[int], list[dict[str, Any]]]:
    """Static filter: adjusts c_range for (isl, osl) before running.

    Returns (kept_c_range, skipped_cells).
    """
    needed = (max_sampled_len(isl, range_ratio) +
              max_sampled_len(osl, range_ratio))
    if max_model_len and needed > max_model_len:
        err = (f"ISL+OSL up to {needed} tokens exceeds the server's "
               f"max_model_len {max_model_len}")
        return [], [skipped_cell(isl, osl, c, err) for c in c_range]

    kept: list[int] = []
    skipped: list[dict[str, Any]] = []
    for c in c_range:
        if max_concurrency_limit is not None and c > max_concurrency_limit:
            skipped.append(
                skipped_cell(
                    isl,
                    osl,
                    c,
                    f"concurrency {c} > max_concurrency_limit "
                    f"{max_concurrency_limit}",
                ))
            continue
        rule = avoid_rule_for(isl, osl, c, avoid_rules)
        if rule is not None:
            skipped.append(
                skipped_cell(isl, osl, c, f"avoided by --avoid rule {rule}"))
            continue
        kept.append(c)
    return kept, skipped


def _cell_tps(cell: dict[str, Any]) -> float | None:
    return cell.get("metrics", {}).get("output_tokens_per_s", {}).get("mean")


def check_early_stop(
    ok_cells: list[dict[str, Any]],
    curr_cell: dict[str, Any],
    *,
    stop_below_gain_pct: float | None = None,
    stop_on_failure: bool = False,
) -> str | None:
    """Dynamic early-stopping policy evaluated after each concurrency point.

    Returns the stop error string if remaining higher concurrencies in c_range
    for this (isl, osl) shape should be skipped, or None to continue.
    """
    if curr_cell["status"] == "failed":
        if stop_on_failure:
            return f"earlier concurrency c={curr_cell['concurrency']} failed"
        return None
    if stop_below_gain_pct is not None and len(ok_cells) >= 2:
        tps = [
            t for t in (_cell_tps(c) for c in [*ok_cells, curr_cell])
            if t is not None
        ]
        if len(tps) >= 3 and min(tps[:-1]) > 0:
            best_prev = max(tps[:-2])
            best_curr = max(tps[:-1])
            gain_prev = 100.0 * (tps[-2] - best_prev) / best_prev
            gain_curr = 100.0 * (tps[-1] - best_curr) / best_curr
            if (gain_prev < stop_below_gain_pct
                    and gain_curr < stop_below_gain_pct):
                return (
                    f"throughput saturated at c={curr_cell['concurrency']} "
                    f"({gain_curr:+.1f}% < {stop_below_gain_pct:.1f}% vs best "
                    f"{best_curr:.1f} for 2 steps)")
    return None


def shape_ok(shape_cells: list[dict[str, Any]], stop_on_failure: bool) -> bool:
    """Whether one (isl, osl) shape's cells pass.

    Some cell measured, and none failed unless `--stop-on-failure` made a
    failure the end of the shape. Cells skipped by the static filter or by
    early stopping do not fail it. serving_sweep.py judges each shape of a
    merged sweep by this rule.
    """
    return any(c["status"] == "ok"
               for c in shape_cells) and (stop_on_failure
                                          or not any(c["status"] == "failed"
                                                     for c in shape_cells))


def http_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url: str, timeout_s: float) -> None:
    """Waits for /health to answer 200, raising TimeoutError at the deadline."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{base_url}/health did not answer within {timeout_s:.0f}s")
        time.sleep(5)


def server_alive(base_url: str) -> bool:
    """Whether /health answers 200 right now."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def discover_model(base_url: str) -> dict[str, Any]:
    """Reads the served model's name, loaded path and max_model_len."""
    data = http_json(f"{base_url}/v1/models").get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models lists no models")
    entry = data[0]
    return {
        "id": entry["id"],
        "root": entry.get("root") or entry["id"],
        "max_model_len": entry.get("max_model_len"),
    }


def reset_prefix_cache(base_url: str, attempts: int = 10) -> bool | None:
    """Empties the server's prefix cache, so a run cannot hit the prompts an
    earlier run of the sweep left in it.

    Returns None when the server has no such endpoint (vLLM registers it only
    under VLLM_SERVER_DEV_MODE=1), else whether the reset took. A reset fails
    while blocks are still held, e.g. by requests finishing, so it is retried.
    """
    request = urllib.request.Request(f"{base_url}/reset_prefix_cache",
                                     data=b"",
                                     method="POST")
    for attempt in range(attempts):
        if attempt:
            time.sleep(1)
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                body = resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                e.close()  # It holds the response open.
                return None
            raise
        try:
            if json.loads(body).get("success", True):
                return True
        except (json.JSONDecodeError, AttributeError):
            return True  # An older server answers 200 with no JSON body.
    return False


def cell_name(isl: int, osl: int, concurrency: int) -> str:
    return f"isl{isl}_osl{osl}_c{concurrency}"


def max_sampled_len(length: int, range_ratio: float) -> int:
    # vllm bench serve samples lengths in [(1-r)*len, (1+r)*len].
    return int(length * (1 + range_ratio))


def client_name(args: argparse.Namespace) -> str:
    return ("vllm bench serve"
            if args.engine == "vllm" else FASTAPI_CLIENT.name)


def sweep_mode(args: argparse.Namespace) -> str:
    """`exploratory` when any flag may skip a point, else `fixed`: only a
    fixed sweep is comparable across a study's arms."""
    return ("exploratory" if args.stop_below_gain_pct is not None
            or args.stop_on_failure or args.avoid_rules else "fixed")


def prompts_for(args: argparse.Namespace, concurrency: int) -> int:
    """Prompts in one clean run at `concurrency`: `--num-prompts`, else
    `--prompts-per-concurrency` waves of it, at least `--min-prompts`.
    serving_sweep.py sizes its shapes by the same rule."""
    return args.num_prompts or max(args.min_prompts,
                                   args.prompts_per_concurrency * concurrency)


def build_command(args: argparse.Namespace, model: dict[str, Any], isl: int,
                  osl: int, concurrency: int, result_dir: Path,
                  result_filename: str, seed: int) -> list[str]:
    num_prompts = prompts_for(args, concurrency)
    num_warmups = (args.num_warmups
                   if args.num_warmups is not None else concurrency)
    if args.engine == "custom_fastapi":
        return build_fastapi_command(args, model, isl, osl, concurrency,
                                     num_prompts, num_warmups, result_dir,
                                     result_filename, seed)
    backend, endpoint = ENDPOINTS[args.endpoint]
    cmd = list(args.vllm_cmd) + [
        "bench",
        "serve",
        "--backend",
        backend,
        "--base-url",
        args.base_url,
        "--endpoint",
        endpoint,
        "--model",
        model["id"],
        "--tokenizer",
        args.tokenizer or model["root"],
        "--dataset-name",
        "random",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--random-range-ratio",
        str(args.range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        str(args.request_rate),
        "--num-warmups",
        str(num_warmups),
        "--percentile-metrics",
        ",".join(LATENCY_METRICS),
        "--metric-percentiles",
        ",".join(str(p) for p in PERCENTILES),
        "--seed",
        str(seed),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if not args.no_ignore_eos:
        cmd.append("--ignore-eos")
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd += shlex.split(args.extra_bench_args or "")
    return cmd


def build_fastapi_command(args: argparse.Namespace, model: dict[str, Any],
                          isl: int, osl: int, concurrency: int,
                          num_prompts: int, num_warmups: int, result_dir: Path,
                          result_filename: str, seed: int) -> list[str]:
    """The FastAPI client, given the flags vllm bench serve would get.

    It is always greedy and has no endpoint or request-rate choice: the recipe
    serves one path per task and the client sends as fast as concurrency
    allows.
    """
    cmd = [
        sys.executable,
        str(FASTAPI_CLIENT),
        "--base-url",
        args.base_url,
        "--task",
        args.task,
        "--model",
        model["id"],
        "--input-len",
        str(isl),
        "--output-len",
        str(osl),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--num-warmups",
        str(num_warmups),
        "--metric-percentiles",
        ",".join(str(p) for p in PERCENTILES),
        "--seed",
        str(seed),
        "--request-timeout",
        str(args.run_timeout or 1800.0),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.image_url:
        cmd += ["--image-url", args.image_url]
    if not args.no_ignore_eos:
        cmd.append("--ignore-eos")
    cmd += shlex.split(args.extra_bench_args or "")
    return cmd


def normalize(raw: dict[str, Any]) -> dict[str, float]:
    """Maps one vllm bench serve result to flat summary metric names."""
    metrics: dict[str, float] = {}
    for key, name in THROUGHPUT_KEYS.items():
        if raw.get(key) is not None:
            metrics[name] = raw[key]
    for m in LATENCY_METRICS:
        if raw.get(f"mean_{m}_ms") is not None:
            metrics[f"{m}_mean_ms"] = raw[f"mean_{m}_ms"]
        if raw.get(f"median_{m}_ms") is not None:
            metrics[f"{m}_p50_ms"] = raw[f"median_{m}_ms"]
        for p in PERCENTILES:
            value = raw.get(f"p{p}_{m}_ms")
            if value is not None:
                metrics[f"{m}_p{p}_ms"] = value
    tpot = metrics.get("tpot_mean_ms")
    if tpot:
        # Decode speed one user sees, as opposed to the aggregate rate.
        metrics["output_tokens_per_s_per_user"] = 1000.0 / tpot
    return metrics


def aggregate(replicates: list[dict[str, float]]) -> dict[str, dict]:
    """Mean, stdev and coefficient of variation per metric across replicates."""
    out: dict[str, dict] = {}
    names = sorted({k for r in replicates for k in r})
    for name in names:
        values = [r[name] for r in replicates if name in r]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        out[name] = {
            "mean": mean,
            "stdev": stdev,
            "cv_pct": (100.0 * stdev / mean) if mean else 0.0,
            "values": values,
        }
    return out


def failure_reason(raw: dict[str, Any], max_failed_pct: float) -> str | None:
    # vllm bench serve exits 0 even when every request fails, so the result
    # file is the only place a failed run shows.
    completed = raw.get("completed") or 0
    failed = raw.get("failed") or 0
    total = completed + failed
    if completed == 0:
        errors = [e for e in raw.get("errors") or [] if e]
        first = f": {errors[0][:300]}" if errors else ""
        return f"no request completed{first}"
    if total and 100.0 * failed / total > max_failed_pct:
        return f"{failed}/{total} requests failed (limit {max_failed_pct}%)"
    return None


def _run_client(
        args: argparse.Namespace,
        cmd: list[str],
        log_path: Path,
        reset: bool,
        label: str,
        allowed: tuple[int, ...] = (0, ),
) -> str | None:
    """Resets prefix cache if `reset` and runs `cmd`, returning an error or None."""
    if reset and not reset_prefix_cache(args.base_url):
        return "the server did not reset its prefix cache"
    print(f"[serving_benchmark] {label}", file=sys.stderr, flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd,
                                  stdout=log,
                                  stderr=subprocess.STDOUT,
                                  check=False,
                                  timeout=args.run_timeout or None)
        except subprocess.TimeoutExpired:
            # A server that stops producing tokens mid-run would otherwise
            # hold the client for as long as its own HTTP timeout.
            return (f"{client_name(args)} did not finish within "
                    f"{args.run_timeout:.0f}s; see {log_path}")
    if proc.returncode not in allowed:
        return f"{client_name(args)} exited {proc.returncode}; see {log_path}"
    return None


def run_cell(args: argparse.Namespace, model: dict[str, Any], isl: int,
             osl: int, concurrency: int, raw_dir: Path, first_seed: int,
             reset: bool) -> dict[str, Any]:
    name = cell_name(isl, osl, concurrency)
    cell: dict[str, Any] = {
        "cell": name,
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "status": "ok",
        "replicates": [],
    }
    allowed = (0, 1) if args.engine == "custom_fastapi" else (0, )
    for rep in range(1, args.replicates + 1):
        filename = f"{name}_r{rep}.json"
        seed = args.seed if reset else first_seed + rep - 1
        cell.setdefault("seeds", []).append(seed)
        cmd = build_command(args, model, isl, osl, concurrency, raw_dir,
                            filename, seed)
        cell.setdefault("command", shlex.join(cmd))
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        log_path = raw_dir / f"{name}_r{rep}.log"
        err = _run_client(args,
                          cmd,
                          log_path,
                          reset,
                          f"{name} replicate {rep}/{args.replicates}",
                          allowed=allowed)
        result_path = raw_dir / filename
        if err or not result_path.exists():
            cell["status"] = "failed"
            cell[
                "error"] = err or f"{client_name(args)} wrote no {result_path}"
            break
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        reason = failure_reason(raw, args.max_failed_pct)
        if reason:
            cell["status"] = "failed"
            cell["error"] = reason
            break
        cell["replicates"].append(normalize(raw))

    if cell["replicates"]:
        cell["metrics"] = aggregate(cell["replicates"])
    return cell


def write_summary(output_dir: Path, summary: dict[str, Any]) -> str:
    """summary.json and summary.md, each replaced whole; the table.

    Rewritten after every cell, so a sweep cut short keeps the cells it
    finished. Replaced, never rewritten in place, so whoever copies the files
    mid-sweep (compute sends declared outputs back as they change) never gets
    half of one.
    """
    table = format_table(summary["cells"])
    for name, text in (("summary.json", json.dumps(summary, indent=1)),
                       ("summary.md", table)):
        tmp = output_dir / f".{name}.tmp"
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, output_dir / name)
    return table


def format_table(cells: list[dict[str, Any]]) -> str:
    columns = [
        ("ttft_p50_ms", "TTFT p50 ms"),
        ("ttft_p99_ms", "TTFT p99 ms"),
        ("tpot_p50_ms", "TPOT p50 ms"),
        ("itl_p50_ms", "ITL p50 ms"),
        ("itl_p99_ms", "ITL p99 ms"),
        ("e2el_p50_ms", "E2E p50 ms"),
        ("output_tokens_per_s", "out tok/s"),
        ("total_tokens_per_s", "total tok/s"),
        ("output_tokens_per_s_per_user", "tok/s/user"),
        ("requests_per_s", "req/s"),
    ]
    header = ["cell", "status"] + [label for _, label in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for cell in cells:
        metrics = cell.get("metrics", {})
        row = [cell["cell"], cell["status"]]
        for key, _ in columns:
            m = metrics.get(key)
            if m is None:
                row.append("-")
            elif len(m["values"]) > 1:
                row.append(f"{m['mean']:.2f} ±{m['cv_pct']:.1f}%")
            else:
                row.append(f"{m['mean']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_as_vllm_cli(argv: list[str]) -> int:
    """Runs the vllm CLI with `argv`, keeping the TPU stack out of the process.

    On a pod of the torchtpu image without chips the plain CLI does not start:
    `import torch` autoloads torch_tpu, which brings up libtpu; with torch_tpu
    not loaded, nothing names the PrivateUse1 backend and vllm fails with
    "Expected one of cpu, cuda, ... device type: tpu"; and the torchtpu vllm
    platform plugin imports torch_tpu. A client needs none of them.

    Nothing this module imports at the top loads torch, so this runs before
    torch does. It is for `vllm bench serve`: with no platform, vllm commands
    that build an engine config, even `--version`, fail to infer a device.
    """
    # Forced, not defaulted: a caller's values, say copied from the server's
    # recipe, would bring torch_tpu back. torch reads the first one once, at
    # import; vllm bench serve needs no plugin.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["VLLM_PLUGINS"] = ""
    for name in ("libtpu", "torch_tpu"):
        # `None` makes any later import of it raise ImportError.
        sys.modules.setdefault(name, None)
    import torch  # pylint: disable=import-outside-toplevel
    try:
        # Names the backend as torch_tpu would, for vllm code that still asks
        # for the "tpu" device.
        torch.utils.rename_privateuse1_backend("tpu")
    except RuntimeError:
        pass  # Already named.
    from vllm.entrypoints.cli.main import \
        main as vllm_main  # pylint: disable=import-outside-toplevel
    sys.argv = ["vllm", *argv]
    return vllm_main()


def resolve_vllm_cmd(value: str | None) -> list[str]:
    if value:
        return shlex.split(value)
    if importlib.util.find_spec("vllm") is not None:
        return [sys.executable, str(Path(__file__).resolve()), AS_VLLM_CLI]
    # Right where there are chips, or vllm is not torchtpu's; on a pod of the
    # torchtpu image without them, the plain CLI does not start.
    print(
        f"[serving_benchmark] warning: {sys.executable} has no vllm, so "
        f"vllm runs without {AS_VLLM_CLI} and will not start on a pod "
        "without TPU chips",
        file=sys.stderr,
        flush=True)
    found = shutil.which("vllm")
    if found:
        return [found]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url",
                   default="http://localhost:8000",
                   help="Server root, without /v1 (default: %(default)s)")
    p.add_argument("--output-dir",
                   required=True,
                   type=Path,
                   help="Where raw results, logs and summary.json go")
    p.add_argument("--engine",
                   choices=ENGINES,
                   default="vllm",
                   help="Serving engine, which picks the client "
                   "(default: %(default)s)")
    p.add_argument("--task",
                   choices=FASTAPI_TASKS,
                   default="generate",
                   help="custom_fastapi only: the recipe the server was "
                   "built from (default: %(default)s)")
    p.add_argument("--image-url",
                   default=None,
                   help="custom_fastapi --task vqa only: the image every "
                   "request sends")
    p.add_argument("--isl-osl",
                   default="1024:128",
                   help="Comma-separated ISL:OSL cells (default: %(default)s)")
    p.add_argument("--concurrency",
                   default="1",
                   help="Comma-separated max concurrency levels "
                   "(default: %(default)s)")
    p.add_argument("--replicates",
                   type=int,
                   default=1,
                   help="Runs per cell; report mean, stdev and CV across them")
    p.add_argument("--num-prompts",
                   type=int,
                   default=None,
                   help="Prompts per run (default: max(--min-prompts, "
                   "--prompts-per-concurrency x concurrency))")
    p.add_argument("--prompts-per-concurrency", type=int, default=10)
    p.add_argument("--min-prompts", type=int, default=16)
    p.add_argument("--num-warmups",
                   type=int,
                   default=None,
                   help="Discarded warmup requests (default: concurrency)")
    p.add_argument("--range-ratio",
                   type=float,
                   default=0.0,
                   help="Length jitter, sampled in [(1-r)L, (1+r)L]; 0 keeps "
                   "every request at the nominal length")
    p.add_argument("--request-rate",
                   default="inf",
                   help="Requests/s; inf sends as fast as concurrency allows")
    p.add_argument("--endpoint",
                   choices=sorted(ENDPOINTS),
                   default="completions")
    p.add_argument("--temperature",
                   type=float,
                   default=0.0,
                   help="Sampling temperature (default greedy; pass a "
                   "negative value to omit and use the server default)")
    p.add_argument("--seed",
                   type=int,
                   default=42,
                   help="Prompt seed. When the server cannot reset its "
                   "prefix cache, each later run of the sweep adds one, so "
                   "no run resends an earlier run's prompts")
    p.add_argument("--no-ignore-eos",
                   action="store_true",
                   help="Let requests stop at EOS instead of running to OSL")
    p.add_argument("--model",
                   default=None,
                   help="Served model name (default: from /v1/models)")
    p.add_argument("--tokenizer",
                   default=None,
                   help="Tokenizer path, id or gs:// checkpoint folder "
                   "(default: the path the server loaded, from /v1/models); a "
                   "gs:// folder's tokenizer files are copied here")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-failed-pct",
                   type=float,
                   default=0.0,
                   help="Fail a cell when more requests than this fail")
    p.add_argument("--ready-timeout",
                   type=float,
                   default=60.0,
                   help="Seconds to wait for /health before giving up")
    p.add_argument("--run-timeout",
                   type=float,
                   default=1800.0,
                   help="Seconds one vllm bench serve run may take before it "
                   "is killed and its cell failed; 0 waits forever "
                   "(default: %(default)s)")
    p.add_argument("--vllm-cmd",
                   default=None,
                   help="Command that runs the vllm CLI (default: this "
                   f"script with {AS_VLLM_CLI} when this interpreter has "
                   "vllm, else vllm on PATH, else python -m "
                   "vllm.entrypoints.cli.main)")
    p.add_argument("--extra-bench-args",
                   default="",
                   help="Extra flags appended to every vllm bench serve call")
    p.add_argument("--max-concurrency-limit",
                   type=int,
                   default=None,
                   help="Static filter: skip concurrency points above this "
                   "limit (e.g. max_num_seqs * dp)")
    p.add_argument("--avoid",
                   default="",
                   help="Static filter: comma-separated isl:osl:c rules to "
                   "skip (* matches any, e.g. 8192:*:64)")
    p.add_argument("--stop-below-gain-pct",
                   type=float,
                   default=None,
                   help="Dynamic early-stop: stop sweeping higher "
                   "concurrencies for an ISL:OSL shape after two consecutive "
                   "steps where output_tokens_per_s improves by less than "
                   "this percentage over the best throughput so far "
                   "(e.g. 10.0)")
    p.add_argument("--stop-on-failure",
                   action="store_true",
                   help="Dynamic early-stop: stop sweeping higher "
                   "concurrencies for an ISL:OSL shape after a failed cell")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print the commands without running them")
    args = p.parse_args(argv)
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None
    if args.replicates < 1:
        p.error("--replicates must be at least 1")
    if (args.max_concurrency_limit is not None
            and args.max_concurrency_limit < 1):
        p.error("--max-concurrency-limit must be at least 1")
    try:
        args.cells = parse_cells(args.isl_osl)
        args.concurrencies = parse_ints(args.concurrency)
        args.avoid_rules = parse_avoid_rules(args.avoid)
    except ValueError as e:
        p.error(str(e))
    if args.engine == "custom_fastapi":
        if not args.model:
            p.error("--engine custom_fastapi needs --model: the server has "
                    "no /v1/models to read it from")
        if args.task == "vqa" and not args.image_url:
            p.error("--task vqa needs --image-url")
        if args.stop_below_gain_pct is not None and (args.task != "generate"
                                                     or not args.tokenizer):
            p.error("--stop-below-gain-pct needs output_tokens_per_s, which "
                    "--engine custom_fastapi measures only for --task "
                    "generate with --tokenizer")
    args.vllm_cmd = resolve_vllm_cmd(args.vllm_cmd)
    return args


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == [AS_VLLM_CLI]:
        return run_as_vllm_cli(argv[1:])
    args = parse_args(argv)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fastapi = args.engine == "custom_fastapi"
    try:
        if (args.dry_run and args.model) or fastapi:
            model = {
                "id": args.model,
                "root": args.tokenizer or args.model,
                "max_model_len": None
            }
            if not args.dry_run:
                wait_ready(args.base_url, args.ready_timeout)
        else:
            wait_ready(args.base_url, args.ready_timeout)
            model = discover_model(args.base_url)
        # The FastAPI recipe keeps no prefix cache, so there is none to reset.
        reset = (not args.dry_run and not fastapi
                 and reset_prefix_cache(args.base_url) is not None)
    except (TimeoutError, RuntimeError, urllib.error.URLError, OSError,
            KeyError, ValueError) as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return EXIT_SETUP
    if args.model:
        model["id"] = args.model
    # A gs:// checkpoint folder -- named by --tokenizer, or the path the
    # server loaded -- is copied here, tokenizer and config files only, so
    # vllm bench serve and the FastAPI client read a local directory.
    source = args.tokenizer if fastapi else (args.tokenizer or model["root"])
    if not args.dry_run and str(source or "").startswith(GCS_SCHEME):
        try:
            args.tokenizer = local_tokenizer(source)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print(
                json.dumps({
                    "status":
                    "error",
                    "error":
                    f"could not copy the tokenizer from {source}: {e}"
                }))
            return EXIT_SETUP
    # The client reaches a remote server through a port forward, so the path
    # the server loaded is on the server's filesystem, not necessarily here.
    if (not args.tokenizer and not args.dry_run and not fastapi
            and model["root"].startswith("/")
            and not Path(model["root"]).exists()):
        print(
            json.dumps({
                "status":
                "error",
                "error":
                f"the server loaded {model['root']}, which is not on this "
                "host; pass --tokenizer with the model's Hugging Face id, a "
                "local copy, or the gs:// folder it was loaded from"
            }))
        return EXIT_SETUP

    # No run may hit prompts an earlier run of the sweep left in the server's
    # prefix cache. The cache is reset before every run where the server
    # allows it; otherwise every run gets its own seed. Either way the
    # baseline and every arm send the identical sequence of prompts.
    if not reset and not args.dry_run and not fastapi:
        print(
            "[serving_benchmark] the server has no /reset_prefix_cache "
            "(start it with VLLM_SERVER_DEV_MODE=1); giving each run its "
            "own seed instead",
            file=sys.stderr,
            flush=True)
    run = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": model,
        "client": client_name(args),
        "settings": {
            "engine":
            args.engine,
            "task":
            args.task if fastapi else None,
            "num_prompts":
            args.num_prompts,
            "endpoint":
            None if fastapi else args.endpoint,
            "replicates":
            args.replicates,
            "range_ratio":
            args.range_ratio,
            "request_rate":
            args.request_rate,
            "temperature":
            args.temperature,
            "ignore_eos":
            not args.no_ignore_eos,
            "seed":
            args.seed,
            "max_concurrency_limit":
            args.max_concurrency_limit,
            "avoid":
            args.avoid or None,
            "stop_below_gain_pct":
            args.stop_below_gain_pct,
            "stop_on_failure":
            args.stop_on_failure,
            "sweep_mode":
            sweep_mode(args),
            "prefix_cache":
            ("none: the engine keeps no prefix cache"
             if fastapi else "reset per run" if reset else "seed per run"),
        },
    }
    sweep_plan = [(isl, osl, sorted(set(args.concurrencies)))
                  for isl, osl in args.cells]
    cells: list[dict[str, Any]] = []
    server_down = None
    shapes_ok = True
    for shape_idx, (isl, osl, c_range) in enumerate(sweep_plan):
        kept_c_range, shape_cells = ((
            [],
            [],
        ) if server_down else filter_c_range(
            isl,
            osl,
            c_range,
            max_model_len=model.get("max_model_len"),
            range_ratio=args.range_ratio,
            max_concurrency_limit=args.max_concurrency_limit,
            avoid_rules=args.avoid_rules,
        ))
        ok_cells: list[dict[str, Any]] = []
        stop_error: str | None = None
        for conc in (c_range if server_down else kept_c_range):
            if server_down or stop_error:
                shape_cells.append(
                    skipped_cell(isl, osl, conc,
                                 f"not run: {server_down or stop_error}"))
                continue
            grid_idx = shape_idx * len(c_range) + c_range.index(conc)
            cell = run_cell(
                args,
                model,
                isl,
                osl,
                conc,
                raw_dir,
                args.seed + grid_idx * args.replicates,
                reset,
            )
            shape_cells.append(cell)
            # After a failure, tell a dead server from a bad cell, so the rest
            # of the sweep does not wait out a timeout per cell against nothing.
            if (cell["status"] == "failed" and not args.dry_run
                    and not server_alive(args.base_url)):
                server_down = (f"the server stopped answering {args.base_url}"
                               f"/health during {cell['cell']}")
            else:
                if not args.dry_run:
                    stop_error = check_early_stop(
                        ok_cells,
                        cell,
                        stop_below_gain_pct=args.stop_below_gain_pct,
                        stop_on_failure=args.stop_on_failure,
                    )
                if cell["status"] == "ok":
                    ok_cells.append(cell)
            if not args.dry_run:
                # Kept as each cell ends, so a sweep cut short loses no cell
                # it finished. The final summary, below, replaces it.
                write_summary(
                    args.output_dir, {
                        "status":
                        "partial",
                        "error":
                        server_down,
                        **run, "cells":
                        cells +
                        sorted(shape_cells, key=lambda c: c["concurrency"])
                    })
        if not shape_ok(shape_cells, args.stop_on_failure):
            shapes_ok = False
        shape_cells.sort(key=lambda c: c["concurrency"])
        cells.extend(shape_cells)
    if args.dry_run:
        return EXIT_OK

    ok = shapes_ok and server_down is None
    summary = {
        "status": "ok" if ok else "failed",
        "error": server_down,
        **run, "cells": cells
    }
    table = write_summary(args.output_dir, summary)
    summary_path = args.output_dir / "summary.json"
    print(table, file=sys.stderr)
    for cell in cells:
        if cell["status"] != "ok":
            print(f"{cell['cell']}: {cell['status']}: {cell.get('error')}",
                  file=sys.stderr)
    print(
        json.dumps({
            "status": summary["status"],
            "summary": str(summary_path)
        }))
    if server_down:
        print(server_down, file=sys.stderr)
        return EXIT_SETUP
    return EXIT_OK if ok else EXIT_CELL_FAILED


if __name__ == "__main__":
    sys.exit(main())
