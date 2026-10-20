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
"""Read a finished serving job into a compact, checked-in *snapshot*.

A snapshot is everything the derivation needs, and nothing it does not: the served
command and env, the per-request records of the measured window, the server's counters
split warmup vs profiling, and (optionally) step costs read from the server log. It is
small enough to commit next to a reference config, which is what lets `selftest`
re-derive every reference offline.

Input is a job directory, `<jobs root>/<job-id>/`: the serving job's Kubernetes JobSet manifest
as `recipe.yaml`, its benchmark and server outputs under `outputs/`, and optionally its server
container logs under `logs/`. The root is `DEFAULT_ROOT` (`cli.py --jobs`, default
`<work>/jobs`). This module fetches nothing: the agent fetches the job's files first
(`compute job artifacts`), then points here.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import statistics
from typing import Any

import yaml

DEFAULT_ROOT = os.path.join(os.getcwd(), ".work", "jobs")

# ---------------------------------------------------------------- recipe


def _containers(recipe: dict) -> list[dict]:
    out = []
    for rj in recipe.get("spec", {}).get("replicatedJobs", []):
        pod = rj["template"]["spec"]["template"]["spec"]
        for c in pod.get("containers", []):
            out.append({"replicated_job": rj["name"], **c})
    return out


#: Runs before the recipe script. fd 3 is opened on the capture file *before* restricted mode,
#: because rbash refuses redirection to files but allows `>&3`. Restricted mode also refuses
#: commands with a `/`, `cd`, `exec` and changing PATH; PATH is an empty directory, so every
#: external command (gcloud, ray, python, sleep, tee) is a no-op that succeeds.
_PRELUDE = r'''exec 3>"$WMB_CAPTURE"
unset WMB_CAPTURE
builtin set -r
command_not_found_handle() { return 0; }
set() {
  local a=() x skip=0
  for x in "$@"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case $x in -o|+o|-?*o|+?*o) skip=1;; -?*|+?*) ;; *) a+=("$x");; esac
  done
  [ ${#a[@]} -gt 0 ] && builtin set -- "${a[@]}"
  return 0
}
exec() { [ $# -gt 0 ] && "$@"; return 0; }
vllm() {
  local n
  for n in "$@"; do printf 'A\037%s\0' "$n" >&3; done
  for n in $(compgen -e); do printf 'E\037%s=%s\0' "$n" "${!n}" >&3; done
  builtin kill -KILL 0
}
'''

#: `> file`, `>> file`, `&> file`, `2>/dev/null`; not fd duplication (`2>&1`, `>&2`).
#: rbash turns any file redirect into a failed command, which is harmless everywhere except
#: on the `vllm` call itself, so only lines that call it are stripped.
_FILE_REDIRECT = re.compile(
    r"""(?<=\s)(?:\d|&)?>>?\|?[ \t]*(?!&)(?:"[^"\n]*"|'[^'\n]*'|[^\s;|&)"'<>]+)"""
)
_CALLS_VLLM = re.compile(r"(^|[\s;(])vllm\s|VLLM_ARGS\[@\]")

#: Set by bash or by the prelude, not by the recipe.
_SHELL_ENV = {"PATH", "HOME", "PWD", "OLDPWD", "SHLVL", "_"}


def _eval_script(
        script: str,
        env: dict[str, str],
        timeout_s: float = 30) -> tuple[list[str], dict[str, str]] | None:
    """Run the recipe script in restricted bash until it calls `vllm`, and return the argv
    and the exported environment at that call, exactly as bash expanded them: `export`s,
    `${X:-default}`, `if` branches, line continuations, arrays, `VAR=x vllm ...` prefixes.
    Returns None if the script never reaches `vllm` (e.g. `python -m vllm...`).

    Filesystem tests (`[ -d /vllm-cache/... ]`) see this machine, not the pod."""
    import signal
    import subprocess
    import tempfile
    body = "\n".join(
        _FILE_REDIRECT.sub("", line) if _CALLS_VLLM.search(line) else line
        for line in script.splitlines())
    with tempfile.TemporaryDirectory() as tmp:
        cap, empty = os.path.join(tmp, "capture"), os.path.join(tmp, "bin")
        os.mkdir(empty)
        path = os.path.join(tmp, "recipe.sh")
        with open(path, "w") as f:
            f.write(_PRELUDE + "\n" + body + "\n")
        run_env = {**env, "PATH": empty, "HOME": tmp, "WMB_CAPTURE": cap}
        with open(os.path.join(tmp, "stderr"), "w") as err:
            p = subprocess.Popen(["/bin/bash", path],
                                 cwd=tmp,
                                 env=run_env,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=err,
                                 start_new_session=True)
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        recs = []
        if os.path.exists(cap):
            with open(cap, "rb") as f:
                recs = f.read().decode(errors="replace").split("\0")
    argv = [r[2:] for r in recs if r.startswith("A\x1f")]
    if not argv:
        return None
    out_env = {}
    for r in recs:
        if r.startswith("E\x1f") and "=" in r:
            k, v = r[2:].split("=", 1)
            if k not in _SHELL_ENV:
                out_env[k] = v
    return argv, out_env


def _args_from_tokens(tokens: list[str]) -> tuple[dict[str, Any], list[str]]:
    if "serve" in tokens:
        tokens = tokens[tokens.index("serve") + 1:]
    model = tokens[0] if tokens and not tokens[0].startswith("-") else None
    args: dict[str, Any] = {}
    i = 1 if model else 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        if "=" in t:
            k, v = t[2:].split("=", 1)
            args[k] = v
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[t[2:]] = tokens[i + 1]
            i += 2
        else:
            args[t[2:]] = True
            i += 1
    return args, ([model] if model else [])


def _parse_serve_args(script: str) -> tuple[dict[str, Any], list[str]]:
    """Fallback when `_eval_script` cannot reach `vllm`: pull `vllm serve` arguments out of
    the script text. Handles a `VLLM_ARGS=( ... )` array whose expansion may carry extra flags
    (`"${VLLM_ARGS[@]}" --no-dcp-q-replicate`), and a one-line `vllm serve ...`. Does not
    expand shell variables or follow `\\` continuations.
    """
    tokens: list[str] = []
    m = re.search(r"VLLM_ARGS=\(\s*\n(.*?)\n\s*\)", script, re.S)
    if m:
        body = "\n".join(line for line in m.group(1).splitlines()
                         if not line.strip().startswith('"${')
                         and "OFFLOAD_ARGS" not in line)
        tokens = shlex.split(body)
        tail = re.search(r'"\$\{VLLM_ARGS\[@\]\}"([^|\n]*)', script)
        if tail:
            # stop at the first shell redirection: `--flag 2>&1 | tee log`
            tokens += [
                t for t in shlex.split(re.split(r"\s\d?>", tail.group(1))[0])
            ]
    else:
        line = next((s for s in script.splitlines() if "vllm serve" in s), "")
        tokens = shlex.split(re.split(r"\s\d?>|\|", line)[0])
    return _args_from_tokens(tokens)


def read_recipe(job_dir: str) -> dict[str, Any]:
    recipe = yaml.safe_load(open(os.path.join(job_dir, "recipe.yaml")))
    server = bench = None
    for c in _containers(recipe):
        script = (c.get("args") or [""])[0]
        if "vllm serve" in script or "VLLM_ARGS" in script:
            server = (c, script)
        elif "aiperf" in script or "vllm bench" in script or "REPLAY_CMD" in script:
            bench = (c, script)
    if server is None:
        raise ValueError(f"{job_dir}: no container runs `vllm serve`")
    c, script = server
    cenv = {
        e["name"]: e.get("value")
        for e in c.get("env", []) if "value" in e
    }
    benv = {
        e["name"]: e.get("value")
        for e in (bench[0].get("env", []) if bench else []) if "value" in e
    }
    # the server runs on node 0 (multi-host recipes branch on the jobset index)
    ev = _eval_script(script, {
        "JOBSET_JOB_INDEX": "0",
        **{
            k: str(v)
            for k, v in cenv.items()
        }
    })
    if ev:
        argv, env = ev
        args, model = _args_from_tokens(argv)
        if "JOBSET_JOB_INDEX" not in cenv:
            env.pop("JOBSET_JOB_INDEX", None)
    else:
        args, model = _parse_serve_args(script)
        env = dict(cenv)
    meta = recipe.get("metadata", {})
    return {
        "job_id":
        meta.get("name"),
        "owner":
        _annotated_user(meta.get("annotations") or {}),
        "image":
        c.get("image"),
        "model_path":
        model[0] if model else None,
        "server_args":
        args,
        "env":
        env,
        # what the script itself set or changed at the `vllm` call, on top of container env
        "env_container":
        cenv,
        "env_from_script": {
            k: v
            for k, v in env.items() if cenv.get(k) != v
        },
        "serve_parse":
        "bash-eval" if ev else "regex-fallback (script exports not applied)",
        "bench_env":
        benv,
        "slice_workers":
        next((rj.get("replicas") for rj in recipe["spec"]["replicatedJobs"]
              if rj["name"] == c["replicated_job"]), None),
    }


# ---------------------------------------------------------------- aiperf records


def _find(job_dir: str, pattern: str) -> str | None:
    hits = sorted(
        glob.glob(os.path.join(job_dir, "outputs", "**", pattern),
                  recursive=True))
    return hits[0] if hits else None


def read_requests(job_dir: str) -> list[dict[str, Any]]:
    path = _find(job_dir, "profile_export.jsonl")
    if not path:
        raise ValueError(
            f"{job_dir}: no profile_export.jsonl (not an aiperf job?)")
    rows = []
    for line in open(path):
        d = json.loads(line)
        m, x = d["metadata"], d.get("metrics", {})

        def g(k):
            return (x.get(k) or {}).get("value")

        rows.append({
            "conv": m.get("conversation_id"),
            "turn": m.get("turn_index"),
            "kind": m.get("source_kind"),
            "phase": m.get("benchmark_phase"),
            "t0": m.get("request_start_ns", 0) / 1e9,
            "t1": (m.get("request_end_ns") or 0) / 1e9 or None,
            "isl": g("input_sequence_length") or 0,
            "osl": g("output_sequence_length") or 0,
            "ttft_ms": g("time_to_first_token"),
            "error": d.get("error") is not None,
        })
    rows.sort(key=lambda r: r["t0"])
    t0 = rows[0]["t0"] if rows else 0.0
    for r in rows:
        r["t0"] = round(r["t0"] - t0, 3)
        if r["t1"]:
            r["t1"] = round(r["t1"] - t0, 3)
    return rows


def prefix_chain(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per request (warmup included, so it seeds the chain): the trace-level prefix it
    extends. No cache rule here, only lengths:

      reusable  longest earlier (prompt + output) of the same conversation that fits in this
                prompt, taken to be its prefix; new = isl - reusable
      src_turn  the turn that prefix came from
      gap       the previous turn's prompt + output is LONGER than this prompt, so the chain
                fell back to an older turn (e.g. reasoning stripped from the history)
    """
    seen: dict[str, list[tuple[int, Any]]] = {}
    out = []
    for r in requests:
        if not r.get("isl"):
            continue
        hist = seen.setdefault(r["conv"], [])
        prior = [(e, t) for e, t in hist if e <= r["isl"]]
        base, src = max(prior, key=lambda z: z[0]) if prior else (0, None)
        prev = hist[-1][0] if hist else None
        hist.append((r["isl"] + (r.get("osl") or 0), r.get("turn")))
        out.append({
            **r, "reusable": base,
            "src_turn": src,
            "new": r["isl"] - base,
            "prev_len": prev,
            "gap": bool(prev and prev > r["isl"])
        })
    return out


def concurrency(requests: list[dict[str, Any]],
                phase: str = "profiling") -> dict | None:
    """Requests in flight from the client's start/end times, time-weighted over `phase`."""
    ev = []
    for r in requests:
        if r.get("phase") == phase and r.get("t1"):
            ev += [(r["t0"], 1), (r["t1"], -1)]
    if not ev:
        return None
    ev.sort()
    dur: dict[int, float] = {}
    cur = 0
    for (t, dx), (t2, _) in zip(ev, ev[1:]):
        cur += dx
        dur[cur] = dur.get(cur, 0.0) + (t2 - t)
    tot = sum(dur.values())

    def q(p: float) -> int:
        acc = 0.0
        for c in sorted(dur):
            acc += dur[c]
            if acc >= p * tot:
                return c
        return max(dur)

    return {
        "p50": q(0.5),
        "p90": q(0.9),
        "max": max(dur),
        "mean": round(sum(c * t for c, t in dur.items()) / tot, 2),
        "window_s": round(tot, 1),
        "time_share": {
            c: round(dur[c] / tot, 4)
            for c in sorted(dur)
        }
    }


# ---------------------------------------------------------------- server metrics


def _series(block: dict, name: str, **labels) -> dict | None:
    for key in (name, "vllm:" + name):
        if key in block:
            for s in block[key]["series"]:
                if all(
                        s.get("labels", {}).get(k) == v
                        for k, v in labels.items()):
                    return s["stats"]
    return None


def read_server_metrics(job_dir: str) -> dict[str, Any]:
    """Counters as (profiling - warmup) deltas; aiperf's profiling block is cumulative."""
    path = _find(job_dir, "server_metrics_export.json")
    if not path:
        return {}
    d = json.load(open(path))
    W, P = d.get("warmup_metrics", {}), d.get("metrics", {})

    def delta_total(name, **lab):
        p, w = _series(P, name, **lab), _series(W, name, **lab)
        if p is None:
            return None
        return (p.get("total", 0.0) or 0.0) - (
            (w or {}).get("total", 0.0) or 0.0)

    def delta_count(name):
        p, w = _series(P, name), _series(W, name)
        if p is None:
            return None
        return (p.get("count", 0) or 0) - ((w or {}).get("count", 0) or 0)

    def pct(name,
            keys=("p10_estimate", "p50_estimate", "p90_estimate",
                  "p95_estimate", "p99_estimate", "avg")):
        s = _series(P, name) or {}
        return {k.replace("_estimate", ""): s.get(k) for k in keys if k in s}

    def gauge(name):
        s = _series(P, name) or {}
        return {
            k: s.get(k)
            for k in ("avg", "p10", "p25", "p50", "p75", "p90", "max")
        }

    def buckets(name):
        """Per-bin counts (not cumulative), profiling - warmup, keyed by upper bound."""
        for key in (name, "vllm:" + name):
            if key in P:
                p = P[key]["series"][0].get("buckets") or {}
                w = (W.get(key) or {
                    "series": [{}]
                })["series"][0].get("buckets") or {}
                cum = [(b, p[b] - w.get(b, 0)) for b in p]
                return {
                    b: c - (cum[i - 1][1] if i else 0)
                    for i, (b, c) in enumerate(cum)
                }
        return None

    ranges = d.get("summary", {}).get("phase_time_ranges", {})
    prof = ranges.get("profiling", {})
    accepted_per_pos = []
    for pos in range(8):
        v = delta_total("spec_decode_num_accepted_tokens_per_pos",
                        position=str(pos))
        if v is None:
            break
        accepted_per_pos.append(v)
    return {
        "profiling_duration_s":
        (prof.get("end_ns", 0) - prof.get("start_ns", 0)) / 1e9,
        "prompt_tokens":
        delta_total("prompt_tokens"),
        "prompt_tokens_cached":
        delta_total("prompt_tokens_by_source", source="local_cache_hit"),
        "prompt_tokens_computed":
        delta_total("prompt_tokens_by_source", source="local_compute"),
        "generation_tokens":
        delta_total("generation_tokens"),
        "iterations":
        delta_count("iteration_tokens_total"),
        "spec_drafts":
        delta_total("spec_decode_num_drafts"),
        "spec_draft_tokens":
        delta_total("spec_decode_num_draft_tokens"),
        "spec_accepted":
        delta_total("spec_decode_num_accepted_tokens"),
        "spec_accepted_per_pos":
        accepted_per_pos,
        "preemptions":
        delta_total("num_preemptions"),
        "itl_s":
        pct("inter_token_latency_seconds"),
        "running":
        gauge("num_requests_running"),
        "waiting":
        gauge("num_requests_waiting"),
        "kv_usage":
        gauge("kv_cache_usage_perc"),
        # tokens per engine iteration = computed prompt + generated. vLLM books a request's
        # whole computed prefill in the iteration it completes, so bins above the largest
        # decode-only step are (approximately) per-request computed prompt tokens.
        "iteration_tokens_buckets":
        buckets("iteration_tokens_total"),
    }


#: `vllm:iteration_tokens_total` upper bounds kept per slice: <=1, <=8, <=16, <=32 (decode
#: under MTP: 4 x D tokens), <=512 (a small chunk rides the step), >512 (a prefill chunk).
SLICE_EDGES = ("1.0", "8.0", "16.0", "32.0", "512.0", "+Inf")


def read_step_slices(job_dir: str) -> dict[str, Any] | None:
    """aiperf's 1-s scrape of the step-token histogram and the running gauge, over the
    measured window, in the requests' time origin (`read_requests`). Compact enough to keep
    in the snapshot; `core/mix.py` turns it into the step mix.

    One row per second: [t, steps per SLICE_EDGES bin (6), running max, running min,
    running avg]. The histogram is observed when a step ENDS, so a second with no steps and
    requests running is inside a long step."""
    path, rq = _find(job_dir, "server_metrics_export.json"), _find(
        job_dir, "profile_export.jsonl")
    if not (path and rq):
        return None
    origin = min(
        json.loads(line)["metadata"].get("request_start_ns", 0)
        for line in open(rq)) / 1e9
    d = json.load(open(path))
    M = d.get("metrics", {})
    prof = d.get("summary", {}).get("phase_time_ranges",
                                    {}).get("profiling", {})
    if "vllm:iteration_tokens_total" not in M or not prof:
        return None
    lo, hi = prof["start_ns"] / 1e9, prof["end_ns"] / 1e9
    run = {
        t["start_ns"]: t
        for t in M["vllm:num_requests_running"]["series"][0]["timeslices"]
    }
    rows, last = [], 0.0
    for t in M["vllm:iteration_tokens_total"]["series"][0]["timeslices"]:
        t0 = t["start_ns"] / 1e9
        if not lo <= t0 < hi:
            continue
        cum = [t["buckets"].get(e, 0) for e in SLICE_EDGES]
        n = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
        r = run.get(t["start_ns"])
        rmax, rmin, ravg = (r["max"], r["min"],
                            r["avg"]) if r else (last, last, last)
        last = rmax
        rows.append([round(t0 - origin, 3), *n, rmax, rmin, round(ravg, 3)])
    return {
        "edges":
        list(SLICE_EDGES),
        "window": [round(lo - origin, 3),
                   round(hi - origin, 3)],
        "columns": [
            "t", *(f"n_le{e}" for e in SLICE_EDGES), "run_max", "run_min",
            "run_avg"
        ],
        "rows":
        rows
    }


# ---------------------------------------------------------------- server log (optional)

_SCHED = re.compile(
    r"(\d\d):(\d\d):(\d\d) .*\[vLLM Scheduler\] Step: (\d+) prefill tokens.*"
    r"running_reqs=(\d+)")

#: Startup facts read from the server log. The TPU runtime resolves a hybrid (GDN / mamba)
#: model's page size itself (`--block-size` is only a floor), and logs it with the block count.
_LOG_FACTS = {
    "kv_pool_tokens":
    (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), "GPU KV cache size"),
    "block_size_resolved":
    (re.compile(r"TPU block_size derivation path: .*final_block_size=(\d+)"),
     "final_block_size="),
    "kv_num_blocks":
    (re.compile(r"\bnum_blocks=(\d+) num_groups=\d+"), "num_blocks="),
}

#: vLLM logs the sampling defaults it took from the checkpoint's generation_config.json once at
#: startup (vllm/config/model.py): "Default vLLM sampling parameters have been overridden by the
#: model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`".
_SAMPLING_CHEAP = "sampling parameters have been overridden"
_SAMPLING_RX = re.compile(r"(\{[^{}]*\})")


def parse_sampling_line(line: str) -> dict[str, Any] | None:
    """The sampling defaults from the server's startup line, else None (pure, unit-tested)."""
    import ast
    if _SAMPLING_CHEAP not in line:
        return None
    m = _SAMPLING_RX.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    return {
        k: d[k]
        for k in ("temperature", "top_k", "top_p", "min_p",
                  "repetition_penalty") if k in d
    } if isinstance(d, dict) else None


def read_sampling_defaults(job_dir: str,
                           max_lines: int = 2_000_000
                           ) -> dict[str, Any] | None:
    """The served sampling defaults (`parse_sampling_line`) from the job's server log (the API
    server logs it after engine init: line 338,608 on j-9fb52480), with the log
    it came from; None when the server logged no override (vLLM's own defaults then apply,
    unless the client sent per-request parameters)."""
    import gzip
    for path in _server_logs(job_dir):
        gz = path.endswith(".gz")
        with (gzip.open(path, "rt", errors="replace")
              if gz else open(path, errors="replace")) as f:
            for i, line in enumerate(f):
                d = parse_sampling_line(line)
                if d is not None:
                    return {
                        **d, "source":
                        f"server log {os.path.relpath(path, job_dir)}: "
                        "'Default vLLM sampling parameters have been overridden'"
                    }
                if i > max_lines:
                    break
    return None


def _server_logs(job_dir: str) -> list[str]:
    """`vllm.log` in the outputs, else the server container's own log as the job platform saved it
    (`logs/<job>-vllm-server-0/<pod>/vllm-server.log.gz`), largest first: a restarted job keeps
    the short log of its failed attempt beside the real one."""
    path = _find(job_dir, "vllm.log")
    if path:
        return [path]
    hits = glob.glob(
        os.path.join(job_dir, "logs", "*", "*", "vllm-server.log*"))
    return sorted(hits, key=os.path.getsize, reverse=True)


def read_log(job_dir: str, scan_steps: bool) -> dict[str, Any]:
    """Startup facts always (KV pool size, resolved page size, block count; all near the top);
    prefill step cost only if asked and the log is plain text, since it can be gigabytes. Step
    cost = mean gap between consecutive full-budget prefill steps, the only gap the 1 s
    timestamps can resolve (async scheduling blurs the rest)."""
    import gzip
    out: dict[str, Any] = {}
    logs = _server_logs(job_dir)
    for path in logs:
        gz = path.endswith(".gz")
        with (gzip.open(path, "rt", errors="replace")
              if gz else open(path, errors="replace")) as f:
            for i, line in enumerate(f):
                for key, (rx, cheap) in _LOG_FACTS.items():
                    if key not in out and cheap in line:
                        m = rx.search(line)
                        if m:
                            out[key] = int(m.group(1).replace(",", ""))
                if len(out) == len(_LOG_FACTS) or i > 2_000_000:
                    break
        if "kv_pool_tokens" in out:
            out["log_path"] = os.path.relpath(path, job_dir)
            break
    if logs and "log_path" not in out:
        out["log_path"] = os.path.relpath(logs[0], job_dir)
    if not scan_steps or not out.get("log_path") or out["log_path"].endswith(
            ".gz"):
        return out
    with open(os.path.join(job_dir, out["log_path"]), errors="replace") as f:
        f.seek(0)
        rows = []
        for line in f:
            if "[vLLM Scheduler] Step:" not in line:
                continue
            m = _SCHED.search(line)
            if m:
                h, mi, s, p, r = map(int, m.groups())
                rows.append((h * 3600 + mi * 60 + s, p, r))
    budget = max((p for _, p, _ in rows), default=0)
    gaps = [
        b[0] - a[0] for a, b in zip(rows, rows[1:]) if a[1] >= budget -
        64 and b[1] >= budget - 64 and 0 < b[0] - a[0] <= 20
    ]
    if gaps:
        out["prefill_step_s"] = round(statistics.mean(gaps), 3)
        out["prefill_step_pairs"] = len(gaps)
        out["prefill_budget_tokens"] = budget
    return out


# ---------------------------------------------------------------- snapshot

#: Snapshot line width. A request record (~250 characters) fits on one line.
SNAPSHOT_WIDTH = 260


def dumps_snapshot(snap: Any, width: int = SNAPSHOT_WIDTH) -> str:
    """A snapshot as JSON with one record per line rather than one field per line.

    An object or list that fits in `width` stays on one line; a longer one is
    expanded one entry per line. A list of short flat entries (the step-slice
    rows) is packed several to a line. `json.loads` reads it back unchanged.
    """

    def compact(v: Any) -> str:
        return json.dumps(v, separators=(", ", ": "))

    def flat(v: Any) -> bool:
        return not isinstance(v, (dict, list)) or (isinstance(
            v, list) and not any(isinstance(x, (dict, list)) for x in v))

    def fmt(v: Any, ind: int) -> str:
        pad = " " * ind
        one_line = compact(v)
        if not isinstance(
                v, (dict, list)) or not v or len(one_line) + ind <= width:
            return one_line
        if isinstance(v, dict):
            entries = [
                f"{pad} {json.dumps(k)}: {fmt(x, ind + 1)}"
                for k, x in v.items()
            ]
        else:
            parts = [compact(x) for x in v]
            if all(flat(x) for x in v) and max(map(len, parts)) <= width // 4:
                entries, line = [], ""
                for part in parts:
                    if line and len(line) + len(part) + ind + 3 > width:
                        entries.append(f"{pad} {line}")
                        line = part
                    else:
                        line = f"{line}, {part}" if line else part
                entries.append(f"{pad} {line}")
            else:
                entries = [f"{pad} {fmt(x, ind + 1)}" for x in v]
        open_, close = ("{", "}") if isinstance(v, dict) else ("[", "]")
        return f"{open_}\n" + ",\n".join(entries) + f"\n{pad}{close}"

    return fmt(snap, 0) + "\n"


def _annotated_user(annotations: dict[str, Any]) -> str | None:
    """The submitting user, from the job platform's `<domain>/username` annotation."""
    return next(
        (v for k, v in sorted(annotations.items()) if k.endswith("/username")),
        None)


def read_checkpoint_config(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The checkpoint's config.json, from the GCS mirror the bench container names
    (`GCS_MODEL_SRC`). Read once at onboarding and stored in the snapshot, so later
    derivations are offline."""
    import subprocess
    src = rec.get("bench_env", {}).get("GCS_MODEL_SRC") or rec.get(
        "env", {}).get("GCS_MODEL_SRC")
    if not src:
        # recipes that copy `<prefix>/$MODEL_NAME` (infx-qwen35-agentx) or take MODEL_ID
        # (the generic bench recipes) instead of naming the mirror. The prefix is the job's
        # MODEL_GCS_PREFIX, else WMB_MODEL_GCS_PREFIX (e.g. gs://<bucket>/ckpt/hf); with
        # neither, there is no mirror to read.
        env = {**rec.get("env_container", {}), **rec.get("env", {})}
        mid = env.get("MODEL_NAME") or env.get("MODEL_ID")
        prefix = env.get("MODEL_GCS_PREFIX") or os.environ.get(
            "WMB_MODEL_GCS_PREFIX")
        if prefix and mid and "/" in mid and not mid.startswith(
            ("/", "gs://")):
            src = f"{prefix.rstrip('/')}/{mid}"
    if not src:
        return None
    r = subprocess.run(
        ["gcloud", "storage", "cat", f"{src.rstrip('/')}/config.json"],
        capture_output=True,
        text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"cannot read {src}/config.json: {r.stderr.strip()[:200]}")
    cfg = json.loads(r.stdout)
    cfg["_source"] = f"{src.rstrip('/')}/config.json"
    return cfg


def snapshot(job_id: str,
             root: str | None = None,
             scan_log: bool = True) -> dict[str, Any]:
    job_dir = os.path.join(root or DEFAULT_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise FileNotFoundError(
            f"{job_dir} missing. Fetch the job's files into it first "
            f"(`compute job artifacts --job-id {job_id} --out-dir {job_dir}`)")
    rec = read_recipe(job_dir)
    return {
        "_what":
        "workload-microbenchmark job snapshot: the inputs a derivation reads",
        "job_id": job_id,
        "recipe": rec,
        "checkpoint_config": read_checkpoint_config(rec),
        "requests": read_requests(job_dir),
        "server": read_server_metrics(job_dir),
        "step_slices": read_step_slices(job_dir),
        "log": read_log(job_dir, scan_log),
        "sampling_defaults": read_sampling_defaults(job_dir),
    }
