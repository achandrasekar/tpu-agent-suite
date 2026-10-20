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
"""Model-agnostic accuracy eval for an OpenAI-compatible server.

Runs lm-evaluation-harness against a running server's chat-completions
endpoint, with the invocation vllm-torchtpu's own eval flow uses
(`run_eval_flow.sh`), and writes one `eval_summary.json` whose scores are keyed
by task name -- the names a study protocol's accuracy gates refer to. Given the
`eval_summary.json` of a baseline run, it also applies vllm-torchtpu's
regression gate (`check_regression.py --mode eval`): a score fails when it
drops by more than max(tolerance, 1.645 x the standard error of the
difference).

The model is never configured here: the served name comes from `/v1/models`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

Path = pathlib.Path

# Score keys lm-eval reports, most specific first; the first one a task
# carries is its score. Same order as vllm-torchtpu's check_regression.py.
SCORE_KEYS = (
    "exact_match,strict_match",
    "exact_match,custom-extract",
    "exact_match,flexible-extract",
    "acc,none",
    "exact_match,none",
    "acc_norm,none",
    "pass@1,create_test",
    "pass_at_1,create_test",
    "pass_at_1,none",
)

# One-sided 95%: the gate fails only a drop unlikely to be run-to-run noise.
Z_ONE_SIDED_95 = 1.645
# Guards the exactly-at-threshold comparison against floating-point error.
EPS = 1e-9

# Few-shot examples sent as prior chat turns, the answer continued in place.
MULTITURN_GEN_KWARGS = {
    "continue_final_message": True,
    "add_generation_prompt": False,
    "chat_template_kwargs": {
        "enable_thinking": False
    },
}
SINGLE_TURN_GEN_KWARGS = {"chat_template_kwargs": {"enable_thinking": False}}

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_SETUP = 2
EXIT_EVAL_FAILED = 3


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


def discover_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8")).get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models lists no models")
    return data[0]["id"]


def stderr_key(score_key: str) -> str:
    base, sep, filter_ = score_key.partition(",")
    return f"{base}_stderr{sep}{filter_}"


def build_command(args: argparse.Namespace, model: str, task: str,
                  output_path: Path) -> list[str]:
    model_args = (f"model={model},"
                  f"base_url={args.base_url}/v1/chat/completions,"
                  f"num_concurrent={args.num_concurrent}")
    if args.extra_model_args:
        model_args += f",{args.extra_model_args}"
    cmd = list(args.lm_eval_cmd) + [
        "--tasks",
        task,
        "--seed",
        "0,1234,None,1234",
        "--output_path",
        str(output_path),
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--apply_chat_template",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.num_fewshot is not None:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.single_turn:
        gen_kwargs = SINGLE_TURN_GEN_KWARGS
    else:
        cmd += ["--fewshot_as_multiturn", "true"]
        gen_kwargs = MULTITURN_GEN_KWARGS
    cmd += ["--gen_kwargs", args.gen_kwargs or json.dumps(gen_kwargs)]
    if args.system_instruction:
        cmd += ["--system_instruction", args.system_instruction]
    if args.include_path:
        cmd += ["--include_path", str(args.include_path)]
    # Always on: the per-question answers are how an empty or unparseable
    # reply shows, which would otherwise score as a plain wrong answer.
    cmd.append("--log_samples")
    return cmd


def written_since(output_path: Path, pattern: str, since: float) -> list[Path]:
    """The files matching pattern that this invocation wrote, oldest first.

    A reused --output-dir still holds an earlier attempt's files, which must
    not count toward this run.
    """
    files = [
        p for p in output_path.rglob(pattern) if p.stat().st_mtime >= since
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def unscored_fraction(output_path: Path, since: float) -> tuple[int, int]:
    """Counts answers that came back empty or that the task's filter could
    not extract an answer from, out of all logged answers."""
    unscored = total = 0
    for path in written_since(output_path, "samples_*.jsonl", since):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n} is not JSON: {e}") from e
                filtered = sample.get("filtered_resps") or [""]
                answer = filtered[0]
                answer = answer[0] if isinstance(answer, list) else answer
                total += 1
                if not str(answer).strip() or answer == "[invalid]":
                    unscored += 1
    return unscored, total


def read_scores(output_path: Path, requested: list[str],
                since: float) -> dict[str, dict[str, Any]]:
    """Reads every task and group score lm-eval wrote under output_path.

    A score is `headline` when it is a requested task or one of lm-eval's
    groups (MMLU's overall score and its four categories), as opposed to a
    single subject of a few dozen questions. Only headline scores are printed
    and gated, as in vllm-torchtpu's eval baselines.
    """
    scores: dict[str, dict[str, Any]] = {}
    for path in written_since(output_path, "results_*.json", since):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not JSON: {e}") from e
        groups = set(data.get("groups") or {})
        results = data.get("results") or {}
        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            key = next((k for k in SCORE_KEYS
                        if isinstance(metrics.get(k), (int, float))), None)
            if key is None:
                continue
            stderr = metrics.get(stderr_key(key))
            scores[task] = {
                "metric":
                key,
                "value":
                float(metrics[key]),
                "stderr":
                (float(stderr) if isinstance(stderr, (int, float)) else None),
                "samples":
                metrics.get("sample_len"),
                "headline":
                task in requested or task in groups,
            }
    return scores


def delta_stderr(baseline_se: float | None, run_se: float | None) -> float:
    """Standard error of (run - baseline): both runs are noisy.

    A run without a stderr is assumed as noisy as the baseline; a baseline
    without one is treated as exact.
    """
    if baseline_se is None and run_se is None:
        return 0.0
    if run_se is None:
        run_se = baseline_se
    return math.hypot(baseline_se or 0.0, run_se)


def load_baseline(path: Path) -> dict[str, dict]:
    """Reads the scores of a baseline eval_summary.json.

    A baseline that failed, or that has no headline score, would let every
    arm pass the gate trivially, so it is rejected rather than gated against.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "ok":
        raise ValueError(f"{path} has status {data.get('status')!r}, not "
                         "'ok'; rerun the baseline")
    scores = data["scores"]
    if not any(s.get("headline", True) for s in scores.values()):
        raise ValueError(f"{path} has no headline score to gate against")
    return scores


def check_gate(scores: dict[str, dict], baseline: dict[str, dict],
               tolerance: float) -> list[dict[str, Any]]:
    """Compares each baseline task's score against this run's."""
    rows = []
    for task, base in sorted(baseline.items()):
        if not base.get("headline", True):
            continue
        run = scores.get(task)
        if run is None:
            rows.append({
                "task": task,
                "baseline": base["value"],
                "value": None,
                "passed": False,
                "reason": "not measured in this run"
            })
            continue
        se = delta_stderr(base.get("stderr"), run.get("stderr"))
        allowed = max(tolerance, Z_ONE_SIDED_95 * se)
        delta = run["value"] - base["value"]
        rows.append({
            "task": task,
            "baseline": base["value"],
            "value": run["value"],
            "delta": delta,
            "allowed_drop": allowed,
            "passed": delta >= -allowed - EPS,
        })
    return rows


def format_table(scores: dict[str, dict], gate: list[dict] | None) -> str:
    by_task = {row["task"]: row for row in gate or []}
    header = ["task", "score", "stderr", "samples"]
    if gate is not None:
        header += ["baseline", "delta", "allowed drop", "gate"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    shown = {t for t, s in scores.items() if s.get("headline", True)}
    for task in sorted(shown | set(by_task)):
        s = scores.get(task, {})
        row = [
            task,
            f"{s['value']:.4f}" if s else "-",
            f"{s['stderr']:.4f}" if s.get("stderr") is not None else "-",
            str(s.get("samples") or "-"),
        ]
        if gate is not None:
            g = by_task.get(task)
            if g is None:
                row += ["-", "-", "-", "-"]
            else:
                row += [
                    f"{g['baseline']:.4f}",
                    f"{g['delta']:+.4f}" if "delta" in g else "-",
                    f"{g['allowed_drop']:.4f}" if "allowed_drop" in g else "-",
                    "pass" if g["passed"] else "FAIL",
                ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def last_api_error(log_path: Path) -> str | None:
    """The server's last error message lm-eval logged, if any.

    lm-eval retries a refused request and then dies with a bare HTTP 400; the
    server's reason (a prompt longer than max_model_len, most often) is only
    in its retry warnings.
    """
    marker = "API request failed with error message: "
    found = None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            for line in log:
                if marker in line:
                    found = line.split(marker, 1)[1].strip()
    except OSError:
        return None
    if found and found.endswith(". Retrying..."):
        found = found[:-len(". Retrying...")]
    return found[:500] if found else None


def resolve_lm_eval_cmd(value: str | None) -> list[str]:
    if value:
        return shlex.split(value)
    found = shutil.which("lm_eval") or shutil.which("lm-eval")
    if found:
        return [found]
    return [sys.executable, "-m", "lm_eval"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url",
                   default="http://localhost:8000",
                   help="Server root, without /v1 (default: %(default)s)")
    p.add_argument("--output-dir",
                   required=True,
                   type=Path,
                   help="Where lm-eval output and eval_summary.json go")
    p.add_argument("--tasks",
                   default="mmlu_llama",
                   help="Comma-separated lm-eval tasks (default: %(default)s)")
    p.add_argument("--limit",
                   type=float,
                   default=100,
                   help="Questions per (sub)task, as lm-eval --limit; "
                   "0 runs every question (default: %(default)s)")
    p.add_argument("--num-fewshot",
                   type=int,
                   default=None,
                   help="Override the task's few-shot count")
    p.add_argument("--num-concurrent",
                   type=int,
                   default=32,
                   help="Requests in flight (default: %(default)s)")
    p.add_argument("--single-turn",
                   action="store_true",
                   help="Send few-shot examples in one user turn instead of "
                   "as prior chat turns")
    p.add_argument("--gen-kwargs",
                   default=None,
                   help="JSON replacing the default --gen_kwargs, e.g. to "
                   "suppress reasoning with a template's own flag")
    p.add_argument("--system-instruction",
                   default=None,
                   help="System prompt sent with every question, e.g. the "
                   "answer format a zero-shot task's extraction regex needs")
    p.add_argument("--extra-model-args",
                   default="",
                   help="key=value pairs appended to lm-eval --model_args")
    p.add_argument("--include-path",
                   type=Path,
                   default=None,
                   help="Directory of extra lm-eval task yamls")
    p.add_argument("--max-unscored-pct",
                   type=float,
                   default=10.0,
                   help="Fail when more answers than this are empty or "
                   "unparseable (default: %(default)s)")
    p.add_argument("--model",
                   default=None,
                   help="Served model name (default: from /v1/models)")
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="eval_summary.json of the baseline run to gate against")
    p.add_argument("--tolerance",
                   type=float,
                   default=0.015,
                   help="Smallest allowed score drop, as a fraction "
                   "(default: %(default)s, i.e. 1.5 points)")
    p.add_argument("--ready-timeout",
                   type=float,
                   default=60.0,
                   help="Seconds to wait for /health before giving up")
    p.add_argument("--run-timeout",
                   type=float,
                   default=3600.0,
                   help="Seconds one task's lm-eval run may take before it "
                   "is killed and the task failed; 0 waits forever "
                   "(default: %(default)s)")
    p.add_argument("--lm-eval-cmd",
                   default=None,
                   help="Command that runs lm-eval (default: lm_eval on "
                   "PATH, else python -m lm_eval)")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print the commands without running them")
    args = p.parse_args(argv)
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    args.task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not args.task_list:
        p.error("--tasks names no task")
    if args.limit is not None:
        if args.limit <= 0:
            args.limit = None
        elif args.limit.is_integer():
            args.limit = int(args.limit)
    args.lm_eval_cmd = resolve_lm_eval_cmd(args.lm_eval_cmd)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lm_eval_dir = args.output_dir / "lm_eval"
    lm_eval_dir.mkdir(parents=True, exist_ok=True)

    baseline = None
    if args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except (OSError, KeyError, ValueError) as e:
            print(
                json.dumps({
                    "status": "error",
                    "error": f"bad --baseline: {e}"
                }))
            return EXIT_SETUP

    try:
        if args.dry_run and args.model:
            model = args.model
        else:
            wait_ready(args.base_url, args.ready_timeout)
            model = args.model or discover_model(args.base_url)
    except (TimeoutError, RuntimeError, urllib.error.URLError, OSError,
            KeyError, ValueError) as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return EXIT_SETUP

    # Whole seconds, as a filesystem may store mtimes no finer than that.
    started = math.floor(time.time())
    commands = []
    failed = []
    server_down = None
    for i, task in enumerate(args.task_list):
        cmd = build_command(args, model, task, lm_eval_dir)
        commands.append(shlex.join(cmd))
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        log_path = args.output_dir / f"{task}.log"
        print(f"[serving_eval] {task}", file=sys.stderr, flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(cmd,
                                      stdout=log,
                                      stderr=subprocess.STDOUT,
                                      check=False,
                                      timeout=args.run_timeout or None)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                # lm-eval retries a request that gets no answer, so a hung
                # server would otherwise hold it far longer than any task.
                returncode = None
        if returncode is None:
            failed.append(f"{task}: lm-eval did not finish within "
                          f"{args.run_timeout:.0f}s; see {log_path}")
        elif returncode != 0:
            reason = last_api_error(log_path)
            failed.append(f"{task}: lm-eval exited {returncode}" +
                          (f"; server said: {reason}" if reason else "") +
                          f"; see {log_path}")
        # After a failure, tell a dead server from a bad task, so the rest of
        # the tasks do not wait out a timeout each against nothing.
        if returncode != 0 and not server_alive(args.base_url):
            server_down = (f"the server stopped answering {args.base_url}"
                           f"/health during {task}")
            failed += [
                f"{t}: not run: {server_down}" for t in args.task_list[i + 1:]
            ]
            break
    if args.dry_run:
        return EXIT_OK

    try:
        scores = read_scores(lm_eval_dir, args.task_list, started)
        unscored, answered = unscored_fraction(lm_eval_dir, started)
    except ValueError as e:
        failed.append(f"unreadable lm-eval output: {e}")
        scores, unscored, answered = {}, 0, 0
    for task in args.task_list:
        if task not in scores and not any(task in f for f in failed):
            failed.append(f"{task}: lm-eval wrote no score")
    if answered and 100.0 * unscored / answered > args.max_unscored_pct:
        failed.append(
            f"{unscored}/{answered} answers were empty or had no answer the "
            "task could extract: the score measures the reply format, not "
            "the model. For a reasoning model, raise max_gen_toks, drop the "
            "stop sequences or set the answer format (--gen-kwargs, "
            "--system-instruction).")
    gate = (check_gate(scores, baseline, args.tolerance)
            if baseline is not None else None)

    if failed:
        status = "failed"
    elif gate is not None and not all(row["passed"] for row in gate):
        status = "gate_failed"
    else:
        status = "ok"
    summary = {
        "status": status,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": model,
        "client": "lm-evaluation-harness (local-chat-completions)",
        "settings": {
            "tasks": args.task_list,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "fewshot_as_multiturn": not args.single_turn,
            "gen_kwargs": args.gen_kwargs,
            "system_instruction": args.system_instruction,
            "tolerance": args.tolerance,
        },
        "commands": commands,
        "errors": failed,
        "unscored_answers": {
            "count": unscored,
            "of": answered
        },
        "scores": scores,
        "gate": gate,
    }
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1) + "\n",
                            encoding="utf-8")
    table = format_table(scores, gate)
    (args.output_dir / "eval_summary.md").write_text(table + "\n",
                                                     encoding="utf-8")
    print(table, file=sys.stderr)
    for error in failed:
        print(error, file=sys.stderr)
    print(json.dumps({"status": status, "summary": str(summary_path)}))
    if server_down:
        return EXIT_SETUP
    if failed:
        return EXIT_EVAL_FAILED
    return EXIT_GATE_FAILED if status == "gate_failed" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
