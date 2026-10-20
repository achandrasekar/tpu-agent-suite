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
"""Benchmark a `custom_fastapi` server the way `vllm bench serve` would.

`vllm bench serve` speaks only vLLM's and OpenAI's protocols, so it cannot
reach the FastAPI recipe's `/generate`, `/embed` and `/vqa`. This client takes
the same flags for the same ideas -- `--num-prompts`, `--num-warmups`,
`--max-concurrency`, `--seed`, `--metric-percentiles`, `--save-result` -- and
writes its result under the key names vLLM 0.29.0's saved JSON uses
(`completed`, `request_throughput`, `p99_e2el_ms`, ...), so a report reads
both engines' files the same way.

`--ignore-eos` sends `ignore_eos: true`, which the recipe answers by
generating exactly `--output-len` tokens, so end-to-end latency covers the
same work on every request, as it does under `vllm bench serve --ignore-eos`.

Only end-to-end latency is measured. The recipe does not stream, so the first
token is never observable on its own: TTFT, TPOT and ITL are absent from the
result, as they are from a vLLM result that did not select them, and are
listed under `unsupported_metrics`.

Every prompt is random and unique, and the warmup prompts are not reused in
the measured run. The recipe has no prefix cache, but the warmups still matter:
they absorb XLA compilation of the request's shapes, which would otherwise be
timed as the first requests' latency.

With `--tokenizer`, prompts are `--input-len` tokens of random vocabulary,
and the completions are counted in tokens. Without one, prompts are `--input-len` random words and every token
count is null.

Exit codes: 0 every measured request succeeded; 1 some failed (see
`errors` with `--save-detailed`); 2 the server failed a warmup request or no
measured request succeeded, so there is nothing to report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

TASK_ENDPOINTS = {"generate": "/generate", "embed": "/embed", "vqa": "/vqa"}
# Metrics vLLM can report that need streaming, which the recipe does not do.
STREAMING_METRICS = ("ttft", "tpot", "itl")
# Fields vLLM drops from the saved result unless --save-detailed is given.
DETAILED_FIELDS = ("input_lens", "output_lens", "start_times",
                   "generated_texts", "errors")
# Plain words for prompts when there is no tokenizer to draw from.
WORDS = ("time year people way day man thing woman life child world school "
         "state family student group country problem hand part place case "
         "week company system program question work government number night "
         "point home water room mother area money story fact month lot right "
         "study book eye job word business issue side kind head house service "
         "friend father power hour game line end member law car city name "
         "team minute idea kid body information back parent face others level "
         "office door health person art war history party result change "
         "morning reason research girl guy moment air teacher force education"
         ).split()


def percentile(values: list[float], p: float) -> float:
    """numpy.percentile's default (linear) interpolation, without numpy."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


GCS_SCHEME = "gs://"


def local_tokenizer(name: str) -> str:
    """`name`, or for a gs:// checkpoint folder a local copy of its tokenizer
    and config files (no weights), made by fetch_weights.py.

    In a job, fetch_weights.py is shipped beside this script; in the package
    it is in the tpu-model-onboarding scripts.
    """
    if not name.startswith(GCS_SCHEME):
        return name
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here,
                   os.path.join(os.path.dirname(here),
                                "tpu-model-onboarding")):
        if (os.path.isfile(os.path.join(folder, "fetch_weights.py"))
                and folder not in sys.path):
            sys.path.insert(0, folder)
    try:
        import fetch_weights  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            f"--tokenizer {name} is a gs:// folder: ship fetch_weights.py "
            "beside this script (--input <fetch_weights.py>:fetch_weights.py)"
        ) from e
    # Its progress lines go to stderr: stdout is the client's own output.
    with contextlib.redirect_stdout(sys.stderr):
        return fetch_weights.fetch_gcs(name, small_only=True)


def load_tokenizer(name: str | None):
    if not name:
        return None
    from transformers import \
        AutoTokenizer  # pylint: disable=import-outside-toplevel
    return AutoTokenizer.from_pretrained(local_tokenizer(name))


def sampling_vocab(tokenizer) -> list[int]:
    """Token ids a random prompt may use: all but the special tokens."""
    special = set(tokenizer.all_special_ids)
    return [i for i in range(len(tokenizer)) if i not in special]


def random_prompt(rng: random.Random, length: int, tokenizer,
                  vocab: list[int] | None) -> tuple[str, int]:
    """Return a random prompt of `length` tokens (or words) and its length."""
    if tokenizer is None:
        return " ".join(rng.choice(WORDS) for _ in range(length)), length
    text = tokenizer.decode([rng.choice(vocab) for _ in range(length)])
    # Decoding and re-encoding merges or splits some tokens; trim to length.
    ids = tokenizer.encode(text, add_special_tokens=False)[:length]
    return tokenizer.decode(ids), len(ids)


def build_payload(task: str, prompt: str, output_len: int,
                  image_url: str | None, ignore_eos: bool) -> dict:
    if task == "embed":
        return {"text": prompt}
    payload = {"prompt": prompt, "max_tokens": output_len, "temperature": 0}
    if ignore_eos:
        payload["ignore_eos"] = True
    if task == "vqa":
        payload["image_url"] = image_url
    return payload


def completion_text(task: str, body: dict) -> str:
    if task == "embed":
        return ""
    for key in ("result", "generated_text", "text", "answer", "output"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def send(url: str, payload: dict, timeout: float) -> dict:
    """POST one request; return its timing, response and error."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode() or "{}")
        return {
            "start": start,
            "latency": time.perf_counter() - start,
            "body": body,
            "error": ""
        }
    except (urllib.error.URLError, OSError, ValueError) as e:
        detail = e.read().decode(errors="replace")[:500] if isinstance(
            e, urllib.error.HTTPError) else ""
        return {
            "start": start,
            "latency": time.perf_counter() - start,
            "body": None,
            "error": f"{e} {detail}".strip()
        }


def run_requests(url: str, payloads: list[dict], concurrency: int,
                 timeout: float) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda p: send(url, p, timeout), payloads))


def summarize(outputs: list[dict], input_lens: list[int | None],
              duration: float, task: str, tokenizer, percentiles: list[float],
              percentile_metrics: list[str]) -> dict:
    ok = [i for i, o in enumerate(outputs) if not o["error"]]
    texts = [
        completion_text(task, o["body"]) if not o["error"] else ""
        for o in outputs
    ]
    if task == "embed":
        output_lens = [0] * len(outputs)
    elif tokenizer is None:
        output_lens = [None] * len(outputs)
    else:
        output_lens = [
            len(tokenizer.encode(t, add_special_tokens=False))
            if i in ok else 0 for i, t in enumerate(texts)
        ]
    counted = tokenizer is not None or task == "embed"
    total_input = sum(input_lens[i] for i in ok) if tokenizer else None
    total_output = sum(output_lens[i] for i in ok) if counted else None
    result = {
        "duration":
        duration,
        "completed":
        len(ok),
        "failed":
        len(outputs) - len(ok),
        "total_input_tokens":
        total_input,
        "total_output_tokens":
        total_output,
        "request_throughput":
        len(ok) / duration,
        "request_goodput":
        None,
        "output_throughput":
        total_output / duration if total_output is not None else None,
        "total_token_throughput": (total_input + total_output) / duration
        if total_input is not None and total_output is not None else None,
        "input_lens":
        input_lens,
        "output_lens":
        output_lens,
        "start_times": [o["start"] for o in outputs],
        "generated_texts":
        texts,
        "errors": [o["error"] for o in outputs],
    }
    if "e2el" in percentile_metrics:
        # vLLM reports zeros, not nulls, when nothing completed; so do we.
        e2els = [outputs[i]["latency"] * 1000 for i in ok] or [0.0]
        result["mean_e2el_ms"] = statistics.fmean(e2els)
        result["median_e2el_ms"] = statistics.median(e2els)
        result["std_e2el_ms"] = statistics.pstdev(e2els)
        for p in percentiles:
            word = str(int(p)) if int(p) == p else str(p)
            result[f"p{word}_e2el_ms"] = percentile(e2els, p)
    return result


def print_summary(result: dict, percentiles: list[float]) -> None:
    row = "{:<40} {:<10}".format
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    for label, key in (("Successful requests:",
                        "completed"), ("Failed requests:", "failed"),
                       ("Benchmark duration (s):", "duration"),
                       ("Total input tokens:", "total_input_tokens"),
                       ("Total generated tokens:",
                        "total_output_tokens"), ("Request throughput (req/s):",
                                                 "request_throughput"),
                       ("Output token throughput (tok/s):",
                        "output_throughput"),
                       ("Total token throughput (tok/s):",
                        "total_token_throughput")):
        value = result.get(key)
        print(
            row(
                label, f"{value:.2f}" if isinstance(value, float) else
                "n/a" if value is None else value))
    if "mean_e2el_ms" in result:
        print("{s:{c}^{n}}".format(s="End-to-end Latency", n=50, c="-"))
        print(row("Mean E2EL (ms):", f"{result['mean_e2el_ms']:.2f}"))
        print(row("Median E2EL (ms):", f"{result['median_e2el_ms']:.2f}"))
        for p in percentiles:
            word = str(int(p)) if int(p) == p else str(p)
            print(
                row(f"P{word} E2EL (ms):",
                    f"{result[f'p{word}_e2el_ms']:.2f}"))
    print("=" * 50)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url",
                   required=True,
                   help="Server URL, e.g. http://localhost:8000")
    p.add_argument("--task",
                   choices=sorted(TASK_ENDPOINTS),
                   default="generate",
                   help="Which recipe the server was built from")
    p.add_argument("--endpoint",
                   default=None,
                   help="API path (default: the task's recipe path)")
    p.add_argument("--image-url",
                   default=None,
                   help="Image sent with every --task vqa request")
    p.add_argument("--model",
                   default=None,
                   help="Model id recorded in the result")
    p.add_argument("--tokenizer",
                   default=None,
                   help="Tokenizer for token-exact prompts and token counts: "
                   "a Hugging Face id, a local directory, or a gs:// "
                   "checkpoint folder (its tokenizer files are copied here)")
    p.add_argument("--input-len",
                   type=int,
                   default=1024,
                   help="Prompt length in tokens (words without --tokenizer)")
    p.add_argument("--output-len",
                   type=int,
                   default=128,
                   help="max_tokens sent with each request")
    p.add_argument("--ignore-eos",
                   action="store_true",
                   help="Ask the server to generate exactly --output-len "
                   "tokens, past end-of-sequence")
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--num-warmups",
                   type=int,
                   default=0,
                   help="Discarded warmup requests, on their own prompts")
    p.add_argument("--max-concurrency",
                   type=int,
                   default=1,
                   help="Requests in flight at once")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-timeout",
                   type=float,
                   default=1800,
                   help="Seconds before one request is failed")
    p.add_argument("--percentile-metrics",
                   default="e2el",
                   help='Comma-separated; only "e2el" is measurable here')
    p.add_argument("--metric-percentiles",
                   default="99",
                   help='Comma-separated percentiles, e.g. "50,90,99"')
    p.add_argument("--label", default="custom_fastapi")
    p.add_argument("--save-result", action="store_true")
    p.add_argument("--save-detailed", action="store_true")
    p.add_argument("--result-dir", default=".")
    p.add_argument("--result-filename", default=None)
    p.add_argument("--metadata",
                   nargs="*",
                   default=[],
                   help="KEY=VALUE pairs recorded in the result")
    args = p.parse_args(argv)
    args.percentile_metrics = [
        m.strip() for m in args.percentile_metrics.split(",") if m.strip()
    ]
    unsupported = [m for m in args.percentile_metrics if m != "e2el"]
    if unsupported:
        p.error(f"{','.join(unsupported)} need a streaming endpoint, which "
                "the FastAPI recipe does not have; only e2el can be measured")
    args.metric_percentiles = [
        float(x) for x in args.metric_percentiles.split(",") if x.strip()
    ]
    if any(not 0 < x < 100 for x in args.metric_percentiles):
        p.error("--metric-percentiles must each be in (0, 100)")
    if args.task == "vqa" and not args.image_url:
        p.error("--task vqa needs --image-url")
    if args.max_concurrency < 1 or args.num_prompts < 1:
        p.error("--max-concurrency and --num-prompts must be at least 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    url = args.base_url.rstrip("/") + (args.endpoint
                                       or TASK_ENDPOINTS[args.task])
    tokenizer = load_tokenizer(args.tokenizer)
    vocab = sampling_vocab(tokenizer) if tokenizer else None
    rng = random.Random(args.seed)
    # Warmups first, from the same stream, so no measured prompt repeats one.
    prompts = [
        random_prompt(rng, args.input_len, tokenizer, vocab)
        for _ in range(args.num_warmups + args.num_prompts)
    ]
    payloads = [
        build_payload(args.task, text, args.output_len, args.image_url,
                      args.ignore_eos) for text, _ in prompts
    ]

    if args.num_warmups:
        print(f"Warming up with {args.num_warmups} requests...")
        warmups = run_requests(url, payloads[:args.num_warmups],
                               args.max_concurrency, args.request_timeout)
        failed = [w["error"] for w in warmups if w["error"]]
        if failed:
            print(f"warmup request failed: {failed[0]}", file=sys.stderr)
            return 2

    print(f"Sending {args.num_prompts} requests, "
          f"max concurrency {args.max_concurrency}...")
    start = time.perf_counter()
    outputs = run_requests(url, payloads[args.num_warmups:],
                           args.max_concurrency, args.request_timeout)
    duration = time.perf_counter() - start

    input_lens = [
        n if tokenizer else None for _, n in prompts[args.num_warmups:]
    ]
    result = summarize(outputs, input_lens, duration, args.task, tokenizer,
                       args.metric_percentiles, args.percentile_metrics)
    print_summary(result, args.metric_percentiles)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json = {
        "date": now,
        "endpoint_type": "custom_fastapi",
        "backend": "custom_fastapi",
        "label": args.label,
        "model_id": args.model,
        "tokenizer_id": args.tokenizer,
        "num_prompts": args.num_prompts,
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": args.max_concurrency,
        "unsupported_metrics": list(STREAMING_METRICS),
    }
    for pair in args.metadata:
        key, _, value = pair.partition("=")
        result_json[key.strip()] = value.strip()
    result_json.update(result)
    if not args.save_detailed:
        for field in DETAILED_FIELDS:
            result_json.pop(field, None)

    if args.save_result:
        name = (args.result_filename
                or f"{args.label}-infqps-concurrency{args.max_concurrency}-"
                f"{(args.model or 'model').split('/')[-1]}-{now}.json")
        path = pathlib.Path(args.result_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result_json))
        print(f"result written to {path}")

    if result["completed"] == 0:
        first = next((o["error"] for o in outputs if o["error"]), "")
        print(f"no request succeeded: {first}", file=sys.stderr)
        return 2
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
