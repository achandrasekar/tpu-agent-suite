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
"""Workload -> the step batches `forward_run.py` runs.

The workload is the job's STEP MIX (`core/mix.py`, the default): every step shape the
engine ran -- decode steps of 1-2 / 3-4 / 5-8 requests at three context levels, prefill
chunks behind three prefix levels -- each with its measured step count and share of busy
time. `mix_steps` turns each cell into the batch to run:

  decode cell   B requests x rows tokens at the cell's context, rows = 1 + spec tokens
  mixed cell    the same decode rows plus one prefill chunk behind the cell's prefix,
                filling the token budget (running decodes are scheduled first, vLLM v1),
                so the chunk gets `budget - B * rows` tokens; its decode rows alone are
                timed too, so the chunk's own cost is the difference

A result rolls up as `sum over cells of steps x step time` -- the job's busy time, at the
twin's shapes. Everything is in REAL tokens; `forward_run.py` scales contexts by the twin's
context_scale. Buckets follow the served runner's rule (`token_paddings`).

A hand-typed *point* (`parse_point`, `cached / input / output @ C`) is still accepted: one
request shape with C running. It is an explicit what-if, not a model of the job.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from .cells import _q, with_hit
from .deploy import Deploy


@dataclass(frozen=True)
class Point:
    name: str
    cached: int  # tokens of the prompt already seen (trace-level reusable prefix)
    input: int  # new prompt tokens
    output: int  # generated tokens
    concurrency: int  # requests running on the engine together with this one


_NUM = re.compile(r"^\s*([\d.]+)\s*([kKmM]?)\s*$")


def _tokens(s: str) -> int:
    m = _NUM.match(s)
    if not m:
        raise ValueError(f"not a token count: {s!r}")
    return int(
        round(
            float(m.group(1)) * {
                "": 1,
                "k": 1e3,
                "m": 1e6
            }[m.group(2).lower()]))


def parse_point(s: str) -> Point:
    """`NAME=cached/input/output@concurrency`, counts like 299K, 1.4K, 200."""
    m = re.match(r"^\s*([^=]+)=([^/]+)/([^/]+)/([^@]+)@\s*(\d+)\s*$", s)
    if not m:
        raise ValueError(
            f"--point expects NAME=cached/input/output@C (e.g. P90=299K/6K/1K@8), "
            f"got {s!r}")
    return Point(
        m.group(1).strip(), *(_tokens(m.group(i)) for i in (2, 3, 4)),
        int(m.group(5)))


def job_measured(snap: dict) -> dict[str, Any]:
    """What the source job measured (whole model, served slice), to set beside twin ms:
    output throughput, TTFT and ITL percentiles, and the mixed-step time from the log."""
    prof = [
        r for r in snap["requests"]
        if r.get("phase") == "profiling" and not r.get("error")
    ]
    srv, log = snap.get("server") or {}, snap.get("log") or {}
    ttft = [r["ttft_ms"] for r in prof if r.get("ttft_ms") is not None]
    dur = srv.get("profiling_duration_s")
    itl = srv.get("itl_s") or {}
    return {
        "job_id":
        snap.get("job_id"),
        "output_tok_per_s":
        round(srv["generation_tokens"] /
              dur, 1) if srv.get("generation_tokens") and dur else None,
        "ttft_ms": {
            k: round(_q(ttft, p), 1)
            for k, p in (("p50", 0.5), ("p90", 0.9))
        } if ttft else None,
        "itl_ms": {
            k: round(itl[k] * 1e3, 1)
            for k in ("p50", "p90") if itl.get(k) is not None
        },
        "t_mixed_s":
        log.get("prefill_step_s"),
        "note":
        "measured, served model on the served slice; TTFT includes queueing",
    }


def token_paddings(mnbt: int,
                   env: dict | None = None,
                   compile_sizes: list[int] | None = None) -> list[int]:
    """The compiled token buckets vllm-torchtpu pads every step to (platforms/tpu_platform.py
    `_get_token_paddings`): explicit `compile_sizes` if the job passed them, else sizes doubling
    from 16 with the last bucket = mnbt itself, merged with TPU_TOKEN_BUCKET_EXTRA (<= mnbt).
    The largest bucket is always the step budget, mnbt."""
    if compile_sizes:
        return sorted(set(int(x) for x in compile_sizes))
    out, n = [], 16
    while n < mnbt:
        out.append(n)
        n *= 2
    out.append(mnbt)
    extra = [
        int(v)
        for v in str((env or {}).get("TPU_TOKEN_BUCKET_EXTRA", "")).split(",")
        if v.strip()
    ]
    return sorted(set(out) | {e for e in extra if 0 < e <= mnbt})


def bucket(tokens: int, paddings: list[int] | int) -> int:
    """The bucket a step of `tokens` tokens runs in: the first padding >= tokens (vllm-torchtpu's
    sequence layout planner, `_first_ge`). An int means `token_paddings(int)`."""
    if isinstance(paddings, int):
        paddings = token_paddings(paddings)
    for b in paddings:
        if b >= tokens:
            return b
    raise ValueError(
        f"{tokens} tokens exceed the largest bucket {paddings[-1]} (the step budget)"
    )


def accept_per_step(server: dict, d: Deploy) -> tuple[float, str]:
    """Tokens a decoding request advances per step: 1 + accepted / drafts (measured), else
    1 + spec tokens (the ceiling) when the job has no spec counters."""
    dr, acc = server.get("spec_drafts"), server.get("spec_accepted")
    if dr and acc is not None:
        return round(1 + acc / dr,
                     3), "measured: 1 + spec_accepted / spec_drafts"
    return float(1 + (
        d.spec_tokens if d.spec_method else 0)), "ceiling: 1 + spec tokens"


def point_steps(pt: Point,
                d: Deploy,
                acc: float,
                budget: int | None = None,
                last_chunks: bool = False,
                paddings: list[int] | None = None,
                cache_hit=None) -> dict[str, Any]:
    """The mixed and decode batches for one point (real tokens)."""
    budget = budget or d.mnbt
    paddings = paddings or token_paddings(budget)
    rows = 1 + (d.spec_tokens if d.spec_method else 0)
    C = pt.concurrency
    if C < 1 or C > d.mns:
        raise ValueError(
            f"{pt.name}: concurrency {C} outside 1..max-num-seqs {d.mns}")
    isl = pt.cached + pt.input
    hit = with_hit({"reusable": pt.cached, "isl": isl}, d, cache_hit)["hit"]
    computed = isl - hit
    neighbours = C - 1
    chunk = budget - neighbours * rows
    if chunk <= 0:
        raise ValueError(
            f"{pt.name}: {neighbours} decodes x {rows} rows fill the budget {budget}"
        )
    full, last = divmod(computed, chunk)
    # a neighbour is a request like this one, mid-generation
    mid = isl + pt.output // 2
    dec_rows = [{"q": rows, "seq": mid + rows} for _ in range(neighbours)]
    chunks = [{"q": chunk, "prefix": hit + j * chunk} for j in range(full)]
    mixed = [
        dec_rows + [{
            "q": c["q"],
            "seq": c["prefix"] + c["q"]
        }] for c in chunks
    ]
    decode = [{"q": rows, "seq": mid + rows} for _ in range(C)]
    out = {
        "point": asdict(pt),
        "hit": hit,
        "computed": computed,
        "rows_per_decode": rows,
        "chunk_tokens": chunk,
        "chunks_full": full,
        "last_chunk_tokens": last,
        "mixed": {
            "tokens": budget,
            "bucket": bucket(budget, paddings),
            # the last full chunk (deepest prefix) is the one profiled; all full chunks are
            # the same program with a different prefix, so each is timed
            "profile_index": full - 1 if full else None,
            "batches": mixed,
        },
        "decode": {
            "tokens": C * rows,
            "bucket": bucket(C * rows, paddings),
            "batch": decode,
            "steps_per_request":
            math.ceil(pt.output / acc) if pt.output else 0,
        },
    }
    if last:
        t = last + neighbours * rows
        out["last_chunk"] = {
            "tokens": t,
            "bucket": bucket(t, paddings),
            "run": last_chunks,
            "batch": dec_rows + [{
                "q": last,
                "seq": hit + full * chunk + last
            }]
        }
        if not last_chunks:
            out["last_chunk"][
                "why"] = "its own (smaller) bucket, not compiled; --last-chunks runs it"
    if not full:
        out["mixed"]["note"] = (
            f"computed {computed} < one chunk {chunk}: the only prefill step "
            "is the partial last chunk")
    out["step_mix"] = step_mix(C, full + (1 if last else 0), pt.output, acc)
    return out


def step_mix(C: int, chunks: int, output: int, acc: float) -> dict[str, Any]:
    """Steady state, all C requests at this point: steps per C turns (one turn each).

    A mixed step advances C-1 decoding requests by one step; a decode-only step advances C.
    Decode slots needed = C * output / acc; mixed steps supply C * chunks * (C-1) of them."""
    mixed = C * chunks
    need = C * output / acc
    decode = max(0.0, (need - mixed * (C - 1)) / C)
    return {
        "mixed_steps":
        mixed,
        "decode_steps":
        round(decode, 1),
        "formula":
        "T(C turns) = mixed_steps * t_mixed + decode_steps * t_decode; "
        "output tok/s = C * output / T"
    }


#: `--top K` bounds: fewer than 2 shapes is not a workload, more than 5 is not analyzable by hand
TOP_K_RANGE = (2, 5)
#: a kind (decode / mixed) carrying at least this share of busy time gets a picked shape
TOP_KIND_FLOOR = 0.10
#: forward steps per shape in a `--top` job (timed + profiled), so the job stays under ~5 min warm
TOP_MAX_STEPS = 50


def steps_per_shape(entry: dict,
                    iters: int,
                    repeats: int,
                    profile_iters: int,
                    profile: bool = True) -> int:
    """Forward steps one plan entry runs in forward_run: each of its batches (mixed chunks,
    last chunk if run, decode) `iters` x `repeats` timed, plus `profile_iters` profiled for
    each batch in the profile window (the mixed chunk and the decode rows)."""
    timed = len(entry["mixed"]["batches"]) + bool(entry.get("decode")) \
        + bool(entry.get("last_chunk", {}).get("run"))
    windowed = (entry["mixed"]["profile_index"] is not None) + bool(entry.get("decode")) \
        if profile and entry.get("profile", True) else 0
    return timed * iters * repeats + windowed * profile_iters


def top_cells(cells: list[dict],
              k: int,
              kind_floor: float = TOP_KIND_FLOOR) -> list[dict]:
    """The `k` cells carrying the most busy time, in mix order. A kind whose cells together
    carry >= `kind_floor` of busy time first gets its heaviest cell, so prefill is never left
    out just because decode cells are more numerous."""
    lo, hi = TOP_K_RANGE
    if not lo <= k <= hi:
        raise ValueError(f"--top {k}: pick {lo}..{hi} shapes")
    by_share = sorted(cells, key=lambda c: -c["share"])
    kinds: dict[str, float] = {}
    for c in cells:
        kinds[c["kind"]] = kinds.get(c["kind"], 0.0) + c["share"]
    picked = [
        next(c for c in by_share if c["kind"] == kd)
        for kd, s in sorted(kinds.items(), key=lambda x: -x[1])
        if s >= kind_floor
    ][:k]
    for c in by_share:
        if len(picked) >= k:
            break
        if c not in picked:
            picked.append(c)
    names = {c["name"] for c in picked}
    return [c for c in cells if c["name"] in names]


def mix_steps(mix: dict,
              d: Deploy,
              paddings: list[int],
              min_share: float = 0.0,
              profile_per_class: int = 1,
              top: int | None = None) -> list[dict[str, Any]]:
    """One plan entry per step-mix cell (real tokens). `min_share` drops cells under that
    share of busy time (their weight is reported as dropped). The `profile_per_class`
    heaviest cells of each class are profiled; every cell is timed. `top` keeps only the
    `top_cells` and profiles every one of them."""
    rows = mix["rows_per_decode"]
    kept = [c for c in mix["cells"] if c["share"] >= min_share]
    heavy = set()
    if top:
        kept = top_cells(kept, top)
        heavy = {c["name"] for c in kept}
    by_cls: dict[str, list] = {}
    for c in kept:
        by_cls.setdefault(c["class"], []).append(c)
    for cs in by_cls.values():
        heavy |= {
            c["name"]
            for c in sorted(cs, key=lambda c: -c["share"])[:profile_per_class]
        }
    out = []
    for c in kept:
        dec_rows = [{
            "q": rows,
            "seq": c["context"] + rows
        } for _ in range(c["batch"])]
        e: dict[str, Any] = {
            "point": {
                "name": c["name"],
                "kind": "cell"
            },
            "cell": dict(c),
            "rows_per_decode": rows,
            "profile": c["name"] in heavy
        }
        if c["kind"] == "mixed":
            chunk = d.mnbt - c["batch"] * rows
            if chunk <= 0:
                raise ValueError(
                    f"{c['name']}: {c['batch']} decodes x {rows} rows fill the "
                    f"budget {d.mnbt}")
            e["mixed"] = {
                "tokens": d.mnbt,
                "bucket": bucket(d.mnbt, paddings),
                "profile_index": 0,
                "batches":
                [dec_rows + [{
                    "q": chunk,
                    "seq": c["prefix"] + chunk
                }]]
            }
        else:
            e["mixed"] = {
                "tokens": 0,
                "bucket": None,
                "profile_index": None,
                "batches": []
            }
        if dec_rows:
            e["decode"] = {
                "tokens": len(dec_rows) * rows,
                "bucket": bucket(len(dec_rows) * rows, paddings),
                "batch": dec_rows
            }
        out.append(e)
    return out


def plan(points: list[Point],
         d: Deploy,
         acc: float,
         acc_src: str,
         budget: int | None = None,
         last_chunks: bool = False,
         job: dict | None = None,
         env: dict | None = None,
         compile_sizes: list[int] | None = None,
         cache_hit=None,
         mix: dict | None = None,
         min_share: float = 0.0,
         top: int | None = None) -> dict[str, Any]:
    paddings = token_paddings(budget or d.mnbt, env, compile_sizes)
    steps = (mix_steps(mix, d, paddings, min_share, top=top) if mix else []) + \
        [point_steps(p, d, acc, budget, last_chunks, paddings, cache_hit) for p in points]
    buckets = sorted(
        {s["mixed"]["bucket"]
         for s in steps if s["mixed"]["batches"]}
        | {s["decode"]["bucket"]
           for s in steps if s.get("decode")}
        | {
            s["last_chunk"]["bucket"]
            for s in steps if s.get("last_chunk", {}).get("run")
        })
    return {
        "_what":
        "workload-microbenchmark steps: per point, the mixed and decode batches "
        "(REAL tokens; forward_run scales contexts to the twin)",
        "deploy": {
            "mnbt": budget or d.mnbt,
            "block_x_dcp": d.cache_block_tokens,
            "eagle_drop": d.eagle_drop,
            "spec_tokens": d.spec_tokens,
            "max_num_seqs": d.mns
        },
        "accept_per_step":
        acc,
        "accept_source":
        acc_src,
        # the served runner's full bucket list (largest = mnbt) and the subset this plan compiles
        "serving_token_paddings":
        paddings,
        "compile_buckets":
        buckets,
        "job_measured":
        job,
        "points":
        steps,
        # `--top K`: the fixed shapes, run back to back in ONE job, every one profiled; the
        # cost model counts only these (`share_kept` is their coverage of busy time)
        "top": {
            "k":
            top,
            "max_steps_per_shape":
            TOP_MAX_STEPS,
            "rule":
            f"heaviest cells by share of busy time; a kind with >= "
            f"{TOP_KIND_FLOOR:.0%} of busy gets its heaviest cell first"
        } if top else None,
        "mix":
        ({
            k: v
            for k, v in mix.items() if k != "cells"
        }
         | {
             "cells_kept":
             sum(1 for s in steps if s.get("cell")),
             "cells_total":
             len(mix["cells"]),
             "share_kept":
             round(sum(s["cell"]["share"] for s in steps if s.get("cell")), 4)
         }) if mix else None
    }


def table(pl: dict) -> str:
    L = [
        f"accept/step {pl['accept_per_step']} ({pl['accept_source']}); buckets "
        f"{pl['compile_buckets']}"
    ]
    mx = pl.get("mix")
    cells = [s for s in pl["points"] if s.get("cell")]
    if cells:
        L += [
            f"step mix: measured busy {mx['busy_s']} s + idle {mx['idle_s']} s; share of wall "
            f"{mx['measured_share_of_wall']}; step ms {mx['measured_step_ms']}; "
            f"{mx['cells_kept']}/{mx['cells_total']} cells kept = {100 * mx['share_kept']:.1f}% "
            f"of busy"
        ]
        if pl.get("top"):
            L.append(
                f"top {pl['top']['k']}: {len(cells)} fixed shapes, one job, back to back, "
                f"every shape profiled; they cover {100 * mx['share_kept']:.1f}% of busy and "
                f"the cost model counts only them ({pl['top']['rule']})")
        L += [
            "cell | batch x rows @ context | chunk @ prefix | tokens->bucket | steps | "
            "share of busy | measured step ms | profiled"
        ]
        for s in cells:
            c = s["cell"]
            tb = (f"{s['mixed']['tokens']}->{s['mixed']['bucket']}"
                  if c["kind"] == "mixed" else
                  f"{s['decode']['tokens']}->{s['decode']['bucket']}")
            L.append(
                f"{c['name']:18} | {c['batch']} x {s['rows_per_decode']} @ {c['context']:,} | "
                + (f"{s['mixed']['batches'][0][-1]['q']} @ {c['prefix']:,}"
                   if c["kind"] == "mixed" else "-") +
                f" | {tb} | {c['steps']:,.0f} | {100 * c['share']:.1f}% | "
                f"{c['measured_step_ms']} | {'yes' if s['profile'] else ''}")
    pts = [s for s in pl["points"] if not s.get("cell")]
    if pts:
        L.append(
            "point | cached/input/output @C | hit / computed | chunk x full (+last) | "
            "mixed tok->bucket | decode tok->bucket | steps/request | mixed:decode steps "
            "per C turns")
    for s in pts:
        p, m, dc, mx2 = s["point"], s["mixed"], s["decode"], s["step_mix"]
        L.append(
            f"{p['name']:5} | {p['cached']:,}/{p['input']:,}/{p['output']:,} @{p['concurrency']} "
            f"| {s['hit']:,} / {s['computed']:,} | {s['chunk_tokens']} x {s['chunks_full']}"
            f"{' (+' + str(s['last_chunk_tokens']) + ')' if s['last_chunk_tokens'] else ''} "
            f"| {m['tokens']}->{m['bucket']} | {dc['tokens']}->{dc['bucket']} "
            f"| {dc['steps_per_request']} | {mx2['mixed_steps']}:{mx2['decode_steps']}"
        )
    j = pl.get("job_measured")
    if j:
        L.append(
            f"job {j['job_id']} (measured, served model): {j['output_tok_per_s']} output tok/s, "
            f"TTFT {j['ttft_ms']} ms, ITL {j['itl_ms']} ms, mixed step {j['t_mixed_s']} s"
        )
    return "\n".join(L)
