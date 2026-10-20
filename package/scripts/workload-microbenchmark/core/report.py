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
"""Tables and text laid out from data it is given: selftest bars and checks, summaries, the
request tables, the worked example. No model facts, no thresholds of its own: gates come from
cells.py, everything else from the config / snapshot / plan passed in."""

from __future__ import annotations

import json
from typing import Any


def _get(d: dict, path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(k)]
        elif isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def bars(cfg: dict, snap: dict, model_dir: str) -> dict:
    """Expectations for the reference and two override variants: every value the derivation
    produces by arithmetic (feasibility, layer kinds, KV bytes, each kernel choice, each numeric
    per-rank row, the chunk count) exactly; the values resting on measured data within the
    gates in cells.py."""
    from .cells import CLOSURE_GATE, REPLAY_GATE
    exact = [
        "feasibility.feasible", "facts.layer_kinds",
        "facts.kv_bytes_per_token", "cells.counts.N_A_prefill_chunks"
    ]
    exact += [f"kernels.{k}" for k in (cfg.get("kernels") or {}) if k != "env"]
    exact += [
        f"per_rank.{k}" for k, v in (cfg.get("per_rank") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    expect: dict[str, Any] = {
        p: {
            "eq": _get(cfg, p)
        }
        for p in exact if _get(cfg, p) is not None
    }
    expect["cells.recompute.model_over_server"] = {
        "between": [1 - REPLAY_GATE, 1 + REPLAY_GATE]
    }
    if cfg.get("closure_G0"):
        expect["closure_G0.ratio"] = {
            "between": [1 - CLOSURE_GATE, 1 + CLOSURE_GATE]
        }
    cases = [{"overrides": {}, "expect": expect}]
    # an override that re-derives, and one the feasibility gate must refuse
    d = cfg["deploy"]
    if d["dcp"] > 2:
        cases.append({
            "overrides": {
                "dcp": str(d["dcp"] // 2)
            },
            "expect": {
                "feasibility.feasible": {
                    "eq": True
                },
                "per_rank.cache_block_tokens": {
                    "eq": d["block_size"] * d["dcp"] // 2
                },
                "decidable_at": {
                    "startswith": "L1 per-layer"
                }
            }
        })
    if d["dcp"] > 1:
        mn = cfg["feasibility"]["kv"].get("min_dcp_for_recorded_peak")
        exp = {"feasibility.feasible": {"eq": mn is not None and mn <= 1}}
        if mn:
            exp["feasibility.kv.min_dcp_for_recorded_peak"] = {"eq": mn}
        cases.append({"overrides": {"dcp": "1"}, "expect": exp})
    return {
        "_what":
        "selftest bars: `cli.py selftest` re-derives from the snapshot and checks",
        "snapshot": cfg["_header"]["snapshot"],
        "model_dir": model_dir,
        "cases": cases
    }


def twin_bars(tw: dict) -> dict:
    """Projection expectations: exact overrides/args, and every per-rank row the rule says must
    match does match."""
    return {
        "twin.hf_overrides": {
            "eq": tw["twin"]["hf_overrides"]
        },
        "twin.server_args": {
            "eq": tw["twin"]["server_args"]
        },
        "layers": {
            "eq": tw["layers"]
        },
        "context_scale": {
            "eq": tw["context_scale"]
        },
        "deviations": {
            "whats": sorted(x["what"] for x in tw["deviations"])
        },
        "projection": {
            "statuses": {
                r["per_rank"]: r["status"]
                for r in tw["projection"]
            }
        },
        # the model's own weight estimate, re-derived: catches a change to the byte count
        "memory.weights_real_gib": {
            "approx": tw["memory"]["weights_real_gib"],
            "rel": 0.02
        },
    }


def check(cfg: dict, expect: dict) -> list[str]:
    errs = []
    for path, rule in expect.items():
        v = _get(cfg, path)
        if "eq" in rule and v != rule["eq"]:
            errs.append(f"{path}: {json.dumps(v)} != {json.dumps(rule['eq'])}")
        if "between" in rule and (
                v is None
                or not rule["between"][0] <= v <= rule["between"][1]):
            errs.append(f"{path}: {v} not in {rule['between']}")
        if "approx" in rule and (v is None or abs(v - rule["approx"])
                                 > rule["rel"] * abs(rule["approx"])):
            errs.append(
                f"{path}: {v} not within {rule['rel']:.0%} of {rule['approx']}"
            )
        if "startswith" in rule and not str(v).startswith(rule["startswith"]):
            errs.append(
                f"{path}: {v!r} does not start with {rule['startswith']!r}")
        if "statuses" in rule:
            got = {r["per_rank"]: r["status"] for r in (v or [])}
            if got != rule["statuses"]:
                errs.append(f"{path}: {got} != {rule['statuses']}")
        if "whats" in rule and sorted(x["what"]
                                      for x in (v or [])) != rule["whats"]:
            errs.append(
                f"{path}: {[x['what'] for x in (v or [])]} != {rule['whats']}")
    return errs


def summary(cfg: dict) -> str:
    d, f = cfg["deploy"], cfg["facts"]
    L = [
        f"{cfg['model_dir']}  job {cfg['job_id']}  overrides {cfg['overrides'] or 'none'}",
        f"deploy tp{d['tp']} dp{d['dp']} dcp{d['dcp']} ep={d['ep']} block {d['block_size']} "
        f"mnbt {d['mnbt']} mns {d['mns']} max_len {d['max_model_len']} "
        f"spec {d['spec_method']}x{d['spec_tokens']}",
        f"layer kinds {f['layer_kinds']}   decidable at: {cfg['decidable_at']}"
    ]
    fe = cfg["feasibility"]
    L.append(f"feasible: {fe['feasible']}  " +
             "; ".join(fe["problems"] + fe["notes"]))
    if not fe["feasible"]:
        kv = fe["kv"]
        L.append(
            f"  alternatives: min DCP {kv.get('min_dcp_for_recorded_peak')} for the "
            f"recorded peak; or max-model-len <= {kv.get('max_model_len_that_fits')}"
        )
        return "\n".join(L)
    pr = cfg["per_rank"]
    L.append("per rank: " +
             ", ".join(f"{k}={v}"
                       for k, v in pr.items() if not isinstance(v, str)))
    c = cfg["cells"]
    L.append(f"counts: {c['counts']}")
    L.append(f"recompute: {c['recompute']}")
    for p in c["prefill_cells"]:
        L.append(
            f"  {p['name']}: chunk {p['chunk_tokens']} @ ctx {p['cached_context']} "
            f"+ {p['decode_rows']['batch']} decode rows  w={p['weight']}")
    for q in c["decode_cells"]:
        L.append(
            f"  {q['name']}: {q['batch']} rows x {q['tokens_per_row']} @ {q['contexts']}"
            f"  w={q['weight']}")
    if cfg.get("closure_G0"):
        L.append(f"G0 closure: {cfg['closure_G0']}")
    return "\n".join(L)


def twin_summary(tw: dict) -> str:
    L = [
        f"twin: real {tw['real_deploy']['tp']}x{tw['real_deploy']['dcp']} (TPxDCP) -> "
        f"{tw['twin_world']} cores tp{tw['twin_deploy']['tp']} dcp{tw['twin_deploy']['dcp']}, "
        f"layers {tw['layers']}, context x{tw['context_scale']}"
    ]
    L.append("  per rank (real -> twin):")
    for r in tw["projection"]:
        L.append(
            f"    {r['per_rank']:32s} {str(r['real']):>14s} -> {str(r['twin']):<22s} {r['status']}"
        )
    m = tw["memory"]
    L.append(
        f"  memory/rank: weights {m['weights_real_gib']} -> {m['weights_twin_gib']} GiB, KV "
        f"{m['kv_real_gib']} -> {m['kv_twin_gib']} GiB; ballast to match {m['ballast_to_match_gib']} GiB"
    )
    if tw.get("mtp"):
        L.append(
            f"  MTP: {tw['mtp']['twin_engine']}; not reproduced: {tw['mtp']['not_reproduced']}"
        )
    for x in tw["deviations"]:
        L.append(
            f"  deviation: {x['what']}: {x['real']} -> {x['twin']}  [{x['covered_by']}]"
        )
    L.append(
        f"  hf_overrides (w/o quant list): "
        f"{json.dumps({k: v for k, v in tw['twin']['hf_overrides'].items() if k != 'quantization_config'})}"
    )
    L.append(f"  server args: {json.dumps(tw['twin']['server_args'])}")
    return "\n".join(L)


def _k(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e4 else \
        f"{n / 1e3:.1f}K" if n >= 1e3 else str(int(n))


def _pct(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


_BINS = [
    0, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000,
    512_000
]


def _dist(name: str, v: list[float]) -> list[str]:
    """Histogram of `v` (count, %, cumulative %) over _BINS, then its percentiles."""
    n = len(v)
    L = [
        f"{name}  (n={n})",
        f"  {'range':>17} {'requests':>9} {'%':>6} {'cum %':>6}"
    ]
    lo_hi = [(0, 0)] + [
        (a + 1 if a else 1, b) for a, b in zip(_BINS, _BINS[1:])
    ] + [(_BINS[-1] + 1, None)]
    cum = 0
    for lo, hi in lo_hi:
        c = sum(1 for x in v if x >= lo and (hi is None or x <= hi))
        cum += c
        if not c:
            continue
        rng = "0" if hi == 0 else f"{_k(lo)} - {_k(hi)}" if hi else f"> {_k(lo - 1)}"
        L.append(
            f"  {rng:>17} {c:>9} {100 * c / n:>5.1f}% {100 * cum / n:>5.1f}%")
    ps = [(p, _pct(v, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)]
    L.append("  percentiles: " + "  ".join(f"P{round(p * 100)} {_k(x)}"
                                           for p, x in ps) +
             f"  max {_k(max(v))}  mean {_k(sum(v) / n)}")
    return L


def request_report(job: str,
                   rows: list[dict],
                   d: dict,
                   server: dict,
                   conc: dict | None,
                   conv: str | None = None,
                   bars: bool = False,
                   width: int = 60,
                   gate: float | None = None,
                   pcts: tuple = (0.5, 0.9)) -> str:
    """Per-request tables for a job: how many requests had a cached prefix, the distribution
    of cached / new input / output / computed, concurrency, and the summary those give.
    `rows` = prefix_chain rows with hit/computed (warmup included; tables use measured only)."""
    prof = [r for r in rows if r["phase"] == "profiling"]
    per_row = 1 + (d["spec_tokens"] if d["spec_method"] else 0)
    n = len(prof)
    L = [
        f"job {job}: {n} measured requests ({len(rows) - n} warmup, not counted), "
        f"{len({r['conv'] for r in prof})} conversations; cache block "
        f"{d['block_size']}x{d['dcp']} = {d['block_size'] * d['dcp']} tokens" +
        (", the drafter's block drop applies" if d.get("eagle_drop") else ""),
        ""
    ]
    cached = sum(1 for r in prof if r["reusable"])
    hit = sum(1 for r in prof if r["hit"])
    gaps = sum(1 for r in prof if r["gap"])
    L += [
        "1. How many requests reuse an earlier turn",
        f"  {'':62}{'requests':>9} {'%':>6}"
    ]
    for lab, c in (
        ("measured requests",
         n), ("  with a cached prefix (an earlier turn fits in the prompt)",
              cached),
        ("  cold: no earlier turn fits (first turn, or history changed)",
         n - cached),
        ("  cache hit > 0 under the deployment's cache rule (replayed)", hit),
        ("  prompt shorter than previous turn's prompt+output (!)", gaps)):
        L.append(f"  {lab:<62}{c:>9} {100 * c / n:>5.1f}%")
    L += [
        "",
        "2. Distributions over measured requests (token counts per request)"
    ]
    for name, key in (("cached prefix (trace)",
                       "reusable"), ("new input = isl - cached",
                                     "new"), ("output (osl)", "osl"),
                      ("computed by the server = isl - hit (replayed)",
                       "computed")):
        L += _dist(name, [r[key] for r in prof]) + [""]
    comp, srv = sum(r["computed"]
                    for r in prof), server.get("prompt_tokens_computed")
    L.append("3. Check the replay against the server")
    verdict = (
        f"  ({'OK' if abs(comp / srv - 1) <= gate else 'FAIL'}, gate {gate:.0%})"
        if srv and gate is not None else "")
    L.append(
        f"  total computed: replay {comp:,} / server local_compute {srv:,.0f} = {comp / srv:.3f}"
        + verdict if srv else "  server local_compute counter missing")
    b = server.get("iteration_tokens_buckets")
    if b:
        floor = d["mns"] * per_row  # the largest possible decode-only step
        edges = [(float("inf") if k == "+Inf" else float(k), k) for k in b]
        mine = {k: 0 for _, k in edges}
        for r in prof:
            mine[next(k for e, k in edges if r["computed"] <= e)] += 1
        shown = [(e, k) for e, k in edges if e > floor]
        L += [
            "  computed per request by bin (server: iteration_tokens_total, approximate)",
            "    bin <=  " + " ".join(f"{k.replace('.0', ''):>7}"
                                      for _, k in shown),
            "    server  " + " ".join(f"{b[k]:>7}" for _, k in shown),
            "    replay  " + " ".join(f"{mine[k]:>7}" for _, k in shown)
        ]
    L += [
        "", "4. Concurrency (requests in flight)",
        "  client: from each measured request's start and end time (profile_export.jsonl),",
        "  the share of the window spent at each number of requests in flight"
    ]
    if conc:
        L.append(f"  {'in flight':>11} {'time %':>7} {'cum %':>6}")
        acc = 0.0
        for c, sh in conc["time_share"].items():
            acc += sh
            L.append(f"  {c:>11} {100 * sh:>6.1f}% {100 * acc:>5.1f}%")
        L.append(
            f"  P50 {conc['p50']}  P90 {conc['p90']}  max {conc['max']}  mean {conc['mean']}"
            f"  over {conc['window_s']:,.0f} s   (max-num-seqs {d['mns']})")
    else:
        L.append("  (no request end times in this snapshot)")
    rs = server.get("running") or {}
    if rs:
        L.append(
            "  server: vllm:num_requests_running (as recorded in the job's server metrics): "
            + "  ".join(
                f"{k} {v:.0f}" if isinstance(v, (int, float)) else f"{k} -"
                for k, v in rs.items()
                if k in ("p10", "p25", "p50", "p75", "p90", "max")))
    L += [
        "", "5. Summary",
        f"  {'percentile':<11}| cached / new input / output | concurrency"
    ]
    for p, tag in ((p, f"P{round(p * 100)}") for p in pcts):

        def g(key):
            return _k(_pct([r[key] for r in prof], p))

        L.append(
            f"  {tag:<11}| {g('reusable'):>6} / {g('new'):>5} / {g('osl'):>5}"
            f"{'':6}| {(conc or {}).get(tag.lower(), '-')}x")
    if not bars:
        return "\n".join(L)
    order = list(dict.fromkeys(r["conv"] for r in rows))
    top = max(r["isl"] for r in rows)
    omax = max(r["osl"] for r in rows) or 1
    L += [
        "",
        f"per request: # cache hit, = cached but recomputed, . new input; one char = "
        f"{top / width:,.0f} tokens; output on its own scale (| = {omax / 12:,.0f} tokens)",
        f"{'':4}{'turn':>7} {'isl':>7} {'cached':>7}{'from':>5} {'hit':>7} {'new':>6} "
        f"{'out':>6}  prompt"
    ]
    for i, c in enumerate(order, 1):
        if conv and not (c.startswith(conv) or conv == str(i)):
            continue
        g = [r for r in rows if r["conv"] == c]
        L.append(f"#{i} {c[:16]}  ({len(g)} turns)")
        for r in g:

            def n(v):
                return int(round(v / top * width))

            h, rc, nw = n(r["hit"]), n(r["reusable"] - r["hit"]), n(r["new"])
            tag = ("w" if r["phase"] == "warmup" else "t") + str(r["turn"])
            src = "-" if r["src_turn"] is None else str(r["src_turn"])
            L.append(
                f"{'!' if r['gap'] else ' ':>4}{tag:>7} {_k(r['isl']):>7} {_k(r['reusable']):>7}"
                f"{src:>5} {_k(r['hit']):>7} {_k(r['new']):>6} {_k(r['osl']):>6}  "
                f"{'#' * h}{'=' * rc}{'.' * nw:<{width - h - rc}} "
                f"{'|' * max(1 if r['osl'] else 0, int(round(r['osl'] / omax * 12)))}"
            )
    return "\n".join(L)


def env_block(env: list[dict], source_image: str | None) -> str:
    """The env the twin runs with, one row per key: value, where it came from, and whether it
    selects sharding communication. `review` rows need a decision before the numbers count."""
    L = [
        "", f"ENV passed to the twin (source job image: {source_image})",
        f"  {'name':<44} {'source':<12} {'sharding':<8} value"
    ]
    for r in env:
        v = str(r["value"])
        L.append(
            f"  {r['name']:<44} {r['source'][:12]:<12} {'yes' if r['sharding'] else '':<8} "
            f"{v if len(v) <= 70 else v[:67] + '...'}")
    rev = [r for r in env if r.get("review")]
    if any("NOT evaluated" in r["source"] for r in env):
        L.append(
            "  WARN the recipe script was not evaluated: env it exports is missing"
        )
    for r in rev:
        L.append(f"  REVIEW {r['name']}: {r['review']}")
    L.append(
        "  Not listed: the image's own defaults (Dockerfile ENV, vllm-torchtpu envs.py) and "
        "the pod's plumbing; forward_run records the effective env in result.json."
    )
    return "\n".join(L)


def worked_example(cfg: dict, snap: dict, tw: dict | None = None) -> str:
    d, f, c = cfg["deploy"], cfg["facts"], cfg["cells"]
    rec, s = snap["recipe"], snap["server"]
    pr = cfg["per_rank"]
    lines = [
        f"# Worked example: {cfg['model_dir']} from `{cfg['job_id']}`",
        "",
        f"Generated by `{cfg['_header']['generated_by']}` at {cfg['_header']['generated_at']}"
        f" (harness `{cfg['_header']['harness_commit']}`). Every number below is derived by"
        " `core/` from the checked-in snapshot; `cli.py selftest` re-derives and checks it.",
        "",
        "## 1. Inputs",
        f"- Image `{rec.get('image')}`, owner {rec.get('owner')}.",
        f"- Server args (from the job's recipe): TP {d['tp']}, DP {d['dp']}, DCP {d['dcp']} "
        f"(interleave {d['interleave']}, q-replicate {d['dcp_q_replicate']}), EP {d['ep']}, "
        f"block {d['block_size']}, mnbt {d['mnbt']}, mns {d['mns']}, max-model-len "
        f"{d['max_model_len']}, KV {d['kv_cache_dtype']}, spec {d['spec_method']} x "
        f"{d['spec_tokens']}, prefix caching {d['prefix_caching']}.",
        f"- Checkpoint config: `{cfg['source']['checkpoint_config']}`.",
        f"- Measured window: {c['requests_measured']} requests from {c['conversations']} "
        f"conversations, {s.get('profiling_duration_s', 0):.0f} s.",
        "",
        "## 2. Model facts",
        f"- {f['layers']} layers; kinds {f['layer_kinds']}.",
        f"- KV per token (all layers, from the model's model.py): {f['kv_bytes_per_token']} B"
        + (f"; recurrent state per sequence: {f['state_bytes_seq']} B"
           if f.get("state_bytes_seq") else "") +
        (f"; MTP layers' KV per token: {f['mtp_kv_bytes_token']} B"
         if f.get("mtp_kv_bytes_token") else "") + ".",
        f"- Kernel paths from the recipe env: {json.dumps({k: v for k, v in cfg['kernels'].items() if k != 'env'})}.",
        "",
        "## 3. Per-rank shapes (global / degree)",
    ]
    lines += [f"- `{k}`: {v}" for k, v in pr.items()]
    rc = c["recompute"]
    lines += [
        "",
        "## 4. Workload -> cells",
        f"- Cache block {rc['cache_block_tokens']} tokens; drafter block drop "
        f"{rc['eagle_drop']}. Computed tokens: model {rc['computed_tokens_model']:,} vs server "
        f"{rc['computed_tokens_server']:,.0f} (ratio {rc['model_over_server']}).",
        f"- Computed per request p5/p50/p90 {rc['computed_per_request_p05_p50_p90']} vs new "
        f"content p50 {rc['new_per_request_p50']}.",
        f"- Counts: {c['counts']}.",
        f"- Prefill cells (equal-weight bins of chunk context; full-chunk share "
        f"{c['full_chunk_share']}):",
    ]
    lines += [
        f"  - {p['name']}: chunk {p['chunk_tokens']} at cached {p['cached_context']:,}, "
        f"{p['decode_rows']['batch']} decode rows at {p['decode_rows']['contexts']}, "
        f"weight {p['weight']}" for p in c["prefill_cells"]
    ]
    lines += [
        "- Decode cells (batch around the mean decode rows; contexts = step-weighted "
        "quantiles):"
    ]
    lines += [
        f"  - {q['name']}: {q['batch']} x {q['tokens_per_row']} tokens at {q['contexts']}, "
        f"weight {q['weight']}" for q in c["decode_cells"]
    ]
    if cfg.get("closure_G0"):
        g = cfg["closure_G0"]
        lines += [
            f"- G0: {c['counts']['N_A_prefill_chunks']} x {g['t_A_s']} s + "
            f"{c['counts']['N_B_decode_steps']} x {g['t_B_s']} s = {g['predicted_s']} s vs "
            f"measured {g['measured_s']} s (ratio {g['ratio']}); prefill share "
            f"{g['prefill_share']}."
        ]
    lines += [
        "",
        "## 5. Projection onto the small slice (the twin config)",
        "```",
        twin_summary(tw) if tw else "(infeasible: not projected)",
        "```",
    ]
    lines += [
        "", "## 6. Feasibility", f"- {json.dumps(cfg['feasibility'])}", ""
    ]
    return "\n".join(lines)
