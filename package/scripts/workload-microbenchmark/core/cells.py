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
"""Recorded requests -> representative cells, weights and step counts.

The dynamic run is summarised as two step shapes (P1 in design-notes):
  A  mixed prefill step: one prefill chunk on a cached prefix + the running decodes
  B  decode step: B requests x (1 + spec) tokens
and the counts N_A (prefill chunks) and N_B (decode-only steps) that multiply them.

Everything here is derived from the recorded measured-window requests and the
deployment's cache rules, then cross-checked against a server counter, so a cell is
the job's own mix rather than a guess.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .deploy import Deploy
from .jobdata import prefix_chain


def _q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


#: Gates the checks use (replayed computed tokens vs the server's counter; the G0 closure).
REPLAY_GATE = 0.03
CLOSURE_GATE = 0.10


def recompute(requests: list[dict], d: Deploy, cache_hit=None) -> list[dict]:
    """Per measured request: trace-level reusable prefix, the server's cache hit under
    this deployment, and the tokens it must compute.

    Default hit rule (vLLM's full-attention prefix cache):
      - hashing is in blocks of `block x DCP` tokens;
      - the hit is capped at num_tokens - 1;
      - eagle-style drafters (MTP included) drop one whole block from the hit.
    A model whose cache hits differently (e.g. hybrid models that can only resume from saved
    recurrent state) passes its own `cache_hit(r, d) -> hit tokens` (model.py).
    The reusable prefix is the longest earlier (prompt + output) of the same
    conversation that the prompt extends.
    """
    return [
        with_hit(r, d, cache_hit) for r in prefix_chain(requests)
        if r.get("phase") == "profiling"
    ]


def with_hit(r: dict, d: Deploy, cache_hit=None) -> dict:
    """Apply the hit rule (the model's, or the default above) to one `prefix_chain` row."""
    if cache_hit:
        hit = int(cache_hit(r, d)) if d.prefix_caching and r["reusable"] else 0
        return {**r, "hit": hit, "computed": r["isl"] - hit}
    G = d.cache_block_tokens
    base = r["reusable"]
    if d.prefix_caching and base:
        cap = (math.floor((r["isl"] - 1) / G)) * G
        hit = max(0, min(cap, (base // G) * G) - (G if d.eagle_drop else 0))
    else:
        hit = 0
    return {**r, "hit": hit, "computed": r["isl"] - hit}


def derive(requests: list[dict],
           d: Deploy,
           server: dict[str, Any],
           n_prefill_cells: int = 3,
           cache_hit=None) -> dict[str, Any]:
    reqs = recompute(requests, d, cache_hit)
    if not reqs:
        raise ValueError("no measured-window requests")
    spec = d.spec_tokens if d.eagle_drop or d.spec_method else 0
    tokens_per_decode_row = 1 + spec

    # --- prefill chunks: each request's computed span, cut into scheduler chunks.
    # The chunk is the budget minus the decode rows riding along (scheduler.py fills
    # decode first); the mean decode batch sets how much that is.
    drafts = server.get("spec_drafts") or 0
    iters = server.get("iterations") or 0
    b_mean = (drafts / iters) if drafts and iters else None
    n_dec_in_mixed = max(0, round(b_mean)) if b_mean else 0
    chunk = max(1, d.mnbt - n_dec_in_mixed * tokens_per_decode_row)
    chunks = []
    for r in reqs:
        s = r["hit"]
        while s < r["isl"]:
            n = min(chunk, r["isl"] - s)
            chunks.append((n, s))
            s += n
    n_a = len(chunks)

    # prefill cells: equal-weight quantile bins of chunk context (the cached tokens
    # before the chunk); representative = the bin's median. Chunk size = full budget,
    # plus the partial-chunk share reported separately.
    ctx = sorted(s for _, s in chunks)
    edges = [
        ctx[int(i * len(ctx) / n_prefill_cells)]
        for i in range(n_prefill_cells)
    ] + [math.inf]
    pcells = []
    for i in range(n_prefill_cells):
        sel = [c for c in ctx if edges[i] <= c < edges[i + 1]] if i < n_prefill_cells - 1 \
            else [c for c in ctx if c >= edges[i]]
        if sel:
            pcells.append({
                "name": f"P{i+1}",
                "chunk_tokens": chunk,
                "cached_context": int(statistics.median(sel)),
                "weight": round(len(sel) / len(ctx), 3)
            })
    full_share = sum(1 for n, _ in chunks if n == chunk) / n_a

    # --- decode rows: each request contributes ~osl / accept steps at its context.
    accept = 1 + (server["spec_accepted"] / drafts) if drafts else 1.0
    rows = [(r["isl"], (r.get("osl") or 0) / accept) for r in reqs
            if r.get("osl")]
    tot = sum(w for _, w in rows)

    def ctx_quantile(p: float) -> int:
        acc = 0.0
        for c, w in sorted(rows):
            acc += w
            if acc >= p * tot:
                return int(c)
        return int(sorted(rows)[-1][0])

    def contexts(n: int) -> list[int]:
        return [ctx_quantile((i + 0.5) / n) for i in range(n)]

    r_dec = drafts or round(tot)
    n_b = max(0, iters - n_a) if iters else None
    if b_mean:
        bs = sorted({
            max(1, round(b_mean * 0.4)),
            max(1, round(b_mean)),
            min(d.mns, max(1, round(b_mean * 1.6)))
        })
    else:
        bs = [1]
    wts = {1: [1.0], 2: [0.5, 0.5], 3: [0.25, 0.5, 0.25]}[len(bs)]
    dcells = [{
        "name": f"D{b}",
        "batch": b,
        "tokens_per_row": tokens_per_decode_row,
        "contexts": contexts(b),
        "weight": w
    } for b, w in zip(bs, wts)]
    for p in pcells:
        p["decode_rows"] = {
            "batch": n_dec_in_mixed,
            "tokens_per_row": tokens_per_decode_row,
            "contexts": contexts(n_dec_in_mixed) if n_dec_in_mixed else []
        }

    computed_model = sum(r["computed"] for r in reqs)
    computed_server = server.get("prompt_tokens_computed")
    return {
        "requests_measured": len(reqs),
        "conversations": len({r["conv"]
                              for r in reqs}),
        "prefill_cells": pcells,
        "decode_cells": dcells,
        "full_chunk_share": round(full_share, 3),
        "counts": {
            "N_A_prefill_chunks": n_a,
            "N_B_decode_steps": n_b,
            "R_dec_request_decode_steps": r_dec,
            "B_mean_decode_rows": round(b_mean, 2) if b_mean else None,
            "accept_len": round(accept, 3)
        },
        "recompute": {
            "cache_block_tokens":
            d.cache_block_tokens,
            "eagle_drop":
            d.eagle_drop,
            "computed_tokens_model":
            computed_model,
            "computed_tokens_server":
            computed_server,
            "model_over_server":
            round(computed_model /
                  computed_server, 4) if computed_server else None,
            "computed_per_request_p05_p50_p90": [
                int(_q([r["computed"] for r in reqs], p))
                for p in (0.05, 0.5, 0.9)
            ],
            "new_per_request_p50":
            int(_q([r["new"] for r in reqs], 0.5)),
        },
        "distributions": {
            "isl_p50_p90_max":
            [int(_q([r["isl"] for r in reqs], p)) for p in (0.5, 0.9, 1.0)],
            "osl_p50_mean_p90": [
                int(_q([r["osl"] for r in reqs], 0.5)),
                int(statistics.mean(r["osl"] for r in reqs)),
                int(_q([r["osl"] for r in reqs], 0.9))
            ],
        },
    }


def closure(cells: dict[str, Any], server: dict[str, Any],
            log: dict[str, Any]) -> dict | None:
    """G0: does N_A * t_A + N_B * t_B reproduce the measured window? t_A from the server log
    (full-budget prefill steps), t_B = ITL p50 (a decode step)."""
    t_a, t_b = log.get("prefill_step_s"), (server.get("itl_s")
                                           or {}).get("p50")
    c = cells["counts"]
    if not (t_a and t_b and c["N_B_decode_steps"] is not None):
        return None
    pred = c["N_A_prefill_chunks"] * t_a + c["N_B_decode_steps"] * t_b
    meas = server.get("profiling_duration_s")
    return {
        "t_A_s": t_a,
        "t_B_s": round(t_b, 4),
        "predicted_s": round(pred, 1),
        "measured_s": round(meas, 1) if meas else None,
        "ratio": round(pred / meas, 3) if meas else None,
        "prefill_share": round(c["N_A_prefill_chunks"] * t_a / pred, 3)
    }
