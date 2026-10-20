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
"""The sharding report: sweep, group by combination, open one example per group, chart.

    python3 search/report.py presets/<cell>.json \
        --out <runs>/<run_id>/artifacts/sharding-strategy/<cell>

Writes `<out>/REPORT.md`, the CSVs `sweep.py` writes, and `<out>/png/`. Without
matplotlib the report is still written and the two chart links print the command that
draws them.

What this file does and does not do
===================================
It **groups, orders, formats and links**. It computes nothing: every number in the
report is a field of `serving.Point` (via `sweep.py`) or of `derivation.Derivation`
(via `../roofline-analysis/derivation.py`), printed as returned. The only
arithmetic here is presentation -- a ratio of two printed numbers, a percentage of a
printed total -- and each of those sits next to both operands so a reader can redo it.

The report is built for auditing, not for skimming, and its order says so:

    0  the cell, the operating point, the pins, and how to reproduce
    1  the funnel and the index -- one row per combination, linking to its section
    2  the reasons: what was rejected (no such partition, does not fit) and why the
       blocked-today rows are blocked
    3+ one section per combination with an operating point -- shippable ones first,
       then the ones blocked today, which are forward-looking projections:
         how it works -> every member -> what EP and DCP change ->
         one worked example: weights, KV, collectives, where the time goes,
         and the derivation step by step, each step checked against the sweep
    n  the combined view -- every point on TTFT x throughput and TTFT x T_decode
    n+1 what the model gets wrong, and the artifacts

`[annotate]` blocks are the only prose the agent writes. Everything else is rendered.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.normpath(os.path.join(_HERE, "..", "..", "roofline-analysis")))

import derivation as DV  # noqa: E402
import roofline as R  # noqa: E402
import space as SP  # noqa: E402
import sweep as SW  # noqa: E402

GiB = R.GiB

#: What each rank axis does, in one line each, with where the full account lives.
#: Description, not arithmetic -- the numbers under it are `derivation.py`'s.
AXIS_NOTES = {
    "TP":
    ("**TP** splits every layer's heads and FFN columns across the cores of one "
     "engine. Each attention and FFN exit needs a reduction over the TP group. "
     "MLA's KV latent has no head axis, so **TP does not split the KV cache** — "
     "every TP rank holds the whole cache of its requests. "
     "`references/roofline-analysis/parallelism/tp/SKILL.md`"),
    "DP":
    ("**DP** runs independent engines, each holding its own copy of the "
     "non-expert weights, and splits the offered load `C/DP` per engine. It has "
     "no collective of its own; each engine has its own KV pool and admits its "
     "own requests. `references/roofline-analysis/parallelism/dp/SKILL.md`"),
    "PP":
    ("**PP** cuts the layer stack into stages, one per rank group, so weights and "
     "KV are divided by the stage's share of layers. A stage hands its hidden "
     "state to the next with one collective-permute per forward; decode keeps up "
     "to `PP` batches in flight, and TPOT is the full cycle through every stage. "
     "`references/roofline-analysis/parallelism/pp/SKILL.md`"),
    "PCP":
    ("**PCP** splits the prompt's sequence across ranks during prefill and "
     "replicates TP's weight shards. vLLM refuses it on a multihost slice today. "
     "`references/roofline-analysis/parallelism/pcp/SKILL.md`"),
}
VARIANT_NOTES = (
    "**EP** (a boolean) spreads the routed experts whole across the group `TP×PCP×DP`, "
    "instead of giving every core `1/(TP·PCP)` of every expert, and pays for it with a "
    "dispatch (all-gather) and a collect (reduce-scatter) per MoE layer. "
    "`references/roofline-analysis/parallelism/ep/SKILL.md`",
    "**DCP** subdivides TP and is **the only axis that splits an MLA KV cache**: "
    "each DCP rank keeps `1/DCP` of the sequence, at the price of a query gather and a "
    "partial-softmax merge per attention layer. `parallelism/dcp/SKILL.md`",
)

#: What the model gets wrong, carried from SKILL.md's table so the report is complete
#: on its own. Only the rows that bear on a one-point sharding comparison.
CAVEATS = [
    ("Collectives are fully exposed",
     "no compute/collective overlap is modelled; a "
     "profile puts real exposure at 21.9% of decode and 37.8% of prefill",
     "ICI-bound rows are the most pessimistic — read their margin as a floor"),
    ("DCP's combine is priced, its balance is assumed",
     "the sparse core's `K_idx/DCP` "
     "assumes balanced ownership",
     "treat a DCP row's margin over its `dcp=1` sibling "
     "as unmeasured"),
    ("PCP folds into TP for weights", "a `PCP` row is optimistic",
     "a ceiling on what "
     "fixing multihost PCP would buy"),
    ("The KV pool is sized against the heaviest PP stage, the step against the mean",
     "PP's balance term is `pp_balance`",
     "read PP rows' TPOT as the cycle, not `t_decode`"),
    ("`kernel_efficiency = 1.0`, peak FLOPs and bandwidth, no layout/copy term",
     "every absolute number is a ceiling",
     "quote ratios; no absolute tok/s is a forecast"),
]

# ============================================================ formatting


def f(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def gib(b: float, nd: int = 2) -> str:
    return f"{b / GiB:,.{nd}f}"


def table(head: list[str],
          rows: list[list[str]],
          align: str = "") -> list[str]:
    """Markdown table. `align` is one char per column: l, r, or c."""
    align = (align or "l" * len(head)).ljust(len(head), "l")
    sep = {"l": "---", "r": "---:", "c": ":---:"}
    return ([
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(sep[a] for a in align) + " |"
    ] + ["| " + " | ".join(r) + " |" for r in rows])


def anchor(title: str) -> str:
    keep = "".join(c for c in title.lower() if c.isalnum() or c in " -")
    return keep.strip().replace(" ", "-")


def annotate(what: str) -> list[str]:
    return ["", f"> **[annotate]** {what}", ""]


# ============================================================ grouping


def family_order(fam: str) -> tuple:
    axes = fam.split("×")
    return (len(axes), [
        SP.RANK_AXES.index(a.lower()) if a.lower() in SP.RANK_AXES else 9
        for a in axes
    ])


def group(res: SW.Result) -> dict[str, list[SW.Row]]:
    out: dict[str, list[SW.Row]] = defaultdict(list)
    for r in res.rows:
        out[r.sharding.family].append(r)
    return dict(sorted(out.items(), key=lambda kv: family_order(kv[0])))


def ship(rows: list[SW.Row]) -> list[SW.Row]:
    return [r for r in rows if r.shippable and r.point.feasible]


def best(rows: list[SW.Row]) -> SW.Row | None:
    ok = [r for r in rows if r.point.feasible]
    return max(ok, key=lambda r: r.point.output_tok_s) if ok else None


def by_rate(rows: list[SW.Row]) -> list[SW.Row]:
    return sorted(
        rows,
        key=lambda r:
        (not r.point.feasible, -r.point.output_tok_s, r.sharding.label))


# ============================================================ sections


def stack(r: SW.Row) -> str:
    """Whether today's stack accepts the sharding -- independent of whether it fits."""
    if r.cls == SP.SHIPPABLE and not r.over_cap:
        return "accepts"
    return "‡cap" if r.cls == SP.SHIPPABLE else f"refuses ({r.cls})"


def members_table(rows: list[SW.Row]) -> list[str]:
    body = []
    for r in by_rate(rows):
        p, s = r.point, r.sharding
        cls = stack(r)
        if not p.feasible:
            body.append([
                f"`{s.label}`", cls, "**no**",
                f(p.weights_gib, 1),
                f(p.kv_pool_gib, 1),
                f(p.kv_cap_reqs, 1), "—", "—", "—", "—", "—", "—", "—",
                p.reason.split(" -- ")[0]
            ])
            continue
        body.append([
            f"`{s.label}`", cls, "yes",
            f(p.weights_gib, 1),
            f(p.kv_pool_gib, 1),
            f(p.kv_cap_reqs, 1), f"{p.running:.1f} ({p.gate})",
            f(p.ttft_ms - p.queue_ms),
            f(p.tpot_ms, 2),
            f(p.ttft_ms),
            f(p.output_tok_s, 0),
            f(p.total_tok_s, 0), f"{p.step_bound} · {100 * p.ici_share:.0f}%",
            ""
        ])
    return table([
        "sharding", "stack today", "fits", "W GiB/core", "KV pool GiB",
        "KV cap req", "running (gate)", "T_prefill ms", "TPOT ms", "TTFT ms",
        "out tok/s", "total tokens throughput", "bound · ICI", "why not"
    ], body, "llcrrrlrrrrrll")


def variants_table(rows: list[SW.Row]) -> list[str]:
    """EP on vs off, and DCP>1 vs DCP=1, each against its sibling that differs in that
    one knob only. Both operands printed, so the ratio is checkable."""
    idx = {
        (r.sharding.tp, r.sharding.dp, r.sharding.pp, r.sharding.pcp, r.sharding.dcp, r.sharding.ep):
        r
        for r in rows
    }
    body = []
    for r in by_rate(rows):
        s = r.sharding
        pairs = []
        if s.ep:
            pairs.append(("EP on vs off",
                          idx.get((s.tp, s.dp, s.pp, s.pcp, s.dcp, False))))
        if s.dcp > 1:
            pairs.append((f"DCP {s.dcp} vs 1",
                          idx.get((s.tp, s.dp, s.pp, s.pcp, 1, s.ep))))
        for what, sib in pairs:
            if sib is None:
                continue
            a, b = r.point, sib.point
            if a.feasible and b.feasible:
                ratio = f"{a.output_tok_s / b.output_tok_s:.2f}×"
                body.append([
                    what, f"`{s.label}`", f"`{sib.sharding.label}`",
                    f"{f(a.output_tok_s, 0)} vs {f(b.output_tok_s, 0)}", ratio,
                    f"{f(a.tpot_ms, 2)} vs {f(b.tpot_ms, 2)}",
                    f"{f(a.ttft_ms)} vs {f(b.ttft_ms)}",
                    f"{f(a.kv_cap_reqs)} vs {f(b.kv_cap_reqs)}"
                ])
            else:
                dead = sib if not b.feasible else r
                body.append([
                    what, f"`{s.label}`", f"`{sib.sharding.label}`",
                    f"`{dead.sharding.label}` does not fit", "—", "—", "—", "—"
                ])
    if not body:
        return ["*No EP or DCP variants in this combination.*"]
    return table([
        "variant", "with", "without", "out tok/s", "ratio", "TPOT ms",
        "TTFT ms", "KV cap req"
    ], body, "lllrrrrr")


def weights_table(d: DV.Derivation) -> list[str]:
    tot = sum(t.local_bytes for t in d.weights)
    body = [[
        t.tensor,
        str(t.layers),
        gib(t.global_bytes), f"{t.divisor_expr} = {t.divisor}" +
        (f" · PP {1 / t.stage:.0f}" if t.stage < 1 else ""),
        gib(t.local_bytes, 3), f"{100 * t.local_bytes / tot:.1f}%", t.rule
    ] for t in d.weights]
    body.append([
        "**total**", "",
        gib(sum(t.global_bytes for t in d.weights)), "", f"**{gib(tot, 3)}**",
        "100%", ""
    ])
    return table([
        "tensor", "layers", "whole model GiB", "÷", "per core GiB", "share",
        "why that divisor"
    ], body, "lrrlrrl")


def kv_table(d: DV.Derivation) -> list[str]:
    body = [[
        t.cache,
        str(t.layers),
        f(t.bytes_per_token_layer, 0),
        f(t.global_bytes_per_token / 1024, 2),
        f"{t.divisor_expr} = {t.divisor}",
        f(t.local_bytes_per_token / 1024, 3), t.rule
    ] for t in d.kv]
    return table([
        "cache", "layers", "B / token / layer", "whole model KiB/token", "÷",
        "per core KiB/token", "why that divisor"
    ], body, "lrrrlrl")


def collectives_table(ph: DV.PhaseDetail) -> list[str]:
    if not ph.collectives:
        return [f"*No collective in {ph.name}.*"]
    body = [[
        c["op"],
        f(c["layers"], 0),
        f(c["ms_per_layer"], 4),
        f(c["total_ms"], 3), f"{100 * c['total_ms'] / ph.total_ms:.1f}%",
        c["comment"] or ""
    ] for c in sorted(ph.collectives, key=lambda c: -c["total_ms"])]
    tot = sum(c["total_ms"] for c in ph.collectives)
    body.append([
        "**all collectives**", "", "", f"**{f(tot, 3)}**",
        f"**{100 * tot / ph.total_ms:.1f}%**", ""
    ])
    return table([
        "collective", "layers", "ms / layer", "ms", f"% of {ph.name}",
        "what it carries"
    ], body, "lrrrrl")


def blocks_table(d: DV.Derivation) -> list[str]:
    dec, pre = d.decode, d.prefill
    names = sorted(set(dec.blocks) | set(pre.blocks),
                   key=lambda b: -(dec.blocks.get(b, 0) / dec.total_ms + pre.
                                   blocks.get(b, 0) / pre.total_ms))
    body = [[
        b,
        f(dec.blocks.get(b, 0.0),
          3), f"{100 * dec.blocks.get(b, 0.0) / dec.total_ms:.1f}%",
        f(pre.blocks.get(b, 0.0), 1),
        f"{100 * pre.blocks.get(b, 0.0) / pre.total_ms:.1f}%"
    ] for b in names]
    body.append([
        "**total**", f"**{f(dec.total_ms, 3)}**", "100%",
        f"**{f(pre.total_ms, 1)}**", "100%"
    ])
    return table(["block", "decode ms", "%", "prefill ms", "%"], body, "lrrrr")


def kinds_table(d: DV.Derivation) -> list[str]:
    pre = {k["kind"]: k for k in d.prefill.kinds}
    body = []
    for k in d.decode.kinds:
        p = pre.get(k["kind"], {})
        body.append([
            f"`{k['kind']}`",
            str(k["count"]),
            f(k["ms_per_layer"], 4),
            f(k["total_ms"], 3),
            f(p.get("ms_per_layer", 0.0), 2),
            f(p.get("total_ms", 0.0), 1)
        ])
    return table([
        "layer kind", "count", "decode ms/layer", "decode ms",
        "prefill ms/layer (all chunks)", "prefill ms"
    ], body, "lrrrrr")


def top_ops_table(ph: DV.PhaseDetail, n: int = 8) -> list[str]:
    body = [[
        o["op"], o["block"],
        f(o["layers"], 0),
        f(o["total_ms"],
          3), f"{100 * o['total_ms'] / ph.total_ms:.1f}%", o["bound"],
        f"{f(o['mem_ms'], 4)} / {f(o['comp_ms'], 4)} / {f(o['ici_ms'], 4)}"
    ] for o in sorted(ph.ops, key=lambda o: -o["total_ms"])[:n]]
    return table([
        "op", "block", "layers", "ms", "%", "bound",
        "per layer: HBM / MXU / ICI ms"
    ], body, "llrrrll")


def steps_table(d: DV.Derivation) -> list[str]:
    body = []
    for i, s in enumerate(d.steps, 1):
        nd = 4 if abs(s.value) < 10 else 2
        chk = "—" if s.check is None else (
            f"{s.check:,.{nd}f} ✓" if s.ok else f"**{s.check:,.{nd}f} ✗**")
        body.append([
            str(i), s.what, f"`{s.formula}`", s.inputs,
            f"**{s.value:,.{nd}f}** {s.unit}", chk
        ])
    return table(
        ["#", "quantity", "formula", "inputs", "value", "sweep returned"],
        body, "rlllrr")


def worked_example(res: SW.Result, row: SW.Row,
                   sec: str) -> tuple[list[str], DV.Derivation]:
    sp = res.spec
    d = DV.derive(res.config_for(row.sharding),
                  sp.workload,
                  sp.knobs,
                  sp.concurrency,
                  force=not row.verdict.ok)
    s = row.sharding
    L = [
        f"### {sec}.4 Worked example: `{s.label}`", "",
        ("The best shippable member of this combination" if row.shippable else
         f"**Projection** — the best member, blocked today ({row.cls}: "
         f"{row.verdict.reasons[0] if row.verdict.reasons else row.cap_reason})"
         ) + f" at the cell's operating point. "
        f"`{SP.ServerConfig(s, sp.max_num_seqs, sp.max_num_batched_tokens).flags()}`",
        ""
    ]
    L += [
        f"**{sec}.4.1 How the weights divide** — `roofline.weight_terms`. "
        f"The total is what the admission gate uses.", ""
    ]
    L += weights_table(d)
    if d.stage_weights_gib:
        L += [
            "",
            f"Under PP the table is the **mean** stage. The heaviest stage, which "
            f"is what must fit: {max(d.stage_weights_gib):.2f} GiB "
            f"(stages: {', '.join(f'{x:.2f}' for x in d.stage_weights_gib)})."
        ]
    L += [
        "", f"**{sec}.4.2 How the KV cache divides** — `roofline.kv_terms`.",
        ""
    ]
    L += kv_table(d)
    if d.decode is None:
        L += ["", f"**Does not fit:** {row.point.reason}", ""]
        return L, d
    L += [
        "",
        f"**{sec}.4.3 What it communicates** — every op with ICI time, one engine, "
        f"per core. Decode at {d.decode.tokens:.1f} tokens per batch:", ""
    ]
    L += collectives_table(d.decode)
    L += ["", f"Prefill, summed over its {d.prefill.passes} chunks:", ""]
    L += collectives_table(d.prefill)
    L += [
        "",
        f"**{sec}.4.4 Where the time goes** — the op table rolled into blocks, then "
        f"by layer kind. Both sum to the phase total.", ""
    ]
    L += blocks_table(d)
    L += [""] + kinds_table(d)
    L += ["", "Largest decode ops:", ""] + top_ops_table(d.decode)
    L += [
        "",
        f"**{sec}.4.5 The derivation** — model 1, step by step. Every value is "
        f"recomputed by `derivation.py` from the parts above; *sweep returned* is what "
        f"`serving.operating_point` returned for the same sharding. A ✗ is a finding.",
        ""
    ]
    L += steps_table(d)
    if d.decode_stage_ms:
        L += [
            "",
            "Decode stage times, ms: " + ", ".join(f"{x:.3f}"
                                                   for x in d.decode_stage_ms)
        ]
    return L, d


def family_section(res: SW.Result, fam: str, rows: list[SW.Row],
                   sec: str) -> tuple[list[str], DV.Derivation | None]:
    axes = fam.split("×")
    sh = ship(rows)
    proj = [r for r in rows if r.point.feasible and not r.shippable]
    title = sec_of[fam]
    L = [f"## {title}", ""]
    if not sh:
        rules = sorted({
            (r.verdict.reasons[0] if r.verdict.reasons else r.cap_reason)
            for r in proj
        })
        L += [
            "> **Projection — nothing in this combination runs today.** Blocked by: "
            + "; ".join(f"`{x}`"
                        for x in rules) + ". Every number below is what "
            "lifting that rule would be worth, priced by the same model as the "
            "shippable rows.", ""
        ]
    L += [f"**{sec}.1 How it works.**", ""]
    L += [f"- {AXIS_NOTES[a]}" for a in axes if a in AXIS_NOTES]
    L += [f"- {v}" for v in VARIANT_NOTES]
    L += annotate(
        f"2–4 sentences on why {fam} lands where it does in the combined view "
        f"(§{combined_no}), citing numbers "
        f"from {sec}.2–{sec}.4. Do not introduce a number the tables do not show."
    )
    L += [
        f"**{sec}.2 Every member** — {len(rows)} coherent: {len(sh)} shippable, "
        f"{len(proj)} blocked today (projection), "
        f"{sum(not r.point.feasible for r in rows)} do not fit. Sorted by throughput.",
        ""
    ]
    L += members_table(rows)
    L += [
        "", "↳ `points.csv`, filter `tp,dp,pp,pcp` to this combination. "
        "Columns: `columns.md`.", ""
    ]
    L += [
        f"**{sec}.3 What EP and DCP change** — each row against the sibling that "
        f"differs in that knob only.", ""
    ]
    L += variants_table(rows)
    L += [""]
    ex = best(sh) or best(proj)
    if ex is None:
        L += ["*No member fits, so there is no worked example.*", ""]
        return L, None
    body, d = worked_example(res, ex, sec)
    return L + body + [""], d


def reasons_section(res: SW.Result, fams: dict[str, list[SW.Row]], sec: str,
                    best_ship: SW.Row | None) -> list[str]:
    """Rejections (no operating point at all) and blocks (priced, refused today), each
    grouped by the rule that fired. The projections themselves live in their
    combinations' sections; this is only the why."""
    L = [f"## {sec}", ""]
    n = sec.split(".")[0]
    # ---- rejected: arithmetic (never a config) and memory (no operating point)
    arith: dict[str, list[str]] = defaultdict(list)
    for s_, v in res.enumerated:
        if v.blocking_class == SP.ARITHMETIC:
            arith[v.reason_for(SP.ARITHMETIC) or "?"].append(s_.label)
    mem = [r for r in res.rows if not r.point.feasible]
    L += [
        f"### {n}.1 Rejected — no operating point", "",
        "Not candidates at any width: `arithmetic` means the partition does not exist "
        "(it was never a config); `memory` means it exists and does not fit, and the "
        "numbers say by how much.", ""
    ]
    body = [[
        "`arithmetic`",
        str(len(v)), rule,
        ", ".join(f"`{x}`"
                  for x in sorted(v)[:3]) + (" …" if len(v) > 3 else "")
    ] for rule, v in sorted(arith.items(), key=lambda kv: -len(kv[1]))]
    L += table(["class", "shardings", "reason", "which"], body, "lrll") if body else \
        ["*No arithmetic rejections.*"]
    if mem:
        budget = mem[0].point.reason.split(" against ")[-1]
        L += [
            "",
            f"`memory` — {len(mem)} shardings with no KV pool. Per-core budget: "
            f"{budget}, less the resident overhead in each row's reason "
            f"(`points.csv` → `reason`).", ""
        ]
        L += table(
            ["sharding", "combination", "stack today", "W GiB/core", "cause"],
            [[
                f"`{r.sharding.label}`", r.sharding.family,
                stack(r),
                f(r.point.weights_gib, 1),
                r.point.reason.split(" -- ")[0].removeprefix("no KV pool: ")
            ] for r in sorted(mem,
                              key=lambda r: (family_order(r.sharding.family), r
                                             .point.weights_gib))], "lllrl")
    # ---- blocked today: why the projections are not runnable
    blocked = [r for r in res.rows if not r.shippable]
    by_rule: dict[str, list[SW.Row]] = defaultdict(list)
    for r in blocked:
        rule = (r.verdict.reasons[0]
                if r.verdict.reasons else r.cap_reason) or "?"
        by_rule[rule].append(r)
    L += [
        "", f"### {n}.2 Blocked today — why the projections do not run", "",
        "Coherent, priced by the same model, refused by the stack today. `infra` "
        "is a software refusal (most say \"yet\" in the source), `hardware` is measured "
        "on silicon, `dispatch` is the MoE dispatch cap. These rows appear in full in "
        "their combinations' sections, marked **projection**.", ""
    ]
    body = [[
        rs[0].cls if len({x.cls
                          for x in rs}) == 1 else "mixed",
        str(len(rs)),
        str(sum(x.point.feasible for x in rs)), rule,
        ", ".join(sorted({x.sharding.family
                          for x in rs}, key=family_order))
    ] for rule, rs in sorted(by_rule.items(), key=lambda kv: -len(kv[1]))]
    L += table(["class", "shardings", "of which fit", "the rule (first cited; "
                "`space.csv` has all)", "combinations"], body, "lrrll") if body else \
        ["*Nothing blocked — every coherent sharding is shippable.*"]
    ok = by_rate([r for r in blocked if r.point.feasible])
    if ok and best_ship:
        a, b = ok[0].point, best_ship.point
        L += [
            "",
            f"**What the blocks cost at this operating point:** the best blocked "
            f"row `{ok[0].sharding.label}` ({f(a.output_tok_s, 0)} tok/s) against the "
            f"best shippable `{best_ship.sharding.label}` ({f(b.output_tok_s, 0)}) is "
            f"{a.output_tok_s / b.output_tok_s:.2f}×."
        ]
    L += ["", "↳ every rule for every enumerated sharding: `space.csv`.", ""]
    return L


def combined_section(res: SW.Result, fams: dict[str, list[SW.Row]], sec: str,
                     charts: dict | None, cmd: str) -> list[str]:
    L = [
        f"## {sec}. Combined view", "",
        "Every sharding with an operating point, one point each, at the same load and "
        "knobs. Colour and marker are the combination (the first eight in section "
        "order; the rest fold into grey). **Filled = shippable today; hollow = blocked "
        "today, a projection.** The best member of each coloured combination is "
        "labelled.", ""
    ]
    if charts:
        L += [
            f"![throughput vs TTFT]({os.path.relpath(charts['throughput'], res_out)})",
            "*Fig 1 — output tok/s against TTFT. Up and to the left is better.*",
            "",
            f"![TPOT vs TTFT]({os.path.relpath(charts['decode'], res_out)})",
            "*Fig 2 — per-token decode time (TPOT; under PP the full cycle) against "
            "TTFT. Down and to the left is better.*", ""
        ]
    else:
        L += [f"*Charts not drawn — matplotlib is missing. Run:* `{cmd}`", ""]
    rows = by_rate([r for rs in fams.values() for r in rs if r.point.feasible])
    lead = max((r.point.output_tok_s for r in rows if r.shippable),
               default=1.0)
    L += [
        f"Every sharding that fits ({len(rows)}: "
        f"{sum(r.shippable for r in rows)} shippable, "
        f"{sum(not r.shippable for r in rows)} projections), ranked by throughput. "
        f"*vs best shippable* divides by the best shippable row's tok/s "
        f"({f(lead, 0)}).", ""
    ]
    L += table([
        "#", "sharding", "combination", "today", "out tok/s",
        "total tokens throughput", "vs best shippable", "TTFT ms", "TPOT ms",
        "T_prefill ms", "gate", "bound"
    ], [[
        str(i), f"`{r.sharding.label}`",
        f"[{r.sharding.family}](#{anchor(sec_of[r.sharding.family])})",
        "**shippable**" if r.shippable else f"projection ({r.cls})",
        f(r.point.output_tok_s, 0),
        f(r.point.total_tok_s, 0), f"{r.point.output_tok_s / lead:.2f}",
        f(r.point.ttft_ms),
        f(r.point.tpot_ms, 2),
        f(r.point.ttft_ms - r.point.queue_ms), r.point.gate, r.point.step_bound
    ] for i, r in enumerate(rows, 1)], "rlllrrrrrrll")
    sh = [r for r in rows if r.shippable]
    if len(sh) > 1:
        m = sh[0].point.output_tok_s / sh[1].point.output_tok_s
        L += [
            "",
            f"Margin between the two best shippable rows (`{sh[0].sharding.label}` "
            f"over `{sh[1].sharding.label}`): {m:.2f}×" +
            (" — under 5%, which is two candidates, not a ranking."
             if m < 1.05 else ".")
        ]
    L += annotate(
        "What the two charts show that the table does not — where the "
        "combinations separate, the latency each pays for its throughput, and "
        "which projections would move the frontier if their rule were lifted.")
    return L


res_out = ""  # the report's own directory, for relative links
combined_no = ""  # the combined view's section number, for cross-references
sec_of: dict[str, str] = {}


def section_families(fams: dict[str, list[SW.Row]]) -> list[str]:
    """Combinations that get a section: any member with an operating point. Shippable
    combinations first, then projection-only ones, each in canonical axis order."""
    live = [k for k, rs in fams.items() if any(r.point.feasible for r in rs)]
    return ([k for k in live if ship(fams[k])] +
            [k for k in live if not ship(fams[k])])


def render(res: SW.Result, out: str, charts: dict | None,
           fyi: SW.Result | None, cmd: str) -> str:
    global res_out, combined_no
    res_out = out
    sp, m = res.spec, res.base.model
    fams = group(res)
    order = section_families(fams)
    reasons_sec = "2. Rejections and blocks — the reasons"
    first = 3
    combined_sec = str(first + len(order))
    combined_no = combined_sec
    sec_of.clear()
    for i, fam in enumerate(order, first):
        sec_of[fam] = f"{i}. {fam}" + (""
                                       if ship(fams[fam]) else " (projection)")
    for fam in fams:
        sec_of.setdefault(fam, reasons_sec)

    L = [
        f"# Sharding report: {m.name} on `{sp.topology.name}`", "",
        "**Modelled — nothing here ran on hardware**, and none of it is an "
        "experiment result. Peak FLOPs and bandwidth, collectives fully exposed, "
        "`kernel_efficiency = 1.0`: every absolute number is a ceiling. Read ratios.",
        ""
    ]
    L += table(["", ""], [
        [
            "cell", f"{sp.topology.name} = {sp.topology.chips} chips / "
            f"{sp.topology.devices} cores / {sp.topology.hosts} hosts · "
            f"ISL {sp.isl} / OSL {sp.osl} · {sp.role}"
        ],
        [
            "operating point", f"**C = {sp.concurrency}** fleet-wide · "
            f"`max-num-batched-tokens` {sp.max_num_batched_tokens} · "
            f"`max-num-seqs` {sp.max_num_seqs} (both per DP engine) — "
            f"the same for every sharding"
        ],
        [
            "step model",
            "model 1 (native: prefill and decode as separate passes)"
        ],
        ["pins", f"`{os.path.relpath(sp.roofline_config)}`"],
        [
            "dispatch cap", "on — " +
            sp.dispatch_bound.note if sp.dispatch_bound.enabled else "off"
        ],
        ["generated", datetime.date.today().isoformat()],
    ])
    L += ["", "```bash", cmd, "```", ""]
    L += [
        "**Shippable** rows run on today's stack. **Projection** rows are coherent and "
        "fit, but the stack refuses them today (§2.2 says why); they are priced by the "
        "same model so the report is forward-looking, and are never a benchmark "
        "candidate until their rule is lifted.", "",
        "**Throughput columns are fleet-wide — the whole slice, all "
        f"{sp.topology.chips} chips.** `out tok/s` counts generated tokens only; "
        "**total tokens throughput** counts input and output: "
        f"`req/s × (ISL + OSL)` = `out tok/s × {sp.isl + sp.osl}/{sp.osl}`. "
        f"Divide by {sp.topology.chips} for per chip.", "",
        "Not in this report yet: the concurrency sweep and the `mnbt` ladder. Each "
        "sharding has exactly one operating point, so no statement here is about how "
        "a number moves with load.", ""
    ]

    # ---- 1. funnel + index
    L += ["## 1. The space, and where each combination lands", ""]
    L += table(["stage", "remaining", "removed", "class", "kind", "note"], [[
        x["stage"],
        str(x["count"]),
        str(x["removed"]), x["class"], x["kind"], x["note"]
    ] for x in res.funnel], "lrrlll")
    L += [
        "",
        "↳ `funnel.csv`; one row per enumerated sharding with every rule that "
        "refused it: `space.csv`.", "",
        "A **combination** is the set of axes that consume ranks — TP, DP, PP, PCP. "
        "EP and DCP consume none (EP is a boolean over TP×PCP×DP, DCP subdivides TP), "
        "so they are variants inside each combination, not combinations of their own.",
        ""
    ]
    idx = []
    for fam in order + [k for k in fams if k not in order]:
        rs = fams[fam]
        b = best(ship(rs)) or best(rs)
        idx.append([
            f"[{fam}](#{anchor(sec_of[fam])})",
            str(len(rs)),
            str(sum(r.point.feasible for r in rs)),
            str(len(ship(rs))),
            str(sum(1 for r in rs if not r.shippable and r.point.feasible)),
            (f"`{b.sharding.label}`" +
             ("" if b.shippable else " (proj.)")) if b else "—",
            f(b.point.output_tok_s, 0) if b else "—",
            f(b.point.total_tok_s, 0) if b else "—",
            f(b.point.ttft_ms) if b else "—",
            f(b.point.tpot_ms, 2) if b else "—"
        ])
    L += table([
        "combination", "coherent", "fit", "shippable", "projection",
        "best (shippable first)", "out tok/s", "total tokens throughput",
        "TTFT ms", "TPOT ms"
    ], idx, "lrrrrlrrrr")
    L += [""]

    # ---- 2. reasons
    all_ship = [r for rs in fams.values() for r in ship(rs)]
    L += reasons_section(res, fams, reasons_sec, best(all_ship))

    # ---- 3..k. one section per combination with an operating point
    derivs = []
    for i, fam in enumerate(order, first):
        body, d = family_section(res, fam, fams[fam], str(i))
        L += body
        if d:
            derivs.append(d)

    L += combined_section(res, fams, combined_sec, charts, cmd)

    if fyi is not None:
        a = [
            r.sharding.label for r in by_rate(
                [r for r in res.rows if r.shippable and r.point.feasible])
        ]
        pos = {
            r.sharding.label: r.point.output_tok_s
            for r in fyi.rows if r.shippable and r.point.feasible
        }
        a = [x for x in a if x in pos]
        inv = sum(1 for i in range(len(a)) for j in range(i + 1, len(a))
                  if pos[a[i]] < pos[a[j]])
        lead2 = max(pos, key=pos.get) if pos else "—"
        L += [
            "",
            f"**Model 2 (vLLM mixed step), for comparison:** shippable leader "
            f"`{lead2}`; {inv} of {len(a) * (len(a) - 1) // 2} shippable pairs order "
            f"differently than under model 1. A small count says the ranking does not "
            f"hinge on the step model; a large one says it is a hypothesis about the "
            f"scheduler.", ""
        ]

    # ---- caveats + artifacts
    n = int(combined_sec) + 1
    L += [f"## {n}. What the model gets wrong, and the artifacts", ""]
    L += table(["defect", "effect", "how to read around it"],
               [list(c) for c in CAVEATS])
    bad = [(d.label, s.key) for d in derivs for s in d.steps if not s.ok]
    L += [
        "", f"**Self-check:** {len(derivs)} worked examples, "
        f"{sum(len(d.steps) for d in derivs)} derivation steps, " +
        (f"**{len(bad)} disagree with the sweep: {bad}**" if bad else
         "every step that the sweep also returns agrees with it."), ""
    ]
    L += table(["file", "what"], [
        [
            "`points.csv`",
            "every coherent sharding at the operating point — **start here "
            "for any number not shown**"
        ],
        ["`columns.md`", "what every column means, generated with the data"],
        [
            "`space.csv`",
            "every enumerated sharding, every rule that refused it"
        ],
        ["`funnel.csv`", "§1's accounting"],
        ["`png/`", "the two combined-view charts"],
    ])
    L += ["", "The preset and the command above regenerate all of it.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fyi",
                    action="store_true",
                    help="skip the model 2 comparison")
    a = ap.parse_args()
    spec = SW.load_spec(a.spec)
    res = SW.run(spec, model=1)
    SW.write_csvs(res, a.out)
    fyi = None if a.no_fyi else SW.run(spec, model=2)
    cmd = (f"python3 {os.path.abspath(__file__)} "
           f"{os.path.abspath(a.spec)} --out {os.path.abspath(a.out)}")
    try:
        import plot as PL
        charts = PL.render(
            a.out,
            section_families(group(res)),
            sub=f"{spec.topology.name} · ISL {spec.isl}/OSL {spec.osl} · "
            f"C={spec.concurrency} · mnbt {spec.max_num_batched_tokens}")
    except ImportError:
        charts = None
        print(
            "matplotlib missing -- charts skipped. Install matplotlib (a host "
            "dependency of the sharding-strategy skill) and re-run.")
    text = render(res, a.out, charts, fyi, cmd)
    path = os.path.join(a.out, "REPORT.md")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
