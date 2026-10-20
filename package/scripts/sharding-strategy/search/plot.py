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
"""Render the combined view from `points.csv`. Reads nothing else, so a chart cannot
contradict the table beside it.

    python3 search/plot.py <runs>/<run_id>/artifacts/sharding-strategy/<cell>

matplotlib is a host dependency of the sharding-strategy skill. `report.py` calls `render()` itself when matplotlib imports.

Palette
=======
From the `dataviz` skill's reference palette, in its fixed order, never cycled: colour
follows the family, so a cell with fewer families does not repaint the rest. A scatter
needs an all-pairs-safe set and only the first three slots are, so every family also
gets its own marker -- colour is never the only thing telling two points apart.
Shippable points are filled and projections (blocked today) hollow, in their
combination's colour. Past eight combinations the rest fold into grey rather than
getting a generated hue.

Every axis is stamped `modelled`, because a PNG outlives the markdown it was pasted next
to and an unlabelled ceiling eventually gets read as a measurement.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import space as SP  # noqa: E402

# dataviz reference palette, light mode, in its fixed order. Do not re-order.
SERIES = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#4a3aa7", "#e34948"
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OTHER = "#8a8a85"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#c9c8c2"


def read(d: str) -> list[dict]:
    with open(os.path.join(d, "points.csv")) as f:
        return list(csv.DictReader(f))


def family(r: dict) -> str:
    return SP.family_of(int(r["tp"]), int(r["dp"]), int(r["pp"]),
                        int(r["pcp"]))


def style(ax, title: str, xlabel: str, ylabel: str, sub: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title + ("\n" + sub if sub else ""),
                 color=INK,
                 fontsize=11,
                 loc="left",
                 pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color=MUTED, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def finish(fig, path: str) -> None:
    fig.text(0.005,
             0.005,
             "modelled — nothing here ran on hardware",
             color=INK2,
             fontsize=7)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def scatter(rows: list[dict], families: list[str], y: str, ylabel: str,
            title: str, sub: str, path: str) -> None:
    """`families` is the report's section order; the first eight get a hue and a
    marker, the rest fold into grey. Filled = shippable, hollow = projection. Labels:
    the best shippable member of each combination, and the single best projection --
    labelling every combination's best piles a dozen names into one corner."""
    ok = [r for r in rows if r["feasible"] == "1"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    style(ax, title, "TTFT, ms (log) — modelled", ylabel, sub)
    ax.set_xscale("log")
    slot = {fam: i for i, fam in enumerate(families[:len(SERIES)])}
    handles = []
    for fam in families + ["other"]:
        if fam == "other":
            pts = [r for r in ok if family(r) not in slot]
            col, mk, name = OTHER, "o", "other combinations"
        else:
            if fam not in slot:
                continue
            pts = [r for r in ok if family(r) == fam]
            col, mk, name = SERIES[slot[fam]], MARKERS[slot[fam]], fam
        if not pts:
            continue
        for shp in ("1", "0"):
            sel = [r for r in pts if r["shippable"] == shp]
            if not sel:
                continue
            ax.scatter([float(r["ttft_ms"]) for r in sel],
                       [float(r[y]) for r in sel],
                       s=46 if shp == "1" else 34,
                       marker=mk,
                       facecolors=col if shp == "1" else "none",
                       edgecolors=col,
                       linewidths=0.6 if shp == "1" else 1.0,
                       zorder=3 if shp == "1" else 2)
        shipped = [r for r in pts if r["shippable"] == "1"]
        if fam != "other" and shipped:
            best = max(shipped, key=lambda r: float(r["output_tok_s"]))
            ax.annotate(best["sharding"],
                        (float(best["ttft_ms"]), float(best[y])),
                        xytext=(6, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=INK2)
        handles.append(
            Line2D([], [],
                   color=col,
                   marker=mk,
                   linestyle="",
                   markersize=7,
                   label=name))
    proj = [r for r in ok if r["shippable"] != "1"]
    if proj:  # the one projection worth naming: the fastest
        top = max(proj, key=lambda r: float(r["output_tok_s"]))
        ax.annotate(top["sharding"] + " (projection)",
                    (float(top["ttft_ms"]), float(top[y])),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK2)
    handles += [
        Line2D([], [],
               color=INK2,
               marker="o",
               linestyle="",
               markersize=6,
               label="filled: shippable today"),
        Line2D([], [],
               color=INK2,
               marker="o",
               linestyle="",
               markersize=6,
               markerfacecolor="none",
               label="hollow: projection (blocked today)")
    ]
    ax.legend(handles=handles,
              fontsize=7.5,
              frameon=False,
              loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    finish(fig, path)


def render(d: str,
           families: list[str],
           sub: str = "",
           out: str | None = None) -> dict:
    out = out or os.path.join(d, "png")
    os.makedirs(out, exist_ok=True)
    rows = read(d)
    paths = {
        "throughput": os.path.join(out, "throughput_vs_ttft.png"),
        "decode": os.path.join(out, "tdecode_vs_ttft.png")
    }
    scatter(rows, families, "output_tok_s", "output tok/s, fleet — modelled",
            "Throughput against TTFT, every costed sharding", sub,
            paths["throughput"])
    scatter(rows, families, "tpot_ms",
            "T_decode per output token (TPOT), ms — modelled",
            "Decode step against TTFT, every costed sharding", sub,
            paths["decode"])
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir",
                    help="a --out directory written by sweep.py or report.py")
    ap.add_argument("--out", help="where the PNGs go (default: <dir>/png)")
    a = ap.parse_args()
    rows = read(a.dir)
    ship = {
        family(r)
        for r in rows if r["shippable"] == "1" and r["feasible"] == "1"
    }
    live = sorted({family(r)
                   for r in rows if r["feasible"] == "1"},
                  key=lambda f: (f not in ship, len(f.split("×")), f))
    fams = live
    render(a.dir, fams, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
