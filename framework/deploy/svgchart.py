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
"""Scatter-with-lines charts as standalone SVG, standard library only.

The host that runs the agents has Python and PyYAML and nothing else
promised, so the leaderboard charts cannot depend on matplotlib. This
draws what the InferenceX dashboard draws -- one series per
hardware/stack, points along a concurrency sweep joined in x order,
log or linear axes, point labels -- into an SVG a browser or a doc
renders, plus the same data as CSV next to it.
"""

from __future__ import annotations

import html
import math
from typing import Any

PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")
MARKERS = ("circle", "square", "diamond", "triangle")


def _ticks(lo: float, hi: float, log: bool) -> list[float]:
    if log:
        lo_e = math.floor(math.log10(lo))
        hi_e = math.ceil(math.log10(hi))
        out = []
        for e in range(lo_e, hi_e + 1):
            for m in (1, 2, 5):
                v = m * 10 ** e
                if lo <= v <= hi:
                    out.append(v)
        return out or [lo, hi]
    span = hi - lo or 1.0
    step = 10 ** math.floor(math.log10(span / 5))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= 7:
            step *= mult
            break
    start = math.floor(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e6:
        m = v / 1e6
        return f"{m:.0f}M" if float(m).is_integer() else f"{m:.1f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _marker(kind: str, x: float, y: float, r: float, color: str,
            hollow: bool) -> str:
    fill = "white" if hollow else color
    if kind == "square":
        return (f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}"'
                f' height="{2 * r:.1f}" fill="{fill}" stroke="{color}"'
                ' stroke-width="1.5"/>')
    if kind == "diamond":
        pts = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}"
        return (f'<polygon points="{pts}" fill="{fill}" stroke="{color}"'
                ' stroke-width="1.5"/>')
    if kind == "triangle":
        pts = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y + r:.1f} {x - r:.1f},{y + r:.1f}"
        return (f'<polygon points="{pts}" fill="{fill}" stroke="{color}"'
                ' stroke-width="1.5"/>')
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"'
            f' stroke="{color}" stroke-width="1.5"/>')


def scatter_svg(series: list[dict[str, Any]], *, title: str,
                x_label: str, y_label: str, log_x: bool = True,
                log_y: bool = False, width: int = 960, height: int = 600,
                subtitle: str = "", footnote: str = "") -> str:
    """`series`: [{"name", "points": [{"x", "y", "label"?, "hollow"?,
    "on_line"?, "faded"?}], "color"?, "marker"?, "dashed"?, "width"?}].
    The line joins, in ascending x, the points whose `on_line` is not
    false (a caller drawing a Pareto frontier sets it on the frontier
    only); a `faded` point is drawn translucent and unlabelled (a
    dominated point); a hollow point is one that failed its gate."""
    pad_l, pad_r, pad_t, pad_b = 90, 260, 70, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs = [p["x"] for s in series for p in s["points"]
          if p.get("x") is not None and p.get("y") is not None]
    ys = [p["y"] for s in series for p in s["points"]
          if p.get("x") is not None and p.get("y") is not None]
    if not xs:
        xs, ys = [1.0, 10.0], [0.0, 1.0]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if log_x:
        x_lo, x_hi = max(x_lo, 1e-9) / 1.3, x_hi * 1.3
    else:
        span = (x_hi - x_lo) or 1.0
        x_lo, x_hi = x_lo - 0.05 * span, x_hi + 0.05 * span
    if log_y:
        y_lo, y_hi = max(y_lo, 1e-9) / 1.3, y_hi * 1.3
    else:
        span = (y_hi - y_lo) or 1.0
        y_lo, y_hi = min(0.0, y_lo), y_hi + 0.12 * span
    if x_lo == x_hi:
        x_hi = x_lo + 1
    if y_lo == y_hi:
        y_hi = y_lo + 1

    def sx(v: float) -> float:
        if log_x:
            return pad_l + plot_w * (math.log10(v) - math.log10(x_lo)) / (
                math.log10(x_hi) - math.log10(x_lo))
        return pad_l + plot_w * (v - x_lo) / (x_hi - x_lo)

    def sy(v: float) -> float:
        if log_y:
            return pad_t + plot_h - plot_h * (
                math.log10(v) - math.log10(y_lo)) / (
                math.log10(y_hi) - math.log10(y_lo))
        return pad_t + plot_h - plot_h * (v - y_lo) / (y_hi - y_lo)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"'
           f' height="{height}" viewBox="0 0 {width} {height}"'
           ' font-family="Helvetica, Arial, sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{pad_l}" y="30" font-size="18" font-weight="bold">'
           f'{html.escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="{pad_l}" y="50" fill="#555">'
                   f'{html.escape(subtitle)}</text>')
    # grid + axes
    for v in _ticks(x_lo, x_hi, log_x):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}"'
                   f' y2="{pad_t + plot_h}" stroke="#eee"/>')
        out.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}"'
                   f' text-anchor="middle" fill="#333">{_fmt(v)}</text>')
    for v in _ticks(y_lo, y_hi, log_y):
        y = sy(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}"'
                   f' y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end"'
                   f' fill="#333">{_fmt(v)}</text>')
    out.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}"'
               f' height="{plot_h}" fill="none" stroke="#999"/>')
    out.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 40}"'
               f' text-anchor="middle" font-size="13">{html.escape(x_label)}'
               '</text>')
    out.append(f'<text transform="translate(22,{pad_t + plot_h / 2:.1f})'
               ' rotate(-90)" text-anchor="middle" font-size="13">'
               f'{html.escape(y_label)}</text>')
    # series
    legend_y = pad_t
    for i, s in enumerate(series):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        marker = s.get("marker") or MARKERS[(i // len(PALETTE)) % len(MARKERS)]
        pts = sorted([p for p in s["points"]
                      if p.get("x") is not None and p.get("y") is not None],
                     key=lambda p: p["x"])
        line_pts = [p for p in pts if p.get("on_line", True)]
        if len(line_pts) > 1:
            path = " ".join(f"{'M' if k == 0 else 'L'}{sx(p['x']):.1f},"
                            f"{sy(p['y']):.1f}" for k, p in enumerate(line_pts))
            dash = ' stroke-dasharray="6,4"' if s.get("dashed") else ""
            out.append(f'<path d="{path}" fill="none" stroke="{color}"'
                       f' stroke-width="{s.get("width", 2)}"{dash}'
                       ' opacity="0.9"/>')
        for p in pts:
            x, y = sx(p["x"]), sy(p["y"])
            if p.get("faded"):
                out.append('<g opacity="0.3">' + _marker(
                    marker, x, y, 3.5, color, bool(p.get("hollow"))) + '</g>')
                continue
            out.append(_marker(marker, x, y, 5, color, bool(p.get("hollow"))))
            if p.get("label"):
                out.append(f'<text x="{x + 7:.1f}" y="{y - 6:.1f}"'
                           f' font-size="10" fill="{color}">'
                           f'{html.escape(str(p["label"]))}</text>')
        lx = pad_l + plot_w + 16
        out.append(_marker(marker, lx + 6, legend_y + 6, 5, color, False))
        out.append(f'<text x="{lx + 18}" y="{legend_y + 10}" font-size="12">'
                   f'{html.escape(s["name"])}</text>')
        legend_y += 20
    if any(p.get("hollow") for s in series for p in s["points"]):
        lx = pad_l + plot_w + 16
        out.append(_marker("circle", lx + 6, legend_y + 10, 5, "#555", True))
        out.append(f'<text x="{lx + 18}" y="{legend_y + 14}" font-size="11"'
                   ' fill="#555">hollow: failed the qualifying gate</text>')
        legend_y += 20
    if any(p.get("faded") for s in series for p in s["points"]):
        lx = pad_l + plot_w + 16
        out.append('<g opacity="0.3">' + _marker(
            "circle", lx + 6, legend_y + 10, 3.5, "#555", False) + '</g>')
        out.append(f'<text x="{lx + 18}" y="{legend_y + 14}" font-size="11"'
                   ' fill="#555">faded: dominated (off the frontier)</text>')
        legend_y += 20
    if footnote:
        out.append(f'<text x="{pad_l}" y="{height - 14}" font-size="10"'
                   f' fill="#777">{html.escape(footnote)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
