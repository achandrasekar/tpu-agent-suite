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
"""Deterministic benchmark speedup calculator and candidate debugger for TPU serving.

Speedup is strictly defined as throughput TPS (Tokens Per Second, output tok/s).
Evaluates matched-concurrency and peak-to-peak speedups across baseline and
optimized runs, while printing a complete debug trace of all candidate values
used and considered across concurrency sweeps and regimes.
"""

import fnmatch
import json
import math
import pathlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rewardkit import criterion

Path = pathlib.Path

BENCHMARK_MANIFEST_FILENAME = "benchmark_results.json"
VALID_REGIMES = {"1k/1k", "8k/1k", "1k/8k"}

_REGIME_PATTERNS = {
    "8k/1k":
    re.compile(r"isl8192_osl1024|8k[/_-]?1k|8192[/:_x-]1024|prefill",
               re.IGNORECASE),
    "1k/8k":
    re.compile(r"isl1024_osl8192|1k[/_-]?8k|1024[/:_x-]8192|decode",
               re.IGNORECASE),
    "1k/1k":
    re.compile(r"isl1024_osl1024|1k[/_-]?1k|1024[/:_x-]1024|balanced",
               re.IGNORECASE),
}

_VALID_TABLE_CONCURRENCIES = {
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048
}


def safe_div(
    num: float | None,
    den: float | None,
    decimals: int = 4,
) -> float | None:
    """Safely computes num / den rounded to specified decimals, or None."""
    if num is None or den is None or den == 0.0:
        return None
    return round(float(num) / float(den), decimals)


def _first(
    dicts: Iterable[dict[str, Any]],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    """Returns the first non-None value found across dicts and keys (avoids falsy 0 bugs)."""
    for d in dicts:
        for k in keys:
            if d.get(k) is not None:
                return d[k]
    return default


def _to_float(val: Any, default: float | None = None) -> float | None:
    try:
        return float(str(val).split()[0]) if val is not None else default
    except (ValueError, TypeError, IndexError):
        return default


def _to_int(val: Any, default: int | None = None) -> int | None:
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        m = re.search(r"\b(\d+)\b", val)
        return int(m.group(1)) if m else default
    try:
        return int(round(float(val))) if val is not None else default
    except (ValueError, TypeError):
        return default


def infer_regime_from_filename(text: str) -> str | None:
    """Detects canonical regime name from a filename, heading, or raw string."""
    for canonical, pattern in _REGIME_PATTERNS.items():
        if pattern.search(str(text)):
            return canonical
    return None


def normalize_regime_name(name: str) -> str:
    """Normalizes variations like '1k1k', '1k_1k', '1024:1024', 'balanced' into '1k/1k'."""
    return (infer_regime_from_filename(name)
            or str(name).lower().replace("-", "/").replace("_", "/").strip())


def infer_regime_from_isl_osl(isl: Any, osl: Any) -> str | None:
    """Maps (isl, osl) token counts to canonical regime names."""
    return {
        (1024, 1024): "1k/1k",
        (8192, 1024): "8k/1k",
        (1024, 8192): "1k/8k",
    }.get((_to_int(isl), _to_int(osl)))


def infer_concurrency_from_filename(filename: str) -> int | None:
    """Infers request concurrency from a filename (e.g. 'bench_1k1k_c16.json') if present."""
    m = re.search(
        r"(?:^|[_\-\b])(?:c|conc|concurrency)[_\-]?(\d+)(?:$|[_\-\.])",
        filename.lower(),
    )
    return int(m.group(1)) if m else None


@dataclass
class BenchmarkRunMetrics:
    """Telemetry metrics extracted from a single benchmark run."""

    regime: str
    concurrency: int
    output_tps: float
    duration_s: float = 0.0
    completed: int = 0
    failed: int = 0
    total_tps: float | None = None
    req_per_s: float | None = None
    ttft_p50: float | None = None
    itl_p50: float | None = None
    e2e_p50: float | None = None
    source: str = ""
    is_queued: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        self.regime = normalize_regime_name(self.regime)
        self.concurrency = int(self.concurrency)
        self.output_tps = float(self.output_tps)
        self.duration_s = float(self.duration_s)
        self.completed = int(self.completed)
        self.failed = int(self.failed)
        self.total_tps = float(
            self.total_tps) if self.total_tps is not None else None
        self.req_per_s = float(
            self.req_per_s) if self.req_per_s is not None else None
        self.ttft_p50 = float(
            self.ttft_p50) if self.ttft_p50 is not None else None
        self.itl_p50 = float(
            self.itl_p50) if self.itl_p50 is not None else None
        self.e2e_p50 = float(
            self.e2e_p50) if self.e2e_p50 is not None else None
        self.source = str(self.source)
        self.is_queued = bool(self.is_queued)
        self.notes = str(self.notes)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        default_regime: str = "1k/1k",
        source: str = "",
    ) -> "BenchmarkRunMetrics":
        """Parses a benchmark result dictionary from vllm bench serve, summary, or raw JSON."""
        s = raw.get("summary") if isinstance(raw.get("summary"), dict) else raw
        sources = (s, raw)

        completed = (_to_int(
            _first(sources,
                   ("num_successful", "completed_requests", "completed")),
            0,
        ) or 0)
        failed = (_to_int(
            _first(sources, ("num_failed", "failed_requests", "failed")),
            0,
        ) or 0)
        num_prompts = (_to_int(
            _first(sources, ("num_prompts", "total_requests", "num_requests")),
            0,
        ) or 0)
        if failed == 0 and 0 < completed < num_prompts:
            failed = num_prompts - completed

        isl = _first(sources, ("isl", "random_input_len", "input_len"))
        osl = _first(sources, ("osl", "random_output_len", "output_len"))
        tot_in = _first(sources, ("total_input", "total_input_tokens"))
        tot_out = _first(sources, ("total_output", "total_output_tokens"))
        regime = (infer_regime_from_isl_osl(isl, osl)
                  or (infer_regime_from_isl_osl(
                      float(tot_in) / completed,
                      float(tot_out) / completed) if completed > 0
                      and tot_in is not None and tot_out is not None else None)
                  or _first(sources, ("regime", ), default_regime))
        concurrency = (_to_int(
            _first(
                sources,
                ("concurrency", "max_concurrency", "num_concurrent_requests",
                 "c"),
            ),
            1,
        ) or 1)
        duration_s = (_to_float(
            _first(sources, ("total_wall_time_s", "duration_s", "duration")),
            0.0,
        ) or 0.0)

        output_tps = (_to_float(
            _first(
                sources,
                (
                    "output_throughput_tok_s",
                    "throughput_tok_per_s",
                    "throughput_tokens_per_s",
                    "output_throughput_tok_per_sec",
                    "throughput_tok_s",
                    "output_throughput",
                    "output_tokens_per_s",
                    "throughput",
                ),
            ),
            0.0,
        ) or 0.0)
        total_tps = _to_float(
            _first(
                sources,
                (
                    "total_throughput_tok_per_sec",
                    "total_token_throughput",
                    "total_tokens_per_s",
                ),
            ))
        req_per_s = _to_float(
            _first(sources, ("request_throughput", "requests_per_s")))
        ttft_p50 = _to_float(
            _first(
                sources,
                ("ttft_ms_median", "median_ttft_ms", "p50_ttft_ms", "ttft_ms"),
            ))
        itl_p50 = _to_float(
            _first(
                sources,
                ("itl_ms_median", "median_itl_ms", "p50_itl_ms", "itl_ms"),
            ))
        e2e_p50 = _to_float(
            _first(sources, ("p50_e2el_ms", "median_e2el_ms", "e2el_p50_ms")))

        notes = str(_first(sources, ("notes", ), ""))
        is_queued = bool(
            _first(sources, ("is_queued", ), False)
            or any(k in notes.lower() for k in ("queue", "artifact"))
            or "queue" in str(raw.get("concurrency_label", "")).lower())
        return cls(
            regime=regime,
            concurrency=concurrency,
            output_tps=output_tps,
            duration_s=duration_s,
            completed=completed,
            failed=failed,
            total_tps=total_tps,
            req_per_s=req_per_s,
            ttft_p50=ttft_p50,
            itl_p50=itl_p50,
            e2e_p50=e2e_p50,
            source=source,
            is_queued=is_queued,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "concurrency": self.concurrency,
            "output_tps": self.output_tps,
            "duration_s": self.duration_s,
            "completed": self.completed,
            "failed": self.failed,
            "ttft_p50_ms": self.ttft_p50,
            "itl_p50_ms": self.itl_p50,
            "source": self.source,
            "is_queued": self.is_queued,
            "notes": self.notes,
        }


@dataclass
class RawBenchmarkParseResult:
    """Result of parsing and validating a raw benchmark output file."""

    metrics: BenchmarkRunMetrics | None = None
    detected_regime: str | None = None
    detected_concurrency: int | None = None
    error: str | None = None


@dataclass
class RunDiscoveryResult:
    """Result of loading verified runs from a manifest or legacy pattern discovery."""

    baseline_runs: list[BenchmarkRunMetrics] = field(default_factory=list)
    optimized_runs: list[BenchmarkRunMetrics] = field(default_factory=list)
    summary: str = ""
    inferred: bool = False
    verification_errors: list[str] = field(default_factory=list)


def parse_markdown_tables(
    content: str,
    filename: str,
) -> list[BenchmarkRunMetrics]:
    """Parses benchmark sweep tables from markdown reports with header-aware column mapping."""
    runs: list[BenchmarkRunMetrics] = []
    current_regime = "1k/1k"
    header_map: dict[str, int] = {}
    seen_table_row = False
    non_benchmark_table = False

    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("#"):
            current_regime = infer_regime_from_filename(
                line_clean) or current_regime

        if not (line_clean.startswith("|") and line_clean.endswith("|")):
            header_map, seen_table_row, non_benchmark_table = {}, False, False
            continue

        cells = [c.strip() for c in line_clean.split("|")[1:-1]]
        if not cells or all(set(c) <= set("-: ")
                            for c in cells) or non_benchmark_table:
            continue

        lower_cells = [c.lower() for c in cells]
        if any(k in c for c in lower_cells
               for k in ("throughput", "tok/s", "tps")) and any(
                   "concurrency" in c or c == "conc" for c in lower_cells):
            seen_table_row, header_map = True, {}
            for idx, c in enumerate(lower_cells):
                is_tps = any(k in c for k in ("throughput", "tok/s", "tps"))
                if "concurrency" in c or c == "conc":
                    header_map["concurrency"] = idx
                elif is_tps and ("baseline" in c or "base" in c):
                    header_map["baseline_throughput"] = idx
                elif is_tps and ("optimized" in c or "opt" in c):
                    header_map["optimized_throughput"] = idx
                elif is_tps and not any(x in c for x in (
                        "chip",
                        "header",
                        "spread",
                        "gain",
                        "total",
                        "user",
                        "run-1",
                        "run1",
                )):
                    if "throughput" not in header_map or "throughput" in c:
                        header_map["throughput"] = idx
                elif "ttft" in c:
                    header_map["ttft"] = idx
                elif "itl" in c:
                    header_map["itl"] = idx
                elif any(k in c for k in ("combination", "isl/osl", "regime")):
                    header_map["regime"] = idx
                elif c in ("config", "stage", "variant"):
                    header_map["config"] = idx
                elif "failed" in c and "ok" not in c:
                    header_map["failed"] = idx
                elif "completed" in c:
                    header_map["completed"] = idx
            continue

        clean_cells = [
            re.sub(r"\*\*([^*]+)\*\*", r"\1", c).replace(",", "").strip()
            for c in cells
        ]
        if not seen_table_row:
            seen_table_row = True
            if not any(
                    cell.isdigit() and int(cell) in _VALID_TABLE_CONCURRENCIES
                    for cell in clean_cells):
                non_benchmark_table = True
                continue

        if "regime" in header_map and clean_cells[header_map["regime"]]:
            current_regime = (infer_regime_from_filename(
                clean_cells[header_map["regime"]]) or current_regime)

        if "concurrency" in header_map:
            c = _to_int(clean_cells[header_map["concurrency"]])
            if c is None:
                continue
            if ("baseline_throughput" in header_map
                    and "optimized_throughput" in header_map):
                for col_key, label in (
                    ("baseline_throughput", "Baseline"),
                    ("optimized_throughput", "Optimized"),
                ):
                    tps = _to_float(clean_cells[header_map[col_key]])
                    if tps is not None:
                        runs.append(
                            BenchmarkRunMetrics(
                                regime=current_regime,
                                concurrency=c,
                                output_tps=tps,
                                source=f"{filename}:{label.lower()}",
                                notes=f"Markdown Table Row ({label} Column)",
                            ))
                continue
            if "throughput" in header_map:
                tps = _to_float(clean_cells[header_map["throughput"]])
                if tps is not None:
                    ttft = (_to_float(clean_cells[header_map["ttft"]])
                            if "ttft" in header_map else None)
                    itl = (_to_float(clean_cells[header_map["itl"]])
                           if "itl" in header_map else None)
                    failed = (_to_int(clean_cells[header_map["failed"]], 0)
                              or 0 if "failed" in header_map else 0)
                    if (not failed and "completed" in header_map
                            and "/" in clean_cells[header_map["completed"]]):
                        comp_parts = clean_cells[
                            header_map["completed"]].split("/")
                        failed = max(
                            0,
                            (_to_int(comp_parts[1], 0) or 0) -
                            (_to_int(comp_parts[0], 0) or 0),
                        )
                    row_src = filename
                    if "config" in header_map:
                        cfg = clean_cells[header_map["config"]].lower()
                        if any(k in cfg
                               for k in ("baseline", "stock", "unmodified")):
                            row_src = f"{filename}:baseline"
                        elif any(k in cfg for k in ("optimized", "tuned",
                                                    "final", "patch")):
                            row_src = f"{filename}:optimized"
                    runs.append(
                        BenchmarkRunMetrics(
                            regime=current_regime,
                            concurrency=c,
                            output_tps=tps,
                            failed=failed,
                            ttft_p50=ttft,
                            itl_p50=itl,
                            source=row_src,
                            notes="Markdown Table Row",
                        ))
                continue

        # Fallback for genuine headerless tables
        for idx, cell in enumerate(clean_cells):
            if cell.isdigit() and int(cell) in (_VALID_TABLE_CONCURRENCIES -
                                                {2048}):
                floats = [
                    v for s in clean_cells[idx + 1:]
                    if (v := _to_float(s)) is not None
                ]
                if floats:
                    runs.append(
                        BenchmarkRunMetrics(
                            regime=current_regime,
                            concurrency=int(cell),
                            output_tps=max(floats),
                            source=filename,
                            notes="Markdown Table Row (heuristic)",
                        ))
                break
    return runs


def load_runs_from_path(
    path: Path,
    default_regime: str = "1k/1k",
    tag: str = "",
) -> list[BenchmarkRunMetrics]:
    """Recursively loads benchmark runs from files (JSON, Markdown) or directories."""
    if not path.exists():
        return []

    if path.is_dir():
        skip_dirs = {
            "profile",
            "xprof",
            "tpu-diagnostics",
            "discarded",
        }
        json_runs: list[BenchmarkRunMetrics] = []
        md_runs: list[BenchmarkRunMetrics] = []
        for subfile in sorted(path.rglob("*")):
            if (subfile.is_file() and subfile.suffix in (".json", ".md")
                    and "backup" not in subfile.name.lower()
                    and "discarded" not in str(subfile).lower()):
                rel_parts = {
                    p.lower()
                    for p in subfile.relative_to(path).parts[:-1]
                }
                if not (rel_parts & skip_dirs):
                    loaded = load_runs_from_path(subfile,
                                                 default_regime=default_regime,
                                                 tag=tag)
                    (json_runs
                     if subfile.suffix == ".json" else md_runs).extend(loaded)
        return json_runs or md_runs

    inferred_regime = (infer_regime_from_filename(path.name)
                       or infer_regime_from_filename(str(path))
                       or default_regime)
    source_desc = path.name if not tag else f"{path.name} ({tag})"

    if path.suffix == ".md":
        try:
            parsed = parse_markdown_tables(
                path.read_text(encoding="utf-8", errors="ignore"), source_desc)
            if tag == "baseline":
                return [r for r in parsed if ":optimized" not in r.source]
            if tag == "optimized":
                return [r for r in parsed if ":baseline" not in r.source]
            return parsed
        except Exception:
            return []

    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8",
                                             errors="ignore"))
        except Exception:
            return []

        runs: list[BenchmarkRunMetrics] = []
        if isinstance(data, dict) and isinstance(data.get("comparison"), dict):
            for reg_key, comp_val in data["comparison"].items():
                if not isinstance(comp_val, dict):
                    continue
                reg = normalize_regime_name(reg_key)
                arms = ("baseline", "optimized") if not tag else (tag, )
                for arm in arms:
                    if arm in comp_val and isinstance(comp_val[arm], dict):
                        d = comp_val[arm]
                        runs.append(
                            BenchmarkRunMetrics(
                                regime=reg,
                                concurrency=d.get("concurrency", 1),
                                output_tps=(d.get("peak_throughput_tok_s")
                                            or d.get("throughput", 0.0)),
                                ttft_p50=d.get("ttft_ms"),
                                itl_p50=d.get("itl_ms"),
                                source=f"{source_desc}:{arm}",
                            ))
            return runs

        if isinstance(data, list):
            return [
                r for item in data if isinstance(item, dict) and (
                    r := BenchmarkRunMetrics.from_dict(
                        item, inferred_regime, source_desc)).output_tps > 0.0
            ]

        if isinstance(data, dict):
            if isinstance(data.get("json_files"), dict):
                return [
                    r for fname, item in data["json_files"].items()
                    if isinstance(item, dict) and
                    (r := BenchmarkRunMetrics.from_dict(
                        item,
                        infer_regime_from_filename(fname) or inferred_regime,
                        f"{source_desc}:{fname}",
                    )).output_tps > 0.0
                ]
            for k, val in data.items():
                norm_k = normalize_regime_name(k)
                if norm_k in VALID_REGIMES and isinstance(val, list):
                    runs.extend(
                        r for item in val if isinstance(item, dict) and (
                            r := BenchmarkRunMetrics.from_dict(
                                item, norm_k, source_desc)).output_tps > 0.0)
            if runs:
                return runs
            r = BenchmarkRunMetrics.from_dict(data, inferred_regime,
                                              source_desc)
            return [r] if r.output_tps > 0.0 else []

    return []


def auto_discover_runs(
    root: Path,
) -> tuple[list[BenchmarkRunMetrics], list[BenchmarkRunMetrics], str]:
    """Automatically locates baseline and optimized runs within a trial or artifact directory."""
    all_paths = [
        p for p in root.rglob("*")
        if ".git" not in p.parts and "tests" not in p.parts
    ]
    all_dirs = [p for p in all_paths if p.is_dir()]
    all_files = [p for p in all_paths if p.is_file()]

    def _find_dirs(*names: str) -> list[Path]:
        return [d for d in all_dirs if d.name in names]

    def _find_files(*patterns: str) -> list[Path]:
        return [
            f for f in all_files if any(
                fnmatch.fnmatch(f.name, pat) for pat in patterns)
        ]

    # Priority 1: raw_benchmarks/ or benchmarks/ subdirectories
    raw_bench_dirs = _find_dirs("raw_benchmarks", "benchmarks")
    for b_dir in raw_bench_dirs:
        if (b_dir / "baseline").exists() and (b_dir / "optimized").exists():
            base = load_runs_from_path(b_dir / "baseline", tag="baseline")
            opt = load_runs_from_path(b_dir / "optimized", tag="optimized")
            if base and opt:
                return (
                    base,
                    opt,
                    f"Found sweep directories in {b_dir.name} (baseline & optimized)",
                )

    # Priority 2: baseline_bench/ + optimized_bench/ or opt_bench/ (or serving_bench/ + serving_bench_opt/)
    base_bench_dirs = _find_dirs("baseline_bench", "serving_bench")
    if base_bench_dirs:
        base = max(
            (load_runs_from_path(d, tag="baseline") for d in base_bench_dirs),
            key=len,
        )
        opt_dirs = _find_dirs("optimized_bench", "opt_bench",
                              "serving_bench_opt")
        if (opt_dirs
                and (opt := load_runs_from_path(opt_dirs[0], tag="optimized"))
                and base):
            return (
                base,
                opt,
                "Found baseline_bench and optimized_bench directories",
            )

    # Priority 3: baseline markdown report + benchmark_optimized_*.json
    md_reports = _find_files("*baseline*profiling*report*.md",
                             "*baseline*performance*table*.md")
    opt_jsons = _find_files("benchmark_optimized_*.json")
    if md_reports and opt_jsons:
        base = [
            r for md in md_reports
            for r in load_runs_from_path(md, tag="baseline_report")
        ]
        opt = [
            r for oj in opt_jsons
            for r in load_runs_from_path(oj, tag="optimized_json")
        ]
        if base and opt:
            return (
                base,
                opt,
                f"Extracted baseline from {md_reports[0].name} and optimized from {len(opt_jsons)} benchmark JSONs",
            )

    # Priority 4 & 4b: comparison summaries (JSON or unified Markdown)
    for comp_path in _find_files("benchmark_comparison_summary.json"):
        base = load_runs_from_path(comp_path, tag="baseline")
        opt = load_runs_from_path(comp_path, tag="optimized")
        if base and opt:
            return base, opt, f"Loaded comparison from {comp_path.name}"

    comp_md_patterns = (
        "*benchmark*report*.md",
        "*comparison*report*.md",
        "*performance*comparison*.md",
        "*serving*throughput*report*.md",
        "*results*table*.md",
        "*result*table*.md",
        "*benchmark*table*.md",
    )
    for cmd in _find_files(*comp_md_patterns):
        parsed = load_runs_from_path(cmd)
        base = [r for r in parsed if ":baseline" in r.source]
        opt = [r for r in parsed if ":optimized" in r.source]
        if base and opt:
            return (
                base,
                opt,
                f"Loaded comparison from unified markdown report: {cmd.name}",
            )

    # Priority 5: Separate baseline and optimized performance tables
    base_tables = _find_files("*baseline*performance*table*.md")
    opt_tables = _find_files("*optimized*performance*table*.md")
    if base_tables and opt_tables:
        base = load_runs_from_path(base_tables[0], tag="baseline_table")
        opt = load_runs_from_path(opt_tables[0], tag="optimized_table")
        if base and opt:
            return (
                base,
                opt,
                f"Loaded from markdown tables: {base_tables[0].name} and {opt_tables[0].name}",
            )

    # Priority 5b: Stage directories (e.g. 02_baseline vs 06_optimized)
    cand_base = [
        d for d in all_dirs
        if any(k in d.name.lower()
               for k in ("baseline", "02_base", "01_base", "base_bench"))
    ]
    cand_opt = [
        d for d in all_dirs if any(
            k in d.name.lower()
            for k in ("optimized", "opt_bench", "06_opt", "05_opt", "04_opt"))
    ]
    if cand_base and cand_opt:
        b_runs, b_name = max(
            ((load_runs_from_path(d, tag="baseline"), d.name)
             for d in cand_base),
            key=lambda x: len(x[0]),
        )
        o_runs, o_name = max(
            ((load_runs_from_path(d, tag="optimized"), d.name)
             for d in cand_opt),
            key=lambda x: len(x[0]),
        )
        if b_runs and o_runs:
            return (
                b_runs,
                o_runs,
                f"Discovered baseline ({b_name}: {len(b_runs)} runs) and optimized ({o_name}: {len(o_runs)} runs) stage directories",
            )

    # Priority 6: Campaign directory structure
    for c_dir in _find_dirs("campaign"):
        base_stages = [
            d for d in c_dir.iterdir()
            if d.is_dir() and "baseline" in d.name.lower()
        ]
        if base_stages:
            base = load_runs_from_path(base_stages[0], tag="baseline")
            opt = [
                r for d in sorted(c_dir.iterdir())
                if d.is_dir() and d != base_stages[0]
                for r in load_runs_from_path(d, tag=d.name)
            ]
            if base and opt:
                return (
                    base,
                    opt,
                    f"Discovered campaign: baseline={base_stages[0].name} with multiple iteration stages",
                )

    # Priority 7: Baseline exists, no optimized runs produced
    if base_bench_dirs and (base := load_runs_from_path(base_bench_dirs[0],
                                                        tag="baseline")):
        return (
            base,
            [],
            f"Discovered baseline in {base_bench_dirs[0].name}, but no optimized benchmark runs were produced",
        )
    for b_dir in raw_bench_dirs:
        if (b_dir / "baseline").exists() and (base := load_runs_from_path(
                b_dir / "baseline", tag="baseline")):
            return (
                base,
                [],
                f"Discovered baseline in {b_dir.name}/baseline, but no optimized benchmark runs were produced",
            )

    # Priority 8: Flat benchmark JSON files
    skip_names = {
        "trajectory", "reward", "config", "lock", "manifest", "failed",
        "profile"
    }
    direct_jsons = [
        f for f in all_files if f.suffix == ".json" and f.name not in (
            "result.json",
            BENCHMARK_MANIFEST_FILENAME) and not any(k in f.name
                                                     for k in skip_names)
    ]
    base_runs: list[BenchmarkRunMetrics] = []
    opt_runs: list[BenchmarkRunMetrics] = []
    for jf in direct_jsons:
        p_low = str(jf.relative_to(root)).lower()
        if any(k in p_low for k in ("optimized", "06_opt", "05_opt",
                                    "opt_bench", "/opt/", "opt_")):
            opt_runs.extend(load_runs_from_path(jf, tag="optimized"))
        elif any(k in p_low for k in (
                "baseline",
                "02_base",
                "01_base",
                "base_bench",
                "/base/",
                "base_",
        )):
            base_runs.extend(load_runs_from_path(jf, tag="baseline"))
        else:
            for r in load_runs_from_path(jf, tag=jf.stem):
                (opt_runs
                 if "opt" in r.source.lower() else base_runs).append(r)

    if base_runs:
        if opt_runs:
            return (
                base_runs,
                opt_runs,
                f"Discovered {len(base_runs)} baseline and {len(opt_runs)} optimized runs from {len(direct_jsons)} JSON files by path structure",
            )
        return (
            base_runs,
            [],
            f"Discovered {len(base_runs)} baseline runs from {len(direct_jsons)} JSON files, but no optimized benchmark runs",
        )

    return [], [], "No structured benchmark runs discovered"


def find_manifest_file(root: Path) -> Path | None:
    """Locates benchmark_results.json, preferring specialist artifact subdirectories over top-level fallback paths."""
    search_dirs = [
        root / "artifacts",
        root / "logs" / "artifacts",
        root,
    ]
    if root.name == "workspace":
        search_dirs.append(root.parent / "artifacts")
    if str(root) == "/workspace":
        search_dirs.insert(0, Path("/logs/artifacts"))
    # Prefer specialist artifact subdirectories (<artifacts>/<specialist>/benchmark_results.json)
    for d in search_dirs:
        if d.is_dir():
            for sub_manifest in sorted(
                    d.glob(f"*/{BENCHMARK_MANIFEST_FILENAME}")):
                if sub_manifest.is_file():
                    return sub_manifest
    # Fall back to top-level /logs/artifacts or workspace root only when no specialist directory has it
    for d in search_dirs:
        candidate = d / BENCHMARK_MANIFEST_FILENAME
        if candidate.is_file():
            return candidate
    return None


def resolve_raw_output_path(
    raw_path_str: str,
    manifest_path: Path,
    workspace: Path,
) -> Path:
    """Resolves a raw_output_path from benchmark_results.json across container and host layouts."""
    cleaned = str(raw_path_str).strip()
    manifest_dir = manifest_path.parent
    if cleaned.startswith("/logs/artifacts/"):
        rel = cleaned[len("/logs/artifacts/"):]
        candidates = [
            manifest_dir / rel,
            workspace / "artifacts" / rel,
            workspace / rel,
            Path(cleaned),
        ]
    elif cleaned.startswith("/workspace/"):
        rel = cleaned[len("/workspace/"):]
        candidates = [workspace / rel, manifest_dir / rel, Path(cleaned)]
    else:
        candidates = [
            manifest_dir / cleaned,
            workspace / cleaned,
            workspace / "artifacts" / cleaned,
            Path(cleaned),
        ]
    return next((c for c in candidates if c.is_file()), candidates[0])


def parse_raw_benchmark_file(
    raw_path: Path,
    expected_regime: str = "1k/1k",
    expected_concurrency: int = 1,
) -> RawBenchmarkParseResult:
    """Parses a raw benchmark output file (.json or plain-text log) and extracts metadata for verification."""
    if not raw_path.is_file():
        return RawBenchmarkParseResult(
            error=f"Raw output file not found: {raw_path}")
    if raw_path.suffix.lower() == ".md":
        return RawBenchmarkParseResult(
            error=f"Raw output cannot be a markdown file: {raw_path.name}")

    if raw_path.suffix.lower() == ".json":
        try:
            data = json.loads(
                raw_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            return RawBenchmarkParseResult(
                error=
                f"Failed to parse JSON raw output '{raw_path.name}': {exc}")

        if isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], dict):
                data = data[0]
            else:
                return RawBenchmarkParseResult(error=(
                    f"Raw output '{raw_path.name}' must be a single benchmark run object, "
                    "not a multi-run list"))
        if not isinstance(data, dict):
            return RawBenchmarkParseResult(
                error=f"Raw output '{raw_path.name}' is not a JSON object")
        if "comparison" in data or ("baseline" in data
                                    and "optimized" in data):
            return RawBenchmarkParseResult(error=(
                f"Raw output '{raw_path.name}' is a summary/comparison file, "
                "not a raw benchmark output"))

        s = data.get("summary") if isinstance(data.get("summary"),
                                              dict) else data
        sources = (s, data)

        raw_reg = _first(sources, ("regime", ))
        detected_regime = (
            normalize_regime_name(str(raw_reg)) if raw_reg
            and normalize_regime_name(str(raw_reg)) in VALID_REGIMES else None)
        if not detected_regime:
            detected_regime = infer_regime_from_isl_osl(
                _first(sources, ("isl", "random_input_len", "input_len")),
                _first(sources, ("osl", "random_output_len", "output_len")),
            )

        completed_raw = _first(
            sources, ("num_successful", "completed_requests", "completed"))
        if completed_raw is not None:
            comp_count = _to_int(completed_raw, 0) or 0
            if comp_count <= 0:
                return RawBenchmarkParseResult(
                    detected_regime=detected_regime,
                    error=
                    f"Raw output '{raw_path.name}' recorded 0 completed requests",
                )
            if not detected_regime:
                tot_in = _first(sources, ("total_input", "total_input_tokens"))
                tot_out = _first(sources,
                                 ("total_output", "total_output_tokens"))
                if tot_in is not None and tot_out is not None:
                    detected_regime = infer_regime_from_isl_osl(
                        float(tot_in) / comp_count,
                        float(tot_out) / comp_count)

        detected_regime = detected_regime or infer_regime_from_filename(
            raw_path.name)
        detected_conc = _to_int(
            _first(
                sources,
                ("concurrency", "max_concurrency", "num_concurrent_requests"),
            ))
        if detected_conc is None:
            detected_conc = infer_concurrency_from_filename(raw_path.name)

        metrics = BenchmarkRunMetrics.from_dict(
            data,
            default_regime=detected_regime or expected_regime,
            source=raw_path.name,
        )
        if detected_conc is None:
            metrics.concurrency = expected_concurrency
        if metrics.failed > 0:
            return RawBenchmarkParseResult(
                metrics=metrics,
                detected_regime=detected_regime,
                detected_concurrency=detected_conc,
                error=
                f"Raw output '{raw_path.name}' had {metrics.failed} failed requests",
            )
        if metrics.output_tps <= 0.0:
            return RawBenchmarkParseResult(
                metrics=metrics,
                detected_regime=detected_regime,
                detected_concurrency=detected_conc,
                error=(
                    f"Raw output '{raw_path.name}' has non-positive throughput "
                    f"({metrics.output_tps})"),
            )
        return RawBenchmarkParseResult(
            metrics=metrics,
            detected_regime=detected_regime,
            detected_concurrency=detected_conc,
        )

    # Plain-text benchmark output log (e.g. vllm bench serve stdout)
    try:
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return RawBenchmarkParseResult(
            error=f"Failed to read raw output '{raw_path.name}': {exc}")

    tps_m = re.search(
        r"Output(?:\s+token)?\s+throughput\s*\(tok/s\):\s*([0-9.]+)",
        text,
        re.IGNORECASE,
    )
    if not tps_m:
        return RawBenchmarkParseResult(
            error=
            f"Could not find output token throughput in raw log '{raw_path.name}'"
        )
    output_tps = float(tps_m.group(1))

    comp_m = re.search(r"Successful\s+requests:\s*(\d+)", text, re.IGNORECASE)
    completed = int(comp_m.group(1)) if comp_m else 0
    if comp_m and completed <= 0:
        return RawBenchmarkParseResult(
            error=f"Raw log '{raw_path.name}' recorded 0 successful requests")

    fail_m = re.search(r"Failed\s+requests:\s*(\d+)", text, re.IGNORECASE)
    failed = int(fail_m.group(1)) if fail_m else 0
    if failed > 0:
        return RawBenchmarkParseResult(
            error=f"Raw log '{raw_path.name}' recorded {failed} failed requests"
        )

    conc_m = re.search(
        r"(?:Maximum\s+request\s+concurrency|--max-concurrency[=\s]+)\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    detected_conc = (int(conc_m.group(1)) if conc_m else
                     infer_concurrency_from_filename(raw_path.name))

    isl_m = re.search(r"--random-input-len[=\s]+(\d+)", text, re.IGNORECASE)
    osl_m = re.search(r"--random-output-len[=\s]+(\d+)", text, re.IGNORECASE)
    detected_regime = (infer_regime_from_isl_osl(
        isl_m.group(1), osl_m.group(1)) if (isl_m and osl_m) else None)
    if not detected_regime and completed > 0:
        tot_in_m = re.search(r"Total\s+input\s+tokens:\s*(\d+)", text,
                             re.IGNORECASE)
        tot_out_m = re.search(r"Total\s+generated\s+tokens:\s*(\d+)", text,
                              re.IGNORECASE)
        if tot_in_m and tot_out_m:
            detected_regime = infer_regime_from_isl_osl(
                float(tot_in_m.group(1)) / completed,
                float(tot_out_m.group(1)) / completed,
            )
    detected_regime = detected_regime or infer_regime_from_filename(
        raw_path.name)

    if output_tps <= 0.0:
        return RawBenchmarkParseResult(
            detected_regime=detected_regime,
            detected_concurrency=detected_conc,
            error=
            f"Raw log '{raw_path.name}' has non-positive throughput ({output_tps})",
        )

    ttft_m = re.search(r"Median\s+TTFT\s*\(ms\):\s*([0-9.]+)", text,
                       re.IGNORECASE)
    itl_m = re.search(r"Median\s+ITL\s*\(ms\):\s*([0-9.]+)", text,
                      re.IGNORECASE)
    metrics = BenchmarkRunMetrics(
        regime=detected_regime or expected_regime,
        concurrency=detected_conc
        if detected_conc is not None else expected_concurrency,
        output_tps=output_tps,
        completed=completed,
        failed=failed,
        ttft_p50=float(ttft_m.group(1)) if ttft_m else None,
        itl_p50=float(itl_m.group(1)) if itl_m else None,
        source=raw_path.name,
    )
    return RawBenchmarkParseResult(
        metrics=metrics,
        detected_regime=detected_regime,
        detected_concurrency=detected_conc,
    )


def verify_and_load_manifest(
    manifest_path: Path,
    workspace: Path,
) -> RunDiscoveryResult:
    """Loads benchmark_results.json and cross-checks each entry against its raw output file."""
    baseline_runs: list[BenchmarkRunMetrics] = []
    optimized_runs: list[BenchmarkRunMetrics] = []
    errors: list[str] = []

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return RunDiscoveryResult(
            summary=f"Invalid manifest {manifest_path.name}",
            inferred=False,
            verification_errors=[
                f"Failed to parse manifest '{manifest_path.name}': {exc}"
            ],
        )

    staged_entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, dict):
        if isinstance(data.get("baseline"), list) or isinstance(
                data.get("optimized"), list):
            for stage in ("baseline", "optimized"):
                staged_entries.extend((stage, item)
                                      for item in data.get(stage, [])
                                      if isinstance(item, dict))
        elif isinstance(data.get("results") or data.get("runs"), list):
            staged_entries.extend(
                (str(item.get("stage", "")).lower().strip(), item)
                for item in (data.get("results") or data.get("runs") or [])
                if isinstance(item, dict))
        else:
            errors.append(
                f"Manifest '{manifest_path.name}' must contain 'baseline' and 'optimized' lists"
            )
    elif isinstance(data, list):
        staged_entries.extend(
            (str(item.get("stage", "")).lower().strip(), item) for item in data
            if isinstance(item, dict))
    else:
        errors.append(
            f"Manifest '{manifest_path.name}' must be a JSON object or list")

    for stage, entry in staged_entries:
        if stage not in ("baseline", "optimized"):
            errors.append(
                f"Manifest entry has invalid stage '{stage}': {entry}")
            continue

        raw_regime = entry.get("regime") or entry.get("isl_osl")
        if not raw_regime:
            errors.append(f"Rejected {stage} entry missing 'regime': {entry}")
            continue
        regime = normalize_regime_name(str(raw_regime))
        if regime not in VALID_REGIMES:
            errors.append(
                f"Rejected {stage} entry with unrecognized regime '{raw_regime}'"
            )
            continue

        concurrency = _to_int(entry.get("concurrency"))
        if concurrency is None or concurrency <= 0:
            errors.append(
                f"Rejected {stage} {regime} entry with invalid concurrency '{entry.get('concurrency')}'"
            )
            continue

        if entry.get("failed", 0) or str(entry.get("status",
                                                   "")).lower() == "failed":
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: marked as failed in manifest"
            )
            continue

        raw_tps = _first((entry, ),
                         ("output_tps", "tps", "throughput", "throughput_tps"))
        claimed_tps = _to_float(raw_tps)
        if claimed_tps is None or claimed_tps <= 0.0:
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: invalid output_tps '{raw_tps}'"
            )
            continue

        raw_path_str = _first(
            (entry, ), ("raw_output_path", "raw_path", "raw_output", "file"))
        if not raw_path_str or not str(raw_path_str).strip():
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: missing 'raw_output_path'"
            )
            continue

        raw_file = resolve_raw_output_path(str(raw_path_str), manifest_path,
                                           workspace)
        if raw_file.resolve() == manifest_path.resolve():
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: raw_output_path cannot point to the manifest itself"
            )
            continue

        parsed = parse_raw_benchmark_file(
            raw_file,
            expected_regime=regime,
            expected_concurrency=concurrency,
        )
        if parsed.error or parsed.metrics is None:
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: {parsed.error}")
            continue
        if parsed.detected_regime is not None and parsed.detected_regime != regime:
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: mismatched regime in raw output "
                f"'{raw_file.name}' (manifest={regime}, raw={parsed.detected_regime})"
            )
            continue
        if (parsed.detected_concurrency is not None
                and parsed.detected_concurrency != concurrency):
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: mismatched concurrency in raw output "
                f"'{raw_file.name}' (manifest=c{concurrency}, raw=c{parsed.detected_concurrency})"
            )
            continue

        raw_metrics = parsed.metrics
        measured_tps = raw_metrics.output_tps
        if abs(claimed_tps - measured_tps) > max(0.1, 1e-3 * measured_tps):
            errors.append(
                f"Rejected {stage} {regime} c={concurrency}: manifest TPS ({claimed_tps}) "
                f"does not match raw output TPS ({measured_tps}) in '{raw_file.name}'"
            )
            continue

        raw_metrics.regime = regime
        raw_metrics.concurrency = concurrency
        raw_metrics.source = f"{manifest_path.name} -> {raw_file.name}"
        (baseline_runs
         if stage == "baseline" else optimized_runs).append(raw_metrics)

    summary = (
        f"Verified {len(baseline_runs)} baseline and {len(optimized_runs)} optimized runs "
        f"from {manifest_path.name} ({len(errors)} rejected)")
    return RunDiscoveryResult(
        baseline_runs=baseline_runs,
        optimized_runs=optimized_runs,
        summary=summary,
        inferred=False,
        verification_errors=errors,
    )


def discover_or_verify_runs(root: Path) -> RunDiscoveryResult:
    """Loads verified runs from benchmark_results.json if present, else falls back to legacy discovery (marked as inferred)."""
    manifest_path = find_manifest_file(root)
    if manifest_path is not None:
        return verify_and_load_manifest(manifest_path, root)

    base_runs, opt_runs, discovery_msg = auto_discover_runs(root)
    return RunDiscoveryResult(
        baseline_runs=base_runs,
        optimized_runs=opt_runs,
        summary=f"[INFERRED] {discovery_msg}",
        inferred=True,
    )


class DeterministicSpeedupEvaluator:
    """Evaluates deterministic TPS speedup and formats candidate debugging traces."""

    def __init__(
        self,
        baseline_runs: list[BenchmarkRunMetrics],
        optimized_runs: list[BenchmarkRunMetrics],
        inferred: bool = False,
        verification_errors: list[str] | None = None,
        prefer_matched_concurrency: bool = True,
    ):
        self.baseline_runs = baseline_runs
        self.optimized_runs = optimized_runs
        self.inferred = bool(inferred)
        self.verification_errors = list(verification_errors or [])
        self.prefer_matched_concurrency = prefer_matched_concurrency
        self.failed_runs: list[BenchmarkRunMetrics] = []
        self.baseline_by_regime: dict[str, list[BenchmarkRunMetrics]] = {}
        self.optimized_by_regime: dict[str, list[BenchmarkRunMetrics]] = {}

        for runs, target_dict in (
            (baseline_runs, self.baseline_by_regime),
            (optimized_runs, self.optimized_by_regime),
        ):
            for r in runs:
                if r.failed > 0 or r.output_tps <= 0.0:
                    self.failed_runs.append(r)
                else:
                    target_dict.setdefault(r.regime, []).append(r)

    def evaluate(
        self,
        target_regimes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculates deterministic TPS speedup across regimes with full candidate pool trace."""
        evaluated_regimes = (
            [normalize_regime_name(r)
             for r in target_regimes] if target_regimes else None)
        regime_results: dict[str, Any] = {}
        all_regimes = sorted(
            set(self.baseline_by_regime) | set(self.optimized_by_regime))

        for reg in all_regimes:
            base_candidates = sorted(self.baseline_by_regime.get(reg, []),
                                     key=lambda x: x.concurrency)
            opt_candidates = sorted(self.optimized_by_regime.get(reg, []),
                                    key=lambda x: x.concurrency)
            if not base_candidates:
                continue

            peak_base = max(base_candidates, key=lambda x: x.output_tps)
            peak_opt = max(opt_candidates,
                           key=lambda x: x.output_tps,
                           default=None)
            peak_speedup = ((
                safe_div(peak_opt.output_tps, peak_base.output_tps) or 1.0)
                            if peak_opt else 1.0)

            base_map: dict[int, BenchmarkRunMetrics] = {}
            for b in base_candidates:
                if (b.concurrency not in base_map
                        or b.output_tps > base_map[b.concurrency].output_tps):
                    base_map[b.concurrency] = b

            opt_map: dict[int, BenchmarkRunMetrics] = {}
            for o in opt_candidates:
                if (o.concurrency not in opt_map
                        or o.output_tps > opt_map[o.concurrency].output_tps):
                    opt_map[o.concurrency] = o

            matched_opt = opt_map.get(peak_base.concurrency)
            matched_peak_ratio = (safe_div(matched_opt.output_tps,
                                           peak_base.output_tps)
                                  if matched_opt else None)

            if peak_opt is None:
                selected_opt, selected_speedup = None, 1.0
            elif self.prefer_matched_concurrency and matched_opt is not None:
                selected_opt, selected_speedup = matched_opt, matched_peak_ratio
            else:
                selected_opt, selected_speedup = peak_opt, peak_speedup

            candidate_table: list[dict[str, Any]] = []
            matched_ratios: list[float] = []

            for c in sorted(set(base_map) | set(opt_map)):
                b_pt, o_pt = base_map.get(c), opt_map.get(c)
                b_tps = b_pt.output_tps if b_pt else None
                o_tps = o_pt.output_tps if o_pt else None
                matched_ratio = safe_div(o_tps, b_tps)
                if matched_ratio is not None:
                    matched_ratios.append(matched_ratio)

                is_base_peak = b_pt is not None and c == peak_base.concurrency
                is_selected_peak = is_base_peak and (o_pt is not None
                                                     or peak_opt is None)

                if peak_opt is None:
                    role = ("Baseline Peak (No Opt Runs)"
                            if is_base_peak else "Candidate Point")
                elif is_selected_peak:
                    role = "SELECTED (Matched @ Baseline Peak)"
                elif is_base_peak:
                    role = "Baseline Peak"
                elif o_pt and c == peak_opt.concurrency:
                    role = "Optimized Peak"
                else:
                    role = "Candidate Point"

                candidate_table.append({
                    "concurrency":
                    c,
                    "baseline_tps":
                    b_tps,
                    "baseline_source":
                    b_pt.source if b_pt else None,
                    "optimized_tps":
                    o_tps,
                    "optimized_source":
                    o_pt.source if o_pt else None,
                    "matched_speedup_ratio":
                    matched_ratio,
                    "percentage_gain":
                    (round((matched_ratio - 1.0) *
                           100, 2) if matched_ratio else None),
                    "is_selected_peak":
                    is_selected_peak,
                    "candidate_role":
                    role,
                    "baseline_latency_ms": {
                        "ttft": b_pt.ttft_p50 if b_pt else None,
                        "itl": b_pt.itl_p50 if b_pt else None,
                    },
                    "optimized_latency_ms": {
                        "ttft": o_pt.ttft_p50 if o_pt else None,
                        "itl": o_pt.itl_p50 if o_pt else None,
                    },
                })

            mean_tps_ratio, median_matched = None, None
            if opt_candidates:
                mean_base = sum(
                    b.output_tps
                    for b in base_candidates) / len(base_candidates)
                mean_opt = sum(o.output_tps
                               for o in opt_candidates) / len(opt_candidates)
                mean_tps_ratio = safe_div(mean_opt, mean_base)
            if matched_ratios:
                sr = sorted(matched_ratios)
                mid = len(sr) // 2
                median_matched = (sr[mid] if len(sr) % 2 != 0 else round(
                    (sr[mid - 1] + sr[mid]) / 2.0, 4))

            regime_results[reg] = {
                "regime": reg,
                "values_used": {
                    "baseline_peak_tps":
                    peak_base.output_tps,
                    "baseline_concurrency":
                    peak_base.concurrency,
                    "baseline_source":
                    peak_base.source,
                    "optimized_peak_tps":
                    (selected_opt.output_tps if selected_opt else None),
                    "optimized_concurrency":
                    (selected_opt.concurrency if selected_opt else None),
                    "optimized_source":
                    (selected_opt.source
                     if selected_opt else "No optimized benchmark measured"),
                    "speedup_ratio":
                    selected_speedup,
                    "percentage_gain":
                    (round((selected_speedup - 1.0) * 100, 2) if selected_opt
                     and selected_speedup is not None else 0.0),
                },
                "candidate_pool": candidate_table,
                "alternative_metrics": {
                    "peak_to_peak_speedup": peak_speedup,
                    "matched_at_baseline_peak_speedup": matched_peak_ratio,
                    "mean_tps_speedup": mean_tps_ratio,
                    "median_matched_concurrency_speedup": median_matched,
                },
            }

        # Select target regime result (defaults to 1k/1k)
        if evaluated_regimes:
            primary_regime = next(
                (r for r in evaluated_regimes if r in regime_results),
                evaluated_regimes[0],
            )
        else:
            primary_regime = ("1k/1k" if "1k/1k" in regime_results else next(
                iter(regime_results), "1k/1k"))

        primary_eval = regime_results.get(primary_regime)
        final_speedup = (primary_eval["values_used"]["speedup_ratio"]
                         if primary_eval else 0.0)

        active_regimes = evaluated_regimes or list(regime_results.keys())
        target_ratios = [
            regime_results[r]["values_used"]["speedup_ratio"]
            for r in active_regimes if r in regime_results
        ]
        geom_mean = (round(
            math.prod(target_ratios)**(1.0 / len(target_ratios)), 4)
                     if len(target_ratios) > 1 else None)

        return {
            "final_speedup": final_speedup,
            "speedup_inferred": self.inferred,
            "verification_errors": list(self.verification_errors),
            "failed_runs_excluded": len(self.failed_runs),
            "metric_rule": "Solely Throughput TPS (Output Tokens / Second)",
            "primary_regime": primary_regime,
            "multi_regime_geometric_mean": geom_mean,
            "regimes": regime_results,
        }


def format_debug_trace(results: dict[str, Any]) -> str:
    """Formats a transparent ASCII debug trace showing values used and candidate pool."""
    inferred = results.get("speedup_inferred", False)
    ver_str = (
        "INFERRED (legacy pattern discovery; benchmark_results.json not found)"
        if inferred else
        "VERIFIED (benchmark_results.json cross-checked against raw outputs)")
    lines: list[str] = [
        "=" * 80,
        "DETERMINISTIC TPS SPEEDUP CALCULATION & DEBUG CANDIDATE TRACE",
        "=" * 80,
        f"Metric Rule   : {results.get('metric_rule', 'Solely Throughput TPS')}",
        f"Verification  : {ver_str}",
        f"Primary Target: {results.get('primary_regime')} (Bottleneck Regime)",
        f"Raw Speedup   : {results.get('final_speedup'):.4f}x",
    ]
    if results.get("verification_errors"):
        lines.append("Rejected Rows :")
        lines.extend(f"  - {err}" for err in results["verification_errors"])
    lines.append("-" * 80)

    for reg_name, reg_data in results.get("regimes", {}).items():
        used = reg_data["values_used"]
        lines.extend([
            f"[REGIME: {reg_name}]",
            "  >>> VALUES USED IN SPEEDUP CALCULATION:",
            f"    Baseline Peak TPS  : {used['baseline_peak_tps']:,.2f} tok/s  "
            f"(at concurrency c={used['baseline_concurrency']})",
        ])
        if used.get("baseline_source"):
            lines.append(
                f"                         Source: {used['baseline_source']}")
        if used.get("optimized_peak_tps") is not None:
            lines.append(
                f"    Optimized Peak TPS : {used['optimized_peak_tps']:,.2f} tok/s  "
                f"(at concurrency c={used['optimized_concurrency']})")
            if used.get("optimized_source"):
                lines.append(
                    f"                         Source: {used['optimized_source']}"
                )
            lines.append(
                f"    Calculated Speedup : {used['speedup_ratio']:.4f}x  "
                f"({used['percentage_gain']:+.2f}%)")
        else:
            lines.extend([
                "    Optimized Peak TPS : — (No optimized server benchmarked)",
                f"    Calculated Speedup : {used['speedup_ratio']:.4f}x  "
                "(1.0000x per rule: baseline measured, no opt runs)",
            ])

        lines.extend([
            "",
            "  >>> CANDIDATE POOL (All concurrency points evaluated / values we could have used):",
            f"    {'Concurrency':<12} | {'Baseline TPS':<14} | {'Optimized TPS':<14} | "
            f"{'Matched Ratio':<14} | {'Candidate Role / Status'}",
            f"    {'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*28}",
        ])
        for pt in reg_data.get("candidate_pool", []):
            c_str = f"c={pt['concurrency']}"
            b_str = (f"{pt['baseline_tps']:,.2f}"
                     if pt["baseline_tps"] is not None else "—")
            o_str = (f"{pt['optimized_tps']:,.2f}"
                     if pt["optimized_tps"] is not None else "—")
            r_str = (
                f"{pt['matched_speedup_ratio']:.4f}x ({pt['percentage_gain']:+.1f}%)"
                if pt["matched_speedup_ratio"] is not None else "—")
            lines.append(
                f"    {c_str:<12} | {b_str:<14} | {o_str:<14} | {r_str:<14} | "
                f"{pt['candidate_role']}")

        alt = reg_data.get("alternative_metrics", {})
        lines.extend(["", "  >>> ALTERNATIVE SPEEDUP INTERPRETATIONS:"])
        m_spd = alt.get("matched_at_baseline_peak_speedup")
        p_spd = alt.get("peak_to_peak_speedup")
        if m_spd is not None:
            lines.append(
                f"    * Matched at Baseline Peak (Opt @ Base c)  : {m_spd:.4f}x  <-- [SELECTED]"
            )
            lines.append(
                f"    * Peak-to-Peak Speedup (Max TPS Ratio)     : {p_spd:.4f}x"
            )
        else:
            lines.append(
                f"    * Peak-to-Peak Speedup (Max TPS Ratio)     : {p_spd:.4f}x  <-- [SELECTED]"
            )
        if alt.get("median_matched_concurrency_speedup") is not None:
            lines.append(f"    * Median Matched-Concurrency Speedup       : "
                         f"{alt['median_matched_concurrency_speedup']:.4f}x")
        if alt.get("mean_tps_speedup") is not None:
            lines.append(f"    * Mean TPS Speedup (Mean Opt / Mean Base)  : "
                         f"{alt['mean_tps_speedup']:.4f}x")
        lines.append("-" * 80)

    if results.get("multi_regime_geometric_mean") is not None:
        lines.extend([
            f"Multi-Regime Geometric Mean Speedup: "
            f"{results['multi_regime_geometric_mean']:.4f}x",
            "=" * 80,
        ])
    return "\n".join(lines)


@criterion(description=(
    "The speedup of the optimized model over the baseline model, both served with vLLM on TPU "
    "and benchmarked with the same serving benchmark suite on the bottleneck regime: output "
    "tokens per second of the optimized server divided by peak output tokens per second of baseline."
))
def tpu_speedup(workspace: Path) -> float:
    """Calculates deterministic throughput speedup from verified manifest or legacy fallback."""
    if not workspace or not workspace.exists():
        print(f"[!] tpu_speedup: Workspace does not exist: {workspace}")
        return 0.0

    discovery = discover_or_verify_runs(workspace)
    if not discovery.baseline_runs:
        print(
            f"[!] tpu_speedup: No valid baseline benchmark runs found in workspace: "
            f"{workspace} ({discovery.summary})")
        for err in discovery.verification_errors:
            print(f"    - {err}")
        return 0.0

    evaluator = DeterministicSpeedupEvaluator(
        discovery.baseline_runs,
        discovery.optimized_runs,
        inferred=discovery.inferred,
        verification_errors=discovery.verification_errors,
    )
    res = evaluator.evaluate(target_regimes=["1k/1k"])
    res["discovery_source"] = discovery.summary
    if trace := format_debug_trace(res):
        print(trace)

    speedup = round(float(res.get("final_speedup", 1.0)), 4)
    reg_data = res.get("regimes", {}).get("1k/1k", {})
    used = reg_data.get("values_used", {})
    base_tps = used.get("baseline_peak_tps") or 0.0
    opt_tps = used.get("optimized_peak_tps")
    opt_str = f"{opt_tps:,.2f} tok/s" if opt_tps is not None else "(no opt runs)"

    print(
        f"tpu_speedup: calculated raw ratio = {speedup:.4f}x "
        f"(base={base_tps:,.2f} tok/s, opt={opt_str}, inferred={discovery.inferred})"
    )
    return speedup


if __name__ == "__main__":
    import sys

    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"Final Speedup: {tpu_speedup(target_path):.4f}x")
