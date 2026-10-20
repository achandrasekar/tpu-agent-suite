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
"""How well the ranking finds the row a caller meant.

`rank.py` carries numbers that were measured, not chosen: the page weight
and cap, the question-word filter, the synonym groups, the length below
which a word is not respelt. This is where they were measured. Every one
of them is worth keeping only while it moves these figures, and a change
to the ranking is judged by whether they move, not by whether one
hand-picked query improved.

A query row in `search_eval.jsonl` is `{"task", "kind", "query", "want"}`.
`kind` is `term` for something an agent would pass as a keyword and
`question` for a caller's own words. `want` lists every acceptable row id;
the first one found counts.

    python3 tests/tpu-knowledge-base/search_eval.py
    python3 tests/tpu-knowledge-base/search_eval.py --by kind
    python3 tests/tpu-knowledge-base/search_eval.py --misses
    python3 tests/tpu-knowledge-base/search_eval.py --min-mrr 0.55

Run from `package`.
"""

import argparse
import collections
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts" / "tpu-knowledge-base"
QUERIES = HERE / "search_eval.jsonl"
# The same questions, mistyped. Respelling is judged here and
# nowhere else: the clean set has no typos to fix, so it scores
# the same with the feature on or off.
TYPOS = HERE / "search_typos.jsonl"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load("catalog")
rank = _load("rank")


def queries(known: set[str], path: pathlib.Path = QUERIES) -> list[dict]:
    """The query set, less any row that has since left the tables.

    A query whose every acceptable answer is gone is unanswerable, and
    counting it as a miss would blame the ranking for a data change.
    """
    found = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if set(row["want"]) & known:
                found.append(row)
    return found


def place(table: list[dict], query: dict, limit: int) -> int | None:
    """1-based position of the first acceptable row, or None."""
    want = set(query["want"])
    for position, (_, row, _) in enumerate(
            rank.search(table, query["query"], catalog.page_of, limit,
                        catalog.page_scored), 1):
        if row.get("id") in want:
            return position
    return None


def score(places: list[int | None]) -> dict:
    n = max(len(places), 1)
    return {
        "n": len(places),
        "hit@1": sum(1 for p in places if p == 1) / n,
        "hit@3": sum(1 for p in places if p and p <= 3) / n,
        "hit@5": sum(1 for p in places if p and p <= 5) / n,
        "mrr": sum(1 / p for p in places if p) / n,
    }


def table(groups: dict[str, list[int | None]]) -> str:
    lines = [
        f"{'':<22} {'n':>4} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'mrr':>6}"
    ]
    for name, places in sorted(groups.items()):
        s = score(places)
        lines.append(f"{name:<22} {s['n']:>4} {s['hit@1']:>6.2f} "
                     f"{s['hit@3']:>6.2f} {s['hit@5']:>6.2f} {s['mrr']:>6.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--by", choices=("kind", "task"), help="break down by")
    parser.add_argument("--typos",
                        action="store_true",
                        help="score the mistyped set instead")
    parser.add_argument("--misses",
                        action="store_true",
                        help="list the queries that found nothing")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-mrr",
                        type=float,
                        help="exit 1 below this, for CI")
    args = parser.parse_args(argv)

    rows = catalog.rows()
    asked = queries({r["id"] for r in rows}, TYPOS if args.typos else QUERIES)
    if not asked:
        print("no queries left to score", file=sys.stderr)
        return 1
    places = [place(rows, q, args.limit) for q in asked]

    print(table({"all": places}))
    if args.by:
        groups = collections.defaultdict(list)
        for query, found in zip(asked, places):
            groups[query.get(args.by, "?")].append(found)
        print()
        print(table(dict(groups)))
    if args.misses:
        print()
        for query, found in zip(asked, places):
            if found is None:
                print(f"  miss: {query['query']!r} wanted {query['want']}")

    mrr = score(places)["mrr"]
    if args.min_mrr is not None and mrr < args.min_mrr:
        print(f"\nMRR {mrr:.3f} is below {args.min_mrr}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
