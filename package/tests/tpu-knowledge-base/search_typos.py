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
"""A misspelt copy of a query set, for judging typo tolerance.

Few queries in `search_eval.jsonl` are misspelt, so a typo change could not be
judged on it. This writes one typo into every query, keeping its label: the
longest word of five letters or more loses a letter, swaps two, has one hit a
neighbouring key or doubles one, the four taken in turn. Made by rule rather
than by hand so the typos are not the ones a hand already knows search copes
with, and seeded so the set is the same on every run.

    python3 tests/tpu-knowledge-base/search_typos.py > tests/tpu-knowledge-base/search_typos.jsonl

The source defaults to `search_eval.jsonl`; name another to copy it instead.

A held-out set gets a copy the same way, held out like its source.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

# Each letter's neighbours on a QWERTY keyboard, same row.
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEAR = {
    c: row[max(i - 1, 0):i] + row[i + 1:i + 2]
    for row in _ROWS
    for i, c in enumerate(row)
}


def misspell(word: str, how: int, rng: random.Random) -> str:
    """`word` with one typo of kind `how`, never at its first letter."""
    i = rng.randrange(1, len(word) - 1)
    if how == 0:
        return word[:i] + word[i + 1:]
    if how == 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if how == 2:
        c = word[i].lower()
        return word[:i] + rng.choice(_NEAR.get(c, c)) + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path",
                        nargs="?",
                        default=pathlib.Path(__file__).resolve().parent /
                        "search_eval.jsonl",
                        help="the query set to misspell")
    path = parser.parse_args(argv).path
    rng = random.Random(0)
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for n, row in enumerate(rows):
        words = [m for m in re.finditer(r"[A-Za-z]{5,}", row["query"])]
        if not words:
            continue
        word = max(words, key=lambda m: len(m.group()))
        typo = misspell(word.group(), n % 4, rng)
        query = row["query"][:word.start()] + typo + row["query"][word.end():]
        print(
            json.dumps({
                **row, "kind": f"{row['kind']} typo",
                "query": query
            }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
