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
"""The ranking still finds what it found when its constants were measured.

`search_eval.py` is where `rank.py`'s numbers were chosen. This holds the
figures it reports to a floor, so a change that costs the ranking its
place fails here rather than going unnoticed. The floors sit a few points
under what was measured (MRR 0.75, hit@5 0.89 clean; 0.74 and 0.88
mistyped), so a row added or retired does not trip them.
"""

import importlib.util
import pathlib
import unittest

_spec = importlib.util.spec_from_file_location(
    "search_eval",
    pathlib.Path(__file__).resolve().parent / "search_eval.py")
search_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_eval)

FLOORS = {
    search_eval.QUERIES: {
        "mrr": 0.70,
        "hit@5": 0.85
    },
    search_eval.TYPOS: {
        "mrr": 0.69,
        "hit@5": 0.84
    },
}


class SearchEvalTest(unittest.TestCase):

    def test_each_query_set_scores_above_its_floor(self):
        rows = search_eval.catalog.rows()
        known = {r["id"] for r in rows}
        for path, floors in FLOORS.items():
            asked = search_eval.queries(known, path)
            places = [search_eval.place(rows, q, 5) for q in asked]
            got = search_eval.score(places)
            for metric, floor in floors.items():
                with self.subTest(queries=path.name, metric=metric):
                    self.assertGreater(len(asked), 0)
                    self.assertGreaterEqual(got[metric], floor)


if __name__ == "__main__":
    unittest.main()
