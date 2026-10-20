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
"""What the caller might have meant, when no keyword is what they typed."""

import importlib.util
import pathlib
import unittest

_SCRIPTS = (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
            "tpu-knowledge-base")


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load("catalog")
rank = _load("rank")


class StemTest(unittest.TestCase):

    def test_a_word_and_its_variants_share_a_stem(self):
        for group in (("compile", "compiled", "compilation"),
                      ("profile", "profiling", "profiler"), ("kernel",
                                                             "kernels")):
            with self.subTest(group=group):
                self.assertEqual(len({rank.stem(w) for w in group}), 1, group)

    def test_a_stem_is_never_cut_below_three_letters(self):
        """Otherwise everything short collapses onto everything else."""
        for word in ("es", "ing", "led", "mesh", "heap"):
            with self.subTest(word=word):
                self.assertEqual(rank.stem(word), word)


class QueryTermsTest(unittest.TestCase):

    def test_question_words_are_dropped(self):
        self.assertNotIn("why", rank.query_terms("why is my matmul slow"))
        self.assertIn("matmul", rank.query_terms("why is my matmul slow"))

    def test_a_query_of_only_question_words_keeps_them(self):
        """Dropped only while something is left, or there is nothing to score."""
        self.assertTrue(rank.query_terms("why is it"))

    def test_a_synonym_comes_in_once(self):
        terms = rank.query_terms("gemm matmul")
        self.assertEqual(terms.count(rank.stem("matmul")), 1)


class RespeltTest(unittest.TestCase):

    USED = {"blocking": 3, "mesh": 2, "conv1d": 1, "v5e": 1}

    def test_a_typo_lands_on_the_commonest_spelling(self):
        fixed, swaps = rank.respelt(self.USED, "blocknig")
        self.assertEqual(fixed, "blocking")
        self.assertEqual(swaps, [("blocknig", "blocking")])

    def test_a_word_with_a_digit_is_left_alone(self):
        """One edit from `conv3d` is `conv1d`, real and wrong."""
        self.assertEqual(rank.respelt(self.USED, "conv3d"), ("conv3d", []))

    def test_a_short_word_is_left_alone(self):
        self.assertEqual(rank.respelt(self.USED, "heap"), ("heap", []))

    def test_an_underscore_does_not_block_the_fix(self):
        """`_` is a word character to \\b, so a \\b-bounded sub would miss."""
        fixed, _ = rank.respelt(self.USED, "flash_blocknig")
        self.assertIn("blocking", fixed)


class SearchTest(unittest.TestCase):

    def setUp(self):
        self.rows = catalog.rows()

    def find(self, query):
        return rank.search(self.rows, query, catalog.page_of, limit=5)

    def test_free_text_reaches_the_page_it_is_about(self):
        """None of these names a keyword; all of them name a subject.

        Membership of the five, not first place: five is what a caller is
        handed, and most questions have several fair answers -- the eval
        set labels "how do I capture a TPU profile" with four.
        """
        for query, entry in (("why is my flash attn slow",
                              "flash-attention-to-sdpa"),
                             ("how do I profile a tpu program",
                              "playbook-profile-tpu-program"),
                             ("avoid recompiling every step",
                              "torch-tpu-avoiding-recompilation")):
            with self.subTest(query=query):
                self.assertIn(entry,
                              [row["id"] for _, row, _ in self.find(query)])

    def test_nonsense_scores_nothing(self):
        self.assertEqual(self.find("asdfgh qwerty"), [])

    def test_the_ranking_is_stable(self):
        """A ranked list that reshuffles between runs is not a ranking."""
        first = [(s, e["id"]) for s, e, _ in self.find("tpu memory profile")]
        self.assertEqual(first,
                         [(s, e["id"])
                          for s, e, _ in self.find("tpu memory profile")])

    def test_the_keyword_returned_is_one_the_row_lists(self):
        for score, row, keyword in self.find("flash attention"):
            with self.subTest(entry=row["id"]):
                self.assertIn(keyword, row.get("keywords") or [])
