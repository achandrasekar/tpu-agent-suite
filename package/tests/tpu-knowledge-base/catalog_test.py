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
"""The walk from a key to a row.

A caller writes what it wants, not the row that answers. Folding is
spelling, not similarity: a key either normalizes onto one the table lists,
names a generation, or misses.
"""

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
           "tpu-knowledge-base" / "catalog.py")
_spec = importlib.util.spec_from_file_location("kb_catalog", _SCRIPT)
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)

TABLE = [
    {
        "id": "page-row",
        "page": "vpu.md",
        "keywords": ["vpu", "hosted-xprof"]
    },
    {
        "id": "args-row",
        "url": "https://e.test/a",
        "keywords": ["max-num-seqs"]
    },
    {
        "id": "v6e-row",
        "page": "v6e.md",
        "keywords": ["v6e"]
    },
    {
        "id": "templated",
        "url": "https://e.test/docs/{keyword}"
    },
]


class FoldingTest(unittest.TestCase):

    def test_case_spaces_underscores_and_dashes_fold(self):
        for spelling in ("hosted xprof", "Hosted_XProf", "--hosted-xprof",
                         "  hosted-xprof  "):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    catalog.find(TABLE, spelling)[0]["id"], "page-row")

    def test_a_flag_spelling_reaches_its_row(self):
        self.assertEqual(
            catalog.find(TABLE, "--max-num-seqs")[0]["id"], "args-row")

    def test_folding_is_spelling_not_similarity(self):
        # Near is not a match: a near-miss serves a real page answering a
        # different question, which reads as checked and is not.
        for near in ("hosted xprofs", "max-num-seq", "vp"):
            with self.subTest(key=near):
                self.assertIsNone(catalog.find(TABLE, near))


class GenerationTest(unittest.TestCase):

    def test_the_chip_count_is_dropped(self):
        for key in ("v6e-8", "tpu-v6e-4", "TPU-V6E-16"):
            with self.subTest(key=key):
                self.assertEqual(catalog.find(TABLE, key)[0]["id"], "v6e-row")

    def test_a_listed_generation_keeps_its_own_row(self):
        self.assertEqual(catalog.find(TABLE, "v6e")[0]["id"], "v6e-row")

    def test_a_generation_no_row_lists_falls_to_the_templated_row(self):
        # A chip released tomorrow still resolves, because that row's url is
        # built from the key rather than listing it.
        row, keyword = catalog.find(TABLE, "v9z-16")
        self.assertEqual(row["id"], "templated")
        self.assertEqual(keyword, "v9z")

    def test_a_word_that_is_not_a_generation_still_misses(self):
        self.assertIsNone(catalog.find(TABLE, "not-a-chip"))


class TableTest(unittest.TestCase):

    def test_every_shipped_row_loads(self):
        rows = catalog.rows()
        self.assertGreater(len(rows), 300)
        self.assertTrue(all(r.get("id") for r in rows))

    def test_every_id_is_a_cache_directory_of_its_own(self):
        # A page lives under `<id>/`, beside `ungrounded/`, which the sweep
        # treats as no row's.
        for row in catalog.rows():
            self.assertNotIn("/", row["id"], row["id"])
            self.assertNotEqual(row["id"], "ungrounded")

    def test_groupings_and_watchlists_are_not_rows(self):
        # subjects.jsonl and pending.jsonl are other shapes entirely.
        self.assertTrue(
            all("members" not in r and "public_at" not in r
                for r in catalog.rows()))

    def test_every_shipped_keyword_is_claimed_once(self):
        seen: dict[str, str] = {}
        for row in catalog.rows():
            for keyword in row.get("keywords") or []:
                folded = catalog.normalize(keyword)
                self.assertNotIn(
                    folded, seen,
                    f"{folded} on {row['id']} and {seen.get(folded)}")
                seen[folded] = row["id"]

    def test_every_shipped_key_reaches_its_own_row(self):
        rows = catalog.rows()
        for row in rows:
            for keyword in row.get("keywords") or []:
                found = catalog.find(rows, keyword)
                self.assertIsNotNone(found, keyword)
                self.assertEqual(found[0]["id"], row["id"], keyword)

    def test_the_only_row_without_keys_is_the_templated_one(self):
        # Everything else must be reachable by something a caller can write.
        for row in catalog.rows():
            if not (row.get("keywords") or []):
                with self.subTest(row=row["id"]):
                    self.assertIn("{keyword}", row.get("url", ""))

    def test_a_shipped_page_resolves_to_a_file_that_exists(self):
        for row in catalog.rows():
            page = catalog.page_of(row)
            if page is not None:
                with self.subTest(row=row["id"]):
                    self.assertTrue(page.is_file(), page)

    def test_a_file_row_cannot_escape_the_package(self):
        with self.assertRaisesRegex(ValueError, "escapes package"):
            catalog.page_of({"id": "x", "file": "../../../etc/passwd"})

    def test_a_page_row_cannot_escape_the_pages_dir(self):
        """Only `file` was gated once, so a `page` walked out freely."""
        with self.assertRaisesRegex(ValueError, "escapes pages"):
            catalog.page_of({"id": "x", "page": "../../../etc/passwd"})

    def test_source_kind_is_the_field_that_is_set(self):
        self.assertEqual(catalog.source_kind({"url": "u"}), "url")
        self.assertEqual(catalog.source_kind({"page": "p"}), "page")
        with self.assertRaises(KeyError):
            catalog.source_kind({"id": "nothing"})

    def test_a_header_value_drops_its_trailing_comment(self):
        text = "---\nretrieved: 2026-10-04        # when we fetched\n---\n"
        self.assertEqual(catalog.front_field(text, "retrieved"), "2026-10-04")


if __name__ == "__main__":
    unittest.main()


class AliasTest(unittest.TestCase):
    """Keys a trial actually asked for, which the table now answers to.

    Each of these cost a scored run: the lookup missed, and the agent went
    looking through the installed package by hand. They are aliases in the
    table rather than a matcher in code.
    """

    def test_the_keys_trials_guessed_resolve(self):
        entries = catalog.rows()
        for key, entry in (("vit-mnist", "torch-tpu-get-started-gke"),
                           ("mnist", "torch-tpu-get-started-gke"),
                           ("xprof-remote", "xprof-capture"),
                           ("pprof", "jax-device-memory-profiling"),
                           ("memory.prof", "jax-device-memory-profiling")):
            with self.subTest(key=key):
                found = catalog.find(entries, key)
                self.assertIsNotNone(found, f"{key} resolves to nothing")
                self.assertEqual(found[0]["id"], entry)


class NotYetPublicTest(unittest.TestCase):
    """A row into a repo that is not public yet is still a row.

    Nothing gates it. The team the package serves holds the credential,
    and `kb_summarize.github_note` already tells the harness to
    use its own GitHub access for any github url -- so there is nothing
    for the catalog to decide. `pending.jsonl` is a watchlist of repos and
    when they go public, which is why it is not an entry table.
    """

    def test_a_pending_keyword_still_resolves(self):
        entries = catalog.rows()
        found = catalog.find(entries, "kb-torch-tpu-overview")
        self.assertIsNotNone(found, "a not-yet-public row must still resolve")

    def test_the_watchlist_is_not_an_entry_table(self):
        self.assertIn("pending.jsonl", catalog.NOT_ENTRIES)
        self.assertFalse([r for r in catalog.rows() if r.get("repo")],
                         "a watchlist row leaked into the entries")


class SpellingTest(unittest.TestCase):
    """One generation, however it is written.

    `tpu-v5e`, `tpuv5e` and `v5e` are one chip, and which of them a row
    happens to list must not decide the answer. Trying only the bare form
    sent `v5e-4` past `hw-tpu-v5e`, which lists `tpu-v5e`, to the templated
    vendor url -- a page the package ships, skipped for one it fetches.
    """

    def test_every_spelling_of_a_generation_reaches_its_page(self):
        entries = catalog.rows()
        for key in ("v5e", "tpu-v5e", "tpuv5e", "v5e-4", "tpu-v5e-4",
                    "tpu-v5e-8"):
            with self.subTest(key=key):
                found = catalog.find(entries, key)
                self.assertIsNotNone(found, f"{key} resolves to nothing")
                self.assertEqual(found[0]["id"], "hw-tpu-v5e")

    def test_a_generation_no_row_lists_falls_to_the_templated_row(self):
        found = catalog.find(catalog.rows(), "v99e")
        self.assertIsNotNone(found)
        self.assertIn("{keyword}", found[0].get("url", ""))

    def test_nonsense_after_a_generation_is_not_a_generation(self):
        """`v5e-wombat` must miss rather than build a url that 404s."""
        self.assertIsNone(catalog.find(catalog.rows(), "v5e-wombat"))


class InstructionsForTest(unittest.TestCase):
    """A row's `prompt` is an id, not prose.

    Handed straight to the model it put the bare word `concept` at the end
    of the prompt -- a stray token where the record holds the instructions
    that give the summary its shape. Every url row named one.
    """

    def test_every_kind_a_row_names_is_in_the_table(self):
        known = set(catalog.prompts())
        self.assertTrue(known, "the prompt table went missing")
        named = {row["prompt"] for row in catalog.rows() if row.get("prompt")}
        self.assertTrue(named, "no row names a prompt kind")
        self.assertEqual(named - known, set(),
                         "a row names a prompt kind the table does not hold")

    def test_the_id_itself_never_reaches_the_prompt(self):
        """The bug: `concept` as a word, where instructions should be."""
        for kind in sorted(catalog.prompts()):
            with self.subTest(kind=kind):
                text = catalog.instructions_for(kind, "v6e")
                self.assertGreater(len(text), 200,
                                   f"{kind} expanded to a stub")
                self.assertNotEqual(text.strip(), kind)

    def test_the_keyword_is_substituted(self):
        text = catalog.instructions_for("tpu-spec", "v6e")
        self.assertIn("v6e", text)
        self.assertNotIn("{keyword}", text)

    def test_an_unknown_kind_adds_nothing(self):
        """A row naming no kind, or one the table lost, still summarises."""
        self.assertEqual(catalog.instructions_for(None, "v6e"), "")
        self.assertEqual(catalog.instructions_for("nonesuch", "v6e"), "")

    def test_a_broken_table_names_its_line_as_a_value_error(self):
        # A lookup resolves the kind before serving a cached page and
        # degrades on ValueError; a KeyError from a row with no id would
        # escape that and fail the whole lookup.
        for bad in ('{"id": "ok"}\nnot json\n',
                    '{"id": "ok"}\n{"lead": "x"}\n'):
            with self.subTest(bad=bad.splitlines()[1]):
                table = pathlib.Path(tempfile.mkdtemp()) / "prompt_table.jsonl"
                table.write_text(bad)
                with mock.patch.object(catalog, "PROMPT_TABLE", table):
                    with self.assertRaisesRegex(ValueError,
                                                "prompt_table.jsonl:2"):
                        catalog.prompts()
