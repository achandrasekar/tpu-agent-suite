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
"""Every tpu-knowledge-base case, scored without a model or a cluster.

The smoke check runs two tasks through Harbor and a judge, which gates
that the package builds; it cannot tell whether a key still reaches its
page. That is deterministic, so it is a test: each dataset's `EXPECT`
goes through the real catalog and the real checker, and a key that stops
resolving -- or a page that stops saying its figure -- fails here, in
seconds, rather than in a trial.

Offline. A url source is handed back as a url and cited, never fetched,
so nothing here depends on the network or on a harness CLI.
"""

import contextlib
import importlib.util
import io
import json
import pathlib
import re
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
DATASETS = REPO / "eval" / "datasets" / "tpu-knowledge-base"
LOOKUP_PY = REPO / "package" / "scripts" / "tpu-knowledge-base" / "kb_lookup.py"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expectations() -> list[tuple[str, dict]]:
    """Each case's name and the EXPECT its checker carries."""
    found = []
    for checker in sorted(DATASETS.rglob("tests/*/check_answer.py")):
        text = checker.read_text("utf-8")
        start = text.index("EXPECT: dict =") + len("EXPECT: dict =")
        decoder = json.JSONDecoder()
        expect, _ = decoder.raw_decode(text[start:].lstrip())
        found.append((checker.parents[2].name, expect))
    return found


class DatasetExpectationsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lookup = _load("kb_lookup", LOOKUP_PY)
        # The module, not its function: a function on the class would bind
        # as a method and take `self` as its first argument.
        cls.checker = _load("check_answer",
                            DATASETS / "_shared" / "check_answer.py")
        cls.entries = cls.lookup.catalog.rows()

    def test_every_case_still_reaches_what_it_expects(self):
        """The catalog half of each case, which needs no model.

        The shared checker scores what a reply wrote; it cannot run here,
        because there is no reply. What is deterministic is the lookup:
        the case's keys must resolve, and the url its expectation names
        must be among what the lookup cites. A keyword that stops
        resolving fails here in under a second rather than in a trial.
        """
        cases = _expectations()
        self.assertGreater(len(cases), 20, "the datasets went missing")
        scored = 0
        for name, expect in cases:
            keys = expect.get("keys")
            if not keys:
                continue
            with self.subTest(case=name):
                with mock.patch.object(self.lookup.summarize,
                                       "build",
                                       side_effect=ValueError("offline")), \
                     mock.patch.object(self.lookup.summarize,
                                       "download",
                                       side_effect=OSError("offline")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = self.lookup.route(self.entries, keys, None)
                # Offline, every url is left to read, which is `partial`:
                # still served, just not summarised yet.
                self.assertIn(result["status"], ("cached", "partial"),
                              f"{name}: {keys} served nothing")
                cited = " ".join(r["url"] for r in result["references"])
                for wanted in expect.get("references") or []:
                    self.assertIn(wanted, cited, f"{name}: not cited")
            scored += 1
        self.assertGreater(scored, 20, "nothing was scored")

    def test_csat006_fails_an_answer_written_without_the_lookup(self):
        """The one gating knowledge case must not pass on a sourceless reply.

        With delegation broken, the router answered csat006 itself, citing
        the PagedAttention paper and general docs, and the smoke check
        stayed green. An answer from tpu-knowledge-base cites the source
        the engine-args page names.
        """
        smoke = _load(
            "check_question", REPO / "eval" / "datasets" / "smoke-check" /
            "knowledge-qa-csat006" / "tests" / "question_csat006_kv_block" /
            "check_question.py")
        looked_up = {
            "answer":
            "Block size moves fragmentation, not the pool.",
            "reference": [
                "https://github.com/vllm-project/vllm/blob/main/vllm/config/"
                "cache.py",
                "https://github.com/vllm-project/vllm/blob/main/docs/"
                "configuration/engine_args.md",
            ],
        }
        sourceless = {
            "answer":
            "Block size moves fragmentation, not the pool.",
            "reference":
            "1. Kwon et al., PagedAttention, SOSP '23, "
            "https://arxiv.org/abs/2309.06180",
        }
        self.assertEqual(smoke.check(looked_up, smoke.EXPECT), (True, []))
        ok, why = smoke.check(sourceless, smoke.EXPECT)
        self.assertFalse(ok)
        self.assertIn("never cites vllm/config/cache.py", why)

    def test_every_checker_is_the_shared_one(self):
        """A per-case edit to the scoring is drift.

        What a case may vary is its EXPECT, its docstring and the one line
        describing its criterion. The code that scores the record is the
        same everywhere, so a fix lands once.
        """
        # The smoke check's csat006 scores the same record, under the
        # criterion name its verifier_params and Harbor wiring already use.
        smoke = (REPO / "eval" / "datasets" / "smoke-check" /
                 "knowledge-qa-csat006" / "tests" /
                 "question_csat006_kv_block" / "check_question.py")

        def scoring(text: str) -> str:
            """The file past its docstring, with EXPECT and the criterion
            description taken out."""
            after = text.partition('"""')[2].partition('"""')[2]
            after = re.sub(r"^EXPECT.*?\n\n", "", after, flags=re.S | re.M)
            return re.sub(r"@_criterion\(description=.*?\)", "", after)

        shared = scoring(
            (DATASETS / "_shared" / "check_answer.py").read_text("utf-8"))
        for checker in [
                *sorted(DATASETS.rglob("tests/*/check_answer.py")), smoke
        ]:
            with self.subTest(case=checker.parents[2].name):
                self.assertEqual(scoring(checker.read_text("utf-8")), shared)


if __name__ == "__main__":
    unittest.main()
