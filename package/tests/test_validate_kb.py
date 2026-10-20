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
"""Tests for the knowledge base validator: index catalog completeness and log checks."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from package.data.knowledgebase.scripts import validate_kb

_PAGE = "---\ntitle: t\ntype: concept\n---\n\nbody\n"
_LOG = (
    "# Log\n\n---\n\n## [2026-09-10] ingest | second\n\n**Operation**: ingest\n\n"
    "## [2026-09-01] manual | first\n\n**Operation**: manual\n")


class TestValidateKb(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.kb = self.tmp / "kb"
        (self.kb / "wiki" / "concepts").mkdir(parents=True)
        (self.kb / "wiki" / "concepts" / "a.md").write_text(_PAGE)
        (self.kb / "wiki" / "concepts" / "b.md").write_text(_PAGE)
        (self.kb / "log.md").write_text(_LOG)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_index(self, entries=("a", "b")):
        lines = "".join(f"- [{e}](wiki/concepts/{e}.md) — {e}.\n"
                        for e in entries)
        (self.kb / "index.md"
         ).write_text(f"# Knowledge Base Index\n\n## Concepts\n\n{lines}")

    def test_valid_index_and_log_pass(self):
        self._write_index()
        self.assertEqual(validate_kb.validate_index(self.kb), [])
        self.assertEqual(validate_kb.validate_log(kb_dir=self.kb), [])

    def test_uncataloged_page_is_reported(self):
        self._write_index(entries=("a", ))
        errors = validate_kb.validate_index(self.kb)
        self.assertTrue(
            any("not referenced in index.md: wiki/concepts/b.md" in e
                for e in errors), errors)

    def test_broken_link_is_reported(self):
        self._write_index(entries=("a", "b", "missing"))
        errors = validate_kb.validate_index(self.kb)
        self.assertTrue(
            any("non-existent wiki file: wiki/concepts/missing.md" in e
                for e in errors), errors)

    def test_unknown_log_operation_is_reported(self):
        self._write_index()
        log = self.kb / "log.md"
        log.write_text(log.read_text().replace("] ingest | second",
                                               "] ingest-code | second"))
        errors = validate_kb.validate_log(kb_dir=self.kb)
        self.assertTrue(
            any("uses operation 'ingest-code'" in e for e in errors), errors)

    def _git(self, *args):
        return subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=True).stdout.strip()

    def _init_repo(self):
        self._write_index()
        self._git("init", "-q", "-b", "main")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "init")

    def test_check_git_requires_log_entry_for_kb_changes(self):
        self._init_repo()
        (self.kb / "wiki" / "concepts" / "a.md").write_text(_PAGE + "more\n")
        errors = validate_kb.validate_log(check_git_diff=True, kb_dir=self.kb)
        self.assertTrue(any("no accompanying entry" in e for e in errors),
                        errors)
        (self.kb / "log.md"
         ).write_text("# Log\n\n---\n\n## [2026-09-11] manual | fix\n\n" +
                      _LOG.split("---\n\n")[1])
        self.assertEqual(
            validate_kb.validate_log(check_git_diff=True, kb_dir=self.kb), [])

    def test_check_git_ignores_tooling_changes(self):
        """Editing the validator is code, not a knowledge event, so it needs no log entry."""
        self._init_repo()
        scripts = self.kb / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "validate_kb.py").write_text("# edited\n")

        self.assertEqual(
            validate_kb.validate_log(check_git_diff=True, kb_dir=self.kb), [])

    def test_check_git_against_base_ref(self):
        self._init_repo()
        base = self._git("rev-parse", "HEAD")
        (self.kb / "wiki" / "concepts" / "a.md").write_text(_PAGE + "more\n")
        self._git("commit", "-q", "-am", "page only")
        errors = validate_kb.validate_log(check_git_diff=True,
                                          kb_dir=self.kb,
                                          base=base)
        self.assertTrue(any("no accompanying entry" in e for e in errors),
                        errors)
        (self.kb / "log.md"
         ).write_text("# Log\n\n---\n\n## [2026-09-11] manual | fix\n\n" +
                      _LOG.split("---\n\n")[1])
        self._git("commit", "-q", "-am", "log too")
        self.assertEqual(
            validate_kb.validate_log(check_git_diff=True,
                                     kb_dir=self.kb,
                                     base=base), [])

    def test_log_out_of_order_is_reported(self):
        self._write_index()
        (self.kb / "log.md").write_text(
            "# Log\n\n---\n\n## [2026-09-01] manual | first\n\n## [2026-09-10] ingest | second\n"
        )
        errors = validate_kb.validate_log(kb_dir=self.kb)
        self.assertTrue(any("out of chronological order" in e for e in errors),
                        errors)


if __name__ == "__main__":
    unittest.main()
