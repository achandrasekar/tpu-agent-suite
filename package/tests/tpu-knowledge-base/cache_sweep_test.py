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
"""The session-start sweep: once a day, silent, and `--now` by hand."""

import contextlib
import datetime
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest import mock

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
           "tpu-knowledge-base" / "cache_sweep.py")
_spec = importlib.util.spec_from_file_location("cache_sweep", _SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

ROW = {
    "id": "fetched",
    "url": "https://e.test/doc",
    "prompt": "howto",
    "keywords": ["fetched-thing"]
}


class SweepTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = pathlib.Path(tmp.name)
        for target, name, value in ((sweep.summarize, "CACHE_DIRS",
                                     [self.cache]),
                                    (sweep.summarize.catalog, "rows",
                                     mock.Mock(return_value=[ROW]))):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_session_start_sweeps_once_a_day_and_says_nothing(self):
        with mock.patch.object(sweep.summarize, "purge",
                               return_value={}) as purge, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sweep.main([]), 0)
            self.assertEqual(sweep.main([]), 0)
        self.assertEqual(purge.call_count, 1, "swept twice in one day")
        self.assertEqual(sweep.summarize.catalog.rows.call_count, 1,
                         "read the index on a day that does not sweep")
        self.assertEqual(out.getvalue(), "", "a hook's stdout is its verdict")
        self.assertTrue((self.cache / sweep.summarize.SWEPT).is_file())

    def test_a_build_without_the_catalog_has_nothing_to_sweep(self):
        # A hook is in every build; the catalog ships with an agent.
        with mock.patch.object(sweep, "summarize", None), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sweep.main([]), 0)
            with self.assertRaises(SystemExit), \
                 contextlib.redirect_stderr(io.StringIO()):
                sweep.main(["--now"])
        self.assertEqual(out.getvalue(), "")

    def test_a_broken_index_is_the_lookups_problem_not_the_sessions(self):
        with mock.patch.object(sweep.summarize.catalog, "rows",
                               side_effect=ValueError("line 3")), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sweep.main([]), 0)
            with self.assertRaises(ValueError):
                sweep.main(["--now"])
        self.assertEqual(out.getvalue(), "")
        # The attempt is stamped: tried again tomorrow, not every start.
        self.assertTrue((self.cache / sweep.summarize.SWEPT).is_file())

    def test_now_reports_and_dry_run_removes_nothing(self):
        page = self.cache / ROW["id"] / "answer.md"
        page.parent.mkdir(parents=True)
        days = sweep.summarize.FETCH_TTL_DAYS + 1
        today = datetime.date.today()
        then = (today - datetime.timedelta(days=days)).isoformat()
        page.write_text(
            sweep.summarize.header(ROW, "k",
                                   True).replace(f"retrieved: {today}",
                                                 f"retrieved: {then}"))
        mtime = time.time() - days * sweep.summarize.A_DAY
        os.utime(page, (mtime, mtime))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sweep.main(["--now", "--dry-run"]), 0)
        report = json.loads(out.getvalue())
        self.assertTrue(report["dry_run"])
        self.assertEqual(len(report["removed"]), 1)
        self.assertTrue(page.is_file(), "a dry run deleted")
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.main(["--now"])
        self.assertFalse(page.exists())
        with self.assertRaises(SystemExit), \
             contextlib.redirect_stderr(io.StringIO()):
            sweep.main(["--dry-run"])


if __name__ == "__main__":
    unittest.main()
