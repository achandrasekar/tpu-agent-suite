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
"""Verdict rules for the attention-kernel-tuning task.

These are the rules the whole task turns on, and three of them have never fired
in a real run: no agent has yet shipped a narrower dtype, spread the work across
the slice, or overrun the profile budget. Waiting for one to do it is not a
test strategy -- a signal that has only ever seen negatives has not been shown
to work, which is exactly how a broken delegation check sat green for thirteen
runs. So they are pinned here instead.

`analyze.py` is a task-local script rather than an importable module, so it is
loaded by path.
"""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ANALYZE = (Path(__file__).resolve().parents[1] / "datasets" /
            "xprof-attention" / "kernel-tuning" / "tests" / "analyze.py")
_spec = importlib.util.spec_from_file_location("attention_analyze", _ANALYZE)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def fingerprint(total, sumsq, absmax):
    return {"sum": total, "sumsq": sumsq, "absmax": absmax, "tail": [1.0] * 8}


def session(**kw):
    base = {
        "dir": kw.pop("dir", "x"),
        "device_us": kw.pop("device_us", 1000.0),
        "output_dtype": kw.pop("output_dtype", "bfloat16"),
        "input_dtype": kw.pop("input_dtype", "bfloat16"),
        "output_device_count": kw.pop("output_device_count", 1),
        "fingerprint": kw.pop("fingerprint", fingerprint(100.0, 200.0, 3.0)),
        "checksum": kw.pop("checksum", 100.0),
    }
    base.update(kw)
    return base


class TestClassify(unittest.TestCase):
    """One verdict per session, and only `ok` may win."""

    def setUp(self):
        self.baseline = session(dir="baseline", device_us=6360.0)

    def test_identical_math_is_ok(self):
        s = session()
        analyze.classify(s, self.baseline)
        self.assertEqual(s["numerics"], "ok")
        self.assertEqual(s["rel_diff"], 0.0)

    def test_blocked_attention_drift_is_ok(self):
        """~1 bfloat16 ULP. Rejecting this is what NUMERIC_TOL was widened for:
        a legitimate blocked implementation lands here, and 1e-3 failed them
        all."""
        s = session(fingerprint=fingerprint(100.4, 200.8, 3.01))
        analyze.classify(s, self.baseline)
        self.assertEqual(s["numerics"], "ok")
        self.assertLess(s["rel_diff"], analyze.NUMERIC_TOL)

    def test_wrong_answer_is_mismatch(self):
        s = session(fingerprint=fingerprint(96.7, 187.0, 2.8))
        analyze.classify(s, self.baseline)
        self.assertEqual(s["numerics"], "mismatch")

    def test_returning_a_wider_dtype_is_allowed(self):
        """`jnp.einsum` promotes bf16 inputs, so the reference itself returns
        float32 while its docstring says "bfloat16 in and out". Comparing
        against the baseline's *output* flagged a correct bf16 answer as a
        cheat (run 18, `opt2_sdpa_transposed`, numerically fine at 0.0044).
        Both widths are legitimate; only going below the input is not."""
        for out in ("bfloat16", "float32"):
            with self.subTest(output_dtype=out):
                s = session(output_dtype=out, input_dtype="bfloat16")
                analyze.classify(s, self.baseline)
                self.assertEqual(s["numerics"], "ok")

    def test_narrower_dtype_loses_even_when_numerically_closer(self):
        """The case a tolerance cannot catch. Every fingerprint field is taken
        after `.astype(float32)`, so an fp8 variant that widens at the end can
        sit *closer* to the reference than an honest blocked one while being
        several times faster for doing less arithmetic."""
        cheat = session(output_dtype="float8_e4m3",
                        fingerprint=fingerprint(100.2, 200.4, 3.0))
        honest = session(fingerprint=fingerprint(100.4, 200.8, 3.01))
        analyze.classify(cheat, self.baseline)
        analyze.classify(honest, self.baseline)
        self.assertLess(cheat["rel_diff"], honest["rel_diff"])
        self.assertEqual(cheat["numerics"], "wrong_dtype")
        self.assertEqual(honest["numerics"], "ok")

    def test_multi_device_loses_on_exact_numerics(self):
        """Sharding is the same arithmetic spread wider: numerically perfect,
        worth a different multiple on every topology, and not attention
        tuning."""
        s = session(output_device_count=8)
        analyze.classify(s, self.baseline)
        self.assertEqual(s["rel_diff"], 0.0)
        self.assertEqual(s["numerics"], "multi_device")

    def test_single_device_is_not_flagged(self):
        for count in (1, None):
            with self.subTest(output_device_count=count):
                s = session(output_device_count=count)
                analyze.classify(s, self.baseline)
                self.assertFalse(s["multi_device"])
                self.assertEqual(s["numerics"], "ok")

    def test_dtype_outranks_a_numeric_match(self):
        """Ordering matters: a narrower dtype is disqualifying even when the
        numbers agree exactly."""
        s = session(output_dtype="float8_e5m2")
        analyze.classify(s, self.baseline)
        self.assertEqual(s["rel_diff"], 0.0)
        self.assertEqual(s["numerics"], "wrong_dtype")

    def test_no_baseline_leaves_it_unknown(self):
        s = session(fingerprint=None, checksum=None)
        analyze.classify(s, None)
        self.assertEqual(s["numerics"], "unknown")


class TestSessionDedup(unittest.TestCase):
    """One capture must count once, however many places it was copied to.

    The collector untars an archive that already carries the label directory
    into a directory of the same name, so the same trace lands at
    `xprof/<label>/plugins/...` and `xprof/<label>/<label>/plugins/...`. Both
    normalise to a session, so every capture counted twice: run 23 reported six
    profiles for three labels, filled its budget on the duplicates, and read
    "1 variant" for a run that tried three.
    """

    def _tree(self, root, label, depth_paths, runs):
        for d in depth_paths:
            for i, run in enumerate(runs, 1):
                p = Path(root, d, "plugins", "profile", run)
                p.mkdir(parents=True, exist_ok=True)
                (p / f"attn-{label}-{i}.xplane.pb").write_text("x")

    def test_same_trace_at_two_depths_counts_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._tree(d, "baseline", ["baseline", "baseline/baseline"],
                       ["2026_01_01_00_00_00"])
            self.assertEqual(len(analyze.session_dirs(d)), 1)

    def test_the_superset_wins_when_one_copy_is_partial(self):
        """A label can hold more than one profile run, and the shallower copy
        may carry only some of them. Dropping the superset would lose a real
        capture."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._tree(d, "opt", ["opt/opt"],
                       ["2026_01_01_00_01_00", "2026_01_01_00_02_00"])
            self._tree(d, "opt", ["opt"], ["2026_01_01_00_01_00"])
            dirs = analyze.session_dirs(d)
            self.assertEqual(len(dirs), 1)
            self.assertTrue(dirs[0].endswith("opt/opt"),
                            f"kept the partial copy: {dirs[0]}")

    def test_distinct_labels_are_not_merged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._tree(d, "a", ["a"], ["2026_01_01_00_00_00"])
            self._tree(d, "b", ["b"], ["2026_01_01_00_00_00"])
            self.assertEqual(len(analyze.session_dirs(d)), 2)


class TestBudget(unittest.TestCase):
    """Over-budget is a failure, not a footnote. Nothing has overrun the cap
    since the check existed, so only the passing path has ever been seen in a
    run -- the interesting half is pinned here."""

    def test_exactly_at_the_cap_is_allowed(self):
        b = analyze.budget(taken=6, attempts=5, cap=6)
        self.assertTrue(b["budget_ok"])
        self.assertEqual(b["over_cap"], 0)
        # One baseline plus five attempts.
        self.assertEqual(b["attempts_used"], 5)

    def test_one_past_the_cap_fails(self):
        b = analyze.budget(taken=7, attempts=5, cap=6)
        self.assertFalse(b["budget_ok"])
        self.assertEqual(b["over_cap"], 1)

    def test_extras_do_not_buy_more_attempts(self):
        """Captures past the cap are dropped from the analysis, so an agent
        that takes nine must not be credited with eight attempts."""
        b = analyze.budget(taken=9, attempts=5, cap=6)
        self.assertEqual(b["attempts_used"], 5)
        self.assertEqual(b["over_cap"], 3)
        self.assertFalse(b["budget_ok"])

    def test_short_run_is_within_budget(self):
        b = analyze.budget(taken=2, attempts=5, cap=6)
        self.assertTrue(b["budget_ok"])
        self.assertEqual(b["attempts_used"], 1)

    def test_no_captures_reports_no_attempts(self):
        b = analyze.budget(taken=0, attempts=5, cap=6)
        self.assertEqual(b["attempts_used"], 0)
        self.assertTrue(b["budget_ok"])


class TestBudgetLedger(unittest.TestCase):
    """The cap must not be reversible by the agent it constrains. Counting
    directories alone meant `rm -rf` on one capture bought another."""

    def test_ledger_binds_when_captures_were_deleted(self):
        # Nine starts recorded, three directories left on disk.
        b = analyze.budget(taken=3, attempts=5, cap=6, logged=9)
        self.assertFalse(b["budget_ok"])
        self.assertEqual(b["over_cap"], 3)
        self.assertTrue(b["ledger_discrepancy"])

    def test_directories_bind_when_the_ledger_was_truncated(self):
        b = analyze.budget(taken=8, attempts=5, cap=6, logged=1)
        self.assertFalse(b["budget_ok"])
        self.assertEqual(b["over_cap"], 2)
        self.assertTrue(b["ledger_discrepancy"])

    def test_agreement_is_not_a_discrepancy(self):
        b = analyze.budget(taken=6, attempts=5, cap=6, logged=6)
        self.assertTrue(b["budget_ok"])
        self.assertFalse(b["ledger_discrepancy"])

    def test_missing_ledger_falls_back_to_directories(self):
        """An older image ships no ledger; that must not fail every run."""
        b = analyze.budget(taken=6, attempts=5, cap=6, logged=None)
        self.assertTrue(b["budget_ok"])
        self.assertFalse(b["ledger_discrepancy"])
        self.assertIsNone(b["captures_logged"])

    def test_ledger_count_reads_nonblank_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".captures").write_text("a b\n\nc d\n", encoding="utf-8")
            self.assertEqual(analyze.ledger_count(d), 2)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(analyze.ledger_count(d))


class TestBaselineSelection(unittest.TestCase):
    """The baseline is the session profiled first -- and only if it can be
    read. Falling through to the first *timed* session promotes an agent
    variant to baseline whenever the baseline capture is unreadable, and a
    variant compared against itself looks perfectly fine."""

    @staticmethod
    def _pick(sessions):
        """The rule as `main` applies it."""
        first = sessions[0] if sessions else None
        return (first if first and first["device_us"] else None,
                bool(first and not first["device_us"]))

    def test_readable_first_session_is_the_baseline(self):
        sessions = [
            session(dir="baseline", device_us=6360.0),
            session(dir="opt", device_us=2800.0)
        ]
        base, unreadable = self._pick(sessions)
        self.assertEqual(base["dir"], "baseline")
        self.assertFalse(unreadable)

    def test_unreadable_baseline_does_not_promote_a_variant(self):
        sessions = [
            session(dir="baseline", device_us=None),
            session(dir="opt", device_us=2800.0)
        ]
        base, unreadable = self._pick(sessions)
        self.assertIsNone(
            base, "an optimised variant must not become the "
            "baseline just because the real one is unreadable")
        self.assertTrue(unreadable)

    def test_no_baseline_leaves_every_verdict_unknown(self):
        """The cost of refusing to guess: nothing can be judged, and the run
        fails honestly rather than comparing a variant against itself."""
        s = session(dir="opt")
        analyze.classify(s, None)
        self.assertEqual(s["numerics"], "unknown")


class TestFingerprintDistance(unittest.TestCase):

    def test_worst_component_wins(self):
        """A lone sum can hold steady through cancellation while the values
        underneath move, so the worst of the four statistics decides."""
        a = fingerprint(100.0, 200.0, 3.0)
        b = fingerprint(100.0, 260.0, 3.0)  # sum identical, sumsq 30% off
        self.assertGreater(analyze.fingerprint_distance(a, b), 0.25)


class TestXprofVersion(unittest.TestCase):
    """The image installs a floor, so the toolchain is a property of the run.
  Two scores a week apart were not measured with the same xprof, and the one
  thing that must never happen is the field going missing silently -- that is
  how a floating pin becomes unreproducible without anyone noticing."""

    def test_reports_the_installed_distribution(self):
        """`which` is mocked away on purpose: without it this reads whatever
    xprof happens to be on the developer's PATH, and the test passes or fails
    by accident of the machine."""
        with mock.patch.object(analyze.shutil, "which", return_value=None), \
             mock.patch("importlib.metadata.version",
                        return_value="2.24.0") as v:
            self.assertEqual(analyze.xprof_version(), "2.24.0")
        v.assert_called_once_with("xprof")

    def test_a_missing_distribution_does_not_raise(self):
        """Nothing answers; the run must still produce an analysis."""

        def boom(_name):
            raise Exception("no such distribution")

        with mock.patch("importlib.metadata.version", boom), \
             mock.patch.object(analyze.shutil, "which", return_value=None), \
             mock.patch.object(analyze.subprocess, "run", side_effect=OSError):
            self.assertIsNone(analyze.xprof_version())

    def test_it_names_the_binary_that_measured_not_this_interpreter(self):
        """A rescore pointed XPROF_BIN at a private venv holding
    2.24.2a20260922 while this interpreter carried 2.24.1a20260903, and the
    analysis named the interpreter's. Naming the wrong toolchain is worse than
    naming none."""
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "xprof"
            script.write_text("#!/venv/bin/python3\n# console script\n")
            done = subprocess.CompletedProcess([],
                                               0,
                                               stdout="2.24.2a20260922\n",
                                               stderr="")
            with mock.patch.object(analyze, "XPROF_BIN", str(script)), \
                 mock.patch.object(analyze.shutil, "which", return_value=str(script)), \
                 mock.patch("importlib.metadata.version",
                            return_value="2.24.1a20260903"), \
                 mock.patch.object(analyze.subprocess, "run",
                                   return_value=done) as run:
                self.assertEqual(analyze.xprof_version(), "2.24.2a20260922")
            self.assertEqual(run.call_args[0][0][0], "/venv/bin/python3")

    def test_a_binary_that_is_not_a_console_script_falls_back(self):
        """No shebang to read: this interpreter's metadata is the next best
    claim, and is still better than nothing."""
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / "xprof"
            binary.write_bytes(b"\x7fELF\x02\x01\x01\x00")
            with mock.patch.object(analyze, "XPROF_BIN", str(binary)), \
                 mock.patch.object(analyze.shutil, "which", return_value=str(binary)), \
                 mock.patch("importlib.metadata.version",
                            return_value="2.24.1a20260903"):
                self.assertEqual(analyze.xprof_version(), "2.24.1a20260903")


class TestUnchargedKernelsAreExcluded(unittest.TestCase):
    """`total_device_duration_us` is the session total, and the session is not
  only the layer. Run 28 paid up to 7023 us of `xsc-output-checksum-compute`
  inside the profiled window on variants the baseline never charged, which
  understated its speedup by more than 5x and inverted the ranking."""

    @staticmethod
    def _doc(total, records):
        return json.dumps({
            "total_device_duration_us": total,
            "kernel_records": records
        })

    def _describe(self, total, records):
        with mock.patch.object(analyze, "xprof",
                               return_value=self._doc(total, records)), \
             mock.patch.object(analyze, "run_record", return_value={}):
            return analyze.describe("/logs/artifacts/xprof/opt")

    def test_checksum_time_is_not_attention_time(self):
        """Run 28's opt3: 1786.0 total, 1448.8 of it uncharged."""
        out = self._describe(1786.044, [{
            "kernel_name": "xsc-output-checksum-compute",
            "total_duration_us": 1448.8
        }, {
            "kernel_name": "%copy-start = (bf16[8,4096,128])",
            "total_duration_us": 337.2
        }])
        self.assertAlmostEqual(out["device_us"], 337.244, places=2)
        self.assertAlmostEqual(out["excluded_us"], 1448.8, places=2)
        self.assertEqual(out["session_us"], 1786.044)

    def test_a_session_with_nothing_excluded_is_unchanged(self):
        """The baseline's shape: no uncharged kernel, so nothing moves."""
        out = self._describe(6360.629, [{
            "kernel_name": "%fusion.1 = bf16[8,4096,128]",
            "total_duration_us": 6360.629
        }])
        self.assertAlmostEqual(out["device_us"], 6360.629, places=2)
        self.assertEqual(out["excluded_us"], 0.0)
        self.assertEqual(out["excluded_kernels"], [])

    def test_what_was_removed_is_reported(self):
        """Excluded silently is excluded unaccountably."""
        out = self._describe(100.0, [{
            "kernel_name": "xsc-output-checksum-compute",
            "total_duration_us": 40.0
        }])
        self.assertEqual(out["excluded_kernels"],
                         ["xsc-output-checksum-compute"])

    def test_a_session_that_is_all_uncharged_reports_no_time(self):
        """Not zero -- zero would read as "ran instantly" and win check 1."""
        out = self._describe(50.0, [{
            "kernel_name": "output-checksum",
            "total_duration_us": 50.0
        }])
        self.assertIsNone(out["device_us"])

    def test_ranking_follows_attention_time_not_session_total(self):
        """Run 28 inverted: opt3 is fastest by total, opt4 by attention."""
        opt3 = self._describe(1786.0, [{
            "kernel_name": "xsc-output-checksum-compute",
            "total_duration_us": 1448.8
        }, {
            "kernel_name": "%fusion",
            "total_duration_us": 337.2
        }])
        opt4 = self._describe(1978.9, [{
            "kernel_name": "xsc-output-checksum-compute",
            "total_duration_us": 1861.8
        }, {
            "kernel_name": "%fusion",
            "total_duration_us": 117.1
        }])
        self.assertLess(opt3["session_us"], opt4["session_us"])
        self.assertLess(opt4["device_us"], opt3["device_us"])


class TestKeptTheFastestConfiguration(unittest.TestCase):
    """The check runs as an embedded block inside `test.sh`, so it is exercised
  by extracting the shipped block and running it -- testing a copy would not
  notice the shipped one drifting.

  What it must never do is credit an agent for keeping a configuration it did
  not ship. Run 26 left a mismatching variant in place; the check passed anyway
  by substituting the last *valid* session two profiles earlier."""

    @staticmethod
    def _run(sessions):
        shell = (Path(__file__).resolve().parents[1] / "datasets" /
                 "xprof-attention" / "kernel-tuning" / "tests" / "test.sh")
        text = shell.read_text()
        block = text.split("<<'PYT'\n", 1)[1].split("\nPYT\n", 1)[0]
        with tempfile.TemporaryDirectory() as d:
            traces = Path(d) / "traces.json"
            traces.write_text(
                json.dumps({
                    "sessions": sessions,
                    "best_tiling": None
                }))
            script = Path(d) / "check.py"
            script.write_text(block)
            r = subprocess.run(
                ["python3", str(script),
                 str(traces), ""],
                capture_output=True,
                text=True,
                timeout=60)
        return r.stdout.strip()

    @staticmethod
    def _s(name, us, numerics="ok"):
        return {
            "dir": f"/logs/{name}",
            "device_us": us,
            "numerics": numerics,
            "is_baseline": False,
            "tiles": None
        }

    def test_shipping_a_mismatch_earns_nothing(self):
        """Run 26's shape: the fastest *valid* session is not what it shipped."""
        out = self._run([
            self._s("baseline", 6357.0) | {
                "is_baseline": True
            },
            self._s("opt2_correct", 51501.0),
            self._s("opt4_wrong", 678.0, "mismatch")
        ])
        self.assertTrue(out.startswith("|"), f"should not pass: {out}")
        self.assertIn("opt4_wrong", out)
        self.assertIn("disqualified", out)

    def test_shipping_the_fastest_valid_passes(self):
        """Run 27's shape."""
        out = self._run([
            self._s("baseline", 6358.0) | {
                "is_baseline": True
            },
            self._s("opt1_sdpa", 678.8, "mismatch"),
            self._s("opt2_chunked", 8263.6),
            self._s("opt3_head_chunked", 6159.5)
        ])
        self.assertTrue(out.startswith("1|"), f"should pass: {out}")
        self.assertIn("opt3_head_chunked", out)

    def test_shipping_a_slower_valid_one_fails(self):
        out = self._run([
            self._s("baseline", 6358.0) | {
                "is_baseline": True
            },
            self._s("fast", 900.0),
            self._s("slow", 5000.0)
        ])
        self.assertTrue(out.startswith("|"), f"should not pass: {out}")
        self.assertIn("fastest measured was fast", out)


if __name__ == "__main__":
    unittest.main()
