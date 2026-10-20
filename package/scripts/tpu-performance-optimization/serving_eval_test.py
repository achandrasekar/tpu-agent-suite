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
"""Unit tests for serving_eval.py, against a fake server and fake lm-eval."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import pathlib
import shlex
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import serving_eval  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "serving_eval", _SCRIPTS_DIR / "serving_eval.py")
    serving_eval = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(serving_eval)

# An lm-eval results file for mmlu_llama, trimmed: the overall score, one
# category group and one subject.
RESULTS = {
    "results": {
        "mmlu_llama": {
            "sample_len": 1140,
            "exact_match,strict_match": 0.80,
            "exact_match_stderr,strict_match": 0.0115,
        },
        "mmlu_llama_stem": {
            "sample_len": 380,
            "exact_match,strict_match": 0.75,
            "exact_match_stderr,strict_match": 0.021,
        },
        "mmlu_llama_virology": {
            "sample_len": 20,
            "exact_match,strict_match": 0.60,
            "exact_match_stderr,strict_match": 0.11,
        },
    },
    "groups": {
        "mmlu_llama": {},
        "mmlu_llama_stem": {}
    },
}

# Stands in for lm_eval: writes results.json (with override.json merged into
# its results, when present) under --output_path, and records its argv.
# Hangs when a "hang" file exists.
FAKE_LM_EVAL = r'''
import json, pathlib, sys, time
argv = sys.argv[1:]
here = pathlib.Path(__file__).parent
with open(here / "calls.jsonl", "a") as f:
    f.write(json.dumps(argv) + "\n")
if (here / "hang").exists():
    time.sleep(60)
if (here / "fail").exists():
    print('WARNING API request failed with error message: {"error": '
          '"Input length (3132) exceeds model context (2048)"}. Retrying...')
    sys.exit(7)
data = json.loads((here / "results.json").read_text())
override = here / "override.json"
if override.exists():
    for task, metrics in json.loads(override.read_text()).items():
        data["results"][task].update(metrics)
out = pathlib.Path(argv[argv.index("--output_path") + 1]) / "org__model"
out.mkdir(parents=True, exist_ok=True)
text = json.dumps(data)
if (here / "truncate").exists():
    text = text[:len(text) // 2]
(out / "results_2026-01-01T00-00-00.json").write_text(text)
answer = "" if (here / "empty").exists() else "A."
(out / "samples_mmlu_llama_virology.jsonl").write_text(
    json.dumps({"filtered_resps": [answer], "target": "A."}) + "\n")
'''


class _FakeServer(http.server.BaseHTTPRequestHandler):

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = b""
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "org/model"}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestGate(unittest.TestCase):

    def test_delta_stderr_adds_in_quadrature(self):
        self.assertAlmostEqual(serving_eval.delta_stderr(0.03, 0.04), 0.05)
        # A run without a stderr is assumed as noisy as the baseline.
        self.assertAlmostEqual(serving_eval.delta_stderr(0.03, None),
                               0.03 * 2**0.5)
        self.assertEqual(serving_eval.delta_stderr(None, None), 0.0)

    def test_small_drop_within_noise_passes(self):
        base = {"t": {"value": 0.80, "stderr": 0.01}}
        run = {"t": {"value": 0.78, "stderr": 0.01}}
        row = serving_eval.check_gate(run, base, 0.015)[0]
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["allowed_drop"], 1.645 * 0.01 * 2**0.5)

    def test_large_drop_fails(self):
        base = {"t": {"value": 0.80, "stderr": 0.01}}
        run = {"t": {"value": 0.70, "stderr": 0.01}}
        self.assertFalse(
            serving_eval.check_gate(run, base, 0.015)[0]["passed"])

    def test_tolerance_is_the_floor_without_stderr(self):
        base = {"t": {"value": 0.80, "stderr": None}}
        self.assertTrue(
            serving_eval.check_gate({"t": {
                "value": 0.79,
                "stderr": None
            }}, base, 0.015)[0]["passed"])
        self.assertFalse(
            serving_eval.check_gate({"t": {
                "value": 0.78,
                "stderr": None
            }}, base, 0.015)[0]["passed"])

    def test_missing_task_fails_and_subjects_are_not_gated(self):
        base = {
            "t": {
                "value": 0.8,
                "stderr": 0.01
            },
            "subject": {
                "value": 0.5,
                "stderr": 0.1,
                "headline": False
            },
        }
        rows = serving_eval.check_gate({}, base, 0.015)
        self.assertEqual([r["task"] for r in rows], ["t"])
        self.assertFalse(rows[0]["passed"])


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        (self.dir / "fake_lm_eval.py").write_text(FAKE_LM_EVAL,
                                                  encoding="utf-8")
        (self.dir / "results.json").write_text(json.dumps(RESULTS),
                                               encoding="utf-8")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _FakeServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _main(self, out: str, *extra: str) -> int:
        cmd = shlex.join([sys.executable, str(self.dir / "fake_lm_eval.py")])
        return serving_eval.main([
            "--base-url", self.base_url, "--output-dir",
            str(self.dir / out), "--lm-eval-cmd", cmd, "--ready-timeout", "5",
            *extra
        ])

    def _summary(self, out: str) -> dict:
        return json.loads((self.dir / out / "eval_summary.json").read_text())

    def test_scores_keyed_by_task_with_headline_marked(self):
        self.assertEqual(self._main("run"), serving_eval.EXIT_OK)
        summary = self._summary("run")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["model"], "org/model")
        scores = summary["scores"]
        self.assertEqual(scores["mmlu_llama"]["value"], 0.80)
        self.assertEqual(scores["mmlu_llama"]["stderr"], 0.0115)
        self.assertTrue(scores["mmlu_llama_stem"]["headline"])
        self.assertFalse(scores["mmlu_llama_virology"]["headline"])
        table = (self.dir / "run" / "eval_summary.md").read_text()
        self.assertIn("mmlu_llama_stem", table)
        self.assertNotIn("virology", table)

        argv = json.loads((self.dir / "calls.jsonl").read_text())
        self.assertEqual(argv[argv.index("--tasks") + 1], "mmlu_llama")
        self.assertEqual(argv[argv.index("--limit") + 1], "100")
        self.assertEqual(argv[argv.index("--model") + 1],
                         "local-chat-completions")
        model_args = argv[argv.index("--model_args") + 1]
        self.assertIn("model=org/model", model_args)
        self.assertIn(f"base_url={self.base_url}/v1/chat/completions",
                      model_args)
        self.assertIn("--apply_chat_template", argv)
        self.assertEqual(argv[argv.index("--fewshot_as_multiturn") + 1],
                         "true")

    def test_limit_zero_runs_every_question(self):
        self._main("run", "--limit", "0")
        argv = json.loads((self.dir / "calls.jsonl").read_text())
        self.assertNotIn("--limit", argv)

    def test_gate_against_baseline(self):
        self._main("base")
        baseline = str(self.dir / "base" / "eval_summary.json")
        self.assertEqual(self._main("same", "--baseline", baseline),
                         serving_eval.EXIT_OK)
        (self.dir / "override.json").write_text(
            json.dumps({"mmlu_llama": {
                "exact_match,strict_match": 0.70
            }}))
        self.assertEqual(self._main("worse", "--baseline", baseline),
                         serving_eval.EXIT_GATE_FAILED)
        summary = self._summary("worse")
        self.assertEqual(summary["status"], "gate_failed")
        failed = [r["task"] for r in summary["gate"] if not r["passed"]]
        self.assertEqual(failed, ["mmlu_llama"])

    def test_unusable_baseline_is_a_setup_error(self):
        self._main("base")
        good = json.loads(
            (self.dir / "base" / "eval_summary.json").read_text())
        for name, bad in (("failed", {
                **good, "status": "failed"
        }), ("empty", {
                **good, "scores": {}
        })):
            path = self.dir / f"{name}.json"
            path.write_text(json.dumps(bad))
            with self.subTest(name):
                self.assertEqual(self._main("run", "--baseline", str(path)),
                                 serving_eval.EXIT_SETUP)

    def test_lm_eval_failure_is_reported(self):
        (self.dir / "fail").write_text("")
        self.assertEqual(self._main("run"), serving_eval.EXIT_EVAL_FAILED)
        summary = self._summary("run")
        self.assertEqual(summary["status"], "failed")
        self.assertIn("exited 7", summary["errors"][0])
        self.assertIn("exceeds model context (2048)", summary["errors"][0])

    def test_hung_task_times_out(self):
        (self.dir / "hang").write_text("")
        self.assertEqual(self._main("run", "--run-timeout", "1"),
                         serving_eval.EXIT_EVAL_FAILED)
        errors = self._summary("run")["errors"]
        self.assertIn("did not finish within 1s", errors[0])

    def test_server_down_after_a_failure_stops_the_tasks(self):
        (self.dir / "fail").write_text("")
        with unittest.mock.patch.object(serving_eval,
                                        "server_alive",
                                        return_value=False):
            rc = self._main("run", "--tasks", "mmlu_llama,gsm8k")
        self.assertEqual(rc, serving_eval.EXIT_SETUP)
        calls = (self.dir / "calls.jsonl").read_text().splitlines()
        self.assertEqual(len(calls), 1)
        errors = self._summary("run")["errors"]
        self.assertIn("gsm8k: not run: the server stopped answering",
                      errors[1])

    def test_earlier_attempt_in_output_dir_is_not_counted(self):
        self.assertEqual(self._main("run"), serving_eval.EXIT_OK)
        old = time.time() - 3600
        for path in (self.dir / "run" / "lm_eval").rglob("*.json*"):
            os.utime(path, (old, old))
        (self.dir / "fail").write_text("")
        self.assertEqual(self._main("run"), serving_eval.EXIT_EVAL_FAILED)
        self.assertEqual(self._summary("run")["scores"], {})

    def test_truncated_results_fail_the_run(self):
        (self.dir / "truncate").write_text("")
        self.assertEqual(self._main("run"), serving_eval.EXIT_EVAL_FAILED)
        errors = self._summary("run")["errors"]
        self.assertIn("is not JSON", errors[0])

    def test_empty_answers_fail_the_run(self):
        (self.dir / "empty").write_text("")
        self.assertEqual(self._main("run"), serving_eval.EXIT_EVAL_FAILED)
        summary = self._summary("run")
        self.assertEqual(summary["unscored_answers"], {"count": 1, "of": 1})
        self.assertIn("empty", summary["errors"][0])

    def test_unreachable_server_is_a_setup_error(self):
        rc = serving_eval.main([
            "--base-url", "http://127.0.0.1:1", "--output-dir",
            str(self.dir / "x"), "--ready-timeout", "0"
        ])
        self.assertEqual(rc, serving_eval.EXIT_SETUP)


if __name__ == "__main__":
    unittest.main()
