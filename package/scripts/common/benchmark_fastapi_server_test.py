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
"""The FastAPI benchmark client against a stand-in server.

The result must read like vLLM's: the same keys, the same percentile
arithmetic, and no TTFT a non-streaming endpoint could not have measured.
And no measured request may reuse a prompt, warmups included.
"""

import contextlib
import http.server
import io
import json
import pathlib
import random
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import benchmark_fastapi_server as bench


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Answers like the FastAPI recipe and records every payload."""
    payloads: list = []
    fail_after = None

    def do_POST(self):  # pylint: disable=invalid-name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).payloads.append((self.path, body))
        if (self.fail_after is not None
                and len(self.payloads) > self.fail_after):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"detail": "boom"}')
            return
        answer = ({
            "embedding": [0.1, 0.2]
        } if self.path == "/embed" else {
            "result": "one two three"
        })
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


class _WordTokenizer:
    """One token per whitespace-separated word; ids index a fixed vocab."""
    all_special_ids = [0]
    vocab = ["<pad>"] + [f"w{i}" for i in range(1, 50)]

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1] * len(text.split())


class BenchmarkTest(unittest.TestCase):

    def setUp(self):
        _Recorder.payloads = []
        _Recorder.fail_after = None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _Recorder)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def run_bench(self, *extra):
        argv = [
            "--base-url", self.url, "--input-len", "8", "--output-len", "4",
            "--num-prompts", "6", "--num-warmups", "2", "--max-concurrency",
            "2", "--metric-percentiles", "50,90,99", "--save-result",
            "--save-detailed", "--result-dir", self.dir.name,
            "--result-filename", "r.json", *extra
        ]
        code = bench.main(argv)
        result = json.loads(
            (pathlib.Path(self.dir.name) / "r.json").read_text())
        return code, result

    def test_result_uses_vllm_keys_and_omits_streaming_metrics(self):
        code, result = self.run_bench("--model", "org/m")
        self.assertEqual(code, 0)
        for key in ("duration", "completed", "failed", "request_throughput",
                    "output_throughput", "total_token_throughput",
                    "mean_e2el_ms", "median_e2el_ms", "std_e2el_ms",
                    "p50_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
                    "max_concurrency", "num_prompts", "model_id"):
            self.assertIn(key, result)
        self.assertEqual(result["completed"], 6)
        self.assertFalse([k for k in result if "ttft" in k or "tpot" in k])
        self.assertEqual(result["unsupported_metrics"],
                         ["ttft", "tpot", "itl"])

    def test_warmups_are_sent_but_not_measured(self):
        _, result = self.run_bench()
        self.assertEqual(len(_Recorder.payloads), 8)
        self.assertEqual(len(result["errors"]), 6)

    def test_no_prompt_repeats_across_warmup_and_measurement(self):
        self.run_bench()
        prompts = [body["prompt"] for _, body in _Recorder.payloads]
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_requests_are_greedy_and_capped_at_output_len(self):
        self.run_bench()
        _, body = _Recorder.payloads[0]
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 4)

    def test_ignore_eos_is_sent_only_when_asked(self):
        self.run_bench()
        self.assertNotIn("ignore_eos", _Recorder.payloads[0][1])
        _Recorder.payloads = []
        self.run_bench("--ignore-eos")
        self.assertIs(_Recorder.payloads[0][1]["ignore_eos"], True)

    def test_embed_never_sends_ignore_eos(self):
        self.run_bench("--task", "embed", "--ignore-eos")
        self.assertEqual(set(_Recorder.payloads[0][1]), {"text"})

    def test_token_counts_need_a_tokenizer(self):
        _, result = self.run_bench()
        self.assertIsNone(result["total_input_tokens"])
        self.assertIsNone(result["output_throughput"])

    def test_tokenizer_gives_token_counts(self):
        original = bench.load_tokenizer
        bench.load_tokenizer = lambda name: _WordTokenizer() if name else None
        try:
            _, result = self.run_bench("--tokenizer", "word")
        finally:
            bench.load_tokenizer = original
        self.assertEqual(result["total_input_tokens"], 6 * 8)
        # "one two three" is three tokens per completion.
        self.assertEqual(result["total_output_tokens"], 6 * 3)
        self.assertGreater(result["total_token_throughput"], 0)

    def test_embed_task_posts_text_to_embed(self):
        _, result = self.run_bench("--task", "embed")
        path, body = _Recorder.payloads[0]
        self.assertEqual(path, "/embed")
        self.assertEqual(set(body), {"text"})
        self.assertEqual(result["total_output_tokens"], 0)

    def test_failed_measured_requests_exit_1(self):
        _Recorder.fail_after = 5  # 2 warmups + 3 measured succeed
        code, result = self.run_bench("--max-concurrency", "1")
        self.assertEqual(code, 1)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["failed"], 3)

    def test_failed_warmup_exits_2_without_measuring(self):
        _Recorder.fail_after = 0
        code = bench.main([
            "--base-url", self.url, "--num-prompts", "3", "--num-warmups", "1",
            "--input-len", "4"
        ])
        self.assertEqual(code, 2)
        self.assertEqual(len(_Recorder.payloads), 1)

    def test_streaming_metrics_are_rejected(self):
        with self.assertRaises(SystemExit):
            bench.parse_args(
                ["--base-url", self.url, "--percentile-metrics", "ttft,e2el"])


class PercentileTest(unittest.TestCase):

    def test_matches_numpy_linear_interpolation(self):
        # numpy.percentile([1, 2, 3, 4], q) for q = 50, 90, 99.
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertAlmostEqual(bench.percentile(values, 50), 2.5)
        self.assertAlmostEqual(bench.percentile(values, 90), 3.7)
        self.assertAlmostEqual(bench.percentile(values, 99), 3.97)

    def test_single_value(self):
        self.assertEqual(bench.percentile([7.0], 99), 7.0)


class PromptTest(unittest.TestCase):

    def test_tokenizer_prompts_are_exact_length_and_skip_special_ids(self):
        tok = _WordTokenizer()
        vocab = bench.sampling_vocab(tok)
        self.assertNotIn(0, vocab)
        text, n = bench.random_prompt(random.Random(0), 16, tok, vocab)
        self.assertEqual(n, 16)
        self.assertNotIn("<pad>", text)


class GcsTokenizerTest(unittest.TestCase):
    """A gs:// --tokenizer is read from a local copy of its small files."""

    def test_a_hub_id_or_local_path_is_passed_through(self):
        self.assertEqual(bench.local_tokenizer("org/model"), "org/model")
        self.assertEqual(bench.local_tokenizer("/models/m"), "/models/m")

    def test_a_gcs_folder_is_copied_and_progress_kept_off_stdout(self):
        copies = []
        module = types.ModuleType("fetch_weights")

        def fetch_gcs(url, small_only=False):
            copies.append((url, small_only))
            print("WEIGHTS READY")
            return "/tmp/copy"

        module.fetch_gcs = fetch_gcs
        with mock.patch.dict(sys.modules, {"fetch_weights": module}), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.local_tokenizer("gs://b/m"), "/tmp/copy")
        self.assertEqual(copies, [("gs://b/m", True)])
        self.assertEqual(out.getvalue(), "")

    def test_load_tokenizer_reads_the_copy(self):
        module = types.ModuleType("fetch_weights")
        module.fetch_gcs = lambda url, small_only=False: "/tmp/copy"
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = mock.Mock()
        with mock.patch.dict(sys.modules, {
                "fetch_weights": module,
                "transformers": transformers
        }), contextlib.redirect_stderr(io.StringIO()):
            bench.load_tokenizer("gs://b/m")
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "/tmp/copy")

    def test_without_fetch_weights_it_says_what_to_ship(self):
        with mock.patch.dict(sys.modules, {"fetch_weights": None}):
            with self.assertRaisesRegex(ImportError, "fetch_weights.py"):
                bench.local_tokenizer("gs://b/m")


if __name__ == "__main__":
    unittest.main()
