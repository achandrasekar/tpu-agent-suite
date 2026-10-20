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
"""Unit tests for scripts/tpu-diagnostics/capture_vllm_profile.py."""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import threading
import unittest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import capture_vllm_profile  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "capture_vllm_profile", _SCRIPTS_DIR / "capture_vllm_profile.py")
    capture_vllm_profile = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(capture_vllm_profile)


class _FakeVllmServer(http.server.BaseHTTPRequestHandler):

    def do_POST(self):  # noqa: N802
        if self.path in ("/start_profile", "/stop_profile"):
            self.server.posts.append(self.path)
            self.send_response(200)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


class CaptureVllmProfileTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.pdir = self.root / "vllm_profile"
        self.pdir.mkdir()
        self.out = self.root / "out"

        self.vllm_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                        _FakeVllmServer)
        self.vllm_srv.posts = []
        threading.Thread(target=self.vllm_srv.serve_forever,
                         daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.vllm_srv.server_port}"

        pdir = self.pdir

        class _DirHandler(http.server.SimpleHTTPRequestHandler):

            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(pdir), **kw)

            def log_message(self, *args):
                pass

        self.files_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                         _DirHandler)
        threading.Thread(target=self.files_srv.serve_forever,
                         daemon=True).start()
        self.files_url = f"http://127.0.0.1:{self.files_srv.server_port}"

    def tearDown(self):
        self.vllm_srv.shutdown()
        self.vllm_srv.server_close()
        self.files_srv.shutdown()
        self.files_srv.server_close()
        self.tmp.cleanup()

    def _run(self, *argv: str) -> tuple[int, dict]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = capture_vllm_profile.main(list(argv))
        return rc, json.loads(buf.getvalue())

    def _write_trace(self,
                     phase: str,
                     ts: str,
                     payload: bytes = b"/device:TPU:0") -> None:
        d = self.pdir / phase / "plugins" / "profile" / ts
        d.mkdir(parents=True, exist_ok=True)
        (d / "rank0_host.xplane.pb").write_bytes(payload)

    def test_pure_helpers(self):
        self.assertEqual(
            capture_vllm_profile.default_files_url("http://my-job:8000"),
            "http://my-job:8790")
        self.assertEqual(
            capture_vllm_profile._settle(lambda: {}, quiet_s=0.0,
                                         limit_s=10.0), (False, {}))

    def test_snapshot_then_http_collect_excludes_earlier_session(self):
        self._write_trace("prefill_only", "old_ts")
        snap = self.root / "before.json"
        rc, snap_out = self._run("--files-url", self.files_url, "--snapshot",
                                 "--snapshot-file", str(snap))
        self.assertEqual(rc, 0)
        self.assertEqual(
            snap_out["files"],
            ["prefill_only/plugins/profile/old_ts/rank0_host.xplane.pb"])

        self._write_trace("prefill_only", "new_ts")
        self._write_trace("decode_only", "new_ts")
        rc, res = self._run("--files-url", self.files_url, "--snapshot-file",
                            str(snap), "--out-dir", str(self.out),
                            "--settle-s", "0")
        self.assertEqual(rc, 0)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["phases"], ["decode_only", "prefill_only"])
        self.assertEqual(res["xplane_files"], 2)
        self.assertEqual(res["xplane_files_with_tpu"], 2)
        self.assertFalse(
            (self.out /
             "prefill_only/plugins/profile/old_ts/rank0_host.xplane.pb"
             ).exists())
        self.assertTrue(
            (self.out /
             "prefill_only/plugins/profile/new_ts/rank0_host.xplane.pb"
             ).exists())

    def test_load_cmd_toggles_start_and_stop_profile_and_collects(self):
        writer = self.root / "write_trace.py"
        writer.write_text(
            f"import pathlib\n"
            f"d = pathlib.Path({str(self.pdir)!r}) / 'decode_only/plugins/profile/ts1'\n"
            f"d.mkdir(parents=True, exist_ok=True)\n"
            f"(d / 'rank0.xplane.pb').write_bytes(b'/device:TPU:0')\n",
            encoding="utf-8",
        )
        rc, res = self._run(
            "--base-url",
            self.base_url,
            "--files-url",
            self.files_url,
            "--out-dir",
            str(self.out),
            "--settle-s",
            "0",
            "--load-cmd",
            f"{sys.executable} {writer}",
        )
        self.assertEqual(rc, 0)
        self.assertEqual(self.vllm_srv.posts,
                         ["/start_profile", "/stop_profile"])
        self.assertEqual(res["phases"], ["decode_only"])
        self.assertEqual(res["xplane_files_with_tpu"], 1)

    def test_host_only_trace_fails_validation(self):
        self._write_trace("decode_only", "ts_host", payload=b"/host:CPU")
        rc, res = self._run("--profile-dir", str(self.pdir), "--out-dir",
                            str(self.out), "--settle-s", "0")
        self.assertEqual(rc, 1)
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["xplane_files"], 1)
        self.assertEqual(res["xplane_files_with_tpu"], 0)
        self.assertIn("no /device:TPU: plane", res["error"])


if __name__ == "__main__":
    unittest.main()
