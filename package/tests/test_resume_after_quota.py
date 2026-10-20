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
"""`RESOURCE_EXHAUSTED` is a status name, not a cause.

The Stop hook resumes a conversation the model backend refused, and must not
resume one the *TPU program* failed. Both arrive under the same status name, so
the distinction is a regex, and a regex that drifts silently re-enables a
five-minute retry loop against a kernel that will never fit. The same rule is
applied by the xprof eval's `test.sh`, on the other side of the container
boundary; the last test here pins the two copies together.
"""

import importlib.util
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_HOOK = (_ROOT / "package" / "scripts" / "tpu-agent" / "resume_after_quota.py")
_spec = importlib.util.spec_from_file_location("resume_after_quota", _HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


# Every check below asks the one question the hook asks: retry, or let it stop?
def retryable(text: str) -> bool:
    return bool(
        hook.TRANSIENT.search(text)) and not hook.WORKLOAD_FAULT.search(text)


class TestBackendRefusalsAreRetried(unittest.TestCase):

    def test_quota_exhaustion(self):
        self.assertTrue(
            retryable(
                "429 RESOURCE_EXHAUSTED: You exceeded your current quota"))

    def test_backend_unavailable(self):
        self.assertTrue(retryable("503 UNAVAILABLE: model is overloaded"))

    def test_rate_limit_prose(self):
        self.assertTrue(retryable("The model is experiencing high demand"))


class TestWorkloadFaultsAreNot(unittest.TestCase):
    """A TPU compile or allocation failure is the agent's program being wrong.
  Waiting cannot help, and the hook used to spend five minutes proving it."""

    def test_compile_time_vmem_oom(self):
        self.assertFalse(
            retryable("RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom: "
                      "Requested 40.00M, limit 16.00M"))

    def test_any_e_coded_compiler_status(self):
        self.assertFalse(retryable("RESOURCE_EXHAUSTED: E2100: something new"))

    def test_runtime_hbm_allocation_carries_no_e_code(self):
        """The runtime form the narrower pattern used to miss entirely, which
    would have been retried five times and then counted as an outage."""
        self.assertFalse(
            retryable("RESOURCE_EXHAUSTED: Attempting to allocate 3.50G. "
                      "That was not possible. There are 1.20G free."))

    def test_bare_vmem_oom_any_case(self):
        self.assertFalse(retryable("RESOURCE_EXHAUSTED: vmemoom in kernel"))

    def test_out_of_memory_prose(self):
        self.assertFalse(
            retryable(
                "RESOURCE_EXHAUSTED: Out of memory while trying to allocate bytes"
            ))


class TestTheTwoCopiesAgree(unittest.TestCase):
    """The eval verifier ships its own copy of this rule -- it runs inside the
  task container and cannot import the package. Nothing but a test stops the
  two from drifting, and a drifted copy excuses an agent's own compile failure
  as a backend outage."""

    def test_pattern_is_identical_in_the_eval_verifier(self):
        shell = (_ROOT / "eval" / "datasets" / "xprof-attention" /
                 "kernel-tuning" / "tests" / "test.sh").read_text()
        m = re.search(r"WORKLOAD = re\.compile\(\n(.*?)\n\s*re\.IGNORECASE\)",
                      shell, re.S)
        self.assertIsNotNone(m, "WORKLOAD not found in test.sh")
        shipped = "".join(re.findall(r'r?"([^"]*)"', m.group(1)))
        self.assertEqual(shipped, hook.WORKLOAD_FAULT.pattern)


if __name__ == "__main__":
    unittest.main()
