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
"""Tests for staging.py (eval/staging.py)."""

import unittest
from unittest.mock import patch

from eval.staging import (compute_staging_digest, find_repo_root,
                          stage_composite_package)


class TestStaging(unittest.TestCase):
    """Test suite for composite staging digest calculation and staging flows."""

    def setUp(self) -> None:
        self.repo_root = find_repo_root()

    def test_digest_deterministic_and_unique(self) -> None:
        """Digest calculation is deterministic and sensitive to harness and answers."""
        d1 = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
        )
        d2 = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
        )
        # Deterministic
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 16)

        # A subset must not reuse the whole package's cache.
        d_subset = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
            skills=["tpu-agent"],
        )
        self.assertNotEqual(d1, d_subset)

        # Sensitive to harness
        d_diff_harness = compute_staging_digest(
            self.repo_root,
            harness="antigravity",
            compute_answers={"backend": "local"},
        )
        self.assertNotEqual(d1, d_diff_harness)

        # Sensitive to compute answers
        d_diff_answers = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={
                "backend": "gke_pod",
                "pod": "test-pod"
            },
        )
        self.assertNotEqual(d1, d_diff_answers)

    def test_a_control_arm_stages_nothing(self) -> None:
        """A control arm returns is_empty=True without compiling anything."""
        with patch("subprocess.run") as run:
            res = stage_composite_package("claude-code", control_arm=True)
        self.assertTrue(res.is_empty)
        run.assert_not_called()

    def test_a_subset_is_compiled_alone(self) -> None:
        """The subset mode passes its skills to the compiler."""
        with patch("subprocess.run") as run:
            stage_composite_package("claude-code",
                                    skills=["tpu-agent"],
                                    repo_root=self.repo_root)
        [translate] = [
            c.args[0] for c in run.call_args_list if "translate" in c.args[0]
        ]
        self.assertEqual("tpu-agent",
                         translate[translate.index("--skills") + 1])

    def test_the_whole_package_is_compiled(self) -> None:
        """No --skills: every skill package.release.json releases is built."""
        with patch("subprocess.run") as run:
            stage_composite_package("claude-code", repo_root=self.repo_root)
        [translate] = [
            c.args[0] for c in run.call_args_list if "translate" in c.args[0]
        ]
        self.assertNotIn("--skills", translate)

    def test_central_pkg_dir_scoped_to_staged_root(self) -> None:
        """Verify central package directory is scoped inside staged_root to prevent cross-run pollution."""
        with patch("subprocess.run"):
            res = stage_composite_package(
                harness="claude-code",
                repo_root=self.repo_root,
            )
            self.assertTrue(
                str(res.central_pkg_dir).startswith(str(res.staged_root)))
            self.assertIn("packages/tpu-agent-suite@0.1.0",
                          str(res.central_pkg_dir))


if __name__ == "__main__":
    unittest.main()
