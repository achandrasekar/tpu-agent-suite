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
"""Tests for session trajectory export.

Every test points HOME at a temp dir and builds synthetic transcripts there,
so nothing reads a real session. Uploads go through an injected recorder, so
nothing touches the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import unittest.mock

from framework import runtime, trajectory, utils
from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


def _assistant(blocks: list[dict]) -> dict:
    """One assistant record carrying the given content blocks."""
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": blocks
        }
    }


def _tool_use(tool_id: str, name: str, payload: dict) -> dict:
    """One tool_use content block."""
    return {"type": "tool_use", "id": tool_id, "name": name, "input": payload}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records),
                    encoding="utf-8")


def _slash_command(name: str, args: str = "") -> dict:
    """A user record shaped the way Claude Code logs a typed slash command."""
    text = (f"<command-message>{name}</command-message>\n"
            f"<command-name>/{name}</command-name>\n"
            f"<command-args>{args}</command-args>")
    return {"type": "user", "message": {"role": "user", "content": text}}


# An environment.json in the profiles format, as the manifest records it.
ENVIRONMENT = {
    "revision": 1,
    "default": "default",
    "profiles": {
        "default": {
            "backend": "local"
        }
    },
}


class HomeIsolatedTestCase(unittest.TestCase):
    """Base that redirects HOME so Path.home() resolves into a temp dir."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._prev_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)

    def tearDown(self) -> None:
        """Restore the real HOME so later tests are unaffected."""
        if self._prev_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._prev_home
        self._tmp.cleanup()


class ClaudeCodeResolverTest(HomeIsolatedTestCase):
    """The Claude Code resolver: exact cut, nested subagents, pre-cut exclusion."""

    SESSION = "sess-1"

    def _build(self, records: list[dict], agents: dict[str, dict]) -> None:
        """Lay out a project transcript plus its durable subagents directory."""
        project = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(project / f"{self.SESSION}.jsonl", records)
        subagents = project / self.SESSION / "subagents"
        subagents.mkdir(parents=True, exist_ok=True)
        for agent_id, spec in agents.items():
            (subagents / f"agent-{agent_id}.meta.json").write_text(
                json.dumps(spec["meta"]), encoding="utf-8")
            _write_jsonl(subagents / f"agent-{agent_id}.jsonl",
                         spec.get("records", [{
                             "type": "user"
                         }]))

    def test_cut_is_the_most_recent_activation(self) -> None:
        """A second activation moves the cut forward, dropping the first round."""
        records = [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
            _assistant([_tool_use("t1", "Bash", {})]),
            _assistant([_tool_use("t2", "Skill", {"skill": "tpu-agent"})]),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertTrue(bundle.cut.detected)
        self.assertEqual(bundle.cut.basis, "skill-tool-use")
        self.assertEqual(bundle.cut.key, 2)

    def test_slash_command_is_an_activation(self) -> None:
        """A user-typed /tpu-agent is recorded as tagged text, not a tool call."""
        records = [
            _assistant([_tool_use("t0", "Bash", {})]),
            _slash_command("tpu-agent", "--note hi"),
            _assistant([_tool_use("t2", "Bash", {})]),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertTrue(bundle.cut.detected)
        self.assertEqual(bundle.cut.basis, "slash-command")
        self.assertEqual(bundle.cut.key, 1)

    def test_latest_activation_wins_across_shapes(self) -> None:
        """A slash command after a Skill call moves the cut, and vice versa."""
        records = [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
            _slash_command("tpu-agent"),
            _assistant([_tool_use("t2", "Skill", {"skill": "tpu-agent"})]),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertEqual((bundle.cut.basis, bundle.cut.key),
                         ("skill-tool-use", 2))
        self._build(records[:2], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertEqual((bundle.cut.basis, bundle.cut.key),
                         ("slash-command", 1))

    def test_slash_command_in_text_blocks_counts_but_tool_results_do_not(
            self) -> None:
        """Block-list content is read for text; an echoed transcript is not."""
        tag = "<command-name>/tpu-agent</command-name>"
        as_text = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": tag
                }]
            }
        }
        as_tool_result = {
            "type": "user",
            "message": {
                "role":
                "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "t0",
                    "content": tag
                }]
            }
        }
        self._build([as_text], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertEqual((bundle.cut.basis, bundle.cut.key),
                         ("slash-command", 0))
        self._build([as_tool_result], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertFalse(bundle.cut.detected)

    def test_export_command_is_not_an_activation(self) -> None:
        """The export skill shares the prefix but must not cut the session."""
        records = [
            _assistant([_tool_use("t0", "Bash", {})]),
            _slash_command("tpu-agent-export-trajectory", "--all"),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertFalse(bundle.cut.detected)
        self.assertEqual(bundle.cut.basis, "none")

    def test_excludes_pre_cut_and_includes_nested_subagents(self) -> None:
        """Delegations before the cut are dropped; nested ones are kept."""
        records = [
            _assistant([_tool_use("pre", "Agent", {})]),
            _assistant([_tool_use("cut", "Skill", {"skill": "tpu-agent"})]),
            _assistant([_tool_use("post", "Agent", {})]),
        ]
        agents = {
            # Launched before the cut, so out of scope.
            "old": {
                "meta": {
                    "toolUseId": "pre",
                    "agentType": "old",
                    "spawnDepth": 1
                }
            },
            # Launched after the cut, and itself launches a nested agent.
            "kid": {
                "meta": {
                    "toolUseId": "post",
                    "agentType": "tpu-model-onboarding",
                    "spawnDepth": 1
                },
                "records": [_assistant([_tool_use("deep", "Agent", {})])],
            },
            # Reachable only through "kid", which is what nesting must find.
            "grandkid": {
                "meta": {
                    "toolUseId": "deep",
                    "agentType": "tpu-kernel-development",
                    "spawnDepth": 2
                }
            },
        }
        self._build(records, agents)
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        labels = sorted(ref.label for ref in bundle.transcripts)
        self.assertEqual(
            labels, ["root", "tpu-kernel-development", "tpu-model-onboarding"])

    def test_missing_transcript_is_an_error(self) -> None:
        """A session with no transcript cannot be exported."""
        with self.assertRaises(TpuAgentSuiteError):
            trajectory.resolve_claude_code("absent", Path("/w"), "tpu-agent")

    def test_no_activation_reports_undetected(self) -> None:
        """Without an activation the cut is undetected rather than guessed."""
        self._build([_assistant([_tool_use("t0", "Bash", {})])], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertFalse(bundle.cut.detected)
        self.assertEqual(bundle.cut.basis, "none")

    def test_harness_version_and_models_come_from_the_transcript(self) -> None:
        """Version and model are read back rather than shelled out for."""
        records = [
            {
                "type": "assistant",
                "version": "2.1.267",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "model": "claude-opus-5"
                }
            },
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        self.assertEqual(bundle.harness_version, "2.1.267")
        self.assertEqual(bundle.models["observed"], ["claude-opus-5"])


class GoogleHarnessResolverTest(HomeIsolatedTestCase):
    """The Google harness resolver: step-indexed cut and recursive descendants."""

    ROOT = "conv-root"

    def _brain(self,
               conversation_id: str,
               root: str = "antigravity-cli") -> Path:
        """Brain directory for one conversation, created on demand."""
        path = self.home / ".gemini" / root / "brain" / conversation_id
        (path / ".system_generated" / "subagents").mkdir(parents=True,
                                                         exist_ok=True)
        return path

    def _transcript(self,
                    conversation_id: str,
                    records: list[dict],
                    root: str = "antigravity-cli") -> None:
        """Write a conversation's full transcript."""
        brain = self._brain(conversation_id, root)
        _write_jsonl(
            brain / ".system_generated" / "logs" / "transcript_full.jsonl",
            records)

    def _descriptor(self, parent: str, child: str, spawn_step: int,
                    type_name: str) -> None:
        """Register `child` as a subagent spawned by `parent`."""
        brain = self._brain(parent)
        (brain / ".system_generated" / "subagents" /
         f"{child}.json").write_text(json.dumps({
             "conversationId": child,
             "subagentDescriptor": {
                 "typeName": type_name
             },
             "spawnStepIndex": spawn_step,
         }),
                                     encoding="utf-8")

    def test_cut_uses_the_skill_file_read(self) -> None:
        """Activation is a view_file of the skill document on this harness."""
        self._transcript(self.ROOT, [
            {
                "step_index": 1,
                "tool_calls": [{
                    "name": "run_command",
                    "args": {}
                }]
            },
            {
                "step_index":
                3,
                "tool_calls": [{
                    "name": "view_file",
                    "args": {
                        "AbsolutePath": "/w/.agents/skills/tpu-agent/SKILL.md"
                    },
                }]
            },
        ])
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        self.assertTrue(bundle.cut.detected)
        self.assertEqual(bundle.cut.basis, "skill-file-read")
        self.assertEqual(bundle.cut.key, 3)
        self.assertEqual(bundle.cut.mode, "step_index")

    def test_descendants_are_filtered_by_spawn_step_then_recursed(
            self) -> None:
        """Pre-cut children are dropped; a kept child's own children come along."""
        self._transcript(self.ROOT, [
            {
                "step_index":
                2,
                "tool_calls": [{
                    "name": "view_file",
                    "args": {
                        "AbsolutePath": "/w/.agents/skills/tpu-agent/SKILL.md"
                    },
                }]
            },
        ])
        self._descriptor(self.ROOT, "early", 1, "stale")
        self._descriptor(self.ROOT, "kept", 5, "tpu-sharding")
        self._transcript("early", [{"step_index": 1}])
        self._transcript("kept", [{"step_index": 1}])
        # Nested below "kept" at a step before the root's cut: still in scope,
        # because the cut applies only where spawning happened.
        self._descriptor("kept", "nested", 1, "tpu-diagnostics")
        self._transcript("nested", [{"step_index": 1}])

        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        labels = sorted(ref.label for ref in bundle.transcripts)
        self.assertEqual(labels, ["root", "tpu-diagnostics", "tpu-sharding"])

    def test_bundle_records_the_harness_it_was_resolved_for(self) -> None:
        """The harness name is passed through, not fixed to one family member."""
        self._transcript(self.ROOT, [{
            "step_index": 1
        }],
                         root="some-google-harness")
        bundle = trajectory.resolve_google_harness("some-google-harness",
                                                   self.ROOT, Path("/w"),
                                                   "tpu-agent")
        self.assertEqual(bundle.harness, "some-google-harness")

    def test_brain_roots_of_another_harness_are_not_searched(self) -> None:
        """A root counts only when its name contains the harness name."""
        self._transcript(self.ROOT, [{
            "step_index": 1
        }],
                         root="other-google-harness")
        with self.assertRaises(TpuAgentSuiteError):
            trajectory.resolve_google_harness("antigravity", self.ROOT,
                                              Path("/w"), "tpu-agent")

    def test_subagents_are_found_under_the_same_harness(self) -> None:
        """A child conversation under another harness's root is not collected."""
        self._transcript(self.ROOT, [{"step_index": 1}])
        self._descriptor(self.ROOT, "kept", 1, "tpu-sharding")
        self._transcript("kept", [{"step_index": 1}])
        self._descriptor(self.ROOT, "foreign", 1, "tpu-diagnostics")
        self._transcript("foreign", [{
            "step_index": 1
        }],
                         root="other-google-harness")
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        labels = sorted(ref.label for ref in bundle.transcripts)
        self.assertEqual(labels, ["root", "tpu-sharding"])

    def test_every_google_harness_is_registered_under_its_own_name(
            self) -> None:
        """resolve_session dispatches each family member with its own name."""
        for harness in utils.GOOGLE_HARNESSES:
            with self.subTest(harness=harness):
                self._transcript(self.ROOT, [{"step_index": 1}], root=harness)
                self.assertTrue(trajectory.supported(harness))
                bundle = trajectory.resolve_session(harness, self.ROOT,
                                                    Path("/w"), "tpu-agent")
                self.assertEqual(bundle.harness, harness)


class SlicingAndCapTest(unittest.TestCase):
    """Generic slicing and the per-artifact size cap."""

    def test_index_mode_slices_by_position(self) -> None:
        """Claude Code cuts are record positions."""
        records = [{"n": 0}, {"n": 1}, {"n": 2}]
        cut = trajectory.Cut(detected=True, basis="skill-tool-use", key=1)
        self.assertEqual(trajectory.slice_root_records(cut, records), [{
            "n": 1
        }, {
            "n": 2
        }])

    def test_field_mode_compares_a_record_field(self) -> None:
        """Google harness cuts compare step_index."""
        records = [{"step_index": 1}, {"step_index": 4}]
        cut = trajectory.Cut(detected=True,
                             basis="skill-file-read",
                             key=3,
                             mode="step_index")
        self.assertEqual(trajectory.slice_root_records(cut, records),
                         [{
                             "step_index": 4
                         }])

    def test_undetected_cut_keeps_everything(self) -> None:
        """With no cut the caller has consented to the whole session."""
        records = [{"n": 0}, {"n": 1}]
        cut = trajectory.Cut(detected=False, basis="user-consented-all")
        self.assertEqual(trajectory.slice_root_records(cut, records), records)

    def test_oversized_artifacts_are_dropped_and_reported(self) -> None:
        """A file over the cap is excluded, but never silently."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "small.txt").write_bytes(b"x" * 10)
            (root / "big.bin").write_bytes(b"x" * 100)
            included, dropped = trajectory.collect_run_files(root, cap=50)
            self.assertEqual([p.name for p in included], ["small.txt"])
            self.assertEqual(len(dropped), 1)
            self.assertIn("cap", dropped[0]["reason"])


class ObjectPrefixTest(unittest.TestCase):
    """The destination path layout."""

    def test_timestamp_and_session_share_the_leaf(self) -> None:
        """Lexical order is chronological; user and host are browsable levels."""
        prefix = trajectory.object_prefix("me@example.com", "host-1",
                                          "2026-09-22T23:36:39+0000", "sess-1")
        self.assertEqual(
            prefix,
            "tpu-agent-trajectories/me@example.com/host-1/20260922T233639Z-sess-1"
        )

    def test_unsafe_characters_are_replaced(self) -> None:
        """Path components never carry separators through from an identifier."""
        prefix = trajectory.object_prefix("a/b", "c d", "2026-01-01T00:00:00Z",
                                          "s")
        self.assertIn("a_b", prefix)
        self.assertIn("c_d", prefix)


class PushFeedbackTest(HomeIsolatedTestCase):
    """End-to-end bundling, with uploads captured rather than sent."""

    SESSION = "sess-push"

    def setUp(self) -> None:
        super().setUp()
        project = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(project / f"{self.SESSION}.jsonl", [
            _assistant([_tool_use("t0", "Bash", {})]),
            _assistant([_tool_use("t1", "Skill", {"skill": "tpu-agent"})]),
        ])
        (project / self.SESSION / "subagents").mkdir(parents=True,
                                                     exist_ok=True)

        # A run namespace holding one small artifact.
        self.run_root = self.home / "run"
        (self.run_root / "artifacts").mkdir(parents=True)
        (self.run_root / "artifacts" / "result.txt").write_text("ok")
        self.uploads: list[tuple[str, str]] = []

    def _push(self, **overrides):
        """Run a push with test defaults and a recording uploader."""
        kwargs = dict(
            harness="claude-code",
            session_id=self.SESSION,
            workspace=Path("/w"),
            run_root=self.run_root,
            run_id="run-1",
            entrypoint="tpu-agent",
            message="compilation error",
            bucket="my-bucket",
            user_id="me@example.com",
            package=f"tpu-agent-suite@{utils.TPU_AGENT_SUITE_VERSION}",
            environment=ENVIRONMENT,
            uploaded_at="2026-09-22T23:36:39+0000",
            uploader=lambda path, bucket, blob: self.uploads.append(
                (blob, path.read_bytes() and blob)),
        )
        kwargs.update(overrides)
        return trajectory.export_trajectory(**kwargs)

    def test_configured_bucket_spellings_agree(self) -> None:
        """`gs://b`, `b/` and `b` name one place; the URI must not double up."""
        for spelling in ("gs://my-bucket", "my-bucket/", "my-bucket"):
            with self.subTest(bucket=spelling):
                summary = self._push(bucket=spelling)
                self.assertTrue(summary["uri"].startswith("gs://my-bucket/"),
                                summary["uri"])
                self.assertNotIn("gs://gs://", summary["uri"])

    def test_uploads_bundle_then_manifest(self) -> None:
        """The manifest lands last, so seeing it implies a complete bundle."""
        summary = self._push()
        names = [blob for blob, _ in self.uploads]
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].endswith("bundle.tar.gz"))
        self.assertTrue(names[1].endswith("manifest.json"))
        self.assertTrue(summary["uri"].startswith("gs://my-bucket/"))

    def test_undetected_cut_refuses_without_consent(self) -> None:
        """Shipping a whole session is the user's call, not the tool's."""
        _write_jsonl(
            self.home / ".claude" / "projects" / "-w" /
            f"{self.SESSION}.jsonl",
            [_assistant([_tool_use("t0", "Bash", {})])])
        with self.assertRaises(TpuAgentSuiteError) as caught:
            self._push()
        self.assertIn("--all", str(caught.exception))

    def test_consent_flag_records_its_basis(self) -> None:
        """With --all the manifest says the whole session was sent by consent."""
        _write_jsonl(
            self.home / ".claude" / "projects" / "-w" /
            f"{self.SESSION}.jsonl",
            [_assistant([_tool_use("t0", "Bash", {})])])
        summary = self._push(include_full_session=True)
        self.assertEqual(summary["cut"]["basis"], "user-consented-all")
        self.assertFalse(summary["cut"]["detected"])

    def test_blank_bucket_is_a_decline(self) -> None:
        """No configured destination means export is off, not broken."""
        with self.assertRaises(TpuAgentSuiteError) as caught:
            self._push(bucket="")
        self.assertIn("bucket", str(caught.exception))

    def test_bundle_contents_and_manifest_fields(self) -> None:
        """The tar carries sliced transcripts and artifacts."""
        captured: dict[str, bytes] = {}

        def recorder(path: Path, bucket: str, blob: str) -> None:
            captured[blob] = path.read_bytes()

        summary = self._push(uploader=recorder)
        tar_blob = next(b for b in captured if b.endswith("bundle.tar.gz"))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "b.tar.gz"
            archive.write_bytes(captured[tar_blob])
            with tarfile.open(archive) as tar:
                names = sorted(tar.getnames())
        self.assertIn("transcripts/root.jsonl", names)
        self.assertIn("artifacts/result.txt", names)

        manifest_blob = next(b for b in captured
                             if b.endswith("manifest.json"))
        manifest = json.loads(captured[manifest_blob])
        self.assertTrue(manifest["note"]["provided"])
        self.assertEqual(manifest["note"]["text"], "compilation error")
        self.assertEqual(manifest["environment"], ENVIRONMENT)
        self.assertEqual(manifest["session"]["run_id"], "run-1")
        # The size reported is the compressed bundle, not the sum of inputs.
        self.assertGreater(manifest["contents"]["bundle_bytes"], 0)
        self.assertEqual(summary["transcripts"], 1)

    def test_empty_message_is_recorded_as_not_provided(self) -> None:
        """The qualitative message is optional and its absence is explicit."""
        captured: dict[str, bytes] = {}
        self._push(message="",
                   uploader=lambda p, b, blob: captured.__setitem__(
                       blob, p.read_bytes()))
        manifest_blob = next(b for b in captured
                             if b.endswith("manifest.json"))
        manifest = json.loads(captured[manifest_blob])
        self.assertFalse(manifest["note"]["provided"])

    def test_bundle_includes_session_journal_when_present(self) -> None:
        """journal.jsonl and experiment.md sit beside artifacts/ and ride in the tar."""
        from framework import journal
        (self.run_root / journal.JOURNAL_FILE).write_text('{"id": "H1"}\n',
                                                          encoding="utf-8")
        (self.run_root / journal.RENDERED_FILE).write_text(
            "# Session experiment journal\n", encoding="utf-8")
        captured: dict[str, bytes] = {}
        self._push(uploader=lambda p, b, blob: captured.__setitem__(
            blob, p.read_bytes()))
        tar_blob = next(b for b in captured if b.endswith("bundle.tar.gz"))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "b.tar.gz"
            archive.write_bytes(captured[tar_blob])
            with tarfile.open(archive) as tar:
                names = sorted(tar.getnames())
        self.assertIn(journal.JOURNAL_FILE, names)
        self.assertIn(journal.RENDERED_FILE, names)


class UnsupportedHarnessTest(unittest.TestCase):
    """Codex and Cursor ship the skill but refuse the export."""

    def test_refusal_names_the_harness(self) -> None:
        """A missing resolver entry is what produces the refusal."""
        for harness in ("codex", "cursor"):
            self.assertFalse(trajectory.supported(harness))
            with self.assertRaises(TpuAgentSuiteError) as caught:
                trajectory.resolve_session(harness, "s", Path("/w"),
                                           "tpu-agent")
            self.assertIn(harness, str(caught.exception))
            self.assertIn("not supported", str(caught.exception))


class TimestampNormalizationTest(unittest.TestCase):
    """Prefixes look the same whichever ISO spelling produced them."""

    def test_fractional_seconds_are_dropped(self) -> None:
        """utc_now_iso() carries microseconds; the prefix must not."""
        prefix = trajectory.object_prefix("u", "h",
                                          "2026-09-23T07:38:28.866752Z", "s")
        self.assertTrue(prefix.endswith("20260923T073828Z-s"), prefix)

    def test_session_id_is_truncated(self) -> None:
        """A UUID session id is cut to eight characters; the manifest keeps it whole."""
        prefix = trajectory.object_prefix(
            "u", "h", "2026-09-24T06:38:08Z",
            "90290a6c-e78b-4c5c-bcfc-d7839d28faa6")
        self.assertTrue(prefix.endswith("/20260924T063808Z-90290a6c"), prefix)

    def test_offset_and_zulu_agree(self) -> None:
        """Two spellings of the same instant yield the same prefix."""
        offset = trajectory.object_prefix("u", "h", "2026-09-23T07:38:28+0000",
                                          "s")
        zulu = trajectory.object_prefix("u", "h", "2026-09-23T07:38:28Z", "s")
        self.assertEqual(offset, zulu)


class GoogleHarnessTreeShapeTest(HomeIsolatedTestCase):
    """A nested delegation is reported as nested, not flattened."""

    def test_parent_and_depth_are_recorded(self) -> None:
        """The manifest must describe the tree, not just the membership."""
        brain = self.home / ".gemini" / "antigravity-cli" / "brain"

        def conversation(cid: str) -> pathlib.Path:
            """Create one conversation's brain dir with a stub transcript."""
            path = brain / cid / ".system_generated"
            (path / "subagents").mkdir(parents=True, exist_ok=True)
            _write_jsonl(path / "logs" / "transcript_full.jsonl",
                         [{
                             "step_index": 1
                         }])
            return path

        root = conversation("root")
        conversation("child")
        conversation("grandchild")
        _write_jsonl(root / "logs" / "transcript_full.jsonl", [{
            "step_index":
            1,
            "tool_calls": [{
                "name": "view_file",
                "args": {
                    "AbsolutePath": "/w/.agents/skills/tpu-agent/SKILL.md"
                },
            }],
        }])
        (root / "subagents" / "child.json").write_text(json.dumps({
            "conversationId":
            "child",
            "subagentDescriptor": {
                "typeName": "tpu-sharding"
            },
            "spawnStepIndex":
            2,
        }),
                                                       encoding="utf-8")
        (brain / "child" / ".system_generated" / "subagents" /
         "grandchild.json").write_text(json.dumps({
             "conversationId": "grandchild",
             "subagentDescriptor": {
                 "typeName": "tpu-kernel-development"
             },
             "spawnStepIndex": 1,
         }),
                                       encoding="utf-8")

        bundle = trajectory.resolve_google_harness("antigravity", "root",
                                                   Path("/w"), "tpu-agent")
        by_label = {ref.label: ref for ref in bundle.transcripts}
        self.assertEqual(by_label["tpu-sharding"].parent_id, "root")
        self.assertEqual(by_label["tpu-sharding"].depth, 1)
        self.assertEqual(by_label["tpu-kernel-development"].parent_id, "child")
        self.assertEqual(by_label["tpu-kernel-development"].depth, 2)


class RuntimeCliTest(HomeIsolatedTestCase):
    """The `trajectory export` CLI path a compiled skill actually invokes."""

    RUN_ID = "20260923T070000Z-0123456789ab"
    SESSION = "cli-session"

    def setUp(self) -> None:
        super().setUp()
        # Central store and workspace both live under the redirected HOME.
        self._prev_store = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.home / "store")

        self.workspace = self.home / "ws"
        marker = self.workspace / ".tpu-agent-suite"
        marker.mkdir(parents=True)
        (marker / "current_run_id").write_text(self.RUN_ID, encoding="utf-8")
        utils.write_json(
            utils.get_sessions_root() / f"{self.SESSION}.json", {
                "session_id": self.SESSION,
                "run_id": self.RUN_ID,
                "created_at": "2026-09-23T07:00:00Z",
            })

        run_root = utils.get_run_root(self.RUN_ID)
        (run_root / "artifacts").mkdir(parents=True)
        (run_root / "artifacts" / "out.txt").write_text("result")

        _write_jsonl(
            self.home / ".claude" / "projects" / "-ws" /
            f"{self.SESSION}.jsonl",
            [_assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})])])

    def tearDown(self) -> None:
        """Restore the central store pointer, then the HOME redirect."""
        if self._prev_store is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = self._prev_store
        super().tearDown()

    def _run(self, *argv: str) -> tuple[int, str]:
        """Invoke the CLI with stdout captured, so tests stay readable."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = runtime.main(list(argv))
        return code, buffer.getvalue()

    def test_export_resolves_session_and_run_without_flags(self) -> None:
        """The session is recovered from the run_id."""
        with unittest.mock.patch.object(trajectory.bundle,
                                        "verify_bucket_exists"), \
             unittest.mock.patch.object(trajectory.bundle,
                                        "_default_uploader"):
            code, output = self._run("trajectory", "export", "--harness",
                                     "claude-code", "--workspace",
                                     str(self.workspace), "--run-id",
                                     self.RUN_ID, "--bucket", "gs://my-bucket",
                                     "--user-id", "tester", "--note",
                                     "it broke", "--json")
        self.assertEqual(code, 0)
        summary = json.loads(output)
        # The user id shapes the object path, so it must appear in it.
        self.assertIn("/tester/", summary["objects"][0]["name"])
        self.assertEqual(summary["cut"]["basis"], "skill-tool-use")

    def test_target_resolution_picks_the_session_naming_the_run(self) -> None:
        """The session_id is looked up from the given run_id."""
        session_id, run_id, run_root = runtime.resolve_trajectory_target(
            run_id=self.RUN_ID)
        self.assertEqual(session_id, self.SESSION)
        self.assertEqual(run_id, self.RUN_ID)
        self.assertIsNotNone(run_root)

    def test_unknown_session_is_an_error(self) -> None:
        """Without a session record there is nothing to export."""
        with self.assertRaises(TpuAgentSuiteError):
            runtime.resolve_trajectory_target(session_id="no-such-session")

    def test_passed_run_selects_its_own_session(self) -> None:
        """A second window overwriting the marker does not move the export."""
        other_run = "20260923T080000Z-ba9876543210"
        runtime.record_session_run("second-window", other_run)
        utils.record_active_run(other_run, self.workspace)
        session_id, run_id, run_root = runtime.resolve_trajectory_target(
            run_id=self.RUN_ID)
        self.assertEqual(session_id, self.SESSION)
        self.assertEqual(run_id, self.RUN_ID)
        self.assertEqual(run_root, utils.get_run_root(self.RUN_ID))

    def test_passed_run_without_a_session_is_refused(self) -> None:
        """A run no session record names is not exported."""
        unknown = "20260923T090000Z-000000000000"
        with self.assertRaises(TpuAgentSuiteError) as raised:
            runtime.resolve_trajectory_target(run_id=unknown)
        self.assertIn(unknown, str(raised.exception))

    def test_misspelled_harness_is_rejected_as_an_unknown_name(self) -> None:
        """A typo is an invalid choice, not a harness awaiting support.

        Without the argparse constraint both produce the same "not supported
        at this time" message, which reads as "not ready yet" for what is
        actually a misspelling.
        """
        with self.assertRaises(SystemExit):
            self._run("trajectory", "export", "--harness", "claude",
                      "--workspace", str(self.workspace))

    def test_unsupported_harness_exits_nonzero(self) -> None:
        """Codex reports the refusal through the CLI, not a traceback."""
        code, _ = self._run("trajectory", "export", "--harness", "codex",
                            "--workspace", str(self.workspace))
        self.assertEqual(code, 1)


class TreeJoinsOnAgentIdsTest(HomeIsolatedTestCase):
    """Both harnesses describe the delegation tree the same way."""

    def test_claude_code_root_is_nameable_as_a_parent(self) -> None:
        """A top-level subagent names the root, so the tree joins on ids."""
        project = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(project / "s.jsonl", [
            _assistant([_tool_use("cut", "Skill", {"skill": "tpu-agent"})]),
            _assistant([_tool_use("call", "Agent", {})]),
        ])
        subagents = project / "s" / "subagents"
        subagents.mkdir(parents=True)
        (subagents / "agent-a.meta.json").write_text(json.dumps({
            "toolUseId": "call",
            "agentType": "tpu-kernel-development",
            "spawnDepth": 1
        }),
                                                     encoding="utf-8")
        _write_jsonl(subagents / "agent-a.jsonl", [{"type": "user"}])

        bundle = trajectory.resolve_claude_code("s", Path("/w"), "tpu-agent")
        root = next(r for r in bundle.transcripts if r.kind == "root")
        child = next(r for r in bundle.transcripts if r.kind == "subagent")
        self.assertEqual(root.agent_id, "s")
        self.assertEqual(child.parent_id, root.agent_id)
        self.assertEqual(root.parent_id, "")


class GoogleHarnessTranscriptFallbackTest(HomeIsolatedTestCase):
    """The full transcript is preferred, the compact one is a usable stand-in.

    agy's `--print` mode writes no `transcript_full.jsonl` at all, so a
    resolver that only ever names that file silently yields nothing.
    """

    ROOT = "fallback-root"

    def _logs(self) -> pathlib.Path:
        """The conversation's logs directory, created on demand."""
        logs = (self.home / ".gemini" / "antigravity-cli" / "brain" /
                self.ROOT / ".system_generated" / "logs")
        logs.mkdir(parents=True, exist_ok=True)
        (logs.parent / "subagents").mkdir(parents=True, exist_ok=True)
        return logs

    def test_full_transcript_is_preferred(self) -> None:
        """When both exist the untruncated one wins."""
        logs = self._logs()
        _write_jsonl(logs / "transcript_full.jsonl", [{"step_index": 1}])
        _write_jsonl(logs / "transcript.jsonl", [{"step_index": 1}])
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        root = bundle.transcripts[0]
        self.assertEqual(root.source, "transcript_full")
        self.assertTrue(root.path.name.endswith("transcript_full.jsonl"))

    def test_compact_transcript_is_used_when_full_is_absent(self) -> None:
        """Rather than naming a file that does not exist and reading nothing."""
        logs = self._logs()
        _write_jsonl(logs / "transcript.jsonl", [{"step_index": 1}])
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        root = bundle.transcripts[0]
        self.assertEqual(root.source, "transcript")
        self.assertGreater(root.bytes, 0)

    def test_the_fallback_is_recorded_not_silent(self) -> None:
        """A cut missed because of truncation must be explicable afterwards."""
        logs = self._logs()
        _write_jsonl(logs / "transcript.jsonl", [{"step_index": 1}])
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        self.assertTrue(any("compact transcript" in m for m in bundle.missing),
                        bundle.missing)

    def test_neither_present_names_the_preferred_one(self) -> None:
        """The error should say what was looked for, not the fallback."""
        self._logs()
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   Path("/w"), "tpu-agent")
        self.assertTrue(
            any("transcript_full.jsonl" in m for m in bundle.missing),
            bundle.missing)


class GoogleHarnessCycleGuardTest(HomeIsolatedTestCase):
    """The descendant walk terminates on malformed descriptor graphs."""

    def _conversation(self, cid: str, step: int = 1) -> pathlib.Path:
        """Create a conversation with a stub transcript, return its brain dir."""
        brain = self.home / ".gemini" / "antigravity-cli" / "brain" / cid
        sysgen = brain / ".system_generated"
        (sysgen / "subagents").mkdir(parents=True, exist_ok=True)
        _write_jsonl(sysgen / "logs" / "transcript_full.jsonl",
                     [{
                         "step_index": step
                     }])
        return brain

    def _spawn(self, parent: str, child: str, step: int = 1) -> None:
        """Register `child` as spawned by `parent`."""
        path = (self.home / ".gemini" / "antigravity-cli" / "brain" / parent /
                ".system_generated" / "subagents" / f"{child}.json")
        path.write_text(json.dumps({
            "conversationId": child,
            "subagentDescriptor": {
                "typeName": f"agent-{child}"
            },
            "spawnStepIndex": step,
        }),
                        encoding="utf-8")

    def test_a_cycle_terminates(self) -> None:
        """A <-> B would otherwise recurse until the stack limit."""
        self._conversation("a")
        self._conversation("b")
        self._spawn("a", "b")
        self._spawn("b", "a")
        bundle = trajectory.resolve_google_harness("antigravity", "a",
                                                   Path("/w"), "tpu-agent")
        labels = sorted(r.label for r in bundle.transcripts)
        self.assertEqual(labels, ["agent-b", "root"])

    def test_self_referencing_descriptor_terminates(self) -> None:
        """A descriptor naming its own conversation is not a child."""
        self._conversation("solo")
        self._spawn("solo", "solo")
        bundle = trajectory.resolve_google_harness("antigravity", "solo",
                                                   Path("/w"), "tpu-agent")
        self.assertEqual([r.kind for r in bundle.transcripts], ["root"])

    def test_a_shared_child_is_collected_once(self) -> None:
        """Two parents reaching one conversation must not duplicate it."""
        self._conversation("top")
        for cid in ("left", "right", "shared"):
            self._conversation(cid)
        self._spawn("top", "left")
        self._spawn("top", "right")
        self._spawn("left", "shared")
        self._spawn("right", "shared")
        bundle = trajectory.resolve_google_harness("antigravity", "top",
                                                   Path("/w"), "tpu-agent")
        ids = [r.agent_id for r in bundle.transcripts if r.kind == "subagent"]
        self.assertEqual(sorted(ids), ["left", "right", "shared"])
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()


class StageSessionDirTest(HomeIsolatedTestCase):
    """Staging for Harbor: one session per directory, sliced like the bundle."""

    SESSION = "sess-stage"

    def _resolve(self):
        """Resolve a session that shares its project dir with a neighbour."""
        project = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(project / f"{self.SESSION}.jsonl", [
            _assistant([_tool_use("t0", "Bash", {})]),
            _assistant([_tool_use("t1", "Skill", {"skill": "tpu-agent"})]),
            _assistant([_tool_use("t2", "Bash", {})]),
        ])
        # A second session filed beside it. harbor merges every *.jsonl in the
        # directory it is handed, so this is the transcript that must not
        # follow the staged one.
        _write_jsonl(project / "sess-neighbour.jsonl",
                     [_assistant([_tool_use("n0", "Bash", {})])])
        return trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                              "tpu-agent")

    def test_stages_only_the_requested_session(self) -> None:
        """A neighbour in the same project directory is not staged."""
        bundle = self._resolve()
        with tempfile.TemporaryDirectory() as tmp:
            staged = trajectory.stage_session_dir(bundle, Path(tmp))
            names = sorted(p.name for p in staged.glob("*.jsonl"))
        self.assertEqual(names, [f"{self.SESSION}.jsonl"])

    def test_root_transcript_is_sliced_at_the_cut(self) -> None:
        """Staging drops pre-cut records, matching what write_bundle ships."""
        bundle = self._resolve()
        self.assertTrue(bundle.cut.detected)
        with tempfile.TemporaryDirectory() as tmp:
            staged = trajectory.stage_session_dir(bundle, Path(tmp))
            records = [
                json.loads(line)
                for line in (staged /
                             f"{self.SESSION}.jsonl").read_text().splitlines()
                if line.strip()
            ]
        # The pre-cut Bash call is gone; the activation and its work remain.
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["message"]["content"][0]["input"]["skill"],
                         "tpu-agent")

    def test_layout_matches_what_harbor_globs(self) -> None:
        """Subagents land under subagents/, where harbor's rglob looks."""
        project = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(project / f"{self.SESSION}.jsonl", [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
            _assistant([_tool_use("call", "Agent", {})]),
        ])
        subagents = project / self.SESSION / "subagents"
        subagents.mkdir(parents=True, exist_ok=True)
        (subagents / "agent-a1.meta.json").write_text(json.dumps({
            "toolUseId":
            "call",
            "agentType":
            "tpu-kernel-development",
            "spawnDepth":
            1
        }),
                                                      encoding="utf-8")
        _write_jsonl(subagents / "agent-a1.jsonl", [{"type": "user"}])
        bundle = trajectory.resolve_claude_code(self.SESSION, Path("/w"),
                                                "tpu-agent")
        with tempfile.TemporaryDirectory() as tmp:
            staged = trajectory.stage_session_dir(bundle, Path(tmp))
            roots = sorted(p.name for p in staged.glob("*.jsonl"))
            nested = sorted(p.name for p in staged.rglob("subagents/*.jsonl"))
        self.assertEqual(roots, [f"{self.SESSION}.jsonl"])
        self.assertTrue(nested, "subagent transcript should be staged")


class HarborLazyImportTest(unittest.TestCase):
    """Harbor is optional: the bucket path must survive its absence."""

    def test_trajectory_imports_without_harbor(self) -> None:
        """No module-level harbor import, so an env without it still exports."""
        source = (Path(trajectory.__file__).parent / "bundle.py").read_text()
        for line in source.splitlines():
            if line.startswith(("import harbor", "from harbor")):
                self.fail(f"harbor imported at module scope: {line!r}")


class HarborAvailabilityTest(unittest.TestCase):
    """Both halves of harbor are required and can be installed apart."""

    def setUp(self) -> None:
        trajectory.harbor_missing.cache_clear()

    def tearDown(self) -> None:
        trajectory.harbor_missing.cache_clear()

    def test_usable_when_importable_and_on_path(self) -> None:
        """Nothing missing reads as the empty string."""
        with unittest.mock.patch("importlib.util.find_spec",
                                 return_value=object()), \
             unittest.mock.patch("shutil.which", return_value="/usr/bin/harbor"):
            self.assertEqual(trajectory.harbor_missing(), "")

    def test_library_only_install_is_reported(self) -> None:
        """Importable but no binary: the upload would raise FileNotFoundError."""
        with unittest.mock.patch("importlib.util.find_spec",
                                 return_value=object()), \
             unittest.mock.patch("shutil.which", return_value=None):
            self.assertIn("PATH", trajectory.harbor_missing())

    def test_pipx_install_is_reported(self) -> None:
        """Binary on PATH but not importable: the converter would fail."""
        with unittest.mock.patch("importlib.util.find_spec", return_value=None), \
             unittest.mock.patch("shutil.which", return_value="/usr/bin/harbor"):
            self.assertIn("importable", trajectory.harbor_missing())

    def test_availability_is_decided_without_importing(self) -> None:
        """find_spec locates the package; harbor must not land in sys.modules."""
        for name in [
                m for m in sys.modules
                if m == "harbor" or m.startswith("harbor.")
        ]:
            del sys.modules[name]
        trajectory.harbor_missing()
        self.assertNotIn("harbor", sys.modules)


class DestinationSelectionTest(HomeIsolatedTestCase):
    """Harbor replaces the bucket only when the user chose it."""

    def _session(self) -> None:
        proj = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(proj / "sess.jsonl", [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
        ])
        self._record_run()

    def _record_run(self) -> None:
        """Give session `sess` a recorded run with a directory."""
        run_id = "20260923T070000Z-0123456789ab"
        runtime.record_session_run("sess", run_id)
        utils.get_run_root(run_id).mkdir(parents=True)

    def _args(self, **over):
        base = dict(harness="claude-code",
                    workspace=str(self.home),
                    package="",
                    bucket="",
                    user_id="u",
                    share_org="",
                    destination="",
                    session_id="sess",
                    run_id="",
                    entrypoint="tpu-agent",
                    note="",
                    json=False,
                    include_full_session=True)
        base.update(over)
        return unittest.mock.Mock(**base)

    def test_declining_export_does_not_reach_harbor(self) -> None:
        """A blank bucket means declined, so nothing is uploaded anywhere."""
        self._session()
        with unittest.mock.patch.object(trajectory, "export_to_harbor") as hub:
            with self.assertRaises(TpuAgentSuiteError) as caught:
                runtime.handle_trajectory_cli(self._args())
        hub.assert_not_called()
        self.assertIn("bucket", str(caught.exception))

    def test_harbor_is_not_attempted_just_because_it_is_installed(
            self) -> None:
        """Availability is not consent: the default destination is the bucket."""
        self._session()
        with unittest.mock.patch.object(trajectory, "export_to_harbor") as hub, \
             unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(trajectory, "export_trajectory",
                                        return_value={
                                            "uri": "gs://b/p",
                                            "objects": [],
                                            "transcripts": 1,
                                            "cut": {"detected": True, "basis": "skill-tool-use"},
                                            "dropped": [],
                                        }):
            runtime.handle_trajectory_cli(self._args(bucket="gs://b"))
        hub.assert_not_called()

    def test_choosing_harbor_skips_the_bucket(self) -> None:
        """The two destinations are exclusive, not a try-then-fallback."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(
                 trajectory, "export_to_harbor",
                 return_value={"steps": 1, "destination": "https://hub/x"}) as hub, \
             unittest.mock.patch.object(trajectory, "export_trajectory") as gcs:
            runtime.handle_trajectory_cli(self._args(destination="harbor"))
        hub.assert_called_once()
        gcs.assert_not_called()

    def _stage(self, **kwargs):
        """Stand-in for export_to_harbor, i.e. write a file into the job dir."""
        kwargs["job_dir"].mkdir(parents=True)
        (kwargs["job_dir"] / "trajectory.json").write_text("{}")
        self.staged = kwargs["job_dir"]
        return {
            "steps": 1,
            "destination": "https://hub/x",
            "job_dir": str(kwargs["job_dir"])
        }

    def test_harbor_job_is_staged_outside_runs_and_removed(self) -> None:
        """The job should be staged outside to avoid being read by later agents."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(trajectory, "export_to_harbor",
                                        side_effect=self._stage):
            runtime.handle_trajectory_cli(
                self._args(destination="harbor", json=True))
        runs = utils.get_run_root("20260923T070000Z-0123456789ab").parent
        self.assertFalse(self.staged.is_relative_to(runs))
        self.assertFalse(self.staged.parent.exists())

    def test_failed_harbor_upload_keeps_the_staged_job(self) -> None:
        """The error names the staged job, so it should be kept there."""
        self._session()

        def fail(**kwargs):
            self._stage(**kwargs)
            raise TpuAgentSuiteError("upload failed")

        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(trajectory, "export_to_harbor",
                                        side_effect=fail):
            with self.assertRaises(TpuAgentSuiteError):
                runtime.handle_trajectory_cli(self._args(destination="harbor"))
        self.assertTrue((self.staged / "trajectory.json").is_file())
        shutil.rmtree(self.staged.parent)

    def test_unusable_harbor_is_an_error_not_a_silent_bucket_export(
            self) -> None:
        """Choosing harbor and not having it is reported, not worked around."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value="the `harbor` command is"
                                        " not on PATH"), \
             unittest.mock.patch.object(trajectory, "export_trajectory") as gcs:
            with self.assertRaises(TpuAgentSuiteError) as caught:
                runtime.handle_trajectory_cli(self._args(destination="harbor"))
        gcs.assert_not_called()
        self.assertIn("PATH", str(caught.exception))

    def test_all_refusal_is_not_disguised_as_an_outage(self) -> None:
        """A missing cut raises its own message instead of a fake fallback."""
        proj = self.home / ".claude" / "projects" / "-w"
        # No skill activation, so the cut cannot be found.
        _write_jsonl(proj / "sess.jsonl",
                     [_assistant([_tool_use("t0", "Bash", {})])])
        self._record_run()
        with unittest.mock.patch.object(trajectory,
                                        "harbor_missing",
                                        return_value=""):
            with self.assertRaises(TpuAgentSuiteError) as caught:
                runtime.handle_trajectory_cli(
                    self._args(destination="harbor",
                               include_full_session=False))
        self.assertIn("--all", str(caught.exception))


class HarborPayloadTest(HomeIsolatedTestCase):
    """With no bucket behind it, the job must carry everything the bucket would."""

    def test_manifest_and_bundle_ride_along(self) -> None:
        """The note, the consent basis and the run's artifacts are not dropped."""
        proj = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(proj / "sess.jsonl", [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
        ])
        run_root = self.home / "run"
        (run_root / "artifacts").mkdir(parents=True)
        (run_root / "artifacts" / "log.txt").write_text("boom",
                                                        encoding="utf-8")

        bundle = trajectory.resolve_for_export("claude-code", "sess",
                                               Path("/w"), "tpu-agent", True)
        job = self.home / "job"
        trial = job / "sess__session"
        with unittest.mock.patch.object(trajectory.bundle, "_build_trajectory",
                                        return_value=unittest.mock.Mock(steps=[])), \
             unittest.mock.patch.object(trajectory.bundle, "write_job_dir",
                                        side_effect=lambda *a, **k:
                                        (trial.mkdir(parents=True,
                                                     exist_ok=True), trial)[1]), \
             unittest.mock.patch.object(trajectory.bundle, "subprocess") as sp:
            sp.run.return_value = unittest.mock.Mock(returncode=0,
                                                     stdout="https://hub/x",
                                                     stderr="")
            summary = trajectory.export_to_harbor(bundle=bundle,
                                                  job_dir=job,
                                                  run_root=run_root,
                                                  run_id="r1",
                                                  message="a note",
                                                  package="p",
                                                  environment={},
                                                  share_org="my-team")

        self.assertTrue((trial / trajectory.MANIFEST_NAME).is_file())
        self.assertTrue((trial / trajectory.BUNDLE_NAME).is_file())
        manifest = json.loads((trial / trajectory.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["note"]["text"], "a note")
        self.assertTrue(manifest["note"]["provided"])
        self.assertEqual(manifest["cut"]["basis"], "skill-tool-use")
        with tarfile.open(trial / trajectory.BUNDLE_NAME) as tar:
            names = tar.getnames()
        self.assertIn("artifacts/log.txt", names)
        self.assertEqual(summary["destination"], "https://hub/x")

    def test_sharing_passes_yes_and_a_timeout(self) -> None:
        """--share-org prompts without --yes, and a captured prompt hangs."""
        proj = self.home / ".claude" / "projects" / "-w"
        _write_jsonl(proj / "sess.jsonl", [
            _assistant([_tool_use("t0", "Skill", {"skill": "tpu-agent"})]),
        ])
        bundle = trajectory.resolve_for_export("claude-code", "sess",
                                               Path("/w"), "tpu-agent", True)
        job = self.home / "job2"
        trial = job / "t"
        with unittest.mock.patch.object(trajectory.bundle, "_build_trajectory",
                                        return_value=unittest.mock.Mock(steps=[])), \
             unittest.mock.patch.object(trajectory.bundle, "write_job_dir",
                                        side_effect=lambda *a, **k:
                                        (trial.mkdir(parents=True,
                                                     exist_ok=True), trial)[1]), \
             unittest.mock.patch.object(trajectory.bundle, "subprocess") as sp:
            sp.TimeoutExpired = subprocess.TimeoutExpired
            sp.run.return_value = unittest.mock.Mock(returncode=0,
                                                     stdout="",
                                                     stderr="")
            trajectory.export_to_harbor(bundle=bundle,
                                        job_dir=job,
                                        run_root=None,
                                        run_id="",
                                        message="",
                                        package="",
                                        environment={},
                                        share_org="my-team")
            cmd = sp.run.call_args[0][0]
            kwargs = sp.run.call_args[1]
        self.assertIn("--yes", cmd)
        self.assertEqual(cmd[cmd.index("--share-org") + 1], "my-team")
        self.assertEqual(kwargs["timeout"],
                         trajectory.bundle.UPLOAD_TIMEOUT_SECONDS)


class BlankBucketDeclinesTest(HomeIsolatedTestCase):
    """Naming no bucket is a change of mind, not a broken install."""

    def test_blank_bucket_exports_nowhere(self) -> None:
        """Agreeing to share and naming nothing leaves no destination. That
        is reported before a missing starting point, so the user is asked
        whether to share before whether to share the whole session."""
        _write_jsonl(
            self.home / ".claude" / "projects" / "-w" / "s.jsonl",
            [_assistant([{
                "type": "text",
                "text": "no tpu-agent work"
            }])])
        with self.assertRaises(TpuAgentSuiteError) as caught:
            trajectory.export_trajectory(harness="claude-code",
                                         session_id="s",
                                         workspace=Path("/w"),
                                         run_root=None,
                                         run_id="",
                                         entrypoint="tpu-agent",
                                         message="",
                                         bucket="",
                                         user_id="u",
                                         package="",
                                         environment={})
        self.assertIn("no destination is configured", str(caught.exception))
