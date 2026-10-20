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
"""Unit tests for standalone runtime operations."""

import contextlib
import functools
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any
from unittest import mock

from framework import compiler, runtime, utils

Path = pathlib.Path

_no_lease_sweep = mock.patch("framework.compute.lease.spawn_sweep")


def setUpModule():
    # Every builtin start-run starts a detached sweep of the real job store.
    _no_lease_sweep.start()


def tearDownModule():
    _no_lease_sweep.stop()


class TestRuntime(unittest.TestCase):
    """Unit tests for runtime namespace allocation, hooks, and validations."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()

        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        # Set up installed package structure: the payload under its version,
        # and a user-scope environment.json at the central store's root.
        self.pkg_root = self.central_home / "packages" / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        (self.central_home / "environment.json").write_text(json.dumps({
            "revision":
            1,
            "target":
            "local"
        }),
                                                            encoding="utf-8")
        (self.pkg_root / "schemas").mkdir()
        (self.pkg_root / "schemas" / "out.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string"
                    }
                },
                "required": ["status"],
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_cli_accepts_every_declared_hook_event(self):
        """The dispatcher must accept whatever the manifest schema allows.

        These drifted: the schema gained Stop while the CLI kept a hardcoded
        pair, so a hook the harness dutifully invoked died in argument parsing
        with an invalid-choice error. The run it was meant to rescue ended
        anyway, and the failure looked like the backend rather than like us.
        """
        parser = runtime.build_runtime_parser()
        for event in utils.get_supported_hook_events():
            with self.subTest(event=event):
                args = parser.parse_args([
                    "hook-dispatch", "--harness", "antigravity", "--package",
                    "pkg", "--hook", "h", "--event", event, "--command", "true"
                ])
                self.assertEqual(args.event, event)

    def _main_capturing(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = runtime.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _hook_failure_records(self) -> list[dict[str, Any]]:
        log = self.central_home / utils.HOOK_FAILURE_LOG
        if not log.is_file():
            return []
        return [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
        ]

    def test_hook_dispatch_usage_error_is_settled_by_on_failure(self):
        """A hook-dispatch whose arguments do not parse answers by policy,
        in the harness's shape, exit 0, and is recorded as `startup` --
        instead of argparse's exit 2, which Claude Code reads as a block."""
        base = [
            "hook-dispatch", "--harness", "claude-code", "--package",
            "test-pkg@1.0.0", "--hook", "lenient", "--event", "PreToolUse",
            "--command", "true"
        ]
        code, out, err = self._main_capturing(base +
                                              ["--on-failure", "allow"] +
                                              ["--no-such-flag"])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out)["hookSpecificOutput"]["permissionDecision"],
            "allow")
        self.assertIn("hook failed (startup)", err)
        self.assertIn("--no-such-flag", err)
        records = self._hook_failure_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "startup")
        self.assertEqual(records[0]["hook"], "lenient")
        self.assertEqual(records[0]["event"], "PreToolUse")
        self.assertEqual(records[0]["decision"], "allow")

        # Default policy: a deny in the reply, still exit 0.
        code, out, err = self._main_capturing(base + ["--no-such-flag"])
        self.assertEqual(code, 0)
        h_out = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(h_out["permissionDecision"], "deny")
        self.assertIn("--no-such-flag", h_out["permissionDecisionReason"])

    def test_hook_dispatch_usage_error_without_a_shape_exits_by_policy(self):
        """An unknown event (or harness) leaves no shape to answer in, so the
        exit code carries the policy: 0 for allow, 2 for deny. So does a
        shape lookup that fails outright, when the store has lost the schema
        files it reads: the startup path must not crash into a traceback."""
        for extra, expected in ((["--on-failure", "allow"], 0), ([], 2)):
            with self.subTest(policy=extra or "default"):
                code, out, err = self._main_capturing([
                    "hook-dispatch", "--harness", "claude-code", "--package",
                    "p", "--hook", "h", "--event", "NotAnEvent", "--command",
                    "true"
                ] + extra)
                self.assertEqual(code, expected)
                self.assertEqual(out, "")
                self.assertIn("hook failed (startup)", err)
                self.assertIn("NotAnEvent", err)

        # The store has lost event-mappings.json: the strict parser cannot
        # even be built (its --harness choices come from it), and neither
        # can a reply shape. Not a usage error, yet still a startup failure.
        with mock.patch.object(utils,
                               "get_event_mappings",
                               side_effect=utils.TpuAgentSuiteError(
                                   "missing event mappings")):
            code, out, err = self._main_capturing([
                "hook-dispatch", "--harness", "claude-code", "--package", "p",
                "--hook", "h", "--event", "PreToolUse", "--command", "true",
                "--on-failure", "allow"
            ])
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("hook failed (startup)", err)
        self.assertIn("missing event mappings", err)
        self.assertNotIn("Traceback", err)

        # The shape is known but the reply cannot be made (here the failure
        # log refuses the store): the exit code still carries the policy,
        # and the reason says both what failed.
        with mock.patch.object(
                runtime,
                "_dispatch_failure",
                side_effect=utils.TpuAgentSuiteError("store refused")):
            code, out, err = self._main_capturing([
                "hook-dispatch", "--harness", "claude-code", "--package", "p",
                "--hook", "h", "--event", "PreToolUse", "--command", "true",
                "--no-such-flag"
            ])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("--no-such-flag", err)
        self.assertIn("no reply could be shaped", err)
        self.assertIn("store refused", err)
        self.assertNotIn("Traceback", err)

    def test_hook_dispatch_usage_error_on_handback_answers_the_tool_event(
            self):
        """A hand-back registration names its tool event on the command line;
        a usage error is answered in that event's shape, not SubagentStop's,
        or Claude Code would reject the reply and run the tool."""
        code, out, _ = self._main_capturing([
            "hook-dispatch", "--harness", "claude-code", "--package", "p",
            "--hook", "verify", "--event", "SubagentStop", "--handback-event",
            "PreToolUse", "--command", "true", "--no-such-flag"
        ])
        self.assertEqual(code, 0)
        h_out = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(h_out["hookEventName"], "PreToolUse")
        self.assertEqual(h_out["permissionDecision"], "deny")

    def test_only_hook_dispatch_usage_errors_are_settled(self):
        """Other subcommands keep argparse's exit 2, and hook-dispatch's own
        --help still exits 0: neither is a hook that has to answer."""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                runtime.main(["validate"])
        self.assertEqual(ctx.exception.code, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                runtime.main(["hook-dispatch", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_stop_event_carries_why_the_loop_ended(self):
        """Stop names no tool, so normalization must surface the termination
        fields instead: they are the only thing a handler can decide on."""
        normalized = runtime.normalize_inbound_event(
            raw_payload={
                "terminationReason": "error",
                "error": "Error 503, high demand",
                "executionNum": 2,
                "fullyIdle": True,
            },
            harness="antigravity",
            event="Stop",
            package="pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(normalized["event"], "Stop")
        self.assertEqual(normalized["termination"]["reason"], "error")
        self.assertIn("503", normalized["termination"]["error"])
        self.assertEqual(normalized["termination"]["execution_num"], 2)

    def test_stop_event_carries_the_conversation_id(self):
        """A handler needs it to scope state per conversation.

        Without it the resume hook's retry counter was global: the state file
        hashed the empty string, so concurrent runs spent one budget between
        them and a fresh run inherited a stale count.
        """
        normalized = runtime.normalize_inbound_event(
            raw_payload={
                "terminationReason": "error",
                "conversationId": "abc-123"
            },
            harness="antigravity",
            event="Stop",
            package="pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(normalized["context"]["conversation_id"], "abc-123")

    def test_stop_retry_decision_resumes_the_loop(self):
        """'retry' must become the harness's resume signal. Resuming keeps the
        conversation, so the run continues instead of repeating itself."""
        # Antigravity
        resumed = runtime.translate_canonical_output(
            {
                "decision": "retry",
                "reason": "waited for quota"
            },
            harness="antigravity",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(resumed["decision"], "continue")
        self.assertIn("waited", resumed["reason"])

        ended = runtime.translate_canonical_output(
            {"decision": "allow"},
            harness="antigravity",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(ended["decision"], "stop")

        # Claude Code
        claude_retry = runtime.translate_canonical_output(
            {
                "decision": "retry",
                "reason": "waited for quota"
            },
            harness="claude-code",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(claude_retry["decision"], "block")
        self.assertIn("waited", claude_retry["reason"])
        self.assertEqual(
            claude_retry.get("hookSpecificOutput", {}).get("hookEventName"),
            "Stop",
        )

        claude_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            harness="claude-code",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertNotIn("decision", claude_allow)
        self.assertEqual(
            claude_allow.get("hookSpecificOutput", {}).get("hookEventName"),
            "Stop",
        )

        # Codex
        codex_retry = runtime.translate_canonical_output(
            {
                "decision": "retry",
                "reason": "waited for quota"
            },
            harness="codex",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(codex_retry["decision"], "block")
        self.assertIn("waited", codex_retry["reason"])

        codex_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            harness="codex",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(codex_allow["decision"], "allow")

        # Cursor
        cursor_retry = runtime.translate_canonical_output(
            {
                "decision": "retry",
                "reason": "waited for quota"
            },
            harness="cursor",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(cursor_retry["decision"], "cancel")
        self.assertIn("waited", cursor_retry["reason"])

        cursor_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            harness="cursor",
            event="Stop",
            hook_name="resume-after-quota",
        )
        self.assertEqual(cursor_allow, {})

    def test_completed_result_requires_an_artifact(self):
        """Findings must land somewhere durable before they are summarized."""
        self.assertTrue(
            runtime.check_completed_has_artifact({
                "status": "completed",
                "summary": "all done"
            }))
        self.assertTrue(
            runtime.check_completed_has_artifact({
                "status": "completed",
                "artifacts": []
            }))
        self.assertFalse(
            runtime.check_completed_has_artifact({
                "status":
                "completed",
                "artifacts": [{
                    "file": "report.md"
                }],
            }))

    def test_unfinished_result_needs_no_artifact(self):
        """A failure has nothing to show, and demanding one would push agents
        toward reporting success."""
        for status in ("failed", "invalid_input", "needs_input"):
            with self.subTest(status=status):
                self.assertFalse(
                    runtime.check_completed_has_artifact({"status": status}))

    def test_validate_enforces_the_completed_artifact_rule(self):
        schema = self.temp_dir / "result.json"
        schema.write_text(json.dumps({
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {
                    "type": "string"
                },
                "artifacts": {
                    "type": "array"
                },
            },
        }),
                          encoding="utf-8")
        instance = self.temp_dir / "instance.json"
        instance.write_text(json.dumps({"status": "completed"}),
                            encoding="utf-8")

        errors = runtime.validate_json_file(schema, instance)
        self.assertTrue(errors)

    def _stage_result_schemas(self) -> dict[str, dict]:
        """Every package schema that declares a completed-payload rule.

        Discovered from the schemas themselves rather than from a list here.
        The framework used to hold the table, which meant it named a specific
        package's files; now a schema carries its own rule, and the thing worth
        asserting is that each schema that carries one is coherent.
        """
        schemas_dir = (pathlib.Path(__file__).resolve().parents[2] /
                       "package" / "schemas")
        found = {}
        for path in sorted(schemas_dir.glob("*.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(schema, dict) and "status_requires" in schema:
                found[path.name] = schema
        self.assertTrue(found, "no schema declares status_requires")
        return found

    def test_a_claimed_outcome_owes_its_payload(self):
        """A status that claims an outcome must carry what that outcome means.

        These fields left the schemas' `required` arrays so that a failing
        stage could describe itself; the rule has to reappear here or the
        success status becomes a free pass.
        """
        for schema_name, schema in self._stage_result_schemas().items():
            for status, fields in schema["status_requires"].items():
                with self.subTest(schema=schema_name, status=status):
                    self.assertTrue(
                        runtime.check_status_payload(schema, {
                            "status": status,
                            "run_id": "x"
                        }))
                    payload = {"status": status, "run_id": "x"}
                    payload.update({field: "filled" for field in fields})
                    self.assertFalse(
                        runtime.check_status_payload(schema, payload))

    def test_a_claimed_outcome_cannot_be_satisfied_by_an_empty_value(self):
        """Presence is not enough: `artifacts: []` carries nothing.

        `kernel-result` is the case that matters. Its durability rule used to
        come from `check_completed_has_artifact`, which fires only on the
        literal status `completed` -- so renaming the status to `optimized`
        would have switched that rule off silently. It is declared in
        `status_requires` instead, and an empty array has to fail or the
        declaration is theatre.
        """
        for schema_name, schema in self._stage_result_schemas().items():
            for status, fields in schema["status_requires"].items():
                for field in fields:
                    with self.subTest(schema=schema_name, field=field):
                        payload = {"status": status, "run_id": "x"}
                        payload.update({f: "filled" for f in fields})
                        payload[field] = [] if field == "artifacts" else ""
                        self.assertTrue(
                            runtime.check_status_payload(schema, payload))

    def test_a_false_verdict_still_satisfies_the_rule(self):
        """`succeeded: false` is an answer, not an absence.

        The empty-value check must not read a boolean false as nothing
        reported -- a brief that found nothing to borrow is still a brief.
        """
        schema = self._stage_result_schemas()["summary-result.json"]
        status, fields = next(iter(schema["status_requires"].items()))
        payload = {"status": status, "run_id": "x"}
        payload.update(
            {f: 0 if f == "succeeded" else "filled"
             for f in fields})
        payload["succeeded"] = False
        self.assertFalse(runtime.check_status_payload(schema, payload))

    def test_unfinished_stage_result_owes_nothing(self):
        """The point of the change: a stage that reached no answer can say so.

        Every one of these was previously unrepresentable -- the schema
        required a plan, a latency or a best config, so a failing stage had to
        emit success-shaped fields or emit nothing that validated.

        The statuses come from each schema's own enum rather than a list here:
        a hardcoded one keeps passing after a value is dropped, because an
        unrecognised status has no rule to break.
        """
        for schema_name, schema in self._stage_result_schemas().items():
            owed = schema["status_requires"]
            unfinished = [
                status for status in schema["properties"]["status"]["enum"]
                if status not in owed
            ]
            self.assertTrue(unfinished, f"{schema_name} cannot report failure")
            for status in unfinished:
                with self.subTest(schema=schema_name, status=status):
                    self.assertFalse(
                        runtime.check_status_payload(
                            schema, {
                                "status": status,
                                "failure_reason": "no_device"
                            }))

    def test_every_stage_schema_declares_its_own_id(self):
        """`$id` is what labels the rule's error for the SubagentStop hook.

        The hook holds a loaded schema and no filename, so without `$id` the
        denial names nothing the agent can act on. The rule itself now travels
        in the schema, so a missing `$id` no longer disables it -- but an
        unnamed contract violation is still a bad message to hand an agent
        that has to retry.
        """
        for schema_name, schema in self._stage_result_schemas().items():
            with self.subTest(schema=schema_name):
                self.assertEqual(schema.get("$id"), schema_name)

    def test_every_stage_names_its_own_outcome_and_can_still_fail(self):
        """A status enum says what the stage produced, and can say it failed.

        `kernel-worker-result` once offered both `success` and `completed` with no
        stated difference, and `implement-result` offered `compilation_error`
        -- a cause wearing an outcome's clothes. `planned`, `implemented` and
        `measured` each say something `completed` could not. What every one of
        them still owes is a way to report not getting there, which is the
        property that made the schemas failure-capable at all.
        """
        for schema_name, schema in self._stage_result_schemas().items():
            with self.subTest(schema=schema_name):
                enum = schema["properties"]["status"]["enum"]
                self.assertIn("status", schema["required"])
                self.assertIn("failed", enum)
                self.assertIn("failure_reason", schema["properties"])
                # The success value is the one carrying a payload rule, and it
                # must not be the generic word this package moved off.
                for status in schema["status_requires"]:
                    self.assertIn(status, enum)
                    self.assertNotEqual(status, "completed")

    def test_stage_results_do_not_require_success_only_fields(self):
        """The conditional fields must be out of `required`, or the runtime
        rule never gets a chance to run -- the schema rejects the failure
        first."""
        for schema_name, schema in self._stage_result_schemas().items():
            for fields in schema["status_requires"].values():
                for field in fields:
                    with self.subTest(schema=schema_name, field=field):
                        self.assertIn(field, schema["properties"])
                        self.assertNotIn(field, schema["required"])

    def test_schema_examples_satisfy_their_own_schema(self):
        """An example is what a denied agent is handed to retry against.

        Both hook denials quote `load_schema_example` back to the agent as
        "expected output". An example that does not validate is therefore a
        retry loop: the agent copies the shape it was given and is rejected
        again for the same reason. Tightening a schema without revisiting its
        example is easy and silent, which is why this is pinned.
        """
        schemas_dir = (pathlib.Path(__file__).resolve().parents[2] /
                       "package" / "schemas")
        checked = 0
        for path in sorted(schemas_dir.glob("*.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                continue
            for index, example in enumerate(schema.get("examples", [])):
                checked += 1
                with self.subTest(schema=path.name, example=index):
                    errors = utils.validate_instance(example, schema)
                    errors += runtime.check_status_payload(
                        schema, example, path.name)
                    self.assertFalse(errors)
        self.assertGreater(checked, 0, "no schema declares an example")

    def test_status_requires_must_name_declared_statuses_and_properties(self):
        """A misspelled name has to fail the build, not the rule.

        The rule fires only on the status it is keyed to, so a typo in either
        the status or a field name enforces nothing while the schema keeps
        validating -- it fails open, quietly, in exactly the case it exists to
        catch. Now that each package names its own statuses there is no shared
        vocabulary to fall back on, so a mistyped status is the likelier of the
        two. Catching both at verify time is why the rule is worth declaring in
        the schema at all.
        """
        good = {
            "type": "object",
            "status_requires": {
                "measured": ["latency_ms"]
            },
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["measured", "failed"]
                },
                "latency_ms": {
                    "type": "number"
                },
            },
        }
        self.assertFalse(utils.validate_schema_definition(good))

        misspelled_field = json.loads(json.dumps(good))
        misspelled_field["status_requires"]["measured"] = [
            "latency_ms", "nonexistent_field"
        ]
        errors = utils.validate_schema_definition(misspelled_field)
        self.assertTrue(errors)
        self.assertIn("nonexistent_field", errors[0])

        misspelled_status = json.loads(json.dumps(good))
        misspelled_status["status_requires"] = {"measurred": ["latency_ms"]}
        errors = utils.validate_schema_definition(misspelled_status)
        self.assertTrue(errors)
        self.assertIn("measurred", errors[0])

    def test_unversioned_package_resolves_to_central_install(self):
        """A caller who names a package without its version must still find it.

        Versions are an install detail no agent tracks, so an unversioned
        lookup is the normal case. Missing the versioned directory sends the
        caller to a workspace path holding only part of the package, and
        anything installed solely in central storage then reads as absent --
        with the error naming that content rather than the resolution.
        """
        central = self.central_home / "packages" / "central-pkg@2.3.0"
        (central / "schemas").mkdir(parents=True)
        (central / "schemas" / "schema.json").write_text(json.dumps(
            {"type": "object"}),
                                                         encoding="utf-8")

        # A same-named workspace directory holding a *different* subset is
        # exactly the shape that shadowed the real install.
        decoy = self.workspace_dir / ".tpu-agent-suite" / "central-pkg"
        (decoy / "knowledgebase").mkdir(parents=True)

        resolved = runtime.get_installed_package_root(self.workspace_dir,
                                                      "central-pkg")
        self.assertEqual(resolved.resolve(), central.resolve())
        self.assertTrue((resolved / "schemas" / "schema.json").is_file())

    def test_unversioned_lookup_prefers_the_highest_version(self):
        for version in ("1.9.0", "2.10.0", "2.9.0"):
            (self.central_home / "packages" /
             f"multi-pkg@{version}").mkdir(parents=True)

        resolved = runtime.latest_installed_package("multi-pkg")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "multi-pkg@2.10.0")

    def test_unparseable_version_is_not_a_candidate(self):
        (self.central_home / "packages" /
         "odd-pkg@not-a-version").mkdir(parents=True)
        self.assertIsNone(runtime.latest_installed_package("odd-pkg"))

    def test_start_run_and_namespace(self):
        doc = runtime.start_run(package="test-pkg@1.0.0",
                                workspace=self.workspace_dir)
        self.assertIn("run_id", doc)
        self.assertEqual(doc["package"], "test-pkg@1.0.0")
        self.assertEqual(doc["environment_revision"], 1)

        # Run directory is in central storage (~/.tpu-agent-suite/runs/)
        run_dir = self.central_home / "runs" / doc["run_id"]
        self.assertTrue(run_dir.is_dir())
        self.assertTrue((run_dir / "artifacts").is_dir())
        self.assertTrue((run_dir / "run.json").is_file())
        self.assertEqual(utils.get_active_run_id(self.workspace_dir),
                         doc["run_id"])

    def test_versioned_package_tag_resolution(self):
        # Compiled prompts pass --package as <name>@<version>; the tag must
        # resolve to the versioned directory the installer created.
        versioned_root = self.central_home / "packages" / "test-pkg@1.0.0"
        (versioned_root / "schemas").mkdir(parents=True, exist_ok=True)
        (versioned_root / "schemas" / "tagged.json").write_text(
            json.dumps({"type": "object"}), encoding="utf-8")

        self.assertEqual(
            runtime.get_installed_package_root(self.workspace_dir,
                                               "test-pkg@1.0.0").resolve(),
            versioned_root.resolve(),
        )

        run_doc = runtime.start_run(package="test-pkg@1.0.0",
                                    workspace=self.workspace_dir)
        run_id = run_doc["run_id"]
        self.assertEqual(run_doc["package"], "test-pkg@1.0.0")

    def test_pending_context_is_capped_and_evicts_oldest(self):
        """Every live entry rides along on every model call, so the spool is bounded.

        A run writing context in a loop would otherwise grow it without
        limit. The newest survive: they are the context a result is most
        likely still to need.
        """
        overflow = runtime.MAX_PENDING_CONTEXT_ENTRIES + 3
        for i in range(overflow):
            runtime.record_pending_hook_context(
                session_id="conv-cap",
                hook_name="test-hook",
                context_text=f"CONTEXT-{i:02d}",
                package="test-pkg@1.0.0",
            )
        replayed = runtime.replay_pending_hook_context("conv-cap")
        kept = [
            line for line in replayed.splitlines()
            if line.startswith("CONTEXT-")
        ]
        self.assertEqual(len(kept), runtime.MAX_PENDING_CONTEXT_ENTRIES)
        self.assertNotIn("CONTEXT-00", kept)
        self.assertIn(f"CONTEXT-{overflow - 1:02d}", kept)

    def test_pending_context_is_not_shared_across_sessions(self):
        """Draining one session must not drain, or reveal, another's.

        Entries are consumed rather than re-read, so a spool shared across
        sessions would not merely leak context, it would lose it: whoever
        replayed first would take it. Note the limit of this test -- these two
        IDs are unrelated, so it pins the keying but not the separate decision
        in `replay_pending_hook_context` to skip the parent-conversation
        fallback that `resolve_session_hook_context` performs. Covering that
        needs a real parent/child pair.
        """
        runtime.record_pending_hook_context(
            session_id="conv-owner",
            hook_name="test-hook",
            context_text="CONTEXT-OWNED",
            package="test-pkg@1.0.0",
        )
        self.assertEqual(runtime.replay_pending_hook_context("conv-stranger"),
                         "")
        self.assertEqual(runtime.replay_pending_hook_context("conv-owner"),
                         "CONTEXT-OWNED")

    def test_posttooluse_context_reaches_a_google_harness_model(self):
        """The whole emulation, end to end.

        Antigravity documents the PostToolUse response as a bare {}, so the
        hook must still answer with exactly that -- and the context it
        produced must nonetheless reach the model, as injectSteps on the
        following invocations. `invocationNum > 0` is the model call right
        after a tool call, which is where hook context is wanted.
        """
        session = "conv-e2e-posttooluse"
        post_payload = json.dumps({
            "conversationId": session,
            "workspacePaths": [str(self.workspace_dir)],
            "toolCall": {
                "name": "write_to_file",
                "args": {
                    "TargetFile": "notes.md"
                }
            },
            "stepIdx": 3,
        })
        post_out, _ = runtime.dispatch_hook(
            harness="antigravity",
            package="",
            hook_name="probe-hook",
            event="PostToolUse",
            workspace=self.workspace_dir,
            raw_input_text=post_payload,
            command=
            "printf '%s' '{\"decision\":\"allow\",\"context\":\"SPOOLED-CONTEXT\"}'",
        )
        self.assertEqual(post_out, {},
                         "PostToolUse must answer with the documented {}")

        def invoke(invocation_num: int) -> dict:
            out, _ = runtime.dispatch_hook(
                harness="antigravity",
                package="",
                hook_name="session-run-init",
                event="SessionStart",
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "conversationId":
                    session,
                    "workspacePaths": [str(self.workspace_dir)],
                    "invocationNum":
                    invocation_num,
                    "initialNumSteps":
                    8,
                }),
                command="printf '%s' '{\"decision\":\"allow\"}'",
            )
            return out

        # Replay remains available across rounds so the agent can execute
        # multiple intermediate tool calls before citing the context.
        for invocation_num in range(1, 6):
            self.assertIn({"ephemeralMessage": "SPOOLED-CONTEXT"},
                          invoke(invocation_num).get("injectSteps", []),
                          f"context missing from invocation {invocation_num}")

    def test_centralized_storage_resolution(self):
        (self.central_home / "environment.json").write_text(json.dumps({
            "revision":
            2,
            "target":
            "cloud"
        }),
                                                            encoding="utf-8")
        (self.central_home / "schemas").mkdir()
        (self.central_home / "schemas" / "out.json").write_text(
            json.dumps({"type": "object"}), encoding="utf-8")

        # Start run without workspace arg uses centralized storage and central environment
        doc = runtime.start_run(package="central-pkg@2.0.0")
        self.assertIn("run_id", doc)
        self.assertEqual(doc["package"], "central-pkg@2.0.0")
        self.assertEqual(doc["environment_revision"], 2)

        run_dir = self.central_home / "runs" / doc["run_id"]
        self.assertTrue(run_dir.is_dir())
        self.assertTrue((run_dir / "artifacts").is_dir())

    def test_runtime_cli(self):
        run_doc = runtime.start_run(package="test-pkg@1.0.0",
                                    workspace=self.workspace_dir)
        run_id = run_doc["run_id"]  # noqa: F841

        valid_instance = self.temp_dir / "valid.json"
        valid_instance.write_text(json.dumps({"status": "ok"}),
                                  encoding="utf-8")
        schema_path = self.pkg_root / "schemas" / "out.json"

        ret = runtime.main(
            ["validate", "--schema",
             str(schema_path),
             str(valid_instance)])
        self.assertEqual(ret, 0)

    def test_cli_start_run_reads_workspace_environment(self):
        """`start-run` without --workspace consults the cwd's workspace install."""
        ws = self.temp_dir / "proj_ws"
        pkg_cfg = ws / ".tpu-agent-suite"
        pkg_cfg.mkdir(parents=True)
        (pkg_cfg / "environment.json").write_text(json.dumps({
            "revision": 3,
            "target": "proj"
        }),
                                                  encoding="utf-8")

        old_cwd = Path.cwd()
        buf = io.StringIO()
        try:
            os.chdir(ws)
            with contextlib.redirect_stdout(buf):
                ret = runtime.main(["start-run", "--package", "wspkg@1.0.0"])
        finally:
            os.chdir(old_cwd)

        self.assertEqual(ret, 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["environment_revision"], 3)

    def test_hook_dispatch_claude_allow_and_deny(self):
        script_path = self.pkg_root / "scripts" / "guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "data = json.load(sys.stdin)\n"
            "cmd = data.get('args', {}).get('command', '')\n"
            "if 'block' in cmd:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'blocked keyword'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow', 'message': 'checked'}))\n",
            encoding="utf-8",
        )

        # Test allow in Claude Code
        claude_allow_payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {
                "command": "git branch"
            }
        })
        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=claude_allow_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["systemMessage"], "checked")
        self.assertEqual(res["hookSpecificOutput"]["permissionDecision"],
                         "allow")

        # Test deny in Claude Code
        claude_deny_payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {
                "command": "echo block"
            }
        })
        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=claude_deny_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["hookSpecificOutput"]["permissionDecision"],
                         "deny")
        self.assertEqual(res["hookSpecificOutput"]["permissionDecisionReason"],
                         "blocked keyword")

    def test_hook_dispatch_antigravity_allow_and_deny(self):
        script_path = self.pkg_root / "scripts" / "guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "data = json.load(sys.stdin)\n"
            "cmd = data.get('args', {}).get('command', '')\n"
            "if 'block' in cmd:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'blocked command'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow'}))\n",
            encoding="utf-8",
        )

        # Allow
        agy_allow_payload = json.dumps({
            "toolCall": {
                "name": "run_command",
                "args": {
                    "CommandLine": "ls"
                }
            }
        })
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=agy_allow_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"decision": "allow"})

        # Deny
        agy_deny_payload = json.dumps({
            "toolCall": {
                "name": "run_command",
                "args": {
                    "CommandLine": "echo block"
                }
            }
        })
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=agy_deny_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {
            "decision": "deny",
            "reason": "blocked command"
        })

    def test_hook_dispatch_codex_allow_and_deny(self):
        script_path = self.pkg_root / "scripts" / "guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "data = json.load(sys.stdin)\n"
            "cmd = data.get('args', {}).get('command', '')\n"
            "if 'block' in cmd:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'blocked command'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow'}))\n",
            encoding="utf-8",
        )

        # Allow
        codex_allow_payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {
                "command": "git status"
            },
            "session_id": "sess-123"
        })
        res, code = runtime.dispatch_hook(
            harness="codex",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=codex_allow_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"decision": "allow"})

        # Deny
        codex_deny_payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rm -rf block"
            },
            "session_id": "sess-123"
        })
        res, code = runtime.dispatch_hook(
            harness="codex",
            package="test-pkg@1.0.0",
            hook_name="test-guard",
            event="PreToolUse",
            script_rel="scripts/guard.py",
            workspace=self.workspace_dir,
            raw_input_text=codex_deny_payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {
            "decision": "deny",
            "reason": "blocked command"
        })

    def test_hook_dispatch_codex_post_tool_use(self):
        script_path = self.pkg_root / "scripts" / "post.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "data = json.load(sys.stdin)\n"
            "print(json.dumps({'decision': 'allow', 'context': 'verified write'}))\n",
            encoding="utf-8",
        )
        payload = json.dumps({
            "tool_name": "apply_patch",
            "tool_input": {
                "command": "*** Begin Patch\n*** Add File: a.txt\n+hello\n"
            },
            "tool_response": {
                "status": "ok"
            }
        })
        res, code = runtime.dispatch_hook(
            harness="codex",
            package="test-pkg@1.0.0",
            hook_name="post-guard",
            event="PostToolUse",
            script_rel="scripts/post.py",
            workspace=self.workspace_dir,
            raw_input_text=payload,
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {
            "decision": "allow",
            "additional_context": "verified write"
        })

    def test_hook_dispatch_exit_code_and_schema_validation(self):
        # Exit code non-zero fallback
        script_fail = self.pkg_root / "scripts" / "fail.py"
        script_fail.parent.mkdir(parents=True, exist_ok=True)
        script_fail.write_text(
            "import sys\n"
            "print('Fatal error in guard', file=sys.stderr)\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="fail-guard",
            event="PreToolUse",
            script_rel="scripts/fail.py",
            workspace=self.workspace_dir,
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("Fatal error in guard", res["reason"])

        # Invalid schema property (e.g. invalid 'status' key instead of 'decision')
        script_invalid = self.pkg_root / "scripts" / "invalid.py"
        script_invalid.write_text(
            "import json\n"
            "print(json.dumps({'status': 'allow'}))\n",
            encoding="utf-8",
        )
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="invalid-guard",
            event="PreToolUse",
            script_rel="scripts/invalid.py",
            workspace=self.workspace_dir,
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("violated hook-output.schema.json", res["reason"])

    def test_hook_dispatch_command(self):
        # 1. Successful command emitting empty output -> default allow
        res_claude, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="cmd-hook",
            event="PreToolUse",
            command="true",
            workspace=self.workspace_dir,
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            res_claude["hookSpecificOutput"]["permissionDecision"], "allow")

        res_ag, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="cmd-hook",
            event="PreToolUse",
            command="true",
            workspace=self.workspace_dir,
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_ag["decision"], "allow")

        # 2. Failed command (non-zero exit) -> deny
        res_fail, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="cmd-fail",
            event="PreToolUse",
            command="false",
            workspace=self.workspace_dir,
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_fail["hookSpecificOutput"]["permissionDecision"],
                         "deny")

    def test_hook_dispatch_with_script_args(self):
        script_path = self.pkg_root / "scripts" / "args_guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "passed_args = sys.argv[1:]\n"
            "if '--strict' in passed_args and 'param with spaces' in passed_args:\n"
            "    print(json.dumps({'decision': 'allow', 'message': 'args verified'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': f'unexpected args: {passed_args}'}))\n",
            encoding="utf-8",
        )

        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="test-args-guard",
            event="PreToolUse",
            script_rel="scripts/args_guard.py",
            workspace=self.workspace_dir,
            raw_input_text="{}",
            script_args=["--strict", "param with spaces"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"decision": "allow"})

    def test_hook_dispatch_cli_remainder_args(self):
        script_path = self.pkg_root / "scripts" / "cli_args_guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "if '--mode' in sys.argv and 'fast' in sys.argv:\n"
            "    print(json.dumps({'decision': 'allow'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': f'missing mode: {sys.argv}'}))\n",
            encoding="utf-8",
        )

        captured = io.StringIO()
        with mock.patch("sys.stdout",
                        captured), mock.patch("sys.stdin", io.StringIO("{}")):
            ret = runtime.main([
                "hook-dispatch",
                "--harness",
                "antigravity",
                "--package",
                "test-pkg@1.0.0",
                "--hook",
                "cli-guard",
                "--event",
                "PreToolUse",
                "--script",
                "scripts/cli_args_guard.py",
                "--workspace",
                str(self.workspace_dir),
                "--",
                "--mode",
                "fast",
            ])
        self.assertEqual(ret, 0)
        output_json = json.loads(captured.getvalue().strip())
        self.assertEqual(output_json.get("decision"), "allow")

    def test_normalize_inbound_antigravity_multi_replace(self):
        payload = {
            "toolCall": {
                "name": "multi_replace_file_content",
                "args": {
                    "TargetFile": "/path/to/file.py",
                    "ReplacementChunks": []
                }
            }
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="antigravity",
            event="PreToolUse",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["tool"], "edit")
        self.assertEqual(event["args"]["path"], "/path/to/file.py")

    def test_dispatch_hook_script_timeout(self):
        scripts_dir = (self.central_home / "packages" / "test-pkg@1.0.0" /
                       "scripts")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "slow_guard.py").write_text("print('slow')",
                                                   encoding="utf-8")

        with mock.patch.object(runtime,
                               "_run_hook_process",
                               side_effect=subprocess.TimeoutExpired(
                                   cmd=["python3"], timeout=5)):
            res, code = runtime.dispatch_hook(
                harness="claude-code",
                package="test-pkg@1.0.0",
                hook_name="slow-guard",
                event="PreToolUse",
                script_rel="scripts/slow_guard.py",
                timeout=5,
                workspace=self.workspace_dir,
                raw_input_text="{}",
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            res.get("hookSpecificOutput", {}).get("permissionDecision"),
            "deny",
        )
        self.assertIn(
            "timed out after 5 seconds",
            res.get("hookSpecificOutput", {}).get("permissionDecisionReason",
                                                  ""),
        )

    def test_dispatch_hook_command_timeout(self):
        with mock.patch.object(runtime,
                               "_run_hook_process",
                               side_effect=subprocess.TimeoutExpired(
                                   cmd="sleep 10", timeout=5)):
            res, code = runtime.dispatch_hook(
                harness="antigravity",
                package="test-pkg@1.0.0",
                hook_name="slow-cmd",
                event="PreToolUse",
                command="sleep 10",
                timeout=5,
                workspace=self.workspace_dir,
                raw_input_text="{}",
            )
        self.assertEqual(code, 0)
        self.assertEqual(res.get("decision"), "deny")
        self.assertIn("timed out after 5 seconds", res.get("reason", ""))

    def test_package_isolation(self):
        for pkg in ("pkg-a@1.0.0", "pkg-b@1.0.0"):
            (self.central_home / "packages" / pkg).mkdir(parents=True,
                                                         exist_ok=True)
        run_doc = runtime.start_run(package="pkg-a@1.0.0",
                                    workspace=self.workspace_dir)
        run_id = run_doc["run_id"]

        # Same package access succeeds
        _, _, doc = runtime.get_installed_run_context(
            package="pkg-a@1.0.0", run_id=run_id, workspace=self.workspace_dir)
        self.assertEqual(doc["run_id"], run_id)

        # Different package access is rejected
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            runtime.get_installed_run_context(package="pkg-b@1.0.0",
                                              run_id=run_id,
                                              workspace=self.workspace_dir)
        self.assertIn("refusing access", str(ctx.exception).lower())

    def test_workspace_environment_is_read(self):
        # A project install's environment sits directly in .tpu-agent-suite/.
        pkg_env_dir = self.workspace_dir / ".tpu-agent-suite"
        pkg_env_dir.mkdir(parents=True, exist_ok=True)
        (pkg_env_dir / "environment.json").write_text(json.dumps(
            {"target": "custom_override"}),
                                                      encoding="utf-8")

        env, path = runtime.get_environment(self.workspace_dir, "custom-pkg")
        self.assertEqual(env.get("target"), "custom_override")
        self.assertEqual(path, (pkg_env_dir / "environment.json").resolve())

    def test_normalize_inbound_subagent_events(self):
        # Antigravity PreToolUse invoke_subagent
        ag_payload = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [{
                        "TypeName": "tpu-kernel-development",
                        "Role": "Kernel Optimizer",
                        "Prompt": "Optimize flash attention kernel",
                    }]
                },
            }
        }
        ag_event = runtime.normalize_inbound_event(
            raw_payload=ag_payload,
            harness="antigravity",
            event="PreToolUse",
            package="tpu-agent-suite",
            workspace=self.workspace_dir,
        )
        self.assertEqual(ag_event["event"], "PreToolUse")
        self.assertEqual(ag_event["tool"], "delegate")
        self.assertEqual(ag_event["args"]["subagent"],
                         "tpu-kernel-development")
        self.assertEqual(ag_event["args"]["prompt"],
                         "Optimize flash attention kernel")
        self.assertIsNone(ag_event["output"])

        # Claude Code PreToolUse Agent
        claude_payload = {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "tpu-sharding",
                "prompt": "Distribute model tensors across TPU pod",
            },
        }
        claude_event = runtime.normalize_inbound_event(
            raw_payload=claude_payload,
            harness="claude-code",
            event="PreToolUse",
            package="tpu-agent-suite",
            workspace=self.workspace_dir,
        )
        self.assertEqual(claude_event["event"], "PreToolUse")
        self.assertEqual(claude_event["tool"], "delegate")
        self.assertEqual(claude_event["args"]["subagent"], "tpu-sharding")
        self.assertEqual(claude_event["args"]["prompt"],
                         "Distribute model tensors across TPU pod")
        self.assertIsNone(claude_event["output"])

        # Antigravity PostToolUse invoke_subagent with toolOutput
        ag_post_payload = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [{
                        "TypeName": "tpu-kernel-development"
                    }]
                },
            },
            "toolOutput": {
                "status": "success",
                "speedup": "1.4x"
            },
        }
        ag_post_event = runtime.normalize_inbound_event(
            raw_payload=ag_post_payload,
            harness="antigravity",
            event="PostToolUse",
            package="tpu-agent-suite",
            workspace=self.workspace_dir,
        )
        self.assertEqual(ag_post_event["event"], "PostToolUse")
        self.assertEqual(ag_post_event["tool"], "delegate")
        self.assertEqual(ag_post_event["args"]["subagent"],
                         "tpu-kernel-development")
        self.assertEqual(ag_post_event["output"], {
            "status": "success",
            "speedup": "1.4x"
        })

    def test_translate_subagent_canonical_output(self):
        # Claude Code PreToolUse allow
        out = runtime.translate_canonical_output(
            {
                "decision": "allow",
                "updated_args": {
                    "prompt": "Sanitized prompt"
                }
            },
            "claude-code",
            "PreToolUse",
            "guard",
        )
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PreToolUse")
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertEqual(out["hookSpecificOutput"]["updatedInput"],
                         {"prompt": "Sanitized prompt"})

        # Claude Code PostToolUse block
        out = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Kernel check failed",
                "context": "Please retry with tile size 64"
            },
            "claude-code",
            "PostToolUse",
            "verifier",
        )
        self.assertEqual(out["decision"], "block")
        self.assertEqual(out["reason"], "Kernel check failed")
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PostToolUse")
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"],
                         "Please retry with tile size 64")

        # Antigravity PreToolUse deny
        out = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Unauthorized subagent"
            },
            "antigravity",
            "PreToolUse",
            "guard",
        )
        self.assertEqual(out["decision"], "deny")
        self.assertEqual(out["reason"], "Unauthorized subagent")

    def test_normalize_inbound_subagent_stop_claude_code(self):
        payload = {
            "agent_type": "torch_coder",
            "last_assistant_message":
            "Kernel code generated successfully with 1.4x speedup.",
            "agent_transcript_path": "/path/to/transcript.jsonl",
            "session_id": "sess-123",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="claude-code",
            event="SubagentStop",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "torch_coder")
        self.assertEqual(event["args"]["subagent"], "torch_coder")
        self.assertEqual(event["args"]["agent_transcript_path"],
                         "/path/to/transcript.jsonl")
        self.assertEqual(event["args"]["session_id"], "sess-123")
        self.assertEqual(
            event["output"],
            "Kernel code generated successfully with 1.4x speedup.",
        )
        self.assertEqual(event["context"]["workspace"],
                         str(self.workspace_dir))
        self.assertEqual(event["context"]["package"], "test-pkg")

    def test_normalize_inbound_subagent_stop_codex(self):
        payload = {
            "agent_type": "tpu-sharding",
            "agent_id": "agent-456",
            "turn_id": "turn-789",
            "last_assistant_message":
            "Distributed sharding strategy computed.",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="codex",
            event="SubagentStop",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "tpu-sharding")
        self.assertEqual(event["args"]["agent_id"], "agent-456")
        self.assertEqual(event["args"]["turn_id"], "turn-789")
        self.assertEqual(event["output"],
                         "Distributed sharding strategy computed.")

    def test_normalize_inbound_subagent_stop_cursor(self):
        payload = {
            "subagent_type": "torch_coder",
            "status": "completed",
            "task": "Refactor custom flash attention kernel",
            "summary":
            "Completed kernel implementation and benchmark verification.",
            "modified_files": ["kernel.py", "bench.py"],
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="cursor",
            event="subagentStop",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "torch_coder")
        self.assertEqual(event["args"]["status"], "completed")
        self.assertEqual(event["args"]["task"],
                         "Refactor custom flash attention kernel")
        self.assertEqual(event["args"]["modified_files"],
                         ["kernel.py", "bench.py"])
        self.assertEqual(
            event["output"],
            "Completed kernel implementation and benchmark verification.",
        )

    def test_normalize_inbound_subagent_stop_antigravity(self):
        payload = {
            "role": "torch_coder",
            "conversationId": "conv-999",
            "transcriptPath": "/path/to/log.jsonl",
            "terminationReason": "NO_TOOL_CALL",
            "toolOutput": "Optimized TPU tile size 128x128.",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="antigravity",
            event="Stop",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "torch_coder")
        self.assertEqual(event["args"]["conversationId"], "conv-999")
        self.assertEqual(event["args"]["transcriptPath"], "/path/to/log.jsonl")
        self.assertEqual(event["args"]["terminationReason"], "NO_TOOL_CALL")
        self.assertEqual(event["args"]["stop_reason"], "NO_TOOL_CALL")
        self.assertEqual(event["output"], "Optimized TPU tile size 128x128.")

    def test_normalize_inbound_subagent_stop_unsupported_harness(self):
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            runtime.normalize_inbound_event(
                raw_payload={},
                harness="unsupported_harness",
                event="SubagentStop",
                package="test-pkg",
                workspace=self.workspace_dir,
            )
        self.assertIn("unsupported harness for subagent stop",
                      str(ctx.exception))

    def test_extract_last_transcript_message(self):
        transcript_file = self.workspace_dir / "transcript.jsonl"
        lines = [
            json.dumps({
                "source": "USER_EXPLICIT",
                "content": "Hello"
            }),
            json.dumps({
                "source": "MODEL",
                "content": '{"return": "done"}'
            }),
        ]
        transcript_file.write_text("\n".join(lines), encoding="utf-8")
        msg = runtime.extract_last_transcript_message(str(transcript_file))
        self.assertEqual(msg, '{"return": "done"}')

    def test_get_google_harness_subagent_name_with_disk_descriptor(self):
        cid = "test-descriptor-cid-9999"
        # A fake home, so the descriptor never lands in the developer's own
        # ~/.gemini brain directory.
        fake_home = self.temp_dir / "fake_home"
        parent_dir = fake_home / ".gemini" / "antigravity-cli" / "brain" / "mock-parent-1234" / ".system_generated" / "subagents"
        parent_dir.mkdir(parents=True, exist_ok=True)
        desc_file = parent_dir / f"{cid}.json"
        desc_file.write_text(json.dumps({
            "conversationId": cid,
            "subagentDescriptor": {
                "typeName": "torch-coder"
            }
        }),
                             encoding="utf-8")
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            name = runtime.get_google_harness_subagent_name(
                cid, "", self.workspace_dir)
        self.assertEqual(name, "torch-coder")

    def test_get_google_harness_subagent_name_root_conversation(self):
        cid = "test-root-cid-5678"
        name = runtime.get_google_harness_subagent_name(
            cid, "", self.workspace_dir)
        self.assertIsNone(name)

    def test_google_harness_post_processing(self):
        # 1. Output extracted from transcript when None
        transcript_file = self.workspace_dir / "transcript_post.jsonl"
        transcript_file.write_text(
            json.dumps({
                "source": "MODEL",
                "content": "TPU kernel done"
            }),
            encoding="utf-8",
        )
        subagent, output = runtime.google_harness_post_processing(
            subagent="my-agent",
            subagent_output=None,
            workspace=self.workspace_dir,
            transcript_path=str(transcript_file),
        )
        self.assertEqual(subagent, "my-agent")
        self.assertEqual(output, "TPU kernel done")

        # 2. Existing output preserved
        _, output_kept = runtime.google_harness_post_processing(
            subagent="my-agent",
            subagent_output="already_set",
            workspace=self.workspace_dir,
            transcript_path=str(transcript_file),
        )
        self.assertEqual(output_kept, "already_set")

    def test_translate_canonical_output_subagent_stop(self):
        # 1. Claude Code
        # Allow
        claude_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            "claude-code",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(
            claude_allow.get("hookSpecificOutput", {}).get("hookEventName"),
            "SubagentStop",
        )
        self.assertNotIn("decision", claude_allow)

        # Deny / Block
        claude_deny = runtime.translate_canonical_output(
            {
                "decision":
                "deny",
                "reason":
                "Missing benchmark metric in summary",
                "followup_message":
                "Please include speedup metric before completing",
            },
            "claude-code",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(claude_deny.get("decision"), "block")
        self.assertEqual(
            claude_deny.get("reason"),
            "Please include speedup metric before completing",
        )

        # 2. Codex
        codex_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            "codex",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(codex_allow.get("decision"), "allow")

        codex_block = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Failed output validation",
            },
            "codex",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(codex_block.get("decision"), "block")
        self.assertEqual(codex_block.get("reason"), "Failed output validation")

        # 3. Cursor
        cursor_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            "cursor",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(cursor_allow, {})

        cursor_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Artifact validation failed",
                "followup_message": "Please verify return schema",
            },
            "cursor",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(
            cursor_deny,
            {"followup_message": "Please verify return schema"},
        )

        # 4. Antigravity
        ag_allow = runtime.translate_canonical_output(
            {"decision": "allow"},
            "antigravity",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(ag_allow.get("decision"), "allow")

        ag_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Compilation summary missing",
            },
            "antigravity",
            "SubagentStop",
            "guard",
        )
        self.assertEqual(ag_deny.get("decision"), "continue")
        self.assertEqual(ag_deny.get("reason"), "Compilation summary missing")

    def test_dispatch_hook_subagent_stop_e2e(self):
        scripts_dir = (self.central_home / "packages" / "test-pkg@1.0.0" /
                       "scripts")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        hook_script = scripts_dir / "check_deliverable.py"
        hook_script.write_text(
            "import sys, json\n"
            "data = json.loads(sys.stdin.read())\n"
            "output = data.get('output', '')\n"
            "if 'error' in str(output).lower():\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'Error found in output'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow'}))\n",
            encoding="utf-8",
        )

        # Allow case for Claude Code
        res_allow, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="SubagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "agent_type":
                "torch_coder",
                "last_assistant_message":
                "All TPU tests passed successfully."
            }),
        )
        self.assertEqual(code, 0)
        self.assertNotIn("decision", res_allow)

        # Deny case for Cursor. `status` is what Cursor's mapping points
        # `stop_reason` at, and its vocabulary is not Antigravity's: read as a
        # termination reason it would look like a halt and excuse the subagent
        # from its contract, so only the Google harnesses are asked.
        res_deny, code = runtime.dispatch_hook(
            harness="cursor",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="subagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "subagent_type":
                "torch_coder",
                "status":
                "completed",
                "summary":
                "Execution error: Kernel died."
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_deny.get("followup_message"),
                         "Error found in output")

        # Deny case for Antigravity. The reason says the subagent finished on
        # its own terms, so it is held to its contract rather than excused.
        res_ag, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="SubagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "role": "torch_coder",
                "terminationReason": "NO_TOOL_CALL",
                "toolOutput": "Error during compilation."
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_ag.get("decision"), "continue")
        self.assertEqual(res_ag.get("reason"), "Error found in output")

        # Matcher emulation on Antigravity: matching subagent triggers validation
        res_matched, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="SubagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            matcher="torch_coder|tpu-kernel-development",
            raw_input_text=json.dumps({
                "role": "torch_coder",
                "toolOutput": "Error during compilation."
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_matched.get("decision"), "continue")

        # Matcher emulation on Antigravity: non-matching subagent is bypassed
        res_unmatched, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="SubagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            matcher="torch_coder|tpu-kernel-development",
            raw_input_text=json.dumps({
                "role": "doc_writer",
                "toolOutput": "Error during compilation."
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_unmatched.get("decision"), "allow")

        # Root session stop for Antigravity: bypasses SubagentStop hook even if output has errors
        res_root, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="check-deliverable",
            event="SubagentStop",
            script_rel="scripts/check_deliverable.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId":
                "root-conv-1234",
                "terminationReason":
                "USER_CANCELED",
                "toolOutput":
                "Error occurred in user turn.",
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_root.get("decision"), "allow")

        # A subagent cut short has no result to hold to a contract, and
        # blocking its stop would push an agent that was just stopped back
        # into the loop. An unrecognised reason is treated the same way: the
        # set lists completions, so anything the enum gains later errs toward
        # letting the halt stand.
        for reason in ("USER_CANCELED", "HALTED_STEP", "MAX_INVOCATIONS",
                       "ERROR", "TERMINATION_REASON_FROM_THE_FUTURE"):
            with self.subTest(reason=reason):
                res_halted, code = runtime.dispatch_hook(
                    harness="antigravity",
                    package="test-pkg@1.0.0",
                    hook_name="check-deliverable",
                    event="SubagentStop",
                    script_rel="scripts/check_deliverable.py",
                    workspace=self.workspace_dir,
                    raw_input_text=json.dumps({
                        "role":
                        "torch_coder",
                        "terminationReason":
                        reason,
                        "toolOutput":
                        "Error occurred before abort.",
                    }),
                )
                self.assertEqual(code, 0)
                self.assertEqual(res_halted.get("decision"), "allow")

    def test_hook_dispatch_subagent_delegate(self):
        # Directly test normalize_inbound_event unpacking for Antigravity invoke_subagent
        raw_ag_payload = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [{
                        "TypeName": "tpu-kernel-development",
                        "Prompt": "optimize"
                    }]
                },
            }
        }
        normalized = runtime.normalize_inbound_event(
            raw_payload=raw_ag_payload,
            harness="antigravity",
            event="PreToolUse",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
        )
        self.assertEqual(normalized["tool"], "delegate")
        self.assertEqual(normalized["args"]["subagent"],
                         "tpu-kernel-development")
        self.assertEqual(normalized["args"]["prompt"], "optimize")

        script_path = self.pkg_root / "scripts" / "subagent_guard.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "import json, sys\n"
            "data = json.load(sys.stdin)\n"
            "assert data['tool'] == 'delegate'\n"
            "subagent = data.get('args', {}).get('subagent', '')\n"
            "prompt = data.get('args', {}).get('prompt', '')\n"
            "if subagent == 'blocked-agent':\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'blocked subagent'}))\n"
            "elif data['event'] == 'PostToolUse' and data.get('output', {}).get('status') == 'fail':\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'subagent failed'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow', 'message': f'{subagent}:{prompt}'}))\n",
            encoding="utf-8",
        )

        # Antigravity PreToolUse allow (unpacks Subagents[0])
        res_ag, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="subagent-guard",
            event="PreToolUse",
            script_rel="scripts/subagent_guard.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(raw_ag_payload),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_ag, {"decision": "allow"})

        # Claude Code PreToolUse deny
        res_claude, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="subagent-guard",
            event="PreToolUse",
            script_rel="scripts/subagent_guard.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "tool_name": "Agent",
                "tool_input": {
                    "subagent_type": "blocked-agent",
                    "prompt": "run"
                },
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            res_claude["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(
            res_claude["hookSpecificOutput"]["permissionDecisionReason"],
            "blocked subagent")

        # Antigravity PostToolUse deny on toolOutput
        res_post, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="subagent-guard",
            event="PostToolUse",
            script_rel="scripts/subagent_guard.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "toolCall": {
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [{
                            "TypeName": "tpu-kernel-development"
                        }]
                    }
                },
                "toolOutput": {
                    "status": "fail"
                },
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_post, {
            "decision": "deny",
            "reason": "subagent failed"
        })

        # Antigravity multi-subagent batch: partial allow with overwrite filtering
        raw_multi_ag = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "tpu-kernel-development",
                            "Prompt": "optimize"
                        },
                        {
                            "TypeName": "blocked-agent",
                            "Prompt": "run dangerous task"
                        },
                    ]
                }
            }
        }
        res_multi, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="subagent-guard",
            event="PreToolUse",
            script_rel="scripts/subagent_guard.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(raw_multi_ag),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_multi.get("decision"), "allow")
        self.assertIn("overwrite", res_multi)
        # Blocked subagent is pruned, only allowed subagent remains in overwrite.Subagents
        self.assertEqual(res_multi["overwrite"]["Subagents"],
                         [{
                             "TypeName": "tpu-kernel-development",
                             "Prompt": "optimize"
                         }])

        # Antigravity multi-subagent batch: all denied
        raw_all_denied = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "blocked-agent",
                            "Prompt": "task 1"
                        },
                    ]
                }
            }
        }
        res_all_denied, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="subagent-guard",
            event="PreToolUse",
            script_rel="scripts/subagent_guard.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(raw_all_denied),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res_all_denied.get("decision"), "deny")

    def test_normalize_inbound_session_start_claude_code(self):
        payload = {
            "session_id": "sess-claude-123",
            "source": "startup",
            "cwd": "/workspace/project",
            "transcript_path": "/workspace/.claude/transcript.jsonl",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="claude-code",
            event="SessionStart",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SessionStart")
        self.assertEqual(event["args"]["session_id"], "sess-claude-123")
        self.assertEqual(event["args"]["source"], "startup")
        self.assertEqual(event["args"]["cwd"], "/workspace/project")
        self.assertEqual(event["args"]["transcript_path"],
                         "/workspace/.claude/transcript.jsonl")
        self.assertEqual(event["context"]["workspace"],
                         str(self.workspace_dir))
        self.assertEqual(event["context"]["package"], "test-pkg")

    def test_normalize_inbound_session_start_codex(self):
        payload = {
            "session_id": "sess-codex-456",
            "source": "resume",
            "cwd": "/workspace/codex-proj",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="codex",
            event="SessionStart",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SessionStart")
        self.assertEqual(event["args"]["session_id"], "sess-codex-456")
        self.assertEqual(event["args"]["source"], "resume")
        self.assertEqual(event["args"]["cwd"], "/workspace/codex-proj")

    def test_normalize_inbound_session_start_cursor(self):
        """Cursor's documented sessionStart payload, including its array cwd.

        `workspace_roots` is a list of paths, but `cwd` names a single
        directory, so the normalizer must hand hooks the first root as a
        string rather than the list itself.
        """
        payload = {
            "session_id": "sess-cursor-789",
            "conversation_id": "sess-cursor-789",
            "is_background_agent": False,
            "composer_mode": "agent",
            "workspace_roots": ["/workspace/cursor-proj"],
            "transcript_path": "/workspace/.cursor/transcript.jsonl",
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="cursor",
            event="sessionStart",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SessionStart")
        self.assertEqual(event["args"]["session_id"], "sess-cursor-789")
        self.assertEqual(event["args"]["cwd"], "/workspace/cursor-proj")
        self.assertEqual(event["args"]["transcript_path"],
                         "/workspace/.cursor/transcript.jsonl")
        # Cursor publishes nothing naming what started the session, so the
        # canonical default stands rather than a value invented from the
        # payload.
        self.assertEqual(event["args"]["source"], "startup")

    def test_session_start_cwd_survives_an_empty_workspace_roots(self):
        """An empty array is no answer, so the workspace fallback must hold.

        Cursor sends `workspace_roots` as a list. Unwrapping an empty one used
        to yield `[]`, which then displaced the fallback and handed hooks a
        list where the contract promises a path string.
        """
        event = runtime.normalize_inbound_event(
            raw_payload={
                "session_id": "sess-empty",
                "workspace_roots": [],
            },
            harness="cursor",
            event="sessionStart",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["args"]["cwd"], str(self.workspace_dir))
        self.assertIsInstance(event["args"]["cwd"], str)

    def test_normalize_inbound_session_start_antigravity(self):
        payload = {
            "conversationId": "conv-agy-101",
            "source": "startup",
            "workspacePaths": [str(self.workspace_dir)],
            "transcriptPath": "/path/to/agy.jsonl",
            "invocationNum": 1,
            "initialNumSteps": 0,
        }
        event = runtime.normalize_inbound_event(
            raw_payload=payload,
            harness="antigravity",
            event="PreInvocation",
            package="test-pkg",
            workspace=self.workspace_dir,
        )
        self.assertEqual(event["event"], "SessionStart")
        self.assertEqual(event["args"]["session_id"], "conv-agy-101")
        self.assertEqual(event["args"]["source"], "startup")
        self.assertEqual(event["args"]["cwd"], str(self.workspace_dir))
        self.assertEqual(event["args"]["transcript_path"],
                         "/path/to/agy.jsonl")
        self.assertEqual(event["args"]["invocation_num"], 1)
        self.assertEqual(event["args"]["initial_num_steps"], 0)

    def test_translate_canonical_output_session_start(self):
        # 1. Claude Code Allow with context
        claude_allow = runtime.translate_canonical_output(
            {
                "decision": "allow",
                "context": "TPU v5p environment active.",
                "message": "Welcome!",
            },
            "claude-code",
            "SessionStart",
            "init-guard",
        )
        self.assertEqual(claude_allow.get("systemMessage"), "Welcome!")
        self.assertEqual(
            claude_allow.get("hookSpecificOutput", {}).get("hookEventName"),
            "SessionStart",
        )
        self.assertEqual(
            claude_allow.get("hookSpecificOutput",
                             {}).get("additionalContext"),
            "TPU v5p environment active.",
        )
        self.assertNotIn("decision", claude_allow)

        # Claude Code Deny
        claude_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Missing TPU credentials",
            },
            "claude-code",
            "SessionStart",
            "init-guard",
        )
        self.assertEqual(claude_deny.get("decision"), "block")
        self.assertEqual(claude_deny.get("reason"), "Missing TPU credentials")

        # 2. Codex Allow & Deny
        codex_allow = runtime.translate_canonical_output(
            {
                "decision": "allow",
                "context": "Context for codex",
            },
            "codex",
            "SessionStart",
            "init-guard",
        )
        self.assertEqual(codex_allow.get("decision"), "allow")
        self.assertEqual(codex_allow.get("additional_context"),
                         "Context for codex")

        codex_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "GCE cluster unavailable",
            },
            "codex",
            "SessionStart",
            "init-guard",
        )
        self.assertEqual(codex_deny.get("decision"), "block")
        self.assertEqual(codex_deny.get("reason"), "GCE cluster unavailable")

        # 3. Cursor Allow & Deny
        cursor_allow = runtime.translate_canonical_output(
            {
                "decision": "allow",
                "context": "Context for cursor",
            },
            "cursor",
            "sessionStart",
            "init-guard",
        )
        self.assertEqual(cursor_allow.get("additional_context"),
                         "Context for cursor")

        cursor_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Cluster down",
            },
            "cursor",
            "sessionStart",
            "init-guard",
        )
        self.assertEqual(cursor_deny.get("decision"), "cancel")
        self.assertEqual(cursor_deny.get("reason"), "Cluster down")

        # 4. Antigravity (injectSteps)
        ag_allow = runtime.translate_canonical_output(
            {
                "decision": "allow",
                "context": "Active profile: TPU v6e.",
            },
            "antigravity",
            "SessionStart",
            "init-guard",
        )
        self.assertIn("injectSteps", ag_allow)
        self.assertEqual(len(ag_allow["injectSteps"]), 1)
        self.assertEqual(
            ag_allow["injectSteps"][0]["ephemeralMessage"],
            "Active profile: TPU v6e.",
        )

        ag_deny = runtime.translate_canonical_output(
            {
                "decision": "deny",
                "reason": "Invalid workspace",
            },
            "antigravity",
            "SessionStart",
            "init-guard",
        )
        self.assertIn("injectSteps", ag_deny)
        self.assertEqual(len(ag_deny["injectSteps"]), 1)
        self.assertIn("Invalid workspace",
                      ag_deny["injectSteps"][0]["ephemeralMessage"])

    def test_dispatch_hook_session_start_e2e(self):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        hook_script = scripts_dir / "init_session.py"
        hook_script.write_text(
            "import sys, json\n"
            "data = json.loads(sys.stdin.read())\n"
            "assert data.get('event') == 'SessionStart'\n"
            "print(json.dumps({\n"
            "    'decision': 'allow',\n"
            "    'context': 'TPU environment ready',\n"
            "    'message': 'Session initialized'\n"
            "}))\n",
            encoding="utf-8",
        )

        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="init-session",
            event="SessionStart",
            script_rel="scripts/init_session.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "session_id": "sess-init-123",
                "source": "startup",
                "cwd": str(self.workspace_dir),
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res.get("systemMessage"), "Session initialized")
        self.assertEqual(
            res.get("hookSpecificOutput", {}).get("additionalContext"),
            "TPU environment ready",
        )

    def test_dispatch_hook_session_start_antigravity_turn_bypass(self):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        hook_script = scripts_dir / "fail_on_call.py"
        hook_script.write_text(
            "import sys, json\n"
            "# If called on turn > 1, emit error\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )

        # Turn 2: invocationNum=2 -> should be bypassed, returning allow ({'injectSteps': []}) with exit code 0
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="fail-on-call",
            event="SessionStart",
            script_rel="scripts/fail_on_call.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-123",
                "invocationNum": 2,
                "initialNumSteps": 5,
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"injectSteps": []})

    # The compute note is not replayed; its own test below covers that.
    @mock.patch.object(runtime.compute.cli,
                       "check_target_at_session_start",
                       return_value=None)
    def test_dispatch_hook_session_start_antigravity_turn_2_reinjects_active_run_id(
            self, _check):
        """Turn 2+ replays the run namespace that Turn 1 allocated.

        The builtin emits a bare `run_id`; the context wrapped around it is
        synthesised during translation. Turn 2+ must reproduce that same text
        from cache, so caching a canonical result keyed only on `context` is
        not enough -- it has to cover `run_id` too.
        """
        # Turn 1: the builtin allocates the namespace and seeds the cache.
        res1, code1 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-turn2",
                "invocationNum": 0,
                "initialNumSteps": 1,
            }),
        )
        self.assertEqual(code1, 0)
        turn_1_context = res1["injectSteps"][0]["ephemeralMessage"]
        run_id = runtime.resolve_session_run_id("conv-turn2")
        self.assertIn(run_id, turn_1_context)

        # Turn 2: initialNumSteps=5, invocationNum=0 -> replay cache, allocating nothing.
        res2, code2 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-turn2",
                "invocationNum": 0,
                "initialNumSteps": 5,
            }),
        )
        self.assertEqual(code2, 0)
        self.assertEqual(
            res2, {"injectSteps": [{
                "ephemeralMessage": turn_1_context
            }]})
        # The namespace must be the one from Turn 1, not a freshly allocated one.
        self.assertEqual(runtime.resolve_session_run_id("conv-turn2"), run_id)

        # The context is ephemeral: each model call sees only what is
        # injected into it, so later calls in a turn get it again too.
        for init_steps, invocation in ((1, 2), (5, 3)):
            res3, _ = runtime.dispatch_hook(
                harness="antigravity",
                package="test-pkg@1.0.0",
                hook_name="session-run-init",
                event="SessionStart",
                builtin="start-run",
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "conversationId": "conv-turn2",
                    "invocationNum": invocation,
                    "initialNumSteps": init_steps,
                }),
            )
            self.assertEqual(res3, res2)

    def test_dispatch_hook_session_start_antigravity_turn_1_executes(self):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        hook_script = scripts_dir / "turn1_hook.py"
        hook_script.write_text(
            "import sys, json\n"
            "json.dump({'decision': 'allow', 'context': 'Turn 1 context'}, sys.stdout)\n",
            encoding="utf-8",
        )

        # Turn 1: initialNumSteps=1 (user prompt), invocationNum=0 -> should NOT be bypassed
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="turn1-hook",
            event="SessionStart",
            script_rel="scripts/turn1_hook.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-123",
                "invocationNum": 0,
                "initialNumSteps": 1,
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            res, {"injectSteps": [{
                "ephemeralMessage": "Turn 1 context"
            }]})

    def test_dispatch_hook_session_start_antigravity_multi_hook_context_caching(
            self):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        telemetry_script = scripts_dir / "telemetry.py"
        telemetry_script.write_text(
            "import sys, json\n"
            "json.dump({'decision': 'allow', 'context': 'TPU Cluster: v4-8 Ready'}, sys.stdout)\n",
            encoding="utf-8",
        )

        # Turn 1: Hook A (builtin:start-run)
        res_a1, code_a1 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-multi",
                "invocationNum": 0,
                "initialNumSteps": 1,
            }),
        )
        self.assertEqual(code_a1, 0)
        self.assertEqual(len(res_a1["injectSteps"]), 1)
        self.assertIn("TPU agent suite run namespace:",
                      res_a1["injectSteps"][0]["ephemeralMessage"])

        # Turn 1: Hook B (user script)
        res_b1, code_b1 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="telemetry-hook",
            event="SessionStart",
            script_rel="scripts/telemetry.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-multi",
                "invocationNum": 0,
                "initialNumSteps": 1,
            }),
        )
        self.assertEqual(code_b1, 0)
        self.assertEqual(
            res_b1,
            {"injectSteps": [{
                "ephemeralMessage": "TPU Cluster: v4-8 Ready"
            }]})

        # Overwrite telemetry script to fail if invoked on Turn 2+
        telemetry_script.write_text("import sys\nsys.exit(1)\n",
                                    encoding="utf-8")

        # Turn 2: Hook A should return only Hook A's cached context
        res_a2, code_a2 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-multi",
                "invocationNum": 0,
                "initialNumSteps": 3,
            }),
        )
        self.assertEqual(code_a2, 0)
        self.assertEqual(len(res_a2["injectSteps"]), 1)
        self.assertIn("TPU agent suite run namespace:",
                      res_a2["injectSteps"][0]["ephemeralMessage"])

        # Turn 2: Hook B should return only Hook B's cached context without running script
        res_b2, code_b2 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="telemetry-hook",
            event="SessionStart",
            script_rel="scripts/telemetry.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-multi",
                "invocationNum": 0,
                "initialNumSteps": 3,
            }),
        )
        self.assertEqual(code_b2, 0)
        self.assertEqual(
            res_b2,
            {"injectSteps": [{
                "ephemeralMessage": "TPU Cluster: v4-8 Ready"
            }]})

        # Turn 2: Hook C (unregistered hook) returns empty injectSteps
        res_c2, code_c2 = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="unregistered-hook",
            event="SessionStart",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId": "conv-multi",
                "invocationNum": 0,
                "initialNumSteps": 3,
            }),
        )
        self.assertEqual(code_c2, 0)
        self.assertEqual(res_c2, {"injectSteps": []})

        # Each hook's context lives in its own file, sharing no document.
        context_files = sorted(p.name
                               for p in (utils.get_sessions_root() /
                                         "conv-multi.contexts").iterdir())
        self.assertEqual(context_files,
                         ["session-run-init.json", "telemetry-hook.json"])

    def test_dispatch_hook_session_start_skips_cache_for_single_fire_harness(
            self):
        """Only the per-turn harnesses read the cache back, so only they write it.

        Claude Code fires SessionStart once per session and injects context
        directly, so caching there is pure write amplification on a session
        record nothing will ever consult.
        """
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "ctx.py").write_text(
            "import sys, json\n"
            "json.dump({'decision': 'allow', 'context': 'CC ctx'}, sys.stdout)\n",
            encoding="utf-8",
        )

        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="cc-hook",
            event="SessionStart",
            script_rel="scripts/ctx.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "session_id": "conv-cc",
                "source": "startup"
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["hookSpecificOutput"]["additionalContext"],
                         "CC ctx")
        self.assertIsNone(
            runtime.resolve_session_hook_context("conv-cc", "cc-hook"))

    def test_session_writers_touch_disjoint_files(self):
        """The run record and each hook's context never share a document.

        This is what makes concurrent hook dispatch safe: with no shared file
        there is no read-modify-write to interleave, so neither writer can
        clobber the other.
        """
        runtime.record_session_run(
            session_id="conv-keep",
            run_id="20260913T170000Z-998877665544",
            package="test-pkg@1.0.0",
        )
        session_file = runtime.get_session_file("conv-keep")
        created_at = utils.load_json(session_file)["created_at"]
        session_mtime = session_file.stat().st_mtime_ns

        # Caching hook context must not rewrite the run record at all.
        runtime.record_session_hook_context(
            session_id="conv-keep",
            hook_name="telemetry-hook",
            context_text="TPU Cluster: v4-8 Ready",
            package="test-pkg@1.0.0",
        )
        self.assertEqual(session_file.stat().st_mtime_ns, session_mtime)

        # Re-recording a run must not evict context cached by any hook.
        runtime.record_session_run(
            session_id="conv-keep",
            run_id="20260914T170000Z-112233445566",
            package="test-pkg@1.0.0",
        )
        session_doc = utils.load_json(session_file)
        self.assertEqual(session_doc["run_id"],
                         "20260914T170000Z-112233445566")
        self.assertEqual(session_doc["created_at"], created_at)
        self.assertEqual(
            runtime.resolve_session_hook_context("conv-keep",
                                                 "telemetry-hook"),
            "TPU Cluster: v4-8 Ready")

        # Re-recording identical state must not rewrite the file.
        mtime = session_file.stat().st_mtime_ns
        runtime.record_session_run(
            session_id="conv-keep",
            run_id="20260914T170000Z-112233445566",
            package="test-pkg@1.0.0",
        )
        self.assertEqual(session_file.stat().st_mtime_ns, mtime)

    def test_session_lookups_do_not_touch_the_filesystem(self):
        """A miss must not leave state behind.

        resolve_session_run_id runs on every normalized event, and the context
        lookup runs on every turn of an emulated SessionStart. If the path
        builders created directories, the store would accumulate an empty
        directory per conversation that was merely asked about.
        """
        sessions_root = utils.get_sessions_root()
        shutil.rmtree(sessions_root, ignore_errors=True)

        self.assertIsNone(runtime.resolve_session_run_id("conv-absent"))
        self.assertIsNone(
            runtime.resolve_session_hook_context("conv-absent", "any-hook"))

        self.assertFalse(sessions_root.exists(),
                         "lookups created state for a session never recorded")

    def test_session_record_readers_route_through_the_safe_reader(self):
        """A corrupt record degrades to a cache miss, never a crash.

        load_json_doc's own behaviour is covered in test_utils; what matters
        here is that both resolvers go through it. Reading a record with
        load_json instead would raise inside hook dispatch and take down
        session start for the whole conversation.
        """
        session_file = runtime.get_session_file("conv-bad")
        context_file = runtime.get_hook_context_file("conv-bad", "any-hook")
        for path in (session_file, context_file):
            path.parent.mkdir(parents=True, exist_ok=True)
            # Valid JSON, wrong shape: the case that raised AttributeError.
            path.write_text("[1, 2, 3]", encoding="utf-8")

        self.assertIsNone(
            runtime.resolve_session_hook_context("conv-bad", "any-hook"))
        self.assertIsNone(runtime.resolve_session_run_id("conv-bad"))

    def test_purge_data_clears_runs_and_session_records(self):
        """Purging runs must not leave sessions pointing at deleted namespaces.

        Session records are derived from runs: they name a run_id and cache the
        context advertising its artifact path. Surviving a purge, they would
        hand a live conversation a namespace that no longer exists.
        """
        run_doc = runtime.start_run(package="test-pkg@1.0.0",
                                    workspace=self.workspace_dir)
        runtime.record_session_run(
            session_id="conv-purge",
            run_id=run_doc["run_id"],
            package="test-pkg@1.0.0",
        )
        runtime.record_session_hook_context(
            session_id="conv-purge",
            hook_name="session-run-init",
            context_text="cached context",
            package="test-pkg@1.0.0",
        )
        self.assertTrue(utils.get_run_root(run_doc["run_id"]).is_dir())

        with contextlib.redirect_stdout(io.StringIO()):
            runtime.purge_data(force=True)

        self.assertFalse(utils.get_runs_root().is_dir())
        self.assertIsNone(runtime.resolve_session_run_id("conv-purge"))
        self.assertIsNone(
            runtime.resolve_session_hook_context("conv-purge",
                                                 "session-run-init"))

    def test_purge_data_declined_keeps_everything(self):
        runtime.record_session_run(
            session_id="conv-keep-purge",
            run_id="20260913T170000Z-998877665544",
            package="test-pkg@1.0.0",
        )
        with mock.patch("builtins.input", return_value="n"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                runtime.purge_data()

        self.assertIn("cancelled", err.getvalue())
        self.assertEqual(runtime.resolve_session_run_id("conv-keep-purge"),
                         "20260913T170000Z-998877665544")

    def test_dispatch_hook_session_start_antigravity_subagent_bypassed(self):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        hook_script = scripts_dir / "fail_on_call.py"
        hook_script.write_text(
            "import sys, json\n"
            "# If called for a subagent, exit with error\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )

        # Worktree transcript path indicates subagent -> should be bypassed with allow
        res, code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="subagent-test",
            event="SessionStart",
            script_rel="scripts/fail_on_call.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "conversationId":
                "subagent-conv-456",
                "invocationNum":
                0,
                "initialNumSteps":
                1,
                "transcriptPath":
                "/Users/test/workspace/worktrees/subagent-123/transcript.jsonl",
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"injectSteps": []})

    def test_dispatch_hook_builtin_start_run(self):
        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "session_id": "sess-builtin-1",
                "source": "startup",
                "cwd": str(self.workspace_dir),
            }),
        )
        self.assertEqual(code, 0)
        self.assertIn("systemMessage", res)
        additional_ctx = res.get("hookSpecificOutput",
                                 {}).get("additionalContext", "")
        self.assertIn("TPU agent suite run namespace:", additional_ctx)

        session_file = utils.get_sessions_root() / "sess-builtin-1.json"
        self.assertTrue(session_file.is_file())
        session_data = json.loads(session_file.read_text(encoding="utf-8"))
        self.assertEqual(session_data["session_id"], "sess-builtin-1")
        first_run_id = session_data["run_id"]
        self.assertTrue(first_run_id)

        # Calling start-run again for the same session must be idempotent
        res2, code2 = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "session_id": "sess-builtin-1",
                "source": "startup",
                "cwd": str(self.workspace_dir),
            }),
        )
        self.assertEqual(code2, 0)
        session_data2 = json.loads(session_file.read_text(encoding="utf-8"))
        self.assertEqual(session_data2["run_id"], first_run_id)
        additional_ctx2 = res2.get("hookSpecificOutput",
                                   {}).get("additionalContext", "")
        self.assertIn(f"TPU agent suite run namespace: {first_run_id}",
                      additional_ctx2)

    def _start_session(self, session_id):
        res, code = runtime.dispatch_hook(
            harness="claude-code",
            package="test-pkg@1.0.0",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            timeout=30,
            workspace=self.workspace_dir,
            raw_input_text=json.dumps({
                "session_id": session_id,
                "source": "startup",
                "cwd": str(self.workspace_dir),
            }),
        )
        self.assertEqual(code, 0)
        return res.get("hookSpecificOutput", {}).get("additionalContext", "")

    def test_start_run_adds_the_compute_check_to_the_context(self):
        note = "Compute target checked for this session: t (ok)"
        with mock.patch.object(runtime.compute.cli,
                               "check_target_at_session_start",
                               return_value=note) as check:
            context = self._start_session("sess-check-1")
            self._start_session("sess-check-1")  # resumed: checked again
        self.assertIn("TPU agent suite run namespace:", context)
        self.assertIn(note, context)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(check.call_args.args[-1], 30)  # the hook's timeout

    def test_later_turns_on_google_harnesses_do_not_replay_the_compute_check(
            self):
        note = "Compute target FAILED its check"

        def turn(initial_num_steps):
            res, code = runtime.dispatch_hook(
                harness="antigravity",
                package="test-pkg@1.0.0",
                hook_name="session-run-init",
                event="SessionStart",
                builtin="start-run",
                timeout=30,
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "conversationId": "conv-check",
                    "invocationNum": 0,
                    "initialNumSteps": initial_num_steps,
                }),
            )
            self.assertEqual(code, 0)
            return res["injectSteps"][0]["ephemeralMessage"]

        with mock.patch.object(runtime.compute.cli,
                               "check_target_at_session_start",
                               return_value=note):
            first = turn(1)
            later = turn(5)
        self.assertIn(note, first)
        self.assertIn("TPU agent suite run namespace:", later)
        self.assertNotIn(note, later)

    def test_start_run_survives_a_compute_check_that_raises(self):
        with mock.patch.object(runtime.compute.cli,
                               "check_target_at_session_start",
                               side_effect=RuntimeError("boom")), \
                mock.patch("sys.stderr", io.StringIO()):
            context = self._start_session("sess-check-2")
        self.assertIn("TPU agent suite run namespace:", context)

    def test_multi_session_concurrency_and_run_id_propagation(self):
        # Register two concurrent sessions in the same workspace
        runtime.record_session_run(
            session_id="session-tab-1",
            run_id="20260913T100000Z-aaaaaaaaaaaa",
            package="test-pkg@1.0.0",
        )
        runtime.record_session_run(
            session_id="session-tab-2",
            run_id="20260913T100001Z-bbbbbbbbbbbb",
            package="test-pkg@1.0.0",
        )

        # Verify distinct resolution
        self.assertEqual(
            runtime.resolve_session_run_id("session-tab-1"),
            "20260913T100000Z-aaaaaaaaaaaa",
        )
        self.assertEqual(
            runtime.resolve_session_run_id("session-tab-2"),
            "20260913T100001Z-bbbbbbbbbbbb",
        )

        # Verify normalization assigns respective run_ids to each event's context
        ev1 = runtime.normalize_inbound_event(
            raw_payload={
                "session_id": "session-tab-1",
                "tool_name": "Bash",
                "tool_input": {}
            },
            harness="claude-code",
            event="PreToolUse",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
        )
        ev2 = runtime.normalize_inbound_event(
            raw_payload={
                "session_id": "session-tab-2",
                "tool_name": "Bash",
                "tool_input": {}
            },
            harness="claude-code",
            event="PreToolUse",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
        )
        self.assertEqual(ev1["context"]["run_id"],
                         "20260913T100000Z-aaaaaaaaaaaa")
        self.assertEqual(ev2["context"]["run_id"],
                         "20260913T100001Z-bbbbbbbbbbbb")

    def test_antigravity_child_conversation_resolves_parent_run_id(self):
        parent_cid = "conv-parent-999"
        child_cid = "subagent-child-888"
        parent_run_id = "20260913T120000Z-999999999999"

        # Record run_id for parent session
        runtime.record_session_run(
            session_id=parent_cid,
            run_id=parent_run_id,
            package="test-pkg@1.0.0",
        )

        # Set up real filesystem descriptor in fake gemini brain home
        fake_home = self.temp_dir / "fake_home"
        subagent_file = (fake_home / ".gemini" / "antigravity-cli" / "brain" /
                         parent_cid / ".system_generated" / "subagents" /
                         f"{child_cid}.json")
        subagent_file.parent.mkdir(parents=True, exist_ok=True)
        subagent_file.write_text("{}", encoding="utf-8")

        with mock.patch("pathlib.Path.home", return_value=fake_home):
            # Verify child resolves to parent run_id via disk discovery
            self.assertEqual(
                runtime.resolve_session_run_id(child_cid),
                parent_run_id,
            )
            self.assertIsNone(runtime.resolve_session_run_id("non-existent"), )

            # Inbound normalization for child subagent PreToolUse
            payload = {
                "conversationId": child_cid,
                "toolCall": {
                    "name": "view_file",
                    "args": {
                        "AbsolutePath": "/test/file.py"
                    },
                },
            }
            ev = runtime.normalize_inbound_event(
                raw_payload=payload,
                harness="antigravity",
                event="PreToolUse",
                package="test-pkg@1.0.0",
                workspace=self.workspace_dir,
            )
            self.assertEqual(ev["context"]["session_id"], child_cid)
            self.assertEqual(ev["context"]["run_id"], parent_run_id)


class TestWorkspaceRootResolution(unittest.TestCase):
    """Where a hook decides the workspace is.

    `hooks.json` installs inside the harness's own directory, so a hook command
    runs with that directory as its cwd rather than the workspace. Resolving
    from there has to climb out, or every run-scoped path the hook computes is
    wrong by one level -- silently, since each path still exists.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        (self.workspace / ".tpu-agent-suite").mkdir(parents=True)
        (self.workspace / ".tpu-agent-suite" / "current_run_id").write_text(
            "20260928T225805Z-a09b00bd2610", encoding="utf-8")

    def test_harness_directories_resolve_to_their_parent(self):
        """Every directory a harness owns, including Antigravity's `.agents`."""
        for name in (".agents", ".agent", ".claude", ".cursor"):
            with self.subTest(harness_dir=name):
                harness_dir = self.workspace / name
                harness_dir.mkdir(exist_ok=True)
                with mock.patch.object(pathlib.Path,
                                       "cwd",
                                       return_value=harness_dir):
                    self.assertEqual(runtime.resolve_workspace_root(),
                                     self.workspace.resolve())

    def test_explicit_workspace_wins(self):
        """An argument is an answer, not a hint; nothing below should run."""
        self.assertEqual(runtime.resolve_workspace_root(self.workspace),
                         self.workspace.resolve())

    def test_a_nested_directory_climbs_to_the_marker(self):
        """From real work inside the tree, the marker directory is the root."""
        nested = self.workspace / "src" / "deep"
        nested.mkdir(parents=True)
        with mock.patch.object(pathlib.Path, "cwd", return_value=nested):
            self.assertEqual(runtime.resolve_workspace_root(),
                             self.workspace.resolve())

    def test_a_repository_inside_the_workspace_is_not_the_workspace(self):
        """A session that `cd`s into a repo it created still finds its records.

        The session-start hook writes `.tpu-agent-suite/current_run_id` where
        the harness says the session is. A later command run from a git
        repository below that directory has to resolve the same place, or it
        looks for the active run in a directory that has none.
        """
        repo = self.workspace / "model-repo"
        (repo / ".git").mkdir(parents=True)
        nested = repo / "scripts"
        nested.mkdir()
        for cwd in (repo, nested):
            with self.subTest(cwd=cwd.name):
                with mock.patch.object(pathlib.Path, "cwd", return_value=cwd):
                    self.assertEqual(runtime.resolve_workspace_root(),
                                     self.workspace.resolve())

    def test_an_export_from_a_repository_finds_the_session(self):
        """The trajectory export, run from that repo, names the right session.

        This is the failure it had: the run id was found by climbing, the
        session record was looked up under the repo, and the export stopped
        with "could not determine which session to export".
        """
        run_id = "20260928T225805Z-a09b00bd2610"
        tpu_agent_suite = self.workspace / ".tpu-agent-suite"
        (tpu_agent_suite / "current_run_id").write_text(run_id,
                                                        encoding="utf-8")
        repo = self.workspace / "model-repo"
        (repo / ".git").mkdir(parents=True)
        store = self.workspace / "store"
        with mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(store)}):
            utils.write_json(
                utils.get_sessions_root() / "4b8b541a.json", {
                    "session_id": "4b8b541a",
                    "run_id": run_id,
                    "created_at": "2026-09-28T22:58:05Z",
                })
            utils.get_run_root(run_id).mkdir(parents=True)
            with mock.patch.object(pathlib.Path, "cwd", return_value=repo):
                # The run id comes from climbing out of the repo to the
                # workspace's marker; the session from the central store.
                session_id, found_run, _ = runtime.resolve_trajectory_target(
                    run_id=utils.get_active_run_id(
                        runtime.resolve_workspace_root()) or "")
        self.assertEqual((session_id, found_run), ("4b8b541a", run_id))

    def test_a_directory_without_an_active_run_is_not_the_workspace(self):
        """A bare `.tpu-agent-suite` is not a marker; the one with a run is.

        The installer and the logging hooks create that directory in places
        no session started in, so only the active-run file says a session did.
        """
        stray = self.workspace / "model-repo"
        (stray / ".tpu-agent-suite" / "logs").mkdir(parents=True)
        with mock.patch.object(pathlib.Path, "cwd", return_value=stray):
            self.assertEqual(runtime.resolve_workspace_root(),
                             self.workspace.resolve())

    def test_a_repository_is_not_a_workspace(self):
        """With no active run above the cwd, the cwd answers, not a git root."""
        repo = self.tmp / "bare-repo"
        (repo / ".git").mkdir(parents=True)
        nested = repo / "src"
        nested.mkdir()
        with mock.patch.object(pathlib.Path, "cwd", return_value=nested):
            self.assertEqual(runtime.resolve_workspace_root(),
                             nested.resolve())


class TestCompletedArtifactRuleIsConditional(unittest.TestCase):
    """Only a `completed` result must carry an artifact.

    The rule lives in the runtime rather than the schema because the validator
    has no conditional keywords, and stating it unconditionally would be worse
    than not stating it: an agent that failed has nothing to attach, so the only
    way to satisfy the rule would be to claim success.
    """

    def test_completed_without_an_artifact_is_rejected(self):
        self.assertTrue(
            runtime.check_completed_has_artifact({
                "status": "completed",
                "artifacts": []
            }))

    def test_a_failure_owes_no_artifact(self):
        for status in ("failed", "blocked", "cancelled"):
            with self.subTest(status=status):
                self.assertEqual(
                    runtime.check_completed_has_artifact({
                        "status": status,
                        "artifacts": []
                    }), [])

    def test_a_non_result_is_not_this_rule_s_business(self):
        """Anything that is not a result object passes through untouched."""
        for value in ("text", [], 7, None):
            with self.subTest(value=value):
                self.assertEqual(runtime.check_completed_has_artifact(value),
                                 [])


class TestInstalledPackageResolution(unittest.TestCase):
    """Finding the versioned package directory under the TPU agent suite home."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.packages = self.tmp / "packages"
        self.packages.mkdir()

    def _resolve(self, package):
        with mock.patch.object(utils,
                               "get_tpu_agent_suite_home",
                               return_value=self.tmp):
            return runtime.latest_installed_package(package)

    def test_no_package_named_resolves_to_nothing(self):
        self.assertIsNone(self._resolve(""))

    def test_the_highest_version_wins(self):
        for version in ("0.9.0", "0.10.0", "0.2.0"):
            (self.packages / f"pkg@{version}").mkdir()
        resolved = self._resolve("pkg")
        self.assertIsNotNone(resolved)
        # 0.10.0 beats 0.9.0 numerically, which string ordering gets wrong.
        self.assertEqual(resolved.name, "pkg@0.10.0")

    def test_files_and_other_packages_are_skipped(self):
        (self.packages / "pkg@1.0.0").mkdir()
        (self.packages / "other@9.0.0").mkdir()
        (self.packages / "pkg@8.0.0").write_text("not a directory",
                                                 encoding="utf-8")
        (self.packages / "unversioned").mkdir()
        self.assertEqual(self._resolve("pkg").name, "pkg@1.0.0")


class TestStopEventDisambiguation(unittest.TestCase):
    """Antigravity emits one native `Stop` for two different things.

    A subagent finishing and the root loop ending arrive on the same wire
    event, so only the payload separates them. Getting it backwards sends a
    subagent's completion down the root path, where a resume hook would try to
    restart a conversation that already ended.
    """

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def _normalize(self, payload):
        return runtime.normalize_inbound_event(raw_payload=payload,
                                               harness="antigravity",
                                               event="Stop",
                                               package="pkg",
                                               workspace=self.workspace)

    def test_no_subagent_metadata_is_a_root_stop(self):
        event = self._normalize({
            "terminationReason": "error",
            "error": "429 RESOURCE_EXHAUSTED",
            "conversationId": "conv-1",
        })
        self.assertEqual(event["event"], "Stop")
        self.assertEqual(event["termination"]["error"],
                         "429 RESOURCE_EXHAUSTED")
        self.assertEqual(event["context"]["conversation_id"], "conv-1")

    def test_top_level_subagent_metadata_is_a_subagent_stop(self):
        event = self._normalize({
            "role": "tpu-diagnostics",
            "terminationReason": "NO_TOOL_CALL"
        })
        self.assertEqual(event["event"], "SubagentStop")

    def test_nested_subagent_metadata_is_a_subagent_stop(self):
        """The role can arrive inside the tool call rather than beside it."""
        event = self._normalize({
            "toolCall": {
                "typeName": "tpu-diagnostics"
            },
            "terminationReason": "NO_TOOL_CALL",
        })
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "tpu-diagnostics")


class TestIdleFlagAbsenceIsAnnounced(unittest.TestCase):
    """A missing idle flag is read permissively, so it must not be silent.

    The gate that holds a stop hook until the agent reports itself idle reads
    "the harness did not say" as "allow". That is the safe default, but if
    Antigravity ever renamed `fullyIdle` the gate would go on allowing every
    stop with nothing in the logs to explain it -- the original bug wearing a
    different hat.
    """

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def _normalize(self, harness, event, payload):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            normalized = runtime.normalize_inbound_event(
                raw_payload=payload,
                harness=harness,
                event=event,
                package="pkg",
                workspace=self.workspace)
        return normalized, stderr.getvalue()

    def test_antigravity_payload_without_the_idle_flag_warns(self):
        # Both normalizers reach the flag, and they file it in different
        # places: a root stop under `termination`, a subagent stop under
        # `args`. Either one going quiet is the regression worth catching.
        for event, payload, read in (
            ("Stop", {
                "terminationReason": "NO_TOOL_CALL"
            }, lambda e: e["termination"]["fully_idle"]),
            ("Stop", {
                "role": "tpu-diagnostics"
            }, lambda e: e["args"].get("fully_idle")),
        ):
            with self.subTest(payload=payload):
                normalized, err = self._normalize("antigravity", event,
                                                  payload)
                self.assertIsNone(read(normalized))
                self.assertIn("fullyIdle", err)

    def test_a_reported_flag_and_a_harness_without_one_stay_quiet(self):
        idle, err = self._normalize("antigravity", "Stop", {
            "terminationReason": "NO_TOOL_CALL",
            "fullyIdle": False
        })
        self.assertIs(idle["termination"]["fully_idle"], False)
        self.assertEqual(err, "")

        # Claude Code declares no idle field at all, so there is nothing to
        # have gone missing and nothing to say about it.
        claude, err = self._normalize("claude-code", "Stop",
                                      {"stop_hook_active": False})
        self.assertIsNone(claude["termination"]["fully_idle"])
        self.assertEqual(err, "")


class TestBuiltinValidateSchema(unittest.TestCase):
    """Unit tests for builtin:validate-schema hook execution."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()

        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        # Set up installed package structure in central packages directory with tpu-agent-suite-build.json and schemas
        self.pkg_root = self.central_home / utils.PACKAGES_DIRNAME / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        (self.pkg_root / "schemas").mkdir()

        (self.pkg_root / "schemas" / "helper-request.json").write_text(
            json.dumps({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": ["prompt"],
                "properties": {
                    "prompt": {
                        "type": "string"
                    },
                    "run_id": {
                        "type": "string"
                    },
                },
                "examples": [{
                    "prompt": "Optimize TPU kernel"
                }],
            }),
            encoding="utf-8",
        )

        (self.pkg_root / "schemas" / "helper-result.json").write_text(
            json.dumps({
                "$schema":
                "https://json-schema.org/draft/2020-12/schema",
                "type":
                "object",
                "additionalProperties":
                False,
                "required": ["status"],
                "properties": {
                    "status": {
                        "type":
                        "string",
                        "enum": [
                            "completed", "failed", "invalid_input",
                            "needs_input"
                        ],
                    },
                    "summary": {
                        "type": "string"
                    },
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {
                                    "type": "string"
                                },
                                "run_id": {
                                    "type": "string"
                                },
                            },
                        },
                    },
                },
                "examples": [{
                    "status": "completed",
                    "summary": "done",
                    "artifacts": [{
                        "file": "summary.json"
                    }]
                }],
            }),
            encoding="utf-8",
        )

        # Matches the five shipped request schemas that make run_id mandatory
        # (worker, plan, test, implement, autotune). The helper-request fixture
        # above leaves it optional, so it cannot show what happens when a
        # caller omits a run_id the contract actually demands.
        (self.pkg_root / "schemas" / "strict-request.json").write_text(
            json.dumps({
                "$schema":
                "https://json-schema.org/draft/2020-12/schema",
                "type":
                "object",
                "additionalProperties":
                False,
                "required": ["prompt", "run_id"],
                "properties": {
                    "prompt": {
                        "type": "string"
                    },
                    "run_id": {
                        "type": "string"
                    },
                },
                "examples": [{
                    "prompt": "Optimize TPU kernel",
                    "run_id": "20260914T120000Z-000000000000",
                }],
            }),
            encoding="utf-8",
        )

        # Mirrors kernel-worker-result.json: closed to extra properties, admits
        # "completed", and carries its deliverable in a typed field instead of
        # a generic `artifacts` array.
        (self.pkg_root / "schemas" / "typed-result.json").write_text(
            json.dumps({
                "$schema":
                "https://json-schema.org/draft/2020-12/schema",
                "type":
                "object",
                "additionalProperties":
                False,
                "required": ["status", "kernel_artifact"],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failed", "completed"],
                    },
                    "kernel_artifact": {
                        "type": "object"
                    },
                },
                "examples": [{
                    "status": "completed",
                    "kernel_artifact": {
                        "file": "kernel.py"
                    },
                }],
            }),
            encoding="utf-8",
        )

        (self.pkg_root / "tpu-agent-suite-build.json").write_text(
            json.dumps({
                "version": "1.0.0",
                "source": "test-pkg",
                "contracts": {
                    "helper": {
                        "accepts": "schemas/helper-request.json",
                        "produces": "schemas/helper-result.json",
                    },
                    "orchestrator": {
                        "accepts": "schemas/helper-request.json",
                        "produces": "schemas/helper-result.json",
                    },
                    "typed-worker": {
                        "accepts": "schemas/helper-request.json",
                        "produces": "schemas/typed-result.json",
                    },
                    "strict-worker": {
                        "accepts": "schemas/strict-request.json",
                        "produces": "",
                    },
                    "uncontracted": {
                        "accepts": "",
                        "produces": "",
                    },
                },
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_pre_tool_use_delegate_validation(self):
        """PreToolUse validates delegation arguments against accepts schema, allowing valid and denying invalid payloads."""
        valid_res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": "helper",
                "args": {
                    "subagent": "helper",
                    "prompt": "Optimize TPU kernel",
                },
                "context": {
                    "run_id": "20260913T120000Z-abcdef123456",
                },
            },
        )
        self.assertEqual(valid_res["decision"], "allow")

        invalid_res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": "helper",
                "args": {
                    "subagent": "helper",
                    "prompt": "Optimize TPU kernel",
                    "unsupported_field": 123,
                },
            },
        )
        self.assertEqual(invalid_res["decision"], "deny")
        self.assertIn("accepts schema", invalid_res["reason"])
        self.assertIn("unsupported_field", invalid_res["reason"])
        self.assertIn("Expected payload schema example:",
                      invalid_res["reason"])

    def _delegate_event(self, tool_input):
        """Build a delegate event the way a harness actually produces one."""
        return runtime._normalize_tool_event(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Agent",
                "tool_input": tool_input,
                "session_id": "sess-1",
            },
            "claude-code",
            "PreToolUse",
            "test-pkg@1.0.0",
            self.workspace_dir,
        )

    def test_delegate_validation_ignores_harness_routing_keys(self):
        """A plain-text delegation validates as its prompt, not as the harness's argument bag.

        Normalization seeds the canonical args with the harness's own tool_input,
        so `subagent_type` and `description` ride along into the validator. A
        closed accepts schema rejected both, which made every plain-text
        delegation impossible and left a hand-wrapped JSON prompt as the only
        way through. This goes through `_normalize_tool_event` rather than
        around it on purpose: a hand-built args dict is already clean, so it
        cannot reproduce the leak.
        """
        event = self._delegate_event({
            "subagent_type": "helper",
            "description": "Optimize a kernel",
            "prompt": "Optimize TPU kernel",
        })
        self.assertEqual(event["tool"], "delegate")
        self.assertIn("subagent_type", event["args"])
        self.assertIn("description", event["args"])

        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload=event,
        )
        self.assertEqual(res["decision"], "allow")

    def test_delegate_validation_rejects_unknown_field_in_authored_payload(
            self):
        """Dropping routing keys is not a blanket amnesty for unknown fields.

        A JSON prompt is written by the caller, so every key in it is deliberate
        and a misspelled one is worth a denial rather than a silent drop.
        """
        event = self._delegate_event({
            "subagent_type":
            "helper",
            "description":
            "Optimize a kernel",
            "prompt":
            json.dumps({
                "prompt": "Optimize TPU kernel",
                "unsupported_field": 123,
            }),
        })
        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload=event,
        )
        self.assertEqual(res["decision"], "deny")
        self.assertIn("unsupported_field", res["reason"])
        self.assertIn("Expected payload schema example:", res["reason"])

    def test_missing_required_run_id_is_denied_at_the_caller(self):
        """A run_id the subagent cannot see must be refused while it can be fixed.

        Supplying the context run_id to the validator alone would let the
        delegation through while the subagent still receives a prompt without
        one. It cannot recover it either: subagents bypass SessionStart, so the
        prompt is their only source, and five shipped result schemas require
        the value. The denial has to land on the caller, which holds the
        run_id and can resend.
        """
        run_id = "20260913T120000Z-abcdef123456"
        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": "strict-worker",
                "args": {
                    "subagent": "strict-worker",
                    "prompt": json.dumps({"prompt": "Optimize TPU kernel"}),
                },
                "context": {
                    "run_id": run_id
                },
            },
        )
        self.assertEqual(res["decision"], "deny")
        # The example carries a placeholder run_id, and it is the most
        # copyable thing in the message, so the real one has to be stated.
        self.assertIn(run_id, res["reason"])
        self.assertLess(res["reason"].index(run_id),
                        res["reason"].index("20260914T120000Z-000000000000"))

        supplied = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": "strict-worker",
                "args": {
                    "subagent":
                    "strict-worker",
                    "prompt":
                    json.dumps({
                        "prompt": "Optimize TPU kernel",
                        "run_id": run_id,
                    }),
                },
                "context": {
                    "run_id": run_id
                },
            },
        )
        self.assertEqual(supplied["decision"], "allow")

    def test_delegate_routing_keys_come_from_the_mapping_table(self):
        """Routing keys are read off tool-mappings.json, not hardcoded.

        Hardcoding them is what let claude-code's pair go unhandled, so a
        harness added to the table must contribute its names automatically.
        """
        keys = runtime._delegate_routing_keys()
        self.assertIn("subagent", keys)
        self.assertIn("subagent_type", keys)
        self.assertIn("description", keys)
        # A dotted spec addresses into a nested arg; the bag carries its root.
        self.assertIn("Subagents", keys)
        self.assertNotIn("Subagents.TypeName", keys)

    def test_subagent_stop_validation(self):
        """SubagentStop verifies output JSON schema and prose rejection."""
        run_id = "20260913T120000Z-abcdef123456"

        # 1. Valid result envelope -> allow
        valid_res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event":
                "SubagentStop",
                "subagent":
                "helper",
                "context": {
                    "run_id": run_id
                },
                "output":
                json.dumps({
                    "status":
                    "completed",
                    "summary":
                    "Kernel compiled successfully",
                    "artifacts": [{
                        "file": "summary.json",
                        "run_id": run_id
                    }],
                }),
            },
        )
        self.assertEqual(valid_res["decision"], "allow")

        # 2. Non-JSON prose output -> deny
        prose_res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "helper",
                "output": "I am done with the optimization loop!",
            },
        )
        self.assertEqual(prose_res["decision"], "deny")
        self.assertIn("must be a valid JSON object", prose_res["reason"])
        self.assertIn("Expected output schema example:", prose_res["reason"])

    def test_subagent_stop_validates_despite_running_background_agents(self):
        """Other live agents in `background_tasks` do not excuse a stop.

        Claude Code lists every live agent of the session there with no
        parent link, so a running entry of a type this agent could spawn may
        be a sibling's child. Background tasks are disabled at install, so the
        agent's own children have always returned before it stops; its stop
        is the only one it gets, and skipping it would skip validation for
        good.
        """

        def _stop(agent_type, agent_id, live_agents):
            return json.dumps({
                "hook_event_name":
                "SubagentStop",
                "session_id":
                "sess-nested",
                "agent_id":
                agent_id,
                "agent_type":
                agent_type,
                "last_assistant_message":
                "Done; see the report.",
                "background_tasks": [{
                    "id": live_id,
                    "type": "subagent",
                    "status": "running",
                    "agent_type": live_type,
                } for live_id, live_type in live_agents],
            })

        cases = {
            "sibling of the same type":
            _stop("orchestrator", "agent-orch-1", [
                ("agent-orch-1", "orchestrator"),
                ("agent-orch-2", "orchestrator"),
            ]),
            "a sibling's child":
            _stop("orchestrator", "agent-orch-1", [
                ("agent-orch-1", "orchestrator"),
                ("agent-helper-1", "helper"),
            ]),
            "nothing else running":
            _stop("orchestrator", "agent-orch-1",
                  [("agent-orch-1", "orchestrator")]),
        }
        for label, raw_input_text in cases.items():
            with self.subTest(label):
                res, code = runtime.dispatch_hook(
                    harness="claude-code",
                    package="test-pkg@1.0.0",
                    workspace=self.workspace_dir,
                    hook_name="validate-subagent-result",
                    event="SubagentStop",
                    raw_input_text=raw_input_text,
                    builtin="validate-schema",
                )
                self.assertEqual(code, 0)
                self.assertEqual(res.get("decision"), "block")

    def test_skipped_subagent_stop_says_why_on_stderr(self):
        """A stop the gate filters must not vanish silently.

        Claude Code sends an empty `agent_type` for its own internal agents.
        The hooks skip that stop, and stderr says so and why, so a missing
        study record can be told apart from a hook that never ran.
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res, code = runtime.dispatch_hook(
                harness="claude-code",
                package="test-pkg@1.0.0",
                workspace=self.workspace_dir,
                hook_name="validate-subagent-result",
                event="SubagentStop",
                raw_input_text=json.dumps({
                    "hook_event_name": "SubagentStop",
                    "session_id": "sess-internal",
                    "agent_id": "agent-suggest-1",
                    "agent_type": "",
                    "last_assistant_message": "not JSON",
                }),
                builtin="validate-schema",
            )
        self.assertEqual(code, 0)
        self.assertNotEqual(res.get("decision"), "block")
        self.assertIn("[validate-subagent-result] SubagentStop skipped",
                      stderr.getvalue())
        self.assertIn("no subagent name", stderr.getvalue())

    def test_routine_subagent_stop_skips_stay_quiet(self):
        """Skips this PR did not introduce say nothing on stderr.

        On Antigravity a nameless stop is every root turn and a stop not fully
        idle is every mid-wait pause; both were always skipped and fire far
        too often to log.
        """
        cases = {
            "root turn": {
                "conversationId": "conv-root-1",
                "fullyIdle": True,
            },
            "mid-wait pause": {
                "role": "helper",
                "conversationId": "conv-agy-1",
                "fullyIdle": False,
            },
        }
        for label, payload in cases.items():
            with self.subTest(label):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    res, code = runtime.dispatch_hook(
                        harness="antigravity",
                        package="test-pkg@1.0.0",
                        workspace=self.workspace_dir,
                        hook_name="validate-subagent-result",
                        event="SubagentStop",
                        raw_input_text=json.dumps({
                            **payload, "finalModelOutput":
                            "not JSON"
                        }),
                        builtin="validate-schema",
                    )
                self.assertEqual(code, 0)
                self.assertEqual(res.get("decision"), "allow")
                self.assertNotIn("SubagentStop skipped", stderr.getvalue())

    def test_subagent_stop_waits_until_the_harness_reports_itself_idle(self):
        """Antigravity delegates asynchronously, so a stop it calls not fully idle is a turn boundary.

        Prose that would otherwise be denied has to pass while the subagent's
        own work is still outstanding.
        """
        agy_payload = {
            "role": "helper",
            "conversationId": "conv-agy-1",
            "finalModelOutput": "Delegated the benchmark; waiting on it.",
            "fullyIdle": False,
        }
        res_paused, code_paused = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            hook_name="validate-subagent-result",
            event="SubagentStop",
            raw_input_text=json.dumps(agy_payload),
            builtin="validate-schema",
        )
        self.assertEqual(code_paused, 0)
        self.assertEqual(res_paused.get("decision"), "allow")

        # Once the subagent is idle, the same prose is held to the contract.
        agy_payload["fullyIdle"] = True
        res_done, code_done = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            hook_name="validate-subagent-result",
            event="SubagentStop",
            raw_input_text=json.dumps(agy_payload),
            builtin="validate-schema",
        )
        self.assertEqual(code_done, 0)
        self.assertEqual(res_done.get("decision"), "continue")

    def test_subagent_stop_last_valid_json_wins(self):
        """SubagentStop selects the last candidate JSON that satisfies the produces contract."""
        run_id = "20260913T120000Z-abcdef123456"
        valid_envelope = {
            "status": "completed",
            "summary": "Finished successfully",
            "artifacts": [{
                "file": "summary.json",
                "run_id": run_id
            }],
        }
        invalid_json = {"step": 1, "note": "intermediate draft"}

        # 1. Earlier invalid draft followed by valid deliverable surrounded by prose -> allow
        output_draft_then_valid = (
            "Draft step:\n"
            f"```json\n{json.dumps(invalid_json)}\n```\n"
            "Final result envelope:\n"
            f"```json\n{json.dumps(valid_envelope)}\n```\n"
            "All tests passed!")
        res_draft = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "helper",
                "context": {
                    "run_id": run_id
                },
                "output": output_draft_then_valid,
            },
        )
        self.assertEqual(res_draft["decision"], "allow")

        # 2. Valid deliverable followed by invalid debug dump -> allow (valid envelope wins)
        output_valid_then_debug = (
            f"```json\n{json.dumps(valid_envelope)}\n```\n"
            "Debug output:\n"
            f"```json\n{json.dumps(invalid_json)}\n```")
        res_debug = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "helper",
                "context": {
                    "run_id": run_id
                },
                "output": output_valid_then_debug,
            },
        )
        self.assertEqual(res_debug["decision"], "allow")

    def test_subagent_stop_multiple_invalid_jsons_denied(self):
        """When no candidate satisfies the contract, SubagentStop denies reporting the last error."""
        run_id = "20260913T120000Z-abcdef123456"
        output_all_invalid = (
            "```json\n"
            '{"unsupported": 1}\n'
            "```\n"
            "```json\n"
            '{"status": "completed"}\n'  # Missing required artifacts
            "```")
        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "helper",
                "context": {
                    "run_id": run_id
                },
                "output": output_all_invalid,
            },
        )
        self.assertEqual(res["decision"], "deny")

    def test_typed_artifact_schema_accepts_its_own_declared_example(self):
        """A closed schema without an `artifacts` array must not be unsatisfiable.

        The runtime's artifact rule is not part of any schema, so applying it
        to a schema that reports its deliverable through a typed field and
        forbids additional properties leaves the agent no legal output: the
        payload is denied for lacking `artifacts`, and adding `artifacts` is
        denied as an additional property. Because SubagentStop denials carry a
        followup_message, that pair loops -- and the example quoted back in the
        denial is this very payload.
        """
        schema = json.loads((self.pkg_root / "schemas" /
                             "typed-result.json").read_text(encoding="utf-8"))
        declared_example = schema["examples"][0]

        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "typed-worker",
                "output": json.dumps(declared_example),
            },
        )
        self.assertEqual(res["decision"], "allow")

        # The escape hatch really is closed, so the allow above is the only
        # reachable outcome rather than one of two.
        self.assertTrue(
            utils.validate_instance(
                {
                    **declared_example, "artifacts": [{
                        "file": "kernel.py"
                    }]
                }, schema))

    def test_subagent_stop_still_requires_artifacts_when_schema_declares_them(
            self):
        """Narrowing the artifact rule must not switch it off where it applies.

        `helper-result.json` declares an `artifacts` array, so a completed result
        that omits one is still prose pretending to be a deliverable.
        """
        res = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "helper",
                "output": json.dumps({
                    "status": "completed",
                    "summary": "done",
                }),
            },
        )
        self.assertEqual(res["decision"], "deny")

    def test_subagent_without_contract_allowed(self):
        """Subagents without declared contracts pass through unconditionally."""
        res_pre = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": "uncontracted",
                "args": {
                    "any_arg": "value"
                },
            },
        )
        self.assertEqual(res_pre["decision"], "allow")

        res_stop = runtime._builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload={
                "event": "SubagentStop",
                "subagent": "uncontracted",
                "output": "Raw prose output is fine",
            },
        )
        self.assertEqual(res_stop["decision"], "allow")

    def test_dispatch_hook_multi_subagent_batch_filtering(self):
        """In Antigravity, multiple subagents in a single batch are filtered individually."""
        raw_payload = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "helper",
                            "Prompt":
                            json.dumps({"prompt": "Valid subagent 1"}),
                        },
                        {
                            "TypeName":
                            "helper",
                            "Prompt":
                            json.dumps({
                                "prompt": "Invalid subagent 2",
                                "unexpected": True
                            }),
                        },
                    ]
                }
            }
        }
        response, exit_code = runtime.dispatch_hook(
            harness="antigravity",
            package="test-pkg@1.0.0",
            hook_name="validate-delegate-request",
            event="PreToolUse",
            builtin="validate-schema",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(raw_payload),
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(response["decision"], "allow")
        self.assertIn("overwrite", response)
        overwrite_subs = response["overwrite"]["Subagents"]
        self.assertEqual(len(overwrite_subs), 1)
        self.assertEqual(overwrite_subs[0]["TypeName"], "helper")
        self.assertIn("Valid subagent 1", overwrite_subs[0]["Prompt"])


class TestBuiltinSkillTools(unittest.TestCase):
    """Tests for builtin:skill-tools, which keeps a skill's tool limit for the session."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)
        self.pkg_root = self.central_home / utils.PACKAGES_DIRNAME / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        self._write_build_info({
            "router": ["ask", "delegate", "read", "Skill", "ToolSearch"],
            "exporter":
            None,
        })

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _write_build_info(self, skill_allowed_tools):
        build_info = {"version": "1.0.0", "source": "test-pkg"}
        if skill_allowed_tools is not None:
            build_info["skill_allowed_tools"] = skill_allowed_tools
        (self.pkg_root / "tpu-agent-suite-build.json").write_text(
            json.dumps(build_info), encoding="utf-8")

    def _run(self, event, tool, args=None, session="s1", raw_input=None):
        payload: dict[str, Any] = {
            "event": event,
            "tool": tool,
            "args": args or {},
            "context": {
                "session_id": session
            },
            "raw_input": raw_input or {},
        }
        if event == "PostToolUse":
            payload["success"] = True
        return runtime._builtin_skill_tools(package="test-pkg@1.0.0",
                                            workspace=self.workspace_dir,
                                            event_payload=payload)

    def _invoke_skill(self, name, **kwargs):
        return self._run("PostToolUse", "Skill", {"skill": name}, **kwargs)

    def _shell(self, **kwargs):
        return self._run("PreToolUse", "shell", {"command": "ls"}, **kwargs)

    def test_no_active_skill_allows_everything(self):
        """Without a tool-limited skill, shell is allowed."""
        self.assertEqual(self._shell()["decision"], "allow")

    def test_limited_skill_denies_shell_for_the_rest_of_the_session(self):
        """After the router loads, shell stays denied on later calls."""
        self._invoke_skill("test-pkg:router")
        for _ in range(2):
            res = self._shell()
            self.assertEqual(res["decision"], "deny")
            self.assertIn("router", res["reason"])
            self.assertIn("Delegate", res["reason"])
        self.assertTrue(runtime.get_active_skill_file("s1").exists())

    def test_limited_skill_denies_tools_not_on_its_allow_list(self):
        """Unmapped, MCP and unknown tools are denied without being named."""
        self._invoke_skill("router")
        for tool in ("PowerShell", "mcp__github__create_issue", "CronCreate",
                     "SomeFutureTool"):
            with self.subTest(tool=tool):
                self.assertEqual(
                    self._run("PreToolUse", tool)["decision"], "deny")

    def test_limited_skill_allows_its_own_tools_and_delegation(self):
        """The router can still read, ask, delegate and use always-safe tools."""
        self._invoke_skill("router")
        for tool in ("read", "ask", "delegate", "Skill", "ToolSearch"):
            with self.subTest(tool=tool):
                self.assertEqual(
                    self._run("PreToolUse", tool)["decision"], "allow")

    def test_limited_skill_still_allows_this_packages_setup_commands(self):
        """The main thread can set compute up; nothing else gets the shell."""
        self._invoke_skill("router")
        cmd = functools.partial(runtime.compute.config.runtime_command,
                                "test-pkg@1.0.0")
        runtime_py = runtime.compute.config.RUNTIME
        # The rendered guide names the runtime as `~/...`.
        home = mock.patch.dict(os.environ, {"HOME": str(runtime_py.parent)})
        for command, decision in (
            (cmd("compute",
                 "setup apply --profile p --answers '{\"a\": 1}'"), "allow"),
            (cmd("config", "show").replace(str(runtime_py),
                                           "~/" + runtime_py.name), "allow"),
            (cmd("compute", "run --script x.py"), "deny"),
            (cmd("compute", "setup verify; rm -rf ~"), "deny"),
            (cmd("compute", "setup verify\nrm -rf ~"), "deny"),
            (cmd("compute", "setup plan --profile $(whoami)"), "deny"),
            (cmd("config", "show").replace("test-pkg@1.0.0",
                                           "other@1"), "deny"),
        ):
            with self.subTest(command=command), home:
                self.assertEqual(
                    self._run("PreToolUse", "shell",
                              {"command": command})["decision"], decision)

    def test_subagents_are_not_limited(self):
        """Events from a subagent are allowed and don't change the main thread's limit."""
        sub = {"agent_id": "a1"}
        self._invoke_skill("router")
        self.assertEqual(self._shell(raw_input=sub)["decision"], "allow")
        self._invoke_skill("exporter", raw_input=sub)
        self.assertEqual(self._shell()["decision"], "deny")

    def test_unlimited_package_skill_lifts_the_limit(self):
        """Loading a package skill with no tool limit lifts it."""
        self._invoke_skill("router")
        self._invoke_skill("/exporter")
        self.assertEqual(self._shell()["decision"], "allow")

    def test_skill_from_another_package_keeps_the_limit(self):
        """A skill this package doesn't ship leaves the limit in place."""
        self._invoke_skill("router")
        self._invoke_skill("other-plugin:helper")
        self.assertEqual(self._shell()["decision"], "deny")

    def test_sessions_are_independent(self):
        """A limit in one session doesn't affect another."""
        self._invoke_skill("router", session="s1")
        self.assertEqual(self._shell(session="s2")["decision"], "allow")

    def test_build_without_skill_tools_allows_everything(self):
        """An older build with no skill_tools table never limits tools."""
        self._write_build_info(None)
        self._invoke_skill("router")
        self.assertEqual(self._shell()["decision"], "allow")

    def test_claude_events_are_mapped_end_to_end(self):
        """Claude's Skill and Bash events reach the hook as skill and shell."""

        def dispatch(event, raw):
            return runtime.dispatch_hook(harness="claude-code",
                                         package="test-pkg@1.0.0",
                                         hook_name="skill-tools",
                                         event=event,
                                         builtin="skill-tools",
                                         workspace=self.workspace_dir,
                                         raw_input_text=json.dumps(raw))

        base = {"session_id": "s1", "cwd": str(self.workspace_dir)}
        dispatch(
            "PostToolUse", {
                **base, "hook_event_name": "PostToolUse",
                "tool_name": "Skill",
                "tool_input": {
                    "skill": "tpu-agent:router"
                },
                "tool_response": {
                    "success": True
                }
            })
        res, _ = dispatch(
            "PreToolUse", {
                **base, "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "kubectl get pods"
                }
            })
        self.assertEqual(res["hookSpecificOutput"]["permissionDecision"],
                         "deny")


class TestMissingPackageDegradesGracefully(unittest.TestCase):
    """A hook whose package has gone missing must still answer the harness.

    Hook commands bake a versioned package tag at compile time, so upgrading,
    renaming, or removing the package mid-session leaves the tag pointing at a
    directory that no longer exists. Package resolution raises for that, and an
    escaping error costs the dispatch its stdout: the harness is handed an
    empty string and a non-zero exit instead of a decision.
    """

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        # An empty store: nothing the hook names can resolve.
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.temp_dir / "empty_home")

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_contract_resolution_skips_rather_than_raising(self):
        """Validation degrades to "no contract", the way its other failures do.

        Every other unresolvable input here -- absent build file, unparseable
        build file, unknown agent -- yields no contract rather than raising, so
        package resolution has to behave the same or the builtin cannot return.
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            accepts, produces = runtime._resolve_agent_contract(
                "ghost-pkg@9.9.9", self.workspace_dir, "helper")

        self.assertIsNone(accepts)
        self.assertIsNone(produces)
        # Enforcing nothing without saying so is its own failure.
        self.assertIn("ghost-pkg@9.9.9", stderr.getvalue())

    def test_validate_schema_builtin_still_returns_a_decision(self):
        """The builtin answers even when it cannot find what to validate."""
        with contextlib.redirect_stderr(io.StringIO()):
            res = runtime._builtin_validate_schema(
                package="ghost-pkg@9.9.9",
                workspace=self.workspace_dir,
                event_payload={
                    "event": "SubagentStop",
                    "subagent": "helper",
                    "output": json.dumps({"status": "completed"}),
                },
            )
        self.assertEqual(res["decision"], "allow")

    def test_script_hook_denies_with_the_reason(self):
        """A script hook denies instead of skipping, as it does for any
        script it cannot locate -- but the harness still gets a decision.
        """
        with contextlib.redirect_stderr(io.StringIO()):
            response, code = runtime.dispatch_hook(
                harness="claude-code",
                package="ghost-pkg@9.9.9",
                hook_name="some-script-hook",
                event="PreToolUse",
                script_rel="scripts/whatever.py",
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "session_id": "s1",
                    "tool_name": "Bash",
                    "tool_input": {},
                }),
            )

        self.assertEqual(code, 0)
        specific = response["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn("is not installed", specific["permissionDecisionReason"])


class TestHookTimeoutVerdictBeatsHarness(unittest.TestCase):
    """A hung hook's deny must be written before the harness gives up on it.

    The harness is registered with the hook's timeout plus
    `compiler.HOOK_DISPATCH_HEADROOM_S`, and hook-dispatch gets the bare
    timeout. Dispatch enforces that timeout on the action but finishes a
    little after it -- interpreter start before, the verdict after. Should
    that overrun ever reach the headroom, the harness kills dispatch first,
    the deny is lost, and Claude Code runs the tool the hook was guarding.

    Run as a real subprocess, as the harness runs it, so interpreter start and
    module import are counted in the time.
    """

    TIMEOUT_S = 1

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_hung_hook_is_denied_within_the_headroom(self):
        env = {
            **os.environ, "TPU_AGENT_SUITE_HOME": str(self.temp_dir / "home")
        }
        started = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(runtime.__file__)),
                "hook-dispatch",
                "--harness",
                "claude-code",
                "--package",
                "test-pkg@1.0.0",
                "--hook",
                "hung-hook",
                "--event",
                "PreToolUse",
                "--timeout",
                str(self.TIMEOUT_S),
                "--workspace",
                str(self.workspace_dir),
                "--command",
                "sleep 30",
            ],
            input=json.dumps({
                "session_id": "s1",
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "echo hi"
                },
            }),
            capture_output=True,
            text=True,
            env=env,
            # Far past the headroom: a hang here is a failure, not a stall.
            timeout=60,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(proc.returncode, 0, proc.stderr)
        specific = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn(f"timed out after {self.TIMEOUT_S} seconds",
                      specific["permissionDecisionReason"])
        # The hook's own timeout is what ended it, not something sooner...
        self.assertGreaterEqual(elapsed, self.TIMEOUT_S)
        # ...and the verdict landed while the harness was still listening.
        self.assertLess(elapsed,
                        self.TIMEOUT_S + compiler.HOOK_DISPATCH_HEADROOM_S)

    # A builtin that never returns and keeps writing to stderr, as a stuck
    # builtin that reports progress would; the child then dispatches to it.
    CHATTERING_BUILTIN = (
        "import sys\n"
        "from framework import runtime\n"
        "def chatter(**_kwargs):\n"
        "    while True:\n"
        "        sys.stderr.write('x' * 4096)\n"
        "runtime.BUILTIN_HOOK_ACTIONS['record-event'] = chatter\n"
        "raise SystemExit(runtime.main(sys.argv[1:]))\n")

    def test_an_abandoned_builtin_does_not_crash_dispatch_at_exit(self):
        """Shutdown aborts on a stream lock the thread holds (SIGABRT).

        The verdict may already be out, but to Claude Code a non-zero exit is
        a non-blocking error: the output is ignored and the tool runs.
        """
        proc = subprocess.run(
            [
                sys.executable, "-c", self.CHATTERING_BUILTIN, "hook-dispatch",
                "--harness", "claude-code", "--package", "test-pkg@1.0.0",
                "--hook", "stuck-builtin", "--event", "PreToolUse",
                "--timeout",
                str(self.TIMEOUT_S), "--workspace",
                str(self.workspace_dir), "--builtin", "record-event"
            ],
            input=json.dumps({
                "session_id": "s1",
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {},
            }),
            capture_output=True,
            text=True,
            cwd=Path(runtime.__file__).parents[1],
            env={
                **os.environ, "TPU_AGENT_SUITE_HOME":
                str(self.temp_dir / "home")
            },
            timeout=60,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        specific = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn("timed out", specific["permissionDecisionReason"])


class TestHookFailurePolicy(unittest.TestCase):
    """What a hook that fails to reach a verdict turns into.

    A guard must fail closed; a hook that only observes, or that sits where a
    deny keeps the agent from returning, must not hold it up. Either way the
    failure is recorded, since a hook that fails open is otherwise silent.
    """

    PAYLOAD = {
        "event": "PreToolUse",
        "tool": "shell",
        "args": {},
        "context": {
            "session_id": "s1",
            "run_id": "r1",
            "package": "test-pkg@1.0.0",
        },
    }

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.workspace = self.temp_dir / "workspace"
        self.workspace.mkdir()
        # The failure log is central: point the store at the temp dir so the
        # failures recorded here never reach the real one.
        home = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.temp_dir / "home")})
        home.start()
        self.addCleanup(home.stop)
        self.failure_log = utils.get_tpu_agent_suite_home(
        ) / utils.HOOK_FAILURE_LOG

    def _command(self, command: str, on_failure: str) -> dict:
        with contextlib.redirect_stderr(io.StringIO()):
            return runtime.execute_single_hook_action(
                event_payload=self.PAYLOAD,
                hook_name="h",
                workspace=self.workspace,
                package="test-pkg@1.0.0",
                command=command,
                timeout=5,
                on_failure=on_failure,
            )

    def _builtin(self, handler, on_failure: str, timeout: float = 5) -> dict:
        with contextlib.redirect_stderr(io.StringIO()):
            return runtime.run_builtin_hook_action(
                handler,
                package="test-pkg@1.0.0",
                workspace=self.workspace,
                event_payload=self.PAYLOAD,
                hook_name="h",
                timeout=timeout,
                on_failure=on_failure,
            )

    def _records(self) -> list[dict]:
        return utils.read_jsonl(self.failure_log)

    def test_non_json_output_is_quoted_in_the_reason(self):
        """What the hook printed is the clue to why it failed."""
        denied = self._command("echo not json", "deny")
        self.assertEqual(
            denied["reason"],
            "Hook 'h' emitted non-JSON output on stdout: 'not json'")
        # Capped: the reason reaches the log and the model's context.
        noisy = self._command("printf 'x%.0s' $(seq 600)", "deny")
        self.assertIn(repr("x" * 500 + "..."), noisy["reason"])
        self.assertNotIn("x" * 501, noisy["reason"])
        self.assertEqual([r["kind"] for r in self._records()],
                         ["non_json", "non_json"])

    def test_an_explicit_deny_is_honoured_whatever_the_policy(self):
        """Only a failure to answer is covered; a refusal is an answer."""
        res = self._command(
            "echo '{\"decision\": \"deny\", \"reason\": \"no\"}'", "allow")
        self.assertEqual(res, {"decision": "deny", "reason": "no"})
        self.assertFalse(self.failure_log.exists())

    def test_output_that_cannot_be_decoded_follows_the_policy(self):
        # `text=True` decodes strictly: this used to escape as a
        # UnicodeDecodeError and leave dispatch without a verdict.
        command = r"printf '\377\376'"
        self.assertEqual(self._command(command, "allow"),
                         {"decision": "allow"})
        denied = self._command(command, "deny")
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("UnicodeDecodeError", denied["reason"])
        self.assertEqual([r["kind"] for r in self._records()],
                         ["exception", "exception"])

    def test_a_long_reason_keeps_only_its_end(self):
        """The whole stderr of a hook can be long, and can hold secrets."""
        denied = self._command(
            "printf 'secret%.0s' $(seq 1000) >&2; echo ' ValueError: bad' >&2;"
            " exit 3", "deny")
        [record] = self._records()
        # The denial the model reads and the record are the same text.
        self.assertEqual(denied["reason"], record["reason"])
        self.assertEqual(len(record["reason"]),
                         len("...") + runtime._HOOK_FAILURE_REASON_CHARS)
        self.assertTrue(record["reason"].startswith("..."))
        self.assertTrue(record["reason"].endswith("ValueError: bad"))

    def test_failures_are_recorded_under_either_policy(self):
        # Exit 2 is not read as a block: shell and `argparse` errors exit 2.
        usage_error = "echo 'error: bad arguments' >&2; exit 2"
        self.assertEqual(self._command(usage_error, "allow"),
                         {"decision": "allow"})
        self.assertEqual(self._command(usage_error, "deny"), {
            "decision": "deny",
            "reason": "error: bad arguments"
        })
        records = self._records()
        self.assertEqual([(r["on_failure"], r["decision"]) for r in records],
                         [("allow", "allow"), ("deny", "deny")])
        first = records[0]
        self.assertEqual(first["hook"], "h")
        self.assertEqual(first["event"], "PreToolUse")
        self.assertEqual(first["tool"], "shell")
        self.assertEqual(first["kind"], "exit")
        self.assertEqual(first["session_id"], "s1")
        self.assertEqual(first["run_id"], "r1")
        self.assertEqual(first["package"], "test-pkg@1.0.0")
        self.assertTrue(first["timestamp"])

    def _pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _kill_if_running(self, pid_file: Path) -> None:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)

    def test_a_timed_out_hook_takes_its_children_with_it(self):
        """Killing only `sh` left what it had started running, orphaned."""
        pid_file = self.workspace / "child.pid"
        self.addCleanup(self._kill_if_running, pid_file)
        with contextlib.redirect_stderr(io.StringIO()):
            res = runtime.execute_single_hook_action(
                event_payload=self.PAYLOAD,
                hook_name="h",
                workspace=self.workspace,
                package="test-pkg@1.0.0",
                command="cat >/dev/null; sleep 15 & echo $! > child.pid; wait",
                timeout=1,
                on_failure="deny",
            )
        self.assertIn("timed out", res["reason"])
        child = int(pid_file.read_text())
        # Its parent is gone too, so init reaps it, not us: give it a moment.
        deadline = time.monotonic() + 2
        while self._pid_running(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self._pid_running(child))

    def test_an_unwritable_log_does_not_cost_the_verdict(self):
        # A file where the log's directory belongs: the log cannot be created.
        self.failure_log.parent.parent.mkdir(parents=True, exist_ok=True)
        self.failure_log.parent.write_text("", encoding="utf-8")
        self.assertEqual(self._command("exit 3", "allow"),
                         {"decision": "allow"})
        self.assertEqual(self._command("exit 3", "deny")["decision"], "deny")

    def test_a_builtin_that_raises_follows_the_policy(self):
        broken = mock.Mock(side_effect=ValueError("kaput"))
        self.assertEqual(self._builtin(broken, "allow"), {"decision": "allow"})
        denied = self._builtin(broken, "deny")
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("ValueError: kaput", denied["reason"])
        self.assertEqual([r["kind"] for r in self._records()],
                         ["exception", "exception"])

    def _hang_until_released(self, **_kwargs) -> dict:
        """A builtin that does not answer until the test lets it go."""
        self.release.wait(30)
        return {"decision": "allow"}

    def test_a_builtin_that_hangs_follows_the_policy(self):
        self.release = threading.Event()
        self.addCleanup(self.release.set)

        started = time.monotonic()
        self.assertEqual(
            self._builtin(self._hang_until_released, "allow", timeout=0.2),
            {"decision": "allow"})
        denied = self._builtin(self._hang_until_released, "deny", timeout=0.2)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("timed out", denied["reason"])
        self.assertEqual([r["kind"] for r in self._records()],
                         ["timeout", "timeout"])


class TestDispatchFailure(unittest.TestCase):
    """An error in dispatch itself is settled by the hook's policy.

    Left to escape, it would exit non-zero with nothing on stdout, and the
    harness would decide: Claude Code runs the tool, Antigravity blocks it.
    """

    CLAUDE_INPUT = json.dumps({
        "session_id": "s1",
        "tool_name": "Bash",
        "tool_input": {},
    })

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.workspace = self.temp_dir / "workspace"
        self.workspace.mkdir()
        # The failure log is central: point the store at the temp dir so the
        # failures recorded here never reach the real one.
        home = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.temp_dir / "home")})
        home.start()
        self.addCleanup(home.stop)
        self.failure_log = utils.get_tpu_agent_suite_home(
        ) / utils.HOOK_FAILURE_LOG

    def _dispatch(self, on_failure: str, **kwargs) -> tuple[dict, int]:
        arguments = {
            "harness": "claude-code",
            "package": "test-pkg@1.0.0",
            "hook_name": "h",
            "event": "PreToolUse",
            "command": "true",
            "workspace": self.workspace,
            "raw_input_text": self.CLAUDE_INPUT,
            "on_failure": on_failure,
        }
        arguments.update(kwargs)
        with contextlib.redirect_stderr(io.StringIO()):
            return runtime.dispatch_hook(**arguments)

    def _records(self) -> list[dict]:
        return utils.read_jsonl(self.failure_log)

    def test_a_normalization_error_follows_the_policy(self):
        with mock.patch.object(runtime,
                               "normalize_inbound_event",
                               side_effect=RuntimeError("boom")):
            allowed, allowed_code = self._dispatch("allow")
            denied, denied_code = self._dispatch("deny")
        self.assertEqual((allowed_code, denied_code), (0, 0))
        self.assertEqual(allowed["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        verdict = denied["hookSpecificOutput"]
        self.assertEqual(verdict["permissionDecision"], "deny")
        self.assertIn("Dispatch of hook 'h' failed: RuntimeError: boom",
                      verdict["permissionDecisionReason"])
        records = self._records()
        self.assertEqual([(r["kind"], r["decision"]) for r in records],
                         [("dispatch", "allow"), ("dispatch", "deny")])
        self.assertEqual(records[0]["hook"], "h")
        self.assertEqual(records[0]["event"], "PreToolUse")

    def test_a_bug_under_the_hooks_action_follows_the_policy(self):
        with mock.patch.object(runtime,
                               "_run_hook_process",
                               side_effect=NameError("oops")):
            denied, code = self._dispatch("deny")
        self.assertEqual(code, 0)
        verdict = denied["hookSpecificOutput"]
        self.assertEqual(verdict["permissionDecision"], "deny")
        self.assertIn("NameError: oops", verdict["permissionDecisionReason"])
        self.assertEqual([r["kind"] for r in self._records()], ["dispatch"])

    def test_a_handback_is_answered_as_its_tool_event(self):
        # Claude Code rejects a SubagentStop-shaped reply to the hand-back's
        # PreToolUse as a hook error and runs the tool, losing the deny. The
        # payload is unreadable here: the command line alone decides.
        with mock.patch.object(runtime,
                               "normalize_inbound_event",
                               side_effect=RuntimeError("boom")):
            denied, code = self._dispatch("deny",
                                          event="SubagentStop",
                                          raw_input_text="not json",
                                          handback_event="PreToolUse")
        self.assertEqual(code, 0)
        self.assertNotIn("decision", denied)
        verdict = denied["hookSpecificOutput"]
        self.assertEqual(verdict["hookEventName"], "PreToolUse")
        self.assertEqual(verdict["permissionDecision"], "deny")
        self.assertIn("RuntimeError: boom",
                      verdict["permissionDecisionReason"])


class TestHookFanOut(unittest.TestCase):
    """A batched delegation must reach its verdict inside the harness budget.

    The hook runs once per subagent, and the harness allows the whole dispatch
    only the hook's own timeout plus headroom; hung subagents must not add up.
    """

    TIMEOUT = 1
    # Enough subagents that, checked in turn, they would outlast the budget.
    SUBAGENTS = tuple(f"agent-{i}"
                      for i in range(TIMEOUT +
                                     compiler.HOOK_DISPATCH_HEADROOM_S + 1))
    # Hangs on the first subagent only, and answers at once for the others.
    # Every sub-event carries the whole batch too; match the one it is for.
    SLOW_ON_FIRST = ('payload=$(cat); case "$payload" in '
                     '*\'"subagent": "agent-0"\'*) sleep 15;; esac; '
                     'echo \'{"decision": "allow"}\'')

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.workspace = self.temp_dir / "workspace"
        self.workspace.mkdir()
        # The failure log is central: point the store at the temp dir so the
        # failures recorded here never reach the real one.
        home = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.temp_dir / "home")})
        home.start()
        self.addCleanup(home.stop)
        self.failure_log = utils.get_tpu_agent_suite_home(
        ) / utils.HOOK_FAILURE_LOG
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def _hang_until_released(self, **_kwargs) -> dict:
        """A builtin that does not answer until the test lets it go."""
        self.release.wait(30)
        return {"decision": "allow"}

    def _dispatch(self, on_failure: str, **action) -> tuple[dict, float]:
        """Dispatches one delegation of every subagent; returns the timing."""
        raw = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [{
                        "TypeName": name,
                        "Prompt": "go"
                    } for name in self.SUBAGENTS]
                },
            }
        }
        started = time.monotonic()
        with contextlib.redirect_stderr(io.StringIO()):
            response, code = runtime.dispatch_hook(
                harness="antigravity",
                package="test-pkg@1.0.0",
                hook_name="h",
                event="PreToolUse",
                workspace=self.workspace,
                raw_input_text=json.dumps(raw),
                timeout=self.TIMEOUT,
                on_failure=on_failure,
                **action,
            )
        self.assertEqual(code, 0)
        return response, time.monotonic() - started

    def _assert_within_budget(self, elapsed: float) -> None:
        self.assertLess(elapsed,
                        self.TIMEOUT + compiler.HOOK_DISPATCH_HEADROOM_S)

    def _hung_actions(self) -> dict[str, dict]:
        return {
            "command": {
                "command": "sleep 15"
            },
            "builtin": {
                "builtin": "validate-schema"
            },
        }

    def test_hung_subagents_are_denied_within_the_budget(self):
        with mock.patch.dict(runtime.BUILTIN_HOOK_ACTIONS,
                             {"validate-schema": self._hang_until_released}):
            for kind, action in self._hung_actions().items():
                with self.subTest(kind=kind):
                    response, elapsed = self._dispatch("deny", **action)
                    self._assert_within_budget(elapsed)
                    self.assertEqual(response["decision"], "deny")
                    for name in self.SUBAGENTS:
                        self.assertIn(name, response["reason"])

    def test_a_hung_subagent_costs_only_itself(self):
        response, elapsed = self._dispatch("deny", command=self.SLOW_ON_FIRST)
        self._assert_within_budget(elapsed)
        self.assertEqual(response["decision"], "allow")
        self.assertEqual(response["overwrite"]["Subagents"], [{
            "TypeName": name,
            "Prompt": "go"
        } for name in self.SUBAGENTS[1:]])


class TestSessionStartLease(unittest.TestCase):
    """`start-run` leases the run, then sweeps the runs of ended sessions."""

    PKG = "test-pkg@1.0.0"
    OWNER = {"pid": 4242, "started": "99"}

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir()
        patcher = mock.patch.dict(os.environ,
                                  {"TPU_AGENT_SUITE_HOME": str(self.tmp)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _start(self, owner):
        """Dispatch start-run; returns the lease the sweep saw."""
        seen = []
        with mock.patch("framework.compute.lease.session_owner",
                        return_value=owner), \
                mock.patch("framework.compute.lease.spawn_sweep",
                           side_effect=lambda run: seen.append(
                               runtime.compute.lease.read_owner(run))):
            _, code = runtime.dispatch_hook(
                harness="claude-code",
                package=self.PKG,
                hook_name="hook",
                event="SessionStart",
                builtin="start-run",
                workspace=self.workspace,
                raw_input_text=json.dumps({"session_id": "sess-1"}),
            )
        self.assertEqual(code, 0)
        return seen[0] if seen else None

    def test_the_sweep_runs_after_the_claim_and_a_resume_takes_it_over(self):
        owner = self._start(self.OWNER)
        self.assertEqual((owner["pid"], owner["package"], owner["workspace"]),
                         (4242, self.PKG, str(self.workspace.resolve())))
        self.assertEqual(self._start({**self.OWNER, "pid": 5151})["pid"], 5151)
        # A resume whose owner cannot be identified releases the run.
        self.assertEqual(self._start(None), {})

    def test_failures_never_fail_the_session_start(self):
        with mock.patch("framework.compute.lease.claim",
                        side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(self._start(self.OWNER))
        self.assertIn("could not lease run", err.getvalue())


class TestSessionEndReapHook(unittest.TestCase):
    """The `reap-jobs` builtin: fires only when the session is really over.

    Registered on SessionEnd, it is what cancels a run's leftover workloads.
    On Claude Code SessionEnd is native and always means the end. Antigravity
    emulates it on its per-turn Stop, so the same hook fires at
    every turn boundary and has to leave a just-submitted job alone.
    """

    RUN_ID = "20260101T000000Z-777777777777"
    PKG = "test-pkg@1.0.0"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir(parents=True)
        # Session records are central: keep this class's out of the real store.
        home = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.tmp / "home")})
        home.start()
        self.addCleanup(home.stop)

    def _dispatch(self, harness, event, payload, session="sess-1"):
        runtime.record_session_run(session, self.RUN_ID, package=self.PKG)
        return runtime.dispatch_hook(
            harness=harness,
            package=self.PKG,
            hook_name="session-end-reap",
            event=event,
            builtin="reap-jobs",
            workspace=self.workspace,
            raw_input_text=json.dumps(payload),
        )

    @mock.patch("framework.compute.reap_jobs", return_value=(["job-1"], {}))
    @mock.patch("framework.compute.resolve_driver")
    def test_claude_code_session_end_reaps_the_sessions_run(
            self, mock_resolve_driver, mock_reap_jobs):
        res, code = self._dispatch("claude-code", "SessionEnd", {
            "session_id": "sess-1",
            "reason": "prompt_input_exit",
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_called_once()
        run_id, resolver = mock_reap_jobs.call_args.args
        self.assertEqual(run_id, self.RUN_ID)
        self.assertIn("job-1", res.get("systemMessage", ""))

        # The resolver hands reap a driver for the job's own backend and
        # profile, with the workspace's configuration behind it.
        mock_driver = mock.MagicMock()
        mock_resolve_driver.return_value = mock_driver
        self.assertIs(resolver("job-1"), mock_driver)
        mock_resolve_driver.assert_called_once_with(
            workspace=self.workspace.resolve(),
            package=self.PKG,
            run_id=self.RUN_ID,
            job_id="job-1",
        )

    @mock.patch("framework.compute.reap_jobs")
    def test_claude_code_turn_stop_never_reaps(self, mock_reap_jobs):
        """A Stop is a turn ending, whatever the payload says."""
        res, code = self._dispatch("claude-code", "Stop", {
            "session_id": "sess-1",
            "stop_hook_active": True,
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs")
    def test_unknown_session_is_a_no_op(self, mock_reap_jobs):
        res, code = self._dispatch("claude-code", "SessionEnd", {
            "session_id": "someone-else",
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs", return_value=([], {}))
    def test_antigravity_idle_terminal_stop_reaps(self, mock_reap_jobs):
        res, code = self._dispatch(
            "antigravity", "SessionEnd", {
                "conversationId": "sess-1",
                "terminationReason": "TERMINAL_STEP_TYPE",
                "fullyIdle": True,
            })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_called_once()
        self.assertEqual(mock_reap_jobs.call_args.args[0], self.RUN_ID)

    @mock.patch("framework.compute.reap_jobs")
    def test_antigravity_turn_boundaries_do_not_reap(self, mock_reap_jobs):
        """Every payload here is a loop pausing, not the session ending."""
        payloads = {
            "turn end": {
                "terminationReason": "NO_TOOL_CALL",
                "fullyIdle": True
            },
            "not idle": {
                "terminationReason": "TERMINAL_STEP_TYPE",
                "fullyIdle": False
            },
            "no idle flag at all": {
                "terminationReason": "TERMINAL_STEP_TYPE"
            },
            # The resume hook retries an error; reaping first would cancel
            # the jobs of a run about to be picked up again.
            "transient error": {
                "terminationReason": "ERROR",
                "fullyIdle": True
            },
            # The user stopping the agent is not the user ending the session.
            "user cancelled": {
                "terminationReason": "USER_CANCELED",
                "fullyIdle": True
            },
            "budget exhausted": {
                "terminationReason": "MAX_TOKEN_BUDGET_EXCEEDED",
                "fullyIdle": True
            },
            # A reason the enum gains later must not start cancelling jobs on
            # its own; that is the point of listing completions, not halts.
            "reason we have never seen": {
                "terminationReason": "TERMINATION_REASON_FROM_THE_FUTURE",
                "fullyIdle": True
            },
        }
        for label, extra in payloads.items():
            with self.subTest(label):
                res, code = self._dispatch("antigravity", "SessionEnd", {
                    "conversationId": "sess-1",
                    **extra
                })
                self.assertEqual(code, 0)
                mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs")
    def test_antigravity_subagent_stop_bypasses_the_hook(self, mock_reap_jobs):
        """A subagent finishing arrives on the same Stop; it is not the end."""
        res, code = self._dispatch(
            "antigravity", "SessionEnd", {
                "conversationId": "sess-1",
                "role": "tpu-diagnostics",
                "terminationReason": "NO_TOOL_CALL",
                "fullyIdle": True,
            })
        self.assertEqual(code, 0)
        self.assertEqual(res, {"decision": "stop"})
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs",
                side_effect=RuntimeError("kubectl: unauthorized"))
    def test_reap_failure_is_reported_not_raised(self, mock_reap_jobs):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res, code = self._dispatch("claude-code", "SessionEnd", {
                "session_id": "sess-1",
            })
        self.assertEqual(code, 0)
        self.assertIn("kubectl: unauthorized", stderr.getvalue())
        self.assertIn(self.RUN_ID, stderr.getvalue())


class TestSubagentHandback(unittest.TestCase):
    """A Claude Code subagent's result arrives by one of two roads.

    Outside auto mode a subagent ends its turn and the result is its last
    message (native SubagentStop), in the foreground or the background. In
    auto mode every subagent must call the injected `SubagentHandback` tool
    instead, and only that call is delivered. The same SubagentStop hook is
    registered on both, and the contract must be enforced exactly once
    whichever road is taken.
    """

    PKG = "test-pkg@1.0.0"
    RESULT = {"status": "success", "summary": "done", "artifacts": []}

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        scripts_dir = self.central_home / "packages" / self.PKG / "scripts"
        scripts_dir.mkdir(parents=True)
        self.fires = self.temp_dir / "fires.jsonl"
        # Records every fire, and refuses any output without a JSON status.
        (scripts_dir / "check.py").write_text(
            "import sys, json\n"
            "data = json.loads(sys.stdin.read())\n"
            f"with open({str(self.fires)!r}, 'a') as f:\n"
            "    f.write(json.dumps(data) + '\\n')\n"
            "if '\"status\"' in str(data.get('output') or ''):\n"
            "    print(json.dumps({'decision': 'allow'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'bad',\n"
            "                      'followup_message': 'send JSON'}))\n",
            encoding="utf-8",
        )

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _dispatch(self, payload, matcher=None):
        return runtime.dispatch_hook(
            harness="claude-code",
            package=self.PKG,
            hook_name="check",
            event="SubagentStop",
            script_rel="scripts/check.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(payload),
            matcher=matcher,
        )

    def _handback(self, message, agent_type="tpu-model-onboarding", **extra):
        return {
            "hook_event_name": "PreToolUse",
            "tool_name": "SubagentHandback",
            "agent_type": agent_type,
            "session_id": "sess-1",
            "tool_input": {
                "message": message
            },
            **extra,
        }

    def _fire_count(self):
        if not self.fires.exists():
            return 0
        return len(self.fires.read_text(encoding="utf-8").splitlines())

    def _write_transcript(self, records):
        path = self.temp_dir / "agent-1.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records),
                        encoding="utf-8")
        return path

    @staticmethod
    def _content(block):
        return {"message": {"content": [block]}}

    @staticmethod
    def _prompt(prompt_id, text, meta=False):
        # Every user record Claude Code writes for a run carries its promptId.
        record = {
            "type": "user",
            "promptId": prompt_id,
            "message": {
                "role": "user",
                "content": text
            },
        }
        if meta:
            record["isMeta"] = True
        return record

    def _native_stop(self,
                     transcript,
                     message="Report delivered.",
                     prompt_id=None,
                     permission_mode=None):
        payload = {
            "hook_event_name": "SubagentStop",
            "agent_type": "tpu-model-onboarding",
            "session_id": "sess-1",
            "last_assistant_message": message,
            "agent_transcript_path": str(transcript),
            # The parent session's file on Claude Code; must never be read.
            "transcript_path": str(self.temp_dir / "parent.jsonl"),
        }
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        if permission_mode is not None:
            payload["permission_mode"] = permission_mode
        return payload

    def test_other_tool_events_are_not_handbacks(self):
        for payload in (
                self._handback("x", hook_event_name="PostToolUse"),
                dict(self._handback("x"), tool_name="Read"),
        ):
            event = runtime.normalize_inbound_event(payload, "claude-code",
                                                    "SubagentStop", self.PKG,
                                                    self.workspace_dir)
            self.assertNotIn("handback", event["args"])

    def test_handback_without_event_name_is_matched_by_tool(self):
        payload = self._handback("x")
        del payload["hook_event_name"]
        event = runtime.normalize_inbound_event(payload, "claude-code",
                                                "SubagentStop", self.PKG,
                                                self.workspace_dir)
        self.assertIs(event["args"]["handback"], True)

    def test_refused_handback_answers_as_tool_denial(self):
        res, code = self._dispatch(self._handback("no json here"))
        self.assertEqual(code, 0)
        self.assertNotIn("decision", res)
        out = res["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "PreToolUse")
        self.assertEqual(out["permissionDecision"], "deny")
        self.assertEqual(out["permissionDecisionReason"], "send JSON")

    def _dispatch_script(self, decision, payload):
        scripts_dir = self.central_home / "packages" / self.PKG / "scripts"
        (scripts_dir / "bare.py").write_text(
            f"print({json.dumps(json.dumps({'decision': decision}))})\n",
            encoding="utf-8")
        return runtime.dispatch_hook(
            harness="claude-code",
            package=self.PKG,
            hook_name="bare",
            event="SubagentStop",
            script_rel="scripts/bare.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(payload),
        )

    def test_bare_refusal_still_gives_the_subagent_a_reason(self):
        # A script hook may refuse with no text at all; the denial reason is
        # the only thing the subagent reads back, so it is never empty.
        res, code = self._dispatch_script("deny", self._handback("x"))
        self.assertEqual(code, 0)
        out = res["hookSpecificOutput"]
        self.assertEqual(out["permissionDecision"], "deny")
        self.assertEqual(out["permissionDecisionReason"],
                         "Denied by hook bare")

    def test_retry_means_the_same_on_both_roads(self):
        # `retry` is a Stop decision. A native SubagentStop does not block on
        # it, so a hand-back must not be refused on it either.
        handback, _ = self._dispatch_script("retry", self._handback("x"))
        native, _ = self._dispatch_script(
            "retry", self._native_stop(self._write_transcript([])))
        self.assertEqual(handback["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertNotIn("decision", native)

    def test_handback_from_unmatched_subagent_is_bypassed(self):
        res, _ = self._dispatch(
            self._handback("no json", agent_type="doc_writer"),
            matcher="tpu-model-onboarding|tpu-kernel-development")
        self.assertEqual(res["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertEqual(self._fire_count(), 0)

    def test_one_check_per_delivery_across_both_roads(self):
        # The recorded auto-mode sequence: the subagent writes its report as
        # text and ends its turn before calling the tool, then calls it, then
        # ends its turn again with a closing remark. Only the call reaches
        # the caller, so only the call is checked. The stop's mode decides
        # alone: the transcript, missing here, is not read.
        transcript = self.temp_dir / "missing.jsonl"
        before, _ = self._dispatch(
            self._native_stop(transcript, "draft", permission_mode="auto"))
        handback, _ = self._dispatch(self._handback(json.dumps(self.RESULT)))
        after, _ = self._dispatch(
            self._native_stop(transcript,
                              "Report delivered.",
                              permission_mode="auto"))

        self.assertNotIn("decision", before)
        self.assertEqual(handback["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertNotIn("decision", after)
        fires = [
            json.loads(line)
            for line in self.fires.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(fires), 1)
        self.assertIs(fires[0]["args"]["handback"], True)
        self.assertEqual(fires[0]["subagent"], "tpu-model-onboarding")

    def test_native_stop_outside_auto_mode_is_held(self):
        # Outside auto mode, or with no mode on the payload, the final
        # message is the delivery and is held to the contract.
        transcript = self._write_transcript(
            [self._prompt("p1", "harness test")])
        for mode in ("default", None):
            with self.subTest(permission_mode=mode):
                res, _ = self._dispatch(
                    self._native_stop(transcript,
                                      prompt_id="p1",
                                      permission_mode=mode))
                self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 2)

    def _handback_call(self, call_id, result, is_error=False):
        # A SubagentHandback call and the tool result Claude Code records.
        tool_result = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result
        }
        if is_error:
            tool_result["is_error"] = True
        return [
            self._content({
                "type": "tool_use",
                "id": call_id,
                "name": "SubagentHandback"
            }),
            self._content(tool_result),
        ]

    def _auto_run_then_resume(self):
        # Recorded shape of a subagent that handed back in auto mode (run
        # "p1") and was then resumed in default mode (run "p2") in the same
        # transcript: Claude Code withdraws the tool for the resumed run, and
        # its plain final message is what the caller receives.
        return self._write_transcript([
            self._prompt("p1", "harness test, part 1"),
            *self._handback_call("t1", "Report delivered to your caller."),
            self._prompt("p2", "harness test, part 2", meta=True),
            self._prompt("p2",
                         "<system-reminder>SubagentHandback is not available "
                         "in this run ...</system-reminder>",
                         meta=True),
            self._content({
                "type": "text",
                "text": "handshake-2"
            }),
        ])

    def test_resumed_run_without_handback_is_held(self):
        # The earlier run's hand-back must not exempt the resumed run: its
        # native stop is the delivery and is held to the contract.
        transcript = self._auto_run_then_resume()
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "handshake-2",
                              prompt_id="p2",
                              permission_mode="default"))
        self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 1)

    def test_stop_without_a_known_run_is_held(self):
        # A payload that cannot name its run, or names one no record bears,
        # proves nothing, like an unreadable transcript: an earlier run's
        # hand-back must not exempt it, so the native stop is checked.
        transcript = self._auto_run_then_resume()
        for prompt_id in (None, "p9"):
            with self.subTest(prompt_id=prompt_id):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    res, _ = self._dispatch(
                        self._native_stop(transcript,
                                          "handshake-2",
                                          prompt_id=prompt_id))
                self.assertEqual(res.get("decision"), "block")
                self.assertIn("cannot locate the stopping run",
                              stderr.getvalue())
        self.assertEqual(self._fire_count(), 2)

    def test_prompt_id_named_only_in_prose_does_not_start_the_run(self):
        # The resumed run's id quoted in the earlier run's text must not
        # pull that run's hand-back into the resumed run.
        transcript = self._auto_run_then_resume()
        records = [
            json.loads(line)
            for line in transcript.read_text(encoding="utf-8").splitlines()
        ]
        records.insert(0, self._content({"type": "text", "text": "next: p2"}))
        transcript = self._write_transcript(records)
        res, _ = self._dispatch(
            self._native_stop(transcript, "handshake-2", prompt_id="p2"))
        self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 1)

    def test_run_switched_out_of_auto_mode_is_held(self):
        # Recorded (mode test C): delegated in auto mode, switched to default
        # mid-run. Claude Code refuses every call outside auto mode; the
        # caller's follow-up in the same turn (same prompt id) withdraws the
        # tool and the plain final message is delivered.
        transcript = self._write_transcript([
            self._prompt("p1", "sleep 60, then report"),
            *self._handback_call(
                "t1",
                "Only the auto-mode classifier can allow SubagentHandback: "
                "the session is not in auto mode",
                is_error=True),
            self._prompt("p1",
                         "<system-reminder>SubagentHandback is not available "
                         "in this run ...</system-reminder>",
                         meta=True),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "mode-test-C",
                              prompt_id="p1",
                              permission_mode="default"))
        self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 1)

    def test_auto_mode_run_never_offered_the_tool_is_held(self):
        # Recorded on Claude Code 2.1.241, headless and interactive alike:
        # auto mode, but the run is never given the tool, so its plain final
        # message is what the caller receives. Skipping it on the mode alone
        # let every result through unchecked.
        transcript = self._write_transcript([
            self._prompt("p1", "Scope the onboarding request"),
            self._content({
                "type": "text",
                "text": "Brief and plan written."
            }),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "Brief and plan written.",
                              prompt_id="p1",
                              permission_mode="auto"))
        self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 1)

    def test_auto_mode_run_offered_the_tool_is_a_turn_boundary(self):
        # Recorded on Claude Code 2.1.285: the offer is a meta record at the
        # start of the run, before its first stop, so the draft stop before
        # the call is let through without waiting for the call.
        transcript = self._write_transcript([
            self._prompt("p1", "reply with pong"),
            self._prompt("p1",
                         "<system-reminder>\nYour final report is delivered "
                         "through SubagentHandback: when your work is "
                         "complete, call SubagentHandback({message: <your "
                         "full report>}).</system-reminder>",
                         meta=True),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "draft",
                              prompt_id="p1",
                              permission_mode="auto"))
        self.assertNotIn("decision", res)
        self.assertEqual(self._fire_count(), 0)

    def test_unreadable_transcript_in_auto_mode_is_let_through_and_reported(
            self):
        # The transcript cannot say whether the tool was offered. Holding the
        # stop would block, on every retry, a subagent that was offered it
        # and already delivered, so the mode decides as before; the skip is
        # reported because a run without the tool goes unchecked here.
        missing = self.temp_dir / "never-written.jsonl"
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            res, _ = self._dispatch(
                self._native_stop(missing,
                                  "draft",
                                  prompt_id="p1",
                                  permission_mode="auto"))
        self.assertNotIn("decision", res)
        self.assertEqual(self._fire_count(), 0)
        self.assertIn("SubagentStop skipped: auto mode, but the transcript",
                      err.getvalue())

    def test_offered_tool_is_not_reported(self):
        transcript = self._write_transcript([
            self._prompt("p1", "reply with pong"),
            self._prompt("p1", "Your final report is delivered through "
                         "SubagentHandback.",
                         meta=True),
        ])
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self._dispatch(
                self._native_stop(transcript,
                                  "draft",
                                  prompt_id="p1",
                                  permission_mode="auto"))
        self.assertNotIn("SubagentStop skipped", err.getvalue())

    def test_harness_without_transcript_offer_falls_back_to_the_mode(self):
        # The offer's wording is the harness's, declared in its mapping.
        # Without it the transcript cannot show the run lacked the tool, so
        # the mode decides and the skip is reported.
        mapping = dict(runtime.subagent_handback_mapping("claude-code"))
        del mapping["transcript_offer"]
        transcript = self._write_transcript([
            self._prompt("p1", "Scope the onboarding request"),
        ])
        with mock.patch.object(runtime,
                               "subagent_handback_mapping",
                               return_value=mapping), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            res, _ = self._dispatch(
                self._native_stop(transcript,
                                  "draft",
                                  prompt_id="p1",
                                  permission_mode="auto"))
        self.assertNotIn("decision", res)
        self.assertEqual(self._fire_count(), 0)
        self.assertIn("SubagentStop skipped: auto mode, but the transcript",
                      err.getvalue())

    def test_withdrawn_tool_is_not_an_offer(self):
        # The meta record that withdraws the tool names it too; it must not
        # read as the offer.
        transcript = self._write_transcript([
            self._prompt("p1", "harness test"),
            self._prompt("p1",
                         "<system-reminder>SubagentHandback is not available "
                         "in this run ...</system-reminder>",
                         meta=True),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "handshake",
                              prompt_id="p1",
                              permission_mode="auto"))
        self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 1)

    def test_accepted_handback_exempts_stops_in_any_mode(self):
        # Switching out of auto mode after the call was accepted must not
        # hold the closing remark: that is the block-on-every-retry loop.
        transcript = self._write_transcript([
            self._prompt("p1", "harness test"),
            *self._handback_call("t1", "Report delivered to your caller."),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "Report delivered.",
                              prompt_id="p1",
                              permission_mode="default"))
        self.assertNotIn("decision", res)
        self.assertEqual(self._fire_count(), 0)


if __name__ == "__main__":
    unittest.main()
