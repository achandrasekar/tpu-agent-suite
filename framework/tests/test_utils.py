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
"""Unit tests for shared utils module."""

import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from framework import utils

Path = pathlib.Path


class TestUtils(unittest.TestCase):
    """Unit tests for utils functions and config managers."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_semver_parsing_and_ranges(self):
        v1 = utils.SemVer.parse("1.2.3")
        v2 = utils.SemVer.parse("1.2.4")
        v3 = utils.SemVer.parse("2.0.0")
        self.assertTrue(v1 < v2 < v3)
        self.assertTrue(utils.version_satisfies(v1, ">=1.0.0,<2.0.0"))
        self.assertTrue(utils.version_satisfies(v1, "==1.2.3"))
        self.assertFalse(utils.version_satisfies(v1, ">1.2.3"))
        self.assertTrue(utils.version_satisfies(v2, ">=1.2.0"))

    def test_schema_validator(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 2,
                    "pattern": r"^[A-Z][a-z]+$",
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 120
                },
                "role": {
                    "type": "string",
                    "enum": ["admin", "user"]
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
            },
            "additionalProperties": False,
        }
        # Valid instance
        self.assertEqual(
            utils.validate_instance(
                {
                    "name": "Alice",
                    "age": 30,
                    "role": "admin",
                    "tags": ["lead"]
                },
                schema,
            ),
            [],
        )

        # Missing required property
        errs = utils.validate_instance({"name": "Alice"}, schema)
        self.assertIn("$/age: required property is missing", errs)

        # Type discrimination: boolean vs integer
        errs = utils.validate_instance({"name": "Alice", "age": True}, schema)
        self.assertIn("$/age: expected integer, got bool", errs)

        # Pattern failure
        errs = utils.validate_instance({"name": "alice123", "age": 30}, schema)
        self.assertIn("$/name: string does not match '^[A-Z][a-z]+$'", errs)

        # Additional property rejection
        errs = utils.validate_instance(
            {
                "name": "Alice",
                "age": 30,
                "extra": 123
            }, schema)
        self.assertIn("$/extra: additional property is not allowed", errs)

        # Additional property rejected even with flag if additionalProperties is False
        errs = utils.validate_instance(
            {
                "name": "Alice",
                "age": 30,
                "extra": 123
            },
            schema,
            allow_additional_properties=True,
        )
        self.assertIn("$/extra: additional property is not allowed", errs)

        # Omitted additionalProperties defaults to strict/false when properties exist
        schema_omitted = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
        }
        errs = utils.validate_instance({
            "name": "Alice",
            "unknown": 1
        }, schema_omitted)
        self.assertIn("$/unknown: additional property is not allowed", errs)

        # Explicit additionalProperties: true requires allow_additional_properties flag
        schema_open = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "additionalProperties": True,
        }
        # Without flag: rejected
        errs = utils.validate_instance({
            "name": "Alice",
            "unknown": 1
        }, schema_open)
        self.assertIn("$/unknown: additional property is not allowed", errs)

        # With flag: accepted
        errs = utils.validate_instance(
            {
                "name": "Alice",
                "unknown": 1
            },
            schema_open,
            allow_additional_properties=True,
        )
        self.assertEqual(errs, [])

        # Typed additionalProperties validates values
        schema_typed = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "additionalProperties": {
                "type": "integer"
            },
        }
        errs = utils.validate_instance({
            "name": "Alice",
            "score": 100
        }, schema_typed)
        self.assertEqual(errs, [])
        errs_bad = utils.validate_instance(
            {
                "name": "Alice",
                "score": "invalid"
            }, schema_typed)
        self.assertIn("$/score: expected integer, got str", errs_bad)

    def test_format_is_accepted_but_not_enforced(self):
        schema = {"type": "string", "format": "date-time"}
        self.assertEqual(utils.validate_schema_definition(schema), [])
        self.assertEqual(utils.validate_instance("not a date", schema), [])

    def test_every_repo_schema_is_valid(self):
        """Checks every schema file, whether or not package.release.json lists it."""
        repo_root = Path(__file__).resolve().parents[2]
        not_schemas = {"event-mappings.json", "tool-mappings.json"}  # data
        paths = [
            p for d in ("package/schemas", "framework/schemas")
            for p in (repo_root / d).glob("*.json")
            if p.name not in not_schemas
        ]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(schema=path.name):
                self.assertEqual(
                    utils.validate_schema_definition(utils.load_json(path)),
                    [])

    def test_toml_manager(self):
        toml_file = self.temp_dir / "config.toml"
        toml_file.write_text('[section1]\nfoo = "bar"\n', encoding="utf-8")

        # Apply setting in existing section
        record = utils.TomlConfigManager.apply_set(toml_file, "section1",
                                                   "key1", "val1")
        self.assertIsNotNone(record)

        # Apply setting in new section
        record2 = utils.TomlConfigManager.apply_set(toml_file, "section2",
                                                    "key2", True)
        self.assertIsNotNone(record2)

        # Verify content
        text = toml_file.read_text(encoding="utf-8")
        self.assertIn('key1 = "val1"', text)
        self.assertIn("[section2]", text)
        self.assertIn("key2 = true", text)

    def test_get_event_mappings_completeness_and_integrity(self):
        """Verify event-mappings.json matches supported events and covers all harnesses."""
        events = utils.get_event_mappings()
        supported_events = utils.get_supported_hook_events()
        known_harnesses = {"claude-code", "antigravity", "cursor", "codex"}

        # 1. Every event in supported_hook_events must be mapped
        for event in supported_events:
            self.assertIn(event, events,
                          f"Event '{event}' missing from event-mappings.json")
            entry = events[event]
            self.assertEqual(entry.get("canonical_name"), event)
            harnesses = entry.get("harnesses", {})
            # Every supported harness must have an event_name defined
            for h in known_harnesses:
                self.assertIn(
                    h, harnesses,
                    f"Harness '{h}' missing in event mapping for '{event}'")
                self.assertTrue(
                    harnesses[h].get("event_name"),
                    f"Empty event_name for '{event}' on harness '{h}'")
                if event == "SubagentStop":
                    fields = harnesses[h].get("fields", {})
                    self.assertIsInstance(fields.get("subagent"), list)
                    self.assertTrue(len(fields["subagent"]) > 0)
                    self.assertIsInstance(fields.get("output"), list)
                    self.assertTrue(len(fields["output"]) > 0)
                elif event == "SessionStart":
                    fields = harnesses[h].get("fields", {})
                    self.assertIsInstance(fields.get("session_id"), list)
                    self.assertTrue(len(fields["session_id"]) > 0)
                    self.assertIsInstance(fields.get("cwd"), list)
                    self.assertTrue(len(fields["cwd"]) > 0)
                    # `source` is harness-dependent: Cursor's sessionStart
                    # payload has no field naming what began the session, so
                    # requiring one here would force a fabricated mapping.
                    # Declaring it empty is still a defect -- that reads as
                    # "supported" while resolving to nothing.
                    if "source" in fields:
                        self.assertIsInstance(fields["source"], list)
                        self.assertTrue(
                            len(fields["source"]) > 0,
                            f"Empty source mapping for '{event}' on '{h}';"
                            " omit the key instead")

        # 2. Missing file error handling
        with unittest.mock.patch("framework.utils.get_schemas_dir",
                                 return_value=self.temp_dir):
            with self.assertRaises(utils.TpuAgentSuiteError):
                utils.get_event_mappings()

    def test_json_config_manager(self):
        doc = {"a": {"b": 1}}
        records = []
        utils.JsonConfigManager.merge(doc, {
            "a": {
                "c": 2
            },
            "d": 3
        }, [], records)
        self.assertEqual(doc, {"a": {"b": 1, "c": 2}, "d": 3})
        self.assertEqual(len(records), 2)

    def test_load_json_doc_degrades_to_an_empty_mapping(self):
        """Unusable state files read as absent rather than raising.

        Callers use this to read optional records on paths where an exception
        would abort the surrounding operation, so every failure mode -- missing,
        unparseable, or parseable but not an object -- must yield a mapping.
        """
        missing = self.temp_dir / "nope.json"
        self.assertEqual(utils.load_json_doc(missing), {})

        for payload in ("{not json", "[1, 2, 3]", '"just-a-string"', "null"):
            with self.subTest(payload=payload):
                path = self.temp_dir / "doc.json"
                path.write_text(payload, encoding="utf-8")
                self.assertEqual(utils.load_json_doc(path), {})

        utils.write_json(self.temp_dir / "good.json", {"a": 1})
        self.assertEqual(utils.load_json_doc(self.temp_dir / "good.json"),
                         {"a": 1})

    def test_write_json_leaves_no_scratch_file_behind(self):
        """The temporary file must never outlive the write that created it.

        Scratch files are named for the writing process, so a leaked one is
        never reclaimed and sits beside the document it was meant to replace.
        """
        target = self.temp_dir / "atomic" / "doc.json"
        utils.write_json(target, {"generation": 1})
        utils.write_json(target, {"generation": 2})

        self.assertEqual([p.name for p in target.parent.iterdir()],
                         ["doc.json"])

    def test_write_json_preserves_the_mode_of_an_existing_file(self):
        """Rewriting must not silently widen or narrow permissions.

        Write-then-rename would otherwise hand the file whatever mode the
        umask gave the scratch file, quietly turning 0o600 into 0o644.
        """
        target = self.temp_dir / "mode" / "doc.json"
        utils.write_json(target, {"generation": 1})
        target.chmod(0o600)

        utils.write_json(target, {"generation": 2})

        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        self.assertEqual(utils.load_json(target), {"generation": 2})


class TestMarkdownBlockManager(unittest.TestCase):
    """Unit tests for marker-delimited managed blocks in markdown files."""

    MARKER = "tpu-agent-suite:pkg:kb"

    def setUp(self):
        super().setUp()
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        self.path = self.temp_dir / "CLAUDE.md"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_apply_creates_file_and_reports_creation(self):
        record = utils.MarkdownBlockManager.apply(self.path, self.MARKER,
                                                  "Body line.")
        self.assertTrue(record["file_created"])
        self.assertEqual(record["kind"], "markdown_block")
        self.assertEqual(
            utils.MarkdownBlockManager.extract(
                self.path.read_text(encoding="utf-8"), self.MARKER),
            "Body line.")

    def test_apply_replaces_body_in_place(self):
        self.path.write_text("# Notes\n\nkeep me\n", encoding="utf-8")
        utils.MarkdownBlockManager.apply(self.path, self.MARKER, "first")
        record = utils.MarkdownBlockManager.apply(self.path, self.MARKER,
                                                  "second")
        text = self.path.read_text(encoding="utf-8")

        self.assertFalse(record["file_created"])
        self.assertIn("keep me", text)
        self.assertNotIn("first", text)
        self.assertEqual(text.count(f"<!-- {self.MARKER}:begin -->"), 1)

    def test_strip_preserves_surrounding_content(self):
        self.path.write_text("# Notes\n\nkeep me\n", encoding="utf-8")
        record = utils.MarkdownBlockManager.apply(self.path, self.MARKER,
                                                  "body")
        utils.MarkdownBlockManager.strip(self.path, self.MARKER,
                                         record["sha256"],
                                         record["file_created"])
        text = self.path.read_text(encoding="utf-8")

        self.assertIn("keep me", text)
        self.assertNotIn(self.MARKER, text)
        self.assertTrue(self.path.is_file())

    def test_strip_keeps_locally_edited_block(self):
        record = utils.MarkdownBlockManager.apply(self.path, self.MARKER,
                                                  "body")
        self.path.write_text(self.path.read_text(encoding="utf-8").replace(
            "body", "edited"),
                             encoding="utf-8")
        utils.MarkdownBlockManager.strip(self.path, self.MARKER,
                                         record["sha256"],
                                         record["file_created"])

        self.assertIn("edited", self.path.read_text(encoding="utf-8"))

    def test_malformed_markers_rejected(self):
        self.path.write_text(f"<!-- {self.MARKER}:begin -->\nno end marker\n",
                             encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError):
            utils.MarkdownBlockManager.locate(
                self.path.read_text(encoding="utf-8"), self.MARKER)


class TestHookEventSources(unittest.TestCase):
    """The runtime and the schema must name the same lifecycle events."""

    def test_constant_matches_the_schema(self):
        """The fallback exists because the runtime ships without the schema.
        If the two drift, an installed package can declare an event its own
        dispatcher rejects -- which is how a Stop hook the harness dutifully
        called died in argument parsing."""
        schema = utils.get_package_manifest_schema()
        declared = schema["properties"]["hooks"]["items"]["properties"][
            "event"]["enum"]
        self.assertEqual(list(utils.SUPPORTED_HOOK_EVENTS), list(declared))

    def test_lookup_survives_a_missing_schema(self):
        """An installed package has no manifest schema beside its runtime."""
        with mock.patch.object(
                utils,
                "get_package_manifest_schema",
                side_effect=utils.TpuAgentSuiteError("missing")):
            self.assertEqual(utils.get_supported_hook_events(),
                             list(utils.SUPPORTED_HOOK_EVENTS))

    def test_every_harness_maps_session_id_on_every_event(self):
        """A hook with no session_id cannot resolve its run namespace.

        _build_event_context skips both session_id and run_id when the mapping
        yields nothing, so a harness missing this field on one event silently
        hands its hooks an empty context on that event alone.
        """
        events = utils.get_event_mappings()
        for event_name, event in events.items():
            for harness, cfg in event["harnesses"].items():
                with self.subTest(event=event_name, harness=harness):
                    sources = cfg.get("fields", {}).get("session_id")
                    self.assertIsInstance(
                        sources, list,
                        f"{harness} has no session_id source for {event_name}")
                    self.assertTrue(sources)


class TestExtractJsonObjects(unittest.TestCase):
    """Tests for extracting JSON dictionaries from raw text, code fences, and prose."""

    def test_extract_all_json_objects_from_text(self):
        """Extracts JSON dictionaries from code fences (3 and 4 backticks) and unfenced text."""
        raw = ("Context config: {\"lr\": 0.01}\n\n"
               "Step 1 output:\n"
               "```json\n"
               '{"step": 1, "nested": {"val": [1, 2]}}\n'
               "```\n\n"
               "Final envelope:\n"
               "````json\n"
               '{"step": 2, "status": "completed"}\n'
               "````\n"
               "All done!")
        objs = utils.extract_all_json_objects(raw)
        self.assertEqual(len(objs), 3)
        self.assertEqual(objs[0], {"lr": 0.01})
        self.assertEqual(objs[1], {"step": 1, "nested": {"val": [1, 2]}})
        self.assertEqual(objs[2], {"step": 2, "status": "completed"})

        # Convenience helper extract_json_object returns the last candidate
        self.assertEqual(utils.extract_json_object(raw), {
            "step": 2,
            "status": "completed"
        })

    def test_extract_all_json_objects_non_json(self):
        """Returns empty list for non-JSON, empty, or non-string inputs."""
        self.assertEqual(utils.extract_all_json_objects("Just prose text"), [])
        self.assertEqual(utils.extract_all_json_objects(""), [])
        self.assertEqual(utils.extract_all_json_objects(None), [])
        self.assertIsNone(utils.extract_json_object("Just prose text"))


class TestJsonSchemaValidatorUnionTypes(unittest.TestCase):
    """Schemas whose "type" is an array of type names, e.g. ["string", "null"]."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"]
            }
        },
    }

    def test_union_type_schema_definition_is_valid(self):
        """A union type does not crash and is accepted as a valid definition."""
        self.assertEqual(
            utils.JsonSchemaValidator.validate_schema_definition(self.SCHEMA),
            [])

    def test_union_type_instance_matches_any_member(self):
        """An instance matching any listed type validates; others are rejected."""
        validator = utils.JsonSchemaValidator
        self.assertEqual(
            validator.validate_instance({"name": "x"}, self.SCHEMA), [])
        self.assertEqual(
            validator.validate_instance({"name": None}, self.SCHEMA), [])
        self.assertEqual(validator.validate_instance({"name": 1}, self.SCHEMA),
                         ["$/name: expected string or null, got int"])

    def test_invalid_union_types_are_reported(self):
        """Empty, duplicate, unknown, or non-string type entries are errors."""
        validate = utils.JsonSchemaValidator.validate_schema_definition
        self.assertEqual(validate({"type": []}),
                         ["$/type: type array must not be empty"])
        self.assertEqual(validate({"type": ["string", "string"]}),
                         ["$/type: type array must contain unique values"])
        self.assertEqual(validate({"type": ["string", "bogus"]}),
                         ["$/type: unsupported type 'bogus'"])
        self.assertEqual(validate({"type": {}}),
                         ["$/type: unsupported type {}"])


class TestActiveRun(unittest.TestCase):
    """The workspace's active run: written once, found from any subdirectory."""

    RUN = "20260101T000000Z-111111111111"

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def test_recorded_run_is_found_from_the_workspace_and_below(self):
        self.assertIsNone(utils.get_active_run_id(self.workspace))
        utils.record_active_run(self.RUN, self.workspace)
        self.assertEqual(utils.get_active_run_id(self.workspace), self.RUN)
        nested = self.workspace / "src" / "deep"
        nested.mkdir(parents=True)
        self.assertEqual(utils.get_active_run_id(nested), self.RUN)

    def test_nested_workspace_prefers_its_own_run(self):
        """A workspace inside another finds its own run, not its parent's."""
        inner_run = "20260101T000000Z-222222222222"
        inner = self.workspace / "project_2"
        inner.mkdir()
        utils.record_active_run(self.RUN, self.workspace)
        utils.record_active_run(inner_run, inner)
        self.assertEqual(utils.get_active_run_id(inner), inner_run)
        self.assertEqual(utils.get_active_run_id(inner / "src"), inner_run)
        self.assertEqual(utils.get_active_run_id(self.workspace), self.RUN)

    def test_corrupted_file_reads_as_no_run(self):
        marker = self.workspace / ".tpu-agent-suite" / "current_run_id"
        marker.parent.mkdir(parents=True)
        marker.write_text("not-a-valid-run-id", encoding="utf-8")
        self.assertIsNone(utils.get_active_run_id(self.workspace))

    def test_recording_an_invalid_run_id_raises(self):
        with self.assertRaises(utils.TpuAgentSuiteError):
            utils.record_active_run("invalid-id", self.workspace)


if __name__ == "__main__":
    unittest.main()
