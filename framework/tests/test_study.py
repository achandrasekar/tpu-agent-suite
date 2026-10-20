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
"""Tests for the study store: the layer above a run.

Every test points TPU_AGENT_SUITE_HOME at a temp dir, so nothing here touches a real
~/.tpu-agent-suite.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

from framework import runtime, study, utils
from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


class StudyStoreTestCase(unittest.TestCase):
    """Base that isolates the central store into a temp directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = self._tmp.name
        self.home = Path(self._tmp.name)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = self._prev
        self._tmp.cleanup()

    @staticmethod
    def protocol(**overrides) -> dict:
        """A minimal protocol that satisfies the schema.

        Built here rather than pasted into forty tests so that adding a
        required field breaks one place, not forty, and so no test can
        accidentally assert the behaviour of a protocol that would be
        rejected in practice.
        """
        base = {
            "primary_metric": {
                "name": "tokens_per_s",
                "unit": "tok/s",
                "direction": "maximize",
            },
            "measurement_cell": {
                "topology": "v6e-2x2x1"
            },
            "replicate_count": 3,
            "noise_floor_pct": 0.4,
            "stopping_rule": {
                "max_rounds": 10
            },
            "admissibility_gates": [],
            "mode": "approve",
        }
        base.update(overrides)
        return base

    def start(self, study_id: str, **overrides):
        """Open a study with a valid protocol and identity."""
        return study.start_study(study_id,
                                 protocol=self.protocol(**overrides),
                                 package="tpu-agent-suite",
                                 target="test-model",
                                 owner="tester")


class TestStudyIdValidation(StudyStoreTestCase):

    def test_accepts_hyphen_case_slugs(self):
        for good in ("qwen35-397b-ttft", "a", "a1", "kernel-sweep-2"):
            with self.subTest(good=good):
                utils.validate_study_id(good)

    def test_rejects_shapes_that_would_escape_or_confuse_the_store(self):
        # A study id becomes a directory name, so traversal and separators are
        # the cases that matter, not merely cosmetic ones.
        for bad in ("", "../escape", "a/b", "Upper", "under_score", "-lead",
                    "trail-", "double--dash", "sp ace"):
            with self.subTest(bad=bad):
                with self.assertRaises(TpuAgentSuiteError):
                    utils.validate_study_id(bad)

    def test_traversal_is_refused_before_a_path_is_built(self):
        with self.assertRaises(TpuAgentSuiteError):
            utils.get_study_root("../../etc")


class TestStudyLifecycle(StudyStoreTestCase):

    RUN = "20260917T120000Z-0123456789ab"

    def test_start_writes_the_state_file_under_studies_root(self):
        doc = self.start("qwen35-397b-ttft")
        root = self.home / utils.STUDIES_DIRNAME / "qwen35-397b-ttft"

        self.assertTrue((root / study.STUDY_DOCUMENT).is_file())
        self.assertEqual(doc["status"], study.STUDY_STATUS_OPEN)
        self.assertEqual(doc["protocol_revision"], 1)
        self.assertEqual(doc["protocol"]["primary_metric"]["name"],
                         "tokens_per_s")
        self.assertEqual(doc["package"], "tpu-agent-suite")
        self.assertEqual(doc["target"], "test-model")
        self.assertEqual(doc["owner"], "tester")
        self.assertEqual(doc["runs"], [])

    def test_the_round_counter_is_derived_not_stored(self):
        # Two counters that can disagree eventually will, so there is only one.
        doc = self.start("derived")
        self.assertNotIn("round", doc)
        self.assertEqual(study.study_round(doc), 0)

    def test_a_reader_can_still_see_the_round(self):
        # Deriving a value and not surfacing it is worse than storing it: the
        # round agent checks the stopping rule through the CLI, so an
        # invisible round number makes the rule uncheckable by the one agent
        # that has to check it.
        self.start("visible")
        study.attach_run_to_study("visible", self.RUN)
        view = study.study_view(study.load_study("visible"))
        self.assertEqual(view["round"], 1)

    def test_the_view_does_not_write_back(self):
        self.start("readonly")
        study.study_view(study.load_study("readonly"))
        self.assertNotIn("round", study.load_study("readonly"))

    def test_status_and_list_both_surface_the_round(self):
        self.start("surfaced")
        study.attach_run_to_study("surfaced", self.RUN)
        for d in study.list_studies():
            if d.get("study_id") == "surfaced":
                self.assertEqual(study.study_view(d)["round"], 1)
                break
        else:
            self.fail("study missing from list")

    def test_no_notes_file_is_written_at_creation(self):
        # A file claiming to be the morning narrative that still says
        # "nothing measured yet" on the morning that matters is worse than
        # absent. Rendered on demand instead.
        self.start("quiet-start")
        root = utils.get_study_root("quiet-start")
        self.assertFalse((root / "NOTES.md").exists())

    def test_study_lives_beside_runs_not_inside_one(self):
        # The whole point of the layer: a study outlives the session, so it
        # must not be nested under a run that a later session cannot name.
        self.start("sharding-sweep")
        studies = self.home / utils.STUDIES_DIRNAME
        runs = self.home / utils.RUNS_DIRNAME
        self.assertTrue(studies.is_dir())
        self.assertNotIn(utils.RUNS_DIRNAME,
                         [p.name for p in studies.iterdir()])
        self.assertFalse(runs.exists() and any(runs.rglob("study.json")))

    def test_start_refuses_to_reopen_an_existing_study(self):
        self.start("dup")
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            self.start("dup")
        self.assertIn("already exists", str(ctx.exception))

    def test_reopen_attempt_leaves_the_original_ledger_untouched(self):
        first = self.start("keepme", replicate_count=4)
        with self.assertRaises(TpuAgentSuiteError):
            self.start("keepme", replicate_count=999)
        self.assertEqual(
            study.load_study("keepme")["protocol"], first["protocol"])

    def test_load_missing_study_fails_loudly(self):
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.load_study("never-started")
        self.assertIn("not in", str(ctx.exception))

    def test_close_marks_status_and_records_outcome(self):
        self.start("closing")
        closed = study.close_study("closing", outcome="level with baseline")
        self.assertEqual(closed["status"], study.STUDY_STATUS_CLOSED)
        self.assertEqual(closed["outcome"], "level with baseline")
        self.assertIn("closed_at", closed)

    def test_close_is_idempotent(self):
        self.start("twice")
        study.close_study("twice")
        again = study.close_study("twice")
        self.assertEqual(again["status"], study.STUDY_STATUS_CLOSED)

    def test_reclosing_does_not_move_closed_at(self):
        # closed_at records when the work ended. A cleanup script run at 06:00
        # must not rewrite it to 06:00 and erase when the study really stopped.
        self.start("restamp")
        first = study.close_study("restamp", outcome="tie")
        doc = study.load_study("restamp")
        doc["closed_at"] = "2020-01-01T00:00:00Z"
        utils.write_json(
            utils.get_study_root("restamp") / study.STUDY_DOCUMENT, doc)

        again = study.close_study("restamp")
        self.assertEqual(again["closed_at"], "2020-01-01T00:00:00Z")
        self.assertEqual(again["outcome"], "tie")
        self.assertIsNotNone(first["closed_at"])

    def test_reclosing_can_still_record_an_outcome(self):
        self.start("lateoutcome")
        study.close_study("lateoutcome")
        got = study.close_study("lateoutcome", outcome="halted on plateau")
        self.assertEqual(got["outcome"], "halted on plateau")


class TestProtocolIsRequired(StudyStoreTestCase):
    """Pre-registration is the design's central claim, so it is enforced."""

    def test_a_study_cannot_open_without_a_protocol(self):
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.start_study("bare",
                              protocol={},
                              package="tpu-agent-suite",
                              target="m")
        self.assertIn("invalid study protocol", str(ctx.exception))

    def test_each_governing_field_is_required(self):
        # Every one of these disables a different guard when absent, and all
        # of them fail silently today: no stopping rule means step 2 cannot
        # terminate, no noise floor means the win condition is invented.
        for field in ("primary_metric", "measurement_cell", "replicate_count",
                      "noise_floor_pct", "stopping_rule",
                      "admissibility_gates", "mode"):
            with self.subTest(missing=field):
                proto = self.protocol()
                del proto[field]
                errors = study.validate_study_protocol(proto)
                self.assertTrue(errors, f"{field} should be required")

    def test_a_stopping_rule_must_bound_the_study(self):
        proto = self.protocol(stopping_rule={"plateau_rounds": 3})
        self.assertTrue(study.validate_study_protocol(proto),
                        "max_rounds is what guarantees termination")

    def test_auto_mode_is_bounded_by_the_round_count(self):
        # Auto spends without asking, so it must be bounded by a number a
        # human wrote down. max_rounds is that number, and it is the only
        # bound the runtime can actually evaluate.
        proto = self.protocol(mode="auto")
        self.assertEqual(study.validate_study_protocol(proto), [])
        self.assertEqual(proto["stopping_rule"]["max_rounds"], 10)

    def test_auto_mode_without_a_round_bound_is_refused(self):
        proto = self.protocol(mode="auto", stopping_rule={"plateau_rounds": 3})
        errors = study.validate_study_protocol(proto)
        self.assertTrue(any("max_rounds" in e for e in errors))

    def test_there_is_no_separate_spend_cap(self):
        # A cap phrased in "expensive dispatches" would enforce nothing,
        # because nothing in the package defines which dispatches are
        # expensive. An unenforceable brake is worse than a coarse one.
        schema = study.load_study_protocol_schema()
        self.assertNotIn("expensive_dispatch_cap", schema["properties"])
        self.assertTrue(
            study.validate_study_protocol(
                self.protocol(expensive_dispatch_cap=4)),
            "a field that enforces nothing must not be quietly accepted")

    def test_gates_may_be_empty_but_not_absent(self):
        # Stating "no gates" is a decision; omitting the field is an
        # oversight, and only one of those is acceptable.
        self.assertEqual(
            study.validate_study_protocol(
                self.protocol(admissibility_gates=[])), [])
        proto = self.protocol()
        del proto["admissibility_gates"]
        self.assertTrue(study.validate_study_protocol(proto))

    def test_a_gate_must_say_whether_it_is_co_located(self):
        # The 108-measurements-zero-gates failure in one field.
        proto = self.protocol(admissibility_gates=[{
            "name": "ttft",
            "metric": "ttft_p50_ms",
            "comparator": "lt",
            "bound": 5000,
        }])
        self.assertTrue(study.validate_study_protocol(proto))

    def test_an_unknown_protocol_field_is_rejected(self):
        self.assertTrue(
            study.validate_study_protocol(self.protocol(budget_e2e=4)),
            "a typo must not be silently accepted as a new field")

    def test_the_shipped_example_validates(self):
        # The worked example is what people copy, so it has to be correct.
        schema = study.load_study_protocol_schema()
        for example in schema.get("examples", []):
            self.assertEqual(study.validate_study_protocol(example), [])

    def test_a_rejected_protocol_leaves_no_study_behind(self):
        with self.assertRaises(TpuAgentSuiteError):
            study.start_study(
                "aborted",
                protocol=self.protocol(stopping_rule={"plateau_rounds": 3}),
                package="tpu-agent-suite",
                target="m")
        self.assertFalse(utils.get_study_root("aborted").exists())


class TestIdentityIsNotProtocol(StudyStoreTestCase):
    """Identity lives on the document; the protocol holds only the terms."""

    def test_target_and_owner_are_not_protocol_fields(self):
        # The protocol is amendment-governed: changing it bumps the revision
        # and owes a written reason. Correcting a misspelt owner must not be
        # an amendment to what the experiment measures.
        schema = study.load_study_protocol_schema()
        self.assertNotIn("target", schema["properties"])
        self.assertNotIn("owner", schema["properties"])

    def test_they_land_on_the_document_instead(self):
        doc = self.start("identified")
        self.assertEqual(doc["target"], "test-model")
        self.assertEqual(doc["owner"], "tester")
        self.assertNotIn("target", doc["protocol"])

    def test_target_is_required(self):
        # The grouping key cannot be recovered from a free-form study id.
        for bad in ("", "   "):
            with self.subTest(target=bad):
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.start_study("untargeted",
                                      protocol=self.protocol(),
                                      package="tpu-agent-suite",
                                      target=bad)
                self.assertIn("--target", str(ctx.exception))

    def test_owner_defaults_from_the_environment(self):
        os.environ["USER"] = "someone"
        doc = study.start_study("defaulted",
                                protocol=self.protocol(),
                                package="tpu-agent-suite",
                                target="m")
        self.assertEqual(doc["owner"], "someone")

    def test_owner_falls_back_rather_than_refusing(self):
        # An unattributed study is a nuisance; refusing to open one because
        # the environment is unusual would be worse.
        for var in ("USER", "LOGNAME", "USERNAME"):
            os.environ.pop(var, None)
        self.assertEqual(study.default_owner(), "unknown")


class TestPackageIsRequired(StudyStoreTestCase):

    def test_package_is_required_and_checked_for_shape(self):
        for bad in ("", "Not A Package", "under_score"):
            with self.subTest(bad=bad):
                with self.assertRaises(TpuAgentSuiteError):
                    study.start_study("pkg",
                                      protocol=self.protocol(),
                                      package=bad,
                                      target="m")

    def test_a_valid_but_uninstalled_package_is_accepted(self):
        # Shape only. Coupling study creation to install state would break
        # opening a study in CI, or before an install finishes, and buys no
        # safety in return.
        doc = study.start_study("uninstalled",
                                protocol=self.protocol(),
                                package="not-installed-anywhere",
                                target="m")
        self.assertEqual(doc["package"], "not-installed-anywhere")

    def test_no_package_placeholder_survives(self):
        doc = self.start("labelled")
        self.assertNotEqual(doc["package"], "no-package")


class TestStudyRunAttachment(StudyStoreTestCase):

    RUN_A = "20260917T120000Z-0123456789ab"
    RUN_B = "20260917T130000Z-abcdef012345"

    def test_attach_records_runs_in_order(self):
        self.start("attaching")
        study.attach_run_to_study("attaching", self.RUN_A)
        doc = study.attach_run_to_study("attaching", self.RUN_B)
        self.assertEqual(doc["runs"], [self.RUN_A, self.RUN_B])

    def test_attach_is_idempotent_by_run_id(self):
        self.start("idem")
        study.attach_run_to_study("idem", self.RUN_A)
        doc = study.attach_run_to_study("idem", self.RUN_A)
        self.assertEqual(doc["runs"], [self.RUN_A])

    def test_attach_rejects_a_malformed_run_id(self):
        self.start("badrun")
        with self.assertRaises(TpuAgentSuiteError):
            study.attach_run_to_study("badrun", "not-a-run-id")
        self.assertEqual(study.load_study("badrun")["runs"], [])


class TestStudyListing(StudyStoreTestCase):

    def test_empty_store_lists_nothing(self):
        self.assertEqual(study.list_studies(), [])

    def test_listing_is_freshest_first(self):
        self.start("older")
        self.start("newer")
        # save_study stamps updated_at from the clock; force a known order
        # rather than depending on timer resolution.
        old = study.load_study("older")
        old["updated_at"] = "2020-01-01T00:00:00Z"
        utils.write_json(
            utils.get_study_root("older") / study.STUDY_DOCUMENT, old)

        ids = [d["study_id"] for d in study.list_studies()]
        self.assertEqual(ids, ["newer", "older"])

    def test_a_broken_ledger_is_reported_not_silently_dropped(self):
        # A study that disappears from the board because its file broke is
        # exactly the failure the board exists to catch.
        self.start("healthy")
        broken = utils.get_study_root("healthy").parent / "broken"
        broken.mkdir()
        (broken / study.STUDY_DOCUMENT).write_text("{not json",
                                                   encoding="utf-8")

        listed = {d["study_id"]: d for d in study.list_studies()}
        self.assertIn("broken", listed)
        self.assertIn("error", listed["broken"])
        self.assertIn("healthy", listed)

    def test_directory_without_a_ledger_is_reported(self):
        (self.home / utils.STUDIES_DIRNAME).mkdir(parents=True)
        (self.home / utils.STUDIES_DIRNAME / "hollow").mkdir()
        listed = {d["study_id"]: d for d in study.list_studies()}
        self.assertEqual(listed["hollow"]["error"], "no study.json")


class TestStudyLiveness(StudyStoreTestCase):

    def test_save_refreshes_updated_at_from_the_clock(self):
        self.start("liveness")
        doc = study.load_study("liveness")
        doc["updated_at"] = "2020-01-01T00:00:00Z"
        saved = study.save_study(doc)
        self.assertNotEqual(saved["updated_at"], "2020-01-01T00:00:00Z")

    def test_save_rejects_a_document_with_a_bad_study_id(self):
        with self.assertRaises(TpuAgentSuiteError):
            study.save_study({"study_id": "../escape"})


class RoundRecordTestCase(StudyStoreTestCase):
    """Shared fixtures for the round-record verbs."""

    GATES = [
        {
            "name": "ttft",
            "metric": "ttft_p50_ms",
            "comparator": "lt",
            "bound": 5000,
            "same_execution": True,
        },
        {
            "name": "accuracy",
            "metric": "gsm8k",
            "comparator": "gte",
            "bound": 0.94,
            "same_execution": True,
        },
    ]

    @staticmethod
    def baseline(**overrides) -> dict:
        base = {
            "execution_id": "j-base",
            "metric": {
                "name": "tokens_per_s",
                "value": 100.0,
                "unit": "tok/s",
                "replicates": 3,
                "provenance": "measured",
            },
            "profile": "baseline/xplane.pb",
            "fingerprint": {
                "commit_sha": "abc123",
                "config_digest": "sha256:deadbeef",
                "cell": "v6e-2x2x1/8192-1024/c256",
            },
        }
        metric_overrides = overrides.pop("metric", None)
        base.update(overrides)
        if metric_overrides is not None:
            base["metric"] = {**base["metric"], **metric_overrides}
        return base

    def start_based(self, study_id: str, **overrides):
        """Open a study that already has a baseline, so arms may report a
        delta. Most round tests need one; a delta without a baseline is a
        comparison with nothing on the other side."""
        doc = self.start(study_id, **overrides)
        # The baseline must itself meet the pre-registered replicate count,
        # so derive it rather than hardcoding one that a stricter protocol
        # would reject.
        replicates = doc["protocol"]["replicate_count"]
        study.set_baseline(study_id,
                           self.baseline(metric={"replicates": replicates}))
        return doc

    @staticmethod
    def arm(**overrides) -> dict:
        base = {
            "arm": "a1",
            "kind": "kernel-block",
            "status": "decided",
            "claim": "fuse the epilogue",
            "result": {
                "verdict":
                "tie",
                "execution_id":
                "j-1",
                "metric": {
                    "name": "tokens_per_s",
                    "value": 100.0,
                    "replicates": 3,
                    "delta_pct": 5.0,
                    "provenance": "measured",
                },
                "gates": [
                    {
                        "name": "ttft",
                        "passed": True,
                        "value": 4500.0,
                        "execution_id": "j-1",
                    },
                    {
                        "name": "accuracy",
                        "passed": True,
                        "value": 0.95,
                        "execution_id": "j-1",
                    },
                ],
            },
        }
        result_overrides = overrides.pop("result", None)
        base.update(overrides)
        if result_overrides is not None:
            base["result"] = {**base["result"], **result_overrides}
        return base


class TestRecordRound(RoundRecordTestCase):

    def test_a_clean_round_is_written_whole(self):
        self.start_based("rec", admissibility_gates=self.GATES)
        doc = study.record_round("rec", {"arms": [self.arm()]})
        self.assertEqual(len(doc["arms"]), 1)
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "tie")

    def test_arms_are_replaced_not_merged(self):
        # A patch can leave the state half-updated if the round dies mid-write.
        self.start_based("whole", admissibility_gates=self.GATES)
        study.record_round("whole", {"arms": [self.arm(arm="a1")]})
        doc = study.record_round("whole", {"arms": [self.arm(arm="a2")]})
        self.assertEqual([a["arm"] for a in doc["arms"]], ["a2"])

    def test_a_malformed_record_is_refused(self):
        self.start("bad")
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("bad", {"arms": [{"arm": "a1"}]})
        self.assertIn("invalid round record", str(ctx.exception))

    def test_an_unknown_kind_is_refused(self):
        self.start("kinds")
        with self.assertRaises(TpuAgentSuiteError):
            study.record_round("kinds",
                               {"arms": [self.arm(kind="vibes", result=None)]})

    def test_a_decided_arm_needs_a_result(self):
        self.start("resultless")
        arm = {
            "arm": "a1",
            "kind": "kernel-block",
            "status": "decided",
            "claim": "x",
        }
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("resultless", {"arms": [arm]})
        self.assertIn("decided without a result", str(ctx.exception))

    def test_an_open_arm_needs_no_result(self):
        self.start("openarm", admissibility_gates=self.GATES)
        arm = {
            "arm": "a9",
            "kind": "scheduler-batching",
            "status": "open",
            "claim": "raise max_num_batched_tokens",
        }
        doc = study.record_round("openarm", {"arms": [arm]})
        self.assertEqual(doc["arms"][0]["status"], "open")

    def test_a_failed_write_leaves_the_state_untouched(self):
        self.start_based("atomic", admissibility_gates=self.GATES)
        study.record_round("atomic", {"arms": [self.arm(arm="keep")]})
        with self.assertRaises(TpuAgentSuiteError):
            study.record_round("atomic", {"arms": [{"arm": "broken"}]})
        self.assertEqual(
            [a["arm"] for a in study.load_study("atomic")["arms"]], ["keep"])


class TestRoundHonoursTheProtocol(RoundRecordTestCase):
    """The checks that make pre-registration mean something."""

    def test_an_omitted_gate_is_not_a_passing_gate(self):
        self.start_based("omit", admissibility_gates=self.GATES)
        arm = self.arm(result={
            "gates": [{
                "name": "ttft",
                "passed": True,
                "execution_id": "j-1"
            }]
        })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("omit", {"arms": [arm]})
        self.assertIn("accuracy", str(ctx.exception))
        self.assertIn("unmeasured gate", str(ctx.exception))

    def test_a_gate_measured_elsewhere_is_refused(self):
        # The 108-measurements failure, caught by the schema this time.
        self.start_based("split", admissibility_gates=self.GATES)
        arm = self.arm(
            result={
                "gates": [
                    {
                        "name": "ttft",
                        "passed": True,
                        "execution_id": "j-1"
                    },
                    {
                        "name": "accuracy",
                        "passed": True,
                        "execution_id": "j-2"
                    },
                ]
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("split", {"arms": [arm]})
        message = str(ctx.exception)
        self.assertIn("same_execution", message)
        self.assertIn("do not describe one configuration", message)

    def test_a_win_that_fails_a_gate_is_refused(self):
        # A faster number that violates a gate is inadmissible, not a
        # smaller win. This is the champion that was crowned and withdrawn.
        self.start_based("champ", admissibility_gates=self.GATES)
        arm = self.arm(
            result={
                "verdict":
                "win",
                "gates": [
                    {
                        "name": "ttft",
                        "passed": False,
                        "value": 6000.0,
                        "execution_id": "j-1",
                    },
                    {
                        "name": "accuracy",
                        "passed": True,
                        "value": 0.95,
                        "execution_id": "j-1",
                    },
                ],
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("champ", {"arms": [arm]})
        self.assertIn("inadmissible", str(ctx.exception))

    def test_a_win_inside_the_noise_floor_is_refused(self):
        self.start_based("noisy",
                         admissibility_gates=self.GATES,
                         noise_floor_pct=1.0)
        arm = self.arm(
            result={
                "verdict": "win",
                "metric": {
                    "name": "tokens_per_s",
                    "value": 100.0,
                    "replicates": 3,
                    "delta_pct": 0.13,
                    "provenance": "measured",
                },
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("noisy", {"arms": [arm]})
        self.assertIn("noise floor", str(ctx.exception))
        self.assertIn("That is a tie", str(ctx.exception))

    def test_a_win_under_replicated_is_refused(self):
        self.start_based("thin",
                         admissibility_gates=self.GATES,
                         replicate_count=5)
        arm = self.arm(
            result={
                "verdict": "win",
                "metric": {
                    "name": "tokens_per_s",
                    "value": 100.0,
                    "replicates": 1,
                    "delta_pct": 9.0,
                    "provenance": "measured",
                },
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("thin", {"arms": [arm]})
        self.assertIn("replicate", str(ctx.exception))

    def test_a_legitimate_win_is_accepted(self):
        self.start_based("real",
                         admissibility_gates=self.GATES,
                         noise_floor_pct=0.4)
        arm = self.arm(
            result={
                "verdict": "win",
                "metric": {
                    "name": "tokens_per_s",
                    "value": 112.0,
                    "replicates": 3,
                    "delta_pct": 12.0,
                    "provenance": "measured",
                },
            })
        doc = study.record_round("real", {"arms": [arm]})
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "win")

    def test_a_tie_is_held_to_the_gates_but_not_the_floor(self):
        # A tie still has to have measured its gates; it just need not clear
        # the floor, which is what makes it a tie.
        self.start_based("tie",
                         admissibility_gates=self.GATES,
                         noise_floor_pct=5.0)
        doc = study.record_round("tie", {"arms": [self.arm()]})
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "tie")

    def test_a_study_with_no_gates_requires_none(self):
        self.start_based("ungated", admissibility_gates=[])
        arm = self.arm(result={"gates": []})
        doc = study.record_round("ungated", {"arms": [arm]})
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "tie")

    def test_the_shipped_example_round_validates(self):
        schema = study.load_study_round_schema()
        for example in schema.get("examples", []):
            self.assertEqual(utils.validate_instance(example, schema), [],
                             "the worked example is what people copy")


class TestAWinOwesAMeasurement(RoundRecordTestCase):
    """Every bar is skipped when its input is missing, so absence must fail."""

    def test_a_win_with_no_metric_is_refused(self):
        # Without this the noise floor and replicate checks are both skipped,
        # so omitting the measurement clears every bar by saying nothing.
        self.start_based("silentwin", admissibility_gates=[])
        arm = self.arm(result={"verdict": "win", "gates": [], "metric": None})
        del arm["result"]["metric"]
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("silentwin", {"arms": [arm]})
        self.assertIn("without a metric", str(ctx.exception))

    def test_a_win_must_report_replicates_and_delta(self):
        self.start_based("partial", admissibility_gates=[])
        for missing in ("replicates", "delta_pct"):
            with self.subTest(missing=missing):
                metric = {
                    "name": "tokens_per_s",
                    "value": 100.0,
                    "replicates": 3,
                    "delta_pct": 9.0,
                    "provenance": "measured",
                }
                del metric[missing]
                arm = self.arm(result={
                    "verdict": "win",
                    "gates": [],
                    "metric": metric
                })
                errors = study.check_round_against_protocol(
                    {"arms": [arm]},
                    study.load_study("partial")["protocol"])
                self.assertTrue(any(missing in e for e in errors))

    def test_a_win_must_report_value_for_admissibility_gate(self):
        # A win must report the measurement for every gate declaring a bound.
        # Merely asserting passed=True without a value is inadmissible.
        self.start_based("gateval", admissibility_gates=self.GATES)
        arm = self.arm(
            result={
                "verdict":
                "win",
                "metric": {
                    "name": "tokens_per_s",
                    "value": 112.0,
                    "replicates": 3,
                    "delta_pct": 12.0,
                    "provenance": "measured",
                },
                "gates": [
                    {
                        "name": "ttft",
                        "passed": True,
                        "execution_id": "j-1"
                        # Omitted "value"
                    },
                    {
                        "name": "accuracy",
                        "passed": True,
                        "value": 0.95,
                        "execution_id": "j-1"
                    },
                ]
            })
        errors = study.check_round_against_protocol(
            {"arms": [arm]},
            study.load_study("gateval")["protocol"])
        self.assertTrue(any("ttft" in e and "value" in e for e in errors))

    def test_a_tie_also_owes_a_measurement(self):
        self.start_based("silenttie", admissibility_gates=[])
        arm = self.arm(result={"verdict": "tie", "gates": []})
        del arm["result"]["metric"]
        with self.assertRaises(TpuAgentSuiteError):
            study.record_round("silenttie", {"arms": [arm]})

    def test_invalid_needs_no_measurement(self):
        # `invalid` says the number cannot be trusted, which is a different
        # claim from a number that did not help.
        self.start("untrusted", admissibility_gates=[])
        arm = self.arm(result={"verdict": "invalid", "gates": []})
        del arm["result"]["metric"]
        doc = study.record_round("untrusted", {"arms": [arm]})
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "invalid")


class TestBaseline(RoundRecordTestCase):
    """The number every arm is compared against."""

    def test_a_baseline_is_recorded(self):
        self.start("based")
        doc = study.set_baseline("based", self.baseline())
        self.assertEqual(doc["baseline"]["metric"]["value"], 100.0)

    def test_a_profile_is_required(self):
        # Without per-op attribution the first round cannot rank arms by
        # headroom and falls back to guessing which layer to attack.
        self.start("noprofile")
        bad = self.baseline()
        del bad["profile"]
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("noprofile", bad)
        self.assertIn("invalid baseline", str(ctx.exception))

    def test_a_fingerprint_is_required(self):
        # Nothing can tell whether a baseline is reusable without it, and an
        # e2e run costs hours.
        self.start("nofinger")
        bad = self.baseline()
        del bad["fingerprint"]
        with self.assertRaises(TpuAgentSuiteError):
            study.set_baseline("nofinger", bad)

    def test_a_modelled_baseline_is_refused(self):
        # Every arm is compared against it, so a modelled one turns the study
        # into an evaluation of its own prediction.
        self.start("modelled")
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("modelled",
                               self.baseline(metric={"provenance": "modeled"}))
        self.assertIn("must be measured", str(ctx.exception))

    def test_the_wrong_metric_is_refused(self):
        self.start("wrongmetric")
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("wrongmetric",
                               self.baseline(metric={"name": "ttft_p50_ms"}))
        self.assertIn("not a delta", str(ctx.exception))

    def test_an_under_replicated_baseline_is_refused(self):
        self.start("thinbase", replicate_count=5)
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("thinbase",
                               self.baseline(metric={"replicates": 1}))
        self.assertIn("every comparison rests on", str(ctx.exception))

    def test_an_adopted_baseline_records_where_it_came_from(self):
        # A silently reused number is how a study ends up comparing against
        # something nobody can locate.
        self.start("adopted")
        doc = study.set_baseline(
            "adopted",
            self.baseline(adopted_from_run="20260101T000000Z-0123456789ab"))
        self.assertIn("adopted_from_run", doc["baseline"])

    def test_a_delta_without_a_baseline_is_refused(self):
        # delta_pct claimed to compare against "the study's baseline" while no
        # baseline existed anywhere -- a comparison with nothing on one side.
        self.start("nobase", admissibility_gates=[])
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("nobase",
                               {"arms": [self.arm(result={"gates": []})]})
        self.assertIn("no baseline", str(ctx.exception))

    def test_a_delta_is_accepted_once_a_baseline_exists(self):
        self.start("withbase", admissibility_gates=[])
        study.set_baseline("withbase", self.baseline())
        doc = study.record_round("withbase",
                                 {"arms": [self.arm(result={"gates": []})]})
        self.assertEqual(doc["arms"][0]["result"]["metric"]["delta_pct"], 5.0)

    def test_an_arm_with_no_delta_needs_no_baseline(self):
        self.start("deltafree", admissibility_gates=[])
        arm = self.arm(result={"gates": [], "verdict": "invalid"})
        del arm["result"]["metric"]
        doc = study.record_round("deltafree", {"arms": [arm]})
        self.assertEqual(doc["arms"][0]["result"]["verdict"], "invalid")

    def test_the_shipped_example_validates(self):
        schema = study.load_study_baseline_schema()
        for example in schema.get("examples", []):
            self.assertEqual(utils.validate_instance(example, schema), [])


class TestBaselineReplacement(RoundRecordTestCase):

    def test_replacing_a_baseline_with_measured_arms_is_refused(self):
        # Every delta already recorded compares to the old number. Swapping it
        # silently leaves them comparing to something that is no longer there.
        self.start_based("swap", admissibility_gates=[])
        study.record_round("swap", {"arms": [self.arm(result={"gates": []})]})
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("swap", self.baseline(metric={"value": 999.0}))
        self.assertIn("no longer there", str(ctx.exception))

    def test_replacing_before_any_arm_is_measured_is_allowed(self):
        # Nothing has been compared yet, so nothing is invalidated.
        self.start_based("early")
        doc = study.set_baseline("early",
                                 self.baseline(metric={"value": 42.0}))
        self.assertEqual(doc["baseline"]["metric"]["value"], 42.0)

    def test_setting_the_identical_baseline_again_is_a_no_op(self):
        self.start_based("same", admissibility_gates=[])
        study.record_round("same", {"arms": [self.arm(result={"gates": []})]})
        doc = study.set_baseline("same", self.baseline())
        self.assertEqual(doc["baseline"]["metric"]["value"], 100.0)


class TestArmMetricMatchesProtocol(RoundRecordTestCase):

    def test_an_arm_reporting_another_metric_is_refused(self):
        # A delta between two different quantities is not a delta.
        self.start_based("mismatch", admissibility_gates=[])
        arm = self.arm(
            result={
                "gates": [],
                "metric": {
                    "name": "ttft_p50_ms",
                    "value": 1.0,
                    "replicates": 3,
                    "delta_pct": 2.0,
                    "provenance": "measured",
                },
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("mismatch", {"arms": [arm]})
        self.assertIn("primary metric", str(ctx.exception))


class TestStudyNotes(RoundRecordTestCase):
    """Rendered from the state, so it cannot go stale."""

    def test_notes_report_the_current_round_and_terms(self):
        self.start("noted")
        study.attach_run_to_study("noted", "20260917T120000Z-0123456789ab")
        text = study.render_study_notes("noted")
        self.assertIn("# study noted", text)
        self.assertIn("round: 1 of 10", text)
        self.assertIn("tokens_per_s", text)

    def test_notes_say_plainly_when_there_is_no_baseline(self):
        self.start("bare-notes")
        self.assertIn("No arm can report a delta",
                      study.render_study_notes("bare-notes"))

    def test_notes_reflect_recorded_arms(self):
        self.start("armed", admissibility_gates=[])
        study.set_baseline("armed", self.baseline())
        study.record_round("armed", {"arms": [self.arm(result={"gates": []})]})
        text = study.render_study_notes("armed")
        self.assertIn("**a1**", text)
        self.assertIn("tie", text)
        self.assertIn("[measured]", text)

    def test_notes_are_not_written_to_disk(self):
        self.start("ephemeral")
        study.render_study_notes("ephemeral")
        root = utils.get_study_root("ephemeral")
        self.assertEqual(sorted(p.name for p in root.iterdir()),
                         ["study.json"])


class TestSetProtocol(RoundRecordTestCase):

    def test_amending_keeps_the_superseded_version_and_the_reason(self):
        self.start("amend")
        doc = study.set_protocol(
            "amend",
            self.protocol(replicate_count=7),
            reason="baseline spread was wider than 3 runs could resolve")
        self.assertEqual(doc["protocol_revision"], 2)
        self.assertEqual(doc["protocol"]["replicate_count"], 7)
        self.assertEqual(len(doc["amendments"]), 1)
        prior = doc["amendments"][0]
        self.assertEqual(prior["revision"], 1)
        self.assertEqual(prior["protocol"]["replicate_count"], 3)
        self.assertIn("baseline spread", prior["reason"])

    def test_a_reason_is_required(self):
        # A diff shows what changed and never why, and the amendment worth
        # reviewing later is the one that relaxed a failing bar.
        self.start("silent")
        for blank in ("", "   "):
            with self.subTest(reason=blank):
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.set_protocol("silent", self.protocol(), blank)
                self.assertIn("--reason", str(ctx.exception))

    def test_an_invalid_amendment_is_refused(self):
        self.start("badamend")
        with self.assertRaises(TpuAgentSuiteError):
            study.set_protocol("badamend", {"mode": "auto"}, reason="x")
        self.assertEqual(study.load_study("badamend")["protocol_revision"], 1)

    def test_amendments_accumulate(self):
        self.start("twice-amended")
        study.set_protocol("twice-amended",
                           self.protocol(replicate_count=5),
                           reason="first")
        doc = study.set_protocol("twice-amended",
                                 self.protocol(replicate_count=9),
                                 reason="second")
        self.assertEqual(doc["protocol_revision"], 3)
        self.assertEqual([a["revision"] for a in doc["amendments"]], [1, 2])


class TestSchemasResolveInBothLayouts(StudyStoreTestCase):
    """The repository layout is not the layout users run."""

    def test_schemas_load_from_the_repository_layout(self):
        self.assertTrue(study.load_study_protocol_schema()["required"])
        self.assertTrue(study.load_study_round_schema()["required"])

    def test_schemas_load_from_framework_schemas(self):
        # framework/ ships as is, so its schemas/ is beside utils.py in the
        # repository and in a build alike.
        here = Path(utils.__file__).resolve().parent
        for name in (study.STUDY_PROTOCOL_SCHEMA, study.STUDY_ROUND_SCHEMA):
            self.assertEqual(utils.runtime_schema_path(name),
                             here / "schemas" / name)

    def test_a_missing_schema_names_where_it_looked(self):
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            utils.runtime_schema_path("no-such-schema.json")
        self.assertIn("looked in", str(ctx.exception))


class TestStudyStreams(StudyStoreTestCase):
    """The two cold streams beside the ledger."""

    def test_events_and_journal_are_separate_files(self):
        # Machine-authored and agent-authored records have different writers,
        # schemas and readers; sharing a file makes both unusable.
        self.start("streams")
        study.append_study_event("streams", "hook", event="SessionStart")
        study.append_study_journal("streams", {"dropped_arm": "a3"})

        root = utils.get_study_root("streams")
        self.assertTrue((root / study.STUDY_EVENTS).is_file())
        self.assertTrue((root / study.STUDY_JOURNAL).is_file())
        self.assertEqual(len(study.read_study_events("streams")), 1)
        self.assertEqual(len(study.read_study_journal("streams")), 1)

    def test_append_is_one_line_per_record(self):
        self.start("lines")
        for i in range(5):
            study.append_study_event("lines", "hook", n=i)
        raw = (utils.get_study_root("lines") /
               study.STUDY_EVENTS).read_text(encoding="utf-8")
        self.assertEqual(len(raw.strip().splitlines()), 5)

    def test_events_are_returned_oldest_first(self):
        self.start("ordered")
        for i in range(4):
            study.append_study_event("ordered", "hook", n=i)
        self.assertEqual([e["n"] for e in study.read_study_events("ordered")],
                         [0, 1, 2, 3])

    def test_limit_keeps_the_newest(self):
        self.start("limited")
        for i in range(6):
            study.append_study_event("limited", "hook", n=i)
        got = study.read_study_events("limited", limit=2)
        self.assertEqual([e["n"] for e in got], [4, 5])

    def test_every_record_is_stamped(self):
        self.start("stamped")
        rec = study.append_study_event("stamped", "hook")
        self.assertIn("at", rec)
        self.assertEqual(rec["kind"], "hook")

    def test_a_torn_line_does_not_blind_the_reader(self):
        # A monitor that returns nothing because one line is corrupt is worse
        # than one that returns the other nine.
        self.start("torn")
        study.append_study_event("torn", "hook", n=1)
        path = utils.get_study_root("torn") / study.STUDY_EVENTS
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{ this is not json\n")
        study.append_study_event("torn", "hook", n=2)

        got = study.read_study_events("torn")
        self.assertEqual([e["n"] for e in got], [1, 2])

    def test_reading_an_absent_stream_is_empty_not_an_error(self):
        self.start("quiet")
        self.assertEqual(study.read_study_events("quiet"), [])
        self.assertEqual(study.read_study_journal("quiet"), [])

    def test_streams_reject_a_bad_study_id(self):
        with self.assertRaises(TpuAgentSuiteError):
            study.append_study_event("../escape", "hook")
        with self.assertRaises(TpuAgentSuiteError):
            study.read_study_events("../escape")
        with self.assertRaises(TpuAgentSuiteError):
            study.append_study_journal("../escape", {})
        with self.assertRaises(TpuAgentSuiteError):
            study.read_study_journal("../escape")


class TestOpenStudyIds(StudyStoreTestCase):

    def test_lists_only_open_studies(self):
        self.start("open-one")
        self.start("shut-one")
        study.close_study("shut-one")
        self.assertEqual(study.open_study_ids(), ["open-one"])

    def test_empty_when_nothing_is_open(self):
        self.assertEqual(study.open_study_ids(), [])

    def test_a_broken_ledger_does_not_appear_as_open(self):
        root = self.home / utils.STUDIES_DIRNAME / "broken"
        root.mkdir(parents=True)
        (root / study.STUDY_DOCUMENT).write_text("{bad", encoding="utf-8")
        self.assertEqual(study.open_study_ids(), [])


class TestRunToStudyCardinality(StudyStoreTestCase):
    """One run, one study. The rule the rest of the model rests on."""

    RUN = "20260917T120000Z-0123456789ab"
    OTHER = "20260917T130000Z-abcdef012345"

    def test_a_run_attaches_to_one_study(self):
        self.start("mine")
        doc = study.attach_run_to_study("mine", self.RUN)
        self.assertEqual(doc["runs"], [self.RUN])
        self.assertEqual(study.study_for_run(self.RUN), "mine")

    def test_attaching_the_same_run_twice_is_a_no_op(self):
        # A retried command is not an error.
        self.start("idem")
        study.attach_run_to_study("idem", self.RUN)
        doc = study.attach_run_to_study("idem", self.RUN)
        self.assertEqual(doc["runs"], [self.RUN])

    def test_attaching_a_run_to_a_second_study_is_refused(self):
        self.start("first")
        self.start("second")
        study.attach_run_to_study("first", self.RUN)
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.attach_run_to_study("second", self.RUN)
        message = str(ctx.exception)
        self.assertIn("already attached", message)
        self.assertIn("first", message)
        self.assertIn("a retry is a new round", message)

    def test_a_refused_attach_leaves_both_studies_unchanged(self):
        self.start("keeper")
        self.start("thief")
        study.attach_run_to_study("keeper", self.RUN)
        with self.assertRaises(TpuAgentSuiteError):
            study.attach_run_to_study("thief", self.RUN)
        self.assertEqual(study.load_study("keeper")["runs"], [self.RUN])
        self.assertEqual(study.load_study("thief")["runs"], [])

    def test_a_closed_study_still_owns_its_runs(self):
        # Re-attaching a closed study's run elsewhere would rewrite history
        # rather than extend it.
        self.start("finished")
        study.attach_run_to_study("finished", self.RUN)
        study.close_study("finished")
        self.start("newcomer")
        with self.assertRaises(TpuAgentSuiteError):
            study.attach_run_to_study("newcomer", self.RUN)

    def test_attaching_emits_the_marker_event(self):
        # SessionStart cannot be routed, so the attach carries that signal.
        self.start("marked")
        study.attach_run_to_study("marked", self.RUN)
        events = study.read_study_events("marked")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "attach")
        self.assertEqual(events[0]["run_id"], self.RUN)
        self.assertEqual(events[0]["round"], 1)

    def test_an_unattached_run_belongs_to_nothing(self):
        self.assertIsNone(study.study_for_run(self.OTHER))


class TestEventRouting(StudyStoreTestCase):
    """Strict lookup: no inference, no fan-out."""

    RUN = "20260917T120000Z-0123456789ab"
    OTHER = "20260917T130000Z-abcdef012345"

    def test_an_attached_run_routes_to_its_own_study(self):
        self.start("mine")
        study.attach_run_to_study("mine", self.RUN)
        self.assertEqual(study.studies_for_event(self.RUN), ["mine"])

    def test_a_lone_open_study_does_not_absorb_unrelated_sessions(self):
        # The common case, and the bug: one open study used to receive every
        # session on the machine, including onboarding jobs and scratch work.
        self.start("lonely")
        self.assertEqual(study.studies_for_event(self.OTHER), [])

    def test_an_event_with_no_run_is_dropped(self):
        self.start("lonely")
        self.assertEqual(study.studies_for_event(""), [])

    def test_concurrent_studies_do_not_cross_contaminate(self):
        self.start("study-a")
        self.start("study-b")
        study.attach_run_to_study("study-a", self.RUN)
        study.attach_run_to_study("study-b", self.OTHER)
        self.assertEqual(study.studies_for_event(self.RUN), ["study-a"])
        self.assertEqual(study.studies_for_event(self.OTHER), ["study-b"])

    def test_a_closed_study_receives_nothing_further(self):
        self.start("done")
        study.attach_run_to_study("done", self.RUN)
        study.close_study("done")
        self.assertEqual(study.studies_for_event(self.RUN), [])

    def test_a_stop_event_reaches_only_its_own_study(self):
        # fully_idle in the wrong stream is a false liveness signal, which
        # inverts the one thing the stream exists to report.
        self.start("watched")
        self.start("bystander")
        study.attach_run_to_study("watched", self.RUN)
        runtime._builtin_record_event(
            "p", Path("/ws"), {
                "event": "Stop",
                "context": {
                    "workspace": "/ws",
                    "package": "p",
                    "run_id": self.RUN
                },
                "termination": {
                    "reason": "NO_TOOL_CALL",
                    "fully_idle": True
                },
            })
        watched = [
            e for e in study.read_study_events("watched")
            if e["kind"] == "hook"
        ]
        self.assertEqual(len(watched), 1)
        self.assertTrue(watched[0]["fully_idle"])
        self.assertEqual([
            e for e in study.read_study_events("bystander")
            if e["kind"] == "hook"
        ], [])


class TestRecordEventBuiltin(StudyStoreTestCase):
    """The hook handler that writes the event stream."""

    PAYLOAD = {
        "event": "PreToolUse",
        "tool": "delegate",
        "subagent": "worker",
        "context": {
            "workspace": "/ws",
            "package": "tpu-agent-suite",
            "session_id": "sess-1",
            "run_id": "20260917T120000Z-0123456789ab",
        },
    }

    def test_event_lands_in_the_attached_study(self):
        self.start("recording")
        study.attach_run_to_study("recording", "20260917T120000Z-0123456789ab")
        out = runtime._builtin_record_event("tpu-agent-suite", Path("/ws"),
                                            self.PAYLOAD)
        self.assertEqual(out["decision"], "allow")

        events = [
            e for e in study.read_study_events("recording")
            if e["kind"] == "hook"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "PreToolUse")
        self.assertEqual(events[0]["subagent"], "worker")
        self.assertEqual(events[0]["run_id"], "20260917T120000Z-0123456789ab")

    def test_closed_studies_do_not_receive_events(self):
        self.start("done")
        study.attach_run_to_study("done", "20260917T120000Z-0123456789ab")
        study.close_study("done")
        runtime._builtin_record_event("tpu-agent-suite", Path("/ws"),
                                      self.PAYLOAD)
        self.assertEqual([
            e for e in study.read_study_events("done") if e["kind"] == "hook"
        ], [])

    def test_termination_fields_are_carried(self):
        # fully_idle distinguishes "the model stopped" from "no work is
        # outstanding", which is the difference between healthy and stalled.
        self.start("stopping")
        study.attach_run_to_study("stopping", "20260917T120000Z-0123456789ab")
        runtime._builtin_record_event(
            "tpu-agent-suite", Path("/ws"), {
                "event": "Stop",
                "context": {
                    "workspace": "/ws",
                    "package": "p",
                    "run_id": "20260917T120000Z-0123456789ab"
                },
                "termination": {
                    "reason": "NO_TOOL_CALL",
                    "fully_idle": True
                },
            })
        got = [
            e for e in study.read_study_events("stopping")
            if e["kind"] == "hook"
        ][0]
        self.assertEqual(got["stop_reason"], "NO_TOOL_CALL")
        self.assertTrue(got["fully_idle"])

    def test_always_allows_even_with_no_study_open(self):
        # A monitor must never be able to block the work it monitors.
        out = runtime._builtin_record_event("tpu-agent-suite", Path("/ws"),
                                            self.PAYLOAD)
        self.assertEqual(out["decision"], "allow")

    def test_always_allows_when_the_store_is_unwritable(self):
        self.start("unwritable")
        root = utils.get_study_root("unwritable")
        root.chmod(0o500)
        try:
            out = runtime._builtin_record_event("tpu-agent-suite", Path("/ws"),
                                                self.PAYLOAD)
            self.assertEqual(out["decision"], "allow")
        finally:
            root.chmod(0o700)

    def test_it_is_registered_as_a_builtin(self):
        self.assertIn("record-event", runtime.BUILTIN_HOOK_ACTIONS)


class TestStudyCli(StudyStoreTestCase):

    def test_start_status_attach_list_close_round_trip(self):
        protocol = self.home / "protocol.json"
        protocol.write_text(json.dumps(self.protocol()), encoding="utf-8")

        self.assertEqual(
            runtime.main([
                "study", "start", "--study-id", "cli-study", "--protocol",
                str(protocol), "--package", "tpu-agent-suite", "--target", "m"
            ]), 0)
        self.assertEqual(
            runtime.main(["study", "status", "--study-id", "cli-study"]), 0)
        self.assertEqual(
            runtime.main([
                "study", "attach-run", "--study-id", "cli-study", "--run-id",
                "20260917T120000Z-0123456789ab"
            ]), 0)
        self.assertEqual(runtime.main(["study", "list"]), 0)
        self.assertEqual(
            runtime.main(["study", "events", "--study-id", "cli-study"]), 0)
        self.assertEqual(
            runtime.main([
                "study", "journal", "--study-id", "cli-study", "--limit", "5"
            ]), 0)
        self.assertEqual(
            runtime.main([
                "study", "close", "--study-id", "cli-study", "--outcome", "tie"
            ]), 0)

        doc = study.load_study("cli-study")
        self.assertEqual(doc["status"], study.STUDY_STATUS_CLOSED)
        self.assertEqual(doc["outcome"], "tie")
        self.assertEqual(doc["protocol"]["primary_metric"]["name"],
                         "tokens_per_s")
        self.assertEqual(doc["runs"], ["20260917T120000Z-0123456789ab"])

    def test_cli_returns_nonzero_on_a_bad_study_id(self):
        protocol = self.home / "p.json"
        protocol.write_text(json.dumps(self.protocol()), encoding="utf-8")
        self.assertEqual(
            runtime.main([
                "study", "start", "--study-id", "Bad_Id", "--protocol",
                str(protocol), "--package", "tpu-agent-suite", "--target", "m"
            ]), 1)

    def test_cli_refuses_to_start_without_a_protocol(self):
        # argparse exits 2 on a missing required flag; the point is that it
        # does not quietly open an unusable study.
        with self.assertRaises(SystemExit) as ctx:
            runtime.main([
                "study", "start", "--study-id", "no-protocol", "--package",
                "tpu-agent-suite", "--target", "m"
            ])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_cli_refuses_an_invalid_protocol_file(self):
        bad = self.home / "bad.json"
        bad.write_text(json.dumps({"target": "m"}), encoding="utf-8")
        self.assertEqual(
            runtime.main([
                "study", "start", "--study-id", "bad-protocol", "--protocol",
                str(bad), "--package", "tpu-agent-suite", "--target", "m"
            ]), 1)


class TestComputeArmVerdict(unittest.TestCase):
    """Unit tests for compute_arm_verdict invariants."""

    def test_win_exceeds_noise_floor_maximize(self):
        verdict, delta = study.compute_arm_verdict(
            metric_value=105.0,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "win")
        self.assertEqual(delta, 5.0)

    def test_win_exceeds_noise_floor_minimize(self):
        verdict, delta = study.compute_arm_verdict(
            metric_value=90.0,
            baseline_value=100.0,
            direction="minimize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "win")
        self.assertEqual(delta, 10.0)

    def test_tie_within_noise_floor(self):
        verdict, delta = study.compute_arm_verdict(
            metric_value=100.5,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "tie")
        self.assertEqual(delta, 0.5)

    def test_loss_below_negative_noise_floor(self):
        verdict, delta = study.compute_arm_verdict(
            metric_value=98.0,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "loss")
        self.assertEqual(delta, -2.0)

    def test_admissibility_gate_failure_is_invalid(self):
        # A faster number that breaks an admissibility gate is inadmissible ('invalid'), and cannot count.
        verdict, delta = study.compute_arm_verdict(
            metric_value=120.0,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=False,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "invalid")
        self.assertEqual(delta, 20.0)

    def test_noise_floor_exact_boundary_is_tie(self):
        # Exactly at the noise floor is a tie, not a win.
        verdict, delta = study.compute_arm_verdict(
            metric_value=101.0,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=3,
            required_replicates=3,
        )
        self.assertEqual(verdict, "tie")
        self.assertEqual(delta, 1.0)
        # Verify check_round_against_protocol also rejects a win at exactly the floor
        win_at_floor = {
            "arms": [{
                "arm": "a1",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "test arm",
                "result": {
                    "verdict": "win",
                    "execution_id": "job-1",
                    "metric": {
                        "name": "tok_per_s",
                        "value": 101.0,
                        "unit": "tok/s",
                        "replicates": 3,
                        "delta_pct": 1.0,
                        "provenance": "measured",
                    },
                    "gates": [],
                },
            }]
        }
        protocol = {
            "primary_metric": {
                "name": "tok_per_s",
                "direction": "maximize",
                "unit": "tok/s",
            },
            "noise_floor_pct": 1.0,
            "replicate_count": 3,
            "admissibility_gates": [],
        }
        errors = study.check_round_against_protocol(win_at_floor, protocol)
        self.assertTrue(any("That is a tie" in e for e in errors))

    def test_round_trip_production_matches_validation(self):
        protocol = {
            "primary_metric": {
                "name": "tok_per_s",
                "direction": "maximize",
                "unit": "tok/s",
            },
            "measurement_cell": {
                "topology": "v6e-2x2x1",
            },
            "noise_floor_pct":
            1.0,
            "replicate_count":
            3,
            "stopping_rule": {
                "max_rounds": 10,
            },
            "admissibility_gates": [{
                "name": "eval_loss",
                "metric": "eval_loss",
                "comparator": "lte",
                "bound": 2.5,
                "same_execution": True,
            }],
            "mode":
            "auto",
        }
        self.assertEqual(study.validate_study_protocol(protocol), [])
        # Matrix of test cases: (metric, gate_passed, reps, expected_verdict)
        cases = [
            (105.0, True, 3, "win"),
            (100.5, True, 3, "tie"),
            (105.0, True, 1, "tie"),
            (95.0, True, 3, "loss"),
            (105.0, False, 3, "invalid"),
        ]
        for val, gate_pass, reps, exp_verdict in cases:
            verdict, delta = study.compute_arm_verdict(
                metric_value=val,
                baseline_value=100.0,
                direction="maximize",
                noise_floor_pct=1.0,
                gates_passed=gate_pass,
                replicates=reps,
                required_replicates=3,
            )
            self.assertEqual(verdict, exp_verdict)
            round_record = {
                "arms": [{
                    "arm": "a1",
                    "kind": "kernel-block",
                    "status": "decided",
                    "claim": "test arm",
                    "result": {
                        "status":
                        "completed",
                        "verdict":
                        verdict,
                        "execution_id":
                        "job-1",
                        "metric": {
                            "name": "tok_per_s",
                            "value": val,
                            "unit": "tok/s",
                            "replicates": reps,
                            "delta_pct": delta,
                            "provenance": "measured",
                        },
                        "gates": [{
                            "name": "eval_loss",
                            "passed": gate_pass,
                            "value": 2.0 if gate_pass else 3.0,
                            "execution_id": "job-1",
                        }],
                    },
                }]
            }
            errors = study.check_round_against_protocol(round_record, protocol)
            self.assertEqual(
                errors, [],
                f"check_round_against_protocol failed on {exp_verdict}: {errors}"
            )

    def test_insufficient_replicates_remains_tie(self):
        verdict, delta = study.compute_arm_verdict(
            metric_value=120.0,
            baseline_value=100.0,
            direction="maximize",
            noise_floor_pct=1.0,
            gates_passed=True,
            replicates=1,
            required_replicates=3,
        )
        self.assertEqual(verdict, "tie")
        self.assertEqual(delta, 20.0)

    def test_invalid_baseline_or_direction(self):
        v1, _ = study.compute_arm_verdict(100.0, 0.0, "maximize")
        self.assertEqual(v1, "invalid")
        v2, _ = study.compute_arm_verdict(100.0, 100.0, "unknown")
        self.assertEqual(v2, "invalid")


class StudyLockingAndSectionCapsTestCase(StudyStoreTestCase):
    """Advisory file locking and state section cap enforcement."""

    def test_study_lock_is_reentrant_and_does_not_pollute_list_studies(self):
        with study.study_lock("unopened-study"):
            with study.study_lock("unopened-study"):
                pass
        self.assertEqual(study.list_studies(), [])

    def test_concurrent_attach_run_preserves_all_updates(self):
        import threading

        study.start_study("concurrent-study",
                          self.protocol(),
                          package="tpu-agent-suite",
                          target="m")
        run_ids = [f"20260922T1200{i:02d}Z-0123456789ab" for i in range(8)]
        errors: list[Exception] = []

        def worker(rid: str) -> None:
            try:
                study.attach_run_to_study("concurrent-study", rid)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(r, )) for r in run_ids
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        doc = study.load_study("concurrent-study")
        self.assertEqual(sorted(doc["runs"]), sorted(run_ids))
        self.assertEqual(study.study_round(doc), 8)

    def test_save_study_caps_amendments_and_archives_oldest_to_journal(self):
        study.start_study("capped-amendments",
                          self.protocol(),
                          package="tpu-agent-suite",
                          target="m")
        total = study.MAX_STUDY_AMENDMENTS + 3
        for idx in range(total):
            updated = self.protocol(noise_floor_pct=float(idx + 1))
            study.set_protocol("capped-amendments",
                               updated,
                               reason=f"reason-{idx}")

        doc = study.load_study("capped-amendments")
        self.assertEqual(len(doc["amendments"]), study.MAX_STUDY_AMENDMENTS)
        self.assertEqual([a["reason"] for a in doc["amendments"]],
                         [f"reason-{i}" for i in range(3, total)])

        journal = study.read_study_journal("capped-amendments")
        self.assertEqual(len(journal), 3)
        self.assertEqual([j["evicted"]["reason"] for j in journal],
                         ["reason-0", "reason-1", "reason-2"])

    def test_save_study_caps_arms_evicting_weakest_oldest_to_journal(self):
        study.start_study("capped-arms",
                          self.protocol(),
                          package="tpu-agent-suite",
                          target="m")
        total = study.MAX_STUDY_ARMS + 3
        arms = []
        for idx in range(total):
            if idx == 0:
                status, verdict, delta = "decided", "invalid", None
            elif idx in (1, 2):
                status, verdict, delta = "decided", "loss", -4.0
            elif idx == total - 1:
                status, verdict, delta = "in_flight", None, None
            elif idx == total - 2:
                status, verdict, delta = "open", None, None
            elif idx == 3:
                status, verdict, delta = "decided", "win", 5.0
            else:
                status, verdict, delta = "decided", "tie", 0.5

            arm_entry: dict[str, object] = {
                "arm": f"a{idx}",
                "kind": "kernel-block",
                "status": status,
                "claim": f"claim-{idx}",
            }
            if verdict is not None:
                arm_entry["result"] = {
                    "verdict": verdict,
                    "metric": {
                        "value": 100.0,
                        "delta_pct": delta,
                        "replicates": 3,
                        "provenance": "measured",
                    },
                }
            arms.append(arm_entry)

        doc = study.load_study("capped-arms")
        doc["arms"] = arms
        study.save_study(doc)

        saved = study.load_study("capped-arms")
        self.assertEqual(len(saved["arms"]), study.MAX_STUDY_ARMS)
        kept_ids = [a["arm"] for a in saved["arms"]]
        self.assertNotIn("a0", kept_ids)
        self.assertNotIn("a1", kept_ids)
        self.assertNotIn("a2", kept_ids)
        self.assertIn("a3", kept_ids)
        self.assertIn(f"a{total - 2}", kept_ids)
        self.assertIn(f"a{total - 1}", kept_ids)

        journal = study.read_study_journal("capped-arms")
        self.assertEqual(len(journal), 3)
        self.assertEqual([j["evicted"]["arm"] for j in journal],
                         ["a0", "a1", "a2"])

    def test_cap_eviction_never_evicts_win_over_untested_open_arms(self):
        study.start_study("win-survives-open-flood",
                          self.protocol(),
                          package="tpu-agent-suite",
                          target="m")
        arms: list[dict[str, object]] = [{
            "arm": "proven-win",
            "kind": "kernel-block",
            "status": "decided",
            "claim": "12% proven win",
            "result": {
                "verdict": "win",
                "metric": {
                    "value": 112.0,
                    "delta_pct": 12.0,
                    "replicates": 3,
                    "provenance": "measured",
                },
            },
        }]
        for idx in range(study.MAX_STUDY_ARMS):
            arms.append({
                "arm": f"open-{idx}",
                "kind": "kernel-block",
                "status": "open",
                "claim": f"untested candidate {idx}",
            })

        doc = study.load_study("win-survives-open-flood")
        doc["arms"] = arms
        study.save_study(doc)

        saved = study.load_study("win-survives-open-flood")
        self.assertEqual(len(saved["arms"]), study.MAX_STUDY_ARMS)
        kept_ids = [a["arm"] for a in saved["arms"]]
        self.assertIn("proven-win", kept_ids)
        self.assertNotIn("open-0", kept_ids)

        journal = study.read_study_journal("win-survives-open-flood")
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["evicted"]["arm"], "open-0")


class StudyArmClaimAndReleaseTestCase(RoundRecordTestCase):
    """Atomic arm reservation and release for parallel worker fleets."""

    def test_claim_arm_by_id_transitions_open_to_in_flight(self):
        self.start("claim-id")
        doc = study.load_study("claim-id")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
            {
                "arm": "a1",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c1"
            },
        ]
        study.save_study(doc)

        claimed = study.claim_arm("claim-id",
                                  arm_id="a1",
                                  worker_id="worker-99")
        self.assertEqual(claimed["arm"], "a1")
        self.assertEqual(claimed["status"], "in_flight")
        self.assertEqual(claimed["claimed_by"], "worker-99")
        self.assertIn("claimed_at", claimed)

        reloaded = study.load_study("claim-id")
        arm1 = [a for a in reloaded["arms"] if a["arm"] == "a1"][0]
        self.assertEqual(arm1["status"], "in_flight")
        self.assertEqual(arm1["claimed_by"], "worker-99")

    def test_claim_arm_auto_selects_highest_expected_gain_first(self):
        self.start("claim-order")
        doc = study.load_study("claim-order")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0",
                "share": 0.1,
                "headroom": 0.2
            },
            {
                "arm": "a1",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c1",
                "share": 0.5,
                "headroom": 0.8
            },
            {
                "arm": "a2",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c2",
                "share": 0.3,
                "headroom": 0.5
            },
        ]
        study.save_study(doc)

        c1 = study.claim_arm("claim-order", worker_id="w1")
        self.assertEqual(c1["arm"], "a1")
        c2 = study.claim_arm("claim-order", worker_id="w2")
        self.assertEqual(c2["arm"], "a2")
        c3 = study.claim_arm("claim-order", worker_id="w3")
        self.assertEqual(c3["arm"], "a0")

        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.claim_arm("claim-order")
        self.assertIn("no open arms available", str(ctx.exception))

    def test_claim_arm_refuses_non_open_arm(self):
        self.start("claim-refuse")
        doc = study.load_study("claim-refuse")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "c0"
            },
            {
                "arm": "a1",
                "kind": "kernel-block",
                "status": "in_flight",
                "claim": "c1"
            },
        ]
        study.save_study(doc)

        with self.assertRaises(study.TpuAgentSuiteError) as ctx0:
            study.claim_arm("claim-refuse", arm_id="a0")
        self.assertIn("cannot be claimed", str(ctx0.exception))

        with self.assertRaises(study.TpuAgentSuiteError) as ctx1:
            study.claim_arm("claim-refuse", arm_id="a1")
        self.assertIn("cannot be claimed", str(ctx1.exception))

        with self.assertRaises(study.TpuAgentSuiteError) as ctx_missing:
            study.claim_arm("claim-refuse", arm_id="nonexistent")
        self.assertIn("not found", str(ctx_missing.exception))

    def test_concurrent_multi_arm_claims_partition_distinct_arms(self):
        import threading
        self.start("concurrent-claims")
        doc = study.load_study("concurrent-claims")
        doc["arms"] = [{
            "arm": f"arm-{i}",
            "kind": "kernel-block",
            "status": "open",
            "claim": f"c{i}"
        } for i in range(8)]
        study.save_study(doc)

        claimed_ids: list[str] = []
        errors: list[Exception] = []

        def worker(wid: int):
            try:
                claimed = study.claim_arm("concurrent-claims",
                                          worker_id=f"w-{wid}")
                claimed_ids.append(claimed["arm"])
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i, )) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(claimed_ids), 8)
        self.assertEqual(len(set(claimed_ids)), 8)

        reloaded = study.load_study("concurrent-claims")
        for arm in reloaded["arms"]:
            self.assertEqual(arm["status"], "in_flight")
            self.assertTrue(arm["claimed_by"].startswith("w-"))

    def test_release_arm_transitions_in_flight_to_open_or_blocked(self):
        self.start("release-test")
        doc = study.load_study("release-test")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
        ]
        study.save_study(doc)

        study.claim_arm("release-test", arm_id="a0", worker_id="w-temp")
        released = study.release_arm("release-test",
                                     arm_id="a0",
                                     status="open",
                                     reason="preflight retry")
        self.assertEqual(released["status"], "open")
        self.assertNotIn("claimed_at", released)
        self.assertNotIn("claimed_by", released)
        self.assertEqual(released["release_reason"], "preflight retry")

        study.claim_arm("release-test", arm_id="a0", worker_id="w-temp2")
        blocked = study.release_arm("release-test",
                                    arm_id="a0",
                                    status="blocked",
                                    reason="awaiting cluster capacity")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["release_reason"],
                         "awaiting cluster capacity")

        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("release-test", arm_id="a0")
        self.assertIn("cannot be released", str(ctx.exception))

    def test_failed_dispatches_reopen_then_block_the_arm(self):
        self.start("failed-dispatch")
        doc = study.load_study("failed-dispatch")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
        ]
        study.save_study(doc)

        study.claim_arm("failed-dispatch", arm_id="a0")
        first = study.release_arm("failed-dispatch",
                                  arm_id="a0",
                                  failed=True,
                                  summary="recompile per step")
        self.assertEqual(first["status"], "open")
        self.assertEqual(first["failed_dispatches"], 1)
        self.assertNotIn("result", first)

        # A round that rewrites the arms without the count does not reset it.
        study.record_round(
            "failed-dispatch", {
                "arms": [{
                    "arm": "a0",
                    "kind": "kernel-block",
                    "status": "open",
                    "claim": "c0"
                }]
            })
        self.assertEqual(
            study.load_study("failed-dispatch")["arms"][0]
            ["failed_dispatches"], 1)

        study.claim_arm("failed-dispatch", arm_id="a0")
        second = study.release_arm("failed-dispatch",
                                   arm_id="a0",
                                   failed=True,
                                   summary="NaN in cache update")
        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["failed_dispatches"],
                         study.MAX_FAILED_DISPATCHES)
        self.assertIn("NaN in cache update", second["blocked_on"])

    def test_failed_release_needs_a_summary_and_no_verdict(self):
        self.start("failed-dispatch-edges")
        doc = study.load_study("failed-dispatch-edges")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
        ]
        study.save_study(doc)
        study.claim_arm("failed-dispatch-edges", arm_id="a0")
        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("failed-dispatch-edges",
                              arm_id="a0",
                              failed=True)
        self.assertIn("summary", str(ctx.exception))
        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("failed-dispatch-edges",
                              arm_id="a0",
                              status="invalid",
                              failed=True,
                              summary="x")
        self.assertIn("no verdict", str(ctx.exception))

    def test_claim_arm_then_record_round_succeeds_end_to_end(self):
        self.start_based("claim-record-e2e", admissibility_gates=[])
        # 1. Record open arms with share & headroom via record_round
        study.record_round(
            "claim-record-e2e",
            {
                "arms": [
                    {
                        "arm": "weak-open",
                        "kind": "kernel-block",
                        "status": "open",
                        "claim": "low ceiling",
                        "share": 0.1,
                        "headroom": 0.2,
                    },
                    {
                        "arm": "strong-open",
                        "kind": "kernel-block",
                        "status": "open",
                        "claim": "high ceiling",
                        "share": 0.5,
                        "headroom": 0.8,
                    },
                ]
            },
        )
        # 2. Claim highest expected gain arm (strong-open) and another worker claims weak-open
        claimed = study.claim_arm("claim-record-e2e", worker_id="w-1")
        self.assertEqual(claimed["arm"], "strong-open")
        study.claim_arm("claim-record-e2e", worker_id="w-2")

        # 3. Load study state (which now carries claimed_at and claimed_by) and record round
        current_arms = study.load_study("claim-record-e2e")["arms"]
        for a in current_arms:
            if a["arm"] == "strong-open":
                a["status"] = "decided"
                a["result"] = {
                    "verdict": "win",
                    "execution_id": "job-win-1",
                    "metric": {
                        "name": "tokens_per_s",
                        "value": 110.0,
                        "replicates": 3,
                        "delta_pct": 10.0,
                        "provenance": "measured",
                    },
                    "gates": [],
                }
        recorded = study.record_round("claim-record-e2e",
                                      {"arms": current_arms})
        decided_arm = [
            a for a in recorded["arms"] if a["arm"] == "strong-open"
        ][0]
        self.assertEqual(decided_arm["result"]["verdict"], "win")

    def test_multiprocess_claim_arm_races_without_double_claims(self):
        import subprocess
        import sys

        self.start("mp-claims")
        doc = study.load_study("mp-claims")
        doc["arms"] = [{
            "arm": f"mp-arm-{i}",
            "kind": "kernel-block",
            "status": "open",
            "claim": f"c{i}",
            "share": 0.5,
            "headroom": float(i + 1) / 10.0,
        } for i in range(4)]
        study.save_study(doc)

        env = dict(os.environ)
        env["TPU_AGENT_SUITE_HOME"] = str(self.home)
        repo_root = str(Path(study.__file__).resolve().parents[1])
        env["PYTHONPATH"] = repo_root

        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    ("import json; from framework import study; "
                     f"print(json.dumps(study.claim_arm('mp-claims', worker_id='p-{i}')))"
                     ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            ) for i in range(8)
        ]
        claimed_ids: list[str] = []
        refused = 0
        for proc in procs:
            out, err = proc.communicate(timeout=10)
            if proc.returncode == 0:
                claimed_ids.append(json.loads(out.strip())["arm"])
            else:
                self.assertIn("no open arms available", err)
                refused += 1

        self.assertEqual(len(claimed_ids), 4)
        self.assertEqual(len(set(claimed_ids)), 4)
        self.assertEqual(refused, 4)

    def test_cli_claim_and_release_arm(self):
        self.start("cli-claim")
        doc = study.load_study("cli-claim")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
        ]
        study.save_study(doc)

        rc_claim = runtime.main([
            "study", "claim-arm", "--study-id", "cli-claim", "--worker-id",
            "worker-cli"
        ])
        self.assertEqual(rc_claim, 0)

        reloaded = study.load_study("cli-claim")
        self.assertEqual(reloaded["arms"][0]["status"], "in_flight")
        self.assertEqual(reloaded["arms"][0]["claimed_by"], "worker-cli")

        rc_release = runtime.main([
            "study", "release-arm", "--study-id", "cli-claim", "--arm", "a0",
            "--status", "open", "--reason", "cli release"
        ])
        self.assertEqual(rc_release, 0)

        reloaded2 = study.load_study("cli-claim")
        self.assertEqual(reloaded2["arms"][0]["status"], "open")
        self.assertEqual(reloaded2["arms"][0]["release_reason"], "cli release")

    def test_record_round_merges_concurrent_arms_without_clobbering_decided_or_in_flight(
            self):
        self.start_based("merge-concurrent", admissibility_gates=[])
        study.record_round(
            "merge-concurrent",
            {
                "arms": [
                    {
                        "arm": "a-worker1",
                        "kind": "kernel-block",
                        "status": "open",
                        "claim": "c1",
                    },
                    {
                        "arm": "a-worker2",
                        "kind": "kernel-block",
                        "status": "open",
                        "claim": "c2",
                    },
                    {
                        "arm": "a-worker3",
                        "kind": "kernel-block",
                        "status": "open",
                        "claim": "c3",
                    },
                ]
            },
        )
        # Worker 1 and Worker 2 both snapshot the study before claiming
        stale_snapshot = study.load_study("merge-concurrent")
        study.claim_arm("merge-concurrent", arm_id="a-worker2", worker_id="w2")

        # Worker 1 finishes a-worker1 and records round with only its own arm or stale snapshot
        w1_arms = [
            {
                "arm": "a-worker1",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "c1",
                "result": {
                    "verdict": "win",
                    "execution_id": "job-w1",
                    "metric": {
                        "name": "tokens_per_s",
                        "value": 120.0,
                        "replicates": 3,
                        "delta_pct": 20.0,
                        "provenance": "measured",
                    },
                    "gates": [],
                },
            },
            # Stale copy of a-worker2 as 'open' even though w2 claimed it
            stale_snapshot["arms"][1],
        ]
        merged = study.record_round("merge-concurrent", {"arms": w1_arms})
        by_id = {a["arm"]: a for a in merged["arms"]}
        self.assertEqual(by_id["a-worker1"]["status"], "decided")
        # a-worker2 must preserve its in_flight claim from w2, not regress to open
        self.assertEqual(by_id["a-worker2"]["status"], "in_flight")
        self.assertEqual(by_id["a-worker2"]["claimed_by"], "w2")
        # a-worker3 omitted from w1's payload must still be preserved
        self.assertIn("a-worker3", by_id)
        self.assertEqual(by_id["a-worker3"]["status"], "open")

        # Now w2 records its round using a stale copy of a-worker1 ('open')
        w2_arms = [
            stale_snapshot["arms"][0],
            {
                "arm": "a-worker2",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "c2",
                "result": {
                    "verdict": "tie",
                    "execution_id": "job-w2",
                    "metric": {
                        "name": "tokens_per_s",
                        "value": 100.0,
                        "replicates": 3,
                        "delta_pct": 0.0,
                        "provenance": "measured",
                    },
                    "gates": [],
                },
            },
        ]
        merged2 = study.record_round("merge-concurrent", {"arms": w2_arms})
        by_id2 = {a["arm"]: a for a in merged2["arms"]}
        self.assertEqual(by_id2["a-worker1"]["status"], "decided")
        self.assertEqual(by_id2["a-worker1"]["result"]["verdict"], "win")
        self.assertEqual(by_id2["a-worker2"]["status"], "decided")

    def test_release_arm_supports_invalid_requires_blocked_on_and_clears_stale_fields(
            self):
        self.start("release-edge-cases")
        doc = study.load_study("release-edge-cases")
        doc["arms"] = [
            {
                "arm": "a0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
        ]
        study.save_study(doc)

        # 1. Release with summary sets summary and release_reason; subsequent release without reason clears both
        study.claim_arm("release-edge-cases", arm_id="a0", worker_id="w1")
        rel1 = study.release_arm("release-edge-cases",
                                 arm_id="a0",
                                 status="open",
                                 summary="first failure")
        self.assertEqual(rel1["summary"], "first failure")
        self.assertEqual(rel1["release_reason"], "first failure")

        study.claim_arm("release-edge-cases", arm_id="a0", worker_id="w2")
        rel2 = study.release_arm("release-edge-cases",
                                 arm_id="a0",
                                 status="open")
        self.assertNotIn("summary", rel2)
        self.assertNotIn("release_reason", rel2)

        # 2. status='blocked' without blocked_on or summary/reason raises TpuAgentSuiteError
        study.claim_arm("release-edge-cases", arm_id="a0", worker_id="w3")
        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("release-edge-cases",
                              arm_id="a0",
                              status="blocked")
        self.assertIn("blocked_on", str(ctx.exception))

        rel_blocked = study.release_arm("release-edge-cases",
                                        arm_id="a0",
                                        status="blocked",
                                        blocked_on="awaiting v7x reservation")
        self.assertEqual(rel_blocked["status"], "blocked")
        self.assertEqual(rel_blocked["blocked_on"], "awaiting v7x reservation")

        # 3. Reset to open, claim, and release with status='invalid' -> closes arm as decided + verdict='invalid'
        doc = study.load_study("release-edge-cases")
        doc["arms"][0]["status"] = "open"
        study.save_study(doc)
        study.claim_arm("release-edge-cases", arm_id="a0", worker_id="w4")
        rel_invalid = study.release_arm("release-edge-cases",
                                        arm_id="a0",
                                        status="invalid",
                                        summary="failed syntax preflight")
        self.assertEqual(rel_invalid["status"], "decided")
        self.assertEqual(rel_invalid["result"]["verdict"], "invalid")
        self.assertEqual(rel_invalid["summary"], "failed syntax preflight")

    def test_claim_arm_expires_stale_in_flight_leases_and_formats_pdt_timezone(
            self):
        self.start("claim-ttl")
        doc = study.load_study("claim-ttl")
        doc["arms"] = [
            {
                "arm": "orphaned-arm",
                "kind": "kernel-block",
                "status": "in_flight",
                "claim": "c0",
                "claimed_by": "crashed-worker",
                "claimed_at": "2020-01-01T00:00:00-08:00",
            },
        ]
        study.save_study(doc)

        # Claiming with claim_ttl_seconds=60.0 automatically recovers the expired in_flight lease
        reclaimed = study.claim_arm("claim-ttl",
                                    worker_id="new-worker",
                                    claim_ttl_seconds=60.0,
                                    timezone_name="America/Los_Angeles")
        self.assertEqual(reclaimed["arm"], "orphaned-arm")
        self.assertEqual(reclaimed["claimed_by"], "new-worker")
        self.assertTrue(
            reclaimed["claimed_at"].endswith("-07:00")
            or reclaimed["claimed_at"].endswith("-08:00"),
            f"Expected America/Los_Angeles offset in {reclaimed['claimed_at']}",
        )

    def test_arm_expected_gain_shared_helper_and_zero_tie_warning(self):
        import io
        from contextlib import redirect_stderr

        # Step-function kinds with share=0.0 fall back to headroom
        self.assertAlmostEqual(
            study.arm_expected_gain({
                "kind": "memory-capacity",
                "share": 0.0,
                "headroom": 0.65
            }),
            0.65,
        )
        self.assertAlmostEqual(
            study.arm_expected_gain({
                "kind": "kernel-block",
                "share": 0.4,
                "headroom": 0.5
            }),
            0.20,
        )

        self.start("zero-tie")
        doc = study.load_study("zero-tie")
        doc["arms"] = [
            {
                "arm": "z0",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c0"
            },
            {
                "arm": "z1",
                "kind": "kernel-block",
                "status": "open",
                "claim": "c1"
            },
        ]
        study.save_study(doc)

        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            study.claim_arm("zero-tie", worker_id="w-tie")
        self.assertIn("WARNING:", stderr_buf.getvalue())
        self.assertIn("0.0 expected gain", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
