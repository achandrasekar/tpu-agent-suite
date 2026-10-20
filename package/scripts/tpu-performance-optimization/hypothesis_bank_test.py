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
"""Unit tests for hypothesis_bank.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

_HERE = pathlib.Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent
_PACKAGE_DIR = _HERE.parents[2]
_REPO_ROOT = _HERE.parents[3]
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arm_microbench  # noqa: E402
import evaluate_arm  # noqa: E402
import hypothesis_bank  # noqa: E402

from framework import study  # noqa: E402
from framework import utils  # noqa: E402

STUDY = "modela-bank-test"
RUN_ID = "20260925T120000Z-0123456789ab"
KERNEL = "models/modelA/kernel.py"
SHARDING = "models/modelA/sharding.py"
ATTENTION = "models/modelA/attention.py"


def _load_schema(name: str) -> dict:
    return json.loads(
        (_PACKAGE_DIR / "schemas" / name).read_text(encoding="utf-8"))


RESULT_SCHEMA = _load_schema("hypothesis-bank-result.json")
REQUEST_SCHEMA = _load_schema("hypothesis-bank-request.json")
ITEM_SCHEMA = RESULT_SCHEMA["properties"]["hypotheses"]["items"]
# Reasons the item schema can express; the rest are enforced in code alone.
SCHEMA_CODES = {
    "invalid_shape", "missing_field", "invalid_id", "out_of_range",
    "unknown_field"
}


def _make_protocol() -> dict[str, object]:
    return {
        "primary_metric": {
            "name": "tokens_per_s",
            "unit": "tok/s",
            "direction": "maximize",
        },
        "measurement_cell": {
            "topology": "v7x-2x2x1"
        },
        "replicate_count":
        3,
        "noise_floor_pct":
        1.0,
        "stopping_rule": {
            "max_rounds": 20
        },
        "admissibility_gates": [{
            "name": "eval_loss",
            "metric": "eval_loss",
            "comparator": "lte",
            "bound": 2.20,
            "same_execution": True,
        }],
        "mode":
        "approve",
    }


def _hyp(hyp_id: str,
         kind: str = "kernel-block",
         path: str = KERNEL,
         **extra: object) -> dict[str, object]:
    hyp: dict[str, object] = {
        "id": hyp_id,
        "kind": kind,
        "claim": f"Hypothesis {hyp_id} for {kind} on {path}.",
        "evidence": [{
            "type": "code",
            "ref": f"{path}:L1-L2"
        }],
        "code_refs": [{
            "path": path
        }],
        "share": 0.3,
        "headroom": 0.4,
    }
    hyp.update(extra)
    return hyp


def _without(hyp: dict[str, object], *keys: str) -> dict[str, object]:
    return {k: v for k, v in hyp.items() if k not in keys}


def _win(arm_id: str) -> dict[str, object]:
    return {
        "arm": arm_id,
        "kind": "kernel-block",
        "claim": f"Decided win {arm_id}.",
        "status": "decided",
        "result": {
            "verdict": "win",
            "execution_id": f"job-{arm_id}"
        },
    }


def _code(reason: str) -> str:
    return reason.split(":", 1)[0]


class HypothesisBankTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self._prev_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.root / "home")
        self.repo_root = self.root / "repo"
        model_dir = self.repo_root / "models" / "modelA"
        model_dir.mkdir(parents=True)
        (model_dir / "kernel.py").write_text(
            "def chunk_scan(x):\n    return x + 1\n", encoding="utf-8")
        (model_dir / "sharding.py").write_text(
            "def shard_layers(x):\n    return x\n", encoding="utf-8")
        (model_dir / "attention.py").write_text(
            "def attend(q, k):\n    return q\n", encoding="utf-8")
        study.start_study(
            STUDY,
            protocol=_make_protocol(),
            package="tpu-agent-suite",
            target="modelA",
            owner="tester",
        )

    def tearDown(self) -> None:
        if self._prev_home is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = self._prev_home
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def admit(self, hypotheses: list[object], **kwargs: object) -> dict:
        kwargs.setdefault("run_id", RUN_ID)
        result = hypothesis_bank.admit_hypotheses(STUDY, hypotheses, **kwargs)
        self.assertEqual(utils.validate_instance(result, RESULT_SCHEMA), [],
                         result)
        return result

    def run_main(self, argv: list[str]) -> tuple[int, dict]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()):
            rc = hypothesis_bank.main(argv)
        result = json.loads(out.getvalue())
        self.assertEqual(utils.validate_instance(result, RESULT_SCHEMA), [],
                         result)
        return rc, result

    def arm_ids(self) -> list[str]:
        return [a["arm"] for a in study.load_study(STUDY).get("arms") or []]

    def bank(self) -> dict:
        return hypothesis_bank.load_hypothesis_bank(STUDY)

    def admitted_events(self) -> list[dict]:
        return [
            e for e in study.read_study_events(STUDY)
            if e.get("kind") == "hypotheses_admitted"
        ]

    @staticmethod
    def admitted_ids(result: dict) -> list[str]:
        return [a["hypothesis_id"] for a in result["admitted"]]

    @staticmethod
    def reasons(result: dict) -> dict[str, list[str]]:
        return {r["hypothesis_id"]: r["reasons"] for r in result["rejected"]}

    @staticmethod
    def warnings(result: dict) -> dict[str, list[str]]:
        return {a["hypothesis_id"]: a["warnings"] for a in result["admitted"]}

    def set_arms(self, arms: list[dict]) -> None:
        doc = study.load_study(STUDY)
        doc["arms"] = arms
        study.save_study(doc)

    # -- kept from the synthesizer suite, adapted --------------------------

    def test_admitted_arms_match_study_round_schema_and_round_trip(
            self) -> None:
        res = self.admit(
            [
                _hyp("s1",
                     target_parameters={"block_size": 128},
                     kb_references=["kb/vmem.md"]),
                _hyp("s2", "exposed-collective", SHARDING),
            ],
            repo_root=self.repo_root,
        )
        self.assertEqual(self.admitted_ids(res), ["s1", "s2"])
        self.assertEqual(res["rejected"], [])
        arm_schema = study.load_study_round_schema(
        )["properties"]["arms"]["items"]
        doc = study.load_study(STUDY)
        for arm in doc["arms"]:
            self.assertEqual(study.validate_instance(arm, arm_schema), [], arm)
            self.assertEqual(
                set(arm),
                {"arm", "kind", "claim", "status", "share", "headroom"})
        # Negative control: the same check fires on an arm carrying bank data.
        bad = dict(doc["arms"][0], code_refs=[{"path": KERNEL}])
        self.assertTrue(
            any("additional property" in e
                for e in study.validate_instance(bad, arm_schema)))

        # A planner round echoing the arms back must be accepted.
        study.record_round(STUDY, {"arms": doc["arms"], "next": "echo"})
        self.assertEqual(self.arm_ids(), ["s1", "s2"])

        bank = self.bank()
        self.assertEqual(bank["arms"]["s1"]["target_parameters"],
                         {"block_size": 128})
        self.assertEqual(bank["arms"]["s1"]["fingerprint"],
                         res["admitted"][0]["arms"][0]["fingerprint"])
        self.assertNotIn("target_parameters", bank["arms"]["s2"])
        self.assertEqual(bank["hypotheses"]["s1"]["kb_references"],
                         ["kb/vmem.md"])
        self.assertEqual(bank["hypotheses"]["s1"]["request_id"], RUN_ID)
        [event] = self.admitted_events()
        self.assertEqual(event["run_id"], RUN_ID)
        self.assertEqual(event["arm_ids"], ["s1", "s2"])

    def test_repeat_call_with_fewer_fields_leaves_no_stale_metadata(
            self) -> None:
        self.admit(
            [
                _hyp("r1",
                     target_parameters={"block_size": 128},
                     kb_references=["kb/a.md"])
            ],
            repo_root=self.repo_root,
        )
        first = self.bank()
        # Second call: no repo_root, no target_parameters, no kb_references.
        res = self.admit([_hyp("r2", "exposed-collective", SHARDING)])
        self.assertEqual(self.admitted_ids(res), ["r2"])
        bank = self.bank()
        self.assertEqual(bank["arms"]["r1"], first["arms"]["r1"])
        self.assertEqual(bank["hypotheses"]["r1"], first["hypotheses"]["r1"])
        self.assertNotIn("target_parameters", bank["arms"]["r2"])
        self.assertNotIn("kb_references", bank["hypotheses"]["r2"])
        self.assertEqual(self.arm_ids(), ["r1", "r2"])

        # Repeating the first call changes nothing and says why.
        again = self.admit([_hyp("r1", target_parameters={"block_size": 128})])
        self.assertEqual(again["admitted"], [])
        self.assertEqual([_code(r) for r in self.reasons(again)["r1"]], [
            "duplicate_hypothesis_id", "duplicate_arm_id", "duplicate_recipe"
        ])
        self.assertEqual(self.arm_ids(), ["r1", "r2"])
        self.assertEqual(self.bank(), bank)

    def test_crash_before_study_write_leaves_study_unchanged_and_retry_works(
            self) -> None:
        crashed = [
            _hyp("c1",
                 target_parameters={"block_size": 128},
                 kb_references=["kb/a.md"]),
            _hyp("c2", path=SHARDING, target_parameters={"block_size": 256}),
        ]
        with mock.patch.object(study,
                               "save_study",
                               side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                hypothesis_bank.admit_hypotheses(STUDY, crashed, run_id=RUN_ID)
        self.assertEqual(study.load_study(STUDY).get("arms", []), [])
        # The bank is written first, so the orphans exist but are inert.
        self.assertEqual(sorted(self.bank()["arms"]), ["c1", "c2"])

        # c1 retried with the same recipe: an orphan is not a duplicate.
        # c2 retried with fewer fields: the entry is replaced, not merged.
        res = self.admit([
            _without(crashed[0], "kb_references"),
            _without(crashed[1], "target_parameters"),
        ])
        self.assertEqual(self.admitted_ids(res), ["c1", "c2"])
        self.assertEqual(self.arm_ids(), ["c1", "c2"])
        bank = self.bank()
        self.assertNotIn("kb_references", bank["hypotheses"]["c1"])
        self.assertEqual(bank["arms"]["c1"]["target_parameters"],
                         {"block_size": 128})
        self.assertNotIn("target_parameters", bank["arms"]["c2"])
        self.assertNotIn("target_parameters", bank["hypotheses"]["c2"])
        self.assertEqual(bank["retired"], [])

    def test_concurrent_admit_loses_no_arm_or_bank_entry(self) -> None:
        kinds = ("kernel-block", "exposed-collective", "sharding-overhead",
                 "host-critical-path")
        barrier = threading.Barrier(len(kinds))
        errors: list[BaseException] = []

        def run(i: int) -> None:
            try:
                barrier.wait()
                hypothesis_bank.admit_hypotheses(STUDY, [
                    _hyp(f"k{i}",
                         kinds[i],
                         f"models/f{i}.py",
                         target_parameters={"block_size": 128 * (i + 1)})
                ],
                                                 run_id=RUN_ID)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=(i, )) for i in range(len(kinds))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])
        want = [f"k{i}" for i in range(len(kinds))]
        self.assertEqual(sorted(self.arm_ids()), want)
        bank = self.bank()
        self.assertEqual(sorted(bank["arms"]), want)
        self.assertEqual(sorted(bank["hypotheses"]), want)
        for i in range(len(kinds)):
            self.assertEqual(bank["arms"][f"k{i}"]["target_parameters"],
                             {"block_size": 128 * (i + 1)})
        self.assertEqual(len(self.admitted_events()), len(kinds))

    def test_journal_evicted_recipe_blocks_duplicate(self) -> None:
        self.admit([_hyp("old", target_parameters={"block_size": 128})])
        old_fp = self.bank()["arms"]["old"]["fingerprint"]
        # A round decides the arm and the state cap evicts it to the journal.
        evicted = {
            "arm": "old",
            "kind": "kernel-block",
            "status": "decided",
            "claim": "old claim",
            "result": {
                "verdict": "loss",
                "execution_id": "job-old"
            },
        }
        study.append_study_journal(STUDY, {
            "section": "arms",
            "evicted": evicted
        })
        self.set_arms([])
        # The journal record alone could never match the parameter recipe.
        self.assertNotEqual(hypothesis_bank._claim_fingerprint(evicted),
                            old_fp)

        res = self.admit([
            _hyp("same-recipe",
                 path=ATTENTION,
                 target_parameters={"block_size": 128}),
            # Lookalike with a different recipe must still be admitted.
            _hyp("new-recipe",
                 path=ATTENTION,
                 target_parameters={"block_size": 256}),
        ])
        self.assertEqual(self.admitted_ids(res), ["new-recipe"])
        [reason] = self.reasons(res)["same-recipe"]
        self.assertTrue(reason.startswith("duplicate_recipe:old:"), reason)
        # That call's prune retired the recipe, so it keeps blocking after the
        # journal record leaves the tail.
        self.assertEqual(self.bank()["retired"], [{
            "arm": "old",
            "hypothesis_id": "old",
            "fingerprint": old_fp
        }])
        study.append_study_journal(STUDY, {
            "section": "amendments",
            "evicted": {
                "note": "filler"
            }
        })
        res = self.admit([
            _hyp("third", path=SHARDING, target_parameters={"block_size": 128})
        ],
                         journal_dedup_limit=1)
        [reason] = self.reasons(res)["third"]
        self.assertTrue(reason.startswith("duplicate_recipe:old:"), reason)

    def test_main_dry_run_does_not_mutate_study(self) -> None:
        argv = [
            "--study-id", STUDY, "--run-id", RUN_ID, "--hypotheses-json",
            json.dumps({"hypotheses": [_hyp("d1")]})
        ]
        rc, out = self.run_main(argv + ["--dry-run"])
        self.assertEqual((rc, out["status"]), (0, "completed"))
        self.assertEqual(self.admitted_ids(out), ["d1"])
        self.assertEqual(out["open_arms"], 1)
        self.assertIn("Dry run", out["summary"])
        self.assertEqual(study.load_study(STUDY).get("arms", []), [])
        self.assertFalse(hypothesis_bank.hypothesis_bank_path(STUDY).exists())
        self.assertEqual(self.admitted_events(), [])
        # Negative control: the same call without --dry-run writes.
        rc, out = self.run_main(argv)
        self.assertEqual((rc, self.admitted_ids(out)), (0, ["d1"]))
        self.assertEqual(self.arm_ids(), ["d1"])
        self.assertTrue(hypothesis_bank.hypothesis_bank_path(STUDY).exists())
        self.assertEqual(len(self.admitted_events()), 1)

    def test_bank_prunes_on_state_cap_and_fills_retired(self) -> None:
        hyps = [
            _hyp(f"a{i}", target_parameters={"block_size": 128 * (i + 1)})
            for i in range(20)
        ]
        res = self.admit(hyps, max_new_arms=20)
        # All 20 pass screening. The 16-arm cap evicts the four oldest of the
        # equal-gain open arms on admission, so they are reported as rejected.
        self.assertEqual(self.admitted_ids(res),
                         [f"a{i}" for i in range(4, 20)])
        self.assertEqual(sorted(self.reasons(res)),
                         sorted(f"a{i}" for i in range(4)))
        for i in range(4):
            [reason] = self.reasons(res)[f"a{i}"]
            self.assertTrue(reason.startswith("evicted_by_arm_cap:"), reason)
        self.assertEqual(res["open_arms"], study.MAX_STUDY_ARMS)
        live = set(self.arm_ids())
        self.assertEqual(len(live), study.MAX_STUDY_ARMS)
        bank = self.bank()
        self.assertEqual(set(bank["arms"]), live)
        self.assertEqual(set(bank["hypotheses"]), live)
        fps = {
            h["id"]:
            evaluate_arm.compute_arm_fingerprint(
                kind="kernel-block", target_parameters=h["target_parameters"])
            for h in hyps
        }
        self.assertEqual([(r["arm"], r["fingerprint"])
                          for r in bank["retired"]],
                         [(f"a{i}", fps[f"a{i}"]) for i in range(4)])

        # A round that drops an arm leaves a stale bank entry; the next call,
        # even an empty one, retires it.
        doc = study.load_study(STUDY)
        dropped = doc["arms"].pop(0)["arm"]
        study.save_study(doc)
        self.admit([])
        after = self.bank()
        self.assertEqual(set(after["arms"]), live - {dropped})
        self.assertEqual(len(after["retired"]), 5)
        self.assertEqual(after["retired"][-1]["arm"], dropped)

        # `retired` keeps only the newest entries.
        doc = study.load_study(STUDY)
        dropped_again = doc["arms"].pop(0)["arm"]
        study.save_study(doc)
        with mock.patch.object(hypothesis_bank, "DEFAULT_RETIRED_LIMIT", 2):
            self.admit([])
        self.assertEqual([r["arm"] for r in self.bank()["retired"]],
                         [dropped, dropped_again])

    def test_bounded_journal_dedup_read(self) -> None:
        with mock.patch.object(study,
                               "read_study_journal",
                               wraps=study.read_study_journal) as spy:
            self.admit([_hyp("b1")], journal_dedup_limit=16)
            spy.assert_called_once_with(STUDY, limit=16)
            # Negative control: a limit of 0 (read everything) is refused.
            res = self.admit([_hyp("b2")], journal_dedup_limit=0)
            self.assertEqual(res["status"], "invalid_input")
            spy.assert_called_once()

    # -- spec tests 1-11 ---------------------------------------------------

    def test_two_file_sharding_hypothesis_without_hlo_evidence_is_admitted(
            self) -> None:
        hyp = _hyp(
            "sh1",
            "sharding-overhead",
            SHARDING,
            evidence=[
                {
                    "type": "sharding",
                    "ref": f"{SHARDING}:L1-L2",
                    "note": "weights replicated on every chip"
                },
                {
                    "type": "code",
                    "ref": f"{ATTENTION}:attend"
                },
            ],
            code_refs=[{
                "path": SHARDING,
                "symbol": "shard_layers"
            }, {
                "path": ATTENTION,
                "lines": "L1-L2"
            }],
        )
        res = self.admit([hyp], repo_root=self.repo_root)
        self.assertEqual(self.admitted_ids(res), ["sh1"])
        self.assertEqual(res["admitted"][0]["warnings"], [])
        # Negative control: the path check was live, so a missing file rejects.
        bad = dict(hyp,
                   id="sh2",
                   code_refs=[{
                       "path": SHARDING
                   }, {
                       "path": "models/modelA/missing.py"
                   }],
                   target_parameters={"mesh": "2x2"})
        res = self.admit([bad], repo_root=self.repo_root)
        self.assertEqual(res["admitted"], [])
        self.assertEqual(
            self.reasons(res)["sh2"], [
                "missing_code_ref:models/modelA/missing.py: no such file under"
                " repo_root"
            ])

    def test_step_function_gain_draws_no_low_gain_warning(self) -> None:
        res = self.admit(
            [_hyp("m1", "memory-capacity", share=0.0, headroom=0.6)])
        self.assertEqual(self.admitted_ids(res), ["m1"])
        self.assertEqual(res["admitted"][0]["warnings"], [])
        # Control: kernel-block at share 0.01 x headroom 0.3 = 0.003.
        res = self.admit([_hyp("k1", path=SHARDING, share=0.01, headroom=0.3)])
        self.assertEqual(self.admitted_ids(res), ["k1"])
        [warning] = res["admitted"][0]["warnings"]
        self.assertTrue(warning.startswith("low_expected_gain:"), warning)
        # A missing estimate is a warning too, never a reject.
        res = self.admit([_without(_hyp("n1", path=ATTENTION), "share")])
        self.assertEqual(self.admitted_ids(res), ["n1"])
        [warning] = res["admitted"][0]["warnings"]
        self.assertTrue(warning.startswith("no_gain_estimate:"), warning)
        self.assertIn("ranks last for claim_arm", warning)

    def test_zero_evidence_is_rejected(self) -> None:
        res = self.admit([
            _hyp("e0", evidence=[]),
            _without(_hyp("e1", path=SHARDING), "evidence"),
        ])
        self.assertEqual(res["admitted"], [])
        self.assertEqual([_code(r) for r in self.reasons(res)["e0"]],
                         ["no_evidence"])
        self.assertEqual([_code(r) for r in self.reasons(res)["e1"]],
                         ["missing_field", "no_evidence"])
        # Control: one evidence item is enough.
        res = self.admit([_hyp("e2", path=ATTENTION)])
        self.assertEqual(self.admitted_ids(res), ["e2"])

    def test_unknown_field_and_invalid_kind_are_rejected(self) -> None:
        res = self.admit([
            _hyp("u1", attributed_ops=["custom-call.fused_scan.1"]),
            _hyp("u2", "kernel", SHARDING),
        ])
        self.assertEqual(res["admitted"], [])
        self.assertEqual(
            self.reasons(res)["u1"], [
                "unknown_field:attributed_ops: superseded by evidence[type=hlo-op]"
            ])
        self.assertEqual([_code(r) for r in self.reasons(res)["u2"]],
                         ["invalid_kind"])
        # The echo keeps known keys only.
        echo = {h["id"]: h for h in res["hypotheses"]}
        self.assertNotIn("attributed_ops", echo["u1"])
        # Control: the same proposals, fixed, are admitted.
        res = self.admit([_hyp("u1"), _hyp("u2", "kernel-block", SHARDING)])
        self.assertEqual(self.admitted_ids(res), ["u1", "u2"])

    def test_code_ref_path_escape_is_rejected(self) -> None:
        escapes = ("../../etc/passwd", "sub/../../etc/passwd", "/etc/passwd")
        res = self.admit(
            [
                _hyp(f"p{i}", code_refs=[{
                    "path": path
                }]) for i, path in enumerate(escapes)
            ],
            repo_root=self.repo_root,
        )
        self.assertEqual(res["admitted"], [])
        for i, path in enumerate(escapes):
            self.assertEqual(
                self.reasons(res)[f"p{i}"],
                [f"code_ref_outside_repo:{path}: resolves outside repo_root"])
        # Control: an existing file inside the repo is admitted.
        res = self.admit([_hyp("p9")], repo_root=self.repo_root)
        self.assertEqual(self.admitted_ids(res), ["p9"])

    def test_duplicate_recipe_of_open_arm_is_rejected(self) -> None:
        self.admit([_hyp("first", target_parameters={"block_size": 256})])
        res = self.admit([
            _hyp("second",
                 path=SHARDING,
                 target_parameters={"block_size": 256}),
            _hyp("third",
                 path=ATTENTION,
                 target_parameters={"block_size": 512}),
        ])
        self.assertEqual(self.admitted_ids(res), ["third"])
        [reason] = self.reasons(res)["second"]
        self.assertTrue(reason.startswith("duplicate_recipe:first:"), reason)

        # evaluate_arm.find_duplicate_arm skips open arms, so it misses this.
        def prior() -> list[dict]:
            return hypothesis_bank.with_hypothesis_metadata(
                study.load_study(STUDY)["arms"], self.bank())

        self.assertIsNone(
            evaluate_arm.find_duplicate_arm(
                kind="kernel-block",
                target_parameters={"block_size": 256},
                prior_arms=prior()))
        # ...and it does find the arm once it is in flight.
        study.claim_arm(STUDY, "first", worker_id="w1")
        self.assertEqual(
            evaluate_arm.find_duplicate_arm(
                kind="kernel-block",
                target_parameters={"block_size": 256},
                prior_arms=prior())["arm"], "first")

    def test_variants_expand_to_claimable_arms(self) -> None:
        hyp = _hyp(
            "h",
            target_parameters={"fuse_recurrence": True},
            variants=[
                {
                    "suffix": "a",
                    "target_parameters": {
                        "block_size": 256
                    }
                },
                {
                    "suffix": "b",
                    "target_parameters": {
                        "block_size": 512
                    },
                    "claim": "Variant b claim."
                },
            ],
        )
        res = self.admit([hyp])
        [item] = res["admitted"]
        self.assertEqual([a["arm"] for a in item["arms"]], ["h-a", "h-b"])
        self.assertNotEqual(item["arms"][0]["fingerprint"],
                            item["arms"][1]["fingerprint"])
        doc = study.load_study(STUDY)
        self.assertEqual([a["claim"] for a in doc["arms"]],
                         [f"{hyp['claim']} [a]", "Variant b claim."])
        meta = {
            a["arm"]: a
            for a in hypothesis_bank.with_hypothesis_metadata(
                doc["arms"], self.bank())
        }
        self.assertEqual(meta["h-a"]["target_parameters"], {
            "fuse_recurrence": True,
            "block_size": 256
        })
        self.assertEqual(meta["h-b"]["target_parameters"], {
            "fuse_recurrence": True,
            "block_size": 512
        })
        self.assertEqual(meta["h-a"]["hypothesis_id"], "h")
        self.assertEqual(meta["h-b"]["code_refs"], [{"path": KERNEL}])
        for arm_id in ("h-a", "h-b"):
            self.assertEqual(
                study.claim_arm(STUDY, arm_id, worker_id="w1")["status"],
                "in_flight")

        repeated_suffix = _hyp("h2",
                               path=SHARDING,
                               variants=[{
                                   "suffix": "a"
                               }, {
                                   "suffix": "a",
                                   "target_parameters": {
                                       "block_size": 128
                                   }
                               }])
        # Two variants whose merged recipes are identical: one recipe twice.
        same_recipe = _hyp("h3",
                           path=ATTENTION,
                           target_parameters={"block_size": 128},
                           variants=[{
                               "suffix": "x"
                           }, {
                               "suffix": "y",
                               "target_parameters": {
                                   "block_size": 128
                               }
                           }])
        res = self.admit([repeated_suffix, same_recipe])
        self.assertEqual(res["admitted"], [])
        self.assertEqual([_code(r) for r in self.reasons(res)["h2"]],
                         ["duplicate_variant_suffix"])
        fp = evaluate_arm.compute_arm_fingerprint(
            kind="kernel-block", target_parameters={"block_size": 128})
        self.assertEqual(
            self.reasons(res)["h3"], [
                f"duplicate_recipe:h3-x: arm h3-y has the same recipe"
                f" (fingerprint {fp})"
            ])

    def test_max_new_arms_walks_array_order_first_fit(self) -> None:
        res = self.admit(
            [
                _hyp("o1", share=0.1, headroom=0.1),
                _hyp("o2", path=SHARDING, share=0.2, headroom=0.2),
                # Highest expected gain, last in the array: still deferred.
                _hyp("o3", path=ATTENTION, share=0.9, headroom=0.9),
            ],
            max_new_arms=2,
        )
        self.assertEqual(self.admitted_ids(res), ["o1", "o2"])
        self.assertEqual(res["deferred"], ["o3"])
        self.assertIn("o3 (needs 1 arm(s), 0 left)", res["summary"])
        self.assertEqual(self.arm_ids(), ["o1", "o2"])

        # First fit: a hypothesis needing more arms than remain is deferred,
        # and a later one that fits is still admitted.
        big = _hyp("big",
                   variants=[{
                       "suffix": f"v{i}",
                       "target_parameters": {
                           "block_size": 128 * i
                       }
                   } for i in (1, 2, 3)])
        small = _hyp("small",
                     path=SHARDING,
                     target_parameters={"block_size": 1024})
        res = self.admit([big, small], max_new_arms=2)
        self.assertEqual(res["deferred"], ["big"])
        self.assertEqual(self.admitted_ids(res), ["small"])
        self.assertNotIn("big-v1", self.arm_ids())

    def test_evicted_at_admission_is_rejected_not_admitted(self) -> None:
        wins = [_win(f"w{i}") for i in range(study.MAX_STUDY_ARMS)]
        self.set_arms(wins)
        late = _without(_hyp("late"), "share", "headroom")
        res = self.admit([late])
        self.assertEqual(res["admitted"], [])
        [reason] = self.reasons(res)["late"]
        self.assertTrue(reason.startswith("evicted_by_arm_cap:"), reason)
        self.assertEqual(res["open_arms"], 0)
        self.assertEqual(self.arm_ids(), [w["arm"] for w in wins])
        bank = self.bank()
        self.assertEqual((bank["arms"], bank["hypotheses"]), ({}, {}))
        self.assertEqual(bank["retired"], [{
            "arm":
            "late",
            "hypothesis_id":
            "late",
            "fingerprint":
            evaluate_arm.compute_arm_fingerprint(kind="kernel-block",
                                                 claim=late["claim"]),
        }])
        self.assertEqual(self.admitted_events(), [])
        # Control: with room for one arm, a new hypothesis is admitted.
        self.set_arms(wins[:-1])
        res = self.admit([dict(late, id="late2", claim="Another late idea.")])
        self.assertEqual(self.admitted_ids(res), ["late2"])

    def test_partial_eviction_withdraws_sibling_arms(self) -> None:
        wins = [_win(f"w{i}") for i in range(study.MAX_STUDY_ARMS - 1)]
        self.set_arms(wins)
        pair = _hyp("pair",
                    target_parameters={"fuse": True},
                    variants=[{
                        "suffix": "a",
                        "target_parameters": {
                            "block_size": 256
                        }
                    }, {
                        "suffix": "b",
                        "target_parameters": {
                            "block_size": 512
                        }
                    }])
        res = self.admit([pair])
        self.assertEqual(res["admitted"], [])
        [reason] = self.reasons(res)["pair"]
        # Equal gains: the older arm, pair-a, is the one the cap evicts.
        self.assertTrue(reason.startswith("evicted_by_arm_cap:"), reason)
        self.assertIn("evicted pair-a on admission", reason)
        self.assertEqual(self.arm_ids(), [w["arm"] for w in wins])
        bank = self.bank()
        self.assertEqual(bank["arms"], {})
        self.assertEqual(sorted(r["arm"] for r in bank["retired"]),
                         ["pair-a", "pair-b"])
        # Negative control: with room for both arms it is admitted whole.
        self.set_arms(wins[:-1])
        res = self.admit([dict(pair, id="pair2", target_parameters={})])
        self.assertEqual([a["arm"] for a in res["admitted"][0]["arms"]],
                         ["pair2-a", "pair2-b"])

    def test_hardware_gotcha_is_a_warning_not_a_veto(self) -> None:
        res = self.admit([_hyp("hw1", target_parameters={"block_size": 100})],
                         tpu_version="v7x")
        self.assertEqual(self.admitted_ids(res), ["hw1"])
        [warning] = res["admitted"][0]["warnings"]
        self.assertTrue(warning.startswith("hardware: "), warning)
        self.assertIn("block_size=100", warning)
        # Control: an aligned block size draws no hardware warning.
        res = self.admit([
            _hyp("hw2", path=SHARDING, target_parameters={"block_size": 128})
        ],
                         tpu_version="v7x")
        self.assertEqual(res["admitted"][0]["warnings"], [])
        # Code checks run on existing .py code_refs and name the file.
        (self.repo_root / "models" / "modelA" / "loop.py").write_text(
            "def step(xs):\n    for x in xs:\n        x.item()\n",
            encoding="utf-8")
        res = self.admit([_hyp("hw3", path="models/modelA/loop.py")],
                         repo_root=self.repo_root)
        [warning] = res["admitted"][0]["warnings"]
        self.assertTrue(
            warning.startswith("hardware:models/modelA/loop.py: line 3:"),
            warning)
        # A malformed override is reported, not raised.
        res = self.admit([
            _hyp("hw4",
                 path=ATTENTION,
                 target_parameters={
                     "block_size": 128,
                     "systolic_lane_dim": 0
                 })
        ])
        [warning] = res["admitted"][0]["warnings"]
        self.assertTrue(
            warning.startswith("hardware: check_failed: ZeroDivisionError"),
            warning)

        # An unknown or duplicate generation alias is invalid input: no silent v7x fallback.
        for bad_gen in ("v99", "trillium", "ironwood"):
            res = hypothesis_bank.admit_hypotheses(STUDY, [_hyp("hw5")],
                                                   tpu_version=bad_gen)
            self.assertEqual(res["status"], "invalid_input")
            self.assertEqual(utils.validate_instance(res, RESULT_SCHEMA), [])
        rc, out = self.run_main([
            "--study-id", STUDY, "--hypotheses-json",
            json.dumps([_hyp("hw5")]), "--tpu-version", "v99"
        ])
        self.assertEqual((rc, out["status"]), (2, "invalid_input"))
        self.assertIn("v99", out["error"])
        self.assertNotIn("hw5", self.arm_ids())

    def test_cli_payloads_emit_schema_valid_results_for_every_status(
            self) -> None:
        statuses = set()
        payload = self.root / "hypotheses.json"
        payload.write_text(json.dumps({"hypotheses": [_hyp("f1")]}),
                           encoding="utf-8")
        rc, out = self.run_main([
            "--study-id", STUDY, "--run-id", RUN_ID, "--hypotheses-json",
            str(payload), "--repo-root",
            str(self.repo_root), "--tpu-version", "v7x", "--max-new-arms", "5"
        ])
        statuses.add(out["status"])
        self.assertEqual((rc, out["status"], out["request_id"]),
                         (0, "completed", RUN_ID))
        self.assertEqual(self.admitted_ids(out), ["f1"])

        # Inline JSON, a bare array, no --run-id, an inline profile summary.
        rc, out = self.run_main([
            "--study-id", STUDY, "--hypotheses-json",
            json.dumps([
                _hyp("i1",
                     path=SHARDING,
                     evidence=[{
                         "type": "hlo-op",
                         "ref": "all-gather.7"
                     }])
            ]), "--profile-summary-json",
            json.dumps({"ops": [{
                "name": "all-gather.7",
                "share": 0.1
            }]})
        ])
        self.assertEqual((rc, out["status"]), (0, "completed"))
        self.assertRegex(out["request_id"], utils.RUN_ID_PATTERN)
        self.assertNotEqual(out["request_id"], RUN_ID)
        self.assertEqual([_code(w) for w in out["admitted"][0]["warnings"]],
                         ["inflated_profile_share"])
        for item in out["hypotheses"]:
            self.assertEqual(utils.validate_instance(item, ITEM_SCHEMA), [])

        bad_inputs = (
            ["--hypotheses-json", "{not json"],
            ["--hypotheses-json",
             json.dumps({"candidate_hypotheses": []})],
            ["--hypotheses-json",
             str(self.root / "missing.json")],
            ["--hypotheses-json", "[]", "--run-id", "run-1"],
            ["--hypotheses-json", "[]", "--max-new-arms", "0"],
            ["--hypotheses-json", "[]", "--max-new-arms", "many"],
        )
        for extra in bad_inputs:
            with self.subTest(extra=extra):
                rc, out = self.run_main(["--study-id", STUDY] + extra)
                statuses.add(out["status"])
                self.assertEqual((rc, out["status"]), (2, "invalid_input"))
                self.assertTrue(out["error"])
        # argparse usage errors come back as JSON too.
        rc, out = self.run_main(["--hypotheses-json", "[]"])
        self.assertEqual((rc, out["status"]), (2, "invalid_input"))
        self.assertIn("--study-id", out["error"])

        # A study that was never started: load_study raises TpuAgentSuiteError.
        rc, out = self.run_main(
            ["--study-id", "modela-never-started", "--hypotheses-json", "[]"])
        statuses.add(out["status"])
        self.assertEqual((rc, out["status"]), (1, "failed"))
        self.assertIn("start it first", out["error"])
        self.assertEqual(statuses,
                         set(RESULT_SCHEMA["properties"]["status"]["enum"]))
        self.assertEqual(self.arm_ids(), ["f1", "i1"])

    # -- schemas -----------------------------------------------------------

    def test_code_and_item_schema_agree(self) -> None:
        base = _hyp("ok")
        fixtures = {
            "valid": (base, True),
            "full": (dict(base,
                          details="why",
                          targeted_roofline={"bound": "hbm"},
                          kb_references=["kb/a.md"],
                          target_parameters={"block_size": 256},
                          variants=[{
                              "suffix": "x",
                              "claim": "c",
                              "target_parameters": {}
                          }]), True),
            "missing claim": (_without(base, "claim"), False),
            "empty claim": (dict(base, claim=""), False),
            "id with a space": (dict(base, id="bad id"), False),
            "id not a string": (dict(base, id=7), False),
            "unknown field": (dict(base, target_files=[KERNEL]), False),
            "evidence extra key": (dict(base,
                                        evidence=[{
                                            "type": "code",
                                            "ref": "x",
                                            "url": "y"
                                        }]), False),
            "evidence empty ref": (dict(base,
                                        evidence=[{
                                            "type": "code",
                                            "ref": ""
                                        }]), False),
            "code_ref without path": (dict(base, code_refs=[{
                "lines": "L1"
            }]), False),
            "share above 1": (dict(base, share=1.5), False),
            "share bool": (dict(base, share=True), False),
            "headroom string": (dict(base, headroom="0.4"), False),
            "variant bad suffix": (dict(base, variants=[{
                "suffix": "-x"
            }]), False),
            "variant extra key": (dict(base,
                                       variants=[{
                                           "suffix": "x",
                                           "share": 0.1
                                       }]), False),
            "target_parameters list": (dict(base,
                                            target_parameters=[1]), False),
            "kb_references not strings": (dict(base,
                                               kb_references=[1]), False),
        }
        for name, (hyp, valid) in fixtures.items():
            with self.subTest(name):
                code = [
                    r for r in hypothesis_bank._hypothesis_errors(hyp)
                    if _code(r) in SCHEMA_CODES
                ]
                schema = utils.validate_instance(hyp, ITEM_SCHEMA)
                self.assertEqual(not code, valid, code)
                self.assertEqual(not schema, valid, schema)
        # Rules the framework validator cannot express live in code alone.
        code_only = {
            "kind outside BOTTLENECK_KINDS":
            (dict(base, kind="kernel"), "invalid_kind"),
            "empty evidence": (dict(base, evidence=[]), "no_evidence"),
            "repeated suffix": (dict(base,
                                     variants=[{
                                         "suffix": "x"
                                     }, {
                                         "suffix": "x"
                                     }]), "duplicate_variant_suffix"),
            "non-finite share": (dict(base,
                                      share=float("nan")), "out_of_range"),
        }
        for name, (hyp, reason) in code_only.items():
            with self.subTest(name):
                self.assertEqual(utils.validate_instance(hyp, ITEM_SCHEMA), [])
                self.assertEqual([
                    _code(r) for r in hypothesis_bank._hypothesis_errors(hyp)
                ], [reason])

    def test_request_schema_example_and_fixture(self) -> None:
        self.assertEqual(utils.validate_schema_definition(REQUEST_SCHEMA), [])
        self.assertEqual(
            utils.validate_instance(REQUEST_SCHEMA["examples"][0],
                                    REQUEST_SCHEMA), [])
        fixture = {
            "prompt": "Seed study modela-bank-test.",
            "study_id": STUDY,
            "run_id": RUN_ID,
            "tpu_version": "v6e",
            "min_open_arms": 0,
            "max_new_arms": 1,
            "profile_summary_path": "summary.json",
            "completed_arms": [],
        }
        self.assertEqual(utils.validate_instance(fixture, REQUEST_SCHEMA), [])
        # Negative controls. Hypotheses are the researcher's output now, and
        # duplicate generation aliases (trillium/ironwood) are rejected.
        for bad in (
                dict(fixture, candidate_hypotheses=[]),
                dict(fixture, max_new_arms=0),
                dict(fixture, tpu_version="v99"),
                dict(fixture, tpu_version="trillium"),
                dict(fixture, tpu_version="ironwood"),
                _without(fixture, "run_id"),
        ):
            self.assertTrue(utils.validate_instance(bad, REQUEST_SCHEMA), bad)
        self.assertEqual(
            list(hypothesis_bank.SUPPORTED_TPU_VERSIONS),
            ["v5e", "v5p", "v6e", "v7x"],
        )
        self.assertEqual(
            set(REQUEST_SCHEMA["properties"]["tpu_version"]["enum"]),
            set(hypothesis_bank.SUPPORTED_TPU_VERSIONS))
        self.assertTrue(
            set(hypothesis_bank.SUPPORTED_TPU_VERSIONS).issubset(
                set(arm_microbench.TPU_HARDWARE_SPECS)))

    def test_result_schema_example_is_real_output(self) -> None:
        example = RESULT_SCHEMA["examples"][0]
        repo = self.root / "example-repo"
        (repo / "models" / "modelA").mkdir(parents=True)
        (repo / KERNEL).write_text("def chunk_scan(x):\n    return x\n",
                                   encoding="utf-8")
        study.start_study(example["study_id"],
                          protocol=_make_protocol(),
                          package="tpu-agent-suite",
                          target="modelA-400b",
                          owner="tester")
        res = hypothesis_bank.admit_hypotheses(example["study_id"],
                                               example["hypotheses"],
                                               run_id=example["request_id"],
                                               repo_root=repo,
                                               tpu_version="v7x")
        self.assertEqual(res, example)

    # -- other paths -------------------------------------------------------

    def test_profile_mismatch_is_a_warning(self) -> None:

        def hlo(op: str) -> list[dict[str, str]]:
            return [{"type": "hlo-op", "ref": op}]

        res = self.admit(
            [
                _hyp("pf1",
                     share=0.6,
                     headroom=0.5,
                     evidence=hlo("custom-call.fused_scan.1")),
                _hyp("pf2", path=SHARDING, evidence=hlo("all-gather.99")),
                # Control: a share within tolerance of the profile.
                _hyp("pf3",
                     path=ATTENTION,
                     share=0.3,
                     evidence=hlo("custom-call.fused_scan.1")),
            ],
            profile_summary={
                "ops": [{
                    "name": "custom-call.fused_scan.1",
                    "share": 0.35
                }]
            },
        )
        self.assertEqual(self.admitted_ids(res), ["pf1", "pf2", "pf3"])
        warnings = self.warnings(res)
        self.assertEqual([_code(w) for w in warnings["pf1"]],
                         ["inflated_profile_share"])
        self.assertEqual([_code(w) for w in warnings["pf2"]],
                         ["unverified_profile_ops"])
        self.assertEqual(warnings["pf3"], [])

    def test_overlapping_code_refs_warn(self) -> None:
        self.admit([
            _hyp("ov1", code_refs=[{
                "path": KERNEL,
                "symbol": "chunk_scan"
            }])
        ])
        res = self.admit([
            _hyp("ov2",
                 path=SHARDING,
                 code_refs=[{
                     "path": "./" + KERNEL
                 }, {
                     "path": SHARDING
                 }]),
            _hyp("ov3", path=ATTENTION),
        ])
        warnings = self.warnings(res)
        self.assertEqual(warnings["ov2"], [
            f"overlaps_active_arm:ov1: code_refs ['{KERNEL}'] are also"
            " targeted by these open or in_flight arms; parallel workers"
            " editing the same code may collide"
        ])
        self.assertEqual(warnings["ov3"], [])
        # Control: a decided arm is no longer active.
        doc = study.load_study(STUDY)
        for arm in doc["arms"]:
            if arm["arm"] == "ov1":
                arm.update(status="decided",
                           result={
                               "verdict": "tie",
                               "execution_id": "job-ov1"
                           })
        study.save_study(doc)
        res = self.admit([
            _hyp("ov4",
                 code_refs=[{
                     "path": KERNEL
                 }],
                 target_parameters={"block_size": 256})
        ])
        self.assertFalse(
            any("ov1" in w for w in res["admitted"][0]["warnings"]))

    def test_hypothesis_id_reuse(self) -> None:
        self.admit([_hyp("dh", target_parameters={"block_size": 256})])
        v2 = [{"suffix": "v2", "target_parameters": {"block_size": 512}}]
        res = self.admit([_hyp("dh", path=SHARDING, variants=v2)])
        self.assertEqual([_code(r) for r in self.reasons(res)["dh"]],
                         ["duplicate_hypothesis_id"])
        # Twice in one call.
        res = self.admit(
            [_hyp("twin", path=SHARDING),
             _hyp("twin", path=ATTENTION)])
        self.assertEqual(self.admitted_ids(res), ["twin"])
        self.assertEqual(_code(self.reasons(res)["twin"][0]),
                         "duplicate_hypothesis_id")
        # Control: once its arms have left study.json, the id is free again.
        self.set_arms(
            [a for a in study.load_study(STUDY)["arms"] if a["arm"] != "dh"])
        res = self.admit([_hyp("dh", path=ATTENTION, variants=v2)])
        self.assertEqual(self.admitted_ids(res), ["dh"])
        bank = self.bank()
        self.assertEqual(bank["hypotheses"]["dh"]["code_refs"],
                         [{
                             "path": ATTENTION
                         }])
        self.assertIn("dh", [r["arm"] for r in bank["retired"]])

    def test_malformed_items_are_rejected_individually_and_not_echoed(
            self) -> None:
        res = self.admit([42, {"kind": "kernel-block"}, _hyp("good")])
        self.assertEqual(self.admitted_ids(res), ["good"])
        self.assertEqual(sorted(self.reasons(res)), ["#0", "#1"])
        self.assertEqual([h["id"] for h in res["hypotheses"]], ["good"])

    def test_invalid_input_is_returned_not_raised(self) -> None:
        cases = {
            "study id": {
                "study_id": "Bad_Study"
            },
            "run id": {
                "run_id": "run-1"
            },
            "hypotheses not a list": {
                "hypotheses": {
                    "id": "x"
                }
            },
            "max_new_arms zero": {
                "max_new_arms": 0
            },
            "max_new_arms bool": {
                "max_new_arms": True
            },
            "min_expected_gain nan": {
                "min_expected_gain": float("nan")
            },
            "repo_root missing": {
                "repo_root": self.root / "nope"
            },
            "profile_summary list": {
                "profile_summary": []
            },
            "tpu_version case": {
                "tpu_version": "V7X"
            },
        }
        for name, override in cases.items():
            with self.subTest(name):
                kwargs = {"run_id": RUN_ID, **override}
                study_id = kwargs.pop("study_id", STUDY)
                hypotheses = kwargs.pop("hypotheses", [_hyp("x")])
                res = hypothesis_bank.admit_hypotheses(study_id, hypotheses,
                                                       **kwargs)
                self.assertEqual(res["status"], "invalid_input", res)
                self.assertTrue(res["error"])
                self.assertEqual(utils.validate_instance(res, RESULT_SCHEMA),
                                 [])
                self.assertRegex(res["request_id"], utils.RUN_ID_PATTERN)
        self.assertEqual(self.arm_ids(), [])

    def test_load_hypothesis_bank_defaults_when_absent_or_malformed(
            self) -> None:
        empty = {"hypotheses": {}, "arms": {}, "retired": []}
        self.assertEqual(self.bank(), empty)
        path = hypothesis_bank.hypothesis_bank_path(STUDY)
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.bank(), empty)
        path.write_text(json.dumps({
            "hypotheses": [],
            "arms": {
                "a": 1
            },
            "retired": {}
        }),
                        encoding="utf-8")
        self.assertEqual(self.bank(), empty)


if __name__ == "__main__":
    unittest.main()
