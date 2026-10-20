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

import json
import math
import os

import compare_parity
import torch
from absl.testing import absltest, parameterized
from safetensors.torch import save_file

VOCAB = 8


def make_manifest(default_role, **overrides):
    manifest = {
        "role": default_role,
        "model_id": "tiny/model",
        "device": "cpu" if default_role == "reference" else "tpu",
        "dtype": "float32" if default_role == "reference" else "bfloat16",
        "probe_sha256": "abc123",
        "probes": [{
            "id": "p0",
            "prompt": "hello",
        }],
        "independent_reference": True,
        "modules": ["layers.0.mixer"],
    }
    manifest.update(overrides)
    return manifest


def confident_logits(positions=4, seed=0):
    """Logits whose top-1 beats top-2 by a wide, unambiguous margin."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.rand((1, positions, VOCAB), generator=generator)
    # Push one entry per position far above the rest so small perturbations
    # cannot change the argmax.
    for pos in range(positions):
        logits[0, pos, pos % VOCAB] += 10.0
    return logits


class CompareParityTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path

    def write_capture(self, prefix, manifest, tensors):
        path = os.path.join(self.tmp, prefix)
        with open(f"{path}.json", "w") as f:
            json.dump(manifest, f)
        save_file(tensors, f"{path}.safetensors")
        return path

    def write_pair(self,
                   ref_logits,
                   cand_logits,
                   ref_module=None,
                   cand_module=None,
                   ref_overrides=None,
                   cand_overrides=None):
        if ref_module is None:
            ref_module = torch.zeros((1, 4, VOCAB))
        if cand_module is None:
            cand_module = ref_module.clone()
        ref = self.write_capture(
            "reference",
            make_manifest("reference", **(ref_overrides or {})),
            {
                "p0/logits": ref_logits,
                "p0/module/layers.0.mixer": ref_module,
            },
        )
        cand = self.write_capture(
            "candidate",
            make_manifest("candidate", **(cand_overrides or {})),
            {
                "p0/logits": cand_logits,
                "p0/module/layers.0.mixer": cand_module,
            },
        )
        return ref, cand

    def run_compare(self, ref_prefix, cand_prefix, **kwargs):
        ref_manifest, ref_tensors = compare_parity.load_capture(ref_prefix)
        cand_manifest, cand_tensors = compare_parity.load_capture(cand_prefix)
        return compare_parity.compare(ref_manifest, ref_tensors, cand_manifest,
                                      cand_tensors, **kwargs)

    # ========================================================================
    # The null test: a capture compared against itself must be exactly zero.
    # ========================================================================

    def test_identical_captures_have_zero_difference(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.clone())
        report = self.run_compare(ref, cand)

        self.assertEqual(report["verdict"], compare_parity.PASS)
        self.assertEqual(report["probes"][0]["logits"]["max_abs"], 0.0)
        self.assertEqual(report["probes"][0]["logits"]["mse"], 0.0)
        self.assertEqual(report["probes"][0]["modules"][0]["max_abs"], 0.0)
        self.assertEmpty(report["failures"])

    # ========================================================================
    # Tolerance
    # ========================================================================

    def test_small_difference_passes(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits + 1e-3)
        report = self.run_compare(ref, cand, tolerance=1e-2)
        self.assertEqual(report["verdict"], compare_parity.PASS)

    def test_difference_above_tolerance_fails(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits + 0.5)
        report = self.run_compare(ref, cand, tolerance=1e-2)
        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertRegex(report["failures"][0], "max_abs")

    def test_module_difference_is_reported_per_module(self):
        logits = confident_logits()
        module = torch.zeros((1, 4, VOCAB))
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=module,
                                    cand_module=module + 1.0)
        report = self.run_compare(ref, cand, tolerance=1e-2)

        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertEqual(report["probes"][0]["modules"][0]["name"],
                         "layers.0.mixer")
        self.assertEqual(report["probes"][0]["modules"][0]["max_abs"], 1.0)
        self.assertIn("layers.0.mixer", report["failures"][0])

    def test_relative_error_catches_what_absolute_misses(self):
        # A proportional error on a small-magnitude tensor. This is the case
        # an absolute-only bound is blind to: 5% of 0.02 is 0.001, which
        # clears any absolute tolerance wide enough for logits.
        logits = confident_logits()
        small = torch.full((1, 4, VOCAB), 0.02)
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=small,
                                    cand_module=small * 1.05)
        report = self.run_compare(ref,
                                  cand,
                                  tolerance=5e-2,
                                  rel_tolerance=2e-2)

        module = report["probes"][0]["modules"][0]
        self.assertLess(module["max_abs"], 5e-2)  # absolute bound unbroken
        self.assertGreater(module["max_rel"], 2e-2)  # relative bound broken
        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertIn("max_rel", report["failures"][0])

    def test_relative_error_is_scaled_by_reference_magnitude(self):
        logits = confident_logits()
        big = torch.full((1, 4, VOCAB), 100.0)
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=big,
                                    cand_module=big + 1.0)
        module = self.run_compare(ref, cand,
                                  tolerance=10.0)["probes"][0]["modules"][0]
        self.assertAlmostEqual(module["max_abs"], 1.0, places=4)
        self.assertAlmostEqual(module["max_rel"], 0.01, places=4)
        self.assertAlmostEqual(module["reference_scale"], 100.0, places=3)

    def test_zero_reference_tensor_does_not_divide_by_zero(self):
        logits = confident_logits()
        zeros = torch.zeros((1, 4, VOCAB))
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=zeros,
                                    cand_module=zeros + 1e-3)
        report = self.run_compare(ref, cand)
        module = report["probes"][0]["modules"][0]
        self.assertTrue(math.isfinite(module["max_rel"]))
        # A reference of exactly zero and a candidate that is not cannot be
        # called a match at any relative bound.
        self.assertEqual(report["verdict"], compare_parity.FAIL)

    def test_bf16_rounding_stays_under_the_relative_bound(self):
        # bf16 keeps ~8 mantissa bits, so a pure cast lands near 0.4%. The
        # default relative bound has to sit above that or every clean run
        # fails.
        values = torch.full((1, 4, VOCAB), 0.02)
        logits = confident_logits()
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=values,
                                    cand_module=values.to(
                                        torch.bfloat16).float())
        module = self.run_compare(ref, cand)["probes"][0]["modules"][0]
        self.assertLess(module["max_rel"],
                        compare_parity.DEFAULT_REL_TOLERANCE)

    # ========================================================================
    # Token agreement: a flip only counts when the reference was confident.
    # ========================================================================

    def test_tie_flip_is_not_a_failure(self):
        # Two positions, top-2 separated by a hair. Nudging the runner-up past
        # the leader flips the argmax without meaning anything.
        logits = torch.zeros((1, 2, VOCAB))
        logits[0, 0, 0] = 5.0
        logits[0, 0, 1] = 5.0 - 1e-4
        logits[0, 1, 2] = 3.0
        logits[0, 1, 3] = 3.0 - 1e-4
        candidate = logits.clone()
        candidate[0, 0, 1] += 2e-4  # index 1 now wins position 0
        ref, cand = self.write_pair(logits, candidate)

        report = self.run_compare(ref, cand, tolerance=1e-2)
        tokens = report["probes"][0]["tokens"]
        self.assertEqual(tokens["agree"], 1)
        self.assertEqual(tokens["real_disagreements"], 0)
        self.assertTrue(tokens["disagreements"][0]["explained_by_rounding"])
        self.assertEqual(report["verdict"], compare_parity.PASS)

    def test_confident_flip_is_a_failure(self):
        # The reference is sure: top-1 beats top-2 by 5.0. A flip here is the
        # rewrite being wrong, not rounding.
        logits = torch.zeros((1, 2, VOCAB))
        logits[0, 0, 0] = 10.0
        logits[0, 0, 1] = 5.0
        logits[0, 1, 2] = 3.0
        candidate = logits.clone()
        candidate[0, 0, 1] = 11.0
        ref, cand = self.write_pair(logits, candidate)

        report = self.run_compare(ref, cand, tolerance=5e-2)
        tokens = report["probes"][0]["tokens"]
        self.assertEqual(tokens["real_disagreements"], 1)
        self.assertFalse(tokens["disagreements"][0]["explained_by_rounding"])
        self.assertEqual(tokens["disagreements"][0]["reference_margin"], 5.0)
        self.assertEqual(report["verdict"], compare_parity.FAIL)

    def test_tie_gap_floor_applies_when_runs_are_identical(self):
        # max_abs is 0, so without the floor the tie gap would be 0 and any
        # flip would read as real. Identical runs cannot flip, so this checks
        # the floor is carried into the report.
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.clone())
        report = self.run_compare(ref, cand, min_tie_gap=0.25)
        self.assertEqual(report["probes"][0]["tokens"]["tie_gap"], 0.25)

    def test_absolute_bound_is_off_by_default(self):
        # Tensor scales vary by 40x between layers of one real model, so no
        # constant is meaningful. A large absolute difference on a
        # large-magnitude tensor must not fail on its own.
        logits = confident_logits()
        big = torch.full((1, 4, VOCAB), 300.0)
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=big,
                                    cand_module=big + 5.0)
        report = self.run_compare(ref, cand)
        self.assertIsNone(report["tolerance"])
        self.assertEqual(report["probes"][0]["modules"][0]["max_abs"], 5.0)
        self.assertEqual(report["verdict"], compare_parity.PASS)

    def test_absolute_bound_applies_when_asked_for(self):
        logits = confident_logits()
        big = torch.full((1, 4, VOCAB), 300.0)
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_module=big,
                                    cand_module=big + 5.0)
        report = self.run_compare(ref, cand, tolerance=1.0)
        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertIn("max_abs", report["failures"][0])

    def test_tie_gap_uses_rms_not_max(self):
        # One element wrong by 3.0 flips a margin of 2.0. A max-based
        # threshold would be 2*3.0 = 6.0 and excuse the very flip that
        # error caused; the RMS-based one is 1.5 and catches it.
        logits = torch.zeros((1, 2, VOCAB))
        logits[0, 0, 0] = 10.0
        logits[0, 0, 1] = 8.0  # margin 2.0
        logits[0, 1, 2] = 3.0
        candidate = logits.clone()
        candidate[0, 0, 1] = 11.0
        ref, cand = self.write_pair(logits, candidate)

        report = self.run_compare(ref, cand)
        tokens = report["probes"][0]["tokens"]
        self.assertEqual(report["probes"][0]["logits"]["max_abs"], 3.0)
        self.assertLess(tokens["tie_gap"], 2.0)
        self.assertEqual(tokens["real_disagreements"], 1)
        self.assertEqual(report["verdict"], compare_parity.FAIL)

    def test_rms_is_reported_for_every_tensor(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits + 1e-3)
        probe = self.run_compare(ref, cand)["probes"][0]
        self.assertAlmostEqual(probe["logits"]["rms"], 1e-3, places=6)
        self.assertIn("rms", probe["modules"][0])

    def test_matching_dtypes_report_no_confound(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_overrides={"dtype": "bfloat16"})
        report = self.run_compare(ref, cand)
        self.assertFalse(report["dtype_confound"])

    def test_mismatched_dtypes_are_flagged_as_a_confound(self):
        # The default fixture runs the reference in float32 and the
        # candidate in bfloat16, which is exactly the case this warns about.
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.clone())
        report = self.run_compare(ref, cand)
        self.assertTrue(report["dtype_confound"])
        self.assertIn("precision", report["dtype_confound_note"])
        self.assertIn("NOTE", compare_parity.format_report(report))

    def test_dtype_confound_does_not_by_itself_fail(self):
        # It is a caveat on the numbers, not a defect in the candidate.
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.clone())
        report = self.run_compare(ref, cand)
        self.assertTrue(report["dtype_confound"])
        self.assertEqual(report["verdict"], compare_parity.PASS)

    # ========================================================================
    # Guards against meaningless comparisons
    # ========================================================================

    def test_different_probes_are_rejected(self):
        logits = confident_logits()
        ref, cand = self.write_pair(
            logits,
            logits.clone(),
            cand_overrides={"probe_sha256": "different"})
        with self.assertRaisesRegex(compare_parity.CaptureError,
                                    "different inputs"):
            self.run_compare(ref, cand)

    def test_different_models_are_rejected(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    cand_overrides={"model_id": "other/model"})
        with self.assertRaisesRegex(compare_parity.CaptureError,
                                    "model mismatch"):
            self.run_compare(ref, cand)

    def test_different_hooked_modules_are_rejected(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    cand_overrides={"modules": ["layers.1"]})
        with self.assertRaisesRegex(compare_parity.CaptureError,
                                    "different modules"):
            self.run_compare(ref, cand)

    def test_swapped_roles_are_rejected(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits,
                                    logits.clone(),
                                    ref_overrides={"role": "candidate"})
        with self.assertRaisesRegex(compare_parity.CaptureError,
                                    "expected role 'reference'"):
            self.run_compare(ref, cand)

    def test_missing_tensor_is_rejected(self):
        ref = self.write_capture("reference", make_manifest("reference"),
                                 {"p0/logits": confident_logits()})
        cand = self.write_capture("candidate", make_manifest("candidate"),
                                  {"p0/logits": confident_logits()})
        with self.assertRaisesRegex(compare_parity.CaptureError, "missing"):
            self.run_compare(ref, cand)

    def test_missing_capture_file_is_rejected(self):
        with self.assertRaisesRegex(compare_parity.CaptureError, "not found"):
            compare_parity.load_capture(os.path.join(self.tmp, "nope"))

    # ========================================================================
    # Shape mismatch is a failure, never a silent crop.
    # ========================================================================

    def test_shape_mismatch_fails_without_cropping(self):
        ref, cand = self.write_pair(confident_logits(positions=4),
                                    confident_logits(positions=3))
        report = self.run_compare(ref, cand)
        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertTrue(report["probes"][0]["logits"]["shape_mismatch"])
        self.assertIn("shape", report["failures"][0])

    # ========================================================================
    # A non-independent reference can never report PASS.
    # ========================================================================

    def test_non_independent_reference_is_unverified(self):
        logits = confident_logits()
        ref, cand = self.write_pair(
            logits,
            logits.clone(),
            ref_overrides={"independent_reference": False})
        report = self.run_compare(ref, cand)
        self.assertEqual(report["verdict"], compare_parity.UNVERIFIED)
        self.assertIn("rewrite", report["verdict_reason"])

    def test_non_independent_reference_outranks_failures(self):
        logits = confident_logits()
        ref, cand = self.write_pair(
            logits,
            logits + 5.0,
            ref_overrides={"independent_reference": False})
        report = self.run_compare(ref, cand, tolerance=1e-3)
        self.assertEqual(report["verdict"], compare_parity.UNVERIFIED)
        self.assertNotEmpty(report["failures"])

    # ========================================================================
    # bf16 handling
    # ========================================================================

    def test_bf16_candidate_upcasts_before_diffing(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.to(torch.bfloat16))
        report = self.run_compare(ref, cand, tolerance=1e-1)
        result = report["probes"][0]["logits"]

        self.assertEqual(result["reference_dtype"], "float32")
        self.assertEqual(result["candidate_dtype"], "bfloat16")
        # Non-zero (bf16 rounds) but small. If the subtraction happened in
        # bf16 the error would be inflated by an extra rounding step.
        self.assertGreater(result["max_abs"], 0.0)
        self.assertLess(result["max_abs"], 1e-1)
        self.assertEqual(report["verdict"], compare_parity.PASS)

    # ========================================================================
    # CLI
    # ========================================================================

    def test_main_returns_zero_on_pass_and_writes_json(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits.clone())
        out = os.path.join(self.tmp, "report.json")
        code = compare_parity.main(
            ["--reference", ref, "--candidate", cand, "--out", out])

        self.assertEqual(code, 0)
        with open(out) as f:
            self.assertEqual(json.load(f)["verdict"], compare_parity.PASS)

    def test_main_returns_one_on_fail(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits + 5.0)
        code = compare_parity.main(
            ["--reference", ref, "--candidate", cand, "--tolerance", "0.001"])
        self.assertEqual(code, 1)

    def test_main_returns_two_on_bad_input(self):
        logits = confident_logits()
        ref, cand = self.write_pair(
            logits,
            logits.clone(),
            cand_overrides={"probe_sha256": "different"})
        code = compare_parity.main(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 2)

    def test_format_report_renders_verdict_and_failures(self):
        logits = confident_logits()
        ref, cand = self.write_pair(logits, logits + 5.0)
        text = compare_parity.format_report(
            self.run_compare(ref, cand, tolerance=1e-3))
        self.assertIn("FAIL", text)
        self.assertIn("max_abs", text)
        self.assertIn("layers.0.mixer", text)


if __name__ == "__main__":
    absltest.main()
