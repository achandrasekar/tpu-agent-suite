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
"""Compares a reference capture against a candidate capture.

The reference is the model's own plain-PyTorch path run on CPU in fp32. The
candidate is the unwired code run on TPU. Both captures are produced by
`capture_parity.py`, which feeds each side the same fixed token sequence
(teacher forcing) so every position is compared against the same inputs. This
script does the arithmetic only: it loads no model and touches no accelerator.

Verdicts:
  PASS        the candidate matches the reference within tolerance.
  FAIL        it does not, and the report names where.
  UNVERIFIED  the reference was not independent of the candidate (no built-in
              plain-PyTorch path existed, so both sides ran the same rewritten
              code). The numbers cannot catch an error in the rewrite itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

# A top-1 disagreement is attributed to rounding when the reference's own
# margin between its best and second-best logit is under this many multiples
# of the logit noise. Two logits drifting in opposite directions can bridge
# roughly twice the typical difference, hence 2. The noise is measured as
# RMS, not max: one badly wrong element must not inflate the threshold that
# is supposed to catch it. Measured on mamba-130m, a clean bf16 run showed
# max_abs 2.34 against RMS 0.73 on the same tensor.
TIE_GAP_SLACK = 2.0

# Floor for the tie-gap threshold, used when the two runs agree so closely
# that the observed difference is ~0 and would otherwise call every
# disagreement a real one.
DEFAULT_MIN_TIE_GAP = 1e-3

# The absolute bound is off unless asked for. There is no defensible
# constant: measured on mamba-130m, tensor scales ran from 8.1 at layer 0 to
# 341.9 at layer 23 *within one model*, so any fixed value is simultaneously
# absurd for one layer and vacuous for another. Pass --tolerance when a
# particular deployment has a number it wants held.
DEFAULT_TOLERANCE = None

# The relative bound, calibrated against measurements on real hardware.
# mamba-130m, CPU bf16 reference against TPU v6e bf16 candidate, identical
# code on both sides: device noise alone reached 2.92% at layer 23 and 3.09%
# on the logits. The same run with a deliberate 10% error injected into one
# layer showed 10.85% at that layer. 5% clears the noise with margin and
# catches the bug with margin.
#
# The floor this implies is real: a rewrite wrong by less than roughly 3%
# is indistinguishable from device noise at this probe size, and no
# threshold fixes that. Widen the probe set or compare more modules.
DEFAULT_REL_TOLERANCE = 0.05

# Guards the relative denominator when a reference tensor is all zeros.
REL_DENOM_FLOOR = 1e-6

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"


class CaptureError(Exception):
    """Raised when the two captures cannot be meaningfully compared."""


def load_capture(
        prefix: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Loads the manifest and tensors written under a capture prefix.

  Args:
    prefix: Path prefix shared by `<prefix>.json` and `<prefix>.safetensors`.

  Returns:
    A tuple of the parsed manifest and the tensor mapping.

  Raises:
    CaptureError: If either file is missing or the manifest is not valid JSON.
  """
    manifest_path = Path(f"{prefix}.json")
    tensors_path = Path(f"{prefix}.safetensors")
    for path in (manifest_path, tensors_path):
        if not path.is_file():
            raise CaptureError(f"capture file not found: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CaptureError(f"{manifest_path} is not valid JSON: {e}") from e
    return manifest, load_file(str(tensors_path))


def check_comparable(ref: dict[str, Any], cand: dict[str, Any]) -> None:
    """Rejects capture pairs whose numbers would not mean anything together.

  Args:
    ref: Reference manifest.
    cand: Candidate manifest.

  Raises:
    CaptureError: If the roles, inputs, model or hooked modules disagree.
  """
    if ref.get("role") != "reference":
        raise CaptureError(
            f"expected role 'reference', got {ref.get('role')!r}")
    if cand.get("role") != "candidate":
        raise CaptureError(
            f"expected role 'candidate', got {cand.get('role')!r}")

    # Different inputs make every downstream number meaningless, so this is a
    # hard stop rather than a warning.
    if ref.get("probe_sha256") != cand.get("probe_sha256"):
        raise CaptureError(
            "the two runs used different inputs "
            f"(reference probe {ref.get('probe_sha256')}, "
            f"candidate probe {cand.get('probe_sha256')}); re-run one side "
            "with the same probe file")

    if ref.get("model_id") != cand.get("model_id"):
        raise CaptureError(f"model mismatch: {ref.get('model_id')!r} vs "
                           f"{cand.get('model_id')!r}")

    ref_modules = list(ref.get("modules", []))
    cand_modules = list(cand.get("modules", []))
    if ref_modules != cand_modules:
        only_ref = sorted(set(ref_modules) - set(cand_modules))
        only_cand = sorted(set(cand_modules) - set(ref_modules))
        raise CaptureError(
            "the two runs hooked different modules "
            f"(reference only: {only_ref}, candidate only: {only_cand})")


def _as_f32(tensor: torch.Tensor) -> torch.Tensor:
    """Upcasts to fp32 so subtracting two bf16 tensors adds no rounding."""
    return tensor.detach().to(torch.float32)


def tensor_diff(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    """Measures how far a candidate tensor sits from its reference.

  Shape mismatches are reported as such and never silently cropped: a
  truncating comparison turns a real divergence into a pass.

  Args:
    ref: Reference tensor.
    cand: Candidate tensor.

  Returns:
    A dict carrying either `shape_mismatch` or the difference metrics.
  """
    if ref.shape != cand.shape:
        return {
            "shape_mismatch": True,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }
    ref_f32 = _as_f32(ref)
    delta = (ref_f32 - _as_f32(cand)).abs()
    max_abs = delta.max().item()
    # Normalized by the reference's own scale, so the number means the same
    # thing on logits of magnitude 10 and on activations of magnitude 0.02.
    scale = max(ref_f32.abs().max().item(), REL_DENOM_FLOOR)
    return {
        "shape_mismatch": False,
        "max_abs": max_abs,
        "max_rel": max_abs / scale,
        "reference_scale": scale,
        "rms": delta.pow(2).mean().sqrt().item(),
        "mse": delta.pow(2).mean().item(),
        "reference_dtype": str(ref.dtype).removeprefix("torch."),
        "candidate_dtype": str(cand.dtype).removeprefix("torch."),
    }


def top1_agreement(ref: torch.Tensor, cand: torch.Tensor,
                   tie_gap: float) -> dict[str, Any]:
    """Compares the argmax token at every position of a logits tensor.

  A disagreement only means something when the reference was confident. Where
  the reference's own top two logits sat closer together than `tie_gap`, a
  flip is rounding rather than broken math, so each disagreement is reported
  with the margin it overturned.

  Args:
    ref: Reference logits, `[..., vocab]`.
    cand: Candidate logits, same shape.
    tie_gap: Reference margin at or below which a flip counts as a tie.

  Returns:
    A dict with the agreement count and a record per disagreement.
  """
    ref_flat = _as_f32(ref).reshape(-1, ref.shape[-1])
    cand_flat = _as_f32(cand).reshape(-1, cand.shape[-1])
    positions = ref_flat.shape[0]

    ref_top2 = ref_flat.topk(2, dim=-1)
    ref_best = ref_top2.indices[:, 0]
    # How much the runner-up would have to gain to win this position.
    margins = ref_top2.values[:, 0] - ref_top2.values[:, 1]
    cand_best = cand_flat.argmax(dim=-1)

    disagreements = []
    real_disagreements = 0
    for index in (ref_best != cand_best).nonzero(as_tuple=True)[0].tolist():
        margin = margins[index].item()
        explained = margin <= tie_gap
        if not explained:
            real_disagreements += 1
        disagreements.append({
            "position": index,
            "reference_token": int(ref_best[index].item()),
            "candidate_token": int(cand_best[index].item()),
            "reference_margin": margin,
            "explained_by_rounding": explained,
        })

    return {
        "positions": positions,
        "agree": positions - len(disagreements),
        "tie_gap": tie_gap,
        "real_disagreements": real_disagreements,
        "disagreements": disagreements,
    }


def _probe_ids(manifest: dict[str, Any]) -> list[str]:
    return [probe["id"] for probe in manifest.get("probes", [])]


def _require(tensors: dict[str, torch.Tensor], key: str,
             role: str) -> torch.Tensor:
    if key not in tensors:
        raise CaptureError(f"{role} capture is missing tensor {key!r}")
    return tensors[key]


def tolerance_breaches(key: str, diff: dict[str, Any], tolerance: float | None,
                       rel_tolerance: float) -> list[str]:
    """Names every tolerance a tensor difference broke.

  The relative bound always applies; it is scale-free, and because the
  denominator is the tensor's own peak magnitude rather than each element,
  it does not blow up where the reference is near zero. The absolute bound
  is checked only when the caller supplies one.

  Args:
    key: Tensor key, for the message.
    diff: A `tensor_diff` result.
    tolerance: Absolute ceiling, or None to skip it.
    rel_tolerance: Relative ceiling.

  Returns:
    Human-readable failure strings, empty when the tensor is within both.
  """
    if diff["shape_mismatch"]:
        return [
            f"{key}: shape {diff['reference_shape']} vs "
            f"{diff['candidate_shape']}"
        ]
    breaches = []
    if tolerance is not None and diff["max_abs"] > tolerance:
        breaches.append(f"{key}: max_abs {diff['max_abs']:.4g} exceeds "
                        f"tolerance {tolerance:g}")
    if diff["max_rel"] > rel_tolerance:
        breaches.append(
            f"{key}: max_rel {diff['max_rel']:.2%} exceeds tolerance "
            f"{rel_tolerance:.2%} (reference scale "
            f"{diff['reference_scale']:.4g})")
    return breaches


def compare(
    ref_manifest: dict[str, Any],
    ref_tensors: dict[str, torch.Tensor],
    cand_manifest: dict[str, Any],
    cand_tensors: dict[str, torch.Tensor],
    tolerance: float | None = DEFAULT_TOLERANCE,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    min_tie_gap: float = DEFAULT_MIN_TIE_GAP,
) -> dict[str, Any]:
    """Builds the full comparison report for a capture pair.

  Args:
    ref_manifest: Reference manifest.
    ref_tensors: Reference tensors.
    cand_manifest: Candidate manifest.
    cand_tensors: Candidate tensors.
    tolerance: Ceiling on the largest absolute elementwise difference.
    rel_tolerance: Ceiling on that difference relative to the reference's
      own scale. Both must hold.
    min_tie_gap: Floor for the tie-gap threshold.

  Returns:
    The report dict, including the overall verdict.

  Raises:
    CaptureError: If the captures are not comparable or a tensor is missing.
  """
    check_comparable(ref_manifest, cand_manifest)

    report: dict[str, Any] = {
        "model_id": ref_manifest.get("model_id"),
        "reference": {
            "device": ref_manifest.get("device"),
            "dtype": ref_manifest.get("dtype"),
            "independent": bool(ref_manifest.get("independent_reference")),
        },
        "candidate": {
            "device": cand_manifest.get("device"),
            "dtype": cand_manifest.get("dtype"),
        },
        "tolerance": tolerance,
        "rel_tolerance": rel_tolerance,
        "probes": [],
        "failures": [],
    }

    # Two runs in different dtypes differ for reasons that have nothing to
    # do with the rewrite under test. Measured on mamba-130m, fp32 against
    # bf16 on the same device with the same code showed up to 2.45%. It is
    # not an error -- some models have no usable bf16 path on CPU -- but
    # every number below then carries it, so say so.
    ref_dtype = ref_manifest.get("dtype")
    cand_dtype = cand_manifest.get("dtype")
    report["dtype_confound"] = ref_dtype != cand_dtype
    if report["dtype_confound"]:
        report["dtype_confound_note"] = (
            f"reference ran in {ref_dtype} and candidate in {cand_dtype}, so "
            "the differences below include precision as well as the rewrite; "
            "re-run the reference in the candidate's dtype to isolate it")

    for probe_id in _probe_ids(ref_manifest):
        logits_key = f"{probe_id}/logits"
        ref_logits = _require(ref_tensors, logits_key, "reference")
        cand_logits = _require(cand_tensors, logits_key, "candidate")

        logits = tensor_diff(ref_logits, cand_logits)
        probe: dict[str, Any] = {"id": probe_id, "logits": logits}

        report["failures"].extend(
            tolerance_breaches(logits_key, logits, tolerance, rel_tolerance))
        if not logits["shape_mismatch"]:
            # Scale the tie threshold to the noise these two runs actually
            # showed, so the bar moves with the model instead of being pinned
            # to a constant that fits one of them. RMS rather than max_abs:
            # a localized error blows up the max and would otherwise inflate
            # the threshold meant to catch the flips it caused.
            tie_gap = max(TIE_GAP_SLACK * logits["rms"], min_tie_gap)
            tokens = top1_agreement(ref_logits, cand_logits, tie_gap)
            probe["tokens"] = tokens
            if tokens["real_disagreements"]:
                worst = max(
                    (d for d in tokens["disagreements"]
                     if not d["explained_by_rounding"]),
                    key=lambda d: d["reference_margin"],
                )
                report["failures"].append(
                    f"{probe_id}: {tokens['real_disagreements']} token "
                    f"disagreement(s) too large to be rounding (worst margin "
                    f"{worst['reference_margin']:.4g} at position "
                    f"{worst['position']})")

        modules = []
        for name in ref_manifest.get("modules", []):
            key = f"{probe_id}/module/{name}"
            diff = tensor_diff(_require(ref_tensors, key, "reference"),
                               _require(cand_tensors, key, "candidate"))
            diff["name"] = name
            modules.append(diff)
            report["failures"].extend(
                tolerance_breaches(key, diff, tolerance, rel_tolerance))
        probe["modules"] = modules
        report["probes"].append(probe)

    if not report["reference"]["independent"]:
        # Both sides ran the same rewritten code, so agreement here says only
        # that the port to TPU held -- not that the rewrite is faithful.
        report["verdict"] = UNVERIFIED
        report["verdict_reason"] = (
            "the model had no built-in plain-PyTorch path, so the reference "
            "ran the same rewritten code as the candidate; these numbers "
            "cannot detect an error in the rewrite itself")
    elif report["failures"]:
        report["verdict"] = FAIL
        report["verdict_reason"] = f"{len(report['failures'])} check(s) failed"
    else:
        report["verdict"] = PASS
        report["verdict_reason"] = "within tolerance at every compared point"
    return report


def format_report(report: dict[str, Any]) -> str:
    """Renders the report as the plain-text block printed to stdout."""
    lines = []
    if report.get("dtype_confound"):
        lines.append(f"NOTE: {report['dtype_confound_note']}")
        lines.append("")
    lines += [
        f"model     {report['model_id']}",
        f"reference {report['reference']['device']} "
        f"{report['reference']['dtype']}"
        f"{'' if report['reference']['independent'] else '  (NOT INDEPENDENT)'}",
        f"candidate {report['candidate']['device']} "
        f"{report['candidate']['dtype']}",
        "",
    ]
    for probe in report["probes"]:
        lines.append(f"probe {probe['id']}")
        logits = probe["logits"]
        if logits["shape_mismatch"]:
            lines.append(f"  logits       SHAPE MISMATCH "
                         f"{logits['reference_shape']} vs "
                         f"{logits['candidate_shape']}")
        else:
            lines.append(f"  logits       max_abs {logits['max_abs']:.4g}   "
                         f"max_rel {logits['max_rel']:.2%}   "
                         f"rms {logits['rms']:.3g}")
            tokens = probe["tokens"]
            lines.append(f"  top-1 agree  {tokens['agree']}/"
                         f"{tokens['positions']}")
            for dis in tokens["disagreements"]:
                verdict = ("rounding"
                           if dis["explained_by_rounding"] else "REAL")
                lines.append(f"    position {dis['position']}: margin "
                             f"{dis['reference_margin']:.4g}  -> {verdict}")
        for module in probe["modules"]:
            if module["shape_mismatch"]:
                lines.append(f"  {module['name']}  SHAPE MISMATCH "
                             f"{module['reference_shape']} vs "
                             f"{module['candidate_shape']}")
            else:
                lines.append(f"  {module['name']}  max_abs "
                             f"{module['max_abs']:.4g}   "
                             f"max_rel {module['max_rel']:.2%}   "
                             f"rms {module['rms']:.3g}")
        lines.append("")

    lines.append(f"{report['verdict']}: {report['verdict_reason']}")
    for failure in report["failures"]:
        lines.append(f"  - {failure}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare a reference capture against a candidate capture.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Prefix of the reference capture (without file extension).",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Prefix of the candidate capture (without file extension).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Optional ceiling on the largest absolute elementwise "
        "difference. Off by default: tensor scales vary too much between "
        "models, and between layers of one model, for a constant to mean "
        "anything.",
    )
    parser.add_argument(
        "--rel_tolerance",
        type=float,
        default=DEFAULT_REL_TOLERANCE,
        help="Largest difference relative to the reference tensor's own "
        "scale that is treated as a match. Calibrate from a known-good run "
        "for the model at hand.",
    )
    parser.add_argument(
        "--min_tie_gap",
        type=float,
        default=DEFAULT_MIN_TIE_GAP,
        help="Floor for the margin below which a token flip counts as a tie.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the report as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on PASS, 1 on FAIL/UNVERIFIED, 2 on bad input."""
    args = parse_args(argv)
    try:
        ref_manifest, ref_tensors = load_capture(args.reference)
        cand_manifest, cand_tensors = load_capture(args.candidate)
        report = compare(
            ref_manifest,
            ref_tensors,
            cand_manifest,
            cand_tensors,
            tolerance=args.tolerance,
            rel_tolerance=args.rel_tolerance,
            min_tie_gap=args.min_tie_gap,
        )
    except CaptureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2),
                                  encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0 if report["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
