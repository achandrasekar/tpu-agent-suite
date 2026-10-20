#!/usr/bin/env python3
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
"""Internal-consistency checks for the sharding search.

    python3 search/selftest.py

**Run this before trusting any output.** A curve built on an enumeration error looks
exactly like a curve.

Scope, and why it is smaller than it was
========================================
This file used to be 741 lines because the cost model lived here. It no longer does:
the step arithmetic is `../roofline-analysis/roofline.py --selftest` (23 bars) and
the scheduler model is `.../serving.py --selftest` (17 bars, including Little's Law
against an independent forward derivation and the PP round trip that caught a 13.7x
error), and every worked-example step is `derivation.py --selftest`. **Run all of
them.** What is left here is what this skill actually owns: enumeration, legality, the
funnel, the grouping, the report, and the seam to the roofline -- including a bar that
fails if roofline arithmetic creeps back into this directory.

If you add a bar, break the thing it guards on purpose first and
confirm it refuses. A bar that cannot fail is not a bar, and it is worse than none
because it is counted.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.normpath(os.path.join(_HERE, "..", "..", "roofline-analysis")))

import report as RP  # noqa: E402
import roofline as R  # noqa: E402
import space as SP  # noqa: E402
import sweep as SW  # noqa: E402

PRESET = os.path.join(_HERE, "..", "presets", "GLM-5.2-FP8-2x2x2-8k1k.json")
GLM = SP.ModelFacts("GLM-5.2-FP8",
                    is_moe=True,
                    num_routed_experts=256,
                    top_k=8,
                    num_kv_heads=1,
                    use_mla=True)
DENSE = SP.ModelFacts("dense-gqa8", is_moe=False, num_kv_heads=8)


class Topologies(unittest.TestCase):
    """Any slice, not only the two predefined in `space.TOPOLOGIES`."""

    def test_devices_and_hosts_are_derived_not_tabulated(self):
        for spec, chips, dev, hosts in (("2x2x1", 4, 8, 1),
                                        ("2x2x2", 8, 16, 2), ("2x2x4", 16, 32,
                                                              4), ("4x4x4", 64,
                                                                   128, 16)):
            t = SP.Topology.parse(spec)
            self.assertEqual((t.chips, t.devices, t.hosts),
                             (chips, dev, hosts), spec)

    def test_a_chip_is_two_devices_and_the_factor_is_not_lost(self):
        self.assertEqual(SP.Topology.parse("2x2x2", d2d=False).devices, 8)
        self.assertEqual(SP.Topology.parse("2x2x2", d2d=True).devices, 16)

    def test_the_legal_set_is_a_function_of_the_architecture_not_just_the_slice(
            self):
        """A count quoted for one model does not transfer to another."""
        t = SP.Topology.parse("2x2x2")
        moe = len(SP.legal_shardings(t, GLM))
        dense = len(SP.legal_shardings(t, DENSE))
        # Pinned; EP doubles much of the space. Went 30 -> 32 on 2026-09-23, when PP
        # stopped being illegal: the two new rows are `tp1-dp1-pp16` with and without
        # EP, and nothing was removed. Only *one* PP degree is ever legal on a given
        # slice -- pure PP forces tp=dp=pcp=1, so pp is pinned to the device count.
        self.assertEqual(moe, 32)
        self.assertLess(dense, moe)


class Enumeration(unittest.TestCase):

    def test_every_enumerated_sharding_multiplies_out_or_says_why(self):
        t = SP.Topology.parse("2x2x4")
        for s, v in SP.enumerate_shardings(t, GLM, include_pp=True):
            if s.world != t.devices:
                self.assertEqual(v.blocking_class, SP.ARITHMETIC)

    def test_a_blocked_sharding_is_attributed_to_the_hardest_rule(self):
        """Not the first `if` that fired -- otherwise the funnel depends on source
        order. `dcp=4` under `pcp=8` is ARITHMETIC (dcp must be in {1, 8}); the same
        row also trips several PCP INFRA rules. Arithmetic wins: no feature flag
        conjures ranks that do not exist."""
        v = SP.classify(SP.Sharding(tp=1, dp=1, pcp=8, pp=2, dcp=4),
                        SP.Topology.parse("2x2x2"), GLM)
        self.assertIn(SP.ARITHMETIC, v.classes)
        self.assertIn(SP.INFRA, v.classes)
        self.assertEqual(v.blocking_class, SP.ARITHMETIC)

    def test_arithmetic_is_never_costable(self):
        for s, v in SP.enumerate_shardings(SP.Topology.parse("2x2x2"),
                                           GLM,
                                           include_pp=True):
            if SP.ARITHMETIC in v.classes:
                self.assertFalse(v.costable)

    def test_mixed_pp_is_blocked_and_still_costable(self):
        """The wide view's whole point: an exclusion with no number on it cannot be
        weighed against the cost of removing it.

        `pp>1` alongside `dp>1` is the weakest rule in CONSTRAINTS.md -- reported, not
        read out of `pp_validation.py` -- so it is exactly the one whose price the
        caller needs, to decide whether measuring it is worth a job."""
        v = SP.classify(SP.Sharding(tp=1, dp=8, pp=2, ep=True),
                        SP.Topology.parse("2x2x2"), GLM)
        self.assertEqual(v.blocking_class, SP.INFRA)
        self.assertTrue(v.costable)

    def test_pp_with_tp_inside_a_stage_is_refused(self):
        """`pp_validation.py:48-53`. The stage's TP collectives stall on the first
        serving step; warmup, where every stage runs them, passes -- so this cannot
        be left to the benchmark to discover."""
        v = SP.classify(SP.Sharding(tp=2, dp=1, pp=8, ep=True),
                        SP.Topology.parse("2x2x2"), GLM)
        self.assertEqual(v.blocking_class, SP.INFRA)

    def test_pure_pp_is_legal_and_emits_no_async_scheduling(self):
        """Reversed 2026-09-23: PP used to be hard illegal on tpu7x. It serves, in
        the pure-PP shape, measured at `j-78adee7b-af0e-43cc-9ee0`. The flag matters
        as much as the legality -- `pp_validation.py:54-58` refuses async scheduling
        under PP, and a missing `--no-async-scheduling` kills the job at startup."""
        v = SP.classify(SP.Sharding(tp=1, dp=1, pp=16, ep=True),
                        SP.Topology.parse("2x2x2"), GLM)
        self.assertEqual(v.status, SP.LEGAL, v.reasons)
        flags = SP.ServerConfig(SP.Sharding(tp=1, dp=1, pp=16)).flags()
        self.assertIn("--pipeline-parallel-size=16", flags)
        self.assertIn("--no-async-scheduling", flags)

    def test_a_stage_cannot_hold_zero_layers(self):
        """`get_pp_indices` needs at least one layer per stage. GLM-5.2 has 78, so
        this only bites on a slice wider than the layer stack -- which `2x2x32` is
        not, but a hypothetical model with 8 layers on 16 chips is."""
        tiny = SP.ModelFacts("tiny",
                             num_kv_heads=1,
                             use_mla=True,
                             num_layers=8)
        v = SP.classify(SP.Sharding(tp=1, dp=1, pp=16),
                        SP.Topology.parse("2x2x2"), tiny)
        self.assertEqual(v.blocking_class, SP.ARITHMETIC)


class Seam(unittest.TestCase):
    """This skill must not restate anything the roofline config already pins."""

    def test_model_facts_are_derived_from_the_roofline_config(self):
        cfg = R.load(SW.load_spec(PRESET).roofline_config)
        f = SW.model_facts(cfg)
        self.assertEqual(f.name, cfg.model.name)
        self.assertEqual(f.num_routed_experts, cfg.model.n_experts)
        self.assertEqual(f.top_k, cfg.model.topk)
        self.assertEqual(f.use_mla, cfg.model.use_mla)

    def test_the_preset_restates_no_architecture_pin(self):
        """If a checkpoint number appears in a preset it is a second authority, and
        the stale copy is indistinguishable from the fresh one by anything but a diff
        nobody runs."""
        import json
        raw = json.load(open(PRESET))
        for k in ("model", "numerics", "hardware", "layers", "hidden",
                  "n_experts"):
            self.assertNotIn(
                k, raw, f"preset restates {k!r}; it belongs to the "
                f"roofline config")

    def test_the_two_sharding_dataclasses_agree(self):
        for s in SP.legal_shardings(SP.Topology.parse("2x2x2"), GLM):
            r = SW.as_roofline_sharding(s)
            self.assertEqual(r.label.replace("-ep", ""),
                             s.label.replace("-ep", ""))
            self.assertEqual(r.ep_group() if s.ep else 1, s.ep_group)

    def test_the_cell_overrides_the_slice_and_says_so(self):
        """The cell owns the slice -- that is what makes "same checkpoint, other
        slice" one preset rather than a forked roofline config. An override that
        reshaped the hardware silently would be found three reports later."""
        import contextlib
        import io
        spec = replace(SW.load_spec(PRESET),
                       topology=SP.Topology.parse("2x2x1"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = SW.run(spec)
        self.assertIn("overrides the roofline config", buf.getvalue())
        self.assertEqual(res.base.hardware.world, 8)
        # and only the slice moved
        orig = R.load(spec.roofline_config)
        self.assertEqual(res.base.model, orig.model)
        self.assertEqual(res.base.numerics, orig.numerics)
        self.assertEqual(res.base.hardware.hbm_capacity,
                         orig.hardware.hbm_capacity)


class TheSweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = SW.load_spec(PRESET)
        cls.res = SW.run(cls.spec)

    def test_a_list_in_the_preset_is_refused_not_silently_truncated(self):
        """Concurrency and mnbt sweeps are not implemented. Taking the first element of
        a list would answer a question the preset did not ask."""
        import json
        import tempfile
        raw = json.load(open(PRESET))
        raw["concurrency"] = [16, 256]
        raw["roofline_config"] = os.path.normpath(
            os.path.join(os.path.dirname(PRESET), raw["roofline_config"]))
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(raw, fh)
        with self.assertRaises(SystemExit):
            SW.load_spec(fh.name)
        os.unlink(fh.name)

    def test_every_sharding_is_at_the_same_operating_point(self):
        """What makes the combined chart like-for-like."""
        for r in self.res.rows:
            self.assertEqual(r.point.concurrency, self.spec.concurrency)

    def test_the_funnel_accounts_for_every_enumerated_sharding(self):
        """Every stage's count equals the previous minus what it removed. A funnel
        whose arithmetic does not close is an accounting of exclusions hiding one."""
        rows = self.res.funnel
        self.assertEqual(rows[0]["count"], len(self.res.enumerated))
        for prev, cur in zip(rows, rows[1:]):
            self.assertEqual(cur["count"], prev["count"] - cur["removed"],
                             cur["stage"])

    def test_the_infra_class_is_priced(self):
        """Projections are the forward-looking half of the report. `space.py` marks a
        PCP+DP row costable and `Sharding.check` refused it anyway once -- the class was
        enumerated, counted, and never given a number."""
        pcp = [
            r for r in self.res.rows
            if r.sharding.pcp > 1 and r.sharding.dp > 1
        ]
        self.assertTrue(pcp)
        self.assertTrue(any(r.point.feasible for r in pcp))

    def test_memory_blocked_shardings_are_refused_with_numbers_not_costed(
            self):
        row = next(r for r in self.res.rows if r.sharding.label == "tp1-dp16")
        self.assertFalse(row.point.feasible)
        self.assertIn("no KV pool", row.point.reason)
        self.assertRegex(row.point.reason, r"\d")  # says by how much

    def test_a_blocked_row_is_never_shippable(self):
        for r in self.res.rows:
            if r.cls != SP.SHIPPABLE:
                self.assertFalse(r.shippable, r.sharding.label)


class TheReport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.spec = SW.load_spec(PRESET)
        cls.res = SW.run(cls.spec)
        cls.out = tempfile.mkdtemp()
        SW.write_csvs(cls.res, cls.out)
        cls.text = RP.render(cls.res, cls.out, None, None, "cmd")
        cls.fams = RP.group(cls.res)

    def test_ep_and_dcp_are_variants_not_combinations(self):
        """They consume no ranks, so `tp2-dp8` and `tp2-dp8-dcp2-ep` are one
        combination. A family key that includes them splits every section in four."""
        a = SP.Sharding(tp=2, dp=8)
        b = SP.Sharding(tp=2, dp=8, dcp=2, ep=True)
        self.assertEqual(a.family, b.family)
        self.assertEqual(a.family, "TP×DP")
        self.assertNotEqual(
            SP.Sharding(tp=2, dp=8).family,
            SP.Sharding(tp=16).family)

    def test_every_coherent_sharding_lands_in_exactly_one_combination(self):
        n = sum(len(v) for v in self.fams.values())
        self.assertEqual(n, len(self.res.rows))

    def test_every_sharding_with_an_operating_point_appears_in_the_report(
            self):
        """Shippable and projection alike -- a row the report does not show is a row
        nobody can ask about."""
        for r in self.res.rows:
            if r.point.feasible:
                self.assertIn(f"`{r.sharding.label}`", self.text,
                              r.sharding.label)

    def test_every_rejection_appears_with_its_reason(self):
        for r in self.res.rows:
            if not r.point.feasible:
                self.assertIn(f"`{r.sharding.label}`", self.text,
                              r.sharding.label)
        for s, v in self.res.enumerated:
            if v.blocking_class == SP.ARITHMETIC:
                self.assertIn(v.reason_for(SP.ARITHMETIC), self.text)

    def test_projection_only_combinations_are_marked_as_projections(self):
        for fam, rows in self.fams.items():
            if any(r.point.feasible for r in rows) and not RP.ship(rows):
                self.assertIn(f". {fam} (projection)", self.text, fam)

    def test_every_worked_example_reconciles_with_the_sweep(self):
        """The report's own self-check line. A ✗ anywhere means the derivation and the
        sweep disagree, and the report says so rather than printing either."""
        self.assertIn("every step that the sweep also returns agrees with it",
                      self.text)
        self.assertNotIn(" ✗**", self.text)  # the mismatch cell, not the prose

    def test_no_unfilled_template_field(self):
        for tok in ("<sharding>", "<N>", "<out>", "None ms", "nan", "inf "):
            self.assertNotIn(tok, self.text)


class NoRooflineHere(unittest.TestCase):
    """This skill enumerates, calls, groups and renders. Arithmetic on shapes, bytes or
    times belongs to `roofline-analysis`; a second copy here is a second provenance,
    and the one that drifts is the one nobody diffs."""

    #: What this directory may touch in the roofline modules: loading a config, the
    #: two value types, the byte unit for display, and the one costing entry point.
    ALLOWED = {
        "R": {"load", "Config", "Sharding", "GiB", "WeightTerm", "KVTerm"},
        "SV":
        {"operating_point", "Point", "Workload", "Knobs", "DEFAULT_MODEL"},
        "DV": {"derive", "Derivation", "PhaseDetail", "Step"},
    }

    def test_no_file_here_reaches_into_the_cost_model(self):
        import re
        bad = []
        for name in sorted(os.listdir(_HERE)):
            if not name.endswith(".py") or name == "selftest.py":
                continue
            src = open(os.path.join(_HERE, name)).read()
            for mod, ok in self.ALLOWED.items():
                for attr in re.findall(rf"\b{mod}\.([A-Za-z_]\w*)", src):
                    if attr not in ok:
                        bad.append(f"{name}: {mod}.{attr}")
        self.assertFalse(
            bad, "roofline arithmetic reached from sharding-strategy: " +
            ", ".join(sorted(set(bad))))


if __name__ == "__main__":
    unittest.main(verbosity=1)
