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
"""Step model 1 — NATIVE. Prefill and decode as separate dedicated passes.

**The default.** This is `roofline.py`'s own composition, unchanged, wrapped in the
`step_api.Solution` contract so `serving.py` can drive it. Nothing here is a new
model: every formula below is `roofline.py`'s `run_phase` and `serving`, and
`selftest` pins them against that file rather than restating them.

    T_step     = run_phase(decode:  tokens=running, kv_len=ISL+OSL, batch=running)
    T_prefill  = run_phase(prefill: tokens=ISL, kv_len=ISL, chunked at mnbt)
    TPOT       = T_step
    TTFT       = T_prefill                      <- a floor: no queueing here
    QPS_pre    = 1 / T_prefill                  one request, run on its own
    QPS_dec    = running / (OSL * T_step)
    QPS        = 1 / (1/QPS_pre + 1/QPS_dec)    it must do both, serially
    tok/s      = QPS * OSL                      output tokens

What this model cannot see, by construction
===========================================
**Prefill and decode never share a token budget here**, because they are priced as
separate passes. So `max_num_batched_tokens` cannot touch the decode step at all --
`T_step` is exactly invariant to it -- and `prefill_schedule_interval` does nothing.

What mnbt *can* still do here is change how many chunks the dedicated prefill run is
split into, and that is not small: on the 8K/1K cell T_prefill is **1,816 ms at
mnbt 256 against 1,486 at 4,096**, worth 1.17x of throughput at saturation. Do not
quote README §2's **0.13%** as "mnbt does nothing under model 1" -- that figure is
the 1024..8192 range only, which is the flat part of the curve. **The low rungs are
the ones that discriminate.**

So the difference from model 2 is *where* the knob acts, not whether it acts: under
model 1 a large chunk costs decode latency nothing, because decode is priced alone.
Under model 2 every decode sharing that step waits behind it.

One divergence from `roofline.serving()`, deliberate
====================================================
That function reports `tok_s_fleet = per_core * hardware.world`, which treats every
core as an independent server. True at `dp16-tp1` (16 engines, and the cell it was
written for); wrong by `tp * pcp * pp` anywhere else, because TP ranks cooperate on
one engine rather than each serving requests. This wrapper scales by **`dp`**.
`roofline.py` is left as written -- `selftest` pins the divergence so it cannot drift
unnoticed.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R  # noqa: E402
from step_api import Solution  # noqa: E402

NAME = "model1"
MIXES = False


def _phase_ops(cfg: R.Config,
               ph: R.Phase) -> tuple[float, dict[str, float], str, float]:
    """One phase, through `roofline.run_phase`'s own arithmetic.

    Walks `roofline.chunks_of` / `build_ops` / `multiplicity` -- the same three
    functions `run_phase` walks, in the same order -- and sums them once, because
    `explain.py` needs the per-op breakdown and `PhaseResult.rows` is already
    aggregated. Calling `run_phase` as well and taking its total would build every
    op twice, which doubles a sweep.

    `selftest::test_the_op_breakdown_sums_to_run_phase` asserts this total equals
    `run_phase`'s to 9 places, so it cannot drift into being a second cost model.
    """
    hw = cfg.hardware
    ops: dict[str, float] = {}
    by: dict[str, float] = {}
    ici = total = 0.0
    stage = 1.0 / max(1, cfg.sharding.pp)
    for sub in R.chunks_of(ph):
        for op in R.build_ops(cfg, sub):
            mult = R.multiplicity(cfg.model, op.layer_class) * stage
            ms = op.roofline_ms(hw) * mult
            ops[op.name] = ops.get(op.name, 0.0) + ms
            by[op.bound(hw)] = by.get(op.bound(hw), 0.0) + ms
            ici += op.ici_ms * mult
            total += ms
    return total, ops, (max(by, key=lambda k: by[k]) if by else ""), ici


#: Memoised like model 2's, and for the same reason: once the `kv` or `slots` gate
#: binds, `running` stops moving with concurrency, so a sweep asks for the identical
#: point many times. It matters more here than there -- this model re-runs a CHUNKED
#: prefill pass (up to 32 sub-passes of ~30 ops) on every call, so an uncached sweep
#: over 78 shardings x 6 mnbt rungs x 26 concurrencies does not finish in a coffee
#: break. Keyed by VALUE; `id()` would be wrong because CPython reuses the id of a
#: collected object and a stale hit is a silently wrong number in a plausible table.
_CACHE: dict[tuple, Solution] = {}


def solve(cfg: R.Config, isl: int, osl: int, running: float, mnbt: int,
          **_ignored) -> Solution:
    """Price one config at an admitted running set. `**_ignored` absorbs the knobs
    this model has no term for -- see the module docstring; they are dropped loudly
    in the report rather than silently here."""
    s_, n = cfg.sharding, cfg.numerics
    key = (cfg.model, cfg.hardware, n.default, tuple(sorted(n.sites.items())),
           s_.tp, s_.dp, s_.pp, s_.pcp, s_.dcp, s_.ep, s_.sp, s_.pp_layers,
           isl, osl, mnbt, round(running, 6))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    full = isl + osl
    m = cfg.model
    d = max(1.0, running)
    stages = max(1, cfg.sharding.pp)
    # PP: up to `pp` batches in flight, and a request in flight cannot join another,
    # so the running set is DIVIDED between them. parallelism/pp/SKILL.md s6.
    v = float(max(1, min(stages, int(d)))) if stages > 1 else 1.0
    db = d / v  # decode tokens in ONE batch

    # decode: every running request advances one token, attending over FULL context
    dec_fetch = db * (min(full, m.indexer_topk) if m.indexer_topk else full)
    dec = R.Phase("decode",
                  tokens=db,
                  kv_len=full,
                  batch=max(1, int(round(db))),
                  kv_fetched=max(1.0, dec_fetch))
    t_step, ops, bound, ici = _phase_ops(cfg, dec)

    # prefill: one prompt, on its own, chunked at the token budget
    pre = R.Phase("prefill", tokens=isl, kv_len=isl, batch=1, mnbt=mnbt)
    if stages > 1:
        # its chunks pipeline across the real stages: one per wave at the slowest
        # stage, then the last drains. The engine is busy one wave per chunk.
        chunk_st = [
            R.stage_ms(cfg, c, bucket=mnbt or c.tokens)
            for c in R.chunks_of(pre)
        ]
        t_prefill = sum(max(c) for c in chunk_st)
        ttft = R.pp_ttft_ms(chunk_st)
    else:
        t_prefill, _, _, _ = _phase_ops(cfg, pre)
        ttft = t_prefill

    if t_step <= 0 or t_prefill <= 0:
        return Solution(model=NAME, mixes=MIXES)

    # `_phase_ops` divides layers by pp, so `t_step` and `ops` are the MEAN stage of
    # one batch -- kept for `explain.py`. The pipeline is priced from the real stages.
    if stages > 1:
        stage_ms = tuple(R.stage_ms(cfg, dec, bucket=mnbt or db))
        cycle = R.pp_cycle_ms(list(stage_ms), v)
    else:
        stage_ms, cycle = (t_step, ), t_step
    qp = 1.0 / (t_prefill / 1e3)  # req/s if it only prefilled
    qd = d / (osl * cycle / 1e3)  # req/s if it only decoded
    qc = 1.0 / (1.0 / qp + 1.0 / qd)
    out = Solution(
        decode_tokens=d,
        chunk=float(min(mnbt, isl)) if mnbt else float(isl),
        n_chunks=float(math.ceil(isl / mnbt)) if mnbt else 1.0,
        step_ms=t_step,
        prefill_ms=t_prefill,
        tpot_ms=cycle,
        ttft_core_ms=ttft,
        tok_s_engine=qc * osl,
        bound=bound,
        ici_ms=ici,
        stage_ms=stage_ms,
        ops=ops,
        model=NAME,
        mixes=MIXES,
    )
    _CACHE[key] = out
    return out


def selftest() -> int:
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = R.load(os.path.join(here, "configs", "glm-5.2-fp8-8k1k-2x2x2.json"))

    class T(unittest.TestCase):

        def test_pp_is_priced_from_the_real_stages_not_the_mean(self):
            """parallelism/pp s6: TPOT is the decode cycle over `v` batches of
            running/v; TTFT pipelines the prompt's chunks. Both reduce to the old
            single-stage numbers at pp=1."""
            part = (6, 4, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 4, 8, 4)
            c16 = replace(cfg, sharding=R.Sharding(pp=16, pp_layers=part))
            a = solve(c16, 8192, 1024, 64, 1024)
            self.assertEqual(len(a.stage_ms), 16)
            self.assertAlmostEqual(a.tpot_ms,
                                   R.pp_cycle_ms(list(a.stage_ms), 16),
                                   places=9)
            self.assertGreater(a.tpot_ms, 16 * a.step_ms)  # not the mean stage
            self.assertLess(a.ttft_core_ms, a.n_chunks * a.round_ms / 2)
            # fewer running requests than stages: a steady-state bubble
            lone = solve(c16, 8192, 1024, 4, 1024)
            self.assertAlmostEqual(lone.tpot_ms, lone.round_ms, places=9)

        def test_the_op_breakdown_sums_to_run_phase(self):
            """This wrapper must not become a second cost model. If the parts stop
            summing to `roofline.run_phase`, it has."""
            for ph in (R.Phase("decode", tokens=64, kv_len=9216, batch=64),
                       R.Phase("prefill",
                               tokens=8192,
                               kv_len=8192,
                               batch=1,
                               mnbt=1024)):
                total, ops, _, _ = _phase_ops(cfg, ph)
                self.assertAlmostEqual(sum(ops.values()) / total,
                                       1.0,
                                       places=9)
                self.assertAlmostEqual(total,
                                       R.run_phase(cfg, ph).total_ms,
                                       places=9)

        def test_it_reproduces_roofline_serving_on_the_shipped_config(self):
            """The shipped config is dp16/tp1, where `world` and `dp` coincide, so
            the two must agree exactly. Anywhere else they differ by tp*pcp*pp --
            see the docstring."""
            res = [R.run_phase(cfg, p) for p in cfg.phases]
            sv = R.serving(cfg, res)
            sol = solve(cfg, 8192, 1024, running=64, mnbt=1024)
            self.assertAlmostEqual(sol.step_ms, sv.t_step_ms, places=6)
            self.assertAlmostEqual(sol.prefill_ms, sv.t_prefill_ms, places=6)
            self.assertAlmostEqual(sol.tpot_ms, sv.tpot_ms, places=6)
            self.assertAlmostEqual(sol.ttft_core_ms, sv.ttft_ms, places=6)
            self.assertAlmostEqual(sol.tok_s_engine,
                                   sv.qps_combined * 1024,
                                   places=4)

        def test_the_fleet_divergence_is_real_and_pinned(self):
            """`roofline.serving` scales per-core by `world`; this scales by `dp`.
            At tp16-dp1 that is a factor of 16, and this test exists so nobody
            'fixes' one to match the other without deciding which is right."""
            c2 = replace(cfg, sharding=R.Sharding(tp=16, dp=1))
            res = [R.run_phase(c2, p) for p in c2.phases]
            sv = R.serving(c2, res)
            self.assertEqual(c2.hardware.world, 16)
            self.assertEqual(c2.sharding.dp, 1)
            self.assertAlmostEqual(sv.tok_s_fleet / sv.tok_s_per_core,
                                   16.0,
                                   places=6)

        def test_the_step_is_exactly_invariant_to_the_token_budget(self):
            """The defining property of this model, and the honest way to state it.

            Prefill and decode do not share a budget here, so `mnbt` cannot touch the
            decode step at all -- it only changes how many chunks the DEDICATED
            prefill run is split into. So mnbt moves TTFT and throughput, and never
            TPOT. If T_step ever varies with mnbt, this model has grown a mixing term
            it is not supposed to have and has become model 2.
            """
            sols = [
                solve(cfg, 8192, 1024, running=60, mnbt=m)
                for m in (256, 512, 1024, 2048, 4096, 8192)
            ]
            for s_ in sols[1:]:
                self.assertAlmostEqual(s_.step_ms, sols[0].step_ms, places=9)
            # ...while the prefill run really does move, and the LOW rungs are where
            # it moves: 1816 ms at mnbt 256 against 1486 at 4096 on this cell.
            pre = [s_.prefill_ms for s_ in sols]
            self.assertGreater(pre[0] / min(pre), 1.20)
            self.assertLess(max(pre[2:]) / min(pre[2:]),
                            1.01)  # flat above 1024

        def test_no_prefill_schedule_interval_term(self):
            a = solve(cfg, 8192, 1024, running=64, mnbt=1024)
            b = solve(cfg,
                      8192,
                      1024,
                      running=64,
                      mnbt=1024,
                      prefill_schedule_interval=8)
            self.assertEqual(a.tok_s_engine, b.tok_s_engine)

        def test_a_bigger_running_set_is_never_slower_per_token(self):
            got = [
                solve(cfg, 8192, 1024, running=r, mnbt=1024).tok_s_engine
                for r in (1, 4, 16, 64)
            ]
            self.assertEqual(got, sorted(got))

    r = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(T))
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
