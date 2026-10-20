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
"""Step model 2 — MIXED. One fused step carrying decode and a prefill chunk.

**Not the default.** Reported alongside model 1 as an FYI, because the two disagree
and neither has been settled against hardware. `--roofline-model 2` promotes it.

What it changes, and the one thing that follows from it
=======================================================
vLLM's scheduler builds ONE batch per step: running (decode) requests first, one token
each, then what is left of the token budget goes to chunked prefill
(`v1/core/sched/scheduler.py`). So a steady-state step is mixed, and prefill and decode
**compete for one budget**.

That single structural change is the whole difference from model 1, and everything
else follows from it:

  * `max_num_batched_tokens` acquires a term. Model 1 moves 0.13% across the ladder;
    this moves 1.03x (C=16) to 1.18x (C=1024). **If you need a token-budget knob to
    mean anything, this is the model that can say so** -- which is also exactly why it
    should not be quietly substituted for model 1 in a report.
  * The weights are read ONCE per step, not once per phase. Adding two phases
    double-charges the largest HBM term at decode -- 1.11x on the 8K/1K cell.
  * TTFT becomes a sum over the steps a prompt actually takes while decode runs
    alongside, rather than a dedicated prefill run.

Two assumptions model 1 does not need
=====================================
Both are load-bearing and neither is verified:

  steady state   over its life a request contributes ISL prefill and OSL decode
                 tokens, so in equilibrium the step must process them at that ratio.
                 Breaks under a bursty arrival process, which this has no model of.
  occupancy      a resident request is either decoding or mid-prefill, so
                 `running = d + q`. Assumes the scheduler never holds a request that
                 is doing neither.

Pipeline parallelism is charged the round trip
==============================================
`roofline.run_phase` divides layers by `pp` and returns the MEAN stage, which model 1
reports as the step. A pipeline has two time scales and the mean is neither:
throughput is set by the slowest stage, TPOT by the full traversal. Reporting the mean
as the step made a `pp16` row **13.7x optimistic** in this repo once already, so this
model computes both from the real per-layer costs and charges `max(v*T_slow, T_round)`.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R  # noqa: E402
from step_api import Solution  # noqa: E402

NAME = "model2"
MIXES = True

# ------------------------------------------------------------------ the fused step


def mixed_step(cfg: R.Config, dec: R.Phase, pre: R.Phase | None) -> list[R.Op]:
    """The op list for ONE real vLLM step: `d` decode tokens at full context plus a
    prefill chunk, in a single fused forward pass.

    Pricing the two as separate phases and adding the results charges the weight read
    twice, which at decode is the dominant HBM term; pricing them as one phase at
    `tokens = d + p` charges every query the same context, which is wrong in the other
    direction because a decode query attends over ISL+OSL and a prefill query over its
    chunk.

    So: build both sub-passes, and merge per op --
        FLOPs        sum       both sub-batches run through the same kernel
        activations  sum       they scale with the tokens in the pass
        KV reads     sum       each sub-batch fetches its own context
        weights      once      one kernel launch, one pass over the parameters
        collectives  once at `d + p`, not summed -- one call carries both sub-batches

    `pre=None` is a decode-only step and reduces to `compute_ops(dec)` exactly.
    """
    merged: dict[str, R.Op] = {}
    tokens = float(dec.tokens)
    for ph in (dec, pre):
        if ph is None:
            continue
        if ph is not dec:
            tokens += float(ph.tokens)
        for op in R.compute_ops(cfg, ph):
            prev = merged.get(op.name)
            if prev is None:
                merged[op.name] = replace(op)
                continue
            # The three HBM parts are now carried separately by `roofline.Op`, so this
            # merge states the docstring's rule directly instead of reconstructing the
            # non-weight remainder by subtraction: activations and KV sum, weights are
            # read once. `Op.hbm_bytes` is their sum, so it follows.
            prev.flops += op.flops
            prev.act_bytes += op.act_bytes
            prev.kv_bytes += op.kv_bytes
            prev.w_bytes = max(prev.w_bytes, op.w_bytes)
    step_shape = replace(dec, tokens=int(round(tokens)), mnbt=0)
    return list(merged.values()) + R.collectives(cfg, step_shape)


# ------------------------------------------- pipeline stages  parallelism/pp/SKILL.md

# The stage arithmetic lives in `roofline.py` so both step models price the same
# pipeline -- partition (`Sharding.pp_layers` or `get_pp_indices`), real indexer
# positions, padded hand-off. These names are kept for the tests and callers here.
layer_costs = R.layer_costs
stage_costs = R.stage_costs


def pp_partition(n_layers: int, k: int) -> list[int]:
    """`get_pp_indices`'s default partition. `parallelism/pp/SKILL.md` s3."""
    return R.pp_default_partition(n_layers, k)


def pp_p2p_ms(cfg: R.Config, bucket: float) -> float:
    """One hand-off; every stage pays it. `R.pp_handoff_ms`, pp/SKILL.md s4."""
    return R.pp_handoff_ms(cfg, bucket)


# ------------------------------------------------------------------- the token mix


def token_mix(running: float, isl: int, osl: int,
              mnbt: int) -> tuple[float, float, float, float]:
    """Split one step's token budget into decode and prefill. Returns (d, p, q, chunk).

    Two equations and two ceilings.

      steady state   every request contributes ISL prefill and OSL decode tokens over
                     its life, so the step processes them at that ratio: p/d = ISL/OSL
      occupancy      a resident request is either decoding (1 token) or mid-prefill,
                     so  running = d + q  where q = p / chunk
      budget         d + p <= mnbt
      chunk          what is left of the budget after decode takes its tokens

    `chunk` depends on `d` and `d` depends on `chunk`, so this iterates; it converges
    monotonically in a few passes and 32 is generous.

    `q` is a *time average* and is routinely well below 1: at ISL 8K / OSL 1K a request
    spends 8 steps prefilling and 1,024 decoding, so an engine has a request mid-prefill
    about 0.8% of the time. Rounding it up to 1 would halve decode throughput.
    """
    ratio = isl / osl
    d_budget = mnbt * osl / (isl + osl)
    d = min(d_budget, running)
    chunk = max(1.0, min(float(mnbt) - d, float(isl)))
    q = 0.0
    for _ in range(32):
        chunk = max(1.0, min(float(mnbt) - d, float(isl)))
        q = (d * ratio) / chunk
        nd = min(d_budget, max(0.0, running - q))
        if abs(nd - d) < 1e-12 * max(1.0, d):
            d = nd
            break
        d = nd
    return d, d * ratio, q, chunk


def _step_ops(cfg: R.Config, isl: int, osl: int, d: float, p: float,
              q: float) -> list[R.Op]:
    """decode sub-pass: `d` queries over the FULL context (ISL+OSL, the peak within a
    request, the expensive end on purpose). prefill sub-pass: `p` queries spread over
    `q` requests, each over its own partial context -- ISL/2 on average."""
    m = cfg.model
    full = isl + osl
    dec_fetch = d * (min(full, m.indexer_topk) if m.indexer_topk else full)
    dec = R.Phase("decode",
                  tokens=d,
                  kv_len=full,
                  batch=max(1, int(round(d))),
                  kv_fetched=max(1.0, dec_fetch))
    if p <= 0:
        return mixed_step(cfg, dec, None)
    mean_ctx = isl / 2.0
    pre = R.Phase("prefill",
                  tokens=p,
                  kv_len=mean_ctx,
                  batch=max(1, int(round(q))),
                  kv_fetched=max(1.0, q * mean_ctx))
    return mixed_step(cfg, dec, pre)


# ------------------------------------------------------------------------ the model

#: A sweep asks for the same step many times: once the `kv` or `slots` gate binds,
#: `running` stops moving with concurrency, so every higher rung on the ladder is the
#: identical (d, p, q). Keyed by VALUE -- `id()` would be wrong, because CPython
#: reuses the id of a collected object and a stale hit is a silently wrong number.
_CACHE: dict[tuple, Solution] = {}


def solve(cfg: R.Config,
          isl: int,
          osl: int,
          running: float,
          mnbt: int,
          prefill_schedule_interval: int = 1,
          **_ignored) -> Solution:
    s, hw, n = cfg.sharding, cfg.hardware, cfg.numerics
    stages = max(1, s.pp)
    # A k-stage pipeline wants k batches in flight (max_concurrent_batches = pp), and a
    # request whose token is in flight cannot join another (scheduler.py:558-572), so
    # the running set is DIVIDED between them. pp/SKILL.md s6.
    v = float(max(1, min(stages, int(running))))
    key = (cfg.model, hw, n.default, tuple(sorted(n.sites.items())), s.tp,
           s.dp, s.pp, s.pcp, s.dcp, s.ep, s.sp, s.pp_layers, isl, osl, mnbt,
           round(running, 6))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    d, p, q, chunk = token_mix(running / v, isl, osl, mnbt)
    if d <= 0:
        return Solution(model=NAME, mixes=MIXES)

    ops_list = _step_ops(cfg, isl, osl, d, p, q)
    # An op function returns ONE layer's cost; the step runs it once per layer of its
    # class, and a PP stage holds 1/pp of them. `roofline.run_phase` applies the same
    # two factors -- an op list summed without them is a single layer, and on GLM-5.2
    # that reads 77x fast.
    ops: dict[str, float] = {}
    by: dict[str, float] = {}
    ici = 0.0
    for o in ops_list:
        mult = R.multiplicity(cfg.model, o.layer_class) / stages
        ms = o.roofline_ms(hw) * mult
        ops[o.name] = ops.get(o.name, 0.0) + ms
        by[o.bound(hw)] = by.get(o.bound(hw), 0.0) + ms
        ici += o.ici_ms * mult
    step_ms = sum(ops.values())
    if step_ms <= 0:
        return Solution(model=NAME, mixes=MIXES)

    if stages > 1:
        # every stage launches the hand-off every forward, padded to the largest
        # bucket (mnbt), the last included. pp/SKILL.md s4.
        p2p = pp_p2p_ms(cfg, mnbt or (d + p))
        stage_ms = tuple(x + p2p for x in stage_costs(cfg, ops_list))
    else:
        stage_ms = (step_ms, )
    cycle_ms = R.pp_cycle_ms(list(stage_ms), v)
    dispatch_ms = cycle_ms / v
    n_chunks = math.ceil(isl / chunk) if chunk > 0 else 1.0
    throttle = (prefill_schedule_interval - 1) / 2.0 * dispatch_ms
    # one prompt's chunks pipeline across the stages -- one per wave at the slowest
    # stage, then the last drains -- whatever `v` is. pp/SKILL.md s6.
    ttft_core = (n_chunks * max(stage_ms) + sum(stage_ms) -
                 max(stage_ms) if stages > 1 else n_chunks * dispatch_ms)

    out = Solution(
        decode_tokens=d * v,
        prefill_tokens=p * v,
        prefill_reqs=q * v,
        chunk=chunk,
        n_chunks=n_chunks,
        step_ms=step_ms,
        tpot_ms=cycle_ms,
        ttft_core_ms=ttft_core + throttle,
        tok_s_engine=v * d / (cycle_ms / 1e3),
        bound=max(by, key=lambda k: by[k]) if by else "",
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
    hw = cfg.hardware

    class T(unittest.TestCase):

        # ---- the fused step

        def test_mixed_step_reduces_to_decode_when_there_is_no_prefill(self):
            d = R.Phase("decode", tokens=64, kv_len=9216, batch=64)
            self.assertAlmostEqual(
                sum(o.roofline_ms(hw) for o in mixed_step(cfg, d, None)),
                sum(o.roofline_ms(hw) for o in R.build_ops(cfg, d)),
                places=12)

        def test_mixed_step_reads_the_weights_once(self):
            """Two sub-passes, one pass over the parameters. Adding two phases -- what
            model 1 does -- double-charges the weight read, 1.11x on this cell."""
            d = R.Phase("decode", tokens=64, kv_len=9216, batch=64)
            p = R.Phase("prefill", tokens=960, kv_len=4096, batch=1)
            mix = {o.name: o for o in mixed_step(cfg, d, p)}
            od = {o.name: o for o in R.compute_ops(cfg, d)}
            op_ = {o.name: o for o in R.compute_ops(cfg, p)}
            for k in od:
                self.assertAlmostEqual(mix[k].w_bytes, od[k].w_bytes, places=6)
                self.assertAlmostEqual(mix[k].flops,
                                       od[k].flops + op_[k].flops,
                                       places=3)
                self.assertAlmostEqual(mix[k].hbm_bytes,
                                       od[k].hbm_bytes + op_[k].hbm_bytes -
                                       od[k].w_bytes,
                                       places=3)
            naive = (sum(o.roofline_ms(hw) for o in R.build_ops(cfg, d)) +
                     sum(o.roofline_ms(hw) for o in R.build_ops(cfg, p)))
            self.assertGreater(naive,
                               sum(o.roofline_ms(hw) for o in mix.values()))

        def test_collectives_are_priced_once_at_the_fused_token_count(self):
            d = R.Phase("decode", tokens=64, kv_len=9216, batch=64)
            p = R.Phase("prefill", tokens=960, kv_len=4096, batch=1)
            got = {o.name: o.ici_ms for o in mixed_step(cfg, d, p) if o.ici_ms}
            want = {
                o.name: o.ici_ms
                for o in R.collectives(cfg, replace(d, tokens=1024))
                if o.ici_ms
            }
            self.assertEqual(sorted(got), sorted(want))
            for k in want:
                self.assertAlmostEqual(got[k], want[k], places=9)

        def test_step_time_reconciles_with_run_phase(self):
            """A decode-only step must equal `roofline.run_phase`'s decode phase. An
            op list summed without the per-layer multiplicity is 77x fast, and every
            other bar here is invariant to a constant factor on the step."""
            for ph in (R.Phase("decode", tokens=64, kv_len=9216, batch=64),
                       R.Phase("decode", tokens=8, kv_len=131072, batch=8)):
                mine = sum(
                    o.roofline_ms(hw) *
                    R.multiplicity(cfg.model, o.layer_class)
                    for o in mixed_step(cfg, ph, None))
                self.assertAlmostEqual(mine / R.run_phase(cfg, ph).total_ms,
                                       1.0,
                                       places=9)

        # ---- the token mix: the property model 1 does not have

        def test_the_mix_holds_the_steady_state_ratio_and_the_budget(self):
            d, p, q, chunk = token_mix(16.0, 8192, 1024, 1024)
            self.assertAlmostEqual(p / d, 8192 / 1024, places=9)
            self.assertLessEqual(d + p, 1024 + 1e-6)

        def test_occupancy_closes(self):
            """running = decoding + mid-prefill. Otherwise the batch is invented."""
            d, p, q, chunk = token_mix(16.0, 8192, 1024, 1024)
            self.assertAlmostEqual(d + q, 16.0, places=6)

        def test_mnbt_moves_this_model_and_that_is_the_whole_point(self):
            """Model 1 measures 0.13% here. If this ever goes flat, the mixed step
            has been lost and every token-budget recommendation is vacuous.
            Measured: 1.030x at running=1, rising with the running set."""
            spread = []
            for run in (1, 4, 16, 60):
                got = [
                    solve(cfg, 8192, 1024, run, m).tok_s_engine
                    for m in (256, 512, 1024, 2048, 4096, 8192)
                ]
                self.assertTrue(all(x > 0 for x in got))
                spread.append(max(got) / min(got))
            self.assertGreater(max(spread), 1.02)

        # ---- pipeline parallelism

        def test_pp_is_charged_the_round_trip_not_the_mean_stage(self):
            a = solve(replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=16)),
                      8192, 1024, 64, 1024)
            self.assertEqual(len(a.stage_ms), 16)
            self.assertGreaterEqual(a.round_ms, a.stage_slow_ms * 15)
            self.assertGreaterEqual(a.tpot_ms, a.round_ms - 1e-9)
            self.assertGreater(a.tpot_ms, a.step_ms)  # NOT the mean stage

        def test_pp_reduces_to_nothing_at_k_1(self):
            a = solve(cfg, 8192, 1024, 64, 1024)
            self.assertEqual(a.stage_ms, (a.step_ms, ))
            self.assertAlmostEqual(a.tpot_ms, a.step_ms, places=9)
            self.assertAlmostEqual(a.pp_balance, 1.0, places=9)

        def test_the_partition_matches_get_pp_indices_as_the_spec_states_it(
                self):
            """`pp/SKILL.md` s2: L//k each, remainder counting back from the
            SECOND-to-last, first and last spared because they carry the embeddings.

            Written from getting it wrong: an even split with the remainder on the
            last stages puts a layer on a stage the real scheduler spares, which
            moves T_slow and therefore the whole PP row.
            """
            self.assertEqual(sum(pp_partition(78, 4)), 78)
            self.assertEqual(pp_partition(78, 4), [19, 20, 20, 19])
            self.assertEqual(pp_partition(78, 16), [4] + [5] * 14 + [4])
            self.assertEqual(pp_partition(78, 2), [39, 39])
            for k in (1, 2, 3, 4, 8, 16):
                part = pp_partition(78, k)
                self.assertEqual(len(part), k)
                self.assertEqual(sum(part), 78)
                self.assertLessEqual(max(part) - min(part), 1)

        def test_every_stage_pays_the_padded_handoff(self):
            """pp/SKILL.md s4: one collective-permute that every stage launches every
            forward, sized to the largest bucket (mnbt), not to the tokens in it."""
            c2 = replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=4))
            a = solve(c2, 8192, 1024, 64, 1024)
            raw = stage_costs(
                c2,
                _step_ops(c2, 8192, 1024,
                          *token_mix(16.0, 8192, 1024, 1024)[:3]))
            p2p = pp_p2p_ms(c2, 1024)
            self.assertGreater(p2p, 0.0)
            self.assertAlmostEqual(sum(a.stage_ms) - sum(raw),
                                   4 * p2p,
                                   places=9)
            # DSA: the top-k table rides along, so the hand-off is wider than D alone
            per_tok = cfg.model.hidden * cfg.numerics.b("act_out")
            self.assertGreater(p2p, 1024 * per_tok / cfg.hardware.ici_bw * 1e3)

        def test_one_prompt_pipelines_its_own_chunks(self):
            """pp/SKILL.md s6: at running=1 decode has one batch in flight, but the
            prompt's chunks still fill the pipe -- TTFT is ~n_chunks x T_slow, not
            n_chunks x T_round."""
            c2 = replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=16))
            a = solve(c2, 8192, 1024, 1, 1024)
            self.assertLess(a.ttft_core_ms,
                            a.n_chunks * a.stage_slow_ms + a.round_ms + 1e-9)
            self.assertLess(a.ttft_core_ms, a.n_chunks * a.round_ms / 2)

        def test_stage_costs_partition_every_layer_exactly_once(self):
            d = R.Phase("decode", tokens=64, kv_len=9216, batch=64)
            for k in (1, 2, 4, 8, 16):
                c2 = replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=k))
                ops = R.build_ops(c2, d)
                st = stage_costs(c2, ops)
                self.assertEqual(len(st), k)
                self.assertAlmostEqual(sum(st),
                                       sum(layer_costs(c2, ops)),
                                       places=9)

        def test_layer_costs_see_the_dense_moe_asymmetry(self):
            """If every layer cost the same, balance would be 1.0 and nothing would
            say repartitioning is worth ~6%."""
            c2 = replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=16))
            ops = R.build_ops(
                c2, R.Phase("decode", tokens=64, kv_len=9216, batch=64))
            lc = layer_costs(c2, ops)
            self.assertEqual(len(lc), cfg.model.layers)
            self.assertLess(max(lc[:cfg.model.dense_layers]),
                            min(lc[cfg.model.dense_layers:]))
            bal = (sum(stage_costs(c2, ops)) / 16) / max(stage_costs(c2, ops))
            self.assertTrue(0.90 < bal < 1.0)

        def test_p2p_is_zero_without_pipelining(self):
            self.assertEqual(pp_p2p_ms(cfg, 1024), 0.0)
            self.assertGreater(
                pp_p2p_ms(replace(cfg, sharding=R.Sharding(tp=1, dp=1, pp=16)),
                          1024), 0.0)

        # ---- the cache cannot serve a stale answer

        def test_the_cache_is_keyed_by_value(self):
            a = solve(cfg, 8192, 1024, 16, 1024)
            b = solve(replace(cfg, sharding=R.Sharding(tp=2, dp=8, ep=True)),
                      8192, 1024, 16, 1024)
            self.assertNotAlmostEqual(a.step_ms, b.step_ms, places=6)
            self.assertAlmostEqual(solve(cfg, 8192, 1024, 16, 1024).step_ms,
                                   a.step_ms,
                                   places=12)

    r = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(T))
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
