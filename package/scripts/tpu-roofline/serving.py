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
"""The scheduler front-end: from a roofline and an *offered load* to what a client sees.

    python3 scripts/serving.py --selftest
    python3 scripts/serving.py scripts/configs/<cfg>.json --concurrency 64
    python3 scripts/serving.py scripts/configs/<cfg>.json --roofline-model 2

Three layers, and this is the top one
=====================================
    roofline.py                shapes, FLOPs, HBM, collectives, weights, KV pool
    roofline_model{1,2}.py     how op times compose into a step        THE FORK
    serving.py                 admission, queueing, Little's Law       <- here

**Nothing in this file knows which step model it is driving**, and nothing in it knows
a FLOP count, a bandwidth or a dtype. If a number here could be changed by a hardware
pin or by the choice of step model, it is in the wrong file. What it owns:

  the three gates   running = min(offered, max_num_seqs, kv_capacity)
  the queue         Little's Law on the QUEUE: waiting requests / completion rate
  the cross-check   E2E forward vs concurrency / throughput

The queue term belongs here rather than to either model because it is a scheduler
property, not a roofline one -- it is exactly what `roofline.serving()` flags as
missing with "a floor: no queueing model".

Model 1 is the default. See `step_api.py` for why there are two and what they
disagree about.

Scope — the conditions this model is valid under
================================================
**Unified serving.** Under P/D disaggregation TTFT belongs to the prefill cluster and
TPOT to the decode cluster, fleet throughput is min(prefill, decode) capacity, and the
KV transfer has no term here. Say so rather than reporting these numbers for it.

**A closed loop.** The benchmark runs `--request-rate inf --max-concurrency C`, so `C`
is genuinely held constant and Little's Law applies exactly rather than in expectation.
An open-loop arrival process would need this whole file replaced, not adjusted: with
arrivals independent of completions there is no fixed `C` to apply the identity to, and
the queue term below becomes a waiting-time distribution rather than a ratio.

**One `(ISL, OSL)`.** Claim: over its life a request contributes ISL prefill and OSL
decode tokens, so in steady state the step processes them at that ratio. A **bimodal**
prompt mix has a different steady state from its mean, and nothing here represents the
spread -- two workloads with the same mean ISL can want different shardings.

**No prefix caching.** With it on, a request's prefill token count is not ISL, so both
the ratio above and `n_chunks` are wrong in the same direction (pessimistic).

**One token per request per step.** Speculative decoding breaks Claim 1's lockstep:
a request can advance several tokens in a step, so `TPOT != T_step` and the decode
term stops being linear in the running set.

Each of these is a *scope* condition, not an error bar. Inside them the model can be
wrong by its modelling; outside them it is answering a different question.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R  # noqa: E402
import roofline_model1 as M1  # noqa: E402
import roofline_model2 as M2  # noqa: E402
from step_api import Solution  # noqa: E402

GiB = R.GiB

#: Model 1 is the default, deliberately: it is `roofline.py`'s own composition, and
#: promoting the newer one by default would make every report depend on a modelling
#: choice nobody has settled against hardware.
MODELS = {1: M1, 2: M2}
DEFAULT_MODEL = 1


@dataclass(frozen=True)
class Workload:
    """The request shape. `isl`/`osl` are per request, not per step."""
    isl: int
    osl: int

    @property
    def label(self) -> str:
        return f"{self.isl}/{self.osl}"


@dataclass(frozen=True)
class Knobs:
    """Server flags, all **per DP engine**, exactly as vLLM takes them.

    Fleet-wide is `x DP`. Confusing the two is how a number ends up 16x wrong.
    """
    max_num_batched_tokens: int  # --max-num-batched-tokens
    max_num_seqs: int  # --max-num-seqs
    prefill_schedule_interval: int = 1  # steps between prefill admissions

    @property
    def label(self) -> str:
        return (f"mnbt{self.max_num_batched_tokens}-seqs{self.max_num_seqs}" +
                (f"-psi{self.prefill_schedule_interval}"
                 if self.prefill_schedule_interval > 1 else ""))


@dataclass(frozen=True)
class Point:
    """One config at one offered concurrency. `feasible=False` carries the reason."""
    concurrency: int  # fleet-wide, what `--max-concurrency` takes
    per_engine: float
    feasible: bool
    reason: str = ""
    model: str = ""  # which step model produced the times

    # what the scheduler admitted -- model-independent
    running: float = 0.0
    gate: str = ""  # offered | slots | kv -- which cap bound `running`
    kv_cap_reqs: float = 0.0
    kv_pool_gib: float = 0.0  # reserved for KV at startup
    kv_used_gib: float = 0.0  # what the running set actually occupies of it
    weights_gib: float = 0.0

    @property
    def hbm_used_gib(self) -> float:
        """Weights plus the KV the running set holds.

        NOT weights + pool: vLLM reserves the whole pool up front, so that figure is
        the same for every config at a given sharding and says nothing about the
        operating point. `used` is what distinguishes a config running against its KV
        gate from one with headroom -- and the gap between the two is how much room
        there is to raise concurrency before the sharding has to change.
        """
        return self.weights_gib + self.kv_used_gib

    # what the step model said
    decode_tokens: float = 0.0
    prefill_tokens: float = 0.0
    prefill_reqs: float = 0.0
    chunk: float = 0.0
    n_chunks: float = 0.0
    step_ms: float = 0.0
    prefill_ms: float = 0.0
    step_bound: str = ""
    ici_share: float = 0.0

    # what the client sees
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    queue_ms: float = 0.0
    output_tok_s: float = 0.0  # fleet-wide, output tokens only
    total_tok_s: float = 0.0  # fleet-wide, (ISL+OSL) per request
    req_s: float = 0.0  # fleet-wide
    littles_residual: float = 0.0  # (E2E_forward - c/rate) / TPOT, in steps

    # pipeline parallelism. Inert at pp=1.
    pp_stages: int = 1
    pp_microbatches: float = 1.0
    pp_fill: float = 1.0
    pp_balance: float = 1.0
    stage_slow_ms: float = 0.0
    round_ms: float = 0.0


def operating_point(cfg: R.Config,
                    w: Workload,
                    k: Knobs,
                    concurrency: int,
                    model: int = DEFAULT_MODEL,
                    *,
                    force: bool = False) -> Point:
    """One config at one offered concurrency, under the chosen step model.

    `force` skips `Sharding.check`, and exists because **legality is enforced in two
    places**: `sharding-strategy/search/space.py` classifies a rule as `arithmetic`
    (the partition does not exist) or `infra` (the code refuses it *today*), while
    `Sharding.check` refuses both alike. A wide search decides to price the `infra`
    class precisely so an exclusion carries a number -- and then this function threw
    those rows away anyway, which made `--wide` silently not do the one thing it is
    for.

    **Only pass `force` for a sharding the caller has established is arithmetically
    coherent** (`space.Verdict.costable`). Forcing an `arithmetic` violation divides
    by a shard count that is not one, and returns a number rather than an error.
    """
    mod = MODELS[model]
    s, hw = cfg.sharding, cfg.hardware
    bad = s.check(hw, cfg.model)
    if bad and not force:
        return Point(concurrency, 0.0, False, f"illegal sharding: {bad}",
                     mod.NAME)

    c = concurrency / s.dp
    if c < 1:
        return Point(
            concurrency, c, False,
            f"concurrency {concurrency} < dp {s.dp}: engines would idle",
            mod.NAME)

    # ---- admission. Pure memory arithmetic from `roofline.py`; no step model needed,
    # which is why two models ranked side by side always agree about what fits.
    wcfg = replace(cfg, workload=R.Workload(isl=w.isl, osl=w.osl))
    pool = R.kv_pool_bytes(wcfg)
    weights_gib = R.weight_bytes(wcfg) / GiB
    if pool <= 0:
        cap = hw.hbm_capacity * hw.memory_utilization / GiB
        # "weights do not fit" is wrong when they do and the resident overhead is what
        # tips it -- tp8-dp2 on GLM-5.2 is 86.6 GiB against 88.3, and dies on the 4 GiB
        # of compiled graphs. Same verdict, different lever, so say which.
        why = ("weights alone exceed the budget" if weights_gib >= cap else
               f"weights fit ({cap - weights_gib:.1f} GiB spare) but the "
               f"{hw.resident_overhead_gb:.0f} GiB resident overhead does not")
        return Point(
            concurrency,
            c,
            False, f"no KV pool: {why} -- {weights_gib:.1f} GiB weights + "
            f"{hw.resident_overhead_gb:.0f} overhead against {cap:.1f} usable "
            f"({hw.hbm_capacity / GiB:.0f} x {hw.memory_utilization})",
            mod.NAME,
            weights_gib=weights_gib,
            kv_pool_gib=pool / GiB)
    kv_cap = R.kv_capacity_reqs(wcfg)
    if kv_cap < 1:
        return Point(
            concurrency,
            c,
            False,
            f"KV pool holds {kv_cap:.2f} requests of {w.label} -- under one",
            mod.NAME,
            weights_gib=weights_gib,
            kv_pool_gib=pool / GiB,
            kv_cap_reqs=kv_cap)

    # The three gates. Reported on every row, because the gate names the lever:
    # `offered` -> raise concurrency, `slots` -> raise --max-num-seqs, `kv` -> a
    # sharding with KV headroom, or DCP.
    gates = {"offered": c, "slots": float(k.max_num_seqs), "kv": kv_cap}
    gate = min(gates, key=lambda g: gates[g])
    running = gates[gate]
    kv_used_gib = (running * R.kv_bytes_per_token(wcfg) *
                   (w.isl + w.osl)) / GiB

    # ---- the fork
    sol: Solution = mod.solve(
        wcfg,
        isl=w.isl,
        osl=w.osl,
        running=running,
        mnbt=k.max_num_batched_tokens,
        prefill_schedule_interval=k.prefill_schedule_interval)
    if sol.tok_s_engine <= 0 or sol.tpot_ms <= 0:
        return Point(concurrency,
                     c,
                     False,
                     f"{mod.NAME} returned no operating point at running "
                     f"{running:.1f}",
                     mod.NAME,
                     running=running,
                     gate=gate,
                     kv_cap_reqs=kv_cap,
                     kv_pool_gib=pool / GiB,
                     weights_gib=weights_gib)

    # ---- queueing. Little's Law on the QUEUE: `c - running` requests are waiting and
    # the engine drains them at `rate`. Zero below saturation, by construction.
    req_s_engine = sol.tok_s_engine / w.osl
    queue_ms = 1e3 * max(0.0, c - running) / req_s_engine
    ttft_ms = sol.ttft_core_ms + queue_ms
    e2e_ms = ttft_ms + (w.osl - 1) * sol.tpot_ms

    # ---- the cross-check. Little's Law is an identity: `c` requests in the system
    # completing at `rate` must reside `c/rate`. Nothing above solved for it, so the
    # two sides share only `rate` and disagreement is informative rather than a bug.
    #   model 2 agrees to within one step.
    #   model 1 does NOT, and the gap is the point: its TTFT is a dedicated prefill
    #   run, while its throughput is a harmonic combination that already implies each
    #   request waits behind the others' prefills. The residual measures that
    #   inconsistency, which its own docstring flags as "a floor: no queueing model".
    littles_ms = 1e3 * c / req_s_engine
    residual = (e2e_ms - littles_ms) / sol.tpot_ms if sol.tpot_ms else 0.0

    stages = max(1, s.pp)
    return Point(
        concurrency=concurrency,
        per_engine=c,
        feasible=True,
        model=mod.NAME,
        running=running,
        gate=gate,
        kv_cap_reqs=kv_cap,
        kv_pool_gib=pool / GiB,
        kv_used_gib=kv_used_gib,
        weights_gib=weights_gib,
        decode_tokens=sol.decode_tokens,
        prefill_tokens=sol.prefill_tokens,
        prefill_reqs=sol.prefill_reqs,
        chunk=sol.chunk,
        n_chunks=sol.n_chunks,
        step_ms=sol.step_ms,
        prefill_ms=sol.prefill_ms,
        step_bound=sol.bound,
        ici_share=sol.ici_ms / sol.step_ms if sol.step_ms else 0.0,
        ttft_ms=ttft_ms,
        tpot_ms=sol.tpot_ms,
        e2e_ms=e2e_ms,
        queue_ms=queue_ms,
        output_tok_s=sol.tok_s_engine * s.dp,
        total_tok_s=req_s_engine * (w.isl + w.osl) * s.dp,
        req_s=req_s_engine * s.dp,
        littles_residual=residual,
        pp_stages=stages,
        pp_microbatches=float(max(1, min(stages, int(running)))),
        pp_fill=float(max(1, min(stages, int(running)))) / stages,
        pp_balance=sol.pp_balance,
        stage_slow_ms=sol.stage_slow_ms,
        round_ms=sol.round_ms,
    )


# ============================================================ sweeps and the curve


def concurrency_ladder(dp: int, *, lo: int = 1, hi: int = 8192) -> list[int]:
    """Multiples of DP on a roughly geometric ladder -- engines must stay balanced."""
    out: list[int] = []
    c = dp * max(1, lo)
    while c <= hi:
        out.append(c)
        c = max(c + dp, int(round(c * 1.4 / dp)) * dp)
    return out


def sweep(cfg: R.Config,
          w: Workload,
          k: Knobs,
          concurrencies: Iterable[int] | None = None,
          model: int = DEFAULT_MODEL) -> list[Point]:
    cs = (list(concurrencies) if concurrencies is not None else
          concurrency_ladder(cfg.sharding.dp))
    return [operating_point(cfg, w, k, c, model) for c in cs]


@dataclass(frozen=True)
class CurvePoint:
    ttft_budget_ms: float
    output_tok_s: float
    concurrency: float
    tpot_ms: float
    gate: str
    step_bound: str
    interpolated: bool


def curve(points: Sequence[Point],
          budgets: Sequence[float]) -> list[CurvePoint]:
    """Best throughput reachable under each TTFT budget.

    Throughput and TTFT both rise with concurrency, so the answer is the largest
    concurrency whose TTFT still fits, linearly interpolated to the budget. A budget
    below the cheapest feasible point yields no entry, and that is a real answer: this
    config cannot serve that TTFT at all. It is also invisible unless the caller says
    so, which is why a ladder should carry a catch-all rung far above every config.
    """
    ok = sorted((p for p in points if p.feasible), key=lambda p: p.concurrency)
    out: list[CurvePoint] = []
    for b in budgets:
        under = [p for p in ok if p.ttft_ms <= b]
        if not under:
            continue
        lo = under[-1]
        above = [p for p in ok if p.ttft_ms > b]
        if not above:
            out.append(
                CurvePoint(b, lo.output_tok_s, lo.concurrency, lo.tpot_ms,
                           lo.gate, lo.step_bound, False))
            continue
        hi = above[0]
        span = hi.ttft_ms - lo.ttft_ms
        f = 0.0 if span <= 0 else (b - lo.ttft_ms) / span
        out.append(
            CurvePoint(
                b, lo.output_tok_s + f * (hi.output_tok_s - lo.output_tok_s),
                lo.concurrency + f * (hi.concurrency - lo.concurrency),
                lo.tpot_ms + f * (hi.tpot_ms - lo.tpot_ms), lo.gate,
                lo.step_bound, True))
    return out


# ============================================================ report


def report(cfg: R.Config, w: Workload, k: Knobs, p: Point) -> None:
    s = cfg.sharding
    print(
        f"\n=== serving   {cfg.model.name}   {s.label}   ISL {w.isl} / OSL {w.osl}"
        f"   C={p.concurrency}   {k.label}   [{p.model}]")
    if not p.feasible:
        print(f"    INFEASIBLE: {p.reason}")
        return
    print(f"    admission  offered {p.per_engine:.1f}  slots {k.max_num_seqs}"
          f"  kv {p.kv_cap_reqs:.1f}   -> running {p.running:.1f}/engine"
          f"   bound by '{p.gate}'")
    print(
        f"    HBM        weights {p.weights_gib:.1f} GiB + pool {p.kv_pool_gib:.1f}"
        f" GiB of {cfg.hardware.hbm_capacity / GiB:.0f}"
        f" x {cfg.hardware.memory_utilization}")
    if p.prefill_tokens:
        print(
            f"    the step   {p.decode_tokens:.1f} decode + {p.prefill_tokens:.1f}"
            f" prefill tokens, fused   chunk {p.chunk:.0f}"
            f" -> {p.n_chunks:.0f} chunks/prompt")
    else:
        print(
            f"    the step   {p.decode_tokens:.1f} decode tokens; prefill priced as"
            f" a separate pass ({p.prefill_ms:.1f} ms, {p.n_chunks:.0f} chunks)"
        )
    print(f"    T_step     {p.step_ms:.3f} ms   bound {p.step_bound}"
          f"   ICI {100 * p.ici_share:.1f}%")
    if p.pp_stages > 1:
        print(f"    pipeline   {p.pp_stages} stages, v={p.pp_microbatches:.0f}"
              f"   T_slow {p.stage_slow_ms:.3f}  T_round {p.round_ms:.3f}"
              f"   fill {p.pp_fill:.2f}  balance {p.pp_balance:.3f}")
    print(f"    TTFT       {p.ttft_ms:10.1f} ms   (queue {p.queue_ms:.1f})")
    print(f"    TPOT       {p.tpot_ms:10.3f} ms")
    print(f"    E2E        {p.e2e_ms / 1e3:10.2f} s")
    print(f"    throughput {p.output_tok_s:10.0f} out tok/s fleet"
          f"   ({p.total_tok_s:,.0f} total, {p.req_s:.3f} req/s)")
    print(f"    Little's Law: E2E is off the c/throughput identity by"
          f" {p.littles_residual:+.2f} steps")


def selftest() -> int:
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = R.load(os.path.join(here, "configs", "glm-5.2-fp8-8k1k-2x2x2.json"))
    w = Workload(isl=8192, osl=1024)
    k = Knobs(max_num_batched_tokens=1024, max_num_seqs=256)

    class T(unittest.TestCase):

        # ---- what this file owns is model-independent

        def test_admission_is_identical_under_both_models(self):
            """The gates are pure memory arithmetic, so two models ranked side by
            side must never disagree about what fits. If they do, a step model has
            reached into the admission path."""
            for c in (16, 64, 256, 4096):
                a = operating_point(cfg, w, k, c, model=1)
                b = operating_point(cfg, w, k, c, model=2)
                self.assertEqual((a.feasible, a.gate), (b.feasible, b.gate))
                self.assertAlmostEqual(a.running, b.running, places=9)
                self.assertAlmostEqual(a.kv_cap_reqs, b.kv_cap_reqs, places=9)
                self.assertAlmostEqual(a.weights_gib, b.weights_gib, places=9)

        def test_each_gate_can_bind(self):
            for m in (1, 2):
                self.assertEqual(
                    operating_point(cfg, w, Knobs(1024, 256), 16, m).gate,
                    "offered")
                self.assertEqual(
                    operating_point(cfg, w, Knobs(1024, 4), 1024, m).gate,
                    "slots")
                kv = operating_point(cfg, w, Knobs(1024, 10_000), 100_000, m)
                self.assertEqual(kv.gate, "kv")
                self.assertAlmostEqual(kv.running, kv.kv_cap_reqs, places=6)

        def test_weights_that_do_not_fit_are_refused_not_costed(self):
            big = replace(cfg,
                          sharding=R.Sharding(tp=1,
                                              dp=16))  # non-EP MoE at dp16
            for m in (1, 2):
                p = operating_point(big, w, k, 64, m)
                self.assertFalse(p.feasible)
                self.assertIn("weights alone exceed", p.reason)

        def test_the_no_pool_message_distinguishes_weights_from_overhead(self):
            """Same verdict, different lever. `tp8-dp2` on GLM-5.2 has 86.6 GiB of
            weights against 88.3 usable -- they fit, and the 4 GiB resident overhead
            is what kills it. Calling that 'weights do not fit' sends someone to
            shrink the wrong thing."""
            p = operating_point(replace(cfg, sharding=R.Sharding(tp=8, dp=2)),
                                w, k, 64)
            self.assertFalse(p.feasible)
            self.assertIn("resident overhead", p.reason)
            self.assertIn("weights fit", p.reason)

        def test_force_prices_an_infra_row_and_is_off_by_default(self):
            """`--wide` exists to put a number on a refusal. Legality is enforced in
            two places, and this function refusing as well meant the wide view
            silently did not price the `infra` class at all."""
            pcp = replace(cfg, sharding=R.Sharding(tp=1, dp=8, pcp=2, ep=True))
            off = operating_point(pcp, w, k, 256)
            self.assertFalse(off.feasible)
            self.assertIn("illegal sharding", off.reason)
            on = operating_point(pcp, w, k, 256, force=True)
            self.assertTrue(on.feasible)
            self.assertGreater(on.output_tok_s, 0)

        def test_queueing_is_zero_below_saturation_and_positive_above(self):
            for m in (1, 2):
                lo = operating_point(cfg, w, Knobs(1024, 10_000), 16, m)
                self.assertEqual(lo.gate, "offered")
                self.assertAlmostEqual(lo.queue_ms, 0.0, places=9)
                hi = operating_point(cfg, w, Knobs(1024, 4), 4096, m)
                self.assertGreater(hi.queue_ms, 0.0)

        def test_the_default_is_model_1(self):
            self.assertEqual(DEFAULT_MODEL, 1)
            self.assertEqual(operating_point(cfg, w, k, 64).model, "model1")

        # ---- Little's Law, and what each model does to it

        def test_model2_agrees_with_littles_law_within_one_step(self):
            """The two derivations share only the completion rate. They differ by
            exactly one step because the first generated token is counted in both the
            prefill term and the decode term."""
            for c in (16, 64, 256, 1024):
                p = operating_point(cfg, w, k, c, model=2)
                if p.feasible:
                    self.assertLessEqual(abs(p.littles_residual), 1.0,
                                         f"C={c}: {p.littles_residual:+.3f}")

        def test_model1_understates_ttft_and_the_residual_says_by_how_much(
                self):
            """Not a bug in this file -- a property of model 1, and the reason its own
            docstring calls TTFT "a floor: no queueing model". Its throughput is a
            harmonic combination that already implies a request waits behind the other
            residents' prefills; its TTFT is a dedicated prefill run that does not.
            The residual is that inconsistency, and it is NEGATIVE (forward E2E is
            below the identity) and grows with the running set."""
            resid = [
                operating_point(cfg, w, k, c, model=1).littles_residual
                for c in (16, 64, 256)
            ]
            self.assertTrue(all(r < 0 for r in resid), resid)
            self.assertGreater(abs(resid[-1]), abs(resid[0]))

        # ---- monotonicity a scheduler must obey, under either model

        def test_throughput_and_ttft_both_rise_with_concurrency(self):
            for m in (1, 2):
                ps = [
                    p for p in sweep(cfg, w, k, [16, 32, 64, 128, 256], m)
                    if p.feasible
                ]
                self.assertGreater(len(ps), 3)
                for a, b in zip(ps, ps[1:]):
                    self.assertGreaterEqual(b.output_tok_s,
                                            a.output_tok_s - 1e-6)
                    self.assertGreaterEqual(b.ttft_ms, a.ttft_ms - 1e-6)

        def test_the_curve_is_monotone_in_the_budget(self):
            for m in (1, 2):
                cv = curve(
                    sweep(cfg, w, k, concurrency_ladder(cfg.sharding.dp), m),
                    [2e3, 5e3, 15e3, 60e3, 3.6e6])
                self.assertTrue(cv)
                for a, b in zip(cv, cv[1:]):
                    self.assertGreaterEqual(b.output_tok_s,
                                            a.output_tok_s - 1e-6)

        # ---- the disagreement the two models exist to expose

        def test_the_models_differ_on_mnbt_by_MECHANISM_not_magnitude(self):
            """Both respond to the token budget, and by similar amounts -- 1.17x
            (model 1) against 1.18x (model 2) at saturation on this cell. An earlier
            version of this bar asserted model 1 was flat, which was wrong: it was
            generalising README s2's 0.13%, a figure measured over mnbt 1024..8192
            only -- exactly the flat part of the curve. The low rungs discriminate.

            The real difference is WHERE the knob acts, and it is checkable:

              model 1   `T_step` is EXACTLY invariant to mnbt, because decode is
                        priced on its own. All sensitivity is the dedicated prefill
                        run's chunk count, so mnbt moves TTFT and never TPOT.
              model 2   the budget is shared, so `T_step` itself moves -- and
                        therefore TPOT and inter-token latency do too.

            That is the thing a report must not blur: under model 1 a big chunk is
            free for decode latency; under model 2 every decode in that step waits
            behind it.
            """

            def steps(m):
                return [
                    operating_point(cfg, w, Knobs(mn, 256), 1024, m).step_ms
                    for mn in (256, 512, 1024, 2048, 4096, 8192)
                ]

            s1, s2 = steps(1), steps(2)
            for x in s1[1:]:
                self.assertAlmostEqual(x, s1[0], places=9)  # model 1: flat
            self.assertGreater(max(s2) / min(s2), 1.05)  # model 2: not flat

            def spread(m):
                got = [
                    operating_point(cfg, w, Knobs(mn, 256), 1024,
                                    m).output_tok_s
                    for mn in (256, 512, 1024, 2048, 4096, 8192)
                ]
                return max(got) / min(got)

            self.assertGreater(spread(1), 1.10)  # both respond
            self.assertGreater(spread(2), 1.10)

        # ---- the seam

        def test_only_the_roofline_sees_the_hardware(self):
            src = open(os.path.join(here, "serving.py")).read()
            body = src.split('"""', 2)[2].split("def selftest(")[0]
            for term in ("peak_fp8", "peak_bf16", "hbm_bw", "ici_bw",
                         "mxu_min_dim"):
                self.assertNotIn(term, body, f"{term!r}: the seam has leaked")

        def test_this_file_names_no_step_model_internals(self):
            """It may pick a model; it may not know how one works."""
            src = open(os.path.join(here, "serving.py")).read()
            body = src.split('"""', 2)[2].split("def selftest(")[0]
            for term in ("mixed_step", "token_mix", "stage_costs", "pp_p2p",
                         "run_phase", "build_ops"):
                self.assertNotIn(
                    term, body, f"{term!r}: serving.py reached into a "
                    f"step model")

    r = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(T))
    return 0 if r.wasSuccessful() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--concurrency",
                    type=int,
                    default=0,
                    help="fleet-wide; 0 sweeps a ladder")
    ap.add_argument("--isl", type=int)
    ap.add_argument("--osl", type=int)
    ap.add_argument("--mnbt", type=int, default=0)
    ap.add_argument("--max-num-seqs",
                    type=int,
                    default=0,
                    help="overrides the config's workload.max_num_seqs")
    ap.add_argument("--roofline-model",
                    type=int,
                    choices=(1, 2),
                    default=DEFAULT_MODEL,
                    help="1 = native phases (default), 2 = vLLM mixed step")
    ap.add_argument("--both",
                    action="store_true",
                    help="run model 1 and show model 2 alongside as an FYI")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.config:
        ap.error("need a config path, or --selftest")
    cfg = R.load(a.config)
    if a.isl and a.osl:
        w = Workload(a.isl, a.osl)
    elif cfg.workload:
        w = Workload(cfg.workload.isl, cfg.workload.osl)
    else:
        ap.error("config has no `workload` block; pass --isl and --osl")
    mnbt = a.mnbt or max((p.mnbt for p in cfg.phases if p.mnbt), default=2048)
    seqs = a.max_num_seqs or (cfg.workload.max_num_seqs if cfg.workload else 0)
    if not seqs:
        ap.error("config has no `workload.max_num_seqs`; pass --max-num-seqs")
    k = Knobs(mnbt, seqs)

    if a.concurrency:
        report(cfg, w, k,
               operating_point(cfg, w, k, a.concurrency, a.roofline_model))
        if a.both:
            report(cfg, w, k, operating_point(cfg, w, k, a.concurrency, 2))
        return 0

    print(
        f"\n{cfg.model.name}  {cfg.sharding.label}  ISL {w.isl}/OSL {w.osl}  {k.label}"
        f"   [model {a.roofline_model}" +
        ("  + model 2 FYI" if a.both else "") + "]"
        f"\n{'C':>7}{'run':>7}{'gate':>9}{'step ms':>10}{'bound':>7}"
        f"{'TTFT s':>9}{'TPOT ms':>9}{'out tok/s':>11}{'LL':>7}" +
        (f"{'m2 tok/s':>11}{'m2/m1':>8}" if a.both else ""))
    for p in sweep(cfg, w, k, None, a.roofline_model):
        if not p.feasible:
            continue
        extra = ""
        if a.both:
            q = operating_point(cfg, w, k, p.concurrency, 2)
            extra = (f"{q.output_tok_s:>11,.0f}"
                     f"{q.output_tok_s / p.output_tok_s:>8.3f}"
                     if q.feasible else f"{'-':>11}{'-':>8}")
        print(
            f"{p.concurrency:>7}{p.running:>7.1f}{p.gate:>9}{p.step_ms:>10.3f}"
            f"{p.step_bound:>7}{p.ttft_ms / 1e3:>9.2f}{p.tpot_ms:>9.2f}"
            f"{p.output_tok_s:>11,.0f}{p.littles_residual:>+7.2f}{extra}")
    print(
        "\n  modelled -- a ceiling. LL = Little's Law residual in steps; model 2"
        "\n  agrees within 1, model 1 does not and the gap is its missing queueing"
        " model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
