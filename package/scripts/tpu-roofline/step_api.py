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
"""The step-cost contract: what a serving model may ask a roofline for.

The contract gets its own file because two models fill it and one consumer reads it,
so it belongs to none of the three. Nothing else is here — no arithmetic, no hardware,
no imports from either model. Both `roofline_model1` and `roofline_model2` return a
`Solution`; `serving.py` consumes one and does not know which it got.

    serving.py          gates, queueing, Little's Law        model-independent
        |  running, knobs, workload
        v
    roofline_model{1,2}.solve(...) -> Solution                THE FORK
        |  ops, per layer
        v
    roofline.py         shapes, FLOPs, HBM, collectives       shared, one authority

Why there are two models and not one
====================================
They disagree about **what a step is**, which is a real modelling question with no
settled answer in this repo:

  model 1 (native)  prefill and decode are separate dedicated passes. Their rates
                    combine reciprocally because a request needs both, serially.
                    TTFT is the whole prompt run on its own.

  model 2 (mixed)   one fused pass carries `d` decode tokens at full context AND a
                    prefill chunk, sharing one token budget, reading the weights once
                    -- which is what vLLM's scheduler actually builds. TTFT is a sum
                    over the steps the prompt takes while decode runs alongside.

Neither is obviously right. Model 1 is simpler and is what `roofline.py` has always
done; model 2 is closer to the scheduler but adds assumptions (the steady-state token
ratio, the occupancy closure) that model 1 does not need. **They differ most on the
knobs that act on the token budget** — `max_num_batched_tokens` moves model 1 by
~0.13% and model 2 by 1.03-1.18x — so a report that quotes one without the other is
hiding a disagreement rather than resolving it.

Model 1 is the default. Model 2 is reported alongside as an FYI.

Filling the contract
====================
A model fills every field it can and leaves the rest at zero. Two fields are
model-specific by construction and callers must not assume both are populated:

  `prefill_ms`   model 1 only -- a dedicated prefill run has no meaning in model 2
  `prefill_tokens` / `prefill_reqs` / `chunk`
                 model 2's mixed step. Model 1 reports the chunking it used for the
                 dedicated run, which is not the same quantity; `mixes` says which.

`serving.py` adds queueing on top of `ttft_core_ms`. That term belongs to the
scheduler layer, not the roofline -- it is what `roofline.py`'s own `serving()` flags
as missing with "a floor: no queueing model" -- so neither model owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Solution:
    """One config at one admitted running set, priced. All times in ms, per engine."""

    # ---- what the scheduler ran
    decode_tokens: float = 0.0  # d, per engine per step
    prefill_tokens: float = 0.0  # p, per engine per step. model 2 only
    prefill_reqs: float = 0.0  # q, time-average requests mid-prefill. model 2 only
    chunk: float = 0.0  # prefill tokens one request gets in a step
    n_chunks: float = 0.0  # steps to consume one prompt

    # ---- what it cost
    step_ms: float = 0.0  # one step. Under PP this is the MEAN stage
    prefill_ms: float = 0.0  # whole prompt as a dedicated run. model 1 only
    tpot_ms: float = 0.0  # per output token. PP round trip included
    ttft_core_ms: float = 0.0  # TTFT excluding queueing
    tok_s_engine: float = 0.0  # OUTPUT tokens/s from one DP engine

    # ---- how it was bound, for the report and for `explain.py`
    bound: str = ""  # HBM | MXU | VPU | ICI, by share of the step
    ici_ms: float = 0.0
    stage_ms: tuple[float, ...] = ()  # per PP stage; (step_ms,) at pp=1
    ops: dict[str,
              float] = field(default_factory=dict)  # per-op ms, layers applied

    # ---- provenance. A number in a report must be able to name its model.
    model: str = ""
    mixes: bool = False  # does this model share a token budget?

    @property
    def stage_slow_ms(self) -> float:
        return max(self.stage_ms) if self.stage_ms else self.step_ms

    @property
    def round_ms(self) -> float:
        return sum(self.stage_ms) if self.stage_ms else self.step_ms

    @property
    def pp_balance(self) -> float:
        """mean stage / slowest stage. 1.0 is a perfectly partitioned pipeline."""
        if not self.stage_ms:
            return 1.0
        return (sum(self.stage_ms) / len(self.stage_ms)) / max(self.stage_ms)
