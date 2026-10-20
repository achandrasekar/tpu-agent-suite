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
"""CPU tests for core/xprof_steps.py: profile split and per-step ops.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)


def _ar(t_ns, i):
    return (
        t_ns, 100,
        f"%all-reduce.{i} = bf16[32,6144]{{1,0:T(8,128)}} all-reduce(bf16[32,6144] %x)"
    )


def _op(t_ns, name):
    return (t_ns, 50,
            f"%{name}.1 = s32[64]{{0:T(128)}} fusion(s32[64]{{0}} %p)")


def _synthetic_step(order):
    """One step: embedding AR, then per layer [attention ops] AR [ffn ops] AR."""
    ops, t, i = [_ar(0, 0)], 1000, 1
    for attn, ffn in order:
        for n in attn:
            ops.append(_op(t, n))
            t += 1000
        ops.append(_ar(t, i))
        t += 1000
        i += 1
        for n in ffn:
            ops.append(_op(t, n))
            t += 1000
        ops.append(_ar(t, i))
        t += 1000
        i += 1
    return {
        "module": "tt_jit_backbone(1)",
        "t0": 0,
        "t1": t + 1000,
        "ops": ops,
        "draft_ns": 0,
        "draft_modules": 0
    }


def _glm_trace():
    from core import derive
    return derive.load_model("GLM-5.2-FP8").TRACE


class XprofSplit(unittest.TestCase):
    KINDS = ["dense_idx", "moe_idx", "moe"]
    GOOD = [(["StreamIdxTC", "gather_fusion"], ["dense_mlp"]),
            (["StreamIdxTC", "gather_fusion"], ["router_topk", "gmm_v2"]),
            (["MLA"], ["router_topk", "gmm_v2"])]

    def test_split_and_census(self):
        from core import xprof_steps as xs
        t = _glm_trace()
        sp = xs.split(_synthetic_step(self.GOOD), 3, t)
        self.assertEqual(sp["all_reduces"], 7)
        self.assertEqual(len(sp["layers"]), 3)
        self.assertTrue(
            all(c["ok"] for c in xs.census_check(sp, self.KINDS, t["census"])))

    def test_census_fails_on_wrong_layer_contents(self):
        # negative control: layer 0 and 2 swapped -> kinds no longer match contents
        from core import xprof_steps as xs
        t = _glm_trace()
        sp = xs.split(
            _synthetic_step([self.GOOD[2], self.GOOD[1], self.GOOD[0]]), 3, t)
        self.assertFalse(
            all(c["ok"] for c in xs.census_check(sp, self.KINDS, t["census"])))

    def test_split_refuses_wrong_all_reduce_count(self):
        from core import xprof_steps as xs
        sp = xs.split(_synthetic_step(self.GOOD), 4, _glm_trace())
        self.assertIn("error", sp)

    def test_split_needs_the_models_trace_rules(self):
        # negative control: no model rule -> refuse, never assume a boundary pattern
        from core import xprof_steps as xs
        self.assertIn("error", xs.split(_synthetic_step(self.GOOD), 3, None))

    def test_other_boundary_rule(self):
        # a model with 1 all-reduce per layer and none before layer 0 (e.g. a different path)
        from core import xprof_steps as xs
        ops, t = [], 1000
        for i in range(3):
            ops.append(_op(t, "work"))
            t += 1000
            ops.append(_ar(t, i))
            t += 1000
        step = {
            "module": "tt_jit_x(1)",
            "t0": 0,
            "t1": t + 1000,
            "ops": ops,
            "draft_ns": 0,
            "draft_modules": 0
        }
        sp = xs.split(step, 3, {
            "embed_all_reduces": 0,
            "all_reduces_per_layer": 1
        })
        self.assertEqual(len(sp["layers"]), 3)

    def test_shapes_ignores_layouts(self):
        from core.xprof_steps import _shapes
        out, ins = _shapes(
            "%sort.28 = (s32[8388608]{0:T(1024)}, s32[8388608]{0:T(1024)}) "
            "sort(s32[8388608]{0:T(1024)} %p.1, s32[8388608]{0:T(1024)} %p.2), dimensions={0}"
        )
        self.assertEqual(out, "(s32[8388608], s32[8388608])")
        self.assertEqual(ins, ["s32[8388608]", "s32[8388608]"])
        out, ins = _shapes(
            "%g.1 = s32[8388608]{0:T(1024)} fusion(s32[4096,8]{1,0:T(8,128)} %a, "
            "s32[8388608]{0} %b), kind=kCustom")
        self.assertEqual((out, ins),
                         ("s32[8388608]", ["s32[4096,8]", "s32[8388608]"]))


def _qwen_trace():
    from core import derive
    return derive.load_model("Qwen3.5-397B-A17B-FP8").TRACE


class PerStepOps(unittest.TestCase):
    """Ops the served step runs once (model TRACE per_step_ops) are counted once on the served
    basis, though the twin places them inside a layer."""

    @staticmethod
    def _step(extra_attn):
        # embed AR | gdn: attn-part AR ffn AR | attn: attn-part AR ffn AR; every op 1 us
        def op(t, text, d=1000):
            return (t, d, text)

        ops, t = [_ar(0, 0)], 1000
        layer = [["%fused_conv1d_gdn_x.1 = bf16[4,512]{1,0} custom-call()"],
                 ["%RPAd-x.2 = bf16[1,4,2,2,256]{4,3,2,1,0} custom-call()"] +
                 extra_attn]
        i = 1
        for att in layer:
            for n in att:
                ops.append(op(t, n, 50000 if "copy.42" in n else 1000))
                t += 100000
            ops.append(_ar(t, i))
            t += 1000
            i += 1
            ops.append(op(t, "%gmm_v2-x.3 = bf16[48,1024]{1,0} custom-call()"))
            t += 2000
            ops.append(_ar(t, i))
            t += 1000
            i += 1
        return {
            "module": "tt_jit_backbone(1)",
            "t0": 0,
            "t1": t + 1000,
            "ops": ops,
            "draft_ns": 0,
            "draft_busy_ns": 0,
            "draft_modules": 0
        }

    def test_rope_copy_and_metadata_counted_once(self):
        from core import xprof_steps as xs
        tr = _qwen_trace()
        layers = {"twin": ["gdn", "attn"], "real": {"gdn": 45, "attn": 15}}
        once = [
            "%copy.42 = bf16[262144,64]{1,0} copy(bf16[262144,64] %p)",
            "%rpa_metadata_schedule.5 = (s32[16384], s32[16384]) custom-call()",
            "%copy-done.3 = s32[1277952]{0} copy-done((s32[1277952], u32[]) %c)"
        ]
        a = xs.split(self._step([]), 2, tr)
        b = xs.split(self._step(once), 2, tr)
        self.assertTrue(
            all(c["ok"]
                for c in xs.census_check(b, layers["twin"], tr["census"])))
        self.assertAlmostEqual(b["layers"][1]["per_step_ms"], 0.052, places=6)
        self.assertEqual(a["layers"][1]["per_step_ms"], 0.0)
        pa, pb = xs.project(a, layers, tr), xs.project(b, layers, tr)
        # the attn layer is 0.3 ms longer in the twin (3 ops 100 us apart), of which 0.052 ms
        # is per-step: served grows by 0.052 once + (0.3 - 0.052) x 15, not 0.3 x 15
        d_layer = b["layers"][1]["ms"] - a["layers"][1]["ms"]
        self.assertAlmostEqual(pb["served_step_ms"] - pa["served_step_ms"],
                               0.052 + (d_layer - 0.052) * 15,
                               places=2)
        self.assertAlmostEqual(pb["per_kind_ms"]["attn"],
                               b["layers"][1]["ms"] - 0.052,
                               places=3)

    def test_route_tensors_and_small_copies_stay_per_layer(self):
        # negative control: an MoE route gather and a 2-D copy are not per-step ops
        from core import xprof_steps as xs
        m = xs.per_step_match(_qwen_trace())
        self.assertIsNone(
            m("%gather_offload_async_done.4 = s32[163840]{0} custom-call()"))
        self.assertIsNone(m("%copy.38 = s32[8,4]{1,0} copy(s32[8,4] %p)"))
        self.assertIsNone(
            m("%copy.165 = bf16[16384,4096]{1,0} copy(bf16[16384,4096] %p)"))
        self.assertEqual(
            m("%gather_offload_async_done.1 = bf16[49152,64]{1,0} custom-call()"
              ), "mRoPE position gather")
        self.assertIsNone(
            xs.per_step_match(None)("%copy.42 = bf16[262144,64]{1,0} copy()"))


if __name__ == "__main__":
    unittest.main()
