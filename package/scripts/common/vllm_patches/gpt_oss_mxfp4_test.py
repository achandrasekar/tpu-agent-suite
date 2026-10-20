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
"""Tests for gpt_oss_mxfp4.py, the revision-tolerant gpt-oss MXFP4 patch.

The plugins here are small stand-ins written for the test: each keeps only
the lines the fixes anchor on, in the two layouts the patch knows.
"""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "gpt_oss_mxfp4.py")
APPLIER = os.path.join(HERE, "apply_vllm_patch.sh")

_spec = importlib.util.spec_from_file_location("gpt_oss_mxfp4", SCRIPT)
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)

_PROCESS = '''
    def process_weights_after_loading(self, layer):
        {activation}

        w13_weight = dequantize_mxfp4_packed(
            layer.w13_weight.data.to("cpu"),
            layer.w13_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w13_weight.device)
        w2_weight = dequantize_mxfp4_packed(
            layer.w2_weight.data.to("cpu"),
            layer.w2_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w2_weight.device)

        w13_weight = w13_weight.transpose(1, 2).contiguous()
        w2_weight = w2_weight.transpose(1, 2).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        if layer.w13_bias is not None:
            w13_bias = layer.w13_bias.data

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
        if layer.w2_bias is not None:
            layer.w2_bias = torch.nn.Parameter(
                layer.w2_bias.data.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
'''

_MXFP4 = '''\
import torch
from vllm.model_executor.layers.quantization import register_quantization_config

from vllm_torchtpu.layers.{core}.quant_methods import {import_names}


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config:

    @classmethod
    def get_name(cls):
        return MXFP4


class VllmMxfp4MoEMethod:
{process}
    def _forward_monolithic_tpu(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,{input_ids}
    ) -> torch.Tensor:
        """Forward pass using GMM kernel."""
        return x
'''

_QUANTIZATION = '''\
from vllm_torchtpu.layers.{core} import quant_methods
from vllm_torchtpu.layers.{adapter}.quantization.mxfp4 import VllmMxfp4Config


def get_tpu_quantization_config():
    method_to_config = {{
        None: None,
        quant_methods.MXFP4: VllmMxfp4Config,
    }}
    return method_to_config
'''

_ATTENTION = '''\
from vllm.utils.math_utils import cdiv

TPU_HEAD_SIZE_ALIGNMENT = 128


class Impl:

    def forward(self, query, key, value, kv_cache):
        if kv_cache.ndim == 4:
            pool_head_dim = self.head_size
        elif self._pool_is_seq_along_lane:
            pool_head_dim = kv_cache.shape[-3] * kv_cache.shape[-2]
        else:
            pool_head_dim = kv_cache.shape[-1]
        if pool_head_dim > self.head_size:
            pad_size = pool_head_dim - self.head_size
        return self.num_kv_heads
'''

_ATTENTION_PROBE = '''\
TPU_HEAD_SIZE_ALIGNMENT = 128


def cdiv(a, b):
    return -(-a // b)


class Impl:

    def forward(self, kv_cache):
        if kv_cache.ndim == 4:
            pool_head_dim = self.head_size
        elif self._pool_is_seq_along_lane:
            pool_head_dim = kv_cache.shape[-3] * kv_cache.shape[-2]
        else:
            pool_head_dim = kv_cache.shape[-1]
        if pool_head_dim > self.head_size:
            return "widened"
        assert self.num_kv_heads
        return "kept"
'''

_NEW_PLATFORM = '''\
class TpuPlatform:
    supported_quantization: list[str] = [
        "fp8",
        "mxfp4",
        "deepseek_v4_fp8",
    ]
'''

_OLD_PLATFORM = '''\
class TpuPlatform:
    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4"
    ]
'''


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_plugin(root, layout="new"):
    """A stand-in vllm_torchtpu under `root` in the given layout."""
    core, adapter = ("core", "adapter") if layout == "new" else ("common",
                                                                 "vllm")
    pkg = "vllm_torchtpu"
    write(root, f"{pkg}/__init__.py", "")
    write(root, f"{pkg}/layers/{core}/quant_methods.py",
          'UNQUANTIZED = "unquantized"\nMXFP4 = "mxfp4"\nFP8 = "fp8"\n')
    write(root, f"{pkg}/platforms/tpu_platform.py",
          _NEW_PLATFORM if layout == "new" else _OLD_PLATFORM)
    write(root, f"{pkg}/layers/{adapter}/quantization/__init__.py",
          _QUANTIZATION.format(core=core, adapter=adapter))
    activation = ("activation_str = self._set_tpu_activation(layer)"
                  if layout == "new" else
                  "activation_str = _get_activation_str(layer.activation)")
    input_ids = ("\n        input_ids: torch.Tensor | None = None,"
                 if layout == "new" else "")
    imports = ("MXFP4, get_tpu_quant_method"
               if layout == "new" else "(MXFP4,\n    get_tpu_quant_method)")
    write(
        root, f"{pkg}/layers/{adapter}/quantization/mxfp4.py",
        _MXFP4.format(core=core,
                      import_names=imports,
                      process=_PROCESS.format(activation=activation),
                      input_ids=input_ids))
    if layout == "new":
        write(root, f"{pkg}/layers/{adapter}/attention.py", _ATTENTION)
    else:
        write(root, f"{pkg}/layers/{adapter}/attention.py",
              "class Impl:\n    pass\n")


def snapshot(root):
    files = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                files[os.path.relpath(path, root)] = f.read()
    return files


class PatchTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def run_patch(self, *args):
        return subprocess.run([sys.executable, SCRIPT, *args],
                              capture_output=True,
                              text=True,
                              check=False)

    def read(self, rel):
        with open(os.path.join(self.root, "vllm_torchtpu", rel),
                  encoding="utf-8") as f:
            return f.read()

    def assert_parses(self):
        for rel, text in snapshot(self.root).items():
            with self.subTest(rel):
                ast.parse(text)

    def test_new_layout_applies_every_fix_then_is_idempotent(self):
        make_plugin(self.root, "new")
        done = self.run_patch(self.root)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("vllm patch applied: gpt_oss_mxfp4.py", done.stdout)
        self.assertIn("moe-forward-input-ids: present, upstream", done.stdout)
        self.assert_parses()
        mxfp4 = self.read("layers/adapter/quantization/mxfp4.py")
        self.assertIn("class VllmGptOssMxfp4Config(VllmMxfp4Config)", mxfp4)
        self.assertIn("hidden_p = _gpt_oss_round_up(hidden, 128)", mxfp4)
        self.assertIn("_zero_gpt_oss_unloaded_intermediate(layer)", mxfp4)
        self.assertNotIn(").to(layer.w13_weight.device)", mxfp4)
        self.assertIn("b13.unsqueeze(1).to(device)", mxfp4)
        self.assertIn("b2.unsqueeze(1).to(device)", mxfp4)
        quantization = self.read("layers/adapter/quantization/__init__.py")
        self.assertIn("quant_methods.GPT_OSS_MXFP4: VllmGptOssMxfp4Config",
                      quantization)
        self.assertIn(
            "from vllm_torchtpu.layers.adapter.quantization.mxfp4 import"
            " VllmGptOssMxfp4Config", quantization)
        self.assertIn('GPT_OSS_MXFP4 = "gpt_oss_mxfp4"',
                      self.read("layers/core/quant_methods.py"))
        self.assertIn("own_hd64_pool",
                      self.read("layers/adapter/attention.py"))
        platform = ast.parse(self.read("platforms/tpu_platform.py"))
        allowed = [e.value for e in platform.body[0].body[0].value.elts]
        self.assertEqual(allowed,
                         ["fp8", "mxfp4", "deepseek_v4_fp8", "gpt_oss_mxfp4"])

        patched = snapshot(self.root)
        again = self.run_patch(self.root)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("vllm patch already applied", again.stdout)
        self.assertIn("present, applied earlier by this patch", again.stdout)
        self.assertNotIn(": applied:", again.stdout)
        self.assertEqual(snapshot(self.root), patched)

    def test_hd64_pool_kept_at_every_kv_head_count(self):
        # The plugin's hd64 pool is (blocks, block, cdiv(kv_heads, packing),
        # packing, 128). With one KV head per device (TP 8 for gpt-oss-20b)
        # that has as many rows as a wider shared pool, so the row count
        # alone cannot tell them apart; a layer with sinks must still stay
        # on the hd64 kernel.
        namespace = {}
        exec(patch.edit_hd64_attention_sinks(_ATTENTION_PROBE),
             namespace)  # noqa: S102

        def run(kv_heads, rows, sinks):
            impl = namespace["Impl"]()
            impl.head_size, impl.num_kv_heads = 64, kv_heads
            impl._pool_is_seq_along_lane = False
            impl.sinks = sinks
            pool = type("Pool", (), {
                "ndim": 5,
                "shape": (8, 16, rows, 2, 128)
            })
            return impl.forward(pool)

        for kv_heads in (4, 2, 1):
            with self.subTest(kv_heads=kv_heads):
                hd64_rows = -(-kv_heads // 2)
                self.assertEqual(run(kv_heads, hd64_rows, sinks=object()),
                                 "kept")
                # A shared pool laid out for twice the rows still widens.
                self.assertEqual(run(kv_heads, kv_heads, sinks=None),
                                 "widened")

    def test_old_layout(self):
        make_plugin(self.root, "old")
        done = self.run_patch(self.root)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assert_parses()
        self.assertIn("hd64-attention-sinks: present, not needed", done.stdout)
        self.assertIn("moe-forward-input-ids: applied", done.stdout)
        mxfp4 = self.read("layers/vllm/quantization/mxfp4.py")
        forward = patch.find_method(ast.parse(mxfp4), "VllmMxfp4MoEMethod",
                                    "_forward_monolithic_tpu")
        self.assertEqual([a.arg for a in forward.args.args],
                         ["self", "layer", "x", "router_logits", "input_ids"])
        self.assertIn(
            "from vllm_torchtpu.layers.vllm.quantization.mxfp4 import"
            " VllmGptOssMxfp4Config",
            self.read("layers/vllm/quantization/__init__.py"))
        platform = ast.parse(self.read("platforms/tpu_platform.py"))
        allowed = [e.value for e in platform.body[0].body[0].value.elts]
        self.assertEqual(allowed[-2:], ["mxfp4", "gpt_oss_mxfp4"])
        self.assertIn("vllm patch already applied",
                      self.run_patch(self.root).stdout)

    def test_moved_anchor_fails_naming_the_fix_and_writes_nothing(self):
        make_plugin(self.root, "new")
        path = os.path.join(self.root, "vllm_torchtpu/layers/adapter/"
                            "attention.py")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                text.replace("if pool_head_dim > self.head_size:",
                             "if pool_head_dim >= self.head_size:"))
        before = snapshot(self.root)
        done = self.run_patch(self.root)
        self.assertEqual(done.returncode, 4)
        self.assertIn("vllm patch does not apply", done.stderr)
        self.assertIn("hd64-attention-sinks", done.stderr)
        self.assertNotIn("expert-padding-128", done.stderr)
        self.assertEqual(snapshot(self.root), before)

    def test_fix_already_upstream_is_skipped(self):
        make_plugin(self.root, "new")
        path = os.path.join(self.root,
                            "vllm_torchtpu/layers/core/quant_methods.py")
        with open(path, "a", encoding="utf-8") as f:
            f.write('GPT_OSS_MXFP4 = "gpt_oss_mxfp4"\n')
        done = self.run_patch(self.root)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(
            "quantization-name-registration: present, upstream:"
            " vllm_torchtpu/layers/core/quant_methods.py", done.stdout)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read().count("GPT_OSS_MXFP4 ="), 1)

    def test_config_map_needs_the_config_class(self):
        make_plugin(self.root, "new")
        path = os.path.join(
            self.root, "vllm_torchtpu/layers/adapter/quantization/"
            "mxfp4.py")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                text.replace("class VllmMxfp4Config:",
                             "class TpuMxfp4Config:"))
        done = self.run_patch(self.root)
        self.assertEqual(done.returncode, 4)
        self.assertIn("gpt-oss-config-class", done.stderr)
        self.assertIn("quantization-name-registration", done.stderr)

    def test_missing_plugin(self):
        self.assertEqual(self.run_patch(self.root).returncode, 3)
        self.assertEqual(self.run_patch("a", "b").returncode, 2)

    def test_one_line_allowlist(self):
        text = 'class P:\n    supported_quantization = ["fp8", "mxfp4"]\n'
        patched = patch.edit_platform_allowlist(text)
        values = [
            e.value for e in ast.parse(patched).body[0].body[0].value.elts
        ]
        self.assertEqual(values, ["fp8", "mxfp4", "gpt_oss_mxfp4"])
        self.assertIn(patch.MARKER, patched)
        with self.assertRaises(patch.Present):
            patch.edit_platform_allowlist(patched)

    def test_applier_runs_a_python_patch(self):
        """apply_vllm_patch.sh, as launch_vllm.sh calls it."""
        make_plugin(self.root, "new")
        inputs = os.path.join(self.root, "inputs")
        os.makedirs(inputs)
        shutil.copy(SCRIPT, inputs)
        shutil.copy(APPLIER, inputs)
        env = dict(os.environ,
                   TPU_AGENT_SUITE_PYTHON=sys.executable,
                   PYTHONPATH=self.root)
        for expected in ("vllm patch applied", "vllm patch already applied"):
            done = subprocess.run([
                "bash",
                os.path.join(inputs, "apply_vllm_patch.sh"), "gpt_oss_mxfp4.py"
            ],
                                  cwd=inputs,
                                  capture_output=True,
                                  text=True,
                                  env=env,
                                  check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn(expected, done.stdout)


if __name__ == "__main__":
    unittest.main()
