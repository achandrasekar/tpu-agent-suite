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
"""Unit tests for megakernel_analyzer.py and its vLLM plugin scaffold integration."""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import tempfile
import unittest

Path = pathlib.Path
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


megakernel_analyzer = _load_module(
    "megakernel_analyzer", HERE / "megakernel_analyzer.py"
)
build_plugin = _load_module(
    "build_plugin",
    PACKAGE_ROOT / "scripts" / "tpu-kernel-integrator" / "build_plugin.py",
)


class MegakernelAnalyzerTest(unittest.TestCase):

    def test_tile_hw_divisibility_and_alignment(self):
        tk, tn = megakernel_analyzer.tile_hw(5120, 2560, 1024, 1280)
        self.assertEqual((tk, tn), (1024, 1280))
        self.assertEqual(5120 % tk, 0)
        self.assertEqual(2560 % tn, 0)

        tk2, tn2 = megakernel_analyzer.tile_hw(5120, 2176, 1024, 1280)
        self.assertEqual(5120 % tk2, 0)
        self.assertEqual(2176 % tn2, 0)
        self.assertLessEqual(tk2, 1024)
        self.assertLessEqual(tn2, 1280)

    def test_analyze_qwen_hybrid_linear_and_full_attention(self):
        qwen_cfg = {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }
        spec = megakernel_analyzer.analyze_megakernel(
            qwen_cfg,
            tpu_version="v7x",
            tp_size=8,
            batch_buckets=(1, 9),
        )
        self.assertEqual(spec["status"], "ok")
        self.assertTrue(spec["model"]["uses_hybrid_linear_attention"])
        self.assertEqual(spec["model"]["num_full_attention_layers"], 16)
        self.assertEqual(spec["model"]["num_linear_attention_layers"], 48)
        self.assertEqual(spec["weight_streaming"]["strategy"], "global_bank_ring")
        self.assertEqual(
            spec["collectives"]["strategy"],
            "reduce_scatter_fp32_all_gather_bf16",
        )
        self.assertTrue(spec["vmem_budget"]["fits_in_vmem"])

    def test_analyze_kimi_mla_moe_tp32(self):
        kimi_cfg = {
            "model_type": "kimi_k3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 384,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "quantization_config": {
                "quant_method": "mxfp4",
                "bits": 4,
                "group_size": 32,
            },
        }
        spec = megakernel_analyzer.analyze_megakernel(
            kimi_cfg,
            tpu_version="v7x",
            tp_size=32,
            batch_buckets=(1, 16),
        )
        self.assertTrue(spec["model"]["uses_mla"])
        self.assertTrue(spec["model"]["uses_moe"])
        self.assertEqual(spec["model"]["quantization"]["format"], "mxfp4")
        self.assertEqual(
            spec["weight_streaming"]["strategy"], "bank_ring_plus_pool_alias"
        )
        self.assertIsNotNone(spec["weight_streaming"]["moe_pool_alias"])
        self.assertEqual(
            spec["collectives"]["strategy"],
            "hierarchical_int16_shared_exponent",
        )
        self.assertTrue(spec["vmem_budget"]["fits_in_vmem"])

    def test_head_dim_64_two_heads_per_row_packing_and_nvfp4(self):
        muse_cfg = {
            "model_type": "muse_spark",
            "hidden_size": 4096,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 12288,
            "num_local_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1024,
            "moe_latent_size": 1024,
            "quantization_config": {
                "quant_method": "nvfp4",
                "group_size": 16,
            },
        }
        spec = megakernel_analyzer.analyze_megakernel(
            muse_cfg,
            tpu_version="v7x",
            tp_size=8,
            batch_buckets=(1, 2),
        )
        self.assertTrue(spec["sharding"]["pack_two_heads_per_row"])
        self.assertEqual(spec["model"]["quantization"]["format"], "nvfp4")
        self.assertEqual(
            spec["collectives"]["strategy"], "xor_butterfly_allreduce"
        )

    def test_vmem_auto_tuning_on_v5e(self):
        cfg = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 14336,
            "vocab_size": 128256,
        }
        spec = megakernel_analyzer.analyze_megakernel(
            cfg,
            tpu_version="v5e",
            tp_size=4,
            bank_count=12,
            max_bank_k=1024,
            max_bank_n=1280,
        )
        self.assertTrue(spec["weight_streaming"]["auto_tuned_for_vmem"])
        self.assertTrue(spec["vmem_budget"]["fits_in_vmem"])
        self.assertLessEqual(
            spec["vmem_budget"]["total_allocated_vmem_bytes"],
            spec["vmem_budget"]["vmem_limit_bytes"],
        )

    def test_emit_scaffold_builds_valid_vllm_plugin(self):
        cfg = {
            "model_type": "beam",
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 16384,
            "vocab_size": 131072,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            scaffold_dir = tmp_dir / "scaffold"
            out_json = tmp_dir / "out.json"
            rc = megakernel_analyzer.main([
                "--config",
                json.dumps(cfg),
                "--tpu-version",
                "v7x",
                "--tp-size",
                "8",
                "--batch-buckets",
                "16,32",
                "--op-slug",
                "beam_decode_mk",
                "--target",
                "vllm.model_executor.models.beam:BeamMLP",
                "--emit-scaffold",
                str(scaffold_dir),
                "--out",
                str(out_json),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.is_file())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            scaffold = data["scaffold"]

            for key in (
                "pool_alias_py",
                "megakernel_py",
                "op_module_py",
                "case_module_py",
            ):
                p = Path(scaffold[key])
                self.assertTrue(p.is_file(), f"Missing {p}")
                ast.parse(p.read_text(encoding="utf-8"))

            # Verify the generated op module and megakernel build into a vLLM plugin
            plugin_dir = tmp_dir / "plugin"
            tar_path = tmp_dir / "plugin.tgz"
            manifest = build_plugin.add_op(
                plugin_dir,
                "beam_decode_mk",
                Path(scaffold["op_module_py"]),
                Path(scaffold["megakernel_py"]),
                None,
            )
            build_plugin.pack(plugin_dir, tar_path)
            self.assertIn("beam_decode_mk", manifest["ops"])
            self.assertTrue(tar_path.is_file())


if __name__ == "__main__":
    unittest.main()
