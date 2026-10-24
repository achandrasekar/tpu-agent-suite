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
"""Analyze a model architecture and synthesize a TPU Pallas megakernel spec and scaffold.

Given a Hugging Face model `config.json` (file path, model directory, or inline
JSON string), optional PyTorch model source file, TPU generation, and tensor
parallelism degree, this tool computes:

1. Per-rank tensor-parallel shard dimensions and 128-lane MXU/VPU alignment,
   including automatic two-heads-per-row `[nh // 2, 128]` packing when
   `head_dim == 64` (eliminating the 50% MXU lane waste on 64-wide heads).
2. Weight streaming schedule:
   - Global Bank-Ring HBM-to-VMEM tile schedule (`tile_hw`, `nk`, `nn`, `nt`)
     across all layer projections and the LM head.
   - Raw VMEM pool aliasing (`pool_alias`) schedule for heterogeneous MoE
     experts (`EXPERT_SLOTS` multi-buffered pipeline + shared expert folding).
3. Exact VMEM scratchpad budget breakdown (persistent activations/norms/RoPE,
   double-buffered KV streaming slots, recurrent state scratch, weight streaming
   banks or aliased expert pool, and ICI remote DMA collective buffers)
   validated against the chip's `vmem_limit_bytes`, with automatic tile/bank
   sizing when the requested bank ring exceeds VMEM.
4. In-kernel ICI remote DMA collective strategy (`xor_butterfly_allreduce`,
   `reduce_scatter_fp32_all_gather_bf16` for speculative verification, or
   `hierarchical_int16_shared_exponent` for multi-host TP16/TP32).
5. HBM roofline step-time floor (microseconds/step) and interactivity ceiling
   (tokens/sec) per decode step.

When `--emit-scaffold <dir>` is passed, writes:
  - `<dir>/megakernel_spec.json`
  - `<dir>/pool_alias.py`
  - `<dir>/megakernel.py`
  - `<dir>/op_<slug>.py` (compatible with `tpu-kernel-integrator/build_plugin.py`)
  - `<dir>/case_<slug>.py` (compatible with `tpu-kernel-integrator/inplace_harness.py`)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
import sys
from typing import Any

Path = pathlib.Path

EXIT_OK = 0
EXIT_USAGE = 2

TPU_SPECS: dict[str, dict[str, Any]] = {
    "v5e": {
        "tpu_version": "v5e",
        "vmem_bytes": 16 * 1024 * 1024,
        "default_vmem_limit_bytes": 14 * 1024 * 1024,
        "max_tunable_vmem_bytes": 15 * 1024 * 1024,
        "mxu_tile": 128,
        "num_mxus": 4,
        "hbm_bw_gbps": 819.0,
        "ici_bidi_bw_gbps": 100.0,
        "ici_hop_latency_us": 2.2,
        "bf16_tflops": 197.0,
    },
    "v5p": {
        "tpu_version": "v5p",
        "vmem_bytes": 64 * 1024 * 1024,
        "default_vmem_limit_bytes": 48 * 1024 * 1024,
        "max_tunable_vmem_bytes": 56 * 1024 * 1024,
        "mxu_tile": 128,
        "num_mxus": 4,
        "hbm_bw_gbps": 2765.0,
        "ici_bidi_bw_gbps": 200.0,
        "ici_hop_latency_us": 1.8,
        "bf16_tflops": 459.0,
    },
    "v6e": {
        "tpu_version": "v6e",
        "vmem_bytes": 32 * 1024 * 1024,
        "default_vmem_limit_bytes": 28 * 1024 * 1024,
        "max_tunable_vmem_bytes": 30 * 1024 * 1024,
        "mxu_tile": 256,
        "num_mxus": 2,
        "hbm_bw_gbps": 1638.0,
        "ici_bidi_bw_gbps": 200.0,
        "ici_hop_latency_us": 1.5,
        "bf16_tflops": 918.0,
    },
    "v7x": {
        "tpu_version": "v7x",
        "vmem_bytes": 64 * 1024 * 1024,
        "default_vmem_limit_bytes": 32 * 1024 * 1024,
        "max_tunable_vmem_bytes": 60 * 1024 * 1024,
        "mxu_tile": 256,
        "num_mxus": 2,
        "hbm_bw_gbps": 7370.0,
        "ici_bidi_bw_gbps": 600.0,
        "ici_hop_latency_us": 1.2,
        "bf16_tflops": 2307.0,
    },
}


class AnalyzerError(Exception):
    """Raised when the model config or parameters are invalid."""


def tile_hw(k_dim: int, n_dim: int, max_k: int, max_n: int) -> tuple[int, int]:
    """Compute Largest 128-Aligned Dividing Tile (tk, tn) <= (max_k, max_n)."""
    if k_dim <= 0 or n_dim <= 0:
        raise AnalyzerError(f"Invalid matrix dims ({k_dim}, {n_dim})")
    tk = min(k_dim, max_k)
    while k_dim % tk != 0:
        tk -= 128 if tk > 128 and (tk - 128) > 0 else 1
    tn = min(n_dim, max_n)
    while n_dim % tn != 0:
        tn -= 128 if tn > 128 and (tn - 128) > 0 else 1
    return tk, tn


def load_raw_config(config_arg: str) -> dict[str, Any]:
    """Load config.json from a file path, directory path, or JSON string."""
    s = config_arg.strip()
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise AnalyzerError(f"Invalid inline JSON config: {e}") from e
        if not isinstance(data, dict):
            raise AnalyzerError("Inline JSON config must be an object")
        return data

    p = Path(s)
    if p.is_dir():
        p = p / "config.json"
    if not p.is_file():
        raise AnalyzerError(f"Config file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AnalyzerError(f"Cannot read config {p}: {e}") from e
    if not isinstance(data, dict):
        raise AnalyzerError(f"Config in {p} must be a JSON object")
    return data


def inspect_model_source(source_path: Path | None) -> dict[str, Any]:
    """Extract class names and custom layer hints from optional PyTorch source."""
    if source_path is None:
        return {"classes": [], "has_custom_attention": False, "has_moe": False}
    if not source_path.is_file():
        raise AnalyzerError(f"Model source file not found: {source_path}")
    try:
        text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError) as e:
        raise AnalyzerError(f"Cannot parse model source {source_path}: {e}") from e

    classes = [
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    lower = text.lower()
    return {
        "classes": classes,
        "has_custom_attention": any(
            k in lower for k in ("deltanet", "kda", "mla", "linear_attn")
        ),
        "has_moe": any(
            k in lower for k in ("routed_experts", "fused_moe", "megablox", "experts")
        ),
    }


def detect_quantization(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Detect weight quantization format and bytes per element."""
    qcfg = raw_cfg.get("quantization_config") or {}
    q_method = str(
        qcfg.get("quant_method")
        or qcfg.get("format")
        or raw_cfg.get("quant_method")
        or ""
    ).lower()
    bits = qcfg.get("bits") or qcfg.get("num_bits") or qcfg.get("w_bit")
    group_size = int(qcfg.get("group_size") or qcfg.get("block_size") or 128)

    if "nvfp4" in q_method or (
        "fp4" in q_method and int(qcfg.get("group_size") or 16) == 16
    ):
        return {
            "format": "nvfp4",
            "weight_bytes_per_elem": 0.5,
            "scale_bytes_per_elem": 1.0 / 16.0,
            "group_size": 16,
            "in_vmem_dequant": (
                "tpu.bitcast(packed_u8, jnp.float4_e2m1fn).astype(jnp.bfloat16)"
                " * scale_f8e4m3fn"
            ),
        }
    if "mxfp4" in q_method or "fp4" in q_method or bits == 4:
        return {
            "format": "mxfp4",
            "weight_bytes_per_elem": 0.5,
            "scale_bytes_per_elem": 1.0 / 32.0,
            "group_size": 32,
            "in_vmem_dequant": (
                "nibble_shift_and_e2m1_piecewise_convert * exp2(e8m0_scale - 127)"
            ),
        }
    if (
        "fp8" in q_method
        or "compressed-tensors" in q_method
        or "f8" in q_method
        or bits == 8
    ):
        return {
            "format": "fp8_e4m3fn",
            "weight_bytes_per_elem": 1.0,
            "scale_bytes_per_elem": 4.0 / float(max(group_size, 128)),
            "group_size": max(group_size, 128),
            "in_vmem_dequant": (
                "lax.dot_general(x_bf16, w_f8e4m3fn,"
                " preferred_element_type=jnp.float32) * scale_f32"
            ),
        }
    return {
        "format": "bfloat16",
        "weight_bytes_per_elem": 2.0,
        "scale_bytes_per_elem": 0.0,
        "group_size": 0,
        "in_vmem_dequant": "native_bf16_mxu",
    }


def parse_model_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize Hugging Face config (including nested text_config) into canonical fields."""
    text_cfg = raw_cfg.get("text_config")
    cfg = dict(raw_cfg)
    if isinstance(text_cfg, dict):
        for k, v in text_cfg.items():
            cfg.setdefault(k, v)
        # Prefer text_config's core transformer fields over outer vision wrapper
        for k in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "layer_types",
            "full_attention_interval",
        ):
            if k in text_cfg:
                cfg[k] = text_cfg[k]

    model_type = str(cfg.get("model_type") or "custom_transformer")
    architectures = list(raw_cfg.get("architectures") or cfg.get("architectures") or [])

    hidden_size = int(cfg.get("hidden_size") or cfg.get("d_model") or 4096)
    num_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 32)
    num_q_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 32)
    num_kv_heads = int(
        cfg.get("num_key_value_heads")
        or cfg.get("num_kv_heads")
        or num_q_heads
    )
    head_dim = int(cfg.get("head_dim") or (hidden_size // max(num_q_heads, 1)))
    intermediate_size = int(
        cfg.get("intermediate_size")
        or cfg.get("ffn_dim")
        or (hidden_size * 4)
    )
    vocab_size = int(cfg.get("vocab_size") or 128256)
    rms_norm_eps = float(
        cfg.get("rms_norm_eps") or cfg.get("layer_norm_eps") or 1e-6
    )
    hidden_act = str(cfg.get("hidden_act") or cfg.get("activation_function") or "silu")

    # MLA detection (DeepSeek / Kimi K3 style)
    kv_lora_rank = int(cfg.get("kv_lora_rank") or 0)
    q_lora_rank = int(cfg.get("q_lora_rank") or 0)
    qk_nope_head_dim = int(cfg.get("qk_nope_head_dim") or 0)
    qk_rope_head_dim = int(cfg.get("qk_rope_head_dim") or 0)
    v_head_dim = int(cfg.get("v_head_dim") or head_dim)
    uses_mla = kv_lora_rank > 0

    # Hybrid linear / recurrent attention detection (Qwen3.8 GatedDeltaNet / Kimi KDA)
    linear_num_k_heads = int(cfg.get("linear_num_key_heads") or 0)
    linear_num_v_heads = int(cfg.get("linear_num_value_heads") or 0)
    linear_k_dim = int(cfg.get("linear_key_head_dim") or 0)
    linear_v_dim = int(cfg.get("linear_value_head_dim") or 0)
    linear_conv_dim = int(cfg.get("linear_conv_kernel_dim") or 0)
    full_attn_interval = int(cfg.get("full_attention_interval") or 0)
    layer_types = cfg.get("layer_types")

    num_linear_layers = 0
    num_full_attn_layers = num_layers
    if isinstance(layer_types, list) and len(layer_types) == num_layers:
        num_linear_layers = sum(
            1 for t in layer_types if "linear" in str(t).lower() or "kda" in str(t).lower()
        )
        num_full_attn_layers = num_layers - num_linear_layers
    elif full_attn_interval > 1:
        num_full_attn_layers = num_layers // full_attn_interval
        num_linear_layers = num_layers - num_full_attn_layers

    # MoE detection
    num_routed_experts = int(
        cfg.get("num_local_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or 0
    )
    num_experts_per_tok = int(
        cfg.get("num_experts_per_tok")
        or cfg.get("num_selected_experts")
        or (8 if num_routed_experts > 0 else 0)
    )
    moe_intermediate_size = int(
        cfg.get("moe_intermediate_size")
        or cfg.get("expert_ffn_dim")
        or intermediate_size
    )
    num_shared_experts = int(
        cfg.get("n_shared_experts")
        or cfg.get("num_shared_experts")
        or 0
    )
    first_k_dense_replace = int(cfg.get("first_k_dense_replace") or 0)
    moe_latent_size = int(cfg.get("moe_latent_size") or 0)
    num_moe_layers = (
        max(0, num_layers - first_k_dense_replace)
        if num_routed_experts > 0
        else 0
    )
    num_dense_layers = num_layers - num_moe_layers

    quant = detect_quantization(raw_cfg)

    return {
        "model_type": model_type,
        "architectures": architectures,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "rms_norm_eps": rms_norm_eps,
        "hidden_act": hidden_act,
        "uses_mla": uses_mla,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "uses_hybrid_linear_attention": num_linear_layers > 0,
        "num_linear_attention_layers": num_linear_layers,
        "num_full_attention_layers": num_full_attn_layers,
        "linear_num_key_heads": linear_num_k_heads,
        "linear_num_value_heads": linear_num_v_heads,
        "linear_key_head_dim": linear_k_dim,
        "linear_value_head_dim": linear_v_dim,
        "linear_conv_kernel_dim": linear_conv_dim,
        "uses_moe": num_routed_experts > 0,
        "num_routed_experts": num_routed_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "num_shared_experts": num_shared_experts,
        "first_k_dense_replace": first_k_dense_replace,
        "num_moe_layers": num_moe_layers,
        "num_dense_layers": num_dense_layers,
        "moe_latent_size": moe_latent_size,
        "quantization": quant,
    }


def compute_projection_tiles(
    projections: list[tuple[str, int, int, int]],
    max_bank_k: int,
    max_bank_n: int,
) -> tuple[list[dict[str, Any]], int]:
    """Compute (tk, tn, nk, nn, nt) for each projection family."""
    schedule: list[dict[str, Any]] = []
    total_tiles = 0
    for name, k_dim, n_dim, layer_count in projections:
        if k_dim <= 0 or n_dim <= 0 or layer_count <= 0:
            continue
        tk, tn = tile_hw(k_dim, n_dim, max_bank_k, max_bank_n)
        nk = k_dim // tk
        nn = n_dim // tn
        nt = nk * nn
        total_tiles += nt * layer_count
        schedule.append({
            "name": name,
            "k_dim": k_dim,
            "n_dim": n_dim,
            "tk": tk,
            "tn": tn,
            "nk": nk,
            "nn": nn,
            "tiles_per_layer": nt,
            "layer_count": layer_count,
            "total_tiles": nt * layer_count,
        })
    return schedule, total_tiles


def analyze_megakernel(
    raw_cfg: dict[str, Any],
    *,
    tpu_version: str = "v7x",
    tp_size: int = 8,
    batch_buckets: tuple[int, ...] = (1, 8),
    max_seq_len: int = 4096,
    bank_count: int = 12,
    max_bank_k: int = 1024,
    max_bank_n: int = 1280,
    expert_slots: int = 4,
    vmem_limit_bytes: int | None = None,
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the full TPU Pallas megakernel architecture specification."""
    if tpu_version not in TPU_SPECS:
        raise AnalyzerError(
            f"Unsupported tpu_version {tpu_version!r}; choose from {sorted(TPU_SPECS)}"
        )
    if tp_size <= 0 or (tp_size & (tp_size - 1)) != 0:
        raise AnalyzerError(f"tp_size must be a positive power of 2; got {tp_size}")

    hw = TPU_SPECS[tpu_version]
    model = parse_model_config(raw_cfg)
    quant = model["quantization"]
    w_bytes = float(quant["weight_bytes_per_elem"])
    s_bytes = float(quant["scale_bytes_per_elem"])
    eff_elem_bytes = w_bytes + s_bytes

    hidden_size = model["hidden_size"]
    num_layers = model["num_hidden_layers"]
    num_q_heads = model["num_attention_heads"]
    num_kv_heads = model["num_key_value_heads"]
    head_dim = model["head_dim"]
    inter_size = model["intermediate_size"]
    vocab_size = model["vocab_size"]

    q_heads_per_rank = max(1, num_q_heads // tp_size)
    kv_heads_per_rank = max(1, num_kv_heads // tp_size)
    q_dim_per_rank = q_heads_per_rank * head_dim
    kv_dim_per_rank = kv_heads_per_rank * head_dim
    inter_per_rank = max(128, inter_size // tp_size)
    vocab_per_rank = max(128, math.ceil(vocab_size / tp_size / 128) * 128)

    # Check 64-wide head packing optimization (Muse-Spark / Llama-style head_dim=64)
    pack_two_heads_per_row = (head_dim == 64) and (q_heads_per_rank % 2 == 0)

    alignment_warnings: list[str] = []
    if hidden_size % 128 != 0:
        alignment_warnings.append(
            f"hidden_size ({hidden_size}) is not a multiple of 128 lanes"
        )
    if inter_per_rank % 128 != 0:
        alignment_warnings.append(
            f"inter_per_rank ({inter_per_rank}) is not a multiple of 128 lanes; pad to {(inter_per_rank + 127) // 128 * 128}"
        )
    if q_dim_per_rank % 128 != 0 and not pack_two_heads_per_row:
        alignment_warnings.append(
            f"q_dim_per_rank ({q_dim_per_rank}) is not a multiple of 128 lanes"
        )

    # Build per-rank projection list for bank-ring tiling
    projections: list[tuple[str, int, int, int]] = []
    full_attn_layers = model["num_full_attention_layers"]
    linear_layers = model["num_linear_attention_layers"]
    dense_layers = model["num_dense_layers"]
    moe_layers = model["num_moe_layers"]

    if model["uses_mla"]:
        q_lora = model["q_lora_rank"] or 1536
        kv_lora = model["kv_lora_rank"]
        qk_nope = model["qk_nope_head_dim"] or 128
        qk_rope = model["qk_rope_head_dim"] or 64
        v_dim = model["v_head_dim"] or 128
        projections.extend([
            ("attn_wq_a", hidden_size, q_lora, full_attn_layers),
            ("attn_wq_b", q_lora, q_heads_per_rank * (qk_nope + qk_rope), full_attn_layers),
            ("attn_wkv_a", hidden_size, kv_lora + qk_rope, full_attn_layers),
            ("attn_wkv_b", kv_lora, q_heads_per_rank * (qk_nope + v_dim), full_attn_layers),
            ("attn_wo", q_heads_per_rank * v_dim, hidden_size, full_attn_layers),
        ])
    else:
        qkv_out = q_dim_per_rank * 2 + 2 * kv_dim_per_rank
        projections.extend([
            ("attn_qkv", hidden_size, qkv_out, full_attn_layers),
            ("attn_o", q_dim_per_rank, hidden_size, full_attn_layers),
        ])

    if linear_layers > 0:
        lin_k_heads = max(1, (model["linear_num_key_heads"] or num_kv_heads) // tp_size)
        lin_v_heads = max(1, (model["linear_num_value_heads"] or num_q_heads) // tp_size)
        lin_k_dim = model["linear_key_head_dim"] or head_dim
        lin_v_dim = model["linear_value_head_dim"] or head_dim
        qkvz_out = 2 * lin_k_heads * lin_k_dim + 2 * lin_v_heads * lin_v_dim
        projections.extend([
            ("lin_in_proj_qkvz", hidden_size, qkvz_out, linear_layers),
            ("lin_out_proj", lin_v_heads * lin_v_dim, hidden_size, linear_layers),
        ])

    if dense_layers > 0:
        projections.extend([
            ("mlp_gate_up", hidden_size, 2 * inter_per_rank, dense_layers),
            ("mlp_down", inter_per_rank, hidden_size, dense_layers),
        ])

    # LM head is always streamed through the bank ring
    projections.append(("lm_head", hidden_size, vocab_per_rank, 1))

    target_vmem_limit = vmem_limit_bytes or hw["default_vmem_limit_bytes"]
    max_batch = max(batch_buckets) if batch_buckets else 1

    # Persistent + KV + recurrent + collective VMEM usage
    persistent_vmem_bytes = (
        max_seq_len * head_dim * 4  # RoPE cos/sin
        + num_layers * hidden_size * 4 * 2  # layer norms
        + max_batch * hidden_size * 4 * 4  # act_acc, ffn_acc, residual scratch
        + max_batch * max(q_dim_per_rank, inter_per_rank) * 4 * 2  # q/inter accumulators
    )
    kv_chunk_seq = min(max_seq_len, 512)
    kv_streaming_vmem_bytes = (
        2 * 2 * kv_heads_per_rank * kv_chunk_seq * head_dim * 2
    )
    recurrent_state_vmem_bytes = 0
    if linear_layers > 0:
        lin_v_heads = max(1, (model["linear_num_value_heads"] or num_q_heads) // tp_size)
        lin_k_dim = model["linear_key_head_dim"] or head_dim
        lin_v_dim = model["linear_value_head_dim"] or head_dim
        recurrent_state_vmem_bytes = lin_v_heads * lin_k_dim * lin_v_dim * 4

    collective_scratch_vmem_bytes = (
        2 * max(max_batch, 8) * hidden_size * 4 if tp_size > 1 else 0
    )

    # Pool-aliased MoE VMEM calculation
    pool_alias_moe_vmem_bytes = 0
    moe_schedule: dict[str, Any] | None = None
    if model["uses_moe"]:
        moe_inter = model["moe_intermediate_size"]
        moe_in_dim = model["moe_latent_size"] or hidden_size
        # In TP+EP hybrid or TP-sharded experts, each rank streams its shard of active experts
        moe_inter_per_slot = max(128, moe_inter // max(1, tp_size // 4))
        w13_slot_bytes = int(moe_in_dim * (2 * moe_inter_per_slot) * eff_elem_bytes)
        w2_slot_bytes = int(moe_inter_per_slot * moe_in_dim * eff_elem_bytes)
        pool_alias_moe_vmem_bytes = expert_slots * (w13_slot_bytes + w2_slot_bytes)
        moe_schedule = {
            "strategy": "pool_alias_multi_slot_expert_ring",
            "expert_slots": expert_slots,
            "num_routed_experts": model["num_routed_experts"],
            "num_experts_per_tok": model["num_experts_per_tok"],
            "num_shared_experts": model["num_shared_experts"],
            "moe_in_dim": moe_in_dim,
            "moe_inter_per_slot": moe_inter_per_slot,
            "w13_slot_bytes": w13_slot_bytes,
            "w2_slot_bytes": w2_slot_bytes,
            "total_pool_alias_bytes": pool_alias_moe_vmem_bytes,
            "singleton_expert_compaction": max_batch > 1,
        }

    # Auto-tune bank_count / max_bank_k / max_bank_n if requested ring exceeds budget
    usable_vmem = int(target_vmem_limit * 0.85)
    fixed_vmem = (
        persistent_vmem_bytes
        + kv_streaming_vmem_bytes
        + recurrent_state_vmem_bytes
        + collective_scratch_vmem_bytes
        + pool_alias_moe_vmem_bytes
    )
    tuned_bank_count = bank_count
    tuned_max_k = max_bank_k
    tuned_max_n = max_bank_n

    def ring_bytes(bc: int, mk: int, mn: int) -> int:
        return int(bc * mk * mn * eff_elem_bytes)

    auto_tuned = False
    while (
        fixed_vmem + ring_bytes(tuned_bank_count, tuned_max_k, tuned_max_n)
        > usable_vmem
        and (tuned_bank_count > 2 or tuned_max_n > 256 or tuned_max_k > 256)
    ):
        auto_tuned = True
        if tuned_bank_count > 4:
            tuned_bank_count -= 2
        elif tuned_max_n > 512:
            tuned_max_n = max(256, tuned_max_n - 256)
        elif tuned_max_k > 512:
            tuned_max_k = max(256, tuned_max_k - 256)
        else:
            tuned_bank_count = max(2, tuned_bank_count - 1)
            break

    weight_bank_ring_vmem_bytes = ring_bytes(
        tuned_bank_count, tuned_max_k, tuned_max_n
    )
    total_vmem_bytes = fixed_vmem + weight_bank_ring_vmem_bytes
    recommended_vmem_limit_bytes = max(
        target_vmem_limit,
        min(
            hw["max_tunable_vmem_bytes"],
            int(math.ceil(total_vmem_bytes / (1024 * 1024) * 1.18)) * 1024 * 1024,
        ),
    )

    tile_schedule, total_tiles = compute_projection_tiles(
        projections, tuned_max_k, tuned_max_n
    )

    # Select in-kernel ICI collective strategy
    if tp_size == 1:
        collective_strategy = "none"
        collective_hops_per_allreduce = 0
    elif tp_size <= 8 and max_batch <= 2:
        collective_strategy = "xor_butterfly_allreduce"
        collective_hops_per_allreduce = int(math.log2(tp_size))
    elif tp_size <= 8:
        collective_strategy = "reduce_scatter_fp32_all_gather_bf16"
        collective_hops_per_allreduce = 2
    else:
        collective_strategy = "hierarchical_int16_shared_exponent"
        collective_hops_per_allreduce = int(math.log2(tp_size))

    # Estimate HBM roofline step time
    dense_weight_bytes_per_rank = sum(
        p["k_dim"] * p["n_dim"] * p["layer_count"] * eff_elem_bytes
        for p in tile_schedule
    )
    active_moe_bytes_per_rank = 0.0
    if moe_schedule is not None:
        active_experts = (
            model["num_experts_per_tok"] + model["num_shared_experts"]
        )
        bytes_per_expert = (
            moe_schedule["w13_slot_bytes"] + moe_schedule["w2_slot_bytes"]
        )
        active_moe_bytes_per_rank = float(
            moe_layers * active_experts * bytes_per_expert / max(1, min(tp_size, 8))
        )

    kv_read_bytes_per_rank = float(
        full_attn_layers * 2 * kv_heads_per_rank * min(max_seq_len, 1024) * head_dim * 2
    )
    total_hbm_bytes_per_step = (
        dense_weight_bytes_per_rank
        + active_moe_bytes_per_rank
        + kv_read_bytes_per_rank
    )
    hbm_floor_us = (total_hbm_bytes_per_step / 1e9) / hw["hbm_bw_gbps"] * 1e6
    allreduces_per_step = 2 * num_layers
    ici_overhead_us = (
        allreduces_per_step
        * collective_hops_per_allreduce
        * hw["ici_hop_latency_us"]
        * 0.35  # overlapped with next projection prefetch / non-dependent compute
    )
    estimated_step_time_us = hbm_floor_us + ici_overhead_us
    max_interactivity_tok_per_s = (
        1e6 / estimated_step_time_us if estimated_step_time_us > 0 else 0.0
    )

    streaming_strategy = (
        "bank_ring_plus_pool_alias" if model["uses_moe"] else "global_bank_ring"
    )

    return {
        "status": "ok",
        "model": model,
        "source_inspection": source_info or {"classes": [], "has_custom_attention": False, "has_moe": False},
        "hardware": hw,
        "sharding": {
            "tp_size": tp_size,
            "q_heads_per_rank": q_heads_per_rank,
            "kv_heads_per_rank": kv_heads_per_rank,
            "q_dim_per_rank": q_dim_per_rank,
            "kv_dim_per_rank": kv_dim_per_rank,
            "inter_per_rank": inter_per_rank,
            "vocab_per_rank": vocab_per_rank,
            "pack_two_heads_per_row": pack_two_heads_per_row,
            "alignment_warnings": alignment_warnings,
        },
        "weight_streaming": {
            "strategy": streaming_strategy,
            "bank_count": tuned_bank_count,
            "max_bank_k": tuned_max_k,
            "max_bank_n": tuned_max_n,
            "auto_tuned_for_vmem": auto_tuned,
            "total_stream_tiles": total_tiles,
            "tile_schedule": tile_schedule,
            "moe_pool_alias": moe_schedule,
        },
        "vmem_budget": {
            "vmem_capacity_bytes": hw["vmem_bytes"],
            "vmem_limit_bytes": recommended_vmem_limit_bytes,
            "fits_in_vmem": total_vmem_bytes <= recommended_vmem_limit_bytes,
            "utilization_ratio": round(
                total_vmem_bytes / float(recommended_vmem_limit_bytes), 4
            ),
            "total_allocated_vmem_bytes": total_vmem_bytes,
            "breakdown_bytes": {
                "persistent_vmem_bytes": persistent_vmem_bytes,
                "kv_streaming_vmem_bytes": kv_streaming_vmem_bytes,
                "recurrent_state_vmem_bytes": recurrent_state_vmem_bytes,
                "weight_bank_ring_vmem_bytes": weight_bank_ring_vmem_bytes,
                "pool_alias_moe_vmem_bytes": pool_alias_moe_vmem_bytes,
                "collective_scratch_vmem_bytes": collective_scratch_vmem_bytes,
            },
        },
        "collectives": {
            "strategy": collective_strategy,
            "tp_size": tp_size,
            "hops_per_allreduce": collective_hops_per_allreduce,
            "allreduces_per_step": allreduces_per_step,
            "primitives": (
                ["pltpu.make_async_remote_copy", "pltpu.SemaphoreType.DMA"]
                if tp_size > 1
                else []
            ),
        },
        "roofline": {
            "batch_buckets": list(batch_buckets),
            "dense_weight_bytes_per_rank": int(dense_weight_bytes_per_rank),
            "active_moe_bytes_per_rank": int(active_moe_bytes_per_rank),
            "kv_read_bytes_per_rank": int(kv_read_bytes_per_rank),
            "total_hbm_bytes_per_step": int(total_hbm_bytes_per_step),
            "hbm_floor_us": round(hbm_floor_us, 2),
            "ici_overlapped_overhead_us": round(ici_overhead_us, 2),
            "estimated_step_time_us": round(estimated_step_time_us, 2),
            "max_interactivity_tok_per_s": round(max_interactivity_tok_per_s, 1),
        },
    }


def render_pool_alias_py() -> str:
    """Render `pool_alias.py` for raw VMEM pool aliasing in Pallas TPU megakernels."""
    return '''# Copyright 2026 Google LLC
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
"""Raw VMEM byte-pool aliasing utility for Pallas TPU megakernels.

Allows heterogeneous projection tiles and multi-slot MoE expert buffers to
share a single flat uint32 VMEM allocation (`pltpu.VMEM((words, 128), jnp.uint32)`)
and reinterpret slices into typed 2D/3D Refs without triggering XLA/Mosaic VMEM OOMs.
"""

from __future__ import annotations

import math
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _dtype_bytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def storage_words(shapes_and_dtypes: list[tuple[tuple[int, ...], object]]) -> int:
    """Return the number of (words, 128) uint32 rows required for the largest view."""
    max_bytes = 0
    for shape, dtype in shapes_and_dtypes:
        numel = math.prod(shape)
        # Align each view to 128*4 = 512 bytes in VMEM
        view_bytes = math.ceil(numel * _dtype_bytes(dtype) / 512) * 512
        max_bytes = max(max_bytes, view_bytes)
    return max(8, math.ceil(max_bytes / 512) * 8)


def pool_scratch_spec(shapes_and_dtypes: list[tuple[tuple[int, ...], object]]):
    """Allocate a single raw VMEM uint32 scratchpad sized for the largest candidate view."""
    words = storage_words(shapes_and_dtypes)
    return pltpu.VMEM((words, 128), jnp.uint32)
'''


def render_megakernel_py(spec: dict[str, Any], op_slug: str) -> str:
    """Render a self-contained Pallas TPU megakernel module (`megakernel.py`)."""
    model = spec["model"]
    sharding = spec["sharding"]
    ws = spec["weight_streaming"]
    vmem = spec["vmem_budget"]

    hidden_size = model["hidden_size"]
    inter_per_rank = sharding["inter_per_rank"]
    bank_count = ws["bank_count"]
    max_bank_k = ws["max_bank_k"]
    max_bank_n = ws["max_bank_n"]
    vmem_limit = vmem["vmem_limit_bytes"]
    eps = model["rms_norm_eps"]
    act_fn = "gelu" if "gelu" in model["hidden_act"].lower() else "silu"

    return f'''# Copyright 2026 Google LLC
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
"""Generated Pallas TPU Megakernel for `{op_slug}` ({model["model_type"]}).

Synthesized by `megakernel_analyzer.py`:
  - HIDDEN_SIZE = {hidden_size}
  - INTER_PER_RANK = {inter_per_rank}
  - BANK_COUNT = {bank_count}, MAX_BANK_K = {max_bank_k}, MAX_BANK_N = {max_bank_n}
  - VMEM_LIMIT_BYTES = {vmem_limit}
  - Collective strategy = {spec["collectives"]["strategy"]}

Exposes `computation(x, w_gate_up, w_down)` for `tpu-kernel-development` and
`tpu_agent_suite_kernels` vLLM serving integration, plus `tile_hw` and
`reference_computation` for numerical parity verification.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HIDDEN_SIZE = {hidden_size}
INTER_PER_RANK = {inter_per_rank}
BANK_COUNT = {bank_count}
MAX_BANK_K = {max_bank_k}
MAX_BANK_N = {max_bank_n}
VMEM_LIMIT_BYTES = {vmem_limit}
RMS_NORM_EPS = {eps}
ACT_FN = "{act_fn}"


def tile_hw(k_dim: int, n_dim: int,
            max_k: int = MAX_BANK_K,
            max_n: int = MAX_BANK_N) -> tuple[int, int]:
    """Return largest 128-aligned (tk, tn) <= (max_k, max_n) dividing (k_dim, n_dim)."""
    tk = min(k_dim, max_k)
    while k_dim % tk != 0:
        tk -= 128 if tk > 128 else 1
    tn = min(n_dim, max_n)
    while n_dim % tn != 0:
        tn -= 128 if tn > 128 else 1
    return tk, tn


def _activate(x: jax.Array) -> jax.Array:
    if ACT_FN == "gelu":
        return jax.nn.gelu(x, approximate=True)
    return jax.nn.silu(x)


def reference_computation(x: jax.Array, w_gate_up: jax.Array,
                          w_down: jax.Array) -> jax.Array:
    """High-precision JAX reference oracle matching the fused decode block."""
    # w_gate_up: [2 * INTER_PER_RANK, HIDDEN_SIZE], w_down: [HIDDEN_SIZE, INTER_PER_RANK]
    gate_up = jnp.dot(
        x, w_gate_up.T, preferred_element_type=jnp.float32
    ).astype(x.dtype)
    inter = w_gate_up.shape[0] // 2
    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    hidden = (_activate(gate.astype(jnp.float32)).astype(x.dtype)) * up
    return jnp.dot(
        hidden, w_down.T, preferred_element_type=jnp.float32
    ).astype(x.dtype)


def _megakernel_body(x_hbm, w_gu_hbm, w_dn_hbm, out_hbm,
                     x_vmem, acc_gu_vmem, h_vmem, acc_out_vmem,
                     gu_banks, dn_banks, sem_gu, sem_dn):
    """Grid-less Pallas TPU megakernel body with Bank-Ring async DMA streaming."""
    m_dim = x_vmem.shape[0]
    k_dim = x_vmem.shape[1]
    inter_2 = w_gu_hbm.shape[0]
    inter = inter_2 // 2

    # 1. Load token activation into VMEM once
    copy_x = pltpu.make_async_copy(x_hbm, x_vmem, sem_gu.at[0])
    copy_x.start()
    copy_x.wait()

    tk_gu, tn_gu = tile_hw(k_dim, inter)
    nk_gu = k_dim // tk_gu
    nn_gu = inter // tn_gu

    # 2. Stream gate & up projections through VMEM bank ring and fuse activation in VMEM
    for n_idx in range(nn_gu):
        acc_gate = jnp.zeros((m_dim, tn_gu), dtype=jnp.float32)
        acc_up = jnp.zeros((m_dim, tn_gu), dtype=jnp.float32)
        for k_idx in range(nk_gu):
            slot_g = (2 * (n_idx * nk_gu + k_idx)) % BANK_COUNT
            slot_u = (2 * (n_idx * nk_gu + k_idx) + 1) % BANK_COUNT
            g_slice = w_gu_hbm.at[
                pl.ds(n_idx * tn_gu, tn_gu), pl.ds(k_idx * tk_gu, tk_gu)
            ]
            u_slice = w_gu_hbm.at[
                pl.ds(inter + n_idx * tn_gu, tn_gu), pl.ds(k_idx * tk_gu, tk_gu)
            ]
            cg = pltpu.make_async_copy(
                g_slice,
                gu_banks.at[slot_g, pl.ds(0, tn_gu), pl.ds(0, tk_gu)],
                sem_gu.at[slot_g],
            )
            cu = pltpu.make_async_copy(
                u_slice,
                gu_banks.at[slot_u, pl.ds(0, tn_gu), pl.ds(0, tk_gu)],
                sem_gu.at[slot_u],
            )
            cg.start()
            cu.start()
            cg.wait()
            cu.wait()

            x_tile = x_vmem[:, k_idx * tk_gu : (k_idx + 1) * tk_gu]
            w_g = gu_banks[slot_g, :tn_gu, :tk_gu]
            w_u = gu_banks[slot_u, :tn_gu, :tk_gu]
            acc_gate = acc_gate + lax.dot_general(
                x_tile,
                w_g,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            acc_up = acc_up + lax.dot_general(
                x_tile,
                w_u,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        act_tile = (
            _activate(acc_gate.astype(x_vmem.dtype).astype(jnp.float32)).astype(
                x_vmem.dtype
            )
            * acc_up.astype(x_vmem.dtype)
        )
        h_vmem[:, n_idx * tn_gu : (n_idx + 1) * tn_gu] = act_tile

    # 3. Stream down projection through VMEM bank ring accumulating in FP32
    tk_dn, tn_dn = tile_hw(inter, k_dim)
    nk_dn = inter // tk_dn
    nn_dn = k_dim // tn_dn

    for n_idx in range(nn_dn):
        acc_out = jnp.zeros((m_dim, tn_dn), dtype=jnp.float32)
        for k_idx in range(nk_dn):
            slot = (n_idx * nk_dn + k_idx) % BANK_COUNT
            dn_slice = w_dn_hbm.at[
                pl.ds(n_idx * tn_dn, tn_dn), pl.ds(k_idx * tk_dn, tk_dn)
            ]
            cd = pltpu.make_async_copy(
                dn_slice,
                dn_banks.at[slot, pl.ds(0, tn_dn), pl.ds(0, tk_dn)],
                sem_dn.at[slot],
            )
            cd.start()
            cd.wait()
            h_tile = h_vmem[:, k_idx * tk_dn : (k_idx + 1) * tk_dn]
            w_d = dn_banks[slot, :tn_dn, :tk_dn]
            acc_out = acc_out + lax.dot_general(
                h_tile,
                w_d,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        acc_out_vmem[:, n_idx * tn_dn : (n_idx + 1) * tn_dn] = acc_out.astype(
            x_vmem.dtype
        )

    # 4. Write final output back to HBM
    copy_out = pltpu.make_async_copy(acc_out_vmem, out_hbm, sem_dn.at[0])
    copy_out.start()
    copy_out.wait()


@functools.partial(jax.jit)
def computation(x: jax.Array, w_gate_up: jax.Array,
                w_down: jax.Array) -> jax.Array:
    """Execute the grid-less Pallas TPU megakernel with Bank-Ring DMA streaming."""
    m_dim, k_dim = x.shape
    inter = w_gate_up.shape[0] // 2
    tk_gu, tn_gu = tile_hw(k_dim, inter)
    tk_dn, tn_dn = tile_hw(inter, k_dim)

    return pl.pallas_call(
        _megakernel_body,
        out_shape=jax.ShapeDtypeStruct((m_dim, k_dim), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((m_dim, k_dim), x.dtype),
                pltpu.VMEM((m_dim, tn_gu), jnp.float32),
                pltpu.VMEM((m_dim, inter), x.dtype),
                pltpu.VMEM((m_dim, k_dim), x.dtype),
                pltpu.VMEM((BANK_COUNT, tn_gu, tk_gu), w_gate_up.dtype),
                pltpu.VMEM((BANK_COUNT, tn_dn, tk_dn), w_down.dtype),
                pltpu.SemaphoreType.DMA((BANK_COUNT,)),
                pltpu.SemaphoreType.DMA((BANK_COUNT,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(x, w_gate_up, w_down)


def main() -> None:
    """Verify numerical parity and benchmark the megakernel on TPU."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    m_dim = 16
    x = jax.random.normal(k1, (m_dim, HIDDEN_SIZE), dtype=jnp.bfloat16) * 0.1
    w_gu = (
        jax.random.normal(
            k2, (2 * INTER_PER_RANK, HIDDEN_SIZE), dtype=jnp.bfloat16
        )
        * 0.02
    )
    w_dn = (
        jax.random.normal(
            k3, (HIDDEN_SIZE, INTER_PER_RANK), dtype=jnp.bfloat16
        )
        * 0.02
    )

    ref = jax.jit(reference_computation)(x, w_gu, w_dn)
    out = computation(x, w_gu, w_dn)
    jax.block_until_ready((ref, out))

    max_abs = float(jnp.max(jnp.abs(ref.astype(jnp.float32) - out.astype(jnp.float32))))
    print(f"Parity check (m={{m_dim}}, k={{HIDDEN_SIZE}}, inter={{INTER_PER_RANK}}): max_abs={{max_abs:.6f}}")

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        y = computation(x, w_gu, w_dn)
        jax.block_until_ready(y)
    dt_us = (time.perf_counter() - t0) / iters * 1e6
    print(f"Megakernel latency: {{dt_us:.2f}} us/call")


if __name__ == "__main__":
    main()
'''


def render_op_module_py(
    spec: dict[str, Any], op_slug: str, target: str
) -> str:
    """Render `op_<slug>.py` fulfilling `tpu_agent_suite_kernels.OP_CONTRACT`."""
    model = spec["model"]
    buckets = tuple(spec["roofline"]["batch_buckets"])
    act_fn = "gelu" if "gelu" in model["hidden_act"].lower() else "silu"

    return f'''# Copyright 2026 Google LLC
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
"""`tpu_agent_suite_kernels` vLLM plugin op module for `{op_slug}`.

Synthesized by `megakernel_analyzer.py`. Package into a vLLM general plugin with:
  python3 build_plugin.py \\
    --out-dir <plugin_dir> \\
    --op {op_slug} \\
    --op-module op_{op_slug}.py \\
    --kernel megakernel.py \\
    --tar <plugin_dir>.tgz
"""

import jax
import jax.numpy as jnp
import torch

TARGET = "{target}"

# Decode and speculative-verification token buckets owned by the megakernel.
# Prefill buckets fall back to `fallback_fn`.
OWNED_BUCKETS = {buckets!r}

FLIPPED = "_tpu_weight_flipped"


def eligible(module):
    """Return None when `module` matches the megakernel's expected configuration."""
    gu = getattr(module, "gate_up_proj", None)
    dn = getattr(module, "down_proj", None)
    if gu is None or dn is None:
        return "missing gate_up_proj or down_proj"
    if getattr(gu, "bias", None) is not None or getattr(dn, "bias", None) is not None:
        return "bias not supported"
    if (gu.input_size % 128) or (dn.input_size_per_partition % 128):
        return "shard not 128-aligned"
    return None


def kernel_fn(x, w_gate_up, w_down):
    # pylint: disable=g-import-not-at-top
    from tpu_agent_suite_kernels import kernel_{op_slug} as kernel
    return kernel.computation(x, w_gate_up.T, w_down.T)


def fallback_fn(x: jax.Array, w_gate_up: jax.Array,
                w_down: jax.Array) -> jax.Array:
    """Exact fallback math matching vllm_torchtpu's [in, out] weight layout."""
    gate_up = jnp.dot(
        x, w_gate_up, preferred_element_type=jnp.float32
    ).astype(x.dtype)
    inter = w_gate_up.shape[1] // 2
    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    act = (
        jax.nn.gelu(gate.astype(jnp.float32), approximate=True)
        if "{act_fn}" == "gelu"
        else jax.nn.silu(gate.astype(jnp.float32))
    )
    h = act.astype(x.dtype) * up
    return jnp.dot(
        h, w_down, preferred_element_type=jnp.float32
    ).astype(x.dtype)


def fake(x: torch.Tensor, w_gate_up: torch.Tensor,
         w_down: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        x.shape[0], w_down.shape[1], dtype=x.dtype, device=x.device
    )


def call(module, op, x):
    gu, dn = module.gate_up_proj, module.down_proj
    if not (getattr(gu, FLIPPED, False) and getattr(dn, FLIPPED, False)):
        raise RuntimeError(
            "{op_slug} expects [in, out] weights flipped by vllm_torchtpu"
        )
    lead = x.shape[:-1]
    out = op(x.reshape(-1, x.shape[-1]), gu.weight, dn.weight)
    if getattr(dn, "reduce_results", False) and getattr(dn, "tp_size", 1) > 1:
        # pylint: disable=g-import-not-at-top
        from vllm.distributed import tensor_model_parallel_all_reduce
        out = tensor_model_parallel_all_reduce(out)
    return out.reshape(*lead, out.shape[-1])
'''


def render_case_module_py(spec: dict[str, Any], op_slug: str) -> str:
    """Render `case_<slug>.py` fulfilling `inplace_harness.CASE_CONTRACT`."""
    model = spec["model"]
    sharding = spec["sharding"]
    hidden_size = model["hidden_size"]
    inter_per_rank = sharding["inter_per_rank"]
    owned = list(spec["roofline"]["batch_buckets"])
    all_buckets = tuple(sorted(set(owned + [64, 256])))

    return f'''# Copyright 2026 Google LLC
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
"""`inplace_harness.py` verification case for `{op_slug}`."""

import torch

OP = "{op_slug}"
HIDDEN = {hidden_size}
INTER_PER_RANK = {inter_per_rank}
BUCKETS = {all_buckets!r}
ATOL = 2e-2
RTOL = 2e-2


class _LinearStub(torch.nn.Module):
    """Standalone vLLM-compatible linear stub for inplace_harness verification."""

    def __init__(self, in_features: int, out_features: int, tp_size: int = 1):
        super().__init__()
        self.input_size = in_features
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.bias = None
        self.reduce_results = False
        self.tp_size = tp_size
        # Stored in [out, in] before vllm_torchtpu flips to [in, out]
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
        )


class MegakernelDecodeBlock(torch.nn.Module):
    """Target module matching `OP_CONTRACT` for `{op_slug}`."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = _LinearStub(HIDDEN, 2 * INTER_PER_RANK)
        self.down_proj = _LinearStub(INTER_PER_RANK, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_gu = (
            self.gate_up_proj.weight
            if getattr(self.gate_up_proj, "_tpu_weight_flipped", False)
            else self.gate_up_proj.weight.t()
        )
        w_dn = (
            self.down_proj.weight
            if getattr(self.down_proj, "_tpu_weight_flipped", False)
            else self.down_proj.weight.t()
        )
        gate_up = torch.matmul(x.float(), w_gu.float()).to(x.dtype)
        gate, up = gate_up[:, :INTER_PER_RANK], gate_up[:, INTER_PER_RANK:]
        h = torch.nn.functional.silu(gate.float()).to(x.dtype) * up
        return torch.matmul(h.float(), w_dn.float()).to(x.dtype)


def build(quant_config):
    del quant_config
    return MegakernelDecodeBlock()


def inputs(bucket: int, generator: torch.Generator):
    return (
        torch.randn(bucket, HIDDEN, generator=generator).to(torch.bfloat16),
    )
'''


def sanitize_slug(raw_slug: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", raw_slug.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"mk_{slug}" if slug else "mk_decode"
    return slug[:48]


def emit_scaffold(
    out_dir: Path,
    spec: dict[str, Any],
    *,
    op_slug: str | None = None,
    target: str | None = None,
) -> dict[str, str]:
    """Write the megakernel spec, Pallas kernel, pool_alias, and vLLM plugin files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_type = spec["model"]["model_type"]
    slug = sanitize_slug(op_slug or f"{model_type}_megakernel")
    resolved_target = (
        target
        or f"vllm.model_executor.models.{model_type}:MegakernelDecodeBlock"
    )

    spec_path = out_dir / "megakernel_spec.json"
    pool_path = out_dir / "pool_alias.py"
    kernel_path = out_dir / "megakernel.py"
    op_path = out_dir / f"op_{slug}.py"
    case_path = out_dir / f"case_{slug}.py"

    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    pool_path.write_text(render_pool_alias_py(), encoding="utf-8")
    kernel_path.write_text(render_megakernel_py(spec, slug), encoding="utf-8")
    op_path.write_text(
        render_op_module_py(spec, slug, resolved_target), encoding="utf-8"
    )
    case_path.write_text(render_case_module_py(spec, slug), encoding="utf-8")

    return {
        "op_slug": slug,
        "target": resolved_target,
        "spec_json": str(spec_path),
        "pool_alias_py": str(pool_path),
        "megakernel_py": str(kernel_path),
        "op_module_py": str(op_path),
        "case_module_py": str(case_path),
    }


def parse_batch_buckets(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return (16, 32)
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError as e:
        raise AnalyzerError(f"Invalid --batch-buckets {raw!r}: {e}") from e
    if any(v <= 0 for v in vals):
        raise AnalyzerError("--batch-buckets values must be positive integers")
    return vals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Hugging Face config.json, model dir, or inline JSON string",
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=None,
        help="Optional PyTorch model source file for AST inspection",
    )
    parser.add_argument(
        "--tpu-version",
        default="v7x",
        choices=sorted(TPU_SPECS),
        help="Target Cloud TPU generation (default: v7x)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallelism degree across TPU devices (default: 8)",
    )
    parser.add_argument(
        "--batch-buckets",
        default="16,32",
        help="Comma-separated decode/verify token batch buckets (default: 16,32)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum KV cache sequence length per request (default: 4096)",
    )
    parser.add_argument(
        "--bank-count",
        type=int,
        default=12,
        help="Number of VMEM weight-streaming ring banks (default: 12)",
    )
    parser.add_argument(
        "--max-bank-k",
        type=int,
        default=1024,
        help="Maximum K tile dimension per VMEM weight bank (default: 1024)",
    )
    parser.add_argument(
        "--max-bank-n",
        type=int,
        default=1280,
        help="Maximum N tile dimension per VMEM weight bank (default: 1280)",
    )
    parser.add_argument(
        "--expert-slots",
        type=int,
        default=4,
        help="Number of pool_alias VMEM expert streaming slots for MoE models (default: 4)",
    )
    parser.add_argument(
        "--vmem-limit-bytes",
        type=int,
        default=None,
        help="Optional explicit VMEM budget in bytes",
    )
    parser.add_argument(
        "--op-slug",
        default=None,
        help="Optional op slug for generated tpu_agent_suite_kernels files",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional vLLM module.path:ClassName target for generated op module",
    )
    parser.add_argument(
        "--emit-scaffold",
        type=Path,
        default=None,
        help="Optional directory to emit pool_alias.py, megakernel.py, op_<slug>.py, and case_<slug>.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON analysis output",
    )
    args = parser.parse_args(argv)

    try:
        raw_cfg = load_raw_config(args.config)
        source_info = inspect_model_source(args.model_source)
        buckets = parse_batch_buckets(args.batch_buckets)
        spec = analyze_megakernel(
            raw_cfg,
            tpu_version=args.tpu_version,
            tp_size=args.tp_size,
            batch_buckets=buckets,
            max_seq_len=args.max_seq_len,
            bank_count=args.bank_count,
            max_bank_k=args.max_bank_k,
            max_bank_n=args.max_bank_n,
            expert_slots=args.expert_slots,
            vmem_limit_bytes=args.vmem_limit_bytes,
            source_info=source_info,
        )
        if args.emit_scaffold is not None:
            spec["scaffold"] = emit_scaffold(
                args.emit_scaffold,
                spec,
                op_slug=args.op_slug,
                target=args.target,
            )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
    except AnalyzerError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return EXIT_USAGE

    print(json.dumps(spec, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
