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
"""The deployment (from the server command) and the per-rank shapes it implies.

`Facts` is the model in canonical terms. Each model's `model.py` builds it from the
checkpoint's config.json, because key names differ between families. `Deploy` is the
served configuration, read from `vllm serve` arguments. `per_rank()` is the arithmetic
that turns the two into what ONE rank of the deployment sees, which is what every L1
benchmark is shaped from. There is no roofline pricing here, only division.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Facts:
    name: str
    layers: int
    dense_layers: int  # first layers with a dense FFN
    hidden: int
    heads: int  # query heads
    kv_heads: int  # MLA: equals heads in config; the cache has no head axis
    mla: bool
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int = 0
    experts: int = 0
    topk: int = 0
    expert_ffn: int = 0
    shared_ffn: int = 0  # n_shared_experts * expert width
    dense_ffn: int = 0
    indexer_heads: int = 0
    indexer_head_dim: int = 0
    indexer_topk: int = 0
    indexer_layer_ids: tuple[int, ...] = ()
    indexer_cache_bytes_per_token: int = 0
    kv_cache_bytes_per_token_per_layer: int = 0  # at the served KV dtype
    mtp_layers: int = 0
    vocab: int = 0
    # Filled by the model's model.py, never computed here: core only counts and divides.
    kinds: tuple[str, ...] = (
    )  # layer kind of each layer, in order (model's own names)
    kv_bytes_token: int = 0  # KV-cache bytes per token, all layers, before sharding
    state_bytes_seq: int = 0  # recurrent (e.g. linear-attention) state per sequence, all layers
    mtp_kv_bytes_token: int = 0  # KV bytes per token of the MTP / draft layers

    @property
    def layer_kinds(self) -> dict[str, int]:
        """Layer count per kind, in order of first appearance (kind names are the model's)."""
        out: dict[str, int] = {}
        for k in self.kinds:
            out[k] = out.get(k, 0) + 1
        return out

    @property
    def kv_bytes_per_token(self) -> int:
        return self.kv_bytes_token


@dataclass(frozen=True)
class Deploy:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    ep: bool = False
    interleave: int = 1
    block_size: int = 16
    mnbt: int = 8192
    mns: int = 256
    max_model_len: int = 0
    kv_cache_memory_bytes: int = 0
    kv_cache_dtype: str = "auto"
    spec_method: str | None = None
    spec_tokens: int = 0
    prefix_caching: bool = False
    dcp_q_replicate: bool = True

    @property
    def world(self) -> int:
        return self.tp * self.dp * self.pp * self.pcp

    @property
    def ep_group(self) -> int:
        return self.tp * self.dp * self.pcp if self.ep else 1

    @property
    def cache_block_tokens(self) -> int:
        """Prefix-cache hash granularity: block x DCP (vLLM kv_cache_utils.py:649-651)."""
        return self.block_size * self.dcp

    @property
    def eagle_drop(self) -> bool:
        """vLLM's prefix cache drops one whole block from every hit for eagle-style drafters,
        MTP included."""
        return is_eagle_like(self.spec_method) and self.spec_tokens > 0

    def label(self) -> str:
        s = f"tp{self.tp}-dp{self.dp}"
        for k in ("pp", "dcp", "pcp"):
            if getattr(self, k) > 1:
                s += f"-{k}{getattr(self, k)}"
        return s + ("-ep" if self.ep else "")


def is_eagle_like(method: str | None) -> bool:
    """Spec methods vLLM runs as an eagle-style drafter (eagle, eagle3, and every *mtp variant,
    e.g. mtp, deepseek_mtp, qwen3_next_mtp)."""
    m = (method or "").lower()
    return "eagle" in m or "mtp" in m


def _b(v: Any) -> bool:
    return v is True or str(v).lower() in ("1", "true", "yes")


def deploy_from_args(args: dict[str, Any],
                     env: dict[str, str] | None = None) -> Deploy:
    env = env or {}
    spec = args.get("speculative-config")
    spec = json.loads(spec) if isinstance(spec, str) else (spec or {})
    q_rep = not ("no-dcp-q-replicate" in args
                 or env.get("VLLM_DCP_Q_REPLICATE") == "0")
    return Deploy(
        tp=int(args.get("tensor-parallel-size", 1)),
        dp=int(args.get("data-parallel-size", 1)),
        pp=int(args.get("pipeline-parallel-size", 1)),
        dcp=int(args.get("decode-context-parallel-size", 1)),
        pcp=int(args.get("prefill-context-parallel-size", 1)),
        ep=_b(args.get("enable-expert-parallel", False)),
        interleave=int(args.get("cp-kv-cache-interleave-size", 1)),
        block_size=int(args.get("block-size", 16)),
        mnbt=int(args.get("max-num-batched-tokens", 8192)),
        mns=int(args.get("max-num-seqs", 256)),
        max_model_len=int(args.get("max-model-len", 0)),
        kv_cache_memory_bytes=int(float(args.get("kv-cache-memory-bytes", 0))),
        kv_cache_dtype=str(args.get("kv-cache-dtype", "auto")),
        spec_method=spec.get("method"),
        spec_tokens=int(spec.get("num_speculative_tokens", 0) or 0),
        prefix_caching=_b(args.get("enable-prefix-caching", False)),
        dcp_q_replicate=q_rep,
    )


def apply_overrides(d: Deploy, overrides: dict[str, str]) -> Deploy:
    """`dcp=8`, `tp=16`, `block_size=64`, `mnbt=8192`, `max_model_len=524288`, ... Unknown
    keys are an error: a typo must not silently derive the base config."""
    fields = {f for f in d.__dataclass_fields__}
    kw = {}
    for k, v in overrides.items():
        if k.startswith("env."):
            continue
        if k not in fields:
            raise ValueError(
                f"unknown override {k!r}; known: {sorted(fields)} or env.<NAME>"
            )
        cur = getattr(d, k)
        kw[k] = _b(v) if isinstance(
            cur, bool) else (type(cur)(v) if cur is not None else v)
    return replace(d, **kw)


def kv_per_token_rank(f: Facts, d: Deploy, hook=None) -> float:
    """Paged-KV bytes one rank holds per token. Default: the whole token's KV / DCP (MLA keeps
    the latent whole on every TP rank). A model whose KV splits over TP (GQA heads, replicated
    past KV-head count) passes its own `kv_bytes_per_token_per_rank(f, d)` (model.py)."""
    return hook(f, d) if hook else f.kv_bytes_per_token / d.dcp


def per_rank(f: Facts,
             d: Deploy,
             pool_tokens: int | None = None,
             extra=None,
             kv_rank=None) -> dict[str, Any]:
    """What one rank sees. Every value is global / degree; nothing is priced. Generic rows only;
    model-specific rows (attention variant, indexer, recurrent state) come from the model's
    `per_rank_rows(f, d, pool_tokens)` passed as `extra`, and per-rank KV from its
    `kv_bytes_per_token_per_rank` passed as `kv_rank`."""
    tp_attn = max(1, d.tp)
    out: dict[str, Any] = {
        "tokens_per_step":
        f"bucket (<= {d.mnbt}), replicated"
        if d.dp == 1 else f"bucket / {d.dp} (DP)",
        "q_heads_projected":
        f.heads // tp_attn,
        "kv_tokens_per_rank":
        f"L / {d.dcp}" if d.dcp > 1 else "L",
        "kv_divisor":
        d.dcp,
        "cache_block_tokens":
        d.cache_block_tokens,
        "experts_per_rank":
        f.experts // d.ep_group if f.experts else 0,
        "routed_pairs_per_rank":
        f"T x {f.topk} / {d.ep_group} (mean; use the busiest rank)"
        if f.experts else None,
        "kv_bytes_per_token_per_rank":
        round(kv_per_token_rank(f, d, kv_rank), 1),
    }
    if f.dense_ffn:
        out["dense_ffn_per_rank"] = f.dense_ffn // d.tp
    if f.shared_ffn:
        out["shared_expert_ffn_per_rank"] = f.shared_ffn // d.tp
    if pool_tokens:
        out["kv_pool_tokens"] = pool_tokens
        out["kv_pool_tokens_per_rank"] = pool_tokens // d.dcp
        out["pages_per_rank"] = pool_tokens // d.dcp // d.block_size
    if extra:
        out.update(extra(f, d, pool_tokens))
    return out


def to_json(obj: Any) -> Any:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
