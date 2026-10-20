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
"""Worked example op: Gemma-4's dense MLP through a fused Pallas decode kernel.

Installed by build_plugin.py as `tpu_agent_suite_kernels.op_gemma4_decode_mlp`, with
the kernel (tpu-kernel-development's best artifact, exposing `computation`) copied to
`tpu_agent_suite_kernels.kernel_gemma4_decode_mlp`.

The span is `Gemma4MLP.forward`: gate_up_proj (MergedColumnParallelLinear) ->
GeluAndMul(tanh) -> down_proj (RowParallelLinear, which all-reduces across
tensor-parallel ranks). The bridged op computes this rank's partial sum; the
all-reduce the replaced RowParallelLinear owned is re-issued in `call`.

Weights reach the op as the server holds them. vllm_torchtpu flips
unquantized linear weights from vLLM's [out, in] to [in, out] after loading
(the layer's `_tpu_weight_flipped`), so `fallback_fn` contracts against axis
0. The kernel was tuned on [out, in]: `kernel_fn` transposes for it, which
is a copy of both weights on every call, and the block timing charges it.
"""

import jax
import jax.numpy as jnp
import torch

TARGET = "vllm.model_executor.models.gemma4:Gemma4MLP"

# The kernel was tuned and tested at M = 1, 8, 32; decode runs at the padded
# buckets 16 and 32. Every other bucket (prefill: 64 and up) falls back.
OWNED_BUCKETS = (16, 32)

# Set by vllm_torchtpu on a linear layer whose weight it flipped to [in, out].
FLIPPED = "_tpu_weight_flipped"


def eligible(module):
    """None when this instance is the configuration the kernel implements."""
    # pylint: disable=g-import-not-at-top
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    gu, dn = module.gate_up_proj, module.down_proj
    if not (isinstance(gu.quant_method, UnquantizedLinearMethod)
            and isinstance(dn.quant_method, UnquantizedLinearMethod)):
        return "quantized weights"
    if gu.bias is not None or dn.bias is not None:
        return "bias"
    if getattr(module.act_fn, "approximate", None) != "tanh":
        return f"activation {type(module.act_fn).__name__}"
    if not dn.input_is_parallel:
        return "down_proj input is not parallel"
    if (gu.input_size % 128) or (dn.input_size_per_partition % 128):
        return "shard not 128-aligned"
    return None


def kernel_fn(x, w_gate_up, w_down):
    # pylint: disable=g-import-not-at-top
    from tpu_agent_suite_kernels import kernel_gemma4_decode_mlp as kernel
    return kernel.computation(x, w_gate_up.T, w_down.T)


def fallback_fn(x: jax.Array, w_gate_up: jax.Array,
                w_down: jax.Array) -> jax.Array:
    """The unpatched module's math, op for op, as vLLM computes it.

    gate_up_proj returns bf16, GeluAndMul applies gelu_tanh in f32 and casts
    back, down_proj returns bf16. Matching the casts is what makes the
    fallback buckets equal to the original, not merely close. Weights are
    [in, out].
    """
    gate_up = jnp.dot(x, w_gate_up,
                      preferred_element_type=jnp.float32).astype(x.dtype)
    inter = w_gate_up.shape[1] // 2
    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    h = jax.nn.gelu(gate.astype(jnp.float32), approximate=True).astype(
        x.dtype) * up
    return jnp.dot(h, w_down,
                   preferred_element_type=jnp.float32).astype(x.dtype)


def fake(x: torch.Tensor, w_gate_up: torch.Tensor,
         w_down: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0],
                       w_down.shape[1],
                       dtype=x.dtype,
                       device=x.device)


def call(module, op, x):
    gu, dn = module.gate_up_proj, module.down_proj
    # Read while torch.compile traces, so this costs nothing per call.
    if not (getattr(gu, FLIPPED, False) and getattr(dn, FLIPPED, False)):
        raise RuntimeError("gemma4_decode_mlp expects the [in, out] weights "
                           "vllm_torchtpu serves; these were not flipped")
    lead = x.shape[:-1]
    out = op(x.reshape(-1, x.shape[-1]), gu.weight, dn.weight)
    if dn.reduce_results and dn.tp_size > 1:
        # pylint: disable=g-import-not-at-top
        from vllm.distributed import tensor_model_parallel_all_reduce
        out = tensor_model_parallel_all_reduce(out)
    return out.reshape(*lead, out.shape[-1])
