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
"""inplace_harness case for the gemma4_decode_mlp example op.

One tensor-parallel rank of Gemma-4-31B served at TP=4: hidden 5376,
intermediate 21504 / 4 = 5376, bf16. The block the harness times is what the
decoder layer runs around the MLP -- pre-feedforward RMSNorm, the MLP,
post-feedforward RMSNorm, residual add -- because XLA fuses the pre-norm into
the gate_up matmul, and a kernel that wins alone can lose that fusion.
"""

import torch

OP = "gemma4_decode_mlp"
HIDDEN = 5376
INTER_PER_RANK = 21504 // 4
EPS = 1e-6
# Decode buckets the kernel owns, and prefill-sized ones that must fall back.
BUCKETS = (16, 32, 64, 256)
ATOL = 2e-2
RTOL = 2e-2


def build(quant_config):
    # pylint: disable=g-import-not-at-top
    from vllm.model_executor.models import gemma4
    return gemma4.Gemma4MLP(HIDDEN,
                            INTER_PER_RANK,
                            "gelu_pytorch_tanh",
                            quant_config=quant_config,
                            prefix="model.layers.0.mlp")


class Block(torch.nn.Module):
    """pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> +x."""

    def __init__(self, mlp):
        super().__init__()
        # pylint: disable=g-import-not-at-top
        from vllm.model_executor.models import gemma4
        self.pre = gemma4.RMSNorm(HIDDEN, eps=EPS)
        self.mlp = mlp
        self.post = gemma4.RMSNorm(HIDDEN, eps=EPS)

    def forward(self, x):
        return self.post(self.mlp(self.pre(x))) + x


def build_block(mlp):
    return Block(mlp)


def inputs(bucket, generator):
    return (torch.randn(bucket, HIDDEN,
                        generator=generator).to(torch.bfloat16), )
