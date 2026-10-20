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
"""Fakes shared by this directory's tests: a target module, op and case.

They stand in for vLLM, JAX and torch_tpu so the plugin, the builder and the
harness can be tested on a CPU host with torch alone.
"""

import itertools
import os
import sys
import textwrap

TARGET_MODULE = '''
import torch


class FlipMethod:
    """vllm_torchtpu's linear method: once loaded, W is flipped to [in, out]."""

    def process_weights_after_loading(self, layer):
        layer.weight = torch.nn.Parameter(layer.weight.data.t().contiguous(),
                                          requires_grad=False)
        layer.flipped = True


class Mlp(torch.nn.Module):
    """A vLLM-style module: W is created [out, in] and served [in, out]."""

    # Set by the "fast" op, so the original path is the slow one. Slowed with
    # repeated work rather than a sleep: tests compile with fullgraph=True,
    # and a sleep would break the graph.
    slow = False

    def __init__(self, hidden=8, eligible=True, prefix=""):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden, hidden))
        self.quant_method = FlipMethod()
        self.flipped = False
        self.is_eligible = eligible
        self.prefix = prefix

    def _matmul(self, x):
        return x @ (self.weight if self.flipped else self.weight.T)

    def forward(self, x):
        out = torch.relu(self._matmul(x))
        if Mlp.slow:
            for _ in range(300):
                out = torch.relu(self._matmul(x))
        return out
'''

# KERNEL selects the op's kernel: "same" (the original math at the original
# speed), "wrong" (off by one), "fast" (the original math, while the module's
# own forward is made slow, so the op has something to win), "breaks" (the
# right math, but `call` does something torch.compile cannot trace),
# "unloaded" (written for W as created, [out, in], not as served).
OP_MODULE = '''
import threading

import torch

import fake_target

TARGET = "fake_target:Mlp"
OWNED_BUCKETS = (2, 4)
KERNEL = "{kernel}"
fake_target.Mlp.slow = KERNEL == "fast"


def eligible(module):
    return None if module.is_eligible else "not eligible"


def kernel_fn(x, w):
    out = torch.relu(x @ (w.T if KERNEL == "unloaded" else w))
    return out + 1 if KERNEL == "wrong" else out


def fallback_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # W as the server holds it: [in, out].
    return torch.relu(x @ w)


def fake(x, w):
    return torch.empty(x.shape[0], w.shape[1], dtype=x.dtype)


_lock = threading.Lock()


def call(module, op, x):
    if KERNEL == "breaks":
        with _lock:
            pass
    return op(x, module.weight)
'''

CASE_MODULE = '''
import torch

OP = "fake_op"
BUCKETS = (2, 4, 8)
DTYPE = torch.float32
ATOL = 1e-5
RTOL = 1e-5


def build(quant_config):
    import fake_target
    return fake_target.Mlp()


def inputs(bucket, generator):
    return (torch.randn(bucket, 8, generator=generator),)
'''


def write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(text))
    return path


_OP_IDS = itertools.count()


def python_make_op(name, fn, fake):
    """Stands in for torch_tpu's jax_op: a torch custom op over `fn`.

    Opaque to torch.compile, as a jax_op is, so a compiled block sees one
    node and the dispatch inside it runs at call time.
    """
    import torch  # pylint: disable=import-outside-toplevel
    op = torch.library.custom_op(
        f"tpu_agent_suite_test::{name}_{next(_OP_IDS)}", mutates_args=())(fn)
    op.register_fake(fake)
    return op


def forget_plugin_modules():
    for mod in list(sys.modules):
        if mod == "tpu_agent_suite_kernels" or mod.startswith(
            ("tpu_agent_suite_kernels.", "fake_target")):
            del sys.modules[mod]
