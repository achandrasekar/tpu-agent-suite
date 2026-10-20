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
"""Locks the slot rules that decide what the whole run is measured against.

The one that most needs a test is CUDA's: it is reference-only, and the way
that rule fails is silent. If a `.cu` ever becomes a primary candidate again,
the loop ports it, measures against the port, and reports a real speedup about
a computation nobody asked to be ported -- which is exactly the route that
measured worse than handing the CUDA to the planner as context. Nothing
downstream catches that, so it is asserted here.
"""

import os
import sys
import tempfile
import textwrap
import unittest

scripts_dir = os.path.dirname(__file__)
sys.path.insert(0, scripts_dir)

import classify_inputs  # noqa: E402

CUDA_KERNEL = textwrap.dedent("""
    __global__ void add_kernel(const float* a, const float* b, float* out) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      out[i] = a[i] + b[i];
    }
""")

TORCH_MODULE = textwrap.dedent("""
    import torch
    import torch.nn as nn


    class Model(nn.Module):

        def forward(self, a, b):
            return a + b


    def get_inputs():
        return [torch.randn(8), torch.randn(8)]
""")

JAX_MODULE = textwrap.dedent("""
    import jax.numpy as jnp


    def computation(a, b):
        return jnp.add(a, b)
""")

INLINE_CUDA_MODULE = TORCH_MODULE + textwrap.dedent("""
    from torch.utils.cpp_extension import load_inline

    cuda_sources = "__global__ void k() {}"
    ext = load_inline(name="ext", cpp_sources="", cuda_sources=cuda_sources)
""")


class SlotCandidacyTest(unittest.TestCase):

    def test_cuda_is_never_a_primary_candidate(self):
        # Even with a `__global__` entry point, which used to earn the slot.
        self.assertEqual(classify_inputs.slot_candidacy("cuda", True),
                         ["reference"])
        self.assertEqual(classify_inputs.slot_candidacy("cuda", False),
                         ["reference"])

    def test_torch_and_jax_keep_the_primary_slot(self):
        self.assertEqual(classify_inputs.slot_candidacy("pytorch", True),
                         ["primary"])
        self.assertEqual(classify_inputs.slot_candidacy("jax", True),
                         ["primary"])

    def test_inline_cuda_torch_module_still_fills_both_slots(self):
        self.assertEqual(
            classify_inputs.slot_candidacy("pytorch_with_inline_cuda", True),
            ["primary", "reference"])


class ProposeSlotsTest(unittest.TestCase):

    def _propose(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, text in files.items():
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                paths.append(path)
            records = classify_inputs.classify_paths(paths)
            return classify_inputs.propose_slots(records)

    def test_cuda_alone_is_refused_with_an_actionable_message(self):
        primary, references, ambiguity = self._propose({"k.cu": CUDA_KERNEL})
        self.assertIsNone(primary)
        self.assertIsNotNone(ambiguity)
        self.assertIn("CUDA is the only input supplied", ambiguity)
        self.assertIn("reference_paths", ambiguity)
        # Still reported as a reference, so the caller can be told where to
        # put it rather than only that it was rejected.
        self.assertEqual([r["kind"] for r in references], ["cuda"])

    def test_torch_primary_with_cuda_reference_resolves(self):
        primary, references, ambiguity = self._propose({
            "model.py": TORCH_MODULE,
            "k.cu": CUDA_KERNEL,
        })
        self.assertIsNone(ambiguity)
        self.assertEqual(primary["language"], "pytorch")
        self.assertEqual([r["kind"] for r in references], ["cuda"])

    def test_jax_primary_with_cuda_reference_resolves(self):
        primary, references, ambiguity = self._propose({
            "kernel.py": JAX_MODULE,
            "k.cu": CUDA_KERNEL,
        })
        self.assertIsNone(ambiguity)
        self.assertEqual(primary["language"], "jax")
        self.assertEqual([r["kind"] for r in references], ["cuda"])

    def test_kernelbench_layout_is_its_own_reference(self):
        primary, references, ambiguity = self._propose(
            {"task.py": INLINE_CUDA_MODULE})
        self.assertIsNone(ambiguity)
        self.assertEqual(primary["language"], "pytorch_with_inline_cuda")
        self.assertEqual([r["kind"] for r in references], ["cuda"])
        self.assertTrue(references[0]["embedded"])

    def test_no_entry_point_message_does_not_offer_cuda(self):
        _, _, ambiguity = self._propose({"notes.py": "x = 1\n"})
        self.assertIsNotNone(ambiguity)
        self.assertIn("only as a reference", ambiguity)


if __name__ == "__main__":
    unittest.main()
