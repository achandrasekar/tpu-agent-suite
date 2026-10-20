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
"""The original `attention()`, frozen, as the reference for parity checking.

Not a secret: it is the code the agent is handed on line one of the task. It
lives here so the verifier has a copy the agent cannot edit -- run 13 rewrote
the workload wholesale, which would leave nothing to compare against if the
reference came from the same file.

Runs on CPU. The function means the same thing on any backend, so equivalence
needs no chip; only the timing does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def attention(q, k, v):
    """Causal multi-head attention. (HEADS, SEQ, DIM), bfloat16 in and out."""
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = jnp.einsum("hqd,hkd->hqk", q, k) * scale
    causal = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    return jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v)
