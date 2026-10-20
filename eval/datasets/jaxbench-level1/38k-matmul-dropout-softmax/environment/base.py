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

# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    dtype = jnp.float32

    key = jax.random.key(0)
    rand_key = jax.random.key(42)
    ka, kb = jax.random.split(rand_key, 2)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka,
                               (out_features, in_features), dtype=dtype) * 0.02
    bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02

    dynamic_args = [x, weight, bias]
    static_args = []

    return dynamic_args, static_args


# Computation
def computation(x, weight, bias):
    x = x @ weight.T + bias
    x = jax.nn.softmax(x, axis=1)
    return x
