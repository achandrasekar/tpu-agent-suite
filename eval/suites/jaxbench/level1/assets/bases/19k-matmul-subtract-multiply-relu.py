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
def get_inputs(dtype=jnp.float32):
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    subtract_value = 2.0
    multiply_value = 1.5

    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb = jax.random.split(rand_key, 2)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka,
                               (in_features, out_features), dtype=dtype) * 0.02
    bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02

    dynamic_args = [x, weight, bias]
    static_args = [subtract_value, multiply_value]
    return dynamic_args, static_args


# Computation
def computation(x, weight, bias, subtract_value, multiply_value):
    x = jnp.matmul(x, weight) + bias
    x = x - subtract_value
    x = x * multiply_value
    x = jax.nn.relu(x)
    return x
