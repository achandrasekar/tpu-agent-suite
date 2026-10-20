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
    key = jax.random.key(0)
    rand_key = jax.random.key(42)
    ka, kb, kc = jax.random.split(rand_key, 3)
    k1, k2 = jax.random.split(key)
    batch_size, in_features, out_features = 4096, 8192, 8192
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    y = jax.random.uniform(k2, (batch_size, out_features), dtype=dtype)
    bmm_weight = jax.random.normal(ka, (out_features, in_features),
                                   dtype=dtype) * 0.02
    bmm_bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02
    in_weight = jnp.ones(out_features, dtype=dtype)
    in_bias = jax.random.normal(kc, out_features, dtype=dtype) * 0.02

    dynamic_args = [x, y, bmm_weight, bmm_bias, in_weight, in_bias]
    static_args = []
    return dynamic_args, static_args


# Computation
def computation(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    eps = 1e-5
    x = x @ bmm_weight.T + bmm_bias
    x = jnp.expand_dims(jnp.expand_dims(x, 2), 3)
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * jnp.reshape(in_weight,
                        (1, -1, 1, 1)) + jnp.reshape(in_bias, (1, -1, 1, 1))
    x = jnp.squeeze(jnp.squeeze(x, axis=3), axis=2)
    x = x + y
    x = x * y
    return x
