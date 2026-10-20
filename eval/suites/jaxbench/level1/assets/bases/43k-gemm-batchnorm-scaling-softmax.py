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
    ka, kb, kc = jax.random.split(rand_key, 3)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka,
                               (in_features, out_features), dtype=dtype) * 0.02
    bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02
    bn_scale = jnp.ones(out_features, dtype=dtype)
    bn_bias = jax.random.normal(kc, out_features, dtype=dtype) * 0.02
    bn_mean = jnp.zeros(out_features, dtype=dtype)
    bn_var = jnp.ones(out_features, dtype=dtype)
    scale = jnp.ones((1, ), dtype=dtype)

    dynamic_args = [x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale]
    static_args = []
    return dynamic_args, static_args


# Computation
def computation(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    bn_eps = 1e-5
    x = jnp.matmul(x, weight) + bias
    x_normalized = (x - bn_mean) / jnp.sqrt(bn_var + bn_eps)
    x = bn_scale * x_normalized + bn_bias
    x = scale * x
    x = jax.nn.softmax(x, axis=1)
    return x
