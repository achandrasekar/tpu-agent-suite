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
    CONFIG = {
        'name': 'mixtral_8x7b_ragged_dot',
        'model': 'Mixtral-8x7B',
        'operator': 'ragged_dot',
        'num_groups': 8,
        'M': 8192,
        'K': 4096,
        'N': 14336,
    }
    dtype = jnp.bfloat16
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    G, M, K, N = CONFIG['num_groups'], CONFIG['M'], CONFIG['K'], CONFIG['N']
    x = jax.random.normal(k1, (G, M // G, K), dtype=dtype)
    weights = jax.random.normal(k2, (G, K, N), dtype=dtype) * 0.02
    dynamic_args = [x, weights]
    static_args = []
    return dynamic_args, static_args


# Computation
def computation(x, weights):
    return jnp.einsum('gmk,gkn->gmn', x, weights)
