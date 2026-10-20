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
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    D, H, W = 32, 64, 64

    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb = jax.random.split(rand_key, 2)
    x = jax.random.uniform(key, (batch_size, in_channels, D, H, W),
                           dtype=dtype)
    weight = jax.random.normal(
        ka, (out_channels, in_channels, kernel_size, kernel_size, kernel_size),
        dtype=dtype) * 0.02
    bias = jax.random.normal(kb, out_channels, dtype=dtype) * 0.02

    dynamic_args = [x, weight, bias]
    static_args = []
    return dynamic_args, static_args


# Computation
def computation(x, weight, bias):
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    x = jax.lax.conv_general_dilated(x,
                                     kernel,
                                     window_strides=(1, 1, 1),
                                     padding=((0, 0), (0, 0), (0, 0)),
                                     dimension_numbers=('NDHWC', 'DHWIO',
                                                        'NDHWC'))
    x = x + bias.reshape(1, 1, 1, 1, -1)
    x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
    x = jnp.tanh(x)
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    return x
