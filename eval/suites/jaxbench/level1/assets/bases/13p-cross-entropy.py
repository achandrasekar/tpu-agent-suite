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
        'name': 'llama3_8b_cross_entropy',
        'model': 'Llama-3.1-8B',
        'operator': 'fused_cross_entropy',
        'batch_tokens': 8192,
        'hidden_dim': 4096,
        'vocab_size': 128256,
    }
    dtype = jnp.bfloat16
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG[
        'vocab_size']
    hidden = jax.random.normal(k1, (B, H), dtype=dtype)
    weight = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
    labels = jax.random.randint(k3, (B, ), 0, V)
    dynamic_args = [hidden, weight, labels]
    static_args = []
    return dynamic_args, static_args


# Computation
def computation(hidden, weight, labels):
    logits = jnp.dot(hidden, weight)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)
