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
"""torch <-> JAX bridge with the kernel/fallback choice made per bucket.

vLLM traces a model once with a symbolic token count and compiles that one
graph for every padded bucket with guard evaluation off, so a branch on the
token count in torch code is decided once and applied to every bucket. The
JAX function behind a `jax_op` is traced per concrete input shape instead,
so this is where the choice goes: owned buckets run the kernel, every other
bucket runs the fallback (the original math, in JAX).

That function runs as Python only while JAX traces it, once per shape, so it
is also where the evidence goes: each (bucket, path) is appended to `MARKERS`
and printed once, and nothing is printed from inside a compiled graph.
"""

from __future__ import annotations

import functools
import sys
import threading

MARKERS: list[dict] = []
_lock = threading.Lock()


def _record(name: str, bucket: int, path: str) -> None:
    with _lock:
        if any(m["op"] == name and m["bucket"] == bucket for m in MARKERS):
            return
        MARKERS.append({"op": name, "bucket": bucket, "path": path})
    print(f"tpu_agent_suite_kernels: {name} bucket={bucket} path={path}",
          file=sys.stderr,
          flush=True)


def dispatch(name, owned_buckets, kernel_fn, fallback_fn, bucket_of=None):
    """One JAX function: `kernel_fn` on owned buckets, `fallback_fn` elsewhere.

    It carries `fallback_fn`'s signature and annotations (`functools.wraps`):
    `jax_op` derives the torch op's schema from them, so `fallback_fn` must be
    annotated with array and scalar types, and `kernel_fn` must take the same
    arguments.
    """
    owned = frozenset(int(b) for b in owned_buckets)

    @functools.wraps(fallback_fn)
    def fn(*args, **kwargs):
        bucket = int(
            bucket_of(*args, **kwargs) if bucket_of else args[0].shape[0])
        path = "kernel" if bucket in owned else "original"
        _record(name, bucket, path)
        chosen = kernel_fn if path == "kernel" else fallback_fn
        return chosen(*args, **kwargs)

    fn.__name__ = f"tpu_agent_suite_{name}"
    fn.__qualname__ = fn.__name__
    return fn


def make_op(name, fn, fake):
    """Registers `fn` as the torch op `tpu_agent_suite::<name>` via torch_tpu's jax_op.

    No mesh is passed: under vLLM tensor parallelism on torch_tpu each process
    owns one device, and the op works on that rank's local shards, as
    vllm_torchtpu's own jax_op call sites do.
    """
    try:
        # pylint: disable=g-import-not-at-top
        from torch_tpu._internal import pallas
        jax_op = pallas.jax_op
    except ImportError:
        import torch  # pylint: disable=g-import-not-at-top
        jax_op = torch.tpu.pallas.jax_op
    op = jax_op(f"tpu_agent_suite::{name}", fn)
    op.register_fake(fake)
    return op
