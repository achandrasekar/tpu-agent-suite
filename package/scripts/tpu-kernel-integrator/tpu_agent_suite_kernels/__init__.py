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
"""Optimized kernels in a served vLLM model, switched on by one variable.

vLLM loads this package in every process -- API server, engine core, each
tensor-parallel worker -- through its `vllm.general_plugins` entry point.
`register()` does nothing unless `TPU_AGENT_SUITE_KERNELS` names at least one op, so
the same package can sit on PYTHONPATH for a baseline run and an optimized run
and only the variable differs between them.

Each op is a module `tpu_agent_suite_kernels.op_<op>` (see `OP_CONTRACT`). For an
enabled op, `register()` patches the op's target class:

  * `__init__` runs the original, then asks the op whether this instance is
    one the kernel implements (`eligible`), records the answer on the
    instance and logs `tpu_agent_suite_kernels: <op> <prefix> KERNEL|ORIGINAL`;
  * `forward` runs the op's replacement for eligible instances and the
    original for every other.

The replacement calls one torch op bridged to JAX (`bridge.make_op`). The
JAX function behind it chooses kernel or fallback from the concrete shape, so
the choice is made per compiled bucket rather than once at trace time.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading

from . import bridge

ENV = "TPU_AGENT_SUITE_KERNELS"

# Names an op module must define. `bucket_of` is optional.
OP_CONTRACT = (
    "TARGET",  # "module.path:ClassName" whose forward is replaced
    "OWNED_BUCKETS",  # token counts (dim 0) the kernel serves
    "eligible",  # eligible(module) -> None, or the reason it is not
    "kernel_fn",  # JAX: the optimized kernel
    "fallback_fn",  # JAX: the same math as the original, for other buckets
    "fake",  # torch: output metadata for torch.compile tracing
    "call",  # call(module, op, *args, **kwargs) -> output
)

_lock = threading.Lock()
_registered: dict[str, object] = {}
_ops: dict[str, object] = {}


def log(msg: str) -> None:
    print(f"tpu_agent_suite_kernels: {msg}", file=sys.stderr, flush=True)


def enabled_ops(env: dict | None = None) -> list[str]:
    value = (env if env is not None else os.environ).get(ENV, "")
    return [op.strip() for op in value.split(",") if op.strip()]


def load_op(name: str):
    module = importlib.import_module(f"{__name__}.op_{name}")
    missing = [n for n in OP_CONTRACT if not hasattr(module, n)]
    if missing:
        raise AttributeError(f"op {name} does not define {missing}")
    return module


def _target_class(spec: str):
    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def bridged_op(name: str):
    """The torch op for `name`, built once per process outside tracing."""
    op = _ops.get(name)
    if op is not None:
        return op
    with _lock:
        if name not in _ops:
            mod = load_op(name)
            _ops[name] = bridge.make_op(
                name,
                bridge.dispatch(name, mod.OWNED_BUCKETS,
                                mod.kernel_fn, mod.fallback_fn,
                                getattr(mod, "bucket_of", None)), mod.fake)
        return _ops[name]


def patch(name: str) -> object:
    """Patch the op's target class; returns the class. Idempotent."""
    with _lock:
        if name in _registered:
            return _registered[name]
    mod = load_op(name)
    cls = _target_class(mod.TARGET)
    orig_init, orig_forward = cls.__init__, cls.forward
    flag = f"_tpu_agent_suite_{name}"
    op_attr = f"_tpu_agent_suite_{name}_op"

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        reason = mod.eligible(self)
        setattr(self, flag, reason is None)
        prefix = kwargs.get("prefix", "") or getattr(self, "prefix", "")
        if reason is None:
            # Built here, at model construction, and kept on the instance:
            # forward only reads it. vLLM compiles the model as one graph, and
            # anything forward does that Dynamo cannot trace -- taking a lock
            # to build the op, for one -- breaks that graph.
            setattr(self, op_attr, bridged_op(name))
            log(f"{name} {prefix} KERNEL")
        else:
            log(f"{name} {prefix} ORIGINAL ({reason})")

    def forward(self, *args, **kwargs):
        if getattr(self, flag, False):
            return mod.call(self, getattr(self, op_attr), *args, **kwargs)
        return orig_forward(self, *args, **kwargs)

    __init__.__wrapped__ = orig_init
    forward.__wrapped__ = orig_forward
    cls.__init__, cls.forward = __init__, forward
    with _lock:
        _registered[name] = cls
    log(f"{name} patched {mod.TARGET} (pid {os.getpid()})")
    return cls


def register() -> None:
    """vLLM general-plugin entry point."""
    for name in enabled_ops():
        try:
            patch(name)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # A plugin that cannot patch must not take the server down with an
            # unrelated error; the caller's start-up gate sees no KERNEL lines.
            log(f"{name} NOT PATCHED: {type(e).__name__}: {e}")
