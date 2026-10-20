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
"""Causal multi-head attention, and a harness that profiles whatever it does.

Edit `attention()`. Everything below it is measurement scaffolding: it fixes
the shapes, checks the result still matches what this version produces, writes
an XProf trace, and prints timings. Run it once per variant with a different
--label so the traces do not overwrite each other.

    python3 attention.py --label baseline
"""

from __future__ import annotations

import hashlib
import os
import re

# Keep this block, and keep it above `import jax`. Without it the profile still
# writes, still exits 0 and still looks complete -- it just carries no low-level
# instruction data, which is not visible until something asks for it.
#
# Profiling flags belong to the TPU runtime and are read when the backend comes
# up, so they are set before jax is imported. Appended, not assigned: a TPU node
# injects its own LIBTPU_INIT_ARGS (--xla_tpu_use_enhanced_launch_barrier) into
# the container, so `setdefault` silently keeps that value and never sets ours,
# while a plain assignment would drop a flag the node needs.
os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    filter(None, (
        os.environ.get("LIBTPU_INIT_ARGS", "").strip(),
        "--xla_xprof_register_llo_debug_info=true",
    )))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HEADS, SEQ, DIM = 8, 4096, 128


def attention(q, k, v):
    """Causal multi-head attention.

  q, k, v are each (HEADS, SEQ, DIM), bfloat16. Returns the same shape.
  """
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = jnp.einsum("hqd,hkd->hqk", q, k) * scale
    causal = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    return jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v)


# --------------------------------------------------------------------------
# Measurement scaffolding. Editing below this line is allowed but not the point.
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label",
                   required=True,
                   help="names this variant's trace directory")
    p.add_argument("--logdir",
                   default="/logs/artifacts/xprof",
                   help="collected off the machine; elsewhere is discarded")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    shape = (HEADS, SEQ, DIM)
    q = jax.random.normal(kq, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(kk, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(kv, shape, dtype=jnp.bfloat16)

    fn = jax.jit(attention)
    out = jax.block_until_ready(fn(q, k, v))  # compile

    # Read from this file's own text, before the record is built, so a failure
    # here cannot lose the run.
    try:
        _src = open(__file__, encoding="utf-8").read()
        _src_sha = hashlib.sha256(_src.encode("utf-8")).hexdigest()[:16]
        # Patterns assembled from fragments so this file cannot match itself:
        # spelled out, the detector reported the untouched baseline as
        # containing a kernel, because the regex is part of the source it reads.
        # The experimental-kernels module path belongs here and was dropped
        # when these were rewritten as fragments: run 19's agent imported
        # splash attention from it -- a TPU kernel by any reading -- and the run
        # still reported that no kernel API was attempted.
        #
        # Note the shape of every pattern below: assembled from pieces so this
        # file never contains the tokens it searches for. Spelling any of them
        # out, in code *or in a comment*, makes the untouched baseline match
        # itself. That has now happened twice.
        _kern = "|".join(("pal" + "las_call", "Block" + "Spec", "plt" + "pu",
                          r"experimental\." + "pallas"))
        _lib = "|".join(
            ("splash_" + "attention", "dot_product_" + "attention"))
        _src_kernel = bool(re.search(_kern, _src))
        _src_library = bool(re.search(_lib, _src))
    except OSError:
        _src_sha, _src_kernel, _src_library = None, None, None

    # Counted before the record is built so a failure here cannot lose the run.
    try:
        _device_count = len(jax.devices())
    except Exception:  # noqa: BLE001
        _device_count = None
    try:
        _output_devices = len(out.sharding.device_set)
    except Exception:  # noqa: BLE001
        _output_devices = None

    logdir = os.path.join(args.logdir, args.label)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"
    }

    jax.profiler.start_trace(logdir, profiler_options=options)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = fn(q, k, v)
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - t0
    jax.profiler.stop_trace()

    # Written beside the trace, not just printed. The verifier reads this file
    # rather than the agent's prose: run 13 shipped an `attention()` that was
    # 9.4x faster and computed something else, and said so in its own output
    # (checksum -47726.24 against the original's -49370.52) while still
    # reporting it as the result. A number in a log is a claim; a number in the
    # collected artifacts is evidence.
    record = {
        "label": args.label,
        "device": str(jax.devices()[0].device_kind),
        "shape": {
            "heads": HEADS,
            "seq": SEQ,
            "dim": DIM
        },
        "iters": args.iters,
        "wall_per_iter_ms": round(elapsed / args.iters * 1e3, 4),
        "output_checksum": float(jnp.sum(out.astype(jnp.float32))),
        # A sum alone is a weak fingerprint -- cancellation can hide a real
        # difference. These four together cannot coincide by accident, and they
        # are computed on the chip that ran the kernel, so they carry the
        # hardware's own precision rather than a CPU's idea of it.
        "output_fingerprint": {
            "sum": float(jnp.sum(out.astype(jnp.float32))),
            "sumsq": float(jnp.sum(jnp.square(out.astype(jnp.float32)))),
            "absmax": float(jnp.max(jnp.abs(out.astype(jnp.float32)))),
            "tail":
            [float(x) for x in out.astype(jnp.float32).reshape(-1)[-8:]],
        },
        # The output's real dtype, recorded before the float32 casts above
        # erase it. Every fingerprint field is computed on `.astype(float32)`,
        # so a candidate that computed in a narrower type and widened at the
        # end is indistinguishable from one that did not -- and "faster at
        # lower precision" is not the optimisation this task asks for.
        #
        # The CPU parity check cannot cover this: it says nothing at all when
        # the candidate needs the chip, which is every Pallas kernel, and when
        # it does run at bfloat16 it reports a legal one-ULP reordering as
        # divergent. A recorded dtype is decided by the run that actually
        # happened.
        # How much hardware the answer actually used. The job requests a whole
        # node -- 8 chips on a v6e 2x4, 4 on a tpu7x 2x2x1 -- while this
        # workload is single-device by construction, so seven of eight sit
        # idle. That makes "shard it across the rest of the slice" the largest
        # win available, and it is not attention tuning: the same arithmetic
        # spread wider, worth a different multiple on every topology, and
        # invisible in a per-device timing. Recorded here so the grader can
        # tell the two apart instead of trusting a variant's name -- the run
        # labelled `sharded_naive` left behind an unmodified baseline file.
        "device_count": _device_count,
        "output_device_count": _output_devices,
        # What this capture actually ran. The verifier keeps one copy of this
        # file per *run*, which is whatever the agent left behind -- so a
        # variant that was profiled and then overwritten leaves no trace of
        # what it was. Run 18 profiled a session named `opt1_sdpa` whose
        # collected source contained no such call.
        #
        # That gap hides the question the task is named for: an agent that
        # writes a Pallas kernel and cannot compile it is indistinguishable
        # from one that never tried, because neither produces a custom call.
        # A digest identifies a variant across captures; the flags say what
        # kind of thing it was, read from this file's own text.
        "source_sha256": _src_sha,
        "source_uses_kernel_api": _src_kernel,
        "source_uses_library_attention": _src_library,
        "output_dtype": str(out.dtype),
        "input_dtype": str(q.dtype),
        "logdir": logdir,
        # Reported so the flags are observable rather than assumed. Two agents
        # added this line themselves after rewriting the file; both then set the
        # variable in a way that left it holding only the node's own flag, and
        # neither noticed, because a profile with no instruction data still
        # writes, still exits 0 and still looks complete.
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    try:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "run.json"), "w") as fh:
            json.dump(record, fh, indent=2)
    except OSError:
        pass
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
