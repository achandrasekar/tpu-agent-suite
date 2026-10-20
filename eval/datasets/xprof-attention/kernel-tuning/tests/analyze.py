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
"""Reads the traces the agent produced and reports what they show.

The verifier decides from this, not from anything the agent wrote. Sessions are
ordered by mtime, because the agent names them and a name is a claim; when it
profiled is not.

Emits one JSON document on stdout:

    {"sessions": [{"dir","order","device_us","custom_call","tiles"}, ...],
     "baseline_us", "first_kernel_us", "best_kernel_us", "distinct_tilings",
     "llo_success", "errors"}
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

# The tile metadata is a JSON document serialised into a JSON string inside
# another JSON document, so the backslash count in front of a quote varies with
# encoding depth. Match the key and skip punctuation; matching an escaped quote
# works on one chip's output and silently fails on another's.
TILE_KEY = re.compile(r"block_(q|kv|kv_compute)[^0-9A-Za-z-]{0,8}(\d+)")

# Work inside the profiled window that the workload never asked for, and which
# must not be charged to the attention computation.
#
# `total_device_duration_us` is the session total, and the session is not only
# the layer. Run 28 measured a kernel named `xsc-output-checksum-compute`
# running inside the profiled loop -- absent from the baseline, present in every
# variant, and between 0 and 7023 us depending on the session:
#
#     baseline          6360.6 total       0.0 excluded    6360.6 attention
#     opt1_splash      14148.1          7022.6            7125.5
#     opt2_splash_512   1962.0           766.9            1195.1
#     opt3_splash_1024  1786.0          1448.8             337.2
#     opt4_splash_2048  1978.9          1861.8             117.1
#     opt5_final_1024   1989.8          1669.2             320.7
#
# Charging that to the variant and not to the baseline understated the run's
# speedup by more than 5x and inverted the ranking: fastest by total is
# opt3, fastest by attention time is opt4. It is not this task's own
# fingerprint, which is computed after `stop_trace()`.
#
# Excluded by name and reported rather than silently dropped -- `excluded_us`
# and `excluded_kernels` ride along on every session, so a reader can see what
# was taken out and object to it.
EXCLUDED_KERNEL = re.compile(r"checksum", re.IGNORECASE)

# Relative tolerance on the output fingerprint. There are three populations in
# the measurements, not two, and the first cut of this constant sat in the wrong
# gap:
#
#   ~1e-6         identical math -- exact rewrites, sharding across cores
#                 (1.8e-6 splash, 5.6e-6 a hand-written kernel, 0 ULP sharded)
#   ~3e-3..5e-3   blocked / flash attention. Softmax computed per block with a
#                 running max and rescale accumulates in a different order, so
#                 bf16 lands a few parts per thousand away. Systematic, not
#                 noise: three different block sizes in one run gave an
#                 identical 0.00533, and five implementations across two runs
#                 fell in 0.0031-0.0053.
#   ~2e-2..3e-2   genuinely wrong -- wrong layout, or precision dropped outright
#
# At 1e-3 the gate rejected every blocked implementation, which is the solution
# class the task exists to elicit. 1e-2 accepts flash attention and still
# refuses the wrong answers, with roughly a factor of four either side. Tighter
# is not available without a reference that shares the candidate accumulation
# order, which is the thing under test.
NUMERIC_TOL = float(os.environ.get("NUMERIC_TOL", "1e-2"))


def base_us_ok(res: dict) -> bool:
    return isinstance(res.get("baseline_us"), (int, float))


def fingerprint_distance(a: dict, b: dict) -> float:
    """Worst relative difference across the recorded statistics.

  Both fingerprints come from runs on the same chip, so hardware precision is
  common to them and what is left is the algorithm. That is the property a
  CPU-side parity check cannot offer for a TPU kernel: it can tell you two
  formulas agree on a CPU, not that the kernel that ran agrees on the chip.
  """
    worst = 0.0
    for key in ("sum", "sumsq", "absmax"):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            continue
        denom = max(abs(x), 1e-12)
        worst = max(worst, abs(x - y) / denom)
    for x, y in zip(a.get("tail") or [], b.get("tail") or []):
        denom = max(abs(x), 1e-6)
        worst = max(worst, abs(x - y) / denom)
    return worst


# Overridable so the verifier can be exercised against canned output without a
# TPU, and so a machine with the release-channel `xprof` on PATH -- a different
# program with the same name and no subcommands -- can be pointed at the right
# binary instead of silently producing empty results.
XPROF_BIN = os.environ.get("XPROF_BIN", "xprof")

# Generous on purpose: see the timeout branch below. Overridable for a host
# where even this is not enough.
XPROF_TIMEOUT = int(os.environ.get("XPROF_TIMEOUT", "900"))

# Paths whose reads ran out of budget rather than yielding nothing.
TIMED_OUT: set[str] = set()


def xprof(sub: str, path: str, *extra: str) -> str:
    """Runs one subcommand, returning stdout. absl writes INFO to stderr, which
  buries the JSON if the two are merged -- so they are kept apart."""
    try:
        r = subprocess.run([XPROF_BIN, sub, path, *extra],
                           capture_output=True,
                           text=True,
                           timeout=XPROF_TIMEOUT)
        return r.stdout
    except subprocess.TimeoutExpired:
        # Distinguished from "no data", because they mean opposite things: a
        # timeout says the trace was too big to read in the budget, and the
        # variant may be the best one in the run. Measured: a plain-JAX
        # baseline trace is ~110 MB and reads in 13s; a splash-attention
        # variant produced 1.06 GB and needed 143s on an idle host. At the old
        # 240s that margin vanishes under load, and the variant was recorded as
        # having no timing at all -- so the task quietly discarded exactly the
        # kernels it asks for.
        TIMED_OUT.add(path)
        return ""
    except Exception:
        return ""


def xprof_version() -> str | None:
    """The xprof that produced every number below. It comes from the
  nexus-host the image is built on, and the verifier can repair or replace it,
  so the version is a property of the run, not of the repo, and a score that
  cannot name it cannot be reproduced or compared against an earlier one.

  Asked of the interpreter behind `XPROF_BIN`, not of this one. They are not
  always the same: the verifier falls back to a private venv when the shared
  interpreter cannot be repaired, and a rescore can point `XPROF_BIN` anywhere.
  Reading this interpreter's metadata reported 2.24.1a20260903 for numbers a
  2.24.2a20260922 binary had produced -- naming the wrong toolchain, which is
  worse than naming none. `xprof` has no `--version`, so the shebang of the
  console script is what identifies its environment."""
    try:
        path = shutil.which(XPROF_BIN) or XPROF_BIN
        with open(path, "rb") as fh:
            shebang = fh.readline(512).decode("utf-8", "replace")
        if shebang.startswith("#!"):
            interp = shebang[2:].strip().split()[0]
            r = subprocess.run([
                interp, "-c", "import importlib.metadata as m;"
                "print(m.version('xprof'))"
            ],
                               capture_output=True,
                               text=True,
                               timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
    except Exception:
        pass
    # A binary that is not a Python console script, or an interpreter that
    # cannot answer: this interpreter's own metadata is the next best claim.
    try:
        import importlib.metadata as md
        return md.version("xprof")
    except Exception:
        return None


def session_dirs(root: str) -> list[str]:
    """Trace directories, oldest first. A session is the directory holding a
  non-empty .xplane.pb, normalised above plugins/profile/<run>/."""
    seen: dict[str, float] = {}
    traces: dict[str, set[str]] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".xplane.pb"):
                continue
            p = os.path.join(dirpath, f)
            try:
                if os.path.getsize(p) == 0:
                    continue
            except OSError:
                continue
            top = dirpath.split("/plugins/profile/")[0]
            seen[top] = min(seen.get(top, 1e18), os.path.getmtime(p))
            traces.setdefault(top, set()).add(f)

    # One capture can land at two depths. The collector extracts a tarball that
    # already carries the label directory into a directory of the same name, so
    # the same trace file exists at `xprof/<label>/plugins/...` *and*
    # `xprof/<label>/<label>/plugins/...`. Both normalise to a "top", so every
    # capture was counted twice: run 23 reported six profiles for three labels,
    # filled its budget on the duplicates, and read "1 variant" for a run that
    # tried three.
    #
    # Deduplicated on the trace filenames themselves, which are unique per pod
    # and per capture, rather than on the path -- the same evidence is the same
    # capture however many places it was copied to. The shallower path wins so
    # the label stays readable.
    # Subset, not equality: a label can hold more than one profile run, and the
    # copy at the shallower depth may carry only some of them. Whichever path
    # holds the superset is the complete capture; the other is a partial copy
    # of the same work.
    for a in sorted(seen, key=len):
        for b in [x for x in seen if x != a and x.startswith(a + "/")]:
            ta, tb = traces.get(a) or set(), traces.get(b) or set()
            if not ta or not tb:
                continue
            if ta <= tb:
                seen.pop(a, None)
                traces[b] = tb
                break
            if tb <= ta:
                seen.pop(b, None)
    return [d for d, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def run_record(d: str) -> dict:
    """The workload's own JSON, written beside the trace it produced.

  Read from the collected artifacts rather than from the agent's transcript:
  the point is a number the agent did not get to narrate.
  """
    for cand in (os.path.join(d, "run.json"),
                 os.path.join(d, "plugins", "profile", "run.json")):
        try:
            with open(cand) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            continue
    return {}


def describe(d: str) -> dict:
    raw = xprof("get_kernel_stats", d, "--include_summary=True")
    rec = run_record(d)
    out = {
        "dir": d,
        "device_us": None,
        "kernel_us": None,
        "custom_call": False,
        "tiles": None,
        "source_sha256": rec.get("source_sha256"),
        "source_uses_kernel_api": rec.get("source_uses_kernel_api"),
        "source_uses_library_attention":
        rec.get("source_uses_library_attention"),
        "output_device_count": rec.get("output_device_count"),
        "device_count": rec.get("device_count"),
        "output_dtype": rec.get("output_dtype"),
        "input_dtype": rec.get("input_dtype"),
        "checksum": rec.get("output_checksum"),
        "fingerprint": rec.get("output_fingerprint"),
        "max_abs_diff": rec.get("max_abs_diff")
    }
    try:
        doc = json.loads(raw)
    except Exception:
        return out
    total = doc.get("total_device_duration_us")
    excluded, excluded_names = 0.0, []
    for rec in doc.get("kernel_records", []):
        if EXCLUDED_KERNEL.search(str(rec.get("kernel_name", ""))):
            excluded += rec.get("total_duration_us") or 0.0
            excluded_names.append(str(rec.get("kernel_name"))[:60])
    out["session_us"] = total
    out["excluded_us"] = round(excluded, 3)
    out["excluded_kernels"] = excluded_names
    # Never negative, and never zero from subtraction alone: a session whose
    # every kernel was excluded has no attention time to report, which is a
    # different statement from "it ran in no time".
    out["device_us"] = (None if total is None else
                        (round(total - excluded, 3) or None))
    for rec in doc.get("kernel_records", []):
        blob = json.dumps(rec)
        # The kernel's own name, not any substring of the record. An agent that
        # profiles with `--xla_enable_custom_call_region_trace=true` gets records
        # whose *metadata* names tpu_custom_call on a program that has none, so
        # the plain-JAX baseline was classified as a kernel, left the run with no
        # baseline at all, and failed "kernel beats the original" on a run that
        # had beaten it. Flag choice is the agent's; classification must not
        # depend on it.
        name = str(rec.get("kernel_name", ""))
        if "tpu_custom_call" not in name and "custom-call(" not in name:
            continue
        out["custom_call"] = True
        found: dict[str, int] = {}
        for m in TILE_KEY.finditer(blob):
            found.setdefault(m.group(1), int(m.group(2)))
        # The kernel's own duration, not the session total. Asked for "the
        # attention computation", an agent reasonably reports this -- the
        # session total also carries the random-number generation and the
        # harness's own work. Both are recorded so the check can accept either.
        for key in ("total_duration_us", "duration_us", "total_self_time_us"):
            if isinstance(rec.get(key), (int, float)):
                out["kernel_us"] = rec[key]
                break
        if "q" in found and "kv" in found:
            out["tiles"] = [found["q"], found["kv"]]
            break
    return out


def ledger_count(root: str) -> int | None:
    """How many captures `run_on_tpu.sh` recorded starting, or None if no
    ledger survived. Its absence is not proof of nothing: a run that never
    captured and a run that deleted the evidence look alike here, which is why
    the count is compared against the directories rather than trusted alone."""
    try:
        with open(os.path.join(root, ".captures"), encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def dtype_bits(name: str | None) -> int | None:
    """Width of a dtype name, or None when it cannot be read.

  Names carry their own width -- `bfloat16`, `float32`, `float8_e4m3`,
  `int8` -- so the trailing number is the answer and no jax import is needed
  in the verifier.
  """
    if not name:
        return None
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else None


def budget(taken: int,
           attempts: int,
           cap: int,
           logged: int | None = None) -> dict:
    """Account for the profiling budget, and decide whether it was respected.

  The verdict is settled here rather than re-derived by the verifier's shell,
  so the rule has one home and can be tested without a TPU: `budget_ok` is what
  the scored check reads.

  Overrunning is an instruction-following failure, not a rounding error. The
  budget is the one instruction in the prompt carrying a number, and captures
  past the cap were already being dropped from the analysis -- which let a run
  take nine, have three quietly ignored, and score as though it had taken six.
  """
    # The larger of the two counts binds. Enforcing the cap from the
    # directories alone made it reversible -- deleting a capture bought
    # another -- and the cap exists to measure whether the instruction was
    # followed, so it must not be undone by the agent it constrains.
    effective = max(taken, logged or 0)
    return {
        "profiles_taken": taken,
        "captures_logged": logged,
        # Fewer ledger lines than directories means the ledger was truncated;
        # more means captures were removed. Either way the two disagree, and a
        # reader should see that rather than a tidy number.
        "ledger_discrepancy": bool(logged is not None and logged != taken),
        "profile_cap": cap,
        "attempt_cap": attempts,
        # One baseline plus the attempts: profiles = attempts + 1. Counting the
        # baseline as an attempt would give an agent that profiles the original
        # -- which the task asks for -- one fewer try than one that skips it.
        "attempts_used": max(0,
                             min(taken, cap) - 1),
        "over_cap": max(0, effective - cap),
        "budget_ok": effective <= cap,
    }


def classify(s: dict, baseline: dict | None) -> dict:
    """Decide one session's verdict against the baseline, in place.

  Three ways to be disqualified, in increasing order of "it did not answer the
  question asked":

    mismatch      it computed something else
    wrong_dtype   it computed at a narrower precision
    multi_device  it computed on more of the slice than one chip

  Only `ok` can be kept, satisfy the gate, or win, so every caller tests for
  that rather than for the absence of any one failure. Pulled out of `main` to
  be readable and testable on its own: these are the rules the whole task turns
  on, and they were previously thirty lines in the middle of a function that
  also shells out to xprof.
  """
    base_fp = baseline.get("fingerprint") if baseline else None
    base_ck = baseline.get("checksum") if baseline else None
    ck, fp = s.get("checksum"), s.get("fingerprint")

    if base_fp and fp:
        # Every component, worst case wins. Comparing four statistics of the
        # same tensor closes the gap a lone sum leaves open: cancellation can
        # hold a sum steady while the values underneath differ.
        s["rel_diff"] = fingerprint_distance(base_fp, fp)
        s["numerics"] = "ok" if s["rel_diff"] <= NUMERIC_TOL else "mismatch"
    elif base_ck not in (None, 0) and ck is not None:
        s["rel_diff"] = abs(ck - base_ck) / abs(base_ck)
        s["numerics"] = "ok" if s["rel_diff"] <= NUMERIC_TOL else "mismatch"
    else:
        s["numerics"] = "unknown"

    # Precision is a separate question from agreement, and it outranks it. The
    # fingerprints are all computed on `.astype(float32)`, so a variant that
    # computed in a narrower type and widened at the end can land inside
    # NUMERIC_TOL and read as "ok" -- closer to the reference than an honest
    # blocked variant, while being faster for doing less arithmetic.
    #
    # Measured against the *input* width, not the baseline's output. Comparing
    # output dtypes looked right and was not: `jnp.einsum` promotes bf16 inputs,
    # so the baseline returns float32 while its own docstring says "bfloat16 in
    # and out" -- and run 18 duly flagged `opt2_sdpa_transposed`, which returned
    # the documented bf16 and was numerically fine at 0.0044, as though it had
    # cheated. Punishing a candidate for returning the contract's own dtype is
    # a false positive, and the rule that matters is narrower: do not compute
    # below the precision you were given. float8 and int8 fall; bf16 and float32
    # both pass.
    in_bits = dtype_bits(
        s.get("input_dtype") or (baseline or {}).get("input_dtype"))
    out_bits = dtype_bits(s.get("output_dtype"))
    s["dtype_changed"] = bool(in_bits and out_bits and out_bits < in_bits)
    if s["dtype_changed"]:
        s["numerics"] = "wrong_dtype"

    # Single chip, enforced rather than assumed. The task is to make the kernel
    # faster on one device; spreading the same arithmetic across the rest of the
    # slice is a different achievement, worth a different multiple on a v6e 2x4
    # than on a tpu7x 2x2x1, and it would make scores incomparable across
    # hardware. It wins on the timing while answering a question nobody asked.
    used = s.get("output_device_count")
    s["multi_device"] = bool(used and used > 1)
    if s["multi_device"]:
        s["numerics"] = "multi_device"
    return s


def collect_sessions(res: dict) -> tuple[str, list[str]]:
    """Reads the captures, applies the budget, and describes each session."""
    root = sys.argv[1] if len(sys.argv) > 1 else "/logs/artifacts/xprof"
    dirs = session_dirs(root)

    # Equal effort, enforced here rather than asked for in the prompt. Agents
    # write their own dispatch -- most runs ignore `run_on_tpu.sh` entirely --
    # so a cap the workload enforces is one the agent can route around, and a
    # cap only stated in the instruction is one it can overrun by accident.
    # Reading the first N by mtime is the one place the count is certain.
    #
    # Five, because the observed spread was 3 to 9 captures and the best result
    # of the series used six (one baseline, five variants). Comparing a run that
    # took nine against one that took three measures budget, not judgment.
    # One baseline plus the attempts: profiles = attempts + 1. Counting the
    # baseline as an attempt would give an agent that profiles the original --
    # which the task asks for -- one fewer try than one that skips it.
    attempts = int(os.environ.get("MAX_ATTEMPTS", "5"))
    cap = int(os.environ.get("MAX_PROFILE_RUNS", str(attempts + 1)))
    res.update(budget(len(dirs), attempts, cap, ledger_count(root)))
    # A capture that wrote a trace but yields no timing is spent, not free. It
    # cost an admission, a pod and a slot in the budget, and xprof can say
    # nothing about it -- `flash_attention` did exactly this twice. Counted
    # against the cap like any other attempt and reported, so a run that burns
    # its budget on captures nothing can read is visible rather than just short.

    dirs = dirs[:cap]
    for i, d in enumerate(dirs):
        s = describe(d)
        s["order"] = i
        res["sessions"].append(s)

    return root, dirs


def choose_baseline(res: dict) -> dict | None:
    """Picks the baseline and reports what could not be read."""
    # The baseline is the session profiled **first**, and only if it can be
    # read. "Cheapest session with no custom call" picked an agent's own 678 us
    # variant in run 13 and reported the 6358 us original as never run. Falling
    # through to the first *timed* session makes the same mistake by another
    # route, and silently -- a variant compared against itself looks fine.
    # `flash_attention` produced an unreadable capture twice, so this is real.
    # Computed after the loop that fills `sessions`, not before it, where it
    # returned an empty list for the life of the task.
    res["unreadable_profiles"] = [
        s["dir"].split("/")[-1] for s in res["sessions"] if not s["device_us"]
    ]
    res["timed_out_profiles"] = [
        s["dir"].split("/")[-1] for s in res["sessions"]
        if s["dir"] in TIMED_OUT
    ]

    ordered = res["sessions"]
    baseline = ordered[0] if ordered and ordered[0]["device_us"] else None
    res["baseline_unreadable"] = bool(ordered and not ordered[0]["device_us"])
    res["baseline_us"] = baseline["device_us"] if baseline else None
    res["baseline_dir"] = baseline["dir"] if baseline else None
    # Named explicitly because `custom_call` alone reads as "this session ran a
    # kernel", and the unmodified program trips it: XLA lowers `slice-start` and
    # friends to custom calls of its own, so the plain-JAX baseline shows
    # `custom_call=True` while containing no attention kernel at all. A reader
    # comparing sessions should not have to know that.
    for s in res["sessions"]:
        s["is_baseline"] = (s is baseline)
        s["attention_kernel"] = bool(s["tiles"])

    # Does each variant still compute the original function? Checked against the
    # baseline's own checksum, so it needs no reference implementation the agent
    # could edit away. Relative, because these are sums over four million
    # values: a correct rewrite differs in the seventh digit (-49370.52 vs
    # -49370.61, 1.8e-6) and a wrong one in the second (-47726.24, 3.3e-2).
    res["baseline_checksum"] = baseline.get("checksum") if baseline else None
    for s in res["sessions"]:
        classify(s, baseline)

    return baseline


def select_kernels(res: dict, baseline: dict | None) -> list[dict]:
    """Classifies every session against the baseline and returns the sessions
  still eligible to win."""
    # Candidates for "the optimisation". A variant that computes something else
    # is not a faster attention, however fast it is -- run 13 shipped a 9.4x
    # speedup whose checksum was off in the second digit, and nothing stopped
    # it. Only sessions that are not known-wrong can win.
    kernels = [
        s for s in res["sessions"]
        if s["device_us"] and s is not baseline and s["numerics"] == "ok"
    ]
    # Did any capture run source that reached for a kernel API, whether or not
    # a custom call reached the trace? Twenty runs have produced zero custom
    # calls, and until now nothing distinguished "wrote a kernel that would not
    # compile" from "never tried" -- the two look identical in a trace, and the
    # collected source is only whatever the agent left behind at the end.
    res["kernel_api_attempted"] = [
        s["dir"] for s in res["sessions"] if s.get("source_uses_kernel_api")
    ]
    res["library_attention_used"] = [
        s["dir"] for s in res["sessions"]
        if s.get("source_uses_library_attention")
    ]
    res["distinct_sources"] = len({
        s["source_sha256"]
        for s in res["sessions"] if s.get("source_sha256")
    })
    res["multi_device"] = [
        s["dir"] for s in res["sessions"] if s.get("multi_device")
    ]
    res["dtype_changed"] = [
        s["dir"] for s in res["sessions"] if s.get("dtype_changed")
    ]
    res["wrong_but_faster"] = [
        {
            "dir": s["dir"].split("/")[-1],
            "device_us": s["device_us"],
            "rel_diff": round(s.get("rel_diff", 0), 6)
        } for s in res["sessions"]
        if s["numerics"] in ("mismatch", "wrong_dtype",
                             "multi_device") and s["device_us"]
        and base_us_ok(res) and s["device_us"] < res["baseline_us"]
    ]

    return kernels


def derive_timings(res: dict, kernels: list[dict]) -> None:
    """Turns the eligible sessions into the timing figures the verifier scores:
  first kernel, first working kernel, best, last, and the breadth of the
  search that produced them."""
    res["first_kernel_us"] = kernels[0]["device_us"] if kernels else None

    # The first kernel that was actually an improvement -- not the first kernel
    # tried, which is usually the default tiling and slower than the original.
    # Tuning quality is measured from here: everything before it is the port to
    # a kernel, everything after it is the tile search the task is really about.
    base_us = res["baseline_us"]
    working = [s for s in kernels if base_us and s["device_us"] < base_us]
    res["first_working_kernel_us"] = working[0][
        "device_us"] if working else None
    res["best_kernel_us"] = min((s["device_us"] for s in kernels),
                                default=None)
    res["last_kernel_us"] = kernels[-1]["device_us"] if kernels else None
    res["distinct_tilings"] = sorted(
        {tuple(s["tiles"])
         for s in kernels if s["tiles"]})
    res["distinct_tilings"] = [list(t) for t in res["distinct_tilings"]]

    # How many configurations it actually measured. Tile metadata only exists
    # for JAX's splash attention, which writes `BlockSizes` into the trace; an
    # agent that writes its own Pallas kernel emits none, and run 14 did exactly
    # that -- `flash_attn_tuned` and `unrolled_4block` are tiling experiments by
    # any reading, and the tile regex saw zero. Counting profiled variants
    # measures the search rather than the library it was performed with.
    variants = [s for s in kernels if s["device_us"]]
    res["variants_tried"] = len(variants)
    # Whichever evidence of breadth exists; tilings are the sharper of the two
    # because they name what changed, so they win when present.
    res["configs_tried"] = max(len(res["distinct_tilings"]), len(variants))

    # The fastest session, and the tiling that produced it. "Did it keep the
    # best thing it measured" is a fairer question than "did the last attempt
    # beat the first": an agent whose first try is already optimal has not
    # failed at anything.
    best = min(kernels, key=lambda s: s["device_us"], default=None)
    res["best_tiling"] = best["tiles"] if best else None
    res["best_kernel_record_us"] = best["kernel_us"] if best else None


def read_llo(res: dict, dirs: list[str]) -> None:
    """Was LLO data present -- the one signal that says the profiling flags
  reached the backend in time rather than after the trace was cut."""
    # LLO presence is how we know LIBTPU_INIT_ARGS reached the backend in time.
    # Set too late and the run still exits 0 with a clean, empty trace.
    res["llo_success"] = False
    res["llo_detail"] = None
    for d in dirs:
        raw = xprof("get_llo_analysis", d)
        try:
            doc = json.loads(raw or "{}")
        except Exception:
            res["llo_detail"] = f"{d.split('/')[-1]}: unparseable ({raw[:80]!r})"
            continue
        if doc.get("success"):
            res["llo_success"] = True
            res["llo_detail"] = (
                f"{d.split('/')[-1]}: {doc.get('total_instructions')} "
                f"instructions, {doc.get('total_modules')} modules")
            break
        # Why it failed matters: an empty trace means the flags were set too
        # late, which exits 0 and looks like success everywhere else.
        res["llo_detail"] = (
            f"{d.split('/')[-1]}: success={doc.get('success')} "
            f"{str(doc.get('reason') or doc.get('error') or '')[:60]}")


def main() -> int:
    """The pipeline, in the order the verifier depends on it. Each step reads
  and extends `res`, which is the JSON `test.sh` scores from."""
    res: dict = {
        "sessions": [],
        "errors": [],
        "xprof_version": xprof_version()
    }
    root, dirs = collect_sessions(res)
    baseline = choose_baseline(res)
    kernels = select_kernels(res, baseline)
    derive_timings(res, kernels)
    read_llo(res, dirs)
    json.dump(res, sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
