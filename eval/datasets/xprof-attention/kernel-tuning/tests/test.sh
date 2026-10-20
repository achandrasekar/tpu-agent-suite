#!/usr/bin/env bash
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
#
# Two separate questions, kept separate:
#
#   CORRECTNESS -- did the program get faster, and does the agent's own number
#   match the trace. Decided by analyze.py reading the traces, never by the
#   agent's prose. Both arms can pass this; that is the point.
#
#   INTEGRATION -- did the package get used: did tpu-agent delegate to
#   tpu-diagnostics, did the XProf skill get read, did the XProf CLI get
#   driven. Reported for both arms, scored for neither. The control arm has
#   no package and cannot pass it, so folding it into the reward would say the
#   package failed whenever the control succeeded.
set -uo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
ART="${ARTIFACTS_DIR:-/logs/artifacts}"
mkdir -p "$LOGS"
# Overridable for the same reason ARTIFACTS_DIR is: the integration signals and
# the FINAL-line check are the two things that read the agent's own output, and
# a hardcoded path makes both unverifiable outside a live trial.
AGENT_LOG_DIR="${AGENT_LOG_DIR:-/logs/agent}"
AGENT_LOGS=("$AGENT_LOG_DIR"/*)
HERE="$(cd "$(dirname "$0")" && pwd)"

# `command -v xprof` is not enough. Another distribution can install a binary
# with the same name and different subcommands, so a machine that has it passes
# the check and then produces empty output from every call -- silently, with
# exit 0. Test the distribution, not the binary name.
#
# The pin is the one nexus-host installed, read from its constraints, so there
# is no second copy of the floor to keep in step. This is the repair path, not
# the normal one: the image already carries that version, and this runs only
# when the distribution is missing entirely. An image whose constraints name no
# `xprof` (one built while the package tracked xprof-nightly) gets the newest
# release.
XPROF_PIN="$(grep -i -m1 '^xprof==' /etc/nexus-host/constraints.txt 2>/dev/null || true)"
XPROF_PIN="${XPROF_PIN:-xprof}"
if ! python3 -c "import importlib.metadata as m; m.version('xprof')" 2>/dev/null; then
  echo "  installing $XPROF_PIN (no xprof distribution present)"
  # Logged, not discarded, and with a venv to fall back on. The agent shares
  # this interpreter for a whole trial and can leave it in a state pip will not
  # repair; run 10's verifier reported "no xprof", analysed nothing, and
  # left no way to tell whether the install had failed or never run.
  pip install -q --break-system-packages --index-url https://pypi.org/simple \
      "$XPROF_PIN" >"$LOGS/pip.log" 2>&1 \
    || pip install -q --index-url https://pypi.org/simple \
      "$XPROF_PIN" >>"$LOGS/pip.log" 2>&1 \
    || { echo "  system install failed; falling back to a private venv (see pip.log)"
         python3 -m venv "$LOGS/xv" >>"$LOGS/pip.log" 2>&1 \
           && "$LOGS/xv/bin/pip" install -q --index-url https://pypi.org/simple \
                "$XPROF_PIN" >>"$LOGS/pip.log" 2>&1 \
           && XPROF_BIN="$LOGS/xv/bin/xprof"; }
fi
export XPROF_BIN="${XPROF_BIN:-xprof}"
python3 - "$XPROF_BIN" <<'PYV'
import importlib.metadata as m, subprocess, sys
try:
    print(f"  xprof {m.version('xprof')}")
except Exception:
    # A venv fallback is invisible to this interpreter's metadata, so ask the
    # binary itself before declaring the analysis dead.
    try:
        n = len(subprocess.run([sys.argv[1], "--help"], capture_output=True,
                               text=True, timeout=60).stdout.splitlines())
    except Exception:
        n = 0
    print(f"  xprof via {sys.argv[1]} ({n} help lines)" if n > 20 else
          "  WARNING: no xprof; trace analysis will return nothing")
try:
    print(f"  WARNING: xprof-nightly {m.version('xprof-nightly')} is also installed; "
          "they share a package directory and the wrong one may answer")
except Exception:
    pass
PYV

# Keep the code the agent left behind. Run 8 failed the LLO check and there was
# no way to tell whether the agent had dropped the env block or the block itself
# was broken -- the workload is not collected, so the evidence died with the
# container. It was the block: `setdefault` loses to the LIBTPU_INIT_ARGS a TPU
# node injects. Cheap to keep, and the only record of what actually ran.
mkdir -p "$ART/workload"
cp /workspace/*.py "$ART/workload/" 2>/dev/null || true

# The image checks, run from inside the image Harbor built. They used to live in
# a build-time hook in `eval/run.sh`; that runner is gone and this is the
# better home anyway, because the verifier is already executing in the artifact
# under test rather than inspecting a tag it hopes is the right one.
#
# The failure being guarded: a rename once left `attention_stages.py` -- the
# three-stage script with `V2_BLOCKS = dict(block_q=512, block_kv=1024)` in it --
# in /workspace, and the agent opened the answer four seconds into the run. It
# was scored before anyone noticed.
echo "=== image ==="
WANT="$(grep -oE '/workspace/[A-Za-z0-9_.-]+\.py' "$HERE/../instruction.md" 2>/dev/null \
        | head -1 | xargs -r basename)"
WS="$(find /workspace -mindepth 1 -maxdepth 1 2>/dev/null \
        | sed 's|.*/||' | sort | paste -sd' ' -)"
echo "  /workspace: ${WS:-<unreadable>}"
if [ -n "$WANT" ] && [ -d /workspace ]; then
  [ -f "/workspace/$WANT" ] \
    && echo "  instruction names $WANT, and the image ships it" \
    || echo "  WARNING: instruction names $WANT but /workspace does not have it"
  STRAY=""
  for f in /workspace/*.py; do
    [ -e "$f" ] || continue
    b="${f##*/}"
    [ "$b" = "$WANT" ] || STRAY="${STRAY:+$STRAY }$b"
  done
  [ -z "$STRAY" ] || echo "  WARNING: unexpected workload file(s): $STRAY"
  LEAK="$(grep -lE 'block_q=|BlockSizes\(|splash_attention|V2_BLOCKS' /workspace/*.py 2>/dev/null | paste -sd' ' -)"
  [ -z "$LEAK" ] || echo "  WARNING: answer present in the image: $LEAK"
fi

echo "=== what the traces show ==="
python3 "$HERE/analyze.py" "$ART/xprof" > "$LOGS/traces.json" 2>"$LOGS/analyze.err" || true
python3 - "$LOGS/traces.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("  analyze.py produced nothing:", e); raise SystemExit
for s in d["sessions"]:
    tag = "  <- baseline" if s.get("is_baseline") else ""
    print(f"  [{s['order']}] {s['dir'].split('/')[-1]:24} device_us={s['device_us']} "
          f"attention_kernel={s.get('attention_kernel')} tiles={s['tiles']}{tag}")
print(f"  baseline={d['baseline_us']} first_kernel={d['first_kernel_us']} "
      f"best_kernel={d['best_kernel_us']} tilings={d['distinct_tilings']} llo={d['llo_success']}")
# The toolchain follows the nexus-host the image is built on, and the agent can
# change it during a trial, so the score has to say which xprof measured it.
print(f"  measured with xprof {d.get('xprof_version') or '<unknown>'}")
PY

pass=0; total=0
# Float arithmetic because one check is graded rather than binary; shell integer
# arithmetic silently truncates a half mark to zero.
check() { total=$((total+1))
  if [ "$2" = 1 ]; then echo "  PASS  $1"; pass=$(awk "BEGIN{print $pass+1}")
  else echo "  FAIL  $1"; fi; }
check_graded() { total=$((total+1)); pass=$(awk "BEGIN{print $pass+$2}")
  case "$2" in
    1) echo "  PASS  $1" ;;
    0) echo "  FAIL  $1" ;;
    *) echo "  PART  $1" ;;
  esac; }
# The whole last line containing a marker, not a table cell -- used to read the
# FINAL line, which is prose rather than a row.
line_with() { { grep -ohE "$1[^\"\\]{0,200}" "${AGENT_LOGS[@]}" 2>/dev/null || true; } | tail -1; }
q() { python3 -c "
import json,sys
try: d=json.load(open('$LOGS/traces.json'))
except Exception: print(''); raise SystemExit
v=d.get('$1')
print('' if v is None else (len(v) if isinstance(v,list) else v))"; }

BASE=$(q baseline_us); BEST=$(q best_kernel_us)
NTILE=$(q configs_tried); LLO=$(q llo_success)
NTILE_EXACT=$(q distinct_tilings); NVAR=$(q variants_tried)
NSESS=$(python3 -c "
import json
try: print(len(json.load(open('$LOGS/traces.json'))['sessions']))
except Exception: print(0)")
NCC=$(python3 -c "
import json
try: print(sum(1 for s in json.load(open('$LOGS/traces.json'))['sessions'] if s['custom_call']))
except Exception: print(0)")

# Counting model-side refusals honestly, which the first version did not.
#
#   * It summed across files that mirror each other. Run 12's agent directory
#     holds the same conversation in `trajectory.json`, in
#     `antigravity-cli.trajectory.jsonl` and again in
#     `antigravity-conversation.db`, so 24 real refusals were reported as 57.
#     Taking the maximum across files counts the fullest copy once.
#
#   * `RESOURCE_EXHAUSTED` is not always the model. Twelve of run 12's matches
#     were `RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom` -- the TPU
#     compiler refusing a tile too large for VMEM, which is the agent probing
#     the search space and exactly what the task asks it to do. Counting that as
#     an outage would excuse a genuine failure as infrastructure.
refusal_census() {
  python3 - "${AGENT_LOGS[@]}" <<'PYI'
import glob, os, re, sys
files = []
for p in sys.argv[1:]:
    for pat in ("**/*.jsonl", "**/*.txt", "**/*.err", "**/*.json", "**/*.db"):
        files += glob.glob(os.path.join(p, pat), recursive=True)
    if os.path.isfile(p):
        files.append(p)

PATTERNS = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "Error 503", "Error 429",
            "exceeded your current quota", "agent executor error")
# The TPU stack's own out-of-memory refusals, which borrow the same status name:
# `E1001: CompileTimeScopedVmemOom` at compile time, and an HBM allocation the
# runtime cannot satisfy, which carries no `E` code. Both are the agent's
# program being wrong, so neither may count as a backend outage.
#
# Kept textually identical to WORKLOAD_FAULT in tpu-agent's
# `resume_after_quota.py`, which applies the same rule on the other side of the
# container boundary and cannot import this. Change both together.
WORKLOAD = re.compile(
    r"RESOURCE_EXHAUSTED:\s*E\d+:"
    r"|CompileTimeScopedVmemOom|VmemOom"
    r"|Attempting to allocate|Out of memory while trying to allocate"
    r"|RAN OUT OF MEMORY",
    re.IGNORECASE)

hits = {}
for f in set(files):
    try:
        text = open(f, errors="ignore").read()
    except OSError:
        continue
    text = WORKLOAD.sub("", text)
    for k in PATTERNS:
        n = len(re.findall(k, text))
        if n > hits.get(k, 0):
            hits[k] = n
hits = {k: v for k, v in hits.items() if v}
print(";".join(f"{k}x{v}" for k, v in hits.items()))
PYI
}

# Reported on every run, gate failure included. This used to sit after the
# gate, so a trial that failed the gate wrote no signals.json at all --
# losing the delegation, skill and tool evidence exactly when it is most
# needed, since a failed run is the one worth diagnosing. Run A failed the
# gate and left no record of having read the skill for the first time.
python3 - "$LOGS/traces.json" <<'PYB'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
n, c, o = d.get("profiles_taken"), d.get("profile_cap"), d.get("over_cap") or 0
if n is not None:
    extra = f" -- {o} beyond the cap ignored" if o else ""
    print(f"  budget: {n} profiles taken, cap {c}{extra}")
    bad = d.get("unreadable_profiles") or []
    if bad:
        print(f"  unreadable: {len(bad)} capture(s) produced no timing "
              f"({', '.join(bad[:3])}) -- spent, not free")
PYB

report_signals() {
echo "=== integration (reported, not scored) ==="
python3 - "$LOGS" "${AGENT_LOGS[@]}" <<'PY'
import collections, glob, json, os, re, subprocess, sys
logs, paths = sys.argv[1], sys.argv[2:]

# The real subcommand list, to check matches against. Agent transcripts are
# streamed, so a name can be split across two `text_delta` chunks and the tail
# of the first one still looks like a command: run 11 reported `get_a`, which
# came from `"text_delta":"`xprof get_a"`. Counting fragments as tool calls
# inflates the one integration signal that is meant to be countable.
SUBS = set()
try:
    # stdout *and* stderr: absl/fire prints the COMMANDS block to stderr, so
    # reading stdout alone returned an empty set -- and an empty set meant the
    # filter below accepted everything, which is how run 12 still reported
    # `get_avg`, `get_kernel_st` and `get_ove`.
    _r = subprocess.run([os.environ.get("XPROF_BIN", "xprof"), "--help"],
                        capture_output=True, text=True, timeout=60)
    _h = (_r.stdout or "") + (_r.stderr or "")
    SUBS = set(re.findall(
        r"\b(get_\w+|list_\w+|verify_\w+|compute_\w+|check_\w+|aggregate_\w+)\b", _h))
except Exception:
    pass
blobs = []
for p in paths:
    for pat in ("**/*.json", "**/*.jsonl", "**/*.txt"):
        blobs += glob.glob(os.path.join(p, pat), recursive=True)
    if os.path.isfile(p): blobs.append(p)
text = ""
for b in blobs:
    try: text += open(b, errors="ignore").read()
    except OSError: pass
# Counts calls, not names. The instruction used to name the agents, so the
# string was in every transcript before the agent acted and this read True in
# all thirteen runs -- including a control arm with no package installed.
#
# No "was the analyst present" signal is reported, deliberately: the logs are
# one flat blob of commands *and* output, so `tpu-diagnostics.md` matches whether
# the agent found the file or merely looked, and the prompt tells it to look.
# Anchoring to the install path did not help -- the control arm's log contains
# `ls /root/.agents/agents/tpu-diagnostics.md` returning nothing.
# `skill_from_package` below discriminates, because only a real install writes
# under /root.
analyst_probed = bool(re.search(r'which\s+[^\n]{0,80}tpu-diagnostics', text))
# The key agy actually writes is `type_name`, inside `subagent_info.subagents`.
# Guessing at `subagent`/`subagent_type`/`agent_name` matched none of them, so
# run 10 -- the first run in the series that genuinely delegated, with
# `"step_type":"subagent"` and a 0.68s `invoke_subagent` call in the stream --
# was reported as `n/a (no subagent invocation recorded by this harness)`.
# Taken from an observed payload now rather than from a guess, and counted, so
# "delegated once" and "delegated eleven times" are not the same reading.
# Two shapes, per the agy orchestration guide, and a run can use either:
#   subagent_info.subagents[] -> "type_name": "tpu-diagnostics"
#   tool_info.args_json       -> "TypeName": "tpu-diagnostics", batched under
#                                "Subagents" or flat, and escaped because it is
#                                JSON inside a JSON string.
# The lowercase-only pattern would have missed the flat `TypeName` form
# entirely -- the same class of miss as keying on `subagent` when agy writes
# `type_name`, which is what hid the first real delegation in this series.
# Every subagent the run invoked, not just the analyst. The package arm used to
# carry one delegate; the whole-package build gives the router six, and only
# `tpu-kernel-development` leads to a kernel. Matching a single name would report a
# correct route to the wrong-for-this-task specialist -- or to the right one --
# as no delegation at all.
_delegation_re = (r'\\?"(?:type_name|TypeName|subagent|subagent_type|agent_name)'
                  r'\\?"\s*:\s*\\?"([a-z0-9][a-z0-9-]{2,40})\\?"')
_delegated_to = collections.Counter(re.findall(_delegation_re, text))
_delegations = [k for k in _delegated_to if k == "tpu-diagnostics"]
# A call that was issued but never came back is not a delegation that happened;
# agy writes the tool step first and the `subagent` step on completion.
_dispatched = re.findall(r'"tool_name"\s*:\s*"invoke_subagent"', text)

# Count identity, not matches. One delegation shows up in both `tool_info` and
# `subagent_info`, and each step is written twice (ACTIVE then DONE), so
# counting regex hits reported a single delegation as "2x". Every spawned
# subagent gets one conversation id, which is the thing worth counting.
_conversations = set(
    re.findall(r'\\?"conversation_?[Ii]d\\?"\s*:\s*\\?"([0-9a-fA-F-]{36})',
               text))
# The stream opens with `{"event":"init","conversation_id":...}` -- the parent's
# own id, which is not a delegation. Counting it reported one delegation as two.
_conversations -= set(
    re.findall(r'"event"\s*:\s*"init"[^}]{0,80}?"conversation_id"\s*:\s*"([0-9a-fA-F-]{36})',
               text))
invoked = bool(_delegations)

# Where the skill came from matters as much as whether it exists: a control
# agent once pip-installed xprof-nightly and copied the wheel's skill into
# /workspace/.claude/skills/xprof. That is the agent rebuilding the package's
# affordance, not the package providing it, and counting the two the same makes
# the control look equipped. The package itself never copies that skill: its
# tpu-diagnostics prints it from the wheel, so a `skills/xprof` on disk is always
# the agent's own doing.
#
# The two are told apart by *name*, and that is what this got wrong for every
# run on the package-mounts path. The TPU agent suite package compiles to a single skill
# called `tpu-agent` -- `tpu-agent-suite-install.json` copies `skills/tpu-agent`
# to `.agents/skills/tpu-agent` -- while `skills/xprof` belongs to the wheel
# and only a self-install produces it. Matching `skills/xprof` for both meant
# `skill_from_package` could never be True however well the mount worked, and a
# run was written off as a mount failure on that evidence alone.
PKG_SKILL = "tpu-agent"   # installed by the TPU agent suite package
WHEEL_SKILL = "xprof"        # ships inside the xprof wheel

# Disk beats transcript. Both trees are still present when the verifier runs, so
# whether the package arrived is a question about the filesystem rather than
# about what the agent happened to echo. The roots are the ones the antigravity
# wrapper fans customizations out to, plus claude-code's.
_HOME = os.path.expanduser("~")
CUSTOMIZATION_ROOTS = [
    "/workspace/.agents", "/workspace/.claude",
    "/root/.agents", "/root/.claude",
    f"{_HOME}/.gemini/config", f"{_HOME}/.gemini/antigravity-cli",
    f"{_HOME}/.agents", f"{_HOME}/.claude",
]

def _installed_at(kind, leaf):
    return sorted({r for r in CUSTOMIZATION_ROOTS
                   if os.path.isfile(os.path.join(r, kind, leaf))})

pkg_skill_roots = _installed_at("skills", f"{PKG_SKILL}/SKILL.md")
wheel_skill_roots = _installed_at("skills", f"{WHEEL_SKILL}/SKILL.md")
analyst_roots = _installed_at("agents", "tpu-diagnostics.md")

skill_self_installed = bool(wheel_skill_roots)
# tpu-diagnostics reads the wheel's skill through locate_xprof_skill.py, which
# prints it under this banner; no copy lands on disk for the roots above.
skill_printed = "===== SKILL.md =====" in text
skill_seen = bool(re.search(
    r"(/root|/workspace|\.gemini/config|\.gemini/antigravity-cli)"
    r"[^\s\"]*skills/" + PKG_SKILL + r"/", text))
skill_from_package = bool(pkg_skill_roots) or skill_seen

sig = {
    "tpu_diagnostics_probed_for": analyst_probed,
    "delegated_to_tpu_diagnostics": (
        (f"yes, {len(_conversations)} subagent conversation(s)"
         if _conversations else "yes") if invoked
        else ("dispatched but no completion recorded"
              if _dispatched
              else "n/a (no subagent invocation recorded by this harness)")),
    "subagent_conversations": len(_conversations),
    # Which specialists it reached for. Names, not counts: each delegation
    # writes an ACTIVE and a DONE record, so a per-name tally reads 2 for one
    # call. `subagent_conversations` is the number that means something.
    "delegated_to": sorted(_delegated_to),
    "skill_materialised": skill_from_package or skill_printed or bool(re.search(
        r"skills/(" + PKG_SKILL + "|" + WHEEL_SKILL + r")/(SKILL\.md|references/)",
        text)),
    "skill_from_package": skill_from_package,
    "skill_self_installed_by_agent": skill_self_installed,
    "skill_tool_invoked": bool(re.search(
        r'"skill"\s*:\s*"(' + PKG_SKILL + "|" + WHEEL_SKILL + r')"', text)),

    # Where the mount actually landed, reported as paths rather than a bare
    # bool. An empty list next to `skill_from_package: False` says the package
    # never arrived; a populated one next to an agent that ignored it says
    # something entirely different, and the two used to be indistinguishable.
    "package_skill_installed_at": pkg_skill_roots,
    "tpu_diagnostics_installed_at": analyst_roots,

    # Did it verify its own numerics with `verify_numerical_parity`, which the
    # skill marks MANDATORY for this case? Run 13 is the cost of skipping it: a
    # 9.4x speedup shipped with the checksum off in the second digit.
    # Reported, not scored -- the control arm has no skill naming the tool, so
    # scoring it would measure which image was built.
    # How the run spent itself. Run 17 made 760 `manage_subagents` calls against
    # two captures -- a busy-wait -- and graded as merely slow. Counted from DONE
    # records only: each call writes ACTIVE and DONE, so matching the tool name
    # doubles every total.
    "tool_calls": dict(
        sorted(collections.Counter(
            re.findall(
                r'"state"\s*:\s*"DONE","step_type"\s*:\s*"[a-z_]+",'
                r'"tool_name"\s*:\s*"(\w+)"', text)).items(),
               key=lambda kv: (-kv[1], kv[0]))),

    "verified_numerics_with_xprof": bool(
        re.search(r"verify_numerical_parity", text)),
    "cli_subcommands": sorted(
        {m for m in re.findall(
            r"\bxprof\s+(get_\w+|list_\w+|verify_\w+|compute_\w+|check_\w+|aggregate_\w+)",
            text) if not SUBS or m in SUBS}),
}
sig["skill_consulted"] = sig["skill_materialised"] or sig["skill_tool_invoked"]

# Antigravity surfaces skills as files under `.agents/skills` with no
# named tool to invoke, so `skill_tool_invoked` cannot fire there however well
# the agent uses the skill. Reported `n/a` on such a harness rather than
# `False`: for eleven runs it sat next to `skill_materialised: True` and read as
# an agent shortcoming, when it is a property of the harness.
if not os.path.isdir(os.path.expanduser("~/.claude/skills")) and not sig["skill_tool_invoked"]:
    sig["skill_tool_invoked"] = "n/a (no skill tool on this harness)"

json.dump(sig, open(os.path.join(logs, "signals.json"), "w"), indent=2)
for k in ("tpu_diagnostics_probed_for", "verified_numerics_with_xprof",
          "delegated_to_tpu_diagnostics", "subagent_conversations",
          "skill_consulted", "skill_materialised", "skill_from_package",
          "skill_self_installed_by_agent", "skill_tool_invoked",
          "package_skill_installed_at", "tpu_diagnostics_installed_at"):
    print(f"  {k}: {sig[k]}")
print(f"  cli_subcommands ({len(sig['cli_subcommands'])}): {' '.join(sig['cli_subcommands']) or '<none>'}")
_tc = sig["tool_calls"]
_dt = sig["delegated_to"]
print(f"  delegated_to ({len(_dt)}): " + (" ".join(_dt) or "<none>"))
print(f"  tool_calls ({sum(_tc.values())}): "
      + (" ".join(f"{k}={v}" for k, v in _tc.items()) or "<none>"))
PY

# Reported on a passing run too, not only when the gate fails. A trial that is
# refused sixty times and still limps to a score is not the same measurement as
# one that ran clean, and scoring it silently hides that: the census existed but
# only ever printed on the failure path, so every score in this series was
# published without saying what conditions produced it.
REFUSALS="$(refusal_census)"
echo "  model_refusals: ${REFUSALS:-none}"
[ -n "$REFUSALS" ] && echo "  (a refused turn is retried by the harness; these did not end the run)"

}

echo "=== gate ==="
GATE=1
[ "${NSESS:-0}" -ge 2 ] || { echo "  GATE FAIL  fewer than 2 profiles captured ($NSESS)"; GATE=0; }
# Gate on the goal, not on the solution I expected. Requiring a
# `tpu_custom_call` encoded "the answer is a Pallas kernel": run 13 replaced
# `attention()` with `jax.nn.dot_product_attention` and the gate said "nothing
# was written" about a 9.4x change. What has to be true is that something was
# measured faster than the original and still computes it -- how is the agent's
# business.
# A slower result is a result. The gate asks whether the work happened and the
# numbers can be trusted -- a variant profiled, timed, and still computing the
# original -- not whether it won. Requiring a speedup here voided runs that did
# the task and did it badly, which is exactly the outcome a regression baseline
# needs to be able to record. Run D profiled four variants, one of them faster,
# and scored nothing because the fast one was numerically wrong.
#
# "Beats the original" remains check 1, where a loss costs a point instead of
# the whole trial.
MEASURED=$(python3 - "$LOGS/traces.json" <<'PYM'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0); raise SystemExit
print(sum(1 for s in d.get("sessions") or []
          if s.get("device_us") and not s.get("is_baseline")
          and s.get("numerics") == "ok"))
PYM
)
[ "${MEASURED:-0}" -ge 1 ] || { echo "  GATE FAIL  no variant was profiled that still computes the original"; GATE=0; }
if [ "$GATE" -ne 1 ]; then
  # Distinguish "the agent could not do it" from "the model was unavailable".
  # Both write reward 0, so without this a scored comparison silently averages
  # infrastructure outages into the result -- and a control arm that absorbs
  # them is not a control. Four of six runs during development died this way.
  # A refusal is not an outage. Every long run collects a few, and the harness
  # retries them; what marks a trial as unfair is refusals heavy enough to have
  # stopped the work. Run J failed the gate honestly -- it profiled five
  # variants, and every one was slower, unreadable or numerically wrong -- and a
  # single transient refusal would have excused it as infrastructure, hiding a
  # real agent failure from the comparison the classifier exists to protect.
  #
  # Threshold, not presence. Ten is above the noise every healthy run shows
  # (runs I and J: 0 and 1) and far below a genuine outage (E: 60, G: 110,
  # H: 152, run 16: 190).
  INFRA="$(refusal_census)"
  INFRA_N="$(printf '%s' "$INFRA" | tr ';' '\n' | sed -n 's/.*x\([0-9]\+\)$/\1/p' \
             | sort -rn | head -1)"
  [ "${INFRA_N:-0}" -lt "${INFRA_MIN:-10}" ] && INFRA=""
  echo 0 > "$LOGS/reward.txt"
  if [ -n "$INFRA" ]; then
    echo "  NOTE: model-side failures present -- $INFRA"
    echo "  This trial did not get a fair attempt; exclude it from any scored comparison."
    printf '{"reward":0,"reason":"infrastructure","infra_failure":true,"infra_errors":"%s","sessions":%s,"custom_call_sessions":%s}\n' \
      "$INFRA" "${NSESS:-0}" "${NCC:-0}" > "$LOGS/correctness.json"
  else
    printf '{"reward":0,"reason":"gate","infra_failure":false,"sessions":%s,"custom_call_sessions":%s}\n' \
      "${NSESS:-0}" "${NCC:-0}" > "$LOGS/correctness.json"
  fi
  report_signals
  echo "FAIL: no measured optimization happened"; exit 1
fi
echo "  GATE OK  $NSESS profiles, $NCC with a kernel"

echo "=== correctness (scored) ==="
# 1. The point of the task, and by how much. Graded rather than binary since
#    run 8: that run produced no kernel at all, chunked the computation in plain
#    JAX for 1.06x, and scored 6.5/7 -- higher than the run that wrote real
#    Pallas flash attention for 9.6x and was merely sloppy about which variant
#    it kept. A pass/fail "is it faster" cannot tell those apart, and on a task
#    called attention-kernel-tuning that ordering is backwards.
#
#    Thresholds from the measured series, not taste: the baseline is ~6360 us,
#    a working flash kernel lands at 659-780 us (8-9.6x), and blocked variants
#    that genuinely help land well past 2x. 1.06x is what you get from
#    rearranging the same XLA ops.
#
#    Four tiers rather than three, because a ceiling at 2x made 2.27x and 9.6x
#    score the same and a regression from one to the other invisible -- every
#    check would still have passed. 5x is set where it is because 659 us was
#    actually reached on this workload, and reached *without* a custom call, so
#    the top tier is earnable rather than aspirational.
if [ -n "$BASE" ] && [ -n "$BEST" ] && awk "BEGIN{exit !($BEST < $BASE)}"; then
  SPEEDUP="$(awk "BEGIN{printf \"%.2f\", $BASE/$BEST}")"
  if awk "BEGIN{exit !($BASE/$BEST >= 5.0)}"; then
    check_graded "beats the original by ${SPEEDUP}x ($BEST us < $BASE us)" 1
  elif awk "BEGIN{exit !($BASE/$BEST >= 2.0)}"; then
    check_graded "beats the original by ${SPEEDUP}x -- short of the 5x this workload allows" 0.75
  elif awk "BEGIN{exit !($BASE/$BEST >= 1.25)}"; then
    check_graded "beats the original, but only ${SPEEDUP}x ($BEST us < $BASE us)" 0.5
  else
    check_graded "beats the original by ${SPEEDUP}x -- within noise of the baseline" 0.25
  fi
else check "beats the original (best=${BEST:-?} baseline=${BASE:-?})" 0; fi

# 1a. Did it write a kernel at all. Reported by the gate since the beginning
#     ("6 profiles, 0 with a kernel") and scored by nothing, which is how a run
#     with zero custom calls took the top score in the series. Kept out of the
#     gate deliberately: an agent that tries for a kernel and cannot land one
#     should still be graded on everything else, and scoring 0 outright would
#     make "tried and failed" indistinguishable from "never tried".
if [ "${NCC:-0}" -ge 1 ]; then
  check "produced a custom-call kernel ($NCC of ${NSESS:-?} profiles)" 1
else
  check "produced a custom-call kernel (none in ${NSESS:-?} profiles)" 0
fi

# 1c. Did it stay inside the profiling budget. Scored, not merely reported:
#     the budget is the one instruction in the prompt with a number attached,
#     so overrunning it is an instruction-following failure and the cheapest
#     honest test of whether the agent read its constraints. Extra captures
#     were already being dropped from the analysis (`dirs[:cap]`), which meant
#     a run could take nine, have three quietly ignored, and score as though it
#     had taken six -- run I reported a figure from an unscored seventh.
#
#     Counted from what landed on disk, not from what the agent says it did.
#     The verdict comes from `analyze.py`, which settles it in one place and is
#     unit-tested; this reads `budget_ok` rather than re-deriving the rule in
#     shell where nothing can exercise it.
BUDGET="$(python3 -c "
import json
try: d=json.load(open('$LOGS/traces.json'))
except Exception: raise SystemExit
print(int(bool(d.get('budget_ok'))), d.get('profiles_taken') or 0, d.get('profile_cap') or 0, d.get('over_cap') or 0, int(bool(d.get('ledger_discrepancy'))), d.get('captures_logged') if d.get('captures_logged') is not None else -1)" 2>/dev/null)"
# `read`, not `set --`: this runs at the top level, where `set --` would
# discard the script's own arguments.
read -r BUDGET_OK NPROF PCAP OVER SKEW LOGGED <<<"${BUDGET:-1 ? ? 0 0 -1}"
# Reported, not scored. Deleting a capture no longer buys another -- the
# workload counts a ledger as well as the directories -- so what is left is a
# run whose evidence does not match its own record. That is worth a reader's
# attention and not, on its own, proof of bad faith: a failed capture directory
# is a plausible thing to tidy away.
[ "${SKEW:-0}" = "1" ] && echo "  NOTE: $LOGGED captures recorded, $NPROF on disk -- profiles were removed after capture"
if [ "$BUDGET_OK" = "1" ]; then
  check "stayed within the profiling budget (${NPROF:-?} of cap ${PCAP:-?})" 1
else
  check "stayed within the profiling budget (${NPROF:-?} of cap ${PCAP:-?} -- $OVER over)" 0
fi

# 1b. And still computes it. The prompt says "I need the same computation,
#     faster"; until now nothing checked the first half. Run 13 shipped a 9.4x
#     speedup whose output checksum was off in the second digit, noticed the
#     discrepancy in its own reasoning, and reported it as the result anyway.
#     Decided from run.json beside the traces, never from the agent's prose.
# First-party parity, run on CPU against a frozen copy of the original. This is
# the check the skill calls mandatory for exactly this case, and it is stronger
# than comparing checksums: multi-regime inputs and a ULP verdict rather than
# one number that could coincide. Measured on run 13's rewrite -- `false`, max
# ULP 33403, 4 of 5 batches failed -- and `true`, 0 ULP, for the original
# against itself.
#
# Inconclusive rather than failing when the candidate cannot run here: a Pallas
# kernel needs the chip, and "we could not execute it on CPU" is not evidence of
# anything. The checksum comparison below covers that case, because the TPU run
# already produced the numbers.
PARITY=""
PARITY_DIR="$(mktemp -d)"
AGENT_ATTN="$ART/workload/attention.py"
if [ -f "$AGENT_ATTN" ] && [ -f "$HERE/reference_attention.py" ]; then
  cp "$HERE/reference_attention.py" "$PARITY_DIR/reference_attention.py"
  cp "$AGENT_ATTN" "$PARITY_DIR/candidate_attention.py"
  PARITY="$(cd "$PARITY_DIR" && PYTHONPATH="$PARITY_DIR" timeout 300 "$XPROF_BIN" \
      verify_numerical_parity \
      --kernel_ref="reference_attention.attention" \
      --kernel_candidate="candidate_attention.attention" \
      --shapes="[(2,256,64),(2,256,64),(2,256,64)]" \
      --dtype_str="bfloat16" --tier="fast_agent" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
if d.get('status') == 'ERROR' or d.get('is_numerically_equivalent') is None:
    raise SystemExit  # could not run it here; say nothing
ok = d['is_numerically_equivalent']
print(f\"{1 if ok else 0}|xprof verify_numerical_parity: \"
      f\"{'equivalent' if ok else 'DIVERGENT'}, max ULP {d.get('overall_max_ulp')}, \"
      f\"{d.get('failed_batches_count')}/{d.get('total_batches_count')} batches failed\")
")"
fi
rm -rf "$PARITY_DIR"

NUM="$(python3 - "$LOGS/traces.json" <<'PYN'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("0|no traces"); raise SystemExit
sess = d.get("sessions") or []
if d.get("baseline_checksum") is None:
    print("0|no checksum from the baseline run -- did the harness still write run.json?")
    raise SystemExit
wrong = d.get("wrong_but_faster") or []
# What it *shipped*, which is the last thing it profiled -- not the fastest
# thing that happens to be correct. Run 13 left a wrong variant in place while
# a correct one sat earlier in the run; judging on "is some correct session
# present" would have called that a pass.
kept = [s for s in sess if s.get("device_us") and not s.get("is_baseline")]
if not kept:
    print("0|no variant was profiled"); raise SystemExit
last = kept[-1]
name = last["dir"].split("/")[-1]
note = (f"; also tried {len(wrong)} faster-but-wrong "
        f"({', '.join(w['dir'] for w in wrong[:3])})") if wrong else ""
if last.get("numerics") == "unknown":
    print(f"0|{name} wrote no checksum; cannot verify it{note}")
elif last.get("numerics") in ("mismatch", "wrong_dtype"):
    print(f"0|{name} computes something else "
          f"(checksum off by {last.get('rel_diff', 0):.2g}){note}")
else:
    print(f"1|{name} matches the original "
          f"(rel {last.get('rel_diff', 0):.2g}){note}")
PYN
)"
# On-chip evidence decides; CPU parity is corroboration.
#
# Both statements are about equivalence, but only one is about the thing that
# ran. The fingerprints come from the baseline and the candidate executing on
# the same TPU, so the hardware's precision -- bf16 multiply with f32
# accumulate, the MXU's rounding, whatever XLA reassociated -- is common to both
# and what remains is the algorithm. A CPU parity check controls for hardware
# too, but for the wrong hardware: it can show two formulas agree on a CPU, not
# that the kernel that ran agrees on the chip. A tiling that changes
# accumulation order is exactly the case it would miss.
#
# So parity is reported alongside, and only decides when there is no on-chip
# evidence to decide from -- which is the case when the agent replaced the
# harness and stopped writing run.json.
[ -n "$PARITY" ] && echo "  also: ${PARITY#*|}"
if [ "${NUM%%|*}" = "1" ] || [ -z "$PARITY" ]; then
  check "the result it kept still computes the original -- ${NUM#*|}" "${NUM%%|*}"
else
  check "the result it kept still computes the original -- ${PARITY#*|}" "${PARITY%%|*}"
fi

# 2. Did it iterate, or take the first thing that compiled. Counted as tilings
#    where the trace names them and as profiled variants otherwise -- the
#    question is whether it searched, not whether it searched with the library
#    whose metadata I knew how to read.
NTILE_DESC="${NTILE_EXACT:-0} tilings"
[ "${NTILE_EXACT:-0}" -ge 2 ] || NTILE_DESC="${NVAR:-0} variants (kernel writes no tile metadata)"
if [ "${NTILE:-0}" -ge 2 ]; then
  check "tried more than one configuration -- $NTILE_DESC" 1
else
  check "tried more than one configuration -- $NTILE_DESC" 0
fi

# 3. Did it keep the best thing it measured. "Did the last attempt beat the
#    first" punishes an agent whose first try was already optimal, which is not
#    a failure of anything. What matters is that the configuration it left in
#    place is the one its own traces show to be fastest.
#
#    Judged from the last trace, not from the FINAL line: the prompt asks for a
#    time and a description, never for tile numbers, so grepping prose for them
#    failed an agent whose final profile *was* the fastest tiling and whose only
#    sin was calling it "optimized execution". Prose still counts when it does
#    name them. Near-ties are ties: 1932.9 and 1935.6 us are the same config
#    measured twice.
LAST_TILES="$(python3 - "$LOGS/traces.json" "$(line_with 'FINAL')" <<'PYT'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("|no measurements recovered"); raise SystemExit
best, sess = d.get("best_tiling"), d.get("sessions") or []
kept = [s for s in sess if s.get("tiles")]
last = kept[-1]["tiles"] if kept else None
if not best:
    # No tile metadata: a kernel the agent wrote itself names nothing. Ask the
    # same question of the timings instead -- is the variant it left in place
    # the fastest one it measured, within noise.
    profiled = [s for s in sess if s.get("device_us")
                and not s.get("is_baseline")]
    timed = [s for s in profiled if s.get("numerics") == "ok"]
    if not timed:
        print("|no measurements recovered"); raise SystemExit
    # What it *shipped* is the last variant it profiled, valid or not -- the
    # same definition the parity check above uses. Taking the last *valid* one
    # instead credits an agent for discipline it did not exercise: run 26 left
    # a mismatching variant in place, and this check passed anyway by quietly
    # substituting the correct session two profiles earlier. A disqualified
    # result cannot be kept, satisfy the gate, or win -- including here.
    shipped = profiled[-1]
    if shipped.get("numerics") != "ok":
        print(f"|kept {shipped['dir'].split('/')[-1]}, which is "
              f"{shipped.get('numerics')} -- a disqualified result is not a "
              f"configuration it can be credited for keeping")
        raise SystemExit
    fastest = min(timed, key=lambda s: s["device_us"])
    ok = shipped["device_us"] <= fastest["device_us"] * 1.03
    print(f"{'1' if ok else ''}|kept {shipped['dir'].split('/')[-1]} "
          f"({shipped['device_us']:.0f} us); fastest measured was "
          f"{fastest['dir'].split('/')[-1]} ({fastest['device_us']:.0f} us)")
    raise SystemExit
bt = f"{best[0]}/{best[1]}"
ok = last == best
if not ok and last and kept:
    bu, lu = d.get("best_kernel_us"), kept[-1].get("device_us")
    ok = bool(bu and lu and lu <= bu * 1.03)  # same config, measured twice
if not ok:
    txt = sys.argv[2] if len(sys.argv) > 2 else ""
    ok = all(str(x) in txt for x in best[:2])
print(f"{'1' if ok else ''}|{bt}, last trace {last}")
PYT
)"
if [ "${LAST_TILES%%|*}" = "1" ]; then
  check "kept the fastest configuration it measured (${LAST_TILES#*|})" 1
else
  check "kept the fastest configuration it measured (${LAST_TILES#*|})" 0
fi

# 4. LIBTPU_INIT_ARGS reached the backend in time. Set too late and the run
#    still exits 0 with a clean, empty trace -- a silent failure.
LLO_WHY="$(python3 -c "
import json
try: print(json.load(open('$LOGS/traces.json')).get('llo_detail') or 'no detail')
except Exception: print('no detail')")"
if [ "$LLO" = "True" ]; then
  check "LLO data present -- $LLO_WHY" 1
else
  check "LLO data absent -- $LLO_WHY (flags set after backend init?)" 0
fi

# 5. The cited number is one it could only have got by measuring.
FINAL_LINE="$({ grep -ohE 'FINAL:[^"\\]{0,160}' "${AGENT_LOGS[@]}" 2>/dev/null || true; } | tail -1)"
FINAL_US="$(echo "$FINAL_LINE" | grep -oE '[0-9]+\.?[0-9]*' | head -1)"
echo "  FINAL line: ${FINAL_LINE:-<none>}"
# Either reference is defensible: the session total, or the kernel's own
# duration. The prompt asks for "the attention computation"; the session total
# also carries the harness's own work, so an agent citing the kernel is
# arguably more correct than one citing the total. Accept both.
KREC="$(python3 -c "
import json
try:
    v=json.load(open('$LOGS/traces.json')).get('best_kernel_record_us')
    print('' if v is None else v)
except Exception: print('')")"
# Four defensible references, not one. Agents cite per-iteration time as often
# as totals -- the harness runs 10 iterations, so a 2522 us session and a cited
# 252 us are the same measurement. Three runs running, this check called the
# agent wrong while the agent was right. One python call, no shell arithmetic.
MATCH="$(python3 - "$LOGS/traces.json" "${FINAL_US:-}" "${WORKLOAD_ITERS:-10}" <<'PYM'
import json, sys
path, final, iters = sys.argv[1], sys.argv[2], float(sys.argv[3] or 10)
if not final:
    print("none"); raise SystemExit
v = float(final)
try: d = json.load(open(path))
except Exception: print("none"); raise SystemExit
refs = {}
for name, key in (("session", "best_kernel_us"), ("kernel", "best_kernel_record_us")):
    x = d.get(key)
    if isinstance(x, (int, float)) and x > 0:
        refs[name] = x
        refs[name + "/iter"] = x / iters
# Within 10%, not within a factor of two. The four references above already
# absorb the reason the band was wide -- session vs kernel, total vs
# per-iteration -- so anything left over is a citation that does not match.
# Genuine matches land within 0.2%: 193.6~193.7, 437.7~437.2, 193~193.4. The
# old band passed run I citing 136.28 against 100.8, a 35% gap, because its
# real answer came from a seventh profile the grader never read.
for name, r in refs.items():
    if 0.90 < v / r < 1.10:
        print(f"{name}={r:.4g}"); raise SystemExit
print("none")
PYM
)"
if [ "$MATCH" != "none" ]; then
  check "reported device time matches a trace figure ($FINAL_US ~ $MATCH)" 1
else
  check "reported device time matches a trace figure (said '${FINAL_US:-none}'; session $BEST, kernel ${KREC:-n/a})" 0
fi

# 6. How well, not just whether. Graded, and the only check that is.
#
#    The other five saturate: a run that ports to a kernel, tries two tilings
#    and stops scores the same 1.000 as one that tunes properly. Measured -- the
#    package arm reached 1937 us and the control 2483 us, 22% apart, both 5/5.
#
#    Graded on the gain *after* the first kernel that beat the original, not on
#    total speedup. Total speedup is dominated by the port from JAX to Pallas,
#    which both arms manage; the tile search is the skill this task exists to
#    measure. On the two real runs that separates them cleanly -- 1.281x against
#    1.000x.
#
#    Half credit for a flat search that was genuinely a search: an agent whose
#    first working tiling happens to be near-optimal has not failed at anything,
#    and three or more distinct tilings is evidence it looked. Thresholds are
#    overridable because they are calibrated on v6e, where ~1.28x is the whole
#    tuning headroom; another chip has another.
TUNE_FULL="${TUNE_FULL_GAIN:-1.10}"
TUNE_MIN_TILINGS="${TUNE_MIN_TILINGS:-3}"
GRADE="$(python3 - "$LOGS/traces.json" "$TUNE_FULL" "$TUNE_MIN_TILINGS" <<'PYG'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("0|no traces"); raise SystemExit
full, need = float(sys.argv[2]), int(sys.argv[3])
first, best = d.get("first_working_kernel_us"), d.get("best_kernel_us")
# Breadth, however the run expressed it: named tilings when the trace has
# them, profiled variants when the agent wrote its own kernel.
n = d.get("configs_tried") or len(d.get("distinct_tilings") or [])
if not first or not best:
    print("0|no working kernel to tune from"); raise SystemExit
gain = first / best
if gain >= full:
    print(f"1|tuned {gain:.2f}x past its first working kernel "
          f"({first:.0f} -> {best:.0f} us, {n} configs)")
elif n >= need:
    print(f"0.5|searched {n} configs but gained only {gain:.2f}x "
          f"({first:.0f} -> {best:.0f} us)")
else:
    print(f"0|stopped at its first working kernel ({gain:.2f}x, {n} configs)")
PYG
)"
check_graded "tuned the kernel it got working -- ${GRADE#*|}" "${GRADE%%|*}"

report_signals

SCORE=$(awk "BEGIN{printf \"%.3f\", $pass/$total}")
PASS_FMT=$(awk "BEGIN{printf (($pass==int($pass))?\"%d\":\"%.1f\"), $pass}")
echo "$SCORE" > "$LOGS/reward.txt"
python3 - "$pass" "$total" "$SCORE" "$LOGS" "${REFUSALS:-}" <<'PY' > "$LOGS/correctness.json"
import json, sys
p, t, s, logs = sys.argv[1:5]
refusals = sys.argv[5] if len(sys.argv) > 5 else ""
tr = json.load(open(f"{logs}/traces.json"))
sg = json.load(open(f"{logs}/signals.json"))
# `p` is not always an integer: a half-credit check ("PART") makes it 6.5,
# and `int("6.5")` raises. The traceback went to stderr, the redirect had
# already truncated the file, and the verdict was written as zero bytes --
# taking `infra_failure` with it, on the one run in the series that earned a
# PART. Whole numbers still serialise as ints so existing consumers see no
# change.
_p = float(p)
json.dump({"correctness_passed": int(_p) if _p.is_integer() else _p,
           "correctness_total": int(t),
           "correctness_score": float(s),
           "baseline_us": tr.get("baseline_us"), "best_kernel_us": tr.get("best_kernel_us"),
           "first_kernel_us": tr.get("first_kernel_us"),
           "first_working_kernel_us": tr.get("first_working_kernel_us"),
           "tuning_gain": (round(tr["first_working_kernel_us"] / tr["best_kernel_us"], 3)
                           if tr.get("first_working_kernel_us") and tr.get("best_kernel_us")
                           else None),
           "distinct_tilings": tr.get("distinct_tilings"),
           "model_refusals": refusals or None,
           # Explicit, on every verdict. A consumer should not have to infer
           # "did this trial get a fair attempt" from the presence of a string.
           "infra_failure": False,
           "integration": sg}, sys.stdout, indent=2)
PY
echo "score: $PASS_FMT/$total = $SCORE"
exit 0
