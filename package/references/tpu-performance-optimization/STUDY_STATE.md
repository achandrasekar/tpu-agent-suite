# The study state, and the two cold streams beside it

```
L1  study          the effort. Outlives every session it spends.
 └─ L2  round      one step of it. Carried out by exactly one run.
     └─ L3  arm    one candidate change under test.
         └─ L4  execution   one compute job that measures it.
```

`L1` to `L4` label the levels for reading only. No field carries an `L` prefix;
`study_id`, `arm` and `execution_id` are the names in the files. The run is not
a level: it is the session you are in, and the round is the study step that
session carries out. That pairing is enforced — a run attaches to at most one
study, so the round number is the count of attached runs, and a retry is a new
round rather than a second attempt at the same one.

Four files live under a study. They have different writers, different readers
and different rules, and conflating any two of them breaks something.

| file | written by | read by a round? | is it a recovery source? |
| :--- | :--- | :--- | :--- |
| `study.json` | the round, through the runtime | **yes, and only this** | yes — the only one |
| `hypothesis_bank.json` | `hypothesis_bank.py`, under study lock | no (sidecar for hypothesis metadata) | no |
| `events.jsonl` | lifecycle hooks, automatically | no (bounded tail only) | **never** |
| `studies/<study_id>/journal.jsonl` | the round, on overflow | no (bounded tail for dedup in `hypothesis_bank.py`) | never |

## Opening a study

A study is opened by a person, before any round runs. Nothing in the package
opens one, and a round handed a `study_id` that does not exist stops rather
than creating it.

```bash
{{runtime}} study start \
    --study-id qwen35-397b-ttft \
    --package tpu-agent-suite \
    --target Qwen3.5-397B-A17B-FP8 \
    --protocol protocol.json
```

`--study-id`, `--package`, `--target` and `--protocol` are required.
`--owner` is optional and defaults to the invoking user.

`--target` is what is being optimized, and it is the grouping key: "every
study on this model" is the query people actually run, and a free-form study
id cannot stand in for it. It lives on the study document rather than inside
the protocol, because the protocol is amendment-governed -- correcting a
misspelt name must not produce a formal amendment to what the experiment
measures.

`--protocol` points at a file matching `study-protocol.json`: the primary
metric, the admissibility gates, the measurement cell, the replicate count, the
noise floor, the stopping rule, the mode, and the cap when the mode is `auto`.
The schema ships beside the runtime and carries a worked example.

It is required because pre-registration is the whole claim. A study opened
without one has no stopping rule, so nothing ends it; no noise floor or
replicate count, so the win condition is invented round by round; no gates, so
everything is admissible; and no cap, so `auto` spends until something else
stops it. Every one of those fails silently while the board reports the study
open, fresh and healthy.

Deciding these after the first measurement is deciding them too late, which is
how a number that was going to be a tie becomes a win.

Amending a protocol later is legal and recorded. Amending it silently is not.

## The verbs

Everything reaches the study through the runtime. Nothing opens `study.json`
by path -- that is the drift these files exist to prevent.

| verb | who runs it | what it does |
| :--- | :--- | :--- |
| `start` | human | opens a study; needs a protocol |
| `set-baseline` | human or round | the number every arm is judged against |
| `attach-run` | round | claims this session's run for the study |
| `record-round` | round | the only write to the state |
| `append-journal` | round | archives a row that overflowed a cap |
| `set-protocol` | human | amends the terms, with a written reason |
| `status` | anyone | the state, plus the derived round |
| `notes` | human | the readable view, rendered on demand |
| `events` / `journal` | monitor | the two cold streams |
| `claim-arm` | worker or round | atomically claims an open arm for parallel execution (and reclaims stale `in_flight` leases past `--claim-ttl-seconds`) |
| `release-arm` | worker or round | releases an `in_flight` arm back to `open`, `blocked` (with `--blocked-on`), or `invalid` (with `--summary`) |
| `list` | anyone | every study, freshest first |
| `close` | round | ends it, with a one-line outcome |

If a worker fails a preflight check or hits a setup error during a round, it calls `release-arm` (`--status open`, `blocked`, or `invalid`). If a worker crashes mid-round without calling `release-arm` or `record-round`, the next `claim-arm` automatically expires any `in_flight` lease older than `--claim-ttl-seconds` (default 3600s) back to `open` so the arm is never permanently lost.

## Setting the baseline

Every arm is judged against one measurement. It is set once, before any arm is
decided, and until it exists **no arm may report a delta** -- the runtime
refuses the write, because a delta is a comparison and there would be nothing
on the other side of it.

```bash
{{runtime}} study set-baseline --study-id <id> --baseline baseline.json
```

`baseline.json` matches `study-baseline.json` and requires four things:

- the **execution** that produced it, so a later reader can find it
- the **metric**, which must be the protocol's primary metric, must be
  `measured` rather than modelled, and must meet the pre-registered replicate
  count. Every arm is compared against this number, so measuring it less
  carefully than the arms is backwards, and a modelled one turns the study
  into an evaluation of its own prediction.
- a **profile**. Without per-op attribution the first round has nothing to
  rank arms by and falls back to guessing which layer to attack, which is the
  failure the whole design exists to avoid.
- a **fingerprint**: commit sha, config digest, measurement cell. Reuse is not
  built yet, but nothing can decide whether an existing baseline is reusable
  without these, and an end-to-end run costs hours.

If the baseline was measured by someone else rather than by this study, set
`adopted_from_run`. An adopted baseline must be visibly adopted; a silently
reused number is how a study ends up compared against something nobody can
locate.

## `study.json` — the study state

The one hot file. A round reads it whole at every decision point and branches on
what it just read, never on what it remembers reading earlier.

It is reached through the runtime rather than by path:

```bash
{{runtime}} study status --study-id <id>
```

Written back whole, never patched. A partial write on an interrupted round is
the one corruption this design refuses to be exposed to, and the runtime's
writer makes the replacement atomic so a monitor polling the file sees either
the old document or the new one, never a torn middle.

`updated_at` is stamped from the clock inside the write path. It is the liveness
signal — the thing that distinguishes a study that is working from one that died
at 01:00 — so it is never a value a caller supplies.

### Caps are load-bearing

Every list section has a cap (`16` for `arms`, `10` for `amendments`). When a
section overflows, the weakest row moves to `journal.jsonl` **in the same round
that overflows it**, oldest first (`invalid` → `loss` → `tie` → `open` →
`in_flight` → `win` for `arms`, so proven `win` and `in_flight` arms outrank
untested `open` candidates). Never later, and never by deleting.

The caps exist so that round 60 costs what round 2 cost. A study state that grows
without bound is read in full every round, so its growth is paid for on every
single round thereafter, forever. Both of the loops this package delegates into
have this problem today — one keeps an unbounded `history` array inside its
state file, the other an unbounded trajectory log inside its own markdown
ledger — which is precisely the thing not to copy.

## `events.jsonl` — machine-authored, for watching

Appended by lifecycle hooks, one JSON object per line. **Hooks record, not
agents, because a failing agent is the one most likely to stop reporting.**
Monitoring that depended on the agent would fail at the same moment the agent
did.

Two hook events reach a study, plus one marker:

| written on | what it says |
| :--- | :--- |
| `SubagentStop` | a dispatched specialist finished |
| `Stop` | the session ended, with its reason and `fully_idle` |
| attach | a run joined this study, emitted by `attach-run` |

`SessionStart` is deliberately absent. It fires before any agent turn, so the
run exists and belongs to no study yet, and a strictly-routed session-start
event could never be attributed to one. The attach marker carries the same
information at the first moment it can be.

`fully_idle` separates "the model stopped talking" from "no work is
outstanding", which is the difference between a finished round and a stalled
one. It is the field a monitor actually reads.

A monitoring daemon or a dashboard reads this freely — neither has a context
window to spend. A round does not, beyond a bounded tail.

**It is never a recovery source.** If the study state is gone, the round stops and
says so. Rebuilding state from events yields a second source of truth, and the
one that gets believed is whichever was read most recently.

## `journal.jsonl` — agent-authored, for the record

`studies/<study_id>/journal.jsonl`. Not the session journal
(`{{runtime}} journal`), which belongs to one run and sits beside its
`artifacts/`. Where capped sections overflow to. Unbounded, append-only, never opened by a
round (a bounded tail, `--journal-dedup-limit`, default 64, is read by
`hypothesis_bank.py` for dedup). It exists so that a dropped row is archived
rather than lost, and so a human reconstructing what happened on the third
night has something to read.

## Why two streams and not one

Different writers, different schemas, different readers. A fixed-schema event
stream that also holds free-form overflow cannot be validated, and an overflow
archive interleaved with machine events cannot be cheaply tailed. Keeping them
apart costs one file and buys both properties.
