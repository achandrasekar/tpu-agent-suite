# Why a round is shaped the way it is

Each rule below was expensive to learn. They are recorded here so a future
change has to argue with the reason rather than only with the rule.

## One run, one study, and what follows from it

Three relations, enforced rather than assumed:

```
study -> round    1:many   the only long-lived scope
round == run      1:1      enforced by `attach-run`
session           a harness correlation key, not a domain concept
```

`attach-run` refuses a run already attached to any study, open or closed.
That single rule is what the rest of the model rests on, and three things
follow from it rather than needing separate arrangements.

An event routes to exactly one study, because ownership is a fact to read
rather than a guess to make. Before it, one open study absorbed every session
on the machine, and another study's `fully_idle` in your stream is not noise,
it is a false liveness signal.

The round counter can be the count of attached runs, because they are the
same thing. Two counters that can disagree eventually will.

And `session` stops being something the study layer models. It is the
harness's identifier for a conversation, useful for correlating a log, and
nothing in the study layer should branch on it. If you find yourself adding a
session case here, the ownership question has moved somewhere it does not
belong.

**One round per session is therefore a constraint, not a convention.** A
future batching mode that ran several rounds in one session would need this
rule relaxed rather than extended. That is deliberate: batching rounds
reintroduces the context growth the whole design exists to prevent.

One consequence worth knowing. `SessionStart` fires before any agent turn, so
at that moment the run exists and belongs to nothing, and a strictly-routed
session-start event could never reach a study. The attach emits the marker
instead, at the first moment it can be attributed.

## One arm per round, then exit

The alternative — a driver that keeps going until the study ends — puts every
arm's plan, dispatch, output and reasoning into one context. Cost per arm then
climbs with the number of arms already done, and the run dies of its own
history somewhere in the middle of the night.

Exiting after one arm makes round 60 cost what round 2 cost. That single
property is what turns "an optimization loop" into "an optimization loop that
can run for days".

## The study state is the only thing a round reads

Not the event stream, not the overflow journal, not any artifact, not the
scrollback. A round that reads one artifact "just to check" has reintroduced
unbounded growth through the back door, because the next round will read two.

Artifacts are referenced by path so a human or a specialist can open them. The
orchestrator passes the path along without opening it.

## The orchestrator holds no tools that can do the work

This package's router agent is read-only for a reason: an orchestrator that
holds the tools to do the work will do the work. With the full contract in its
prompt naming every specialist, continuing with tools already in hand never
presents the decision to delegate, so the delegation simply does not happen.

A round needs `write` for the study state and `shell` for the runtime CLI, so it
cannot be locked down as hard. The compensating rule is stated in the round's
own contract: never write kernel code, never edit a recipe, never read a trace.
If a round finds itself debugging, that is the bug.

## Tie is the default verdict

**The effects being chased are smaller than the noise of a careless harness.**
So a positive delta is a tie until it clears the noise floor at the
pre-registered replicate count.

Two published results make the point better than an assertion would.[^noise]
Benchmark noise in a large continuous-integration system runs into double
digits on sensitive tests, and change-point detection -- a real improvement
over threshold alerting -- still leaves a false positive rate most of the way
to a coin flip. Meanwhile a week of autonomous kernel evolution on current
hardware bought low single-digit percent over a vendor library.

A win is therefore the unusual outcome, and a round reporting a large one
should be treated as a harness fault until it survives a repeat.

[^noise]: Ingram et al., *The Use of Change Point Detection to Identify
    Software Performance Regressions in a Continuous Integration System*
    (MongoDB, ICPE 2020), arXiv:2003.00584; and the AVO agentic kernel
    evolution results, arXiv:2603.24517. Figures are theirs, not ours, and
    are cited rather than restated so a reader can check them.

## Gates are bound to the same execution as the metric

A metric and its gates measured on different executions do not describe the
same configuration, however identical the configuration was. This is not
hypothetical: a champion has been crowned on throughput and withdrawn later
when the gate was finally applied to the same run, and a whole corpus of
throughput numbers can accumulate with no co-located accuracy score at all.

A metric without its gates, measured together, is not a weaker result. It is
not a result.

## The protocol is pre-registered

The primary metric, the gates, the measurement cell, the replicate count and the
stopping rule are fixed before the baseline runs. Deciding how many times to
measure *after* seeing the first number is deciding it too late, and it is how
a post-hoc rationalisation acquires the authority of a plan.

Amending a protocol is legal. Amending it silently is not: the revision changes
and the superseded version stays.

## Provenance labels are a closed set

`measured`, `inferred`, `modeled`, `projected`. Not `verified`, not `confirmed`,
not `derived`. A modelled ceiling reported where it reads as measured is the
most common way a study starts believing its own predictions.

## The empty queue

Nothing open, nothing in flight, nothing blocked *looks* like success and is the
most dangerous state in an unattended run. Overnight there is nobody to notice
that the cause was "every arm failed to launch" rather than "every arm is
decided". So it is a blocked row and a stopped round, never an invented arm.
