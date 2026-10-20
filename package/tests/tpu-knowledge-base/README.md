# Testing tpu-knowledge-base

Nothing in this directory ships. The manifest's `copy` list is `index`,
`pages`, `prompts`, `templates`, and a build confirms no file from here reaches
`dist/`. That is deliberate and load-bearing twice over: `golden.jsonl` and
`answers.jsonl` hold the answers, so an agent that could read them would be
marking its own work; and
the notes below describe a testing mode that is not a thing a user should ever
be told to do.

Which is why this file exists rather than a section in
`instructions/tpu-knowledge-base.md`. The agent contract describes live usage. This
describes how the pipeline is checked, and the two audiences must not be
confused — an instruction to generate the same entry three times and compare is
sensible for a test and wasteful nonsense for a caller with a question.

## The three layers

| | what it asks | how to run |
| --- | --- | --- |
| unit tests | does each piece behave | `make test-package` |
| Harbor suite | is a generated entry *correct* | `eval/run.py -c claude-code/agent-suite --dataset tpu-knowledge-base` |
| variability | is it *reliably* right | `python3 tests/check_stability.py <keyword>` |

Two surfaces, not three. The unit tests are offline, deterministic and run on
every commit. The golden set is now twenty-one Harbor cases -- thirteen
keywords, six `ask` questions, and the entry and answer critic mutations -- and
`golden.py` is gone: the row loop is the cases,
`--repeats` is `n_attempts`, the tally is `reward.toml`.

Variability is the exception, below, and it is a script rather than a surface.

Running one keyword locally, which is the loop for anything under development:

    python3 scripts/index.py get pallas-tpu        # the real cache-miss path
    python3 tests/check_entry.py pallas-tpu        # what Harbor runs, verbatim
    python3 tests/check_critic.py                  # the six critic mutations

And one question, for `ask`. Answers are never cached, so the reply is saved
and handed to the checker by path; `answers.jsonl` holds the rows:

    python3 scripts/index.py ask tpu7x "What is the HBM capacity per chip of TPU7x?" --json > /tmp/a.json
    python3 tests/check_answer.py tpu7x-hbm --answer /tmp/a.json
    python3 tests/check_answer_critic.py           # the four answer-critic mutations

## Variability, and why it stays local

The variability pass generates the same keyword several times and asks whether
the answers agree. What it compares is the *shape* -- the set of headings, the
count of `not stated` -- and not the text or the numbers, because comparing
those fails on every run for reasons that are not faults. That was measured
rather than assumed: across three runs of three keywords the headings were
identical every time while the number set was not, and every number was
grounded. Runs legitimately differ in how much detail they report.

It is deliberately **local-only and on demand**. Two reasons, and the first is
not cost -- the pass is about 27 model calls and 115 K input tokens, roughly
half of one correctness run, and cheaper than a single generation of
`vllm-engine-args`.

The reason is that Harbor cannot express it. `n_attempts` multiplies the trial
list and each attempt is an independent trial with its own container and its
own verifier run, so no verifier can see another attempt's output. The only way
to run it inside a trial is to have the *verifier* generate, and then the
trial's score stops meaning "did the agent do this right" and starts meaning
that blended with "is the generator stable" -- two questions under one number.

Second: it is the check with the least evidence behind it. Three keywords, and
its clearest finding so far was negative. Keeping it available and manual costs
nothing against the status quo, because it has always been opt-in and has never
run in CI.

If it should ever become automatic, the way to do it without corrupting a trial
score is a dedicated task whose *instruction* tells the agent to generate three
times -- `index.py get <keyword> --force` regenerates past a warm cache -- with
the verifier reading the three results rather than producing them.

It lives in `check_stability.py`, separate from `check_entry.py` because that
one must not generate and this one must. It briefly did not live anywhere:
retiring `golden.py` deleted the implementation while this file went on
describing the pass as available.
