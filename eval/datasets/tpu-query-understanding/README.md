# tpu-query-understanding

Evals for tpu-agent's query-understanding reference
(`references/tpu-agent/query-understanding.md`): the planning step
tpu-agent follows before it delegates a new onboarding request to
tpu-model-onboarding.
Each task scores **the first turn only** -- a passing run plans, asks the user
its questions, and stops. What happens after the user answers is onboarding's
to be evaluated, not this dataset's.

| Task | Prompt | A pass means |
| :--- | :--- | :--- |
| `gpt-oss-20b` | The prompt that motivated query understanding, verbatim: "gpt oss 20b", no `torch_xla` | Reference followed, tpu-knowledge-base asked what constrains the model, `openai/gpt-oss-20b` offered as a linked default to confirm, questions with defaults, nothing delegated to tpu-model-onboarding, no benchmark added to the plan |
| `ambiguous-model` | "llama 3 8b" | A "which repository" question with a linked `meta-llama` default and what else the name could mean, nothing delegated to tpu-model-onboarding |
| `codebase-not-supported` | A GitHub repository instead of a Hub model | Says onboarding from a codebase is not supported yet, asks whether the model is on the Hub, nothing delegated to tpu-model-onboarding |

Every task is scored by RewardKit from two inputs:

- `tests/checks/checks.py` -- deterministic checks over the trajectory's tool
  calls: whether the reference was read, and whether any specialist but
  tpu-knowledge-base was started before the user answered. They match Claude Code,
  Antigravity and Codex tool names alike. The part above the `TASK CRITERIA`
  marker is shared and must stay identical across tasks.
- `tests/expectations/judge.toml` -- an LLM judge over the trajectory for what
  only prose shows: the right model and link, no scope added, questions with
  defaults.

A reference runs in tpu-agent's context, so everything it does is in
tpu-agent's own trajectory; the checks need nothing else.

## Running

```bash
./eval/run.py --config claude-code/agent-suite --dataset tpu-query-understanding
```

The planning turn touches no cluster and waits on no chip. To compare against
the package without query understanding, run the same dataset on a build
without it and put the two jobs side by side in Harbor.

## Reference runs

The checks were calibrated against the golden trajectories in Harbor job
[`tpu-query-understanding-goldens`](https://hub.harborframework.com/jobs/9a9e799b-301a-4865-b14b-6eb7f85a67ec),
shared with `google-pytorch`. All are the gpt-oss-20b prompt on a tpu7x
`2x2x1` GKE setup with Claude Code; tool results are dropped.

| Trial | What it is | Checks |
| :--- | :--- | :--- |
| `gpt-oss-20b__baseline` | Recorded, package without query understanding. Delegated about 30 s in as `mode: benchmark`; 145 min, $22.88 | 0 |
| `gpt-oss-20b__subagent-turn1` | Constructed, from the earlier subagent design | -- (superseded; that design is gone) |
| `gpt-oss-20b__subagent-turn2` | Constructed, from the earlier subagent design | -- (superseded) |

The checks were rewritten for the skill; the baseline scores 0 on
them.

### Live runs

2026-09-30, Claude Code with Claude Opus 5 on Vertex, whole package, tpu7x
`2x2x1` on `bodaborg-tpu7x-nap`. The control is the package at `main`
(`b75408f0`), which has no query understanding.

| Task | With the skill (`0e44439b`) | Without it (`main`) |
| :--- | :--- | :--- |
| `ambiguous-model` | 1.0 -- checks 1.0, judge 1.0 | 0.0 -- delegated to tpu-model-onboarding |
| `codebase-not-supported` | 1.0 -- checks 1.0, judge 1.0 | 0.17 -- sent the GitHub repository to tpu-model-onboarding |
| `gpt-oss-20b` | 0.875 -- checks 1.0, judge 0.75 (the suggested id was not linked) | 0.29 -- delegated to tpu-model-onboarding, then asked tpu-knowledge-base |

With the skill a trial took about 9 min, most of it tpu-knowledge-base answering
with citations ($1.66 for `gpt-oss-20b`); without it every trial ran to the
20 min timeout inside onboarding. In `--print` mode the skill puts its
questions in the final reply rather than the ask tool, since no one can
answer.

The Harbor jobs --
[with the skill](https://hub.harborframework.com/jobs/7de40c37-7773-42ab-b715-23993b21d622),
[without it](https://hub.harborframework.com/jobs/461f001d-b43e-43a6-a66b-b96bf138fd2d)
-- show 0 for every trial but the control's `gpt-oss-20b`: the judge had no
Vertex project, and a judge that cannot run scores the task 0. The scores
above re-ran each trial's verifier on its recorded trajectory with the project
now set in `task.toml`; a live trial with that fix scored
`codebase-not-supported` 1.0.
