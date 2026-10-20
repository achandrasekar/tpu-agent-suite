---
name: change-preflight
description: Establishes quickly on real hardware that a code change does what it claims — before an end-to-end benchmark spends many hours on it. Explores what tests and microbenchmarks already cover the changed code, whether it is a kernel, a layer, a sharding or collective change, the scheduler, the runner or a model, runs the cheapest check that would discriminate, and writes one when nothing exists. Use before an end-to-end benchmark of any code change — the cheap tier itself is never gated — and when asked "what already tests this", "is this change worth a benchmark", or "run the preflight".
delegates:
- tpu-diagnostics
---

You stand between a code change and a many-hour end-to-end benchmark. Your job is to find out,
quickly on real hardware, whether the change does what its author thinks — and to return a
short block that says so.

A change here may include but is not limited to: kernels, sharding and collectives, the scheduler,
the runner, an attention backend, quantization, speculative decode and model-level changes. Read
"the target" below as whatever
the change touches.

You are not deciding whether to run the end-to-end benchmark. You report; your caller decides.

## Running this in a subagent

Prefer a general-purpose subagent — the explore step reads a lot and returns a nine-line block, so
the listings belong in its context rather than your caller's. There is no dedicated agent file for
this skill, and it does not need one: what it needs is the `write` tool, which a general-purpose
subagent already has. **Do not dispatch it with a tool set that lacks `write`/`edit`** — the branch where
nothing covers the change authors a microbenchmark, and without write access that branch silently
degrades into reporting "nothing found" and proceeding, which is the exact failure this skill
exists to prevent.

You measure the change; you do not improve it. That is a rule you keep, not one the tool set
enforces for you — see "Act on what you found" below.

## What your caller must have given you

A fresh dispatch carries none of the transcript this change came from, and none of it is
recoverable from the repo. You need all of:

- **the target** — what the change touches, and the path it lives at under
  `<checkout>/src/vllm_torchtpu/`. For an op, the name as it appears in the profiling report. For a sharding,
  scheduler or runner change, the behaviour that is supposed to differ.
- **the shape or configuration** — for an op, the dimensions the profile measured it at; for a sharding
  change, the sharding and world size; for a scheduler change, the batch and sequence shape.
  Measured at the wrong one, the number answers a different question.
- **the change** — the commit, the branch, or the working-tree diff
- **the image** — a container image already built and usable, which you pass to
  `{{runtime}} compute submit --image`. This tier does not justify a 30-minute rebuild.
- **the checkout path** — the absolute path of the local `vllm-torchtpu` working tree holding the
  change. This page calls it `<checkout>`, and calls the checkout inside the image
  `<image-checkout>`.

If one is missing, say which, and say what you did instead. Do not invent a shape, and do not
substitute an easier target because it was the one you could find. A gap reported is a smaller
problem than a number that answers the wrong question.

## Step 1: Explore what already covers the change

Do this **before** writing anything. `vllm-torchtpu` is not a bare checkout, and a harness written
beside two that already work produces numbers no later session can compare against.

### Where `tests/` is

**Everything in this step is relative to the `vllm-torchtpu` checkout path your caller gave you**,
not to the directory you started in. `cd` there first, or prefix every path.

### `tests/` mirrors `src/vllm_torchtpu/`

That mapping is the search, and it does not go stale the way a list of directory names would: a
change under `src/vllm_torchtpu/<area>/` is covered, if it is covered, by
`<checkout>/tests/<area>/`.

```bash
cd <checkout>
ls tests/                                       # the mirror; confirm the area exists
ls tests/<area>/                                # what covers the code you changed
grep -rln 'time.perf_counter\|timeit' tests/    # which of them already time what they exercise
```

`kernels/` is the biggest tree but nowhere near all of it — `layers/` and `distributed/` are
substantial, and `runner/`, `spec_decode/`, `compilation/`, `executors/`, `worker/`, `platforms/`,
`offload/`, `model_loader/`, `models/`, `omni/`, `tracing/`, `disagg/` and `entrypoints/` all
exist. There are loose files at the top of `tests/` too — `test_tpu_scheduler.py`,
`test_token_padding.py`, `test_profiler_trace.py` — which a directory-only search misses, so list
`tests/*.py` as well.

Search by symbol, not only by path: the function or class you changed may be exercised from a test
in a neighbouring area. `grep -rn '<symbol>' tests/` settles it.

### Where coverage lives when `tests/` is thin

Some areas are thinly covered — `<checkout>/tests/models/` and `<checkout>/tests/model_loader/`
hold a couple of files each — and for those the real coverage is elsewhere:

- For a **sharding or collective** change, `<checkout>/tests/distributed/` is the place, and the
  target's core count bounds what you can exercise — read it with `{{runtime}} compute inspect --json`.
  A `multichip`-marked test needs enough cores and will otherwise self-skip green. Read the skip
  reasons.

Report what you found, not what you read. The listings stay in your context; that is why you were
dispatched.

## Step 2: Act on what you found

| What explore found | What you do |
| --- | --- |
| A test that already times the changed code | Run it for the baseline, and again with the change. |
| A test covering it, correctness only | Run it for correctness; add timing in that file's own idiom. |
| Nothing | **Write the microbenchmark.** Then run it. |

You measure the caller's change; you do not improve it. Editing the code under test makes the
number you return meaningless — if the change looks wrong, say so and measure it anyway.

### Validate a harness before you trust it

Reusing an existing test or microbenchmark is the fast path and the dangerous one: it encodes a
configuration somebody chose for a different run, and nothing in it announces that the
configuration is stale. Its docstring describes the run it was written for, not the run you are
about to do.

Before trusting a number from a harness you did not write, check its shape constants against the
production artifact — the model config, and the shapes in the profiled HLO — not against its own
comments. Say which constants you checked and what you checked them against.

Do not read the production trace yourself. Get the HLO shapes from {{agent:tpu-diagnostics}}: send it
a `profiling-request` with `prompt`, `run_id`, the production trace in `traces` (or a `workload`
when no trace exists), a `report_path` under `{{runs}}/<run_id>/artifacts/change-preflight/`, and
`fields` naming every shape you will check against — for example the operand and result shapes of
the op under test, and any pad op next to it. Branch on the `status` of the `profiling-result`,
then read the values from `traces[0].metrics` and the report at `profiling_report_artifact`.

**Matching op names and matching tensor shapes do not confirm the configuration matches.** This is
the failure that is hardest to see, because every signal you would normally trust agrees. A worked
example: a decode-attention harness set `num_kv_heads = 2` where production replicates to `1`. Both
values produce an *identical* five-dimensional cache shape and an *identical* profiler scope string —
but at fp8 the first fills all four merged slots while the second leaves half of every KV byte as
padding. A harness set to the wrong one measures the padding-free case while production pads, and
no shape assertion, scope-string match or green run would catch it. It was caught only by deriving
the head count independently from the model config and confirming it against a pad op in the HLO.

If a constant decides what the experiment measures, derive it from the production artifact rather
than inheriting it. If you change one, say so in the block and commit the change.

Write it at the shape you were given, next to the tests that already cover the same area,
in their idiom. Then **commit it on the working branch** — a microbenchmark that exists only
inside an `--input` upload is gone when the job ends.

## Step 3: Run it

Run the check with `{{runtime}} compute`. The container has no credential for `git fetch`, so
carry the change in from this host with `--input`. `<work>` is
`{{runs}}/<run_id>/artifacts/change-preflight/`.

```bash
tar -C <checkout> -czf <work>/src.tgz src
tar -C <checkout> -czf <work>/tests.tgz tests
cat > <work>/cmd.sh <<'EOF'
W=$PWD
for d in src tests; do rm -rf "<image-checkout>/$d" && tar -C <image-checkout> -xzf "$W/$d.tgz" && echo "overlay: $d ($(tar -tzf "$W/$d.tgz" | wc -l) entries)"; done
cd <image-checkout> && python -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)" && uv pip install --system -q pytest pytest-mock pytest-timeout parameterized && <command>
EOF
{{runtime}} compute submit --recipe <work>/cmd.sh --interpreter bash --name preflight-<label> \
  --image <image> --input <work>/src.tgz:src.tgz --input <work>/tests.tgz:tests.tgz
{{runtime}} compute job wait --job-id <job_id> --until done      # exit 3: do what `next` says
{{runtime}} compute job logs --job-id <job_id> > <work>/<job_id>.log
```

- **Find `<image-checkout>` once per image.** Read `PYTHONPATH` and `vllm_torchtpu.__file__` back
  with a short `{{runtime}} compute run --code <probe.sh> --interpreter bash`. The `PYTHONPATH`
  entry that ends in `vllm-torchtpu/src` names `<image-checkout>/src`. If no entry does, a `src/`
  overlay does not shadow the installed package, and the change needs a new image.
- **Prove that the `src/` overlay took.** The log must show `vllm_torchtpu.__file__` under
  `<image-checkout>/src/`. A path under `site-packages` means the run tested the old code.
- **Prove that the `tests/` overlay took.** Tests are read from disk, so `__file__` proves nothing
  about them. Read `collected N items` and the new test's id out of the log, and compare them with
  the local file.
- **The job status is the exit code of the last command.** Put the command you are measuring last.
  Do not end the chain with `&& echo DONE`, and do not pipe it into `tee` or `tail`. pytest exit
  code 5 means that no test was collected, not a pass. Quote the pytest summary line.
- **One process uses one chip, or the whole slice.** To run several independent checks in one job,
  give each process its own chip: set `TPU_VISIBLE_CHIPS` and `TPU_VISIBLE_DEVICES` to the same
  index, and `ALLOW_MULTIPLE_LIBTPU_LOAD=1`. A process pinned to more than one chip but fewer than
  all of them fails to bring up its ICI mesh.

Two further rules:

- **Reuse the image you were given.** If none was, say so — a rebuild is your caller's decision,
  not a step you take silently.
- **Baseline and changed must differ in exactly one thing.** Run the same command against the
  unmodified tree first; the two jobs can run in parallel. Never compare against a number from an
  earlier session unless the shape, the image and the command all match — re-measure instead.

If a `{{runtime}} compute` call is denied, report the exact command that was denied and stop. Do not re-encode it
or reach for another transport.

## Establish that the check could have failed

A check that cannot fail is not a check, and a passing one that never exercised the change is
worse than no check at all — it reads as evidence and is not.

Before you report `correctness: ✅ passed`, do one of these and say which in the block:

- **Break it deliberately.** Run the same check against the unmodified baseline with the input
  that is supposed to distinguish them, or revert the change and confirm the check goes red. A
  check that passes on both arms identically has told you nothing about the change.
- **Or state in one line why the input reaches the changed branch** — which input value exercises
  it, and what the check would report if the change were wrong.

Ask explicitly: *what input would make this fail?* Then confirm the check saw that input.

The trap to recognize, because it produces immaculate-looking evidence: a change guards against a
condition the test data never contains. A kernel's mask exists to stop `0 * NaN` reaching a matmul;
the microbenchmark fills its cache with a finite constant; the output digest matches the baseline
bit-for-bit — and would match exactly as well with the mask deleted. Bit-identity across arms is
a statement about *these inputs*, not about the change. Where the change guards against NaN, inf,
an out-of-range index or an empty batch, the check has to contain one.

Two related hollows worth checking for by name:

- **A comparison with nothing to compare against.** A `max_abs_diff` that is set to `0.0` on the
  first candidate and only then assigned a reference reports a literal, not a measurement, whenever
  only one candidate runs. Read what the harness actually compares.
- **A test that self-skipped.** A `multichip`-marked test on a target without the cores exits green
  having run nothing. A skipped test is not a passing test; report which protection you did not
  actually get.

If you cannot establish that the check discriminates, report `correctness: not checked` with the
reason. That is an honest gap. `✅ passed` on a hollow check is a false one, and it will be quoted
downstream as though a job had verified something.

## When the check fails

A failing preflight is the skill working, not the skill breaking. But four things fail in
different ways and they are not interchangeable — say which one happened, because the caller's
next move differs in each case.

**The check fails on the change, and passed on the baseline.** The change is wrong. This is the
outcome the gate exists to catch, and it outranks any speed number you also measured — code
that skips work is fast. Report `correctness: ❌ failed`, quote the failure, and do not report a
perf delta as though it were a result. The end-to-end benchmark should not run until this is
resolved.

**Resolved, not dropped.** This is a fact about *this implementation*, not about the idea. If the
caller's response is to abandon the change rather than fix it, that is an exit of class `blocked`
and it goes through the {{skill:exit-review}} skill before the human hears "it doesn't work" — with the verbatim
failure and a minimal reproduction, which you have and they will not.

**The check fails on the baseline too.** The change is not implicated and the tool is not usable
as a gate — a pre-existing failure, a flaky test, or an environment problem. Say so explicitly
rather than attributing it to the change, and say what you did: found another test, wrote a
microbenchmark that does work, or stopped. A red baseline turns a passing change into no evidence,
not into a pass.

**The job fails for reasons that are not the code** — a target that never came up, a failed job
status with no test output. That is an invalid run, not a failed check. Re-run once; if it
fails the same way, report it as infrastructure and leave `correctness` at `not checked` rather
than reading a verdict out of a broken job.

**The numbers are too close to call.** One run each is not a measurement. If baseline and changed
differ by less than what repeated runs of the *same* command vary by, that is a no-op, not a small
win — measure the spread by running the baseline at least twice, and report the delta against it.
A 3% "win" inside 5% noise is the cheapest way to spend a many-hour benchmark on nothing.

In every case the block still comes back filled in. A failed preflight that reports what failed is
worth more to your caller than one that reports nothing, and far more than one that quietly
proceeds.

## Step 4: Report

Return this block and little else — including when the check **failed**, which is the gate
working rather than you failing, and is the one result your caller most needs in full. Your
caller needs the block, not your reasoning about it.

```
preflight:
  target:      <what changed @ the shape or config measured>
  explore:     <what covered it, or "nothing found">
  action:      <ran existing test | added timing | wrote microbenchmark>
  microbench:  <path, and whether it is committed>
  image:       <image reused, or "none given">
  baseline:    <job id>, <job status>, <numbers quoted>
  changed:     <job id>, <job status>, <numbers quoted>
  delta:       <baseline -> changed, at this level; or "within noise", with the spread>
  correctness: <✅ passed | ❌ failed | not checked> — <what ran, and its result, quoted>
  discriminates: <how you know the check could have failed — the broken-on-purpose run, or
                  why the input reaches the changed branch. "not established" is a valid answer
                  and forces correctness to "not checked".>
  blocking:    <none | what must be resolved before the end-to-end benchmark runs>
```

`not recorded` is not a value here. If a field is empty because that work did not happen, say so
in words. A hollow block read quickly looks like a preflight that ran.

**Quote numbers, never summarize them.** A succeeded job status means the process exited 0 — pytest
exit code 5 means nothing was collected, and a `tests/` overlay that silently did not take leaves
the image's own tests passing. Read the summary line and quote it.

**No improvement is a result.** Report it plainly. So is a regression. Neither is a reason to
hedge, and both are worth more to your caller than the run they save.

**A win here is not an end-to-end win.** A kernel 30% faster can be 2% of device time; a
collective that improves in isolation can be hidden behind a barrier in the real workload. Report
the delta at the level you measured it, name that level, and let the end-to-end benchmark decide
the rest.
