---
scripts:
  - scripts/tpu-model-onboarding/capture_parity.py
  - scripts/tpu-model-onboarding/compare_parity.py
  - scripts/tpu-model-onboarding/fetch_weights.py
---
<!-- disableFinding(LINE_OVER_80) -->

# Numerical Parity Check

After the coder converts model code to run on torch_tpu backend - including
replacement of any CUDA-locked code with native-PyTorch, something has to establish
that the converted code still computes the same numbers. This skill helps address
this concern.

The model runs twice on the same frozen inputs and the two sets of tensors are
compared:

-   **reference** -- CPU, the model's own built-in plain-PyTorch path, **no
    patches applied**. A CUDA kernel cannot execute on a CPU-only host, so if
    the model loads and runs here, what ran was the model's own code. That
    independence from the rewrite is the entire reason this comparison means
    anything.
-   **candidate** -- the accelerator, loading the same patch module the
    inference server loads.

Both runs are **teacher-forced**: one forward pass over a fixed token
sequence, never free generation. Letting each side generate its own
continuation makes one flipped token change every token after it, which says
nothing about whether the rewrite is correct.

## When to run it

-   **MANDATORY** when `preflight-check` reported `ONBOARDING_REQUIRED`. The
    unwiring playbooks rewrite attention, scans and norms; nothing else in the
    inference workflow checks that those rewrites preserved the math.
-   Optional otherwise, when the user asks for it.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`, `<parity_work> = <work>/parity`.
Return `invalid_input` if `run_id` is missing. Write only to `<parity_work>`, never to CWD or package files.

`<patches_path>` below is the patches module tpu-unwiring returned.

### When tpu-unwiring already ran

tpu-unwiring builds the probe and the candidate capture as its own proof that
the patched model runs, under `<work>/parity/<model_slug>/`, and returns their
paths as `probe_path` and `candidate_capture`. Use that directory as
`<parity_work>`, skip Step 1 and the candidate half of Step 2, and run only the
reference capture and Step 3. The verdict is yours to read: tpu-unwiring never
compares.

## Step 1: Build the probe

The probe freezes the inputs both runs receive. Build it **once** and give the
same pair of files to both runs.

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/capture_parity.py}} \
  --output probe.json:<parity_work>/probe.json \
  --output probe.safetensors:<parity_work>/probe.safetensors \
  -- --make_probe --model {model_id} --out probe.json \
     --prompt "{a representative prompt}" \
     --modules "{comma,separated,module,names}"
```

`--modules` takes the qualified names of the modules the patches replaced,
which the generated patch module exposes as `PATCHED_MODULES`. Hooking them is
what makes a failure localizable: without it you learn only that something is
wrong, not which layer. Do not skip it.

For a model whose tokenizer will not load, pass `--input_ids "1,2,3"` instead
of `--prompt`.

The probe writes **two files** and they must stay together: the JSON names the
inputs, the safetensors file holds them. The digest covers both, so a run
against different inputs is refused rather than silently compared.

## Step 2: Capture both runs

Reference, on CPU, with/without patches:

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/capture_parity.py}} \
  --input <parity_work>/probe.json:probe.json \
  --input <parity_work>/probe.safetensors:probe.safetensors \
  [--input <patches_path>:model_patches.py] \
  --output reference.json:<parity_work>/reference.json \
  --output reference.safetensors:<parity_work>/reference.safetensors \
  -- --role reference --probe probe.json --out reference --device cpu [--patches model_patches.py]
```

**If the reference run fails because the model cannot load without the
patches**, that model has no built-in plain-PyTorch path. Pass `--patches` to
the reference too. The comparison then reports `UNVERIFIED` rather than
`PASS`, which is correct: both sides are running the same rewritten code, so
agreement cannot tell you the rewrite is faithful. Report that outcome to the
user in those words. Never present it as a pass.

Candidate, on the accelerator, with the patches:

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/capture_parity.py}} \
  --input <parity_work>/probe.json:probe.json \
  --input <parity_work>/probe.safetensors:probe.safetensors \
  --input <patches_path>:model_patches.py \
  --output candidate.json:<parity_work>/candidate.json \
  --output candidate.safetensors:<parity_work>/candidate.safetensors \
  -- --role candidate --probe probe.json --out candidate --patches model_patches.py
```

Both sides load the model in the dtype the probe names, so precision is not
one of the things the comparison is measuring.

**When `{model_id}` is a `gs://` checkpoint folder**, add
`--input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py`
to `--make_probe` and to both captures. Each job then copies the checkpoint
to its own disk instead of downloading from the hub: the probe copies only the
config and tokenizer files, each capture the whole model, and the probe keeps
naming the `gs://` folder so both captures read the same one. The job reads
the bucket with its own credentials; a `WEIGHTS FAILED: no permission to read`
line means it has none for that bucket, and is the user's to fix, not a
reason to fall back to the hub. The CPU reference holds the whole model in
memory, so give it a target with RAM of at least 1.3 times the checkpoint's
size.

## Step 3: Compare

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/compare_parity.py}} \
  --input <parity_work>/reference.json:reference.json \
  --input <parity_work>/reference.safetensors:reference.safetensors \
  --input <parity_work>/candidate.json:candidate.json \
  --input <parity_work>/candidate.safetensors:candidate.safetensors \
  --output parity_report.json:<parity_work>/parity_report.json \
  -- --reference reference --candidate candidate --out parity_report.json
```

Exit code 0 is `PASS`, 1 is `FAIL` or `UNVERIFIED`, 2 means the two captures
could not be meaningfully compared at all (different inputs, different model,
different hooked modules).

## Reading the result

```
probe p0
  logits       max_abs 2.938   max_rel 3.09%   rms 0.823
  top-1 agree  13/13
  backbone.layers.0.mixer   max_abs 0.0625   max_rel 0.70%
  backbone.layers.11.mixer  max_abs 0.875    max_rel 10.85%
```

-   **`max_rel`** is the difference as a fraction of that tensor's own scale.
    Grade on this, not `max_abs`: tensor magnitudes vary by more than 40x
    between layers of one model, so no absolute number means the same thing
    twice.
-   **`top-1 agree`** counts positions where both runs pick the same token. A
    disagreement is only reported as real when the reference's own margin
    between its best and second-best logit was too wide for rounding to have
    bridged it; the margin is printed so you can judge.
-   **The per-module rows are the point.** A measured case: a deliberate 10%
    error in one layer left every token matching and the final logits
    *slightly closer* than the clean run, while the broken layer read 10.85%
    against 0.70% for its neighbour. End-to-end output alone would have
    reported that model healthy.

When the verdict is `FAIL`, name the module with the highest `max_rel` to the
user and point at the playbook that rewrote it. That module is where to look.

## What this cannot tell you

State these limits plainly rather than overstating the result:

-   **There is a floor of roughly 3%.** Measured on mamba-130m, CPU bf16
    against TPU v6e bf16 with identical code on both sides, device noise alone
    reached 3.09%. A rewrite wrong by less than that is indistinguishable from
    it. Widening the probe set and hooking more modules helps; lowering the
    threshold only produces false alarms.
-   **A pass covers the probes you ran.** It is evidence the rewrite is
    faithful on those inputs, not a proof over all inputs.
-   **`UNVERIFIED` is not a weak pass.** It means the check could not be
    performed.
