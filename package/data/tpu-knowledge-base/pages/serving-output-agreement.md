<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: serving-output-agreement
keyword: teacher-forced-agreement
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-01
source_updated: not applicable
generator: authored
---

# Did a change alter a server's output? Score teacher-forced, not greedy

Use this when a changed LLM server should give the same answers as a baseline:
the same server with a new kernel, say, or a TPU server against CPU. Score the
same fixed token sequences on both, rather than lining up what each generates.

## Why greedy generation misleads

In greedy generation each server picks its next token from its own previous
tokens. One near-tie that rounds the other way early in a sequence changes the
rest of it. After that the texts differ, and every later token counts as a
mismatch, even if the change is correct.

Measured on one change (see Observed): greedy generations agreed on 80.7%
of tokens, and only 13 of 24 sequences were identical. Teacher-forced over the
same model, the top token agreed at 99.2% of positions.

## Method

1. **Fix the sequences.** Take prompts plus continuations, for example the
   baseline's own greedy continuations, and freeze them as token ids. Cover the
   lengths you serve, including long context.
2. **Score them on both servers in one forward pass.** For each position,
   record the top token and the log-probability of the baseline's top token.
   vLLM returns these per prompt position as `prompt_logprobs`.
3. **Run the baseline against itself.** This is the control. If it isn't
   exact, the server isn't deterministic, and differences smaller than its
   self-difference say nothing about the change.
4. **Report:**
   - top-1 agreement: positions where both pick the same token, out of all;
   - mean and max |Δlogprob| for the baseline's top token;
   - for each disagreement, the baseline's margin between its first and
     second token.

## Reading the result

- **Pass:** every disagreement is a near-tie: the baseline's margin between its
  first and second token is no larger than the max |Δlogprob| the change
  produced at positions where the two servers agree, so a perturbation of
  that size can flip it. Judge |Δlogprob| against the control, not against a
  fixed bf16 figure: with an exact control, any nonzero |Δlogprob| is the
  change's own effect.
- **Fail:** any disagreement where the baseline's margin is wider than that.
  The measured perturbation can't explain it, so look for a bug.
- Report greedy agreement as extra information only, never as the verdict.

The package's tpu-model-onboarding numerical-parity step reads model rewrites on the
same principle: teacher-forced inputs, and a disagreement counts only when the
reference's margin was too wide for rounding. This page makes "too wide"
concrete, against the perturbation the change measurably produced.

## Observed

Two tpu-agent sessions on TPU7x in September 2026, with vLLM and
torch_tpu. These are their results, not published figures:

| comparison | top-1 agreement | \|Δlogprob\| | control |
| --- | --- | --- | --- |
| Qwen3-32B, optimized `batched_rpa` decode kernel against baseline | 508 / 512 (99.2%) | mean 0.007, max 0.25 | baseline against itself: 512 / 512, no difference |
| Gemma-4-31B-it, TPU server against the Transformers model on CPU, bf16 | 175 / 179 | not recorded | none, so it doesn't meet step 3 |

The Gemma session reported each of its four disagreements as a near-tie. The
Qwen session recorded the agreement and |Δlogprob| but not its four
disagreements' margins, so whether it passes the rule above is not settled.
Its control was exact, so its nonzero |Δlogprob| is the kernel's own effect:
the change isn't bit-exact. Neither session wrote down a pass/fail rule
before it measured, which is why this page exists.

## What it doesn't cover

- Only the sequences you scored. Long-context and multimodal inputs need their
  own probes; the Gemma session sent no image.
- Speed. Pair it with a benchmark on the same server.

## Evidence

This entry has no source URL. The method follows from how greedy decoding
works, and is the rule the package's tpu-model-onboarding numerical-parity step
uses. The numbers come from two sessions and are on no public page; re-verify
by running the method.
