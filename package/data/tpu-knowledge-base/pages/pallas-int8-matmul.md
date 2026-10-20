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
entry: pallas-int8-matmul
keyword: pallas-int8-matmul
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-12
source_updated: not applicable
generator: authored
---

# INT8 matmul in a Pallas kernel: int32 accumulation is mandatory

On v6e (Trillium) the MXU multiplies `int8` operands natively, at 1,836
TOPs per chip against 918 TFLOPs bf16 (`index.py get hw-tpu-v6e`). Inside a
Pallas kernel you don't upcast to bf16 first. You do have to accumulate in
int32, and to cast the result back yourself before storing it to an 8-bit
ref.

## The two rules, and the errors that enforce them

Internal experiments (not published, 2026-05-01) ran `jnp.int8` operands
through `pl.pallas_call` on Trillium:

1. **Accumulate in int32.** Mosaic lowers the dot to a native `tpu.matmul`
   on `vector<128x128xi8>` blocks. Without `preferred_element_type=jnp.int32`
   on `jnp.dot` / `jax.lax.dot`, compilation fails:

   ```
   'tpu.matmul' op Expected matmul acc to be 32-bit
   ```

2. **Cast before storing to an int8 ref.** Storing the int32 result
   directly into an `int8` output ref fails verification:

   ```
   Invalid dtype for `swap`. Ref dtype: int8. Value dtype: int32.
   ```

   That check is public JAX's ref store, which requires the stored value's
   dtype to match the ref's
   ([`jax/_src/state/primitives.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py),
   checked at `8fabfbc55101`, 2026-10-13). Apply `.astype(jnp.int8)`, or
   whatever requantization you need, to the accumulator first. The narrowing cast runs on the VPU.

With both in place the kernel matched a CPU reference exactly. Re-check
these on another generation or JAX release: the write-ups give neither the
JAX version nor any chip other than Trillium.

```python
acc = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.int32)
o_ref[...] = acc.astype(jnp.int8)  # or scale to bf16 and store that
```

## Outside Pallas

These are Mosaic rules. Plain XLA `dot_general` on int8 operands isn't
covered by them. A tpu-agent session in October 2026 (Gemma-4-31B-it,
v6e-8, TP=4 x DP=2, vLLM 0.29.0 with torch_tpu) ran the dense layers as int8
x int8 `dot_general` with `preferred_element_type=int32`, then a scale
multiply, through XLA. Weights were per-channel int8; activations were
per-token dynamic int8. It compiled and ran: 2,426.12 ± 31.60 output tok/s
against 1,831.60 ± 1.70 (+32%). But it scored 205 of 210 on the session's
known-answer probe, where bf16 scored 210. The session put the loss down to
int8 activations, so the change was reverted. Speed is the easy part on v6e;
whether activations can be int8 is an accuracy question
(`index.py get concept-quantization`).

## Evidence

Both rules come from an internal TPU world-model wiki, which is not
published. The `tpu.matmul` 32-bit message is in no public JAX source, so it
stands as observed there; the `swap` message is public JAX's (rule 2). Six of its pages say the same thing: an axiom
page, its validation experiment and the observations around them. They were
read on 2026-10-09 and are not cited by location. The peak rates are the
Cloud TPU v6e page's, through the shipped chip page.
