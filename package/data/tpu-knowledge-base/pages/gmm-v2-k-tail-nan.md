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
entry: gmm-v2-k-tail-nan
keyword: gmm-v2-k-tail
url: https://github.com/openxla/tokamax/commit/34069eac36b6f8990fff19027b6884fe11bd4834
authority: first-party-verified
retrieved: 2026-10-01
source_updated: 2026-09-21
generator: authored
---

# tokamax `gmm_v2` NaN when K isn't a multiple of `tile_k`: fixed in 0.0.15.dev20260922

tokamax's experimental grouped matmul, `gmm_v2`, can return NaN for rows whose
inputs are finite. That happens when the contraction size K doesn't divide
evenly into the kernel's `tile_k`. The MoE layers that call it then produce NaN, even
though the same model is fine on CPU.

## Cause

The final K block reads past K. Before the fix, only `rhs` was zeroed there, so
`lhs` values past K were multiplied by zero. Finite values give zero, but NaN
or Inf give NaN. Whether those values are NaN depends on what is in memory, so
the failure can come and go between runs and inputs.

The fix, tokamax commit `34069eac36` (2026-09-21), also zeroes `lhs` past K in
that block. Its test, `test_gmm_unaligned_k_to_tpu_num_lanes_with_nans`,
fills the bytes past K with NaN and checks that the output matches the
reference.

## Which tokamax has it

| tokamax | `gmm_v2` K-tail fix |
| --- | --- |
| 0.0.14 (latest stable as of 2026-10-01) and earlier | no |
| 0.0.15.dev20260917 and earlier dev builds | no |
| 0.0.15.dev20260922 and later | yes |

So `pip install -U tokamax` installs 0.0.14 and doesn't fix it. To check
what is installed, rather than trusting the version:

```bash
python -c "import inspect; from tokamax._src.ops.experimental.gmm_v2 import gmm_v2 as m; print('fixed' if 'mask_lhs' in inspect.getsource(m) else 'has the K-tail bug')"
```

An image or environment built before 2026-09-22 has the bug unless it pinned
a later dev build.

## What to do

- **Upgrade:** `pip install --pre 'tokamax>=0.0.15.dev20260922'`. Check that
  whatever imports `gmm_v2` (vLLM's TPU MoE layers, for example) still works
  with that version.
- **If you can't upgrade:** zero-pad K up to a whole number of `tile_k`
  blocks, so nothing is read past it.

## Observed

A tpu-agent session serving GPT-OSS-20B with vLLM and torch_tpu
(September 2026) used an image built in August 2026. It got NaN from the MoE
layers, traced it layer by layer to `gmm_v2`'s final K block, and worked
around it by padding the MoE dimensions to 3072. That fits the bug above, but
the session didn't test the upstream fix, so it isn't confirmed to be the same
bug.

## Evidence

- Documented: the cause and the fix are in the commit and its test, at the URL
  above.
- Checked for this page: which releases contain the fix. Each tokamax wheel on
  PyPI was downloaded and its `gmm_v2.py` checked for the fix's `mask_lhs`
  lines. 0.0.14 and 0.0.15.dev20260917 lack them; 0.0.15.dev20260922 and
  0.0.15.dev20260929 have them. No page states this.
- Observed: the GPT-OSS-20B session, from a trajectory review. It is on no
  public page.
