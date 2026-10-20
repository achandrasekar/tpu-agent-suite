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
entry: tiled-layout-padding-and-reshape
keyword: packing-dimension
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-12
source_updated: not applicable
generator: authored
---

# Tiled HBM layouts: second-minor padding and reshapes that copy

TPU arrays in HBM are stored in 2D tiles over their two minor dimensions,
`T(8,128)` for a 32-bit rank-2 array (`index.py get openxla-tiled-layout`).
Two costs follow, and neither shows up in the logical shape: padding when
the second-minor dimension isn't a multiple of the tile, and a physical
copy when a reshape changes the tiling.

## Padding in the second-minor dimension

A ragged dimension in the second-minor slot is rounded up to the tile. An
internal experiment (not published, 2026-05-02) measured bf16 arrays with
`on_device_size_in_bytes()`:

| shape | logical bytes | on device | waste |
| --- | --- | --- | --- |
| `(1, 137, 1024)` | 280,576 | 294,912 | 14,336 (5.1%): 137 padded to 144 |
| `(1, 137, 8, 128)` | 280,576 | 280,576 | 0 |

The same experiment timed an elementwise loop fusion over each shape: 713.06
µs against 678.11 µs, a 5.1% difference that matches the padding.

The fix is a packing dimension. Split the minor dimension so the last two
dims fill the tile exactly (`1024` -> `8 x 128`). That pushes the ragged
dimension out to third-minor, where it isn't tiled and so isn't padded. The
experiment doesn't say which TPU generation it ran on. The 137 -> 144 rounding
is the 8-row tile it describes.

## Reshapes that move data

A reshape that changes rank can change the tiling, and then XLA has to move
the data. The same internal source shows `x.reshape(-1)` on a `(5, 512)` f32
array:

```
entry_computation_layout={(f32[5,512]{1,0:T(8,128)})->f32[2560]{0:T(1024)}}
```

The rank-2 `T(8,128)` input becomes a rank-1 `T(1024)` output: a physical
reorganization in HBM, not a free view. This matters most when a reshape is
used to dodge a Pallas BlockSpec rule, or sits between an op and a kernel on
every step (`index.py get matmul-pallas-architectures` makes the same point
for Pallas). 8-bit arrays have their own tilings and their own copies (`index.py
get xla-8bit-layout-copies`).

## What to check

- Compare `on_device_size_in_bytes()` with the logical size for arrays whose
  second-minor dimension is ragged (sequence lengths, token counts).
- In optimized HLO, follow the layout annotations (`{1,0:T(8,128)}`) across a
  reshape. A changed tile means a copy.

## Evidence

Both measurements come from an internal TPU world-model wiki, which is not
published. Its pages were read on 2026-10-09 and are not cited by location.
The tile format itself is OpenXLA's documented one.
