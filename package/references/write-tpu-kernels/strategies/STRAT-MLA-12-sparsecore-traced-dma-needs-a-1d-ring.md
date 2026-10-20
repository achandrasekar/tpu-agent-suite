```yaml
id: STRAT-MLA-12
title: A SparseCore DMA with a traced offset and extent lowers — into a 1-D ring slot with multiple_of hints, not a 2-D ring row
hardware: tpu-v7x
unit: sparsecore
software: jax 0.10.2 / jaxlib 0.10.2 / libtpu 0.0.44.1
stage: [4]
lever: pipelining
generality: unit-wide
status: measured
source: raw/2026-09-dsanew-STRATEGY.md § V2; raw/2026-09-dsanew-final_report.md § Levers V2
recheck: compile a 2-D (nbuf, len) VMEM ring sliced as ring.at[slot, pl.ds(0, ext)] on the new toolchain; if it lowers, the 1-D form is no longer required
origin: campaign dsa-mla-256k-c16, 2026-09
```

# STRAT-MLA-12 — Traced-extent SparseCore DMAs want a flat ring

**Rule.** On SparseCore, `make_async_copy` from HBM at a traced offset with a traced extent works, but
only into a destination whose slice is provably tile-aligned: a **1-D** VMEM ring of `nbuf × slot_len`
words addressed at `pl.multiple_of(slot * slot_len, 16)`, with `pl.multiple_of(ext, 16)` and
`pl.multiple_of(off, 16)`. Slicing a **2-D** `(nbuf, slot_len)` ring as `ring.at[slot, pl.ds(0, ext)]`
is rejected at compile time.

**Mechanism.** The 2-D ring gets a `#tpu.tiled<(2,128)>` layout, so a dynamic-size slice along the tiled
row dimension cannot be proven a multiple of the tile; Mosaic refuses rather than guessing. A flat ring
with explicit 16-word multiples gives the verifier the proof it needs. Zero-extent copies must be skipped
under `pl.when(ext > 0)` on both `.start()` and `.wait()` (every started DMA must be waited — hardware-facts §8).

**Evidence.** Failing form: `INTERNAL: Slice sizes along tiled dimensions must be
aligned to tiles. Failed to verify that the size at dimension 0 is divisible by the tile dimension 2`
on `memref<2x16512xi32, #tpu.tiled<(2,128),[129,1]>>`. The flat-ring form compiled and ran bit-exact
against the static-BlockSpec kernel across row lengths 0…264192, and in production
e2e. No upstream test does a contiguous traced-size DMA on SC; the nearest is
`tests/pallas/tpu_sparsecore_pallas_test.py:920-945@ce57049` (traced-size destination slice).

**Scope.** SparseCore vector-subcore kernels, i32 data, this toolchain. The TensorCore analogue (traced
extent into a fixed buffer) is hardware-facts §8. Sharpens [[STRAT-MLA-08]], which covers traced *grid
lengths* and `pl.when` around `emit_pipeline`, not traced DMA extents.
