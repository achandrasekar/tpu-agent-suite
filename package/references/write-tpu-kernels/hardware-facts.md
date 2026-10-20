```yaml
id: hardware-facts
title: Chip constants and compiler hard limits
hardware: tpu-v7x
unit: any
software: jax 0.10.2 / jaxlib 0.10.2 / libtpu 0.0.44.1, python 3.12.14
stage: [1]
lever: any
generality: narrow
status: measured
```

# Hardware facts & limits — TPU v7x (Ironwood)

Each entry states the **fact** first, then the evidence that establishes it. Read whole, at step 1 —
see `PROTOCOL.md`. Most of what is here **silently invalidates a design** rather than answering a
question: nobody asks whether `get_tpu_info()` reports per chip or per core, they assume, and by the
time the assumption is tested the kernel is written.

**Environment:** TPU v7x, `jax` 0.10.2 / `jaxlib` 0.10.2 / `libtpu` 0.0.44.1, Python 3.12.14 —
everything below unless a line says otherwise.

Tags: an untagged figure was measured on tpu7x; `[source: path@sha]` was read out of pinned
source; `[unverified]` believed but not established. Nothing here is a throughput, so nothing goes
stale between runs the way a timing does — except the handful of *achieved* bandwidths, which are
labelled as achieved and are timings wearing a ceiling's clothes.

---

## 1. Device denominator & scaling

**`pltpu.get_tpu_info()` describes one TensorCore — half a physical chip, which is one chiplet.**
It reports `num_cores=1`, `is_split_chip=True`, `is_megacore=False`: the two physical TensorCores are
addressed as separate devices. A Pallas kernel runs on one, so **these are already the right
denominator for capacity and FLOPs — use them as reported.** For *bandwidth*, see §2: the reported
`3.70e12` is a rounded theoretical peak and no transfer reaches it.


**The constructor divides only some fields, and that is the part that catches people.**
`ChipVersion.num_physical_tensor_cores_per_chip` is **2** for v2, v3, v4, v5p, 7, 7x and 8i, and
**1** for v4i, v5e and v6e [source: `jax/_src/pallas/mosaic/tpu_info.py@jax 0.10.2`]. For 7x:

| field | as reported | in source |
| --- | --- | --- |
| `hbm_capacity_bytes` | 103,000,000,000 (95.93 GiB) | `206_000_000_000 // 2` |
| `mem_bw_bytes_per_second` | 3.70e12 | `7.40e12 // 2` |
| `bf16_ops_per_second` | 1.155e15 | `2.31e15 // 2` |
| `fp8_ops_per_second` | 2.30e15 | `4.60e15 // 2` |
| `vmem_capacity_bytes` | 67,108,864 (64 MiB) | **not divided** — `# 64 MiB per core` |
| `smem_capacity_bytes` | 1,048,576 (1 MiB) | **not divided** — `# 1 MiB per core` |
| `cmem_capacity_bytes` | 0 | literal `0`; 7x has no CMEM |
| `int8_ops_per_second` / `int4_ops_per_second` | 0 | literal `0, # Not Available` |
| `sparse_core.vmem_capacity_bytes` | 524,288 | **not divided** — 512 KiB *per vector subcore* |

The probe reports a longer divided-field list because it scans every generation's constructor;
`cmem` is divided on v4 and `int8`/`int4` on v5p. **The four divided rows above are the ones that
apply to this chip.** Two ways to get it wrong, failing in opposite directions: comparing a divided
field against a chip-level datasheet (off by 2×), and multiplying an undivided field back up
(**there is no 128 MiB of VMEM**; a budget written against one compiles right up until it doesn't fit).

**The structural constants the same object carries**, none of them divided, and several of them
load-bearing elsewhere on this page [source: `tpu_info.py@jax 0.10.2`, read back by executing
`get_tpu_info_for_chip(ChipVersion.TPU_7X, 1)` under the pinned wheel]:

| field | value | used by |
| --- | ---: | --- |
| `num_lanes` / `num_sublanes` | 128 / 8 | the relayout axes, §5 |
| `num_mxus` | **2** | why `_0` alone is half the MXU traffic, §9 |
| `mxu_column_size` | 256 | MXU geometry; gen ≥ 6 is 256, below it 128 |
| `num_accumulators` | 128 | — |
| `sparse_core.num_cores` × `num_subcores` | **2 × 16 = 32** | the lockstep wave, §10 |
| `sparse_core.num_lanes` | 16 | — |
| `sparse_core.dma_granule_size_bytes` | **32** | the granule the gather width story rests on, §10 |

Two of those retire an inference. **The 32-subcore wave and the 32-byte DMA granule are declared in
source**, not only inferred from a bandwidth curve and a timing ladder — so where §10 says "inferred",
the count itself is now `[source:]` and only the *cost model* around it is measured.

---

## 2. Memory system & bandwidth ceilings

**TPU v7x is two chiplets in one package; a chiplet is 1 TensorCore + 2 SparseCores + its own HBM
stack. There is no chip-wide memory pool.** The fields §1 divides are divided because the resource
genuinely *is* per-chiplet — which is also why `is_split_chip=True`. A Pallas kernel's memory
denominator is a chiplet's, and it has no path to the other one.

Five ceilings inside one chiplet
[unverified: vendor figures supplied 2026-09-16; no probe and no pinned source in this tree — see §11]:

| ceiling | per chiplet | per chip (= 2×) | of theoretical |
| --- | ---: | ---: | --- |
| **theoretical peak** | **3686 GB/s** | 7372 GB/s | 100% |
| **practical peak** — protocol + Core Memory Network overhead | **3207 GB/s** | 6414 GB/s | 87.0% |
| one SparseCore, **unidirectional** (CMN command limit) | **1700 GB/s** | — | 46.1% |
| one SparseCore, **read + write concurrently** | **2626 GB/s** | — | 71.2% |
| TensorCore **and** SparseCore reading concurrently | **< 3207 GB/s** | — | unquantified |

**Three figures for the same peak, none of them equal, all within 0.4% — know which one you used.**
The per-chip column is arithmetic, `2 ×` the chiplet figure, and nothing measured two chiplets
together. The vendor quoted the chip peak as **7380** GB/s, which is not 2 × 3686 (= 7372) — their
own pair disagrees by 0.1%. jax reports **3.70e12 B/s = 3700 GB/s** per chiplet, rounded to three
significant figures from a chip constant of `7.40e12`, and that is **0.37% above 3686**. The figure
that closes against the physics is **3686.4 GB/s** = 7.2 Gbps/pin × 4096 bits ÷ 8, so 3686 is the
precise one and 3700 is the rounded-up one. **[unverified]** — the 7.2 Gbps/pin derivation is not
pinned to a source in this tree either.

**Build the roofline on 3207, not on 3686 and not on the reported 3.70e12.** A design whose modelled
bandwidth lands between 3207 and 3686 has not got thin headroom — it is already over the ceiling,
with the same standing as a roofline whose per-call time lands under the dispatch intercept (§7 — a
range, and one you must pick the right end of). The 0.37% spread
between 3686 and 3700 is far below the 13% gap to the practical peak, so it changes no conclusion —
but quote one and name it, rather than mixing them across a comparison.

**The SparseCore's ceiling is the unit's, not the chiplet's.** One SparseCore cannot exceed 1700 GB/s
in one direction *however wide its access*, because the limit is a CMN **command** rate, not a data
rate. Two cores would be 3400, above the chiplet's practical peak — so on an SC-only workload the
binding constraint switches from unit to memory system somewhere between one core and two.

**A read-modify-write stream gets 2626, not 2 × 1700.** Budget the *sum* of both streams against it;
that is 1.55× the unidirectional figure, and it is what a scatter-add is priced against.

**Mixing TC and SC traffic costs more than either alone** — 512-byte against 128-byte accesses on the
same stack. **How much less is not quantified**; an overlap of a TC matmul with an SC gather is a
probe, not a subtraction.

**Capacity: the vendor's 96 GiB per chiplet and `hbm_capacity_bytes = 103,000,000,000` are the same
quantity to 0.08%**, not exactly equal — 96 GiB is 103,079,215,104 B, and jax's 103e9 B is
95.926 GiB. Decimal on one side, binary on the other, and jax's constant is the rounded one. Check
the unit before reconciling a 7% gap in any capacity on this page. **For a capacity roofline use
103e9 B**, the number the runtime will actually report.

---

## 3. MXU native dtypes & precision

**The MXU natively supports exactly 12 dtype pairs**, swept over all 9×9: `(f32|bf16) × (f32|bf16)`, and
`(f32|bf16|f8_e4m3fn|f8_e5m2) × (f8_e4m3fn|f8_e5m2)` — four and eight, disjoint. One boolean
expression, and re-enumerating it over the same 9×9 under the pinned wheel returns the same 12
[source: `tpu_info.py:is_matmul_supported@jax 0.10.2`, executed].

**Native integer pairs: zero.** That is the whole explanation for `int8_ops_per_second = 0`, which
looks like an unpopulated field and is not. An int8 `jnp.dot` is promoted and run as bf16, measuring
**746.8 Tops/s against bf16's 764.8**; `float16` is not on the list either and measures
**729.3 Tops/s with a 14% spread** against 1–4% everywhere else, which is what an emulated path
looks like from outside (8192³ `jnp.dot` under XLA). **On 7x the only narrow dtype that buys MXU throughput is fp8** — quantizing to int8 is
a bandwidth optimization and never a compute one. int4 was never run; see §11.

**`f32 × f32` is native and the MXU still does not multiply f32** — it runs bf16 passes, and
`jax.lax.Precision` selects how many. Relative max error against a numpy float64 reference at 1024³:

| arm | relative max error |
| --- | --- |
| f32 operands, `DEFAULT` | 2.2104e-03 |
| **bf16 operands, f32 accumulator** | **2.2104e-03** |
| f32 operands, `HIGH` (**XLA only**) | 1.2995e-05 |
| f32 operands, `HIGHEST` | 1.4283e-07 |

**`f32_DEFAULT / bf16 = 1.000×` — not close, identical.** Default precision throws the mantissa away
before the MXU sees it: bf16 accuracy at f32 bandwidth, the worst cell in the table. If you did not
name a precision, storing the operands as f32 bought you nothing.

**A Pallas kernel has two rungs, not three.** `dot_general` under `Precision.HIGH` raises
`NotImplementedError: Unsupported dot precision: HIGH`. Naming
the middle rung fails the build rather than falling back — do not copy a three-row XLA precision
ladder into a kernel.

**The throughput cliff is level-dependent and the two figures are not interchangeable.** `HIGHEST` is
**3.377× `DEFAULT` inside Pallas** at `[256,256]` f32 — 617.4 µs vs 182.8 µs, spread 1.5%
— and a **4.5× cliff at 8192³ in XLA**, the honest f32 row at
147.5 Tops/s. Quote the one matching the
level and the layer you are designing at. **How many bf16 passes each level costs is not recorded
anywhere in this tree** — a claim like "`HIGHEST` is 3-pass" cannot be checked here.

**Consequence you pay at step 5, so decide it now: an XLA reference is not automatically the accurate
side of a Pallas comparison.** An f32 `jnp.einsum` at default precision lowers to the MXU and
multiplies in bf16 — the table above, read from the other end. A Pallas kernel that contracts on the
VPU instead (`jnp.sum(a * b)`) stays in **true f32** and is therefore **more accurate than the
reference judging it**, while the mismatch is still reported against the kernel; the tempting fix,
loosening the tolerance, ships a false accuracy claim about the wrong side. Measured: two `d_k`=128
contractions inside an f32 XLA reference lost **~6e-04**, failing a **4e-4** bar, while the same
kernel matched a numpy float64 oracle to **1.456e-07**. **That contradiction is the diagnostic** — fails
the XLA reference, matches the oracle to seven digits; nothing about a genuinely broken kernel
produces that pattern. So **pin `Precision.HIGHEST` in any reference `_fwd` that contracts in f32**
(`HIGHEST` *is* honoured at Pallas level; at XLA level it has been seen not to be
[unverified]), and **measure the reference against the f64 oracle too**, so
which side is wrong is proven rather than argued. **CPU interpret mode cannot catch this — there is
no MXU** — so it passes locally and fails on hardware, reading as a hardware-specific kernel bug it
is not.

---

## 4. Sublane tiling & block legality

**Sublane tiling on generation 7 is a closed form**, `num_sublanes * (32 // bitwidth)` =
`8 * (32 // bitwidth)` [source: `tpu_info.py:get_sublane_tiling@jax 0.10.2`]:

| dtype | bits | returns | evidence |
| --- | --- | --- | --- |
| float32 | 32 | 8 | measured + source |
| bfloat16 / float16 | 16 | 16 | same |
| int8 / float8_e4m3fn | 8 | 32 | same |
| **int4** | 4 | **64** | **[source:]** only — the probe never confirmed this row on hardware |

All four rows were re-read by calling `get_sublane_tiling` directly under the pinned wheel
[source: `tpu_info.py:get_sublane_tiling@jax 0.10.2`, executed]. **This is a pure-Python function
with no device call**, so executing it settles what the compiler will be told — it does not settle
what the hardware does with that, which is why the int4 row stays `[source:]` and not `[measured:]`.

The job printed `matches formula: False`, and that was **the probe's arithmetic, not jax's**: the
probe recomputed int4's bitwidth as `itemsize * 8`, and a sub-byte dtype has an itemsize of 1, so it
predicted 32 against the function's 64. The function itself uses `dtypes.itemsize_bits`, which
returns 4, and returns 64. The probe fix is on a branch and **has not been re-run on hardware**, so
nothing has exercised an int4 block on this chip — see §11.

**It is documented as a guess.** The docstring, verbatim typo included: *"Note that this is a
heurustic and depends on the settings of the XLA flags."* Generation 7 is the first where XLA enables
large 2nd-minor tiling by default. This is a statement about the current flag set, not the silicon.

**The divisibility rule — last two dims divisible by (8, 128), or equal to the array's extent — is
necessary and not sufficient.** Pure-copy kernel over a (256, 128) array, sweeping the block's
sublane dim:

| dtype | tiling | accepted | rejected |
| --- | --- | --- | --- |
| float32 | 8 | 8, 16, 32, 64 | — |
| bfloat16 | 16 | 8, 16, 32, 64 | — |
| **int8** | **32** | **32, 64** | **8, 16** (`Failed to set window params`) |

Every one of those divides 256, so the extent rule is not what fired. **f32 and bf16 blocks follow
the (8, 128) rule; an int8 block needs at least 32 rows.** Note bf16 has a tiling of 16 and accepts
8-row blocks quite happily — so this is *not* "block rows ≥ tiling". Whether the rule generalizes to
sub-4-byte types is **[unverified]**: float8 and int4 were not swept.

---

## 5. Register layout & relayout costs

**A value in registers occupies two axes the hardware treats completely differently** — the **lane**
axis (last, 128 wide) and the **sublane** axis (second-to-last, tiled per §4). Moving between them is
not free, and **no floor in this suite has a term for it**: it is not FLOPs and not HBM bytes, so a
roofline scores it at zero, and in a profile it folds into whatever op it was fused into.

**A leading axis is any axis outside the last two** — a tile index, so operators on it are
bookkeeping. The same operator on a trailing axis moves data through the XLU, and the source line
looks identical either way.

f32, on a value produced by arithmetic and consumed by arithmetic so nothing can fold the move into a
contraction; each arm against a control with the same output element count that performs no move
(`grid=(64,)` with constant `index_map`s so the loop is DMA-free, carriers (512,512) and (2048,256)):

| move | cost | Δns/step at 2048×256 |
| --- | --- | --- |
| add / drop / permute a **leading** axis | **free** | −2.6 … +5.2 (inside control spread) |
| slice on a lane boundary (multiple of 128) | **free** | +1.0 / +4.7 |
| **roll by a whole tile** — 8 sublanes, 128 lanes | **free** | −7.3 / −0.9 |
| roll by 1 along **sublanes** | 0.22–0.40 ps/elem | +212 (`pltpu.roll`: +330) |
| roll by 1 along **lanes** | 0.74–0.80 ps/elem | +420 (`pltpu.roll`: +414) |
| **swap lanes ↔ sublanes** (`swapaxes(-1,-2)`) | 0.79–0.82 ps/elem | **+430** |
| `jnp.repeat(x, 2, axis=-2)` — interleaved sublane | 3.9 ps/elem | **+4155** |
| `jnp.repeat(x, 2, axis=-1)` — interleaved lane | **does not compile** | VMEM: 265.64M of 63.94M |
| strided read, any axis (`x[::2]`, `ref[::2]`) | **does not compile** | three refusals, below |

**The transpose costs about two-thirds.** At 2048×256 the swap adds 430 ns to a 644 ns step (+66.8%) whose body is
otherwise a multiply and an add — comparable to the arithmetic beside it, which is exactly the regime
where a roofline scoring it at zero picks the wrong option.

**The cost is tile alignment, not the axis.** Rolling by a whole tile is free *on either axis*;
rolling by one costs 212–420 ns *on either axis*. Where a sliding window's stride is yours to choose,
choosing a tile multiple is worth more than any tuning around the roll.

**A strided read is a compile error, not a slow path.** Three distinct refusals, both shapes: f32 on
a value, `'vector.extract_strided_slice' op expected strides to be confined to [1, 2)`; f32 at the
ref, `Not Implemented: The last dim size is not 128 in original base memref`; bf16 at the ref,
`not implemented: Strided load with non 32-bit data`. The second names a gate that was not tested, so
a strided read may work on a differently shaped ref — do not design on that hope without a probe.

**Mosaic defers relayouts, and forcing one early to shrink a VREG count can lose.** Reshaping a value
into its consumer's layout *before* a run of elementwise ops is correct as a VREG count and was still
a **measured regression of +2.48% at one shape and +1.97% at another**, resolving at ratio 6.69 and
2.12, surviving composition. The compiler was deferring the
relayout to a cheaper place, or folding it into a neighbour, and hoisting took that away.
**A VREG-count argument is not a cost model** — nor is an op count, a byte count, or any static tally
wherever the compiler has latitude about *when* to do the work. Treat a favourable static tally as a
reason to measure, never as the result.

**Scope: f32 only. The bf16 sheet is unusable** — its `identity_2d` control measured 785.5 ns against
f32's 644.1 at the same shape, which a control should not do, and five bf16 arms came out negative.
Only two bf16 arms agree with their f32 counterparts (roll-by-1 sublane +220 ns, interleaved sublane
repeat +5025 ns); **the bf16 transpose in particular is not measured**, and a kernel that transposes
activations is usually transposing bf16 ones.

**A transpose handed straight to `dot_general` is a different measurement** — absorbed into the
contraction's dimension numbers, costing nothing. "Transposes are free" and this table do not
conflict; establish which question was asked before quoting either. When no layout choice resolves
the mismatch, it is an algebra question: restructure so the move never happens.

---

## 6. Memory capacities & allocations

**The default scoped VMEM limit is 32 MiB — exactly half the 64 MiB physical capacity — and it is a
knob.** A 48 MiB f32 scratch, written and read at different runtime offsets so it cannot be optimized
away: default `None` gives
`RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom`; `pltpu.CompilerParams(vmem_limit_bytes=128 * 2**20)`
is accepted. The largest single scratch under the default bisects to **31 MiB**. Raising
the flag moves the *scoped* limit, not the silicon — asking for 128 MiB does not create 128 MiB.

**Bytes per element track itemsize, with no floor.** Largest usable `(units, 32, 128)` scratch,
bisected per dtype, written and read at two independent offsets:

| dtype | elements | bytes | bytes/element |
| --- | --- | --- | --- |
| float32 | 8,384,512 | 33,538,048 (31.984 MiB) | 4 |
| bfloat16 | 16,773,120 | 33,546,240 (31.992 MiB) | 2 |
| int8 | 33,550,336 | 33,550,336 (31.996 MiB) | 1 |

A clean 1 : 2 : 4, all three within 12 KiB of the same ceiling. **int8 is one byte per element — no
2-byte floor — so halving the dtype really does double what fits.** The answer is the *ratio* between
three arms in one job against an independently bisected ceiling, and it is only trustworthy because
all three landed below physical capacity. An earlier version of this probe laddered bf16 past 1 GiB
without a rejection — 16× the physical VMEM, so not a capacity at all, but a static shape check with
the allocator never running, because the kernel wrote and read the same offset and the buffer was
eliminated. **Always read a capacity probe back at a different offset than you wrote it at.**

**The usable-capacity, per-operand-budget and operand-count tables are not recorded anywhere in this
tree** — see §11.

---

## 7. Dispatch & call overheads

**Two floors sit under every kernel measurement and they are routinely confused:**

| cost | size | where | does batching remove it? |
| --- | --- | --- | --- |
| host round trip | ~150 µs, up to 183–220 µs at small shapes | host | **yes** |
| per-`pallas_call` intercept | **20–36 µs** TensorCore, **~15 µs** SparseCore — see the scope note | **device** | **no** |

A fitted intercept plus ramp of **20–36 µs per `pallas_call`** that no grid tuning removes — 20.5 µs
at a 384 KiB/step block, which is **33 grid steps' worth**.
Corroborated on a real kernel: **36.0 µs per call on a chained clock** against **2.98 + 4.64 µs** of
modelled work, leaving **28.2 µs surviving the chain** as reported. Note the subtraction gives 28.38, not
28.2 — the reported residual is 0.18 µs below the difference of the two reported figures, so one of
the three is rounded. **Treat the floor as ~28 µs, not as 28.2 exactly**; the gap is 0.6% and does
not move any conclusion here, but do not build a second figure on top of 28.2 as if it were exact.

**The instrument that tells the two apart is a chained clock**: time `M` invocations inside one `jit`,
chained by a data dependency XLA cannot fold, divide by `M`. Host dispatch divides away; device cost
does not. In the same job an XLA incumbent at a small-batch shape measured **117.6 µs chained against
364.8 µs unchained** — dispatch was **30% of its own wall clock**. **Quote the chained column and say
which clock you read**; the costs are additive, not alternatives.

**Scope, and it is wider than this section used to claim. The intercept is a range, not a constant,
and it has never been measured as a single number.** Everything above is a **TensorCore**
`pallas_call` on a **chained clock**. Two other readings in this tree land elsewhere:

| instrument | figure | source |
| --- | ---: | --- |
| TensorCore `pallas_call`, chained, fitted intercept + ramp | **20–36 µs** | measured |
| TensorCore, real kernel, chained, residual after modelled work | **~28 µs** | measured |
| SparseCore gather, chained, fitted intercept net of live work | **15.22 µs** | [[STRAT-MLA-11]] |
| TensorCore attention kernel, **in-program xprof median** | **8.56 µs/layer** | [[STRAT-MLA-02]] |

**So the honest floor is ~8–36 µs depending on unit and instrument, and which end applies to you is
an open question.** Two things make it one rather than a spread to average. The unit differs —
SparseCore's fitted intercept is roughly half the TensorCore range, on the only kernel where it has
been fitted. And the isolated chained intercept is **not additive per call in-program**: the same
campaign notes that 3 × 15.22 µs would exceed the 40.08 µs/layer the whole layer costs, so whatever
the chained clock is charging, an in-program kernel is not paying all of it
[source: `raw/2026-09-sctraversal-final_report.md` §5, caveat 2].

**What to do with it.** A `pallas_call` under roughly 50 grid steps is intercept-dominated and fusing
two calls beats any block-size tuning there — that consequence is unaffected. But **do not use any
single number here as a hard gate on a roofline.** A modelled per-call time below ~28 µs is *not*
impossible by construction; it is impossible for a chained TensorCore `pallas_call` and demonstrably
reachable elsewhere in this tree. If a floor near this range decides your design, **measure the
intercept for your unit and your instrument** — the recipe is [[STRAT-MLA-11]]'s: sweep the grid
extent at constant live work and read the fit's intercept. It is one 4-arm sweep. See §11.

---

## 8. Dynamic indexing & pipelining rules

The facts most likely to invalidate a design late, because the failure is a compile error after the
kernel is written rather than a wrong number.

**A DMA's transferred extent may be traced; its destination shape may not.** A DMA of a runtime
number of rows into a fixed 64-row VMEM scratch, length from SMEM, at 8, 40 and 64 — every arm moved
the right rows and left the rest untouched.
That is the whole rule: **allocate statically, transfer dynamically.**

**A DMA's source offset may be traced for f32, and needs an annotation for bf16.** A 64-row DMA out of
an HBM ref at an offset from SMEM:

| dtype | offsets 0, 64, 128, 320 | with `pl.multiple_of(offset, 64)` |
| --- | --- | --- |
| float32 | all accepted, all correct | accepted |
| **bfloat16** | **all four rejected** | **accepted, correct** |

Every offset tested is a multiple of 64, so **the rejection is about provability, not the value.**
bf16 fails even at offset 0 with `Offsets along tiled dimensions must be aligned to tiles`. Reach for
`pl.multiple_of` by default: on f32 it costs nothing, on bf16 it is the difference between compiling
and not.

**A traced index into a VMEM ref is rejected on both axes.** `source[pl.ds(i, 8), :]` and
`source[:, pl.ds(i, 128)]` both fail with `E2003: CompileTimeMosaicUnprovenMemoryAccessAlignment`. **No lane/sublane
asymmetry** — the axis is not the variable, the mechanism is. A traced offset works through
`make_async_copy` out of HBM and does not work as a direct VMEM slice: **express data-dependent
addressing as a DMA, not as an index.** Whether `pl.multiple_of` rescues the VMEM case is
**[unverified]**; the error names the same class, so it is the first thing to try.

**Buffer depth is not a free parameter.** `pl.Buffered(buffer_count=n)` as the `pipeline_mode` of a
top-level grid `BlockSpec`: 1 and 2 accepted and correct; 3 and 4 give
`LoweringException: Only single (1) and double (2) buffering are supported. Got 3.`. **Triple buffering does not
exist at the outer grid on this version** — any design arguing "deepen the pipeline to hide a long
transfer" needs another lever. The `pltpu.emit_pipeline` position was not tested and may differ.

**Every semaphore you signal must be waited before the kernel exits; only a genuinely zero-size
transfer is exempt.** A zero-row async copy was started and never waited, and the kernel completed
with correct output — so a
ragged path issuing a copy of `end - start` rows needs no `pl.when` guard. A *real* DMA left unwaited
halts the core:

```
E0200: RuntimeUnexpectedCoreHalt: Core halted unexpectedly
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0xc3:
Semaphore (scratch argument 1) has a nonzero value upon exit from a …
```

Not a wrong answer and not a slow kernel — the core stops, and every subsequent launch **in the same
process** fails with something that reads like unrelated infrastructure noise. That is why the probe
is `destructive=True` and runs alone. **If you see `E0200` after a run of green probes, suspect an
undrained semaphore in the probe before it, not the hardware.**

---

## 9. Instruments that lie

Two counters on this chip do not measure what their names say, and both were discovered by
contradiction, days apart, after a conclusion built on them had been published and had to be
retracted. These are the xprof/LLO counter planes (`VF_CHIP_DIE0_TC_…` on `/device:TPU:0`) at this
libtpu.

**`COUNT_CYCLES` is a free-running counter over an enable window, not a measure of execution.** The
window is decoupled from the kernel, the module and the capture: in one job it accumulated over
**683.64 ms against 49.73 ms of module execution** — 13.7× longer than the program ran — and the
implied clock across its four captures is **3.1, 3.3, 8.0 and 22.9 GHz**.
**Never put it in a denominator.** Anything of the form `X / COUNT_CYCLES` is a real quantity over a
meaningless one.

**`COUNT_MXU_BUSY_0` counts MXU *idle*, not busy.** It equals
`COUNT_CYCLES − COUNT_MATPUSH_CYCLES_MXU_0` exactly in all 12 captures, and `MXU_BUSY_0 + _1 + _2` sums **exactly** to `COUNT_CYCLES`. It is a partition of the enable window, so it inherits that window's
non-determinism. The same partition reading holds for `XLU_BUSY_*`. And because `num_mxus = 2` with
**identical** push counts to both (§1), **`_0` alone is half the traffic, not a per-unit breakdown.**

**Pinning a different `jax` / `jaxlib` wheel varies neither compiler.** Both the Mosaic TPU and the
XLA:TPU pipelines ship inside `libtpu.so`, so the wheel pin in this page's frontmatter is not the
thing that selects a compiler. Symbol counts, `libtpu.so` against jaxlib: `apply-vector-layout` 5/0,
`infer-vector-layout` 4/0, `canonicalize-mosaic` 2/0, `relayout-insertion` 3/0, `VectorLayout`
741/32, `LatencyHidingScheduler` 165/0, `MemorySpaceAssignment` 254/0, `xla_tpu_enable` 1021/0;
jaxlib's `_tpu_ext.so` is a **3624-byte stub**. **An A/B across jax
versions that holds `libtpu` fixed is null by construction** — reading its ≈0 delta as "the compiler
is not the cause" is reading a result out of an experiment that had none.

**Run the determinism control before you read any counter at all** — the same variant at the same
shape in two jobs, or two replicate captures in one, discarding every counter that moves. One extra
arm, and it is the only thing separating a counter that measures the program from one that measures
elapsed time. Under it `COUNT_CYCLES` moved **4.52–5.77% between clean replicates and 81.24% when
the host stalled** and `COUNT_MXU_BUSY_0` moved **83.37%**, while these nine were bit-identical at
**0.0000%** across four arms: `COUNT_MATPUSH_MATRICES_MXU_0` / `_1`,
`COUNT_MATPUSH_CYCLES_MXU_0`, `COUNT_MATMUL_VREG_F32_MXU_0`, `COUNT_VECTOR_ISSUE`,
`COUNT_SCALAR_ISSUE`, `COUNT_BUNDLES_INTO_VIF`, `COUNT_VST_INSTRUCTION`, `COUNT_VLD_INSTRUCTION_0`,
`COUNT_ROTATE_PERMUTE_INSTRUCTION_XLU_0`. **Those nine are sound as raw counts and as deltas between
two arms in the same job.** They are not sound as fractions, and they are whole-chip aggregates
including non-kernel ops, so a per-µs rate built from one is a mismatched ratio and is not an
occupancy.

**A percentage carries an air of having been normalised by something real.** Two cheap defences.
*Sanity-check the implied physical constant* — cycles over wall time is a clock speed, and the one
division above is what caught `COUNT_CYCLES`. *Prefer a raw count and state its denominator inline*:
the same counters yield "1.65 memory ops per arithmetic op" or 1.84 or 0.59 depending which you
pick, and "97.1% of cycles" turned out to be `COUNT_VECTOR_ISSUE / COUNT_SCALAR_ISSUE` with
`scalar 100.0%` printed beside it all session. **A resource at exactly 100.0% is a quantity divided
by itself, never a finding.**

---

## 10. SparseCore architecture & performance

Four constants, here rather than in a strategy entry because each silently invalidates a design.

**VMEM has 32 banks, and the conflict is linear in ways.** A lane-private table strided by a multiple
of 32 puts every lane on one bank; cost fits `base + (ways − 1) × 4.7 µs`, and 16 lanes sharing a
bucket serialize 16-way. **The bank count is inferred from that
ladder, not read out of a spec** — `SparseCoreInfo` declares subcores and lanes but not banks, so 32
here reproduces the observed 16-way ceiling and the odd-stride fix, and that is as far as the
evidence goes. Do not confuse it with the 32 subcores below, which is a different 32 and *is* in
source. Design rule: [[STRAT-SC-06]].

**A wave is 32 vector subcores in lockstep, and it costs its longest row.** `sparse_core.num_cores=2`
× `num_subcores=16` [source: `tpu_info.py@jax 0.10.2`] — **one chiplet's** SparseCore complement; a
chip has four cores. Row `w` is `w * (cores * subcores) + flat`, so any per-row variable extent is
governed by `E[max over 32]`, not the mean — for `rl ~ U(0,n)` that is `32/33 ≈ 0.97n`. The **count** is source; the **cost model** is measured.
Modelling rule: [[STRAT-SC-05]].

**The pooled-VMEM denominator is 2,097,152 words per core.** 524,288 B per vector subcore ÷ 4 =
131,072 words, × 16 subcores. This is the figure to quote a working set against; the
`sparse_core.vmem_capacity_bytes` line in §1 is **per subcore and not poolable**, which is the easiest
number on this page to misuse. Two adjacent figures were never measured and must be re-probed before use:
the largest scratch the compiler actually accepts, **480 KiB**, and SparseCore SMEM topping out at
**4 KiB** against the TensorCore's ~1 MiB [unverified].

**Achieved streaming bandwidth is 338 GB/s per chiplet, aggregate over all 32 subcores**
— reproducing 337 GB/s measured earlier, with 23.7× scaling across the 2×16
fan-out. **Aggregate is the operative word: there is no per-subcore escape hatch**, so a transfer
lever cannot be justified by dividing this by 32. Two qualifiers travel with it and are load-bearing:

- **It is per chiplet, not per chip.** Each probe ran on one device, and a device is one chiplet. A
  chip-level total would be ~676 GB/s and nothing here measured one.
- **It is an achieved bandwidth, not a ceiling.** It is **10.5%** of the 3207 GB/s practical peak and
  **9.9%** of the chiplet's two-SparseCore 3400 GB/s ceiling — 2 × 1700, §2. **Match the denominator
  to the aggregate:** 338 is summed over both cores, so against *one* core's 1700 it is **19.9%**, and
  quoting ~10% against 1700 is a 2× error. Either way the gap it describes is the 32-byte DMA
  granule and the access width — not the silicon's limit. [[STRAT-MLA-06]] measures **509–709 GB/s on
  640-byte indirect rows**, 1.5–2.1× this figure and still only 15–21% of 3400. **Neither number is a
  roofline**; price a gather against its row width, and treat 1700/3400 as the only ceiling in this
  area.

Restating against the practical peak matters for the TensorCore too: the long-quoted **9.1% SC /
78.8% TC** were both taken against the *theoretical* 3.7 TB/s; against 3207 GB/s they are **10.5%** and
**90.9%**. The TensorCore is not leaving 21% on the table — it is within 9% of what the memory system
can deliver, which is a different conclusion about where a TC-side transfer lever could go.

**"Good at data-dependent addressing" does not survive as a single fact — read wins, read-modify-write
loses.** Gather is **51.8× in the SparseCore's favour** against an XLA gather, for a table resident in
SC VMEM and a **1-element row** (replicated at 50.01×). That qualifier is not decoration: sweeping row width, the ratio falls
61.25× → 19.06× → 12.98× → 3.31× → **0.93×** at w = 1/4/16/64/256 f32, so **a 1 KiB row is an XLA
win**, and useful throughput saturates at ~230 GB/s by w=16 — the 32-byte granule
(`sparse_core.dma_granule_size_bytes = 32` [source: `tpu_info.py@jax 0.10.2`]), visible directly. That sweep varies `chunk` with width; at fixed `chunk=128` the
shape and the crossing survive but the w=1 headline is **40.02×**. Scatter-add goes the other way — **6.6× against** the
SparseCore, 226.0 µs vs 34.4 µs — and only **3.38× better** when
fanned across all 32 subcores, still 1.9× behind XLA. The
streaming unit's 23.7× fan-out does not transfer to it.

---

## 11. Not established

- **The five bandwidth ceilings in §2 are `[unverified]` in this tree.** Supplied as vendor figures on
  2026-09-16; no probe produced them and no pinned source path backs them. They are consistent with
  what *is* measured **to within 0.4%** — 3686 GB/s per chiplet against jax's rounded `3.70e12`
  (3700), and 96 GiB against jax's 103e9 B (95.93 GiB). That corroborates the *framing*; it does not
  corroborate the two numbers that are new (3207, and the SparseCore pair), and it is not exact
  agreement — see §2. **What would settle them:** a streaming probe walking HBM with wide TC
  accesses, reporting achieved GB/s against wall clock — if the practical peak is real it plateaus at
  ~3207, not ~3686 — and the same probe SC-side, driving one core then two. Until then a roofline on
  3207 is the conservative choice and one on 3686 the optimistic one; **say which you used.**
- **The TC/SC concurrent-access penalty is a direction, not a number.** "Below 3207 GB/s" is all that
  is claimed. Any design needing its size needs a probe.
- **The per-call dispatch intercept has never been measured as one quantity** — §7 now carries four
  readings spanning **8.56 to 36 µs**, across two units and two instruments, and no probe has held
  the kernel fixed while varying only the unit or only the clock. This page previously stated a flat
  "below ~28 µs is unreachable by construction", which was a TensorCore chained-clock result written
  as a chip fact; four entries in `strategies/` measure below it. **What would settle it:** the
  [[STRAT-MLA-11]] sweep — grid extent at constant live work, read the intercept — run on a
  TensorCore and a SparseCore kernel in the same job, then the same two kernels timed in-program.
  Until then the intercept is a per-unit, per-instrument quantity you measure, not a constant you
  quote, and **a roofline must not reject a design for landing under it.**
- **Four figures here were never measured**, tagged `[unverified]`: `HIGHEST` not
  being honoured at XLA level (§3); SparseCore's **480 KiB** largest accepted scratch and **4 KiB**
  SMEM (§10); and **XLA's chosen layout not matching `get_sublane_tiling`** — the returned values in
  §4 are what the function returns; whether XLA's layout agrees was measured separately and did not.
- **Two things are recorded nowhere in this tree:** the usable-capacity, per-operand-budget and
  operand-count tables (§6), and the map of how many bf16 passes each `jax.lax.Precision` level costs
  (§3).
- **The relayout table is f32 only; bf16 is not measured** and its control misbehaved rather than
  merely being absent — see §5. Two arms remain unexplained: the negative `concat_lane_2x` (it differs
  from its control in output rank, so the consumer's store differs — more likely a wrong control than
  a free concat), and the f32 strided-read gate on "last dim size is not 128", which may make a
  strided read legal at some shapes.
- **int4 was never run.** It appears in §4's tiling table because the formula produces a value for it.
  Whether an int4 matmul promotes the way int8 does, and whether an int4 block obeys a 64-row minimum
  the way int8 obeys 32, are both **[unverified]**.
- **float8 block legality was not swept**, so "int8 needs 32 rows" has one data point behind the
  general rule, not two.
- **`buffer_count` was measured at the top-level grid only**; the `pltpu.emit_pipeline` position is
  **[unverified]**.
- **Interpret mode does not check dynamically-indexed VMEM scratch** — scratch addressed by a traced
  index such as `lax.rem(p, 2)`, which is exactly how a double-buffered kernel addresses its crossing
  buffers. A pipelined design validated **23/23 green** with nothing to catch. **[unverified]** —
  judgement from the shape of the hole, not a probe that demonstrated a miss. What *is* established
  is that interpret mode sees no MXU (§3) and cannot see Mosaic lowering bugs either, so the cheap
  correctness tier is thinner than it looks.
- **The counter facts in §9 are this libtpu and this capture mechanism.** Whether `COUNT_CYCLES`
  behaves the same under a different capture path, and whether the `MXU_BUSY` partition holds on
  non-7x chips, are both **[unverified]**. The determinism-control numbers come from the GDN conv1d
  workspace, which has **no `raw/` copy in this tree**.
- **Nothing here touches multi-device.** Every capacity, rate and limit is one TensorCore, except the
  per-chip column of §2's table, which is arithmetic on the per-chiplet figure and not a measurement
  of two chiplets running together.
- Every shard of the single-chip probe job logged `INVALID_ARGUMENT: unexpected worker hostname`
  from `libtpu_init_utils.cc` because `TPU_WORKER_HOSTNAMES` is unset in the single-chip recipe. It
  appears on the passing shards as well as the halting one — harness noise, not a signal about any
  probe.
