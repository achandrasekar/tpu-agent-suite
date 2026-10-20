# 2026-09-dsamla — WHITEBOARD excerpt (arbitration record only)

**Verbatim excerpt.** This is six sections of an 1,500-line running log, copied without edit.

**What was dropped and why.** The log's other sections are the campaign's forward narrative --
target selection, shape derivation, upstream mining, roofline setup, per-job dispatch notes and
result tables. They are block-specific and reconstructable from the final report. What is kept here
is the *arbitration* record: every place a later measurement retracted an earlier number, together
with the reasoning as it stood before the retraction. Those are the parts nobody would reconstruct,
and the parts a reader checking a promoted entry is most likely to disbelieve.

**Which entry each section serves.**

| section | serves |
| --- | --- |
| §13 | STRAT-MLA-04 (comparator/instrument failure modes); correctness story in the report |
| §14 | STRAT-MLA-04 (a relative-error bar that charged the kernel for cancellation) |
| §15 | **STRAT-MLA-04** — the CSE'd gather: a loop carry that did not reach the gather's inputs |
| §16 | **STRAT-MLA-06** — the 338 GB/s refinement *and* the decision not to act on it |
| §19 | **STRAT-MLA-07** — the widening roofline ratio; also the residual traversal cost |
| §22 | **STRAT-MLA-02, STRAT-MLA-03, STRAT-MLA-10** — the batch=1 and geometry reversals |

---


<!-- excerpt: §13 — Job #5: the correctness failures were mine (void comparator) -->

## 13. Job #5 landed: three shards, and the correctness failures were mine `[source: j-b844c441-6f28-4d41-96a1]`

### 13a. All eight correctness verdicts are void -- the comparator was wrong

Five arms reported `!! WRONG` with an identical signature at every geometry:

    nope_ok=False  rope_ok=True  first_bad_row=2048  n_bad=6144

That is not a kernel bug, it is `device_slope.py` comparing the wrong rows.
`dsa_gather` returns nope already reshaped to `(N, WORD_BYTES, TILE_LANE_BYTES)` -- one row per
index -- so the live region is the first `LIVE`=2048 rows. I sliced `[:LIVE * WORD_BYTES]` = 8192
rows. Rows 2048..8191 lie in skipped windows, which T0 deliberately leaves untouched, so they hold
whatever the allocator last left there. `8192 - 2048 = 6144` is exactly `n_bad`, and rope was clean
precisely because rope was sliced correctly.

Which also explains the otherwise baffling pattern -- `when` "wrong" at `(32,4)` but "correct" at the
other three, `grid` "wrong" everywhere. Untouched memory sometimes happens to match and sometimes
does not. **So I cannot currently say whether `grid` is broken; the evidence that said so was
garbage on both sides.** Fixed, with the reasoning recorded at the slice so it is not re-introduced.

The same class of bug killed chip 1: `unit_test.py:209` differenced a `(1,64,576)` kernel output
against a `(1,64,512)` oracle. The kernel returns the q width (512 lora + 64 rope) and only the
leading 512 carry the attention output. Fixed by slicing; the existing "stock also matches the
oracle" check is what would catch a wrong slice, which is why it earns its place.

**The suite has still never completed a run.** 8 PASS, 0 FAIL, no `RESULT:` line. Nothing about T0's
correctness is established yet -- the 8 passes cover lowering, gather byte-exactness and the R=1
padding rows, not the oracle comparison that aborted.

### 13b. The overlap control cleared the slopes

    independent 32.30 us   dependent 34.02 us   ratio 1.05x   -> no meaningful overlap

So §12c(b) is dead: the K calls do not overlap and the slopes are per-call latencies. The
above-roofline reading stands and must be (a) a wrong 338 GB/s fact or (c) a wrong byte count. Since
both candidates now only affect how the number is *described*, not T0's value, this stops being
load-bearing -- but the hardware fact should not be cited again until it is rechecked.

### 13c. Timings, believable modulo the correctness re-run

Stock rises monotonically as the window shrinks -- 30.75, 33.90, 35.60, **45.91** us at
(32,4)/(16,4)/(8,4)/(4,4) -- confirming §12b and killing §9a for good. The skip arms all land
9-19 us. Taking only `when`, whose correctness is at least plausible:

| geometry | stock us | when us | ratio |
| --- | --- | --- | --- |
| (32,4) | 30.75 | 18.91 | 1.63x |
| (16,4) | 33.90 | 14.01 | 2.20x |
| (8,4) | 35.60 | 12.97 | 2.37x |
| (4,4) | 45.91 | 12.64 | **2.43x vs shipping**, saving 18.11 us/call |

Note `when` barely improves past (8,4) -- 12.97 -> 12.64 -- while stock degrades 35.60 -> 45.91. The
skip ceiling rises 8x -> 16x but the measured skip does not follow it, so something other than
window count binds the skipped path. `(8,4)` may be the better operating point; the re-run decides.

### 13d. The end-to-end number does not yet exist, and section B is not it

`benchmark.py` section B put the whole attention at 231 us with skip deltas of -5 to +19 us and
12-17% spread -- i.e. +-30 us of noise around a ~22 us effect. **It cannot discriminate**, and its
0.98x-1.09x range is equally consistent with the full gather win landing and with none of it
landing. Section C is the same: batch 8 looks ~7% better than batch 16, inside the same noise.

So the honest position: T0 is worth 2.4x **on the gather in isolation**, and *nothing is known* about
what that is worth to the attention, let alone to a served token. `attention_slope.py` applies the
same K-regression to `sparse_ragged_paged_attention` and is the measurement that decides whether any
of this ships. Prediction: if the gather is fully exposed the attention slope falls by ~18 us; if it
falls by much less, the gather overlaps other work and T0 is worth a fraction of its microbenchmark.


<!-- excerpt: §14 — Job #6: correctness settled, the attention reading withdrawn -->

## §14 Job #6 (j-2c4dfcd3-d154-4231-8ec4) -- correctness settled, the attention reading withdrawn

Three shards, all ran. rc=1 on chip 2 is a correctness result, not an infra fault.

### §14a T0 is correct. Both skip modes, all four geometries.

`device_slope.py` with the fixed comparator: **8 of 8 CORRECT**, not one `!! WRONG`.
Both `when` and `grid` are bit-identical to stock on the live rows at (32,4),
(16,4), (8,4) and (4,4). The five failures in job #5 were entirely my slice bug;
the three passes were equally void. This is the first real correctness evidence
for T0 and it is clean.

`unit_test.py` reached `RESULT:` for the first time -- 14 PASS, 9 FAIL. **None of
the 9 is attributable to T0**, and the file says why in each case:

- 6 are paired `X / matches float64 oracle` + `X / stock also matches`, with
  *identical* `max rel` in each pair, while `real rows bit-identical to stock`
  passes `exact` in all four configs. A bar both arms fail identically is a bar,
  not a regression. Diagnosis: `out` is a convex combination of ~2048 zero-mean
  rows, so it lands ~sqrt(topk) below the values it averages and a relative
  bound charges the kernel for the cancellation. `mean rel` was 4.3e-3 at every
  config -- one bf16 ulp, flat -- while `max rel` grew with R, which is the
  signature of cancellation and not of error. Fixed: the oracle now returns the
  weighted mean of |v| alongside the result and the bar is 64 bf16 ulps of
  *that*. 64 is tight, not generous: sqrt(2048)*2^-8 = 0.18 is what an honest
  bf16 accumulation is allowed to drift, i.e. ~180 ulps.
- 1 is `multichunk_P128_R70 / padding rows are exactly zero | max |pad|=4.875`,
  unpaired. Arithmetic, not assumption: `batch_end = cdiv(70,16)*16 = 80`, so
  rows 80..127 are never reached by the grid and keep the input `q` verbatim.
  max|N(0,1)| over 48*64*576 = 1.7M draws is ~5.0, and 4.875 is that in bf16.
  Stock behaviour. The check asserted a property the kernel never promised.
  Fixed by splitting at `batch_end`: [R, batch_end) must be exactly zero (this
  is the check that actually guards the skip, via the kv_len pin at
  kernel.py:715), [batch_end, P) must still equal the input `q` -- untouched,
  which is stricter than the "finite" it replaces, since leaked gathered
  garbage would also be finite.

### §14b `attention_slope.py`'s verdict is withdrawn. The instrument has a hole.

It reported skip worth ~1.0x on the whole attention and `attention_kernel_batch_size=4`
worth 2.67x. One pair of numbers from the same job kills it:

      attention, (4,4) batch=4        11.64 us
      the gather it contains, (4,4)   41.20 us

An attention call cannot be a quarter of a gather it performs. And the gather
volume does not vary with the batch at all -- `kernel.py:794` gathers
`chunk_size * topk` rows where `chunk_size` comes from
`gather_and_attention_chunk_size` (64), never from `attention_kernel_batch_size`.
So in that instrument the gather is off the measured critical path.

Mechanism: gather is SparseCore, `_attention_kernel` is TensorCore, and the K
chained calls are independent (only `q` is perturbed), so gather(k+1) issues
under attention(k). The chain measures max(gather, attn), not the sum.

`device_slope.py`'s 1.02x overlap control could never have caught this -- it
chained SparseCore to SparseCore.

**Production has no such overlap.** Within a layer gather->attention is serial;
across layers, layer L+1's indices derive from its own `q`, hence from layer L's
MLP. Nothing for the gather to hide under. So the dependent chain is the
production-faithful model and the independent one is a lower bound on exposed
cost. `attention_dep_slope.py` measures both per arm; dep/indep *is* the result.

### §14c What `attention_kernel_batch_size` actually is

Not a tiling knob. `kernel.py:818-824`:

    batch_end = start_pos + cdiv(min(cu_q_lens[distribution[2]], end_pos)
                                 - start_pos, kernel_batch_size) * kernel_batch_size

The grid stops at the last batch containing a real token, so at R=1 in a P=16
bucket batch 16 walks 16 token slots to serve 1, batch 8 walks 8, batch 4 walks
4. Measured 31.07 / 20.65 / 11.64 us is almost exactly proportional to 16:8:4.

That is T0's own thesis -- stop paying for padding -- already half-implemented on
the TensorCore side. It is a one-line config change, it composes with T0 rather
than competing (they truncate different units), and at this cell it may be the
larger of the two. Its cost is elsewhere: at R=16 the truncation buys nothing and
the finer grid may cost per-batch overhead. Neither arm is measured yet with the
gather serialized, and correctness at batch=4 is entirely unchecked -- the timing
instrument never compared an output.

### §14d Standing caution on the numbers still in play

Both batch=4 arms had the weakest fits in the table (r2=0.9832, 0.9175) against
0.9982 for the shipping arm, and the 177 us harness floor is ~15x the slope being
extracted. Those two figures rest on a small residual and are the ones to re-take
first.


<!-- excerpt: §15 — Job #7: T0 verified correct; both attention instruments void (CSE) -->

## §15 Job #7 (j-a8845b56-1b50-40a6-9289) -- T0 verified correct; both attention instruments void

### §15a Correctness: `RESULT: PASS (all checks)`, 26 PASS / 0 FAIL

The first clean run of the suite, at production geometry (topk=2048, 64 heads,
page_size=1024) across four configs, which nothing in the repo's own tests
reaches. What it establishes:

- real rows **bit-identical to stock** in all four configs (`exact`), at both
  skip modes and all four gather geometries (the latter from `device_slope`);
- written padding rows [R, batch_end) **exactly zero** -- the check that
  actually guards the skip, since it is what the kv_len pin at kernel.py:715
  buys;
- unreached rows [batch_end, P) **still equal the input `q`**, bit-for-bit;
- **0.4-0.8 ulp** against the float64 oracle on a 64-ulp bar, with stock landing
  on the identical figure to the tenth of an ulp.

Both fixes from §14a were right, and the ulp figures say the new bar is not
merely looser -- 0.8 of an ulp is three orders of magnitude inside it.

T0's correctness is no longer an open question. Everything still open is
performance.

### §15b The overlap hypothesis was wrong. The defect is CSE, in my instrument.

dep/indep came back 1.00, 1.03, 1.14, 0.97, 1.22, 1.02 across six arms -- no
meaningful overlap anywhere. So §14b's explanation of the gap is refuted by the
control built to test it. The gap itself survived:

      (4,4) skip=0 batch=4, serialized   14.99 us
      the gather it performs, alone      41.20 us

With no overlap available, a call cannot be a third of something it contains.
The gather is simply **not in the measurement**.

Cause: `sparse_ragged_paged_attention` derives its gather from
`(cache, topk_indices, page_indices)` alone. `attention_slope.py` perturbed `q`,
so those three were identical across all K calls and XLA computed the gather
once. `attention_dep_slope.py` v1 inherited it and then routed the dependency
edge through `q` as well -- serializing the attention kernels around a gather
that was still hoisted out of the loop.

This is one bug, and it is the same bug as §13's comparator and §14a's oracle
bar: an instrument that looked like it was measuring the thing named in its
docstring. Rule earned: **when the containing call measures cheaper than a part
it contains, stop and find the hole -- do not reach for a mechanism that
explains it.** Both times the arithmetic was sitting in the output.

Void as a result: every "skip is worth ~1.0x on the attention" number in §13-§14.
Not void: the batch arms, which measure the attention kernel correctly -- but
they measure it with the gather deleted, so `batch=4 is 2.67x on the attention`
must be restated as `2.67x on the attention kernel, gather excluded`.

### §15c What the numbers now imply, pending the re-run

Composing the two instruments, each measuring the half it measures honestly:

      gather (device_slope, indices vary per call) : 34.82 us  -> 10.61 us (8,4 grid)
      attention kernel (batch arms, gather CSEd)   : 31.07 us  -> 11.64 us (batch=4)

If they are serial and additive, shipping is ~66 us/layer and the two
optimizations attack disjoint halves. That is a prediction to test, not a
result: it assumes additivity, and the whole point of §15b is that composing
numbers from instruments with different holes is how the last two rounds went
wrong.

### §15d v2 proves the fix instead of asserting it

Indices are now rotated per call (`(topk + off_k + carry) mod seq_slots`) --
same count, same scatter, same sortedness bar one wrap, no -1 introduced so
`kv_lens` and the skippable mask are untouched -- and the dependency edge rides
the index path. The instrument counts Mosaic custom calls in the *optimized*
HLO at K=1 and K=8 and reports kernels-per-call: 3 means gather_page_ids +
dsa_gather + _attention_kernel all replicate and the slope contains a real
gather; 1 means the gather is still hoisted. Any arm scoring <= 2.5 is named
and its timings are disclaimed above the table rather than beside it.


<!-- excerpt: §16 — the 338 GB/s contradiction, and the decision NOT to overturn it -->

## §16 The 338 GB/s contradiction, probably resolved -- and not in favour of overturning it

Recorded in §12 as "do not cite the 338 GB/s SparseCore streaming figure again
until rechecked", because `device_slope.py` measured 509-605 GB/s on the read
side across four geometries (602.3 / 605.2 / 535.2 / 509.0 at 640 B/row), i.e.
1.5-1.8x over a figure `hardware-facts.md:590` carries as verified by
**three** independent jobs (j-74a24183, j-996520b8, j-48095a41).

Three jobs beat one side-effect measurement, so the resolution is unlikely to be
"the constant is wrong". The likely reconciliation is in the access width, and
the archived treatment says so itself: `hardware-facts.md:619` notes useful
throughput saturating at **~230 GB/s by w=16**, explicitly attributing the
ceiling to the 32-byte DMA granule. The DSA gather reads a **640-byte
contiguous row** per index -- 20 whole granules, no partial-granule waste --
which is the favourable end of that curve, not the one the campaign measured.

So the honest statement is narrower than either "338 is wrong" or "do not cite
it": **338 GB/s is not a ceiling for wide-row indirect reads.** At 640 B/row the
measured rate is 509-605 GB/s, four geometries, consistent to +-9%.

Not acting on this beyond the note. Amending a thrice-verified constant in a
shared checked-in reference on the strength of an instrument whose last three
versions each had a hole in them would be the wrong trade, and the measurement
was a by-product rather than the question. It goes in the final report as a
flagged refinement with the four numbers attached, for someone to confirm
deliberately.

Worth carrying separately: this makes the read-concentration result (41x for
same-address vs scattered, r2=1.0000) the more transferable of the two
bandwidth findings, since it is a ratio measured within one instrument and does
not depend on the absolute byte accounting being right.


<!-- excerpt: §19 — Roofline Gate: it does not close, and why that redirects -->

## §19 Roofline Gate -- it does not close, and the reason says where to go next

Deferred until a real baseline existed, because a gate against a guessed
baseline is just a roofline, and a roofline is never a result.

Machine: tpu7x. HBM 3.70e12 B/s, bf16 1.155e15 FLOP/s. Row = 512 B fp8 `nope`
+ 128 B bf16 `rope` = **640 B**, confirmed against the array shapes rather than
assumed (`n_slots x LKV_DIM = PAGES x PAGE_SIZE x WORD_BYTES x TILE_LANE_BYTES`).

### §19a Compute is irrelevant here; only movement matters

At R=1 the attention arithmetic is `1 x 64 heads x 2048 keys x (576+512) x 2`
= 0.29 GFLOP = **0.25 us** of bf16 MXU. At R=16 it is 4.56 GFLOP = 3.95 us.
Against a measured 27-64 us, the matmuls are rounding error. Nothing in DSA
decode is compute bound and no MXU-side optimization can matter. Ignore the
FLOP column from here on.

### §19b Traffic model, and the gap that widens as work is removed

Three movements per layer: the gather reads the cache, the gather writes the
gathered buffer, the attention kernel reads that buffer back. The gather moves
its live token count; the attention moves `batch_end` tokens.

      arm                            MB    floor   measured   ratio
      (32,4) skip=0 batch=16 SHIP  62.91  17.00us   64.16us    3.8x
      (32,4) skip=1 batch=16       23.59   6.38us   52.24us    8.2x
      (8,4)  skip=1 batch=16       23.59   6.38us   44.95us    7.0x
      (4,4)  skip=1 batch=4         7.86   2.13us   27.19us   12.8x

The ratio column is the finding. **Every optimization so far has moved the
kernel further from its roofline, not closer.** 3.8x -> 12.8x. That is the
signature of removing data movement from something whose cost is not data
movement: the bytes fall 8x and the time falls 2.4x.

So the gate does not close -- 12.8x of headroom remains -- but it does not
close for a reason that redirects rather than blocks. Bandwidth was never the
constraint at decode. Fixed per-invocation and per-window cost is.

### §19c The machine's own demonstrated rate is the useful floor

A theoretical 3.7 TB/s is not reachable by an indirect gather, so take the
floor the hardware itself demonstrated, on this kernel, in this job:

      (32,4) stock, 16 tok   32768 rows / 29.59 us = 0.90 ns/row = 709 GB/s
      (8,4)  skip,   1 tok    2048 rows / 12.35 us = 6.03 ns/row = 106 GB/s
      (4,4)  skip,   1 tok    2048 rows / 11.81 us = 5.77 ns/row = 111 GB/s

The stock arm proves the SparseCore can retire a 640-byte indirect row every
0.90 ns. At that demonstrated rate, one token's 2048 rows should cost
**1.85 us**. The skipped gather costs ~12. So roughly **10 us per layer is
fixed traversal cost that the skip does not remove** -- windows walked and
predicated off rather than not walked at all.

That is worth about as much again as everything measured so far (10 us x 78 =
0.78 ms/step), and it is the next lever. Note also that `grid` mode, which was
supposed to be exactly this fix, does not deliver it: at (8,4) `when` reads
12.35 and `grid` 12.47. Whatever the residual is, shortening the emitted grid
does not remove it. That is a finding about `grid` mode, not a reason to prefer
`when`, and it should be stated in the report rather than buried.

### §19d What the gate authorizes

Continue: yes, the headroom is real and large. But **not within this task's
remit.** The user's question was the cost of chunking, and that is answered and
addressed. The residual fixed cost is a separate, larger investigation into
SparseCore window traversal, and it should be carried as a named next step with
its own estimate rather than folded into this change.

Also worth recording, because it contradicts a shared reference: 709 GB/s on
640-byte indirect rows is a fourth independent sighting above the 338 GB/s in
`hardware-facts.md:590`. See §16 -- still not amended, still flagged.


<!-- excerpt: §22 — Job #11: the prefill result reverses two conclusions -->

## §22 Job #11 (j-9773d307-2353-472a-adec) -- the prefill result reverses two conclusions

Both shards rc=0, first attempt, `k/call` 3.00 (P<=64) and 5.00 (P=128, two
chunks, so five custom calls per invocation -- consistent, not an anomaly).

### §22a The two shards agree, which is what makes this trustworthy

Same cell (P=R=64), two scripts, two chips, same seed:

      arm            prefill_batch   geom_final     diff
      (8,4) b16           210.55       210.90     +0.17%
      (8,4) b4            203.96       205.50     +0.76%
      (8,4) b1            224.58       225.36     +0.35%
      SHIP                187.34       179.38     -4.25%

The three candidate arms agree to under 1% across independent scripts and chips.
SHIP disagrees by 4.25%, which is a real caveat on the baseline rather than on
the candidates; **use the lower (179.38) whenever claiming a win**, so the
comparison is stated against the incumbent's better number.

### §22b batch=1 is the WORST arm at real prefill, not the best

This is the reversal, and §20c was right to refuse the extrapolation.

      P=R=64    b16 0.89x   b4 0.92x   b1 0.83x     (vs SHIP, (8,4))
      P=R=128   b16 0.87x   b4 0.90x   b1 0.85x
      (32,4):   b16 0.98x   b4 1.00x   b1 0.89x     at R=64

At P=16 the batch=1 row read 1.07x and looked like a free default. At a full
64-token chunk it is 0.89x. The curve does not just flatten, it **turns over**:
b16 -> b4 improves, b4 -> b1 regresses. 16 grid steps beat 4, and 64 lose to 4.
Per-grid-step overhead is real; it was simply invisible at a chunk four times
too short to show it. Every number in §20b's R=16 column stands, and none of it
generalised.

Recorded as the fourth instance of the same lesson: **a monotone trend measured
on a short axis is not evidence about the long axis, least of all when the axis
being shortened is the one carrying the cost.**

### §22c The gather geometry change is dropped

      geometry (8,4) vs (32,4), both at batch=1:
        R=1  : +0.81 us/layer  (2.6% of the arm)
        R=64 : -23.47 us/layer (13% regression)

`_NUM_ROW_SUBCHUNKS` is a module-level constant in `dsa_gather.py`. Changing it
retunes the gather for every caller, every shape, including shapes this
investigation never measured. That is a large review surface and a large blast
radius, and it is being bought with 2.6% on one shape while costing 13% on
another. Drop it. The shipping geometry stays untouched.

This also retires the (8,4)-vs-(4,4) question from §18c entirely -- neither
ships, so the (16,4) r2=0.9363 outlier no longer needs explaining either.

### §22d The candidate, and why each knob is at the value it is

      skip_padding_gather          False -> True
      attention_kernel_batch_size     16 -> 4
      gather geometry                    unchanged

At (32,4), from `geom_final`:

      R=1  of P=64:  SHIP 131.90  ->  skip+b4  40.16   = 3.28x
      R=64 of P=64:  SHIP 179.38  ->  skip+b4 179.09   = 1.00x

batch=4 is the turning point of the curve in §22b, and it is the only batch that
is both a large decode win and non-negative at prefill. skip and batch compose
rather than compete: skip attacks the gather (which is pure waste at R<P) and
batch attacks the attention kernel's padding traversal.

Correctness for exactly this configuration is already in hand -- `unit_test.py`
runs at the shipping geometry (it prints `row_chunk_size = 512`) and
`decode_P16_R1_batch4`, `full_P16_R16_batch4` and `multichunk_P128_R70_batch4`
all pass with real rows bit-identical to stock. No new correctness job needed.

`final_candidate.py` now measures the composed candidate whole against the
incumbent, interleaved in one process over four cells, which is the form the
skill requires and which no measurement so far has taken.
