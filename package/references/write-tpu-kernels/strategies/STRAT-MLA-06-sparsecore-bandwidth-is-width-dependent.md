```yaml
id: STRAT-MLA-06
title: The SparseCore streaming figure is access-width dependent; 338 GB/s is not a ceiling for wide-row indirect reads
hardware: tpu-v7x
unit: sparsecore
software: any
stage: [1, 2]
lever: layout
generality: unit-wide
status: measured
source: raw/2026-09-dsamla-WHITEBOARD-excerpt.md §16; raw/2026-09-dsamla-STRATEGY.md § STRAT-DMA-03
recheck: **this entry is the recheck.** Measure achieved bandwidth at your row width with settled byte accounting before using either number
origin: campaign (DSA/MLA, GLM-5.2-FP8), 2026-09
```

# STRAT-MLA-06 — Price a gather against its row width, not against one streaming number

**Rule.** SparseCore's quoted streaming bandwidth is a figure for a particular access width, not a
single ceiling. A gather whose rows span many whole DMA granules sustains substantially more than
the number quoted for narrow reads. Check the row width before pricing a gather against a streaming
figure — and before concluding, from a roofline built on it, that you have headroom or that you have
none.

**Mechanism.** The earlier saturation figure attributes the ceiling to the **32-byte DMA granule**:
cost is paid per granule, so a read touching part of a granule wastes the rest. Narrow or
partial-granule reads sit at the bad end of that curve. A wide contiguous row is many whole granules
with no partial-granule waste and sits at the good end. The two regimes are the same hardware fact
at different widths, which is why one number cannot serve both.

**Evidence.** Five sightings above the figure, at a **640-byte contiguous row** (20 whole granules).
Four geometries: **602.3 / 605.2 / 535.2 / 509.0 GB/s**, consistent to ±9%; and **709 GB/s** (32768 rows × 640 B in 29.59 µs) at the
shipping gather geometry. The incumbent figure is
**338 GB/s** aggregate per chiplet, which two earlier jobs reproduce at
**337 GB/s** — three independent sightings of the same figure, though
only one of them is the 338 itself. `hardware-facts.md` §10 carries all three, with **~230 GB/s**
attributed to the granule by `w = 16`.

**`hardware-facts.md`'s measured number was deliberately left unamended**, and this entry does not
supersede it. Three targeted jobs outrank a by-product measurement, and the *absolute* figure here
rests on a byte accounting that was never settled — on one reading of the rope row width the same
data reads 1058 GB/s. What survives is the **qualitative** claim, which all five sightings support
at 1.5–2.1× over the incumbent: 338 is not the ceiling at this width. What does not survive is any
specific replacement number.

**Independent corroboration of the qualitative claim, from the other side.** `hardware-facts.md`
§2 (Memory system & bandwidth ceilings) now carries a vendor SparseCore ceiling of **1700 GB/s per core unidirectional**,
3400 per chiplet [unverified: vendor figures, 2026-09]. That puts 338 GB/s at ~10% of the unit's
own limit and every sighting in this entry at 15–21% — so the headroom this entry inferred from a
width sweep is independently implied by the hardware limit, and *nothing* in this area is close
enough to the wall to serve as a roofline. It does not validate any number above; the vendor figure
is itself unverified in this tree, and two unverified quantities agreeing is weaker than either
being measured.

**Scope.** v7x SparseCore indirect reads. The mechanism — granule utilization sets achieved
bandwidth — is general; every number in this entry is not.

**Does not apply to** narrow or partial-granule reads. That is the regime the 338 and 230 GB/s
figures describe, and there they stand unchallenged.
