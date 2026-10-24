---
name: write-tpu-kernels
description: Designs, proves, builds, probes and optimizes custom Pallas/Mosaic TPU kernels for a block of model compute, across prefill, decode_first and decode_last, or returns an evidenced verdict that no kernel is warranted. Design-heavy (70/30 design-to-code) and delivered in two stages — stage A reaches the floor of today's decomposition, stage B moves toward the fully-fused floor — with both floors priced by the roofline-analysis skill. Builds an accuracy test (float64 oracle, native-dtype reference, fault-injection positive control) and a phase-accurate microbenchmark with paged KV-cache inputs first; learns every Pallas API from a cited upstream example via the code-reference skill; profiles the incumbent from compiled artifacts; audits implementation affordances (buffering, async writes, tail tiles, hoisted setup) as well as cost classes; searches for unit trades such as moving VPU work onto the MXU; and gates every performance attribution on a removal arm the way it gates correctness on a fault the oracle catches. Use when creating, tuning or profiling a custom JAX Pallas or Mosaic TPU kernel for attention, MoE, GEMM, norms or a fused region, or when asked to "write a kernel for", "speed up this kernel", "fuse these ops". Not for high-level JAX/Flax code, CPU/GPU-only ops, or general Python debugging.
delegates:
- tpu-diagnostics
scripts:
- scripts/tpu-kernel-development/megakernel_analyzer.py
- scripts/tpu-kernel-integrator/build_plugin.py
- scripts/tpu-kernel-integrator/inplace_harness.py
references:
- references/write-tpu-kernels/
- references/tpu-kernel-development/concepts/megakernel.md
---

# `write-tpu-kernels`

Turn a block of compute into a custom TPU kernel that runs near its hardware floor at its original
precision, in every phase it is called in — or deliver an evidenced verdict that no custom kernel
is warranted.

**Runs on the main thread.** It orchestrates subagents and stops for a human; subagent nesting
depth is 3, so it must not itself be launched inside one.

## What your caller must give you

- **The block** — file and function, or the existing kernel to beat, in a named `vllm-torchtpu`
  checkout on a named branch.
- **The model and the deployment cell** — model, ISL, OSL, concurrency — enough to determine the
  served config. Shapes you derive; a caller-supplied shape is a hypothesis to check.
- **The sharding** — TP/DP/EP degrees and the mesh. The kernel sees one device's slice.
- **The precisions** — storage, movement and accumulation per operand.
- **A configured compute target.** Check it with
  `{{runtime}} compute inspect --json` and record the chip type and chip
  count it reports.

If any of these is missing, {{built-in:ask_user_question}} for it before Step 1. A guessed input produces a confident kernel for a
different problem.

---

## Core rules

**Design**

1. **70/30 design before code, per stage.** Steps 0–3 get most of the effort, and the re-freeze
   that opens stage B is design time too. A wrong dataflow costs ten times more to fix in code than
   on paper. Design-time probes that answer a *fact* — does this lower, what does this primitive
   cost, what did the compiler build — are design, and are encouraged. Timing a candidate is not,
   and waits for the freeze.
2. **Deliver in two stages.** Stage A reaches the floor of today's decomposition. Stage B moves
   toward the floor of full fusion (up to a multi-layer or full-model decode/verify megakernel — see
   {{reference:tpu-kernel-development/concepts/megakernel.md}}, analyzed and scaffolded via
   `{{script:tpu-kernel-development/megakernel_analyzer.py}}` and packaged for `vllm-torchtpu`
   serving via `{{script:tpu-kernel-integrator/build_plugin.py}}` and
   `{{script:tpu-kernel-integrator/inplace_harness.py}}`). Both are designed up front; A ships
   first, and A must not rule out B.
3. **Ask two questions, not one.** *What is this computation and where must it run* — the op DAG,
   the floors, the cost classes. **And** *what does a competent kernel of this shape contain* — the
   implementation affordances. The first finds the right physics; only the second finds a blocking
   write.
4. **Elimination before tuning.** Before padding or retiling a structure, ask whether a native
   primitive, a fold into an existing matmul, or a different unit deletes it.
5. **Every phase, every time.** The kernel serves prefill, `decode_first` and `decode_last`. Floors,
   gates, predictions and results are per phase, and a phase that regresses is a regression.

**Evidence**

6. **A correctness claim needs a fault the oracle catches.** A green oracle is evidence only after
   an injected fault has turned it red on the cell class production runs.
7. **A performance claim needs a removal that moves the clock.** "X costs N" is a hypothesis until
   an arm without X measured about N less. Instruments that agree are independent only if they
   reach the answer by different routes.
8. **Every number carries its origin.** `[measured: <job_id>]`, `[modelled]`, `[fact: <id>]`,
   `[source: path@sha]`, `[assumed]` — in notes too. A roofline is always `[modelled]`.
9. **A null holds only on the axis and configuration it was measured on.** Record both, or it will
   harden into a false constraint.

**Sources and process**

10. **Learn the API from upstream, never from memory.** Pallas and Mosaic APIs move between
    releases, and a remembered signature is the most common source of wasted compiles. Every API
    the kernel uses is copied in shape from a working, cited example found through
    {{skill:code-reference}} — see "Examples before API" in Step 2.
11. **Upstream is an example, not an oracle.** `jax` and `tokamax` kernels target other generations,
    precisions and shapes, and are not optimal for yours. Take the mechanism, the API usage and the
    test structure; re-derive the buffer math.
12. **Zero hardcoded constants or dtypes.** Dtypes from the spec; `pltpu.get_tpu_info()` and
    `pltpu.get_sublane_tiling(dtype)` at runtime.
13. **The main thread never reads raw logs.** Scripts, dry-runs, polling, trace parsing and upstream
    reading happen in subagents that return a compact handover or citations.

---

## Conventions

| | |
| --- | --- |
| **Time delta** | `(new − old) / old × 100`. **Negative is faster.** |
| **Speedup** | `old / new`. `> 1.0×` is faster. |
| **Phases** | `prefill`, `decode_first`, `decode_last`, exactly as {{skill:roofline-analysis}} names them. |
| **Precision** | storage and accumulator dtypes stated separately (e.g. `fp8_e4m3fn` in, `fp32` accumulate, `bf16` out). |
| **Carried rules** | a wiki rule labelled for another chip or unit is `[assumed]` until re-measured. |
| **Upstream citations** | `path:line-line@sha`, as {{skill:code-reference}} returns them. |
| **Frozen text** | probe READMEs and whiteboard versions are never edited after their md5 is recorded; corrections go in the next version. |

---

## Workspace and the wiki

The campaign workspace is `<campaign>` = `{{runs}}/<run_id>/artifacts/write-tpu-kernels/<slug>/`.
The wiki is installed with this skill at {{reference:write-tpu-kernels/}}.

```
<campaign>/                     DISPOSABLE — the campaign
├── WHITEBOARD.v1.md   design; frozen at the gate. Re-freezes add v2, v3 … beside it.
├── STRATEGY.md        run log: every win, rejection and null, as it happens
├── roofline/          {{skill:roofline-analysis}} config, data and report for F1 and F2
├── instruments/       accuracy test + microbenchmark, frozen before Job #1
├── incumbent/         compile-only profile of the incumbent
├── probes/<nn>-<slug>/  probe.py, frozen README.md, <job_id>.txt
├── variants/          candidates, one file each — created only after the freeze gate
├── final/             kernel_patch.diff + final_report.md
└── wiki/              entries promoted at the end, for merging into the wiki

write-tpu-kernels wiki (installed, read-only)   DURABLE — the wiki
├── PROTOCOL.md        read/write protocol — read before touching this tree
├── INDEX.md           one row per entry, and a by-stage list
├── hardware-facts.md  compiler hard limits, layout rules, instruments that lie
├── strategies/        one earned rule per file, each hardware-labelled
└── raw/               verbatim campaign sources the entries cite
```

**Reading.** At each step, open {{reference:write-tpu-kernels/INDEX.md}}, take the rows its by-stage list gives for
that step, and read only those bodies. Check each entry's `hardware:` label against your chip
before using a number; a mismatch means re-measure, not carry. Cite what you use as `[fact: <id>]`.

| step | wiki stage |
| --- | --- |
| 0–1 — context, operating points, instruments | `hardware-facts.md` whole + stage 1 |
| 2 — floors, dataflow, cost classes, affordances | stage 2 |
| 3 — algebra, unit trading, freeze | stage 3 |
| 4 — lowering | stage 4 — the APIs that lie |
| 5 — probing | stage 5 + `hardware-facts.md` "Instruments that lie" — **before** dispatch |

**Constants have one source each.** Throughput numbers — HBM bandwidth, MXU and VPU rates,
SparseCore bandwidth, per-call intercept — come from {{skill:roofline-analysis}} (its TPU hardware
spec), so two floors can never disagree. {{reference:write-tpu-kernels/hardware-facts.md}} owns limits and behaviour: VMEM capacity, tiling
legality, relayout costs, what lowers. If the two disagree on a shared number, stop and report it
rather than picking one.

---

# The flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 0  SYSTEM CONTEXT (paper)                                               │
│ op DAG by class · eliminate by unit capability · 2× bytes per unit boundary  │
│ · denominator T_block/T_step · is the inherited decomposition still right?   │
└───────────────────────────────────────┬──────────────────────────────────────┘
┌───────────────────────────────────────▼──────────────────────────────────────┐
│ STEP 1  OPERATING POINTS, INSTRUMENTS, BASELINE                              │
│ cell → roofline phases (prefill / decode_first / decode_last)                │
│ → accuracy test + microbenchmark (paged KV, ragged), upstream test as model  │
│ → commit → incumbent profile from artifacts → Job #1 per phase               │
└───────────────────────────────────────┬──────────────────────────────────────┘
┌───────────────────────────────────────▼──────────────────────────────────────┐
│ STEP 2  TWO FLOORS, AND THE DESIGN OF BOTH STAGES                            │
│ F1 today's decomposition · F2 full fusion (legality-marked) · per phase gate │
│ · examples before API ({{skill:code-reference}}) · dataflow + working set · primitives │
│ · cost classes with verdicts · affordances declared · stage-B ladder         │
└───────────────────────────────────────┬──────────────────────────────────────┘
┌───────────────────────────────────────▼──────────────────────────────────────┐
│ STEP 3  ALGEBRA, UNIT TRADING, FREEZE GATE                                   │
└───────────────────────────────────────┬──────────────────────────────────────┘
          ┌─────────────────────────────▼────────────────────────────┐
          │ STAGE A  Step 4 lower + affordance audit                 │
          │          Step 5 probe → re-freeze on refutation → deliver│
          └─────────────────────────────┬────────────────────────────┘
                                        │ A delivered, and the fusion prize
                                        │ F1 → F2 is above noise
          ┌─────────────────────────────▼────────────────────────────┐
          │ STAGE B  re-freeze on A's measured numbers (Step 3 again)│
          │          fusion ladder, one boundary per rung:           │
          │          Step 4 → Step 5 → deliver, per rung             │
          └──────────────────────────────────────────────────────────┘
```

---

## Step 0 — System context

**On paper, before Step 1.** Steps 1–5 make a decomposition fast; Step 0 asks whether it is the
right one.

1. **Write the op DAG.** Every op with shapes and dtypes, classified as exactly one of: dense
   contraction · elementwise · reduction · transcendental · data-dependent addressing · layout
   change. The class, not the op's name, is what maps to silicon.
2. **Eliminate by capability.** Most ops have one legal unit. Check each unit's hard limits
   (`hardware-facts.md`) before costing anything — a unit without `dot_general` cannot hold a
   matmul, whatever its bandwidth.
3. **Price every unit boundary as a full HBM round trip** — the producer writes the intermediate,
   the consumer reads it back: `2 × intermediate_bytes`. A placement locally worse on one op that
   deletes a boundary usually wins. Note which boundaries *could* be removed and what would have to
   lower for that; this is the raw material for stage B.
4. **Record the denominator** — `T_block / T_step` per phase — as `measured` (cite the profile),
   `modelled` (show the arithmetic), or `UNKNOWN`. Concurrent units cost `max()`, not `sum()`: an
   op already hidden under another is worth at most `max(0, T_hidden − T_cover)`. Under `UNKNOWN`
   the deliverable is device time only, and establishing the denominator is the top carried item.
5. **Name the assumption behind the inherited decomposition** and check whether anything since —
   the wiki, a new primitive, a toolchain bump — has falsified it.

---

## Step 1 — Operating points, instruments, baseline

Everything later is judged by what this step builds. Build it before designing anything.

### 1.1 Operating points — the roofline's phases, not round numbers

Get the served config for the cell — the model's config and the deployment's serving flags — then
write the {{skill:roofline-analysis}} config for it. Its phase list is the kernel's operating points:

| phase | what the kernel sees | `kv_len` |
| --- | --- | --- |
| `prefill` | a chunk of `mnbt` tokens | the context before the chunk |
| `decode_first` | `batch` requests, one token each | ISL |
| `decode_last` | the same | ISL + OSL |

Take the **per-device** tokens, batch and `kv_len` the roofline prints after sharding — the kernel
runs on one device's slice. Record them in `WHITEBOARD.v1.md` with the exact dtypes and the
serving flags that shape them (page size, max batch, cache precision, bucket ladders).

### 1.2 Start from an upstream test, not a blank file

Before writing either instrument, dispatch {{skill:code-reference}} for the upstream kernel
nearest this block and **the unit test and benchmark that prove it** — e.g. the paged or ragged
paged attention tests for a decode attention kernel, the megablox / ragged-dot tests for a grouped
GEMM. Borrow their structure, not their shapes:

- **the tokamax per-op convention** — a shared correctness suite written before any backend
  exists, which the incumbent and every candidate both run; and one shape/dtype spec that the test
  and the benchmark both consume, so they can never disagree on shapes. Here that spec is the three
  phases.
- **how upstream builds realistic inputs** — page tables, ragged lengths, masks — and its numerical
  comparison helpers.

### 1.3 The accuracy test — `<campaign>/instruments/test_accuracy.py`

Three parts, run on every phase cell plus edge cells:

| part | role |
| --- | --- |
| **math oracle** | `float64` (NumPy, or JAX `Precision.HIGHEST`). Separates algorithmic drift from rounding when the math is reordered. |
| **native reference** | the op in its original dtypes with its real accumulator — calibrates `atol` / `rtol`. |
| **fault control** | deliberately broken variants the test must reject — one per fault class you are guarding (wrong mask, off-by-one page, stale accumulator, dropped tail) — with the catch rate reported **per cell class**. |

**Edge cells** belong here because prefill and decode break differently:

- extents that are not a multiple of the block size;
- `kv_len` of 0, 1, and one either side of a page boundary;
- a one-token prefill chunk;
- the maximum batch.

A fault class that is blind on the cell class production runs means the gate has not been built,
however many cells pass.

### 1.4 The microbenchmark — `<campaign>/instruments/bench.py`

Its cells are the three phases, exactly, so every measured number joins a modelled one:

- **KV cache, when the kernel reads one** — paged, at production page size and KV dtype, filled to
  the phase's `kv_len`, with block tables drawn from a **shuffled** page pool. Contiguous pages
  flatter the DMA pattern and measure a kernel production never runs.
- **Prefill** runs as a chunk with its prior context already in the cache, not as a fresh prompt.
- **Ragged decode** is one extra cell, with request lengths spread across
  `[decode_first, decode_last]`. The uniform cells stay, since they are what the roofline prices.
- **Kernels without KV** (GEMM, MoE, norms): the cells reduce to token counts — `mnbt` for
  prefill, `batch` for decode.
- **Value-dependent cost** — routing skew, top-k, masks — uses production's distribution, not
  uniform random.
- **Reporting:** per rep, per cell: time, compile seconds, and the median with rep 0 held out.
  Obey M1 and M2 (Step 5) from the first version.

**Freeze `instruments/` before Job #1** — commit it if `<campaign>` is under version control, and
record each file's md5 in the whiteboard. From here on the evaluation criteria are read-only.

### 1.5 The incumbent's profile — from artifacts, not source

One compile-only job (plus a short run if needed) records what the compiler actually built. The
profile needs this campaign's inputs and harness, so capture the trace yourself and send it to
{{agent:tpu-diagnostics}} to read:

1. Capture: write `<campaign>/incumbent/capture.py`. It sets
   `LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"` in `os.environ` before it imports
   jax, builds the incumbent exactly as `<campaign>/instruments/bench.py` builds it, warms it up outside the
   trace, runs one input size under `jax.profiler.trace`, and packs the trace directory into one
   archive. Run it:
   `{{runtime}} compute run --code <campaign>/incumbent/capture.py --input <incumbent file> --output incumbent_trace.tar.gz:<campaign>/incumbent/incumbent_trace.tar.gz`.
   Unpack the archive under `<campaign>/incumbent/trace/`, then delete
   `<campaign>/incumbent/incumbent_trace.tar.gz`: the unpacked trace is what gets read.
2. Read: send {{agent:tpu-diagnostics}} a `profiling-request` (see {{schema:profiling-request.json}}):

   ```json
   {"prompt": "Read the compile and trace data of the incumbent kernel <name>. Report what the compiler built.", "run_id": "<run_id>", "traces": [{"path": "<campaign>/incumbent/trace", "label": "incumbent"}], "report_path": "<campaign>/incumbent/tpu_profiling_report.md", "fields": ["program_or_scope_name", "tile_parameters", "grid", "vmem_allocations", "buffer_counts", "tpu_custom_call_count_per_call", "xla_ops_in_jitted_region"]}
   ```

3. Read the returned `profiling-result`. Proceed only on `status: reported`. Its report gives:

- program or scope name, and the tile parameters its name carries;
- grid, VMEM allocations and buffer counts;
- `tpu_custom_call` count per call;
- the XLA ops inside the jitted region around it.

Identify the incumbent by program id or by the shape string in its own name, never by a phase
label or a directory name — both are metadata and both can be wrong. This costs one job and it
often explains the incumbent's speed before any design starts. Write it to `incumbent/` and
summarize it in the whiteboard.

**Capture cheaply.** Default to the trace-side capture above, which needs no dump files.
Use one input size. Use the dump-side capture (`--xla_jf_dump_to`, `--xla_mosaic_dump_to`) only
for a question the trace cannot answer, and name that question in the dispatch. When you dump:

- write to the target's local disk (e.g. `/tmp`), never to a directory that is staged back whole,
  because every staged byte is a transfer;
- copy back only the kernel's own final-pass files, packed into one archive named by `--output`.
  The compiler writes the whole program after every pass and dumps every other op in the program
  too;
- stop the job if the copied set exceeds 1 GB.

A dump written to the output directory for four input sizes once produced 22 GB, about 99% of
it intermediate passes and other ops, and spent ~40 of the job's 49 minutes uploading.

### 1.6 Job #1 — the baseline, per phase

Run the incumbent — the file production runs, verbatim, md5-recorded — at every cell. Print the
**XLA op breakdown of the measured region**: that is the template every candidate's region is
compared against (M4). Record per-phase times as `[measured: <job_id>]`.

---

## Step 2 — Two floors, and the design of both stages

### 2.1 Two floors via {{skill:roofline-analysis}}

Dispatch {{skill:roofline-analysis}} with the model, precisions, sharding, slice, the three
phases, and the block as `file:function` with its op DAG from Step 0. Ask for two numbers per phase:

| | **F1 — today's decomposition** | **F2 — full fusion** |
| --- | --- | --- |
| boundaries | today's op and kernel boundaries kept | none inside the block |
| HBM | every intermediate written and read back | the block's external inputs, weights, KV and outputs only |
| time | Σ over ops of `max(HBM, compute)` — Path A per op | `max(total HBM / BW, total compute)` — DMA hidden under compute |
| its gap | **baseline → F1 is tuning headroom** | **F1 → F2 is the fusion prize** |

Include the per-call intercept and any collective on top, as the roofline prices them. Keep the
analyzer's per-op binding resource; Step 3 reads it.

**F2 is a target only if fusion is legal.** If Step 0 put two ops on different units with no
handoff that lowers without HBM, F2 is a `projection` — report it as carried, and do not plan a
stage B toward it.

Both floors are `[modelled]`. Step 5 replaces their inputs with measured ones as soon as a
measurement contradicts them.

### 2.2 The gate, per phase

| per-phase condition | action |
| --- | --- |
| baseline ≤ 1.25 × F1, and the fusion prize is under noise or F2 is a projection | **stop at floor** — `stopped at floor` or `baseline optimal`, through {{skill:exit-review}} |
| baseline ≤ 1.25 × F1, fusion prize above noise | skip stage A for this phase; stage B |
| baseline > 1.25 × F1 | stage A; plan stage B if the prize is above noise |
| bytes already at the theoretical minimum | **gate inapplicable** — say so. The levers left are rows, descriptors and events; rank them with the fixed-cost fit (cost class 6) |

"Above noise" is absolute: `(F1 − F2) × calls per step` against the benchmark's own spread, and
against the denominator. A prize worth 0.1% of step time is not worth a stage.

### 2.3 Examples before API — {{skill:code-reference}}

Agents write plausible-looking Pallas that does not compile, or compiles into something other than
what was meant. The fix is to design and write from working examples. Upstream `jax` carries the
largest body of TPU Pallas tests that exists, and `jax`'s shipped TPU kernels and tokamax's op
directories are the best worked examples of complete kernels. Dispatch {{skill:code-reference}} —
it returns `path:line-line@sha` citations and keeps the checkouts out of your context — for three
things:

1. **Design exemplars** — the shipped kernels nearest this block in shape class (paged or ragged
   attention, grouped GEMM, splash attention, ragged gather and scatter, SparseCore kernels). Read
   them for *mechanism*: how they pipeline, tile, handle ragged tails, lay out scratch, split the
   grid. Then ask of each: what generation, dtype and shape was it built for, and which of its
   choices were forced by a limitation your target does not share?
2. **One working example per API.** For every Pallas/Mosaic API the design will use —
   `pallas_call` with its grid spec, `BlockSpec` and index maps, `PrefetchScalarGridSpec`,
   `emit_pipeline`, `make_async_copy` and semaphores, memory spaces, `pl.when`, scratch shapes,
   SparseCore primitives — the upstream test that exercises it in the form you need.
3. **Lowering questions** — "does this `jnp` op lower inside a TPU kernel", "what does this error
   mean" — answered by the test that covers it, before any job is spent finding out.

Record the result in the whiteboard as an **API ledger**: one row per API the design uses, with
its citation and the exact form (argument names, shapes, memory space) the kernel will copy. An API
with no row is written from memory, and goes back to {{skill:code-reference}} before Step 4.

### 2.4 Dataflow, working set, primitives

- **Pipeline diagram** — DMA in, compute and DMA out across steps T−1, T, T+1, with the buffer each
  stage uses. **Name the pipelining axis.**
- **Working set** — `Buffer_Count × Σ(tile elements × sizeof(dtype)) + scratch ≤ VMEM limit`, per
  phase, arithmetic shown. Declared scratch that does not fit may fail to compile rather than spill,
  so treat this as a hard gate.
- **Accumulators** — dtype and residency stated.
- **Primitive audit** — before designing any data structure, list the native primitives the target
  unit offers (`hardware-facts.md`, and the exemplars from 2.3): scans, cross-lane reductions,
  masked loads, indirect DMA, packing intrinsics. Before copying an idiom from an exemplar, ask which
  primitive the original was avoiding and whether your unit has the same hole.
- **Structural deletion test** — answer in the whiteboard: *which intermediate buffers, tables or
  scratch structures can be removed outright* by a native primitive, rather than made cheaper?

### 2.5 Cost classes — each with a critical-path verdict

Fill every row. A row's status is either:

```
examined: <work> → <critical-path contribution>  [by: ablation | overlap measurement | argued]
UNEXAMINED: <why>
```

A work count is not a verdict. Inside a pipelined kernel, work concurrent with a longer transfer
contributes `max(0, T_work − T_cover)`, often zero. An `argued` verdict is allowed at design time and
is `[modelled]`; it must be backed by an ablation or overlap measurement before it ranks a lever or
appears in a report as an attribution (M8).

| # | cost class | instrument that reveals the work |
| --- | --- | --- |
| 1 | structural elimination | the primitive audit and deletion test above |
| 2 | bulk memory traffic | F1/F2 bytes against measured time |
| 3 | per-element ALU in the innermost body | the body's op list written out by hand × trip count |
| 4 | cross-lane serializing ops | a per-chunk cost model, not the trip count |
| 5 | bank / port conflicts | latency that moves with input *distribution* at constant op count |
| 6 | launch / fixed overhead | the intercept of an M-sweep (M2); a large intercept caps any work-deleting lever |
| 7 | boundary and layout handoff | shapes and pads at the wrapper; `2 ×` bytes per boundary |
| 8 | inter-unit concurrency | can the block overlap other work at all — `max()`, not `sum()` |

Rows 1, 3, 4 and 5 have no natural profiler view — a profile shows where time goes in the code you
wrote, never the lever you did not write. That is why the list is written, and why **the
unexamined rows are a deliverable.**

### 2.6 Implementation affordances — declared now, audited in Step 4

The cost classes ask where time could go. This list asks what a competent kernel of this shape
contains. Score every row `planned`, `NA: <reason>` or `absent: <reason>`. You may add rows; you may
not remove them. The exemplars from 2.3 show how upstream implements most of these — cite them in
the API ledger.

| affordance | how Step 4 checks it in the code or artifact |
| --- | --- |
| every critical-path HBM→VMEM transfer is double- or multi-buffered | BlockSpec / `emit_pipeline` buffer count ≥ 2, or the ring depth of a manual loop |
| every VMEM→HBM write is async, and waited on as late as possible | no `.wait()` directly after its own `.start()` |
| tail and ragged tiles run a narrower body | a distinct path or masked shape for the last grid step |
| zero-trip and common-case fast paths | a branch ahead of loop setup |
| no host-computable work inside the jit | the region's XLA breakdown has only the custom call(s) plus declared ops |
| per-call setup (plans, index tables, constants) hoisted | built outside the jitted callable, or cached |
| the epilogue is chunked or overlapped with the next load | it does not serialize after the last DMA |
| the accumulator stays resident across the contraction loop | no HBM round trip of partial sums |
| grid dimension semantics permit the core split | `parallel` on independent axes |
| data-dependent indices arrive by scalar prefetch | not loaded inside the kernel body |

### 2.7 Design both stages

- **Stage A** — designed in full: the per-op or per-kernel changes that reach F1, with the
  boundaries kept.
- **Stage B** — sketched as a **ladder of boundary removals**, ranked by bytes saved, each rung with
  its own modelled floor. Full fusion is where the ladder ends, not the first rung built. Ask
  {{skill:code-reference}} how upstream fuses across the same boundary (a matmul with its epilogue, a
  gather into its consumer) before sketching a rung.
- **Does A rule out B?** Check every stage-A choice that fixes an interface: output layout and dtype,
  tiling axes, grid order. If the fused kernel could not consume it, change A now.

---

## Step 3 — Algebra, unit trading, and the freeze gate

### 3.1 Algebraic rewrites

Standing techniques, not measured rules — prove each against the `float64` oracle before relying
on it:

- concatenating narrow contractions into one wide one so systolic lanes do not idle;
- factoring shared terms (`Q·A + Q·(K·B) → Q·(A + K·B)`);
- online accumulation (Welford- or flash-style) instead of materialized intermediates.

### 3.2 Unit trading — a search, not a list

Moving work between units — above all VPU work onto an idle MXU — is one of the largest and least
obvious levers, and no profile suggests it: a profile shows a busy VPU, never the matrix that would
empty it. So it is searched for, on this kernel, every time.

**The thinking tool: split each op into its linear and non-linear parts.** Any linear map is a
matmul by some constant matrix and can run on the MXU; the non-linear part stays. Few ops are
purely non-linear — a softmax is an `exp` plus a sum, a norm is a square, a sum and a scale — so the
question is not "is there a known trick for this op" but "what is this op's linear part, and what
matrix expresses it". Try the **folds** first: work absorbed into a matmul that already exists — a
scale folded into a weight, a transpose into `dot_general` dimension numbers — deletes a pass
without adding MXU work.

**The decision rule — four costs, per phase:**

1. **Padding.** A thin matmul (a reduction to width 1) is padded to full MXU width; price the
   padded FLOPs with {{skill:roofline-analysis}}, not the useful ones.
2. **Critical path and slack.** It pays only if the VPU pass is on the critical path (M8) *and* the
   MXU has idle capacity in that phase. Decode usually has MXU slack; prefill often binds on the
   MXU. Expect a trade to win in one phase and lose in another, and allow per-phase dispatch.
3. **Precision.** An fp32 reduction through a bf16 MXU pass loses bits; `Precision.HIGHEST` or a
   hi/lo split costs passes. The oracle decides.
4. **The constant matrix.** It needs VMEM and a source — built once and hoisted, or generated per
   tile, which is VPU work of its own.

The same rule runs in reverse — at tiny M, a matmul can lose to the VPU — and covers the other
trades: a transpose into dimension numbers, a gather onto a SparseCore, a small matmul off the MXU.

**The search.** Every VPU or XLU pass the cost-class verdicts put on the critical path, and every op
whose unit is still a choice after Step 0, gets a row in the whiteboard's **unit-trading table**:

| op | linear part | candidate formulation | padded cost per phase | slack per phase | precision plan | matrix source | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |

Every row ends in one of: a candidate, "linear part too small to pay" with the number, or "no
formulation found" with what was searched. A row may not be left out.

Run the search in a **research subagent**, so the main thread keeps its context. Give it the op list,
the phase shapes, the linear/non-linear split, and the decision rule. It searches with
{{skill:code-reference}} (how jax and tokamax kernels express the same ops — especially a `dot` or `einsum` against a
constant, or work folded into dimension numbers), the wiki's stage-3 rows, and first principles,
and returns candidates with citations, not code. A candidate that survives the pricing becomes a
probe arm, alongside a removal arm showing the pass it replaces is on the critical path.

### 3.3 The freeze gate — stop before coding

No file appears under `variants/` until every item is checked in the current whiteboard version.
Then record the whiteboard's md5; from here it is frozen (see the conventions).

| # | item | what proves it |
| --- | --- | --- |
| | **Computation** | |
| 1 | operating points locked | three phases, per-device shapes, exact dtypes, from the roofline config |
| 2 | floors per phase | F1 and F2, with F2's legality marked |
| 3 | primitives audited, deletion evaluated | the inventory, and why each surviving buffer survives |
| 4 | algebra proved | every rewrite checked against the `float64` oracle |
| 5 | unit-trading table complete | every row ends in a verdict |
| | **Implementation** | |
| 6 | incumbent profile recorded | program name, tiles, VMEM, buffer counts, from artifacts |
| 7 | API ledger complete | every API the design uses has a cited upstream example in the form it will be used |
| 8 | affordances declared | every row scored |
| 9 | working set proven, per phase | exact bytes ≤ VMEM limit, arithmetic shown |
| 10 | pipeline axis named | which axis, and what overlaps on it |
| 11 | stage A does not rule out stage B | the interface check from 2.7 (stage-A gate only) |
| | **Evidence plan** | |
| 12 | cost classes filled | every row a verdict with its instrument, or `UNEXAMINED: <why>` |
| 13 | fault control designed | the faults, and which cell classes discriminate each, per phase |
| 14 | predictions written | signed, falsifiable, per lever, per phase (M5) |
| 15 | sweep drivers computed | the work driver at every cell of every planned sweep (M6) |
| 16 | denominator recorded | `measured` / `modelled` / `UNKNOWN`, plus the overlap bound |

---

## Step 4 — Lowering, then the affordance audit

1. **Translate** the frozen design into `variants/<name>.py`, copying each API's form from its
   ledger row. Keep the wrapper (grid, block sizes, config) and the kernel body in separate
   functions, as upstream does, so a second variant can reuse the wrapper. Earlier variants stay
   importable beside the new one — an edit in place leaves no baseline.
2. **Interpreter dry-run** — `pltpu.InterpretParams()` on CPU catches shape mismatches, bad indexing
   and unaligned slices at no TPU cost.
3. **A compile or lowering error goes to {{skill:code-reference}} first.** Ask for the upstream test that
   exercises the failing API or op, and diff your usage against it, before trying variations. A
   rejected construct is a fact about your usage until an upstream example shows it failing too.
4. **Accuracy test** — the candidate passes at every phase cell **and** every injected fault fails.
5. **Affordance audit** — check the written code and its compiled artifact against every row of
   2.6, with the check that row names. An affordance declared `planned` and absent in code is a bug.
   This is where a slip in writing gets caught, which a design-time checklist cannot catch.

---

## Step 5 — Probe, measure, deliver

### 5.1 Arms

One job, one arm per chip, one frozen question in `probes/<nn>-<slug>/README.md`. Read the chip
count from `{{runtime}} compute inspect --json`; a four-chip target runs
four arms. The probe's recipe `run.sh` runs `python -u probe.py` and writes the result file. Submit
it with the incumbent file, the candidate files and the instruments as inputs:
`{{runtime}} compute submit --recipe <campaign>/probes/<nn>-<slug>/run.sh --input <campaign>/probes/<nn>-<slug>/probe.py --input <campaign>/probes/<nn>-<slug>/README.md --input <incumbent file> --input <campaign>/variants/<name>.py --output <result file>`,
and save `{{runtime}} compute job logs --job-id <job_id>` as `probes/<nn>-<slug>/<job_id>.txt`.

- **Arm 0** is always the incumbent, verbatim (M4).
- The other arms are levers from different cost classes, or removal arms backing an attribution
  (M8). A one-parameter sweep is allowed only when the question *is* a knee, and M6 holds.

### 5.2 The measurement rules

**Clock rules — is this number a number?**

**M1 — Build every callable before the timing loop.** A callable built inside the repeat loop
recompiles every rep and times the compiler. JAX caches on the identity of the wrapped callable,
and `functools.partial` has no value equality — so hoist the function object, not just the `jit`.
Compile every `(arm, cell)` into a dict first, then time by indexing it. Print per-unit compile
seconds; budget the compile count; hold rep 0 out of the median. A late rep costing what rep 0 cost
means that arm is recompiling.

**M2 — The chained clock, and the hoist check that guards it.** A chained-invocation clock must
barrier the **loop carry**, never the loop-invariant operands; otherwise XLA hoists the call out of
the loop, M invocations run once, and — because a hoisted loop still returns the right answer — no
correctness check catches it. Three defences, all required:

1. assert the HLO's `tpu_custom_call` count equals the expected multiple of M, per unit;
2. take per-call cost from an M-sweep least-squares slope, never divide-by-M (the intercept is cost
   class 6);
3. treat a raw `T(M_hi) / T(1)` near 1, or any reading under the floor, as a hoist.

Never stack heavy cell groups on one chip — contention can look like a hoist.

**M3 — Dispatch, never block.** Submit with `{{runtime}} compute submit` and wait with
`{{runtime}} compute job wait` in the background; never `sleep` in the foreground. Submit every independent job before harvesting any — but a dependent job waits: a
correctness gate that has not returned has not passed. Poll around the job's real ETA, and fill the
wait with work decided before dispatch.

**Claim rules — does the number mean what the report says?**

**M4 — Same artifact, same job, same surrounding work.** Arm 0 is the file production runs,
byte-identical and md5-gated, never a rebuild; incumbent and candidate run in the same job,
interleaved on the same chips. **Both measured regions contain the same non-kernel work**: print
both XLA op breakdowns and diff them. A non-kernel op in one region and not the other fails the job
unless it is accounted for and moved into both.

**M5 — Freeze predictions before dispatch.** Signed, falsifiable, per lever and per phase, plus
the abort condition that would make the probe void rather than informative. Record the README's md5
before submit and re-check it on the target. A null is trustworthy only if the prediction is visibly
older than the data.

**M6 — The sweep must move its driver.** Before dispatch, compute the work driver — grid steps,
tiles, bytes moved, DMA count — at every cell, from the same code that builds the kernel. Print it,
and assert it changes in the direction the prediction assumes. A driver that is flat across the
sweep voids the sweep before it costs a job.

**M7 — Correctness: show the oracle can fail.** Report the fault control's catch rate per cell
class with every correctness claim. A class blind on production's cells is named as blind, and is
not a pass.

**M8 — Performance: show the cost can be removed.** The counterpart of M7.

- *"X costs N µs"* needs a removal arm — the kernel without X, allowed to give wrong numbers and
  labelled timing-only — whose delta is about N, at that phase.
- *A gap split across several parts* is a sum of direct per-part arms, published against an
  independently measured whole; the gap between them, in either direction, is **unattributed**.
  One-at-a-time ablation deltas neither sum to the whole nor bound it when the parts contend for
  the same slots, and a residual by subtraction is a remainder, not a mechanism.
- *Instruments that agree* count as independent only if they reach the answer by different routes;
  a static dump and a hardware counter reading the same quantity count once.
- Until an attribution is backed this way it is `[modelled]`: it may suggest a probe, but it may not
  justify a redesign or appear in a report as a result.

**M9 — Measure at the scope the work lives, and compose with a ladder.** A lever that deletes work
*between* stages reads as zero on either stage's clock. Ship on whole-call time per phase against
the incumbent, never on a kernel-scope roofline fraction. Arms sharing a baseline cannot measure
interaction, so confirm a stack with a forward ladder (each rung = previous + one lever) plus at
least one leave-one-out. Disagreement over ~3 pp means the levers interact — say so rather than
publishing per-lever numbers. A residual over 3% between the ladder's top and the full candidate
voids the attribution. No phase may regress.

### 5.3 Re-freeze when the design's numbers are refuted

**Trigger:** a measurement lands outside the frozen range of a modelled number the lever ranking
depends on — a floor, a byte count, a regime split, an attribution.

**Action, before the next probe:**

1. Write `WHITEBOARD.v<n+1>.md` beside the untouched `v<n>`.
2. Open it with a modelled → measured table.
3. Recompute the floors and the lever ranking.
4. Re-run the freeze gate on what changed.
5. Dispatch {{skill:critical-review}} on the new version's attributions, with the raw artifacts behind
   them.

Predictions made under `v<n>` are still scored against `v<n>`. Building on a lever the last probe
refuted is the failure this exists to stop. **Stage B always opens with a re-freeze** on stage A's
measured numbers.

### 5.4 Routing from the profiling result

Capture the probe's trace inside the probe job, as in 1.5, and send it to {{agent:tpu-diagnostics}} in
a `profiling-request` whose `fields` name every signal in the table below (XLU transpose and
relayout time, VPU utilization, DMA wait, spills, time per phase). Its `profiling-result` carries
`levers` ranked by share, and the report at `report_path` gives the numbers. Route on what binds:

| signal | route |
| --- | --- |
| XLU transposes / relayouts | Step 3 — layout, scan order, or a unit trade into dimension numbers |
| VPU saturated, on the critical path | Step 3.2 — unit trading |
| DMA wait | Step 2.6 — buffering depth, async writes, the pipelining axis |
| spills | tile knee, in Step 5 |
| ratio to floor **widens** as work is removed | the floor models the wrong resource — fixed cost binds. Go back to Step 0 or cost class 6, not another sweep |

Cap each question at 2–3 probes. Not converging means the whiteboard is wrong, not the parameters.

### 5.5 Deliver a stage

A stage — and each stage-B rung — is delivered when it has:

- a whole-call, per-phase result against the incumbent in one job, with replicates;
- a correctness job with the fault control;
- every attribution backed per M8;
- a {{skill:critical-review}} verdict, dispatched with job ids, artifact paths and the exact
  command or recipe each job ran — the reviewer recomputes, it does not read prose.

Then apply the stage-B gate: A delivered, and the fusion prize above noise against the updated F1.

---

## Subagent handover

Subagents write `probes/<nn>-<slug>/{probe.py, README.md, <job_id>.txt}` and return **only** this.
Every field in angle brackets is filled from the job; no number is ever copied from a template.

```yaml
subagent_handover:
  probe_id: "<slug>"
  question: "<the frozen question>"
  prediction_freeze: { readme_md5: "<md5>", frozen_at: "<iso8601, before submit>" }
  job_id: "<job_id>"
  status: "SUCCESS | FAILED_ALIGNMENT | COMPILER_REJECT | TIMEOUT | VOID"
  clock_check:            # M1/M2
    custom_call_count: "<expected vs observed, units checked>"
    m_sweep_slope_used: <true|false>
    hoist_ratio: "<T(M_hi)/T(1)>"
  region_symmetry: "<non-kernel op diff between arm 0 and candidates: none | listed>"   # M4
  driver_check: "<driver per cell, direction asserted: pass | VOID>"                   # M6
  arms:
    - name: "<arm>"
      per_phase: { prefill: "<us, delta>", decode_first: "<us, delta>", decode_last: "<us, delta>" }
      oracle: "<PASS | FAIL>"
  fault_control: "<caught n/N per cell class; blind classes named>"                    # M7
  attributions:                                                                        # M8
    - { claim: "<X costs N us at phase>", backing: "<removal arm id | UNBACKED>" }
  profiling_summary: "<what binds, per phase, from the tpu-diagnostics report at report_path>"
  answer: "<one sentence answering the frozen question>"
  predictions_scored: "<each prediction: within range | refuted, with the number>"
  refreeze_triggered: "<no | which modelled number, modelled vs measured>"
  route_action: "Adopt | Re-freeze | Step 0 | Step 2 | Step 3 | Step 5"
  distilled_strategy: |
    lever: <cost class or affordance>; hardware: <chip>; phase: <phases>
    result: <delta> [measured: <job_id>]; axis/config: <required for a null>
    mechanism: <why the silicon did this>; scope: <what it does not apply to>
```

The main thread updates the whiteboard (or opens the next version), copies `distilled_strategy`
into `STRATEGY.md` **including when the arm lost**, and moves on.

---

## Finishing: promote, report, review

### 1. Promote `STRATEGY.md` into the wiki

Follow {{reference:write-tpu-kernels/PROTOCOL.md}}. The filter is one question: *would this change what someone
optimizing a different kernel does next month?* A win with no mechanism fails it; a measured null
with a mechanism and its axis passes. Also sweep the final report's rejections, refuted predictions,
correctness findings, unit-trading verdicts, API-ledger surprises (an upstream example that did not
work on this pin) and carried items — they hold as many promotable facts as `STRATEGY.md` does.
Copy sources into `raw/` first, verbatim; add one `INDEX.md` row per entry. The installed wiki is
not edited in place: write the new `raw/` files, `strategies/` entries and `INDEX.md` rows under
`<campaign>/wiki/`, with the same layout, and list them in the summary for the package maintainer to
merge.

### 2. Write `<campaign>/final/final_report.md` and return the summary

```yaml
write_tpu_kernels_summary:
  block: "<file:function @ per-device shapes, dtypes (in/acc/out), sharding>"
  hardware_target: "<chip>"
  status: "completed | stopped at floor | baseline optimal | blocked"
  denominator: { t_block_over_t_step: "<per phase, tagged | UNKNOWN>", overlap_bound: "<us | not overlappable>" }
  floors:                               # [modelled], from {{skill:roofline-analysis}}
    prefill:      { F1_us: "<>", F2_us: "<>", F2_legality: "legal | projection: <why>", binds: "<>" }
    decode_first: { F1_us: "<>", F2_us: "<>", F2_legality: "<>", binds: "<>" }
    decode_last:  { F1_us: "<>", F2_us: "<>", F2_legality: "<>", binds: "<>" }
  stages:
    A: { status: "delivered | skipped: <why> | not reached", job_id: "<job_id>", critical_review: "<verdict>" }
    B: { status: "<rungs delivered> | not entered: <why>", job_id: "<job_id>", critical_review: "<verdict>" }
  result:                               # whole-call, per phase, against the verbatim incumbent, one job
    prefill:       { baseline_us: "<>", candidate_us: "<>", speedup: "<>" }
    decode_first:  { baseline_us: "<>", candidate_us: "<>", speedup: "<>" }
    decode_last:   { baseline_us: "<>", candidate_us: "<>", speedup: "<>" }
    ragged_decode: { baseline_us: "<>", candidate_us: "<>", speedup: "<>" }
    replicates: "<n draws, agreeing to x%>"
  correctness: { oracles: "float64 + native", fault_control: "<catch rates; blind classes>", job_id: "<job_id>" }
  levers:
    - { class: "<cost class | affordance | unit trade>", change: "<>", mechanism: "<>", delta_per_phase: "<>", backing: "<arm id>" }
  rejected:
    - { lever: "<>", verdict: "<null on the <named> axis/config>", job_id: "<job_id>" }
  cost_class_coverage:
    examined: <count>
    unexamined: ["<class>: <why>"]      # [] only if every row was instrumented
  affordances_absent: ["<affordance>: <why>"]
  unit_trading: { tried: ["<op>: <verdict>"], no_formulation: ["<op>: <what was searched>"] }
  refreezes: ["<v2: what was refuted>"]
  carried:                              # results that are not kernel changes, each with a route
    - "<finding> -> <owner or skill>"
  strategies_promoted:
    - { id: "<id>", hardware: "<chip>", generality: "<narrow | unit-wide | general>", path: "<campaign>/wiki/strategies/<file>" }
  final_diff: "<campaign>/final/kernel_patch.diff"
```

`cost_class_coverage.unexamined`, `affordances_absent` and `carried` are not optional. The first two
stop the next campaign believing this one looked everywhere; the third is how findings that are not
kernel changes reach the person who owns them. A candidate headed for `accepted` still owes an
end-to-end {{skill:accuracy-check}} — list it under `carried` for the caller.

### 3. Reviews before returning

| status | before returning |
| --- | --- |
| `completed` | **{{skill:critical-review}}, always**, on `final_report.md` with the raw artifacts: job ids, file paths, the recipe read-back. The main thread does not review its own claims. |
| `stopped at floor` \| `baseline optimal` | a `floor`-class exit through {{skill:exit-review}} — always if `cost_class_coverage.unexamined` or `affordances_absent` is non-empty. Give it the whiteboard versions, both floors and the measured values; a floor computed from the wrong quantity stops the search authoritatively. |
| `blocked` | **always** {{skill:exit-review}}, with the verbatim failure, a minimal reproduction, the repairs tried, and the upstream examples the failing usage was diffed against. A compiler that rejected the kernel is a fact about the attempt, not about the kernel. |

If the verdict is `EXIT PREMATURE`, run its top continuation before returning. The design budget
exists because a second design costs less than the first.
