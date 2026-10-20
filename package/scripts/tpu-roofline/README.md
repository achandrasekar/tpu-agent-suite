# `scripts/` — the roofline script

```
roofline.py                 the arithmetic. One function per op. Not regenerated per model.
configs/<model>-<cell>.json data only: shapes, numerics, hardware, sharding, phases, workload.

step_api.py                 the step-cost contract both step models fill
roofline_model1.py          NATIVE: prefill and decode as separate passes   <- DEFAULT
roofline_model2.py          MIXED:  one fused vLLM step
serving.py                  admission gates, queueing, Little's Law. Model-agnostic.
derivation.py               one operating point written out step by step (model 1),
                            each step checked against serving.operating_point
```

```bash
python3 scripts/roofline.py --selftest                       # always first
python3 scripts/roofline.py scripts/configs/<cfg>.json --csv out.csv
python3 scripts/roofline.py <cfg>.json --layer-csv layers.csv   # per-layer op tables
python3 scripts/roofline.py <cfg>.json --no-layers              # ...or suppress them

python3 scripts/roofline_model1.py                           # 6 bars
python3 scripts/roofline_model2.py                           # 15 bars
python3 scripts/serving.py --selftest                        # 12 bars
python3 scripts/serving.py configs/<cfg>.json --concurrency 256 --both
python3 scripts/derivation.py --selftest                     # every step vs the sweep
python3 scripts/derivation.py configs/<cfg>.json --concurrency 256 --mnbt 1024
```

`roofline.weight_terms` / `roofline.kv_terms` are the per-tensor split behind
`weight_bytes` / `kv_bytes_per_token` — the aggregates are built by summing them, so a
report that prints the terms cannot disagree with the admission number.

**`model/common/<op>/<op>.md` is the specification; this file implements it.** Each op is a named
function whose comment cites its spec section, so the two can be diffed. Adding a model is writing
a config. Adding an *op* means adding a function here, behind a selftest — never a private copy of
the script.

---

## 0. Two step models, and why there are two

`roofline.py` answers *what does this shape cost*. Turning per-op times into a **step**, and a step
into TTFT / TPOT / throughput, needs one more decision — **what a step is** — and that decision is
not settled in this repo. So it is a fork rather than a default, and both sides are reported.

| | **model 1 — native, DEFAULT** | **model 2 — mixed** |
| --- | --- | --- |
| the step | prefill and decode are separate dedicated passes; their *rates* combine reciprocally | ONE fused pass: `d` decode tokens at full context **plus** a prefill chunk, sharing one token budget, weights read once |
| TTFT | the dedicated prefill run | a sum over the steps a prompt takes while decode runs alongside |
| `mnbt` acts on | the prefill run's chunk count. **`T_step` is exactly invariant to it** | the shared budget, so `T_step` — and therefore TPOT and ITL — move too |
| `prefill_schedule_interval` | no term at all | a TTFT throttle |
| PP | `run_phase`'s mean stage — no fill, no balance, no send | `T_slow` / `T_round`, `k−1` p2p sends, `eta_bal`, per `parallelism/pp/pp.md` |
| extra assumptions | none beyond this file | the steady-state token ratio `p/d = ISL/OSL`, and the occupancy closure `running = d + q` |
| implements | §1 below, unchanged | §1 with the composition replaced |

**Model 1 is the default** because it is this file's own composition; promoting model 2 would make
every downstream report depend on a modelling choice nobody has checked against hardware. Model 2 is
closer to what vLLM's scheduler actually builds (`v1/core/sched/scheduler.py` fills the token budget
with decode first, then chunked prefill) but buys that with two assumptions model 1 does not need.

Three things to hold onto when reading both:

- **Admission is model-independent.** The three gates are pure memory arithmetic out of this file,
  so the two models never disagree about *what fits* — only about time. `serving.py`'s selftest
  enforces it.
- **They are not two estimates of one quantity. Do not average them, and do not pick the
  convenient one.** What the delta is for: small ⇒ the conclusion does not depend on the modelling
  choice, which is the strongest thing an analytic search can say; large ⇒ the ordering is a
  hypothesis about the scheduler. `sweep.py` prints the **inversion count** for exactly this.
- **Little's Law separates them.** `serving.py` computes E2E forward and compares it to
  `concurrency / throughput`, an identity nothing upstream used. Model 2 agrees within one step.
  Model 1 does **not**, and the residual is the size of its missing queueing model: its throughput
  is a harmonic combination that already implies a request waits behind the other residents'
  prefills, while its TTFT is a dedicated run that does not.

Measured on GLM-5.2-FP8 at `2x2x2`, 8K/1K, shippable set: **0 inverted pairs out of 105** at C=256
(model 2 reads 1.12–1.15× higher across the board), and **12 of 190** over a full concurrency sweep.
So on this cell the two agree on the ranking and differ on the level — which is the good case, and
is worth stating rather than assuming.

---

## 1. Methodology

### The model

```
roofline_ms(op) = max(hbm_ms, compute_ms) + ici_ms
E2E             = Σ_ops roofline_ms(op) × layers_of_that_class
```

Compute and HBM overlap **within** an op; collectives are **fully exposed**. Both are optimistic.

### Compute

```
flops   = 2 × tokens × pad(contracting_dim) × output_dim × n_matrices
comp_ms = flops / peak(dtype_A, dtype_B)
```

`flops` is a **count of operations — dtype-independent**. Precision changes the *rate*, not the
work, so dtype appears only in the denominator. `peak` takes **every operand**: one wide operand
halves the pass, so w8a16 runs at the bf16 rate, not fp8.

Non-matmul work — norms, residuals, RoPE, softmax, SwiGLU, router, top-k, permute — runs on the
**VPU at ~1/16 the MXU rate**. Priced separately; FLOP counts there are per-element estimates.

### HBM

```
hbm_bytes = weights + activations_in + activations_out  (+ KV for attention)
```

`act_in` and `act_out` are separate dtypes: under w8a8 the input is read quantized and the
accumulator written wide.

### Communication

Volume from `parallelism/<scheme>/<scheme>.md`, price from `cost_estimator/collective.py` (axis-aware:
it knows D2D from ICI, torus from line). **Never a closed-form ring formula here.**

### Serving — from op times to the four numbers

**This subsection is step model 1** (§0), implemented in `roofline_model1.py` and reached through
`serving.py`. `roofline.serving()` is the same arithmetic as a standalone convenience for a
single-config report. Model 2 replaces everything from `T_prefill` downward; its composition is in
its own module docstring.

Symbols, all **per core**:

| | |
| --- | --- |
| `ISL`, `OSL` | input and output tokens per request |
| `max_model_len` | vLLM's **cap** on `ISL + OSL` (`scheduler.py:893` bounds prompt + generated cumulatively). Not a workload, and not an input here — two deployments with the same cap can have very different ISL/OSL |
| `mnbt` | the per-step token budget (`max_num_batched_tokens`) |
| `n_c` | **number of prefill chunks** = `ceil(ISL / mnbt)`, i.e. how many forward passes the prompt is split into |
| `T_chunk(i)` | time of prefill pass `i`, which attends over `i·mnbt + mnbt/2` of context |
| `T_prefill` | the whole prompt = `Σ_{i=0}^{n_c−1} T_chunk(i)` |
| `T_step` | one decode step |
| `batch` | requests decoding concurrently on this core |

The decode phase's `kv_len` is `ISL + OSL` — the context at the **last** generated token, i.e. the
peak within a request, not the mean (`ISL + OSL/2`). That is the expensive end on purpose.

```
T_prefill = Σ T_chunk(i)        a SUM over n_c unequal chunks, not n_c × T_chunk
T_step                          one decode step; all requests advance one token, lockstep

TPOT = T_step
TTFT = T_prefill                ← a floor: no queueing model
E2E  = TTFT + (OSL − 1) × TPOT
```

The chunks are unequal because later ones attend over more context, so `T_prefill` has to be
summed rather than multiplied.

Throughput is **not** `tokens / T`, because a core interleaves prefill and decode rather than doing
both at once. Price each alone, then combine — a request needs both *serially*, so times add and
rates combine as reciprocals:

```
QPS_prefill = (tokens/ISL) / T_prefill
QPS_decode  = batch / (OSL × T_step)
QPS         = 1 / (1/QPS_prefill + 1/QPS_decode)
throughput  = QPS × (ISL + OSL)
```

`prefill share = (1/QPS_prefill) / (1/QPS_prefill + 1/QPS_decode)` says which phase a win moves.

### The two output tables

Two views of one set of numbers, and they answer different questions.

**The flat phase table** is every op, sorted by cost, with its layer multiplicity applied. Use it to
find the expensive op.

**The per-layer tables** (`layer_rows`, one per *kind* of layer per phase) are one representative
layer in **dataflow order**, plus a layer-sums table. Use them to read a layer the way it executes.
GLM-5.2's 78 layers are three distinct layers repeated, and the phase total has already averaged the
indexer across the 57 layers that do not run it — which is exactly the thing a per-layer table shows
and a phase total hides.

Every row is summed **in the script**, on purpose (§4). Three `role`s come back:

| role | |
| --- | --- |
| `op` | one operator |
| `fused` | a parent over sub-ops that are one kernel on the device. Its children follow with `parent` set and **must not be summed again** |
| `total` | the whole layer kind; `layer_total_ms = roofline_ms × count` and these reproduce the phase total exactly |

So building a report is filtering, ordering and naming — never arithmetic. **Naming is deliberately
not here**: the script prints the key (`moe_idx`), and the tpu-roofline prompt §4 owns the map to "Full Indexer
MoE Layer", because that is model vocabulary rather than cost.

Two properties worth knowing before quoting a row:

- **Merged rows (`fused`, `total`) do not satisfy `max(mem, compute) + ici = roofline`.** They sum
  parts that run one after another. Only leaf rows obey the formula. Recomputing the max over summed
  columns would claim an overlap nothing here models.
- **The occupancy columns are HBM residency on this core**, not traffic (`model/common/
  REQUIREMENTs.md` §TRANSFER and OCCUPANCY), so they do not reproduce the Memory Time column. Times
  are summed over every pass in the phase, so a chunked prefill's 8 passes are already in; occupancy
  is the max over passes, since passes run one after another. A `TOTAL` peaks activations over its
  ops and sums weights and KV.

`ici_lat_ms` / `ici_bw_ms` come straight from `collective.py`'s alpha/beta split, with its `t_fixed`
folded into latency. The halves respond to different fixes: bandwidth time falls when the payload
does, latency does not — a mostly-latency collective is not fixed by quantizing what it carries.

---

## 2. Highlights — the things that bite

Each of these was found by being wrong about it first.

- **MXU padding.** The contracting dim pads to 256. `MLA: QK Rope` contracts over `D_rope = 64`, so
  **three quarters of that op is padding waste** — ~36 ms of a 1,494 ms prefill.
- **Activations dominate prefill HBM.** `Query Down Proj` reads 92 MiB, of which weights are 12.
  A weights-only model misses 87% of it.
- **Sparsity buys compute, not bandwidth, at prefill.** Many queries each picking `topk` keys union
  to the whole context. Only decode gets the bandwidth saving.
- **`L_moe ≠ L`, and `L_idx ≠ L_moe`.** GLM-5.2 is 78 layers: 75 MoE, 21 running the indexer.
  The profile confirms collectives fire 75×/step.
- **EP issues three collectives, not two.** The router-logit all-gather is `E` wide, not `D`, so it
  does not look like the others — 11–12% of EP's collective time.
- **The MoE token permute is 4.8% of prefill** and is neither a GEMM nor a collective, which is
  exactly why it was missing until the spec reconciliation.
- **Chunk size barely matters here — over the range that was measured, which is the trap.**
  `mnbt` 8192→1024 moves prefill **0.13%**, because collective volume and sparse-attention compute
  are both chunk-invariant; only the weight re-read scales. **Do not read that as "mnbt does
  nothing".** 1024→8192 is the flat *top* of the curve. Extend the ladder downward and the same
  model-1 prefill run goes **1,486 ms at mnbt 4096 → 1,816 ms at 256**, worth **1.17x of
  throughput** at saturation; model 2 gives 1.18x. The low rungs are the ones that discriminate,
  and this figure has already been over-generalised once downstream.
  Where the two models genuinely differ is *what* the knob touches, not how much: under model 1
  `T_step` is exactly invariant to `mnbt`, so a large chunk costs decode latency nothing; under
  model 2 the step grows and every decode in it waits behind the chunk. Still a property of *this*
  cell — re-check it on a dense model rather than carrying it.

## 3. Known gaps — expect a profile to be slower

- No overlap beyond compute-with-HBM inside one op; real collectives are async and still 21.9%
  (decode) / 37.8% (prefill) exposed.
- **No layout/copy term** — measured at 7.2% of prefill, 4.6% of decode.
- Perfect expert routing, no hot-expert tail.
- No queueing, no `max_num_seqs`, no straggler **in this file**. The KV pool and its request
  capacity **are** computed (`hbm x utilization - weights - overhead`, per
  `vllm-torchtpu utils.py:171`) and reported next to the configured batch — as a number to read,
  not a gate that rejects. `serving.py` turns all three into real admission gates and adds a
  queueing term; use it when the batch is an *output* of the load rather than an input.
- Collective fixed cost ~3.4× underpriced at the configured 1.0 µs hop latency (13.5 µs measured vs
  4.0 µs of hop latency on a 4-axis slice; buglist.md B5), which lands on small decode-sized
  collectives.
- `kernel_efficiency = 1.0`, no launch overhead. **The output is a ceiling. Use ratios.**

## 4. Why a script and not an agent

Measured 2026-09-10: five agents, same pins, same instruction → prefill anchors of **19 ms, 6,216,
7,641, 10,554 and 2,410,850 ms**; a byte-for-byte re-dispatch moved one config **1,118×**. The same
agents reproduced the *pins* to within 4%. Transcription is reliable, derivation is not.

## 5. Validation and reconciliation

`--selftest` pins 23 identities, 14 of them against an independently hand-built workbook roofline
for GLM-5.2-FP8. **Every compute time matches to 5 decimal places**, after rescaling the
workbook from its 1,785.62 TFLOP/s fp8 peak to the published 2,307 (see the selftest docstring). Six of the rest pin the
per-layer tables as a *regrouping* rather than a second cost model: the layer kinds reconcile with
the flat multiplicities, the kind totals reproduce the phase total to 9 places, a fused parent
equals its children, Memory Time comes from traffic and not from the occupancy columns,
and the ICI split sums to what `collective.py` priced. Where the E2E differs it is in the
collective term, from two causes: the workbook's self-flagged GiB-vs-GB ICI bug (1.074×), and its
flat 12-link hierarchical collective model against this one's per-axis ring decomposition.

Reconciling against `model/common/` on 2026-09-21 found **two ops missing entirely** — the MoE
router GEMM and the token permute/unpermute (+5.2% prefill together). **Two divergences remain
unsettled**, both source reads:

1. `mla/mla.md` says prefill uses **Flow A** (no weight absorption); this script uses Flow B for
   both phases, as the workbook does.
2. `dsa/dsa.md`:106 generates `Q_rope` from `D` directly; this script folds it into the q-LoRA
   up-projection.

### Changing `roofline.py`: check `model/common/` in the same edit

This section is about the **package source**, not the installed payload. The installed copy is
read-only; make these changes in the package repository and rebuild.

`model/common/` is the **specification** and
`roofline.py` **implements** it. A change to either is a claim
about both, and the pair drifting apart is how this agent produces confident
wrong numbers. Decide which case yours is **before** you edit:

| what you found | fix |
| --- | --- |
| code contradicts a correct spec | code only — but say which spec line you were failing |
| spec is wrong or unqualified | **both**, spec first |
| neither says anything | **both**: the spec gets the rule, the code the arithmetic |

Two checks on any op you touch, each of which has caught a real bug:

1. **Does the output have a different width from the input?** KV-cache writes, top-k indices, fp8
   GEMMs accumulating to bf16. Every single-dtype helper in `roofline.py` has been wrong about one.
2. **Grep the other specs for the same shape of claim** — "kept in VMEM", "can be ignored if
   fused", "`* bytes`" with no dtype. A wrong assumption is rarely written once.

Then re-run `--selftest` **and** reconcile the op's FLOPs against the spec formula in Python: the
selftest pins the code against itself, not against `model/common/`.
