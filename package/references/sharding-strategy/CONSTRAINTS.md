---
scripts:
- scripts/sharding-strategy/search/space.py
- scripts/sharding-strategy/search/sweep.py
---
# What sharding this stack actually supports

The catalogue {{script:sharding-strategy/search/space.py}} encodes. Every row is a line of code, cited, with the sha it
was read at.

**These live in checkouts nobody here controls and they drift.** A rule that has silently been
lifted costs you the winning config; a rule that has silently been *added* costs you a
2.5-4 hour benchmark that dies in `__post_init__`. Step 1 of `SKILL.md` is re-reading this
table against the checkout you are about to serve from — not because it is likely to have
moved, but because the whole search is downstream of it.

Read at:

| checkout | sha | read on |
| --- | --- | --- |
| `vllm` (upstream) | `0.27.0+empty`, installed at `<vllm-torchtpu>/uv_venv/lib/python3.12/site-packages/vllm` | 2026-09-17; PP rows 2026-09-23 |
| `vllm-torchtpu` | `d43dae9d` (2026-09-14) | 2026-09-17; PP rows 2026-09-23 |

> [!WARNING]
> The PP rows were read at `d43dae9d` but **measured on an image built from main @ 2026-09-20**,
> six days newer. The checkout is the older of the two: `pp_validation.py` may have grown rules
> since. Re-read it against the image you will serve from before trusting the PP refusal list.

> [!NOTE]
> **There is no `vllm/` clone at the repo root on this machine**, so the upstream rules below were
> re-read from the vllm *installed into `<vllm-torchtpu>/uv_venv`* — which is the copy the server
> actually imports, and therefore better evidence than a side clone, but it is identified by a
> package version rather than a sha. The three `config/parallel.py` rules are unchanged in
> substance since `d6c2fec9f`; only their line numbers moved, and they are updated below.

## The six axes

Five multiply out to the world size; one does not, and one is not a degree at all.

| axis | flag | what it splits | in `world`? |
| --- | --- | --- | --- |
| TP | `--tensor-parallel-size` | heads and FFN columns, within a layer | yes |
| DP | `--data-parallel-size` | requests — one independent engine per rank | yes |
| PCP | `--prefill-context-parallel-size` | the prompt, across ranks, during prefill only | yes |
| PP | `--pipeline-parallel-size` | the layer stack into stages | yes |
| DCP | `--decode-context-parallel-size` | the KV cache, **re-using the TP ranks** | **no** |
| EP | `--enable-expert-parallel` | the experts — a **boolean**, not a degree | n/a |

`world = TP x PCP x PP x DP`. DCP subdivides TP rather than adding ranks, so
`tp16-dp1-dcp4` is still sixteen devices.

Declarations: `vllm/config/parallel.py:122` (PP), `:124` (TP), `:126` (PCP), `:129` (DP),
`:165` (EP), `:342` (DCP).

## EP is a boolean, and that is the constraint people trip on

> The EP group spans the TP x PCP x DP ranks.
> — `vllm/config/parallel.py:515`

There is no `--expert-parallel-size`. Turning EP on spreads the experts across **every** rank
in the TP x PCP x DP product, so you cannot ask for "TP=2 on attention, EP=8 on the MoE" —
the EP group is whatever the other axes already made. The only lever is the boolean, and
`Sharding.ep_group` computes what you get.

This matters because on a large MoE the expert weights are most of the model, and EP is the
only axis that divides them. The sweep ({{script:sharding-strategy/search/sweep.py}}) will report `weights do not fit in HBM` for
GLM-5.2 at `tp1-dp16` and fit comfortably at `tp1-dp16-ep`, which is the same fact.

## Hard refusals

| rule | where | encoded as |
| --- | --- | --- |
| `PCP does not support data parallelism yet` | `vllm/config/parallel.py:528` | `pcp > 1 and dp > 1` |
| when `pcp == 1`, `tp % dcp == 0` — "DCP reuses the TP ranks" | `vllm/config/parallel.py:529-532` | `tp % dcp` |
| when `pcp > 1`, `dcp in {1, pcp, tp*pcp}` | `vllm/config/parallel.py:533-539` | same set |
| `PCP runner path does not support KV consumer/decode workers` | `vllm-torchtpu:.../pcp_validation.py:103` | `role == "decode"` |
| `PCP runner path does not support DCP yet` | `pcp_validation.py:107` | `pcp > 1 and dcp > 1` |
| `PCP runner path does not support pipeline parallelism yet` | `pcp_validation.py:110` | `pcp > 1 and pp > 1` |
| **`PCP on an MoE model requires --enable-expert-parallel`** — **NEW, added between `1dc0f5ce` and `d43dae9d`** | `pcp_validation.py:113` | `pcp > 1 and is_moe and not ep` |
| `PCP runner path does not support TPU multihost yet` | `pcp_validation.py:132` | `pcp > 1 and hosts > 1` |
| `Prefill context parallelism is not supported together with multihost data parallelism` | `vllm-torchtpu:.../tpu_platform.py:1074` | subsumed by the two above |
| `Pipeline parallelism on TPU does not support tensor parallelism inside a stage yet` | `vllm-torchtpu:.../pp_validation.py:48-53` | `pp > 1 and tp > 1` |
| `… does not support async scheduling yet` — pass `--no-async-scheduling` | `pp_validation.py:54-58` | emitted by `ServerConfig.flags()` |
| `… does not support speculative decoding yet` | `pp_validation.py:59-60` | out of scope (unified serving) |
| `… does not support KV transfer connectors other than TPURaidenConnector yet` | `pp_validation.py:61-66` | out of scope (no `kv_transfer_config`) |
| `… does not support the legacy Ray executor yet` — set `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` | `pp_validation.py:76-81` | recipe-level, not a sharding |
| **Only pure PP** — `pp > 1` with `dp > 1` is unsupported. **No code refuses it**; user-reported, unmeasured | see "PP" below | `pp > 1 and dp > 1` |

**The multihost rule is the one that decides things.** It takes PCP off the 2-host
(`2x2x2`) slice entirely, whatever else is set — so on 16 devices the whole PCP family is
gone before any costing happens. On a 1-host slice PCP survives only at `dp=1`.

**The new MoE rule matters most to the wide view**, which is the one that prices what a
blocking rule costs. On an MoE checkpoint a PCP row now needs *two* gaps closed, not one:
lifting multihost PCP does **not** make `pcp>1` runnable unless EP is also on, because
without it the experts would shard across the PCP ranks and "the TPU MoE kernels do not
implement" that. A projection that quotes the PCP-without-EP rows as "worth X once
multihost lands" is quoting the wrong cheque. `pcp>1, ep=True` needs only the one fix.

Also new at `d43dae9d` and **out of scope here** (this skill is unified serving, no
speculative decoding): PCP + spec-decode requires `method=mtp`, `num_speculative_tokens=1`
and `kv_role=kv_producer` (`pcp_validation.py:120-130`). Not encoded in `space.py`.

Note the KV-consumer rule is `is_kv_producer is False`, not "is not a producer"
(`pcp_validation.py:90`). A unified server has no `kv_transfer_config` at all, so the field
is `None` and PCP is **not** refused. PCP is available outside disaggregation; it is only
refused on the decode half of a disaggregated pair. {{script:sharding-strategy/search/space.py}} takes
`role="unified" | "prefill" | "decode"` for exactly this distinction.

## PP — works, in one shape

> [!IMPORTANT]
> **Reversed 2026-09-23.** This section previously said "PP does not run on tpu7x", on a
> controlled 2026-09-10 measurement. That measurement was correct when taken and is now
> **stale**: the stack was fixed underneath it. PP is wired, validated and serving.

`pipeline_parallel_size > 1` **runs**, on one host and across hosts, in the **pure-PP** shape —
`TP=1`, `DP=1`, `PCP=1`, the whole slice spent on stages.

### What landed

| commit | date | what |
| --- | --- | --- |
| `012e4f24` | 2026-09-09 | Pipeline parallelism on TPU |
| `8ff26610` | 2026-09-09 | Raiden KV resharding for pipeline-parallel engines |
| `15085885` | 2026-09-11 | Carry pipeline forwards with pushes; drain the wave only when idle |

The old section's three failure sites are gone with them, and the reason the 2026-09-10 result
looked so solid is now visible: its image was built from `136176940f81`, the merge-base of
`1dc0f5ce` with `origin/main` — i.e. from **before** `012e4f24`. The control arm was honest; it
controlled for the wrong thing. The old evidence is preserved below.

`platforms/pp_validation.py` is the new authority, called from `tpu_platform.py:807`, and
`init_pp_distributed_environment`'s zero callers is no longer the signal it was — the hand-off
goes through the executor push hook `15085885` added, not that function.

### The measurement that reverses it

| | |
| --- | --- |
| job | `j-78adee7b-af0e-43cc-9ee0` — **Succeeded**, 2026-09-21, 82 min |
| owner | another team member (not ours; read-only) |
| recipe | `torchtpu-vllm-glm-5.2-v7x-16-profile-pp-rox-candidate` |
| image | `…/torchtpu-vllm:<main-0920 tag>` (main @ 2026-09-20) |
| slice | 16 chips, **two Ray nodes** — so the old multihost `PjRtClient` site is cleared too |
| sharding | `--pipeline-parallel-size=16 --tensor-parallel-size=1 --data-parallel-size=1`, no EP |
| partition | `VLLM_PP_LAYER_PARTITION=6,4,4,4,4,8,4,4,4,8,4,4,4,4,8,4` |
| workload | GLM-5.2-FP8, ISL 131072 / OSL 1024, 32 prompts at concurrency 16, `mnbt=1024` |
| result | 32/32 successful; 2041.86 tok/s total, 15.83 tok/s output; median TTFT 355 s, TPOT 430 ms |

All flags read back from the job's recorded recipe, not from the job-creation command.

> [!WARNING]
> **Succeeded ≠ fast.** A 355 s median TTFT at 128 k is bubble-dominated — 16 stages, one
> in-flight wave, no TP to shorten a stage. This row establishes **legality only**. It is not a
> baseline, it is not an A/B, and it has no control arm at another sharding on the same image.
> Treat any PP throughput claim as unmeasured until the sweep prices it.

### `VLLM_PP_LAYER_PARTITION` — the knob PP needs

Stage boundaries are an **env var**, not a flag: upstream `vllm/distributed/utils.py:143`,
consumed by `get_pp_indices`. It takes a comma list of per-stage layer counts that must sum to
the model's layer count. GLM-5.2-FP8 has **78** layers
(`U-260909-02`), and the partition above
sums to 78 — not the 72 it is sometimes quoted as. It is deliberately uneven (`6,…,8,…`),
because a flat `78/16` split leaves the ragged stage straddling the dense/MoE boundary at
`first_k_dense_replace=3` and the four `8`s absorb it.

{{script:sharding-strategy/search/space.py}} does not emit this var — `ServerConfig.flags()` returns flags. A PP row taken
to a benchmark needs the partition set in the recipe alongside them.

### The one rule that is not read from code

Everything in the refusal table above is a line of `pp_validation.py`. **`pp > 1` with
`dp > 1` is not.** `pp_validation.py` lets it through; the constraint is reported as PP being
SPMD while DP is MPMD, and no job has tested it. `space.py` carries it as a **hard `INFRA`
refusal anyway** — the asymmetry is deliberate: the cost of a wrong `illegal` is a config the
search never proposes, and the cost of a wrong `shippable` is a 2.5–4 h benchmark that dies.
One `pp8-dp2` job retires the rule in either direction; until then it is the weakest row in
this file and it is labelled as such.

EP under pure PP is a no-op, incidentally: `ep_group` spans `TP × PCP × DP`, which is 1.

### Retired — the 2026-09-10 evidence

Kept because "PP is broken" was quoted downstream and the reversal has to be traceable. These
jobs ran on `sha256:3da9737739c0984053f188d2c451a4bda963c99a38661a6da85d2fac0b0156a4`, built
from `136176940f81`, **pre-`012e4f24`**. They do not describe today's stack.

| arm | config | job | result |
| --- | --- | --- | --- |
| A | `TP=1 PP=2 DP=8`, GLM-5.2-FP8, `2x2x2` | `j-4990f863-f695-4dd6-b715` | Failed — `PjRtClient is not initialized` in 8 workers |
| B (control) | `TP=2 PP=1 DP=8`, byte-identical otherwise | `j-4ff31951-1d16-4189-a247` | Succeeded |
| cheap | `EngineArgs(pipeline_parallel_size=2)` on 4 chips | `j-f0772f70-723a-4945-92f9` | `pp=2` engine died at two single-host sites |
| cheap control | same with `pp=1 tp=2` | `j-9916978e-533e-4a4f-bee8` | engine ran, generated tokens |

Failure sites, all now fixed: `tpu_runner.py:6534` (`assert intermediate_tensors is not None`),
`tpu_runner.py:4765` → `torch_tpu/_internal/sync/sync.py:59` (`_synchronize_tensor()`
incompatible arguments), `tpu_worker.py:381` via `ray_distributed_executor_v2.py:476`
(`PjRtClient is not initialized`).

**The lesson worth keeping is not about PP.** A controlled A/B pins a *difference*; it does not
pin a *date*. Arm A and arm B shared an image, so they isolated PP perfectly and both aged out
together eleven days later. Any rule in this file sourced from a measurement rather than a line
of code needs its image re-checked against `origin/main` before it is quoted, not just its
control arm re-read.

## Model-shaped refusals

These are not vLLM rules; they are arithmetic that vLLM will fail on later.

- **`tp` must divide `num_kv_heads`** for a GQA/MHA model. An MLA model has one shared
  latent and is exempt — which is also why TP does not shrink its KV cache.
- **The EP group must divide the routed expert count.** 256 experts over a 16-rank group is
  16 each; 256 over 24 is a crash.

## The one bound that is not encodable

GLM-5.2-FP8 carries a recorded MoE dispatch cap — `topk x DP x max-num-batched-tokens` must
stay at 65536, or a SparseCore faults the first batch
(`U-260909-17`). It is in `space.py` as
`DispatchBound`, and it is **off by default**, because the arithmetic in the source comment
only closes at `topk = 4` while `config.json` says 8. Until that is resolved, rebuilding the
bound for a different DP degree halves or doubles the wrong factor and lands on the faulting
shape.

Leaving it off means the search may propose a config that faults. That is the cheaper of the
two failures: a fault is a loud twenty-minute loss, and a wrongly-derived bound silently
excludes the winner and you never learn it existed. If you turn it on, set `topk` from
something you verified in the dispatch code, not from `config.json`.

## Re-verifying

```bash
grep -n "does not support data parallelism\|DCP reuses the TP ranks\|EP group spans" \
    <vllm>/vllm/config/parallel.py
grep -n "NotImplementedError" -A2 <checkout>/src/vllm_torchtpu/platforms/pcp_validation.py
grep -n "unsupported = " -A2 <checkout>/src/vllm_torchtpu/platforms/pp_validation.py
```

If a rule moved, fix {{script:sharding-strategy/search/space.py}} and this table in the same edit, and bump the shas
above. A stale rule here is worse than no rule: it looks checked.
