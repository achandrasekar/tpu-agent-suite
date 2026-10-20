---
name: pipeline-parallelism-roofline
description: Pipeline Parallelism (PP) for roofline analysis — how vllm-torchtpu runs stages in lockstep collective-permute waves, the layer partition (`get_pp_indices`, `VLLM_PP_LAYER_PARTITION`, now `Sharding.pp_layers`) and how to search it, what the hand-off carries, why decode needs `pp` disjoint request groups while one prompt pipelines itself, and what it does to HBM.
scripts:
- scripts/roofline-analysis/roofline.py
references:
- references/roofline-analysis/parallelism/SKILL.md
---

# Pipeline Parallelism (PP)

PP splits the layer stack into `k` contiguous stages, one device per stage on TPU today. A token
traverses every stage; throughput is set by the slowest one.

**PP's cost is not the transfer.** What PP costs is fill, balance, and a `k`-fold stretch in
per-request latency — §6. The transfer is small but not free, and on TPU it is padded (§4).

See [`../SKILL.md`]({{reference:roofline-analysis/parallelism/SKILL.md}}) for the variant schema, the `source` field, and all constraints
and cross-scheme interactions.

**Source for this file:** vllm-torchtpu `origin/main` at `fde6c3994`, read 2026-09-23, plus the
upstream vLLM installed in `<vllm-torchtpu>/uv_venv` (`0.27.0+empty`). Implemented by
`{{script:roofline-analysis/roofline.py}}` (`stage_layers`, `stage_ms`, `pp_handoff_ms`, `stage_weight_bytes`,
`best_pp_partition`) and used by both step models.

## 1. Notation

| symbol | meaning |
| --- | --- |
| `PP`, `k` | Pipeline depth — number of stages |
| `L`, `L_i` | Layers total, and layers on stage `i` |
| `D` | Hidden dimension |
| `B_pad` | The padded hand-off buffer: the largest token bucket, i.e. `max_num_batched_tokens` |
| `K_idx` | Top-k indices per token, on sparse-attention (DSA) models |
| `v` | Batches in flight = `min(k, floor(running))` |
| `T_i` | Stage `i`'s time for one forward, including the hand-off |
| `T_slow`, `T_round` | `max_i T_i`, `sum_i T_i` |

## 2. How a TPU pipeline runs

`distributed/pp_wave.py`, `pp_shift.py`, `pp_push.py`:

* **Lockstep waves.** After each forward, **all stages run one collective-permute together**: each
  sends its output to the next stage and the last stage's goes to stage 0, which ignores it — one
  closed cycle (`pp_shift.py`). Launch `p` on one stage pairs with launch `p` on every other, so the
  stages advance together and **a wave lasts as long as the slowest forward in it**.
* **Pushes.** When the engine must wait for a result before a forward has reached the last stage,
  it issues pushes — launches carrying nothing (`pp_wave.py:270-285`). A push is **the permute
  only, no model forward**, so an idle stage costs a launch, not a pass over its layers.
* **Settle.** Once nothing is in flight, every stage is brought to the same launch count and the
  burst closes.

## 3. Where the boundaries fall — the partition

**Default: upstream `get_pp_indices`** (`vllm/distributed/utils.py:127-167`). Every stage gets
`L // k`; the remainder goes to the stages counting back **from the second-to-last**
(`partitions[-i] += 1` for `i = 2 .. r+1`). GLM-5.2's 78 layers: `k=4 -> [19, 20, 20, 19]`,
`k=16 -> [4, 5×14, 4]`.

**Explicit: `VLLM_PP_LAYER_PARTITION`**, a comma list of per-stage layer counts that must have
`k` entries summing to `L` (`utils.py:143-154`). An **environment variable, not a flag**. In the
roofline it is **`Sharding.pp_layers`** — set it in a config's `sharding` block as a list, and
`search/` can set it per row. Empty means the default.

**Search: `roofline.best_pp_partition(cfg, ph, k)`** returns the contiguous partition minimising
`T_slow` for a representative pass `ph`, subject to every stage's weights fitting in HBM, with the
lighter heaviest stage as the tie-break. It is a min-max linear partition over the real per-layer
costs, exact by dynamic programming.

**Why the partition is a knob.** The partition that balances memory does not balance time.
GLM-5.2's first 3 layers are dense and ~43× lighter than an MoE layer; 21 of 78 layers also run the
indexer, at `max(l - offset + 1, 0) % freq == 0` (`vllm/model_executor/models/deepseek_v2.py:
1104-1105`; GLM-5.2 `freq=4, offset=3` -> layers `0, 1, 2, 6, 10, …, 74`). `Model.indexer_freq` /
`indexer_offset` carry that rule; without them the indexer layers are spread evenly, which is a
guess. The one PP job measured on this stack used `6,4,4,4,4,8,4,4,4,8,4,4,4,4,8,4` (§8).

## 4. The hand-off — what crosses a boundary

| tensor | shape | dtype site | source |
| --- | --- | --- | --- |
| hidden state + residual, **summed** | `[B_pad, D]` | `act_out` (model dtype) | `pp_wave.py`: sending the sum halves the bytes |
| top-k index table, **sparse models only** | `[B_pad, K_idx]` | `indexer_indices` (int32) | `runner/tpu_runner.py:5492-5553`, at **every** boundary — "every stage passes them on" |

**Padded to the largest bucket.** "Every launch runs the same compiled program (one buffer padded
to the largest token bucket)" (`pp_wave.py`). A 16-token decode step ships `max_num_batched_tokens`
rows. On GLM-5.2 at `mnbt = 1024` that is 12 MiB of hidden state plus 8 MiB of top-k per launch.

```
T_handoff = B_pad * (D * b(act_out) + K_idx * b(indexer_indices)) / link_bw + t_hop
```

**Every stage pays it**, the last included: the permute is one collective every stage launches on
every wave. It is an XLA collective-permute, priced directly because `collective.py` has no
permute primitive; one link rate is charged whatever the placement (§9).

## 5. What PP does to HBM

Per device.

| tensor | per-device | effect |
| --- | --- | --- |
| Weights | the layers of **this stage**, then `/ TP`; plus the embedding on stage 0 and the LM head on the last | ÷k on a balanced partition |
| KV cache | this stage's layers only; the indexer cache only for its indexer layers | ÷k — the gate it moves |
| Hand-off buffers | `B_pad` rows | small |

**Size the pool against the heaviest stage.** `roofline.weight_bytes` and `kv_bytes_per_token`
return the **maximum over stages**, not the mean. On a partition with 8-layer stages among
4-layer ones the heaviest holds ~1.6× the mean; crediting the mean admits a batch the busiest device
cannot hold. (Weights and KV maxima are taken independently, which is conservative when they fall
on different stages.)

**Weight bytes per emitted token are unchanged by PP alone.** PP divides the decode tokens in one
forward by `k` (§6), so it is free only if concurrency scales with `k`.

## 6. Micro-batching — why decode needs requests and prefill does not

The engine keeps up to `k` batches in flight (`vllm/config/vllm.py:540-550`, `max_concurrent_batches
= pp` without async scheduling; `v1/engine/core.py:625` `step_with_batch_queue`).

* **Decode needs `k` disjoint request groups.** A request whose token is in flight has nothing new
  to schedule until its output returns (`v1/core/sched/scheduler.py:558-572`, `num_new_tokens ==
  0`), so it sits in at most one in-flight batch. The running set is **divided**:
  `v = min(k, floor(running))`, each batch holding `running / v`. `v < k` is a **steady-state**
  bubble, not training's flush bubble.
* **Prefill does not.** A prompt's next chunk *can* be scheduled while its previous chunk is in
  flight, so one long prompt fills the pipeline by itself: chunk `c+1` enters stage 0 while chunk
  `c` is in stage 1.

```
decode   cycle = max(T_round, v * T_slow)        one token per request per cycle  -> TPOT
prefill  TTFT  = sum_c T_slow(c) + (T_round(c_last) - T_slow(c_last))            one chunk per wave, then drain
```

Both reduce to the single-stage step at `k = 1`.

**Time-budget chunking — named, not modelled.** `TPU_PP_DYNAMIC_CHUNKS` (default on) sizes prefill
chunks by **time** rather than tokens, because a wave lasts as long as its slowest step and a chunk
deep in a long prompt attends to more context (`core/pp_chunks.py`). When pending prefill cannot
fill every stage it spreads that work across the stages as small steps
(`core/pp_chunk_scheduler.py`). The roofline cuts chunks at `mnbt` tokens; the pipelining above
captures most of the effect, the per-chunk time equalisation is not modelled.

**Report fill and balance separately.** `eta_bal = (T_round / k) / T_slow` names a repartition;
`v / k` names more concurrency.

| | effect of `PP = k` |
| --- | --- |
| weight bytes / device | ÷k, heaviest stage |
| KV bytes / token / device | ÷k, heaviest stage |
| per-request decode latency | ×k (`T_round`) |
| requests needed to keep decode busy | ×k |
| one long prompt's TTFT | ~`n_chunks * T_slow`, not `n_chunks * T_round` |

## 7. Variants

| variant | partition | source |
| --- | --- | --- |
| **A — default** | `get_pp_indices` | `vllm/distributed/utils.py:127-167` |
| **B — explicit** | `Sharding.pp_layers` = `VLLM_PP_LAYER_PARTITION` | `utils.py:143-154`; searched by `best_pp_partition` |

Both: one collective-permute per forward over the PP group, every stage; `overlap` `serial`.

## 8. The one measurement

`j-78adee7b-af0e-43cc-9ee0` (another team member's job, 2026-09-21, image main @
2026-09-20): GLM-5.2-FP8, `pp16-tp1-dp1`, no EP, partition `6,4,4,4,4,8,4,4,4,8,4,4,4,4,8,4`,
ISL 131072 / OSL 1024, 32 prompts at concurrency 16, `mnbt 1024`. **32/32; 2041.86 tok/s total,
15.83 output; median TTFT 355 s, TPOT 430 ms.** Legality evidence and a calibration point; not a
baseline and not an A/B (CONSTRAINTS.md in `sharding-strategy`).

**Priced against it (2026-09-23), `serving.py --concurrency 16 --mnbt 1024 --both`:**

| | measured | model 1 | model 2 | model 1 before this file's code |
| --- | --- | --- | --- | --- |
| TPOT | 430 ms | 380 ms | 392 ms | 13.9 ms (mean stage) |
| output tok/s | 15.83 | 36 | 36 | 311 |
| median TTFT | 355 s | 4.3 s | 3.4 s | 2.4 s |

* **Decode agrees to ~10%.** The stage model — real partition, `v = 16`, padded hand-off — is what
  moved model 1 from 30× optimistic to 12% optimistic.
* **Prefill is ~10× optimistic.** The benchmark sends 16 128 k prompts at once, so TTFT includes
  queueing behind other prompts' prefill, which this steady-state model does not see; and
  time-budget chunking plus the SMEM pair cap shrink chunks deep in a 128 k prompt, multiplying
  the waves (§6, not modelled). Treat PP TTFT at long context as a lower bound.
* **The partition is modelled as the job's main loss.** Its three 8-layer stages give
  `T_slow` 23.7 ms against 14.9 for the default `[4, 5×14, 4]` (balance 0.59 vs 0.94), and 73.5 GiB
  of weights on the heaviest stage against 46.0. `best_pp_partition` finds nothing better than the
  default beyond moving the dense layers onto stage 0. Modelled TPOT at the default partition:
  239 ms, **−37%**. A hypothesis — one rerun of the same recipe with the partition unset settles
  it.

## 9. Open

* **No stage placement.** One link rate for every boundary, whether D2D, ICI, or across hosts.
* **Time-budget chunk sizing** (§6) is not modelled.
* **Embedding and LM head are HBM-only.** Their bytes are on stages 0 and `k-1`; their compute has
  no op row, so it is in no stage time.
* **No microbatch heterogeneity.** All `v` batches are the same shape; a chunk-prefilling batch
  beside decoding ones makes `T_slow` a moving straggler — the thing time-budget chunking exists to
  fight.
* **`collective.py` has no permute primitive**, so §4 is priced outside the estimator.
