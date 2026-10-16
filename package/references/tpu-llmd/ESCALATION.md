# The fleet <-> engine loop

The llm-d study moves the *fleet*: shape (`xP:yD`, replicas, hosts),
router, KV reuse, and the engine knobs on the fleet whitelist. Everything
below that -- kernels, compilation buckets, sharding, the engine's own
scheduler -- is `tpu-performance-optimization`'s domain, measured on one replica with
its own protocol. A production objective can need both, in turn: a fleet
whose best shape still misses TPOT at target rate has an engine problem,
and an engine tuned in isolation does not know which role or operating
point the fleet needs it at. The loop connects the two without merging
them.

## When the hand-off happens

`engine_escalation.py prepare` decides, at the point the fleet's stopping
rule fires, and escalates only when **all** of these hold:

| condition | read from |
| :--- | :--- |
| the loop is enabled | `llmd-study.json` `engine_loop.enabled` (default true; `search.engine_loop` in the workload spec) |
| the objective is unmet | a failed SLO gate at `target_qps` in the incumbent's bench report, or an `objective_target` on the primary metric the incumbent has not reached |
| the fleet search is at a plateau | the orchestrator's `plateau_rounds` rule, or no open arm left |
| the budget allows | fewer than `engine_loop.max_escalations` (default 3) |
| a role is left | the attributed role is not in `engine_loop.exhausted_roles` |

A met objective never escalates: a fleet that keeps its SLO at target
rate is done, whatever the metric could still gain. A search with open
arms never escalates: the fleet is cheaper to move than the engine.

## What goes across

Attribution (`size_pd.attribute`) names the **role** -- `prefill` when the
TTFT gate fails or the prefill pool queues, `decode` for TPOT or the
decode pool, `aggregated` when nothing in the fleet metrics names a pool
-- and the bottleneck **kind** from the study taxonomy (`memory-capacity`
when KV occupancy or preemptions say so, else `scheduler-batching`; the
profile may rename it inside the engine study).

The `engine-escalation.json` (`{{schema:engine-escalation.json}}`) carries:

- the **operating point** one replica of that role runs at in the
  incumbent fleet: ISL/OSL means, measured prefix hit rate, per-replica
  concurrency (`requests_running_mean` of the pool), offered rate, the
  role's merged engine knobs and the fleet shape;
- the **engine objective**: prefill maximizes prompt tokens/s under the
  tightest TTFT bound, decode maximizes generation tokens/s under the
  tightest TPOT bound, an aggregated engine answers for both;
- a **`study-protocol`** for the engine study (cell = one replica of the
  role's slice at that concurrency; replicates, noise floor and mode
  inherited from the fleet protocol; `max_rounds` 8, `plateau_rounds` 3);
- an **inference-perf microbenchmark** that loads one replica the way the
  fleet loads the role (prefill: the workload's prompt length with a
  one-token answer; decode: a short prompt with the workload's output
  length; closed loop at the measured concurrency);
- **hypothesis seeds** in `hypothesis_bank.py`'s shape. Each seed with a
  mechanical change carries `target_parameters.serving = {args, env,
  image}`; seeds for directions the profile must decide (padding
  buckets, the dominant kernel) carry an empty `serving` and are the
  researcher's to develop. Flags the fleet's inner sweep already walked
  are listed as `excluded_knobs`.

The orchestrator opens the engine study from the protocol, admits the
seeds, records the escalation on the fleet study, and returns
`engine_escalation`. The fleet study stays open.

## Who drives the engine rounds

`tpu-coworker`, the only level with a user. In `approve` mode it asks
before the first engine round; in `auto` mode it proceeds. It advances
`tpu-performance-optimization` rounds on the engine study until that study closes,
passing the escalation path in each `study-request` prompt so the
researcher and executor read the operating point, the microbenchmark and
the return contract. Then it comes back to `tpu-llmd` with
`engine_study_id`.

## What comes back

`recipe_bump.py` reads the engine study's state and its hypothesis bank,
takes the winning arm with the largest `delta_pct`, applies its
`serving` delta to the recipe (args flag by flag, env merged, image
replaced), refreshes `measured.peak_prefill_throughput` /
`peak_decode_throughput` from the arm's metric when it is that role's
tokens/s, and writes `serving-recipe.v<N+1>.json` with
`source.engine_study_id` plus an amendment naming the **successor fleet
study** (`<study_id>-v<N+1>`) and the baseline fingerprint
(`commit_sha` = the new recipe's digest) its first measurement carries.

The fleet search then re-baselines in that successor: the runtime refuses
to replace a baseline that arms were measured against, so the old study
is closed with the bump as its outcome and keeps its ledger as history,
and the **incumbent shape** measured on the new recipe becomes the
successor's baseline. A search never compares a measurement on one engine
with a measurement on another. A win that recorded no
serving delta (`no_delta`) is reported, not applied; a study with no win
marks the role exhausted.

## What ends the loop

The objective met; `max_escalations`; `max_chip_hours`; or the fleet at
a plateau with every attributed role exhausted. Each of those is a
different sentence in the final report, and the ledger shows which one.
