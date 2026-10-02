---
scripts:
  - scripts/tpu-llmd/seed_arms.py
---
# The search: how an llm-d study climbs

The study is a **two-level hill-climb** over one configuration document,
`llmd-config.json` (`{{schema:llmd-config.json}}`). The outer level moves
**cold** knobs, one arm per round, each a fresh release. The inner level
moves **hot** and **warm** knobs on the live release, inside the arm, within
a fixed budget. The best admissible configuration so far is the
**incumbent**; every arm starts from it, and a win replaces it.

## Why two levels

A cold change (replica count, topology, router profile) costs a full release
cycle: model weights pulled, servers warmed, router re-installed -- tens of
minutes on a TPU fleet. A hot change (a scorer weight) is a ConfigMap edit
the EPP reloads in seconds; a warm change (an engine flag) is a rolling
restart of the model servers. Treating every knob as cold would make a
twelve-round study explore twelve points. Letting the executor sweep the
cheap knobs inside each arm means every cold point is measured at its *own*
best tuning, which is the only fair comparison between, say, three replicas
at `max_num_seqs=256` and two at `max_num_seqs=512`.

## The knob classes

`deploy reconfigure` classifies a config path and refuses cold ones
(`COLD_PATH`), so the boundary is enforced, not advised:

| class | paths | mechanism | who moves it |
| :--- | :--- | :--- | :--- |
| hot | `router.plugins.<plugin>.weight`, `router.plugins.<plugin>.<parameter>` | EPP ConfigMap rewrite; the picker reloads it | executor, inner sweep |
| warm | `engine.*` (whitelisted flags), `kv.block_size` | model-server Deployment rollout | executor, inner sweep |
| cold | `topology.*`, `router.profile`, `kv.precise_routing`, `kv.tiered_host`, `kv.pd_connector`, `router.flow_control`, `router.predicted_latency` | new release | orchestrator, one arm per round |

`router.profile` is cold even though it is "just" the EPP config, because a
profile change swaps the plugin set and the data producers behind it; the
scorer weights that are hot only make sense within one profile.

## The staged order of arms

`{{script:tpu-llmd/seed_arms.py}}` writes the arms in the order the study
should spend on them, and the orchestrator picks the highest-ranked open arm:

1. **Capacity** (`baseline`, `a1` = N0-1, `a2` = N0+1). N0 is
   `ceil(target_qps / knee_rps)` from the recipe's measured knee, clamped
   to the chip budget. Nothing else is worth measuring until the fleet is
   roughly the right size: a router profile measured on a saturated fleet
   tells you about saturation. `a1` has the highest modelled headroom
   (goodput per chip rises by N0/(N0-1) if attainment holds); `a2` has zero
   by construction and only wins when the baseline failed a gate.
2. **KV reuse** (`a3` precise prefix routing, `a4` tiered host KV) --
   seeded **blocked** today because the runtime does not render them yet.
   They stay in the state so the plan is visible, and they unblock when a
   package release adds the renderer paths, not when a round decides to.
3. **Routing policy** (`a5` prefix-weighted, `a6` load-only, `a7`
   session-sticky when the workload is multi-turn). All hot inside, cold
   between. `a6` is the control: if routing with no prefix affinity at all
   is not worse, prefix reuse is not this workload's lever and `a3`/`a4`
   should be de-prioritised when they open.
4. **P/D disaggregation** (`a8`) -- blocked for the same reason as 2, seeded
   only when the catalog entry lists it as a suitable optimisation.

The `kind` on each arm is the coarse study taxonomy (`fleet-topology`,
`kv-reuse`, `routing-policy`, `engine-fleet-knob`, `platform-capacity`);
the `arm_kind` in `arm_plans.json` is the precise one the executor acts on.

## The inner sweep: what to move and how

The executor may move only the `sweep_knobs` the plan names for the arm,
within `inner_sweep_budget` steps (default 8). Each step is one
`reconfigure`, one `wait` if warm, one `slo`-stage bench, one ledger entry.
Suggested ladders; stop a knob after two consecutive non-improving steps:

| knob | ladder | reading the result |
| :--- | :--- | :--- |
| `engine.max_num_seqs` | x2 from the recipe value, then x0.5, then bisect toward the better side | goodput rises until TPOT hits its bound; the step that first fails `tpot` marks the ceiling, step back one |
| `engine.max_num_batched_tokens` | x2, x0.5 | moves TTFT against TPOT; prefill-heavy workloads tolerate higher, chat workloads lower |
| `engine.gpu_memory_utilization` | +0.05 up to 0.95 | larger KV cache raises admissible concurrency; a pod `OOM`/restart on `wait` is the ceiling, revert |
| `router.plugins.prefix-cache-scorer.weight` | 1, 3, 5 (others fixed) | raises prefix hit rate in `metrics.json`; past the knee it starves balance and `kv_util_pod_stddev` rises |
| `router.plugins.queue-scorer.weight` / `kv-cache-utilization-scorer.weight` | 1, 2, 4 | the balance side of the same trade |
| `router.plugins.prefix-cache-affinity-filter.peakPrefillThroughput` | set from the recipe's measured prefill tokens/s per replica, then x0.8, x1.2 | the threshold at which the affinity filter stops sending to the warm pod; the recipe's number is the right starting point, not the chart default |
| `router.plugins.prefix-cache-affinity-filter.affinityThreshold` | 0.6, 0.8, 0.95 | how much prefix overlap counts as affinity |
| `router.plugins.session-affinity-scorer.weight` | 1, 2, 4 | multi-turn only |
| `kv.block_size` | 16, 32, 64 | a prerequisite for precise routing (64); on its own trades fragmentation against indexing cost |

The plugin names and parameters are those of llm-d-router at the pinned
release
(https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/router/epp/configuration.md;
affinity-filter parameters at
https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity/README.md).
The renderer rejects a plugin name the profile does not contain, so a typo
is a refused render, not a silent no-op.

## The incumbent rule

- The arm's score is its **best admissible `slo`-stage ledger entry**, and
  `score` copies that entry's config snapshot beside the result as
  `config_path`. A sweep stage never scores an arm.
- The orchestrator copies `config_path` to the incumbent **only on `win`**.
  A tie's config may be equally good; it is not adopted, because adopting
  ties lets noise walk the incumbent.
- Each arm's `sets` are applied to the **current** incumbent, so a later
  arm inherits every earlier win's hot/warm tuning. This is what makes the
  study converge on one configuration rather than a table of independent
  results.
- A `loss` or `invalid` leaves the incumbent untouched and the next arm
  starts from the same place.

## When the evidence points past the fleet

Two patterns say the recipe, not the fleet, is the limit, and both become
`recommendations[]` rather than arms:

- every sweep point fails the same tenant's `tpot` gate, including at the
  lowest `max_num_seqs` tried -- single-replica decode is already over the
  bound (`domain: tpu-performance-optimization`);
- the `sweep` stage's knee is below `target_qps / replicas` at the smallest
  admissible fleet -- the recipe's measured knee was optimistic
  (`domain: tpu-model-onboarding`).

The study records the verdict it measured and reports the finding. It does
not reopen the recipe; that is a different study with a different protocol.
